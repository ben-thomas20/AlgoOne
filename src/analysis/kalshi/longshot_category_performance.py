"""Category performance analysis for Longshot YES Seller strategy.

Analyzes which market categories provide the best opportunities for the
longshot YES seller strategy, identifying where maker-taker gaps are largest.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd

from src.analysis.util.categories import CATEGORY_SQL, GROUP_COLORS, get_group
from src.common.analysis import Analysis, AnalysisOutput
from src.common.interfaces.chart import ChartConfig, ChartType, UnitType


class LongshotCategoryPerformance(Analysis):
    """Analyze longshot strategy performance by category."""

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
        min_price: int = 1,
        max_price: int = 10,
    ):
        super().__init__(
            name="longshot_category_performance",
            description="Longshot YES Seller strategy performance by market category",
        )
        base_dir = Path(__file__).parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base_dir / "data" / "kalshi" / "trades")
        self.markets_dir = Path(markets_dir or base_dir / "data" / "kalshi" / "markets")
        self.min_price = min_price
        self.max_price = max_price

    def run(self) -> AnalysisOutput:
        """Execute category performance analysis."""
        with self.progress("Analyzing performance by category"):
            con = duckdb.connect()

            df = con.execute(
                f"""
                WITH resolved_markets AS (
                    SELECT ticker, result
                    FROM '{self.markets_dir}/*.parquet'
                    WHERE status = 'finalized'
                      AND result IN ('yes', 'no')
                ),
                market_categories AS (
                    SELECT 
                        ticker,
                        {CATEGORY_SQL} AS category
                    FROM '{self.markets_dir}/*.parquet'
                )
                SELECT
                    c.category,
                    COUNT(*) as n_trades,
                    AVG(CASE WHEN t.taker_side = 'yes' AND m.result = 'no' THEN 1 ELSE 0 END) as win_rate,
                    SUM(CASE 
                        WHEN t.taker_side = 'yes' AND m.result = 'no' 
                        THEN t.count * (100 - (100 - t.yes_price)) / 100.0
                        WHEN t.taker_side = 'yes' AND m.result = 'yes'
                        THEN -t.count * (100 - t.yes_price) / 100.0
                        ELSE 0
                    END) as total_profit_usd,
                    AVG((100 - t.yes_price) / 100.0) as avg_cost_basis,
                    SUM(t.count) as total_contracts,
                    SUM(t.count * t.yes_price / 100.0) as total_volume_usd
                FROM '{self.trades_dir}/*.parquet' t
                INNER JOIN resolved_markets m ON t.ticker = m.ticker
                LEFT JOIN market_categories c ON t.ticker = c.ticker
                WHERE t.yes_price BETWEEN {self.min_price} AND {self.max_price}
                  AND t.taker_side = 'yes'
                GROUP BY c.category
                ORDER BY win_rate DESC
                """
            ).df()

            # Add group classification
            df["group"] = df["category"].apply(get_group)
            
            # Calculate returns
            df["expected_win_rate"] = 1 - df["avg_cost_basis"]
            df["excess_return_pct"] = (df["win_rate"] - df["expected_win_rate"]) * 100
            df["avg_return_pct"] = (df["total_profit_usd"] / (df["total_contracts"] * df["avg_cost_basis"])) * 100
            df["roi_pct"] = (df["total_profit_usd"] / df["total_volume_usd"]) * 100

            # Group by high-level category
            grouped = df.groupby("group").agg({
                "n_trades": "sum",
                "win_rate": "mean",
                "total_profit_usd": "sum",
                "total_contracts": "sum",
                "total_volume_usd": "sum",
                "excess_return_pct": "mean",
                "avg_return_pct": "mean"
            }).reset_index()
            grouped = grouped.sort_values("excess_return_pct", ascending=False)

        fig = self._create_figure(df, grouped)
        chart = self._create_chart(grouped)

        return AnalysisOutput(figure=fig, data=grouped, chart=chart)

    def _create_figure(self, detailed_df: pd.DataFrame, grouped_df: pd.DataFrame) -> plt.Figure:
        """Create category performance visualization."""
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # 1. Excess Return by Category
        ax1 = fig.add_subplot(gs[0, :])
        categories = grouped_df["group"]
        excess_returns = grouped_df["excess_return_pct"]
        colors = [GROUP_COLORS.get(cat, "#gray") for cat in categories]
        
        bars = ax1.barh(categories, excess_returns, color=colors, alpha=0.7)
        ax1.axvline(x=0, color="black", linestyle="-", linewidth=0.8)
        ax1.set_xlabel("Excess Return (%)")
        ax1.set_title("Longshot YES Seller Strategy: Excess Return by Category")
        ax1.grid(True, alpha=0.3, axis="x")
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax1.text(width, bar.get_y() + bar.get_height()/2, 
                    f'{width:.2f}%', ha='left' if width > 0 else 'right',
                    va='center', fontsize=9)

        # 2. Trade Volume by Category
        ax2 = fig.add_subplot(gs[1, 0])
        volume_sorted = grouped_df.sort_values("total_volume_usd", ascending=True)
        colors_vol = [GROUP_COLORS.get(cat, "#gray") for cat in volume_sorted["group"]]
        ax2.barh(volume_sorted["group"], volume_sorted["total_volume_usd"] / 1e6, 
                color=colors_vol, alpha=0.7)
        ax2.set_xlabel("Total Volume ($M)")
        ax2.set_title("Trade Volume by Category")
        ax2.grid(True, alpha=0.3, axis="x")

        # 3. Win Rate vs Expected by Category
        ax3 = fig.add_subplot(gs[1, 1])
        categories_sorted = grouped_df.sort_values("win_rate", ascending=False)["group"]
        x = range(len(categories_sorted))
        width = 0.35
        
        # Get win rates for sorted categories
        win_rates = []
        expected_rates = []
        for cat in categories_sorted:
            row = grouped_df[grouped_df["group"] == cat].iloc[0]
            win_rates.append(row["win_rate"] * 100)
            # Expected is average of (1 - cost_basis) for that category
            cat_rows = detailed_df[detailed_df["group"] == cat]
            expected = (1 - cat_rows["avg_cost_basis"].mean()) * 100
            expected_rates.append(expected)
        
        ax3.bar([i - width/2 for i in x], win_rates, width, label="Actual Win Rate", 
               color="#2ecc71", alpha=0.7)
        ax3.bar([i + width/2 for i in x], expected_rates, width, label="Expected Win Rate",
               color="#95a5a6", alpha=0.7)
        ax3.set_ylabel("Win Rate (%)")
        ax3.set_title("Actual vs Expected Win Rate")
        ax3.set_xticks(x)
        ax3.set_xticklabels(categories_sorted, rotation=45, ha="right")
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis="y")

        fig.suptitle(f"Category Performance Analysis (Price Range: {self.min_price}-{self.max_price}¢)", 
                    fontsize=14, fontweight="bold")
        plt.tight_layout()
        return fig

    def _create_chart(self, df: pd.DataFrame) -> ChartConfig:
        """Create interactive chart configuration."""
        chart_data = [
            {
                "category": row["group"],
                "excess_return": round(row["excess_return_pct"], 2),
                "avg_return": round(row["avg_return_pct"], 2),
                "n_trades": int(row["n_trades"]),
                "volume_millions": round(row["total_volume_usd"] / 1e6, 2),
            }
            for _, row in df.iterrows()
        ]

        return ChartConfig(
            type=ChartType.BAR,
            data=chart_data,
            xKey="category",
            yKeys=["excess_return"],
            title="Excess Return by Category",
            yUnit=UnitType.PERCENT,
            xLabel="Category",
            yLabel="Excess Return (%)",
        )
