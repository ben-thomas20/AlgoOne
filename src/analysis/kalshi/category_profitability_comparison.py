"""Compare performance of All Categories vs Top 5 profitable categories.

This analysis demonstrates the improvement gained by filtering to only
profitable categories vs trading all categories indiscriminately.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd

from src.analysis.util.categories import CATEGORY_SQL, GROUP_COLORS, get_group
from src.common.analysis import Analysis, AnalysisOutput
from src.common.interfaces.chart import ChartConfig, ChartType, UnitType


class CategoryProfitabilityComparison(Analysis):
    """Compare All Categories vs Top 5 filtered strategy."""

    TOP_5_CATEGORIES = {
        "Science/Tech",
        "Esports",
        "Entertainment",
        "Media",
        "Politics",
    }

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
        min_price: int = 1,
        max_price: int = 10,
    ):
        super().__init__(
            name="category_profitability_comparison",
            description="Compare All Categories vs Top 5 profitable categories",
        )
        base_dir = Path(__file__).parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base_dir / "data" / "kalshi" / "trades")
        self.markets_dir = Path(markets_dir or base_dir / "data" / "kalshi" / "markets")
        self.min_price = min_price
        self.max_price = max_price

    def run(self) -> AnalysisOutput:
        """Execute comparison analysis."""
        with self.progress("Analyzing category performance"):
            con = duckdb.connect()

            # Get all category performance
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
                        THEN t.count * t.yes_price / 100.0
                        WHEN t.taker_side = 'yes' AND m.result = 'yes'
                        THEN -t.count * (100 - t.yes_price) / 100.0
                        ELSE 0
                    END) as total_profit_usd,
                    AVG((100 - t.yes_price) / 100.0) as avg_cost_basis,
                    SUM(t.count) as total_contracts,
                    SUM(t.count * (100 - t.yes_price) / 100.0) as total_capital_invested
                FROM '{self.trades_dir}/*.parquet' t
                INNER JOIN resolved_markets m ON t.ticker = m.ticker
                LEFT JOIN market_categories c ON t.ticker = c.ticker
                WHERE t.yes_price BETWEEN {self.min_price} AND {self.max_price}
                  AND t.taker_side = 'yes'
                GROUP BY c.category
                """
            ).df()

            df["group"] = df["category"].apply(get_group)
            
            # Calculate returns
            df["avg_return_pct"] = (df["total_profit_usd"] / df["total_capital_invested"]) * 100
            df["expected_win_rate"] = 1 - df["avg_cost_basis"]
            df["excess_return_pct"] = (df["win_rate"] - df["expected_win_rate"]) * 100
            
            # Mark if in Top 5
            df["in_top5"] = df["group"].isin(self.TOP_5_CATEGORIES)
            
            # Calculate aggregates
            top5_stats = df[df["in_top5"]].agg({
                "n_trades": "sum",
                "total_profit_usd": "sum",
                "total_capital_invested": "sum",
            })
            
            all_stats = df.agg({
                "n_trades": "sum",
                "total_profit_usd": "sum",
                "total_capital_invested": "sum",
            })
            
            top5_return = (top5_stats["total_profit_usd"] / top5_stats["total_capital_invested"]) * 100
            all_return = (all_stats["total_profit_usd"] / all_stats["total_capital_invested"]) * 100
            
            comparison = pd.DataFrame([
                {
                    "strategy": "All Categories",
                    "n_trades": int(all_stats["n_trades"]),
                    "total_profit": all_stats["total_profit_usd"],
                    "capital_invested": all_stats["total_capital_invested"],
                    "return_pct": all_return,
                },
                {
                    "strategy": "Top 5 Only",
                    "n_trades": int(top5_stats["n_trades"]),
                    "total_profit": top5_stats["total_profit_usd"],
                    "capital_invested": top5_stats["total_capital_invested"],
                    "return_pct": top5_return,
                },
            ])
            
            comparison["improvement"] = comparison["return_pct"] - all_return

        fig = self._create_figure(df, comparison)
        chart = self._create_chart(df, comparison)

        return AnalysisOutput(figure=fig, data=comparison, chart=chart)

    def _create_figure(self, category_df: pd.DataFrame, comparison_df: pd.DataFrame) -> plt.Figure:
        """Create visualization."""
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # 1. Returns by Category (sorted)
        ax1 = fig.add_subplot(gs[0, :])
        sorted_df = category_df.sort_values("avg_return_pct", ascending=True)
        colors = ["#2ecc71" if row["in_top5"] else "#95a5a6" for _, row in sorted_df.iterrows()]
        bars = ax1.barh(sorted_df["group"], sorted_df["avg_return_pct"], color=colors, alpha=0.7)
        ax1.axvline(x=0, color="black", linestyle="-", linewidth=0.8)
        ax1.set_xlabel("Average Return (%)")
        ax1.set_title("Return by Category (Green = Top 5, Gray = Excluded)")
        ax1.grid(True, alpha=0.3, axis="x")

        # 2. Trade Volume Comparison
        ax2 = fig.add_subplot(gs[1, 0])
        strategies = comparison_df["strategy"]
        trades = comparison_df["n_trades"] / 1000
        ax2.bar(strategies, trades, color=["#95a5a6", "#2ecc71"], alpha=0.7)
        ax2.set_ylabel("Trades (thousands)")
        ax2.set_title("Trade Volume Comparison")
        ax2.grid(True, alpha=0.3, axis="y")
        
        # Add value labels
        for i, (strategy, trade_count) in enumerate(zip(strategies, trades)):
            ax2.text(i, trade_count, f"{trade_count:.0f}k", ha="center", va="bottom")

        # 3. Return Comparison
        ax3 = fig.add_subplot(gs[1, 1])
        returns = comparison_df["return_pct"]
        colors_comp = ["#95a5a6", "#2ecc71"]
        bars = ax3.bar(strategies, returns, color=colors_comp, alpha=0.7)
        ax3.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
        ax3.set_ylabel("Average Return (%)")
        ax3.set_title("Return Comparison")
        ax3.grid(True, alpha=0.3, axis="y")
        
        # Add value labels
        for i, (strategy, ret) in enumerate(zip(strategies, returns)):
            ax3.text(i, ret, f"{ret:.2f}%", ha="center", va="bottom" if ret > 0 else "top")

        fig.suptitle("Category Filtering Impact: All vs Top 5", fontsize=14, fontweight="bold")
        plt.tight_layout()
        return fig

    def _create_chart(self, category_df: pd.DataFrame, comparison_df: pd.DataFrame) -> ChartConfig:
        """Create chart configuration."""
        chart_data = [
            {
                "category": row["group"],
                "return": round(row["avg_return_pct"], 2),
                "in_top5": "Top 5" if row["in_top5"] else "Excluded",
            }
            for _, row in category_df.iterrows()
        ]

        return ChartConfig(
            type=ChartType.BAR,
            data=chart_data,
            xKey="category",
            yKeys=["return"],
            title="Returns by Category",
            yUnit=UnitType.PERCENT,
            xLabel="Category",
            yLabel="Average Return (%)",
        )
