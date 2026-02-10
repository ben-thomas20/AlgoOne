"""Backtest the Longshot YES Seller strategy.

This analysis validates the strategy of acting as a maker by selling YES contracts
at longshot prices (1-10 cents), exploiting the systematic overpayment by takers
for affirmative outcomes.

Strategy: Place maker limit orders to sell YES contracts at 1-10 cent prices,
effectively buying NO contracts at 90-99 cents. The research shows makers buying
NO at these prices have positive excess returns due to the "Optimism Tax".
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.analysis.util.categories import CATEGORY_SQL, get_group
from src.analysis.util.categories import CATEGORY_SQL, get_group
from src.common.analysis import Analysis, AnalysisOutput
from src.common.interfaces.chart import ChartConfig, ChartType, UnitType


class LongshotYesSellerBacktest(Analysis):
    """Backtest the Longshot YES Seller strategy on historical Kalshi data."""

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
        min_price: int = 1,
        max_price: int = 10,
        min_volume_usd: float = 100.0,
    ):
        super().__init__(
            name="longshot_yes_seller_backtest",
            description="Backtest of Longshot YES Seller market-making strategy",
        )
        base_dir = Path(__file__).parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base_dir / "data" / "kalshi" / "trades")
        self.markets_dir = Path(markets_dir or base_dir / "data" / "kalshi" / "markets")
        self.min_price = min_price
        self.max_price = max_price
        self.min_volume_usd = min_volume_usd

    def run(self) -> AnalysisOutput:
        """Execute the backtest analysis."""
        with self.progress("Loading and filtering markets"):
            con = duckdb.connect()

            # Get resolved markets with category and volume filtering
            markets_df = con.execute(
                f"""
                WITH market_volumes AS (
                    SELECT 
                        ticker,
                        SUM(count * yes_price / 100.0) AS volume_usd
                    FROM '{self.trades_dir}/*.parquet'
                    GROUP BY ticker
                ),
                market_categories AS (
                    SELECT 
                        ticker,
                        {CATEGORY_SQL} AS category
                    FROM '{self.markets_dir}/*.parquet'
                )
                SELECT 
                    m.ticker,
                    m.result,
                    c.category,
                    COALESCE(v.volume_usd, 0) AS volume_usd
                FROM '{self.markets_dir}/*.parquet' m
                LEFT JOIN market_volumes v ON m.ticker = v.ticker
                LEFT JOIN market_categories c ON m.ticker = c.ticker
                WHERE m.status = 'finalized'
                  AND m.result IN ('yes', 'no')
                  AND COALESCE(v.volume_usd, 0) >= {self.min_volume_usd}
                """
            ).df()

            markets_df["group"] = markets_df["category"].apply(get_group)

        with self.progress("Simulating maker orders (selling YES at longshot prices)"):
            # Simulate strategy: We are MAKERS selling YES (buying NO) at longshot prices
            # When taker buys YES at 5 cents, we sell YES at 5 cents (buy NO at 95 cents)
            # We win if result = 'no', lose if result = 'yes'
            
            trades_df = con.execute(
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
                    t.ticker,
                    t.yes_price,
                    t.no_price,
                    t.taker_side,
                    t.count,
                    t.created_time,
                    m.result,
                    c.category,
                    -- Strategy: We sell YES (buy NO) when taker buys YES at longshot prices
                    CASE 
                        WHEN t.taker_side = 'yes' AND t.yes_price BETWEEN {self.min_price} AND {self.max_price}
                        THEN 1 ELSE 0 
                    END AS is_strategy_trade,
                    -- We buy NO at (100 - yes_price) cents
                    100 - t.yes_price AS our_cost_basis,
                    -- We win if result is NO
                    CASE 
                        WHEN t.taker_side = 'yes' AND m.result = 'no' THEN 1 
                        WHEN t.taker_side = 'yes' AND m.result = 'yes' THEN 0
                        ELSE NULL
                    END AS we_won,
                    -- Capital at risk per contract
                    (100 - t.yes_price) / 100.0 AS capital_at_risk,
                    -- Trade volume
                    t.count * t.yes_price / 100.0 AS trade_volume_usd
                FROM '{self.trades_dir}/*.parquet' t
                INNER JOIN resolved_markets m ON t.ticker = m.ticker
                LEFT JOIN market_categories c ON t.ticker = c.ticker
                WHERE t.yes_price BETWEEN {self.min_price} AND {self.max_price}
                  AND t.taker_side = 'yes'
                """
            ).df()

            trades_df["group"] = trades_df["category"].apply(get_group)

        with self.progress("Calculating strategy performance"):
            # Overall performance
            total_trades = len(trades_df)
            total_capital = trades_df["capital_at_risk"].sum() * trades_df["count"].sum()
            win_rate = trades_df["we_won"].mean()
            
            # Calculate returns
            # If we win (NO outcome), we get 100 cents per contract, profit = 100 - cost_basis
            # If we lose (YES outcome), we lose our cost basis
            trades_df["profit_per_contract"] = trades_df.apply(
                lambda row: (100 - row["our_cost_basis"]) if row["we_won"] == 1 else -row["our_cost_basis"],
                axis=1
            )
            trades_df["total_profit"] = trades_df["profit_per_contract"] * trades_df["count"] / 100.0
            trades_df["return_pct"] = (trades_df["profit_per_contract"] / trades_df["our_cost_basis"]) * 100

            total_profit = trades_df["total_profit"].sum()
            avg_return = trades_df["return_pct"].mean()
            median_return = trades_df["return_pct"].median()

            # Expected value calculation
            trades_df["expected_prob"] = trades_df["our_cost_basis"] / 100.0
            trades_df["excess_return"] = trades_df["we_won"] - trades_df["expected_prob"]
            avg_excess_return = trades_df["excess_return"].mean()

            # Performance by price
            price_performance = trades_df.groupby("yes_price").agg({
                "we_won": ["count", "mean", "std"],
                "excess_return": "mean",
                "return_pct": ["mean", "std"],
                "count": "sum",
                "capital_at_risk": "mean"
            }).reset_index()
            price_performance.columns = [
                "yes_price", "n_trades", "win_rate", "win_std",
                "excess_return", "avg_return_pct", "return_std", "total_contracts", "avg_cost"
            ]

            # Performance by category
            category_performance = trades_df.groupby("group").agg({
                "we_won": ["count", "mean"],
                "excess_return": "mean",
                "return_pct": "mean",
                "total_profit": "sum",
                "count": "sum"
            }).reset_index()
            category_performance.columns = [
                "category", "n_trades", "win_rate", "excess_return", "avg_return_pct", "total_profit", "total_contracts"
            ]
            category_performance = category_performance.sort_values("excess_return", ascending=False)

        with self.progress("Calculating risk metrics"):
            # Sharpe Ratio (annualized, assuming 250 trading days)
            daily_returns = trades_df.groupby(trades_df["created_time"].dt.date)["return_pct"].mean()
            sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(250) if len(daily_returns) > 1 else 0

            # Maximum drawdown
            cumulative_returns = (1 + trades_df.sort_values("created_time")["return_pct"] / 100).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min() * 100

            # Kelly Criterion optimal bet size (f* = p - q/b where p=win_prob, q=1-p, b=odds)
            p = win_rate
            q = 1 - p
            # Average odds: if we risk X to win (100-X), b = (100-X)/X
            avg_cost = trades_df["our_cost_basis"].mean()
            b = (100 - avg_cost) / avg_cost
            kelly_fraction = (p - q / b) if b > 0 else 0

        # Prepare summary statistics
        summary = {
            "total_trades": total_trades,
            "total_capital_usd": total_capital,
            "win_rate": win_rate * 100,
            "avg_return_pct": avg_return,
            "median_return_pct": median_return,
            "total_profit_usd": total_profit,
            "avg_excess_return_pct": avg_excess_return * 100,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown_pct": max_drawdown,
            "kelly_fraction": kelly_fraction * 100,
            "price_range": f"{self.min_price}-{self.max_price}¢",
        }

        fig = self._create_figure(price_performance, category_performance, summary)
        chart = self._create_chart(price_performance, category_performance)

        # Combine all data for CSV export
        export_data = pd.DataFrame([summary])
        export_data["price_performance"] = [price_performance.to_dict("records")]
        export_data["category_performance"] = [category_performance.to_dict("records")]

        return AnalysisOutput(figure=fig, data=export_data, chart=chart)

    def _create_figure(
        self, 
        price_perf: pd.DataFrame, 
        category_perf: pd.DataFrame,
        summary: dict
    ) -> plt.Figure:
        """Create comprehensive backtest visualization."""
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # 1. Win Rate by Price
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(price_perf["yes_price"], price_perf["win_rate"] * 100, 
                marker="o", linewidth=2, markersize=6, color="#2ecc71")
        ax1.plot(price_perf["yes_price"], (100 - price_perf["yes_price"]), 
                linestyle="--", color="gray", alpha=0.5, label="Expected (NO win rate)")
        ax1.axhline(y=50, color="red", linestyle=":", alpha=0.3)
        ax1.set_xlabel("YES Price (cents)")
        ax1.set_ylabel("Our Win Rate (%)")
        ax1.set_title("Strategy Win Rate by Price\n(We buy NO when takers buy YES)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Excess Return by Price
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.bar(price_perf["yes_price"], price_perf["excess_return"] * 100, 
               color=["#2ecc71" if x > 0 else "#e74c3c" for x in price_perf["excess_return"]])
        ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
        ax2.set_xlabel("YES Price (cents)")
        ax2.set_ylabel("Excess Return (%)")
        ax2.set_title("Mispricing by Price\n(Positive = Market Underprices NO)")
        ax2.grid(True, alpha=0.3)

        # 3. Average Return % by Price
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.bar(price_perf["yes_price"], price_perf["avg_return_pct"], 
               color="#3498db", alpha=0.7)
        ax3.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
        ax3.set_xlabel("YES Price (cents)")
        ax3.set_ylabel("Average Return (%)")
        ax3.set_title("Average Return by Price")
        ax3.grid(True, alpha=0.3)

        # 4. Trade Volume by Price
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.bar(price_perf["yes_price"], price_perf["total_contracts"] / 1000, 
               color="#9b59b6", alpha=0.7)
        ax4.set_xlabel("YES Price (cents)")
        ax4.set_ylabel("Contracts (thousands)")
        ax4.set_title("Strategy Trade Volume by Price")
        ax4.grid(True, alpha=0.3)

        # 5. Performance by Category
        ax5 = fig.add_subplot(gs[2, 0])
        categories = category_perf["category"].head(10)
        excess_returns = category_perf["excess_return"].head(10) * 100
        colors = ["#2ecc71" if x > 0 else "#e74c3c" for x in excess_returns]
        ax5.barh(categories, excess_returns, color=colors, alpha=0.7)
        ax5.axvline(x=0, color="black", linestyle="-", linewidth=0.8)
        ax5.set_xlabel("Excess Return (%)")
        ax5.set_title("Strategy Performance by Category (Top 10)")
        ax5.grid(True, alpha=0.3, axis="x")

        # 6. Summary Statistics
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.axis("off")
        summary_text = f"""
STRATEGY BACKTEST SUMMARY
{'=' * 40}

Total Trades: {summary['total_trades']:,}
Win Rate: {summary['win_rate']:.2f}%
Avg Return: {summary['avg_return_pct']:.2f}%
Median Return: {summary['median_return_pct']:.2f}%

Total Profit: ${summary['total_profit_usd']:,.2f}
Avg Excess Return: {summary['avg_excess_return_pct']:.2f}%

Risk Metrics:
  Sharpe Ratio: {summary['sharpe_ratio']:.2f}
  Max Drawdown: {summary['max_drawdown_pct']:.2f}%
  Kelly Fraction: {summary['kelly_fraction']:.2f}%

Price Range: {summary['price_range']}
Min Volume: ${self.min_volume_usd:.0f}
        """
        ax6.text(0.1, 0.5, summary_text, fontsize=11, family="monospace",
                verticalalignment="center")

        fig.suptitle("Longshot YES Seller Strategy Backtest", fontsize=16, fontweight="bold")
        return fig

    def _create_chart(
        self, 
        price_perf: pd.DataFrame,
        category_perf: pd.DataFrame
    ) -> ChartConfig:
        """Create interactive chart configuration."""
        chart_data = [
            {
                "price": int(row["yes_price"]),
                "win_rate": round(row["win_rate"] * 100, 2),
                "expected_win_rate": round(100 - row["yes_price"], 2),
                "excess_return": round(row["excess_return"] * 100, 2),
                "avg_return": round(row["avg_return_pct"], 2),
                "n_trades": int(row["n_trades"]),
            }
            for _, row in price_perf.iterrows()
        ]

        return ChartConfig(
            type=ChartType.LINE,
            data=chart_data,
            xKey="price",
            yKeys=["win_rate", "expected_win_rate"],
            title="Longshot YES Seller Strategy: Win Rate by Price",
            yUnit=UnitType.PERCENT,
            xLabel="YES Contract Price (cents)",
            yLabel="Win Rate (%)",
        )
