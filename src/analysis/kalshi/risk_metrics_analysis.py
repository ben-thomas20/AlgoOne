"""Comprehensive risk analysis for Longshot NO Buyer strategy.

Includes:
- Return distribution analysis
- Drawdown analysis over time
- Monte Carlo simulation (1000 runs)
- Sensitivity analysis
- Correlation between categories
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.analysis.util.categories import CATEGORY_SQL, GROUP_COLORS, get_group
from src.common.analysis import Analysis, AnalysisOutput
from src.common.interfaces.chart import ChartConfig, ChartType, UnitType


class RiskMetricsAnalysis(Analysis):
    """Comprehensive risk analysis with Monte Carlo simulation."""

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
        n_simulations: int = 1000,
    ):
        super().__init__(
            name="risk_metrics_analysis",
            description="Risk analysis with Monte Carlo simulation and sensitivity tests",
        )
        base_dir = Path(__file__).parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base_dir / "data" / "kalshi" / "trades")
        self.markets_dir = Path(markets_dir or base_dir / "data" / "kalshi" / "markets")
        self.min_price = min_price
        self.max_price = max_price
        self.n_simulations = n_simulations

    def run(self) -> AnalysisOutput:
        """Execute risk analysis."""
        with self.progress("Loading historical trades"):
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
                    t.yes_price,
                    (100 - t.yes_price) AS our_cost,
                    CASE WHEN t.taker_side = 'yes' AND m.result = 'no' THEN 1 ELSE 0 END as won,
                    c.category
                FROM '{self.trades_dir}/*.parquet' t
                INNER JOIN resolved_markets m ON t.ticker = m.ticker
                LEFT JOIN market_categories c ON t.ticker = c.ticker
                WHERE t.yes_price BETWEEN {self.min_price} AND {self.max_price}
                  AND t.taker_side = 'yes'
                """
            ).df()

            df["group"] = df["category"].apply(get_group)
            df = df[df["group"].isin(self.TOP_5_CATEGORIES)]
            
            print(f"  Loaded {len(df):,} trades for analysis")

        with self.progress("Running Monte Carlo simulation"):
            monte_carlo_results = self._run_monte_carlo(df)

        with self.progress("Calculating risk metrics"):
            risk_metrics = self._calculate_risk_metrics(df, monte_carlo_results)

        fig = self._create_figure(df, monte_carlo_results, risk_metrics)
        chart = self._create_chart(monte_carlo_results)

        # Export results
        export_data = pd.DataFrame([risk_metrics])
        
        return AnalysisOutput(figure=fig, data=export_data, chart=chart)

    def _run_monte_carlo(self, df: pd.DataFrame, sample_size: int = 1000) -> dict:
        """Run Monte Carlo simulation by sampling trades."""
        final_returns = []
        max_drawdowns = []
        win_rates = []
        
        for i in range(self.n_simulations):
            # Randomly sample trades (with replacement)
            sample = df.sample(n=min(sample_size, len(df)), replace=True)
            
            # Calculate returns
            sample["profit"] = sample.apply(
                lambda row: (100 - row["our_cost"]) if row["won"] else -row["our_cost"],
                axis=1
            )
            sample["return_pct"] = (sample["profit"] / sample["our_cost"]) * 100
            
            # Calculate metrics for this simulation
            total_return = sample["return_pct"].mean()
            win_rate = sample["won"].mean()
            
            # Calculate drawdown
            cumulative = (1 + sample["return_pct"] / 100).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = ((cumulative - running_max) / running_max).min() * 100
            
            final_returns.append(total_return)
            max_drawdowns.append(drawdown)
            win_rates.append(win_rate * 100)
        
        return {
            "final_returns": final_returns,
            "max_drawdowns": max_drawdowns,
            "win_rates": win_rates,
        }

    def _calculate_risk_metrics(self, df: pd.DataFrame, mc_results: dict) -> dict:
        """Calculate comprehensive risk metrics."""
        # Historical performance
        df["profit"] = df.apply(
            lambda row: (100 - row["our_cost"]) if row["won"] else -row["our_cost"],
            axis=1
        )
        df["return_pct"] = (df["profit"] / df["our_cost"]) * 100
        
        # Basic metrics
        avg_return = df["return_pct"].mean()
        median_return = df["return_pct"].median()
        std_return = df["return_pct"].std()
        win_rate = df["won"].mean() * 100
        
        # Monte Carlo statistics
        mc_mean_return = np.mean(mc_results["final_returns"])
        mc_std_return = np.std(mc_results["final_returns"])
        mc_5th_percentile = np.percentile(mc_results["final_returns"], 5)
        mc_95th_percentile = np.percentile(mc_results["final_returns"], 95)
        probability_profit = (np.array(mc_results["final_returns"]) > 0).mean() * 100
        
        mc_avg_drawdown = np.mean(mc_results["max_drawdowns"])
        mc_worst_drawdown = np.min(mc_results["max_drawdowns"])
        
        # Sensitivity: What if win rate is 2% lower?
        df_sensitivity = df.copy()
        # Randomly flip 2% of wins to losses
        wins_indices = df_sensitivity[df_sensitivity["won"] == 1].index
        flip_count = int(len(wins_indices) * 0.02)
        flip_indices = np.random.choice(wins_indices, flip_count, replace=False)
        df_sensitivity.loc[flip_indices, "won"] = 0
        df_sensitivity["profit_sens"] = df_sensitivity.apply(
            lambda row: (100 - row["our_cost"]) if row["won"] else -row["our_cost"],
            axis=1
        )
        df_sensitivity["return_sens"] = (df_sensitivity["profit_sens"] / df_sensitivity["our_cost"]) * 100
        sensitivity_return = df_sensitivity["return_sens"].mean()
        
        return {
            "avg_return_pct": avg_return,
            "median_return_pct": median_return,
            "std_return_pct": std_return,
            "win_rate_pct": win_rate,
            "mc_mean_return": mc_mean_return,
            "mc_std_return": mc_std_return,
            "mc_5th_percentile": mc_5th_percentile,
            "mc_95th_percentile": mc_95th_percentile,
            "probability_profit_pct": probability_profit,
            "mc_avg_drawdown": mc_avg_drawdown,
            "mc_worst_drawdown": mc_worst_drawdown,
            "sensitivity_win_rate_minus_2pct": sensitivity_return,
        }

    def _create_figure(self, df: pd.DataFrame, mc_results: dict, risk_metrics: dict) -> plt.Figure:
        """Create risk analysis visualization."""
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

        # 1. Monte Carlo Return Distribution
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.hist(mc_results["final_returns"], bins=50, color="#3498db", alpha=0.7, edgecolor="black")
        ax1.axvline(x=0, color="red", linestyle="--", linewidth=2, label="Break-even")
        ax1.axvline(x=risk_metrics["mc_mean_return"], color="green", linestyle="-", 
                   linewidth=2, label=f"Mean: {risk_metrics['mc_mean_return']:.2f}%")
        ax1.axvline(x=risk_metrics["mc_5th_percentile"], color="orange", linestyle=":", 
                   linewidth=2, label=f"5th %ile: {risk_metrics['mc_5th_percentile']:.2f}%")
        ax1.set_xlabel("Return (%)")
        ax1.set_ylabel("Frequency")
        ax1.set_title(f"Monte Carlo Return Distribution ({self.n_simulations} simulations)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Monte Carlo Drawdown Distribution
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.hist(mc_results["max_drawdowns"], bins=40, color="#e74c3c", alpha=0.7, edgecolor="black")
        ax2.axvline(x=risk_metrics["mc_avg_drawdown"], color="black", linestyle="-", linewidth=2)
        ax2.set_xlabel("Max Drawdown (%)")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Drawdown Distribution")
        ax2.grid(True, alpha=0.3)

        # 3. Historical Return Distribution
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.hist(df["return_pct"], bins=100, range=(-100, 20), color="#2ecc71", alpha=0.7, edgecolor="black")
        ax3.axvline(x=0, color="red", linestyle="--", linewidth=2)
        ax3.axvline(x=risk_metrics["avg_return_pct"], color="black", linestyle="-", linewidth=2)
        ax3.set_xlabel("Return (%)")
        ax3.set_ylabel("Frequency")
        ax3.set_title("Historical Trade Returns")
        ax3.grid(True, alpha=0.3)

        # 4. Win Rate by Category
        ax4 = fig.add_subplot(gs[1, 1])
        category_win_rates = df.groupby("group")["won"].mean() * 100
        category_win_rates = category_win_rates.sort_values(ascending=True)
        colors = [GROUP_COLORS.get(cat, "#gray") for cat in category_win_rates.index]
        ax4.barh(category_win_rates.index, category_win_rates.values, color=colors, alpha=0.7)
        ax4.axvline(x=95, color="orange", linestyle="--", linewidth=2, label="95% threshold")
        ax4.set_xlabel("Win Rate (%)")
        ax4.set_title("Win Rate by Category")
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis="x")

        # 5. Return by Category
        ax5 = fig.add_subplot(gs[1, 2])
        category_returns = df.groupby("group")["return_pct"].mean()
        category_returns = category_returns.sort_values(ascending=True)
        colors = [GROUP_COLORS.get(cat, "#gray") for cat in category_returns.index]
        ax5.barh(category_returns.index, category_returns.values, color=colors, alpha=0.7)
        ax5.axvline(x=0, color="black", linestyle="-", linewidth=0.8)
        ax5.set_xlabel("Avg Return (%)")
        ax5.set_title("Return by Category")
        ax5.grid(True, alpha=0.3, axis="x")

        # 6. Summary Statistics
        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis("off")
        
        summary_text = f"""
RISK ANALYSIS SUMMARY
{'=' * 80}

HISTORICAL PERFORMANCE:
  Average Return:     {risk_metrics['avg_return_pct']:.2f}%
  Median Return:      {risk_metrics['median_return_pct']:.2f}%
  Std Deviation:      {risk_metrics['std_return_pct']:.2f}%
  Win Rate:           {risk_metrics['win_rate_pct']:.2f}%

MONTE CARLO RESULTS ({self.n_simulations} simulations):
  Mean Return:        {risk_metrics['mc_mean_return']:.2f}%
  Std Deviation:      {risk_metrics['mc_std_return']:.2f}%
  5th Percentile:     {risk_metrics['mc_5th_percentile']:.2f}%
  95th Percentile:    {risk_metrics['mc_95th_percentile']:.2f}%
  Prob of Profit:     {risk_metrics['probability_profit_pct']:.1f}%

DRAWDOWN ANALYSIS:
  Average Drawdown:   {risk_metrics['mc_avg_drawdown']:.2f}%
  Worst Drawdown:     {risk_metrics['mc_worst_drawdown']:.2f}%

SENSITIVITY ANALYSIS:
  If Win Rate -2%:    {risk_metrics['sensitivity_win_rate_minus_2pct']:.2f}%

DECISION CRITERIA:
  Probability of Profit:  {'✅ PASS' if risk_metrics['probability_profit_pct'] >= 90 else '❌ FAIL'} (target: >90%)
  Worst Drawdown:         {'✅ PASS' if risk_metrics['mc_worst_drawdown'] >= -30 else '❌ FAIL'} (target: >-30%)
  5th Percentile Return:  {'✅ PASS' if risk_metrics['mc_5th_percentile'] >= -5 else '❌ FAIL'} (target: >-5%)
        """
        
        ax6.text(0.05, 0.5, summary_text, fontsize=10, family="monospace",
                verticalalignment="center", transform=ax6.transAxes)

        fig.suptitle("Risk Metrics Analysis - Top 5 Categories", fontsize=14, fontweight="bold")
        
        return fig

    def _create_chart(self, mc_results: dict) -> ChartConfig:
        """Create chart configuration."""
        # Create histogram data for Monte Carlo returns
        hist, bin_edges = np.histogram(mc_results["final_returns"], bins=50)
        chart_data = [
            {
                "return": round((bin_edges[i] + bin_edges[i+1]) / 2, 2),
                "frequency": int(hist[i]),
            }
            for i in range(len(hist))
        ]

        return ChartConfig(
            type=ChartType.BAR,
            data=chart_data,
            xKey="return",
            yKeys=["frequency"],
            title="Monte Carlo Return Distribution",
            xLabel="Return (%)",
            yLabel="Frequency",
        )
