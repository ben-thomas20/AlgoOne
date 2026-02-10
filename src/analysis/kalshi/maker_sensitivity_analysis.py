"""Sensitivity analysis for maker strategy with different fill rates.

Tests how the maker strategy performs under different assumptions about
order fill rates (30%, 50%, 70%) to understand robustness.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.analysis.kalshi.longshot_yes_maker_backtest import LongshotYesMakerBacktest
from src.common.analysis import Analysis, AnalysisOutput
from src.common.interfaces.chart import ChartConfig, ChartType, UnitType


class MakerSensitivityAnalysis(Analysis):
    """Run maker backtest with different fill rates."""

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
    ):
        super().__init__(
            name="maker_sensitivity_analysis",
            description="Sensitivity analysis for maker strategy with different fill rates",
        )
        self.trades_dir = trades_dir
        self.markets_dir = markets_dir

    def run(self) -> AnalysisOutput:
        """Run backtests with different fill rates."""
        fill_rates = [0.30, 0.50, 0.70]
        results = []

        for fill_rate in fill_rates:
            print(f"\nRunning backtest with {fill_rate:.0%} fill rate...")
            
            backtest = LongshotYesMakerBacktest(
                trades_dir=self.trades_dir,
                markets_dir=self.markets_dir,
                fill_rate=fill_rate,
                random_seed=42,  # Same seed for consistency
            )
            
            output = backtest.run()
            metrics = output.data.iloc[0].to_dict()
            
            results.append({
                "fill_rate": fill_rate * 100,
                "final_capital": metrics["final_capital"],
                "total_return_pct": metrics["total_return_pct"],
                "total_pnl": metrics["total_pnl"],
                "n_trades": metrics["n_trades"],
                "win_rate": metrics["win_rate"],
                "avg_return_pct": metrics["avg_return_pct"],
                "sharpe_ratio": metrics["sharpe_ratio"],
                "max_drawdown_pct": metrics["max_drawdown_pct"],
            })
        
        results_df = pd.DataFrame(results)
        
        fig = self._create_figure(results_df)
        chart = self._create_chart(results_df)
        
        return AnalysisOutput(figure=fig, data=results_df, chart=chart)

    def _create_figure(self, df: pd.DataFrame) -> plt.Figure:
        """Create visualization of sensitivity analysis."""
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle("Maker Strategy Sensitivity Analysis: Fill Rate Impact", 
                     fontsize=14, fontweight="bold")
        
        # 1. Total Return vs Fill Rate
        ax = axes[0, 0]
        ax.plot(df["fill_rate"], df["total_return_pct"], marker="o", linewidth=2, color="#2ecc71")
        ax.set_xlabel("Fill Rate (%)")
        ax.set_ylabel("Total Return (%)")
        ax.set_title("Total Return vs Fill Rate")
        ax.grid(True, alpha=0.3)
        
        # 2. Final Capital vs Fill Rate
        ax = axes[0, 1]
        ax.plot(df["fill_rate"], df["final_capital"], marker="o", linewidth=2, color="#3498db")
        ax.axhline(y=10000, color="gray", linestyle="--", label="Starting Capital")
        ax.set_xlabel("Fill Rate (%)")
        ax.set_ylabel("Final Capital ($)")
        ax.set_title("Final Capital vs Fill Rate")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Number of Trades vs Fill Rate
        ax = axes[0, 2]
        ax.plot(df["fill_rate"], df["n_trades"], marker="o", linewidth=2, color="#e74c3c")
        ax.set_xlabel("Fill Rate (%)")
        ax.set_ylabel("Trades Completed")
        ax.set_title("Trades Completed vs Fill Rate")
        ax.grid(True, alpha=0.3)
        
        # 4. Win Rate vs Fill Rate
        ax = axes[1, 0]
        ax.plot(df["fill_rate"], df["win_rate"], marker="o", linewidth=2, color="#9b59b6")
        ax.set_xlabel("Fill Rate (%)")
        ax.set_ylabel("Win Rate (%)")
        ax.set_title("Win Rate vs Fill Rate")
        ax.grid(True, alpha=0.3)
        
        # 5. Sharpe Ratio vs Fill Rate
        ax = axes[1, 1]
        ax.plot(df["fill_rate"], df["sharpe_ratio"], marker="o", linewidth=2, color="#f39c12")
        ax.set_xlabel("Fill Rate (%)")
        ax.set_ylabel("Sharpe Ratio")
        ax.set_title("Sharpe Ratio vs Fill Rate")
        ax.grid(True, alpha=0.3)
        
        # 6. Max Drawdown vs Fill Rate
        ax = axes[1, 2]
        ax.plot(df["fill_rate"], df["max_drawdown_pct"], marker="o", linewidth=2, color="#e74c3c")
        ax.set_xlabel("Fill Rate (%)")
        ax.set_ylabel("Max Drawdown (%)")
        ax.set_title("Max Drawdown vs Fill Rate")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

    def _create_chart(self, df: pd.DataFrame) -> ChartConfig:
        """Create chart configuration."""
        chart_data = [
            {
                "fill_rate": row["fill_rate"],
                "total_return": row["total_return_pct"],
                "final_capital": row["final_capital"],
            }
            for _, row in df.iterrows()
        ]
        
        return ChartConfig(
            type=ChartType.LINE,
            data=chart_data,
            xKey="fill_rate",
            yKeys=["total_return"],
            title="Total Return vs Fill Rate",
            xLabel="Fill Rate (%)",
            yLabel="Total Return (%)",
        )
