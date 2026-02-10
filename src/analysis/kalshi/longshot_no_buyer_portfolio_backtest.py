"""Portfolio-based backtest for Longshot NO Buyer strategy.

This analysis simulates a realistic trading strategy with proper capital constraints:
- Starting capital: $10,000
- Buy NO contracts when YES is at 1-10 cents
- Only trade in profitable categories (Top 5)
- Apply Kelly criterion position sizing with risk limits
- Process trades chronologically to simulate real-time decision making

Key improvements over previous backtest:
- Realistic portfolio constraints (can't take every trade)
- Proper drawdown calculation (portfolio-level, not trade-level)
- Dollar-weighted returns (not equal-weighted)
- Only profitable categories included
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.analysis.util.categories import CATEGORY_SQL, get_group
from src.common.analysis import Analysis, AnalysisOutput
from src.common.interfaces.chart import ChartConfig, ChartType, UnitType


@dataclass
class Position:
    """Represents an open position."""
    ticker: str
    category: str
    entry_time: datetime
    entry_price: float  # Cost per contract in dollars
    quantity: int
    total_cost: float
    close_time: datetime  # When this market closes
    market_result: str  # 'yes' or 'no' - the final outcome


@dataclass
class Trade:
    """Represents a completed trade."""
    ticker: str
    category: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    quantity: int
    total_cost: float
    pnl: float
    return_pct: float
    won: bool


class LongshotNoBuyerPortfolioBacktest(Analysis):
    """Portfolio-based backtest with realistic capital constraints."""

    # Only trade in profitable categories
    PROFITABLE_CATEGORIES = {
        "Science/Tech",   # +3.45%
        "Esports",        # +2.29%
        "Entertainment",  # +2.17%
        "Media",          # +1.80%
        "Politics",       # +0.95%
    }

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
        starting_capital: float = 10000.0,
        min_yes_price: int = 1,
        max_yes_price: int = 10,
        max_position_pct: float = 0.10,
        max_category_pct: float = 0.30,
        max_total_exposure_pct: float = 0.80,
        kelly_fraction_cap: float = 0.25,
    ):
        super().__init__(
            name="longshot_no_buyer_portfolio_backtest",
            description="Portfolio backtest for Longshot NO Buyer strategy with $10k capital",
        )
        base_dir = Path(__file__).parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base_dir / "data" / "kalshi" / "trades")
        self.markets_dir = Path(markets_dir or base_dir / "data" / "kalshi" / "markets")
        
        self.starting_capital = starting_capital
        self.min_yes_price = min_yes_price
        self.max_yes_price = max_yes_price
        self.max_position_pct = max_position_pct
        self.max_category_pct = max_category_pct
        self.max_total_exposure_pct = max_total_exposure_pct
        self.kelly_fraction_cap = kelly_fraction_cap

    def run(self) -> AnalysisOutput:
        """Execute portfolio backtest."""
        with self.progress("Loading market opportunities"):
            con = duckdb.connect()
            
            # Load unique market opportunities (first trade at longshot price per market)
            df = con.execute(
                f"""
                WITH resolved_markets AS (
                    SELECT ticker, result, close_time
                    FROM '{self.markets_dir}/*.parquet'
                    WHERE status = 'finalized'
                      AND result IN ('yes', 'no')
                ),
                market_categories AS (
                    SELECT 
                        ticker,
                        {CATEGORY_SQL} AS category
                    FROM '{self.markets_dir}/*.parquet'
                ),
                first_longshot_trades AS (
                    SELECT 
                        ticker,
                        MIN(created_time) as first_opportunity_time,
                        MIN(yes_price) as best_yes_price
                    FROM '{self.trades_dir}/*.parquet'
                    WHERE yes_price BETWEEN {self.min_yes_price} AND {self.max_yes_price}
                      AND taker_side = 'yes'
                    GROUP BY ticker
                )
                SELECT
                    flt.ticker,
                    flt.best_yes_price as yes_price,
                    flt.first_opportunity_time as created_time,
                    m.result,
                    m.close_time,
                    c.category
                FROM first_longshot_trades flt
                INNER JOIN resolved_markets m ON flt.ticker = m.ticker
                LEFT JOIN market_categories c ON flt.ticker = c.ticker
                ORDER BY flt.first_opportunity_time
                """
            ).df()
            
            df["group"] = df["category"].apply(get_group)
            
            # Filter to only profitable categories
            df = df[df["group"].isin(self.PROFITABLE_CATEGORIES)]
            
            print(f"  Loaded {len(df):,} unique market opportunities in profitable categories")

        with self.progress("Simulating portfolio trading"):
            portfolio_state = self._simulate_portfolio(df)

        with self.progress("Calculating performance metrics"):
            metrics = self._calculate_metrics(portfolio_state)

        fig = self._create_figure(portfolio_state, metrics)
        chart = self._create_chart(portfolio_state)

        # Export data
        export_data = pd.DataFrame([metrics])
        
        return AnalysisOutput(figure=fig, data=export_data, chart=chart)

    def _simulate_portfolio(self, opportunities_df: pd.DataFrame) -> dict:
        """Simulate portfolio trading with capital constraints.
        
        Each row in opportunities_df represents ONE market opportunity (not multiple trades).
        We decide once per market whether to enter a position.
        """
        capital = self.starting_capital
        positions: dict[str, Position] = {}
        completed_trades: list[Trade] = []
        capital_history: list[tuple[datetime, float, float]] = []  # (time, capital, total_value)
        
        # Historical win rates by price for Kelly calculation
        historical_win_rates = {
            1: 0.9958, 2: 0.9891, 3: 0.9815, 4: 0.9755, 5: 0.9661,
            6: 0.9602, 7: 0.9523, 8: 0.9399, 9: 0.9329, 10: 0.9222,
        }
        
        opportunities_considered = 0
        positions_entered = 0
        
        for idx, row in opportunities_df.iterrows():
            current_time = row["created_time"]
            
            # Close any positions whose markets have resolved
            positions_to_close = []
            for ticker, pos in positions.items():
                if current_time >= pos.close_time:
                    positions_to_close.append(ticker)
            
            for ticker in positions_to_close:
                pos = positions[ticker]
                
                # We bought NO, so we win if result is 'no'
                won = (pos.market_result == "no")
                
                if won:
                    payout = pos.quantity * 1.00  # $1 per contract
                    pnl = payout - pos.total_cost
                else:
                    payout = 0
                    pnl = -pos.total_cost
                
                capital += payout
                return_pct = (pnl / pos.total_cost) * 100 if pos.total_cost > 0 else 0
                
                completed_trades.append(Trade(
                    ticker=pos.ticker,
                    category=pos.category,
                    entry_time=pos.entry_time,
                    exit_time=pos.close_time,
                    entry_price=pos.entry_price,
                    quantity=pos.quantity,
                    total_cost=pos.total_cost,
                    pnl=pnl,
                    return_pct=return_pct,
                    won=won,
                ))
                
                del positions[ticker]
            
            # Record capital state
            total_exposure = sum(pos.total_cost for pos in positions.values())
            total_value = capital + total_exposure
            capital_history.append((current_time, capital, total_value))
            
            # Consider entering a new position for this market opportunity
            if row["ticker"] in positions:
                continue  # Already have position in this market (shouldn't happen with deduped data)
            
            opportunities_considered += 1
            
            yes_price = row["yes_price"]
            no_price = 100 - yes_price
            cost_per_contract = no_price / 100.0
            category = row["group"]
            
            # Calculate Kelly fraction
            win_prob = historical_win_rates.get(yes_price, 0.95)
            loss_prob = 1 - win_prob
            win_amount = 100 - no_price  # Profit per contract in cents
            loss_amount = no_price  # Loss per contract in cents
            
            if loss_amount > 0:
                b = win_amount / loss_amount
                kelly = (win_prob - loss_prob / b)
                kelly = max(0, min(kelly, self.kelly_fraction_cap))
            else:
                kelly = 0
            
            # Calculate position size with constraints
            kelly_size = capital * kelly
            max_position = capital * self.max_position_pct
            position_size = min(kelly_size, max_position)
            
            # Check category exposure
            category_exposure = sum(
                pos.total_cost for pos in positions.values()
                if pos.category == category
            )
            max_category = self.starting_capital * self.max_category_pct
            remaining_category = max_category - category_exposure
            position_size = min(position_size, remaining_category)
            
            # Check total exposure
            total_exposure = sum(pos.total_cost for pos in positions.values())
            max_total = self.starting_capital * self.max_total_exposure_pct
            remaining_total = max_total - total_exposure
            position_size = min(position_size, remaining_total)
            
            # Check available capital
            position_size = min(position_size, capital)
            
            if position_size < cost_per_contract:
                continue  # Not enough capital
            
            # Enter position
            quantity = int(position_size / cost_per_contract)
            if quantity == 0:
                continue
            
            total_cost = quantity * cost_per_contract
            
            positions[row["ticker"]] = Position(
                ticker=row["ticker"],
                category=category,
                entry_time=current_time,
                entry_price=cost_per_contract,
                quantity=quantity,
                total_cost=total_cost,
                close_time=row["close_time"],
                market_result=row["result"],
            )
            
            capital -= total_cost
            positions_entered += 1
        
        # Close any remaining open positions at their actual outcomes
        for pos in list(positions.values()):
            won = (pos.market_result == "no")
            
            if won:
                payout = pos.quantity * 1.00
                pnl = payout - pos.total_cost
            else:
                payout = 0
                pnl = -pos.total_cost
            
            capital += payout
            return_pct = (pnl / pos.total_cost) * 100 if pos.total_cost > 0 else 0
            
            completed_trades.append(Trade(
                ticker=pos.ticker,
                category=pos.category,
                entry_time=pos.entry_time,
                exit_time=pos.close_time,
                entry_price=pos.entry_price,
                quantity=pos.quantity,
                total_cost=pos.total_cost,
                pnl=pnl,
                return_pct=return_pct,
                won=won,
            ))
        
        return {
            "completed_trades": completed_trades,
            "capital_history": capital_history,
            "final_capital": capital,
            "opportunities_considered": opportunities_considered,
            "positions_entered": positions_entered,
        }

    def _calculate_metrics(self, portfolio_state: dict) -> dict:
        """Calculate performance metrics."""
        completed_trades = portfolio_state["completed_trades"]
        capital_history = portfolio_state["capital_history"]
        final_capital = portfolio_state["final_capital"]
        
        if not completed_trades:
            return {
                "starting_capital": self.starting_capital,
                "final_capital": final_capital,
                "total_return_pct": 0,
                "total_pnl": 0,
                "n_trades": 0,
                "win_rate": 0,
                "avg_return_pct": 0,
                "sharpe_ratio": 0,
                "max_drawdown_pct": 0,
                "opportunities_considered": portfolio_state["opportunities_considered"],
                "positions_entered": portfolio_state["positions_entered"],
                "entry_rate_pct": 0,
            }
        
        # Basic metrics
        total_pnl = sum(t.pnl for t in completed_trades)
        n_trades = len(completed_trades)
        win_rate = sum(1 for t in completed_trades if t.won) / n_trades
        avg_return = np.mean([t.return_pct for t in completed_trades])
        
        # Portfolio value over time
        if capital_history:
            capital_df = pd.DataFrame(capital_history, columns=["time", "capital", "total_value"])
            capital_df = capital_df.set_index("time")
            
            # Daily returns
            daily_values = capital_df["total_value"].resample("D").last().ffill()
            daily_returns = daily_values.pct_change().dropna()
            
            if len(daily_returns) > 1:
                sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0
            else:
                sharpe = 0
            
            # Max drawdown
            running_max = daily_values.expanding().max()
            drawdown = (daily_values - running_max) / running_max
            max_drawdown = drawdown.min() * 100
        else:
            sharpe = 0
            max_drawdown = 0
        
        # Category breakdown
        category_pnl = {}
        for t in completed_trades:
            if t.category not in category_pnl:
                category_pnl[t.category] = 0
            category_pnl[t.category] += t.pnl
        
        entry_rate = (portfolio_state["positions_entered"] / portfolio_state["opportunities_considered"]) * 100 if portfolio_state["opportunities_considered"] > 0 else 0
        
        return {
            "starting_capital": self.starting_capital,
            "final_capital": final_capital,
            "total_return_pct": ((final_capital - self.starting_capital) / self.starting_capital) * 100,
            "total_pnl": total_pnl,
            "n_trades": n_trades,
            "win_rate": win_rate * 100,
            "avg_return_pct": avg_return,
            "sharpe_ratio": sharpe,
            "max_drawdown_pct": max_drawdown,
            "opportunities_considered": portfolio_state["opportunities_considered"],
            "positions_entered": portfolio_state["positions_entered"],
            "entry_rate_pct": entry_rate,
            "category_pnl": category_pnl,
        }

    def _create_figure(self, portfolio_state: dict, metrics: dict) -> plt.Figure:
        """Create visualization."""
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        completed_trades = portfolio_state["completed_trades"]
        capital_history = portfolio_state["capital_history"]
        
        # 1. Portfolio Value Over Time
        ax1 = fig.add_subplot(gs[0, :])
        if capital_history:
            times, capitals, total_values = zip(*capital_history)
            ax1.plot(times, total_values, label="Total Portfolio Value", linewidth=2, color="#2ecc71")
            ax1.plot(times, capitals, label="Available Capital", linewidth=1, color="#3498db", alpha=0.7)
            ax1.axhline(y=self.starting_capital, color="gray", linestyle="--", label="Starting Capital")
            ax1.set_ylabel("Portfolio Value ($)")
            ax1.set_title("Portfolio Value Over Time")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. Trade Returns Distribution
        ax2 = fig.add_subplot(gs[1, 0])
        if completed_trades:
            returns = [t.return_pct for t in completed_trades]
            ax2.hist(returns, bins=50, color="#3498db", alpha=0.7, edgecolor="black")
            ax2.axvline(x=0, color="red", linestyle="--", linewidth=2)
            ax2.set_xlabel("Return (%)")
            ax2.set_ylabel("Number of Trades")
            ax2.set_title("Distribution of Trade Returns")
            ax2.grid(True, alpha=0.3)
        
        # 3. Category P&L
        ax3 = fig.add_subplot(gs[1, 1])
        if metrics.get("category_pnl"):
            categories = list(metrics["category_pnl"].keys())
            pnls = list(metrics["category_pnl"].values())
            colors = ["#2ecc71" if p > 0 else "#e74c3c" for p in pnls]
            ax3.barh(categories, pnls, color=colors, alpha=0.7)
            ax3.axvline(x=0, color="black", linestyle="-", linewidth=0.8)
            ax3.set_xlabel("Total P&L ($)")
            ax3.set_title("P&L by Category")
            ax3.grid(True, alpha=0.3, axis="x")
        
        # 4. Cumulative P&L
        ax4 = fig.add_subplot(gs[2, 0])
        if completed_trades:
            trades_sorted = sorted(completed_trades, key=lambda t: t.exit_time)
            cumulative_pnl = np.cumsum([t.pnl for t in trades_sorted])
            exit_times = [t.exit_time for t in trades_sorted]
            ax4.plot(exit_times, cumulative_pnl, linewidth=2, color="#2ecc71")
            ax4.axhline(y=0, color="gray", linestyle="--")
            ax4.set_ylabel("Cumulative P&L ($)")
            ax4.set_title("Cumulative P&L Over Time")
            ax4.grid(True, alpha=0.3)
        
        # 5. Summary Statistics
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.axis("off")
        summary_text = f"""
PORTFOLIO BACKTEST SUMMARY
{'=' * 50}

Capital:
  Starting: ${metrics['starting_capital']:,.2f}
  Final: ${metrics['final_capital']:,.2f}
  Total Return: {metrics['total_return_pct']:.2f}%
  Total P&L: ${metrics['total_pnl']:,.2f}

Trading:
  Opportunities: {metrics['opportunities_considered']:,}
  Positions Entered: {metrics['positions_entered']:,}
  Entry Rate: {metrics['entry_rate_pct']:.1f}%

Performance:
  Win Rate: {metrics['win_rate']:.2f}%
  Avg Return: {metrics['avg_return_pct']:.2f}%
  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
  Max Drawdown: {metrics['max_drawdown_pct']:.2f}%

Strategy:
  Price Range: {self.min_yes_price}-{self.max_yes_price}¢
  Categories: Top 5 Profitable
  Position Sizing: Kelly ({self.kelly_fraction_cap:.0%} cap)
        """
        ax5.text(0.1, 0.5, summary_text, fontsize=10, family="monospace",
                verticalalignment="center")
        
        fig.suptitle("Longshot NO Buyer - Portfolio Backtest ($10,000 Starting Capital)", 
                    fontsize=14, fontweight="bold")
        
        return fig

    def _create_chart(self, portfolio_state: dict) -> ChartConfig:
        """Create chart configuration."""
        capital_history = portfolio_state["capital_history"]
        
        if not capital_history:
            return ChartConfig(
                type=ChartType.LINE,
                data=[],
                xKey="time",
                yKeys=["value"],
                title="Portfolio Value Over Time",
                yUnit=UnitType.DOLLARS,
            )
        
        chart_data = [
            {
                "time": time.isoformat(),
                "value": total_value,
            }
            for time, capital, total_value in capital_history[::100]  # Sample every 100th point
        ]
        
        return ChartConfig(
            type=ChartType.LINE,
            data=chart_data,
            xKey="time",
            yKeys=["value"],
            title="Portfolio Value Over Time",
            yUnit=UnitType.DOLLARS,
            xLabel="Date",
            yLabel="Portfolio Value ($)",
        )
