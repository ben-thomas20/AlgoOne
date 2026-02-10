"""Portfolio-based backtest for Longshot YES Maker strategy.

This analysis simulates a realistic MAKER strategy where we post limit orders to
sell YES at longshot prices (1-10¢) and wait for takers to hit our orders.

Key difference from previous backtests:
- We act as MAKER (provide liquidity via limit orders)
- Research shows makers earn +1.12% on average, takers lose -1.12%
- We capture the "Optimism Tax" by selling YES to biased takers

Strategy:
- Post limit orders to sell YES at 1-10¢ (equivalent to buying NO at 90-99¢)
- Target Top 6 categories with highest maker-taker gaps
- Use 50% fill rate assumption (conservative from research)
- Apply Kelly criterion position sizing with risk limits
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.util.categories import CATEGORY_SQL, GROUP_COLORS, get_group
from src.common.analysis import Analysis, AnalysisOutput
from src.common.interfaces.chart import ChartConfig, ChartType, UnitType


@dataclass
class MakerPosition:
    """Represents an open maker position (we sold YES, bought NO)."""
    ticker: str
    category: str
    entry_time: datetime
    yes_price_sold: int  # Price at which we sold YES (1-10¢)
    cost_per_contract: float  # Our cost = (100 - yes_price) / 100
    quantity: int
    total_cost: float
    close_time: datetime
    market_result: str  # 'yes' or 'no'


@dataclass
class CompletedTrade:
    """Represents a completed maker trade."""
    ticker: str
    category: str
    entry_time: datetime
    exit_time: datetime
    yes_price_sold: int
    cost_per_contract: float
    quantity: int
    total_cost: float
    pnl: float
    return_pct: float
    won: bool


class LongshotYesMakerBacktest(Analysis):
    """Backtest maker strategy posting limit orders to sell YES at longshots."""

    # Top 6 categories by maker-taker gap (from Becker research)
    TOP_6_CATEGORIES = {
        "Sports",        # +2.23pp gap, 60.4% volume
        "Politics",      # +1.02pp gap, 6.8% volume
        "Crypto",        # +2.69pp gap, 9.3% volume
        "Entertainment", # +4.79pp gap, 2.1% volume
        "Media",         # +7.28pp gap, 0.8% volume
        "World Events",  # +7.32pp gap, 0.3% volume
    }

    # Historical maker win rates when buying NO (from research)
    # These are HIGHER than taker win rates, which is the maker edge
    MAKER_WIN_RATES_BUYING_NO = {
        1: 0.9957,   # Selling YES at 1¢ = buying NO at 99¢
        2: 0.9911,
        3: 0.9835,
        4: 0.9776,
        5: 0.9691,
        6: 0.9638,
        7: 0.9567,
        8: 0.9455,
        9: 0.9391,
        10: 0.9290,
    }

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
        starting_capital: float = 10000.0,
        min_yes_price: int = 1,
        max_yes_price: int = 10,
        fill_rate: float = 0.50,
        max_position_pct: float = 0.10,
        max_category_pct: float = 0.30,
        max_total_exposure_pct: float = 0.80,
        kelly_fraction_cap: float = 0.25,
        random_seed: int = 42,
    ):
        super().__init__(
            name="longshot_yes_maker_backtest",
            description="Portfolio backtest for Longshot YES Maker strategy with $10k capital",
        )
        base_dir = Path(__file__).parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base_dir / "data" / "kalshi" / "trades")
        self.markets_dir = Path(markets_dir or base_dir / "data" / "kalshi" / "markets")
        
        self.starting_capital = starting_capital
        self.min_yes_price = min_yes_price
        self.max_yes_price = max_yes_price
        self.fill_rate = fill_rate
        self.max_position_pct = max_position_pct
        self.max_category_pct = max_category_pct
        self.max_total_exposure_pct = max_total_exposure_pct
        self.kelly_fraction_cap = kelly_fraction_cap
        
        # Set random seed for reproducibility
        np.random.seed(random_seed)

    def run(self) -> AnalysisOutput:
        """Execute maker strategy backtest."""
        with self.progress("Loading market opportunities"):
            opportunities = self._load_opportunities()
            print(f"  Found {len(opportunities):,} market opportunities in Top 6 categories")

        with self.progress("Simulating maker strategy"):
            portfolio_state = self._simulate_maker_strategy(opportunities)

        with self.progress("Calculating performance metrics"):
            metrics = self._calculate_metrics(portfolio_state)

        fig = self._create_figure(portfolio_state, metrics)
        chart = self._create_chart(portfolio_state)

        # Export data
        export_data = pd.DataFrame([metrics])
        
        return AnalysisOutput(figure=fig, data=export_data, chart=chart)

    def _load_opportunities(self) -> pd.DataFrame:
        """Load markets where we could post maker orders to sell YES."""
        con = duckdb.connect()
        
        # Find best price per market in 1-10¢ range where takers bought YES
        df = con.execute(
            f"""
            WITH market_opportunities AS (
                SELECT 
                    ticker,
                    MIN(created_time) as first_opportunity_time,
                    MAX(yes_price) as best_yes_price
                FROM '{self.trades_dir}/*.parquet'
                WHERE yes_price BETWEEN {self.min_yes_price} AND {self.max_yes_price}
                  AND taker_side = 'yes'
                GROUP BY ticker
            ),
            resolved_markets AS (
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
            )
            SELECT
                mo.ticker,
                mo.best_yes_price,
                mo.first_opportunity_time,
                rm.result,
                rm.close_time,
                mc.category
            FROM market_opportunities mo
            INNER JOIN resolved_markets rm ON mo.ticker = rm.ticker
            LEFT JOIN market_categories mc ON mo.ticker = mc.ticker
            ORDER BY mo.first_opportunity_time
            """
        ).df()
        
        df["group"] = df["category"].apply(get_group)
        
        # Filter to Top 6 categories
        df = df[df["group"].isin(self.TOP_6_CATEGORIES)]
        
        return df

    def _simulate_maker_strategy(self, opportunities: pd.DataFrame) -> dict:
        """Simulate posting limit orders to sell YES and waiting for fills."""
        capital = self.starting_capital
        positions: dict[str, MakerPosition] = {}
        completed_trades: list[CompletedTrade] = []
        capital_history: list[tuple[datetime, float, float]] = []
        
        opportunities_considered = 0
        orders_posted = 0
        fills_received = 0
        
        for idx, row in opportunities.iterrows():
            current_time = row["first_opportunity_time"]
            
            # Close any positions whose markets have resolved
            positions_to_close = []
            for ticker, pos in positions.items():
                if current_time >= pos.close_time:
                    positions_to_close.append(ticker)
            
            for ticker in positions_to_close:
                pos = positions[ticker]
                
                # We sold YES (bought NO), so we win if result is 'no'
                won = (pos.market_result == "no")
                
                if won:
                    payout = pos.quantity * 1.00  # $1 per contract
                    pnl = payout - pos.total_cost
                else:
                    payout = 0
                    pnl = -pos.total_cost
                
                capital += payout
                return_pct = (pnl / pos.total_cost) * 100 if pos.total_cost > 0 else 0
                
                completed_trades.append(CompletedTrade(
                    ticker=pos.ticker,
                    category=pos.category,
                    entry_time=pos.entry_time,
                    exit_time=pos.close_time,
                    yes_price_sold=pos.yes_price_sold,
                    cost_per_contract=pos.cost_per_contract,
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
            
            # Consider posting a maker order for this market
            if row["ticker"] in positions:
                continue  # Already have position in this market
            
            opportunities_considered += 1
            
            # Simulate fill: 50% of maker orders get filled
            if np.random.random() > self.fill_rate:
                continue
            
            orders_posted += 1
            
            yes_price = row["best_yes_price"]
            cost_per_contract = (100 - yes_price) / 100.0  # We buy NO at this price
            category = row["group"]
            
            # Calculate Kelly position size
            win_prob = self.MAKER_WIN_RATES_BUYING_NO.get(yes_price, 0.95)
            win_amount = 1.0 - cost_per_contract  # Profit if we win
            loss_amount = cost_per_contract  # Loss if we lose
            
            if loss_amount > 0:
                b = win_amount / loss_amount
                kelly = (win_prob * b - (1 - win_prob)) / b
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
            
            # Enter position as MAKER
            quantity = int(position_size / cost_per_contract)
            if quantity == 0:
                continue
            
            total_cost = quantity * cost_per_contract
            
            positions[row["ticker"]] = MakerPosition(
                ticker=row["ticker"],
                category=category,
                entry_time=current_time,
                yes_price_sold=yes_price,
                cost_per_contract=cost_per_contract,
                quantity=quantity,
                total_cost=total_cost,
                close_time=row["close_time"],
                market_result=row["result"],
            )
            
            capital -= total_cost
            fills_received += 1
        
        # Close any remaining open positions
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
            
            completed_trades.append(CompletedTrade(
                ticker=pos.ticker,
                category=pos.category,
                entry_time=pos.entry_time,
                exit_time=pos.close_time,
                yes_price_sold=pos.yes_price_sold,
                cost_per_contract=pos.cost_per_contract,
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
            "orders_posted": orders_posted,
            "fills_received": fills_received,
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
                "orders_posted": portfolio_state["orders_posted"],
                "fills_received": portfolio_state["fills_received"],
                "fill_rate_pct": 0,
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
        category_trades = {}
        for t in completed_trades:
            if t.category not in category_pnl:
                category_pnl[t.category] = 0
                category_trades[t.category] = 0
            category_pnl[t.category] += t.pnl
            category_trades[t.category] += 1
        
        actual_fill_rate = (portfolio_state["fills_received"] / portfolio_state["orders_posted"]) * 100 if portfolio_state["orders_posted"] > 0 else 0
        
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
            "orders_posted": portfolio_state["orders_posted"],
            "fills_received": portfolio_state["fills_received"],
            "fill_rate_pct": actual_fill_rate,
            "category_pnl": category_pnl,
            "category_trades": category_trades,
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
            ax1.set_title("Portfolio Value Over Time (Maker Strategy)")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. Trade Returns Distribution
        ax2 = fig.add_subplot(gs[1, 0])
        if completed_trades:
            returns = [t.return_pct for t in completed_trades]
            ax2.hist(returns, bins=50, color="#3498db", alpha=0.7, edgecolor="black")
            ax2.axvline(x=0, color="red", linestyle="--", linewidth=2)
            ax2.axvline(x=np.mean(returns), color="green", linestyle="-", linewidth=2, label=f"Mean: {np.mean(returns):.2f}%")
            ax2.set_xlabel("Return (%)")
            ax2.set_ylabel("Number of Trades")
            ax2.set_title("Distribution of Trade Returns")
            ax2.legend()
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
MAKER STRATEGY BACKTEST SUMMARY
{'=' * 50}

Capital:
  Starting: ${metrics['starting_capital']:,.2f}
  Final: ${metrics['final_capital']:,.2f}
  Total Return: {metrics['total_return_pct']:.2f}%
  Total P&L: ${metrics['total_pnl']:,.2f}

Trading (Maker Orders):
  Opportunities: {metrics['opportunities_considered']:,}
  Orders Posted: {metrics['orders_posted']:,}
  Fills Received: {metrics['fills_received']:,}
  Fill Rate: {metrics['fill_rate_pct']:.1f}%

Performance:
  Trades Completed: {metrics['n_trades']:,}
  Win Rate: {metrics['win_rate']:.2f}%
  Avg Return/Trade: {metrics['avg_return_pct']:.2f}%
  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
  Max Drawdown: {metrics['max_drawdown_pct']:.2f}%

Strategy:
  Role: MAKER (sell YES at longshots)
  Price Range: {self.min_yes_price}-{self.max_yes_price}¢
  Categories: Top 6 (High Maker-Taker Gap)
  Position Sizing: Kelly ({self.kelly_fraction_cap:.0%} cap)
  Fill Rate Assumed: {self.fill_rate:.0%}
        """
        ax5.text(0.1, 0.5, summary_text, fontsize=10, family="monospace",
                verticalalignment="center")
        
        fig.suptitle("Longshot YES Maker Strategy - Portfolio Backtest ($10,000 Starting Capital)", 
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
            title="Portfolio Value Over Time (Maker Strategy)",
            yUnit=UnitType.DOLLARS,
            xLabel="Date",
            yLabel="Portfolio Value ($)",
        )
