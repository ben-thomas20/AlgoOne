"""Simple dashboard for monitoring trading strategy performance."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from src.trading.order_manager import OrderManager
from src.trading.position_manager import PositionManager


class TradingDashboard:
    """Dashboard for tracking and visualizing trading performance."""

    def __init__(
        self,
        position_manager: PositionManager,
        order_manager: OrderManager,
        output_dir: Path | str = "output/trading",
    ):
        """Initialize dashboard.
        
        Args:
            position_manager: Position manager instance
            order_manager: Order manager instance
            output_dir: Directory to save dashboard outputs
        """
        self.position_manager = position_manager
        self.order_manager = order_manager
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_performance_summary(self) -> dict:
        """Generate performance summary metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        portfolio_stats = self.position_manager.get_portfolio_stats()
        order_stats = self.order_manager.get_order_stats()

        # Calculate additional metrics
        total_value = portfolio_stats["current_bankroll"] + portfolio_stats["total_exposure"]
        
        # Win rate from closed positions
        closed_positions = self.position_manager.closed_positions
        if closed_positions:
            # Count positions with positive PnL
            winning_positions = sum(
                1 for pos in closed_positions
                if (pos.current_value or 0) - pos.total_cost > 0
            )
            win_rate = winning_positions / len(closed_positions)
        else:
            win_rate = 0.0

        # Fill rate
        total_orders = order_stats["total_orders"]
        filled_orders = order_stats["filled_orders"]
        fill_rate = (filled_orders / total_orders * 100) if total_orders > 0 else 0

        return {
            "timestamp": datetime.now().isoformat(),
            "portfolio": portfolio_stats,
            "orders": order_stats,
            "total_value": total_value,
            "win_rate_pct": win_rate * 100,
            "fill_rate_pct": fill_rate,
        }

    def save_performance_log(self, filename: str = "performance_log.jsonl"):
        """Append current performance to log file.
        
        Args:
            filename: Name of log file
        """
        log_path = self.output_dir / filename
        summary = self.generate_performance_summary()

        # Append to JSONL file
        with open(log_path, "a") as f:
            f.write(json.dumps(summary) + "\n")

        print(f"Performance logged to {log_path}")

    def export_positions_csv(self, filename: str = "positions.csv"):
        """Export current positions to CSV.
        
        Args:
            filename: Name of CSV file
        """
        positions = self.position_manager.get_position_summary()
        
        if not positions:
            print("No positions to export")
            return

        df = pd.DataFrame(positions)
        csv_path = self.output_dir / filename
        df.to_csv(csv_path, index=False)
        print(f"Positions exported to {csv_path}")

    def export_orders_csv(self, filename: str = "orders.csv"):
        """Export order history to CSV.
        
        Args:
            filename: Name of CSV file
        """
        orders = [
            {
                "order_id": order.order_id,
                "ticker": order.ticker,
                "side": order.side.value,
                "quantity": order.quantity,
                "limit_price": order.limit_price,
                "status": order.status.value,
                "filled_quantity": order.filled_quantity,
                "avg_fill_price": order.avg_fill_price,
                "created_at": order.created_at.isoformat(),
                "updated_at": order.updated_at.isoformat(),
                "notes": order.notes,
            }
            for order in self.order_manager.orders.values()
        ]

        if not orders:
            print("No orders to export")
            return

        df = pd.DataFrame(orders)
        csv_path = self.output_dir / filename
        df.to_csv(csv_path, index=False)
        print(f"Orders exported to {csv_path}")

    def print_dashboard(self):
        """Print dashboard to console."""
        summary = self.generate_performance_summary()
        portfolio = summary["portfolio"]
        orders = summary["orders"]

        print("\n" + "=" * 70)
        print(" " * 20 + "TRADING DASHBOARD")
        print("=" * 70)
        print(f"Timestamp: {summary['timestamp']}")
        print()

        print("PORTFOLIO METRICS:")
        print(f"  Initial Bankroll:    ${portfolio['initial_bankroll']:>12,.2f}")
        print(f"  Current Bankroll:    ${portfolio['current_bankroll']:>12,.2f}")
        print(f"  Total Exposure:      ${portfolio['total_exposure']:>12,.2f} ({portfolio['exposure_pct']:.1f}%)")
        print(f"  Total Value:         ${summary['total_value']:>12,.2f}")
        print()

        print("POSITIONS:")
        print(f"  Open:                {portfolio['open_positions']:>12,}")
        print(f"  Closed:              {portfolio['closed_positions']:>12,}")
        print()

        print("PERFORMANCE:")
        print(f"  Realized P&L:        ${portfolio['realized_pnl']:>12,.2f}")
        print(f"  Unrealized P&L:      ${portfolio['unrealized_pnl']:>12,.2f}")
        print(f"  Total P&L:           ${portfolio['total_pnl']:>12,.2f}")
        print(f"  Total Return:        {portfolio['total_return_pct']:>12.2f}%")
        print(f"  Win Rate:            {summary['win_rate_pct']:>12.1f}%")
        print()

        print("ORDER STATISTICS:")
        print(f"  Total Orders:        {orders['total_orders']:>12,}")
        print(f"  Filled:              {orders['filled_orders']:>12,}")
        print(f"  Open:                {orders['open_orders']:>12,}")
        print(f"  Cancelled:           {orders['cancelled_orders']:>12,}")
        print(f"  Fill Rate:           {summary['fill_rate_pct']:>12.1f}%")
        print(f"  Capital Deployed:    ${orders['total_capital_deployed']:>12,.2f}")
        print()

        # Print open positions
        positions = self.position_manager.get_position_summary()
        if positions:
            print("OPEN POSITIONS:")
            print(f"  {'Ticker':<15} {'Side':<5} {'Qty':>8} {'Avg Price':>10} {'Cost':>12} {'Value':>12} {'P&L':>12}")
            print("  " + "-" * 85)
            for pos in positions:
                pnl_str = f"${pos['pnl']:.2f}" if pos['pnl'] is not None else "N/A"
                value_str = f"${pos['current_value']:.2f}" if pos['current_value'] is not None else "N/A"
                print(f"  {pos['ticker']:<15} {pos['side']:<5} {pos['quantity']:>8,} "
                      f"{pos['avg_price']:>9.1f}¢ ${pos['cost']:>10.2f} {value_str:>12} {pnl_str:>12}")

        print("=" * 70)

    def export_all(self):
        """Export all dashboard data."""
        self.save_performance_log()
        self.export_positions_csv()
        self.export_orders_csv()
        print(f"\nAll data exported to {self.output_dir}/")
