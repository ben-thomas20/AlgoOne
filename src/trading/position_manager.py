"""Position manager for tracking portfolio exposure and risk."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from src.trading.order_manager import Fill, Order, OrderSide, OrderStatus


@dataclass
class Position:
    """Represents a position in a market."""

    ticker: str
    side: OrderSide  # YES or NO
    quantity: int
    avg_entry_price: float  # cents
    total_cost: float  # USD
    current_value: Optional[float] = None  # USD (mark-to-market)
    unrealized_pnl: Optional[float] = None  # USD
    opened_at: datetime = field(default_factory=datetime.now)
    
    def __repr__(self) -> str:
        pnl_str = f", PnL=${self.unrealized_pnl:.2f}" if self.unrealized_pnl is not None else ""
        return (
            f"Position({self.ticker}, {self.side.value} {self.quantity}@{self.avg_entry_price:.1f}¢, "
            f"cost=${self.total_cost:.2f}{pnl_str})"
        )

    def update_market_value(self, current_price: float):
        """Update position's mark-to-market value.
        
        Args:
            current_price: Current market price for our side (in cents)
        """
        # Current value if we sold at current price
        # We own NO contracts, so we'd sell NO at current NO bid
        self.current_value = (self.quantity * current_price) / 100.0
        self.unrealized_pnl = self.current_value - self.total_cost


class PositionManager:
    """Manages positions and portfolio risk for the trading strategy.
    
    Responsibilities:
    - Track open positions
    - Calculate portfolio exposure
    - Apply Kelly criterion for position sizing
    - Enforce risk limits (max per market, category concentration)
    """

    def __init__(
        self,
        initial_bankroll: float,
        max_position_pct: float = 0.10,  # Max 10% per market
        max_category_pct: float = 0.30,  # Max 30% per category
        max_total_exposure_pct: float = 0.80,  # Max 80% total exposure
    ):
        """Initialize position manager.
        
        Args:
            initial_bankroll: Starting capital in USD
            max_position_pct: Maximum % of bankroll per market
            max_category_pct: Maximum % of bankroll per category
            max_total_exposure_pct: Maximum % of bankroll deployed
        """
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        self.max_position_pct = max_position_pct
        self.max_category_pct = max_category_pct
        self.max_total_exposure_pct = max_total_exposure_pct
        
        self.positions: dict[str, Position] = {}  # ticker -> Position
        self.closed_positions: list[Position] = []
        self.realized_pnl = 0.0

    def add_position_from_fill(
        self,
        fill: Fill,
        category: Optional[str] = None,
    ) -> Position:
        """Add or update a position from an order fill.
        
        Args:
            fill: Fill to add to positions
            category: Market category (for risk management)
            
        Returns:
            Updated Position
        """
        if fill.ticker in self.positions:
            # Update existing position
            pos = self.positions[fill.ticker]
            
            # Update average entry price (weighted average)
            total_quantity = pos.quantity + fill.quantity
            pos.avg_entry_price = (
                (pos.avg_entry_price * pos.quantity + fill.price * fill.quantity)
                / total_quantity
            )
            pos.quantity = total_quantity
            pos.total_cost += fill.total_cost
        else:
            # Create new position
            pos = Position(
                ticker=fill.ticker,
                side=fill.side,
                quantity=fill.quantity,
                avg_entry_price=float(fill.price),
                total_cost=fill.total_cost,
                opened_at=fill.timestamp,
            )
            self.positions[fill.ticker] = pos

        # Update bankroll
        self.current_bankroll -= fill.total_cost

        return pos

    def close_position(
        self,
        ticker: str,
        exit_price: float,
        closed_at: Optional[datetime] = None,
    ) -> Optional[float]:
        """Close a position and realize P&L.
        
        Args:
            ticker: Ticker to close
            exit_price: Exit price in cents
            closed_at: Timestamp of close
            
        Returns:
            Realized P&L in USD
        """
        if ticker not in self.positions:
            return None

        pos = self.positions[ticker]
        closed_at = closed_at or datetime.now()

        # Calculate realized P&L
        # If we own NO at 95¢ and market resolves NO, we get $1 per contract
        # Profit = $1 * quantity - total_cost
        if pos.side == OrderSide.NO:
            # Assuming we're closing because market resolved or we're exiting
            exit_value = (pos.quantity * exit_price) / 100.0
        else:
            exit_value = (pos.quantity * exit_price) / 100.0

        pnl = exit_value - pos.total_cost

        # Update state
        self.realized_pnl += pnl
        self.current_bankroll += exit_value
        self.closed_positions.append(pos)
        del self.positions[ticker]

        return pnl

    def get_position(self, ticker: str) -> Optional[Position]:
        """Get position for a ticker.
        
        Args:
            ticker: Market ticker
            
        Returns:
            Position if exists
        """
        return self.positions.get(ticker)

    def get_total_exposure(self) -> float:
        """Get total capital deployed across all positions.
        
        Returns:
            Total exposure in USD
        """
        return sum(pos.total_cost for pos in self.positions.values())

    def get_category_exposure(self, category: str, categories_map: dict[str, str]) -> float:
        """Get total exposure in a specific category.
        
        Args:
            category: Category name
            categories_map: Dict mapping tickers to categories
            
        Returns:
            Total exposure in that category (USD)
        """
        return sum(
            pos.total_cost
            for ticker, pos in self.positions.items()
            if categories_map.get(ticker) == category
        )

    def calculate_position_size(
        self,
        kelly_fraction: float,
        price_per_contract: float,
        category: Optional[str] = None,
        categories_map: Optional[dict[str, str]] = None,
    ) -> int:
        """Calculate optimal position size with risk constraints.
        
        Args:
            kelly_fraction: Kelly criterion optimal fraction
            price_per_contract: Cost per contract in USD
            category: Market category
            categories_map: Dict mapping tickers to categories
            
        Returns:
            Number of contracts to trade
        """
        # Kelly-suggested dollar amount
        kelly_amount = self.current_bankroll * kelly_fraction

        # Apply max position size limit
        max_position_amount = self.current_bankroll * self.max_position_pct
        position_amount = min(kelly_amount, max_position_amount)

        # Apply max total exposure limit
        current_exposure = self.get_total_exposure()
        max_total_exposure = self.initial_bankroll * self.max_total_exposure_pct
        remaining_capacity = max_total_exposure - current_exposure
        position_amount = min(position_amount, remaining_capacity)

        # Apply category concentration limit
        if category and categories_map:
            category_exposure = self.get_category_exposure(category, categories_map)
            max_category_exposure = self.initial_bankroll * self.max_category_pct
            remaining_category_capacity = max_category_exposure - category_exposure
            position_amount = min(position_amount, remaining_category_capacity)

        # Convert to number of contracts
        if position_amount <= 0 or price_per_contract <= 0:
            return 0

        num_contracts = int(position_amount / price_per_contract)
        return max(num_contracts, 0)

    def get_portfolio_stats(self) -> dict:
        """Get portfolio statistics.
        
        Returns:
            Dictionary of portfolio metrics
        """
        total_exposure = self.get_total_exposure()
        total_value = self.current_bankroll + total_exposure
        
        # Calculate unrealized P&L
        unrealized_pnl = sum(
            pos.unrealized_pnl for pos in self.positions.values()
            if pos.unrealized_pnl is not None
        )

        return {
            "initial_bankroll": self.initial_bankroll,
            "current_bankroll": self.current_bankroll,
            "total_exposure": total_exposure,
            "exposure_pct": (total_exposure / self.initial_bankroll) * 100,
            "open_positions": len(self.positions),
            "closed_positions": len(self.closed_positions),
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "total_pnl": self.realized_pnl + unrealized_pnl,
            "total_return_pct": ((total_value - self.initial_bankroll) / self.initial_bankroll) * 100,
        }

    def get_position_summary(self) -> list[dict]:
        """Get summary of all open positions.
        
        Returns:
            List of position dictionaries
        """
        return [
            {
                "ticker": pos.ticker,
                "side": pos.side.value,
                "quantity": pos.quantity,
                "avg_price": pos.avg_entry_price,
                "cost": pos.total_cost,
                "current_value": pos.current_value,
                "pnl": pos.unrealized_pnl,
            }
            for pos in self.positions.values()
        ]

    def check_risk_limits(self, proposed_cost: float, category: Optional[str] = None) -> tuple[bool, str]:
        """Check if a proposed trade violates risk limits.
        
        Args:
            proposed_cost: Cost of proposed trade in USD
            category: Category of the market
            
        Returns:
            Tuple of (allowed, reason)
        """
        # Check total exposure limit
        current_exposure = self.get_total_exposure()
        max_total = self.initial_bankroll * self.max_total_exposure_pct
        if current_exposure + proposed_cost > max_total:
            return False, f"Would exceed max total exposure ({self.max_total_exposure_pct:.0%})"

        # Check single position limit
        max_position = self.initial_bankroll * self.max_position_pct
        if proposed_cost > max_position:
            return False, f"Exceeds max position size ({self.max_position_pct:.0%})"

        # Check available capital
        if proposed_cost > self.current_bankroll:
            return False, "Insufficient bankroll"

        return True, "OK"
