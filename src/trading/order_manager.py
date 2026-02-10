"""Order manager for executing trades on Kalshi.

NOTE: This is a framework for order management. The Kalshi API client in this
codebase is read-only (no authentication). To execute real trades, you would need to:
1. Add authentication to the Kalshi client
2. Implement order placement endpoints
3. Add order status tracking
4. Handle order fills and cancellations

For now, this provides the structure for paper trading and future live trading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import uuid4


class OrderSide(Enum):
    """Side of the order (what we're buying)."""

    YES = "yes"
    NO = "no"


class OrderType(Enum):
    """Type of order."""

    LIMIT = "limit"
    MARKET = "market"


class OrderStatus(Enum):
    """Order status."""

    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Represents a trading order."""

    order_id: str
    ticker: str
    side: OrderSide  # YES or NO
    order_type: OrderType
    quantity: int  # Number of contracts
    limit_price: Optional[int]  # Price in cents (for limit orders)
    status: OrderStatus
    created_at: datetime
    updated_at: datetime
    filled_quantity: int = 0
    avg_fill_price: Optional[float] = None
    notes: str = ""

    def __repr__(self) -> str:
        return (
            f"Order({self.order_id[:8]}, {self.ticker}, "
            f"{self.side.value} {self.quantity}@{self.limit_price}¢, "
            f"status={self.status.value})"
        )


@dataclass
class Fill:
    """Represents an order fill."""

    fill_id: str
    order_id: str
    ticker: str
    side: OrderSide
    quantity: int
    price: int  # cents
    timestamp: datetime
    total_cost: float  # USD

    def __repr__(self) -> str:
        return f"Fill({self.ticker}, {self.side.value} {self.quantity}@{self.price}¢)"


class OrderManager:
    """Manages order lifecycle for the trading strategy.
    
    Strategy-specific: Places MAKER orders to sell YES (buy NO) at longshot prices.
    When YES is trading at 5 cents, we place a limit order to sell YES at 5 cents,
    which is equivalent to buying NO at 95 cents.
    """

    def __init__(self, paper_trading: bool = True):
        """Initialize order manager.
        
        Args:
            paper_trading: If True, simulate orders without executing
        """
        self.paper_trading = paper_trading
        self.orders: dict[str, Order] = {}
        self.fills: dict[str, Fill] = {}

    def create_longshot_yes_seller_order(
        self,
        ticker: str,
        yes_price: int,
        quantity: int,
        notes: str = "",
    ) -> Order:
        """Create a limit order to sell YES at the current bid price.
        
        This is the core of the longshot YES seller strategy. We place a MAKER
        order to sell YES at the bid, which means we're buying NO at (100 - yes_price).
        
        Args:
            ticker: Market ticker
            yes_price: Current YES bid price in cents
            quantity: Number of contracts to trade
            notes: Optional notes about the order
            
        Returns:
            Created Order object
        """
        # We're selling YES, which means we're taking the NO position
        # In Kalshi terms, we buy NO contracts at (100 - yes_price)
        order_id = str(uuid4())
        now = datetime.now()

        order = Order(
            order_id=order_id,
            ticker=ticker,
            side=OrderSide.NO,  # We're buying NO
            order_type=OrderType.LIMIT,
            quantity=quantity,
            limit_price=100 - yes_price,  # Our entry price for NO
            status=OrderStatus.PENDING,
            created_at=now,
            updated_at=now,
            notes=f"Longshot YES Seller: {notes}",
        )

        self.orders[order_id] = order
        return order

    def submit_order(self, order: Order) -> bool:
        """Submit an order to the exchange.
        
        Args:
            order: Order to submit
            
        Returns:
            True if order was submitted successfully
        """
        if self.paper_trading:
            # Paper trading: just mark as open
            order.status = OrderStatus.OPEN
            order.updated_at = datetime.now()
            print(f"[PAPER TRADING] Submitted order: {order}")
            return True
        else:
            # Live trading would call Kalshi API here
            # This requires authentication and order placement endpoints
            raise NotImplementedError(
                "Live trading requires Kalshi API authentication. "
                "Set paper_trading=True for simulation."
            )

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order.
        
        Args:
            order_id: ID of order to cancel
            
        Returns:
            True if order was cancelled successfully
        """
        if order_id not in self.orders:
            return False

        order = self.orders[order_id]
        
        if order.status not in [OrderStatus.OPEN, OrderStatus.PENDING]:
            return False

        if self.paper_trading:
            order.status = OrderStatus.CANCELLED
            order.updated_at = datetime.now()
            print(f"[PAPER TRADING] Cancelled order: {order}")
            return True
        else:
            # Live trading would call Kalshi API here
            raise NotImplementedError("Live trading not implemented")

    def simulate_fill(
        self,
        order_id: str,
        quantity: int,
        price: int,
    ) -> Optional[Fill]:
        """Simulate an order fill (for paper trading).
        
        Args:
            order_id: Order to fill
            quantity: Quantity filled
            price: Fill price in cents
            
        Returns:
            Fill object if successful
        """
        if order_id not in self.orders:
            return None

        order = self.orders[order_id]
        
        if order.status not in [OrderStatus.OPEN, OrderStatus.PENDING]:
            return None

        fill_id = str(uuid4())
        fill = Fill(
            fill_id=fill_id,
            order_id=order_id,
            ticker=order.ticker,
            side=order.side,
            quantity=quantity,
            price=price,
            timestamp=datetime.now(),
            total_cost=(quantity * price) / 100.0,
        )

        self.fills[fill_id] = fill
        
        # Update order
        order.filled_quantity += quantity
        if order.avg_fill_price is None:
            order.avg_fill_price = float(price)
        else:
            # Weighted average
            total_filled = order.filled_quantity
            order.avg_fill_price = (
                (order.avg_fill_price * (total_filled - quantity) + price * quantity)
                / total_filled
            )

        if order.filled_quantity >= order.quantity:
            order.status = OrderStatus.FILLED
        else:
            order.status = OrderStatus.PARTIALLY_FILLED

        order.updated_at = datetime.now()

        print(f"[PAPER TRADING] Order filled: {fill}")
        return fill

    def get_open_orders(self) -> list[Order]:
        """Get all open orders.
        
        Returns:
            List of open orders
        """
        return [
            order for order in self.orders.values()
            if order.status in [OrderStatus.OPEN, OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED]
        ]

    def get_filled_orders(self) -> list[Order]:
        """Get all filled orders.
        
        Returns:
            List of filled orders
        """
        return [
            order for order in self.orders.values()
            if order.status == OrderStatus.FILLED
        ]

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get an order by ID.
        
        Args:
            order_id: Order ID
            
        Returns:
            Order if found
        """
        return self.orders.get(order_id)

    def get_fills_for_order(self, order_id: str) -> list[Fill]:
        """Get all fills for an order.
        
        Args:
            order_id: Order ID
            
        Returns:
            List of fills
        """
        return [fill for fill in self.fills.values() if fill.order_id == order_id]

    def get_total_capital_deployed(self) -> float:
        """Get total capital deployed across all filled orders.
        
        Returns:
            Total capital in USD
        """
        return sum(fill.total_cost for fill in self.fills.values())

    def get_order_stats(self) -> dict:
        """Get order statistics.
        
        Returns:
            Dictionary of order statistics
        """
        return {
            "total_orders": len(self.orders),
            "open_orders": len([o for o in self.orders.values() if o.status == OrderStatus.OPEN]),
            "filled_orders": len([o for o in self.orders.values() if o.status == OrderStatus.FILLED]),
            "cancelled_orders": len([o for o in self.orders.values() if o.status == OrderStatus.CANCELLED]),
            "total_fills": len(self.fills),
            "total_capital_deployed": self.get_total_capital_deployed(),
        }
