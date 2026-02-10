"""Main trading strategy implementation: Longshot NO Buyer.

This module orchestrates the scanner, scorer, order manager, and position manager
to implement the complete trading strategy focused on Top 5 profitable categories.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Optional

from src.indexers.kalshi.client import KalshiClient
from src.trading.order_manager import OrderManager, OrderStatus
from src.trading.position_manager import PositionManager
from src.trading.scanner import MarketScanner, TradingOpportunity
from src.trading.scorer import OpportunityScorer, ScoredOpportunity


class LongshotNoBuyerStrategy:
    """Implements the Longshot NO Buyer trading strategy.
    
    Strategy Overview:
    1. Scan markets for YES contracts at 1-10 cent prices
    2. Filter to Top 5 profitable categories only (Science/Tech, Esports, Entertainment, Media, Politics)
    3. Score opportunities using historical win rates and expected returns
    4. Buy NO contracts (equivalent to fading the longshot YES)
    5. Size positions using Kelly criterion with risk limits
    6. Monitor and manage positions
    """

    def __init__(
        self,
        client: KalshiClient,
        initial_bankroll: float = 1000.0,
        paper_trading: bool = True,
        min_confidence_score: float = 60.0,
        max_positions: int = 20,
    ):
        """Initialize the trading strategy.
        
        Args:
            client: Kalshi API client
            initial_bankroll: Starting capital in USD
            paper_trading: If True, simulate trades without executing
            min_confidence_score: Minimum confidence score to trade
            max_positions: Maximum number of concurrent positions
        """
        self.client = client
        self.initial_bankroll = initial_bankroll
        self.paper_trading = paper_trading
        self.min_confidence_score = min_confidence_score
        self.max_positions = max_positions

        # Initialize components
        self.scanner = MarketScanner(client)
        self.scorer = OpportunityScorer(min_confidence_score=min_confidence_score)
        self.order_manager = OrderManager(paper_trading=paper_trading)
        self.position_manager = PositionManager(initial_bankroll=initial_bankroll)

        # State tracking
        self.is_running = False
        self.last_scan_time: Optional[datetime] = None
        self.scan_count = 0
        self.trades_executed = 0

    def scan_and_score_opportunities(self) -> list[ScoredOpportunity]:
        """Scan markets and score trading opportunities.
        
        Returns:
            List of scored opportunities, sorted by confidence
        """
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Scanning markets...")
        
        # Scan for opportunities
        opportunities = self.scanner.scan_markets()
        print(f"  Found {len(opportunities)} raw opportunities")

        # Score opportunities
        scored = self.scorer.score_opportunities(
            opportunities,
            bankroll=self.position_manager.current_bankroll,
        )
        print(f"  {len(scored)} opportunities passed filters (score >= {self.min_confidence_score})")

        self.last_scan_time = datetime.now()
        self.scan_count += 1

        return scored

    def execute_opportunity(
        self,
        scored_opp: ScoredOpportunity,
    ) -> bool:
        """Execute a trading opportunity.
        
        Args:
            scored_opp: Scored opportunity to trade
            
        Returns:
            True if trade was executed
        """
        opp = scored_opp.opportunity

        # Check if we already have a position in this market
        if self.position_manager.get_position(opp.ticker):
            print(f"  ⏭️  Skipping {opp.ticker}: already have position")
            return False

        # Calculate position size
        price_per_contract = opp.our_entry_price / 100.0  # Convert cents to USD
        num_contracts = self.position_manager.calculate_position_size(
            kelly_fraction=scored_opp.kelly_fraction,
            price_per_contract=price_per_contract,
            category=opp.group,
            categories_map={opp.ticker: opp.group},
        )

        if num_contracts == 0:
            print(f"  ⏭️  Skipping {opp.ticker}: position size = 0 (risk limits)")
            return False

        # Check risk limits
        proposed_cost = num_contracts * price_per_contract
        allowed, reason = self.position_manager.check_risk_limits(proposed_cost, opp.group)
        if not allowed:
            print(f"  ❌ Cannot trade {opp.ticker}: {reason}")
            return False

        # Create and submit order
        order = self.order_manager.create_longshot_yes_seller_order(
            ticker=opp.ticker,
            yes_price=opp.yes_bid,
            quantity=num_contracts,
            notes=f"Score: {scored_opp.confidence_score:.1f}, Edge: {scored_opp.edge_estimate:.2%}",
        )

        success = self.order_manager.submit_order(order)

        if success:
            print(f"  ✅ Placed order: {opp.ticker} - {num_contracts} contracts @ {opp.our_entry_price}¢")
            print(f"      Cost: ${proposed_cost:.2f}, Expected return: {scored_opp.edge_estimate:.2%}")
            
            # For paper trading, simulate immediate fill
            if self.paper_trading:
                fill = self.order_manager.simulate_fill(
                    order_id=order.order_id,
                    quantity=num_contracts,
                    price=opp.our_entry_price,
                )
                if fill:
                    self.position_manager.add_position_from_fill(fill, category=opp.group)
                    self.trades_executed += 1

            return True

        return False

    def run_single_scan(self) -> dict:
        """Run a single scan cycle and execute trades.
        
        Returns:
            Dictionary with scan statistics
        """
        start_time = time.time()

        # Check if we're at max positions
        if len(self.position_manager.positions) >= self.max_positions:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] At max positions ({self.max_positions})")
            return {
                "opportunities_found": 0,
                "trades_executed": 0,
                "reason": "at_max_positions",
            }

        # Scan and score
        scored_opportunities = self.scan_and_score_opportunities()

        if not scored_opportunities:
            print("  No tradeable opportunities found")
            return {
                "opportunities_found": 0,
                "trades_executed": 0,
                "reason": "no_opportunities",
            }

        # Execute top opportunities
        trades_executed = 0
        for scored_opp in scored_opportunities[:5]:  # Try top 5
            if len(self.position_manager.positions) >= self.max_positions:
                break

            if self.execute_opportunity(scored_opp):
                trades_executed += 1

        elapsed = time.time() - start_time
        print(f"  Scan completed in {elapsed:.1f}s, executed {trades_executed} trades")

        return {
            "opportunities_found": len(scored_opportunities),
            "trades_executed": trades_executed,
            "elapsed_seconds": elapsed,
            "reason": "success",
        }

    def get_status(self) -> dict:
        """Get current strategy status.
        
        Returns:
            Dictionary with strategy status
        """
        portfolio_stats = self.position_manager.get_portfolio_stats()
        order_stats = self.order_manager.get_order_stats()

        return {
            "is_running": self.is_running,
            "paper_trading": self.paper_trading,
            "scan_count": self.scan_count,
            "last_scan": self.last_scan_time.isoformat() if self.last_scan_time else None,
            "trades_executed": self.trades_executed,
            "portfolio": portfolio_stats,
            "orders": order_stats,
        }

    def print_status(self):
        """Print current strategy status."""
        status = self.get_status()
        portfolio = status["portfolio"]

        print("\n" + "=" * 60)
        print("LONGSHOT YES SELLER STRATEGY STATUS")
        print("=" * 60)
        print(f"Mode: {'PAPER TRADING' if self.paper_trading else 'LIVE TRADING'}")
        print(f"Scans: {status['scan_count']}")
        print(f"Trades: {status['trades_executed']}")
        print()
        print("PORTFOLIO:")
        print(f"  Bankroll: ${portfolio['current_bankroll']:.2f} (initial: ${portfolio['initial_bankroll']:.2f})")
        print(f"  Exposure: ${portfolio['total_exposure']:.2f} ({portfolio['exposure_pct']:.1f}%)")
        print(f"  Positions: {portfolio['open_positions']} open, {portfolio['closed_positions']} closed")
        print(f"  P&L: ${portfolio['total_pnl']:.2f} ({portfolio['total_return_pct']:.2f}%)")
        print(f"    Realized: ${portfolio['realized_pnl']:.2f}")
        print(f"    Unrealized: ${portfolio['unrealized_pnl']:.2f}")
        print("=" * 60)
