"""Market scanner for identifying longshot YES selling opportunities on Kalshi."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from src.analysis.util.categories import get_group
from src.indexers.kalshi.client import KalshiClient
from src.indexers.kalshi.models import Market


@dataclass
class TradingOpportunity:
    """Represents a potential trading opportunity."""

    ticker: str
    title: str
    category: str
    group: str
    yes_bid: int  # cents
    yes_ask: int  # cents
    no_bid: int  # cents
    no_ask: int  # cents
    volume: int
    open_interest: int
    close_time: Optional[datetime]
    
    # Strategy-specific fields
    our_entry_price: int  # Price we'd pay to buy NO (= 100 - yes_bid)
    implied_probability: float  # = our_entry_price / 100
    expected_return: float  # Historical expected return for this category/price
    
    def __repr__(self) -> str:
        return (
            f"TradingOpportunity(ticker={self.ticker}, "
            f"category={self.group}, yes_bid={self.yes_bid}¢, "
            f"our_cost={self.our_entry_price}¢, expected_return={self.expected_return:.2%})"
        )


class MarketScanner:
    """Scans Kalshi markets for Longshot NO Buyer opportunities.
    
    Strategy: Buy NO contracts when YES is priced at 1-10 cents (longshot),
    in only the Top 5 profitable categories identified through backtesting.
    This filters out unprofitable categories like Sports, Crypto, and Weather.
    """

    # Top 5 profitable categories (from portfolio backtest and category analysis)
    INCLUDED_CATEGORIES = {
        "Science/Tech",   # +3.45%
        "Esports",        # +2.29%
        "Entertainment",  # +2.17%
        "Media",          # +1.80%
        "Politics",       # +0.95%
    }

    # Historical expected returns by category (from portfolio backtest)
    CATEGORY_EXPECTED_RETURNS = {
        "Science/Tech": 0.0345,
        "Esports": 0.0229,
        "Entertainment": 0.0217,
        "Media": 0.0180,
        "Politics": 0.0095,
    }

    def __init__(
        self,
        client: KalshiClient,
        min_yes_price: int = 1,
        max_yes_price: int = 10,
        min_volume: int = 100,
        min_open_interest: int = 10,
    ):
        """Initialize the market scanner.
        
        Args:
            client: Kalshi API client
            min_yes_price: Minimum YES price in cents (default: 1)
            max_yes_price: Maximum YES price in cents (default: 10)
            min_volume: Minimum total volume to consider market liquid
            min_open_interest: Minimum open interest to consider market active
        """
        self.client = client
        self.min_yes_price = min_yes_price
        self.max_yes_price = max_yes_price
        self.min_volume = min_volume
        self.min_open_interest = min_open_interest

    def _extract_category(self, market: Market) -> str:
        """Extract category from event ticker."""
        if not market.event_ticker:
            return "independent"
        
        # Extract prefix (e.g., "NFLGAME" from "NFLGAME-2024-...")
        match = re.match(r"^([A-Z0-9]+)", market.event_ticker)
        if match:
            return match.group(1)
        return "independent"

    def _is_valid_opportunity(self, market: Market, category: str, group: str) -> bool:
        """Check if market meets criteria for trading opportunity."""
        # Must have valid pricing
        if market.yes_bid is None or market.yes_ask is None:
            return False
        
        # YES must be at longshot price (1-10 cents)
        if not (self.min_yes_price <= market.yes_bid <= self.max_yes_price):
            return False
        
        # Must be open for trading
        if market.status != "open":
            return False
        
        # Exclude low-liquidity markets
        if market.volume < self.min_volume:
            return False
        
        if market.open_interest < self.min_open_interest:
            return False
        
        # Only include Top 5 profitable categories
        if group not in self.INCLUDED_CATEGORIES:
            return False
        
        return True

    def scan_markets(self, limit: int = 200) -> list[TradingOpportunity]:
        """Scan current markets and return valid trading opportunities.
        
        Args:
            limit: Maximum number of markets to fetch per API call
            
        Returns:
            List of TradingOpportunity objects sorted by expected return
        """
        opportunities = []
        
        # Fetch markets from Kalshi
        markets = self.client.list_all_markets(limit=limit)
        
        for market in markets:
            category = self._extract_category(market)
            group = get_group(category)
            
            if not self._is_valid_opportunity(market, category, group):
                continue
            
            # Calculate our entry price (we buy NO at 100 - yes_bid)
            our_entry_price = 100 - market.yes_bid
            implied_prob = our_entry_price / 100.0
            
            # Get expected return for this category (only Top 5 should pass validation)
            expected_return = self.CATEGORY_EXPECTED_RETURNS.get(group, 0.0)
            
            opportunity = TradingOpportunity(
                ticker=market.ticker,
                title=market.title,
                category=category,
                group=group,
                yes_bid=market.yes_bid,
                yes_ask=market.yes_ask,
                no_bid=market.no_bid,
                no_ask=market.no_ask,
                volume=market.volume,
                open_interest=market.open_interest,
                close_time=market.close_time,
                our_entry_price=our_entry_price,
                implied_probability=implied_prob,
                expected_return=expected_return,
            )
            
            opportunities.append(opportunity)
        
        # Sort by expected return (highest first)
        opportunities.sort(key=lambda x: x.expected_return, reverse=True)
        
        return opportunities

    def scan_single_market(self, ticker: str) -> Optional[TradingOpportunity]:
        """Scan a single market by ticker.
        
        Args:
            ticker: Market ticker to scan
            
        Returns:
            TradingOpportunity if valid, None otherwise
        """
        try:
            market = self.client.get_market(ticker)
            category = self._extract_category(market)
            group = get_group(category)
            
            if not self._is_valid_opportunity(market, category, group):
                return None
            
            our_entry_price = 100 - market.yes_bid
            implied_prob = our_entry_price / 100.0
            expected_return = self.CATEGORY_EXPECTED_RETURNS.get(group, 0.0)
            
            return TradingOpportunity(
                ticker=market.ticker,
                title=market.title,
                category=category,
                group=group,
                yes_bid=market.yes_bid,
                yes_ask=market.yes_ask,
                no_bid=market.no_bid,
                no_ask=market.no_ask,
                volume=market.volume,
                open_interest=market.open_interest,
                close_time=market.close_time,
                our_entry_price=our_entry_price,
                implied_probability=implied_prob,
                expected_return=expected_return,
            )
        except Exception as e:
            print(f"Error scanning market {ticker}: {e}")
            return None

    def get_top_opportunities(self, n: int = 10) -> list[TradingOpportunity]:
        """Get the top N trading opportunities by expected return.
        
        Args:
            n: Number of top opportunities to return
            
        Returns:
            List of top N opportunities
        """
        opportunities = self.scan_markets()
        return opportunities[:n]
