"""Opportunity scorer that evaluates trading signals using historical data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import duckdb
import pandas as pd
from pathlib import Path

from src.trading.scanner import TradingOpportunity


@dataclass
class ScoredOpportunity:
    """Trading opportunity with detailed scoring."""

    opportunity: TradingOpportunity
    confidence_score: float  # 0-100, higher is better
    historical_win_rate: float
    historical_avg_return: float
    edge_estimate: float  # Expected profit per dollar risked
    time_to_expiry_hours: Optional[float]
    volume_score: float  # 0-100 based on liquidity
    category_score: float  # 0-100 based on category performance
    
    # Risk metrics
    kelly_fraction: float  # Optimal bet size per Kelly criterion
    recommended_position_size: float  # As fraction of bankroll
    
    def __repr__(self) -> str:
        return (
            f"ScoredOpportunity({self.opportunity.ticker}, "
            f"score={self.confidence_score:.1f}, "
            f"win_rate={self.historical_win_rate:.1%}, "
            f"edge={self.edge_estimate:.2%})"
        )


class OpportunityScorer:
    """Scores trading opportunities for Longshot NO Buyer strategy.
    
    Uses portfolio backtest results to estimate win rates, returns, and optimal
    position sizes for Top 5 profitable categories only.
    """

    # Historical win rates by price (from backtest - Top 5 categories)
    HISTORICAL_WIN_RATES = {
        1: 0.9958,
        2: 0.9891,
        3: 0.9815,
        4: 0.9755,
        5: 0.9661,
        6: 0.9602,
        7: 0.9523,
        8: 0.9399,
        9: 0.9329,
        10: 0.9222,
    }

    # Historical average returns by price (from portfolio backtest)
    HISTORICAL_RETURNS = {
        1: 0.0058,
        2: 0.0093,
        3: 0.0118,
        4: 0.0161,
        5: 0.0169,
        6: 0.0215,
        7: 0.0240,
        8: 0.0216,
        9: 0.0251,
        10: 0.0246,
    }

    # Category expected returns (from category profitability analysis)
    # Only Top 5 profitable categories - all others return None (filtered by scanner)
    CATEGORY_EXPECTED_RETURNS = {
        "Science/Tech": 0.0345,    # +3.45%
        "Esports": 0.0229,         # +2.29%
        "Entertainment": 0.0217,   # +2.17%
        "Media": 0.0180,           # +1.80%
        "Politics": 0.0095,        # +0.95%
    }

    # Category multipliers for scoring (relative to Politics baseline)
    CATEGORY_MULTIPLIERS = {
        "Science/Tech": 3.63,   # 3.45% / 0.95%
        "Esports": 2.41,        # 2.29% / 0.95%
        "Entertainment": 2.28,  # 2.17% / 0.95%
        "Media": 1.89,          # 1.80% / 0.95%
        "Politics": 1.0,        # Baseline
    }

    def __init__(
        self,
        max_kelly_fraction: float = 0.25,
        min_confidence_score: float = 50.0,
        min_volume_score: float = 30.0,
    ):
        """Initialize the opportunity scorer.
        
        Args:
            max_kelly_fraction: Maximum Kelly fraction to recommend (default: 25%)
            min_confidence_score: Minimum confidence score to consider trading
            min_volume_score: Minimum volume score to consider trading
        """
        self.max_kelly_fraction = max_kelly_fraction
        self.min_confidence_score = min_confidence_score
        self.min_volume_score = min_volume_score

    def _calculate_kelly_fraction(
        self, 
        win_prob: float, 
        win_amount: float, 
        loss_amount: float
    ) -> float:
        """Calculate Kelly criterion optimal bet size.
        
        Formula: f* = (p * b - q) / b
        where p = win probability, q = 1-p, b = win_amount / loss_amount
        
        Args:
            win_prob: Probability of winning
            win_amount: Amount won if successful
            loss_amount: Amount lost if unsuccessful
            
        Returns:
            Optimal fraction of bankroll to bet
        """
        if loss_amount == 0:
            return 0.0
        
        b = win_amount / loss_amount
        q = 1 - win_prob
        
        kelly = (win_prob * b - q) / b
        
        # Apply safety cap
        return min(max(kelly, 0), self.max_kelly_fraction)

    def _calculate_volume_score(self, opportunity: TradingOpportunity) -> float:
        """Calculate liquidity/volume score (0-100).
        
        Higher score for markets with more volume and open interest.
        """
        # Score based on volume
        volume_score = min(100, (opportunity.volume / 1000) * 10)
        
        # Score based on open interest
        oi_score = min(100, (opportunity.open_interest / 100) * 10)
        
        # Weighted average (60% volume, 40% open interest)
        return volume_score * 0.6 + oi_score * 0.4

    def _calculate_time_score(self, opportunity: TradingOpportunity) -> float:
        """Calculate time-to-expiry score (0-100).
        
        Prefer markets with sufficient time to resolve (not too close to expiry).
        """
        if opportunity.close_time is None:
            return 50.0  # Neutral if unknown
        
        time_to_expiry = (opportunity.close_time - datetime.now()).total_seconds() / 3600  # hours
        
        if time_to_expiry < 1:
            return 10.0  # Very low score for markets closing soon
        elif time_to_expiry < 24:
            return 30.0 + (time_to_expiry / 24) * 40  # 30-70 for 0-24 hours
        elif time_to_expiry < 168:  # 1 week
            return 70.0 + (time_to_expiry / 168) * 20  # 70-90 for 1-7 days
        else:
            return 90.0  # High score for long-dated contracts

    def score_opportunity(
        self, 
        opportunity: TradingOpportunity,
        bankroll: float = 1000.0
    ) -> Optional[ScoredOpportunity]:
        """Score a trading opportunity.
        
        Args:
            opportunity: TradingOpportunity to score
            bankroll: Total bankroll for position sizing
            
        Returns:
            ScoredOpportunity with detailed scoring, or None if category not in Top 5
        """
        # Only score Top 5 categories (scanner should filter, but double-check)
        if opportunity.group not in self.CATEGORY_EXPECTED_RETURNS:
            return None
        
        # Get historical metrics for this price level
        yes_price = opportunity.yes_bid
        historical_win_rate = self.HISTORICAL_WIN_RATES.get(yes_price, 0.95)
        historical_return = self.HISTORICAL_RETURNS.get(yes_price, 0.015)
        
        # Apply category multiplier
        category_multiplier = self.CATEGORY_MULTIPLIERS.get(opportunity.group, 1.0)
        adjusted_return = historical_return * category_multiplier
        adjusted_win_rate = historical_win_rate + (1 - historical_win_rate) * (category_multiplier - 1) * 0.1
        adjusted_win_rate = min(adjusted_win_rate, 0.999)  # Cap at 99.9%
        
        # Calculate edge (expected profit per dollar risked)
        # If we risk X cents to win (100-X) cents
        cost = opportunity.our_entry_price
        win_amount = 100 - cost
        edge = (adjusted_win_rate * win_amount - (1 - adjusted_win_rate) * cost) / cost
        
        # Calculate Kelly fraction
        kelly_fraction = self._calculate_kelly_fraction(
            adjusted_win_rate,
            win_amount,
            cost
        )
        
        # Calculate component scores
        volume_score = self._calculate_volume_score(opportunity)
        time_score = self._calculate_time_score(opportunity)
        category_score = category_multiplier * 50  # 0-100 scale
        
        # Calculate time to expiry
        time_to_expiry_hours = None
        if opportunity.close_time:
            time_to_expiry_hours = (opportunity.close_time - datetime.now()).total_seconds() / 3600
        
        # Overall confidence score (weighted average)
        confidence_score = (
            adjusted_win_rate * 40 +  # 40% weight on win rate
            (edge + 0.05) * 200 +  # 30% weight on edge (scaled)
            volume_score * 0.15 +  # 15% weight on volume
            time_score * 0.15  # 15% weight on time
        )
        confidence_score = min(max(confidence_score, 0), 100)
        
        # Recommended position size (capped Kelly fraction)
        recommended_size = kelly_fraction * bankroll
        
        return ScoredOpportunity(
            opportunity=opportunity,
            confidence_score=confidence_score,
            historical_win_rate=adjusted_win_rate,
            historical_avg_return=adjusted_return,
            edge_estimate=edge,
            time_to_expiry_hours=time_to_expiry_hours,
            volume_score=volume_score,
            category_score=category_score,
            kelly_fraction=kelly_fraction,
            recommended_position_size=recommended_size,
        )

    def score_opportunities(
        self,
        opportunities: list[TradingOpportunity],
        bankroll: float = 1000.0,
    ) -> list[ScoredOpportunity]:
        """Score multiple opportunities and return sorted by confidence.
        
        Args:
            opportunities: List of opportunities to score
            bankroll: Total bankroll for position sizing
            
        Returns:
            List of ScoredOpportunity sorted by confidence score
        """
        scored = [self.score_opportunity(opp, bankroll) for opp in opportunities if self.score_opportunity(opp, bankroll) is not None]
        
        # Filter by minimum thresholds
        scored = [
            s for s in scored
            if s.confidence_score >= self.min_confidence_score
            and s.volume_score >= self.min_volume_score
        ]
        
        # Sort by confidence score (highest first)
        scored.sort(key=lambda x: x.confidence_score, reverse=True)
        
        return scored

    def get_tradeable_opportunities(
        self,
        opportunities: list[TradingOpportunity],
        bankroll: float = 1000.0,
        max_opportunities: int = 10,
    ) -> list[ScoredOpportunity]:
        """Get the best tradeable opportunities.
        
        Args:
            opportunities: List of opportunities to evaluate
            bankroll: Total bankroll
            max_opportunities: Maximum number to return
            
        Returns:
            Top opportunities that meet trading criteria
        """
        scored = self.score_opportunities(opportunities, bankroll)
        return scored[:max_opportunities]
