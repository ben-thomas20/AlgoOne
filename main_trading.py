#!/usr/bin/env python3
"""Main entry point for the Longshot NO Buyer trading strategy.

Usage:
    # Paper trading (simulate without real trades)
    uv run main_trading.py paper --bankroll 1000
    
    # Live trading (requires authentication)
    uv run main_trading.py live --bankroll 1000
    
    # Single scan (run once and exit)
    uv run main_trading.py scan
    
    # Check status
    uv run main_trading.py status
"""

import argparse
import sys
import time
from pathlib import Path

from src.indexers.kalshi.client import KalshiClient
from src.trading.strategy import LongshotNoBuyerStrategy


def paper_trading(args):
    """Run strategy in paper trading mode."""
    print("=" * 60)
    print("LONGSHOT NO BUYER - PAPER TRADING (Top 5 Categories)")
    print("=" * 60)
    print(f"Initial Bankroll: ${args.bankroll}")
    print(f"Scan Interval: {args.interval}s")
    print(f"Max Positions: {args.max_positions}")
    print("=" * 60)
    print()

    client = KalshiClient()
    strategy = LongshotNoBuyerStrategy(
        client=client,
        initial_bankroll=args.bankroll,
        paper_trading=True,
        min_confidence_score=args.min_score,
        max_positions=args.max_positions,
    )

    strategy.is_running = True
    scan_num = 0

    try:
        while scan_num < args.max_scans or args.max_scans == 0:
            scan_num += 1
            print(f"\n--- Scan #{scan_num} ---")
            
            result = strategy.run_single_scan()
            
            # Print status every 5 scans
            if scan_num % 5 == 0:
                strategy.print_status()

            # Sleep before next scan
            if scan_num < args.max_scans or args.max_scans == 0:
                print(f"\nSleeping for {args.interval}s...\n")
                time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        strategy.is_running = False
        print("\n" + "=" * 60)
        print("FINAL STATUS")
        strategy.print_status()
        
        # Print position summary
        positions = strategy.position_manager.get_position_summary()
        if positions:
            print("\nOPEN POSITIONS:")
            for pos in positions:
                print(f"  {pos['ticker']}: {pos['side']} {pos['quantity']}@{pos['avg_price']:.1f}¢ "
                      f"(cost: ${pos['cost']:.2f})")


def single_scan(args):
    """Run a single scan and exit."""
    print("Running single scan...")
    
    client = KalshiClient()
    strategy = LongshotNoBuyerStrategy(
        client=client,
        initial_bankroll=args.bankroll,
        paper_trading=True,
        min_confidence_score=args.min_score,
        max_positions=args.max_positions,
    )

    result = strategy.run_single_scan()
    strategy.print_status()


def check_status(args):
    """Check strategy status (placeholder for persistent strategies)."""
    print("Status checking requires persistent strategy instance.")
    print("This feature is not yet implemented.")
    print()
    print("To run the strategy, use:")
    print("  uv run main_trading.py paper --bankroll 1000")


def live_trading(args):
    """Run strategy in live trading mode."""
    print("=" * 60)
    print("LIVE TRADING NOT IMPLEMENTED")
    print("=" * 60)
    print()
    print("Live trading requires:")
    print("  1. Kalshi account with API access")
    print("  2. API authentication (email + password or API key)")
    print("  3. Order placement endpoints in KalshiClient")
    print()
    print("The current KalshiClient only supports read-only operations.")
    print("To enable live trading:")
    print("  1. Add authentication to src/indexers/kalshi/client.py")
    print("  2. Implement order placement methods")
    print("  3. Set paper_trading=False in strategy initialization")
    print()
    print("For now, use paper trading to test the strategy:")
    print("  uv run main_trading.py paper --bankroll 1000")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Longshot YES Seller Trading Strategy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Paper trading command
    paper_parser = subparsers.add_parser("paper", help="Run in paper trading mode")
    paper_parser.add_argument(
        "--bankroll",
        type=float,
        default=1000.0,
        help="Initial bankroll in USD (default: 1000)",
    )
    paper_parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Seconds between scans (default: 300)",
    )
    paper_parser.add_argument(
        "--max-scans",
        type=int,
        default=0,
        help="Maximum number of scans (0 = unlimited, default: 0)",
    )
    paper_parser.add_argument(
        "--max-positions",
        type=int,
        default=20,
        help="Maximum concurrent positions (default: 20)",
    )
    paper_parser.add_argument(
        "--min-score",
        type=float,
        default=60.0,
        help="Minimum confidence score to trade (default: 60)",
    )

    # Live trading command
    live_parser = subparsers.add_parser("live", help="Run in live trading mode")
    live_parser.add_argument("--bankroll", type=float, default=1000.0)
    live_parser.add_argument("--interval", type=int, default=300)
    live_parser.add_argument("--max-positions", type=int, default=20)
    live_parser.add_argument("--min-score", type=float, default=60.0)

    # Single scan command
    scan_parser = subparsers.add_parser("scan", help="Run a single scan")
    scan_parser.add_argument("--bankroll", type=float, default=1000.0)
    scan_parser.add_argument("--max-positions", type=int, default=20)
    scan_parser.add_argument("--min-score", type=float, default=60.0)

    # Status command
    status_parser = subparsers.add_parser("status", help="Check strategy status")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "paper":
        paper_trading(args)
    elif args.command == "live":
        live_trading(args)
    elif args.command == "scan":
        single_scan(args)
    elif args.command == "status":
        check_status(args)


if __name__ == "__main__":
    main()
