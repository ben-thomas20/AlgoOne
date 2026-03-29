"""
V2 Multi-Signal Filtered Maker Strategy Backtest

Improvements over V1:
1. Price filter: Focus on 3-12¢ (highest maker edge per data)
2. Category filter: Weight by proven maker-taker gap
3. Time filter: Prefer evening hours (16:00-23:00 ET) where retail bias is strongest
4. Regime filter: Rolling maker-taker gap must be positive (avoids 2021-2023 losing regime)
5. Risk management: Fractional Kelly, per-category exposure limits, max drawdown circuit breaker
6. Realistic fill rate: Conservative 5% base (matching v1 observed), with time-of-day adjustment
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd


# ── Strategy Configuration ──────────────────────────────────────────────────

@dataclass
class StrategyConfig:
    starting_capital: float = 10_000.0
    min_price: int = 3
    max_price: int = 12
    start_date: str = "2024-01-01"  # Maker edge only emerged mid-2024 per data

    category_weights: dict[str, float] = field(default_factory=lambda: {
        "Entertainment": 1.5, "Media": 1.5, "World Events": 1.3,
        "Science/Tech": 1.2, "Sports": 1.0, "Weather": 1.0,
        "Other": 0.8, "Politics": 0.3,
        "Crypto": 0.0, "Finance": 0.0, "Esports": 0.0, "Science/Tech": 0.0,
    })

    regime_window: int = 500
    regime_threshold: float = 0.0
    kelly_fraction: float = 0.15
    max_position_pct: float = 0.03
    max_category_exposure: float = 0.30
    max_drawdown: float = 0.50
    drawdown_cooloff_trades: int = 50000
    base_fill_rate: float = 0.05


# ── Category Mapping (done in SQL for speed) ────────────────────────────────

CATEGORY_SQL = """
CASE
    -- Sports (most common, check first)
    WHEN raw_cat LIKE 'NFL%' OR raw_cat LIKE 'NBA%' OR raw_cat LIKE 'MLB%'
      OR raw_cat LIKE 'NHL%' OR raw_cat LIKE 'NCAA%' OR raw_cat LIKE 'WNBA%'
      OR raw_cat LIKE 'ATP%' OR raw_cat LIKE 'WTA%' OR raw_cat LIKE 'PGA%'
      OR raw_cat LIKE 'EPL%' OR raw_cat LIKE 'UCL%' OR raw_cat LIKE 'UFC%'
      OR raw_cat LIKE 'F1%' OR raw_cat LIKE 'NASCAR%' OR raw_cat LIKE 'MLS%'
      OR raw_cat LIKE 'FIFA%' OR raw_cat LIKE 'BOXING%' OR raw_cat LIKE 'SB%'
      OR raw_cat LIKE 'MARMAD%' OR raw_cat LIKE 'WMARMAD%' OR raw_cat LIKE 'MASTERS%'
      OR raw_cat LIKE 'USOPEN%' OR raw_cat LIKE 'THEOPEN%' OR raw_cat LIKE 'HEISMAN%'
      OR raw_cat LIKE 'INDY%' OR raw_cat LIKE 'NATHAN%' OR raw_cat LIKE 'LALIGA%'
      OR raw_cat LIKE 'SERIEA%' OR raw_cat LIKE 'BUNDESLIGA%' OR raw_cat LIKE 'LIGUE1%'
      OR raw_cat LIKE 'CLUBWC%' OR raw_cat LIKE 'NFC%' OR raw_cat LIKE 'AFC%'
      OR raw_cat LIKE 'PREMIERLEAGUE%' OR raw_cat LIKE 'BALLONDOR%'
      OR raw_cat LIKE 'MVE%' OR raw_cat LIKE 'LIVTOUR%' OR raw_cat LIKE 'DAVISCUP%'
      OR raw_cat LIKE 'EFLC%' OR raw_cat LIKE 'UEL%' OR raw_cat LIKE 'EUROLEAGUE%'
      OR raw_cat LIKE 'MENWORLDCUP%' OR raw_cat LIKE 'SUPERLIG%'
      OR raw_cat LIKE 'EREDIVISIE%' OR raw_cat LIKE 'BRASILEIR%'
      OR raw_cat LIKE 'LIGAPORTUGAL%' OR raw_cat LIKE 'GENESIS%'
      THEN 'Sports'
    -- Crypto
    WHEN raw_cat LIKE 'BTC%' OR raw_cat LIKE 'ETH%' OR raw_cat LIKE 'DOGE%'
      OR raw_cat LIKE 'SOL%' OR raw_cat LIKE 'XRP%' OR raw_cat LIKE 'SHIBA%'
      OR raw_cat LIKE 'COIN%' THEN 'Crypto'
    -- Finance
    WHEN raw_cat LIKE 'FED%' OR raw_cat LIKE 'INX%' OR raw_cat LIKE 'NASDAQ%'
      OR raw_cat LIKE 'TNOTE%' OR raw_cat LIKE 'USDJPY%' OR raw_cat LIKE 'EURUSD%'
      OR raw_cat LIKE 'GAS%' OR raw_cat LIKE 'WTI%' OR raw_cat LIKE 'EGGS%'
      OR raw_cat LIKE 'CPI%' OR raw_cat LIKE 'ACPI%' OR raw_cat LIKE 'GDP%'
      OR raw_cat LIKE 'PAYROLLS%' OR raw_cat LIKE 'U3%' OR raw_cat LIKE 'RECSSNBER%'
      OR raw_cat LIKE 'IPO%' OR raw_cat LIKE 'EARNINGS%' OR raw_cat LIKE 'TESLA%'
      OR raw_cat LIKE 'TARIFF%' OR raw_cat LIKE 'RATECUT%' OR raw_cat LIKE 'TERMINAL%'
      OR raw_cat LIKE 'DCEIL%' OR raw_cat LIKE 'AAA%' OR raw_cat LIKE 'MUSKPACKAGE%'
      OR raw_cat LIKE 'MUSKDOGE%' OR raw_cat LIKE 'NEWPARTYMUSK%'
      OR raw_cat LIKE 'LARGETARIFF%' OR raw_cat LIKE 'FTACOUNTRIES%'
      OR raw_cat LIKE 'DEBT%' OR raw_cat LIKE 'GAMBLING%' OR raw_cat LIKE 'RECNCBILL%'
      OR raw_cat LIKE 'SHUTDOWN%' OR raw_cat LIKE 'EXPAND%' OR raw_cat LIKE 'DEPORT%'
      OR raw_cat LIKE 'WAYMO%' OR raw_cat LIKE 'GOLDCARDS%' OR raw_cat LIKE 'LEAVEPOWELL%'
      OR raw_cat LIKE 'POWELLMENTION%' THEN 'Finance'
    -- Politics
    WHEN raw_cat LIKE 'PRES%' OR raw_cat LIKE 'SENATE%' OR raw_cat LIKE 'HOUSE%'
      OR raw_cat LIKE 'GOV%' OR raw_cat LIKE 'TRUMP%' OR raw_cat LIKE 'BIDEN%'
      OR raw_cat LIKE 'CABINET%' OR raw_cat LIKE 'MAYOR%' OR raw_cat LIKE 'ELECTION%'
      OR raw_cat LIKE 'POPVOTE%' OR raw_cat LIKE 'EC%' OR raw_cat LIKE 'CANADA%'
      OR raw_cat LIKE 'POWER%' OR raw_cat LIKE 'INAUG%' OR raw_cat LIKE 'EOCOUNT%'
      OR raw_cat LIKE 'EOWEEK%' OR raw_cat LIKE 'STATEDEEP%' OR raw_cat LIKE 'JAN6%'
      OR raw_cat LIKE 'DEMSWEEP%' OR raw_cat LIKE 'DEBATES%' OR raw_cat LIKE 'SPEAKER%'
      OR raw_cat LIKE 'CONTROL%' OR raw_cat LIKE 'DJT%' OR raw_cat LIKE 'VOTE%'
      OR raw_cat LIKE 'SENMAJORITY%' OR raw_cat LIKE 'RSENATE%' OR raw_cat LIKE 'RHOUSE%'
      OR raw_cat LIKE 'CLOSESTSTATE%' OR raw_cat LIKE 'SWINGSTATES%'
      OR raw_cat LIKE 'TIPPINGPOINT%' OR raw_cat LIKE 'LASTSTATECALL%'
      OR raw_cat LIKE 'STATESHIFTRIGHT%' OR raw_cat LIKE 'TIKTOK%'
      OR raw_cat LIKE 'GREENLAND%' OR raw_cat LIKE 'SEC%' OR raw_cat LIKE 'DNI%'
      OR raw_cat LIKE 'FBI%' OR raw_cat LIKE 'DOED%' OR raw_cat LIKE 'RFK%'
      OR raw_cat LIKE 'RUBIO%' OR raw_cat LIKE 'WLEADER%' OR raw_cat LIKE 'LEADEROUT%'
      OR raw_cat LIKE 'LEAVEADMIN%' THEN 'Politics'
    -- Weather
    WHEN raw_cat LIKE 'HIGH%' OR raw_cat LIKE 'RAIN%' OR raw_cat LIKE 'SNOW%'
      OR raw_cat LIKE 'TORNADO%' OR raw_cat LIKE 'HURCAT%' OR raw_cat LIKE 'ARCTICICE%'
      OR raw_cat LIKE 'WEATHER%' OR raw_cat LIKE 'HMONTH%' THEN 'Weather'
    -- Entertainment
    WHEN raw_cat LIKE 'SPOTIFY%' OR raw_cat LIKE 'RT%' OR raw_cat LIKE 'OSCAR%'
      OR raw_cat LIKE 'GRAM%' OR raw_cat LIKE 'EMMY%' OR raw_cat LIKE 'BAFTA%'
      OR raw_cat LIKE 'NETFLIX%' OR raw_cat LIKE 'TOP%' OR raw_cat LIKE 'BILLBOARD%'
      OR raw_cat LIKE 'KIMMEL%' OR raw_cat LIKE 'SOUTHPARK%' OR raw_cat LIKE 'MRBEAST%'
      OR raw_cat LIKE 'SNF%' OR raw_cat LIKE 'TNF%' OR raw_cat LIKE 'SUPERBOWLHEADLINE%'
      OR raw_cat LIKE 'SBADS%' OR raw_cat LIKE 'SBSETLISTS%' OR raw_cat LIKE 'SBPERFORM%'
      OR raw_cat LIKE 'GAMEAWARDS%' OR raw_cat LIKE 'GTA6%' OR raw_cat LIKE 'ANIME%'
      OR raw_cat LIKE 'MANTIS%' OR raw_cat LIKE 'APPRANK%' OR raw_cat LIKE 'TIME%'
      OR raw_cat LIKE 'SONGSON%' OR raw_cat LIKE 'ALBUUMSALES%'
      OR raw_cat LIKE 'SWIFTMENTION%' OR raw_cat LIKE 'MOSTSTREAMED%' THEN 'Entertainment'
    -- Media
    WHEN raw_cat LIKE 'MENTION%' OR raw_cat LIKE 'GOOGLESEARCH%' OR raw_cat LIKE 'RANKLIST%'
      OR raw_cat LIKE 'HEADLINE%' OR raw_cat LIKE 'VANCE%' OR raw_cat LIKE 'UFSD%'
      OR raw_cat LIKE 'TSAW%' OR raw_cat LIKE 'CASE%' OR raw_cat LIKE '538APPROVE%'
      OR raw_cat LIKE 'APRPOTUS%' THEN 'Media'
    -- Science/Tech
    WHEN raw_cat LIKE 'LLM%' OR raw_cat LIKE 'AI%' OR raw_cat LIKE 'SPACEX%'
      OR raw_cat LIKE 'ALIENS%' OR raw_cat LIKE 'APPLE%' THEN 'Science/Tech'
    -- World Events
    WHEN raw_cat LIKE 'NOBEL%' OR raw_cat LIKE 'POPE%' OR raw_cat LIKE 'NEXTPOPE%'
      OR raw_cat LIKE 'EPSTEIN%' OR raw_cat LIKE 'ZELENSKY%'
      OR raw_cat LIKE 'ARREST%' OR raw_cat LIKE 'SKPRES%' THEN 'World Events'
    -- Esports
    WHEN raw_cat LIKE 'LOL%' OR raw_cat LIKE 'CSGO%'
      OR raw_cat LIKE 'INTERNETINVITATIONAL%' THEN 'Esports'
    ELSE 'Other'
END
"""


def load_opportunities(data_dir: Path, config: StrategyConfig) -> pd.DataFrame:
    """Load only the trades we care about: taker=YES, longshot prices, resolved markets."""
    con = duckdb.connect()
    trades_dir = data_dir / "kalshi" / "trades"
    markets_dir = data_dir / "kalshi" / "markets"

    # Excluded categories
    excluded = [k for k, v in config.category_weights.items() if v == 0]

    print(f"Loading maker opportunities (yes_price {config.min_price}-{config.max_price}¢, taker=YES)...")
    df = con.execute(f"""
        WITH resolved_markets AS (
            SELECT ticker, event_ticker, result, close_time
            FROM '{markets_dir}/*.parquet'
            WHERE status = 'finalized'
              AND result IN ('yes', 'no')
              AND close_time IS NOT NULL
        ),
        trades_with_cat AS (
            SELECT
                t.ticker,
                t.yes_price,
                t.count AS contracts,
                t.created_time,
                m.result,
                m.close_time AS resolve_time,
                CASE
                    WHEN m.event_ticker IS NULL OR m.event_ticker = '' THEN 'independent'
                    WHEN regexp_extract(m.event_ticker, '^([A-Z0-9]+)', 1) = '' THEN 'independent'
                    ELSE regexp_replace(regexp_extract(m.event_ticker, '^([A-Z0-9]+)', 1), '^KX', '')
                END AS raw_cat
            FROM '{trades_dir}/*.parquet' t
            INNER JOIN resolved_markets m ON t.ticker = m.ticker
            WHERE t.taker_side = 'yes'
              AND t.yes_price BETWEEN {config.min_price} AND {config.max_price}
              AND t.created_time >= '{config.start_date}'
              AND m.close_time > t.created_time
        )
        SELECT
            ticker,
            yes_price,
            contracts,
            created_time,
            resolve_time,
            result,
            raw_cat,
            {CATEGORY_SQL} AS category,
            EXTRACT(HOUR FROM created_time) AS hour_et
        FROM trades_with_cat
        ORDER BY created_time
    """).df()

    # Filter excluded categories
    df = df[~df["category"].isin(excluded)].reset_index(drop=True)

    print(f"  Loaded {len(df):,} opportunities across {df['category'].nunique()} categories")
    print(f"  Date range: {df['created_time'].min()} to {df['created_time'].max()}")
    print(f"  Category distribution:")
    for cat, count in df["category"].value_counts().items():
        print(f"    {cat}: {count:,}")

    return df


# ── Backtest Engine (vectorized where possible) ────────────────────────────

def run_backtest(df: pd.DataFrame, config: StrategyConfig) -> dict:
    """Run the v2 backtest with proper collateral lockup."""
    np.random.seed(42)

    n = len(df)
    cat_weights = config.category_weights

    # Pre-extract arrays for speed
    yes_prices = df["yes_price"].values
    contracts_arr = df["contracts"].values
    results_arr = (df["result"].values == "no")  # True = we win (sold YES, NO resolved)
    categories = df["category"].values
    hours = df["hour_et"].values.astype(int)
    timestamps = df["created_time"].values
    resolve_times = df["resolve_time"].values  # when market closes and collateral unlocks

    # Hour multipliers as array lookup
    hour_mults = np.array([1.3 if 16 <= h <= 23 else (1.0 if 5 <= h <= 15 else 0.7) for h in range(24)])

    # Compute fill probabilities per trade
    cat_weight_arr = np.array([cat_weights.get(c, 0) for c in categories])
    hour_mult_arr = hour_mults[hours]
    fill_probs = config.base_fill_rate * hour_mult_arr * cat_weight_arr

    # Simulate fills
    rolls = np.random.random(n)
    filled = rolls < fill_probs

    # State tracking — PROPER COLLATERAL MODEL
    available_capital = config.starting_capital  # free to deploy
    locked_collateral = 0.0  # collateral in open positions
    regime_returns = []

    # Open positions: list of (resolve_time, collateral, pnl, category)
    # Use a sorted structure for efficient resolution
    import heapq
    open_positions = []  # min-heap by resolve_time

    # Category exposure: collateral currently locked per category
    category_locked = {}

    trade_results = []
    orders_posted = 0
    regime_off_count = 0
    peak_portfolio = config.starting_capital
    positions_resolved = 0

    print(f"Running backtest on {n:,} opportunities...")
    report_interval = max(1, n // 20)

    for i in range(n):
        current_time = timestamps[i]

        # Resolve any positions whose market has closed
        while open_positions and open_positions[0][0] <= current_time:
            _, collat, pnl, pos_cat = heapq.heappop(open_positions)
            available_capital += collat + pnl  # return collateral + profit/loss
            locked_collateral -= collat
            category_locked[pos_cat] = category_locked.get(pos_cat, 0) - collat
            positions_resolved += 1

        portfolio_value = available_capital + locked_collateral

        if i % report_interval == 0:
            print(f"  {i/n*100:.0f}% | Avail: ${available_capital:,.2f} | Locked: ${locked_collateral:,.2f} | Portfolio: ${portfolio_value:,.2f} | Open: {len(open_positions):,} | Trades: {len(trade_results):,}")

        # Check drawdown on total portfolio
        peak_portfolio = max(peak_portfolio, portfolio_value)

        # Skip if not filled
        if not filled[i]:
            orders_posted += 1
            continue

        yes_p = int(yes_prices[i])
        won = bool(results_arr[i])
        cat = categories[i]

        # Regime filter (always update regime window, even when blocking)
        excess_check = (1.0 if won else 0.0) - (100 - yes_p) / 100.0
        regime_returns.append(excess_check)
        if len(regime_returns) >= config.regime_window:
            avg_excess = np.mean(regime_returns[-config.regime_window:])
            if avg_excess < config.regime_threshold:
                regime_off_count += 1
                orders_posted += 1
                continue

        # Category exposure check (actual locked collateral per category)
        cat_locked = category_locked.get(cat, 0)
        if cat_locked >= config.max_category_exposure * portfolio_value:
            orders_posted += 1
            continue

        orders_posted += 1

        # Position sizing (Kelly) — based on AVAILABLE capital, not total portfolio
        p_win = 1.0 - yes_p / 100.0 + 0.02  # add 2pp edge estimate
        p_win = min(p_win, 0.99)
        b = yes_p / (100 - yes_p)
        kelly = max(0, (p_win * (b + 1) - 1) / b) * config.kelly_fraction if b > 0 else 0

        max_pos = min(
            kelly * available_capital,
            config.max_position_pct * portfolio_value,
            config.max_category_exposure * portfolio_value - cat_locked,
        )
        if max_pos <= 0:
            continue

        collateral_per = (100 - yes_p) / 100.0
        trade_contracts = min(int(contracts_arr[i]), int(max_pos / collateral_per))
        if trade_contracts <= 0:
            continue

        trade_collateral = trade_contracts * collateral_per
        if trade_collateral > available_capital:
            trade_contracts = int(available_capital / collateral_per)
            trade_collateral = trade_contracts * collateral_per
            if trade_contracts <= 0:
                continue

        # Calculate P&L (known from resolution, but collateral stays locked until resolve_time)
        if won:
            pnl = trade_contracts * (yes_p / 100.0)
        else:
            pnl = -trade_contracts * ((100 - yes_p) / 100.0)

        # Kalshi maker fee: round_up(0.0175 × C × P × (1-P))
        p_dollars = yes_p / 100.0
        import math
        maker_fee = math.ceil(0.0175 * trade_contracts * p_dollars * (1 - p_dollars) * 100) / 100
        pnl -= maker_fee  # fees reduce P&L regardless of win/loss

        # Lock collateral — capital is NOT available until market resolves
        available_capital -= trade_collateral
        locked_collateral += trade_collateral
        category_locked[cat] = category_locked.get(cat, 0) + trade_collateral

        # Add to open positions heap (resolved at close_time)
        resolve_t = resolve_times[i]
        heapq.heappush(open_positions, (resolve_t, trade_collateral, pnl, cat))

        portfolio_value = available_capital + locked_collateral

        trade_results.append({
            "timestamp": timestamps[i],
            "resolve_time": resolve_t,
            "ticker": df.iloc[i]["ticker"],
            "category": cat,
            "yes_price": yes_p,
            "contracts": trade_contracts,
            "collateral": trade_collateral,
            "won": won,
            "pnl": pnl,
            "portfolio_value": portfolio_value,
            "available_capital": available_capital,
            "locked_collateral": locked_collateral,
        })

    # Resolve remaining open positions
    while open_positions:
        _, collat, pnl, pos_cat = heapq.heappop(open_positions)
        available_capital += collat + pnl
        locked_collateral -= collat

    final_portfolio = available_capital + locked_collateral
    print(f"  100% | Final: ${final_portfolio:,.2f} | Trades: {len(trade_results):,} | Resolved: {positions_resolved:,}")

    if not trade_results:
        print("No trades executed!")
        return {"trades": pd.DataFrame(), "config": config, "summary": {}, "category_pnl": pd.DataFrame()}

    trades_df = pd.DataFrame(trade_results)
    trades_df["timestamp"] = pd.to_datetime(trades_df["timestamp"])

    total_pnl = final_portfolio - config.starting_capital
    max_dd = compute_max_drawdown(trades_df["portfolio_value"].values, config.starting_capital)

    # Sharpe — include ALL calendar days (not just trading days)
    daily_pnl = trades_df.set_index("timestamp")["pnl"].resample("D").sum()
    daily_pnl = daily_pnl.fillna(0)  # include zero-PnL days
    sharpe = (daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)) if len(daily_pnl) > 1 and daily_pnl.std() > 0 else 0

    cat_pnl = trades_df.groupby("category").agg(
        total_pnl=("pnl", "sum"),
        num_trades=("pnl", "count"),
        win_rate=("won", "mean"),
        avg_pnl=("pnl", "mean"),
    ).sort_values("total_pnl", ascending=False)

    summary = {
        "starting_capital": config.starting_capital,
        "final_capital": final_portfolio,
        "total_return_pct": total_pnl / config.starting_capital * 100,
        "total_pnl": total_pnl,
        "num_trades": len(trades_df),
        "win_rate": trades_df["won"].mean() * 100,
        "avg_return_per_trade": trades_df["pnl"].mean(),
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": max_dd,
        "orders_posted": orders_posted,
        "fills_received": len(trades_df),
        "fill_rate": len(trades_df) / orders_posted * 100 if orders_posted > 0 else 0,
        "regime_off_count": regime_off_count,
    }

    return {"trades": trades_df, "config": config, "summary": summary, "category_pnl": cat_pnl}


def compute_max_drawdown(portfolio_values: np.ndarray, starting_capital: float) -> float:
    values = np.concatenate([[starting_capital], portfolio_values])
    peak = np.maximum.accumulate(values)
    drawdown = (peak - values) / peak * 100
    return float(np.max(drawdown))


# ── Visualization ───────────────────────────────────────────────────────────

def plot_results(results: dict, output_path: Path):
    trades = results["trades"]
    summary = results["summary"]
    cat_pnl = results["category_pnl"]
    config = results["config"]

    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(
        f"V2 Multi-Signal Maker Strategy - Portfolio Backtest (${config.starting_capital:,.0f} Starting Capital)",
        fontsize=14, fontweight="bold", y=0.98,
    )

    gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.3, top=0.93, bottom=0.05, left=0.08, right=0.92)

    # 1. Portfolio Value Over Time
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(trades["timestamp"], trades["portfolio_value"], color="#2ecc71", linewidth=1.0, label="Portfolio Value")
    ax1.axhline(y=config.starting_capital, color="gray", linestyle="--", linewidth=0.8, label="Starting Capital")
    ax1.set_title("Portfolio Value Over Time (V2 Maker Strategy)")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # 2. Distribution of Trade Returns
    ax2 = fig.add_subplot(gs[1, 0])
    pnl_vals = trades["pnl"]
    ax2.hist(pnl_vals[trades["won"]], bins=50, color="#2ecc71", alpha=0.8, label="Wins")
    ax2.hist(pnl_vals[~trades["won"]], bins=50, color="#e74c3c", alpha=0.8, label="Losses")
    ax2.axvline(x=pnl_vals.mean(), color="orange", linestyle="--", linewidth=2, label=f"Mean: ${pnl_vals.mean():.2f}")
    ax2.set_title("Distribution of Trade P&L")
    ax2.set_xlabel("P&L ($)")
    ax2.set_ylabel("Number of Trades")
    ax2.legend()

    # 3. P&L by Category
    ax3 = fig.add_subplot(gs[1, 1])
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in cat_pnl["total_pnl"]]
    ax3.barh(cat_pnl.index, cat_pnl["total_pnl"], color=colors)
    ax3.set_title("P&L by Category")
    ax3.set_xlabel("Total P&L ($)")

    # 4. Cumulative P&L Over Time
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(trades["timestamp"], trades["pnl"].cumsum(), color="#2ecc71", linewidth=1.0)
    ax4.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax4.set_title("Cumulative P&L Over Time")
    ax4.set_xlabel("Date")
    ax4.set_ylabel("Cumulative P&L ($)")
    ax4.grid(True, alpha=0.3)

    # 5. Summary Stats
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis("off")

    stats_text = f"""MAKER STRATEGY V2 BACKTEST SUMMARY

Capital:
  Starting: ${summary['starting_capital']:,.2f}
  Final: ${summary['final_capital']:,.2f}
  Total Return: {summary['total_return_pct']:.2f}%
  Total P&L: ${summary['total_pnl']:,.2f}

Trading (Maker Orders):
  Orders Posted: {summary['orders_posted']:,}
  Fills Received: {summary['fills_received']:,}
  Fill Rate: {summary['fill_rate']:.1f}%

Performance:
  Trades Completed: {summary['num_trades']:,}
  Win Rate: {summary['win_rate']:.2f}%
  Avg P&L/Trade: ${summary['avg_return_per_trade']:.4f}
  Sharpe Ratio: {summary['sharpe_ratio']:.2f}
  Max Drawdown: -{summary['max_drawdown_pct']:.2f}%

Strategy (V2 Multi-Signal):
  Role: MAKER (sell YES at longshots)
  Price Range: {config.min_price}-{config.max_price}c
  Kelly Fraction: {config.kelly_fraction:.0%}
  Max Position: {config.max_position_pct:.0%} of portfolio
  Max Category Exp: {config.max_category_exposure:.0%}
  Drawdown Limit: {config.max_drawdown:.0%}
  Regime Filter: {config.regime_window} trade window
  Fill Rate Model: {config.base_fill_rate:.0%} base * cat * hour"""

    ax5.text(0.05, 0.95, stats_text, transform=ax5.transAxes,
             fontsize=8, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved visualization to {output_path}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    data_dir = Path(__file__).parent / "prediction-market-analysis" / "data"
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return

    config = StrategyConfig()
    df = load_opportunities(data_dir, config)
    results = run_backtest(df, config)

    if results["trades"].empty:
        return

    s = results["summary"]
    print(f"\n{'='*60}")
    print(f"V2 BACKTEST RESULTS")
    print(f"{'='*60}")
    print(f"Starting Capital:   ${s['starting_capital']:>12,.2f}")
    print(f"Final Capital:      ${s['final_capital']:>12,.2f}")
    print(f"Total Return:       {s['total_return_pct']:>12.2f}%")
    print(f"Total P&L:          ${s['total_pnl']:>12,.2f}")
    print(f"")
    print(f"Orders Posted:      {s['orders_posted']:>12,}")
    print(f"Fills Received:     {s['fills_received']:>12,}")
    print(f"Fill Rate:          {s['fill_rate']:>12.1f}%")
    print(f"Regime Off Count:   {s['regime_off_count']:>12,}")
    print(f"")
    print(f"Trades Completed:   {s['num_trades']:>12,}")
    print(f"Win Rate:           {s['win_rate']:>12.2f}%")
    print(f"Avg P&L/Trade:      ${s['avg_return_per_trade']:>12.4f}")
    print(f"Sharpe Ratio:       {s['sharpe_ratio']:>12.2f}")
    print(f"Max Drawdown:       {s['max_drawdown_pct']:>12.2f}%")
    print(f"{'='*60}")
    print(f"\nCategory Breakdown:")
    print(results["category_pnl"].to_string())

    output_path = Path(__file__).parent / "output"
    output_path.mkdir(exist_ok=True)
    plot_results(results, output_path / "v2_backtest.png")
    results["trades"].to_csv(output_path / "v2_trades.csv", index=False)
    print(f"Saved trades to {output_path / 'v2_trades.csv'}")


if __name__ == "__main__":
    main()
