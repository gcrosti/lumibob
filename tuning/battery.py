"""
battery — Phase 3 five-regime test battery.

Defines the five standard market regimes and provides helpers to:
  - check which regimes have warm price data in the DB
  - run a full backtest for one regime with a given parameter set
  - collect detailed metrics from portfolio_snapshots and trades
  - compare two parameter sets side-by-side with a gate check

Regimes
-------
Each regime is chosen to stress a distinct market environment:

  1. Calm bull       2017-01 → 2017-12  Low vol, steady uptrend
  2. Vol shock       2020-02 → 2020-06  COVID crash + fast recovery
  3. Sideways        2022-01 → 2022-12  Bear market / pairs' natural habitat
  4. Trend bull      2023-04 → 2023-12  Trend-following bull post-bottom
  5. Mixed recent    2024-01 → 2024-09  OOS versus anything trained on ≤ 2023

PASS criterion for Phase 3 gate:
    Best-trial params outscore baseline params in ≥ 3 of the completed regimes.
    (Warm-only runs count; cold regimes are skipped rather than failed.)
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_DB_URL = os.getenv('DB_URL', 'postgresql://postgres:lumibob@localhost:5432/lumibob')

# ---------------------------------------------------------------------------
# Regime definitions
# ---------------------------------------------------------------------------

class Regime(NamedTuple):
    name: str
    start: date
    end: date
    description: str


REGIMES: list[Regime] = [
    Regime('calm_bull_2017',   date(2017, 1, 3),  date(2017, 12, 29), 'Calm bull — low vol, steady uptrend'),
    Regime('vol_shock_2020',   date(2020, 2, 3),  date(2020, 6, 30),  'Vol shock — COVID crash + recovery'),
    Regime('sideways_2022',    date(2022, 1, 3),  date(2022, 12, 30), 'Sideways high-vol — pairs\' natural habitat'),
    Regime('trend_bull_2023',  date(2023, 4, 3),  date(2023, 12, 29), 'Trend-following bull — post-bottom'),
    Regime('mixed_2024',       date(2024, 1, 2),  date(2024, 9, 30),  'Mixed recent — OOS vs. ≤2023 training'),
]


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class RegimeResult:
    regime: Regime
    params_label: str
    run_id: str | None = None
    # Returns
    return_pct: float | None = None
    spy_return_pct: float | None = None
    beats_spy: bool | None = None
    # Risk
    sharpe: float | None = None
    max_drawdown_pct: float | None = None
    # Activity
    n_trades: int | None = None
    avg_active_pairs: float | None = None
    avg_cash_ratio: float | None = None
    # Composite
    score: float | None = None
    # Status
    skipped: bool = False
    skip_reason: str = ''
    error: str = ''


@dataclass
class BatteryResult:
    params_label: str
    params: dict[str, Any]
    regime_results: list[RegimeResult] = field(default_factory=list)

    def completed(self) -> list[RegimeResult]:
        return [r for r in self.regime_results if not r.skipped and r.run_id is not None]

    def beats_spy_count(self) -> int:
        return sum(1 for r in self.completed() if r.beats_spy)

    def mean_score(self) -> float | None:
        scores = [r.score for r in self.completed() if r.score is not None]
        return float(np.mean(scores)) if scores else None


# ---------------------------------------------------------------------------
# Cache warmth check
# ---------------------------------------------------------------------------

# Minimum number of trading days we consider "covered" in a date window.
# 5 trading days ≃ 1 week — deliberately loose; real coverage is symbol-count driven.
_MIN_COVERAGE_DAYS = 5
# lookback_window default (calendar days) — data must start this early.
_LOOKBACK_CALENDAR_DAYS = 140


def check_warmth(regime: Regime) -> tuple[bool, str]:
    """
    Return (is_warm, reason) for a regime.

    "Warm" means stock_prices contains rows spanning from at most
    (regime.start - LOOKBACK_CALENDAR_DAYS) to at least regime.end.
    Both ends must be within a 5-trading-day tolerance.
    """
    lookback_start = regime.start - timedelta(days=_LOOKBACK_CALENDAR_DAYS)

    try:
        with psycopg2.connect(_DB_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        MIN(time)::date AS earliest,
                        MAX(time)::date AS latest,
                        COUNT(DISTINCT time::date) AS trading_days
                    FROM stock_prices
                    WHERE time >= %s AND time <= %s
                    """,
                    (
                        datetime.combine(lookback_start, datetime.min.time()),
                        datetime.combine(regime.end, datetime.min.time()),
                    ),
                )
                row = cur.fetchone()
    except Exception as exc:
        return False, f'DB error: {exc}'

    if row is None or row[2] is None or row[2] < _MIN_COVERAGE_DAYS:
        return False, f'no data in range {lookback_start} → {regime.end}'

    earliest, latest, n_days = row

    # Boundary checks (20-day tolerance on the lookback start).
    if earliest > lookback_start + timedelta(days=20):
        return False, (
            f'lookback data starts {earliest}, need ≤ {lookback_start + timedelta(days=20)}'
        )
    if latest < regime.end - timedelta(days=5):
        return False, f'data ends {latest}, need ≥ {regime.end - timedelta(days=5)}'

    # Density check — require ≥ 70% of expected trading days to detect gaps
    # (e.g. Jan–Jul missing from a full-year range).
    total_calendar = (regime.end - lookback_start).days
    expected_trading_days = int(total_calendar * 5 / 7 * 0.94)
    min_required = max(_MIN_COVERAGE_DAYS, int(expected_trading_days * 0.70))
    if n_days < min_required:
        return False, (
            f'data gap detected: {n_days} days found, need ≥ {min_required} '
            f'(70% of ~{expected_trading_days} expected)'
        )

    return True, f'{n_days} trading days {earliest} → {latest}'


# ---------------------------------------------------------------------------
# Single-regime runner
# ---------------------------------------------------------------------------

def run_regime(
    regime: Regime,
    params: dict[str, Any],
    params_label: str,
    budget: float = 10_000,
) -> RegimeResult:
    """
    Run one BobsBrain backtest for *regime* with *params* and return metrics.
    """
    from lumibot.backtesting import YahooDataBacktesting
    from BobsBrain import BobsBrain

    result = RegimeResult(regime=regime, params_label=params_label)
    start_ts = datetime.now(timezone.utc)

    logger.info(
        '[battery] Running %s  label=%s  %s → %s',
        regime.name, params_label, regime.start, regime.end,
    )

    try:
        BobsBrain.backtest(
            YahooDataBacktesting,
            datetime(regime.start.year, regime.start.month, regime.start.day),
            datetime(regime.end.year, regime.end.month, regime.end.day),
            budget=budget,
            parameters=params,
            show_plot=False,
            show_tearsheet=False,
            save_tearsheet=False,
        )
    except Exception as exc:
        logger.exception('[battery] %s / %s raised: %s', regime.name, params_label, exc)
        result.error = str(exc)
        return result

    run_id = _find_run_id(after_ts=start_ts)
    if run_id is None:
        result.error = 'run_id not found in DB after backtest'
        return result

    result.run_id = run_id
    _fill_metrics(result)
    return result


def _find_run_id(after_ts: datetime, retries: int = 5, delay: float = 3.0) -> str | None:
    """
    Query the DB for the most recently completed run started after *after_ts*.

    Retries up to *retries* times with *delay* seconds between attempts to
    handle the case where BobsBrain sets completed_at slightly after backtest()
    returns.  Also accepts runs whose completed_at is NULL but which have
    sufficient portfolio snapshots (≥ 5), treating them as completed runs that
    failed to write their completion timestamp (e.g. after a transient crash
    that Lumibot recovered from).
    """
    for attempt in range(retries):
        with psycopg2.connect(_DB_URL) as conn:
            with conn.cursor() as cur:
                # Primary: cleanly completed runs.
                cur.execute(
                    """
                    SELECT run_id
                    FROM backtest_runs
                    WHERE started_at >= %s
                      AND mode = 'backtest'
                      AND completed_at IS NOT NULL
                    ORDER BY started_at DESC
                    LIMIT 1
                    """,
                    (after_ts,),
                )
                row = cur.fetchone()
                if row:
                    return row[0]

                # Fallback: runs whose completed_at is NULL but have data,
                # indicating a transient crash that still produced results.
                cur.execute(
                    """
                    SELECT r.run_id
                    FROM backtest_runs r
                    WHERE r.started_at >= %s
                      AND r.mode = 'backtest'
                      AND r.completed_at IS NULL
                      AND (SELECT COUNT(*) FROM portfolio_snapshots p
                           WHERE p.run_id = r.run_id) >= 5
                    ORDER BY r.started_at DESC
                    LIMIT 1
                    """,
                    (after_ts,),
                )
                row = cur.fetchone()
                if row:
                    logger.warning(
                        '_find_run_id: run %s has completed_at=NULL but has '
                        'portfolio data — treating as partial completion.',
                        row[0],
                    )
                    return row[0]

        if attempt < retries - 1:
            time.sleep(delay)

    return None


def _fill_metrics(result: RegimeResult) -> None:
    """Populate result fields by querying portfolio_snapshots and trades."""
    run_id = result.run_id
    with psycopg2.connect(_DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT time, portfolio_value, spy_value, active_pairs, cash_ratio
                FROM portfolio_snapshots
                WHERE run_id = %s
                ORDER BY time
                """,
                (run_id,),
            )
            snap_rows = cur.fetchall()
            cur.execute(
                "SELECT COUNT(*) FROM trades WHERE run_id = %s",
                (run_id,),
            )
            result.n_trades = cur.fetchone()[0]

    if not snap_rows or len(snap_rows) < 3:
        result.error = f'only {len(snap_rows)} snapshots'
        return

    snaps = pd.DataFrame(snap_rows, columns=['time', 'portfolio_value', 'spy_value', 'active_pairs', 'cash_ratio'])
    pv = snaps['portfolio_value'].astype(float)
    spy = snaps['spy_value'].astype(float)

    result.return_pct = float((pv.iloc[-1] / pv.iloc[0] - 1) * 100)

    if not spy.isna().all() and spy.iloc[0] > 0:
        result.spy_return_pct = float((spy.iloc[-1] / spy.iloc[0] - 1) * 100)
        result.beats_spy = result.return_pct > result.spy_return_pct

    daily_ret = pv.pct_change().dropna()
    if daily_ret.std() > 1e-9:
        result.sharpe = float(daily_ret.mean() / daily_ret.std() * np.sqrt(252))

    roll_max = pv.cummax()
    dd = ((pv - roll_max) / roll_max).min()
    result.max_drawdown_pct = float(abs(dd) * 100)

    ap = snaps['active_pairs'].dropna()
    if not ap.empty:
        result.avg_active_pairs = float(ap.mean())

    cr = snaps['cash_ratio'].dropna()
    if not cr.empty:
        result.avg_cash_ratio = float(cr.mean())

    # Composite score — delegate to BacktestObjective.score_run to avoid
    # duplicating the formula (which would silently diverge if the objective
    # changes, e.g. when spy_penalty_weight differs between phases).
    from tuning.objective import BacktestObjective
    from datetime import date as _date

    # Construct a minimal scorer just for score_run(); train window is unused
    # because score_run() reads from the DB by run_id, not from the objective.
    scorer = BacktestObjective(
        train_start=result.regime.start,
        train_end=result.regime.end,
        budget=10_000,
        spy_penalty_weight=1.0,   # Phase 3 battery uses the full SPY constraint
    )
    result.score = scorer.score_run(run_id)


# ---------------------------------------------------------------------------
# Battery runner
# ---------------------------------------------------------------------------

def run_battery(
    params: dict[str, Any],
    params_label: str,
    budget: float = 10_000,
    warm_only: bool = True,
    regimes: list[Regime] | None = None,
) -> BatteryResult:
    """
    Run all (or a subset of) regimes and return a BatteryResult.

    Parameters
    ----------
    warm_only : bool
        If True, skip regimes whose price data is not already in the DB.
        If False, run all regimes regardless (cold fetch will be slow).
    regimes : list[Regime] | None
        Override the default REGIMES list.
    """
    battery = BatteryResult(params_label=params_label, params=params)
    regime_list = regimes or REGIMES

    for regime in regime_list:
        is_warm, warmth_note = check_warmth(regime)
        if not is_warm and warm_only:
            logger.info('[battery] Skipping %s — cold cache (%s)', regime.name, warmth_note)
            result = RegimeResult(
                regime=regime,
                params_label=params_label,
                skipped=True,
                skip_reason=f'cold cache: {warmth_note}',
            )
            battery.regime_results.append(result)
            continue

        logger.info('[battery] %s — %s', regime.name, warmth_note)
        result = run_regime(regime, params, params_label, budget=budget)
        battery.regime_results.append(result)

    return battery


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(
    result_a: BatteryResult,
    result_b: BatteryResult,
) -> None:
    """Print a side-by-side comparison table for two BatteryResults."""
    _W = 78
    print()
    print('=' * _W)
    print(f'  Phase 3 Battery Report — {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    print('=' * _W)
    print(f'  {"":25s}  {"── " + result_a.params_label + " ──":>22s}  {"── " + result_b.params_label + " ──":>22s}')
    print(f'  {"Regime":<25s}  {"Ret%":>6} {"SPY%":>6} {"Shrp":>5} {"DD%":>5}  {"Ret%":>6} {"SPY%":>6} {"Shrp":>5} {"DD%":>5}  {"Winner":<8}')
    print('-' * _W)

    map_a = {r.regime.name: r for r in result_a.regime_results}
    map_b = {r.regime.name: r for r in result_b.regime_results}

    wins_a = wins_b = ties = 0

    for regime in REGIMES:
        ra = map_a.get(regime.name)
        rb = map_b.get(regime.name)

        def _fmt(r: RegimeResult | None) -> str:
            if r is None or r.skipped:
                return f'{"COLD":>6} {"":>6} {"":>5} {"":>5}'
            if r.error and not r.run_id:
                return f'{"ERR":>6} {"":>6} {"":>5} {"":>5}'
            ret = f'{r.return_pct:+.1f}' if r.return_pct is not None else '  N/A'
            spy = f'{r.spy_return_pct:+.1f}' if r.spy_return_pct is not None else '  N/A'
            shr = f'{r.sharpe:+.2f}' if r.sharpe is not None else ' N/A'
            dd  = f'{r.max_drawdown_pct:.1f}' if r.max_drawdown_pct is not None else 'N/A'
            return f'{ret:>6} {spy:>6} {shr:>5} {dd:>5}'

        def _winner(ra: RegimeResult | None, rb: RegimeResult | None) -> str:
            nonlocal wins_a, wins_b, ties
            if ra is None or rb is None or ra.skipped or rb.skipped:
                return 'SKIP'
            if ra.score is None or rb.score is None:
                return 'N/A'
            if ra.score > rb.score + 0.01:
                wins_a += 1
                return result_a.params_label[:8]
            if rb.score > ra.score + 0.01:
                wins_b += 1
                return result_b.params_label[:8]
            ties += 1
            return 'TIE'

        winner = _winner(ra, rb)
        short_name = regime.name.replace('_', ' ')[:25]
        print(f'  {short_name:<25s}  {_fmt(ra)}  {_fmt(rb)}  {winner:<8}')

    print('-' * _W)
    print(f'  {"Wins":>25s}  {wins_a:>2d} regime(s){"":>26s}{wins_b:>2d} regime(s)')

    # Scores
    sc_a = result_a.mean_score()
    sc_b = result_b.mean_score()
    sc_a_str = f'{sc_a:.4f}' if sc_a is not None else 'N/A'
    sc_b_str = f'{sc_b:.4f}' if sc_b is not None else 'N/A'
    print(f'  {"Mean composite score":>25s}  {sc_a_str:>5}{"":>26s}{sc_b_str:>5}')
    print('=' * _W)
    print()


# ---------------------------------------------------------------------------
# Gate check
# ---------------------------------------------------------------------------

def gate_check(
    result_good: BatteryResult,
    result_bad: BatteryResult,
    min_wins: int = 3,
) -> bool:
    """
    Return True (PASS) if result_good outscores result_bad in ≥ min_wins
    of the completed (non-skipped) regimes.

    Prints a verdict line.
    """
    completed_names = {
        r.regime.name
        for r in result_good.regime_results
        if not r.skipped and r.run_id is not None
    } & {
        r.regime.name
        for r in result_bad.regime_results
        if not r.skipped and r.run_id is not None
    }

    if not completed_names:
        print('[gate] No completed regimes to compare — INCONCLUSIVE')
        return False

    map_good = {r.regime.name: r for r in result_good.regime_results}
    map_bad  = {r.regime.name: r for r in result_bad.regime_results}

    wins = 0
    for name in completed_names:
        rg = map_good[name]
        rb = map_bad[name]
        if rg.score is not None and rb.score is not None and rg.score > rb.score:
            wins += 1

    total = len(completed_names)
    verdict = 'PASS' if wins >= min_wins else 'FAIL'
    print(
        f'\n[gate] Phase 3 gate check: {result_good.params_label} beats '
        f'{result_bad.params_label} in {wins}/{total} completed regimes '
        f'(need ≥ {min_wins}) — [{verdict}]'
    )
    return verdict == 'PASS'
