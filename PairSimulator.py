"""
PairSimulator — vectorized cash-switching backtest for a lead/lag stock pair.

Simulates the MA-crossover strategy over historical price data so that pair
quality can be validated *before* committing to a position.  The simulator is
intentionally stateless and pure-pandas so it adds no I/O dependencies and
can be called inline during before_market_opens().

Usage pattern in BobsBrain discovery loop:

    simulator = PairSimulator()
    result = simulator.optimize(lead_series, lag_series, max_lag=5)
    if result.total_return <= 0 or result.num_trades < 2:
        continue          # reject: strategy wouldn't have been profitable
    # use result.lag, result.short_ma, result.long_ma for the new pair
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class SimResult:
    """
    Outcome of a single simulation run.

    total_return    -- strategy return as a fraction over the simulation window,
                       e.g. 0.05 means +5%
    sharpe          -- annualised Sharpe ratio (assumes 252 trading days/year);
                       nan when there are no return observations
    max_drawdown    -- largest peak-to-trough decline in cumulative strategy
                       value, e.g. -0.03 means -3%; 0.0 when never in drawdown
    win_rate        -- fraction of completed buy→sell cycles that were
                       profitable; nan when num_trades == 0
    num_trades      -- number of completed buy→sell cycles over the window;
                       a value of 0 or 1 means the signal barely fired and
                       the pair should be rejected
    avg_holding_days -- mean number of bars held per trade; nan when num_trades == 0
    lag             -- lag offset (in trading days) used for this result
    short_ma        -- short moving-average window used for this result
    long_ma         -- long moving-average window used for this result
    """
    total_return: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    num_trades: int
    avg_holding_days: float
    lag: int
    short_ma: int
    long_ma: int


@dataclass
class ZScoreSimResult:
    """
    Outcome of a single Z-score spread simulation run.

    Fields mirror SimResult where applicable, with MA parameters replaced by
    Z-score parameters.

    total_return      -- strategy return as a fraction over the simulation window
    sharpe            -- annualised Sharpe ratio (252 trading days/year)
    max_drawdown      -- largest peak-to-trough decline in cumulative value
    win_rate          -- fraction of completed buy→sell cycles that were profitable
    num_trades        -- number of completed buy→sell cycles
    avg_holding_days  -- mean bars held per trade
    zscore_window     -- rolling window used to compute the spread Z-score
    entry_threshold   -- Z-score below which a buy is triggered (positive value)
    exit_threshold    -- Z-score above which a sell is triggered (positive value)
    """
    total_return: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    num_trades: int
    avg_holding_days: float
    zscore_window: int
    entry_threshold: float
    exit_threshold: float


class PairSimulator:
    """
    Vectorized cash-switching simulator for a single lead/lag pair.

    The simulation mirrors the live strategy exactly:
    - Shift the lead series by `lag` days to get the predictive signal
    - Compute short and long rolling MAs on the shifted lead
    - Position = 1 (long lag stock) when short MA > long MA, else 0 (cash)
    - Daily strategy return = position[t-1] * daily_return(lag)[t]

    This models "cash switching": you're either fully invested in the lag stock
    or fully in cash based on the lead signal.
    """

    # Grid bounds used by optimize().  short_ma ∈ [1, _MAX_SHORT_MA] and
    # long_ma ∈ [short_ma+1, _MAX_LONG_MA].  Keeping them as class constants
    # makes it easy to widen the search space in one place.
    _MAX_SHORT_MA: int = 4
    _MAX_LONG_MA: int = 5

    def run(
        self,
        lead: pd.Series,
        lag_stock: pd.Series,
        lag: int,
        short_ma: int,
        long_ma: int,
    ) -> SimResult:
        """
        Simulate the cash-switching strategy for one parameter combination.

        Parameters
        ----------
        lead        : close-price series for the lead stock
        lag_stock   : close-price series for the lag stock
        lag         : number of days to shift the lead series
        short_ma    : short MA window (must be < long_ma)
        long_ma     : long MA window

        Returns a SimResult with all computed metrics.
        """
        if short_ma >= long_ma:
            raise ValueError(f"short_ma ({short_ma}) must be less than long_ma ({long_ma})")

        shifted = lead.shift(lag)

        short = shifted.rolling(window=short_ma, min_periods=short_ma).mean()
        long_ = shifted.rolling(window=long_ma, min_periods=long_ma).mean()

        # Signal: 1 when short MA > long MA (bullish), 0 otherwise
        signal = (short > long_).astype(float)
        # Position on day t is determined by yesterday's signal (no look-ahead)
        position = signal.shift(1).fillna(0)

        # Daily returns of the lag stock
        lag_returns = lag_stock.pct_change().fillna(0)

        # Strategy daily P&L
        strategy_returns = position * lag_returns

        total_return = float((1 + strategy_returns).prod() - 1)

        # Sharpe ratio (annualised)
        mean_ret = strategy_returns.mean()
        std_ret = strategy_returns.std()
        if std_ret > 0:
            sharpe = float((mean_ret / std_ret) * np.sqrt(252))
        else:
            sharpe = float('nan')

        # Max drawdown
        cumulative = (1 + strategy_returns).cumprod()
        rolling_peak = cumulative.cummax()
        drawdown = (cumulative - rolling_peak) / rolling_peak
        max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0

        # Trade-level statistics: find entry/exit pairs
        position_int = position.astype(int)
        entries = (position_int.diff() == 1).values
        exits = (position_int.diff() == -1).values

        entry_indices = np.where(entries)[0]
        exit_indices = np.where(exits)[0]

        # Match each entry to the next exit
        completed_trades: list[tuple[int, int]] = []
        ei = 0
        for entry_idx in entry_indices:
            while ei < len(exit_indices) and exit_indices[ei] <= entry_idx:
                ei += 1
            if ei < len(exit_indices):
                completed_trades.append((entry_idx, exit_indices[ei]))
                ei += 1

        num_trades = len(completed_trades)

        if num_trades > 0:
            lag_values = lag_stock.values
            wins = sum(
                1 for e, x in completed_trades
                if x < len(lag_values) and lag_values[x] > lag_values[e]
            )
            win_rate = float(wins / num_trades)
            avg_holding_days = float(
                np.mean([x - e for e, x in completed_trades])
            )
        else:
            win_rate = float('nan')
            avg_holding_days = float('nan')

        return SimResult(
            total_return=total_return,
            sharpe=sharpe,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            num_trades=num_trades,
            avg_holding_days=avg_holding_days,
            lag=lag,
            short_ma=short_ma,
            long_ma=long_ma,
        )

    def optimize(
        self,
        lead: pd.Series,
        lag_stock: pd.Series,
        max_lag: int = 5,
    ) -> SimResult:
        """
        Grid search over lag ∈ [1..max_lag] and all valid (short_ma, long_ma)
        combinations where short_ma ∈ [1.._MAX_SHORT_MA] and
        long_ma ∈ [short_ma+1.._MAX_LONG_MA].

        Returns the SimResult with the highest total_return.  When all
        combinations produce non-positive returns the result with the highest
        total_return (least negative) is still returned — callers are
        responsible for applying the total_return > 0 acceptance threshold.

        The grid is: max_lag lags × 10 MA combos = up to 50 simulations,
        all vectorised, so this is fast enough to call per-pair during discovery.
        """
        best: SimResult | None = None

        for lag in range(1, max_lag + 1):
            for short_ma in range(1, self._MAX_SHORT_MA + 1):
                for long_ma in range(short_ma + 1, self._MAX_LONG_MA + 1):
                    try:
                        result = self.run(lead, lag_stock, lag, short_ma, long_ma)
                    except Exception:
                        continue
                    if best is None or result.total_return > best.total_return:
                        best = result

        if best is None:
            # Fallback: should never happen since max_lag >= 1 always yields combos
            return SimResult(
                total_return=0.0, sharpe=float('nan'), max_drawdown=0.0,
                win_rate=float('nan'), num_trades=0, avg_holding_days=float('nan'),
                lag=1, short_ma=2, long_ma=5,
            )

        return best

    # ------------------------------------------------------------------
    # Z-score spread mean reversion simulation
    # ------------------------------------------------------------------

    def run_zscore(
        self,
        lead: pd.Series,
        lag_stock: pd.Series,
        zscore_window: int,
        entry_threshold: float,
        exit_threshold: float,
    ) -> ZScoreSimResult:
        """
        Simulate the Z-score spread mean reversion strategy for one parameter set.

        The spread is log(lag) - hedge_ratio * log(lead), normalised to a rolling
        Z-score.  Position = 1 (long lag) when zscore < -entry_threshold, 0 (cash)
        when zscore > -exit_threshold.  Between the two thresholds the prior
        position is held (hysteresis band avoids excessive churn).

        Parameters
        ----------
        lead            : close-price series for the lead stock
        lag_stock       : close-price series for the lag stock
        zscore_window   : rolling window for spread mean/std
        entry_threshold : enter long when zscore < -entry_threshold
        exit_threshold  : exit long when zscore > -exit_threshold
        """
        common = lead.index.intersection(lag_stock.index)
        if len(common) < zscore_window + 5:
            return ZScoreSimResult(
                total_return=0.0, sharpe=float('nan'), max_drawdown=0.0,
                win_rate=float('nan'), num_trades=0, avg_holding_days=float('nan'),
                zscore_window=zscore_window,
                entry_threshold=entry_threshold,
                exit_threshold=exit_threshold,
            )

        lead_c = lead.loc[common]
        lag_c = lag_stock.loc[common]

        log_lead = np.log(lead_c.astype(float).clip(lower=1e-9))
        log_lag = np.log(lag_c.astype(float).clip(lower=1e-9))

        try:
            hedge = float(np.polyfit(log_lead.values, log_lag.values, 1)[0])
        except (np.linalg.LinAlgError, ValueError):
            return ZScoreSimResult(
                total_return=0.0, sharpe=float('nan'), max_drawdown=0.0,
                win_rate=float('nan'), num_trades=0, avg_holding_days=float('nan'),
                zscore_window=zscore_window,
                entry_threshold=entry_threshold,
                exit_threshold=exit_threshold,
            )

        spread = log_lag - hedge * log_lead
        roll_mean = spread.rolling(window=zscore_window, min_periods=zscore_window).mean()
        roll_std = spread.rolling(window=zscore_window, min_periods=zscore_window).std()
        zscore = (spread - roll_mean) / roll_std.replace(0, np.nan)

        # Build position array with hysteresis: enter at -entry, exit at -exit
        position = np.zeros(len(zscore))
        in_position = False
        for i, z in enumerate(zscore.values):
            if np.isnan(z):
                position[i] = 0.0
                in_position = False
                continue
            if not in_position and z < -entry_threshold:
                in_position = True
            elif in_position and z > -exit_threshold:
                in_position = False
            position[i] = 1.0 if in_position else 0.0

        position_series = pd.Series(position, index=zscore.index)
        # Use previous day's position (no look-ahead)
        position_lagged = position_series.shift(1).fillna(0)

        lag_returns = lag_c.pct_change().fillna(0)
        strategy_returns = position_lagged * lag_returns

        total_return = float((1 + strategy_returns).prod() - 1)

        mean_ret = strategy_returns.mean()
        std_ret = strategy_returns.std()
        sharpe = float((mean_ret / std_ret) * np.sqrt(252)) if std_ret > 0 else float('nan')

        cumulative = (1 + strategy_returns).cumprod()
        rolling_peak = cumulative.cummax()
        drawdown = (cumulative - rolling_peak) / rolling_peak
        max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0

        pos_int = position_lagged.astype(int)
        entries = (pos_int.diff() == 1).values
        exits = (pos_int.diff() == -1).values
        entry_indices = np.where(entries)[0]
        exit_indices = np.where(exits)[0]

        completed_trades: list[tuple[int, int]] = []
        ei = 0
        for entry_idx in entry_indices:
            while ei < len(exit_indices) and exit_indices[ei] <= entry_idx:
                ei += 1
            if ei < len(exit_indices):
                completed_trades.append((entry_idx, exit_indices[ei]))
                ei += 1

        num_trades = len(completed_trades)
        lag_values = lag_c.values
        if num_trades > 0:
            wins = sum(
                1 for e, x in completed_trades
                if x < len(lag_values) and lag_values[x] > lag_values[e]
            )
            win_rate = float(wins / num_trades)
            avg_holding_days = float(np.mean([x - e for e, x in completed_trades]))
        else:
            win_rate = float('nan')
            avg_holding_days = float('nan')

        return ZScoreSimResult(
            total_return=total_return,
            sharpe=sharpe,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            num_trades=num_trades,
            avg_holding_days=avg_holding_days,
            zscore_window=zscore_window,
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
        )

    def optimize_zscore(
        self,
        lead: pd.Series,
        lag_stock: pd.Series,
    ) -> ZScoreSimResult:
        """
        Grid search over Z-score parameters to find the combination with the
        highest total_return.

        Search space (18 combinations):
            zscore_window    ∈ [10, 20, 30]
            entry_threshold  ∈ [1.5, 2.0, 2.5]
            exit_threshold   ∈ [0.0, 0.5]

        Returns the ZScoreSimResult with the highest total_return.  When all
        combinations yield non-positive returns the least-negative result is
        returned — callers must apply their own acceptance threshold.
        """
        _WINDOWS = [10, 20, 30]
        _ENTRIES = [1.5, 2.0, 2.5]
        _EXITS = [0.0, 0.5]

        best: Optional[ZScoreSimResult] = None

        for window in _WINDOWS:
            for entry in _ENTRIES:
                for exit_t in _EXITS:
                    if exit_t >= entry:
                        continue  # exit must be tighter than entry
                    try:
                        result = self.run_zscore(lead, lag_stock, window, entry, exit_t)
                    except Exception:
                        continue
                    if best is None or result.total_return > best.total_return:
                        best = result

        if best is None:
            return ZScoreSimResult(
                total_return=0.0, sharpe=float('nan'), max_drawdown=0.0,
                win_rate=float('nan'), num_trades=0, avg_holding_days=float('nan'),
                zscore_window=20, entry_threshold=2.0, exit_threshold=0.5,
            )

        return best
