from typing import NamedTuple

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint


class SpreadScores(NamedTuple):
    """Cointegration and mean-reversion quality scores for a pair."""
    coint_score: float      # 0–1; higher = stronger cointegration
    halflife_score: float   # 0–1; higher = faster mean-reversion
    coint_pvalue: float     # raw ADF p-value (stored for caching; not a gate)
    halflife_days: float | None  # AR(1) half-life in trading days; None if non-stationary


class EntryMetrics(NamedTuple):
    """Everything the entry criteria need, from one spread computation.

    ``expected_gross_bps`` is the quantity the Z-score normalises away: how
    many basis points reverting from here to the exit threshold is worth.  The
    Z-score says how *stretched* a spread is relative to its own history;
    this says how much that stretch is worth in money, which is what has to
    clear trading costs.  See
    docs/deepdives/2026-08-01_score-selection-and-the-missing-entry-gate.md.
    """
    z: float                  # signed Z-score at the latest bar
    spread_std_bps: float     # spread std over the Z window, bps of gross notional
    expected_gross_bps: float # (|z| - exit_threshold) * spread_std_bps, floored at 0


def halflife_to_score(halflife_days: float | None, max_halflife_days: float) -> float:
    """Map an AR(1) half-life (trading days) to a [0, 1] mean-reversion score.

    Log-spaced rather than linear.  Predicted half-lives cluster tightly at the
    low end (~1-3 days), and the old ``1 - hl / ceiling`` map collapsed that
    whole cluster to ~0.96 — a near-constant component with no discriminatory
    power (see docs/deepdives/2026-07-17_pass-a-score-signal-and-exploitability).
    In log space a 1-day half-life scores 1.0, ``max_halflife_days`` scores 0.0,
    and the dense low-half-life region is spread across the range so the
    component actually varies pair-to-pair.

    Returns 0.0 for missing / non-finite / non-stationary half-lives, and for a
    degenerate ceiling (<= 1 day, where the log denominator collapses).
    """
    if halflife_days is None or not np.isfinite(halflife_days):
        return 0.0
    ceil_log = np.log(float(max_halflife_days))
    if ceil_log <= 0.0:
        return 0.0
    hl = min(max(float(halflife_days), 1.0), float(max_halflife_days))
    return float(np.clip((ceil_log - np.log(hl)) / ceil_log, 0.0, 1.0))


class StockEvaluator:
    """
    Evaluates the relationship between two stocks
    """

    def get_correlation_dual(
        self,
        lead: pd.Series,
        lag: pd.Series,
        long_window: int = 90,
        short_window: int = 20,
    ) -> tuple[float, float]:
        """
        Return trailing Pearson correlations at two horizons computed on
        **log-returns** (not raw prices), which is regime-appropriate for
        detecting current co-movement.

        Returns (corr_long, corr_short).  Either value may be NaN when
        there are insufficient overlapping observations.
        """
        log_lead = np.log(lead.astype(float).clip(lower=1e-9))
        log_lag = np.log(lag.astype(float).clip(lower=1e-9))
        lr_lead = log_lead.diff().dropna()
        lr_lag = log_lag.diff().dropna()
        common = lr_lead.index.intersection(lr_lag.index)
        ll = lr_lead.loc[common]
        lg = lr_lag.loc[common]

        corr_long = (
            ll.iloc[-long_window:].corr(lg.iloc[-long_window:])
            if len(common) >= long_window else float('nan')
        )
        corr_short = (
            ll.iloc[-short_window:].corr(lg.iloc[-short_window:])
            if len(common) >= short_window else float('nan')
        )
        return corr_long, corr_short

    def compute_z_depth(
        self,
        lead: pd.Series,
        lag: pd.Series,
        window: int = 20,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
    ) -> tuple[float, float | None]:
        """
        Continuous [0, 1] score measuring how far the spread has diverged.

        NOT ON THE LIVE PATH since 2026-08-01 — retained for the cached replay
        studies in tuning/studies/, whose caches carry a z_depth column.
        BobsBrain uses ``compute_entry_metrics`` instead.  The reason: this test
        is on *signed* z, so a spread at z=+5 scores 0.0 ("not diverged"), which
        makes it a direction flag rather than a depth measure.  Among candidates
        that pass a ``z <= -entry_threshold`` gate it is identically 1.0 — a
        constant, carrying no ranking information.  See
        docs/deepdives/2026-08-01_score-selection-and-the-missing-entry-gate.md §3c.

        Returns (z_depth, raw_z).  z_depth is 0.0 when the spread has not
        diverged past exit_threshold, scales linearly to 1.0 at entry_threshold,
        and is clamped at 1.0 beyond that.
        """
        zscore = self.compute_zscore(lead, lag, window)
        if zscore.empty:
            return 0.0, None
        z = float(zscore.iloc[-1])
        if np.isnan(z):
            return 0.0, None
        if z >= -exit_threshold:
            return 0.0, z
        depth = min((-z - exit_threshold) / (entry_threshold - exit_threshold), 1.0)
        return max(depth, 0.0), z

    def get_correlation(self, lead_stock, lag_stock, lag):
        """
        Evaluates the Pearson correlation between two stocks after applying a
        lag to the lead stock.

        Returns float('nan') when the correlation cannot be computed (e.g.
        insufficient overlapping data, zero-variance series). Callers must
        guard against NaN before comparing to a threshold.
        """
        shifted_lead = lead_stock.shift(lag)
        return shifted_lead.corr(lag_stock)

    def get_action(self, lead_stock, lag_stock, lag, short_ma=2, long_ma=5):
        """
        determines whether to buy or sell lag stock based on MA crossover of lead stock.
        short MA above long MA on lead stock signals upward momentum -> buy lag stock.
        """
        shifted_lead = lead_stock.shift(lag)
        short = shifted_lead.rolling(window=short_ma, min_periods=1).mean()
        long = shifted_lead.rolling(window=long_ma, min_periods=1).mean()
        if short.iloc[-1] > long.iloc[-1]:
            return 'buy'
        elif short.iloc[-1] < long.iloc[-1]:
            return 'sell'
        return 'hold'

    def compute_zscore(self, lead: 'pd.Series', lag: 'pd.Series', window: int = 20) -> 'pd.Series':
        """
        Compute the rolling Z-score of the log-price spread between a cointegrated pair.

        The spread is defined as:
            spread = log(lag) - hedge_ratio * log(lead)
        where hedge_ratio is the OLS slope from regressing log(lag) on log(lead)
        over the full available window.  This is the same residual that the
        Engle-Granger cointegration test uses internally.

        The Z-score normalizes the spread using a rolling mean and std of length
        `window`, so the value reflects how many standard deviations the current
        spread has deviated from its recent equilibrium.

        Returns a pd.Series of Z-scores aligned to the input index.
        Returns a series of NaN when inputs are too short or all-constant.
        """
        import pandas as pd

        log_lead = np.log(lead.astype(float).clip(lower=1e-9))
        log_lag = np.log(lag.astype(float).clip(lower=1e-9))

        common = log_lead.index.intersection(log_lag.index)
        if len(common) < window + 2:
            return pd.Series(np.nan, index=common)

        ll = log_lead.loc[common]
        lg = log_lag.loc[common]

        try:
            hedge = float(np.polyfit(ll.values, lg.values, 1)[0])
        except (np.linalg.LinAlgError, ValueError):
            return pd.Series(np.nan, index=common)

        spread = lg - hedge * ll
        roll_mean = spread.rolling(window=window, min_periods=window).mean()
        roll_std = spread.rolling(window=window, min_periods=window).std()
        zscore = (spread - roll_mean) / roll_std.replace(0, np.nan)
        return zscore

    def compute_entry_metrics(
        self,
        lead: 'pd.Series',
        lag: 'pd.Series',
        window: int = 20,
        exit_threshold: float = 0.5,
    ) -> 'EntryMetrics | None':
        """
        Signed Z-score plus the spread's absolute scale, in one pass.

        ``compute_zscore`` divides the spread by its rolling standard deviation
        and discards that denominator.  The denominator is the size of a typical
        dislocation in log-spread units; scaled by ``1 / (1 + |hedge|)`` and
        1e4 it becomes bps of the pair's gross notional — directly comparable
        to round-trip costs.

        Returns None when the inputs are too short, the hedge fit fails, or the
        latest Z-score / std is not finite.
        """
        log_lead = np.log(lead.astype(float).clip(lower=1e-9))
        log_lag = np.log(lag.astype(float).clip(lower=1e-9))
        common = log_lead.index.intersection(log_lag.index)
        if len(common) < window + 2:
            return None
        ll, lg = log_lead.loc[common], log_lag.loc[common]
        try:
            hedge = float(np.polyfit(ll.values, lg.values, 1)[0])
        except (np.linalg.LinAlgError, ValueError):
            return None

        spread = lg - hedge * ll
        roll_mean = spread.rolling(window=window, min_periods=window).mean()
        roll_std = spread.rolling(window=window, min_periods=window).std()
        std = float(roll_std.iloc[-1])
        if not np.isfinite(std) or std <= 0:
            return None
        z = float((spread.iloc[-1] - roll_mean.iloc[-1]) / std)
        if not np.isfinite(z):
            return None

        # bps of gross notional: both legs are transacted, so the move is
        # scaled by the total notional (1 long + |hedge| short).
        std_bps = std / (1.0 + abs(hedge)) * 1e4
        expected = max(abs(z) - exit_threshold, 0.0) * std_bps
        return EntryMetrics(z=z, spread_std_bps=std_bps, expected_gross_bps=expected)

    def get_zscore_action(
        self,
        lead: 'pd.Series',
        lag: 'pd.Series',
        window: int,
        entry_threshold: float,
        exit_threshold: float,
    ) -> tuple[str, float | None]:
        """
        Return the trading action and current Z-score for the lag stock.

        A very negative Z-score means the lag stock is unusually cheap relative
        to the lead — we buy expecting mean reversion upward.  Once the spread
        reverts past exit_threshold (closer to zero or above), we sell.

        Returns a (action, current_zscore) tuple where:
            action        -- 'buy', 'sell', or 'hold'
            current_zscore -- the most recent Z-score value, or None if NaN/empty

        Returning both values from a single compute_zscore call avoids the
        caller having to invoke compute_zscore a second time to read the z value.
        """
        zscore = self.compute_zscore(lead, lag, window)
        if zscore.empty:
            return 'hold', None
        z = float(zscore.iloc[-1])
        if np.isnan(z):
            return 'hold', None
        if z < -entry_threshold:
            return 'buy', z
        if z > -exit_threshold:
            return 'sell', z
        return 'hold', z

    def is_cointegrated(self, lead_stock, lag_stock, p_threshold: float = 0.05) -> bool:
        """
        Test whether two price series are cointegrated using the Engle-Granger
        two-step method on log-prices (consistent with compute_zscore).

        Returns True if the p-value is below p_threshold.  Returns False when
        the test cannot be computed (insufficient data, all-NaN series).
        """
        try:
            lead_clean = lead_stock.dropna()
            lag_clean = lag_stock.dropna()
            common_index = lead_clean.index.intersection(lag_clean.index)
            if len(common_index) < 10:
                return False
            log_lead = np.log(lead_clean.loc[common_index].astype(float).clip(lower=1e-9))
            log_lag = np.log(lag_clean.loc[common_index].astype(float).clip(lower=1e-9))
            _, p_value, _ = coint(log_lead, log_lag)
            return float(p_value) < p_threshold
        except Exception:
            return False

    def compute_spread_scores(
        self,
        lead: pd.Series,
        lag: pd.Series,
        coint_pvalue_ceiling: float = 0.20,
        max_halflife_days: float = 60.0,
    ) -> SpreadScores:
        """
        Score a pair's cointegration quality and mean-reversion speed.

        Both series are log-transformed, matching the spread definition in
        compute_zscore.  The cointegration test runs on log-prices; the
        half-life is derived from an AR(1) fit on the OLS residual spread.

        Parameters
        ----------
        coint_pvalue_ceiling:
            Fixed normalisation constant (not tunable).  Pairs with p-value
            at or above this ceiling score 0 on cointegration.
        max_halflife_days:
            Half-lives at or above this value score 0.  Should match
            BobsBrain.max_halflife_days so scores are comparable across pairs.

        Returns
        -------
        SpreadScores with (coint_score, halflife_score, coint_pvalue, halflife_days).
        Returns SpreadScores(0.0, 0.0, 1.0, None) on any error or bad input.
        """
        _null = SpreadScores(0.0, 0.0, 1.0, None)
        try:
            lead_clean = lead.dropna()
            lag_clean = lag.dropna()
            common = lead_clean.index.intersection(lag_clean.index)
            if len(common) < 20:
                return _null

            log_lead = np.log(lead_clean.loc[common].astype(float).clip(lower=1e-9))
            log_lag = np.log(lag_clean.loc[common].astype(float).clip(lower=1e-9))

            _, p_value, _ = coint(log_lead.values, log_lag.values)
            p_value = float(p_value)
            coint_score = max(0.0, 1.0 - p_value / coint_pvalue_ceiling)

            # AR(1) half-life on the OLS residual spread (same OLS as compute_zscore)
            try:
                hedge = float(np.polyfit(log_lead.values, log_lag.values, 1)[0])
            except (np.linalg.LinAlgError, ValueError):
                return SpreadScores(coint_score, 0.0, p_value, None)

            spread = (log_lag - hedge * log_lead).values
            if len(spread) < 3:
                return SpreadScores(coint_score, 0.0, p_value, None)

            # rho from OLS of spread[t] on spread[t-1]
            y = spread[1:]
            x = spread[:-1]
            rho = float(np.polyfit(x, y, 1)[0])

            abs_rho = abs(rho)
            if abs_rho >= 1.0:
                return SpreadScores(coint_score, 0.0, p_value, None)

            halflife_days = float(-np.log(2) / np.log(abs_rho))
            halflife_days = max(1.0, min(halflife_days, 252.0))
            halflife_score = halflife_to_score(halflife_days, max_halflife_days)

            return SpreadScores(coint_score, halflife_score, p_value, halflife_days)

        except Exception:
            return _null
