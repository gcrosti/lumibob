import numpy as np
from statsmodels.tsa.stattools import coint


class StockEvaluator:
    """
    Evaluates the relationship between two stocks
    """
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

        log_lead = np.log(lead.clip(lower=1e-9))
        log_lag = np.log(lag.clip(lower=1e-9))

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

    def get_zscore_action(
        self,
        lead: 'pd.Series',
        lag: 'pd.Series',
        window: int,
        entry_threshold: float,
        exit_threshold: float,
    ) -> str:
        """
        Return the trading action for the lag stock based on the current Z-score.

        A very negative Z-score means the lag stock is unusually cheap relative
        to the lead — we buy expecting mean reversion upward.  Once the spread
        reverts past exit_threshold (closer to zero or above), we sell.

        Returns:
            'buy'  when zscore < -entry_threshold
            'sell' when zscore > -exit_threshold
            'hold' otherwise (spread within normal range, or Z-score is NaN)
        """
        zscore = self.compute_zscore(lead, lag, window)
        if zscore.empty:
            return 'hold'
        z = float(zscore.iloc[-1])
        if np.isnan(z):
            return 'hold'
        if z < -entry_threshold:
            return 'buy'
        if z > -exit_threshold:
            return 'sell'
        return 'hold'

    def is_cointegrated(self, lead_stock, lag_stock, p_threshold: float = 0.05) -> bool:
        """
        Test whether two price series are cointegrated using the Engle-Granger
        two-step method. Returns True if the p-value from the cointegration test
        is below p_threshold, indicating a statistically significant long-run
        equilibrium relationship between the two series.

        Cointegration is independent of the MA-crossover signal — it validates
        that the lead/lag relationship is structural rather than coincidental,
        complementing the Pearson correlation pre-filter.

        Returns False when the test cannot be computed (e.g. insufficient data,
        all-NaN series) so callers can treat it as a safe rejection.
        """
        try:
            lead_clean = lead_stock.dropna()
            lag_clean = lag_stock.dropna()
            common_index = lead_clean.index.intersection(lag_clean.index)
            if len(common_index) < 10:
                return False
            _, p_value, _ = coint(lead_clean.loc[common_index], lag_clean.loc[common_index])
            return float(p_value) < p_threshold
        except Exception:
            return False
