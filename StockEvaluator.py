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
