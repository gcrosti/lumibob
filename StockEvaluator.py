class StockEvaluator:
    """
    Evaluates the relationship between two stocks
    """
    def get_correlation(self, lead_stock, lag_stock, lag):
        """
        evaluates the correlation between two stocks,
        after applying a lag to lead stock
        """
        shifted_lead = lead_stock.shift(lag)
        return shifted_lead.corr(lag_stock)

    def get_action(self, lead_stock, lag_stock, lag):
        """
        determines whether to buy lag stock based on lead stock
        """
        pass

    def find_optimal_moving_average(self, simulator, lag = 1):
        """
        finds the moving average window between two stocks that will generate
        the highest return when running a full cash switching strategy
        returns short moving average, long moving average tuple
        """
        max_return = 0.0
        short_moving_average = 1
        long_moving_average = 2

        for short_ma in range(1,5):
            for long_ma in range(short_ma + 1,6):
                r = simulator.run_simulation(short_ma,long_ma,lag=lag)[0]
                if r > max_return:
                    max_return = r
                    short_moving_average = short_ma
                    long_moving_average = long_ma

        return short_moving_average, long_moving_average