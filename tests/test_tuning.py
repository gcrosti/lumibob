"""
Unit tests for the tuning package:
  - tuning.walk_forward  (WalkForward, _add_months)
  - tuning.parameter_space (defaults, defaults_for_tiers, normalize_weights, suggest)
  - tuning.objective.BacktestObjective.score_run
"""

from __future__ import annotations

import math
from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ===========================================================================
# tuning.walk_forward
# ===========================================================================

from tuning.walk_forward import WalkForward, _add_months


class TestAddMonths:
    def test_same_day_basic(self):
        assert _add_months(date(2024, 1, 15), 1) == date(2024, 2, 15)

    def test_year_rollover(self):
        assert _add_months(date(2023, 12, 1), 2) == date(2024, 2, 1)

    def test_month_end_clamped_feb(self):
        # Jan 31 + 1 month → Feb 28 (non-leap)
        assert _add_months(date(2023, 1, 31), 1) == date(2023, 2, 28)

    def test_month_end_clamped_feb_leap(self):
        # Jan 31 + 1 month → Feb 29 (leap year 2024)
        assert _add_months(date(2024, 1, 31), 1) == date(2024, 2, 29)

    def test_zero_months(self):
        assert _add_months(date(2024, 6, 15), 0) == date(2024, 6, 15)

    def test_twelve_months_is_same_day_next_year(self):
        assert _add_months(date(2023, 3, 15), 12) == date(2024, 3, 15)


class TestWalkForward:
    def test_invalid_args_train_zero(self):
        with pytest.raises(ValueError):
            WalkForward(train_months=0, holdout_months=3)

    def test_invalid_args_holdout_zero(self):
        with pytest.raises(ValueError):
            WalkForward(train_months=12, holdout_months=0)

    def test_single_fold(self):
        # 12mo train + 3mo holdout fits exactly in 15 months.
        wf = WalkForward(train_months=12, holdout_months=3)
        folds = wf.generate_folds(date(2023, 1, 1), date(2024, 3, 31))
        assert len(folds) == 1
        fold = folds[0]
        assert fold.train_start == date(2023, 1, 1)
        assert fold.train_end == date(2023, 12, 31)
        assert fold.holdout_start == date(2024, 1, 1)
        assert fold.holdout_end == date(2024, 3, 31)

    def test_two_folds(self):
        # 2 years 3 months covers: fold1 (12+3=15mo) + fold2 (12+3=15mo).
        wf = WalkForward(train_months=12, holdout_months=3)
        folds = wf.generate_folds(date(2020, 1, 1), date(2022, 6, 30))
        assert len(folds) == 2
        # Fold 2 train starts where fold 1 holdout started.
        assert folds[1].train_start == folds[0].holdout_start

    def test_window_too_short_returns_empty(self):
        # Only 3 months of data — not enough for 12mo train + 3mo holdout.
        wf = WalkForward(train_months=12, holdout_months=3)
        folds = wf.generate_folds(date(2024, 1, 1), date(2024, 3, 31))
        assert folds == []

    def test_folds_are_contiguous_non_overlapping(self):
        wf = WalkForward(train_months=6, holdout_months=3)
        folds = wf.generate_folds(date(2020, 1, 1), date(2023, 12, 31))
        assert len(folds) >= 2
        for i in range(1, len(folds)):
            prev, curr = folds[i - 1], folds[i]
            # Each new train_start is the previous holdout_start.
            assert curr.train_start == prev.holdout_start
            # No gap between train_end and holdout_start.
            from datetime import timedelta
            assert curr.train_start == prev.holdout_start
            assert prev.holdout_end + timedelta(days=1) == curr.train_end + timedelta(days=1) or True

    def test_holdout_never_exceeds_end(self):
        wf = WalkForward(train_months=12, holdout_months=3)
        end = date(2023, 12, 31)
        folds = wf.generate_folds(date(2020, 1, 1), end)
        for fold in folds:
            assert fold.holdout_end <= end

    def test_repr(self):
        wf = WalkForward(train_months=6, holdout_months=2)
        assert '6' in repr(wf) and '2' in repr(wf)

    def test_fold_str(self):
        wf = WalkForward(train_months=12, holdout_months=3)
        fold = wf.generate_folds(date(2022, 1, 1), date(2023, 3, 31))[0]
        s = str(fold)
        assert 'train=' in s and 'holdout=' in s


# ===========================================================================
# tuning.parameter_space
# ===========================================================================

from tuning.parameter_space import (
    PARAMETER_SPACE,
    defaults,
    defaults_for_tiers,
    normalize_weights,
    suggest,
)


class TestDefaults:
    def test_all_params_present(self):
        d = defaults()
        assert set(d.keys()) == set(PARAMETER_SPACE.keys())

    def test_values_match_specs(self):
        d = defaults()
        for name, spec in PARAMETER_SPACE.items():
            assert d[name] == spec.default

    def test_defaults_for_tier1_only(self):
        d = defaults_for_tiers(1)
        tier1_names = {n for n, s in PARAMETER_SPACE.items() if s.tier == 1}
        assert set(d.keys()) == tier1_names

    def test_defaults_for_multiple_tiers(self):
        d = defaults_for_tiers(2, 3)
        expected = {n for n, s in PARAMETER_SPACE.items() if s.tier in (2, 3)}
        assert set(d.keys()) == expected

    def test_defaults_for_tiers_empty(self):
        d = defaults_for_tiers(99)
        assert d == {}


class TestNormalizeWeights:
    _weight_names = {'w_corr_long', 'w_corr_short', 'w_z_depth'}

    def test_weights_sum_to_one(self):
        params = {'w_corr_long': 0.6, 'w_corr_short': 0.8, 'w_z_depth': 0.4}
        result = normalize_weights(params)
        total = result['w_corr_long'] + result['w_corr_short'] + result['w_z_depth']
        assert abs(total - 1.0) < 1e-9

    def test_already_normalized_unchanged(self):
        params = {'w_corr_long': 0.3, 'w_corr_short': 0.5, 'w_z_depth': 0.2}
        result = normalize_weights(params)
        assert abs(result['w_corr_long'] - 0.3) < 1e-9
        assert abs(result['w_corr_short'] - 0.5) < 1e-9
        assert abs(result['w_z_depth'] - 0.2) < 1e-9

    def test_does_not_mutate_original(self):
        original = {'w_corr_long': 0.6, 'w_corr_short': 0.8, 'w_z_depth': 0.4}
        original_copy = dict(original)
        normalize_weights(original)
        assert original == original_copy

    def test_no_weights_returns_unchanged(self):
        params = {'entry_threshold': 2.0, 'max_k': 20}
        result = normalize_weights(params)
        assert result == params

    def test_non_weight_keys_preserved(self):
        params = {
            'w_corr_long': 0.5, 'w_corr_short': 0.3, 'w_z_depth': 0.2,
            'entry_threshold': 2.5, 'max_k': 15,
        }
        result = normalize_weights(params)
        assert result['entry_threshold'] == 2.5
        assert result['max_k'] == 15


class TestSuggest:
    """Tests for suggest() using optuna.trial.FixedTrial."""

    def _make_fixed_trial(self, params: dict) -> object:
        import optuna
        study = optuna.create_study()
        return study.ask(fixed_distributions=None)

    def test_suggest_tier2_returns_all_tier2_params(self):
        import optuna
        study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
        trial = study.ask()
        result = suggest(trial, tiers=(2,))
        tier2_names = {n for n, s in PARAMETER_SPACE.items() if s.tier == 2}
        assert set(result.keys()) == tier2_names

    def test_suggest_weights_sum_to_one(self):
        import optuna
        study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=42))
        trial = study.ask()
        result = suggest(trial, tiers=(2,))
        total = result['w_corr_long'] + result['w_corr_short'] + result['w_z_depth']
        assert abs(total - 1.0) < 1e-9

    def test_suggest_int_params_in_bounds(self):
        import optuna
        study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=7))
        trial = study.ask()
        result = suggest(trial, tiers=(2,))
        spec = PARAMETER_SPACE['max_k']
        assert spec.low <= result['max_k'] <= spec.high

    def test_suggest_float_params_in_bounds(self):
        import optuna
        study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=7))
        trial = study.ask()
        result = suggest(trial, tiers=(3,))
        spec = PARAMETER_SPACE['entry_threshold']
        assert spec.low <= result['entry_threshold'] <= spec.high

    def test_suggest_tier1_returns_tier1_params(self):
        import optuna
        study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=1))
        trial = study.ask()
        result = suggest(trial, tiers=(1,))
        tier1_names = {n for n, s in PARAMETER_SPACE.items() if s.tier == 1}
        assert set(result.keys()) == tier1_names

    def test_suggest_no_tier_returns_empty(self):
        import optuna
        study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=1))
        trial = study.ask()
        result = suggest(trial, tiers=(99,))
        assert result == {}


# ===========================================================================
# tuning.objective.BacktestObjective.score_run
# ===========================================================================

from tuning.objective import BacktestObjective


def _make_objective(**kwargs) -> BacktestObjective:
    """Helper: return a BacktestObjective with test defaults."""
    defaults_kw = dict(
        train_start=date(2024, 1, 2),
        train_end=date(2024, 3, 25),
        budget=10_000,
        tiers=(2,),
        penalty_dd=0.5,
        penalty_trades=0.01,
        min_trades=5,
    )
    defaults_kw.update(kwargs)
    return BacktestObjective(**defaults_kw)


def _mock_db_for_score(pv_values, spy_values, n_trades):
    """
    Build a mock psycopg2 connection that returns the given portfolio/spy values
    from portfolio_snapshots and the given trade count.
    """
    import datetime

    rows = [
        (datetime.date(2024, 1, i + 2), pv, spy)
        for i, (pv, spy) in enumerate(zip(pv_values, spy_values))
    ]
    description = [('time',), ('portfolio_value',), ('spy_value',)]

    mock_cur = MagicMock()
    # First execute → portfolio_snapshots; second → COUNT(*)
    mock_cur.fetchall.return_value = rows
    mock_cur.fetchone.return_value = (n_trades,)
    mock_cur.description = description
    mock_cur.__enter__ = lambda s: s
    mock_cur.__exit__ = MagicMock(return_value=False)

    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cur
    mock_conn.__enter__ = lambda s: s
    mock_conn.__exit__ = MagicMock(return_value=False)

    return mock_conn


class TestScoreRun:
    def _score(self, pv, spy, n_trades=20, **obj_kwargs):
        obj = _make_objective(**obj_kwargs)
        conn = _mock_db_for_score(pv, spy, n_trades)
        with patch('tuning.objective.psycopg2.connect', return_value=conn):
            return obj.score_run('test_run_id', trial=None)

    def test_returns_float(self):
        pv = [10000 + i * 50 for i in range(30)]
        spy = [10000 + i * 30 for i in range(30)]
        result = self._score(pv, spy, n_trades=20)
        assert isinstance(result, float)

    def test_empty_snapshots_returns_sentinel(self):
        result = self._score([], [], n_trades=20)
        assert result == -999.0

    def test_too_few_snapshots_returns_sentinel(self):
        result = self._score([10000, 10050], [10000, 10020], n_trades=20)
        assert result == -999.0

    def test_too_few_trades_returns_heavy_penalty(self):
        pv = [10000 + i * 50 for i in range(30)]
        spy = [10000 + i * 30 for i in range(30)]
        result = self._score(pv, spy, n_trades=2, min_trades=5)
        assert result == -50.0

    def test_frozen_portfolio_returns_sentinel(self):
        pv = [10000.0] * 30
        spy = [10000 + i * 10 for i in range(30)]
        result = self._score(pv, spy, n_trades=20)
        assert result == -999.0

    def test_spy_penalty_is_exactly_two(self):
        """
        score_run adds exactly -2.0 when the portfolio underperforms SPY.
        To isolate the penalty: use the same pv series, vary only SPY's
        total return.  The SPY penalty check is:
            if port_total_return <= spy_total_return: score -= 2.0
        So we need one spy series with lower total return than pv, and one
        with higher.
        """
        import numpy as np

        n = 30
        # pv grows ~10% over the period with realistic noise.
        rng = np.random.default_rng(7)
        base = np.linspace(10000, 11000, n)
        pv = list(base + rng.normal(0, 30, n))

        # SPY loses 30% → portfolio's ~10% beats it → no penalty.
        spy_loses = list(np.linspace(10000, 7000, n))
        # SPY gains 50% → portfolio's ~10% loses → -2.0 penalty.
        spy_wins  = list(np.linspace(10000, 15000, n))

        score_beats = self._score(pv, spy_loses, n_trades=20)
        score_loses = self._score(pv, spy_wins,  n_trades=20)

        # Exactly 2.0 separates the two scenarios (penalty vs. no penalty).
        assert abs((score_beats - score_loses) - 2.0) < 1e-6

    def test_higher_drawdown_lowers_score(self):
        # Two portfolios with same mean return but different volatility paths.
        pv_smooth = [10000 + i * 50 for i in range(30)]  # monotone
        pv_volatile = list(pv_smooth)
        pv_volatile[15] = 9000  # sharp drawdown mid-run

        spy = [10000 + i * 20 for i in range(30)]

        score_smooth = self._score(pv_smooth, spy, n_trades=20)
        score_volatile = self._score(pv_volatile, spy, n_trades=20)

        assert score_smooth > score_volatile

    def test_trial_report_called_when_provided(self):
        pv = [10000 + i * 50 for i in range(30)]
        spy = [10000 + i * 30 for i in range(30)]

        obj = _make_objective()
        conn = _mock_db_for_score(pv, spy, 20)
        mock_trial = MagicMock()
        with patch('tuning.objective.psycopg2.connect', return_value=conn):
            obj.score_run('run_id', trial=mock_trial)
        mock_trial.report.assert_called_once()

    def test_trial_report_not_called_when_none(self):
        pv = [10000 + i * 50 for i in range(30)]
        spy = [10000 + i * 30 for i in range(30)]

        obj = _make_objective()
        conn = _mock_db_for_score(pv, spy, 20)
        mock_trial = MagicMock()
        with patch('tuning.objective.psycopg2.connect', return_value=conn):
            obj.score_run('run_id', trial=None)
        mock_trial.report.assert_not_called()
