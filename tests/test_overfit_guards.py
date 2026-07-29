"""Tests for tuning/overfit_guards.py — synthetic data with KNOWN answers.

No DB, no network.  Each guard is exercised with a dataset whose correct verdict is
known by construction, and we assert the guard recovers it.
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tuning'))

from tuning.overfit_guards import (
    all_null,
    complexity_ratio,
    holdout_gap,
    null_baseline,
    report_panel,
    seed_stability,
    unit_level_signal,
)


class TestUnitLevelSignal(unittest.TestCase):
    def _make_df(self, seed: int, signal_strength: float, n_per_group: int = 300):
        """Two groups; `sig` genuinely drives the outcome, `noise` is independent."""
        rng = np.random.default_rng(seed)
        frames = []
        for grp in ('a', 'b'):
            sig = rng.normal(0, 1, n_per_group)
            noise = rng.normal(0, 1, n_per_group)
            outcome = signal_strength * sig + rng.normal(0, 1, n_per_group)
            frames.append(pd.DataFrame({
                'fold': grp, 'sig': sig, 'noise': noise, 'forward_gross': outcome,
            }))
        return pd.concat(frames, ignore_index=True)

    def test_signal_component_ci_excludes_zero_correct_sign(self):
        df = self._make_df(seed=1, signal_strength=0.8)
        res = unit_level_signal(df, ['sig', 'noise'], 'forward_gross', 'fold',
                                n_boot=400, seed=0)
        for grp in ('a', 'b'):
            rho, lo, hi = res[grp]['sig']
            self.assertGreater(rho, 0.0)            # correct positive sign
            self.assertGreater(lo, 0.0)             # CI strictly excludes 0
            self.assertGreater(hi, lo)

    def test_noise_component_ci_includes_zero(self):
        df = self._make_df(seed=1, signal_strength=0.8)
        res = unit_level_signal(df, ['sig', 'noise'], 'forward_gross', 'fold',
                                n_boot=400, seed=0)
        for grp in ('a', 'b'):
            _rho, lo, hi = res[grp]['noise']
            self.assertLessEqual(lo, 0.0)
            self.assertGreaterEqual(hi, 0.0)        # CI straddles 0

    def test_all_null_false_when_signal_present(self):
        df = self._make_df(seed=1, signal_strength=0.8)
        res = unit_level_signal(df, ['sig', 'noise'], 'forward_gross', 'fold',
                                n_boot=400, seed=0)
        self.assertFalse(all_null(res))

    def test_all_null_true_for_pure_noise_panel(self):
        rng = np.random.default_rng(7)
        frames = []
        for grp in ('a', 'b'):
            n = 300
            frames.append(pd.DataFrame({
                'fold': grp,
                'c1': rng.normal(0, 1, n),
                'c2': rng.normal(0, 1, n),
                'forward_gross': rng.normal(0, 1, n),   # independent of components
            }))
        df = pd.concat(frames, ignore_index=True)
        res = unit_level_signal(df, ['c1', 'c2'], 'forward_gross', 'fold',
                                n_boot=400, seed=0)
        self.assertTrue(all_null(res))

    def test_negative_signal_sign_recovered(self):
        rng = np.random.default_rng(3)
        n = 400
        sig = rng.normal(0, 1, n)
        df = pd.DataFrame({
            'fold': 'a', 'sig': sig,
            'forward_gross': -0.9 * sig + rng.normal(0, 1, n),
        })
        res = unit_level_signal(df, ['sig'], 'forward_gross', 'fold', n_boot=400, seed=0)
        rho, lo, hi = res['a']['sig']
        self.assertLess(rho, 0.0)
        self.assertLess(hi, 0.0)                    # CI strictly below 0
        self.assertFalse(all_null(res))

    def test_deterministic_under_seed(self):
        df = self._make_df(seed=1, signal_strength=0.8)
        a = unit_level_signal(df, ['sig'], 'forward_gross', 'fold', n_boot=200, seed=42)
        b = unit_level_signal(df, ['sig'], 'forward_gross', 'fold', n_boot=200, seed=42)
        self.assertEqual(a['a']['sig'], b['a']['sig'])


class TestHoldoutGap(unittest.TestCase):
    def _groups(self):
        """Three groups; in each, y = 2*x_good + noise (x_bad is pure noise)."""
        rng = np.random.default_rng(11)
        groups = {}
        for name in ('f1', 'f2', 'f3'):
            n = 200
            x_good = rng.normal(0, 1, n)
            x_bad = rng.normal(0, 1, n)
            y = 2.0 * x_good + rng.normal(0, 0.5, n)
            groups[name] = pd.DataFrame({'x_good': x_good, 'x_bad': x_bad, 'y': y})
        return groups

    def test_generalizing_fit_has_small_gap_positive_holdout(self):
        groups = self._groups()

        def fit(train):
            # OLS slope of y on the genuinely predictive feature.
            b = np.polyfit(train['x_good'], train['y'], 1)[0]
            return {'b': b}

        def evalf(df, params):
            pred = params['b'] * df['x_good']
            # R^2-style score: 1 - SSE/SST (higher is better, generalizes across folds).
            sse = float(np.sum((df['y'] - pred) ** 2))
            sst = float(np.sum((df['y'] - df['y'].mean()) ** 2))
            return 1.0 - sse / sst

        res = holdout_gap(fit, evalf, groups)
        self.assertGreater(res['mean_holdout'], 0.5)      # generalizes
        self.assertLess(abs(res['mean_gap']), 0.15)       # small train/holdout gap

    def test_overfitting_fit_has_large_gap_negative_holdout(self):
        groups = self._groups()

        def fit(train):
            # Memorize the training pool by fitting a high-degree polynomial on the
            # NOISE feature — great in-sample, useless out-of-sample.
            coeffs = np.polyfit(train['x_bad'], train['y'], 12)
            return {'coeffs': coeffs}

        def evalf(df, params):
            pred = np.polyval(params['coeffs'], df['x_bad'])
            sse = float(np.sum((df['y'] - pred) ** 2))
            sst = float(np.sum((df['y'] - df['y'].mean()) ** 2))
            return 1.0 - sse / sst

        res = holdout_gap(fit, evalf, groups)
        self.assertLess(res['mean_holdout'], 0.0)         # negative held-out score
        self.assertGreater(res['mean_gap'], 0.3)          # large positive gap

    def test_requires_two_groups(self):
        with self.assertRaises(ValueError):
            holdout_gap(lambda d: None, lambda d, p: 0.0, {'only': pd.DataFrame({'y': [1]})})


class TestNullBaseline(unittest.TestCase):
    def _make(self, seed: int, predictive: bool, n_dates: int = 8, n_per: int = 60):
        """Panel of (fold,date) cells; score either ranks the outcome or is random."""
        rng = np.random.default_rng(seed)
        rows = []
        for d in range(n_dates):
            outcome = rng.normal(0, 1, n_per)
            if predictive:
                score = outcome + rng.normal(0, 0.3, n_per)   # score ~ outcome
            else:
                score = rng.normal(0, 1, n_per)               # score independent
            for i in range(n_per):
                rows.append({'fold': 'f1', 'date': d,
                             'forward_gross': outcome[i], 'score': score[i]})
        return pd.DataFrame(rows)

    def test_predictive_score_high_percentile(self):
        df = self._make(seed=5, predictive=True)
        res = null_baseline(df, 'fold', 'date', 'forward_gross', df['score'], k=10,
                            n_perm=300, seed=0)
        self.assertGreater(res['percentile'], 95.0)          # vs random-k
        self.assertGreater(res['shuffle']['percentile'], 95.0)
        self.assertGreater(res['z'], 2.0)

    def test_random_score_near_fiftieth_percentile(self):
        df = self._make(seed=5, predictive=False)
        res = null_baseline(df, 'fold', 'date', 'forward_gross', df['score'], k=10,
                            n_perm=300, seed=0)
        self.assertGreater(res['percentile'], 15.0)
        self.assertLess(res['percentile'], 85.0)             # indistinguishable from null

    def test_real_beats_null_mean_when_predictive(self):
        df = self._make(seed=9, predictive=True)
        res = null_baseline(df, 'fold', 'date', 'forward_gross', df['score'], k=10,
                            n_perm=200, seed=1)
        self.assertGreater(res['real'], res['null_mean'])

    def test_deterministic_under_seed(self):
        df = self._make(seed=5, predictive=True)
        a = null_baseline(df, 'fold', 'date', 'forward_gross', df['score'], k=10,
                          n_perm=100, seed=3)
        b = null_baseline(df, 'fold', 'date', 'forward_gross', df['score'], k=10,
                          n_perm=100, seed=3)
        self.assertEqual(a['real'], b['real'])
        self.assertEqual(a['percentile'], b['percentile'])


class TestSeedStability(unittest.TestCase):
    def test_stable_study_not_flagged(self):
        def study(seed):
            rng = np.random.default_rng(seed)
            return {'w': 0.50 + rng.normal(0, 0.005), 'objective': 1.0 + rng.normal(0, 0.01)}

        res = seed_stability(study, seeds=[0, 1, 2, 3, 4], param_keys=['w'], threshold=0.1)
        self.assertEqual(res['flagged'], [])
        self.assertFalse(res['per_key']['w']['flagged'])
        self.assertLess(res['per_key']['w']['range'], 0.1)

    def test_unstable_study_flagged(self):
        def study(seed):
            rng = np.random.default_rng(seed)
            return {'w': rng.uniform(0.0, 1.0), 'objective': rng.uniform(0.0, 1.0)}

        res = seed_stability(study, seeds=[0, 1, 2, 3, 4], param_keys=['w'], threshold=0.1)
        self.assertIn('w', res['flagged'])
        self.assertTrue(res['per_key']['w']['flagged'])
        self.assertGreater(res['per_key']['w']['range'], 0.1)

    def test_objective_spread_reported(self):
        def study(seed):
            return {'w': 0.5, 'objective': float(seed)}

        res = seed_stability(study, seeds=[0, 1, 2], param_keys=['w'])
        self.assertAlmostEqual(res['objective_spread'], 2.0)

    def test_per_key_threshold_mapping(self):
        def study(seed):
            rng = np.random.default_rng(seed)
            return {'tight': 0.5 + rng.normal(0, 0.001), 'loose': rng.uniform(0, 1)}

        res = seed_stability(study, seeds=[0, 1, 2, 3], param_keys=['tight', 'loose'],
                             threshold={'tight': 0.1, 'loose': 0.1})
        self.assertNotIn('tight', res['flagged'])
        self.assertIn('loose', res['flagged'])

    def test_no_threshold_flags_nothing(self):
        def study(seed):
            rng = np.random.default_rng(seed)
            return {'w': rng.uniform(0, 1)}

        res = seed_stability(study, seeds=[0, 1, 2], param_keys=['w'], threshold=None)
        self.assertEqual(res['flagged'], [])


class TestComplexityRatio(unittest.TestCase):
    def test_red_when_free_params_meet_or_exceed_units(self):
        self.assertEqual(complexity_ratio(6, 3)['flag'], 'red')       # the Pass A v4 case
        self.assertEqual(complexity_ratio(5, 5)['flag'], 'red')       # boundary: equal

    def test_amber_within_three_x(self):
        self.assertEqual(complexity_ratio(5, 6)['flag'], 'amber')     # 1 < ratio < 3
        self.assertEqual(complexity_ratio(4, 11)['flag'], 'amber')    # 2.75x

    def test_green_at_or_above_three_x(self):
        self.assertEqual(complexity_ratio(4, 12)['flag'], 'green')    # exactly 3x
        self.assertEqual(complexity_ratio(6, 600)['flag'], 'green')   # pair-level units

    def test_ratio_value(self):
        r = complexity_ratio(6, 3)
        self.assertAlmostEqual(r['ratio'], 0.5)
        self.assertEqual(r['n_free_params'], 6)
        self.assertEqual(r['n_independent_units'], 3)

    def test_zero_free_params_raises(self):
        with self.assertRaises(ValueError):
            complexity_ratio(0, 100)


class TestReportPanel(unittest.TestCase):
    def test_renders_all_sections_without_printing(self):
        complexity = complexity_ratio(6, 3)
        seed = {'flagged': ['w'], 'objective_spread': 0.9}
        holdout = {'mean_gap': 0.8, 'mean_holdout': -0.2, 'per_group': {}}
        null = {'real': 0.1, 'percentile': 48.0, 'z': -0.1,
                'random': {}, 'shuffle': {}, 'null_mean': 0.1, 'null_sd': 0.1}
        unit_signal = {'a': {'sig': (float('nan'),) * 3}}   # degenerate -> null
        text = report_panel(complexity=complexity, unit_signal=unit_signal, null=null,
                            holdout=holdout, seed=seed, print_fn=None)
        self.assertIn('RED', text)                 # complexity red
        self.assertIn('FAIL', text)                # multiple failing gates
        self.assertIn('complexity', text)
        self.assertIn('unit-signal', text)
        self.assertIn('holdout-gap', text)

    def test_passing_panel_shows_pass(self):
        complexity = complexity_ratio(6, 600)
        holdout = {'mean_gap': 0.05, 'mean_holdout': 0.6, 'per_group': {}}
        null = {'real': 0.5, 'percentile': 99.0, 'z': 4.0,
                'null_mean': 0.0, 'null_sd': 0.1}
        seed = {'flagged': [], 'objective_spread': 0.02}
        text = report_panel(complexity=complexity, null=null, holdout=holdout,
                            seed=seed, print_fn=None)
        self.assertIn('GREEN', text)
        self.assertIn('PASS', text)
        self.assertNotIn('FAIL', text)


if __name__ == '__main__':
    unittest.main()
