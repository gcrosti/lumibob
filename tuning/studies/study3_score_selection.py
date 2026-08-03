"""Score selection anatomy — does the composite select profitable pairs, and
what does the live strategy actually enter?

Backs `docs/deepdives/2026-08-01_score-selection-and-the-missing-entry-gate.md`.
Descriptive only: no fitting, no gate. Pooled statistics are reported alongside
date-clustered intervals because replay-pool observations share legs and are
not independent.

Sections:
  1  component distributions and score-variance decomposition (z_depth binary?)
  2  score decile vs outcome (win rate vs win size — PR #50 reproduction)
  3  top-K selection at several K, and per-ranker comparison incl. random
  4  date-clustered CI on the top-20 deficit (is "anti-predictive" established?)
  5  what dislocation the LIVE strategy actually entered, from the gate runs

Run:  DB_URL=... python -m tuning.studies.study3_score_selection
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from DatabaseClient import DatabaseClient

CACHE_DIR = os.path.join(os.path.dirname(__file__), '_scoring_cache')
RETRO = os.path.join(os.path.dirname(__file__), 'study2_event_retro',
                     '_event_cache', 'pair_outcomes.parquet')
FOLDS = ('sideways_2022', 'bull_2023', 'mixed_2023_q4')
GATE_RUNS = ('4f419e', '4b26c6', 'bcb308')
K_DEFAULT = 20

# Live composite (post-WS1): 3 components, correlations clamped to [0, 1].
W_CL, W_CS, W_ZD = 0.3, 0.5, 0.2


def load_pool() -> pd.DataFrame:
    p = pd.read_parquet(os.path.join(CACHE_DIR, 'e2_paths.parquet'))
    cache = pd.concat([pd.read_parquet(os.path.join(CACHE_DIR, f'{f}.parquet'))
                       for f in FOLDS], ignore_index=True)
    cache['date'] = pd.to_datetime(cache.date)
    keys = ['fold', 'date', 'lead', 'lag']
    df = p.merge(cache[keys + ['corr_long', 'corr_short', 'z_depth', 'z_entry',
                               'coint_pvalue', 'halflife_days']],
                 on=keys, how='left')
    df['s_cl'] = np.clip(np.nan_to_num(df.corr_long), 0, 1)
    df['s_cs'] = np.clip(np.nan_to_num(df.corr_short), 0, 1)
    df['score'] = W_CL * df.s_cl + W_CS * df.s_cs + W_ZD * df.z_depth
    return df


def section_components(df: pd.DataFrame) -> None:
    print('=' * 72)
    print('1 — Component distributions and score-variance decomposition')
    print('=' * 72)
    print(f'{"component":12s} {"sd":>7s} {"frac at max":>12s} {"weight":>7s} {"w*sd":>7s}')
    for c, w in [('s_cl', W_CL), ('s_cs', W_CS), ('z_depth', W_ZD)]:
        sd = df[c].std()
        at_max = (df[c] >= df[c].max() - 1e-9).mean() * 100
        print(f'{c:12s} {sd:7.3f} {at_max:11.1f}% {w:7.2f} {w * sd:7.3f}')
    print(f'\ncomposite sd {df.score.std():.4f}')
    print('If z_depth has the largest w*sd it dominates ranking despite the '
          'smallest weight — see the deepdive §3c.')


def section_deciles(df: pd.DataFrame) -> None:
    print('\n' + '=' * 72)
    print('2 — Score decile vs outcome (win RATE vs win SIZE)')
    print('=' * 72)
    d = df.assign(dec=pd.qcut(df.score, 10, labels=False, duplicates='drop'))
    t = d.groupby('dec').agg(
        n=('gross', 'size'), score=('score', 'mean'), mean=('gross', 'mean'),
        median=('gross', 'median'),
        win_rate=('gross', lambda s: (s > 0).mean() * 100),
        win_size=('gross', lambda s: s[s > 0].mean()),
        cat_pct=('cat', lambda s: s.mean() * 100))
    print(t.round(2).to_string())
    print('\nFlat win rate with declining win size = PR #50 reproduced: the '
          'score ranks by correlation, and tighter spreads pay less.')


def section_topk(df: pd.DataFrame) -> None:
    print('\n' + '=' * 72)
    print('3 — Top-K selection and ranker comparison')
    print('=' * 72)
    print(f'{"K":>5s} {"n":>6s} {"mean":>9s} {"median":>8s} {"cat%":>6s} {"win%":>6s}')
    for K in (5, 10, 20, 40, 80, 10 ** 6):
        sel = df.sort_values('score', ascending=False).groupby('date').head(K)
        lbl = 'all' if K > 1000 else str(K)
        print(f'{lbl:>5s} {len(sel):6d} {sel.gross.mean():+9.1f} '
              f'{sel.gross.median():+8.1f} {sel.cat.mean() * 100:5.1f}% '
              f'{(sel.gross > 0).mean() * 100:5.1f}%')

    rankers = {
        'composite': df.score, 'corr_long': df.s_cl, 'corr_short': df.s_cs,
        'z_depth': df.z_depth, 'z_entry': df.z_entry,
        'neg_halflife': -df.halflife_days.fillna(99),
        'random': pd.Series(np.random.default_rng(3).random(len(df)), index=df.index),
    }
    print(f'\ntop-{K_DEFAULT} by each ranker:')
    print(f'{"ranker":14s} {"mean":>9s} {"median":>8s} {"cat%":>6s} {"win size":>9s}')
    for name, key in rankers.items():
        sel = (df.assign(k=key).sort_values('k', ascending=False)
                 .groupby('date').head(K_DEFAULT))
        print(f'{name:14s} {sel.gross.mean():+9.1f} {sel.gross.median():+8.1f} '
              f'{sel.cat.mean() * 100:5.1f}% {sel.gross[sel.gross > 0].mean():+9.1f}')


def section_cluster_ci(df: pd.DataFrame, n_boot: int = 2000, seed: int = 5) -> None:
    print('\n' + '=' * 72)
    print('4 — Date-clustered CI on the top-20 deficit')
    print('=' * 72)
    top = df.sort_values('score', ascending=False).groupby('date').head(K_DEFAULT)
    obs = top.gross.mean() - df.gross.mean()
    rng = np.random.default_rng(seed)
    dates = df.date.unique()
    boot = []
    for _ in range(n_boot):
        sub = pd.concat([df[df.date == d] for d in rng.choice(dates, len(dates))])
        tt = sub.sort_values('score', ascending=False).groupby('date').head(K_DEFAULT)
        boot.append(tt.gross.mean() - sub.gross.mean())
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f'top-20 mean {top.gross.mean():+.1f} (median {top.gross.median():+.1f}) | '
          f'pool mean {df.gross.mean():+.1f} (median {df.gross.median():+.1f})')
    print(f'deficit {obs:+.1f} bps | date-clustered 95% CI {lo:+.1f} .. {hi:+.1f} '
          f'-> {"established" if hi < 0 else "NOT established (includes 0)"}')


def section_live_entries(db: DatabaseClient) -> None:
    print('\n' + '=' * 72)
    print('5 — What dislocation did the LIVE strategy actually enter?')
    print('=' * 72)
    if not os.path.exists(RETRO):
        print(f'skipped: {RETRO} missing (run study2_event_retro first)')
        return
    with db._conn() as conn:  # noqa: SLF001 — read-only study query
        ent = pd.read_sql("""
            SELECT p.id AS pair_id, p.score_z_depth, p.composite_score
            FROM pairs p
            WHERE p.run_id = ANY(%(runs)s) AND p.composite_score IS NOT NULL
              AND EXISTS (SELECT 1 FROM trades t WHERE t.pair_id = p.id
                          AND t.side = 'buy' AND t.leg = 'long')
        """, conn, params=dict(runs=list(GATE_RUNS)))
    out = pd.read_parquet(RETRO)
    m = ent.merge(out[['pair_id', 'gross', 'fold']], on='pair_id')
    m['score_z_depth'] = m.score_z_depth.astype(float)
    print(f'{"dislocation at discovery":40s} {"n":>4s} {"%":>6s} {"mean":>8s} {"median":>8s}')
    for name, sel in [
            ('z_depth = 0 (no buy-side dislocation)', m.score_z_depth <= 1e-9),
            ('0 < z_depth < 1 (partial)',
             (m.score_z_depth > 1e-9) & (m.score_z_depth < 1 - 1e-9)),
            ('z_depth = 1 (full: z <= -entry)', m.score_z_depth >= 1 - 1e-9)]:
        g = m[sel]
        if len(g):
            print(f'{name:40s} {len(g):4d} {len(g) / len(m) * 100:5.1f}% '
                  f'{g.gross.mean():+8.1f} {g.gross.median():+8.1f}')
    print(f'\nfully dislocated share of live entries: '
          f'{(m.score_z_depth >= 1 - 1e-9).mean() * 100:.0f}% '
          f'(the replay pool is 100% by construction — population mismatch)')
    print('\nscore-outcome rank correlation among live entries:')
    for fold, g in m.groupby('fold'):
        print(f'  {fold:14s} n={len(g):3d} spearman = '
              f'{spearmanr(g.composite_score, g.gross).statistic:+.3f}')
    print(f'  {"POOLED":14s} n={len(m):3d} spearman = '
          f'{spearmanr(m.composite_score, m.gross).statistic:+.3f}')


def main() -> None:
    df = load_pool()
    section_components(df)
    section_deciles(df)
    section_topk(df)
    section_cluster_ci(df)
    db = DatabaseClient(os.environ['DB_URL'])
    section_live_entries(db)
    db.close()


if __name__ == '__main__':
    main()
