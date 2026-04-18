"""
walk_forward — rolling walk-forward fold generator.

Produces a sequence of (train_start, train_end, holdout_start, holdout_end)
date tuples by sliding a fixed-size training window forward in time.

Usage::

    wf = WalkForward(train_months=12, holdout_months=3)
    folds = wf.generate_folds(date(2020, 1, 1), date(2024, 12, 31))
    for fold in folds:
        study = optuna.create_study(...)
        objective = BacktestObjective(fold.train_start, fold.train_end, ...)
        study.optimize(objective, n_trials=50, timeout=4*3600)
        # evaluate best params on fold.holdout_start → fold.holdout_end

For Phase 1 (proof study), generate_folds produces a single fold.
For Phase 4, it produces many folds across multiple years.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date, timedelta


@dataclass(frozen=True)
class Fold:
    train_start: date
    train_end: date
    holdout_start: date
    holdout_end: date

    def __str__(self) -> str:
        return (
            f'train={self.train_start}→{self.train_end}  '
            f'holdout={self.holdout_start}→{self.holdout_end}'
        )


class WalkForward:
    """
    Rolling walk-forward fold generator.

    Parameters
    ----------
    train_months : int
        Length of each training window in calendar months.
    holdout_months : int
        Length of each holdout (out-of-sample) window in calendar months.
    """

    def __init__(self, train_months: int = 12, holdout_months: int = 3) -> None:
        if train_months < 1 or holdout_months < 1:
            raise ValueError('train_months and holdout_months must be >= 1')
        self.train_months = train_months
        self.holdout_months = holdout_months

    def generate_folds(self, start: date, end: date) -> list[Fold]:
        """
        Generate all valid folds within [start, end].

        A fold is valid when its holdout window ends on or before *end*.
        The training window of each subsequent fold begins at the end of the
        previous fold's holdout window (non-overlapping, contiguous).

        Returns
        -------
        list[Fold]
            Empty if the total window is too short to fit even one fold.
        """
        folds: list[Fold] = []
        train_start = start

        while True:
            train_end = _add_months(train_start, self.train_months) - timedelta(days=1)
            holdout_start = train_end + timedelta(days=1)
            holdout_end = _add_months(holdout_start, self.holdout_months) - timedelta(days=1)

            if holdout_end > end:
                break

            folds.append(Fold(train_start, train_end, holdout_start, holdout_end))

            # Slide forward by one holdout period (non-overlapping).
            train_start = holdout_start

        return folds

    def __repr__(self) -> str:
        return f'WalkForward(train_months={self.train_months}, holdout_months={self.holdout_months})'


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _add_months(d: date, months: int) -> date:
    """Add *months* calendar months to *d*, clamping to month-end if needed."""
    total_months = d.month + months
    year = d.year + (total_months - 1) // 12
    month = (total_months - 1) % 12 + 1
    day = min(d.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)
