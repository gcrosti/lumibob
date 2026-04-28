#!/usr/bin/env bash
# after_battery.sh
#
# Waits for the Phase 3 baseline battery (phase3_baseline_only.py) to finish,
# then verifies the DB results, prints a status summary, and opens the
# Phase 3.5 deep-dive notebooks in Jupyter.
#
# Usage (run in a separate terminal):
#   bash scripts/after_battery.sh [battery_pid]
#
# If no PID is given it auto-detects the running phase3_baseline_only process.

set -euo pipefail

PROJ=/Users/gcrosti/PycharmProjects/LumiBob
BASELINE_LOG=/tmp/phase3_baseline.log
REPORT_LOG=/tmp/deepdive_ready.log
BATTERY_PID=${1:-$(pgrep -f "phase3_baseline_only" 2>/dev/null | tail -1 || echo "")}

if [[ -z "$BATTERY_PID" ]]; then
    echo "[after_battery] No running baseline battery found. Has it already finished?"
    echo "[after_battery] Running verification and notebook launch anyway..."
else
    echo "[after_battery] Waiting for baseline battery (PID $BATTERY_PID) to finish..."
    # tail -f the log in the background so progress is visible
    tail -f "$BASELINE_LOG" &
    TAIL_PID=$!
    # Poll until the process exits
    while kill -0 "$BATTERY_PID" 2>/dev/null; do
        sleep 30
    done
    kill "$TAIL_PID" 2>/dev/null || true
    echo ""
    echo "[after_battery] Battery process exited at $(date)."
fi

# --- Verify DB results ---
echo "[after_battery] Verifying baseline runs in DB..."
python3 - > "$REPORT_LOG" 2>&1 <<'PYEOF'
import psycopg2
from datetime import date

DB_URL = 'postgresql://postgres:lumibob@localhost:5432/lumibob'

REGIME_WINDOWS = {
    'calm_bull_2017':  (date(2017, 1, 1),  date(2018, 1, 1)),
    'vol_shock_2020':  (date(2020, 2, 1),  date(2020, 7, 1)),
    'sideways_2022':   (date(2022, 1, 1),  date(2023, 1, 1)),
}

# Known best-trial run IDs
BEST_TRIAL_IDS = {
    'calm_bull_2017': '3f7def',
    'vol_shock_2020': 'feac3e',
    'sideways_2022':  '815f18',
}

conn = psycopg2.connect(DB_URL)
cur = conn.cursor()

print()
print('=' * 70)
print('  Phase 3 — Baseline run verification')
print('=' * 70)

all_run_ids = []
for regime, (ws, we) in REGIME_WINDOWS.items():
    cur.execute("""
        SELECT r.run_id, r.started_at, r.completed_at,
               MIN(p.time)::date AS ps_start,
               MAX(p.time)::date AS ps_end,
               COUNT(p.*) AS n_snapshots
        FROM backtest_runs r
        JOIN portfolio_snapshots p ON p.run_id = r.run_id
        WHERE r.mode = 'backtest'
          AND p.time >= %s AND p.time < %s
        GROUP BY r.run_id, r.started_at, r.completed_at
        HAVING COUNT(p.*) >= 20
        ORDER BY r.started_at DESC
        LIMIT 3
    """, (ws, we))
    rows = cur.fetchall()

    bt_id = BEST_TRIAL_IDS[regime]
    found_baseline = [r for r in rows if r[0] != bt_id]
    baseline_id = found_baseline[0][0] if found_baseline else None
    baseline_cpl = found_baseline[0][2] if found_baseline else None
    baseline_snaps = found_baseline[0][5] if found_baseline else 0

    status = 'COMPLETE' if (baseline_id and baseline_cpl) else \
             'PARTIAL'  if (baseline_id and not baseline_cpl and baseline_snaps >= 5) else \
             'MISSING'
    print(f'  {regime:<22} baseline={baseline_id or "NOT FOUND":>8}  '
          f'snapshots={baseline_snaps:>4}  status={status}')
    if baseline_id:
        all_run_ids.append(('baseline', regime, baseline_id))
    all_run_ids.append(('best_trial', regime, bt_id))

print()
print('  Run ID reference for notebooks:')
print(f'  {"label":<12} {"regime":<22} {"run_id"}')
print('  ' + '-' * 50)
for label, regime, rid in all_run_ids:
    print(f'  {label:<12} {regime:<22} {rid}')
print('=' * 70)
print()

conn.close()
PYEOF

cat "$REPORT_LOG"

# --- Launch Jupyter ---
echo "[after_battery] Launching Jupyter for Phase 3.5 deep-dive notebooks..."
cd "$PROJ"
jupyter lab notebooks/ &
JUPYTER_PID=$!
echo "[after_battery] Jupyter started (PID $JUPYTER_PID)."
echo "[after_battery] Open the phase35_0*.ipynb notebooks and run in order."
echo "[after_battery] Full report saved to: $REPORT_LOG"
