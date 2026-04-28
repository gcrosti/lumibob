#!/usr/bin/env bash
# watch_and_cutover.sh
#
# Watches /tmp/phase3_full.log for the moment sideways_2022 completes
# (detected by the battery attempting to start trend_bull_2023 or baseline).
# At that point it:
#   1. Kills the full battery process
#   2. Queries the DB for the sideways_2022 run ID
#   3. Launches phase3_baseline_only with all 3 known run IDs
#
# Usage:
#   bash scripts/watch_and_cutover.sh [battery_pid]

set -euo pipefail

LOG=/tmp/phase3_full.log
BASELINE_LOG=/tmp/phase3_baseline.log
PROJ=/Users/gcrosti/PycharmProjects/LumiBob

# Known best-trial run IDs from previous completed runs
CALM_BULL_ID="3f7def"
VOL_SHOCK_ID="feac3e"

# Auto-detect battery PID if not passed
BATTERY_PID=${1:-$(pgrep -f "phase3_battery" 2>/dev/null | head -1 || echo "")}

if [[ -z "$BATTERY_PID" ]]; then
    echo "[watch] Could not find battery PID. Pass it as argument."
    exit 1
fi

echo "[watch] Monitoring $LOG (battery PID=$BATTERY_PID)..."
echo "[watch] Will kill after sideways_2022 completes and launch 3-regime baseline."

# Poll every 60 seconds for the transition.
# "Running trend_bull_2023" or "Running baseline" both mean sideways_2022 is done.
while true; do
    if grep -q "Running trend_bull_2023\|Running mixed_2024\|Running baseline\|label=baseline" "$LOG" 2>/dev/null; then
        echo "[watch] Transition detected at $(date) — sideways_2022 complete."
        break
    fi
    sleep 60
done

# Allow a few seconds for the last DB writes to flush.
sleep 15

echo "[watch] Killing full battery (PID $BATTERY_PID)..."
kill "$BATTERY_PID" 2>/dev/null || true
sleep 5

# --- Find sideways_2022 run ID from DB ---
echo "[watch] Looking up sideways_2022 run ID in DB..."
SIDEWAYS_ID=$(python3 - <<'PYEOF'
import psycopg2, sys
conn = psycopg2.connect('postgresql://postgres:lumibob@localhost:5432/lumibob')
cur = conn.cursor()
cur.execute("""
    SELECT r.run_id
    FROM backtest_runs r
    JOIN portfolio_snapshots p ON p.run_id = r.run_id
    WHERE r.mode = 'backtest'
      AND p.time BETWEEN '2021-08-01' AND '2023-02-01'
    GROUP BY r.run_id, r.started_at, r.completed_at
    HAVING COUNT(p.*) >= 100
    ORDER BY r.started_at DESC
    LIMIT 1
""")
row = cur.fetchone()
conn.close()
if row:
    print(row[0])
PYEOF
)

if [[ -z "$SIDEWAYS_ID" ]]; then
    echo "[watch] WARNING: sideways_2022 run ID not found — will use auto-discovery in script."
    SIDEWAYS_ARGS=""
else
    echo "[watch] sideways_2022 run ID: $SIDEWAYS_ID"
    SIDEWAYS_ARGS="--sideways-id $SIDEWAYS_ID"
fi

echo "[watch] Launching targeted 3-regime baseline battery..."
cd "$PROJ"
caffeinate -i python -u -m tuning.studies.phase3_baseline_only \
    --calm-bull-id "$CALM_BULL_ID" \
    --vol-shock-id "$VOL_SHOCK_ID" \
    $SIDEWAYS_ARGS \
    > "$BASELINE_LOG" 2>&1 &

BASELINE_PID=$!
echo "[watch] Baseline battery started (PID $BASELINE_PID)."
echo "[watch] Log: $BASELINE_LOG"
echo "[watch] Monitor with: tail -f $BASELINE_LOG"
