# Cloud Tuning — Study 1 Onward

> Created: 2026-05-19.
> Study 0 runs locally (already in flight). Study 1 Pass A through Study 3
> run on AWS EC2 with a migrated DB copy.

---

## Why cloud

Warm-cache trial times are 20–85 min depending on short_leg_fraction activity.
The original plan assumed 2–4 min/trial. Running Studies 1–3 single-threaded
locally would take 3–10 days. A 16-vCPU EC2 spot instance with 10 parallel
Optuna workers reduces wall-clock to hours for each study at negligible cost.

---

## Architecture

```
Local machine                    EC2 (c6i.4xlarge, 16 vCPU)
─────────────────                ───────────────────────────
pg_dump lumibob ──────────────►  psql lumibob (TimescaleDB)
                                 ├── portfolio_snapshots
                                 ├── stock_prices (96 MB)
                                 ├── pair_coint_cache
                                 └── optuna study tables

git push ─────────────────────►  git pull + pip install

Monitor via SSH tunnel ◄───────  10 × python -m tuning.studies.study1_pass_a
```

All Optuna workers share the same PostgreSQL RDB storage on EC2. Workers
pull unclaimed trials, run backtests locally on EC2, and write results back
to the shared DB. The `load_if_exists=True` flag makes each worker safe to
restart or add at any point.

---

## Instance selection

| Study | Window per trial | Workers | Instance | Spot price |
|---|---|---|---|---|
| Study 1 Pass A | 3 months | 8 | c6i.2xlarge (8 vCPU) | ~$0.10/hr |
| Study 1 Pass B | 3 months | 8 | c6i.2xlarge | ~$0.10/hr |
| Study 2 | 3 months | 10 | c6i.4xlarge (16 vCPU) | ~$0.20/hr |
| Study 3 | 3 months | 12 | c6i.4xlarge | ~$0.20/hr |

Use spot instances (70–80% cheaper than on-demand). Set a max price of
2× the current spot price; interruptions are rare at that margin. Optuna
resumes automatically on restart because study state lives in PostgreSQL.

Use **Amazon Linux 2023** or **Ubuntu 24.04 LTS** as the AMI.
Attach a 20 GB gp3 EBS volume (root is sufficient — DB lives in EC2 memory
and the volume stores TimescaleDB data).

---

## One-time setup

### 1. Launch the instance

```bash
# AWS CLI — adjust region/key-pair as needed
aws ec2 run-instances \
  --image-id ami-0c02fb55956c7d316 \   # Amazon Linux 2023, us-east-1
  --instance-type c6i.2xlarge \
  --key-name lumibob-key \
  --security-group-ids sg-XXXXXXXX \   # allow SSH (22) + PostgreSQL (5432) from your IP only
  --instance-market-options '{"MarketType":"spot"}' \
  --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":20,"VolumeType":"gp3"}}]'
```

Open port 5432 on the security group **only to your local IP** so you can
connect Optuna workers locally if needed and monitor via psql.

### 2. Install dependencies on EC2

```bash
ssh -i lumibob-key.pem ec2-user@<EC2_IP>

# TimescaleDB via official repo (Amazon Linux 2023)
sudo dnf install -y postgresql16-server postgresql16-contrib
sudo postgresql-setup --initdb
sudo systemctl enable --now postgresql

# Add TimescaleDB extension
curl -s https://packagecloud.io/install/repositories/timescale/timescaledb/script.rpm.sh | sudo bash
sudo dnf install -y timescaledb-2-postgresql-16
sudo timescaledb-tune --quiet --yes
sudo systemctl restart postgresql

# Python dependencies
sudo dnf install -y python3.12 python3.12-pip git
git clone https://github.com/gcrosti/lumibob.git
cd lumibob
pip3.12 install -r requirements.txt
```

### 3. Create the DB and schema

```bash
sudo -u postgres createuser --superuser lumibob
sudo -u postgres createdb -O lumibob lumibob
sudo -u postgres psql lumibob -c "CREATE EXTENSION IF NOT EXISTS timescaledb;"
cd ~/lumibob
sudo -u postgres psql lumibob < schema.sql
```

---

## Data migration

The full DB dump is ~300–400 MB compressed. Transfer takes 2–5 minutes
on a typical home connection.

### Dump locally (run on local machine)

```bash
# Full dump — includes stock_prices, pair_coint_cache, tickers,
# backtest_runs, and all Optuna study tables.
pg_dump -h localhost -U postgres -F c -f lumibob_full.dump lumibob

# Check size
ls -lh lumibob_full.dump
```

### Transfer and restore on EC2

```bash
# Copy to EC2
scp -i lumibob-key.pem lumibob_full.dump ec2-user@<EC2_IP>:~/

# On EC2 — restore into the already-created schema
pg_restore -h localhost -U lumibob -d lumibob \
  --no-owner --no-acl --clean --if-exists \
  lumibob_full.dump

# Verify
psql -U lumibob -d lumibob -c "SELECT COUNT(*) FROM stock_prices;"
# Expected: ~1,551,674 rows
```

### .env file

```bash
# Copy .env (Alpaca keys, DB_URL, FRED key) to EC2
scp -i lumibob-key.pem .env ec2-user@<EC2_IP>:~/lumibob/.env

# Update DB_URL in the EC2 .env to point to local PostgreSQL
echo 'DB_URL=postgresql://lumibob@localhost:5432/lumibob' >> ~/lumibob/.env
```

---

## Study design revisions for cloud

The original plan used a single 30-month training window per trial for
Study 1. At 20–30 min/trial on a 3-month window, a 30-month window costs
10× more (~200–300 min/trial), making 200 trials unworkable even in cloud.

**Revised design: multi-fold studies matching Study 0's pattern.**

Each trial runs on one of 3–4 pre-defined 3-month folds picked round-robin.
TPE learns a parameter distribution that generalises across regimes rather
than overfitting to one long window. The gate (Spearman rho > 0.15 in ≥2
folds) is evaluated by running the best-trial params on each fold separately.

### Revised trial counts

| Study | Folds | Trials/fold | Total | Workers | Wall-clock |
|---|---|---|---|---|---|
| Study 1 Pass A | 3 | 30 | 90 | 8 | ~6 hrs |
| Study 1 Pass B | 3 | 20 | 60 | 8 | ~4 hrs |
| Study 2 | 8 | 20 | 160 | 10 | ~8 hrs |
| Study 3 | 12 | 25 | 300 | 12 | ~12 hrs |

### Folds used across studies

| Label | Window | Regime |
|---|---|---|
| `sideways_2022` | 2022-02-01 → 2022-04-30 | Sideways / mean-reverting |
| `bull_2023` | 2023-04-01 → 2023-06-30 | Bull trending |
| `mixed_2023_q4` | 2023-09-01 → 2023-11-30 | Mixed / transitional |
| `bear_2022_q2` | 2022-06-01 → 2022-08-31 | Bear / high vol |
| `bull_2024_q1` | 2024-01-02 → 2024-03-29 | Bull (rate-cut anticipation) |
| `sideways_2023_q1` | 2023-01-01 → 2023-03-31 | Sideways / banking stress |
| `recovery_2023_q3` | 2023-07-01 → 2023-09-29 | Recovery / low vol |
| `volatile_2022_q3` | 2022-09-01 → 2022-11-30 | High vol / bear |

Study 1 uses the first 3 (sideways, bull, mixed). Study 2 uses all 8.
Study 3 adds 4 additional 2024–2025 folds if a 2024–2025 price cache
warm-up run is first completed.

---

## Code changes required

### `tuning/studies/study1_pass_a.py` — switch to multi-fold

The current `TRAIN_START` / `TRAIN_END` constants need to be replaced with
a fold-rotating objective. The cleanest approach: create a wrapper objective
that picks a fold round-robin from a list and delegates to `BacktestObjective`.

```python
# tuning/studies/study1_pass_a.py — revised TRAIN_START/END block

FOLDS = [
    (date(2022, 2, 1), date(2022, 4, 30)),   # sideways
    (date(2023, 4, 1), date(2023, 6, 30)),   # bull
    (date(2023, 9, 1), date(2023, 11, 30)),  # mixed
]

class FoldRotatingObjective:
    """Rotates through folds so TPE sees all regimes across 90 trials."""
    def __init__(self, folds, **objective_kwargs):
        self._folds = folds
        self._kwargs = objective_kwargs
        self._call_count = 0

    def __call__(self, trial):
        start, end = self._folds[self._call_count % len(self._folds)]
        self._call_count += 1
        obj = BacktestObjective(train_start=start, train_end=end, **self._kwargs)
        return obj(trial)
```

The gate check logic in `_run_gate()` runs the best-trial params on each
fold independently and checks rho > 0.15 in ≥2 of 3 — no change needed.

### `tuning/studies/study1_pass_b.py` — create new

Pass B uses the best-trial params from Pass A as `base_params`, fixes all
Tier 2 signal params, and optimises only the discovery / position params.
Create `study1_pass_b.py` following the same fold-rotating pattern with
`tiers=(2,)` but only the Pass B Tier 2 params free (exclude signal params
fixed from Pass A). Pure Sharpe objective (`discriminatory_weight=0.0`).

---

## Execution

### Start workers (run on EC2)

```bash
cd ~/lumibob

# Study 1 Pass A — 8 workers
for i in $(seq 1 8); do
  RUN_MODE=backtest TUNE_N_TRIALS=90 \
    python3.12 -m tuning.studies.study1_pass_a \
    >> /tmp/study1a_w${i}.log 2>&1 &
done
echo "Workers: $(pgrep -c -f study1_pass_a)"

# Monitor progress (run locally via SSH tunnel)
# ssh -L 5432:localhost:5432 ec2-user@<EC2_IP>
# Then locally: psql postgresql://lumibob@localhost:5432/lumibob
```

### Monitor from local machine

```bash
# Open SSH tunnel (keep this terminal open)
ssh -i lumibob-key.pem -L 5433:localhost:5432 ec2-user@<EC2_IP> -N &

# Query EC2 Optuna study via tunnel
psql postgresql://lumibob@localhost:5433/lumibob -c "
  SELECT state, COUNT(*) FROM trials
  WHERE study_id = (SELECT study_id FROM studies WHERE study_name='study1_pass_a_v1')
  GROUP BY state;
"
```

Or use the Python polling script:

```bash
DB_URL=postgresql://lumibob@localhost:5433/lumibob \
  python -c "
import os, optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
storage = optuna.storages.RDBStorage(url=os.environ['DB_URL'], engine_kwargs={'pool_pre_ping': True, 'pool_size': 1})
study = optuna.load_study(study_name='study1_pass_a_v1', storage=storage)
completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
print(f'{len(completed)}/90 completed   best={study.best_value:.4f}')
"
```

### Study sequencing

Run studies in order; do not proceed past a gate failure without
investigating root cause.

```
Study 1 Pass A  →  gate (rho > 0.15 in ≥2 folds)  →  Study 1 Pass B
Study 1 Pass B  →  feeds base_params into Study 2
Study 2         →  gate (regime-conditioned > static in ≥8/12 folds)  →  Study 3
Study 3         →  gate (positive Sharpe on 2025 holdout in ≥3/4 quarters)
```

---

## Retrieving results

After each study completes, dump the Optuna tables and run results back to
local for record-keeping. Do not dump the full DB each time — only the tables
that changed.

```bash
# On EC2 — dump Optuna tables + new run data
pg_dump -h localhost -U lumibob -F c \
  -t studies -t study_directions -t study_user_attributes -t study_system_attributes \
  -t trials -t trial_params -t trial_values -t trial_intermediate_values \
  -t trial_user_attributes -t trial_system_attributes \
  -t backtest_runs -t portfolio_snapshots -t trades -t pairs \
  -f lumibob_results_study1a.dump lumibob

# Transfer to local
scp -i lumibob-key.pem ec2-user@<EC2_IP>:~/lumibob_results_study1a.dump .

# Restore into local DB (merge — no --clean flag to avoid overwriting existing data)
pg_restore -h localhost -U postgres -d lumibob \
  --no-owner --no-acl \
  lumibob_results_study1a.dump
```

---

## Cost estimate

| Study | Instance | Hours | Spot cost |
|---|---|---|---|
| Study 1 Pass A | c6i.2xlarge | 6 | ~$0.60 |
| Study 1 Pass B | c6i.2xlarge | 4 | ~$0.40 |
| Study 2 | c6i.4xlarge | 8 | ~$1.60 |
| Study 3 | c6i.4xlarge | 12 | ~$2.40 |
| Data transfer | — | — | ~$0.10 |
| **Total** | | | **~$5.10** |

Spot prices fluctuate. Budget $15 to be safe. Instance can be stopped
(not terminated) between studies to preserve the EBS volume.

---

## Shutdown checklist

After all studies complete and results are retrieved locally:

- [ ] Dump final results (`pg_dump` as above)
- [ ] Transfer dump to local machine
- [ ] Verify local DB has all Optuna study records
- [ ] `aws ec2 stop-instances` (stop, not terminate — preserves EBS)
- [ ] Terminate instance only after confirming local DB is complete
- [ ] Delete EBS volume after termination

---

## Prerequisites before cloud launch

In order:

1. **Study 0 gate passes** (running locally now — expect result ~tonight)
2. **Study 1 Pass A script updated** to use `FoldRotatingObjective`
3. **Study 1 Pass B script created** (`tuning/studies/study1_pass_b.py`)
4. **EC2 setup and DB migration completed** (1–2 hr one-time effort)
5. **Price cache warm-up**: Run one baseline backtest on each fold from EC2
   to confirm price data is fully cached before launching parallel workers
   (prevents Alpaca rate-limit contention across 8 simultaneous workers
   all fetching the same uncached symbols)
