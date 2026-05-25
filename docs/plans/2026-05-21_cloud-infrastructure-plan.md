# Cloud Infrastructure Plan — LumiBob

> Created: 2026-05-21.
> Supersedes: `docs/plans/2026-05-19_cloud-tuning-study1-onward.md`

---

## Phased Approach

| Phase | What runs in cloud | Status |
|---|---|---|
| 1 | All data storage + all Optuna tuning | **Current** |
| 2 | Paper trading (strategy runs daily, monitored) | Future |
| 3 | Live trading (hardened, alerting, audit trail) | Future |

---

## Cloud Concepts — What You're Working With

Before setup steps, here's what each AWS term means in plain language.

**AWS (Amazon Web Services)**
A service that lets you rent computers, storage, and networking by the hour. You pay only for what you use and can shut things down when done.

**EC2 instance**
A virtual machine running in AWS's data center — a remote Linux computer you rent by the hour. You connect to it over SSH exactly as you would any Linux server. When done, you can *stop* it (like shutting down a laptop — data is preserved, compute billing stops) or *terminate* it (like throwing the computer away — data is gone unless you backed it up first).

**Instance type**
The hardware size of the rented computer. `t3.medium` = 2 vCPUs + 4 GB RAM, cheap and suitable for always-on light workloads. `c6i.2xlarge` = 8 vCPUs + 16 GB RAM, optimized for CPU-intensive work like backtests. Bigger = more expensive per hour.

**Spot instance**
AWS sells unused capacity at 70–80% discount. The catch: AWS can reclaim it with a 2-minute warning if they need the capacity back. Fine for tuning (Optuna stores all state in the DB — an interrupted worker loses at most one in-flight trial and resumes cleanly). Not suitable for paper or live trading.

**On-demand instance**
Full price, never interrupted. Used for anything that needs to be reliably up — including the DB instance that holds all your data.

**AMI (Amazon Machine Image)**
A disk image defining the OS and pre-installed software for a new instance — like a Docker base image but for a full VM. We use Amazon Linux 2023.

**EBS (Elastic Block Store)**
A persistent hard drive that attaches to an EC2 instance. EBS survives a stop/start cycle — your TimescaleDB data is preserved when the instance is stopped. If you terminate, EBS is deleted by default unless you configure it otherwise, or take a snapshot first.

**S3 (Simple Storage Service)**
AWS's file storage service. Think of it as a very cheap, extremely durable place to store files in the cloud. You create a *bucket* (a named container) and upload files to it. Durability is effectively guaranteed — AWS replicates your data across multiple data centers. Cost is ~$0.023/GB/month. We use it to store daily database backups.

**Key pair**
An SSH key pair. AWS generates it, you download the `.pem` file (the private key). This is your only way to SSH into an instance. Don't lose it; don't share it.

**Security group**
A firewall that controls which ports are open and from which sources. Think of it as an allowlist. You'll have two: one for the DB instance (always-on), one for the tuning instance (started per study). A critical feature: a security group can grant access to *another* security group — meaning "any instance wearing the tuning badge can reach the DB on port 5432," without needing to know specific IP addresses.

**Elastic IP**
A static public IP address. By default, every time you stop and restart an EC2 instance it gets a new public IP — breaking your SSH config. An Elastic IP stays fixed. Free while attached to a running instance; small hourly charge if allocated but unattached.

**VPC (Virtual Private Cloud)**
Your isolated private network inside AWS. All your instances share the same VPC, which means they can talk to each other over private IP addresses — fast, free of data-transfer charges, and without going through the public internet. AWS provides a default VPC in each region; use that.

**Private IP**
Every EC2 instance gets a private IP address within your VPC (e.g. `172.31.x.x`). Unlike the public IP, the private IP never changes between stop/start cycles. The tuning instance uses the DB instance's private IP to connect to PostgreSQL — no public internet hop needed.

**IAM role**
A set of AWS permissions you attach to an EC2 instance. Without it, the instance can't access other AWS services — even services you own. Think of it as giving the instance an ID badge that says "this machine is allowed to write files to S3 bucket X." You attach the role at launch; the instance then uses it automatically without needing API keys on disk.

**Region**
AWS has data centers worldwide. All your resources — instances, EBS volumes, security groups, Elastic IP — must be in the same region. `us-east-1` (Northern Virginia) is the most common, cheapest, and has the widest instance-type availability.

---

## Architecture

Two instances, always in the same VPC.

```
Your laptop
│
│  SSH tunnel (for local analysis + monitoring)
│
▼
DB instance  (t3.medium, on-demand, always-on)
├── TimescaleDB — all data lives here permanently
│   ├── stock_prices       ← price cache, grows continuously
│   ├── backtest_runs      ← all run results
│   ├── trades / pairs     ← filled orders, discovered pairs
│   └── Optuna study tables
│
├── cron: nightly ticker + price refresh (no laptop needed)
├── cron: nightly pg_dump → S3 (automated backup)
└── EBS: 200 GB gp3

          VPC private network (free, no public internet)
                      │
                      ▼
Tuning instance  (c6i.2xlarge, spot, started per study)
├── Optuna workers (8 parallel) → connect to DB instance
├── No local DB — all reads/writes go to DB instance
└── EBS: 20 GB (OS + code only)
```

**Why two instances?**

The DB instance needs to be always-on so the nightly data refresh runs unattended and so your data is always accessible. The tuning instance is pure compute — expensive to run continuously, but cheap as a spot instance started only when a study is in flight. Separating them means you pay for heavy compute only when you need it.

**How results are retrieved**

There is no retrieval step. Because the DB lives permanently in the cloud, study results are already there when workers finish. You open an SSH tunnel from your laptop and query the cloud DB directly — the same SQL queries you'd run against a local DB.

---

## Data Strategy

### What is stored today and why

The current schema (`schema.sql`) covers: `tickers`, `stock_prices`, `pairs`, `pair_coint_cache`, `backtest_runs`, `portfolio_snapshots`, `trades`, `failed_tickers`, `tuning_studies`, and `active_parameters`.

Two forces shaped these decisions:

1. **What the strategy needs** — the tables are well-designed for their purpose. `pair_coint_cache` is a good example of explicitly persisting expensive computation so it is shared across runs rather than recomputed.
2. **Local disk constraints** — the 2-year rolling retention policy on `stock_prices` exists to keep the local database from growing indefinitely. There is no strategy reason for it. More price history covers more market regimes and directly improves tuning quality.

### What changes with cloud

**Remove the `stock_prices` retention policy.**
The 2-year auto-drop is a disk management workaround. On a 200 GB EBS volume it is unnecessary. Removing it means historical price data accumulates indefinitely, giving tuning studies access to a wider range of market regimes. The compression policy (8× reduction after 90 days) stays — it keeps storage efficient without discarding anything.

Alpaca's free data plan provides up to 5 years of historical OHLCV. That is the practical ceiling regardless of storage capacity. Aim to cache the full 5 years for all symbols in the tradeable universe before starting Study 2 (which uses 8 folds spanning 2022–2024).

**One-time price history backfill.**
After migrating to EC2, run a backfill job to fetch the maximum available history from Alpaca for all symbols currently in `tickers`. This is a one-time operation. After that, the nightly refresh cron keeps the cache current.

### Pending code changes (to make before migration)

These are correctness issues that should be resolved before the local DB is migrated to EC2, so the cloud DB starts clean.

**1. Add `ticker_metadata` to `schema.sql`**

`ticker_metadata` is not defined in `schema.sql`. It is created at runtime by `DatabaseClient.migrate_ticker_metadata()` — a method that runs on every strategy startup. This means a fresh EC2 setup using `schema.sql` alone produces an incomplete schema; the table only appears after the strategy runs for the first time.

The fix: add `ticker_metadata` to `schema.sql` as a proper `CREATE TABLE IF NOT EXISTS` block with all columns already consolidated (replacing the current pattern of a base `CREATE TABLE` followed by three `ALTER TABLE` statements). The runtime migration method stays as a compatibility shim for any existing database that predates this change — its `IF NOT EXISTS` clauses make it a no-op on a fresh schema.

The consolidated definition to add to `schema.sql`:

```sql
CREATE TABLE IF NOT EXISTS ticker_metadata (
    symbol      TEXT        PRIMARY KEY,
    sector      TEXT,
    is_etf      BOOLEAN     NOT NULL DEFAULT FALSE,
    fetched_at  TIMESTAMPTZ NOT NULL,
    sic_code    INT,
    sic_sector  TEXT,
    source      TEXT        DEFAULT 'sec_edgar'
);
```

**2. Fix the stale `source` default in `migrate_ticker_metadata()`**

The `ALTER TABLE` that adds the `source` column in `DatabaseClient.migrate_ticker_metadata()` uses `DEFAULT 'yfinance'`. The strategy has not used yfinance for sector data in some time — SEC EDGAR is the actual source. Any existing database that gets this `ALTER` applied receives a misleading default.

Fix: change `DEFAULT 'yfinance'` to `DEFAULT 'sec_edgar'` in `DatabaseClient.migrate_ticker_metadata()`. One line.

---

## One-Time AWS Setup

Do this once. These resources persist across all phases.

### Step 1 — Create an AWS account

Go to [aws.amazon.com](https://aws.amazon.com) and sign up. You'll need a credit card. Immediately set up a billing alert: Billing → Budgets → Create budget → Monthly cost budget → alert at $80. This prevents surprise charges.

### Step 2 — Install the AWS CLI

The AWS CLI lets you manage AWS from your terminal instead of clicking through the web console.

```bash
brew install awscli

aws configure
# Prompted for:
#   AWS Access Key ID     → IAM → Users → your user → Security credentials → Create access key
#   AWS Secret Access Key → shown once at creation; save it somewhere safe
#   Default region        → us-east-1
#   Default output format → json
```

### Step 3 — Create a key pair

```bash
aws ec2 create-key-pair \
  --key-name lumibob-key \
  --query 'KeyMaterial' \
  --output text > ~/.ssh/lumibob-key.pem

chmod 400 ~/.ssh/lumibob-key.pem
```

`chmod 400` makes the file owner-read-only. SSH refuses to use keys with looser permissions.

### Step 4 — Create security groups

You need two. The DB security group allows your laptop in, and also allows any instance wearing the tuning badge in.

```bash
# Get your default VPC
VPC_ID=$(aws ec2 describe-vpcs \
  --filters Name=isDefault,Values=true \
  --query 'Vpcs[0].VpcId' --output text)

# Your current home IP
MY_IP=$(curl -s https://checkip.amazonaws.com)/32

# --- DB security group ---
DB_SG=$(aws ec2 create-security-group \
  --group-name lumibob-db-sg \
  --description "LumiBob DB instance" \
  --vpc-id $VPC_ID \
  --query 'GroupId' --output text)

# SSH and PostgreSQL from your laptop only (for now — tuning SG added below)
aws ec2 authorize-security-group-ingress \
  --group-id $DB_SG \
  --ip-permissions \
    "IpProtocol=tcp,FromPort=22,ToPort=22,IpRanges=[{CidrIp=$MY_IP}]" \
    "IpProtocol=tcp,FromPort=5432,ToPort=5432,IpRanges=[{CidrIp=$MY_IP}]"

# --- Tuning security group ---
TUNING_SG=$(aws ec2 create-security-group \
  --group-name lumibob-tuning-sg \
  --description "LumiBob tuning instances" \
  --vpc-id $VPC_ID \
  --query 'GroupId' --output text)

# SSH from your laptop (useful for debugging)
aws ec2 authorize-security-group-ingress \
  --group-id $TUNING_SG \
  --ip-permissions \
    "IpProtocol=tcp,FromPort=22,ToPort=22,IpRanges=[{CidrIp=$MY_IP}]"

# Allow any tuning instance to reach the DB on port 5432
# This uses a security group reference — not an IP address.
# Any instance wearing lumibob-tuning-sg can connect to the DB.
aws ec2 authorize-security-group-ingress \
  --group-id $DB_SG \
  --protocol tcp --port 5432 \
  --source-group $TUNING_SG

echo "DB SG:     $DB_SG"
echo "Tuning SG: $TUNING_SG"
```

If your home IP changes (ISP reset), update the rules:

```bash
OLD_IP=x.x.x.x/32
NEW_IP=$(curl -s https://checkip.amazonaws.com)/32

for SG in $DB_SG $TUNING_SG; do
  aws ec2 revoke-security-group-ingress --group-id $SG \
    --ip-permissions "IpProtocol=tcp,FromPort=22,ToPort=22,IpRanges=[{CidrIp=$OLD_IP}]"
  aws ec2 authorize-security-group-ingress --group-id $SG \
    --ip-permissions "IpProtocol=tcp,FromPort=22,ToPort=22,IpRanges=[{CidrIp=$NEW_IP}]"
done
```

### Step 5 — Allocate an Elastic IP for the DB instance

```bash
DB_EIP=$(aws ec2 allocate-address --domain vpc \
  --query 'AllocationId' --output text)
echo "DB Elastic IP allocation: $DB_EIP"
```

### Step 6 — Create an S3 bucket for backups

S3 bucket names must be globally unique across all AWS customers. Using your account ID in the name guarantees uniqueness.

```bash
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
BUCKET=lumibob-backups-$ACCOUNT_ID

aws s3 mb s3://$BUCKET --region us-east-1

# Auto-delete backups older than 30 days
aws s3api put-bucket-lifecycle-configuration \
  --bucket $BUCKET \
  --lifecycle-configuration '{
    "Rules": [{
      "ID": "expire-old-backups",
      "Status": "Enabled",
      "Filter": {"Prefix": ""},
      "Expiration": {"Days": 30}
    }]
  }'

echo "Backup bucket: s3://$BUCKET"
```

### Step 7 — Create an IAM role for the DB instance

This gives the DB instance permission to write backups to your S3 bucket. Without it, the instance can't access S3 even though you own both.

```bash
# Create a trust policy — allows EC2 to assume this role
cat > /tmp/trust.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": "ec2.amazonaws.com"},
    "Action": "sts:AssumeRole"
  }]
}
EOF

# Create the role
aws iam create-role \
  --role-name lumibob-db-role \
  --assume-role-policy-document file:///tmp/trust.json

# Attach a policy allowing writes to the backup bucket only
cat > /tmp/s3-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": ["s3:PutObject", "s3:GetObject", "s3:ListBucket"],
    "Resource": [
      "arn:aws:s3:::$BUCKET",
      "arn:aws:s3:::$BUCKET/*"
    ]
  }]
}
EOF

aws iam put-role-policy \
  --role-name lumibob-db-role \
  --policy-name lumibob-s3-backup \
  --policy-document file:///tmp/s3-policy.json

# Create an instance profile (the thing you actually attach to an EC2 instance)
aws iam create-instance-profile \
  --instance-profile-name lumibob-db-profile

aws iam add-role-to-instance-profile \
  --instance-profile-name lumibob-db-profile \
  --role-name lumibob-db-role
```

---

## Launch the DB Instance

```bash
DB_INSTANCE=$(aws ec2 run-instances \
  --image-id ami-0c02fb55956c7d316 \
  --instance-type t3.medium \
  --key-name lumibob-key \
  --security-group-ids $DB_SG \
  --iam-instance-profile Name=lumibob-db-profile \
  --block-device-mappings '[{
    "DeviceName":"/dev/xvda",
    "Ebs":{"VolumeSize":200,"VolumeType":"gp3","DeleteOnTermination":false}
  }]' \
  --tag-specifications \
    'ResourceType=instance,Tags=[{Key=Name,Value=lumibob-db}]' \
    'ResourceType=volume,Tags=[{Key=Name,Value=lumibob-db-data}]' \
  --query 'Instances[0].InstanceId' \
  --output text)

echo "DB instance: $DB_INSTANCE"

# Wait for it to start (~60 seconds)
aws ec2 wait instance-running --instance-ids $DB_INSTANCE

# Attach the Elastic IP
aws ec2 associate-address \
  --instance-id $DB_INSTANCE \
  --allocation-id $DB_EIP

# Get the Elastic IP address and private IP
DB_PUBLIC_IP=$(aws ec2 describe-addresses \
  --allocation-ids $DB_EIP \
  --query 'Addresses[0].PublicIp' --output text)

DB_PRIVATE_IP=$(aws ec2 describe-instances \
  --instance-ids $DB_INSTANCE \
  --query 'Reservations[0].Instances[0].PrivateIpAddress' --output text)

echo "Public IP (for SSH from laptop): $DB_PUBLIC_IP"
echo "Private IP (for tuning workers): $DB_PRIVATE_IP"
```

Add to `~/.ssh/config` on your laptop:

```
Host lumibob-db
  HostName <DB_PUBLIC_IP>
  User ec2-user
  IdentityFile ~/.ssh/lumibob-key.pem
```

Now `ssh lumibob-db` connects without specifying the IP or key.

---

## DB Instance Setup (Run Once After Launch)

SSH in and run the following blocks in order.

```bash
ssh lumibob-db
```

**Install PostgreSQL 16 + TimescaleDB:**

```bash
sudo dnf install -y postgresql16-server postgresql16-contrib
sudo postgresql-setup --initdb
sudo systemctl enable --now postgresql

curl -s https://packagecloud.io/install/repositories/timescale/timescaledb/script.rpm.sh | sudo bash
sudo dnf install -y timescaledb-2-postgresql-16
sudo timescaledb-tune --quiet --yes
sudo systemctl restart postgresql
```

`timescaledb-tune` reads your instance's RAM and CPU count and adjusts PostgreSQL's memory settings accordingly — on a `t3.medium` it will configure things differently than on a `c6i.2xlarge`.

**Create the database:**

```bash
sudo -u postgres createuser --superuser lumibob
sudo -u postgres createdb -O lumibob lumibob
sudo -u postgres psql lumibob -c "CREATE EXTENSION IF NOT EXISTS timescaledb;"
```

**Install Python and the codebase:**

```bash
sudo dnf install -y python3.12 python3.12-pip git
git clone https://github.com/gcrosti/lumibob.git
cd lumibob
pip3.12 install -r requirements.txt
```

**Apply the schema:**

```bash
sudo -u postgres psql lumibob < schema.sql
```

**Copy secrets:**

```bash
# On your laptop
scp .env lumibob-db:~/lumibob/.env
```

On EC2, open `~/lumibob/.env` and set `DB_URL` to the local database (the DB instance talks to its own PostgreSQL, not through the network):

```
DB_URL=postgresql://lumibob@localhost:5432/lumibob
```

---

## Data Migration (Run Once)

Moves all existing data from your laptop to the cloud DB. This is a one-way migration — after this, the cloud DB is the source of truth.

**On your laptop — dump the local DB:**

```bash
pg_dump -h localhost -U postgres -F c -f lumibob_full.dump lumibob
ls -lh lumibob_full.dump   # check size — expect 300–400 MB

scp lumibob_full.dump lumibob-db:~/
```

**On EC2 — restore:**

```bash
pg_restore -h localhost -U lumibob -d lumibob \
  --no-owner --no-acl --clean --if-exists \
  lumibob_full.dump

# Verify
psql -U lumibob -d lumibob -c "SELECT COUNT(*) FROM stock_prices;"
# Expected: ~1,551,674 rows
```

**Keep your local DB for 2–4 weeks** as a cold backup while you gain confidence in the cloud setup. After that it can be decommissioned (`dropdb lumibob` locally, or just left alone if disk space isn't a concern).

---

## Automated Backup Setup

Set up two cron jobs on the DB instance. One backs up the database nightly to S3; the other refreshes price and ticker data.

**Open the crontab editor on the DB instance:**

```bash
# On EC2
crontab -e
```

This opens a text editor (nano by default). Add the following lines. Times are UTC — adjust if you want a specific local time.

```
# Nightly DB backup to S3 at 3:00 AM UTC
0 3 * * * pg_dump -h localhost -U lumibob -F c lumibob 2>/tmp/pgdump_err.log | aws s3 cp - s3://lumibob-backups-<ACCOUNT_ID>/lumibob-$(date +\%Y\%m\%d).dump && echo "Backup OK $(date)" >> /tmp/backup.log

# Nightly ticker and price refresh at 4:00 AM UTC (after backup)
0 4 * * * cd /home/ec2-user/lumibob && DB_URL=postgresql://lumibob@localhost:5432/lumibob python3.12 -m scripts.refresh_data >> /tmp/refresh.log 2>&1
```

**What these do:**

The backup cron runs `pg_dump` (the same command you'd run manually), pipes the output directly to `aws s3 cp`, and uploads it as a dated file to your S3 bucket. No intermediate file on disk. The S3 lifecycle rule you set up earlier deletes backups older than 30 days automatically.

The refresh cron runs the nightly data fetch — tickers from Alpaca, any price gaps in `StockDataCache` — so the price cache stays current without your laptop being involved.

**Verify the backup runs correctly:**

Wait until after 3:00 AM UTC the next day, then check:

```bash
aws s3 ls s3://lumibob-backups-<ACCOUNT_ID>/
# Should show a .dump file with today's date
```

---

## Connecting From Your Laptop

Since the DB now lives on EC2, you connect via an SSH tunnel. A tunnel works by forwarding a port on your laptop to a port on the remote machine — so tools on your laptop that expect a local database just work, without any configuration changes.

**Open a tunnel:**

```bash
# Open in a background terminal — keep it running while you work
ssh -L 5433:localhost:5432 lumibob-db -N &
```

This makes EC2's PostgreSQL (port 5432) available on your laptop as port 5433.

**Connect:**

```bash
psql postgresql://lumibob@localhost:5433/lumibob
```

**Update `DB_URL` for local analysis sessions:**

```bash
export DB_URL=postgresql://lumibob@localhost:5433/lumibob
```

You can add this to your shell profile with a toggle alias if you switch between local and cloud contexts frequently:

```bash
alias db-cloud='export DB_URL=postgresql://lumibob@localhost:5433/lumibob && ssh -L 5433:localhost:5432 lumibob-db -N &'
```

---

## Running Tuning Studies

### Launch the tuning instance

The tuning instance is started fresh for each study and stopped when done. You pay for it only while workers are running.

```bash
TUNING_INSTANCE=$(aws ec2 run-instances \
  --image-id ami-0c02fb55956c7d316 \
  --instance-type c6i.2xlarge \
  --key-name lumibob-key \
  --security-group-ids $TUNING_SG \
  --instance-market-options '{"MarketType":"spot","SpotOptions":{"MaxPrice":"0.20"}}' \
  --block-device-mappings '[{
    "DeviceName":"/dev/xvda",
    "Ebs":{"VolumeSize":20,"VolumeType":"gp3","DeleteOnTermination":true}
  }]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=lumibob-tuning}]' \
  --query 'Instances[0].InstanceId' \
  --output text)

aws ec2 wait instance-running --instance-ids $TUNING_INSTANCE

TUNING_IP=$(aws ec2 describe-instances \
  --instance-ids $TUNING_INSTANCE \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

echo "Tuning instance: $TUNING_INSTANCE  IP: $TUNING_IP"
```

Note `DeleteOnTermination:true` here — the tuning instance's EBS volume holds only the OS and code, not data. There's nothing to preserve.

Add a temporary entry to `~/.ssh/config`:

```
Host lumibob-tuning
  HostName <TUNING_IP>
  User ec2-user
  IdentityFile ~/.ssh/lumibob-key.pem
```

### Set up the tuning instance

```bash
ssh lumibob-tuning

sudo dnf install -y python3.12 python3.12-pip git
git clone https://github.com/gcrosti/lumibob.git
cd lumibob
pip3.12 install -r requirements.txt

# Copy .env — but override DB_URL to point to the DB instance's private IP
scp .env lumibob-tuning:~/lumibob/.env
```

On the tuning instance, open `~/lumibob/.env` and set:

```
DB_URL=postgresql://lumibob@<DB_PRIVATE_IP>:5432/lumibob
```

This is the private IP of the DB instance. Workers connect to the DB over the VPC's internal network — no public internet hop.

### Price cache warm-up

Before launching 8 parallel workers, run one trial single-threaded. This ensures all symbols for the study's date window are cached in `stock_prices`. Without it, 8 workers simultaneously request uncached data from Alpaca, hit rate limits, and fail or corrupt results.

```bash
cd ~/lumibob
python3.12 -m tuning.studies.study1_pass_a
# Interrupt with Ctrl+C after the first trial finishes (~75 min)
```

### Launch parallel workers

```bash
cd ~/lumibob
for i in $(seq 1 8); do
  python3.12 -m tuning.studies.study1_pass_a \
    >> /tmp/study1a_w${i}.log 2>&1 &
done

pgrep -c -f study1_pass_a   # should print 8
```

Workers run in the background on EC2. You can close your SSH session — they'll keep running.

### Monitor from your laptop

The SSH tunnel to the DB instance gives you access to trial progress at any time. Open the tunnel (if not already open), then:

```bash
psql postgresql://lumibob@localhost:5433/lumibob -c "
  SELECT state, COUNT(*)
  FROM trials
  WHERE study_id = (
    SELECT study_id FROM studies WHERE study_name = 'study1_pass_a_v1'
  )
  GROUP BY state;
"
```

Or with Python:

```bash
DB_URL=postgresql://lumibob@localhost:5433/lumibob python3 - << 'EOF'
import os, optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.load_study(
    study_name="study1_pass_a_v1",
    storage=optuna.storages.RDBStorage(os.environ["DB_URL"])
)
done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
print(f"{len(done)}/90 complete   best={study.best_value:.4f}")
EOF
```

### Stop the tuning instance when the study finishes

There is no data retrieval step — results are already in the cloud DB.

```bash
aws ec2 terminate-instances --instance-ids $TUNING_INSTANCE
```

Terminate (not stop) the tuning instance — it holds no data worth preserving and terminating avoids the small idle EBS charge.

### Starting the next study

```bash
# Launch a fresh tuning instance (same command as above)
# Pull any code changes on the new instance
git pull
# Update DB_URL in .env to DB_PRIVATE_IP (same as before)
# Warm cache if new date windows are involved
# Launch workers
```

The DB instance is always running — nothing to start there.

### Study sequencing

```
Study 1 Pass A  →  gate (rho > 0.15 in ≥2/3 folds)  →  Study 1 Pass B
Study 1 Pass B  →  feeds base_params into Study 2
Study 2         →  gate (regime-conditioned > static in ≥8/12 folds)  →  Study 3
Study 3         →  gate (positive Sharpe on 2025 holdout in ≥3/4 quarters)
                →  Phase 2 (paper trading) readiness review
```

Do not proceed past a gate failure without investigating root cause.

---

## Cost

### Ongoing (Phase 1)

| Resource | Type | Monthly cost |
|---|---|---|
| DB instance (t3.medium, on-demand, 24/7) | Compute | ~$30 |
| DB EBS volume (200 GB gp3) | Storage | ~$16 |
| S3 backup storage (~500 MB/day × 30 days) | Storage | ~$0.35 |
| Elastic IP (attached to running instance) | Network | Free |
| **Monthly total** | | **~$47** |

### Per study (tuning compute only)

| Study | Instance | Hours | Spot cost |
|---|---|---|---|
| Study 1 Pass A | c6i.2xlarge | 6 | ~$0.60 |
| Study 1 Pass B | c6i.2xlarge | 4 | ~$0.40 |
| Study 2 | c6i.4xlarge | 8 | ~$1.60 |
| Study 3 | c6i.4xlarge | 12 | ~$2.40 |
| **Studies 1–3 total** | | | **~$5** |

Budget ~$60/month and $20 for the initial study burst. Spot prices fluctuate — check current rates in the AWS Console under EC2 → Spot Requests → Pricing History.

The $47/month ongoing cost is the price of having data always available and refreshed nightly without manual effort. For Phase 2 and 3 this instance would have run anyway; for Phase 1 (tuning only) it's a judgment call on whether that convenience is worth it.

---

## Phase 2 — Paper Trading (Future)

*Design notes only. No action until Phase 1 studies are complete and gate-passed.*

### What changes from Phase 1

The DB instance is already running — no architecture change needed there. Two things are added:

**1. Strategy execution on a schedule.**
The strategy runs once per trading day via a cron job on the DB instance (or a dedicated compute instance if load warrants it). EC2's clock is UTC; the cron must be offset to run at the right ET time.

```
# Run strategy at 9:28 AM ET on weekdays (14:28 UTC)
28 14 * * 1-5 cd /home/ec2-user/lumibob && RUN_MODE=paper python3.12 main.py >> /tmp/paper.log 2>&1
```

**2. Monitoring.**
You need to know if the strategy fails silently. Minimum viable approach: after each run, the strategy writes a heartbeat record to the DB. A second cron job runs at end-of-day, checks whether today's heartbeat exists, and posts a Slack message if not.

A Slack webhook is one HTTP request — no Slack SDK needed:

```bash
curl -X POST -H 'Content-type: application/json' \
  --data '{"text":"LumiBob: no paper trading snapshot today"}' \
  https://hooks.slack.com/services/YOUR/WEBHOOK/URL
```

### Open decisions before Phase 2

- **Where to run the strategy**: on the DB instance (simplest — everything in one place) or on a separate compute instance (cleaner separation, slightly more complex). The DB instance's `t3.medium` is sufficient for single-threaded daily execution.
- **Holiday calendar**: cron doesn't know about market holidays. The strategy must check whether the market is open at startup and exit cleanly if not.
- **Time zone**: set `TZ=America/New_York` in `/etc/environment` on the instance so Python datetime operations default to ET. Verify before going live.

---

## Phase 3 — Live Trading (Future)

*Design notes only. No action until paper trading has been stable for a meaningful period.*

### What changes from Phase 2

Paper trading tolerates downtime. A missed paper trade is a data gap. A missed live trade — or a crash mid-order — has financial consequences.

**1. Secrets off disk.**
Move Alpaca API keys from `.env` to AWS Secrets Manager. The strategy fetches them at startup via the `boto3` SDK. No secrets on disk means a compromised instance doesn't automatically mean compromised money.

**2. Kill switch.**
A DB flag checked at strategy startup (`SELECT value FROM config WHERE key = 'trading_halted'`) lets you halt trading from any machine with DB access — no SSH to the instance required. Essential for fast response if something goes wrong.

**3. Backup hardening.**
Move from 30-day to 90-day S3 retention. Add point-in-time recovery by enabling PostgreSQL WAL archiving to S3 — this lets you restore to any minute in the past, not just the most recent nightly dump.

**4. Runbook.**
Document — before going live, not after — what to do if the instance goes down mid-day, if an order fails to fill, and how to exit all positions immediately.

### Open decisions before Phase 3

- **Managed DB (RDS) vs. self-managed**: RDS PostgreSQL with the TimescaleDB extension is fully managed — automated backups, minor version upgrades, multi-AZ failover — but adds ~$50–100/month. Worth evaluating once the schema is stable and you have real money at stake.
- **Hard position cap**: enforce a maximum portfolio value at the infrastructure or DB level, independent of strategy parameters. The strategy should not be the only thing standing between a bug and an outsized position.
