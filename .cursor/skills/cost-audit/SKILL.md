---
name: cost-audit
description: AWS cost audit workflow for LumiBob. Use when asked to audit AWS spend, explain a bill, investigate a cost overage, or check for wasted cloud resources — e.g. "why is the AWS bill higher than expected", "audit our cloud costs", "run a cost audit".
---

# Cost Audit Agent

The goal is to explain **what is actually being billed** and surface **waste** (resources that cost money but do nothing). Always work from the real bill via the AWS CLI — never estimate from the infra plan (`docs/plans/2026-05-21_cloud-infrastructure-plan.md`), whose cost table can drift from reality (e.g. it lists the Elastic IP as free; AWS now bills all public IPv4 at ~$3.60/mo).

Prerequisite: `aws sts get-caller-identity` must succeed. If it fails, AWS CLI is not configured — stop and tell the user.

Reference baseline: the plan budgets **~$47/month** for Phase 1 (t3.medium DB ~$30, 200 GB gp3 EBS ~$16, S3 ~$0.35, Elastic IP listed as free). Compare actuals against this.

---

## Step 1 — Cost by service, month over month

Pull the last two full months plus the current partial month. Remember the current month is partial — annualise its run-rate before comparing, don't compare raw totals.

```bash
aws ce get-cost-and-usage \
  --time-period Start=<YYYY-MM-01, two months back>,End=<first of next month> \
  --granularity MONTHLY --metrics UnblendedCost \
  --group-by Type=DIMENSION,Key=SERVICE --output json
```

Flag any service that is non-trivial and unexpected. Known-normal services: EC2-Compute, EC2-Other (EBS), VPC (Elastic IP), S3, RDS (tiny). Anything else warrants a look.

## Step 2 — Break EC2 down by usage type

"EC2 - Other" and "EC2 - Compute" are opaque at the service level. Split them:

```bash
aws ce get-cost-and-usage \
  --time-period Start=<start>,End=<end> \
  --granularity MONTHLY --metrics UnblendedCost \
  --group-by Type=DIMENSION,Key=USAGE_TYPE \
  --filter '{"Dimensions":{"Key":"SERVICE","Values":["EC2 - Other","Amazon Elastic Compute Cloud - Compute","Amazon Virtual Private Cloud"]}}' \
  --output json
```

Read the usage-type names carefully — they encode the answer:
- `EBS:VolumeUsage.gp3` — storage. Compare $/month to expected size: gp3 is ~$0.08/GB-mo, so 200 GB ≈ $16. Double that means double the volumes.
- `BoxUsage:*` — **on-demand** compute.
- `SpotUsage:*` — **spot** compute.
- `PublicIPv4:InUseAddress` — the Elastic IP (~$3.60/mo; unavoidable while you need SSH).
- A `USE2-` / `USW2-` / other region prefix — a resource **outside us-east-1**, which the plan says should not exist.

**The single most expensive recurring mistake to look for:** a `BoxUsage` line for a large instance type (c6i/c6a/etc). Tuning is supposed to run on **spot** (`SpotUsage`). On-demand costs ~3–4× spot — a `BoxUsage:c6a.4xlarge` line means a tuning run was launched without `MarketType=spot`.

## Step 3 — Sweep for orphaned / misplaced resources

Cost Explorer tells you *what* is billed; these commands tell you *which resource* so it can be cleaned up. Check every region you might have touched — waste hides in regions you forgot about.

```bash
# Unattached EBS volumes (State=available) — pure waste, in the main region
aws ec2 describe-volumes --region us-east-1 \
  --query 'Volumes[?State==`available`].{ID:VolumeId,GB:Size,Type:VolumeType,Created:CreateTime,Tags:Tags}' --output json

# Anything at all in a region the plan says you don't use (repeat per suspect region)
aws ec2 describe-instances --region us-east-2 \
  --query 'Reservations[].Instances[].{ID:InstanceId,Type:InstanceType,State:State.Name,Launched:LaunchTime}' --output table
aws ec2 describe-volumes --region us-east-2 \
  --query 'Volumes[].{ID:VolumeId,GB:Size,State:State}' --output table

# Idle Elastic IPs (allocated but not associated) — billed while unattached
aws ec2 describe-addresses --region us-east-1 \
  --query 'Addresses[?AssociationId==null].{IP:PublicIp,Alloc:AllocationId}' --output table
```

**Before calling any volume orphaned, confirm it is truly unused, not a deliberate backup.** An `available` volume is unattached, but that alone does not make it safe to delete. Check:
- Its `Name` tag — a volume tagged identically to the live DB volume but unattached is a leftover from a failed/duplicated launch.
- Its parent snapshot (`aws ec2 describe-snapshots --snapshot-ids <id>`). If the snapshot is a generic AMI base image (e.g. an `amzn2-ami-*` OS image), the volume holds no real data — it is a discarded root volume, safe to delete. If the snapshot is a genuine data backup, do **not** treat it as waste.

Root cause of orphaned volumes: the plan launches the DB instance with `DeleteOnTermination:false`, so a terminated launch leaves its volume behind. This is intentional for the *live* DB volume, but leaves cruft after any failed/retried launch.

## Step 4 — Report

Produce a short written report — do **not** write a `docs/` file unless asked. Structure:

1. **Totals** — last two months plus current run-rate, vs the ~$47 plan baseline.
2. **Line-item table** — plan vs actual per usage type, with a ✓/✗ verdict on each. Separate **recurring** overage (inflates every month) from **one-time** spikes (e.g. a tuning burst).
3. **Waste found** — each orphaned/misplaced resource by ID, with its monthly cost and the evidence it is safe to remove.
4. **Bottom line** — one sentence: what the true steady-state cost is once waste is removed, and what (if anything) remains a genuine gap vs the plan.

## Step 5 — Remediation (only if the user asks)

Deleting volumes and terminating instances is **destructive** — never do it as part of the audit. Report findings and stop. Act only on an explicit instruction, and only after the Step 3 safety checks confirm the resource is truly orphaned. When you do act, verify afterward (`describe-volumes` / `describe-instances`) that the resource is gone and that terminating an instance also removed its volume (older instances may have `DeleteOnTermination:false` on the root volume, leaving a new orphan behind).
