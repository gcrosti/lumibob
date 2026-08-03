#!/usr/bin/env bash
# launch_tuning_instance.sh
#
# Launches a LumiBob tuning EC2 instance. Spot market options are hardcoded
# below and cannot be overridden from the command line — this exists because
# a hand-typed `aws ec2 run-instances` command (per the template in
# docs/plans/2026-05-21_cloud-infrastructure-plan.md) launched on-demand
# instead of spot on 2026-07-16/17, costing ~3-4x spot price for that run
# (see the 2026-08-01 cost audit). Always go through this script instead of
# retyping the raw AWS CLI command.
#
# Usage:
#   scripts/launch_tuning_instance.sh [instance-type]
#
# instance-type defaults to c6i.2xlarge (matches the plan's Study 1 sizing).
# Pass a different type for heavier studies, e.g.:
#   scripts/launch_tuning_instance.sh c6i.4xlarge
#
# Requires the AWS CLI to be configured and the lumibob-tuning-sg security
# group to already exist (created once per the cloud infrastructure plan).

set -euo pipefail

INSTANCE_TYPE="${1:-c6i.2xlarge}"
AMI_ID="${TUNING_AMI_ID:-ami-02b2c1b57c5105166}"
KEY_NAME="${TUNING_KEY_NAME:-lumibob-key}"
SPOT_MAX_PRICE="${TUNING_SPOT_MAX_PRICE:-0.20}"

echo "[launch_tuning_instance] Resolving lumibob-tuning-sg..."
TUNING_SG=$(aws ec2 describe-security-groups \
  --filters "Name=group-name,Values=lumibob-tuning-sg" \
  --query 'SecurityGroups[0].GroupId' --output text)

if [[ -z "$TUNING_SG" || "$TUNING_SG" == "None" ]]; then
  echo "[launch_tuning_instance] ERROR: lumibob-tuning-sg not found." >&2
  echo "[launch_tuning_instance] Create it first — see docs/plans/2026-05-21_cloud-infrastructure-plan.md" >&2
  exit 1
fi

echo "[launch_tuning_instance] Launching $INSTANCE_TYPE on spot (max \$${SPOT_MAX_PRICE}/hr)..."
TUNING_INSTANCE=$(aws ec2 run-instances \
  --image-id "$AMI_ID" \
  --instance-type "$INSTANCE_TYPE" \
  --key-name "$KEY_NAME" \
  --security-group-ids "$TUNING_SG" \
  --instance-market-options "{\"MarketType\":\"spot\",\"SpotOptions\":{\"MaxPrice\":\"${SPOT_MAX_PRICE}\"}}" \
  --block-device-mappings '[{
    "DeviceName":"/dev/xvda",
    "Ebs":{"VolumeSize":20,"VolumeType":"gp3","DeleteOnTermination":true}
  }]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=lumibob-tuning}]' \
  --query 'Instances[0].InstanceId' \
  --output text)

echo "[launch_tuning_instance] Waiting for $TUNING_INSTANCE to enter running state..."
aws ec2 wait instance-running --instance-ids "$TUNING_INSTANCE"

TUNING_IP=$(aws ec2 describe-instances \
  --instance-ids "$TUNING_INSTANCE" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

echo "[launch_tuning_instance] Tuning instance: $TUNING_INSTANCE  IP: $TUNING_IP  Type: $INSTANCE_TYPE  Market: spot"
echo "$TUNING_INSTANCE" > /tmp/lumibob_tuning_instance_id
