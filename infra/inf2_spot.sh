#!/usr/bin/env bash
# Run neuron/run_inf2.py on a one-time Spot Inferentia2 instance that uploads its results to S3 and terminates itself.
#
#   cp infra/env.example infra/.env && $EDITOR infra/.env     # subnet, security group, instance profile, bucket
#   ./infra/inf2_spot.sh                                        # launch; prints how to follow progress
#   ./infra/inf2_spot.sh fetch                                  # copy s3://$BUCKET/$PREFIX/$RUN/ to results/$RUN/
#   DRY_RUN=1 ./infra/inf2_spot.sh                              # ec2 run-instances --dry-run only
#
# The instance needs: an instance profile that can read/write s3://$BUCKET/$PREFIX/*, outbound internet (Hugging Face
# download), and the Neuron 2.28 DLAMI (the last SDK with torch-neuronx + NxDI support for Inf2/Trn1).
# Hard budget: the box powers off (= terminates) after $MINUTES even if the script hangs.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/.." && pwd)
[ -f "$HERE/.env" ] && source "$HERE/.env"
REGION=${REGION:-us-east-1}
ITYPE=${ITYPE:-inf2.xlarge}
MINUTES=${MINUTES:-120}
MODEL=${MODEL:-Qwen/Qwen3.5-0.8B}
SCRIPT=${SCRIPT:-neuron/run_inf2.py}
RUN_ARGS=${RUN_ARGS:-"--bucket 256 --steps 32"}
PREFIX=${PREFIX:-qwen-gdn-static}
VOLUME_GB=${VOLUME_GB:-150}
AMI_NAME=${AMI_NAME:-"Deep Learning AMI Neuron PyTorch Inference vLLM 0.13 (Ubuntu 24.04) 20260227"}
VENV=${VENV:-/opt/aws_neuronx_venv_pytorch_inference_vllm_0_13}
TAG_REPO=${TAG_REPO:-qwen-gdn-static-neuron}
: "${SUBNETS:?set SUBNETS (space-separated) in infra/.env}" "${SG:?set SG}" "${INSTANCE_PROFILE:?set INSTANCE_PROFILE}" "${BUCKET:?set BUCKET}"
AWS="aws --region $REGION"
STATE=$HERE/.state; mkdir -p "$STATE"

if [ "${1:-}" = fetch ]; then
  source "$STATE/last-run.env"
  mkdir -p "$ROOT/results/$RUN"
  $AWS s3 sync "s3://$BUCKET/$PREFIX/$RUN/" "$ROOT/results/$RUN/" --exclude "*.neff" --exclude "*.pt" --exclude "neuronx-cc/*"
  $AWS ec2 describe-instances --instance-ids "$IID" --query 'Reservations[0].Instances[0].State.Name' --output text
  exit 0
fi

RUN=${RUN:-$(date -u +%Y%m%dT%H%M%SZ)-$ITYPE}
AMI=$($AWS ec2 describe-images --owners amazon --filters "Name=name,Values=$AMI_NAME" \
  --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text)
[ "$AMI" != None ] || { echo "AMI '$AMI_NAME' not found in $REGION"; exit 1; }
echo "AMI $AMI  ($AMI_NAME)"

# ship the working tree (not .git, checkpoints or results)
tar -C "$ROOT" --exclude .git --exclude results --exclude 'infra/.state' --exclude 'infra/.env' --exclude ckpt \
    --exclude .venv --exclude __pycache__ --exclude paper -czf "$STATE/code.tgz" .
[ "${DRY_RUN:-0}" = 1 ] || $AWS s3 cp --quiet "$STATE/code.tgz" "s3://$BUCKET/$PREFIX/$RUN/code.tgz"

cat > "$STATE/user-data.sh" <<EOF
#!/bin/bash
shutdown -P +$((MINUTES + 2)) "budget reached"          # backstop: poweroff == terminate
exec > >(tee -a /var/log/run.log) 2>&1
set -x
S3=s3://$BUCKET/$PREFIX/$RUN
export AWS_DEFAULT_REGION=$REGION
sync_out() { aws s3 sync /home/ubuntu/out "\$S3/" --exclude "*.neff" --exclude "*.pt" --exclude "neuronx-cc/*" --quiet; aws s3 cp --quiet /var/log/run.log "\$S3/run.log"; }
finish() { sync_out; cd /home/ubuntu/out && find neuronx-cc -type f \\( -name "*.txt" -o -name "*.json" -o -name "log*" -o \\( -name "*.neff" -size -50M \\) \\) -exec aws s3 cp --quiet {} "\$S3/{}" \\; || true; systemctl poweroff; }
trap finish EXIT
# swap so neuronx-cc does not OOM on a 16 GiB box
fallocate -l 32G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
mkdir -p /home/ubuntu/work /home/ubuntu/out && cd /home/ubuntu/work
aws s3 cp --quiet "\$S3/code.tgz" . && tar xzf code.tgz
export PATH=$VENV/bin:/opt/aws/neuron/bin:\$PATH
python -c "from huggingface_hub import snapshot_download as d; d('$MODEL', local_dir='ckpt', allow_patterns=['*.json','*.safetensors','*.txt'])"
( while sleep 120; do sync_out; done ) &
timeout $((MINUTES - 8))m python -u $SCRIPT ckpt /home/ubuntu/out $RUN_ARGS
echo "exit=\$?" > /home/ubuntu/out/exit_code
EOF

EXPIRES=$(date -u -v+"${MINUTES}"M +%FT%TZ 2>/dev/null || date -u -d "+$MINUTES min" +%FT%TZ)
launch() {  # $1 = subnet; extra args passed through
  local subnet=$1; shift
  $AWS ec2 run-instances "$@" --image-id "$AMI" --instance-type "$ITYPE" --subnet-id "$subnet" --security-group-ids "$SG" \
    --instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}' \
    --instance-initiated-shutdown-behavior terminate --iam-instance-profile Name="$INSTANCE_PROFILE" \
    --metadata-options HttpTokens=required,HttpEndpoint=enabled \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":'"$VOLUME_GB"',"VolumeType":"gp3","DeleteOnTermination":true}}]' \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$PREFIX-$RUN},{Key=expires-at,Value=$EXPIRES},{Key=repo,Value=$TAG_REPO}]" \
    --user-data "file://$STATE/user-data.sh" --query 'Instances[0].InstanceId' --output text
}

if [ "${DRY_RUN:-0}" = 1 ]; then
  launch "${SUBNETS%% *}" --dry-run 2>&1 | tail -1; exit 0
fi
IID=""
for SUBNET in $SUBNETS; do
  if IID=$(launch "$SUBNET" 2>"$STATE/err.txt"); then break; fi
  echo "  $SUBNET: $(tail -1 "$STATE/err.txt")"; IID=""
done
[ -n "$IID" ] || { echo "no Spot capacity for $ITYPE in $REGION right now"; exit 1; }
printf 'IID=%s\nRUN=%s\nREGION=%s\nBUCKET=%s\nPREFIX=%s\n' "$IID" "$RUN" "$REGION" "$BUCKET" "$PREFIX" > "$STATE/last-run.env"
echo "launched $IID ($ITYPE spot, terminates by $EXPIRES)"
echo "  progress: aws --region $REGION s3 cp s3://$BUCKET/$PREFIX/$RUN/run.log - | tail"
echo "  results:  ./infra/inf2_spot.sh fetch"
