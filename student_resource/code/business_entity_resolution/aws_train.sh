#!/bin/bash
# AWS full-train + inference. Tested path: Ubuntu 22.04 AMI.
# Recommended: r6i.2xlarge (8 vCPU, 64 GB) or r6i.4xlarge for headroom.
# Use Spot to spend ~70% less; total run is a few hours, ~$2-5 on Spot.
set -euo pipefail

# 1. Code + data on the box (adjust: S3 sync or scp of student_resource/)
# aws s3 sync s3://<bucket>/student_resource ./student_resource
cd student_resource

# 2. Env
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r code/business_entity_resolution/requirements.txt

# 3. Train (full S2+S3 pool; matcher trained on 300k S1s, validated on 50k)
nohup python -u -m code.business_entity_resolution.src.train_matcher \
  --train-dir dataset/train --out models \
  --top-k 30 --val-size 50000 --max-train-s1 300000 --seed 42 \
  > train.log 2>&1 &
echo "training in background, follow with: tail -f train.log"
wait
cat models/threshold.txt

# 4. Inference (chunked; the long step — expect 2-5h on 8 vCPU)
THR=$(cat models/threshold.txt)
nohup python -u -m code.business_entity_resolution.src.infer \
  --test-dir dataset/test --model models/lgbm.txt \
  --threshold "$THR" --out output --top-k 30 --chunk-s1 100000 \
  > infer.log 2>&1 &
wait

# 5. Validate + package
python utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test --check-ids
