# Practice-run learning-rate sweep for the Blind arm on the 24 development situations.
# LRS and STEPS come from the environment; each run saves adapters every 30 steps.
set -uo pipefail
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
W=/workspace; OUT=$W/out; mkdir -p $OUT
fail() { echo "STAGE FAILED: $1"; echo "$1" >> $OUT/failed_stages.txt; }
. $W/venv/bin/activate
cd $W/code
for lr in ${LRS:-1e-4 3e-4}; do
  echo "=== $(date -u +%FT%TZ) train blind lr=$lr"
  python capstone/train_grpo.py --arm blind --seed 0 --train $W/data/train-dev24.jsonl \
    --out $OUT/train/blind-s0-lr$lr --max-steps "${STEPS:-60}" --lr $lr --save-steps 30 \
    || fail "train lr=$lr"
done
SKIP_PARENT=1 bash $W/code/capstone/jobs/eval_session.sh
