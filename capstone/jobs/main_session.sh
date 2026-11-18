# Main runs (plan of record, frozen 2026-11-18): Blind x4, Aware x4, Shuffled x3 on the 24
# development situations; lr 2e-5, 60 steps, 16 prompts x 8, T=1.0, LoRA r16, beta 0.
# Seeds 1-4 (seed 0 was the practice run used to pick the learning rate).
# Runs are interleaved by seed so that a cut-short session still leaves balanced arms, and each
# run is evaluated on the 2,528 held-out requests as soon as it finishes.
# Runs detached on the Pod; capstone/pod.py finish pulls /workspace/out and deletes the Pod.
set -uo pipefail
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
W=/workspace; OUT=$W/out; mkdir -p $OUT
stage() { echo "=== $(date -u +%FT%TZ) $*"; }
fail() { echo "STAGE FAILED: $1"; echo "$1" >> $OUT/failed_stages.txt; }

stage env
df -h /workspace | tail -1
if [ ! -x $W/venv/bin/python ]; then
  pip install -q uv
  uv venv -q -p 3.12 $W/venv
fi
. $W/venv/bin/activate
uv pip install -q "trl[vllm]==0.29.1" "vllm==0.11.0" "transformers==4.57.1" "datasets<5" peft hf_transfer || { fail env; exit 1; }
uv pip install -q --no-deps -e $W/code || { fail env; exit 1; }
uv pip freeze > $OUT/freeze.txt; nvidia-smi > $OUT/nvidia-smi.txt

stage model
# Bounded download with one retry; progress goes to the log so a stall is visible.
for attempt in 1 2; do
  timeout 900 python -c "from huggingface_hub import snapshot_download as d; d('allenai/Llama-3.1-Tulu-3-8B-DPO', revision='a7beb67e33ffd01cc87ac3b46cadc1000985b8db', local_dir='$W/models/tulu-dpo', ignore_patterns=['*.pth', 'original/*'])" && break
  echo "model download attempt $attempt failed"; [ $attempt = 2 ] && { fail model; exit 1; }
done
du -sh $W/models/tulu-dpo
sha256sum $W/models/tulu-dpo/*.safetensors > $OUT/weights.sha256

cd $W/code
stage baseline
python capstone/generate.py --requests $W/data/requests.jsonl --out $OUT/eval --parent \
  || fail baseline

RUNS="blind:1 aware:1 shuffled:1 blind:2 aware:2 shuffled:2 blind:3 aware:3 shuffled:3 blind:4 aware:4"
for run in $RUNS; do
  arm=${run%%:*}; seed=${run##*:}; name=$arm-s$seed
  stage "train $name"
  python capstone/train_grpo.py --arm $arm --seed $seed --train $W/data/train-dev24.jsonl \
    --out $OUT/train/$name --max-steps 60 --lr 2e-5 --save-steps 30 \
    || { fail "train $name"; continue; }
  stage "eval $name"
  python capstone/generate.py --requests $W/data/requests.jsonl --out $OUT/eval \
    --adapter "$name-checkpoint-30=$OUT/train/$name/checkpoint-30" \
    --adapter "$name-final=$OUT/train/$name/final" \
    || fail "eval $name"
  find $OUT/train/$name -name "optimizer.pt" -delete
  rm -rf $OUT/train/$name/checkpoint-60  # identical to final
done
stage done
