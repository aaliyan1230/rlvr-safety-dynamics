# Re-run of the gate session after pinning transformers 4.57.1 (vLLM 0.11 misread rope_theta
# under transformers 5). Discards the invalid first outputs, then baseline, nudge, two practice
# learning rates and held-out evaluation.
set -uo pipefail
W=/workspace; OUT=$W/out
. $W/venv/bin/activate
uv pip install -q "transformers==4.57.1" || exit 1
python -c "from transformers import AutoConfig; c=AutoConfig.from_pretrained('$W/models/tulu-dpo'); assert c.rope_theta == 500000.0, c.rope_theta"  || exit 1
mkdir -p $W/invalid-tf5 && mv $OUT $W/invalid-tf5/out-$(date +%s) && mkdir -p $OUT
uv pip freeze > $OUT/freeze.txt
LRS="2e-5 5e-5" STEPS=60 bash $W/code/capstone/jobs/lr_sweep.sh
SKIP_PARENT=0 bash $W/code/capstone/jobs/eval_session.sh
