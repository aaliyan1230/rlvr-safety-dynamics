# Second half of the gate re-run: fix the datasets pin, then the two practice runs and their
# held-out evaluation (the parent baseline and nudge already ran under transformers 4.57.1).
set -uo pipefail
W=/workspace; OUT=$W/out
. $W/venv/bin/activate
uv pip install -q "datasets<5" "transformers==4.57.1" || exit 1
python -c "import datasets, trl, transformers; print(datasets.__version__, trl.__version__, transformers.__version__)" || exit 1
uv pip freeze > $OUT/freeze.txt
sed -i '/^train lr=/d' $OUT/failed_stages.txt 2>/dev/null
LRS="2e-5 5e-5" STEPS=60 bash $W/code/capstone/jobs/lr_sweep.sh
