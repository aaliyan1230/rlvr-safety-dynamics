#!/usr/bin/env bash
# Idempotent per-project environment setup, generalized from
# crosslingual-rule-following's infra/runpod/setup.sh for reuse across
# multiple research repos on a single shared Lambda persistent filesystem.
#
# Design (same philosophy as the RunPod version, extended to multi-project):
#   - The venv, uv's package cache, and the HF model cache all live on the
#     PERSISTENT filesystem (mounted under /lambda/nfs/<name> on Lambda), not
#     the pod's ephemeral root disk. A new pod attached to the same
#     filesystem reuses everything instead of reinstalling torch or
#     re-downloading model weights.
#   - The HF cache and uv cache are SHARED across every project on the
#     filesystem (model weights and package wheels are reusable regardless
#     of which repo asked for them).
#   - Each project gets its OWN venv (PROJECT_NAME-keyed), because different
#     research repos may pin conflicting package versions. The shared uv
#     cache means creating a second project's venv is still a local
#     operation, not a re-download.
#   - Lock-file-first: if the project repo already has a requirements lock
#     file committed, install from that (bit-for-bit reproducible across the
#     whole team). Otherwise floor-pin from requirements.txt and freeze a
#     lock file for the project to commit.
#   - torch is uninstalled/reinstalled unconditionally against a pinned CUDA
#     wheel index every run, per NVIDIA/RunPod-style guidance: cheap after
#     the first run thanks to uv's cache, and guarantees everyone ends up on
#     the same build regardless of what the base image shipped.
#
# Usage (called by bin/run.sh over SSH; can also be run by hand on a pod):
#   PROJECT_NAME=crosslingual-rule-following \
#   REPO_DIR=/lambda/nfs/research/repos/crosslingual-rule-following \
#   REQ_FILE=infra/runpod/requirements.txt \
#   LOCK_FILE=infra/runpod/requirements.lock.txt \
#   bash bootstrap_pod.sh
#
# Safe to re-run: skips steps that are already done except the torch
# reinstall, which is deliberately unconditional.

set -euo pipefail

# ---- 0. locate the persistent filesystem ----------------------------------
# Lambda mounts persistent filesystems at /lambda/nfs/<filesystem-name>.
# If WORKSPACE_DIR isn't given explicitly, auto-detect the (first) mounted
# filesystem so this script doesn't need the filesystem's name hardcoded.
if [ -z "${WORKSPACE_DIR:-}" ]; then
    if [ -d /lambda/nfs ]; then
        WORKSPACE_DIR="$(find /lambda/nfs -mindepth 1 -maxdepth 1 -type d | head -n1)"
    fi
fi
if [ -z "${WORKSPACE_DIR:-}" ] || [ ! -d "$WORKSPACE_DIR" ]; then
    echo "ERROR: no persistent filesystem found under /lambda/nfs and WORKSPACE_DIR not set." >&2
    echo "  Did this instance get launched with --filesystem attached?" >&2
    exit 1
fi

: "${PROJECT_NAME:?Set PROJECT_NAME (e.g. the repo name) to key the per-project venv}"
: "${REPO_DIR:?Set REPO_DIR to the already-cloned project repo path on this pod}"
REQ_FILE_REL="${REQ_FILE:-requirements.txt}"
LOCK_FILE_REL="${LOCK_FILE:-requirements.lock.txt}"
REQ_FILE="$REPO_DIR/$REQ_FILE_REL"
LOCK_FILE="$REPO_DIR/$LOCK_FILE_REL"

VENV_DIR="${VENV_DIR:-$WORKSPACE_DIR/venvs/$PROJECT_NAME}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$WORKSPACE_DIR/hf_cache}"
UV_CACHE_DIR="${UV_CACHE_DIR:-$WORKSPACE_DIR/uv_cache}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

echo "== bootstrap_pod: $PROJECT_NAME =="
echo "Workspace (persistent): $WORKSPACE_DIR"
echo "Repo:                   $REPO_DIR"
echo "Venv (per-project):     $VENV_DIR"
echo "HF cache (shared):      $HF_CACHE_DIR"
echo "uv cache (shared):      $UV_CACHE_DIR"
echo

# ---- 1. GPU sanity check ---------------------------------------------------
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found. Is this instance actually GPU-backed?" >&2
    exit 1
fi
echo "-- GPU --"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo

# ---- 2. System packages (best-effort) --------------------------------------
SUDO=""
[ "$(id -u)" = "0" ] || SUDO="sudo"
if command -v apt-get >/dev/null 2>&1; then
    MISSING_PKGS=""
    command -v git >/dev/null 2>&1 || MISSING_PKGS="$MISSING_PKGS git"
    command -v tmux >/dev/null 2>&1 || MISSING_PKGS="$MISSING_PKGS tmux"
    command -v gcc >/dev/null 2>&1 || MISSING_PKGS="$MISSING_PKGS build-essential"
    if [ -n "$MISSING_PKGS" ]; then
        echo "-- Installing system packages:$MISSING_PKGS --"
        $SUDO apt-get update -qq
        # shellcheck disable=SC2086
        $SUDO apt-get install -y -qq $MISSING_PKGS >/dev/null
    fi
fi

# ---- 3. uv ------------------------------------------------------------------
export UV_CACHE_DIR
mkdir -p "$UV_CACHE_DIR"
if ! command -v uv >/dev/null 2>&1; then
    echo "-- Installing uv --"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# ---- 4. Per-project, isolated venv ------------------------------------------
mkdir -p "$WORKSPACE_DIR/venvs"
if [ ! -d "$VENV_DIR" ]; then
    echo "-- Creating venv at $VENV_DIR --"
    uv venv "$VENV_DIR"
else
    echo "-- Reusing existing venv at $VENV_DIR --"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# ---- 5/6. uv project, if the repo has one ----------------------------------
# A pyproject + uv.lock pins the CUDA build of torch itself, which the
# torch-then-requirements path below cannot do: installing torch first and
# resolving the rest afterwards lets a dependency (vllm) pull its own torch
# straight from PyPI and clobber the CUDA-matched wheel. That is exactly how
# this project ended up with a cu130 torch on a CUDA 12.8 driver.
if [ -f "$REPO_DIR/pyproject.toml" ] && [ -f "$REPO_DIR/uv.lock" ]; then
    echo "-- uv sync --frozen (pyproject + uv.lock) --"
    ( cd "$REPO_DIR" && VIRTUAL_ENV="$VENV_DIR" uv sync --frozen --active )

    # A venv on the shared filesystem outlives many runs and accumulates state
    # from earlier, different resolutions. uv treats a package already present
    # as satisfied, so a torch that needs nvidia-*-cu12 can sit next to leftover
    # cu13 libraries and fail only at import. Repair once, then insist.
    check_torch() {
        python - <<'PY'
import sys
try:
    import torch
except Exception as e:
    print(f"torch import failed: {e}")
    sys.exit(1)
print(f"torch {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
sys.exit(0 if torch.cuda.is_available() else 1)
PY
    }

    if ! check_torch; then
        echo "-- torch unusable, forcing a clean reinstall from the lock --"
        ( cd "$REPO_DIR" && VIRTUAL_ENV="$VENV_DIR" uv sync --frozen --active --reinstall )
        check_torch || {
            echo "CUDA still unavailable after reinstall - check the driver vs the lock's CUDA build" >&2
            exit 1
        }
    fi

# ---- 6b. Legacy path: torch pinned separately, then requirements ------------
else
    echo "-- Installing torch/torchvision ($TORCH_INDEX_URL) --"
    uv pip uninstall torch torchvision torchaudio >/dev/null 2>&1 || true
    uv pip install torch torchvision --index-url "$TORCH_INDEX_URL"

    if [ -f "$LOCK_FILE" ]; then
        echo "-- Installing from $LOCK_FILE_REL (reproducible, team-shared) --"
        uv pip install -r "$LOCK_FILE"
    elif [ -f "$REQ_FILE" ]; then
        echo "-- No lock file yet. Installing floor-pinned $REQ_FILE_REL --"
        uv pip install -r "$REQ_FILE"
        echo "-- Freezing to $LOCK_FILE_REL --"
        uv pip freeze | grep -v -E '^(torch|torchvision|torchaudio)==' > "$LOCK_FILE"
        echo
        echo "  >>> First install for this project. Lock file generated at:"
        echo "      $LOCK_FILE"
        echo "  >>> Commit and push it so everyone installs the exact same versions."
        echo
    else
        echo "WARNING: no pyproject/uv.lock, no $LOCK_FILE_REL, no $REQ_FILE_REL - venv has only torch." >&2
    fi
fi

# ---- 7. Shared HF cache + per-project activation dotfile -------------------
mkdir -p "$HF_CACHE_DIR"
mkdir -p "$WORKSPACE_DIR/envs"
ENV_FILE="$WORKSPACE_DIR/envs/${PROJECT_NAME}.env"
cat > "$ENV_FILE" <<EOF
export HF_HOME="$HF_CACHE_DIR"
export TRANSFORMERS_CACHE="$HF_CACHE_DIR"
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export WORKSPACE_DIR="$WORKSPACE_DIR"
export REPO_DIR="$REPO_DIR"
export VENV_DIR="$VENV_DIR"
export UV_CACHE_DIR="$UV_CACHE_DIR"
export PATH="\$HOME/.local/bin:\$PATH"
source "$VENV_DIR/bin/activate"
EOF
echo "-- Wrote activation dotfile: $ENV_FILE --"

echo
echo "== bootstrap complete for $PROJECT_NAME =="
echo "Every future shell/pod attach for this project:"
echo "    source $ENV_FILE"
echo "    export HF_TOKEN=hf_xxx   # not persisted anywhere, set it fresh each pod/session"
echo
python -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
