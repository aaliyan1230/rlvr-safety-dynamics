#!/usr/bin/env bash
# One-shot orchestrator: launch a Lambda pod (or reuse a running one), bring
# up its per-project env on the shared persistent filesystem, clone/pull the
# target repo, run a command, then ALWAYS stop the pod - this is the single
# entry point coding agents (or you) should call to run an experiment.
#
# Usage:
#   bin/run.sh --repo git@github.com:org/repo.git --cmd "python train.py" --yes
#
# Required:
#   --repo URL           git remote to clone (SSH form, e.g. git@github.com:org/repo.git)
#   --cmd "..."          command to run on the pod, inside the project's venv
#   --yes                required to actually launch a NEW pod (costs money).
#                         Get explicit user confirmation in chat before passing this.
#
# Optional:
#   --project-name NAME  defaults to the repo basename
#   --branch NAME         defaults to main
#   --req-file PATH       requirements file path inside the repo (default requirements.txt)
#   --lock-file PATH      lock file path inside the repo (default requirements.lock.txt)
#   --instance-type NAME  override auto-selected instance type
#   --pod-id ID           reuse an already-running pod instead of launching a new one
#                         (skips --yes requirement; nothing new is billed by this call)
#   --keep-alive          do NOT stop the pod when done (prints a loud reminder + the
#                         exact stop command). Use for interactive follow-up work only.
#
# Exit behavior: a trap ALWAYS stops any pod this invocation launched, on
# success, on error, or on Ctrl-C - unless --keep-alive was passed.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAMBDA_CLI="python3 $HERE/lambda/lambda_cli.py"

# ---- args -------------------------------------------------------------------
REPO=""
PROJECT_NAME=""
BRANCH="main"
CMD=""
REQ_FILE="requirements.txt"
LOCK_FILE="requirements.lock.txt"
INSTANCE_TYPE_OVERRIDE=""
POD_ID=""
KEEP_ALIVE=0
CONFIRM=0

while [ $# -gt 0 ]; do
    case "$1" in
        --repo) REPO="$2"; shift 2 ;;
        --project-name) PROJECT_NAME="$2"; shift 2 ;;
        --branch) BRANCH="$2"; shift 2 ;;
        --cmd) CMD="$2"; shift 2 ;;
        --req-file) REQ_FILE="$2"; shift 2 ;;
        --lock-file) LOCK_FILE="$2"; shift 2 ;;
        --instance-type) INSTANCE_TYPE_OVERRIDE="$2"; shift 2 ;;
        --pod-id) POD_ID="$2"; shift 2 ;;
        --keep-alive) KEEP_ALIVE=1; shift ;;
        --yes) CONFIRM=1; shift ;;
        -h|--help) grep '^#' "$0" | sed 's/^# \?//'; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

[ -n "$REPO" ] || { echo "ERROR: --repo is required" >&2; exit 1; }
[ -n "$CMD" ] || { echo "ERROR: --cmd is required" >&2; exit 1; }
[ -z "$PROJECT_NAME" ] && PROJECT_NAME="$(basename "$REPO" .git)"

# ---- config -------------------------------------------------------------
eval "$(python3 "$HERE/lambda/config_to_env.py")"
SSH_PUBLIC_KEY_FILE="${SSH_PUBLIC_KEY_FILE/#\~/$HOME}"
SSH_PRIVATE_KEY_FILE="${SSH_PRIVATE_KEY_FILE/#\~/$HOME}"
SSH_USER="${SSH_USER:-ubuntu}"

if [ -z "$FILESYSTEM" ] || [ -z "$REGION" ]; then
    echo "ERROR: lambda/config.yaml has no filesystem/region set yet." >&2
    echo "  Run the first-time bootstrap in README.md before using run.sh." >&2
    exit 1
fi

echo "== GPU pod run =="
echo "Repo:         $REPO ($BRANCH)"
echo "Project:      $PROJECT_NAME"
echo "Filesystem:   $FILESYSTEM  (region $REGION)"
# Redact secrets: --cmd routinely carries HF_TOKEN=... and this banner is
# echoed straight into logs and terminal scrollback.
echo "Command:      $(printf '%s' "$CMD" | sed -E 's/((HF_TOKEN|HUGGINGFACE_TOKEN|OPENAI_API_KEY|GEMINI_API_KEY|LAMBDA_API_KEY)=)[^ ]*/\1<redacted>/g')"
echo

# ---- ssh agent: make sure the key is loaded for forwarded git auth --------
if ! ssh-add -l 2>/dev/null | grep -q "$(ssh-keygen -lf "$SSH_PUBLIC_KEY_FILE" | awk '{print $2}')"; then
    echo "-- Loading $SSH_PRIVATE_KEY_FILE into ssh-agent (needed for forwarded git auth) --"
    ssh-add "$SSH_PRIVATE_KEY_FILE"
fi

INSTANCE_ID=""
WE_LAUNCHED=0

cleanup() {
    if [ "$WE_LAUNCHED" = "1" ] && [ "$KEEP_ALIVE" != "1" ] && [ -n "$INSTANCE_ID" ]; then
        echo
        echo "-- Stopping pod $INSTANCE_ID (cleanup) --"
        $LAMBDA_CLI stop --id "$INSTANCE_ID" || echo "WARNING: stop failed - check manually with:  $LAMBDA_CLI list" >&2
    elif [ -n "$INSTANCE_ID" ] && [ "$KEEP_ALIVE" = "1" ]; then
        echo
        echo "!!! --keep-alive was set: pod $INSTANCE_ID is STILL RUNNING and billing. !!!"
        echo "!!! Stop it yourself when done:                                            !!!"
        echo "!!!   python3 $HERE/lambda/lambda_cli.py stop --id $INSTANCE_ID            !!!"
    fi
}
trap cleanup EXIT INT TERM

# ---- launch or reuse --------------------------------------------------------
if [ -n "$POD_ID" ]; then
    INSTANCE_ID="$POD_ID"
    echo "-- Reusing existing pod $INSTANCE_ID (no new billing from this call) --"
else
    if [ "$CONFIRM" != "1" ]; then
        echo "Refusing to launch a new pod without --yes." >&2
        echo "Get the human's confirmation in chat first, then re-run with --yes." >&2
        exit 2
    fi

    if [ -n "$INSTANCE_TYPE_OVERRIDE" ]; then
        INSTANCE_TYPE="$INSTANCE_TYPE_OVERRIDE"
    else
        echo "-- Finding an instance type with live capacity in $REGION --"
        read -r INSTANCE_TYPE _ < <($LAMBDA_CLI find-capacity --region "$REGION" --prefer "${PREFERRED_INSTANCE_TYPES[@]}")
    fi

    echo "-- Launching $INSTANCE_TYPE in $REGION --"
    # tail -1: launch echoes a multi-line confirmation banner before the id.
    # Capturing all of it sent the banner to terminate as a UUID, so the trap
    # that is meant to guarantee shutdown 400'd and left the pod billing.
    INSTANCE_ID=$($LAMBDA_CLI launch \
        --instance-type "$INSTANCE_TYPE" \
        --region "$REGION" \
        --ssh-key "$SSH_KEY_NAME" \
        --filesystem "$FILESYSTEM" \
        --name "${PROJECT_NAME}-agent-run" \
        --yes | tail -1)
    WE_LAUNCHED=1
    echo "Launched instance: $INSTANCE_ID"
fi

# ---- wait for SSH ------------------------------------------------------------
echo "-- Waiting for SSH --"
IP=$($LAMBDA_CLI wait-ssh --id "$INSTANCE_ID" --timeout 600)
echo "Instance reachable at $IP"

# Lambda recycles public IPs between tenants, so a fresh pod routinely lands on an
# IP already in known_hosts under a previous instance's key. accept-new admits an
# UNKNOWN host but refuses a CHANGED one, so that stale entry hard-fails the run
# after the pod is already launched and billing. Drop any entry for this IP first.
# This is not a weakening of host verification: we launched this instance via the
# API and took its IP from the API response, so first-contact trust here is exactly
# what accept-new already grants a never-seen host.
ssh-keygen -R "$IP" >/dev/null 2>&1 || true

SSH_OPTS=(-i "$SSH_PRIVATE_KEY_FILE" -o StrictHostKeyChecking=accept-new -A)

# ---- optional hard runtime cap (opt-in; off unless set in config.yaml) ----
if [ -n "${MAX_RUNTIME_HOURS:-}" ]; then
    (
        sleep "$(python3 -c "print(int(float('$MAX_RUNTIME_HOURS') * 3600))")"
        echo "-- max_runtime_hours ($MAX_RUNTIME_HOURS h) exceeded - force-stopping $INSTANCE_ID --" >&2
        $LAMBDA_CLI stop --id "$INSTANCE_ID" || true
    ) &
    WATCHDOG_PID=$!
    trap "kill $WATCHDOG_PID 2>/dev/null || true; cleanup" EXIT INT TERM
fi

START_TS=$(date +%s)

# Known directly from config (Lambda mounts filesystem "$FILESYSTEM" at this
# fixed path) - no need to rediscover it on the remote each time, which
# sidesteps a whole class of local-vs-remote shell-quoting bugs.
POD_WORKSPACE="/lambda/nfs/$FILESYSTEM"
REMOTE_REPO_DIR="$POD_WORKSPACE/repos/$PROJECT_NAME"

# ---- clone/pull the repo on the pod (forwarded-agent git auth, nothing persisted)
echo "-- Syncing repo on pod --"
ssh "${SSH_OPTS[@]}" "$SSH_USER@$IP" bash -s <<EOF
set -euo pipefail
mkdir -p "$POD_WORKSPACE/repos"
export GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=accept-new"
if [ -d "$REMOTE_REPO_DIR/.git" ]; then
    cd "$REMOTE_REPO_DIR"
    git fetch origin
    # A persistent filesystem can retain pod-local WIP from an interrupted
    # research run. Preserve it before syncing the committed experiment code;
    # never make the next run fail or silently discard that WIP.
    #
    # Two ordering rules, both learned the hard way.
    #
    # 1. Stash BEFORE the checkout. Untracked files are exactly what makes a
    #    checkout refuse, so a stash placed after it can never run - the checkout
    #    has already aborted the script.
    # 2. Escape the dollar-parens below. This heredoc is UNQUOTED, so an unescaped
    #    substitution runs on the LAUNCHING machine: the status check would report
    #    on the laptop's clean repo, evaluate false, and skip the stash entirely.
    #
    # NOTE: everything here is still shell-expanded, comments included. Do not put
    # backticks or command substitutions in these comments - they execute locally
    # and splice their output into the script.
    if ! git diff --quiet || [ -n "\$(git status --porcelain --untracked-files=all)" ]; then
        git stash push --include-untracked -m "pre-run pod WIP \$(date -u +%Y%m%dT%H%M%SZ)"
    fi
    git checkout "$BRANCH"
    git pull --ff-only origin "$BRANCH"
else
    git clone --branch "$BRANCH" "$REPO" "$REMOTE_REPO_DIR"
fi
EOF

# ---- bootstrap env + run the command ----------------------------------------
echo "-- Bootstrapping env + running command --"
ssh "${SSH_OPTS[@]}" "$SSH_USER@$IP" \
    "WORKSPACE_DIR='$POD_WORKSPACE' PROJECT_NAME='$PROJECT_NAME' REPO_DIR='$REMOTE_REPO_DIR' REQ_FILE='$REQ_FILE' LOCK_FILE='$LOCK_FILE' bash -s" \
    < "$HERE/pod-env/bootstrap_pod.sh"

ssh "${SSH_OPTS[@]}" "$SSH_USER@$IP" bash -lc "
set -euo pipefail
source '$POD_WORKSPACE/envs/${PROJECT_NAME}.env'
cd '$REMOTE_REPO_DIR'
$CMD
"

END_TS=$(date +%s)
ELAPSED_MIN=$(( (END_TS - START_TS) / 60 ))
echo
echo "== done in ~${ELAPSED_MIN} min of pod time (excludes launch/boot wait) =="
# cleanup trap handles stopping the pod.
