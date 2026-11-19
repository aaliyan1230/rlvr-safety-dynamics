"""Upload /workspace/out from the Pod to the private HF dataset (default artifact path).

Runs on the Pod. Reads the destination prefix from /workspace/hf_prefix, written by
`pod.py run --hf-prefix`; does nothing when that file is absent. Upload is incremental
(unchanged files are skipped), so the job calls it after every run and once at the end.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from huggingface_hub import HfApi

REPO = "ic-org/rlvr-safety-dynamics-shared"
PREFIX_FILE = Path("/workspace/hf_prefix")


def main() -> None:
    if not PREFIX_FILE.exists():
        print("hf_sync: no /workspace/hf_prefix; skipping")
        return
    prefix = PREFIX_FILE.read_text().strip()
    out = sys.argv[1] if len(sys.argv) > 1 else "/workspace/out"
    t0 = time.time()
    HfApi().upload_folder(
        repo_id=REPO,
        repo_type="dataset",
        folder_path=out,
        path_in_repo=prefix,
        ignore_patterns=["optimizer.pt", "*.safetensors.tmp"],
        commit_message=f"capstone: Pod upload to {prefix}",
    )
    print(f"hf_sync: {out} -> {REPO}/{prefix} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
