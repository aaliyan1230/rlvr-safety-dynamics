"""Push, verify and pull run folders in a private Hugging Face dataset.

The token comes from the environment (``HF_TOKEN``) and never leaves this machine. Run data is
refused unless the target dataset is private. Every uploaded file is hashed locally first, and
``verify`` re-downloads files and compares those hashes.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

MANIFEST_NAME = "sync_manifest.json"
RECORD_NAME = "sync_record.json"
EXCLUDE_PATTERNS = (
    ".env*",
    "*.key",
    "*.pem",
    "id_rsa*",
    "id_ed25519*",
    "known_hosts*",
    "ssh-failure.json",
    "watchdog.log",
    RECORD_NAME,
    ".DS_Store",
    "__pycache__/*",
)


def _excluded(relative: str) -> bool:
    name = relative.rsplit("/", 1)[-1]
    return any(fnmatch.fnmatch(name, p) or fnmatch.fnmatch(relative, p) for p in EXCLUDE_PATTERNS)


def _check_path_in_repo(path: str) -> str:
    parts = Path(path).parts
    if not path or Path(path).is_absolute() or ".." in parts:
        raise ValueError("path_in_repo must be a relative path without '..'")
    return "/".join(parts)


def local_manifest(run_dir: Path) -> tuple[dict[str, str], list[str]]:
    files, excluded = {}, []
    for path in sorted(p for p in run_dir.rglob("*") if p.is_file()):
        relative = str(path.relative_to(run_dir))
        if relative == MANIFEST_NAME:
            continue
        if _excluded(relative):
            excluded.append(relative)
        else:
            files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return files, excluded


def default_api():
    from huggingface_hub import HfApi

    return HfApi()


def push(
    run_dir: Path,
    repo_id: str,
    path_in_repo: str,
    *,
    message: str,
    api=None,
) -> dict:
    """Upload ``run_dir`` to ``repo_id/path_in_repo``; return and save a sync record."""
    path_in_repo = _check_path_in_repo(path_in_repo)
    api = api or default_api()
    files, excluded = local_manifest(run_dir)
    if not files:
        raise ValueError("nothing to upload")
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    info = api.dataset_info(repo_id)
    if getattr(info, "private", None) is not True:
        raise RuntimeError(f"{repo_id} is not private; refusing to upload run data")
    manifest = {"files": files, "excluded": excluded, "path_in_repo": path_in_repo}
    (run_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    commit = api.upload_folder(
        folder_path=str(run_dir),
        repo_id=repo_id,
        repo_type="dataset",
        path_in_repo=path_in_repo,
        commit_message=message,
        ignore_patterns=list(EXCLUDE_PATTERNS),
    )
    record = {
        "repo_id": repo_id,
        "path_in_repo": path_in_repo,
        "commit": getattr(commit, "oid", None) or getattr(commit, "commit_oid", None),
        "files": len(files),
        "excluded": excluded,
        "manifest_sha256": hashlib.sha256((run_dir / MANIFEST_NAME).read_bytes()).hexdigest(),
        "pushed_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    if not record["commit"]:
        raise RuntimeError("upload did not return a commit identifier")
    (run_dir / RECORD_NAME).write_text(json.dumps(record, indent=2) + "\n")
    return record


def verify(run_dir: Path, repo_id: str, path_in_repo: str, revision: str, *, download=None) -> dict:
    """Re-download every manifest file at ``revision`` and compare its SHA-256 to the local one."""
    path_in_repo = _check_path_in_repo(path_in_repo)
    if download is None:
        from huggingface_hub import hf_hub_download as download
    files, _ = local_manifest(run_dir)
    missing, mismatched = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for relative, expected in files.items():
            try:
                got = download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    filename=f"{path_in_repo}/{relative}",
                    revision=revision,
                    local_dir=tmp,
                )
            except Exception:  # missing file or network error: report, do not guess
                missing.append(relative)
                continue
            if hashlib.sha256(Path(got).read_bytes()).hexdigest() != expected:
                mismatched.append(relative)
    return {
        "checked": len(files),
        "missing": missing,
        "mismatched": mismatched,
        "ok": not missing and not mismatched,
    }


def pull(repo_id: str, path_in_repo: str, dest: Path, revision: str, *, snapshot=None) -> Path:
    path_in_repo = _check_path_in_repo(path_in_repo)
    if dest.exists():
        raise FileExistsError(f"{dest} already exists")
    if snapshot is None:
        from huggingface_hub import snapshot_download as snapshot
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(
            snapshot(
                repo_id=repo_id,
                repo_type="dataset",
                revision=revision,
                allow_patterns=[f"{path_in_repo}/*"],
                local_dir=tmp,
            )
        )
        source = root / path_in_repo
        if not source.is_dir():
            raise FileNotFoundError(f"{path_in_repo} not found in {repo_id}@{revision}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        import shutil

        shutil.copytree(source, dest)
    return dest
