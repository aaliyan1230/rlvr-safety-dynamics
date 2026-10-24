"""Verified private Hugging Face backup using only the local operator's credentials."""

from __future__ import annotations

import gzip
import io
import json
import tarfile
from pathlib import Path, PurePosixPath

from .checkpoint_study import atomic_json
from .io import sha256_file


def backup_private(
    directory: Path, *, repository: str, prefix: str, receipt: Path, api=None, downloader=None
) -> dict:
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

    path = PurePosixPath(prefix)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("invalid private artifact prefix")
    api = api or HfApi()
    downloader = downloader or hf_hub_download
    info = api.repo_info(repository, repo_type="dataset")
    if info.private is not True:
        raise ValueError("existing shared artifact repository is not private; do not upload")
    files = sorted(p for p in directory.rglob("*") if p.is_file())
    if not files or any(p.is_symlink() for p in directory.rglob("*")):
        raise ValueError("backup needs regular evidence files without symlinks")
    if any(p.name.startswith(".env") or p.name in ("known_hosts", "id_ed25519") for p in files):
        raise ValueError("credential/local connection material must not enter an artifact backup")
    archive_path = receipt.parent / (receipt.stem + "-backup.tar.gz")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"files": {str(p.relative_to(directory)): sha256_file(p) for p in files}}
    with archive_path.open("wb") as target:
        with gzip.GzipFile(filename="", fileobj=target, mode="wb", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for source in files:
                    content = source.read_bytes()
                    member = tarfile.TarInfo(str(source.relative_to(directory)))
                    member.mode, member.mtime, member.size = 0o600, 0, len(content)
                    archive.addfile(member, io.BytesIO(content))
    manifest["archive_sha256"] = sha256_file(archive_path)
    commit = api.create_commit(
        repo_id=repository,
        repo_type="dataset",
        commit_message=f"Preserve private OLMo evidence: {prefix}",
        operations=[
            CommitOperationAdd(
                path_in_repo=f"{prefix}/evidence.tar.gz", path_or_fileobj=str(archive_path)
            ),
            CommitOperationAdd(
                path_in_repo=f"{prefix}/manifest.json",
                path_or_fileobj=json.dumps(manifest, sort_keys=True).encode(),
            ),
        ],
    )
    revision = commit.oid
    pinned_info = api.repo_info(repository, repo_type="dataset", revision=revision)
    if pinned_info.private is not True:
        raise ValueError("private backup visibility could not be verified")
    remote_archive = Path(
        downloader(repository, f"{prefix}/evidence.tar.gz", repo_type="dataset", revision=revision)
    )
    remote_manifest = Path(
        downloader(repository, f"{prefix}/manifest.json", repo_type="dataset", revision=revision)
    )
    if (
        sha256_file(remote_archive) != manifest["archive_sha256"]
        or json.loads(remote_manifest.read_text()) != manifest
    ):
        raise ValueError("off-provider backup verification failed")
    result = {
        "repository": repository,
        "private_verified": True,
        "revision": revision,
        "prefix": prefix,
        "manifest": manifest,
        "download_verification": True,
    }
    atomic_json(receipt, result)
    return result
