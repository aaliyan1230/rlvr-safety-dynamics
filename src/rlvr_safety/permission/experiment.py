"""Experiment specs, spending-limit checks and deterministic run bundles.

An *experiment spec* names the models, workloads, readouts, generation settings, launch limits and
frozen gates of one run. ``write_bundle`` turns it into a self-contained, hashed tarball that the
runner uploads; the same spec always produces the same archive bytes.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import tarfile
import urllib.request
from collections import Counter
from collections.abc import Callable
from datetime import date
from pathlib import Path

from ..io import read_jsonl, sha256_file, write_jsonl
from .benchmark import prepare_requests, validate_requests
from .checks import check_bank
from .prompts import build_conditions
from .runtime import lock_versions, package_name
from .schema import is_identifier, validate_bank

SPEC_VERSION = "permission-experiment-2026-10-07"
USER_AGENT = "OpenAI File Downloader, XaiImageApiFetch/1.0"
READOUTS = {"generate", "choice_logprobs"}
PLAN_KEYS = {"mcq", "executable", "option_free", "wordings"}
REVIEW_REQUIREMENTS = {"none", "accepted"}
CODE_FILES = (
    "__init__.py",
    "io.py",
    "provenance.py",
    "permission/__init__.py",
    "permission/schema.py",
    "permission/environment.py",
    "permission/prompts.py",
    "permission/scoring.py",
    "permission/benchmark.py",
    "permission/generation.py",
    "permission/runtime.py",
)
CONTROLLER_FILES = ("permission_run.py", "pilot.py", "runpod_cli.py")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
IMAGE_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
RUNTIME_LOCK = Path("infra/runpod/permission-runtime.lock")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_spec(spec: dict) -> None:
    _require(spec.get("spec_version") == SPEC_VERSION, f"spec_version must be {SPEC_VERSION}")
    _require(is_identifier(spec.get("experiment_id")), "experiment_id must be an identifier")
    models = spec.get("models")
    _require(isinstance(models, list) and models, "at least one model is required")
    labels = set()
    for model in models:
        _require(isinstance(model.get("repo"), str) and "/" in model["repo"], "model repo")
        _require(HEX40.match(model.get("revision", "")) is not None, "model revision must be a SHA")
        _require(is_identifier(model.get("label")), "model label must be an identifier")
        _require(model["label"] not in labels, "duplicate model label")
        labels.add(model["label"])
        template = model.get("chat_template_sha256", "auto")
        _require(
            template == "auto" or re.fullmatch(r"[0-9a-f]{64}", template) is not None,
            "chat_template_sha256 must be 'auto' or a SHA-256",
        )
    workloads = spec.get("workloads")
    _require(isinstance(workloads, list) and workloads, "at least one workload is required")
    names = set()
    for workload in workloads:
        _require(is_identifier(workload.get("name")), "workload name must be an identifier")
        _require(workload["name"] not in names, "duplicate workload name")
        names.add(workload["name"])
        _require(isinstance(workload.get("bank"), str), "workload bank path required")
        plan = workload.get("plan")
        _require(isinstance(plan, dict) and plan, "workload plan required")
        _require(set(plan) <= PLAN_KEYS, f"unknown plan keys: {sorted(set(plan) - PLAN_KEYS)}")
        _require(bool(set(plan) & {"mcq", "executable", "option_free"}), "plan has no format")
        sources = workload.get("sources")
        _require(sources is None or (isinstance(sources, list) and sources), "sources filter")
    readouts = spec.get("readouts")
    _require(
        isinstance(readouts, list) and set(readouts) <= READOUTS and "generate" in readouts,
        f"readouts must include 'generate' and be within {sorted(READOUTS)}",
    )
    generation = spec.get("generation", {})
    _require(generation.get("do_sample") is False, "generation must be greedy (do_sample false)")
    for key in ("max_new_tokens", "seed", "per_response_seconds"):
        _require(type(generation.get(key)) is int and generation[key] >= 0, f"generation.{key}")
    _require(isinstance(spec.get("runtime_pins"), dict) and spec["runtime_pins"], "runtime_pins")
    launch = spec.get("launch", {})
    for key in ("gpu", "image", "cloud", "min_cuda_version"):
        _require(isinstance(launch.get(key), str) and launch[key], f"launch.{key}")
    digest = launch.get("image_digest", "auto")
    _require(
        digest == "auto" or IMAGE_DIGEST_RE.match(digest) is not None,
        "launch.image_digest must be 'auto' or sha256:<64 hex>",
    )
    _require(launch.get("gpu_count") == 1, "launch.gpu_count must be 1")
    _require(
        isinstance(launch.get("datacenters"), list) and launch["datacenters"], "launch.datacenters"
    )
    for key in ("hourly_ceiling", "max_minutes", "max_launch_usd", "container_disk_gb"):
        _require(isinstance(launch.get(key), (int, float)) and launch[key] > 0, f"launch.{key}")
    _require(spec.get("require_review", "none") in REVIEW_REQUIREMENTS, "require_review")
    for key in ("envelope", "gates"):
        _require(isinstance(spec.get(key), str), f"{key} path required")


def load_spec(path: Path) -> dict:
    spec = json.loads(path.read_text())
    validate_spec(spec)
    return spec


def check_envelope(launch: dict, envelope: dict, spent_usd: float, today: date) -> dict:
    """Compare a launch plan with the recorded spending envelope.

    Returns ``{"problems": [...], "warnings": [...]}``; any problem blocks the launch.
    """
    limits = envelope["limits"]
    problems: list[str] = []
    if today > date.fromisoformat(envelope["valid_through"]):
        problems.append(f"envelope expired on {envelope['valid_through']}")
    checks = (
        ("gpu_count", launch["gpu_count"] == limits["gpu_count"]),
        ("hourly_ceiling", launch["hourly_ceiling"] <= limits["hourly_ceiling_usd"]),
        ("max_minutes", launch["max_minutes"] <= limits["max_minutes_per_launch"]),
        ("max_launch_usd", launch["max_launch_usd"] <= limits["max_usd_per_launch"]),
        ("image", launch["image"] == limits["image"]),
        ("cloud", launch["cloud"] == limits["cloud"]),
        ("container_disk_gb", launch["container_disk_gb"] <= limits["storage_limit_gb"]),
        ("total_budget", spent_usd + launch["max_launch_usd"] <= limits["max_usd_total"]),
    )
    problems += [f"launch exceeds envelope: {name}" for name, ok in checks if not ok]
    warnings = []
    if "not-yet-recorded" in envelope.get("status", ""):
        warnings.append("envelope notes that Aaliyan's exact approval wording is not yet recorded")
    return {"problems": problems, "warnings": warnings}


def _get_json(url: str, opener: Callable = urllib.request.urlopen) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with opener(request, timeout=60) as response:
        return json.loads(response.read())


def fetch_model_metadata(repo: str, revision: str, opener: Callable = urllib.request.urlopen):
    """Pinned file list with LFS hashes, from the public Hugging Face API (no token sent)."""
    raw = _get_json(
        f"https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true", opener
    )
    if raw.get("sha") != revision or raw.get("id") != repo:
        raise ValueError("Hugging Face returned a different repository or revision")
    siblings = []
    for item in raw.get("siblings", []):
        entry = {"rfilename": item["rfilename"], "size": item.get("size")}
        if item.get("lfs", {}).get("sha256"):
            entry["lfs"] = {"sha256": item["lfs"]["sha256"]}
        siblings.append(entry)
    return {"id": raw["id"], "sha": raw["sha"], "siblings": siblings}


def fetch_chat_template_sha(repo: str, revision: str, opener: Callable = urllib.request.urlopen):
    """SHA-256 of the native chat template string, as the provider will compute it."""
    base = f"https://huggingface.co/{repo}/resolve/{revision}/"
    try:
        template = _get_json(base + "tokenizer_config.json", opener).get("chat_template")
    except Exception:  # fall back to a separate template file
        template = None
    if not isinstance(template, str):
        request = urllib.request.Request(
            base + "chat_template.jinja", headers={"User-Agent": USER_AGENT}
        )
        with opener(request, timeout=60) as response:
            template = response.read().decode()
    if not template:
        raise ValueError("no native string chat template found")
    return hashlib.sha256(template.encode()).hexdigest()


def fetch_image_digest(image: str, opener: Callable = urllib.request.urlopen) -> str:
    """The registry digest that currently backs a Docker Hub image tag.

    Only Docker Hub references are supported; other registries must pin
    ``launch.image_digest`` explicitly because this project cannot resolve them.
    """
    reference, separator, pinned = image.partition("@")
    if separator:
        _require(IMAGE_DIGEST_RE.fullmatch(pinned) is not None, "invalid image reference digest")
        return pinned
    parts = reference.split("/")
    if len(parts) > 1 and ("." in parts[0] or ":" in parts[0] or parts[0] == "localhost"):
        if parts[0] != "docker.io":
            raise ValueError(f"cannot resolve a digest for non-Docker-Hub image: {image}")
        path = "/".join(parts[1:])
    else:
        path = reference
    name, _, tag = path.rpartition(":")
    if not name:
        name, tag = path, "latest"
    if "/" not in name:
        name = "library/" + name
    raw = _get_json(f"https://hub.docker.com/v2/repositories/{name}/tags/{tag}", opener)
    digest = raw.get("digest")
    if not isinstance(digest, str) or IMAGE_DIGEST_RE.match(digest) is None:
        raise ValueError(f"registry returned no immutable digest for {image}")
    return digest


def immutable_image(launch: dict) -> str:
    """Registry reference used in the launch request; a tag alone is never sufficient."""
    image, separator, reference_digest = launch["image"].partition("@")
    digest = launch.get("image_digest", "")
    _require(IMAGE_DIGEST_RE.fullmatch(digest) is not None, "a frozen image digest is required")
    _require(not separator or reference_digest == digest, "image reference and digest differ")
    if ":" in image.rsplit("/", 1)[-1]:
        image = image.rsplit(":", 1)[0]
    return image + "@" + digest


def build_workload(spec: dict, root: Path) -> tuple[list[dict], list[dict], dict]:
    """Scenarios, requests and a per-workload report. Banks must pass the mechanical checks."""
    scenarios: dict[str, dict] = {}
    requests: list[dict] = []
    report: dict = {"workloads": {}, "review_statuses": {}}
    for workload in spec["workloads"]:
        bank = list(read_jsonl(root / workload["bank"]))
        sources = workload.get("sources")
        if sources is not None:
            missing = set(sources) - {r["source_id"] for r in bank}
            _require(
                not missing, f"workload {workload['name']} names unknown sources {sorted(missing)}"
            )
            bank = [r for r in bank if r["source_id"] in sources]
        min_wordings = workload.get("min_wordings", 1)
        validate_bank(bank, min_wordings=min_wordings)
        if spec.get("require_review", "none") == "accepted":
            unreviewed = [r["source_id"] for r in bank if r["review"]["status"] != "accepted"]
            _require(not unreviewed, f"sources without accepted human review: {unreviewed}")
        conditions = build_conditions(bank, workload["plan"], min_wordings=min_wordings)
        checks = check_bank(bank, conditions)
        _require(checks["passed"], f"mechanical checks failed: {checks['failures'][:2]}")
        for row in bank:
            existing = scenarios.get(row["source_id"])
            _require(
                existing is None or existing == row, f"conflicting records for {row['source_id']}"
            )
            scenarios[row["source_id"]] = row
        built = prepare_requests(conditions)
        for request in built:
            request["workload"] = workload["name"]
        requests.extend(built)
        report["workloads"][workload["name"]] = {
            "requests": len(built),
            "by_format": dict(Counter(c["format"] for c in conditions)),
            "sources": len(bank),
            "warnings": len(checks["warnings"]),
        }
    ordered = [scenarios[k] for k in sorted(scenarios)]
    validate_requests(ordered, requests)
    report["review_statuses"] = dict(Counter(s["review"]["status"] for s in ordered))
    report["requests"] = len(requests)
    return ordered, requests, report


def _normalized_add(archive: tarfile.TarFile, path: Path, name: str) -> None:
    info = archive.gettarinfo(str(path), arcname=name)
    info.mtime, info.uid, info.gid, info.uname, info.gname, info.mode = 0, 0, 0, "", "", 0o644
    with path.open("rb") as handle:
        archive.addfile(info, handle)


def write_bundle(
    spec_path: Path,
    out_dir: Path,
    *,
    root: Path,
    code_root: Path | None = None,
    metadata_fetcher: Callable = fetch_model_metadata,
    template_fetcher: Callable = fetch_chat_template_sha,
    digest_fetcher: Callable = fetch_image_digest,
) -> dict:
    """Write ``out_dir`` (a bundle) plus ``out_dir.tar.gz`` and its checksum; return a summary."""
    spec = load_spec(spec_path)
    code_root = code_root or Path(__file__).resolve().parents[1]
    scenarios, requests, report = build_workload(spec, root)
    resolved = json.loads(json.dumps(spec))
    launch = resolved["launch"]
    declared_digest = launch.get("image_digest", "auto")
    first_component = launch["image"].split("/", 1)[0]
    other_registry = (
        "/" in launch["image"]
        and ("." in first_component or ":" in first_component or first_component == "localhost")
        and first_component != "docker.io"
    )
    if "@" in launch["image"]:
        fetched_digest = fetch_image_digest(launch["image"])
    elif other_registry and declared_digest != "auto":
        fetched_digest = declared_digest
    else:
        fetched_digest = digest_fetcher(launch["image"])
    _require(IMAGE_DIGEST_RE.fullmatch(fetched_digest) is not None, "invalid resolved image digest")
    if declared_digest not in {"auto", fetched_digest}:
        raise ValueError("pinned image digest differs from the registry")
    launch["image_digest"] = fetched_digest
    launch["image_ref"] = immutable_image(launch)
    lock_path = root / RUNTIME_LOCK
    locked = lock_versions(lock_path.read_text())
    missing = {
        name: version
        for name, version in spec["runtime_pins"].items()
        if locked.get(package_name(name), "").split("+")[0] != version.split("+")[0]
        or ("+" in version and locked.get(package_name(name)) != version)
    }
    _require(not missing, f"runtime pins missing from {RUNTIME_LOCK}: {sorted(missing)}")
    controller_root = code_root.parents[1] / "infra/runpod"
    resolved["controller_sha256"] = {
        name: sha256_file(controller_root / name) for name in CONTROLLER_FILES
    }
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "model_metadata").mkdir()
    for model in resolved["models"]:
        metadata = metadata_fetcher(model["repo"], model["revision"])
        (out_dir / "model_metadata" / f"{model['label']}.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
        fetched = template_fetcher(model["repo"], model["revision"])
        if model.get("chat_template_sha256", "auto") not in {"auto", fetched}:
            raise ValueError(f"{model['label']}: pinned chat template hash differs from the repo's")
        model["chat_template_sha256"] = fetched
    (out_dir / "experiment.json").write_text(json.dumps(resolved, indent=2, sort_keys=True) + "\n")
    gates_source = root / spec["gates"]
    (out_dir / "gates.json").write_bytes(gates_source.read_bytes())
    write_jsonl(out_dir / "scenarios.jsonl", scenarios)
    write_jsonl(out_dir / "requests.jsonl", requests)
    pins = [
        f"{name}=={version}" for name, version in spec["runtime_pins"].items() if name != "torch"
    ]
    (out_dir / "runtime-pins.txt").write_text("\n".join(pins) + "\n")
    (out_dir / "runtime-lock.txt").write_bytes(lock_path.read_bytes())
    workload_script = code_root.parents[1] / "infra/runpod/permission_workload.py"
    (out_dir / "permission_workload.py").write_bytes(workload_script.read_bytes())
    for name in CODE_FILES:
        target = out_dir / "src/rlvr_safety" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((code_root / name).read_bytes())
    files = {
        str(p.relative_to(out_dir)): sha256_file(p)
        for p in sorted(out_dir.rglob("*"))
        if p.is_file()
    }
    manifest = {
        "experiment_id": spec["experiment_id"],
        "spec_version": SPEC_VERSION,
        "requests": len(requests),
        "review_statuses": report["review_statuses"],
        "files": files,
    }
    (out_dir / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    archive_path = out_dir.parent / (out_dir.name + ".tar.gz")
    with (
        archive_path.open("wb") as raw,
        gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as zipped,
    ):
        with tarfile.open(fileobj=zipped, mode="w") as archive:
            for name in sorted([*files, "bundle_manifest.json"]):
                _normalized_add(archive, out_dir / name, name)
    digest = sha256_file(archive_path)
    (out_dir.parent / (out_dir.name + ".tar.gz.sha256")).write_text(digest + "\n")
    return {
        "bundle": str(out_dir),
        "archive": str(archive_path),
        "sha256": digest,
        "requests": len(requests),
        "report": report,
        "launch": resolved["launch"],
    }
