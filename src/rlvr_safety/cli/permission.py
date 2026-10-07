"""Single entry point for the permission benchmark pipeline (``rlvr-permission``)."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

from ..agent_tasks import core
from ..agent_tasks.jobs import authoring, freeform_prelabel, mcq_judge
from ..io import read_jsonl, write_jsonl
from ..permission import analysis, labeling, review, sync
from ..permission.benchmark import run_benchmark
from ..permission.checks import check_bank
from ..permission.experiment import build_workload, load_spec, write_bundle
from ..permission.prompts import build_conditions
from ..permission.schema import validate_bank
from ..permission.scripted import POLICIES, ScriptedProvider

JOBS = {"author": authoring, "judge": mcq_judge, "prelabel": freeform_prelabel}


def _print(value) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False))


def _read_json(path: Path):
    return json.loads(path.read_text())


# ---- bank checks ---------------------------------------------------------------------------
def cmd_check_bank(args) -> int:
    bank = list(read_jsonl(args.bank))
    validate_bank(bank, min_wordings=args.min_wordings)
    plan = _read_json(args.plan) if args.plan else None
    conditions = build_conditions(bank, plan, min_wordings=args.min_wordings) if plan else None
    report = check_bank(bank, conditions)
    _print({k: report[k] for k in ("passed", "failures", "warnings")})
    return 0 if report["passed"] else 1


# ---- agent-task jobs -----------------------------------------------------------------------
def _job_plan(args) -> int:
    module = JOBS[args.job]
    if args.job == "author":
        existing = (
            sorted({r["skeleton_id"] for r in read_jsonl(args.existing_bank)})
            if args.existing_bank
            else []
        )
        tasks, refs = authoring.build_tasks(_read_json(args.brief), existing)
    else:
        rows = list(read_jsonl(args.run_dir / "results.jsonl"))
        requests = {r["episode_id"]: r for r in read_jsonl(args.run_dir / "requests.jsonl")}
        scenarios = {s["source_id"]: s for s in read_jsonl(args.run_dir / "scenarios.jsonl")}
        tasks, refs = module.build_tasks(rows, requests, scenarios)
    if not tasks:
        _print({"tasks": 0, "note": "nothing to do"})
        return 0
    manifest = core.write_plan(args.job_dir, module.JOB, module.RUBRIC_ID, tasks, refs)
    _print({"job": module.JOB, "tasks": len(manifest["tasks"]), "job_dir": str(args.job_dir)})
    return 0


def _job_run(args) -> int:
    command = shlex.split(args.command)
    backend = core.CommandBackend(command, args.agent, args.model_reported, timeout=args.timeout)
    _print({"answered": core.run_backend(args.job_dir, backend, limit=args.limit)})
    return 0


def _job_ingest(args) -> int:
    _print(core.ingest(args.job_dir, JOBS[args.job].validate, operator=args.operator))
    return 0


def _job_audit(args) -> int:
    report = core.audit(args.job_dir)
    _print(report)
    return 0 if report["ok"] else 1


def _read_ingested(job_dir: Path) -> list[dict]:
    return list(read_jsonl(job_dir / "ingested.jsonl"))


def _job_apply(args) -> int:
    """Turn ingested responses into their final artifact (bank, judge scores, prelabels)."""
    ingested = _read_ingested(args.job_dir)
    if args.job == "author":
        bank = authoring.materialize(ingested, accountable_owner=args.owner)
        validate_bank(bank, min_wordings=1)
        report = check_bank(bank)
        write_jsonl(args.out, bank)
        args.out.with_suffix(".checks.json").write_text(json.dumps(report, indent=2) + "\n")
        _print(
            {
                "records": len(bank),
                "checks_passed": report["passed"],
                "warnings": len(report["warnings"]),
            }
        )
        return 0 if report["passed"] else 1
    refs = _read_json(args.job_dir / "refs.json")
    requests = {r["episode_id"]: r for r in read_jsonl(args.run_dir / "requests.jsonl")}
    if args.job == "judge":
        rows = mcq_judge.judge_scores(ingested, refs, requests)
    else:
        scenarios = {s["source_id"]: s for s in read_jsonl(args.run_dir / "scenarios.jsonl")}
        results = list(read_jsonl(args.run_dir / "results.jsonl"))
        rows = freeform_prelabel.prelabels(ingested, refs, scenarios, results)
    write_jsonl(args.out, rows)
    _print({"records": len(rows), "out": str(args.out)})
    return 0


# ---- labelling -----------------------------------------------------------------------------
def cmd_label(args) -> int:
    if args.action == "export":
        _print(labeling.export_csv(args.job_dir, args.csv))
    elif args.action == "import":
        _print(
            labeling.import_csv(args.job_dir, args.csv, args.labeller, allow_partial=args.partial)
        )
    else:
        first = (
            labeling.load_ai_labels(args.job_dir)
            if args.first == "ai"
            else labeling.load_human_labels(args.job_dir, args.first)
        )
        second = (
            labeling.load_ai_labels(args.job_dir)
            if args.second == "ai"
            else labeling.load_human_labels(args.job_dir, args.second)
        )
        _print(labeling.agreement(first, second))
    return 0


# ---- human semantic review ---------------------------------------------------------------
def cmd_review(args) -> int:
    bank = list(read_jsonl(args.bank))
    if args.action == "export":
        counts = review.export_queue(bank, args.reviewers.split(","), args.out_dir)
        _print({"rows_per_reviewer": counts, "out_dir": str(args.out_dir)})
    elif args.action == "packet":
        args.out.write_text(review.render_packet(bank))
        _print({"records": len(bank), "out": str(args.out)})
    else:
        updated, audit = review.apply_reviews(bank, args.queues)
        if args.out.exists():
            raise FileExistsError(f"{args.out} already exists; choose a new file")
        write_jsonl(args.out, updated)
        args.out.with_suffix(".review-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        _print(
            {
                "records": len(updated),
                "statuses": {k: v["status"] for k, v in audit["records"].items()},
            }
        )
    return 0


# ---- bundle, dry run, analysis -------------------------------------------------------------
def cmd_bundle(args) -> int:
    _print(write_bundle(args.spec, args.out_dir, root=args.root))
    return 0


def cmd_dry_run(args) -> int:
    """Run a real spec offline with a scripted provider and analyze it with the spec's gates."""
    spec = load_spec(args.spec)
    scenarios, requests, report = build_workload(spec, args.root)
    provider = ScriptedProvider(scenarios, args.policy)
    out = args.out_dir / args.policy / "benchmark"
    summary = run_benchmark(scenarios, requests, provider, out)
    rows = list(read_jsonl(out / "results.jsonl"))
    gates = _read_json(args.root / spec["gates"])
    gates["expected_rows"] = len(requests)
    metrics = analysis.analyze(
        rows,
        {s["source_id"]: s for s in scenarios},
        {r["episode_id"]: r for r in requests},
        gates=gates,
    )
    (args.out_dir / args.policy / "analysis.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (args.out_dir / args.policy / "report.md").write_text(
        analysis.render_markdown(f"scripted:{args.policy}", metrics)
    )
    _print({"workload": report, "summary": summary, "gates_passed": metrics.get("gates_passed")})
    return 0


def cmd_analyze(args) -> int:
    bundle = args.bundle
    spec = _read_json(bundle / "experiment.json")
    scenarios = {s["source_id"]: s for s in read_jsonl(bundle / "scenarios.jsonl")}
    requests = {r["episode_id"]: r for r in read_jsonl(bundle / "requests.jsonl")}
    gates = _read_json(bundle / "gates.json")
    gates["expected_rows"] = len(requests)
    judge = None
    if args.judge_job_dir:
        refs = _read_json(args.judge_job_dir / "refs.json")
        judge = mcq_judge.judge_scores(_read_ingested(args.judge_job_dir), refs, requests)
    all_metrics, sections = {}, []
    for model in spec["models"]:
        label = model["label"]
        rows = list(read_jsonl(args.run_dir / label / "benchmark/results.jsonl"))
        labels = None
        if args.label_job_dir and args.labellers:
            task_labels = labeling.consensus(args.label_job_dir, args.labellers.split(","))
            source_of = {r["episode_id"]: r["source_id"] for r in rows}
            labels = labeling.episode_role_labels(
                args.label_job_dir, task_labels, scenarios, source_of
            )
        metrics = analysis.analyze(
            rows, scenarios, requests, gates=gates, judge_scores=judge, labels=labels
        )
        all_metrics[label] = metrics
        sections.append(analysis.render_markdown(label, metrics))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "analysis.json").write_text(json.dumps(all_metrics, indent=2) + "\n")
    header = (
        f"# Analysis of `{spec['experiment_id']}`\n\n"
        f"Analysis version `{analysis.ANALYSIS_VERSION}`.\n\n"
    )
    (args.out / "report.md").write_text(header + "\n".join(sections))
    _print({label: {"gates_passed": m.get("gates_passed")} for label, m in all_metrics.items()})
    return 0


# ---- sync ----------------------------------------------------------------------------------
def cmd_sync(args) -> int:
    if args.action == "push":
        _print(sync.push(args.run_dir, args.repo, args.path, message=args.message))
    elif args.action == "verify":
        report = sync.verify(args.run_dir, args.repo, args.path, args.revision)
        _print(report)
        return 0 if report["ok"] else 1
    else:
        _print({"pulled": str(sync.pull(args.repo, args.path, args.dest, args.revision))})
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rlvr-permission", description=__doc__)
    sub = parser.add_subparsers(dest="group", required=True)

    check = sub.add_parser("check-bank", help="mechanical checks on a scenario/control bank")
    check.add_argument("--bank", type=Path, required=True)
    check.add_argument("--plan", type=Path, help="prompt plan JSON; adds prompt-level checks")
    check.add_argument("--min-wordings", type=int, default=1)
    check.set_defaults(func=cmd_check_bank)

    for name, job in JOBS.items():
        group = sub.add_parser(name, help=f"{job.JOB} agent-task job")
        actions = group.add_subparsers(dest="action", required=True)
        plan = actions.add_parser("plan")
        plan.add_argument("--job-dir", type=Path, required=True)
        if name == "author":
            plan.add_argument("--brief", type=Path, required=True)
            plan.add_argument("--existing-bank", type=Path)
        else:
            plan.add_argument(
                "--run-dir",
                type=Path,
                required=True,
                help="benchmark output dir with results/requests/scenarios",
            )
        plan.set_defaults(func=_job_plan, job=name)
        run = actions.add_parser(
            "run", help="answer pending tasks with any command (stdin->stdout)"
        )
        run.add_argument("--job-dir", type=Path, required=True)
        run.add_argument("--command", required=True, help="e.g. 'codex exec --json'")
        run.add_argument("--agent", required=True)
        run.add_argument("--model-reported", required=True)
        run.add_argument("--timeout", type=int, default=300)
        run.add_argument("--limit", type=int)
        run.set_defaults(func=_job_run, job=name)
        ingest = actions.add_parser("ingest")
        ingest.add_argument("--job-dir", type=Path, required=True)
        ingest.add_argument("--operator", required=True)
        ingest.set_defaults(func=_job_ingest, job=name)
        audit = actions.add_parser("audit")
        audit.add_argument("--job-dir", type=Path, required=True)
        audit.set_defaults(func=_job_audit, job=name)
        apply = actions.add_parser("apply", help="materialize ingested responses")
        apply.add_argument("--job-dir", type=Path, required=True)
        apply.add_argument("--out", type=Path, required=True)
        if name == "author":
            apply.add_argument("--owner", required=True, help="accountable human owner")
        else:
            apply.add_argument("--run-dir", type=Path, required=True)
        apply.set_defaults(func=_job_apply, job=name)

    label = sub.add_parser("label", help="blinded human labelling sheets and agreement")
    actions = label.add_subparsers(dest="action", required=True)
    export = actions.add_parser("export")
    export.add_argument("--job-dir", type=Path, required=True)
    export.add_argument("--csv", type=Path, required=True)
    imp = actions.add_parser("import")
    imp.add_argument("--job-dir", type=Path, required=True)
    imp.add_argument("--csv", type=Path, required=True)
    imp.add_argument("--labeller", required=True)
    imp.add_argument("--partial", action="store_true")
    agree = actions.add_parser("agree")
    agree.add_argument("--job-dir", type=Path, required=True)
    agree.add_argument("--first", required=True, help="a labeller name or 'ai'")
    agree.add_argument("--second", required=True, help="a labeller name or 'ai'")
    label.set_defaults(func=cmd_label)

    rev = sub.add_parser("review", help="human semantic review of a bank")
    actions = rev.add_subparsers(dest="action", required=True)
    exp = actions.add_parser("export", help="write blank queue sheets for named reviewers")
    exp.add_argument("--bank", type=Path, required=True)
    exp.add_argument("--reviewers", required=True, help="comma-separated, at least two")
    exp.add_argument("--out-dir", type=Path, required=True)
    pkt = actions.add_parser("packet", help="render a readable review packet")
    pkt.add_argument("--bank", type=Path, required=True)
    pkt.add_argument("--out", type=Path, required=True)
    app = actions.add_parser("apply", help="copy recorded decisions into a new bank file")
    app.add_argument("--bank", type=Path, required=True)
    app.add_argument("--queues", type=Path, nargs="+", required=True)
    app.add_argument("--out", type=Path, required=True)
    rev.set_defaults(func=cmd_review)

    bundle = sub.add_parser("bundle", help="build a deterministic experiment bundle")
    bundle.add_argument("--spec", type=Path, required=True)
    bundle.add_argument("--out-dir", type=Path, required=True)
    bundle.add_argument("--root", type=Path, default=Path.cwd())
    bundle.set_defaults(func=cmd_bundle)

    dry = sub.add_parser("dry-run", help="offline run of a spec with a scripted provider")
    dry.add_argument("--spec", type=Path, required=True)
    dry.add_argument("--policy", choices=POLICIES, required=True)
    dry.add_argument("--out-dir", type=Path, required=True)
    dry.add_argument("--root", type=Path, default=Path.cwd())
    dry.set_defaults(func=cmd_dry_run)

    analyze = sub.add_parser("analyze", help="analyze retrieved results against the frozen gates")
    analyze.add_argument("--bundle", type=Path, required=True)
    analyze.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="retrieved output folder holding <model label>/benchmark",
    )
    analyze.add_argument("--out", type=Path, required=True)
    analyze.add_argument("--judge-job-dir", type=Path)
    analyze.add_argument("--label-job-dir", type=Path)
    analyze.add_argument("--labellers", help="comma-separated labellers whose consensus is used")
    analyze.set_defaults(func=cmd_analyze)

    syncp = sub.add_parser("sync", help="private Hugging Face dataset sync")
    actions = syncp.add_subparsers(dest="action", required=True)
    for action in ("push", "verify", "pull"):
        cmd = actions.add_parser(action)
        cmd.add_argument("--repo", required=True)
        cmd.add_argument("--path", required=True, help="folder inside the dataset repo")
        if action == "push":
            cmd.add_argument("--run-dir", type=Path, required=True)
            cmd.add_argument("--message", required=True)
        elif action == "verify":
            cmd.add_argument("--run-dir", type=Path, required=True)
            cmd.add_argument("--revision", required=True)
        else:
            cmd.add_argument("--dest", type=Path, required=True)
            cmd.add_argument("--revision", required=True)
    syncp.set_defaults(func=cmd_sync)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (
        ValueError,
        FileNotFoundError,
        FileExistsError,
        RuntimeError,
        subprocess.SubprocessError,
    ) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
