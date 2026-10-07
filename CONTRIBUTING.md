# Working on RLVR safety dynamics

The main repository contains the completed study, code, datasets, and compact
result artifacts. The SPAR permission-following benchmark is under development:
its pipeline code is in `src/rlvr_safety/permission/` (see its `CHANGELOG.md`), and no benchmark
results have been released. Current plans, team assignments,
and manuscript materials live in the separate private
[docs repository](https://github.com/aaliyan1230/rlvr-safety-dynamics-docs).

## Clone and check the code

Use Python 3.11 or newer and Git. These checks run locally without GPUs,
model downloads, or API credentials:

```bash
git clone git@github.com:aaliyan1230/rlvr-safety-dynamics.git
cd rlvr-safety-dynamics
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[analysis,dev]'
make check
```

On Windows, use the virtual-environment activation command for your shell;
the Makefile requires a `make`-compatible environment. A core-only install,
`python -m pip install -e .`, is sufficient for `make check` and is what CI uses.

The check validates data, runs unit and smoke tests, regenerates deterministic
results, and verifies artifact hashes. Generated tracked files should remain
unchanged: check with `git diff --exit-code` afterward.

## Add the private team docs

Members with access to the docs repository should clone it **inside the main
checkout as `local/`**:

```bash
git clone git@github.com:aaliyan1230/rlvr-safety-dynamics-docs.git local
```

Start with `local/README.md`. It links the full research plan, latest paired
work split, visual, literature review, and existing manuscript. The visual
is a standalone HTML file that can be opened in a browser without a server.

If you use a coding agent, expose the private project instructions at the
main checkout root on macOS/Linux:

```bash
ln -s local/AGENTS.md AGENTS.md
```

Only create this symlink if `AGENTS.md` does not already exist. On systems
without symlink support, an ignored local copy works too. The inherited
personal skills symlink is optional and is not needed to install or test
the project. Neither `local/` nor the root `AGENTS.md` belongs in the public
repository.

## Keep commits in the right repository

- Code, reusable data, reviewed public documentation, and released artifacts:
  commit from the main checkout.
- Working plans, internal review, research notes, and manuscript drafts:
  commit from `local/` using its separate Git history and remote.
- Credentials, applicant exports, virtual environments, caches, and temporary
  previews: keep out of both repositories.

Use a focused branch and pull request for code changes. Preserve the frozen
study and add new experiments separately; use the exact source and model
provenance required by the research plan. Run the relevant tests and `make
check` before proposing a change that affects the pipeline or its artifacts.

## GPU work

RunPod is the sole configured GPU provider. See the [GPU guide](infra/runpod/README.md)
for REST v2 inventory/billing commands, the validated bounded pilot, and remaining
acceptance work for general experiments. Account access and SSH setup are separate
from local reproduction. Keep the API key in an ignored owner-only `.env`, never
on the Pod. Reconcile costs and obtain approval of the concrete GPU/hourly rate
before each paid launch. Cloning or `make check` never starts a GPU resource.
