# CLAUDE.md — Kronos

Project instructions for Claude Code. Configuration only; do not launch worker agents until the human asks.

## Repository map
- **Root**: foundation model for financial K-lines (`shiyu-coder/Kronos`)
- **`model/`**: `Kronos`, `KronosTokenizer`, `KronosPredictor` (main library entry)
- **`examples/`**: forecast scripts (`prediction_example.py`, batch, backtest helpers)
- **`tests/`**: pytest regression (`test_kronos_regression.py` + CSV fixtures under `tests/data/`)
- **`finetune/`**: Qlib-based finetune pipeline (`config.py`, train scripts, `qlib_test.py`)
- **`webui/`**: Flask UI (`run.py`, `app.py`, `templates/`)
- **`requirements.txt`**: core Python deps; **`webui/requirements.txt`**: UI extras
- Default branch observed: `master`

## Runtime
- Python **3.10+** (project `.venv` is **3.12.x**)
- Torch CPU is sufficient for baseline tests; CUDA may be absent
- Always prefer: `/workspace/Kronos/.venv/bin/python -m …`  
  Bare `pip`/`python` on PATH may hit unrelated envs (e.g. hummingbot 3.13) and break `pandas==2.2.2` installs

## Setup commands
```bash
cd /workspace/Kronos
/workspace/Kronos/.venv/bin/python -m pip install -r requirements.txt
/workspace/Kronos/.venv/bin/python -m pip install pytest
```
Optional (finetune only, not required for baseline):
```bash
/workspace/Kronos/.venv/bin/python -m pip install pyqlib
# plus local Qlib data — see README finetune section
```
Optional (web UI):
```bash
/workspace/Kronos/.venv/bin/python -m pip install -r webui/requirements.txt
```

## Baseline checks
```bash
/workspace/Kronos/.venv/bin/python -m pytest tests/ -q
/workspace/Kronos/.venv/bin/python -c "from model import Kronos, KronosTokenizer, KronosPredictor; import torch; print(torch.__version__)"
```
Preflight result (do not regress without contract change): **4 passed** on CPU.

README does not document pytest; tests still exist and are the baseline gate.

## Worktree isolation
- Project setting: `.claude/settings.json` → `"worktree": { "baseRef": "head" }`.
- Workers with `isolation: worktree`: **backend-engineer**, **frontend-engineer**, **test-engineer**.
- Worktree checkouts live under `.claude/worktrees/` (gitignored).
- Every such worker must, in order: report **cwd / branch / starting SHA** → run **Setup commands** from `CLAUDE.md` in that cwd → edit **only** assigned files → **commit** and return SHA → report **checks** and **blockers** before handoff.
- **Parallel batch rule:** while the first parallel batch runs, keep the **integration branch tip unchanged** so every worker shares one starting commit. Integration (merge/rebase of worker commits) is Release Engineer’s job after the batch, not during it.
- Do not launch workers until the human asks.

## Agent roles (see `.claude/agents/`)
| Agent | Role |
| --- | --- |
| `architect` | Contracts, assignment, corrections |
| `backend-engineer` | Assigned backend implementation |
| `frontend-engineer` | Assigned `webui/` / interface work |
| `test-engineer` | Acceptance tests + verification |
| `code-reviewer` | Read-only review of code + evidence |
| `release-engineer` | Integrate commits, draft PR (never merge) |

Worker return schema (required): **changed files**, **commit SHA**, **checks performed** (with exit codes), **unresolved blockers**. Stay inside the Architect assignment.

## Completion rules
1. No application changes without an Architect assignment (or explicit human ask).
2. Do not launch workers until the human says to.
3. Keep **experimental Agent Teams disabled** (see `.claude/settings.json`).
4. Never push/merge to `master`; Release Engineer opens a **draft** PR only.
5. Do not “repair” failures by weakening tests unless the Architect updates the contract.
6. Record every check command and exit code in the handoff.
7. Prefer smallest diff; do not drive-by refactor.

## Out of scope for default sessions
- Installing system packages or recreating venvs unless asked
- Enabling Agent Teams
- Contacting external services beyond Hugging Face model download needed by tests
