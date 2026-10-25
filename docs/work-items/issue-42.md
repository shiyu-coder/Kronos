# Work item: Issue #42 — Refactoring (uv, ruff, pre-commit)

- **Issue:** [shiyu-coder/Kronos#42](https://github.com/shiyu-coder/Kronos/issues/42)
- **State:** OPEN
- **Title:** Refactoring...
- **Requester:** @cove9988 · **Owner reply:** @shiyu-coder — open to merge; prefer **series of focused PRs**
- **Contract status:** DRAFT — awaiting human approval (do not launch workers)

## Source summary
Issue asks to introduce Python project-management / quality tooling:
1. **uv** for dependency / project management
2. **ruff** for linting
3. **pre-commit** hooks

Owner accepted the direction if delivered as small, reviewable PRs.

## Acceptance criteria
- [ ] Project can be installed / synced with **uv** without breaking the documented baseline: `.venv` + `pip install -r requirements.txt` path still works **or** README documents the new uv-only path clearly.
- [ ] **ruff** is configured (config file + documented command) and runs clean on an agreed file set, **or** known violations are tracked with an explicit allowlist / incremental plan.
- [ ] **pre-commit** config exists with hooks that call the agreed checks (at least ruff; optionally format).
- [ ] Changes land as **focused PRs** (suggested split below), each with green baseline checks.
- [ ] Baseline gate remains: `/workspace/Kronos/.venv/bin/python -m pytest tests/ -q` → **4 passed** (or documented equivalent under uv env).
- [ ] No change to model weights / Hugging Face revision pins used by regression tests unless Test Engineer + Architect explicitly amend this contract.

## Shared API / data contracts
- **Public Python API unchanged** unless a later PR amends this contract:
  - `from model import Kronos, KronosTokenizer, KronosPredictor`
- **Regression fixtures** under `tests/data/` (`regression_input.csv`, `regression_output_*.csv`) stay bit-compatible; do not regenerate without Test Engineer sign-off.
- **Dependency contract (v1 proposal):**
  - Keep `requirements.txt` as the consumer-facing install list **or** generate/export it from uv (`pyproject.toml`) so README “pip install -r requirements.txt” stays true.
  - Pin policy: prefer locking via uv (`uv.lock`) while preserving README-compatible install.
- **Tooling config locations (proposed ownership):**
  - `pyproject.toml` / `uv.lock` — Backend + Release
  - `ruff` config (in `pyproject.toml` or `ruff.toml`) — Backend
  - `.pre-commit-config.yaml` — Backend + Release
  - `README.md` install/docs — Architect reviews; Backend edits; Release includes in PR body
- **No HTTP/UI API** in this issue unless Frontend is explicitly assigned a docs-only webui note (default: **out of scope**).

## File ownership by worker

| Worker | Owns (write) | Must not touch |
| --- | --- | --- |
| **Architect** | This work-item, contract clarifications, assignment notes | Product logic in `model/` |
| **Backend Engineer** | `pyproject.toml`, `uv.lock`, `requirements.txt` sync, `ruff` config, `.pre-commit-config.yaml`, minimal packaging metadata | `webui/` templates/UI; regenerating golden CSVs |
| **Frontend Engineer** | *None for #42 unless Architect amends* | Everything (default idle) |
| **Test Engineer** | `tests/` only if hooks/env break imports; optional CI script under `tests/` or docs for how to run pytest under uv | Broad refactors of `model/` |
| **Code Reviewer** | Read-only | All writes |
| **Release Engineer** | Branch integration, draft PR series, PR descriptions | New feature code beyond merge conflicts |

### Suggested PR series (focused)
1. **PR-A:** add `pyproject.toml` + uv workflow; keep `requirements.txt` compatible
2. **PR-B:** add ruff config + fix/allowlist for first clean run
3. **PR-C:** add pre-commit hooks wiring ruff (+ document usage)
4. *(optional later)* README polish / CI — only after A–C approved

## Task dependencies
```
Architect contract (this file, approved)
    → Backend PR-A (uv / pyproject)
        → Test Engineer verify install + pytest
            → Code Reviewer on PR-A tip
                → Release Engineer draft PR-A
    → Backend PR-B (ruff) [after PR-A merged or stacked]
        → Test + Review + Release
    → Backend PR-C (pre-commit) [after PR-B]
        → Test + Review + Release
Frontend: blocked / N/A unless contract amended
```

## Required checks & handoff evidence
Every worker handoff must include: **changed files**, **commit SHA**, **checks performed (cmd + exit code)**, **unresolved blockers**.

### Baseline (every PR tip)
```bash
/workspace/Kronos/.venv/bin/python -m pip install -r requirements.txt   # or documented uv sync
/workspace/Kronos/.venv/bin/python -m pytest tests/ -q                 # expect 4 passed
```

### Tooling checks (once introduced)
```bash
uv --version
uv sync   # or documented equivalent
ruff check .
pre-commit run --all-files   # after hooks land
```

### Release evidence
- Draft PR URL(s), stacked or sequential as owner requested
- Link back to issue #42
- Note: **never merge to `master`** from agents

## Missing requirements (flags)
- Issue does **not** specify: Python version floor beyond README 3.10+, whether to replace pip entirely, which ruff rule set, format-on-save, CI provider, or whether `webui/requirements.txt` joins the uv workspace.
- Issue does **not** define success metrics (0 ruff errors vs incremental).
- Owner asked for **series of PRs** but not the split; split above is a proposal.
- No mention of `finetune/` / `pyqlib` extras — treat as **out of scope** until named.
- GPU/CUDA not required for this issue; do not block on CUDA.

## Overlapping ownership (flags)
- **`requirements.txt` vs `pyproject.toml`:** Backend and Release both care — Backend owns content; Release owns PR packaging; Architect resolves conflicts.
- **`README.md`:** Backend may edit install section; Architect must approve wording; avoid Frontend edits.
- **`tests/`:** Test Engineer owns assertion changes; Backend must not weaken regression tolerances to silence tooling.
- **pre-commit vs ruff:** both Backend; order is PR-B then PR-C to avoid double-ownership fights in one diff.

## Explicit non-goals
- Enabling Claude experimental Agent Teams
- Launching workers before this contract is **APPROVED**
- Behavioral changes to forecasting / tokenizer / predictor
- Merging PRs to `master`

## Approval
- [x] Human approves this contract (or lists amendments)
- [ ] Architect may then launch workers per dependency order

**Reviewed SHA of repo at contract draft:** `67b630e67f6a18c9e9be918d9b4337c960db1e9a` (`master`)
