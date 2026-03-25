---
name: pr-reviewer
description: Structured PR review workflow for LumiBob. Use when asked to review a pull request, e.g. "review PR #N", "can you review this PR", "evaluate this PR".
---

# PR Reviewer

Never merge the PR. Output a structured written review only.

## Tooling

- `gh pr view <N> --json title,body,files,additions,deletions,commits`
- `gh pr diff <N>`
- `pytest tests/ -v` — only on APPROVE verdict

---

## Review Steps

**Step 1 — Load PR context**
Run both gh commands above. Report: title, purpose, files changed, net line delta, commit count and messages.

**Step 2 — Description quality**
- Accurately reflects the diff (flag discrepancies)
- Explains *why*, not just *what*
- Calls out known limitations or intentional trade-offs

**Step 3 — Test plan soundness**
- Covers happy path end-to-end
- Enumerates edge cases and failure paths
- Reproducible — are manual steps concrete?

**Step 4 — Code quality**
- **Modularity**: single-purpose functions/classes, appropriate separation of concerns
- **Simplicity**: no over-complication, redundant logic, or unnecessary abstractions
- **Readability**: names communicate intent
- **Comments**: non-obvious decisions explained; obvious code left uncommented
- **Scope discipline**: PR focused on one concern — flag bundled unrelated changes
- **Dead / legacy code**: flag as **blocking** any of:
  - Branches whose condition is always true/false given the new logic (e.g. `if signal_type == 'zscore': ... else: <old path>` when all pairs are now Z-score)
  - Fields written into dicts or objects that are never read by live code
  - Imports, helpers, or methods no longer called after the change
  - `else` branches kept "for backward compatibility" when no caller can produce the old input

**Step 5 — Performance**
- O(N²) loops where a better structure exists
- Row-wise pandas iteration where vectorised operations apply
- Repeated I/O inside loops that could be batched or cached
- Unnecessary recomputation of values derivable once

**Step 6 — Test coverage**
For every function/method touched in the diff: is there a corresponding test? Were existing tests updated? Flag untested functions by name.

**Step 7 — Error handling**
- Exceptions caught at the right abstraction level
- Failure modes communicated clearly (logged, raised, or returned)

**Step 8 — Breaking changes**
- Public-facing behavior, function signatures, or return types changed
- Log/DB schema altered without updating the relevant rule

**Step 9 — Security**
- Hardcoded credentials, API keys, or secrets
- User input passed to shell commands, file paths, or eval

**Step 10 — Dependency hygiene** *(only if `requirements.txt` changed)*
- New dependency necessary and justified, pinned to a version, actively maintained

**Step 11 — Commit hygiene**
- Commits atomic and scoped; messages descriptive (not "fix", "WIP", "changes")

---

## Verdict

Issue exactly one of:

**APPROVE** — all criteria met; list any non-blocking nits separately.

**REQUEST CHANGES** — one or more blocking issues. For each:
- The step it violates (e.g. "Step 4 — Dead code")
- File and line number(s) from the diff
- A concrete suggested fix

---

## Post-Approval Test Run

If verdict is **APPROVE**, run `pytest tests/ -v`. Report total run/pass/fail counts and any failure messages. If any tests fail, escalate to **REQUEST CHANGES** citing the failing tests.
