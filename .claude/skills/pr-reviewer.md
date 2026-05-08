# PR Reviewer

Use when asked to review a pull request, e.g. "review PR #N", "can you review this PR", "evaluate this PR".

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
  - Branches whose condition is always true/false given the new logic
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

**Step 11 — Tuning engine consistency** *(skip only if the diff touches no strategy logic)*

Any change to how the strategy discovers, scores, enters, or exits pairs may require tuning engine updates. Check each of the following:

- **New parameter exposed**: is it in `tuning/parameter_space.py` with the correct tier, default, and search bounds? If it is a composite score weight, is it added to `_WEIGHT_NAMES`?
- **Parameter removed or renamed**: does `parameter_space.py` still reference it? Is it still in `BobsBrain.initialize()` defaults?
- **Composite score changed**: does `normalize_weights()` in `parameter_space.py` handle the full new weight set?
- **Settings dict in `create_run()`**: does it include every new tunable param so runs are self-describing in the DB?
- **Tier boundaries**: does any new param belong in a tier other than what was chosen?

Flag any mismatch as **blocking** — an unregistered parameter cannot be optimized by Optuna and silently uses its default across all tuning runs.

**Step 12 — Documentation** *(README and inline docs)*

- If the PR changes user-visible behaviour, is `README.md` updated?
- If the PR adds or changes operational steps, are those reflected in the README or relevant docs?
- Do new public methods/functions have docstrings that explain *what* and *why*?
- Flag missing README updates as **blocking** when the gap would leave a user unable to configure or operate the changed behaviour correctly; flag as **non-blocking nit** otherwise.

**Step 13 — Commit hygiene**
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
