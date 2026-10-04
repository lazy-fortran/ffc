# ffc agent rules

## Goals and architectural freedom

Issues specify desired behavior, indispensable contracts and independent
acceptance evidence. Make architectural decisions as soon as required and as
late as possible. Implementing agents may simplify or replace internal designs;
[shared principles](https://github.com/lazy-fortran/fo/blob/main/doc/GOAL_DRIVEN_DEVELOPMENT.md)
apply. Substantially reduce maintained code through
[Fo #205](https://github.com/lazy-fortran/fo/issues/205), without moving complexity
between dependencies or weakening language support and useful oracles.

## Destination and first reads

Implementation and focused verification are authorized under the workspace
master plan. Start the FFC pilot once its actual Fo/dispatcher blockers pass. The entire
Fo feature or cleanup backlog is not a prerequisite. Take the earliest unblocked
FFC goal without waiting for another permission request. Read the workspace master PLAN.md/AGENTS.md when available; the
repository plan is independently readable.

Make ffc compile, link and correctly execute all modern standard Fortran through
Fortran 2023, including ISO parallel facilities. Read [PLAN.md](PLAN.md) for the
project map, active stage, issue order and verifier. It is the only ffc roadmap.
Read the [fo Gremlin plan](https://github.com/lazy-fortran/fo/blob/main/PLAN.md)
for the provider engine. Repair only actual consumer blockers before the pilot.
Load affected contract/reference documents when their feature/ABI trigger fires.

## Gremlin mode and work ownership

Use fo Gremlin mode for continuous randomized background testing. CLI and MCP
must have the same shared engine and feature set. If MCP cannot reload after a fix,
use the updated CLI immediately. **Fix fo whenever its build, cache, selection,
process lifecycle or result transport fails; do not add an ffc workaround.**
Proposed commands in PLAN are unavailable until their implementation is verified.

Serial mode works only in the main session without subagents; a local model
may be that session. Parallel mode uses a luna coordinator and independent luna
workers up to configured capacity. Serial escalation uses the GPT skill with
Sol; parallel escalation transfers only the failing task to a native Sol worker.
Each implementation/fix
worker gets a writable Git worktree; one controller owns protected integration,
authoritative state, issue updates, main commits/pushes and installed tools.
Workers may commit task branches, returning exact commits or base-plus-patch
SHA256 receipts; they never self-promote. After repeated substantive failure on
an individual task, transfer its frozen evidence and exclusive ownership to sol.

Freeze source, tests/harness, runtime, dependency vector and compiler artifacts
per lane/generation. Keep last compilable-generation tests running during failed
candidate builds; the next successful build preempts only that lane's obsolete
campaign. Preserve completed receipts; interrupted cases remain untested. An old
failure does not interrupt the edit: rerun it first on the newest version, then
prioritize a reproducible current regression. Old passes transfer only through
an unchanged complete action key. Tests never mutate a worker's live build tree.

Random sampling and shuffled dispatch are separate controls. Use short subsets,
seed/order replay, targeted tests, rotating coverage debt and bounded wall/case
budgets. Continue implementation while tests run; no full-suite wait per edit.
Controller checks a combined temporary candidate before advancing integration.
Worker and integrated-generation evidence remain distinct. Final completion
still requires comprehensive frozen-version verification.

Canonical same-project starts attach one owner; different lanes obey aggregate
build/test capacity. Keep separate mutable build trees with shared valid CAS
entries. Never broad-pkill, delete live locks/caches or race global installs.
Use compact snapshots and bounded generated-artifact retention; preserve active
and pinned receipts and valuable uncommitted work. Scratch lives under /var/tmp.

## Components and build

Source → backend-neutral FortFront typed queries → ffc lowering/runtime ABI →
LIRIC session C API through ISO_C_BINDING → object/executable. ffc owns Fortran
semantics below the frontend boundary; LIRIC owns native backend emission.
Preserve supported public lowering and module/submodule behavior; internal
facades and organization may evolve when required. Binding identity, not spelling, owns symbol lookup.
No private FortFront-arena workaround, LLVM/MLIR/HLFIR revival or text-IR path.

`app/` owns the CLI; `src/` the library; `test/` original named behavioral cases
and their shared dispatcher. Keep fpm.toml library source at src/. fo derives
SUBMODULE ancestry from source, never filename order/shims. Avoid reintroducing duplicated implementation or source-order workarounds;
internal representations remain open when independent correctness is preserved.

```bash
export LIBRARY_PATH="$PWD/../liric/build"
fo build
fo test test_session_stop_code_compiler
FO_JOBS=1 fo test --random 12 --seed 1729
fo exec ffc -- empty.f90 -o empty
```

Build LIRIC in its own repository if needed. Use fo for the development loop;
direct fallback commands are only for diagnosing/fixing a broken fo workflow.

## Code, tests and contracts

- Compiler sources use explicit typing, scope-top declarations, snake_case and
  derived types ending in _t. Keep responsibilities cohesive and code readable;
  arbitrary file/procedure limits are not architectural goals.
- Fortran .and./.or. do not short-circuit. Split any guard protecting allocation,
  indexing, association or optional arguments into separate statements.
- Add a source-level behavioral case, then run
  `python3 tools/make_suite_cases.py --apply` only when adding/removing a case.
  Existing cases keep their original filenames and public names.
- Every support claim needs an independent executable/reference oracle plus
  invalid neighbors and boundaries. State/patch-shape checks are not behavioral
  tests. Never weaken/delete tests or XFAIL rows to create green.
- Update SUPPORT_CONTRACT and affected runtime/descriptor ABI docs with the
  semantic change. Native public LIRIC APIs are the default; missing provider
  contracts get atomic issues in the owning repository.
- Run focused affected/reproducer checks and report current failures precisely.
  Broad verification continues independently; never wait for CI while work remains. Inspect already completed regression receipts before
  promotion. Full exact-version suites/epochs are milestone/release gates.

## Communication and delivery

Use English for technical/scientific material. Messages/issues/PR text on
Chris's behalf end with Chris&AI on a separate final line. Preserve unrelated
edits and stage explicit paths. Rewrite plan state and modify/close issues when
work moves; commit and push main regularly under controller ownership.

Never run/schedule/install/resume work on faepop* or faepcr* without an explicit
user request naming the host. faepmac1/faepmac2 are exempt for authorized compute.
Every response/tool call stays below about 300 lines or 8k tokens; read summaries,
write long files incrementally, and repeat this rule in every worker prompt.
