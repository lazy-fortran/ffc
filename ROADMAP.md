# ffc convergence roadmap

Snapshot: 2026-08-06, Europe/Vienna. This document is the execution plan for
bringing `ffc` to its declared language scope. The completion, reliability,
and performance criteria are defined below. Generated reports own live counts.

## Goal and denominator

`ffc` targets standard Fortran through F2023 plus accepted Lazy Fortran. The
maintained FortFront F90 and LF corpora are full targets. Every pinned
LFortran and `gfortran.dg` case is classified, but invalid tests, harness-only
tests, deleted features, and vendor extensions are not language-support
claims. They stay visible as reviewed exclusions and never count as passes.

Coarrays, OpenMP, OpenACC, MPI, and device backends remain outside the current
compiler scope. [#473](https://github.com/lazy-fortran/ffc/issues/473) owns the
F2023 delta. Fortran Synthesis is a later vertical feature chain: standard
[#756](https://github.com/lazy-fortran/standard/issues/756), FortFront
[#2976](https://github.com/lazy-fortran/fortfront/issues/2976), ffc
[#632](https://github.com/lazy-fortran/ffc/issues/632), then fo
[#120](https://github.com/lazy-fortran/fo/issues/120). It does not enlarge the
present denominator before its contract is accepted.

Each corpus row must declare exactly one mode:

- compile and run with an output or exit-status oracle.
- compile only, followed by a consumer compile/link/run when it produces a
  module or object.
- reject, with a diagnostic category/location oracle and a nearby valid
  program.
- exclude, with a machine-validated category and reviewable reason.

Completion means one locked provenance epoch observed every row and produced
zero `FAIL`, `XPASS`, `XFAIL`, `FLAKYPASS`, timeout, OOM, unreviewed `SKIP`, or
`NOREF` among in-scope cases. Every pass has an independent oracle. Production
contains zero `.inc` files, no compatibility lowering path remains, supported
ABI/schema matrices pass, and performance thresholds pass with measured
uncertainty.

## Audited state: do not infer a percentage

The implementation baselines for this gate are `ffc` `5f13422`, FortFront
`ac02b4d0`, fo `32ef96d`, and LIRIC `3facb898`; the current roadmap tips are
`8d92ba6`, `5676f6e4`, `220bbb8`, and `14ca403` respectively. The code
baselines are not green: ffc and fo still lack a completed current-head suite,
while FortFront's latest completed run failed; the latter commits are
documentation-only updates.
The last checked ffc parent was `61cbceb` (CI run 30807386910, red for the
GCC14 submodule-link and formatting gates); the last checked FortFront parent
was `ac02b4d0` (run 31127135731, Windows/aggregate failed and Ubuntu
cancelled); and
the last checked fo ancestor was `e3cff007` (run 31122586327, failed/cancelled).
The external pins are LFortran
`caf87b660f803148f000046392a5da803f9fc630` and GCC
`395e3d8131c189cd58e8c8061cdc77d1c44e3822`.

The current state is not a valid parity baseline:

- [main CI run 30807386910](https://github.com/lazy-fortran/ffc/actions/runs/30807386910)
  is the last checked ffc parent and is red. The GCC14 submodule-link defect
  is fixed by ffc `45194aa`; the 18 order shims are removed by `763ba0c`.
  The formatting debt remains unchecked on current main.
- The live corpus plan contains 11,046 files: 563 FortFront F90, 265
  FortFront LF, 4,280 LFortran, and 5,938 `gfortran.dg`.
  `--sample 900` selects one global 900-file sample, not 900 per suite.
- The checked-in dashboard is pinned to older components and reports 10,924
  files. Its Markdown and TSV digests disagree.
- Current manifests contain 5,528 XFAIL rows, 288 FAIL-owner rows, 11 NOREF
  rows, and 2,303 SKIP rows. Of 5,552 rows with issue ownership, 4,524 point
  to 105 closed issues. Those are inventory facts, not current outcomes.
- There are 70 tracked production `.inc` files containing 79,155 lines. The
  5,073-line host module has 44 direct include sites. Another 323 calls use
  `find_symbol_compat` (322 invocation sites), compared with 12
  binding-keyed lookup sites.
- The only open ffc pull requests, [#596](https://github.com/lazy-fortran/ffc/pull/596)
  and [#677](https://github.com/lazy-fortran/ffc/pull/677), conflict with main
  and have failed checks. Replace their still-needed work with small current
  branches. Do not merge either branch as-is.

Until the harness repair and full provenance census land, publish no aggregate
PASS percentage and promote no row from the stale dashboard.

## Order of work

Independent worktrees may prepare changes in parallel. Main accepts one
passing, rebased commit at a time, in this order.

1. The GCC14 facade/link fix (`ffc` `45194aa`), fo parent+ancestor DAG oracle
   (`fo` `32ef96d`), and removal of the 18 ffc `_order.f90` shims (`ffc`
   `763ba0c`) and orphan include deletion (`5f13422`) are landed. Make the
   fo/format gates blocking.
2. #531's shared immutable-arena context, benchmark provenance, orphan-include
   deletion, and behavior/resource oracle are landed through `ffc` `5f13422`;
   keep the measured regression gate blocking before the census.
3. Repair observation and classification: record one immutable result per
   case, classify it offline, fix timeout/phase reporting, add the rejection
   gate, revalidate owners, and run one full provenance census.
4. Stop silent wrong code and nondeterminism: #671, #673, #626, #628, #649,
   then the remaining crashes in #576.
5. Finish binding identity and module contracts: #584 plus the procedure,
   generic, external-unit, and Lazy-specialization issues that depend on it.
6. Converge on one descriptor/expression/lifetime model: #337, #338, #348,
   #643, then #339 and their array, character, derived, and polymorphic users.
7. Extract the remaining lowering families into modules in the waves below.
   Fix the owned feature cluster while extracting it. Do not preserve a bad
   interface for the sake of a mechanical move.
8. Burn down the remaining classified feature clusters, then run exact union,
   platform, sanitizer, ABI, and performance release gates.

## Target architecture

Breaking internal changes are expected. Compatibility is kept only at a
published repository boundary and only for a bounded migration window.

- FortFront supplies an immutable typed program snapshot, declaration binding
  identities, scope/association facts, and structured diagnostics. `ffc`
  never searches private arena layout or falls back by spelling when a
  reference can be resolved by identity.
- `lowering_context_t` becomes a temporary facade over narrow services:
  binding/declarations, canonical types and descriptors, control-flow state,
  module artifacts, runtime/LIRIC emission, and diagnostics. The AST arena is
  immutable shared state, never repeatedly value-copied per procedure.
- One versioned descriptor contract represents arrays, sections, pointers,
  allocatables, character values, and polymorphic payloads. Views store base,
  element type/size, rank, bounds, strides, extent, contiguity, ownership, and
  lifetime. There is no parallel hidden-extent or bespoke-shape path.
- One typed expression engine owns conversions, calls, array plans, side
  effects, temporaries, and result lifetimes. Statement engines consume it.
  PRINT, WRITE, WHERE, FORALL, and assignment do not reimplement expression
  semantics.
- `.fmod` has a versioned schema and stable declaration/procedure identities.
  Unknown major versions, target-incompatible artifacts, and runtime ABI
  mismatches fail explicitly before code generation.
- Ordinary modules are the default implementation boundary: `private` by
  default, explicit `public` APIs, and `use, only` dependencies. A submodule is
  used only behind a deliberately stable parent interface. Extracted code may
  not depend on private host association.
- The module DAG is real source metadata. fo `32ef96d` now records both
  ancestor and immediate-parent edges with a child-first compile/run oracle;
  the 18 `_order.f90` build shims were removed by ffc `763ba0c`.
- Production `.inc` reaches zero. A monotonic check prevents additions, but it
  is only an architecture check. Every migration also needs a behavioral
  oracle.

The migration policy follows LLVM's practice for large internal API changes:
introduce the canonical boundary, migrate consumers, switch the default, and
delete the old path promptly rather than maintain two implementations
([developer policy](https://llvm.org/docs/DeveloperPolicy.html),
[opaque-pointer migration](https://llvm.org/docs/OpaquePointers.html)). GNU's
description of `INCLUDE` as literal textual insertion explains why an include
is not an interface boundary
([GNU Fortran documentation](https://gcc.gnu.org/onlinedocs/gcc-3.4.4/g77/INCLUDE.html)).
Submodules remain useful for stable interfaces and smaller rebuild cones, not
as access to a god module's private state
([WG5 N1828](https://wg5-fortran.org/N1801-N1850/N1828.pdf)).

## Complete `.inc` removal plan

Every production include is assigned exactly once below. Within a wave, land
the listed units as small independently green commits unless the note says a
pair is one semantic unit. Each extraction deletes the include in that commit.

### Wave 0: green build and dead code (landed)

- Orphan `liric_session_arrays.inc` (237 lines) was deleted by ffc `5f13422`;
  its live routines are in `liric_session_memory_bindings.f90`.
- The 18 `_order.f90` shims were removed by ffc `763ba0c` after fo's
  behavioral build-order test passed. Verify the descendant closure in every
  subsequent clean gate; do not recreate either path.

### Wave 1: typed service seams

Create the binding, immutable declaration/type, descriptor/ABI, CFG, emission,
diagnostic, and `.fmod` services described above. Stop adding fields to the
context facade. This wave removes no include by itself. It breaks the cycles
that made the current submodule extraction fail.

### Wave 2: leaves and expression engines

Extract in this order:

`reject_const_init` (362), `alloc_descriptor` (222), `inferred` (347),
`c_ptr` (362), `transfer` (439), `integer` (576), `complex` (1,330),
`intrinsics_extra` (1,322), `intrinsics` (2,141), `expr_lowering` (1,796),
`logical_reduction` (1,615), and `reduction_expr` (954).

The first extraction needs paired accepted/rejected compiler tests. Expression
work must preserve exactly-once side effects and expose #671 before refactoring
call evaluation.

### Wave 3A: structured control

`block_concurrent` (47), `do_while` (157), `goto` (343), `control` (395),
`associate` (440), `loops` (660), `select` (1,365), `where` (1,275), and
`forall` (159).

The loop/control slice includes #626 and #671 fall-through/termination
oracles. The FORALL slice fixes #673 with statement-level temporaries wherever
simultaneous assignment requires old RHS values.

### Wave 3B: calls, procedures, and characters

`proc_dummy` (322), `statement_function` (390), `arguments` (2,616),
`functions` (3,297), `functions_tail` (2,077), `character` (2,642),
`character_tail` (741), `deferred_char` (1,046), and `lazy_monomorph` (153).

Define call, result, and character descriptor contracts first. Each nested
`functions`/`character` tail moves with its parent or becomes a separately
named module with an explicit API. It does not remain textual inclusion.

### Wave 4: descriptors, storage, and arrays

`alloc_array_result` (171), `scalar_allocatable` (290), `vector_subscript`
(349), `complex_arrays` (516), `assumed_shape_descriptor` (616),
`assumed_shape_extent` (642), `char_arrays` (800), `runtime_alloc` (904),
`allocatable` (2,925), `array_elements` (4,494), and `arrays` (6,771).

Sequence by contract. First, #337 establishes section views. Next come #338
pointer arrays, #348 character descriptors, and #643 derived allocatable
arrays. #339 then deletes legacy shape metadata. #399 adds gather views and
higher array expressions. Do not retain competing descriptor representations.

### Wave 5: I/O

`internal_write` (175), `write_ops` (189), `internal_read` (270),
`io_implied_do` (325), `inquire` (350), `internal_write_compound` (369),
`read_al` (404), `read_ops` (736), `open_close` (1,057), `io_typecheck`
(1,066), `print_expr` (988), and `namelist` (1,619).

Fix #628 before freezing `open_close`. Internal write and its compound child
become one module or two explicit modules. All I/O expressions reuse the
typed expression/call engine so side effects occur exactly once.

### Wave 6: derived types, units, persistence, and module ABI

`save` (293), `equivalence` (423), `pdt` (451), `save_static` (491),
`submodules` (593), `data` (1,082), `derived_ctor` (1,091), `declarations`
(1,234), `module_vars` (1,444), `interface` (1,481), `common` (1,487),
`polymorphic` (1,276), `derived_module_ops` (2,115), `derived_types` (2,413),
and `derived_type_ops` (4,600).

This wave starts only after binding IDs, canonical descriptors, and the
versioned `.fmod` schema exist. Its files are too coupled for mechanical
parallel extraction before those seams land.

### Wave 7: orchestration

Extract `top` (1,357) last. It ends as a small program-unit and statement
dispatcher importing the family modules. The final gate finds no production
`.inc`, `_order` shim, private host dependency, or text-name fallback for a
resolvable reference.

## Minimum-run validation strategy

Optimize for distinct root causes removed per compiler build, not sampled pass
count.

### One observation, many classifications

Compilation, execution, selection, and oracle evaluation are independent of
XFAIL/SKIP manifests. A case runs once and writes an immutable raw observation:
source/dependency closure, exact compiler/runtime/tool hashes, flags, target
and environment, phase, exit status, normalized diagnostic/crash signature,
output hashes, time, peak RSS, semantic tags, and coverage. Normal,
XFAIL-disabled, and dashboard views classify that same record offline. Never
compile the same input twice merely to change a manifest.

Only completed hermetic results are cached. The key contains every declared
input, tool, flag, environment value, target, runtime ABI, corpus revision,
and harness version. Timeouts, OOMs, interrupted runs, and infrastructure
errors are never cached. This follows the action-hermeticity model documented
by [Bazel](https://bazel.build/concepts/hermeticity) and its
[remote cache](https://bazel.build/remote/caching).

### Census, clustering, and selection

After the harness is trustworthy, run one full provenance census. Cluster
failures by pipeline phase, normalized signature, feature tags, owner, and
execution behavior. Reduce one representative per cluster while preserving
the signature and independent oracle. Syntax-aware reduction is preferred.
C-Reduce demonstrates why language-aware passes beat generic line deletion
([PLDI paper](https://doi.org/10.1145/2254064.2254104)).

Build the fast semantic core with greedy set cover over observed compiler
coverage, standard-feature tags, oracle modes, dependency families, known
failure clusters, and important pairwise interactions. Backtest it against
pre-fix commits. It is accepted only if it exposes every retained historical
defect. NIST's combinatorial-testing work motivates explicit low-order
interaction coverage rather than an arbitrary fixed sample
([NIST ACTS](https://csrc.nist.gov/Projects/automated-combinatorial-testing-for-software/combinatorial-methods-in-testing/interactions-involved-in-software-failures)).

Replace repeated random seeds with deterministic, non-overlapping,
duration-balanced shards. A shard epoch's exact union covers the locked
manifest at one compiler/configuration. Sampling remains a discovery and rate
estimate only: with 11,046 cases, a clean random sample of 900 has only about
an 8.15% chance to hit one hidden failure and can still conceal about 35
failures at a one-sided 95% bound. LLVM lit already supports prior-failures
first, slowest-first ordering, coverage, timing, and explicit sharding
([lit manual](https://llvm.org/docs/CommandGuide/lit.html)).

### Gate funnel

| Gate | Required run | When |
| --- | --- | --- |
| edit | reduced reproducer plus positive/negative or compile-link-run independent oracle | after a semantic edit |
| slice | affected module closure, changed-code coverage selection, and feature/cluster representatives | before integration |
| pre-commit | full maintained `fo` suite, ABI/schema contracts, and focused corpus rows from one built compiler | once per final rebased commit |
| PR | clean build, maintained suite, semantic core, compiler/platform lane required by touched contracts | once per head revision |
| main/nightly | next non-overlapping shard, owner/expiry audit, flake probes, dashboard from raw records | once per merged revision or schedule |
| epoch/release | exact union of all shards at one provenance plus full platform, sanitizer, ABI, and performance gates | milestone only |

Order known failures first, then slow tests, and stop on the first new
signature within a developer slice. Run the rest of a cluster after the fix.
Agents may implement and prepare oracles concurrently in isolated worktrees,
but heavy builds run serially with `FO_JOBS=1` on this host. Each branch owns
one module family. Rebase on current main, merge each passing commit,
rerun its focused post-merge gate, push, and update its docs in the same
commit. Never accumulate a multi-feature integration branch.

### Independent oracles

- Standard runnable code uses a checked-in expected output or assertion when
  that oracle is authoritative. Only cases without an authoritative
  self-checking oracle require pinned gfortran plus a second independent
  compiler; a disagreement is `ORACLE_PENDING`, not PASS.
- Lazy syntax uses a standard-Fortran desugared twin, small independent
  evaluator, or proven metamorphic relation.
- Rejection requires the expected category/location and a valid neighboring
  program. This catches over-rejection.
- A module/object case compiles a separate consumer, links it, and checks its
  behavior. Artifact existence is not a test.
- Generated and mutated programs must be initialized and defined. Csmith's
  differential strategy found hundreds of compiler bugs with well-defined
  programs ([PLDI paper](https://doi.org/10.1145/1993498.1993532)). EMI adds a
  metamorphic oracle by changing unexecuted code for controlled inputs
  ([PLDI paper](https://doi.org/10.1145/2594291.2594334)).
- Coverage-guided overnight generation starts from the minimized real corpus.
  Retain only new coverage, a new feature/value interaction, or a new failure
  signature. Periodically merge/minimize the corpus as documented for
  [libFuzzer](https://llvm.org/docs/LibFuzzer.html).

### XFAIL, flakes, and timeouts

Every XFAIL row has an exact selector, open owner, expected phase/signature,
introduced revision, last-confirmed revision, and expiry. A changed signature
is FAIL. An XPASS, expired XFAIL, timeout, OOM, or pass-on-retry blocks its
gate. Retrying one failed case is classification. A pass on retry remains
`FLAKYPASS` and never contributes to 100%.

For each root cause, reproduce it and establish the oracle. Apply the fix and
run the complete affected cluster. Remove the exact manifest rows in the same
commit. Classify any new signature rather than broadening the XFAIL. Dashboard
regeneration is batched from immutable records and never reruns cases.

Timeouts are phase-specific and derived from historical distributions. One
isolated extended retry identifies a slow case. It does not turn the result
into PASS or justify raising a global timeout. #478 and #531 share this work:
fix classification and superlinear compilation, then set thresholds from the
new distributions.

## Performance gates

Correctness is checked before timing. Record cold full build, warm
single-module edit, frontend/lowering/backend phase time, peak RSS, artifact
size, and generated-program runtime separately. Keep a representative
CTMark-like compile corpus rather than timing every conformance file on every
commit.

Use a controlled host, pinned core/governor, randomized old/new interleaving,
and multiple repetitions. Report effect size and confidence interval. Gate
only when the interval lies wholly beyond a predeclared practical regression
threshold. Do not gate on a single elapsed time. LLVM LNT recommends
sequential builds and controlled CPU conditions for compile-time measurement
([LNT guide](https://llvm.org/docs/lnt/tests.html)). Rigorous experimental
design and confidence reporting are described by
[Kalibera and Jones](https://doi.org/10.1145/2464157.2464160).

The immediate clean parent with identical FortFront/LIRIC/toolchain is the
causal performance differential. A preceding released `ffc` binary remains a
separate trend baseline. #531's 5,000-line program must first match gfortran
output, then demonstrate the expected compile-time/RSS improvement without
moving cost into another phase.

## Active issue map

The list below is generated from open GitHub issues as of 2026-08-06. Closed
issue numbers must not own new manifest rows.

| Workstream | Open ffc issues |
| --- | --- |
| wrong code, hangs, crashes, nondeterminism | [#576](https://github.com/lazy-fortran/ffc/issues/576), [#626](https://github.com/lazy-fortran/ffc/issues/626), [#628](https://github.com/lazy-fortran/ffc/issues/628), [#649](https://github.com/lazy-fortran/ffc/issues/649), [#671](https://github.com/lazy-fortran/ffc/issues/671), [#673](https://github.com/lazy-fortran/ffc/issues/673) |
| binding, calls, procedures, modules, Lazy identity | [#415](https://github.com/lazy-fortran/ffc/issues/415), [#433](https://github.com/lazy-fortran/ffc/issues/433), [#437](https://github.com/lazy-fortran/ffc/issues/437), [#449](https://github.com/lazy-fortran/ffc/issues/449), [#453](https://github.com/lazy-fortran/ffc/issues/453), [#456](https://github.com/lazy-fortran/ffc/issues/456), [#461](https://github.com/lazy-fortran/ffc/issues/461), [#467](https://github.com/lazy-fortran/ffc/issues/467), [#522](https://github.com/lazy-fortran/ffc/issues/522), [#579](https://github.com/lazy-fortran/ffc/issues/579), [#582](https://github.com/lazy-fortran/ffc/issues/582), [#584](https://github.com/lazy-fortran/ffc/issues/584), [#609](https://github.com/lazy-fortran/ffc/issues/609) |
| descriptors, arrays, characters, derived values | [#337](https://github.com/lazy-fortran/ffc/issues/337), [#338](https://github.com/lazy-fortran/ffc/issues/338), [#339](https://github.com/lazy-fortran/ffc/issues/339), [#348](https://github.com/lazy-fortran/ffc/issues/348), [#399](https://github.com/lazy-fortran/ffc/issues/399), [#419](https://github.com/lazy-fortran/ffc/issues/419), [#422](https://github.com/lazy-fortran/ffc/issues/422), [#435](https://github.com/lazy-fortran/ffc/issues/435), [#458](https://github.com/lazy-fortran/ffc/issues/458), [#459](https://github.com/lazy-fortran/ffc/issues/459), [#462](https://github.com/lazy-fortran/ffc/issues/462), [#465](https://github.com/lazy-fortran/ffc/issues/465), [#643](https://github.com/lazy-fortran/ffc/issues/643), [#669](https://github.com/lazy-fortran/ffc/issues/669) |
| control, I/O, rejection | [#345](https://github.com/lazy-fortran/ffc/issues/345), [#455](https://github.com/lazy-fortran/ffc/issues/455), [#460](https://github.com/lazy-fortran/ffc/issues/460), [#581](https://github.com/lazy-fortran/ffc/issues/581) |
| corpus truth, build cost, observability | [#475](https://github.com/lazy-fortran/ffc/issues/475), [#478](https://github.com/lazy-fortran/ffc/issues/478), [#531](https://github.com/lazy-fortran/ffc/issues/531), [#532](https://github.com/lazy-fortran/ffc/issues/532), [#540](https://github.com/lazy-fortran/ffc/issues/540), [#663](https://github.com/lazy-fortran/ffc/issues/663) |
| scope | [#473](https://github.com/lazy-fortran/ffc/issues/473), [#632](https://github.com/lazy-fortran/ffc/issues/632) |

Architectural supersets guide implementation, not issue closure:

- #584 and FortFront #2975 establish binding-keyed scope/call identity for the
  procedure and module cluster. Closed #327/#330/#332/#457 were foundations,
  not completion.
- The descriptor/expression/lifetime model covers #337/#338/#339/#345/#399/
  #419/#422/#435/#458/#459/#465/#643. #673 is the remaining wrong-code slice
  of #345.
- FortFront #2980/#2994 establish Lazy specialization identity before ffc #437
  emits it and #433 serializes it.
- #478/#531 share measurement and timeout work. #532/#540 share one provenance
  refresh. Rebaseline the stale umbrellas #576/#609 and replace them with
  exact live signatures.

## Cross-repository gates

| Owner | Active contract before ffc can close the dependent work |
| --- | --- |
| FortFront | rejection/accepted-side correctness [#2883](https://github.com/lazy-fortran/fortfront/issues/2883), [#2897](https://github.com/lazy-fortran/fortfront/issues/2897), [#2924](https://github.com/lazy-fortran/fortfront/issues/2924), [#2951](https://github.com/lazy-fortran/fortfront/issues/2951), [#2970](https://github.com/lazy-fortran/fortfront/issues/2970), [#2986](https://github.com/lazy-fortran/fortfront/issues/2986), [#2987](https://github.com/lazy-fortran/fortfront/issues/2987), [#2993](https://github.com/lazy-fortran/fortfront/issues/2993), and continuation-comment lexer defect [#2996](https://github.com/lazy-fortran/fortfront/issues/2996). Parser/identity [#2973](https://github.com/lazy-fortran/fortfront/issues/2973), [#2975](https://github.com/lazy-fortran/fortfront/issues/2975), [#2980](https://github.com/lazy-fortran/fortfront/issues/2980), [#2994](https://github.com/lazy-fortran/fortfront/issues/2994) |
| LIRIC | Restore producer artifacts and explicit build-failure reporting in [#533](https://github.com/krystophny/liric/issues/533) before relying on the semantic public-session gate [#523](https://github.com/krystophny/liric/issues/523); current Bench Matrix, scheduled compatibility, and nightly runs are red |
| fo | [#117](https://github.com/lazy-fortran/fo/issues/117) for the assignment-name-ending-in-function oracle before formatting changes, [#119](https://github.com/lazy-fortran/fo/issues/119) for unbounded JSON results, parent/submodule DAG discovery and removal of ffc order shims, and [#103](https://github.com/lazy-fortran/fo/issues/103) for diagnostics |
| fluff | [#262](https://github.com/lazy-fortran/fluff/issues/262): verify every test executable can fail the build before fo #59 consumes deep-lint JSON |
| fx | [#36](https://github.com/lazy-fortran/fx/issues/36): concurrent cache survival is an editor reliability gate, not corpus parity |
| standard | #745 arrays, #753 module signatures, and #756 Synthesis are design inputs until accepted. Current conformance does not wait for future proposals |

Pinned gfortran is an oracle with known hazards, not an authority in every
case. Track the relevant GCC wrong-code/acceptance defects and require a second
oracle whenever a case intersects one.

## Roadmap and merge discipline

- Generate the state block and issue-owner audit from one locked observation
  store. Keep live totals in that generated dashboard rather than copying them
  into roadmap prose.
- Every semantic, ABI, corpus-classification, or architecture commit updates
  the affected contract documentation in the same commit.
- A change that cannot pass its independent oracle does not merge. Never
  weaken a test, widen an XFAIL, or call an infrastructure failure evidence.
- Use isolated worktrees under `/mnt/storage`. Keep one owner per module
  family, serialize heavy builds, and merge/push small green commits early.
- Record exact component revisions, commands, failing set, phase/signatures,
  and artifact locations. Compare sets and signatures, not only counts.
- If main is red, stop feature merges. Restore the gate or explicitly revert
  the offending slice before proceeding.
- Refresh this roadmap when scope, architecture order, or an active
  cross-repository contract changes. Historical session narratives belong in
  git history, not appended below the live plan.
