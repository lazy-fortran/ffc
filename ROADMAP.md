# ffc convergence roadmap

Snapshot: 2026-08-07, Europe/Vienna. This document is the execution plan for
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

The current implementation heads for this gate are ffc `eb39ce6` (code
baseline, with typed array-shape extraction plus the
GCC14 descendant-link export fix), FortFront `704cd27f` (semantic code
baseline, with separate module-procedure dummy resolution and implicit
DIMENSION dummy preservation,
IMPLICIT NONE
undeclared-name diagnostics, nested binding identity, continuation-comment,
and inline-IF fixes), fo `32ef96d`,
and LIRIC `3facb898`. The code baselines are not green: ffc and fo still lack
a completed current-head suite, while the latest FortFront run is pending.
The ffc focused observation gate is green locally;
the checked-in parity snapshot remains stale and is not a release baseline.
The final local `fo` gate at `10b2af1` built 435/435 units and ran 350 tests;
the observation smoke passed and the run retains 26 named pre-existing
compiler/dashboard failures. That exit is not a clean release gate.
The current ffc merge CI is [run 31136869798](https://github.com/lazy-fortran/ffc/actions/runs/31136869798)
(in progress); the prior run [31136459600](https://github.com/lazy-fortran/ffc/actions/runs/31136459600)
was docs-only and had the known formatting failure while its build/test job
continued. FortFront merge CI is [run 31137139530](https://github.com/lazy-fortran/fortfront/actions/runs/31137139530)
(queued for current main `b4f1d2f2`); and
the last checked fo ancestor was `e3cff007` (run 31122586327, failed/cancelled).
The external pins are LFortran
`caf87b660f803148f000046392a5da803f9fc630` and GCC
`395e3d8131c189cd58e8c8061cdc77d1c44e3822`.

The current state is not a valid parity baseline:

- The current-head CI links above are intentionally not treated as green until
  they complete. The GCC14 submodule-link failure seen in [run
  31134735351](https://github.com/lazy-fortran/ffc/actions/runs/31134735351)
  is addressed by the explicit descendant-helper exports in ffc `5f1677d`
  (merged as `eb39ce6`). The failure was independently reproduced with
  GCC14.2 on `faepkub4`; a clean `fpm build` and the array-shape module
  consumer oracle now link and pass there. Formatting and aggregate debt
  remain visible until the merge CI above completes.
- The live corpus plan contains 11,046 files: 563 FortFront F90, 265
  FortFront LF, 4,280 LFortran, and 5,938 `gfortran.dg`.
  `--sample 900` selects one global 900-file sample, not 900 per suite.
- The checked-in dashboard is pinned to older components and reports 10,924
  files. Its Markdown and TSV digests disagree.
- Current manifests contain 5,527 XFAIL rows, 288 FAIL-owner rows, 12 NOREF
  rows, and 2,305 SKIP rows. Of 5,552 rows with issue ownership, 4,524 point
  to 105 closed issues. Those are inventory facts, not current outcomes.
- There are 67 tracked production `.inc` files containing 77,394 lines. The
  5,219-line host module has 42 direct include sites. Another 322 invocation
  sites use `find_symbol_compat`, compared with 12 binding-keyed lookup sites.
- The only open ffc pull requests, [#596](https://github.com/lazy-fortran/ffc/pull/596)
  and [#677](https://github.com/lazy-fortran/ffc/pull/677), conflict with main
  and have failed checks. Replace their still-needed work with small current
  branches. Do not merge either branch as-is.

Until the full provenance census and dashboard regeneration land, publish no
aggregate PASS percentage and promote no row from the stale dashboard.

This ordering follows the established compiler-testing practice of keeping
compile-only, execute, and directive outcomes distinct in [GCC's DejaGnu
testsuites](https://gcc.gnu.org/onlinedocs/gccint/Testsuites.html), and of
using differential plus metamorphic oracles for compiler correctness (the OSDI
[*Gauntlet* study](https://www.usenix.org/system/files/osdi20-ruffy.pdf)). The
locked epoch makes those oracles reproducible: one immutable input/toolchain
descriptor, one raw observation, then offline classification. Accelerator
gates use the same rule and pin the NVHPC/CUDA compiler-driver compatibility
required by [NVIDIA's release notes](https://docs.nvidia.com/hpc-sdk/hpc-sdk-release-notes/index.html).

## Order of work

Independent worktrees may prepare changes in parallel. Main accepts one
passing, rebased commit at a time, in this order.

1. The GCC14 facade/link fix (`ffc` `45194aa`), fo parent+ancestor DAG oracle
   (`fo` `32ef96d`), and removal of the 18 ffc `_order.f90` shims (`ffc`
   `763ba0c`) and orphan include deletion (`5f13422`) are landed. Make the
   fo/format gates blocking.
2. #531's shared immutable-arena context, benchmark provenance, orphan-include
   deletion, and behavior/resource oracle are landed through `ffc` `5f13422`.
   The benchmark now has a blocking 10% median wall-time and peak-RSS gate,
   immutable worktree provenance, and a focused positive/negative test;
   record one idle-host baseline/candidate report before closing the issue.
3. The observation/classification repair is landed in `ffc` `21adf72`, and
   the allocatable descriptor include is now a typed descendant submodule in
   `ffc` `935fd5d`: one
   immutable schema-2 result per case with source/closure, compiler flags,
   environment/target/runtime/harness/toolchain digests, diagnostics/output
   hashes, timing/RSS, semantic tags, coverage mode, strict offline views,
   atomic publication, exact repeat merging, and FAIL/XPASS/FLAKY gates. The
   reference cache is success-only, hash-validated, and hermetic across
   environment/target/flags/runtime/harness/corpus/closure. Run one full locked
   provenance census next, then regenerate the stale dashboard before
   promoting rows.
   The breaking execution-epoch tranche now gives every raw row one immutable
   epoch, a declared action mode, and separate ffc/reference compile/run action,
   exit, timeout, and signal evidence. Its supervisor distinguishes actual
   timeouts/signals from deliberate exits 124/137, and strict validators reject
   mixed epochs or inconsistent evidence. General Fortran INCLUDE closure
   snapshotting now hashes canonical suite-relative names and the same copied
   bytes passed to both compilers, including nested INCLUDE files. Instrumented
   coverage remains; schema-2 records `coverage_mode=none` with an empty digest
   until that collector lands.
   SKIP/NOREF remain operational dispositions and are not silently PASS.
   The 1,094-line OPEN/CLOSE/file-unit WRITE lowering slice is now a typed
   `session_program_lowering_open_close.f90` descendant with 30 explicit
   module-procedure interfaces; the textual include is deleted. Its existing
   dynamic `STATUS=` and OPEN/WRITE/CLOSE compiler oracles remain the behavioral
   gate while the remaining I/O slices are migrated.
4. Stop silent wrong code and nondeterminism. #671's binding-keyed host
   storage fix and #673's fixed-intrinsic FORALL snapshot are landed with
   gfortran oracles; #626's nested-DO tail now has an end-to-end ffc oracle and
   is green against FortFront `f3ab76ba` or newer. That FortFront parser fix
   replaces the faulty per-construct span scanners with one case-insensitive
   terminator scanner, so a two-word `end do` closes exactly once and cannot
   absorb statements after a nested loop. Keep the ffc regression
   `test_session_nested_do_tail_compiler` in every locked epoch; do not add an
   ffc filename/order workaround. Close the tracker only after the dependency
   pin and current-head CI prove the same behavior. Continue with #649, then
   the remaining crashes in #576. #649's character MINLOC/MAXLOC path now compares complete
   blank-padded values and preserves first-match/mask semantics; its
   `minmaxloc_11.f90` XFAIL is removed. The RANDOM_NUMBER branch fixture is
   explicitly `nondeterministic-runtime-value` NOREF, so it is built and
   terminated by both processors without a coin-flip structural comparison.
   #628's dynamic `STATUS=` value path is also landed; retain its runtime
   regression while the remaining I/O families are extracted. #669 is now a
   complete parser/lowering vertical slice: FortFront `aa5880ae` preserves
   ordinary `c(i)` subscripts while retaining `c(i)(l:u)` as an
   `array_slice_node`, and ffc `9f26886` lowers the nested view through the
   typed `session_program_lowering_character.f90` descendant. Its read,
   literal-write, overlapping self-assignment, and assumed-length actual
   argument are compared byte-for-byte with gfortran; the touched routines
   are removed from the character and argument includes. Keep the focused
   oracle pinned to that FortFront revision before corpus promotion.
   The accepted-side rejection gate for #663 is now a committed, runnable
   `scripts/corpus_rejection_gate.sh` plus `make check-rejection-gate` and an
   expectation-neutral FortFront baseline. It records one compile disposition
   and all per-file diagnostics, compares only baseline `ACCEPTED` to current
   `REJECTED` transitions, and runs `gfortran -fsyntax-only` on each transition
   as independent validity triage. A syntax-oracle result never suppresses the
   gate: intended new rejections require a reviewed `--allow` entry and a
   nearby accepted oracle. Keep this gate ahead of every rejection-rule change
   and refresh its baseline only in a locked provenance epoch. The committed
   baseline is the 833-file FortFront `ee5caf7b` snapshot (626 accepted, 207
   rejected) measured with ffc `bb30c20`.
   A bounded live audit of #576 at ffc `8ffc35e` (2026-08-07) split all 563
   live `fortfront-f90` files and all 265 `fortfront-lf` files into parallel
   chunks (including a three-file tail check after the initial eight chunks).
   The checked-in gate totals (517 and 264) are stale against this live file
   set and need a corpus-truth refresh, not a manifest exemption.
   It found no emitted-program runtime segfault, but did reproduce one stable
   compiler crash: `fortfront-lf/issue_2064_logical_return_inferred_as_integer.lf`
   exits 139 from compile with signal 11 on three independent runs
   (`crash_signature_sha256=700fd752c286a890...`). The non-short-circuit
   external-procedure guard in `lower_logical_call` now fixes this crash, with
   `test_session_logical_result_call_compiler` as the positive behavioral
   oracle; retain the corpus FAIL until the current-head gate reclassifies it.
   A bounded 2026-08-07 owner audit confirms that #540/#532 cannot be repaired
   by deleting rows from the current manifests: the live owner validator
   reports 105 closed issue IDs (82 ffc, 23 FortFront) across owner-bearing
   XFAIL/FAIL_OWNER sets, while retained schema-v2 reports cover only the 563
   and 265 FortFront suites. Keep every row visible and run one locked
   four-suite provenance epoch before promoting XPASS rows, regenerating the
   dashboard, or changing ownership; existing owner/XPASS validators already
   have independent fake-GitHub behavioral fixtures.
5. Finish binding identity and module contracts: #584 plus the procedure,
   generic, external-unit, and Lazy-specialization issues that depend on it.
6. Converge on one descriptor/expression/lifetime model: #337, #338, #348,
   #643, then #339 and their array, character, derived, and polymorphic users.
7. Extract the remaining lowering families into modules in the waves below.
   Fix the owned feature cluster while extracting it. Do not preserve a bad
   interface for the sake of a mechanical move.
8. Burn down the remaining classified feature clusters, then run exact union,
   platform, sanitizer, ABI, and performance release gates.

### #584 binding-identity audit (2026-08-07)

The current main slice has a single FortFront binding-triple lookup for host
storage (`declaration_node_index`, `declaration_entity_index`, and
`scope_node_index`) and a typed `.fmod` export record for a public procedure
whose direct-session ABI is not yet callable. The existing derived-dummy
subroutine regression covers the latter boundary. This audit adds the
function-shaped counterpart in `test_session_read_fmod_compiler`: a separate
module and `USE ONLY` consumer must compile and run, and the same source must
also compile and run with gfortran as the accepted-side oracle. The test is
green on main with `fo test test_session_read_fmod_compiler` (437/437 build
units; 1/1 test) when the LIRIC library directory is supplied through
`LIBRARY_PATH`. No production `.inc` path or text-name fallback was added.

This is a boundary regression, not closure of #584. The remaining corpus
failures require FortFront facts that ffc cannot reconstruct safely:

- FortFront #2974 must emit one binding/declaration entity for every name in a
  compound declaration after a non-constant array specification
  (`legacy_array_sections_03.f90`). ffc must not infer the missing scalar from
  its use site.
- FortFront #2975's nested-`ASSOCIATE` owner-binding correction is now landed
  at `d1c6a894`; the ffc direct-session regression
  `test_session_associate_selectors_compiler` passes against that revision.
  FortFront `8d3e5a8a` also resolves separate module-procedure dummies from the
  parent interface identity; the ffc consumer gate must use that fact rather
  than reconstructing dummy bindings by spelling.
  The remaining host-storage work must consume that identity and reject an
  absent edge rather than re-register by spelling.
- Host-associated polymorphic `CLASS(t), ALLOCATABLE` selectors in
  `class_is_1_ok.f90` and `type_is_1_ok.f90` need a public FortFront binding and
  dynamic-type fact for the contained procedure. The direct one-level
  `CLASS IS`/`TYPE IS` smoke is green, but it is not evidence for the missing
  nested/corpus shapes.
- `associate_18.f90` needs the imported-derived-type procedure and its
  versioned `.fmod` identity to survive a sibling-module boundary. Until that
  contract is available, keep the corpus row classified as a real rejection,
  not XFAIL/NOREF, and do not add a spelling-based exporter fallback.

The next implementation split is therefore FortFront facts first (#2974,
#2975 and the polymorphic host binding), then one ffc `.fmod`/consumer oracle
for the imported-derived function. A full #584 close requires all four
positive corpus cases plus an invalid/ambiguous binding negative control and
the valid-corpus rejection gate; the new function regression is only the
cheapest executable guard against regression at the module export boundary.

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

The 222-line `alloc_descriptor` leaf is now
`session_program_lowering_alloc_descriptor.f90`: a real descendant unit with
nine explicit implementation interfaces. Its textual include and host include
site are gone; the allocatable lifecycle compiler oracle and a clean sequential
build preserve allocation, shape, flag, stride, and deallocation behavior.

The 362-line `reject_const_init` leaf is now
`session_program_lowering_reject_const_init.f90`: a real descendant unit with
nine explicit interfaces. Its paired accepted/rejected oracle covers variables,
named constants, fixed and assumed shapes, implied-do indices, overflow, I/O
`ASYNCHRONOUS=`, and a user-function reference resolved through the shared
rejection engine. The textual include and host include site are gone.

The 1,094-line `open_close` service is now
`session_program_lowering_open_close.f90`: a typed descendant with explicit
interfaces for OPEN/STATUS, CLOSE, unit resolution, IOSTAT/IOMSG, and file-unit
WRITE lowering. The textual include and host include site are gone. The
existing dynamic-STATUS regression, FILE-variable path, positional-unit path,
and OPEN/WRITE/CLOSE round-trip all remain green after the move; no behavior
was accepted from a state-only check.

Extract the remaining leaves in this order:

`inferred` (347), `c_ptr` (362), `transfer` (439), `integer` (576),
`complex` (1,330),
`intrinsics_extra` (1,322), `intrinsics` (2,141), `expr_lowering` (1,796),
`logical_reduction` (1,615), and `reduction_expr` (954).

The first extraction needs paired accepted/rejected compiler tests. Expression
work must preserve #671's gfortran-oracle exactly-once side-effect regression
while refactoring call evaluation.

### Wave 3A: structured control

`block_concurrent` (47), `do_while` (157), `goto` (343), `control` (395),
`associate` (440), `loops` (660), `select` (1,365), `where` (1,275), and
`forall` (159).

The loop/control slice includes #626 and #671 fall-through/termination
oracles. The FORALL slice fixes #673 with statement-level temporaries wherever
simultaneous assignment requires old RHS values. The #673 tranche now
snapshots each fixed-size intrinsic target through a raw stack copy before its
index nest, routes RHS and mask reads to that snapshot, and leaves stores on
the original target. Multi-statement bodies lower as one complete nest per
statement, preserving statement-level ordering. The
test_session_forall_alias_compiler oracle covers reverse/ascending aliasing
and a two-statement body. Descriptor-backed, runtime-shaped, character,
derived, and nonconforming-shape FORALL targets remain explicitly outside
this tranche.

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
`pointer` (1,707), `allocatable` (2,925), `array_elements` (4,494), and
`arrays` (6,475).

The first 296-line declaration-shape slice of `arrays` is now the typed
`session_program_lowering_array_shape.f90` descendant. Its eleven explicit
module-procedure interfaces classify assumed-shape, assumed-rank,
assumed-size, and runtime-bound declarations without private host association;
the moved routines are deleted from the textual include. The
`test_session_array_shape_module_compiler` differential oracle compiles the
same rank-2 assumed-shape and assumed-size program with ffc and gfortran and
compares their output. Existing assumed-shape, assumed-size, and runtime-bound
compiler tests remain focused regression gates. This is a shape-classifier
seam only; descriptor storage, sections, and allocation remain in their
contract-owned waves below.

Sequence by contract. First, #337 establishes section views. Next come #338
pointer arrays, #348 character descriptors, and #643 derived allocatable
arrays. #339 then deletes legacy shape metadata. #399 adds gather views and
higher array expressions. Do not retain competing descriptor representations.

#### #643 vertical slice (current)

The direct-session lowerer now exercises the first complete `type(t),
allocatable :: a(:)`/`a(:,:)` path through the canonical descriptor: declaration,
rank-1/rank-2 `allocate`, component element stores/loads, `size`, `lbound`,
`ubound`, `allocated`, `deallocate`, and reallocation. `SIZE` of a rank-2
allocatable reads and multiplies both descriptor extents; bounds inquiries read
the descriptor rather than stale compile-time symbol fields. The regression
`test_session_derived_alloc_array_compiler` is checked against the same source
compiled by gfortran (the independent oracle output is 3/60 for rank 1 and
1/2/1/4/8/1212 for rank 2). No inline `{data, extent}` representation is
added. Deep-copy assignment of allocatable derived arrays, finalization on
scope exit, `ALLOCATE(SOURCE=)`/`MOLD=`, polymorphic dynamic element sizes, and
non-unit-bound element addressing remain separate gates; do not mark #643
complete until each has a positive and negative behavioral oracle.

### Wave 5: I/O

`internal_write` (175), `write_ops` (189), `internal_read` (270),
`io_implied_do` (325), `inquire` (350), `internal_write_compound` (369),
`read_al` (404), `read_ops` (736), `open_close` (1,057), `io_typecheck`
(1,066), `print_expr` (988), and `namelist` (1,619).

The #628 dynamic `STATUS=` operand now keeps its character buffer through
lowering (including `NEWUNIT=`), and the runtime compares the value
case-insensitively while trimming fixed-length padding;
`test_session_open_status_variable_compiler` is the independent regression
oracle. Keep that contract while freezing `open_close`. Internal write and its
compound child
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
  #419/#422/#435/#458/#459/#465/#643. #673's fixed-intrinsic snapshot slice is
  landed; allocatable, character, derived, and nonconforming-shape FORALL
  targets remain explicitly pending.
- FortFront #2980/#2994 establish Lazy specialization identity before ffc #437
  emits it and #433 serializes it.
- #478/#531 share measurement and timeout work. #532/#540 share one provenance
  refresh. Rebaseline the stale umbrellas #576/#609 and replace them with
  exact live signatures. The current #576 signature is the deterministic
  `issue_2064_logical_return_inferred_as_integer.lf` compile SIGSEGV recorded
  above; its root-cause fix and positive oracle are landed pending the
  current-head corpus gate.

## Cross-repository gates

| Owner | Active contract before ffc can close the dependent work |
| --- | --- |
| FortFront | rejection/accepted-side correctness [#2883](https://github.com/lazy-fortran/fortfront/issues/2883), [#2897](https://github.com/lazy-fortran/fortfront/issues/2897), [#2924](https://github.com/lazy-fortran/fortfront/issues/2924), [#2951](https://github.com/lazy-fortran/fortfront/issues/2951), [#2970](https://github.com/lazy-fortran/fortfront/issues/2970), [#2986](https://github.com/lazy-fortran/fortfront/issues/2986), and [#2987](https://github.com/lazy-fortran/fortfront/issues/2987). #2993 is landed in FortFront `704cd27f`; the separate-module dummy resolver is `8d3e5a8a`, and the implicit-DIMENSION preservation oracle is `e5e8157b`; continuation-comment lexer defect [#2996](https://github.com/lazy-fortran/fortfront/issues/2996) and parser/identity [#2973](https://github.com/lazy-fortran/fortfront/issues/2973), [#2975](https://github.com/lazy-fortran/fortfront/issues/2975), [#2980](https://github.com/lazy-fortran/fortfront/issues/2980), [#2994](https://github.com/lazy-fortran/fortfront/issues/2994) remain active |
| LIRIC | Restore producer artifacts and explicit build-failure reporting in [#533](https://github.com/krystophny/liric/issues/533) before relying on the semantic public-session gate [#523](https://github.com/krystophny/liric/issues/523); current Bench Matrix, scheduled compatibility, and nightly runs are red |
| fo | [#117](https://github.com/lazy-fortran/fo/issues/117) for the assignment-name-ending-in-function oracle before formatting changes, [#119](https://github.com/lazy-fortran/fo/issues/119) for unbounded JSON results, and [#103](https://github.com/lazy-fortran/fo/issues/103) for diagnostics; the parent/submodule DAG and ffc order-shim prerequisite are already landed in fo `32ef96d`/ffc `763ba0c` |
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
