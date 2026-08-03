# ffc 100% roadmap <!-- slop-ok: requested document type -->

Snapshot: 2026-08-03, Europe/Vienna. This is the execution source for bringing
ffc to 100% of its declared compiler scope. Start each work session here.

## Scope

Build ffc to the Fortran standard through F2023, plus Lazy Fortran. The
standard is the target; the maintained FortFront, LFortran, and gfortran.dg <!-- slop-ok: technical semicolon -->
corpora measure progress toward it and are not themselves the goal. Prioritize
public compiler boundaries, shared lowering engines, and stable ABIs before
isolated corpus cases.

Excluded for now:

- coarrays, images, teams, events, and collectives <!-- slop-ok: scope list -->
- OpenMP, OpenACC, and MPI
- GPU and device backends
- vendor extensions outside standard Fortran
- deleted legacy language features
- compiler-option and DejaGNU-harness behavior not modeled by the runner

F2023 is inside the target and currently unscoped. The closed `[ffc-f2023-*]`
trackers covered F95-through-F2018 language coverage, not the delta F2023
introduced; ffc #473 owns auditing that delta and splitting it into atomic <!-- slop-ok: technical semicolon -->
issues. Until it lands, treat "newest standard" as an intention rather than a
measured position.

Do not count excluded files as passes. Do not create implementation issues for
them unless this scope changes explicitly.

## Definition of 100%

ffc is 100% working only when all of these gates hold at the same pinned
revisions:

- the maintained ffc suite passes in full with `fo`;
- every in-scope runnable conforming case in each declared corpus is `PASS`,
  with required output matching its reference;
- every in-scope rejection case is rejected with the expected diagnostic;
- no claimed in-scope runnable case is `XFAIL`, `XPASS`, `FAIL`, or `NOREF`;
- every `SKIP` and exclusion has machine-validated ownership and never counts as
  a pass;
- ffc uses only public FortFront and LIRIC interfaces, with the runtime, module,
  descriptor, lifetime, dispatch, and generated-code contracts verified; and <!-- slop-ok: acceptance contracts -->
- full local pipelines and required CI matrices are green.

`NOREF` cases must be classified before they enter the 100% denominator. A case
with undefined output, missing linkage, an unsupported harness contract, or an
explicit scope exclusion is not a pass.

ffc #430 landed the classification machinery: NOREF cases now carry an approved
category and a mandatory reason, validated by the manifest and the parity
report checker. The denominator is still not final, because the categorized
manifests must be populated across the remaining unclassified cases and the
dashboard regenerated from a provenance-verified full-corpus run. Until that
regeneration lands, no conformance percentage quoted here is defensible. The
two FortFront corpora matter most: they are maintained in-tree and are genuine
100% targets.

Neither external corpus is a 100% target as a whole. `gfortran.dg` carries
error-detection, deprecated, and vendor-extension tests; the LFortran suite <!-- slop-ok: technical semicolon -->
exercises that compiler's own extension surface. Gate the runnable,
standard-conforming subset of each and record the exclusions.

## Repositories and ownership

| Stage | Repository | Contract |
| --- | --- | --- |
| language and typed frontend | [FortFront](https://github.com/lazy-fortran/fortfront) | parsing, semantic resolution, typed public queries, diagnostics |
| lowering and driver | [ffc](https://github.com/lazy-fortran/ffc) | typed lowering, code-generation policy, module and runtime ABI |
| backend session | [LIRIC](https://github.com/krystophny/liric) | LLVM-independent public session ABI, verification, object emission |
| workflow | [fo](https://github.com/lazy-fortran/fo) | dependency discovery, build, test, diagnostics, conformance commands, and the cheap text-level lint tier |
| deep static analysis | [fluff](https://github.com/lazy-fortran/fluff) | every AST-based rule, plus source-preserving formatting and structured lint output |
| editor client | [fx](https://github.com/lazy-fortran/fx) | user-facing diagnostics and language-service integration |
| language design | [Lazy Fortran standard](https://github.com/lazy-fortran/standard) | accepted language, runtime, ownership, layout, and reproducibility contracts | <!-- slop-ok: contract categories -->
| reference corpus | [LFortran](https://github.com/lfortran/lfortran) | pinned integration tests |
| reference corpus | [GCC](https://gcc.gnu.org/git/gcc.git) ([GitHub mirror](https://github.com/gcc-mirror/gcc)) | pinned `gfortran.dg` tests |

fo and fluff divide source analysis rather than competing over it, following
the split every modern toolchain converged on: Go separates `go vet` from
staticcheck, Rust separates cargo from clippy, C and C++ separate compiler
warnings from clang-tidy. fo owns the cheap tier that needs no parse tree
(unused imports, short-circuit reliance, gfortran's own warnings) and must keep
working with nothing else installed. fluff owns every rule that needs an AST,
reached through `fo lint --deep` (fo #59) as a subprocess exchanging JSON. That
subprocess boundary is deliberate: it holds fo's dependency closure at fx plus
OpenMP, so `fo build` and `fo test` never require FortFront.

The pipeline is source to FortFront typed public API, ffc lowering and runtime
ABI, LIRIC session API, then object or executable. Private frontend AST imports
are forbidden. A missing typed frontend fact is a FortFront issue, not an ffc
AST workaround.

## Current state

## Authoritative handoff (2026-08-03)

This checkpoint supersedes older historical bullets in this file when they
disagree with the current repository state.

- ffc `main` is `5de3ecf` and is pushed to `origin/main`. It includes the
  rank-one-through-rank-four runtime scalar array-section implementation and
  its independent compiler-test oracle. The implementation is not promoted
  in the conformance manifest yet: the exact `array_section_01.f90` normal
  and XFAIL-disabled gates still have to be rerun from a working LIRIC-linked
  primary checkout.
- `array_section_01.f90` remains the first owned XFAIL tranche, owned by ffc
  #337. `arrays_02_size.f90` and every other active XFAIL/FAIL remain behind
  it. No manifest row is removed merely because code has been pushed.
- The random sample remains at 900 per suite. Increase it only after repeated
  fresh subsets are 100% behavioral PASS with zero XFAIL, XPASS, FAIL,
  timeout, and OOM dispositions. Whole-corpus runs are release gates only.
- The fresh worker build for the rank-four patch reached all 450 compile
  units and then failed at link because that worktree lacked `-lliric`; this
  is an environment/setup limitation, not promotion evidence. Keep the
  focused primary-checkout rebuild as a required next step.

### Cross-repository blockers

| Owner | Issue | Effect on ffc | Required handoff |
| --- | --- | --- | --- |
| FortFront | [#2883](https://github.com/lazy-fortran/fortfront/issues/2883), [#2924](https://github.com/lazy-fortran/fortfront/issues/2924), [#2951](https://github.com/lazy-fortran/fortfront/issues/2951) | public binding/scope facts and rejection gates must stay sound for ffc #584 | land the serial rejection train with valid-corpus gates, then rerun ffc #584 |
| FortFront | [#2994](https://github.com/lazy-fortran/fortfront/issues/2994) | Lazy specialization names can carry the wrong default-real ABI | resolve the frontend specialization identity, then verify ffc #437/#433 |
| LIRIC | [#523](https://github.com/krystophny/liric/issues/523) | serialized dominance/print definitions can invalidate generated-code checks | fix and rerun ffc runtime/ABI gates; keep `LIBRARY_PATH` explicit |
| fo | [#59](https://github.com/lazy-fortran/fo/issues/59), [#103](https://github.com/lazy-fortran/fo/issues/103), [#56](https://github.com/lazy-fortran/fo/issues/56) | deep lint and structured diagnostics are not yet one stable workflow | complete fluff JSON integration, diagnostic mapping, then the LSP path |
| fluff | [#262](https://github.com/lazy-fortran/fluff/issues/262), [PR #269](https://github.com/lazy-fortran/fluff/pull/269) | test failures must be reported honestly before fo consumes deep-lint output | finish the red formatter/quality regressions and merge only with all checks green |
| fx | [#36](https://github.com/lazy-fortran/fx/issues/36) | concurrent cache corruption is an editor reliability issue, not an ffc conformance blocker | fix before claiming dependable editor diagnostics |
| standard | [#745](https://github.com/lazy-fortran/standard/issues/745), [#753](https://github.com/lazy-fortran/standard/issues/753), [#756](https://github.com/lazy-fortran/standard/issues/756) | shape/broadcasting, module signatures, and synthesis contracts can change future ABI decisions | accept/specify contracts first; map each accepted change to an implementation issue |

Current main revisions:

- ffc `5de3ecf`, main (rank-one-through-rank-four runtime scalar array-section
  lowering is pushed but not yet manifest-promoted; module integer parameters
  are bound before derived-type
  layout, promoting `derived_types_121.f90`; SIZE keyword/positional mapping
  promotes `arrays_01_size.f90`; selected-real-kind module dummy metadata now preserves
  the resolved f64 ABI and promotes `issue_1771_module_parameter_types.f90`
  after normal/no-XFAIL exact gates, a focused compiler regression, and an
  independent gfortran oracle; deferred-character calls use canonical
  visible-argument preparation, with exact `derived_types_121.f90` still
  XFAIL because its separate-compilation path loses imported `ilp` kind
  metadata) on top of
  keyword AINT/ANINT actuals and the array-valued `intrinsics_115.f90` are
  promoted on top of the deferred-shape issue-1968 lowering, ALLOCATED keyword
  arguments and scalar `DATA p / NULL() /` pointer disassociation, the
  incomplete-expression diagnostic, the comparison-typecheck submodule, the
  FLOOR optional `KIND=8` public-session lowering and f64-to-i64 conversion,
  the storage-rejection submodule, bare-character SELECT CASE fix and
  contained f64 calls in f32 expressions, the lazy whole-array constructor
  reallocation, mixed-kind unary real, ANY DIM assignment, and array-constructor
  02/03 promotions, the complex ABS/IEEE NaN, allocatable inquiry, and
  allocatable complex-array promotions, the modules36 fixed-character-array
  promotion, the #584 assumed-size-array FAIL fix, modules34/modules35 XFAIL
  closure, schema-10 `.fmod` compatibility, and strict sampled conformance
  gating; sampled manifest dispositions through seed 1037)
- Open-PR audit on 2026-08-03: ffc #596 and #677 remain red/stale, fo #116 is
  conflicting with an unresolved correctness review, fluff #269 is red,
  fortfem #56 has requested changes, and fortnum #59--#61 are a red stacked
  train. None is eligible for squash merge; finish each with local behavioral
  evidence and review resolution before merging, and never bypass a red or
  stale gate.
- FortFront `5ff07184`, main (public incomplete-expression diagnostics on top of recursive nested-array-postfix parsing, nested DO WHILE construct parsing, declaration-shape parsing, semantic conformance checks, full logical expressions in I/O argument lists, preserved logical literal kind suffixes, continued declaration statement-boundary handling, full BIND(C) interface return-kind preservation, and contextual `error` identifiers)
- LIRIC `5436e5c`, main (#528)
- fo `af075f4`, main (PR #118; #102)
- fluff `b7fdd2a`, main (PR #277; #244)
- fx `9e16f11`, main
- standard `b43232757183`, main
- pinned LFortran corpus `caf87b660f80`
- pinned GCC gfortran.dg corpus `395e3d8131c1`

Maintained verification:

- ffc `fo build` 450/450 and FortFront `fo build` 379/379 pass at the current
  revisions. Focused descriptor, parser, semantic, and diagnostic tests pass.
- The full local `fo` test pipelines still contain adjacent ffc and FortFront
  failures outside this bounded tranche, including logical-literal, derived-
  type, code-generation, and conformance-timeout cases. The constant-bound
  failure fixed in this tranche is no longer among the focused failures.
- fo main: full pipeline and lint passed.
- Installed fo: `/home/ert/.local/bin/fo`, version 0.3.2.
- fo MCP system test: 70 passed, 0 failed; both framing modes passed.

Latest bounded status:

- Completed 900-file runs remain disposition-clean through seeds 1035, 1036,
  and 1037. Seed 1037 became clean after exact classifications were added for
  its previously unexpected results.
- The formerly XFAIL `associate_18.f90` now passes: public procedures whose
  call ABI is unsupported remain visible in `.fmod` for `USE ONLY` validation,
  without being registered as callable externals. The next sample increase is
  deferred while owned XFAIL implementation work continues.
- The current XFAIL-first tranche is green without manifest help: `while_05.f90`
  passes in the LFortran suite and `do_while_1.f90` passes in the gfortran.dg
  suite as ordinary cases. Their XFAIL entries were removed after the focused
  behavioral checks passed. Earlier DO WHILE promotions remain green as well.
- The next XFAIL-first tranche is also green: `boolean_assign_bare_true.lf`
  and `boolean_assign_bare_false.lf` pass in the FortFront-LF suite with
  `XFAIL=0`, `XPASS=0`, and `FAIL=0`. Their entries were removed only after
  independent lazy-mode runtime checks passed.
- The scalar logical tranche is green as well: `logical3.f90` passes with
  `XFAIL=0`, `XPASS=0`, and `FAIL=0` after shared `.xor.` and `.eqv.` lowering
  was corrected. The gfortran run is the behavioral oracle.
- The logical reduction tranche is green: `logical_dot_product.f90` passes in
  gfortran.dg with `XFAIL=0`, `XPASS=0`, and `FAIL=0` after logical
  `DOT_PRODUCT` was lowered as `ANY(a .AND. b)`.
- The integer-to-logical scalar tranche is green: `logical4.f90` and
  `logical_casting_01.f90` pass against the gfortran behavioral oracle with
  `PASS=2`, `XFAIL=0`, `XPASS=0`, and `FAIL=0`. Arithmetic integer expressions
  now lower through one scalar nonzero conversion, including the i1-to-i32
  storage boundary, and their XFAIL entries were removed only after named
  no-manifest runs passed.
- The reverse scalar/array cast is green too: `logical_to_integer_cast.f90`
  passes against the same behavioral oracle with `PASS=1`, `XFAIL=0`, `XPASS=0`,
  and `FAIL=0`. Integer lowering accepts logical literals and scalar logical
  identifiers, while whole logical arrays reuse their i32 storage.
- The logical array-expression tranche is green:
  `logical_arrays_logical_binop_01.f90` passes in lfortran with `PASS=1`,
  `XFAIL=0`, `XPASS=0`, and `FAIL=0` after shared logical element lowering,
  nested-mask reduction materialisation, and the FortFront I/O full-expression
  parser fix. The XFAIL entry was removed only after the named behavioral run
  passed. The sample count remains 900.
- The typed file-I/O tranche is green: `logical_kind_01.f90` passes in the
  LFortran conformance suite with `PASS=1`, `XFAIL=0`,
  `XPASS=0`, and `FAIL=0`. `INQUIRE(SIZE=)` covers literal files and
  connected units, while scalar stream writes preserve integer and logical
  kind widths. The independent inquiry compiler test and runtime-link
  contract test also pass. The sample count remains 900.
- The logical kind transfer tranche is green: `logical_kind_02.f90` passes in
  the LFortran suite with `PASS=1`, `XFAIL=0`, `XPASS=0`, and `FAIL=0`.
  `TRANSFER` preserves canonical logical truth and zero padding when a scalar
  logical kind is transferred into a same-width integer byte array. The
  independent logical-transfer compiler test also passes. The sample count
  remains 900.
- The logical kind inquiry/literal tranche is green: `logical_kind_04.f90` and
  `logical_kind_05.f90` pass in the LFortran suite with `PASS=2`, `XFAIL=0`, <!-- slop-ok: technical status counts -->
  `XPASS=0`, and `FAIL=0`. ffc lowers logical `KIND` and `STORAGE_SIZE`, while
  FortFront preserves numeric, named, and mixed-case logical kind suffixes.
  Independent ffc and FortFront regressions pass; the sample count remains
  900. The character-valued `ERROR STOP` tranche is green too:
  `logical_kind_06.f90` passes with `PASS=1`, `XFAIL=0`, `XPASS=0`, and
  `FAIL=0`; the independent STOP-banner regression verifies the dynamic
  message and exit status. The nested `LOGICAL` conversion-kind tranche is
  green too: `logical_kind_07.f90` passes with `PASS=1`, `XFAIL=0`, `XPASS=0`,
  and `FAIL=0`; the independent inquiry-fold regression verifies default and
  explicit kinds. The allocatable logical-mask tranche is green too:
  `logical_not_01.f90` passes in the LFortran suite with `PASS=1`, `XFAIL=0`,
  `XPASS=0`, and `FAIL=0`; shared whole-array lowering now covers allocatable
  comparison masks, array printing, and `ANY(.NOT. logical-array)`. The
  independent whole-array compiler test passes, and the XFAIL entry was
  removed only after the normal-manifest run. The formatted file-I/O tranche is
  green too: `logical_testing.f90` passes with `PASS=1`, `XFAIL=0`, `XPASS=0`,
  and `FAIL=0`; file-unit formatted writes now accept character expressions
  such as `TRIM`, and the independent IOSTAT/IOMSG regression verifies a
  scratch-file character-write/logical-read round trip. The dynamic MATMUL
  tranche is green too: `matmul_01.f90` and `matmul_02.f90` pass in the
  LFortran suite with `PASS=2`, `XFAIL=0`, `XPASS=0`, and `FAIL=0`. Runtime
  rank-2 allocatable MATMUL, runtime array-expression reduction, `CPU_TIME`
  widening, explicit-shape whole-array dummy aliasing, explicit lower-bound
  assumed-shape descriptors, and mixed-rank runtime section expressions now
  share the corrected lowering paths. The direct generated `matmul_02`
  executable reports zero error for both kernels, and the focused array-section,
  multi-declaration, diagnostic, and allocatable-function-result tests pass.
  The allocatable-result MATMUL tranche is green too: `matmul_03.f90`,
  `matmul_04.f90`, and `matmul_05.f90` pass with `PASS=3`, `XFAIL=0`,
  `XPASS=0`, and `FAIL=0`; `matmul_06.f90` now passes as well after rank-2
  automatic array results were registered through the raw sret ABI and
  materialised into bounded temporaries. The combined `matmul_01`-`matmul_06`
  run is `PASS=6` with `XFAIL=0`, `XPASS=0`, and `FAIL=0`. The typed
  TRANSPOSE tranche is green too: `matrix_01_transpose.f90` passes against the
  gfortran oracle with `PASS=1`, `XFAIL=0`, `XPASS=0`, and `FAIL=0`; complex
  `RESHAPE` literals and typed integer/real/double/logical transpose storage are
  covered by the same run. The compile-time parameter TRANSPOSE tranche is
  green too: `matrix_03_transpose_param.f90` passes against the gfortran oracle
  with `PASS=1`, `XFAIL=0`, `XPASS=0`, and `FAIL=0`. The mixed-kind integer
  MIN/MAX tranche is green: `max_02.f90` and `min_02.f90` compile and exit
  normally with `PASS=2`, `XFAIL=0`, `XPASS=0`, `FAIL=0`, and `NOREF=2`
  because they read uninitialized values; deterministic initialized mixed-kind
  checks match gfortran exactly. `min_01.f90` also passes with `PASS=1`,
  `XFAIL=0`, `XPASS=0`, and `FAIL=0`; typed `MIN0`, `AMIN0`, `MIN1`, `AMIN1`,
  and `DMIN1` reuse the scalar min/max engines. `minmax_01.f90` also passes
  with `PASS=3`, `XFAIL=0`, `XPASS=0`, and `FAIL=0`; legacy `MAX0` and `MAX1`
  now share the same typed path. The bounded XFAIL work since this snapshot
  promoted `minpack_01.f90`, `minpack_03.f90`, `module_array_init.f90`,
  `module_function_with_nopass.f90`, `module_function_without_nopass.f90`,
  `modules_03.f90`, `modules_05.f90`, `modules_08.f90`, `modules_09.f90`,
  and `modules_20.f90` after independent gfortran/runtime checks. Paired
  module cases are normal named `PASS` runs; module-only companions are
  checked as compile-only/no-reference units where they have no main program.
  The promoted runs are `XFAIL=0`, `XPASS=0`, and `FAIL=0`; keep the sample
  count at 900.
- The BIND(C) XFAIL tranche is now green: the exact named run of
  `modules_15.f90`, `modules_18.f90`/`modules_18b.f90`, and
  `modules_19.f90`/`modules_19b.f90` reports `PASS=5`, `XFAIL=0`, `XPASS=0`,
  `FAIL=0`, and `NOREF=2` under the normal manifest. `modules_15` also matches
  the independent gfortran executable output exactly. ffc now emits typed
  VALUE parameters for BIND(C) bodies, marks those bodies with the C ABI, and
  keeps BIND(C) interface calls on the C argument path; the `modules_15` XFAIL
  row was removed only after both no-manifest and normal-manifest runs passed.
- The next module-identity tranche is green: exact named runs of
  `modules_22.f90` and `modules_22_module.f90` report `PASS=2`, `XFAIL=0`,
  `XPASS=0`, and `FAIL=0` under both the normal and no-manifest configurations
  (`NOREF=1` for the module-only companion). Derived `integer(8)` components
  now use their two-slot layout and i64 load/store path; the XFAIL rows were
  removed only after the focused independent regression passed.
- The character-component module tranche is green as well: exact named runs of
  `modules_23.f90` and `modules_23_module.f90` report `PASS=2`, `XFAIL=0`,
  `XPASS=0`, and `FAIL=0` under both normal and no-manifest configurations
  (`NOREF=1` for the module-only companion). Character component expressions
  now use the fixed-length assignment path, and the frontend accessibility
  workaround is limited to its known false diagnostic; the XFAIL rows were
  removed only after the independent component regression passed.
- The `modules_25.f90` class/runtime-character tranche (#350/#417) is green:
  exact named runs of all three sources report `PASS=3`, `XFAIL=0`, `XPASS=0`,
  and `FAIL=0` in both normal and no-manifest modes (`NOREF=3` for compile-only
  module units). Inherited derived-component ABI flags, named-class dummy
  descriptors, and explicit module companion sources now work; the XFAIL rows
  were removed only after both bounded oracle runs passed.
- The `modules_26.f90` interface-procedure/runtime-archive case (#376) is green:
  exact named normal-manifest and no-manifest runs both report `PASS=1`,
  `XFAIL=0`, `XPASS=0`, and `FAIL=0`. Runtime-archive loading and the
  independent gfortran comparison also pass; interface dummy extents now stay
  runtime bounds. The XFAIL row was removed only after both bounded runs passed.
- The `modules_27_module2.f90` generic module-registration case (#457) is green:
  exact named normal-manifest and no-manifest runs both report `PASS=1`,
  `XFAIL=0`, `XPASS=0`, and `FAIL=0` (`NOREF=1` for the module-only unit).
  Complex pointer dummy registration now follows the resolved declaration path,
  and the independent gfortran compile/run oracle passes. The XFAIL row was
  removed only after both bounded runs passed.
- The modules28 separate-compilation family (#328/#447) is green: exact named
  runs of `modules_28.f90`, `modules_28_module1.f90`, and
  `modules_28_module2.f90` report `PASS=3`, `XFAIL=0`, `XPASS=0`, and `FAIL=0`
  in both normal and no-manifest modes (`NOREF=2` for the module-only units).
  Imported derived types are registered before dependent layouts, repeated
  component metadata is rebuilt safely, and the independent gfortran program
  oracle passes. All three XFAIL rows were removed only after both bounded runs.
- The focused array-shape rejection regression is green again: the accepted
  masked `FORALL` case with `vec(j)=real(j)` now passes after guarding the
  scalar-conversion lookup from invalid index-0 access; the focused compiler
  test and independent gfortran oracle both pass.
- Architecture migration has its first verified seams: diagnostics and
  constant-folding are real module/submodule units, the scalar-kind helpers
  and scalar-expression engine are real module/submodule units, FMod token
  helpers are a real module, literal-utils is a real submodule, and the unused
  `session_program_lowering_text.inc` fragment is gone, and declaration-conflict,
  generic-rejection, result, array-constructor, purity, and pointer rejection
  checks are now real submodules with explicit build-order units. A clean
  sequential `fo build` and focused behavior tests pass. The remaining rejection and
  other host-coupled fragments stay live until their dependencies are extracted
  safely.
- The `modules_15b.f90` module-interface companion compiles with ffc and
  gfortran as an explicit `NOREF=compile-only` case. Its runnable companion is
  now covered by the verified BIND(C) ABI tranche above.
- The modules29 separate-compilation family is green: exact named normal and
  no-XFAIL runs report `PASS=3`, `XFAIL=0`, `XPASS=0`, and `FAIL=0` (`NOREF=2`
  for the module-only companions). The independent gfortran module-chain
  compile/link/run oracle and a bounded seed-1729 unit sample (`10/10`) also
  pass. The implementation exports direct USE dependencies recursively,
  preserves opaque public subroutine interfaces in `.fmod`, and FortFront
  `9ff6605e` treats `error` as a contextual identifier. The stale modules29
  XFAIL rows were removed only after these checks.
- The modules30 separate-compilation family is green: exact named normal and
  XFAIL-disabled runs of `modules_30.f90` and `modules_30_module2.f90` report
  `PASS=2`, `XFAIL=0`, `XPASS=0`, and `FAIL=0` (`NOREF=1` for the module-only
  companion). The independent gfortran four-module-chain compile/link/run
  oracle passes. FFC now preserves per-dummy kinds in opaque public procedure
  interfaces, keeping supported character dummies callable while unsupported
  derived dummies retain the opaque path. The two XFAIL rows were removed only
  after these checks.
- The modules31 separate-compilation family is green: exact named normal and
  XFAIL-disabled runs of `modules_31.f90`, `modules_31_module1.f90`, and
  `modules_31_module2.f90` report `PASS=3`, `XFAIL=0`, `XPASS=0`, and `FAIL=0`
  (`NOREF=2` for the two module-only companions). Independent ffc and
  gfortran object compile/link/run chains both print `running modules_31
  program`. The three stale XFAIL rows were removed only after focused
  receiver-slot, fmod, type-bound, character-component, and separate-
  compilation tests were green. The modules33 four-source separate-
  compilation family is now green as well: normal and XFAIL-disabled exact
  runs report `PASS=4`, `XFAIL=0`, `XPASS=0`, and `FAIL=0`; the independent
  gfortran module-chain oracle and the complete ffc object/link/run chain both
  print `running modules_33 program`. FFC now preserves multi-specific
  type-bound generic metadata across schema-11 `.fmod` files, resolves
  imported generic calls, and handles scalar nested receivers without
  contaminating later derived layouts. The positive direct-session generic
  dispatch regression and the batched five-target focused test pass. The next
  XFAIL-first tranche was the modules34 sibling family. It is now green:
  exact normal and XFAIL-disabled runs report `PASS=5`, `XFAIL=0`, `XPASS=0`,
  `FAIL=0`, and `NOREF=4`; independent ffc and gfortran module-chain
  compile/link/run oracles both print `running modules_34 program`. FFC now
  re-exports public derived types imported through `USE`, and the two stale
  modules34 XFAIL rows were removed only after the named behavioral evidence.
  The modules35 XFAIL is green too: `modules_35.f90` reports `PASS=1`,
  `XFAIL=0`, `XPASS=0`, and `FAIL=0` against the gfortran oracle. The fix
  handles character allocatable-array descriptor passing, bounded rank-1 slot
  copies, and zero-length constructor assignment. Schema-10 `.fmod` reads are
  now backward-compatible with a literal binding fixture, while writers stay
  on schema 11. `sync_all_01.f90` and `sync_memory_01.f90` are explicitly
  classified as out-of-scope coarray/image-control cases.
- The #584 assumed-size-array FAIL closure is green on the exact bounded set
  `arrays_06_size.f90`, `arrays_07_size.f90`, `arrayprint_01.f90`, and
  `array_bound_5.f90`: `PASS=4`, `XFAIL=0`, `XPASS=0`, and `FAIL=0`, matching
  independent gfortran outputs. Variable-driven bounds now avoid the
  assumed-size classifier, and runtime reductions use descriptor-aware loads.
- `session_program_lowering_enum.inc` is now a real
  `session_program_lowering_enum` submodule with an explicit build-order
  module; enum and module-constant focused tests pass. The modules36 XFAIL
  is now green: exact normal and XFAIL-disabled runs report
  `PASS=1`, `XFAIL=0`, `XPASS=0`, and `FAIL=0`; an independent gfortran
  compile/run oracle agrees. Fixed-length character-array components now carry
  element-count and packed-slot metadata, `SIZE(component)` folds in
  specification expressions, and `ANY(.NOT.array)` accepts FortFront's unary
  operator spelling. Element access remains an explicit diagnostic until its
  character-array ABI is implemented; the XFAIL row was removed only after
  the exact behavioral checks and focused tests passed.
- The focused constant-expression rejection regression is green again:
  `test_session_reject_const_01_compiler` passes after `HUGE` bounds stop being
  misclassified as runtime extents. The constant-overflow checker is now a real
  `session_program_lowering_reject_const_overflow` submodule with an explicit
  build-order unit; its focused behavioral oracle remains green.
- No whole-corpus run has been performed under this policy. `XFAIL`, `NOREF`,
  and `SKIP` are classifications, not implementation passes.
- The exact XFAIL-first tranche of `abs_04.f90`, `abs_06.f90`,
  `allocated_01.f90`, `allocated_04.f90`, `allocated_05.f90`,
  `array_constructor_02.f90`, and `array_constructor_03.f90` is fully
  promoted. Normal and XFAIL-disabled LFortran runs both report
  `PASS=7`, `XFAIL=0`, `XPASS=0`, and `FAIL=0`; independent gfortran output
  checks and focused complex/reduction compiler tests agree.
- Luna also completed and promoted `any_01.f90`: normal and XFAIL-disabled
  exact runs both report `PASS=1`, `XFAIL=0`, `XPASS=0`, and `FAIL=0`, with the
  independent gfortran output oracle and the focused array-mask compiler test
  agreeing. The fix covers `ANY(..., DIM)` assignment into assumed-shape
  runtime arrays.
- Luna then promoted `array_op_03.f90`: normal and XFAIL-disabled exact runs
  both report `PASS=1`, `XFAIL=0`, `XPASS=0`, and `FAIL=0`, with the focused
  scalar-expression test and independent gfortran oracle green. Mixed-kind
  f64 operands now lower at f64 before safe conversion into an f32 context.
- The two red `fortfront-lf` sample cases from seed 1038 are now green:
  `test_209_all.lf` and `test_209_complex.lf` pass in normal and XFAIL-disabled
  exact runs with `PASS=2`, `XFAIL=0`, `XPASS=0`, and `FAIL=0`. The independent
  gfortran oracle and focused allocatable-constructor compiler regression
  agree; whole-array constructor operands are materialized before old
  allocation is released.
- The `functions_11.f90` XFAIL is now promoted: normal and XFAIL-disabled exact
  runs report `PASS=1`, `XFAIL=0`, `XPASS=0`, and `FAIL=0`. The contained f64
  function call now uses its f64 ABI before conversion to an f32 assignment;
  the independent gfortran regression and focused compiler test pass.
- The next XFAIL queue remains explicit: `array_section_01.f90` still rejects
  scalar RHS broadcasting in runtime section assignment and a first attempted
  dynamic-loop patch emitted malformed LIR; `derived_types_121.f90` still needs
  the deferred-character dummy contract propagated into class/derived actual
  lowering. Neither case has been promoted or hidden in its manifest.
- Bounded Luna triage on 2026-08-03 rejected several proposed shortcuts without
  edits or manifest drift: `issue_1968_lazy_function_result.lf` still has an
  invalid inferred-array dimension node; `allocatable_component_struct_array_01.f90`
  still rejects `variants_array(1)` as a derived actual; `arrays_01_size.f90`
  and `intrinsics_114/115.f90` still fail with XFAIL disabled; and `sin_01.f90`
  still rejects `DSIN` as an unsupported scalar real call. `case_05.f90` is the
  only active repair from this probe and currently mismatches the behavioral
  oracle (`Invalid grade` instead of the `B` branch). None of these rows was
  removed or hidden; keep the sample at 900 and the XFAIL-first queue intact.
- `case_05.f90` is now promoted: bare `character` declarations materialize
  default length one, so the initialized `grade = 'B'` reaches the matching
  empty SELECT CASE arm and prints the expected value. Normal and XFAIL-
  disabled exact runs both report `PASS=1`, `XFAIL=0`, `XPASS=0`, and `FAIL=0`,
  with the focused SELECT CASE test and gfortran output oracle green.
- `session_program_lowering_reject_storage.inc` is now a real
  `session_program_lowering_reject_storage` submodule with an explicit build
  order module; the include and its references are gone. `fo build` passes
  448/448, `git diff --check` is clean, and an isolated storage-rejection
  source produced identical diagnostic text and exit status before and after
  migration. No manifest row was changed, and the sample remains 900.
- `intrinsics_114.f90` is now promoted. ffc unwraps FortFront keyword actuals
  for AINT/ANINT and resolves `KIND=4/8`; normal and XFAIL-disabled named runs
  report `PASS=1`, `XFAIL=0`, `XPASS=0`, and `FAIL=0`, with the independent
  gfortran output oracle agreeing. Its XFAIL row was removed only after the
  rebuilt main binary passed.
- `intrinsics_115.f90` is now promoted. Shared array-expression lowering handles
  compile-time parameter initializers and runtime AINT/ANINT arrays; normal and
  XFAIL-disabled exact runs both report `PASS=1`, `XFAIL=0`, `XPASS=0`, and
  `FAIL=0`, with focused parameter/runtime regressions and the independent
  gfortran output oracle agreeing. Its XFAIL row was removed only after the
  rebuilt main binary passed.
  `issue_1771_module_parameter_types.f90` still fails before LIRIC with
  `mismatched scalar kind in argument to square`, although its gfortran oracle
  prints `Square: 6.25`. Keep both rows in the queue; do not count normal
  XFAIL-wrapped runs as passes.
- The following disjoint probe also remained red: `arrays_02_size.f90`
  still fails during ffc compilation; `issue_2495_data_null_intrinsic.f90`
  reaches an ffc lowering failure (`data-stmt-object 'ptr2' has the POINTER
  attribute`) while FortFront's focused parser test passes; and the attempted
  `reject_const_init.inc` migration builds but fails its independent rejection
  oracle because invalid input compiles and exits zero. None was integrated or
  promoted.
- `arrays_01_size.f90` remains red for a narrower reason: ffc’s SIZE lowering
  treats the `kind=4` keyword as the `DIM` argument and reports `size dim out
  of range for: a`; FortFront’s public parse and semantic diagnostics are
  clean. Keep it ahead of sample expansion without a manifest change.
- The issue-1968 lowering blocker is now promoted: ffc handles FortFront’s zero
  deferred-shape sentinel for the assumed-shape lazy result. The focused lazy
  function test and both exact gates report `PASS=1`, `XFAIL=0`, `XPASS=0`, and
  `FAIL=0`; its XFAIL row was removed only after the rebuilt main binary passed.
- The remaining high-impact queue is explicit: `array_section_01.f90` still
  requires a known positive RHS extent for runtime scalar broadcasting and a
  prior attempt emitted malformed LIR (`instruction type missing`);
  `derived_types_121.f90` still reaches `direct LIRIC session cannot pass this
  scalar argument`. Keep both ahead of sample expansion and do not classify
  their failures as manifest workarounds.
- `floor_01.f90` is now promoted. Public-session FLOOR lowering accepts the
  optional `KIND=8` result through an f64-to-i64 `FPTOSI` wrapper while retaining
  the default integer kind. Normal and XFAIL-disabled exact runs both report
  `PASS=1`, `XFAIL=0`, `XPASS=0`, and `FAIL=0`; the focused integer-intrinsic
  compiler test and independent gfortran oracle agree. The sample remains 900.
- `cmp_typecheck.inc` is now a real comparison-typecheck submodule with the
  missing public relational-operator interface supplied. The focused behavioral
  compiler test passes, `fo build` is `450/450`, and no corpus manifest changed.
- `allocated_1.f90` is now promoted. ALLOCATED unwraps scalar and array keyword
  arguments before lowering; its focused compiler oracle and independent
  gfortran output agree, and both exact gates report `PASS=1`, `XFAIL=0`,
  `XPASS=0`, and `FAIL=0`. Its XFAIL row was removed only after the rebuilt
  main binary passed.
- `issue_2495_data_null_intrinsic.f90` is now promoted. Scalar pointer DATA
  with `NULL()` lowers to disassociation, while an independent negative
  gfortran oracle still rejects a non-NULL pointer initializer. Normal and
  XFAIL-disabled exact runs both report `PASS=1`, `XFAIL=0`, `XPASS=0`, and
  `FAIL=0`; its XFAIL row was removed only after both gates passed.
- `issue_256_incomplete_expression.f90` is now promoted. FortFront emits the
  focused incomplete-expression diagnostic while preserving valid logical
  literals and continuations; the independent public-API regression passes,
  FortFront builds `379/379`, and ffc normal and XFAIL-disabled exact runs both
  report `PASS=1`, `XFAIL=0`, `XPASS=0`, `FAIL=0` (the invalid fixture is
  classified `NOREF=1` because gfortran rejects it). Its XFAIL row was removed
  only after these gates passed.
- Fresh strict pass-only sampling remains bounded at 900. Seed 1038 supplied a
  red baseline: its two `fortfront-lf` failures are now fixed, while twenty
  LFortran failures were observed before the run was stopped. No sample
  increase or follow-up seeds are justified until the remaining XFAIL/FAIL
  queue is repaired.
- The last full local `fo` workflow, before the latest sampled fixes, is a
  unit/conformance diagnostic, not a
  corpus gate: build `446/446` and static checks `463/463` pass, while the
  339-test phase reports `316` passes and `23` known adjacent failures. Keep
  those failures in the next XFAIL/FAIL queue; do not hide them with manifest
  edits or expand the random sample until the selected queue tranche is zero.

All corpus work stays bounded: use deterministic random subsets, never the
whole corpus, and increase the sample only after repeated 100%-clean subsets.
Finish the owned XFAIL-first tranche at zero before moving to another corpus
area or increasing the count. Keep compiler jobs sequential and bounded to
avoid OOM.

The conformance script's exit status is not by itself a clean-sample proof:
`NOREF`, `SKIP`, and `XFAIL` can be reported without failing the wrapper, and
module-only siblings legitimately produce `NOREF`. For every tranche and
random sample, inspect the summary and require zero `FAIL`, `XPASS`, `XFAIL`,
timeouts, and OOMs; treat every `NOREF`/`SKIP` as an explicit reviewed
classification, never as a behavioral pass. Keep each invocation isolated
with `TMPDIR=$(mktemp -d)` and a private `--ref-cache`. The opt-in
`--require-pass-only --sample N` mode now filters XFAIL/SKIP/NOREF entries,
avoids standalone module-only files without an executable oracle, uses an
isolated scratch directory, and fails unless every selected record is a
behavioral PASS. Use it for random progress checks; keep `N=900` until
repeated fresh seeds are clean.

## XFAIL-zero work gate

XFAIL work always comes before corpus expansion. Each work cycle selects an
owned XFAIL tranche, fixes the implementation or its independent behavioral
oracle, and removes the XFAIL only when the case passes. We do not move to a
different corpus area, broaden the suite, or increase the random-sample count
until the current in-scope XFAIL tranche is at zero. The final conformance gate
requires zero in-scope XFAILs across every declared suite. Classification is
not a substitute for fixing the behavior.

## Fastest honest development loop

The normal development loop is one bounded XFAIL/FAIL tranche, not a full
pipeline or a full corpus run:

Before editing, record the clean `main` baseline for the exact tranche and one
small random unit-test sample. This separates a pre-existing FAIL from a
regression and avoids spending a full build on unrelated work. After the
patch, rerun the same named tranche and sample with `--no-build` wherever the
binary is unchanged.

1. Select the smallest owned tranche from the XFAIL/FAIL manifests. Read the
   exact source, owner/reason, prerequisite sources, and existing focused
   tests. `FAIL` and `XPASS` are blockers or promotion candidates; they are
   never hidden by editing a manifest.
2. Trace the implementation with `rg`, then use an independent behavioral
   oracle. For runnable cases this is normally gfortran output and exit status;
   for rejection cases it is the documented compile/rejection contract. A
   test that only checks repository state is not an oracle.
3. Build once per code change and keep the compiler sequential to protect RAM.
   The current `fo test` wrapper rebuilds its 446-test target on each separate
   invocation, so batch all focused names from one code change in a single
   command; never launch several `fo test` builds concurrently:

   ```bash
   export LIBRARY_PATH=<liric-build>
   export FO_JOBS=1
   fo build
   fo test <focused-target-1> <focused-target-2> ...
   ```

   Reuse that compiler for every exact case in the tranche. Do not rebuild for
   each corpus file, and do not use bare `fo` as the corpus development loop.
4. Run the exact named selection with the normal manifest, then once with
   XFAIL disabled. The second command must expose a real `FAIL` if the fix is
   not complete:

   ```bash
   scripts/conformance_check.sh --no-build --suite <suite> \
     --file <suite-relative-file> --ref-cache <private-ref-cache>
   FFC_XFAIL_MANIFEST=/dev/null scripts/conformance_check.sh --no-build \
     --suite <suite> --file <suite-relative-file> \
     --ref-cache <private-ref-cache>
   ```

   Run those commands separately: the normal-manifest command is expected to
   return nonzero for the named `XPASS` until its row is removed; that is a
   promotion signal, not permission to ignore another failure.
   Use `--files-from <tranche-list>` for a multi-file module family. Give each
   run a unique report/TMPDIR when invoking runners directly; use one stable
   reference cache per worktree so unchanged gfortran results are not rebuilt.
   For a module family, select the runnable file together with its sibling
   modules so the harness resolves the dependency closure. Do not add a
   standalone `compile-only` NOREF row for a source that needs sibling `.fmod`
   files; that bypasses dependency setup and creates a false failure.
5. Promote an XFAIL only after the named case has an independent oracle match,
   the no-XFAIL run is `PASS`, the normal run reports the expected `XPASS`, and
   the focused compiler tests are green. Remove only that exact manifest row,
   rerun the normal selection, and require `PASS` with `XFAIL=0`, `XPASS=0`,
   and `FAIL=0`. Never convert a failure into XFAIL to make a run green.
6. After the owned tranche reaches zero, run the current stratified sample at
   three fresh seeds, for example:

   ```bash
   for seed in 1038 1039 1040; do
     scripts/conformance_check.sh --no-build --sample 900 --seed "$seed" \
       --ref-cache <private-ref-cache>
   done
   ```

   The sample is clean only when every selected disposition
   is `PASS` (with no `FAIL`, `XPASS`, `XFAIL`, `FLAKY`, timeout, or OOM). Keep
   the count unchanged while any owned XFAIL/FAIL remains. Only repeated
   100%-clean subsets authorize a small increase, such as +100 or +200 files;
   then repeat the fresh-seed check before increasing again.
7. Parallelize only read-only audits, independent oracle preparation, and
   disjoint code work in separate worktrees. Do not run multiple heavy ffc
   builds or corpus gauntlets concurrently on this memory-constrained host;
   separate worktrees do not make RAM free. The primary checkout must stay on
   `main`; agents must not switch its branch or edit it while another worker is
   active. Integrate a finished worktree with a reviewed patch/commit, merge one
   green patch at a time, rebase it on `main`, and rerun the focused build/check
   before pushing.

The efficient order is: one baseline, one implementation build, one focused
test, two exact corpus checks, then bounded random samples. Only disjoint
analysis or code work belongs in parallel worktrees; heavy builds and
gauntlets remain sequential because worktrees do not reduce RAM use.

The final whole-corpus run is a release/provenance gate after all declared
XFAIL/XPASS/FAIL work and manifest ownership are resolved. It is not a routine
progress measurement. The fast path is: zero the owned XFAIL tranche, prove it
with an independent oracle, repeat clean random subsets, then widen the
sample modestly.

## Active task list (2026-08-03)

1. Completed: promote `module_array_init.f90`,
   `module_function_with_nopass.f90`, `module_function_without_nopass.f90`,
   `modules_03.f90`, `modules_05.f90`, `modules_08.f90`, `modules_09.f90`,
   and `modules_20.f90` only after independent gfortran behavioral oracles
   and bounded named runs reached `XFAIL=0`, `XPASS=0`, and `FAIL=0`.
2. Completed: promote the BIND(C) tranche consisting of `modules_15.f90`,
   `modules_18.f90`/`modules_18b.f90`, and `modules_19.f90`/`modules_19b.f90`
   only after the exact C-plus-gfortran oracle and normal-manifest run were
   both green. The `modules_22.f90`/`modules_22_module.f90` (#584) pair is now
   green; `modules_24.f90` (#417), the three-file `modules_25.f90`
   class/runtime-character tranche (#350/#417), and `modules_26.f90` (#376)
   are now green after bounded normal and no-manifest runs. `modules_27_module2.f90`
   (#457) and the modules28 family (`modules_28.f90`, `modules_28_module1.f90`,
   and `modules_28_module2.f90`) are now green after bounded normal and
   no-manifest runs. The modules29 family (`modules_29.f90`,
   `modules_29_module2.f90`, and `modules_29_module3.f90`) is now green after
   exact normal/no-XFAIL checks and the independent gfortran module-chain
   oracle. The modules30 family (`modules_30.f90` and
   `modules_30_module2.f90`) is also green after exact normal/no-XFAIL checks
   and the independent gfortran module-chain oracle; its two XFAIL rows are
   removed. The modules31 and modules33 families are complete as recorded
   above. The modules34 and modules35 XFAIL tranches are complete. The #584
   assumed-size FAIL closure is green on its bounded cases. Keep
   The `modules_36.f90` (#417) XFAIL-first tranche is complete: its fixed
   character-array declaration and `SIZE` path pass the exact normal and
   XFAIL-disabled checks plus the independent gfortran oracle. Keep the next
   owned XFAIL/FAIL tranche first and do not increase the sample count yet.
3. Continue replacing the remaining textual `.inc` fragments in the lowerer with real
   Fortran modules/submodules in dependency order. The first verified seams
   are diagnostics, constant folding, scalar-kind/scalar-expression lowering,
   and FMod token support; literal-utils, declaration-conflict, generic,
   result, array-constructor, purity, and pointer rejection are now real
   submodules. Next extract the remaining host-coupled rejection fragments.
   The modules35 fix necessarily touched two existing lowering fragments but
   introduced no new include. Remove each remaining include only after a sequential
   `fo build` plus focused behavioral checks are green. This architecture
   migration runs alongside the named XFAIL work and never authorizes a
   whole-corpus run.
4. Audit and remap stale manifest owners before the next sampling gate. The
   historical owners `ffc#375`, `ffc#447`, `ffc#448`, `ffc#350`, `ffc#457`,
   `ffc#328`, `ffc#342`, and `ffc#412` are closed; their named XFAIL rows stay
   until the corresponding behavior is independently green, but no new work
   is assigned to those issues. The live architecture/corpus gates are
   `ffc#584`, `ffc#609`, `ffc#576`, and `ffc#663`, followed by the open
   descriptor/storage issues in the blocker order below.
5. After the active XFAIL tranche and owner audit are clean, repeat bounded
   900-file subsets. Only repeated clean subsets can authorize the roadmap's
   next sample increase.

Sampled verification history, 2026-08-01:

Per the user's runtime constraint, post-fix corpus checks used deterministic
random subsets rather than whole-corpus runs. The conformance gate reported no
unexpected `FAIL` or `XPASS` in any suite after each listed rerun:

| Seeds | Files per suite | Suites |
| --- | ---: | --- |
| 101, 202, 303 | 20 | fortfront-f90, fortfront-lf, lfortran, gfortran-dg |
| 404, 505 | 30 | fortfront-f90, fortfront-lf, lfortran, gfortran-dg |
| 606, 707 | 40 | fortfront-f90, fortfront-lf, lfortran, gfortran-dg |
| 808 | 50 | fortfront-f90, fortfront-lf, lfortran, gfortran-dg |
| 909 | 50 | fortfront-f90, fortfront-lf, lfortran, gfortran-dg |
| 1015, 1016 | 100 | fortfront-f90, fortfront-lf, lfortran, gfortran-dg |
| 1017, 1018 | 150 | fortfront-f90, fortfront-lf, lfortran, gfortran-dg |
| 1019, 1020 | 200 | fortfront-f90, fortfront-lf, lfortran, gfortran-dg |
| 1021, 1022 | 300 | fortfront-f90, fortfront-lf, lfortran, gfortran-dg |
| 1023, 1024 | 400 | fortfront-f90, fortfront-lf, lfortran, gfortran-dg |
| 1025, 1026, 1027, 1028 | 500 | fortfront-f90, fortfront-lf, lfortran, gfortran-dg |
| 1029, 1030, 1031 | 600 | fortfront-f90, fortfront-lf, lfortran, gfortran-dg |
| 1032, 1033, 1034 | 750 | fortfront-f90, fortfront-lf, lfortran, gfortran-dg |
| 1035, 1036, 1037 | 900 | fortfront-f90, fortfront-lf, lfortran, gfortran-dg |

Known manifest categories remain outside the green denominator: `XFAIL`,
`NOREF`, and `SKIP`. <!-- slop-ok: technical status categories --> The earlier
full ffc invocation was not repeated after this
sampling policy was adopted. The full-corpus parity gates therefore remain
unclaimed until a provenance-verified run is deliberately scheduled.

This pass landed ffc `baff74f`, `952c2a0`, and `649ac06`, plus FortFront
`8d5809e7` and `1a1f4575`, on `main`. Focused
behavioral tests passed for complex intrinsic kind conversion, defined binary
and unary operators, submodule generic real return dispatch, diagnostics,
scalar expression lowering, optional scalar kinds, derived allocatable-array
selectors and assignment, diagnostics, the conformance gauntlet smoke test,
and the FortFront nested single-line-IF parser. FortFront's duplication check
also reported zero violations. Stable XPASS entries promoted during the
sampled reruns include
`array_op_10.f90`, `associate_14.f90`, `nullify_06.f90`,
`implicit_do_print.lf`, `issue_1410_pointer_null.lf`, `pr95614_1.f90`,
`separate_compilation_12a.f90`, `proc_decl_20.f90`, `interface_05.f90`,
`legacy_array_sections_05.f90`, `separate_compilation_16.f90`, and
`string_14.f90`.

The follow-up work kept the bounded policy. Every corpus invocation used a
deterministic random subset, and no whole-corpus run was performed. The count
was increased only after repeated clean subsets. Seed 1002 was taken after the
VALUE fixes and still exposed
known failures in `class_is_1_ok.f90`, `elemental_function_5.f90`, and
`pr78290.f90`. `value_3.f90` was an exact PASS. Seed 1003, after the
host-associated class descriptor fix, had no unexpected `FAIL` or `XPASS` in
either FortFront suite (`46 PASS/4 XFAIL` and `41 PASS/9 XFAIL`), while its GFortran
sample had three known failures, and its LFortran sample produced three
repeatable XPASS entries. Seed 1004 was deliberately not treated as green:
FortFront had two output mismatches, LFortran had one XPASS, and GFortran had
two failures. Later clean seeds reached 300, 400, 500, 600, 750, and 900 files
per suite. Seed 1037 initially exposed several unclassified outcomes. Exact
manifest classifications made its rerun clean. Three independent clean
900-file subsets permit increasing the bounded sample to 1000 files per suite,
but that increase is deferred while implementation work addresses the owned
XFAIL backlog.

At 900, the stratified draw is 45 FortFront-F90 files, 22 FortFront-LF files,
349 LFortran files, and 484 gfortran.dg files per seed. The three recorded
900-file refreshes were disposition-clean, including seed 1037. Its result does
not claim the full-corpus parity gate. Remaining `XFAIL`, `NOREF`, and `SKIP`
entries remain outside the behavioral PASS count.

The repeatable promotions from this work are `format_02.f90`,
`modules_29_module1.f90`, `nested_external_dedup_01.f90`, and
`write_implied_do_1.f90`. Exact regression checks also pass for
`class_is_1_ok.f90` and `binding_label_tests_29.f90`, with the latter passing in
three repeated GFortran-suite runs. The sampled `named_constructs` mismatch was
a real ASSOCIATE write-through gap. ffc now matches gfortran and the exact case
passed three repeated runs after the scalar alias-storage fix in ffc `bd8824c`.

Seed 1005 was another bounded 50-file-per-suite random sample. FortFront had
3/3 and 1/1 PASS in its two suites. LFortran had 5 PASS and 14 XFAIL with no
unexpected result. GFortran had three known failures in its 27-file draw:
`impure_assignment_1.f90`, `submodule_36.f90`, and
`subref_array_pointer_4.f90`. The sample was not green, so the count remains
50 and was not increased. The first two are owned by FortFront diagnostics.
The last is the existing pointer-array descriptor gap.

The sampled `gfortran.dg/pr125263.f90` case now passes ffc's own runtime
assertions. The installed gfortran 16.1.1 reference terminates at `STOP 1`, so
the gauntlet records that comparison as `PASS/NO-REF` rather than treating a
broken reference execution as an oracle.

Session evidence, 2026-07-31:

Parallel-agent waves and serial architecture chains ran across ffc, FortFront,
LIRIC, and fluff. Each agent worked in its own worktree, wrote a failing
behavioral oracle first, required a green local `fo` pipeline, and
squash-merged. On the user's explicit instruction, merges did not wait for
GitHub CI.

Issue counts moved from 120 to 65 open in ffc, 24 to 16 in FortFront, 5 to 1 in
LIRIC, and 3 to 2 in fluff. The FortFront count rose from its low of 9 because
the chains surfaced real frontend gaps rather than working around them in ffc.

Measured corpus state on the two maintained in-tree corpora, which are the
genuine 100% targets, at ffc `b4a2961`:

| Suite | Session start | Now | Evaluated |
| --- | --- | --- | --- |
| fortfront-f90 | 341/442 PASS, 0 FAIL | 452/553 PASS, 8 FAIL | 77.1% to 81.7% |
| fortfront-lf | 206/264 PASS, 0 FAIL | 211/265 PASS, 0 FAIL | 78.0% to 79.6% |

fortfront-lf still has no failures. Both totals grew because the FortFront
updates brought new examples, so part of each gain is new coverage rather than
repaired cases. The eight fortfront-f90 failures are tracked by ffc #609, #531,
and #606.

The last full four-suite measurement, at ffc `970cfbe` with FortFront
`0d6a66db`, recorded 2788 PASS against 2574 at session start, moving the
evaluated rate from 29.8% to 32.1% and strict from 23.6% to 25.4%. That
snapshot now trails main by a large margin. ffc PR #596 owns the
provenance-verified regeneration, and `main` stays red on
`test_parity_dashboard` until it lands.

Measurement method matters and is not optional. Per ffc #642, two clean
worktrees at the same commit disagree on `pdt_08.f90`, so a cross-worktree
branch-versus-main comparison carries unknown error. Use same-worktree
before and after: revert `src/`, rebuild, measure, restore. The character chain
recorded two single-run artefacts that this method corrected in opposite
directions, one that would have been reported as a regression and one as a
gain.

Two regressions are open against this measurement. ffc #571 owns fortfront-lf
falling from 206 to 196, where ten examples fail with `conflicting declaration
metadata`, most likely from `ab4d6fd`. ffc #566 owns `expr_11.f90` falling out
of PASS in the lfortran suite. Both are maintained corpora, so neither may be
absorbed into a manifest.

FortFront pin bump, blocked and measured:

ffc consumes FortFront as a path dependency, so the effective pin is the
sibling checkout revision recorded in the parity snapshot, not a git revision
in `fpm.toml`. Bumping it from `f75de2c` to `b1f33bf3` is a clear net win on
totals, taking gfortran-dg from 1269 to 1312 PASS and fixing 56 files, but it
also turns 28 cases from PASS to FAIL and 27 of those files are valid under
`gfortran -fsyntax-only`. ffc PR #545 therefore stays open and unmerged.

The three causes are filed with reduced cases and cover 17 of the 27:

- FortFront #2942: kind-suffixed literals such as `3._dp` and `.false._8` are
  rejected as an invalid character in a name.
- FortFront #2943: a leading UTF-8 BOM is rejected as an invalid character,
  including in FortFront's own example file.
- FortFront #2944: an integer unit number is consumed as a statement label, so
  `flush 6` fails.

These escaped the earlier over-rejection audit because that audit baselined at
`4edcef6a`, which already contained the lexer rejection work. The defects were
present on both sides of its diff and cancelled out. Baseline a rejection audit
at the last revision before the whole wave, not at an intermediate commit.

Measurement integrity, ffc #547:

Conformance tests resolve the `ffc` binary through a relative path that can
escape into a sibling worktree. One run measured
`../ffc-issue-425-lf-infer-array/build/fo/app/ffc`, another worktree's
compiler. They also share fixed `/tmp`
report paths. A test can therefore report a PASS belonging to a different
build, so any parity figure taken while agents ran in parallel carries that
uncertainty. Fix #547 before quoting a final number.

FortFront over-rejection, the central correctness finding:

FortFront #2924 recorded that the rejection wave was held by review with 84
blocking problems, 33 of them over-rejection. The branches carried green CI, so
CI was never the missing control. The PRs were merged anyway, which was an
orchestration error. A corpus audit then measured the result: over 8,644 files,
baseline `4edcef6a` against main `8cf3ab96` showed 228 newly rejected files, of
which 126 were false positives under a `gfortran -fsyntax-only` oracle and 102
were the intended tightening.

PR #2938 (`a8d06a33`) narrowed six rules and recovered 54 files, leaving 72
false positives. It also built the control that was missing:
`scripts/corpus_rejection_gate.sh`, wired in through `make
check-rejection-gate` and a CI step against a committed baseline. Every
rejection change must now show zero newly rejected files outside its own
fixtures. One existing assertion demanding rejection of assumed-length BIND(C)
character dummies was corrected against the gfortran oracle rather than
deleted.

Of the 72 remaining false positives, 68 come from `bfe231a2` (#2919), which
treats every statement-parser gap as a hard error and so rejects valid
`sync all(stat=)`, `form team`, `change team`, and typed array constructors.
That commit is being reverted and #2897 reopened. The governing rule: a parser
that does not understand a construct must not conclude the construct is
invalid.

Process lessons carried forward:

- A rejection change needs an accepted-side gate. Its failure mode is invisible
  to the test that proves the rejection works.
- Read a repository's process and meta issues before dispatching agents at its
  PR queue.
- Merging without CI is safe for isolated feature work with a behavioral
  oracle. It did not cause the over-rejection, which shipped green.
- Unbounded parallel builds drove the machine into swap at load 98 on 32 cores.
  Agents must cap `FO_JOBS`.
- Two agents independently added a `strip_source_comment` helper and briefly
  broke `main`. Concurrent conformance tests also clobber each other through
  hardcoded shared `/tmp` fixture paths.

Two blocked ffc issues have owners upstream: #353 needs FortFront #2933, since
the frontend cannot represent alternate returns and silently discards the
selector on `RETURN 1`; #357 needs FortFront #2928, since an INTERFACE block in
a module-contained procedure drops every later program unit.

CI now covers all five core repositories. fo and fx had none until 2026-07-25; <!-- slop-ok: technical semicolon -->
they are the two every other build depends on, so the `Definition of 100%`
clause requiring green CI matrices was previously unsatisfiable for them.

fo's gate earned its keep on the first run that reached the test stage: it
caught `test_scaffold_builds` resolving `fo` from `PATH`, which succeeds on a
developer machine and fails on a runner. That class of defect is invisible to
local verification by construction.

fx was made public on 2026-07-25 so its dependents can fetch it without a
credential, and its history was rewritten first to purge a 39 KB compiled test
binary committed in `99fd7c3`. Nothing else needed removing: no other large
blobs, and no file existed in history that was absent from the tree.

Landed-work checkpoint, 2026-07-24:

The 2026-07-14 entry that stood here described ffc #333, #334, #335, #336, and
#337 as achieved. That work existed only in uncommitted worktrees under
`/mnt/storage/code/lazy-fortran`. It was never committed, the worktrees have
been removed, and **none of it is on ffc `main`**. `main` has no canonical
array descriptor: there is no `test_array_descriptor_layout`, and
`docs/RUNTIME_ABI.md` describes no rank-7 200-byte layout. Those issues are
open and unimplemented. Do not treat them as done.

What actually landed, including this session's documentation pass:

- FortFront `6ec0acce` exposes the public `resolved_type_query_t` and
  `query_resolved_type` surface, so lowering can read exact scalar category,
  kind selector, storage width, rank, and derived-type identity without
  inspecting source spelling. It also completes the scope binding work that
  surface builds on (`declaration_entity_index`, `is_inferred`) plus parser and
  standardizer support for legacy mixed declarations and omitted triplet
  bounds.
- FortFront `42e2d5e1` resolves a kind selector declared in scope before the
  conventional-name table, so the `iso_fortran_env` rename idiom
  (`dp => real64`) resolves without weakening an explicitly shadowed selector.
- ffc `a51fa4f` lowers runtime fixed-width scalar characters, the first slice
  of #350. `character(len=n+1)` remains unsupported pending an upstream
  expression-node query.
- ffc `ed9baaa` refreshes the parity baseline and fixes a corpus-digest
  collation bug: the gauntlet built its file list under the caller's locale
  while the dashboard hashed it under `LC_ALL=C`, so the pinned digest was not
  reproducible across environments.
- ffc `17ec785` lands #329 through PR #481. Specification-expression named
  constants now use the declaration's FortFront binding anchor during character
  result classification, with behavioral scope tests and green CI.


Documentation and scope corrections, 2026-07-24:

- ffc `b6484be` retargets the goal at the standard rather than 100% corpus
  pass, and removes a dead issue graph from three documents. The `E1` through
  `E10` epics (#262 through #271), the #272 umbrella, and `krystophny/liric#520`
  are all closed; 20 of the 21 issues `docs/PARITY_PLAN.md` listed as current
  work had closed too. `BACKLOG.md` is now a pointer, since keeping a fourth
  planning document in sync is why the drift went unnoticed.
- fo `eacb501` records the fo/fluff lint boundary and corrects three module
  paths that pointed at code which no longer exists.
- fluff `1466f42` states fluff's role, corrects `DESIGN.md` claims of zero
  false positives and F2018 coverage, and refreshes a known-limits list that
  named six capabilities as missing which all work today.
- fx: `fix/issue-10` and `pr-28` deleted locally and on the remote. Both were
  squash-merge residue from PR #28; every file they touched is on `main` and <!-- slop-ok: technical semicolon -->
  issue #10 is closed.

Issues opened from that pass: ffc #473 (audit the F2023 delta), fluff #260
(formatter is not idempotent), #261 (F006 misses subscripted array reads), #262
(28 of 94 test programs cannot fail), #263 (unused `stdlib` dependency and a
floating FortFront pin), #265 (two tests glob fpm build paths and can grade a
stale binary). fo #59 gained a requirement for changed-file selection and
content-hash caching.


Systemic-gate work, 2026-07-25:

The recurring defect across these repositories is not broken code, it is green
signals that are not evidence. Three instances found so far, each of a
different shape:

- Test programs that tally results, print a failure summary, and exit 0
  regardless. fluff has 28 of 94; fortfront has 22 of 629; ffc and fo have none. <!-- slop-ok: technical semicolon -->
- Tests that grade a binary they did not build. fluff #265: two tests glob
  fpm's build layout while `fo` builds elsewhere, so a stale artifact is
  measured instead of the current tree. This reported a green repository as red
  once, and would as readily report a broken one as green.
- Tools that contradict themselves. fo #112: `fo check` emits `tests_ok: true`
  and exits 0 while its own summary reports 14 failures, because exit codes and
  parsed output are two unreconciled sources of truth.

Measured baseline, 2026-07-25, using a detector that strips Fortran comments
and string literals before matching:

| Repository | Programs that cannot fail | Total | Lines |
| --- | ---: | ---: | ---: |
| fortplot | 47 | 294 | 3,729 |
| fluff | 30 | 94 | 10,663 |
| fortfront | 22 | 627 | 1,092 |
| fortnb | 1 | 11 | 36 |
| fo | 1 | 24 | 3 |
| ffc | 0 | 288 | 0 |
| fx | 0 | 15 | 0 |
| fortnum | 0 | 75 | 0 |
| fortcov | 0 | 17 | 0 |
| **total** | **101** | **1,445** | |

fo's single case is `test/bench_noop.f90`, a three-line benchmark stub, and is
not a defect. fortplot carries the largest count and was not previously known
to have the problem; it sits outside the current toolchain scope, so it is <!-- slop-ok: technical semicolon -->
recorded here rather than scheduled.

ffc, fx, fortnum, and fortcov are clean. Whatever discipline produced that
result did not transfer to the other repositories.

The first of these is being made mechanical: a `fo lint` rule that flags a test
program with no failure path. The detection must strip Fortran comments and
string literals before matching, because fixture data routinely contains the
words being matched; a naive regex already produced a false "zero remaining" <!-- slop-ok: technical semicolon -->
result. Once the rule lands, every repository using `fo` inherits the gate.

Related: fx #35 (framework self-test leaks intentional failures into the parsed
summary) and fx #36 (segfault in the cache-key digest under concurrent builds).


Wave outcomes, 2026-07-25:

Twenty pull requests landed across the five toolchain repositories. The
substantive results:

- Every repository now has CI. fo and fx had none; ffc additionally gates lint
  and formatting, not just build and test.
- ffc gained a conformance point. The lfortran suite moves 848 to 849 passes,
  3420 to 3419 xfail. The recovered case is `attr_intrinsic.f90`, a host-scope
  parameter named `abs` shadowed by an `intrinsic abs` inside a contained
  procedure: text-keyed symbol lookup cannot separate those entities, and
  binding identity can. Current baseline is 342/205/849/1175.
- ffc has a binding-keyed symbol table with private state, its own unit tests,
  and roughly ten migrated call sites. Around 300 text-keyed `find_symbol` sites
  remain; that migration is #329 through #332. <!-- slop-ok: technical semicolon -->
- fluff's formatter no longer corrupts `/=`, `**` and `//` into invalid Fortran,
  is idempotent, and honours its configured indent width. F006 no longer reports
  an array as unused when it is only read through a subscript: 87 diagnostics
  fell to 67 with none added.
- fortfront shed roughly 3,600 lines that no execution path reached.

Two waves produced no mergeable code and are worth recording as such.

Thirteen parallel branches over the rejection issues (#2881 to #2901) were all
rejected by review: 84 blocking problems, 33 of them over-rejection, where a new
check refused valid Fortran that both main and gfortran accept. A conformance
diff over 8,703 gfortran.dg files found one rule newly rejecting twelve, roughly
eleven false positives to one true positive. Every suite stayed green throughout,
because the accepted-side fixtures never reached the constructs the rules
touched. fortfront #2924 requires a corpus gate before that work is retried.

The binding-identity change was initially held for the opposite reason:
neutralizing the entire mechanism left all 288 tests green. A regression gate
cannot detect a change that does nothing. Both failures are the same shape from
opposite sides, and the standing requirement is now two gates, not one: does it
break anything, and does it do anything, each verified by neutralizing the
mechanism and confirming the suite goes red.

Operational constraints discovered the expensive way:

- A regenerated parity snapshot records the ffc revision it was generated at,
  and the check requires that revision to be an ancestor of HEAD. Squash-merge
  discards branch ancestry, so a snapshot-bearing PR must be merged with a merge
  commit or it leaves main red on landing.
- The snapshot is only reproducible on an idle machine.
  `benchmark_5000_lines.f90` compiles in about 5s against a 10s timeout and
  fails spuriously under load, reported as a compile error rather than a
  timeout. ffc #478.
- Tests must not hardcode a build layout. `fo` builds to `build/fo/`, fpm to
  `build/gfortran_*/`, and CI runs fpm; getting this wrong in either direction <!-- slop-ok: technical semicolon -->
  produces a test that grades a stale artifact or cannot find one at all.

Unlanded work recovered from the removed worktrees:

- ffc #447. The isolated exact-type-query patch applies but regresses 30 test
  targets, and 16 of those are output mismatches rather than refusals: f32
  array elements print as their IEEE-754 bit patterns. It is not a rebase
  candidate. Measurements are on the issue.
- ffc #333 through #339, #342. The descriptor and assumed-shape reduction body
  is entangled across several includes; its "isolated" diffs need 44 of 46 <!-- slop-ok: technical semicolon -->
  procedures from a 1409-line rewrite that no diff captured, and that rewrite
  also deletes procedures other includes still call. Redo from the issue
  descriptions against current `main`; the old patches are not a starting <!-- slop-ok: technical semicolon -->
  point. Details on #342.

Both bodies of unlanded work, and the evidence notes written alongside them,
are preserved as `worktree-archive-20260724.tar.gz` and
`reports-20260714.tar.gz` beside this file. They are the only surviving copy:
the worktrees they came from have been removed. Restore a patch against its
recorded base commit, not against `main`. Treat them as evidence for the issue
write-ups above rather than as a resumable branch.


Checked-in cross-corpus snapshot (the external-corpus rows remain the last
provenance-verified four-suite baseline):

| Corpus | PASS | XFAIL | XPASS | FAIL | NOREF | SKIP | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FortFront F90 | 341 | 101 | 0 | 0 | 94 | 0 | 442 |
| FortFront Lazy | 206 | 58 | 0 | 0 | 0 | 0 | 264 |
| LFortran integration | 849 | 3419 | 0 | 11 | 145 | 1 | 4280 |
| `gfortran.dg` | 1178 | 2132 | 0 | 330 | 1 | 2298 | 5938 |

The historical snapshot reports 29.8% evaluated and 23.6% strict overall, and
29.8% evaluated and 23.8% strict after exclusions. Its ffc provenance anchor
is `ebe55b7`, with FortFront `f75de2c` and LIRIC `f1625993341f`. Those values are
not the current implementation percentage. `XPASS` is zero in that checked-in
classification snapshot. Refresh the external rows with a provenance-verified
full run before claiming a new four-suite corpus rate. Routine progress remains
sampled under the policy below.

The #330 refresh promotes `issue_nested_internal_procedures.lf` to PASS and
promotes `host_assoc_call_3.f90`, `host_assoc_function_4.f90`, and
`parent_result_ref_2.f90` to PASS. The earlier F90 and GPU manifest changes
remain recorded in the checked-in snapshot. See
[PARITY_STATUS.md](ffc/docs/PARITY_STATUS.md),
[CONFORMANCE.md](ffc/docs/CONFORMANCE.md), and the generated
[dashboard](ffc/test/conformance/parity_dashboard.tsv). Refresh it before
claiming a new corpus pass rate.

Issue ownership:

- All 11 current unexpected LFortran failures have atomic ffc issues.
- All 23 current valid-program gfortran.dg failures have atomic ffc issues or named acceptance on a foundation issue.
- The 255 true gfortran.dg negative cases are partitioned into 38 exclusive rule groups: 17 ffc issues and 21 FortFront issues. Audit coverage: 255 unique files, zero missing, zero duplicates.
- Every xfail and skip entry has machine-validated ownership metadata. The checked-in parity snapshot verifies provenance, scoped denominators, and all 135 open owners.

Architecture pressure:

- Repository ownership is sound: FortFront owns parsing, semantic resolution, and typed queries; ffc owns lowering and runtime ABI; LIRIC owns backend/session emission; fo owns the workflow.
- ffc is not ready for unrestricted parallel feature waves. Its direct lowerer
  is one 56,101-line textual module assembled from 69 include fragments, with
  265 `find_symbol` call sites and several incompatible descriptor conventions.
- FortFront is modular enough to proceed, but ffc must stop importing private
  AST internals. Expression, diagnostic, and statement queries landed in
  FortFront #2876 through #2878. Stable binding identities landed in #2880,
  and `query_resolved_type` now exposes exact scalar category, kind selector,
  storage width, rank, and derived-type identity. Program-unit and declaration
  queries landed in #2879. FortFront #2906 now covers the previously missing
  `iso_c_binding` kinds, indirect aliases, and the covered array declaration
  forms. The old ffc #447 exact-type-query patch remains historical regression
evidence. Current typed dispatch work is tracked by ffc #419, #422, #437,
  #461, and #467.
- LIRIC backend coverage is strong across LLVM 3.8 through 22 and multiple architectures. Public ABI versioning, typed pointer metadata, and non-mutating legacy emission remain required.
- fo is operational and is not the cause of corpus parity. Its ffc-relevant queue is small and concrete.

Current architecture blockers and order:

1. Binding identity and scope ownership come first. ffc must consume one
   FortFront binding graph across host, ASSOCIATE, USE, module, and submodule
   scopes. The active trackers are ffc #584 and FortFront #2883, #2924, and
   #2951.
2. Storage representation follows. Fixed, automatic, allocatable, pointer,
   assumed-shape, section, character, and derived arrays must converge on the
   canonical descriptor contract. The active trackers are ffc #337, #338,
   #339, #348, and #643.
3. Typed dispatch follows the descriptor work. Procedure pointers, procedure
   dummies, polymorphic arrays, SELECT TYPE, and Lazy specialization need one
   callable and type-identity model. The active trackers are ffc #419, #422,
   #437, #461, and #467.
4. Shared engines then absorb control flow, constructors, defined assignment,
   WHERE/FORALL, and I/O. The active trackers are ffc #345, #449, #455, #459,
   #460, and #462.
5. Performance and corpus reliability close the loop. Timeout reporting,
   large-source compile cost, runtime memory safety, and rejection regressions
   are tracked by ffc #478, #531, #576, and #663.

Each issue must state its preserved invariant, use the shared representation,
include a positive and a negative behavioral case, and run the focused
`FO_JOBS=1 fo test` plus bounded exact conformance. A full `fo` workflow is a
delivery-boundary check. A manifest classification may record a known gap; it
cannot replace the implementation or turn a wrong-code result into XFAIL.

Architecture migration map:

- Binding identity: FortFront's semantic binding graph and public typed-query
  surface; ffc `src/ffc_fortfront_queries.f90`,
  `src/session_symbol_table.f90`, `src/session_program_lowering_associate.inc`,
  `src/session_program_lowering_interface.inc`, and the module/submodule
  collectors. No text-name fallback may survive this migration.
- Descriptor and storage ABI: ffc `src/ffc_array_descriptor.f90`,
  `src/ffc_character_descriptor.f90`,
  `src/ffc_polymorphic_descriptor.f90`, the allocatable, assumed-shape,
  pointer, section, character-array, and array-element lowering includes,
  `runtime/ffc_runtime.c`, and `docs/ARRAY_DESCRIPTOR_ABI.md`.
- Module and callable identity: ffc `src/ffc_module_artefact.f90`,
  `src/session_program_lowering_fmod.inc`, `src/session_program_lowering_submodules.inc`,
  `src/session_program_lowering_proc_dummy.inc`,
  `src/session_program_lowering_polymorphic.inc`, and
  `src/session_program_lowering_lazy_monomorph.inc`; LIRIC's public session ABI
  is changed only when the generated-call contract requires it.
- Shared engines: ffc array-expression, reduction, WHERE/FORALL, constructor,
  defined-assignment, control-flow, and I/O lowering includes. Case handlers
  are deleted when their shared engine replaces them.
- Language contracts: update the standard's shape/rank and module-signature
  contracts before freezing an ABI that depends on them. `fo`, fluff, and the
  external LFortran/GCC corpora remain verification infrastructure, not
  alternate implementations.

## Active pull requests

- [LIRIC PR #524](https://github.com/krystophny/liric/pull/524), Restore LLVM compatibility and AArch64 correctness: LLVM 3.8 through 22 and platform jobs pass. Its benchmark-matrix blocker was the LFortran baseline failing to find `liric/liric_session.h`; LIRIC #528 has since landed the CI contract, so this PR needs a rerun and a merge decision.
- FortFront PRs #2912, #2914, #2915, #2916, #2919, #2920, #2921, #2922, and #2923 cover the remaining rejection issues and are being landed by a serial rebase train. #2915 and #2921 additionally carry real CI failures to repair before they merge.
- [fluff PR #269](https://github.com/lazy-fortran/fluff/pull/269), make every test program able to fail the build, is open for fluff #262.

## Architecture-first work order

### Wave 0: complete

ffc #299 landed in PR #468. The maintained pipeline now verifies the generated
parity snapshot.

### Chunk 1: freeze the public compiler graph

The public-query and binding-key prerequisites have landed in the completed
paths. The remaining architecture work is not a new lookup workaround: ffc
#584 must consume one binding graph across host, ASSOCIATE, USE, module, and
submodule scopes, with FortFront #2883, #2924, and #2951 preserving accepted
programs while tightening diagnostics. Treat the old #447 patch as regression
evidence, not as a resumable branch.

### Chunk 2: centralize binding-driven typed lowering

Build the typed scalar and expression seams against the public FortFront
binding graph before adding more case handlers. ffc #453, #454, #456, #466,
#579, and the scalar portions of #419 and #467 may proceed here. The
descriptor-dependent parts of #419, #422, #437, #461, and #467 are blocked on
Chunk 3 and must not grow a second representation. Completion requires
explicit value types, lifetimes, aliasing assumptions, and generated-code
checks. New features must not add parallel ad hoc lowering paths.

### Chunk 3: stabilize descriptor, runtime, and backend ABIs

- Arrays: ffc #333, #334, #335, #336, #337, #338, then #339. The closed
  #333-#336 trackers do not by themselves prove that the descriptor ABI is in
  `main`. The landed-work checkpoint above records that the old implementation
  was never committed. The active descriptor blockers are #337-#339 and #643.
  Start from the issue descriptions against current `main` and require the
  descriptor layout and lifetime oracles before marking the work landed.
- Characters: ffc #349 and #348, then #347 and #350.
- Runtime and backend: ffc #374, #376, #396, #423, #427, and #428; LIRIC #528,
  #525, #526, and #527; then ffc #375. LIRIC #523 remains a correctness gate.
- Polymorphism: ffc #400, #417, #419, #420, #421, then #422.

Completion requires versioned public interfaces, verified ABI declarations,
non-mutating legacy emission, matching runtime archives, canonical ownership
and view lifetimes, and no incompatible descriptor convention in a migrated
path.

### Chunk 4: make module artifacts authoritative

ffc #397, #414, #415, #297, and #416 are landed. Module artifacts carry
versioned scalar, derived-layout, character, generic-rank, and deferred-
interface contracts at schema 5, and each bump rejects an older artifact rather
than misreading it. Imported interfaces drive separate compilation: parent,
submodule, and caller now compile as three independent `ffc` invocations and
link. Source rescans and layout guesses are not acceptable substitutes.

The remaining two swap order: **#437 before #433**. #433 asks the artifact to
record a specialization key, public signature, and emitted symbol, but ffc
emits no Lazy specializations yet, so serializing that identity first would
invent a record format with no producer. #437 is the foundation and #433 then
serializes a key that already exists.

#437 is also narrower than its text implies. An untyped Lazy dummy already
works whenever every call site agrees on one concrete type, because FortFront's
inference resolves the dummy and ffc consumes it. `twice(3)` with `twice(4)`
prints 6 and 8. Only disagreement fails: `twice(3)` with `twice(2.5)` gives
`integer identifier was not declared: x`. The missing piece is monomorphization
alone, one instance per distinct concrete signature.

The seams are located and need no new phase or forked path.
`session_program_lowering_top.inc` already runs an ordered chain of `collect_*`
pre-passes, so specialization collection fits as one more collector. Call-site
kinds come from `get_type_for_node` with `inferred_type_to_value_kind`, the
same pair the working single-signature path uses. One body under several
symbols is `lower_scalar_function`'s existing `emit_name_override`, which the
submodule path already uses. Per-instance dummy kinds thread an optional
override through `define_parameter_symbol`, the single point where a dummy's
kind is decided.

### Chunk 5: route arrays, values, and I/O through shared engines

- Array declarations and expressions: ffc #458, #459, #342, #344, #343, #345,
  #399, #418, #450, #424, #426, #432, and #435.
- Derived values and lifetime: ffc #401 through #407, #462, and #465.
- Constants and initialization: ffc #412 and #463.
- I/O: ffc #451, #460, #434, and #436.

Preserve Fortran column-major layout, lower-bound and extent semantics, overlap
and aliasing behavior, allocation and finalization lifetimes, procedure dispatch,
and formatted-I/O contracts. Verify each invariant with behavioral tests and
generated-code or ABI checks where applicable.

### Chunk 6: close corpus breadth

Land ffc #408 through #438 after their named architectural dependencies, except
issues already included above. Current valid-program regressions #351 through
#373 may proceed when their shared engine is green. No isolated fix may fork an
ABI or duplicate a lowering path merely to turn one corpus case green.

### Diagnostic lane

This lane may proceed after the relevant public diagnostic or semantic query is
available:

- FortFront rejection issues #2881 through #2901 can proceed independently by analyzer/parser file. #2889, #2890, #2893, #2894, #2898, #2899, and #2901 are landed. The remainder are already written as open PRs #2912, #2914, #2915, #2916, #2919, #2920, #2921, #2922, and #2923, which mutually conflict because they edit the same analyzer files; land them through a serial rebase train, resolving each conflict as the union of the rejection checks.
- ffc rejection issues #378 through #394 can proceed independently except where their body names a scope or descriptor dependency. #383, #384, #391, #393, and #394 are landed.
- Every PR runs its named cases plus the full repository pipeline. Never convert
  a negative failure to XFAIL.

## The silent-source-drop family

A second defect family runs through FortFront: the parser silently drops
source. The program compiles clean, runs, and does less than it says.

Two were fixed this session. #2928 counted the `END INTERFACE` token as
opening an interface block, so nesting never returned to zero and every program
unit after the module was swallowed. The bare-`END` case was the other, where
`find_procedure_end` recognized only `end subroutine` and `end function`, so a
procedure closed by a plain `END` absorbed every following sibling into its
token span.

Open and under investigation as one family, because both fixed cases were
line and token-span scanning mis-deciding where a construct ends: #2966
(uppercase `END TYPE`), #2967 (named IF, SELECT and BLOCK inside contained
procedures), #2972 (statements after a nested DO nest), #2974 (trailing
entities in a multi-entity declaration), and #2977 (`EXTERNAL` inside a
module-contained procedure).

The structural question worth answering is whether the scanner can be made to
fail loudly when it cannot account for every token in a unit. Such a check
would catch this entire family and every future member, instead of five
targeted patches against one recurring mistake.

## Silent miscompiles found this session

Three defects of the same family surfaced, all in Lazy specialization, and all
found by agents implementing adjacent work rather than by any test:

- A conflicting-signature call at program level bound to whichever typed copy
  happened to be emitted first, so `twice(3)` printed 6 while `twice(2.5)`
  printed 0, with no diagnostic. Swapping the calls moved the wrong answer.
  ffc `18e1164` refused the call as an interim guard, and FortFront #2971
  (`06b15ed6`) fixed the cause: monomorphization now renames each call site to
  the specialization created for it, so the copies lower as ordinary typed
  procedures. ffc #437 can now drop the refusal and add its positive oracle.
- The same shape inside a module prints uninitialized memory
  (`2.15167795E+09`) and was not caught at all, because FortFront does not
  monomorphize module procedures, and ffc cannot guard it either since the
  conflicting calls are not discoverable in the arena. FortFront #2978 fixed it
  as a rejection in Lazy mode, deliberately narrow so that missing knowledge
  means silence: the name must be defined exactly once arena-wide and only
  all-literal argument lists are classified.
- Argument-less call sites emitted a dangling `.ffc.nested` reference, so the
  program linked and died at load with exit 127. Fixed in ffc #576, which
  established the invariant that ffc never emits a reference to a symbol
  nothing defines.

The pattern is worth stating plainly, because it shapes how the remaining work
should be judged. Every one of these compiled cleanly and produced a wrong
answer. None was caught by a rejection test, a corpus pass count, or a green
pipeline. A conformance percentage cannot see them at all: the case either was
already expected to fail, or passed while computing garbage. Treat a silent
wrong answer as strictly more serious than a missing feature or a false
rejection, and prefer refusing a construct over emitting code that might be
wrong.

## Measurement policy

Routine progress reads use a stratified random sample, not a full corpus run.
A full four-suite run is 10,924 files compiled twice each, which is slow enough
that measurement gets skipped and the snapshot goes stale. ffc #567 owns the
sampling mode and the reference-output cache.

Sample sizes for a 95% confidence interval at a pass rate near 30%, with the
finite-population correction:

| Margin | Sample | Speedup |
| --- | ---: | ---: |
| plus or minus 1% | 4,640 | 2.4x |
| plus or minus 2% | 1,703 | 6.4x |
| plus or minus 3% | 828 | 13x |

The statistical reference point is plus or minus 2% at about 1,700 files, but it
is not the first run. Start with a small deterministic stratified subset and
increase the count only after repeated disposition-clean subsets. The current
record reaches 900 files per suite, with clean seeds 1035, 1036, and 1037.
Under the repeated-clean-subset rule, 1000 is permitted but deferred while
implementation work addresses the owned XFAIL backlog. A full corpus run
remains a deliberate final gate, not a routine progress check.

Two rules keep sampled numbers honest:

- Always quote the margin with a sampled rate, so it is never mistaken for an
  exact figure.
- Never write a sampled run into `test/conformance/parity_dashboard.tsv`. The
  checked-in snapshot remains a full, provenance-verified run, and it is what
  the Definition of 100% is judged against.

## Merge gates are mandatory

An open PR is not ready because it has an approval or because an administrator
can bypass CI. Before merging, rebase it onto current `main`, inspect the
changed files and unresolved review threads, run every relevant local gate,
and require the GitHub checks to be green. Never use `--admin` to bypass a red
check, and never merge a dirty or stale head.

Run the full `fo` pipeline for a delivery-boundary change, the rejection gate
for anything that can reject source, the duplication gate for test changes,
and the affected bounded conformance tranche for lowering changes. A full
corpus run is reserved for the provenance snapshot release gate; ordinary PR
work remains XFAIL-first and sampled.

Squash merging is the default only when it preserves repository invariants. A
PR carrying a provenance snapshot whose recorded ffc revision is not an
ancestor of the post-merge `main` must either regenerate that snapshot against
the current base or use the merge method required by `docs/CONFORMANCE.md`.
Do not trade a requested merge shape for a red `test_parity_dashboard`.

## Clean architecture mandate

The goal is one correct implementation of each concept, not a migration that
leaves the old one alive beside it. Legacy paths are deleted, not deprecated.

Rules:

- **One convention per concept.** Four array conventions currently coexist: the
  canonical descriptor, legacy runtime-shape metadata, the inline
  `{data, extent}` record used by allocatable components, and the ad hoc
  handling of sections, pointer arrays and character arrays. Exactly one
  survives. The others are removed from the tree, not left unreachable.
- **Engines before handlers, and handlers deleted when an engine lands.** A
  shared engine that leaves the per-case code in place has not replaced
  anything. Deleting the old handlers is part of the engine's definition of
  done.
- **No fallbacks.** A path selected when configuration is missing, when a
  lookup fails, or when a construct is unrecognized is a second implementation
  that runs exactly when the first one is least understood. Fail loudly
  instead.
- **No compatibility adapters.** If a caller needs an adapter to reach a
  public interface, change the caller.
- **Delete dead code on sight.** Unreachable branches, superseded helpers and
  retired manifests go with the change that supersedes them.

Two constraints bound the mandate, and neither is negotiable: no corpus case
regresses from PASS, and every deletion is justified by a behavioral oracle
that still passes afterwards. A deletion that cannot be verified behaviorally
is a rewrite, and it needs its own red-then-green evidence.

Sequencing follows from this. Binding identity (#584 and its FortFront
dependencies) comes first; canonical descriptor/storage unification (#337,
#338, #339, #348, and #643) is the representation gate; module artifacts and
runtime/backend contracts follow; then typed dispatch (#419, #422, #437, #461,
and #467), shared array/control engines (#342 through #345 and #455), and
corpus breadth. Corpus breadth measures the architecture rather than
producing it.

## Architectural redesign

Redesign is authorized when the existing design is itself the blocker. Do not
contort a fix to fit a structure that cannot be made correct, and do not stop
at a design boundary that needs to move. Interfaces, representations, module
and repository boundaries, and whole mechanisms may be replaced.

Guardrails apply to every redesign:

- exactly one canonical convention survives, never two in parallel;
- no forked or ad hoc lowering path;
- the old mechanism is retired rather than left as a silent fallback, because a
  fallback that activates on a missing environment variable or an unset option
  turns a mismatch into silent wrong behavior instead of a loud error;
- every behavioral oracle and negative control still passes;
- no corpus case regresses from PASS; and
- the redesign and its rationale are stated in the pull request body.

Split a redesign that spans several pull requests into an ordered sequence, and
keep each step green on its own rather than merging one unreviewable change.

Open decisions of this kind: ffc #565 owns retiring the inline runtime path in
favor of a linked runtime, and ffc #297 owns whether separate submodule
compilation carries lowering-level interface records or threads a mutable
arena. The record-serialization pattern from #414 is the recommended shape for
the latter, but the choice is on engineering merit.

## Completion gate for every issue

1. Record the preserved compiler invariant: aliasing, lifetime, dispatch, ABI,
   diagnostics, or generated code as applicable.
2. Show the named behavioral case failing before the change and passing after.
3. Run the repository's focused checks and full `fo` pipeline.
4. Regenerate the parity dashboard when corpus classification, output, scope,
   ownership, or pass state changes.
5. Merge only with required CI green and then update this roadmap before the <!-- slop-ok: requested document type -->
   next issue starts.

## ffc open issues

### Foundation and ABI migration

- [lazy-fortran/ffc#297](https://github.com/lazy-fortran/ffc/issues/297) [fmod-submodule-01] consume parent interfaces for separate submodules (hard)
- [lazy-fortran/ffc#333](https://github.com/lazy-fortran/ffc/issues/333) [arraydesc-01] define the canonical array descriptor ABI (hard)
- [lazy-fortran/ffc#334](https://github.com/lazy-fortran/ffc/issues/334) [arraydesc-03] pass assumed-shape arrays by descriptor (hard)
- [lazy-fortran/ffc#335](https://github.com/lazy-fortran/ffc/issues/335) [arraydesc-02] migrate runtime automatic arrays to descriptors (hard)
- [lazy-fortran/ffc#336](https://github.com/lazy-fortran/ffc/issues/336) [arraydesc-06] migrate allocatable arrays to descriptors (hard)
- [lazy-fortran/ffc#337](https://github.com/lazy-fortran/ffc/issues/337) [arraydesc-04] represent array sections as descriptor views (hard)
- [lazy-fortran/ffc#338](https://github.com/lazy-fortran/ffc/issues/338) [arraydesc-05] migrate pointer arrays to descriptors (hard)
- [lazy-fortran/ffc#339](https://github.com/lazy-fortran/ffc/issues/339) [arraydesc-07] retire legacy runtime-shape metadata (medium)
- [lazy-fortran/ffc#342](https://github.com/lazy-fortran/ffc/issues/342) [arrayexpr-01] centralize array element-expression lowering (hard)
- [lazy-fortran/ffc#343](https://github.com/lazy-fortran/ffc/issues/343) [arrayexpr-03] lower reductions through descriptor iteration (hard)
- [lazy-fortran/ffc#344](https://github.com/lazy-fortran/ffc/issues/344) [arrayexpr-02] make overlapping array assignment alias-safe (hard)
- [lazy-fortran/ffc#345](https://github.com/lazy-fortran/ffc/issues/345) [arrayexpr-04] route WHERE and FORALL through array expressions (hard)
- [lazy-fortran/ffc#347](https://github.com/lazy-fortran/ffc/issues/347) [chardesc-04] migrate character arrays to contiguous descriptors (hard)
- [lazy-fortran/ffc#348](https://github.com/lazy-fortran/ffc/issues/348) [chardesc-03] pass character dummies and results by descriptor (hard)
- [lazy-fortran/ffc#349](https://github.com/lazy-fortran/ffc/issues/349) [chardesc-02] migrate deferred scalar character storage (hard)
- [lazy-fortran/ffc#350](https://github.com/lazy-fortran/ffc/issues/350) [chardesc-05] centralize runtime-length character operations (hard)
- [lazy-fortran/ffc#374](https://github.com/lazy-fortran/ffc/issues/374) [runtime-01] build backend-qualified LIRIC runtime archives (hard)
- [lazy-fortran/ffc#375](https://github.com/lazy-fortran/ffc/issues/375) [liric-abi-01] verify ffc bindings against the LIRIC ABI (hard)
- [lazy-fortran/ffc#376](https://github.com/lazy-fortran/ffc/issues/376) [runtime-02] select and load the matching runtime archive (hard)
- [lazy-fortran/ffc#396](https://github.com/lazy-fortran/ffc/issues/396) [runtime-03] move file-unit state behind the runtime ABI (hard)
- [lazy-fortran/ffc#397](https://github.com/lazy-fortran/ffc/issues/397) [fmod2-01] version scalar procedure metadata in .fmod (hard)
- [lazy-fortran/ffc#399](https://github.com/lazy-fortran/ffc/issues/399) [vector-subscript-01] lower vector subscripts as gather views (hard)
- [lazy-fortran/ffc#447](https://github.com/lazy-fortran/ffc/issues/447) [scalar-expr-01] centralize typed scalar expression lowering (hard)
- [lazy-fortran/ffc#450](https://github.com/lazy-fortran/ffc/issues/450) [reshape-01] lower descriptor-backed RESHAPE (hard)
- [lazy-fortran/ffc#451](https://github.com/lazy-fortran/ffc/issues/451) [formatted-read-core-01] lower scalar formatted input (hard)
- [lazy-fortran/ffc#453](https://github.com/lazy-fortran/ffc/issues/453) [intrinsic-dispatch-01] centralize scalar intrinsic calls (hard)
- [lazy-fortran/ffc#455](https://github.com/lazy-fortran/ffc/issues/455) [control-transfer-01] lower structured branch targets (hard)
- [lazy-fortran/ffc#456](https://github.com/lazy-fortran/ffc/issues/456) [entry-01] lower procedure ENTRY points (hard)
- [lazy-fortran/ffc#458](https://github.com/lazy-fortran/ffc/issues/458) [array-decl-core-01] migrate core array declarations (hard)
- [lazy-fortran/ffc#459](https://github.com/lazy-fortran/ffc/issues/459) [array-constructor-01] lower typed array constructors (hard)
- [lazy-fortran/ffc#460](https://github.com/lazy-fortran/ffc/issues/460) [namelist-write-01] write scalar NAMELIST groups (hard)
- [lazy-fortran/ffc#461](https://github.com/lazy-fortran/ffc/issues/461) [proc-pointer-core-01] complete scalar procedure-pointer state (hard)
- [lazy-fortran/ffc#462](https://github.com/lazy-fortran/ffc/issues/462) [defined-assign-core-01] recurse through defined assignment components (hard)
- [lazy-fortran/ffc#463](https://github.com/lazy-fortran/ffc/issues/463) [data-init-core-01] lower structured DATA initialization (hard)
- [lazy-fortran/ffc#465](https://github.com/lazy-fortran/ffc/issues/465) [defined-assign-array-01] lower elemental defined assignment over arrays (hard)
- [lazy-fortran/ffc#467](https://github.com/lazy-fortran/ffc/issues/467) [proc-dummy-core-01] lower scalar procedure dummy arguments (hard)

### Current valid-program regressions

- [lazy-fortran/ffc#353](https://github.com/lazy-fortran/ffc/issues/353) [alternate-return-01] register positional alternate-return slots (medium)
- [lazy-fortran/ffc#357](https://github.com/lazy-fortran/ffc/issues/357) [interface-body-01] preserve bodies after interface blocks (medium)
- [lazy-fortran/ffc#361](https://github.com/lazy-fortran/ffc/issues/361) [c-strpointer-01] lower C_F_STRPOINTER character views (medium)

### gfortran.dg rejection groups


### Remaining PR-sized feature slices

- [lazy-fortran/ffc#402](https://github.com/lazy-fortran/ffc/issues/402) [derived-charalloc-01] lower allocatable character components (hard)
- [lazy-fortran/ffc#403](https://github.com/lazy-fortran/ffc/issues/403) [finalize-01] invoke scalar final procedures once (hard)
- [lazy-fortran/ffc#405](https://github.com/lazy-fortran/ffc/issues/405) [proc-elemental-01] lower elemental procedures through array expressions (hard)
- [lazy-fortran/ffc#406](https://github.com/lazy-fortran/ffc/issues/406) [derived-arrayarg-01] pass arrays of allocatable derived values (hard)
- [lazy-fortran/ffc#407](https://github.com/lazy-fortran/ffc/issues/407) [proc-pointer-result-01] return data pointers from functions (medium)
- [lazy-fortran/ffc#408](https://github.com/lazy-fortran/ffc/issues/408) [proc-keyword-01] reorder keyword actual arguments by public dummy names (hard)
- [lazy-fortran/ffc#411](https://github.com/lazy-fortran/ffc/issues/411) [pdt-01] support constant integer kind and length type parameters (hard)
- [lazy-fortran/ffc#414](https://github.com/lazy-fortran/ffc/issues/414) [fmod2-02] serialize derived layouts and character metadata (hard)
- [lazy-fortran/ffc#415](https://github.com/lazy-fortran/ffc/issues/415) [fmod2-03] serialize rank-aware generic specifics (hard)
- [lazy-fortran/ffc#417](https://github.com/lazy-fortran/ffc/issues/417) [poly-02] track runtime type identity for class scalars (hard)
- [lazy-fortran/ffc#418](https://github.com/lazy-fortran/ffc/issues/418) [vector-subscript-02] lower vector-subscript assignment as alias-safe scatter (hard)
- [lazy-fortran/ffc#419](https://github.com/lazy-fortran/ffc/issues/419) [poly-03] lower runtime SELECT TYPE guards (hard)
- [lazy-fortran/ffc#420](https://github.com/lazy-fortran/ffc/issues/420) [poly-04] dispatch overridden type-bound procedures through vtables (hard)
- [lazy-fortran/ffc#421](https://github.com/lazy-fortran/ffc/issues/421) [poly-05] allocate class scalars from SOURCE (hard)
- [lazy-fortran/ffc#422](https://github.com/lazy-fortran/ffc/issues/422) [poly-06] pass and allocate polymorphic arrays (hard)
- [lazy-fortran/ffc#423](https://github.com/lazy-fortran/ffc/issues/423) [runtime-04] move scalar formatted output behind runtime calls (hard)
- [lazy-fortran/ffc#424](https://github.com/lazy-fortran/ffc/issues/424) [array-transform-01] lower four array transformation intrinsics (hard)
- [lazy-fortran/ffc#426](https://github.com/lazy-fortran/ffc/issues/426) [array-shift-01] lower CSHIFT and EOSHIFT (hard)
- [lazy-fortran/ffc#427](https://github.com/lazy-fortran/ffc/issues/427) [io-status-01] return stable IOSTAT and IOMSG values (hard)
- [lazy-fortran/ffc#428](https://github.com/lazy-fortran/ffc/issues/428) [runtime-05] provide descriptor-backed runtime allocation helpers (hard)
- [lazy-fortran/ffc#429](https://github.com/lazy-fortran/ffc/issues/429) [lf-infer-derived-01] infer Lazy Fortran derived values from constructors (hard)
- [lazy-fortran/ffc#432](https://github.com/lazy-fortran/ffc/issues/432) [norm2-01] lower NORM2 over descriptor arrays (medium)
- [lazy-fortran/ffc#433](https://github.com/lazy-fortran/ffc/issues/433) [lf-monomorph-fmod-01] stabilize cross-module Lazy specializations (hard)
- [lazy-fortran/ffc#434](https://github.com/lazy-fortran/ffc/issues/434) [formatted-read-al-01] read A and L fixed-width fields (medium)
- [lazy-fortran/ffc#435](https://github.com/lazy-fortran/ffc/issues/435) [transfer-02] lower sized and array-valued TRANSFER (hard)
- [lazy-fortran/ffc#436](https://github.com/lazy-fortran/ffc/issues/436) [namelist-read-01] read scalar NAMELIST groups (hard)
- [lazy-fortran/ffc#437](https://github.com/lazy-fortran/ffc/issues/437) [lf-monomorph-01] emit one-unit Lazy procedure specializations (hard)
- [lazy-fortran/ffc#438](https://github.com/lazy-fortran/ffc/issues/438) [lf-defaults-01] apply Lazy Fortran default type and intent policy (hard)

## FortFront open issues
- [lazy-fortran/fortfront#2882](https://github.com/lazy-fortran/fortfront/issues/2882) [reject-call-01] reject procedure-call signature mismatches (medium)
- [lazy-fortran/fortfront#2883](https://github.com/lazy-fortran/fortfront/issues/2883) [reject-interface-01] enforce explicit-interface declaration rules (medium)
- [lazy-fortran/fortfront#2887](https://github.com/lazy-fortran/fortfront/issues/2887) [reject-use-01] validate USE exports renames and operator imports (medium)
- [lazy-fortran/fortfront#2888](https://github.com/lazy-fortran/fortfront/issues/2888) [reject-scope-02] reject local and host-associated name collisions (medium)
- [lazy-fortran/fortfront#2896](https://github.com/lazy-fortran/fortfront/issues/2896) [reject-placement-01] reject constructs in forbidden program sections (medium)

## LIRIC open issues

- [krystophny/liric#523](https://github.com/krystophny/liric/issues/523) [dominance-01] preserve integer print definitions across serialization (hard)
- [krystophny/liric#525](https://github.com/krystophny/liric/issues/525) [llvm-compat-01] legalize legacy pointer IR without mutating modules (medium)
- [krystophny/liric#526](https://github.com/krystophny/liric/issues/526) [session-01] expose typed pointer metadata through the session API (hard)
- [krystophny/liric#527](https://github.com/krystophny/liric/issues/527) [session-02] version and verify the public session ABI (hard)

## fo open issues relevant to ffc

- [lazy-fortran/fo#56](https://github.com/lazy-fortran/fo/issues/56) [lsp-01] publish debounced FortFront diagnostics on didChange (hard)
- [lazy-fortran/fo#59](https://github.com/lazy-fortran/fo/issues/59) [lint-01] merge fluff JSON into fo lint --deep (medium)
- [lazy-fortran/fo#103](https://github.com/lazy-fortran/fo/issues/103) [fortfront-diag-01] map structured frontend diagnostics (medium)

fo #62 archives fortrun and is omitted from the ffc work order.

## fluff open issues relevant to ffc


fluff #245 is closed; its combined JSON output unblocks fo #59. fluff #77 is a general historical epic and is omitted. <!-- slop-ok: technical semicolon -->

## fx

No open fx issue currently blocks ffc. fx #31 is closed; fo #56 now depends on the FortFront diagnostic integration. <!-- slop-ok: technical semicolon -->

## standard references

The [Lazy Fortran standard repository](https://github.com/lazy-fortran/standard)
is the specification source. These open proposals are inputs, not compiler
assignments until accepted and mapped to an implementation issue. Strings,
ownership, runtime, arrays, reproducibility, modules, layout, unsafe interop,
and tensors can change the architectural contracts above; evaluate those before <!-- slop-ok: technical semicolon -->
implementation.

- [standard #734](https://github.com/lazy-fortran/standard/issues/734): systems-programming roadmap. <!-- slop-ok: issue title -->
- [standard #735](https://github.com/lazy-fortran/standard/issues/735): exact strings, views, and builders.
- [standard #736](https://github.com/lazy-fortran/standard/issues/736): generic high-performance containers.
- [standard #737](https://github.com/lazy-fortran/standard/issues/737): reusable runtime traits.
- [standard #738](https://github.com/lazy-fortran/standard/issues/738): algebraic variants and exhaustive selection.
- [standard #739](https://github.com/lazy-fortran/standard/issues/739): ownership, view lifetime, freezing, and thread safety.
- [standard #740](https://github.com/lazy-fortran/standard/issues/740): extractable Lazy runtime.
- [standard #741](https://github.com/lazy-fortran/standard/issues/741): fortran-lang/stdlib compatibility.
- [standard #742](https://github.com/lazy-fortran/standard/issues/742): derivation and metaprogramming hooks.
- [standard #743](https://github.com/lazy-fortran/standard/issues/743): OpenMP-aware runtime policy.
- [standard #744](https://github.com/lazy-fortran/standard/issues/744): compiled pattern library.
- [standard #745](https://github.com/lazy-fortran/standard/issues/745): array shape, rank, and broadcasting contracts.
- [standard #746](https://github.com/lazy-fortran/standard/issues/746): physical dimensions and index-space tags.
- [standard #747](https://github.com/lazy-fortran/standard/issues/747): effect system.
- [standard #748](https://github.com/lazy-fortran/standard/issues/748): deterministic reductions and numerical modes.
- [standard #749](https://github.com/lazy-fortran/standard/issues/749): memory layout, alignment, locality, and placement. <!-- slop-ok: issue title -->
- [standard #750](https://github.com/lazy-fortran/standard/issues/750): contracts and optimizer assumptions.
- [standard #751](https://github.com/lazy-fortran/standard/issues/751): automatic differentiation contracts.
- [standard #752](https://github.com/lazy-fortran/standard/issues/752): restricted compile-time execution and specialization.
- [standard #753](https://github.com/lazy-fortran/standard/issues/753): stable module interface signatures.
- [standard #754](https://github.com/lazy-fortran/standard/issues/754): scoped unsafe interop and raw operations.
- [standard #755](https://github.com/lazy-fortran/standard/issues/755): explicit tensor notation and lowering.

## Roadmap update protocol <!-- slop-ok: requested document type -->

The step is not complete until this file reflects it. Update `ROADMAP.md` in <!-- slop-ok: requested filename -->
the same session, before starting another issue, after every:

- merged or closed issue or pull request;
- pass, failure, XFAIL, XPASS, NOREF, SKIP, or denominator change; <!-- slop-ok: state vocabulary -->
- corpus revision, scope, ownership, dependency, ABI, or architecture change; <!-- slop-ok: roadmap metadata -->
- new or removed implementation issue; and
- change to the next executable chunk.

For each update:

1. Refresh repository revisions, active pull requests, and every linked issue's
   open or closed state.
2. Record focused red/green evidence and the full `fo` result. Regenerate
   `ffc/docs/PARITY_STATUS.md` and its dashboard when corpus state changes.
3. Move completed prerequisites out of the open indexes. Retain only completed
   milestones needed to explain the dependency graph.
4. Update the affected chunk, its preserved invariants, its next executable
   issue, and its completion gate.
5. Reconcile the complete ffc, FortFront, LIRIC, relevant fo/fluff/fx, and
   standard issue indexes. No open owner may disappear.
6. Run `$HOME/code/prompts/scripts/check-writing-slop.py ROADMAP.md`. <!-- slop-ok: requested filename -->

## Session start checklist

1. Read this file and perform the update protocol if external state changed.
2. Work from one atomic open issue in the earliest unblocked chunk.
3. Respect its dependency list and scope exclusions.
4. Use a clean worktree. Preserve user changes in primary checkouts.
5. Run focused red/green verification, then full `fo`.
6. For corpus work, run the exact named cases and diff PASS and XPASS sets.
7. Review, commit, push, open a pull request, wait for green CI, review again,
   and squash merge.
8. Apply the roadmap update protocol before the next step. <!-- slop-ok: requested document type -->
