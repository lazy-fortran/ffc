# ffc Development Plan

`ffc` is the compiler driver for Lazy Fortran and LFortran Infer-style
source. The active pipeline is:

```
Fortran / Lazy Fortran source
  -> FortFront typed AST + diagnostics
  -> ffc lowering + runtime ABI
  -> LIRIC C API (via ISO_C_BINDING)
  -> object file / executable
```

FortFront stays backend-neutral. `ffc` owns lowering, ABI decisions,
runtime calls, LIRIC bindings, and object/exe emission.

## Architecture decisions

- New compiler work goes through LIRIC's session C API.
- `ffc` does not add LLVM or MLIR bindings.
- A need for private FortFront AST layout is filed as a FortFront issue,
  not as an ffc workaround.

The retired MLIR/HLFIR experiment lives only in git history. Reference it
by commit hash if you need to look back, but do not revive it without an
explicit decision.

## Current status (2026-08-02)

- Main: `20398f7` (structured DO WHILE lowering, array-valued predicates, bare
  Lazy logical literals, scalar logical connectives, logical DOT_PRODUCT,
  scalar logical/integer casts, and logical array expressions in reductions and
  I/O, typed file-I/O size/stream transfer, logical-kind byte transfer, and
  logical literal KIND/STORAGE_SIZE inquiries, character-valued ERROR STOP,
  nested LOGICAL conversion-kind inquiries, allocatable logical NOT masks, and
  formatted character file writes, runtime rank-2 allocatable MATMUL and
  runtime array-expression reductions, `CPU_TIME` widening, and explicit-shape
  whole-array dummy aliasing, explicit-lower-bound assumed-shape descriptors,
  mixed-rank runtime section expressions, allocatable function-result MATMUL
  matrix-vector lowering, rank-2 automatic array-result materialisation, typed
  integer/real/double/complex/logical TRANSPOSE lowering, compile-time
  parameter TRANSPOSE initialization, mixed-kind integer MIN/MAX lowering, and
  legacy typed MIN aliases, and legacy typed MAX aliases,
  with
  sampled manifest dispositions through seed 1037). FortFront `d556f5b0`.
  LIRIC `5436e5c`.
- `fo build` passes for ffc 428/428 and FortFront 379/379 at those revisions.
- Repeated deterministic random subsets reached 900 files per suite with no
  unexpected `FAIL` or `XPASS` after exact manifest classification, including
  seeds 1035, 1036, and 1037. The formerly XFAIL `associate_18.f90` now
  passes after public-but-unsupported module procedures were preserved in
  `.fmod` exports. The next sample increase is deliberately deferred while
  owned XFAIL implementation work continues.
- The current owned XFAIL tranche is complete: `while_05.f90` and
  `do_while_1.f90` pass as ordinary no-manifest cases, and their XFAIL entries
  were removed. Focused independent regressions for character array results,
  array-expression materialisation, and the DO WHILE header all pass. Keep the
  random sample at 900 until the next owned XFAIL tranche is selected and
  reaches zero.
- The next owned tranche is complete as well: `boolean_assign_bare_true.lf`
  and `boolean_assign_bare_false.lf` pass in the FortFront-LF suite with
  `XFAIL=0`, `XPASS=0`, and `FAIL=0`. The shared literal classifier and value
  conversion now cover bare Lazy `true` and `false`, with independent runtime
  checks in `test_session_inferred_logical_compiler`.
- The scalar logical tranche is green too: `logical3.f90` passes against the
  gfortran behavioral oracle with `XFAIL=0`, `XPASS=0`, and `FAIL=0`. Shared
  lowering now handles `.xor.` and keeps `.eqv.` inversion operands distinct.
- The logical reduction tranche is green: `logical_dot_product.f90` passes in
  gfortran.dg with `XFAIL=0`, `XPASS=0`, and `FAIL=0`. Logical DOT_PRODUCT now
  lowers through the shared array element engine as `ANY(a .AND. b)`.
- The integer-to-logical scalar tranche is green: `logical4.f90` and
  `logical_casting_01.f90` pass against the gfortran behavioral oracle with
  `PASS=2`, `XFAIL=0`, `XPASS=0`, and `FAIL=0`. Shared lowering recognizes
  arithmetic integer expressions, converts nonzero values through an i1-to-i32
  zero-extension, and their XFAIL entries were removed only after the named
  no-manifest run passed.
- The reverse scalar/array cast is green too: `logical_to_integer_cast.f90`
  passes against the same behavioral oracle with `PASS=1`, `XFAIL=0`, `XPASS=0`,
  and `FAIL=0`. Integer lowering now accepts logical literals and scalar
  logical identifiers, while whole logical arrays reuse their i32 storage.
- The logical array-expression tranche is green:
  `logical_arrays_logical_binop_01.f90` passes in the LFortran suite with
  `PASS=1`, `XFAIL=0`, `XPASS=0`, and `FAIL=0`. Shared lowering now handles
  logical connectives, comparisons, scalar logical-function broadcasts,
  constructor-shaped output, and nested masks in reductions. FortFront
  `cc39c3bc` makes I/O argument parsing consume full logical expressions. The
  XFAIL entry was removed after the named behavioral run. The sample count
  remains 900.
- The typed file-I/O tranche is green: `logical_kind_01.f90` passes in the
  LFortran suite with `PASS=1`, `XFAIL=0`, `XPASS=0`, and `FAIL=0`. `INQUIRE`
  `SIZE=` now reports file and connected-unit byte counts, and stream writes
  preserve scalar integer and logical kind widths. The independent inquiry
  compiler test and runtime-link contract test also pass. The sample count
  remains 900.
- The logical kind transfer tranche is green: `logical_kind_02.f90` passes in
  the LFortran suite with `PASS=1`, `XFAIL=0`, `XPASS=0`, and `FAIL=0`.
  `TRANSFER` now preserves canonical logical truth and zero padding when a
  scalar logical kind is transferred into a same-width integer byte array.
  The independent logical-transfer compiler test also passes. The sample
  count remains 900.
- The logical kind inquiry/literal tranche is green: `logical_kind_04.f90` and
  `logical_kind_05.f90` pass in the LFortran suite with `PASS=2`, `XFAIL=0`, <!-- slop-ok: technical status counts -->
  `XPASS=0`, and `FAIL=0`. ffc lowers logical `KIND` and `STORAGE_SIZE`, while
  FortFront preserves numeric, named, and mixed-case logical kind suffixes.
  Independent ffc and FortFront regressions pass; the sample count remains
  900. The character-valued `ERROR STOP` tranche is green too:
  `logical_kind_06.f90` passes with `PASS=1`, `XFAIL=0`, `XPASS=0`, and
  `FAIL=0`; `test_session_stop_message_compiler` independently verifies the
  dynamic message and exit status. The nested `LOGICAL` conversion-kind
  tranche is green too: `logical_kind_07.f90` passes with `PASS=1`, `XFAIL=0`,
  `XPASS=0`, and `FAIL=0`; `test_session_inquiry_fold_compiler` independently
  verifies default and explicit kinds. The allocatable logical-mask tranche is
  green too: `logical_not_01.f90` passes in the LFortran suite with `PASS=1`,
  `XFAIL=0`, `XPASS=0`, and `FAIL=0`; shared whole-array lowering now covers
  allocatable comparison masks, array printing, and `ANY(.NOT. logical-array)`.
  The independent whole-array compiler test passes, and the XFAIL entry was
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
- The `modules_24.f90` class/derived-pointer case (#417) is green: exact
  no-manifest and normal-manifest runs both report `PASS=1`, `XFAIL=0`,
  `XPASS=0`, and `FAIL=0`. Module type collection now records scalar derived
  pointer components as opaque two-slot addresses, initializes them null, and
  routes compatible class-pointer associations through the existing derived
  pointer path. The XFAIL row was removed only after both bounded runs and the
  existing independent derived-pointer regression passed.
- The `modules_25.f90` class/runtime-character tranche (#350/#417) is green:
  exact named runs of `modules_25.f90`, `modules_25_module.f90`, and
  `modules_25_module1.f90` report `PASS=3`, `XFAIL=0`, `XPASS=0`, and `FAIL=0`
  in both normal and no-manifest modes (`NOREF=3` for compile-only module
  units). Inherited derived components now preserve pointer/descriptor flags,
  named class dummies bind through the polymorphic descriptor, and module
  companions are supplied through the explicit extra-source manifest. The
  XFAIL rows were removed only after the independent gfortran/oracle gauntlet
  passed in both modes.
- Architecture migration has its first verified seams: diagnostics and
  constant-folding are real module/submodule units, the scalar-kind helpers
  and scalar-expression engine are real module/submodule units, FMod token
  helpers are a real module, literal-utils is a real submodule, and the unused
  `session_program_lowering_text.inc` fragment is gone, and declaration-conflict
  and generic-rejection checks are now real submodules with explicit build-order
  units. A clean sequential `fo build` and focused behavior tests pass. The
  remaining rejection and other host-coupled fragments stay live until their
  dependencies are extracted safely.
- The `modules_15b.f90` module-interface companion compiles with ffc and
  gfortran as an explicit `NOREF=compile-only` case. Its runnable companion is
  now covered by the verified BIND(C) ABI tranche above.
- No whole-corpus run has been performed under the bounded-sampling policy.
  `XFAIL`, `NOREF`, and `SKIP` are classifications, not behavioral passes.

All corpus work stays bounded: use deterministic random subsets, never the
whole corpus, and increase the sample only after repeated 100%-clean subsets.
Finish the owned XFAIL-first tranche at zero before moving to another corpus
area or increasing the count. Keep compiler jobs sequential and bounded to
avoid OOM.

## XFAIL-zero work gate

XFAIL work always comes before corpus expansion. Each work cycle selects an
owned XFAIL tranche, fixes the implementation or its independent behavioral
oracle, and removes the XFAIL only when the case passes. We do not move to a
different corpus area, broaden the suite, or increase the random-sample count
until the current in-scope XFAIL tranche is at zero. The final conformance gate
requires zero in-scope XFAILs across every declared suite. Classification is
not a substitute for fixing the behavior.

## Active task list (2026-08-02)

1. Completed: promote `module_array_init.f90`,
   `module_function_with_nopass.f90`, `module_function_without_nopass.f90`,
   `modules_03.f90`, `modules_05.f90`, `modules_08.f90`, `modules_09.f90`,
   and `modules_20.f90` only after independent gfortran behavioral oracles
   and bounded named runs reached `XFAIL=0`, `XPASS=0`, and `FAIL=0`.
2. Completed: promote the BIND(C) tranche consisting of `modules_15.f90`,
   `modules_18.f90`/`modules_18b.f90`, and `modules_19.f90`/`modules_19b.f90`
   only after the exact C-plus-gfortran oracle and normal-manifest run were
   both green. The `modules_22.f90`/`modules_22_module.f90` (#584) pair is now
   green; the next XFAIL-first target is `modules_26.f90` (#376), which remains
   in the module/runtime tranche.
3. Continue replacing the remaining textual `.inc` fragments in the lowerer with real
   Fortran modules/submodules in dependency order. The first verified seams
   are diagnostics, constant folding, scalar-kind/scalar-expression lowering,
   and FMod token support; next extract literal-utils support, then the
   host-coupled rejection checks. Remove each include only after a sequential
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

## Architecture-first blocker order

1. Binding identity and scope ownership: ffc #584. FortFront #2883, #2924,
   and #2951.
2. Canonical descriptor and storage representation: ffc #337, #338, #339,
   #348, and #643.
3. Typed dispatch and runtime identity: ffc #419, #422, #437, #461, and #467.
4. Shared control, constructor, assignment, and I/O engines: ffc #345, #449,
   #455, #459, #460, and #462.
5. Performance and corpus safety: ffc #478, #531, #576, and #663.

Every issue must preserve its stated invariant, use the shared representation,
include positive and negative behavioral cases, and run its focused `fo test`
plus `fo`. A manifest classification may record a known gap. It cannot replace
the implementation or turn wrong code into `XFAIL`.

Architecture migration map:

- Binding identity: consume one FortFront binding graph in
  `src/ffc_fortfront_queries.f90`, `src/session_symbol_table.f90`, the
  ASSOCIATE/interface collectors, and module/submodule lowering. Text-name
  lookup and inferred fallback must not remain as a second implementation.
- Descriptor/storage ABI: migrate the array, section, pointer, allocatable,
  assumed-shape, character-array, and polymorphic paths through
  `src/ffc_array_descriptor.f90`, the descriptor helpers, and
  `runtime/ffc_runtime.c`. The canonical layout and ownership/view lifetime
  rules in `docs/ARRAY_DESCRIPTOR_ABI.md` are the gate.
- Module and callable identity: keep `.fmod` authoritative through
  `src/ffc_module_artefact.f90`, `src/session_program_lowering_fmod.inc`,
  `src/session_program_lowering_submodules.inc`, procedure-dummy,
  polymorphic, and Lazy-specialization lowering. LIRIC changes only when the
  public generated-call ABI must change.
- Shared engines: route array expressions, reductions, constructors,
  defined assignment, WHERE/FORALL, control flow, and I/O through one typed
  representation; delete superseded handlers after behavioral proof.

## Path to standard Fortran conformance

The target is the Fortran standard through F2023, minus the parallel and vendor
features excluded below. The standard defines what must work. The corpora only
measure how far along we are. A corpus file is evidence, never the goal, so a
case that tests another compiler's extension surface is excluded rather than
chased.

Excluded, and never counted in any denominator:

- coarrays, images, teams, events, and collective subroutines
- OpenMP, OpenACC, and MPI
- GPU and device backends
- vendor extensions outside the standard
- features deleted from the standard
- compiler-option and DejaGNU-harness behavior the runner does not model

The checked-in parity snapshot in `test/conformance/parity_dashboard.tsv` is a
historical, provenance-verified full-run artifact. Its raw file ratios are not
an implementation percentage. Current progress is measured with stratified
random subsets. A sampled report is an estimate and must not replace the
snapshot. Regenerate the snapshot only with a provenance-verified full-corpus
run after the bounded work is complete.

Those denominators are raw file counts, not the conformance denominator. 241
cases are still `NOREF`, meaning undefined output, missing linkage, or a
harness contract the runner does not model. Until #430 classifies them, the
number that 100% is 100% *of* is not yet known. Finish that classification
before quoting a conformance percentage.

The `E1` through `E10` epics (#262 through #271), the `#272` compliance
umbrella, and the LIRIC coordination issue `krystophny/liric#520` are all
closed. They were split into the atomic issues that now carry the work. Do not
cite them as the live plan.

The live work order is the chunk sequence in the workspace plan: freeze the
public compiler graph, freeze the canonical descriptor/storage contract,
stabilize module artifacts and runtime/backend ABIs, complete typed dispatch,
route arrays and I/O through shared engines, then close corpus breadth. Scalar
typed-lowering infrastructure may proceed while the binding graph is being
completed, but descriptor-dependent dispatch and corpus work do not bypass
these gates. Each chunk names its own atomic issues.

Neither external corpus is a 100% target as a whole. `gfortran.dg` contains
error-detection, deprecated, and vendor-extension tests. The `lfortran`
integration suite exercises that compiler's own extension surface. Gate only
the runnable, standard-conforming subset of each and document the exclusions in
`docs/CONFORMANCE.md`. The two FortFront corpora are maintained in-tree and are
100% targets once their `NOREF` cases are classified.

F2023 is part of the target, and the delta it adds over F2018 is unscoped. The
`[ffc-f2023-*]` trackers (#243 through #255) were closed after being split into
the current issue set, but that split covered F95-through-F2018 language
coverage. The syntax and intrinsics F2023 itself introduced were never
enumerated. #473 owns auditing that delta so "newest standard" does not quietly
mean F2018.

## Shipped baseline: direct LIRIC session backend

Covered features and the public claim live in
`docs/SUPPORT_CONTRACT.md`. Roughly:

- main program, scalar integer / real / logical, fixed-length and
  deferred-length character;
- arithmetic, comparisons, logical conditions;
- block `if` with PHI merges; counted `do` with literal positive and
  negative step; `SELECT CASE` with single- and multi-arm terminating
  bodies (including multi-label arms) and `case default`;
- contained integer / real / logical functions and subroutines, including
  early `return`;
- fixed-size 1-D integer arrays and rank-2 integer arrays with scalar
  element access, array sections with compile-time integer bounds as
  rvalues, whole-array copy, elemental `+`/`-`/`*`, and the array
  intrinsics `size`, `shape`, `sum`, `product`, `maxval`, and `minval`, plus simple
  derived types with scalar integer components;
- minimal `print *, expr`, compound formatted `print fmt, items` with literal
  `I`, `X`, and `F` descriptors, `stop <expr>`, `abs` / `min` / `max` / `mod`
  and integer-to-real `real()`;
- compile-time `//` folding for character literal chains;
- CLI: `-o`, `-c`, `-I <dir>` accepted (`-I` not yet consumed).

This surface is the baseline the compliance epics build on.

## Runtime, ABI, and conformance

The current ABI is documented in `docs/RUNTIME_ABI.md`. Conformance against
external corpora is documented in `docs/CONFORMANCE.md`. Each new feature must
update both documents and add executable tests in the same change.

The Fortran I/O and intrinsics runtime lives in `ffc/runtime/` (the local
`libgfortran` equivalent), linked through LIRIC's `lr_session_set_runtime_archive`.
It carries a stable C ABI so it can split into its own repo once it stabilizes.
LIRIC stays a backend-neutral codegen layer. No Fortran-language semantics land
there. This work is E7 (#268) and E8 (#269).

## FortFront boundary

`ffc` reaches into FortFront's arena (`select type (node => ...)`) for most
lowering. Each new lowering function should prefer a public FortFront query
over reaching into arena internals. A need for private FortFront AST layout is
filed as a FortFront issue, not an ffc workaround.

## Verification

```bash
export LIBRARY_PATH=<liric-build>   # so the LIRIC static library is linkable
fo                                  # static analysis, build, tests, lint, fmt
bash scripts/conformance_check.sh --no-build --sample 900 --seed <seed>
```

Use `fo` for every build and test loop. Call `fpm` directly only to isolate one
named test or to diagnose a `fo` failure. CI runs the same workflow on every
push and pull request. Increase the sample count only after repeated clean
subsets, and do not run the whole corpus for routine progress checks. Run `fo`
before pushing.
