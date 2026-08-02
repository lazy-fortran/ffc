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

- Main: `41cbc2d` (structured DO WHILE lowering, array-valued predicates, bare
  Lazy logical literals, scalar logical connectives, logical DOT_PRODUCT,
  scalar logical/integer casts, and logical array expressions in reductions and
  I/O, typed file-I/O size/stream transfer, logical-kind byte transfer, and
  logical literal KIND/STORAGE_SIZE inquiries, character-valued ERROR STOP,
  nested LOGICAL conversion-kind inquiries, allocatable logical NOT masks, and
  formatted character file writes, runtime rank-2 allocatable MATMUL and
  runtime array-expression reductions, `CPU_TIME` widening, and explicit-shape
  whole-array dummy aliasing, explicit-lower-bound assumed-shape descriptors,
  mixed-rank runtime section expressions, allocatable function-result MATMUL
  matrix-vector lowering, with sampled manifest dispositions through seed
  1037). FortFront `4948ec2a`. LIRIC `5436e5c`.
- `fo build` passes for ffc 405/405 and FortFront 379/379 at those revisions.
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
  `XPASS=0`, and `FAIL=0`; the combined `matmul_01`-`matmul_05` run is
  `PASS=5` with no unexpected result. The next XFAIL-first tranche is
  `matmul_06.f90`; keep the sample count at 900.
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
public compiler graph, centralize typed lowering, stabilize the
descriptor/runtime/backend ABIs, make module artifacts authoritative, route
arrays and I/O through shared engines, then close corpus breadth. Each chunk
names its own atomic issues.

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
