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

## Current status (2026-08-03)

- Main: `96cfefb` (ALLOCATED keyword arguments and scalar `DATA p / NULL() /`
  pointer disassociation are promoted, on top of the incomplete-expression
  diagnostic, on top of
  the comparison typechecking submodule, on top of
  the FLOOR optional `KIND=8` lowering and public-session
  f64-to-i64 conversion are now green, on top of the storage-rejection checks
  as a real submodule, on top of
  the bare-character SELECT CASE fix and the contained f64 calls in f32
  expressions, on top of the lazy
  whole-array constructor reallocation fix and the
  mixed-kind unary real promotion and the
  ANY DIM assignment promotion and the
  array-constructor 02/03 promotion, the
  complex ABS/IEEE NaN, allocatable inquiry, and allocatable complex-array
  promotions, and the modules36 fixed-character-array promotion on top of the
  enum-lowering module extraction, the #584 assumed-size-array
  FAIL fix, modules34/modules35 XFAIL closure, schema-10 `.fmod`
  compatibility, and strict sampled conformance gating, on top of the
  modules31/modules33 separate-compilation support:
  one-specific and multi-specific generic type-bound bindings, serialized
  type-bound bindings in `.fmod` schema 11,
  imported vtable ownership and module-mangled type-bound calls, and rank-1
  fixed-length allocatable character-component declaration/layout metadata;
  runtime character-array element access remains intentionally unsupported;
  structured DO WHILE lowering, array-valued predicates, bare
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
  legacy typed MIN aliases, legacy typed MAX aliases, and mixed opaque module
  dummy metadata, and corrected constant HUGE array-bound classification,
  with
  sampled manifest dispositions through seed 1037). FortFront `5ff07184`.
  LIRIC `5436e5c`.
- `fo build` passes for ffc 450/450 and FortFront 379/379 at those revisions.
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
- The exact XFAIL-first tranche of `abs_04.f90`, `abs_06.f90`,
  `allocated_01.f90`, `allocated_04.f90`, `allocated_05.f90`,
  `array_constructor_02.f90`, and `array_constructor_03.f90` is now fully
  promoted. Normal and XFAIL-disabled LFortran runs both report
  `PASS=7`, `XFAIL=0`, `XPASS=0`, and `FAIL=0`; independent gfortran output
  checks and the focused complex/reduction compiler tests agree. FortFront
  `9ff6605e` supplies the recursive nested-array-postfix parser fix used by
  the nested ABS case.
- Luna also completed and promoted `any_01.f90`: normal and XFAIL-disabled
  exact runs both report `PASS=1`, `XFAIL=0`, `XPASS=0`, and `FAIL=0`, with the
  independent gfortran output oracle and `test_session_array_mask_reduction_compiler`
  agreeing. The fix covers `ANY(..., DIM)` assignment into assumed-shape
  runtime arrays.
- Luna then promoted `array_op_03.f90`: normal and XFAIL-disabled exact runs
  both report `PASS=1`, `XFAIL=0`, `XPASS=0`, and `FAIL=0`, with the focused
  scalar-expression test and independent gfortran oracle green. Mixed-kind
  f64 operands now lower at f64 before safe conversion into an f32 context.
- The two red `fortfront-lf` sample cases from seed 1038 are now green:
  `test_209_all.lf` and `test_209_complex.lf` pass in normal and XFAIL-disabled
  exact runs with `PASS=2`, `XFAIL=0`, `XPASS=0`, and `FAIL=0`. The independent
  gfortran oracle and `test_session_allocatable_constructor_compiler` agree;
  whole-array constructor operands are materialized before old allocation is
  released.
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
- The next bounded probe remains red and did not alter manifests:
  `intrinsics_114.f90`/`intrinsics_115.f90` still fail XFAIL-disabled with
  `ffc direct-session lowering only supports integer expressions`, while
  `issue_1771_module_parameter_types.f90` still fails XFAIL-disabled with
  `mismatched scalar kind in argument to square`. Keep both rows in the queue;
  do not count the normal XFAIL-wrapped runs as passes.
- The following disjoint probe also remained red: `arrays_02_size.f90`
  still fails during ffc compilation; `issue_2495_data_null_intrinsic.f90`
  reaches an ffc lowering failure (`data-stmt-object 'ptr2' has the POINTER
  attribute`) while FortFront's focused parser test passes; and the attempted
  `reject_const_init.inc` migration builds but fails its independent rejection
  oracle because invalid input compiles and exits zero. None was integrated or
  promoted.
- The high-impact follow-up remains explicitly blocked: `array_section_01.f90`
  still emits malformed LIR (`instruction type missing`); `derived_types_121.f90`
  still reaches `direct LIRIC session cannot pass this scalar argument`; and
  FortFront-LF `issue_1968_lazy_function_result.lf` still leaves an invalid
  inferred dimension index. These attempts made no manifest changes or main
  commits; keep them ahead of sample expansion.
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
- The `modules_26.f90` interface-procedure/runtime-archive case (#376) is
  green: exact normal-manifest and no-manifest runs both report `PASS=1`,
  `XFAIL=0`, `XPASS=0`, and `FAIL=0`. Runtime-archive loading and an
  independent gfortran comparison also pass; interface dummy extents such as
  `real :: x(n)` now remain runtime bounds instead of being looked up as named
  compile-time parameters. The XFAIL row was removed only after all bounded
  checks passed.
- The `modules_27_module2.f90` generic module-registration case (#457) is
  green: exact normal-manifest and no-manifest runs both report `PASS=1`,
  `XFAIL=0`, `XPASS=0`, and `FAIL=0` (`NOREF=1` for the module-only unit).
  Complex pointer dummy registration now follows the resolved declaration
  path, and an independent gfortran compile/run oracle passes. The XFAIL row
  was removed only after both bounded checks passed.
- The focused array-shape rejection regression is green again: the accepted
  masked `FORALL` case with `vec(j)=real(j)` now passes after guarding the
  scalar-conversion lookup from invalid index-0 access. The focused compiler
  test and independent gfortran oracle both pass.
- The modules28 separate-compilation family (#328/#447) is green: exact named
  runs of `modules_28.f90`, `modules_28_module1.f90`, and
  `modules_28_module2.f90` report `PASS=3`, `XFAIL=0`, `XPASS=0`, and `FAIL=0`
  in both normal and no-manifest modes (`NOREF=2` for the module-only units).
  Imported derived types are registered before dependent layouts, repeated
  component metadata is rebuilt safely, and the independent gfortran program
  oracle passes. All three XFAIL rows were removed only after both bounded runs.
- Architecture migration has its first verified seams: diagnostics and
  constant-folding are real module/submodule units, the scalar-kind helpers
  and scalar-expression engine are real module/submodule units, FMod token
  helpers are a real module, literal-utils is a real submodule, and the unused
  `session_program_lowering_text.inc` fragment is gone, and declaration-conflict
  and generic-rejection checks are now real submodules with explicit build-order
  units; array-constructor, purity, and pointer rejection are now real submodules with
  explicit build-order units as well. A clean sequential `fo build` and focused behavior
  tests pass. The
  remaining rejection and other host-coupled fragments stay live until their
  dependencies are extracted safely.
- The `modules_15b.f90` module-interface companion compiles with ffc and
  gfortran as an explicit `NOREF=compile-only` case. Its runnable companion is
  now covered by the verified BIND(C) ABI tranche above.
- The modules29 separate-compilation family is green: exact named runs of
  `modules_29.f90`, `modules_29_module2.f90`, and `modules_29_module3.f90`
  report `PASS=3`, `XFAIL=0`, `XPASS=0`, and `FAIL=0` in both normal and
  no-XFAIL modes (`NOREF=2` for the two module-only companions). The
  independent gfortran module-chain compile/link/run oracle passes, and a
  bounded unit sample at seed 1729 passed `10/10`. FFC now exports direct USE
  dependencies recursively and preserves opaque public subroutine interfaces
  in `.fmod`; FortFront `9ff6605e` correctly treats `error` as a contextual
  identifier. The three stale modules29 XFAIL rows were removed only after
  these checks.
- The modules30 family is green: exact normal and XFAIL-disabled runs of
  `modules_30.f90` and `modules_30_module2.f90` report `PASS=2`, `XFAIL=0`,
  `XPASS=0`, and `FAIL=0` (`NOREF=1` for the module-only companion). The
  independent gfortran four-module-chain compile/link/run oracle passes.
  FFC now preserves per-dummy kinds in opaque public procedure interfaces,
  keeping supported character dummies callable while unsupported derived
  dummies retain the opaque path. The two XFAIL rows were removed only after
  these checks.
- The modules31 separate-compilation family is green: exact named normal and
  XFAIL-disabled runs of `modules_31.f90`, `modules_31_module1.f90`, and
  `modules_31_module2.f90` report `PASS=3`, `XFAIL=0`, `XPASS=0`, and `FAIL=0`
  (`NOREF=2` for the two module-only companions). The independent ffc object
  compile/link/run chain and the equivalent gfortran chain both print
  `running modules_31 program`. The three stale XFAIL rows were removed only
  after the receiver-slot regression and focused ABI tests were green. The
  next task was the modules33 sibling closure. That four-source family is now
  green: normal and XFAIL-disabled exact runs report `PASS=4`, `XFAIL=0`,
  `XPASS=0`, and `FAIL=0`; the independent gfortran module-chain oracle and
  complete ffc object/link/run chain both print `running modules_33 program`.
  FFC now preserves multi-specific type-bound generic metadata through
  schema-11 `.fmod` files, resolves imported generic calls, and handles scalar
  nested receivers without contaminating later derived layouts. The positive
  direct-session generic dispatch regression and the batched five-target
  focused test pass. The modules34 XFAIL-first tranche is now green: exact
  normal and XFAIL-disabled runs report `PASS=5`, `XFAIL=0`, `XPASS=0`,
  `FAIL=0`, and `NOREF=4`; independent ffc and gfortran module-chain
  compile/link/run oracles both print `running modules_34 program`. FFC now
  re-exports public derived types imported through `USE`, and both stale
  modules34 XFAIL rows were removed only after the named behavioral evidence.
  The modules35 XFAIL is green too: `modules_35.f90` reports `PASS=1`,
  `XFAIL=0`, `XPASS=0`, and `FAIL=0` against the gfortran oracle. The fix
  handles character allocatable-array descriptor passing, bounded rank-1 slot
  copies, and zero-length constructor assignment. Schema-10 `.fmod` reads are
  covered by a literal binding fixture while writers remain on schema 11.
  `sync_all_01.f90` and `sync_memory_01.f90` are explicitly classified as
  out-of-scope coarray/image-control cases.
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
- No whole-corpus run has been performed under the bounded-sampling policy.
  `XFAIL`, `NOREF`, and `SKIP` are classifications, not behavioral passes.

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
regression and prevents spending a full build on unrelated work. After the
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
   files; that bypasses the dependency setup and creates a false failure.
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
   separate worktrees do not make RAM free. Merge one green patch at a time,
   rebase it on `main`, and rerun the focused build/check before pushing.

The efficient order is therefore: one baseline, one implementation build, one
focused test, two exact corpus checks, then one or more bounded random samples.
The only safe parallel work is disjoint analysis or code in separate
worktrees; heavy builds and gauntlets remain sequential because worktrees do
not reduce RAM use.

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
   both green. The `modules_22.f90`/`modules_22_module.f90` (#584) pair,
   `modules_24.f90` (#417), the three-file `modules_25.f90` class/runtime-
   character tranche (#350/#417), and `modules_26.f90` (#376) are now green
   after bounded normal and no-manifest runs. `modules_27_module2.f90` (#457)
   and the modules28 family (`modules_28.f90`, `modules_28_module1.f90`, and
   `modules_28_module2.f90`) are now green after bounded normal and
   no-manifest runs. The modules29 family (`modules_29.f90`,
   `modules_29_module2.f90`, and `modules_29_module3.f90`) is now green after
   the exact normal/no-XFAIL checks and independent gfortran module-chain
   oracle. Its stale XFAIL rows are removed. The modules30 family
   (`modules_30.f90` and `modules_30_module2.f90`) is also green after exact
   normal/no-XFAIL checks and the independent gfortran module-chain oracle;
   its two XFAIL rows are removed. The modules31 and modules33 families are
   complete as recorded above. The modules34 and modules35 XFAIL tranches are
   complete. The #584 assumed-size FAIL closure is green on its bounded cases.
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
   introduced no new include, and enum lowering is now a real submodule.
   Remove each remaining include only after a sequential
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
include positive and negative behavioral cases, and run its focused
`FO_JOBS=1 fo test` plus bounded exact conformance. A full `fo` workflow is a
delivery-boundary check. A manifest classification may record a known gap; it
cannot replace the implementation or turn wrong code into `XFAIL`.

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
export FO_JOBS=1                    # avoid parallel compiler OOM
fo build                            # once per code change
fo test <focused-target>            # smallest relevant unit/regression target
bash scripts/conformance_check.sh --no-build --suite <suite> \
  --files-from <tranche-list> --ref-cache <private-ref-cache>
```

Use `fo build` once and reuse the binary across the named checks. Call `fpm`
directly only to isolate one named test or diagnose a `fo` failure. Use
`conformance_check.sh --sample N --seed S` only after the active XFAIL tranche
is zero; require several 100%-clean seeds before increasing `N`. CI runs the
same bounded workflow on every push and pull request. A full `fo` workflow is a
delivery-boundary check, not permission to run the external corpus wholesale.
