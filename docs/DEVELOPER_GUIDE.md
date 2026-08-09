# ffc Developer Guide

## Architecture

`ffc` connects FortFront to LIRIC:

```text
source -> FortFront typed AST -> ffc lowering and ABI -> LIRIC C API -> executable
```

FortFront stays backend-neutral. `ffc` owns lowering, runtime calls, ABI
decisions, and behavioural executable tests.

## Build

The default fpm library source is `src/`. The retired MLIR/HLFIR
experiment lives only in git history.

Use `fo` revision `32ef96d` or newer so `SUBMODULE` parent identifiers become
real build-graph edges. No filename ordering or `_order.f90` helper modules are
part of the source contract.

```bash
LIBRARY_PATH=/path/to/liric/build fo build
LIBRARY_PATH=/path/to/liric/build fo test
```

## Development rules

- Add new executable behaviour to the direct LIRIC session path; keep the
  CLI on `session_program_lowering`.
- Keep compiler clients on the two-procedure `session_program_lowering`
  facade. Lowering implementation units belong under
  `session_program_lowering_impl` and expose only the procedures their
  descendants require.
- Extract growing implementation units into modules or submodules with
  explicit interfaces. Let `fo` derive submodule ancestry from the source; do
  not add ordering shims or production `include` fragments.
- Keep constant-initialization validation in
  `session_program_lowering_reject_const_init.f90`. It is an immutable-AST
  descendant service with explicit interfaces; changes require both accepted
  and rejected executable/compiler oracles in
  `test_session_reject_const_01_compiler`.
- Add focused behavioural tests under `test/`. Each file is a standalone
  `program test_*` picked up by fpm auto-discovery.
- Treat a need for private FortFront AST layout as a FortFront API issue
  (see #58 / #173) rather than an `ffc` workaround.
- Before claiming support for a feature, update
  `docs/SUPPORT_CONTRACT.md`.
- If a feature changes calling convention, storage, or runtime calls,
  update `docs/RUNTIME_ABI.md` in the same change.
- Never `git add .` or `git add -A`. Stage paths explicitly.

## Feature order

The supported surface is in `docs/SUPPORT_CONTRACT.md`. Broadly the order
in which features have been added (and the order new slices should
follow) is:

1. Empty programs and integer `stop`.
2. Scalar integer declarations and assignments.
3. Integer arithmetic and comparisons.
4. Minimal `print *, expr`.
5. Block `if`, fallthrough integer merges, counted `do`.
6. Real and logical scalars; minimal character output.
7. Contained integer / real / logical functions and subroutines.
8. Fixed-size 1-D integer arrays.
9. Simple derived types with scalar integer components.
10. Deferred-length character (assignment, concatenation, self-aliasing).
11. `SELECT CASE` with terminating arms (single, multi, multi-label).
12. Early `return` inside contained subroutines and functions.
13. CLI `-I <dir>` accepted (storage only; consumption is future work).

The remaining work is the self-hosting tracker (#167) plus open
issues for individual slices.

## Verification

Use the repo-declared `fo` targets. Do not invoke build-tree binaries directly.

```bash
LIBRARY_PATH=/path/to/liric/build fo test          # full suite
LIBRARY_PATH=/path/to/liric/build fo test test_session_empty_program_compiler
```

### NVHPC 26.5 submodule check

The narrow-integer memory bindings have a focused public-API oracle in
`test/test_liric_memory_submodule_api.f90`. It creates a LIRIC session and
checks the i8/i16 operand payloads and type handles, so a compile-only check
cannot hide an ABI mismatch. For the NVHPC lane, compile the memory parent and
its three submodules into a fresh module directory before running the oracle;
do not reuse GNU `.mod` or `.smod` files. NVHPC 26.5 emits a warning for the
long submodule name, but the compile and API executable must both return zero.
The existing `test_session_integer_kind_i8_i16_compiler` adds an independent
gfortran differential check. A timeout while compiling the large
`session_program_lowering` unit is a toolchain-duration observation, not a
passing full-lane result.

### GNU runtime-length character result check

The smallest reproducer for the bounded runtime-character-result ownership
slice is already present in
test/test_session_character_function_result_compiler.f90: a contained make(k)
returns character(len=k) and the caller assigns r = make(4). Before the fix,
GNU main 3a6cc46 reproduced corrupted ffc output instead of gfortran's ZZZZ;
the exact captured run is
/var/tmp/ert/ffc-current-test_session_character_function_result_compiler.log.
That log also contains Valgrind invalid reads from the same ownership error.

The compiler path must recognize a contained character result as a descriptor
transfer before the generic character-expression classifier. The focused
behavioral oracle is test/test_session_runtime_character_result_compiler.f90:
it asserts the returned length and bytes in the executable, explicitly
deallocates the deferred result, and independently runs the executable under
Valgrind. Verify it with a distinct cache:

~~~bash
LIBRARY_PATH=/mnt/storage/code/lazy-fortran/liric/build \
FO_CACHE_DIR=/var/tmp/ert/ffc-runtime-char-focused-cache \
fo test test_session_runtime_character_result_compiler
~~~

The 2026-08-09 GNU result was a complete 456/456-unit build and 1/1 focused
test pass. At that checkpoint the larger character-result test still had
runtime-length-expression, nested-concatenation, and print-temporary
failures. The next exact reproduction on d74a188 was:

~~~text
LIBRARY_PATH=/mnt/storage/code/lazy-fortran/liric/build \
FO_CACHE_DIR=/var/tmp/ert/ffc-next-char-result-cache \
fo test test_session_character_function_result_compiler
~~~

It built 456/456 and reported `len_expr_of_dummy` as `Hello, A` instead of
gfortran's `Hello, Ada`, `fixed_dest_truncates` as `Hello, B` instead of
`Hello, Bob`, and garbage bytes for `nested_concat_result` instead of
`abcde`; the print-temporary leak oracle also reported invalid reads and
uninitialized bytes. `is_char_expr_call` recognizes `//`, so its generic
deferred-character branch was selected before deferred concatenation could
compute the result length and transfer ownership. The fix orders the
explicit concatenation dispatch first while preserving contained-result
transfer first.

The repaired aggregate passed 456/456 build and 1/1 test using
`/var/tmp/ert/ffc-order-isolated-cache`. The added
`test_session_runtime_length_expression_character_result_compiler` is the
independent API/compiler oracle: its executable asserts `LEN(r)==10` and the
exact returned bytes `Hello, Ada`, deallocates the result, and is separately
checked with Valgrind by `expect_no_leaks`. Its focused run passed 1/1 with
`/var/tmp/ert/ffc-runtime-length-oracle-cache-2`. Do not infer a full-NVHPC
result from this GNU evidence; the known full-NVHPC timeout is not part of
this slice.

### GNU logical-not reduction check

The smallest current GNU reproduction for the whole-array logical reduction
boundary is `any(.not. a)` with a fixed-size logical array. The existing
regression target also covers whole-array assignment and scalar broadcasts:

~~~bash
LIBRARY_PATH=/mnt/storage/code/lazy-fortran/liric/build \
FO_CACHE_DIR=/var/tmp/ert/ffc-logical-not-fix-cache \
fo test test_session_whole_array_not_compiler
~~~

Unary `.not.` is represented in the AST with only a right operand. The
reduction lowering must evaluate that operand for each element and normalize
the inverse to the i32 logical ABI (`0` false, nonzero true). The independent
compiler/API oracle is
`test_session_logical_not_reduction_oracle_compiler`; it compiles the same
minimal program through FFC and gfortran and compares their executable
outputs. The 2026-08-09 focused GNU run built 456/456 units and passed both
targets. This does not establish aggregate or NVHPC green status.

### Attributing failures from parallel test runs

Do not attribute a broad parallel `fo test` failure to the newest lowering
change from the aggregate result alone. `fo` uses a shared action cache by
default, so concurrent checkouts or dependency changes can reuse a target,
module, or dependency artifact from another checkout. Record the commit and
working-tree state, then reproduce a representative target individually:

```bash
git rev-parse HEAD
git status --short --branch
LIBRARY_PATH=/path/to/liric/build fo test test_session_<target>
```

For a before/after comparison, use separate clean checkouts and distinct
`FO_CACHE_DIR` values. The parent checkout must complete its own build before
its test result is evidence. If that build stops in FortFront (for example on
an implicit-interface diagnostic), report it as a dependency-build failure;
it is not an FFC regression and must not be converted into an expected test
failure. A behavioral oracle that passes individually, together with
unrelated individual failures, is evidence to fix the affected pre-existing
feature or the dependency/cache setup rather than the latest feature commit.

For CLI checks:

```bash
printf 'program main\nend program main\n' > /tmp/empty.f90
LIBRARY_PATH=/path/to/liric/build fo exec ffc -- /tmp/empty.f90 -o /tmp/empty
/tmp/empty
LIBRARY_PATH=/path/to/liric/build fo exec ffc -- /tmp/empty.f90 -c -o /tmp/empty.o
```
