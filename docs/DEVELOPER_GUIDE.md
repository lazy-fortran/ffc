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
