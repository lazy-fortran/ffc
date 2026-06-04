# ffc

`ffc` is the compiler driver for Lazy Fortran and LFortran Infer-style
source. It compiles supported Fortran programs to native object files and
executables via FortFront's typed AST and LIRIC's session C API.

```
Fortran / Lazy Fortran source
  -> FortFront typed AST + diagnostics
  -> ffc lowering + runtime ABI
  -> LIRIC C API (via ISO_C_BINDING)
  -> object file / executable
```

FortFront stays backend-neutral. `ffc` owns lowering, ABI, runtime calls,
LIRIC bindings, and object/exe emission. The retired MLIR/HLFIR
experiment lives only in git history.

## Supported features

The public contract is `docs/SUPPORT_CONTRACT.md`. It lists every
supported construct, its ABI, and every tracked gap with issue links.
Refer to that document instead of this README for the feature list.
Current slices include compound formatted `print` with literal `I`, `X`,
and `F` descriptors on stdout, and fixed-size rank-2 integer arrays with
scalar element access.

## Build

The LIRIC static library must be on the linker path:

```bash
cd ../liric && cmake -S . -B build -G Ninja && cmake --build build
cd ../ffc
export LIBRARY_PATH=../liric/build
fpm build
fpm test
```

`fpm build` produces the `ffc` binary; `fpm test` runs the behavioural
test suite.

Compile a minimal program:

```bash
printf 'program main\nend program main\n' > /tmp/empty.f90
LIBRARY_PATH=../liric/build fpm run ffc -- /tmp/empty.f90 -o /tmp/empty
/tmp/empty
echo $?
```

## Conformance

`docs/CONFORMANCE.md` documents the conformance gauntlet runner that
drives external Fortran test corpora through the full `ffc` pipeline.
It produces an xfail-style report with JSONL output.

## Layout

- `app/ffc.f90` - CLI entry.
- `src/` - lowering, LIRIC bindings, CLI options (fpm auto-discovers).
- `test/` - behavioural tests; each file is a standalone `program test_*`
  picked up by fpm auto-discovery.
- `docs/` - `SUPPORT_CONTRACT.md`, `RUNTIME_ABI.md`, `DEVELOPER_GUIDE.md`,
  `API_REFERENCE.md`, `C_API_USAGE.md`, `MIGRATION_GUIDE.md`,
  `CONFORMANCE.md`.
- `BACKLOG.md`, `DESIGN.md` - planning docs.

## Conventions

- Free-form Fortran 2003+; no implicit typing; declarations at scope top.
- Modules under 500 lines (hard cap 1000); functions under 50 lines
  (hard cap 100). Split into `*.inc` files (already used heavily) when
  the lowerer grows.
- Symbols `snake_case`, derived types end in `_t`.
- Each new supported construct lands as a code change in
  `src/session_program_lowering_*` plus a behavioural test under `test/`.
  Update `README.md` and `docs/SUPPORT_CONTRACT.md` in the same commit.

## Related repositories

- [fortfront](https://github.com/lazy-fortran/fortfront) - frontend,
  transformation, typed AST.
- [liric](https://github.com/lazy-fortran/liric) - backend C API target.
- [standard](https://github.com/lazy-fortran/standard) - intended
  LFortran Standard and Infer behaviour.
