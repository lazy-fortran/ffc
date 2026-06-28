# ffc

Lazy Fortran compiler driver.

```
Fortran / Lazy Fortran source
  -> FortFront typed AST + diagnostics
  -> ffc lowering + runtime ABI
  -> LIRIC C API (via ISO_C_BINDING)
  -> object or executable
```

FortFront stays backend-neutral. `ffc` owns lowering, ABI, runtime
calls, LIRIC bindings, and object/exe emission.

## Layout

- `app/` — CLI entry (`ffc.f90`).
- `src/` — direct-session lowering, LIRIC bindings, CLI parsing.
  Auto-discovered by fpm.
- `test/` — fpm tests. Each file is a standalone `program test_*`.
  Auto-discovered.
- `docs/`, `BACKLOG.md`, `ROADMAP.md`, `DESIGN.md`, `README.md` —
  current status and plans.

The retired MLIR/HLFIR experiment lives only in git history. Reference
it by commit hash if you need to look back; do not revive it without an
explicit decision.

## Build and test

Use `fo` for every build/test loop; call `fpm` directly only to isolate one named test or diagnose a `fo` failure.

The LIRIC static library must be on the linker path:

```bash
export LIBRARY_PATH=/home/ert/code/liric/build   # adjust to your liric build dir
fo build
fo test                                          # full behavioural suite
fpm test test_session_stop_code_compiler         # isolate one named test
fo exec ffc -- empty.f90 -o empty                # use the CLI
```

If you do not have LIRIC built locally, build it first in its own repo
(`cmake -S . -B build -G Ninja && cmake --build build` in `~/code/liric`).
The static library is `build/libliric.a`.

CI runs the same workflow on every push and pull request.

## Conventions

- Free-form Fortran 2003+; no implicit typing; declarations at scope top.
- Modules under 500 lines (hard cap 1000). Functions under 50 lines
  (hard cap 100). Split into `*.inc` files (already used heavily) when
  the lowerer grows.
- Symbols `snake_case`; derived types end in `_t`.
- New compiler work goes through direct LIRIC `lr_session_*` calls. Do
  not add LLVM bindings or revive MLIR/HLFIR without an explicit
  decision.

## Adding a new test

Drop `test/test_<topic>.f90` containing `program test_<topic>` ... `end
program`. fpm auto-discovery picks it up. No `fpm.toml` edit needed.

## Adding a new supported construct

1. Extend the matching `src/session_program_lowering_*.inc` (or add a
   new include).
2. Add `test/test_session_<construct>_compiler.f90` that lowers a
   minimal program, runs the resulting binary, and checks stdout or
   the exit code.
3. If lowering needs a LIRIC API not yet bound in Fortran, extend
   `src/liric_session_bindings.f90` (interface + wrapper method). Keep
   liric itself unchanged unless the C API is genuinely missing — file
   an issue on liric if so.
4. Update `docs/SUPPORT_CONTRACT.md`, `README.md`, and (when the ABI
   changes) `docs/RUNTIME_ABI.md` in the same change.

## Quality gates

1. `fo build` clean.
2. `fo test` all green — never skip, weaken, or label tests as
   "unrelated".
3. CI green on the PR before merging.
4. Update `README.md`'s supported-features list when you add a real
   construct.
