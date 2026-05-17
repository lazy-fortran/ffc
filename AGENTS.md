# ffc

Lazy Fortran compiler driver.

```
Fortran / Lazy Fortran source
  → FortFront typed AST + diagnostics
  → ffc lowering + runtime ABI
  → LIRIC C API (via ISO_C_BINDING)
  → object or executable
```

FortFront stays backend-neutral. `ffc` owns lowering, ABI, runtime calls, LIRIC bindings, and object/exe emission.

## Layout

- `app/` — CLI entry (`ffc.f90`).
- `src/` — active direct-session lowering and LIRIC bindings. Auto-discovered by fpm.
- `test/` — fpm tests. Each file is a standalone `program test_*`. Auto-discovered.
- `legacy_mlir/` — retired MLIR/HLFIR experiment. Not built. Ignore unless porting a specific piece.
- `docs/`, `BACKLOG.md`, `ROADMAP.md`, `DESIGN.md`, `README.md` — current status & plans.

## Build & test

The LIRIC static library must be on the linker path:

```bash
export LIBRARY_PATH=/home/ert/code/liric/build   # adjust to your liric build dir
fpm build
fpm test                           # all 21 MVP tests
fpm test test_session_stop_code_compiler   # one test
fpm run -- empty.f90 -o empty      # use the CLI
```

If you don't have LIRIC built locally, build it first in its own repo (`cmake -S . -B build && cmake --build build` in `~/code/liric`). The static lib is `libliric.a`.

## Conventions

- Free-form Fortran, no implicit typing. Declarations at scope top.
- Modules < 500 lines (hard 1000). Functions < 50 lines (hard 100). Split into `*.inc` files (already used heavily) when the lowerer grows.
- Symbols `snake_case`, derived types end in `_t`.
- New compiler work goes through direct LIRIC `lr_session_*`. Do not add LLVM bindings or revive MLIR/HLFIR without an explicit decision.

## Adding a new test

Drop `test/test_<topic>.f90` containing `program test_<topic>` ... `end program`. fpm auto-discovery picks it up. No fpm.toml edit needed.

## Adding a new supported construct

1. Extend the matching `src/session_program_lowering_*.inc` (or add a new include).
2. Add a `test/test_session_<construct>_compiler.f90` that lowers a minimal program, runs the resulting binary, and checks stdout.
3. If lowering needs a LIRIC API not yet bound in Fortran, extend `src/liric_session_bindings.f90` (interface + wrapper method). Keep liric itself unchanged unless the C API is genuinely missing — file an issue on liric if so.

## Quality gates

1. `fpm build` clean.
2. `fpm test` all green — never skip, weaken, or label tests as "unrelated".
3. Update `README.md`'s supported-features list when you add a real construct.
