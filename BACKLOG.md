# ffc Backlog

This backlog tracks the active FortFront-to-LIRIC compiler path. The
retired MLIR/HLFIR experiment lives only in git history; do not revive it
without an explicit decision.

## P0: Keep main honest

- Keep `fpm.toml` pointing at `src/`.
- Keep executable behavioural tests for every claimed compiler feature.
- Keep CI green; never weaken or skip a failing test to do so.

## P1: Direct LIRIC session lowering

The direct LIRIC session lowerer covers the subset documented in
`docs/SUPPORT_CONTRACT.md`. The currently supported surface includes:

- empty `program main`, integer/real/logical scalars, character literals
  and fixed-length character variables;
- deferred-length `character(len=:), allocatable` including self-aliasing
  and three-way `//` concatenation;
- block `if` with PHI merge for scalars; array-element writes inside `if`
  branches; counted `do` with literal positive and negative step;
- contained integer/real/logical functions and subroutines with scalar
  args; early `return` inside contained functions and subroutines;
- fixed-size 1-D integer arrays, simple derived types with scalar
  integer components;
- single-arm and multi-arm `SELECT CASE` (incl. multi-label and
  `case default`), each arm terminating with `stop`/`return`;
- compile-time `//` folding for character literal chains;
- minimal `print *, expr`, `stop <expr>`, scalar `abs`/`min`/`max`/`mod`
  and integer-to-real `real()` conversion;
- CLI: `-o`, `-c`, `-I <dir>` (`-I` stored, not yet consumed).

Open work is tracked as GitHub issues; the self-hosting roadmap is in #167.

## P2: FortFront API boundary

Tracked by #58 / #173. `ffc` still walks the FortFront AST arena with
`select type` for most lowering. New compiler-facing FortFront queries
should replace those reach-ins one slice at a time. Concrete queries the
roadmap needs are listed in #173.

## P3: Runtime and ABI

The current direct-session ABI is in `docs/RUNTIME_ABI.md`. Each new
feature must update that document and grow executable tests in the same
change. Tracked categories:

- non-integer scalar procedure ABI: see open issues under #50 / #164.
- character value + length representation: #51 and the deferred-length
  series tracked from #167.
- print/runtime call surface beyond the current `printf` shim: #55.
- array descriptors and allocatables: #53, plus the slice issues
  #184–#186.

## Verification

```bash
LIBRARY_PATH=<liric-build> fpm build
LIBRARY_PATH=<liric-build> fpm test
```

CI runs the same on every push and pull request.
