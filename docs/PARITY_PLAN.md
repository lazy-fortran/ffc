# Corpus Parity Plan

Goal: 100% parity for standard Fortran programs that are in scope for `ffc`:
compile success, correct rejection of invalid programs, runtime output matching
`gfortran -w`, and tracked compile/runtime performance. The measured corpora are
FortFront examples, LFortran `integration_tests`, and the conformant runnable
part of GCC `gfortran.dg`.

OpenMP, OpenACC, vendor extensions, compiler-flag tests, and directive-harness
tests are not parity targets. ISO Fortran parallel features are in scope for
full standard parity, including coarrays and image-control semantics, but they
are late runtime work because they need single-image and multi-image semantics.

Live scoreboard: issue #299. This file is the durable architecture and issue
map. Update it when issue state, manifest counts, standard scope, or phase
boundaries change.

## Current State

Current checked-in manifest counts in this checkout:

| Suite | XFAIL entries | SKIP entries |
|-------|--------------:|-------------:|
| fortfront-f90 | 103 | 0 |
| fortfront-lf  | 60  | 0 |
| lfortran      | 3425 | 0 |
| gfortran-dg   | 2121 | 2371 |

Latest recorded full-dashboard state in #299 after wave 14:

| Suite | PASS | Notes |
|-------|-----:|-------|
| fortfront-f90 | 336 | Near complete; one local/CI gfortran-version XPASS can appear. |
| fortfront-lf  | 204 | No gfortran oracle for lazy syntax. |
| lfortran      | 837 | Remaining backlog is mostly feature stacking and architecture. |
| gfortran-dg   | 1176 | Positive gains are offset by invalid-program acceptance debt. |

Local verification command:

```bash
LIBRARY_PATH=/path/to/liric/build fpm test
```

passed, including `test_session_integer8_function_compiler`.

## Requirements

- **REQ-001 Correct positive behavior:** every valid in-scope corpus file
  either compiles and runs with output/exit matching `gfortran -w`, or is listed
  in an xfail manifest with an owned issue.
- **REQ-002 Correct negative behavior:** invalid standard programs reject with a
  source diagnostic. Negative `gfortran.dg` acceptance is a correctness bug, not
  feature progress.
- **REQ-003 No silent partial lowering:** unsupported constructs must stop with
  diagnostics, never compile to wrong output.
- **REQ-004 Architecture before breadth:** descriptor, scope, module-artifact,
  procedure-ABI, runtime, and diagnostic foundations land before more
  per-feature patches that would duplicate those mechanisms.
- **REQ-005 Fast iteration:** `fo` remains the normal loop. Native `ffc`
  backends optimize compile latency; LLVM/LIRIC performance work must not slow
  the default edit-test cycle.
- **REQ-006 Issue quality:** every open implementation issue is PR-sized,
  tagged by difficulty, and contains requirements, files, scaffolding, positive
  and negative tests, non-goals, and copy-pasteable verification.

## In-Scope Standard Surface

Full parity covers these standard Fortran families, ordered by likely substrate
dependency rather than by standard year:

- Source forms and legacy standard features: fixed/free form, continuation,
  labels, `FORMAT`, `DATA`, `COMMON`, `EQUIVALENCE`, `BLOCK DATA`, arithmetic
  IF, assigned/computed GOTO, and standard-era Hollerith where valid.
- Program units and scoping: main programs, external procedures, internal
  procedures, modules, submodules, host/use association, `import`,
  accessibility, renaming, and intrinsic/non-intrinsic module distinction.
- Declarations and typing: implicit rules, `implicit none(type, external)`,
  attributes, parameters, named constants, kind/len selectors, enums, and BOZ.
- Scalars and expressions: integer, real, complex, logical, character, standard
  operators and precedence, conversions, constant folding, and type/kind/rank
  checks.
- Arrays: explicit, assumed, deferred, assumed-size, assumed-rank, rank 0-15
  semantics, lower bounds, sections, vector subscripts, constructors,
  implied-DO, `WHERE`, `FORALL`, whole-array ops, and elemental semantics.
- Allocation and descriptors: allocatable scalars, arrays, components, results,
  dummies, pointer association/remapping, automatic/runtime arrays, and
  `allocate` with `source`, `mold`, `stat`, and `errmsg`.
- Derived types and OOP: constructors, defaults, nested/array/allocatable/
  pointer components, PDTs, extension, `class`, `select type`,
  type-bound procedures, generic bindings, and finalizers.
- Procedures and interfaces: explicit and abstract interfaces, generic
  resolution, optional/keyword arguments, procedure pointers, elemental/pure/
  recursive procedures, result variables, and external procedure units.
- Control flow: `if`, `select case`, `select rank`, `select type`, `do`,
  `do concurrent`, `block`, `associate`, `exit`, `cycle`, `stop`, `error stop`,
  and `return`.
- I/O: list-directed, formatted read/write/print, internal I/O, files, units,
  scratch, `inquire`, namelist, stream/asynchronous/wait/flush, and full edit
  descriptors.
- Intrinsics and intrinsic modules: full intrinsic set, `iso_c_binding`,
  `iso_fortran_env`, IEEE modules, kind inquiry, array/string/bit/math
  intrinsics.
- C interop: `bind(c)`, interoperable types/procedures/enums, C pointers,
  procedure pointers, `value`, and name rules.
- ISO parallel features: coarrays, image control, collectives, atomics, teams,
  and events. Keep this isolated behind a runtime plan.

Lazy/LFortran scope from `../standard`:

- LFortran standard defaults: implicit typing off, implicit interfaces off,
  bounds checks on, default `real(8)`, default `intent(in)`, predefined `dp`.
- Infer mode: top-level script/global scope, `:=`, first-assignment inference,
  inferred arrays/derived/scalars, and automatic array reallocation.
- Generics/templates: Fortran 2028 `TEMPLATE`, `REQUIREMENT`, `REQUIRE(S)`,
  `INSTANTIATE`, deferred type/constant/procedure declarations.
- LFortran extensions: inline instantiation syntaxes and trait syntax including
  `implements`, `sealed`, and `initial`.
- Monomorphization across modules and infer-mode scopes with stable ABI names.

## Architecture Foundations

These issues should be treated as architecture work. Prefer hard-model review,
small child issues, and zero-regression gates.

| Issue | Difficulty | Foundation | Why it blocks parity |
|-------|------------|------------|----------------------|
| #307 | hard | Unified array descriptor ABI | Required for runtime bounds, assumed/deferred/assumed-rank, sections, vector subscripts, pointer arrays, allocatables, and class arrays. |
| #308 | hard | Two-pass symbol/scope resolution | Required for reliable host/use/block scope, forward spec expressions, shadowing, and negative diagnostics. |
| #309 | hard | General array-expression engine | Required for runtime-shaped whole-array ops, reductions, `WHERE`, `FORALL`, constructors, elemental intrinsics, and alias-safe assignment. |
| #310 | hard | Character/string descriptor subsystem | Required for runtime/deferred/assumed-length character variables, procedure interfaces, expression lowering, and I/O. |
| #274/#297 | hard | Rich `.fmod` and submodule ABI | Required for separate compilation, external `use`, generics, derived layouts, dummy names, result specs, submodule parents, and module variables. |
| #273 | hard | Polymorphism/vtable ABI | Required for dynamic `class`, runtime `select type`, overridden type-bound procedures, class allocatables, and finalization policy. |
| #268/#278 | hard | Fortran runtime and I/O | Required to replace ad hoc `printf`/libc lowering with standard I/O, formatted records, file units, and runtime-backed intrinsics. |
| #286/#291/#306 | medium-hard | Diagnostic and rejection gate | Required so broader positive support does not accept invalid programs silently. |

Difficulty labels:

- `difficulty:easy`: local feature or rejection class, no ABI change, one or
  two files plus tests.
- `difficulty:medium`: subsystem work that touches several lowering paths but
  keeps existing ABI.
- `difficulty:hard`: ABI, architecture, cross-repo dependency, or broad
  migration strategy.

## Current Open ffc Work

Umbrellas and epics: #263 through #272 remain the compliance spine. #262 is
implemented; the follow-up integer(8) argument-width regression is covered by
`test_session_integer8_function_compiler`.

Current feature and correctness issues:

- #273 runtime polymorphism.
- #274 separate-file modules.
- #275 top-level external procedures.
- #276 remaining array forms.
- #277 remaining control flow and statements.
- #278 remaining I/O.
- #279 remaining intrinsics.
- #280 compiles-but-mismatches-gfortran.
- #281 Lazy Fortran support.
- #284 duplicate runtime-helper symbols in separate compilation.
- #286 structured diagnostics.
- #290 contained-function ICE.
- #291 negative-test acceptance.
- #295 `gfortran.dg` compile regressions.
- #297 separate-file submodules.
- #304 module-array/function shadowing.
- #305 zero-size array declarations.
- #306 illegal-context array rejection.
- #307, #308, #309, #310 architecture foundations.
- #311 gauntlet worktree path resolution.

Closed or stale entries removed from the active blocker list:

- ffc #257, #258, #260, #294, #296, #298, #300, #303.
- FortFront #2840-#2842, #2844, #2846, #2848-#2858, #2868-#2871.
- LIRIC #520-#522.
- fo #88.

## Dependency State

### FortFront

No live FortFront blocker is open for the parser/AST issues previously listed
here. Future FortFront work should focus on public compiler queries rather than
new ffc private-arena reach-ins:

- scopes and host/use association;
- declarations, attributes, and inferred types;
- procedure signatures, dummy/result metadata, generics;
- module export/import metadata;
- source spans and diagnostics.

### LIRIC

Current live blocker:

- #523: LLVM backend verifier failure for `integer(8)` print (`zext` does not
  dominate `printf` use).

Closed blockers #520-#522 unblocked LLVM AOT and the performance leg. Native
isel/copy-patch remain the fast-compile default.

### fo

Current parity-relevant blocker:

- #98: stale object reuse after path dependency or `.inc` changes. This can
  hide or invent failures during architecture work.

Adjacent but not immediate parity blockers: #55-#57 and #59-#62.

## Phased Plan

### Phase 0: queue hygiene

- Keep only active branches. Finished work is merged; stale worker branches are
  deleted.
- Every open issue is atomic or a real bug. Trackers are closed after child
  issues are filed.
- Apply `difficulty:*` labels in each repository.

### Phase 1: architecture first

1. #308 symbol/scope collection and lookup.
2. #307 descriptor ABI for arrays and descriptor-passing calls.
3. #310 character descriptor.
4. #274/#297 `.fmod` schema and separate compilation.
5. #309 array-expression engine on descriptors.
6. #268/#278 runtime and I/O surface.
7. #273 polymorphism/vtable ABI.

Each foundation should be split into small BDD child issues. Do not land a
large, all-at-once rewrite.

### Phase 2: correctness gates in parallel

- Drive #291 and #306 by diagnostic class. Every class needs invalid fixtures
  and valid near-misses.
- Drive #280 by wrong-output cluster. Each fix compares combined stdout/stderr
  and exit status against `gfortran -w`.
- Close #290, #295, #304, #305 with focused behavioral tests.

### Phase 3: feature families

With the foundations in place, clear feature buckets:

- arrays and descriptors (#264/#276);
- derived/OOP (#265/#273);
- procedure ABI and generics (#266/#274/#297);
- scalar kinds and intrinsics (#267/#269/#279);
- runtime/I/O (#268/#278);
- control flow and legacy forms (#270/#277);
- Lazy/LFortran and Fortran 2028 generic features (#281).

### Phase 4: full gates

- Promote XPASS manifests after every wave.
- Gate FortFront examples, LFortran integration tests, and the conformant
  `gfortran.dg` subset in CI.
- Keep skip reasons explicit.
- Keep performance tracking for native fast-compile and LLVM optimized paths.

## Issue Template Requirements

Each implementation issue should contain:

- `REQ-001`, `REQ-002`, ... requirements in BDD form.
- Exact files to edit.
- Syntax or behavior to implement, with a valid example.
- Scaffold naming actual routines and data structures.
- Positive fixtures with expected output.
- Negative fixtures with diagnostic class.
- Non-goals.
- Copy-pasteable verification commands.
- `difficulty:easy`, `difficulty:medium`, or `difficulty:hard`.

Hard foundation issues should describe strategy and migration steps rather than
pretending to be a single 300-line patch.
