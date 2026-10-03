# Fortran ecosystem plan and status

This is the combined status and plan for the fo build driver, ffc compiler,
FortFront frontend, LIRIC backend, and language-standard repository. The current
status records delivered behavior and failing checks; the active plan retains
its acceptance conditions and evidence. Historical checkpoints remain dated.

## Current status

Checkpoint: 2026-10-02. Implementation work stopped at the user's request;
all five main branches were pushed and verified. This status supersedes the
historical checkpoints below. Remaining language and release work stays open.
The repository revisions identify the delivered code checkpoint.

### Repository delivery

| Repository | Code checkpoint | Remote state |
|---|---|---|
| fo | `50de371` | Pushed |
| fortfront | `2b6bc1a` | Pushed |
| ffc | `73fc71f` | Pushed |
| liric | `9042548` | Pushed; no code changes in this delivery |
| standard | `160032a` | Pushed; no code changes in this delivery |

All implementation, build, and validation work for this delivery lives under
`/var/tmp/lazy-fortran-finish-20261002`. Workers return frozen patches; the
controller commits and promotes them. No review wait is part of this delivery.

### Delivered behavior

- fo discovers marked test modules, preserves public filenames and exit status,
  gives shared PROGRAM names distinct internal identities, and collects source
  dependency keys consistently. Formatter block recognition preserves keyword
  suffixes and identifier assignments.
- FortFront validates declaration namespaces and definition contexts while
  accepting valid split declarations and DATA initialization. Named-loop depth,
  compact dotted operators, unary integer kinds, identifier-hash overflow, and
  identifier-table ownership have focused fixes.
- ffc consolidates original test bodies behind a shared dispatcher. Named
  EXIT/CYCLE resolves the requested enclosing construct. BOZ output follows the
  declared integer width. Derived identity sections and dummy lower bounds use
  the canonical descriptor. Unsafe allocation/index/optional guards are split.
- ffc supports nonliteral MERGE, typed ABS in reductions, and runtime MASK
  reduction paths with documented rank, kind, and contiguity boundaries.
  Rank-one allocation bounds preserve same-size assignment bounds and reset
  resized allocation bounds. A constrained rank-one descriptor TRANSFER slice
  copies supported numeric bit patterns with live source-extent checks.
- The conformance harness supplies immutable input and independent standard
  Fortran references for interactive Lazy Fortran cases, including cache retry.
  Supported historical negative cases now compare execution with gfortran;
  invalid and unsupported neighbors remain checked.
- The Fortran 2023 audit has 45 atomic follow-up issues: FortFront #3022–#3035
  and ffc #767–#797. The disposition table distinguishes support, gaps, and
  explicitly excluded facilities.

### Validation

| Check | Result |
|---|---|
| fo full pipeline | Pass |
| fo CLI behavior checks | Four pass |
| fo MCP checks | 79 pass, zero fail |
| FortFront full suite | 767/767 pass; final pipeline and required recursion checks pass |
| Frontend behavior / namespace parity | 56/56 and 34/34 |
| Frontend sanitizer checks | 4/4 pass |
| Integer B/O/Z output | 75/75 match gfortran |
| Shared test conversion oracle | 67 independent verdicts preserved |
| Existing reported test names | All 500 preserved |
| Shared test build size | 350,501,754 bytes in measured worker checkout |
| Single test edit | Two relinks, 51.89 seconds, named case passes |
| Final combined compiler validation | Build/static/lint pass; full suite: 502 pass, 6 fail, 508 names, 509.16 seconds |
| Final rejection gate | Pass: 919 files, 706 accepted, 213 rejected; no new rejections |
| Four-suite provenance refresh | Deferred; checked-in dashboard remains stale |

Behavioral oracles include controlled falsification and restoration. Compiler
baseline after a correct CLI build was 487 passing and 13 failing tests. The
frozen final run reports 502 passing and six failing tests, listed below.

### Failing checks

The final full suite is not green. No tests were skipped, weakened, or removed
to change this result.

| Test | Observed failure |
|---|---|
| `test_session_class_star_assumed_shape_compiler` | Missing real(8) refusal diagnostic; lowering reaches an undefined procedure at link time. |
| `test_session_class_star_rank2_assumed_shape_compiler` | Missing default-integer/real(8) refusal diagnostic; lowering reaches an undefined procedure at link time. |
| `test_session_pdt_inheritance_compiler` | Unsupported scalar type at line 19 in the inheritance case. |
| `test_parity_dashboard` | Checked-in production snapshot has stale revision/source provenance. |
| `test_fortfront_corpus_conformance` | Hardcoded totals 517/264 disagree with current 646/267 selections. Observed reports contain zero FAIL and zero XPASS, but these reports are not a locked four-suite provenance epoch. |
| `test_conformance_gauntlet_smoke` | Expected rejection of a case listed in both XFAIL and NOREF does not occur. |

The full run used four test workers on a loaded machine. Its 509.16-second wall
time exceeds the plan's 120-second target. The 350 MB disk measurement comes
from the consolidation worker after obsolete binaries were removed; an existing
checkout can retain older build artifacts until its stale-cache cleanup.

### Remaining work

Compiler lint succeeds with 14 unused imports and 93 compiler warnings in
existing code. No warning is anchored to a newly added or modified line.
The documentation checker reports two unchanged prose findings in
docs/RUNTIME_ABI.md; introduced support and audit prose passes.

The plan still contains substantial language and release work. In particular,
CLASS(*) refusal diagnostics, PDT inheritance, higher-rank noncontiguous
sections, trailing-X formatted-record positioning, full corpus parity,
manifest-owner retirement, backend producer artifacts/dominance serialization,
platform release gates, and standard proposal contracts remain open unless
later evidence in this document explicitly closes them. Unsigned integers and
ASYNC=/DEPENDENCY are extensions outside the Fortran 2023 audit denominator.

The test consolidation reduces disk use and link count. Its measured single
edit still takes 51.89 seconds, and the final full-suite timing exceeds the
target. No unsupported corpus case is counted as a pass
by deleting its expectation or changing its scope.

### Evidence

The evidence root contains frozen worker patches and their SHA256 digests,
`manifest.json`, `suite-evidence.json`, `suite-relink-evidence.json`, the final
pipeline logs, numeric/format/descriptor parity receipts, and conformance
observations. In particular, `logs/ffc-suite-final-frozen.log`,
`evidence/final-test-verdicts.json`, `logs/rejection-final.log`,
`logs/ffc-lint-final.log`, and `logs/conformance-oracle-integrated.log` record
the final compiler result. `evidence/ffc-final-controller.patch` freezes the
integrated source before its checkpoint commit. These artifacts are retained in /var/tmp; Git commits
preserve the implementation, test oracles, and public support documentation.

## Active plan

Priorities, in force: **development iteration speed and disk space come
first (W0)**; the modern-Fortran support burn-down follows as Part II and
must be scheduled *around* W0, not on top of it. Every W0 slice makes the
remaining Fortran work cheaper to iterate on.

Repos: `fo` (build driver, native), `ffc` (compiler), `fortfront`
(frontend), `liric` (backend), `standard` (language docs). All five stay
clean at end of every committed slice; ABI-touching work still needs the
full suite + rejection gate **once**, but W0 exists so that "once" is fast.

---

### Evidence path convention

Short pointers written `perf/...` and `logs/...` resolve under
`/var/tmp/ffc-goal/` - so `perf/suite-w38.txt` is
`/var/tmp/ffc-goal/perf/suite-w38.txt`. Pointers are checked for existence
with that base; without it 30 of 39 read as missing when the artifacts are on
disk, which is a pointer-spelling defect and not lost evidence. The one
exception is fixed inline where a slice wrote to a different path.

### Baseline measurements (2026-10-01)

| Cost center | Measured | Root cause |
|---|---|---|
| rebuild after any `.inc` edit | **8m55s** (`touch` + `fo test --only-changed`) | one 7,394-line root module `session_program_lowering.f90` + 42 `.inc` includes → whole TU recompiles → **460 targets relink** |
| full suite | **430s** | `test_fortfront_corpus_conformance` 100s + `test_conformance_gauntlet_smoke` 85s + `test_parity_dashboard` 9s = **45% in 3 serial monoliths**; remaining ~495 tests ≈ 0.5s each |
| no-op `fo build` | 13s per invocation | fpm dependency scan over 460 targets |
| disk per checkout | **16GB** (`ffc/build`) | 509 executables × ~20–30MB, each statically embedding `libffc.a` (28MB, ~2,600 lowering symbols); multiple stale `gfortran_<hash>/` profile dirs |
| compiled program | 76ms hello vs gfortran 41ms | **the compiler itself is NOT the bottleneck** |

Scratch rule (all repos, all agents): work scratch lives in **`/var/tmp`
only**; `/tmp` is tmpfs (RAM-backed, 62% full). Never write large probes,
logs, or suite artifacts to `/tmp`.

### Test and build mechanisms

| Mechanism | Rust | Go | our adoption |
|---|---|---|---|
| test granularity | **one test binary per crate**, all `#[test]` run in-process | one binary per package | W0.3: consolidate ~400 `test_session_*` into one dispatcher binary; test **names** survive as cases |
| link hygiene | `--gc-sections` over per-CGU rlibs | export data + DCE, tiny binaries | W0.2: `-ffunction-sections -fdata-sections -Wl,--gc-sections` on lib + tests |
| debug info | split-debuginfo, line-tables in dev | DWARF cached | W0.1: default `-g0`/line-tables-only; `-g` only in `--debug`/`--asan` profiles |
| reuse/caching | content-addressed `target/`, cheap relinks | build cache | `fo` already hashes per module — keep; W0.2 adds stale-store GC |
| dev linking | `-C prefer-dynamic` | — | W0.5 (optional): shared `libffc.so` for dev profile only |

---

### W0 — Iteration speed & space (TOP PRIORITY)

Order is deliberate: each step is safe alone and compounds.

- [x] **W0.1 Debug-info diet (ffc, one commit, trivial).**
      Default build profile: `-g0` (or `-gline-tables-only`); keep full
      `-g -fcheck=all -fbacktrace` only under `--debug`/`--asan`.
      First measure `du -sh build` and `time fo build` before/after.
      Verify: targeted family tests + full suite fail-name set identical
      once (background). Acceptance: build size −≥50%, rebuild −≥30%.
      **LANDED** — fo `fc73029` (dialect `-g0` default, `--debug`/`--asan`
      keep full `-g`) + ffc `183d86b` (`[extra.fo] debug-info = "g0"`).
      Evidence: `perf/w0_slice_evidence.md`; edit rebuild 535s → 12s build /
      120s build+`--only-changed`; fail names `perf/suite-w03par.fails`.
      After capture `perf/w0_after.env` vs `perf/w0_before.env`
      (16 795 787 477 B → 8 665 725 549 B).
      SCOPE OF THE DIET, measured so the claim is exact: fo/ffc emit no DWARF
      (all `build/fo/obj/*.o` and `build/fo/lib/*.a` have zero `.debug_info`,
      and `strace` confirms `-g0` reaches the test compile). A test executable
      still carries 662 KB `.debug_info` / ~1.6 MB `.debug_*` because **liric's
      deps** ship DWARF — `build/_deps/sleef-build/lib/libsleef.a` holds
      32 686 902 B — and the linker pulls referenced members into every
      binary. Tracked as liric#535; not a regression in the fo/ffc diet.

- [x] **W0.2 Linker DCE + `fo` hygiene (ffc + fo).**
      All four bullets delivered. gc-sections compile+link policy fo `6ed00e4`;
      `fo clean --stale` fo `6ed00e4` freeing 6234.1 MiB (16G to 9.3G); parallel
      test runner fo `3561b50` with `FO_TEST_REPORT_NAMES=1` TEST_RESULT lines and
      shard allowance computed once in fo `63f1641`; `--asan`/`--debug` profiles
      verified symbolizable - full file:line backtraces through fortfront, ffc and
      the test (which is how the three sanitizer defects behind fo#134 / ffc#758 /
      fortfront#3019 were found). Size outcome `ffc/build` **1.7-1.8 GB**
      (`perf/suite-w38.txt`), binaries mean **2.1 MB** (`perf/suite-w40-quiet.txt`
      mean_bin_MB=2.1 bins=501). Gate once at end of W0.2 per process rules:
      **34.25 s**, 705 accepted / 213 rejected, 0 newly rejected
      (`perf/gate-full-w38.log`, `perf/gate-quiet.tsv`).
      ffc lib + test binaries: `-ffunction-sections -fdata-sections`
      compile, `-Wl,--gc-sections` link (test the `--debug`/`--asan`
      profiles stay symbolizable). `fo`: `clean --stale` deletes orphaned
      `build/gfortran_<hash>/` profile dirs and unreferenced store entries;
      wire `FO_JOBS` (already in the binary's vocabulary) into the test
      runner so independent test **targets** run `min(nproc, 16)`-wide;
      parallelize the corpus walkers (corpus conformance, rejection gate,
      gauntlet) — they are independent ~10ms ffc spawns, 32-wide today
      idle: serial. Acceptance: suite wall < 120s, gate wall < 90s,
      per-test binary size shrinks; same fail-name sets.
      **LANDED, acceptance partly met** — fo `6ed00e4` (gc-sections pair +
      `fo clean --stale`: 16 GB → 9.3 GB, 6234 MiB reclaimed), fo `3561b50`
      (test team + `FO_TEST_REPORT_NAMES`), ffc `612f125` `984bf07`
      `957aaf4` `292ed0a` `31bfa14` `5cea955` (gauntlet `--jobs`, shard
      width default, children stop publishing, guard matches serial).
      Met: rejection gate 918 files **705/213 in 27 s**
      (`/var/tmp/ffc-goal/perf/gate-w05b.tsv` + `.stderr.log`, rc=0, no
      newly rejected); mean test binary
      20.9 → **15.5 MB**; corpus suite 217 s serial → **45 s** sharded
      (`shard/S2.obs` vs `shard/E9.obs`: 645 cases, identical order,
      identical status/exits/provenance; only output shas of 12 programs
      differ and two of those were shown to differ run-to-run **serially
      too** for both ffc and gfortran — nondeterministic programs, not
      shard damage); every test reported by name (500/500).
      NOT met: suite wall. Best measured **153 s** (`perf/suite-w04r.txt`,
      ffc `8a1bf71` + fo team 8, fail-set identical to `suite-w04p.fails`),
      down from 248 s: the two critical-path tests now parallelize their own
      internals - `test_conformance_gauntlet_smoke` issues all 15 walk and
      rejection blocks behind one rc-barrier (148.5 s -> 82 s, w04p; each
      block exports its own TMPDIR after a repro showed concurrent same-suite
      runs sharing scratch flip rejections to SKIP=1), and
      `test_conformance_sampling` runs its determinism pair concurrently
      (85 s -> 79 s). Widening the team still does not help; it hurts, the
      walkers inside the team oversubscribe:

      | team | shard width | suite wall | test-time sum |
      |---|---|---|---|
      | 8 | 6 | 330 s (`w07`) | 893 s |
      | 24 | 16 | 361 s (`w10`) | 1241 s |
      | 24 | 1 | 235 s (`w11`) | 779 s |
      | 8 | auto | 153 s (`w04r`) | ~870 s |
      | 16 | auto | 216 s (`w04q`, load1 35.9) | 500 recs |
      | ~10 | auto | 210 s (`w04s`, load1 57.7) | 500 recs |

      fo `89005f8` encodes that: when the team is wider than one it exports
      `FFC_CONFORMANCE_JOBS=1` so a walker inside a team member stops adding
      threads to a busy slot, and an explicit user value still wins (fo
      `fab4714` carries the oracle, falsified by unconditional overwrite ->
      fails exactly that check). Stand-alone walkers keep sharding
      (fortfront-f90 corpus 217 s serial -> 45 s at `--jobs 16`).
      The floor is structural, not tunable: `fo test --all` links 500 test
      binaries (warm run == cold run, 329 vs 330 s, `perf/suite-w08.txt`),
      and `test_conformance_gauntlet_smoke` alone walked the whole corpus
      four times serially (143 s; now 82 s with its 15 blocks concurrent,
      ffc `c29ec68`). The remaining floor is the 500-binary link/test CPU
      sum: ~870 s / team 8 + ~35 s build = 153 s. Both disappear in W0.3;
      fo issue #131 tracks the
      dispatcher support. Until then < 120 s is not reachable by tuning.
      Noise note (2026-10-02): later re-measures w04t/u/v read 186-211 s
      while foreign `lean`/`xdiagno` processes held load1 31-38 on this
      box; w04p/w04r (153 s, load1 ~26) remain the clean comparable runs.
      Re-measured on the current HEAD: rejection gate **wall 41.54 s,
      accepted=705 rejected=213, rc=0** (918 files,
      `perf/gate-final.log` + `perf/gate-final.tsv`) — gate target (< 90 s)
      **MET**. Suite wall `perf/suite-w19.txt`: total=500 pass=489 fail=11,
      **zero new failure names** vs `e339_base_failnames.txt` (the 11 are
      the 10 baseline plus the known corpus XPASS), wall 250 s.
      Fail names: all 10 baseline names present, plus
      `test_fortfront_corpus_conformance` whose sole cause is the
      pre-existing synthetic-XPASS (`FAIL[synthetic-xpass]`, also in
      `logs/suite-w01full.log` at 246.90 s before sharding existed).
      SCORED (fo `6ed00e4` `601606e` `3561b50` `63f1641`, ffc `612f125`
      `984bf07` `31bfa14` `5cea955` `957aaf4`..`7ee4cbe` `fbd62eb`):
      gate **41.54 s < 90 s MET**, 705 accepted / 213 rejected, rc=0
      (`perf/gate-final.log`); `fo clean --stale` freed 6234.1 MiB
      (16G -> 9.3G); suite team runs 12-wide under `FO_JOBS` with a
      shard/oversubscription guard; corpus sharding verified same 201
      records at 72.69 s serial -> 21.82 s `--jobs 8` (3.33x).
      **SIZE TARGET MET** (fo `e4fccad` + ffc `f83fb1c`, `link = "shared"` +
      `pic = "true"`): the library folds once into
      `build/fo/lib/lib_<hash>.so` (20 MB) and every test links against it
      instead of relinking a ~30 MB archive - per-test binary 15.6 MB ->
      **2.1 MB**, `ffc/build` **8.67 GB -> 1.27 GB** (< 2 GB target MET,
      `perf/w0_shared_lib.env`), suite `perf/suite-w33.txt` total=500
      pass=489 fail=11, zero new failure names. Gate re-verified after the
      link change: rc=0, **34.01 s** (< 90 s, faster than the earlier 41.54 s),
      **705 accepted / 213 rejected** unchanged (`perf/gate-shared2.counts`).
      **AUTHORITATIVE RE-VERIFICATION (fo #133).** `which fo` is
      `/home/ert/.local/bin/fo`, a plain ELF copy, not a symlink - and
      `fo build` does not refresh it. Every gate/suite number above was taken
      against a binary older than the source it was supposedly testing
      (installed 18:11 vs build 19:23), which is why a fix appeared not to
      work three times before the staleness was found. After `fo install`, the
      gate is rc=0, **705 accepted / 213 rejected**, wall **34.25 s**
      (`perf/gate-w38.tsv`, `perf/gate-full-w38.log`) with **0 newly rejected
      rows** vs `test/fixtures/corpus_rejection_baseline.tsv`, and the suite
      is `perf/suite-w38.txt` total=500 pass=489 fail=11 with a fail-name set
      identical to `e339_base_failnames.txt`, wall 244 s, `build/` 1.7 GB,
      mean binary 2.1 MB.
      Widening the shardable set (fo `f7dd575`) took the longest test
      145.15 -> 128.76 s and wall 235 -> **226 s** (`perf/suite-w34.txt`).
      **CLOSED AS PROVEN-BOUNDED (w38/w39).** Wall cannot go below its
      longest test, and that test is 145.12 s (`perf/suite-w37.results`).
      Every other candidate cause was measured out: warm vs cold is 235 vs
      244 s, so the rebuild phase is worth only ~9 s and **test execution
      dominates** (`perf/suite-w39-warm.txt`); `FO_JOBS=24` 231 s vs 12 at
      226 s; shard width 4 -> 123.76 s but 16 -> 129.35 s (worse);
      load1 sits at 7.66-8.30 on 32 cores, so idle cores are not the limit
      either. Closing it needs the long test decomposed, which removes an old
      test name and is out of bounds. Numbers below are historical.
      NOT MET, with cause: suite wall < 120 s. It is not links, not team
      width (`FO_JOBS=24` measured 231 s at load1 7.17 vs 226 s at 12,
      `perf/suite-w35.txt`), and not shard width - a direct sweep on the
      long test gives 123.76 s at `FFC_CONFORMANCE_JOBS=4` and **129.35 s at
      16**, i.e. sharding harder is neutral-to-worse because its
      named/list/limited/forwarded modes are each one sequential run. The
      wall decomposes into ~124 s inside that one test plus ~100 s of test
      build, and closing it means changing what the test does, which is out
      of bounds.

      489/11, zero new failure names) because the floor is the longest
      single test, `test_conformance_gauntlet_smoke` at 145.15 s - sum of
      all test times is 736 s so 12 jobs imply a 61 s ideal, and no
      linker work can beat 145 s. Closing it means splitting that test,
      which removes an old test name from the inventory and is out of
      bounds; recorded as a named blocker, not met by dropping coverage.

- [x] **W0.3 Suite consolidation (fo + ffc + fpm.toml) — Rust's `cargo test` model.**
      **Implemented 2026-10-02:** 497 original Fortran bodies retain their
      public names through a shared dispatcher; all 500 original reported
      names survive. The independent 67-case oracle preserves success, STOP,
      host association, and source-edit behavior. Measured build size is
      350,501,754 bytes; a single-case edit relinks two outputs in 51.89 s.
      The original wall-time acceptance target remains unmet until a final
      combined measurement establishes it. See STATUS.md for final failures.

      Create ONE session test executable dispatching by case name in-process;
      `fo test <name>` routes there with the name as argv; `fo test --all`
      unchanged; standalone binaries only for genuinely separate executables.
      Acceptance: `build/` < 2GB; relink cone after one edit <= 5 targets;
      every old test name still runnable and reported by name.
      Measured iteration cost: `ffc/build/fo/obj` = **38 MB** vs
      `ffc/build/fo/bin` = **7.7 GB** — 503 test binaries each re-link ~15 MB
      of libffc, so suite wall and build size are one bottleneck.
      **LANDED IN fo, INERT, ORACLED**: `0526874` config key + marker
      routing, `2747832` fixed a doubled source path (the scanner's `filename`
      is already openable; joining `project_dir/test_dir/` onto it missed every
      marker, so routing silently never fired), `b84f250` maps a marked test to
      the dispatcher's DAG node so it links once. Oracle
      `fo/test/test_dispatcher_routing.f90` (11 checks; falsified — deleting
      the self-dispatch guard fails exactly `dispatcher gets no argv case`).
      **PROVEN, THEN BACKED OUT IN ffc** (`6888ecf`): with the dispatcher live,
      a single named run was genuinely routed — deleting
      `build/fo/bin/test_session_empty_program_compiler` left `fo test
      test_session_empty_program_compiler` passing (`PASS 0.07s`) — but
      `fo test --all` regressed to **2 new failures** (`perf/suite-w24.txt`
      total=501 pass=488 fail=13): `test_session_empty_program_compiler FAIL 3`
      (exit 3 = the dispatcher's own "no such case", so argv in the `--all`
      path is not the bare case name) and `test_ffc_suite FAIL` (the
      dispatcher is scanned as a test and fails when run bare). ffc therefore
      keeps only `tools/make_suite_cases.py` (continuation-fold + skip-refused
      fixes, `6a4a832`), unapplied. Baseline re-confirmed after the backout:
      `perf/suite-w25.txt` total=500 pass=489 fail=11, **zero new failure
      names**, wall 262 s.
      **FALSIFIED AND REVERTED** (ffc `73dbcd5`): folding all 96 wrappers at
      once crashes 93 of them routed (57 exit 1, 36 SIGSEGV) while every one
      passes standalone - `test_rank2_loop_debug` rc=0 alone, rc=134 folded.
      A shared process holding 96 module-scoped tests corrupts them, so the
      batch reverted rather than trimming tests until the number looked green.
      Suite sharding is verified independently: same 201 records, serial
      72.69 s -> `--jobs 8` 21.82 s (3.33x), see
      `/var/tmp/ffc-goal/perf/w0_slice_evidence.md`.
      BLOCKER (fo#132): make the `--all` path pass the bare case name and keep
      the dispatcher out of test discovery; then scale. Population measured: 497
      test programs, 96 convertible mechanically (~1.6 GB of the 8.2 GB), 401
      with internal helpers needing host-associated hoisting — so < 2 GB runs
      through the 401 plus linking libffc once, not through wrappers alone
      (liric#535 covers the separate ~1.6 MB/binary DWARF from `libsleef.a`,
      32.7 MB, which is not fo's diet to enforce).

- [x] **W0.4 Eradicate `.inc` — clean modules + submodules, SRP (ffc).**
      **DONE 0 .inc** — final fold `0676861` (16→0); clean build
      `perf/build-foldT.log` rc=0; suite w04n 489/11 fail-set IDENTICAL
      (`perf/suite-w04n.txt`); build 1.8G<2GB, single-edit rebuild 36.5s
      (`perf/w0-final2.txt`). Tool chain: `640d10c` lead-check, `f646699`
      .f90-includers, `76ec57a` indent-flexible heads + cycle tolerance,
      `e810e3a` declaration-section-only capture (intent corruption fix),
      `cac2e70` single-arg wrap, EXEC-keyword guard.
      **PREVIOUS STATE: counter moving 51 → 38** — thirteen includes folded into
      real `submodule (session_program_lowering_impl)` units, one per commit
      (ffc): `31dcc53` lazy_monomorph, `4bfa875` alloc_array_result,
      `344ce7e` internal_write, `9db0c31` io_implied_do, `0ab72fe`
      internal_read, `996f6f5` io_typecheck, `2ec9fe8` print_expr,
      `5023024` write_ops, `3f9e47e` scalar_allocatable, `5d218f4` submodules
(suite `perf/suite-w04a`), `4d13715` declarations (suite
`perf/suite-w04b`, fail-set identical), `7cba662` character_tail
(suite `perf/suite-w04c`, fail-set identical), `d6ecb7d` functions_tail
+ tool fixes (ROOT include removal, CPython-3.14.7 `re.sub` no-op
worked around with a line filter, `result(...)` now carried through the
long-header wrap); suites `perf/suite-w04b/w04c` fail-name sets
IDENTICAL to the accepted 11; tool in `bb94cf6`.
      `tools/inc_to_submodule.py` does the mechanical half and refuses to
      guess (missing dummy declaration, nested include, duplicate sibling
      definition, leading fragment = procedure continued from another file,
      result-variable declaration order). Oracle for every slice: full-suite
      fail-name set identical to `e339_base_failnames.txt` plus the known
      corpus XPASS (`perf/suite-w16.txt`: total=500 pass=489 fail=11).
      REMAINING 42 fall in three classes, and only the first is mechanical:
      (a) clean leaves — migrate with the tool;
      (b) hub-nested children (`submodules.inc`, `functions_tail.inc`,
      `character_tail.inc`) — their `include` lives in `top.inc` /
      `functions.inc` / `character.inc`, and migrating them surfaces
      interface-order faults in the 8k-line root that need real splitting;
      **RECLASSIFIED by measurement (next unit, easier than stated).**
      `src/session_program_lowering_submodules.inc` was re-read: 597 lines,
      19 procedures, **zero `context%` references and no module-level state
      or derived-type declarations** (`grep -q context%` is the discriminator,
      and it is the only file in `src/*.inc` that fails it). So its obstacle
      is NOT semantic coupling to the host module - it is purely
      include-location indirection, because the `include` directive sits at
      `session_program_lowering_top.inc:1401` rather than in the root. The
      work is therefore: lift it to a real module, replace that one line in
      `top.inc` with a `use`, and let the suite fail-name set prove it.
      `is_submodule_unit` has 2 external uses and
      `submodule_parent_module` 0, so the export surface is small.
      Deliberately not attempted this turn: a 19-procedure lift plus full
      suite verification cannot close the 0-`.inc` target (best case 42 to 41)
      and risks leaving the tree broken at a checkpoint.
      **Exact blocker for the lift, measured:** the signature surface is 5
      derived types + 3 node types. Only 2 resolve inside ffc -
      `lowering_context_t` (`ffc/src/session_program_lowering_types.f90`) and
      `module_info_t` (`ffc/src/ffc_module_artefact.f90`). `function_def_node`,
      `subroutine_def_node`, `module_node`, `submodule_node`,
      `interface_block_node` and `ast_arena_t` are re-exported across the
      fortfront boundary and were not resolved to a specific `only:` list, so
      the new module's `use` clause is not yet known-good. A half-lifted module
      breaks the build for every test, so the prerequisite is a proven
      type-provenance list from fortfront's public API, not more editing.
      (c) fragments that continue a procedure started in another file
      (`save.inc`, `read_ops.inc`, `read_al.inc` reference `c_false_local`,
      declared inside a procedure of `common.inc`) — these require splitting
      the encompassing procedure, which is the layered-DAG work below.
      42 to go, one per commit, leaves before hub.
      **MECHANICAL ROUTE PROVABLY EXHAUSTED (measurement, this close).** The
      rule that made the nine successful lifts work is now extracted from them:
      each lifted unit (`session_program_lowering_scalar_kind.f90` etc.) is a
      **plain module** that makes **zero calls into the host module** - only
      `use session_program_lowering_types, only: ...`. Tested every remaining
      file by comparing its outbound `call` targets against the procedures it
      defines itself: **zero of 42 have zero outbound calls.** `inquire.inc`,
      the smallest at 321 lines, calls 9 foreign procs (`assign_i32_to_symbol`,
      `char_expr_operands`, `compute_len_trim`, `create_printf_format_global`,
      `materialize_character_view`, `reload_i32_symbol`, …);
      `submodules.inc` calls 13 (`lower_scalar_function`,
      `lower_nested_internal_procedures`, `register_internal_function_name`, …).
      So there are no leaves left to pick - the 42 form one mutually-recursive
      scope, and the earlier "-9 clean leaves then dry" was not a shortage of
      effort but the actual frontier. Closing 42 to 0 needs the layered-DAG
      split below (breaking call cycles), not more include folding. Corrects
      the plan's assumption that a leaves-first sweep can continue.
      Also note: type provenance is NOT the blocker - `submodule_node` is in
      fortfront `src/ast/nodes/ast_nodes_data.f90` and ffc already imports it
      at `session_program_lowering.f90:265`; my earlier "unresolved across the
      boundary" claim came from a grep pattern that missed `type, extends(...)`.

      User rule: `.inc` files are **banned**; submodules allowed only
      where interface/body separation genuinely helps (mutual recursion).
      Split the lowering root into a real layered DAG:
      `ffc_kernel` (context types, emit primitives, symbol tables) →
      domain modules (`ffc_lower_char`, `_arrays`, `_derived`, `_io`,
      `_reductions`, `_shapes_descriptors`, `_functions`, `_select`,
      `_intrinsics_*`) → dispatcher module (`session_program_lowering`).
      Cyclic helper calls: introduce small interface parent + submodule
      bodies (or hoist the shared primitive down into `ffc_kernel`).
      Migration style: **one `.inc` per commit, mechanical, byte-identical
      behavior** (targeted oracles + suite fail-name set), leaves first,
      hub last; `touch src/*.f90` no longer needed once includes die
      (fpm/fo see real module units). Monotonic machine check:
      `find src -name '*.inc' | wc -l` strictly decreases every commit
      (roadmap's "zero .inc check" reuses this).
      Acceptance: 0 `.inc`; rebuild after a single-domain edit < 60s.

- [x] **W0.5 (optional, measure first) shared `libffc` for dev.**
      Trigger never fired and it was done anyway. Condition was "if W0.1-W0.4
      leave dev rebuilds still > 60 s" - measured single-domain edit rebuild is
      **12 s** (`perf/w0_slice_evidence.md` W0.1 table), so the precondition was
      not met. Landed regardless because shared linking was the lever that took
      `ffc/build` under 2 GB: PIC in the live flag path (fo `8dfbc80`),
      `pic = "true"` (ffc `e4756c4`), `link = "shared"` + `shared_library()`
      (fo e4fccad), `libffc.so` 20 MB with test binaries **2.1 MB** vs 15.6 MB
      static, found via rpath with no `LD_LIBRARY_PATH`; full suite oracle green
      (`perf/suite-w38.txt` total=500 pass=489 fail=11, fail-name set identical
      to `e339_base_failnames.txt`), cold lib build 33.08 s.
      If W0.1–W0.4 leave dev rebuilds still > 60s: dev profile links
      `libffc.so` (Rust `prefer-dynamic`), release stays static. Not
      started unless the numbers demand it.

**W0 process rules:** measure before/after every slice (time + du,
recorded in `/var/tmp/ffc-goal/perf/`); behavior is frozen during W0 —
W0 commits change *structure only*, never semantics; the rejection gate
runs once at the end of W0.2 and at W0.3 promotion, not per commit.

---

### W1 — Fortran front end / lowering correctness (from prior plan, in order)

Status snapshot at rewrite: ffc `d0f798a` (pushed; #339 slices A+B;
last background suite `fo_all_e339B_v18c.log` must show 489/11 with
fail-name set identical to `/var/tmp/ffc-goal/e339_base_failnames.txt` —
if not, fix before W0 work on ffc); fortfront `8f35952`; liric
`3636871`; fo `9f14389`; standard `160032a`.
Rejection gate: 705 accepted / 213 rejected. Baseline fails: 10 names +
corpus TIMEOUT (#754).

#### W1.a — finish in-flight metadata/descriptor retirement

- [x] **fortfront #3018 literal-base substring ranges dropped at parse**
      **LANDED fortfront `7df1e3e` + follow-up (regression test
      `test_issue_3018_literal_substring`, suite 761/761).** Fix folded in
      fortfront, not split two-repo: `apply_index_postfix` now folds a
      literal base with constant in-range bounds to the truncated literal
      at parse time (`'abcdef'(2:4)` -> `'bcd'`); out-of-range bounds stay
      unfolded because gfortran compile-errors those, and the retained slice
      is refused loudly downstream replacing the old silent drop. Sentinel
      gotcha: `parse_range` spells absent stride/bounds as BOTH -1 and 0.
      ffc `tools/oracle_literal_substring.f90` + `.md5`: byte-exact vs
      gfortran RUNS=24/24 then 20/20 after restore, falsified by off-by-one
      fold (md5 differs, `FALSIFIED_OK`). ffc suite fail-set identical
      (`perf/suite-w04t/u/v.fails` vs `suite-w04p.fails`). Original repro:
      `print *, 'abcdef'(2:4)` compiles rc=0 and prints `abcdef`; gfortran
      prints `bcd`. Cause located: `apply_index_postfix`
      (`fortfront/src/parser/expressions/parser_expression_arrays.f90:859-905`)
      bails via `if (.not. allocated(call_name)) return` - a literal base has
      no target name, so the `(2:4)` postfix is silently discarded. There is
      no substring_node; substrings are call_or_subscript + ranges +
      is_character_substring. Slice: (1) fortfront - when base is a literal
      and collected args contain a range, push call_or_subscript with
      base_expr_index=literal instead of returning base_expr; (2) ffc - fold
      literal-base constant substring ranges to a truncated literal at
      lowering (gfortran byte-exact oracle, RUNS>=20, falsified by dropping
      fold); (3) fortfront suite (763) + ffc suite fail-set identical vs
      e339 baseline.

- [ ] **#337/#338 retire legacy array-shape metadata** (largest cluster;
      835 XFAIL rows tagged #337: 511 lfortran + 324 dg). Row conversion
      only via gauntlet regeneration, never row deletion.
      **BLOCKED IN THIS WORKSPACE - exact cause (ffc #759).** The corpora
      these rows live in are not checked out here.
      `--suite lfortran` resolves to `${FFC_LFORTRAN_DIR:-<parent>/lfortran}`
      and `--suite gfortran-dg` to
      `${FFC_GFORTRAN_DG_DIR:-<parent>/gcc/gcc/testsuite/gfortran.dg}`
      (`scripts/conformance_gauntlet.sh:215-217`); `~/code/lazy-fortran/`
      contains only `ffc fo fortfront liric standard`, so both walk zero files
      - measured `lfortran: PASS=0 XFAIL=0 XPASS=0 FAIL=0 TOTAL=0` - and
      `conformance_check.sh` still exits **0** on that, which is a vacuous
      green and is filed separately. No #337 row can be counted or converted
      until a corpus is provided via those two env vars. Live paths:
      `fortfront-f90` TOTAL=645 PASS=376 XFAIL=0 XPASS=0 FAIL=0 and
      `fortfront-lf` TOTAL=267 PASS=219 XFAIL=47 XPASS=0 FAIL=1
      (`/var/tmp/ffc-goal/perf/conf-check-w1.log`); the single fortfront-lf
      FAIL is `array_index_brackets_*` bracket-extension syntax, proven
      pre-existing by rebuilding at ffc `86f55e5~1` and reproducing it there.
      XPASS=0 across both means my two landed print-lowering fixes
      (`86f55e5` gather print, `506d07b` substring concat) converted **no**
      gauntlet rows - they are correct and oracle-proven, but they do not move
      this cluster, so they are not evidence of #337/#338 progress.
- [x] **#339 retire legacy runtime-shape metadata — slices C+D**
      (A landed `245f976`, B landed `d0f798a`):
      **C AUDIT DONE / D PREMISE FALSIFIED — ffc `5d10a34`.** Guard
      `tools/assert_no_legacy_runtime_shape.py` passes rc=0 and is
      falsifiable (injected `rogue_shape_cache_write` → rc=1 naming
      `session_program_lowering_arrays.inc:349`; tree restored clean).
      Audit: 8 routines in `src/` assign `has_runtime_dim_size(...)`; exactly
      2 are descriptor-backed and both are legitimate producers on opposite
      sides of a call — `bind_assumed_shape_descriptor_params` (callee, reads
      extents OUT of an incoming descriptor) and `define_runtime_array_symbol`
      (caller, creates a descriptor at ALLOCATE). `bind_optional_assumed_shape_descriptor`
      is invoked *by* the canonical binder, so it is one route, not a rival.
      So there were **no legacy cache writes to delete** — deleting the
      `assumed_shape_descriptor.inc` ~:1108–1109 writes would have destroyed
      the only correct producer. The guard now pins the writer set to exactly
      that audited pair so a parallel shape cache cannot be reintroduced.
      Original slice text:
      C: verify producers — `bind_assumed_shape_descriptor_params` cache
      writes are the only writers for descriptor-backed symbols; spot-check
      pointer/allocatable sym kinds.
      D: **delete cache writes** for descriptor-backed symbols
      (`assumed_shape_descriptor.inc` ~:1108–1109) + add
      `assert_no_legacy_runtime_shape(symbol)` (descriptor-backed symbol
      must never be answered from the cache); helper fallback remains for
      descriptor-less producers (#334 sentinel, automatic arrays).
      Oracle `test_reduction_descriptor_extent_parity.sh` (4 shapes,
      RUNS≥20, falsified off-by-dim) reruns green per slice.
- [ ] **#348 character dummies/results by descriptor** (229+208 XFAIL rows).
      Landed: `34b2bc1` rank-1 fixed assumed-shape dummies; `71aa9fe`
      fixed-width result pad/truncate on assignment (#755 closed);
      `6458f90` runtime `len=`. Remaining: literal-base substring ranges
      (`'abcdef'(2:4)`) dropped at parse → **fortfront #3018** (two-repo
      slice; fix sketch in issue: `apply_index_postfix` must keep a
      literal_node base); XFAIL regeneration by owner process.
      **SCOPE NARROWED BY MEASUREMENT (ffc `881c961`).** The locally reachable
      runtime-length character surface was probed and pinned before touching any
      lowering: `trim(x)//y`, `trim//trim`, `repeat(x,n)//y`, `achar(c)//y`,
      `len(trim(x))`, `len(trim(a//b))`, nested `c(1)(2:3)//c(2)(1:2)`,
      `adjustl`/`adjustr` trims, and the part #348 is actually about -
      `character(len=*)` dummies, fixed-length dummies, multi-dummy concat,
      substring of a `len=*)` dummy, character function argument - **all match
      gfortran byte-exactly**. Oracle
      `tools/test_char_runtime_surface_parity.py`: `runs=22 match=22 refused=0
      mismatch=0`, 20 distinct digests, `perf/chrsurf/report.tsv`; falsified by
      making every case run `cs0_ffc` -> `match=1 mismatch=21`, exit **1**
      (clean exit 0, restored exit 0). Therefore the 229+208 rows are **not**
      in print/concat lowering; they are descriptor passing plus the absent
      corpora (#759) and fortfront #3018. Do not re-probe this surface.
- [x] **#399 vector subscripts as gather views.**
      **COMPLETE as scoped.** Slice 1 (ffc `86f55e5`, closes #757): routing-only,
      six lines in `emit_whole_array_expr_print_items` - recognize
      `is_fixed_rank1_vector_gather_candidate`, `materialize_fixed_rank1_vector_
      gather`, hand the cached temp to the existing elementwise printer. No new
      lowering. `print *, a(idx)` byte-exact vs gfortran. Slice 2: scatter target
      `a(idx)=b` **proven already correct** on first probe (`9 20 30 7`), so the
      work was a guard, not a fix (ffc `a46d33c`). Oracle
      `tools/test_vector_subscript_gather_parity.py` covers all three modes:
      `runs=36 match=36 mismatch=0`, plus gap `MATCH=6`, distinct md5 per
      permutation, falsified (`[1,2,3]` 03b31e82… vs `[2,3,1]` 8d364276…).
      Structural proof: full-suite fail-name set identical at w36/w37/w38. Recon done: read-gather
      machinery already exists (`materialize_fixed_rank1_vector_gather`,
      **SLICE 1 LANDED** (ffc `86f55e5`, closes #757): routing-only as
      scoped - six lines in `emit_whole_array_expr_print_items`: gather
      candidate -> materialise into the cached contiguous temp -> hand that
      symbol to the existing elementwise printer, no new lowering. Oracle
      committed at `tools/test_vector_subscript_gather_parity.py` (gfortran
      independent, byte-exact md5, RUNS=24 + 6 whole-array rows):
      `match=24 mismatch=0`, gap `MATCH=6`, distinct md5s per permutation;
      falsified ([1,2,3] `03b31e82...` vs [2,3,1] `8d364276...`). Suite
      fail-name set identical to `/var/tmp/ffc-goal/e339_base_failnames.txt`
      (`perf/suite-w36.txt` 500/489/11, wall 249 s).
      **LOCALIZED 2026-10-01** - ffc#757. Parity harness
      `/var/tmp/ffc-goal/bin/test_vector_subscript_gather_parity.py`
      (gfortran oracle, md5-compared, RUNS=24): the gather **read** path is
      already byte-exact - `runs=24 match=24 mismatch=0` over 12 index
      permutations (ranks 2-6, repeats/rotations/reversals) in both `b =
      a(idx)` and element-wise print. The gap is whole-array print items:
      `gap rows: REFUSED=6`
      (`/var/tmp/ffc-goal/perf/vspar-latest/gap.tsv`). First refusal emitted
      from `src/session_program_lowering_array_elements.inc:5105`
      (`lower_i32_array_subscript`, line 4998) because the print item asks for
      a scalar element address; the second, `print '(9i6)', b`, is a separate
      unsupported feature ("array reads require exactly one subscript") that
      made my first oracle mislabel 12 green gather cases as failures.
      **SLICE 2 PROBE: scatter TARGET already works** (oracle `a46d33c`):
      `a(idx) = b` is byte-identical to gfortran on the first probe
      (`9 20 30 7`), so the oracle now runs three modes over the same 12
      permutations - `runs=36 match=36 mismatch=0` - with scatter
      falsification ([4,1] `794aaa40...` vs [2,3] `12e9908b...`). Slice 2 is
      a regression guard plus slices/ranges off the false-reject list, not a
      new scatter implementation.
 **Slice 1 is routing-only**: print item matching
      `is_fixed_rank1_vector_gather_candidate` → materialize temp →
      existing whole-array print. Oracle
      `test_vector_subscript_gather_parity.sh` (print-gather /
      assign+print / repeated indices, RUNS≥20, falsified). Slice 2:
      vector subscript as scatter TARGET; slices/ranges off false-reject
      list.

#### W1.b — derived types, polymorphism, control flow

- [x] I/O implied-do lowering (print + internal read): flattened walk
      binds loop vars per value, unfolds nesting, expands format groups;
      internal read uses one sscanf with one conversion per target.
      Evidence: ffc `73e8a92`+`4c4c1fa` (print, oracle
      `tools/test_print_implied_do_parity.py` runs=21 match=21,
      `/var/tmp/ffc-goal/perf/pimdo/report.tsv`), ffc `31624a4`
      (internal read, oracle `tools/test_internal_read_implied_do_parity.py`
      runs=20 match=20, `/var/tmp/ffc-goal/perf/irido/report.tsv`,
      falsification recorded). Remaining named gaps: array-constructor
      print items (`wide_multi`), implied-do beside other items,
      fixed-width exact-field stance, stdin implied-do unit.
      Reshape-in-mask fold `9043ff7` (oracle
      `tools/test_reshape_expr_parity.py` runs=20 match=20,
      `/var/tmp/ffc-goal/perf/rsx/report.tsv`, falsification recorded)
      closed the last example gap: fortfront corpus FAIL 3->1 at suite
      `506 passed / 3 failed` (`/var/tmp/ffc-goal/perf/suite-rsx.log`);
      sole remaining corpus FAIL `issue_2455_array_constructor_arg.f90`
      is the correct-posture refusal pending #417 go-ahead.
- [ ] Assumed-shape derived-type dummies (8 live files: no
      compile-time-size whole-array actuals required). rank-1 + rank≥2
      runtime strides landed (`ce89457`, `04a50c4`, nested-write oracle
      `6cfb6a3`); remaining: whole-array SECTION actuals need caller-side
      materialization (`component_storage_rank2`).
- [ ] #422/#419/#449 polymorphic arrays, SELECT TYPE guards (cross-unit
      unsound — static single-type resolution refused), plain scalar
      derived values; #458/#459 core array decls + typed array ctors.
- [ ] #417 cross-unit `class(T)` dummy ABI: class descriptor box
      {data,type_id} — currently a visible refusal (correct posture);
      do not attempt without explicit go-ahead.
- [ ] #435 sized/array TRANSFER; #462/#465 defined-assignment recursion +
      elemental over arrays.
- [ ] #455 structured branch targets incl. named outer EXIT/CYCLE.
      **CONFIRMED wrong-output bug, localized, unfixed** (recorded 09-30
      at `f0d3f84`, silent miscompile, no diagnostic) — implementation
      job, not discovery job.
- [x] ffc **#756** (closed `53e8282`): `sum(x(2:3))` over assumed-shape dummy prints
      0; route section extent via `read_runtime_dim_extent` and give
      `sum` its genuine stride loop; pre-existing at baseline, byte-proof
      in issue.

#### W1.c — acceptance, provenance, gates

- [x] #473 F2023 delta audit → disposition table in
      `docs/SUPPORT_CONTRACT.md` (`supported`/`issue #N`/`out of scope`),
      one atomic issue per standard gap. The complete disposition table is
      `docs/F2023_DELTA.md`; 45 producer/consumer issues remain tracked.
      `uint`, ASYNC=, DEPENDENCY and Synthesis projections are outside F2023.
      Until then "F2023 compliant" is undefined.
- [ ] One locked four-suite provenance epoch (ffc+fortfront+corpora+
      toolchain digests), then regenerate `parity_dashboard.tsv`
      (corpora now present: lfortran 4356, dg 5938; dashboard pin
      `0639933` predates history re-root — stale by construction).
      Schedule AFTER W0.2 parallel runner so it costs minutes.
- [ ] #540 manifest-owner refresh; #532 XPASS reclassification from the
      epoch, never by row deletion.
- [ ] Rebaseline/retire umbrella trackers #576/#609 with live signatures.
- [ ] #478 load-dependent timeout case; #531 benchmark baseline record;
      #649 flake closure after epoch. (Corpus TIMEOUT #754 disappears
      with W0.2 parallel corpus walker + `slow` marking.)

#### W1.d — cross-repo contract work (fortfront / liric / fo / standard)

- [ ] Remaining lowering-family extractions waves 5–7 (I/O families,
      derived types/units, orchestration) — **merge into W0.4**: every
      extraction is now a real module; fix owned clusters while extracting.
- [ ] FortFront cross contracts blocking: #2883, #2897, #2951, #2970,
      #2996 (continuation-comment lexer), #2973 (implied-do I/O AST);
      #3018 (literal-base substrings, see W1.a/#348).
      FortFront known structural limits on the critical path:
      multi-entity declarations share ONE shape record
      (`ast_factory_declarations.f90:264`) — per-entity split is
      parser-side and gates consumer-side fixes.
- [ ] LIRIC #533 producer artifacts restored → #523 dominance/serialization
      gate trustworthy for ffc promotion. (Landed guard: i1→widen store
      only, `3636871`.)
- [ ] fo #117 formatter oracle, #119 unbounded JSON, #103 diagnostics;
      fo/format gates blocking. **Extend fo issue set with W0.2/W0.3
      work**: `clean --stale`, `FO_JOBS` runner parallelism, consolidated
      test routing (`fo test <name>` → dispatcher), `--gc-sections`
      link flags plumbed through profiles.
- [ ] Exact union/platform/sanitizer/ABI/perf release gates; monotonic
      zero-`.inc` check (fed by W0.4); every claimed feature keeps an
      executable byte-exact oracle (RUNS≥20, md5-recorded binaries).
- [ ] Fortran Synthesis chain (standard #756 → fortfront #2976 →
      ffc #632 → fo #120) starts only after normative syntax/semantics
      accepted.

#### W1.e — open diagnostics/misbehavior cluster (older, still open)

- [ ] Symbol collisions — **direction corrected by measurement, filed
      fortfront#3021**. Prior plan §legacy recorded this as over-*refusal*
      ("false positives"). It is over-*acceptance*: four shapes of invalid
      Fortran compile and run under ffc while gfortran rejects them —
      `integer :: shared` beside use-associated `type :: shared` (prints `9`),
      `integer :: bucket`, `common /g/ shared`, and `do shared = 1, 2`
      (gfortran: "Derived type 'shared' cannot be used as a variable").
      fortfront's `check_common_and_construct_names` covers only the COMMON
      **block name** and the do_loop **label**, both verified correct against
      gfortran; the **object-name** half (variable / loop variable / COMMON
      member) is absent, so those shapes reach lowering. Guard
      `tools/test_name_namespace_collision_parity.py` (ffc `5836119`):
      `runs=24 agree=24 disagree=0 known_overaccept=4`,
      `perf/nscoll/report.tsv`, falsified → 19 disagreements exit **1**.
- [ ] Signed/unsigned mixing acceptance (`uint()` family — extension gate, outside F2023).
- [ ] Walrus `:=` redeclaration false positive in LF mode.
- [ ] #584 one FortFront binding identity across host/ASSOCIATE/USE.
- [ ] Versioned `.fmod` schema (stable declaration/procedure identities);
      cross-unit spec-part resolution landed (`63fc564`) is the base.
- [ ] Nested internal procedures support (refusal today).
- [ ] #437→#433 lazy specialization emit/serialize after FortFront lands.

---

### Standing rules (unchanged; binding on every slice above)

1. Every behavioral change carries its own oracle, byte-exact vs
   gfortran, RUNS≥20, md5-recorded, and a shown falsification
   (revert → oracle fails). Structure-only commits (W0) require the
   full-suite fail-name set to be identical instead of a new oracle.
2. Never weaken/delete a test or XFAIL row to go green; baselines only
   by regeneration scripts.
3. ABI changes: full `fo test --all` + `scripts/corpus_rejection_gate.sh`
   once, zero NEW failures vs baseline
   (`/var/tmp/ffc-goal/baseline_fails_complete.txt`, gate 705/213).
4. Contract docs (`RUNTIME_ABI.md`, `ARRAY_DESCRIPTOR_ABI.md`,
   `SUPPORT_CONTRACT.md`) updated in the same commit as the semantic/ABI
   change.
5. One adversarial self-review per commit (fix vs mask, blast radius,
   can-oracle-pass-with-defect, similar sites); blocking findings →
   follow-up commit, max 3 rounds.
6. Build: `fortfront && fpm build`, then `fo build` with
   `LIBRARY_PATH=$PWD/../liric/build`.
7. Scratch in `/var/tmp` only. `/tmp` is RAM.
8. A compiler that silently segfaults is worse than one that visibly
   rejects: never convert a compile-time rejection into a runtime crash.
9. Full suites/gates/gauntlet NEVER ad-hoc-in-loop: once per slice in
   background, only at ABI/phase boundaries otherwise.
10. Cross-unit class-dummy ABI (#417) and any parser-reachable
    cross-unit dispatch semantics: explicit user go-ahead required.
11. Protected hosts rule remains absolute (`faepop*`/`faepcr*`;
    `faepmac1/2` exempt per explicit instruction).

### Evidence ledger (read-only pointers)

- suite baseline fails: `/var/tmp/ffc-goal/e339_base_failnames.txt`
- slice logs: `fo_all_e339_base.log`, `fo_all_e339_v17.log`,
  `fo_all_e339B_v18c.log` (+`_faildiff.txt`) in `/var/tmp/ffc-goal/`
- gate: `/var/tmp/ffc-goal/gate_*_v1*.tsv` (705/213)
- perf baseline for W0: this file's "Baseline measurements" table +
  `/var/tmp/ffc-goal/perf/` (create at W0.1 with `du`/`time` captures)
- probes: `/var/tmp/ffc-goal/{v1,rre1,rre2,rre5,rreB,cc,cprefix}.f90`
- **W0 slice (2026-10-01)**: shas fo `fc73029` `6ed00e4` `3561b50`, ffc
  `183d86b` `612f125` `984bf07`; decomposition + attribution
  `/var/tmp/ffc-goal/perf/w0_slice_evidence.md`; suite wall
  `/var/tmp/ffc-goal/perf/suite-w04.txt` (243 s, 500 names, 488 pass);
  fail names `perf/suite-w04.fails` + `perf/suite-w04.faildiff`;
  gate `/var/tmp/ffc-goal/perf/gate-w02.log` (918 files, 705/213, 24 s);
  edit rebuild `perf/rebuild-edit-w02.log` (12 s) and
  `perf/edit-test-w02.log` (120 s); sharding oracle
  `/var/tmp/ffc-goal/shard/{C.obs,D.obs}` (identical records, serial vs
  `--jobs 6`); fo issues #129 (compdb cannot see link-stage evidence),
  #130 (suite wall bounded by serial children, heartbeat noise, deleted
  logs).

---

## Historical checkpoints

### Checkpoint of 2026-10-01

These records describe the earlier checkpoints. The current status above
supersedes their repository revisions, measurements, and completion claims.

#### Close-state (authoritative; supersedes the sha/metric tables below where they differ)

All five repos clean and pushed, `git log @{u}..HEAD` empty:

| repo | HEAD | added since the table below |
|---|---|---|
| fo | `fd4b818` | `fo clean --help` made inert `f0e99f` (fo#136) + test-harness fixes `fd4b818`. fo suite **rc=0, 40/40 by name** (`logs/fo-suite-w45.log`) |
| ffc | `729cace` | `Iw.m`/`Aw` format fix `f11ff2f`; **#761 logical/comparison print fix `5c3341b`**; guards `13c2f2e` `d7e3cad` `43630ae` `96f4692` `5836119` `665c70b` `58825d6` `edea217`; misleading-`MASK=` diagnostic fix `729cace` |
| fortfront | `8f35952` | no code change; filed #3018, #3019, **#3020 AST dump**, **#3021 over-acceptance** |
| liric | `9042548` | #535 closed (`-g0` scoped to SLEEF sub-build) |
| standard | `160032a` | unchanged |

**New defects found by probing vs gfortran, all with reproduction + measured spec:**
ffc#764 (`index`/`scan` ignore `back=.true.` → wrong index on valid code),
ffc#765 (`B`/`O`/`Z` unsupported, incl. measured fact that **`Z` blank-pads, does not
zero-pad**, and out-of-range incl. `-1` prints width-many `*`, not the bit pattern),
ffc#766 (`MASK=` unsupported on sum/maxval/minval with a **diagnostic that blames
correct arity**; `count(mask=)` separate root cause; `merge` works **only** for literal
scalar args; nested `abs` inside a reduction refuses on INTEGER demanding a real array),
fortfront#3021 (**nine** invalid-program shapes silently accepted, worst being
`integer, parameter :: k=3; k=4` which ffc **executes and prints `4`**),
fortfront#3020, fo#133/#134/#135/#136.

**#761 is the one behavioral compile-error→correct-output fix landed**: `print *, 1<2`
went from refusal to `T`, verified against a **rebuilt HEAD~1 control** — 918 corpus
files compared, **0 changed verdicts** (`/var/tmp/gate-w42-prefix.tsv` vs
`/var/tmp/gate-w42.tsv`). `729cace` is diagnostics-only by design and was likewise
controlled: with the source reverted to HEAD~1 and rebuilt, `fortfront-lf` reports the
identical `FAIL=1` on `read_statement.lf`, proving that failure pre-exists this run.

**LANDED (ffc `5b4eddf`, `a77f418`, `691395d`)**: ffc#764 closed — `BACK=` honored in
`index`/`scan`/`verify` (oracle `tools/test_string_search_back_parity.py`: runs=39
match=36 known_refused=3, falsified: deleting the index condbr surfaces 10 REGRESSION
rows; `perf/suite-w46.txt`, `perf/gate-w46.tsv`). ffc#766 half landed — `MASK=` on
sum/product/maxval/minval/count over fixed-size rank-1 arrays (logical-array and
comparison masks, f32/f64/i32, empty-selection answers gfortran's measured ±HUGE);
DIM=/KIND= stay refused with the pre-#766 pinned wording (fc diagnostics tests pin
strings verbatim). Oracle `tools/test_mask_reduction_merge_parity.py`: runs=42
match=36 known_refused=6, falsified: dropping `mask_index` in the resolver surfaces 14
defects (`perf/mask/report.tsv`). `merge` and nested `abs` remain refused as filed.

**Methodology error recorded, because it produced a false control:** my first attempt
to control `729cace` used `git stash push src/session_program_lowering_arrays.inc`
— but that change was **already committed**, so the stash saved nothing, the
"pre-fix" rebuild recompiled the same code, and both runs reported `FAIL=1` for the
trivial reason they were the same binary. A control that reproduces the observed
number is not a control until you have shown the compared artifacts differ; the valid
version was `git checkout HEAD~1 -- <file>` + rebuild + confirming the binary actually
emitted the old message before running the suite.

#### Landed shas per repo (all clean, `git log @{u}..HEAD` empty in all five)

| repo | HEAD | this run's commits |
|---|---|---|
| fo | `f7dd575` | `-g0` diet `fc73029`; gc-sections + `fo clean --stale` `6ed00e4`; parallel runner `3561b50`; dispatcher routing `0526874`→`1785803`; shard allowance `63f1641`; PIC `839e139`; PIC live path + `shared_library()` `8dfbc80`; `link="shared"` e4fccad; shardable predicate f7dd575 |
| ffc | `506d07b` | manifest `183d86b`; sharding `612f125`→`5cea955`; gate baseline `fbd62eb`; dispatcher `a12374a`; folding falsified `73dbcd5`; PIC `e4756c4`; shared-link contract `da1b601`; **#399 gather print `86f55e5`**; scatter guard `a46d33c`; **#339 guard `5d10a34`**; wall stamps `2bfeaa7`; **#348 substring concat `506d07b`** |
| liric | `9042548` | `-g0` scoped to SLEEF sub-build (closes #535) |
| fortfront | `8f35952` | no code change; issues filed #3018, #3019 |
| standard | `160032a` | unchanged |

#### Metrics achieved vs targets

| target | result | evidence |
|---|---|---|
| `ffc/build` < 2 GB | **MET — 147 MB (0.137 GiB) cold**, 13× under target. The 1.7–2.0 GB figures quoted earlier were a long-lived tree's accumulated cruft (1045 MiB regenerable test binaries in `build/fo/bin`, 814 MiB in `build/fo/lib`), not the build | `logs/cold-build-w43.log`; wiped + rebuilt cold, `du -sh` = 147234816 B |
| rebuild after single-domain edit < 60 s | **MET** (12 s edit-rebuild; 33 s cold lib) | W0.1 table in `perf/w0_slice_evidence.md` |
| rejection gate < 90 s | **MET 34.25 s**, 705 accepted / 213 rejected, **0 newly rejected** | `perf/gate-full-w38.log`, `perf/gate-w38.tsv` |
| every old test name runnable + by name | **MET** 500 names, fail-name set identical | `perf/suite-w38.txt` vs `e339_base_failnames.txt` |
| suite wall < 120 s | **NOT MET** (235–246 s) — proven bounded; **user waived: "we are fast enough"** | `perf/suite-w39-warm.txt` (warm 235 vs cold 244 ⇒ rebuild worth ~9 s); floor = one 145.12 s test in `suite-w37.results` |
| 0 `.inc` | **NOT MET** 42 remain (was 51); next unit reclassified + blocker recorded | W0.4 block above |
| per-test binary shrink | **MET** 15.6 MB → 2.1 MB mean (7.4×), 501 bins | `perf/suite-w38.txt` |
| `--debug`/`--asan` symbolizable | **MET** — asan backtraces resolve file:line through fortfront→ffc→test | `/var/tmp/asan-run.log` |

#### Correctness work landed (each with its own gfortran-oracle, RUNS≥20, falsified)

- **#399 slice 1** `86f55e5` — route whole-array vector-gather print items.
  `runs=24 match=24 mismatch=0` + gap `MATCH=6`; falsified `[1,2,3] 03b31e82…`
  vs `[2,3,1] 8d364276…`. Harness `tools/test_vector_subscript_gather_parity.py`.
- **#399 slice 2** `a46d33c` — scatter target `a(idx)=b` already correct;
  oracle widened to 3 modes `runs=36 match=36`; falsified `794aaa40…` vs `12e9908b…`.
- **#339 slices C+D** `5d10a34` — audit **falsified the premise**: 8 extent
  writers, exactly 2 descriptor-backed and both legitimate; nothing legacy to
  delete. Guard pins the pair; rogue injection → rc=1 naming the offender.
- **#348 slice** `506d07b` — `is_character_operand` missed substrings, so
  `print *, a(2:4)//b(1:2)` fell to the array path. `runs=24 match=24`,
  24 distinct md5s; falsified `d5a07812…` vs `4fd16245…`.
- Structural proof throughout: full-suite fail-name set identical to
  `e339_base_failnames.txt` (500/489/11) at w36, w37, w38.

#### #3018 attempt: falsified the sketch's location, reverted (last turn)

Tried the fix sketch for fortfront#3018 (literal-base substring ranges).
Implemented in `apply_index_postfix`: guard the early `if (.not.
allocated(call_name)) return` so a character `literal_node` whose args contain a
`range_expression_node` gets `push_array_slice(arena, base_expr, arg_indices,
size(arg_indices), …)` instead of returning the bare base; plus `any_has_range`
helper. fortfront built clean, ffc rebuilt, **behaviour unchanged**
(`"abcdef"(2:4)` still `abcdef` vs gfortran `bcd`, also for `len()` and variable
bounds). So the sketch's stated drop point is wrong, or `extract_target_name`
returns a name there and ffc's consumer drops the slice. **Patch reverted** -
unproven code in a parser is worse than the bug, and the defect reproduces
identically at baseline after revert+rebuild. Evidence and next diagnostic in
fortfront#3018 comment; do not re-attempt the same edit.

Two byproducts worth keeping:
- **fo#135 filed**: `fo build` relinks without recompiling changed *path
  dependency* sources - source mtime 23:12, `.mod` still 19:10, binary relinked
  23:12:56; deleting that unit's `.o`+`.mod` forces recompile (23:14, rc=0).
  Makes a correct fix look broken and silently invalidates any measurement.
  Same family as fo#133.
- **No diagnostic route exists for #3018; two dead ends checked.**
  `fortfront --trace` produces nothing about slices on this input, and
  `fortfront/app/debug_ast` looks like the answer but is **not** - see next
  bullet. Filed as **fortfront#3020** (ask: dump the arena actually built,
  reusing the serializer `debug_ast` already has). Until that exists the
  parser-vs-consumer question for #3018 is unanswerable from outside, and the
  next attempt should start from #3020, not from another edit to
  `apply_index_postfix`.
, not a dumper: it prints a
  fixed AST ("Sum negative:", literals 5/3/1) regardless of input. It looks
  like real output and nearly produced a false conclusion here. There is
  currently no working AST dump for #3018 diagnosis - that is the actual gap.

#### Quiet-machine retake + a self-correction (final close)

Machine went idle (load1 2.09, startup-bench finished), so the wall numbers
were retaken at the SAME `FO_JOBS=12` as the loaded baselines - same config,
idle box, so any delta is attributable to load and nothing else.

- **Suite wall quiet: 226 s** at load1 5.20 (`perf/suite-w40-quiet.txt`),
  vs 244 s at load1 8.30 and 235 s warm. Load was worth only ~7 %. So the
  wall is NOT a contention artifact: on an essentially empty 32-core machine it
  is still 226 s, and the floor remains the single 145.12 s test. The bound
  conclusion is now measured, not inferred. Fail-name set identical (500/489/11).
- **Gate quiet: rc=0, 705 ACCEPTED / 213 REJECTED**
  (`perf/gate-quiet.tsv`, `perf/gate-quiet.log`).
- **SELF-CORRECTION.** An earlier line here claimed "0 newly rejected" and was
  computed against the wrong TSV column - status is field 1 (`ACCEPTED` /
  `REJECTED`), not field 2 (path), so that check was comparing paths to
  statuses and proved nothing. Redone properly: a naive diff shows 42 rejected
  rows not in the baseline's rejected set, and all 42 are corpus files absent
  from the baseline entirely; the regression test that matters - files
  `ACCEPTED` in baseline and `REJECTED` now - is **0**. The claim holds, but it
  was not established by the earlier arithmetic and is restated here with the
  method that establishes it.
- **Baseline is stale as data, harmless to the verdict**: it carries 833 rows
  (626 A / 207 R) while 918 files are walked (705 A / 213 R). The gate passes
  because no previously-accepted file regressed; the 85-file growth is simply
  unrecorded. Refreshing `test/fixtures/corpus_rejection_baseline.tsv` is
  maintenance, not a fix, and must not be done by dropping rows.

#### Verifier-integrity fix landed (this close)

- **ffc `ffabc1d` closes #759** — `conformance_check.sh --suite NAME` exited
  **0** on a missing corpus (`TOTAL=0`). Now exits 2 with `NO_CORPUS: <suite>
  <root>`; unknown names rejected too. Verified: absent-corpus rc 0 → **2**
  naming `/home/ert/code/lazy-fortran/lfortran/integration_tests`;
  non-regression on a suite whose corpus exists is unchanged at
  `fortfront-f90: PASS=376 XFAIL=0 XPASS=0 FAIL=0 TOTAL=645` rc=0
  (`perf/conf-f90-after759.log`). Root resolution hoisted to `suite_root()` so
  the explicit and discovery paths cannot drift again. This matters beyond
  tidiness: every "all suites clean" line in this file's history was vacuous
  for lfortran/dg until now.

#### Blocked items — exact blocker

1. **#337/#338 (835 rows) + lfortran/dg share of #348**: corpora not present.
   `../lfortran` and `../gcc/gcc/testsuite/gfortran.dg` are absent; only
   `fortfront-f90` (645) and `fortfront-lf` (267) walk files here.
   Also filed: **ffc#759** — `conformance_check.sh` exits **0** on `TOTAL=0`,
   so "PASS: all suites clean" above is a **vacuous green**, not a pass.
2. **#348 literal-base substring ranges**: needs fortfront **#3018**.
3. **fortfront leaks/UB**: fortfront **#3019** (scope_stack 144 B leak;
   `identifier_table.f90:134` signed overflow).
4. **Stale-binary trap**: `fo build` does not refresh `~/.local/bin/fo`
   (**fo#133**) — invalidates any number taken without `fo install`; all
   headline numbers above were retaken after install at 19:29.
5. **Wall-cap watchdog**: **fo#134** — false "deadlocked" accusation on
   supervising tests; standalone `gauntlet_smoke` genuinely blocks (child
   finished in 11 s, parent idled 89 s). My measured `-g0`-style fix attempt
   was reverted, not shipped, because it could not be shown to fire.
6. **W0.4 `0 .inc`** — mechanical route **provably exhausted**, not merely
   hard. Measured all 42 remaining files by outbound-`call` direction: none has
   zero outbound calls into the host module (min 9, `submodules.inc` 13), while
   all nine lifted units had zero. They are one mutually-recursive scope, so
   reaching 0 requires breaking call cycles (the layered-DAG split), not more
   include folding. My earlier "type provenance unresolved" note was a grep
   artifact and is retracted — `submodule_node` is in fortfront
   `src/ast/nodes/ast_nodes_data.f90`, imported by ffc at
   `session_program_lowering.f90:265`.
7. **ffc#764 `back=.true.` ignored** (`index`, `scan`) — wrong index on valid code.
   Forward results agree, so it is invisible until a caller asks for the rightmost
   match. Guard pins 7 rows as `KNOWN_WRONG_ANSWER`; `verify` honours `back`
   correctly, so the defect is per-intrinsic and an `index`-only fix would leave
   `scan` broken and green.
8. **ffc#765 `B`/`O`/`Z` unsupported** — not a routing fix: there is **no `%b` in C
   printf**, and the width-overflow star rule must run before any `%o`/`%X` route or
   `-1` prints `FFFFFFFF` where Fortran wants `****`. Measured spec is in the issue.
   `G`, `:`, `SP`/`SS`, `P` also refuse honestly (coverage, not wrong answers).
9. **ffc#766 `MASK=`/`DIM=` unsupported on reductions** — the arity half of the
   diagnostic is FIXED at `729cace`; the missing keyword plumbing is not. `merge`
   folds only literal scalar arguments; nested `abs` in a reduction refuses on INTEGER.
   `count(mask=)` fails in keyword-argument resolution (`AST node is not an
   identifier`) on a different code path, deliberately left untouched here.
10. **fortfront#3021 over-acceptance (9 shapes)** — filed, unfixed. Direction was
    inverted in the original W1.e wording: fortfront over-**accepts** invalid code
    (incl. mutating a `parameter`), it does not over-refuse valid code. Needs
    semantic validation on the fortfront side; deliberately not attempted here because
    narrowing acceptance can newly reject valid corpus programs and that needs a full
    suite + gate cycle in two repos, which I did not want to land unverified.
11. **`fortfront-lf` gate `FAIL=1` on `read_statement.lf`** — **pre-existing**, proven
    by a rebuilt HEAD~1 control (identical `FAIL=1`, same file). It is a `read *, x`
    case, so stdin availability in a non-interactive gate run is the suspect cause.
    Left failing rather than added to the XFAIL manifest, because it has not been
    root-caused; promoting it would be hiding it.

**Net: every unmet target is either waived by you (suite wall) or has a measured
structural cause (`.inc` call cycles; absent lfortran/gcc corpora; fortfront
#3018/#3019/#3021 needing semantic validation, `B`/`O`/`Z` needing a binary emitter and
overflow rule). All W0 targets that were reachable landed, and the `ffc/build` number
is now correct at 147 MB cold rather than the ~2 GB cruft figure previously quoted.**

#### Negative results

- **In-process test folding falsified**: 96 cases folded → 3 pass, 93 crash
  while all pass standalone (`73dbcd5` reverted, 1-case state retained).
- **My two print-lowering fixes converted ZERO gauntlet rows** (XPASS=0) —
  correct and oracle-proven, but not #337/#338 progress.
- **Shard widening is counterproductive**: JOBS=4 → 123.76 s, JOBS=16 →
  129.35 s.

##### Diagnostic-route findings (fortfront#3020)

`ffc --help` exposes only `-o -c -I <obj> --backend default|isel|copy-patch|llvm
--json --version` - `--json` is diagnostics, not the tree. `fortfront --trace <f>`
emits nothing mentioning slice/range/substring for `'abcdef'(2:4)`.
`fortfront/app/debug_ast <f>` returns a **fixed** 51-node demo
(`"Sum negative:"`, literals 5/3/1) whatever the input. Three routes tried, all
three useless; that is the recorded state, and it is why the #3018 patch was
reverted instead of guessed at further.

##### #761 FIXED: list-directed comparison/logical printing (`ffc 5c3341b`)

`print *, 1<2` was **refused** ("unsupported integer operator: ... supports +, -,
*, and /") because a comparison reached the integer arithmetic lowerer. Every piece
already worked - `(L1)` printed `T` for integer and character comparisons, a bare
logical variable printed `T`, and the user-overloaded branch beside the patch site
already called `lower_print_logical_value` - only the list-directed route for an
**intrinsic** logical-valued operator was missing. `print *, x > 0` is ordinary
Fortran and did not compile. Fix is routing only, two sites, no new lowering; the
operator predicate is stated locally per submodule because
`is_comparison_operator`'s cross-submodule visibility is unproven (assumed
visibility here is just a build break).

**Hazard caught mid-fix:** an intermediate state printed `1`/`0` instead of `T`/`F` -
routing the *value* without the *printer* turns an honest refusal into a wrong
answer. Oracle pins the spelling; every row also carries arithmetic in the same
statement since the new branch sits in front of the arithmetic lowerer.

Oracle `tools/test_logical_print_parity.py`: `runs=36 match=34 refused=0
mismatch=0 ref_fail=0`, 4 `KNOWN_GAP` (`perf/logprint/report.tsv`), falsified -> 13
MISMATCH exit **1**; restored exit **0**. Verification: **suite w42-logical
500/489/11, fail-name set identical to baseline** (`perf/suite-w42-logical.txt`,
244s, mean binary 2.1 MB); **gate rc=0, 705/213** on both pre-fix and post-fix binaries
(`perf/gate-w42.tsv`, `perf/gate-w42-prefix.tsv`).

**Correction recorded:** I first claimed "0 newly accepted" from a `comm` against
`test/fixtures/corpus_rejection_baseline.tsv`, then produced a bogus
"36 newly accepted" from the same bad comparison. That fixture has **833 rows
(626/207)** while the live corpus run has **918 (705/213)** - different scopes, so
neither number measured my change. The valid control is the **pre-fix binary built
from `HEAD~1`** on the identical corpus: `join` on path gives
**918 files compared, 0 changed verdicts** (`/var/tmp/changed.tsv`, reproducible
from `gate-w42-prefix.tsv` vs `gate-w42.tsv`). A malformed `join -j2` on a
path-first stream joined on the status field and briefly reported 11314 changed -
a tool artifact, discarded. Bare list-directed comparisons simply do not occur in
this corpus, which is why the fix leaves all 918 verdicts untouched.

Also cleared as **not a compiler defect**: `examples/f90/control_flow_associate_construct.f90`
output differs between runs of the *same* binary - and under **gfortran too** (3
runs, 3 distinct digests). The example prints uninitialised values; the source is
nondeterministic, so it must never be used as an output-parity reference. Contract section added in the same commit.

Still open on #761: unary `.not.` (`print "(L1)", .not.(1>2)` prints `T`, list-
directed refuses; no unary-op query exists at the patch site). New wrong-answer
defect filed as **#763**: `print *, 1.lt.2` prints `   1.00000000`, the left
literal - a real literal may end in `.`, so dotted operators mis-split.

##### Non-unit lower bound gap filed (`ffc#762`); allocatable surface pinned (`ffc 43630ae`)

`allocate(a(2:4))` on a rank-1 allocatable is refused although
`docs/SUPPORT_CONTRACT.md` promises rank-1 runtime allocate plus bounds access:
gfortran prints `10 30 2 4 3`, ffc refuses, and the identical program with
`allocate(a(3))` **works** - the explicit lower bound is the only difference. Two
defects at one site: coverage, and a **misleading diagnostic** ("only supports
integer expressions") that names the wrong construct, same family as fo#134's
misattribution. Offset bases are exactly where a parallel shape ledger and a real
descriptor disagree - `base + (i-lbound)*stride`, not `base + (i-1)*stride` - so
this is coverage the #339/#348 retirement currently lacks.

Guard `tools/test_allocatable_intent_optional_parity.py`: `runs=24 match=24
fail=0 known_gap=5 ref_fail=0` (`perf/alio/report.tsv`), falsified -> 23 MISMATCH
exit **1**; restored exit **0**. Five refusals kept by name: three `lb_offset_*`
(#762) plus scalar `character, allocatable` and rank-2 whole-array `reshape`, which
refuse honestly and are listed for prioritisation, not claimed as violations.

##### `fo clean --help` fixed (`fo f06e99f`); W0 build-size evidence corrected

**Defect:** `fo clean --help` printed no usage and **deleted the build tree**
(`build tree cleared: .../ffc/build`). `cmd_clean`'s flag loop knew only
`--cache/--all/--stale/--keep` and silently ignored every other argument, so
`--help` fell through to the destructive clean. The flag typed to ask what a command
does was the flag that destroyed the build, and because incremental recovery is fast
there is no loud signal. Fixed: `--help`/`-h` matched before any filesystem effect;
plain `fo clean` and `--stale` still delete, so it is not an over-correction that
makes `clean` inert. Test `fo/test/test_clean_help.f90` exercises the **binary's**
arg parsing (13 checks) — calling the library directly could never have caught it —
and is **falsified by disabling the guard: 8 checks fail, exit 1**; restored PASS.
fo#136. Installed so `~/.local/bin/fo` carries the fix (fo#133).

**W0 metric corrected by measurement.** I had been reporting `ffc/build` from a
long-lived tree showing **2.0 GB** (exact: 1.921 GiB / 2.063 GB — over 2 GB in
decimal units), and earlier still quoted a stale 1.27 GB. Wiping the tree and
rebuilding cold gave the number that actually answers the criterion:

    cold `fo build` from empty:  wall=13.49s rc=0   logs/cold-build-w43.log
    clean `ffc/build`:           147234816 B = 0.137 GiB

So a **normal** build is **147 MB**, 13x under the 2 GB target; the 2.0 GB figure
was accumulated cruft, `build/fo/bin` alone holding 1045 MiB of regenerable test
binaries plus 814 MiB in `build/fo/lib`. Two facts follow, and they are separate:
the size target is met with margin by the build itself, and **suite artifacts are
never reclaimed**, so a tree used for weeks grows ~14x past its clean footprint —
that reclamation is `fo clean --stale`'s job, and `--help` was the one `clean`
invocation that could not be trusted until this commit.

##### Descriptor + reduction coverage mapped by measurement (`ffc#765`, `ffc#766`, guard `ffc edea217`)

**Support map, type-corrected.** An earlier pass paired `(A3)` with an integer and
`(B8)` with the wrong data, so its refusals were my probe's bugs, not the compiler's.
With declaration, assignment and print separated per case: `I`, `Iw.m`, `F`, `E`,
`ES`, `EN`, `A`, `L`, repeated/nested groups and `nX` inside a compound format are
**accepted and byte-exact**; `B`, `O`, `Z`, `G`, `:`, `SP`/`SS` and `P` scaling
refuse honestly. `ffc#765` carries the measured gfortran spec, which overturns two
assumptions I had: **`Z` right-justifies with blanks, it does not zero-pad** (`%04X`
would be wrong), and an out-of-range value — including `-1` — prints width-many
`*`, **not** its two's-complement pattern, so the overflow test must precede any
`%o`/`%X` route. `B` has no C printf conversion at all.
**`ffc#766`:** `sum/maxval/minval(a, mask=…)` refuse with "requires exactly one array
argument" when arity is *correct* and the unsupported part is the keyword — following
that message deletes the mask and turns 12 into 15 with no diagnostic left.
`count(mask=…)` fails differently (`AST node is not an identifier`) while positional
`count(a>2)` works, localising it to keyword resolution. `merge` unimplemented.
**Nested `abs` is a fourth gap:** `maxval(abs(a))` on INTEGER demands a real array
while `abs(-5)` and `maxval(a)` each work alone.
Guard `tools/test_mask_reduction_merge_parity.py`: `runs=28 match=16
known_refused=12 unexpected=0` (`perf/mask/report.tsv`), falsified → 12 unexpected
exit **1**. Six numeric rows that already match are pinned as regression guards
(`mod` sign convention, `int`/`nint`/`floor`/`ceiling` on -2.7, int8 HUGE, overflow
wraparound) so later mask work cannot silently break working arithmetic.

##### String-search `back=.true.` ignored — wrong answers, `index` AND `scan` (`ffc#764`, guard `ffc 58825d6`)

Wrong answer on **valid** code: `index("abcabc","abc",back=.true.)` gives `1` where
gfortran gives `4`; `index("zabzab","ab",back=.true.)` gives `2` vs `5`;
`scan("aabbc","ab",back=.true.)` gives `1` vs `4`. Forward results agree, so this
hides until a caller asks for the rightmost match — extension stripping, last
separator, suffix tests — where the boundary is silently wrong.
**Correction to my own filing:** `verify` honours `back` correctly, so the defect is
per-intrinsic; a fix aimed only at `index` would leave `scan` broken and green.
Guard `tools/test_string_search_back_parity.py`: `runs=26 match=19 known_wrong=7
regressions=0` (`perf/ssback/report.tsv`), falsified → 7 regressions exit **1**;
scores an honest refusal of the 3-arg form as progress. Test-design trap recorded: a
`character(len=8)` needle holding `"ab"` is `"ab      "`, never occurs, and scored
the bug as MATCH — needles are now literal or exactly sized.

**#3021 widened to nine invalid-program shapes** (`ffc` guard commit, this entry):
a second probing pass found five more programs gfortran rejects and ffc runs —
mismatched `end do`/`end program` labels, duplicate declaration, wrong `len` arity,
and worst of all `integer, parameter :: k=3` then `k=4`, where ffc **performs the
assignment and prints `4`** — a semantic guarantee dropped, not a missed syntax check.
Oracle now `runs=29 agree=29 disagree=0 known_overaccept=9` (`perf/nscoll/report.tsv`).
**Two of the nine now closed by refusal, 2025-10-02:** the named-constant
assignment family in ffc (`8d4d600` guards + `b4a8ded` dummy-exclusion after a
50-test over-rejection was caught by the full suite, `perf/suite-3021.txt` 61
fail -> `perf/suite-3021b.txt` 489/11 fail-set IDENTICAL to accepted); oracle
`tools/test_named_constant_assignment_parity.py` runs=25 match=11 refused_agreed=14,
falsified: guards stubbed -> regressions=14, restored -> 0 (`perf/paramassign/report.tsv`).
END PROGRAM name mismatch in fortfront (`26ccedf`): nscoll known_overaccept 9->7
(both halves of the endprog shape); same suite also fixed the flaky
`test_issue_1572_namelist` ENOENT by moving the orphan program into the build
graph (`be73890`), fo#137 filed for the discovery-vs-build depth defect.
Fortfront suite green 763/763 rc=0 (`logs/ff-suite-1572fix.log`) - first full
green there. LEN arity refusal in ffc (`5ca5af4`): len(s,3,4) refused,
nscoll known_overaccept 6, full suite 489/11 fail-set IDENTICAL
(`perf/suite-len.txt`). dup_decl refusal in ffc (`1f91422`): duplicate
declarations in one scoping unit refused via origin_node_index scan
(BLOCK shadowing byte-exact 2/1); nscoll known_overaccept 5
(`perf/nscoll/report.tsv`). dup_decl `1f91422` REVERTED (`30ef7f5`): the
name+scope+origin scan refused interface dummies shared between interface bodies
and their procedures - 184/500 suite fail (`perf/suite-w04f-inquire.txt`);
dup_decl stays known over-acceptance (nscoll 6). inquire.inc folded to submodule
`session_program_lowering_inquire` (`640d10c`, 37->36 .inc): tool lead-check fix
(comment-only leaders were false-refused) + keyword-stem guard (bare
`submodule (M) inquire` breaks fo ordering - parent .smod never generated);
verification suite w04g 489/11 fail-set IDENTICAL to baseline (perf/suite-w04g.txt).
Folds batch2+3 (`38b2c3a`,`1b465b3`): 13 files (vector_subscript, complex_arrays,
statement_function, read_al, proc_dummy, assumed_shape_extent, read_ops,
logical_reduction, save_static, equivalence, polymorphic, common, save) -> 36->23
.inc; clean build foldE; suites w04h/w04j 489/11 (w04h carried one
conformance_execution_evidence load-flake, PASS standalone; w04j fail-set
IDENTICAL to accepted, `perf/suite-w04j.txt`). W0 re-measure after folds:
build=1.5G<2GB, single-edit rebuild 12.4s<60s (`perf/w0-single-edit.txt`). Compound fold (`f646699`, 23->22 .inc): tool
now accepts submodule .f90 includers; suite w04k 489/11 fail-set IDENTICAL
to accepted (`perf/suite-w04k.txt`). `tools/inc_to_submodule.py` fixes (`76ec57a`): indent-flexible header count
(fragments carry indent-0 and typed/recursive rettype headers - the earlier
"11 header/terminator-mismatch" diagnosis was a regex artifact: continuation-
joined scans show 0 true orphan tails in all 22 files) + cycle tolerance
(cycles need only parent-module interface visibility, not callee-first order).
Fold attempts of allocatable/char_arrays/arguments built interface bodies with
wrong intents (e.g. make_complex_reference_argument context intent(in) vs
intent(inout)) from the tool's per-body declaration capture - two manual
interface repairs failed; all reverted, HEAD build green (w04k suite stands).
char_arrays (`e810e3a`, 22->21) + arguments/allocatable (`32d7257`, 21->19
.inc) folded with the fixed tool; HEADER rettype/indent fix `7081a5d`
(verified vs all four skipped header shapes); full-suite w04l 489/11
fail-set IDENTICAL to accepted (`perf/suite-w04l.txt`).
FINAL: all .inc eradicated (`0676861`, 16->0). Last batches folded with
EXEC-keyword header guard + character merged into existing submodule via
character2 staging (no clobber of its 3 procedures); top.inc folded last
with child-include lines pruned. Clean build foldT rc=0; suite w04n
489/11 fail-set IDENTICAL to accepted (`perf/suite-w04n.txt`); build 1.8G
<2GB, single-edit rebuild 36.5s<60s (`perf/w0-final2.txt`). Remaining W0
gap: suite wall 245s vs <120s target. Attribution complete: sharding and
team width land and work (fo `d8998b9` team=nproc/4, shard default 6;
walkers print jobs=1 PHASE lines as shard children - not unsharded); the
floor is the smoke test's ~20 sequential gauntlet invocations x ~5.5s
(148.5s, perf/suite-w04o.results) plus sampling internals (83.1s). Next:
Attempted and reverted (kept green): full issue/barrier restructure of
test_conformance_gauntlet_smoke.f90 - all 20 blocks backgrounded via
issue(cmd,report) writing REPORT.rc, one poll barrier(n), then rc_ok +
report validation - measured 148.5s -> 80.5s; failed on exactly ONE
check: serial manifest-rejection run `FFC_NOREF_MANIFEST=... --file
ast_coverage_control_flow.f90` (no --report, default scratch) got SKIP=1
instead of its rejection: concurrent same-suite walks pollute shared
walk/cache state under the shared cwd. Retry recipe: same structure PLUS
TMPDIR=ROOT and unique --report on every rejection run; then parallelize
sampling's 4 serial sampled runs (83.1s) the same way. Suite w04o 489/11
fail-set IDENTICAL (perf/suite-w04o.txt).
Tool TODO: emit interface bodies verbatim from the joined definition header +
declaration block before retrying the big-file folds. Remaining 22 .inc: 8 mutually-recursive
groups, 11 header/terminator splits (cross-boundary procedure tails),
2 fragment-lead files, top.inc wrapper - each needs hand splitting.
USE-shadow re-declaration attempted (`d43d1d5`)
and REVERTED (`ab23c5a`): the symbol-slot scan fired across program units in
one session context and over-rejected 184/500 suite tests (perf/suite-useshadow.txt
wall=245s) - the shape needs per-scoping-unit USE tracking, not a flat symbol
scan. Tool fix: inc_to_submodule lead check counted comment-only leads as
fragment statements and refused clean files (inquire.inc now folds: 7 procedures
-> submodule inquire). W0 numbers re-measured after gc'ing 26 stale 31MB
objects_*.a snapshots: `ffc/build` = 1.2G < 2GB, single-domain edit rebuild
12.3s < 60s (`perf/rebuild-len.txt`, `perf/single-edit-rebuild.txt`).
Non-defects verified rather than assumed: implicit typing makes `y=2` legal, and
`a(9)` on `integer :: a(3)` is not a static error without `-fbounds-check`.

fo suite at `fo fd4b818`: **rc=0, 40/40 TEST_RESULT PASS, 0 fail**
(`logs/fo-suite-w45.log`), including `test_clean_help` under the real suite
invocation — it had crashed there while passing standalone, and both invocations are
now verified.

##### Derived-type surface pinned before descriptor work (`ffc 96f4692`)

Pinned as a **baseline, not a defect claim**: component read/write, arithmetic
across components, component-to-component copy, re-assignment and self-reference
(`q%x = q%x*q%x`), two objects of one type, nested `o%a%v` with sum and difference
across branches, character components with `trim`/`len`/`len_trim`/concat, negative
components, and a derived value passed to a function and to a mutating subroutine.
All match gfortran byte-exactly. `runs=23 match=23 fail=0 known_gap=1`
(`perf/dtype/report.tsv`), falsified -> 22 MISMATCH exit **1**; restored exit **0**.

Purpose is forward protection: #339/#348 edit the machinery these shapes travel
through, and #339 slice D already showed what an unguarded refactor does - the
audit found two **legitimate** descriptor producers where the plan expected legacy
writes to delete. Guard keeps that distinction from being re-litigated.

Gap kept visible by name: `new_line("A")` refused as unsupported character
expression. A `select type` row was written then **removed** - `select type (z =>
q)` over a concrete type is invalid Fortran (selector must be polymorphic), so it
tested nothing, and polymorphic dispatch is out of scope per #417.

##### #761 widened and corrected: ALL comparisons/logicals refused in list-directed print

Filed first as character-specific, then **corrected by measurement**: every
comparison and logical in `print *` position is refused, while all of them work in
`if`. `print *, 1<2`, `2==2`, `1/=2`, `1.5>1.0`, `(.true.).and.(.false.)`,
`a=="ab"` all refused; `if (1<2)`, `if (a=="ab")`, and bare `print *, t`
(prints `T`) all work. One route is missing - logical-valued nodes in print
position - and the machinery is proven on both sides of the gap. Refusal texts
name the operand kind, not the route, which is what misled the first probe.
Severity raised: `print *, x > 0` is ordinary Fortran. Issue retitled.

Guard `tools/test_scalar_semantics_parity.py`: `runs=31 match=31 fail=0
known_gap=8` (`perf/scalsem/report.tsv`); the 4 conditional forms in the guarded
set are the executable proof the lowering is correct. 8 refusals kept by name.



`print *, a=="ab"` on a character `a` is refused - `"integer expression used
non-integer identifier: a"` - while `if (a=="ab")` **lowers and runs correctly**.
So the comparison lowering is fine and only the print-item route lacks a branch:
`lower_integer_expression` already reroutes `VALUE_LOGICAL` to
`lower_logical_expression` but has **no `VALUE_CHARACTER` branch**, so character
operands fall through to `case default` and error. Boundary probed: `/=` and
var-to-var also refused as print items; `len`, `a(1:2)`, all conditional forms OK.
Same shape as #399 and #348 - a classifier missing one operand kind, fixed by
routing to machinery that exists, not new lowering.

Guard `tools/test_scalar_semantics_parity.py`: `runs=27 match=27 fail=0
known_gap=3` (`perf/scalsem/report.tsv`), falsified -> 25 MISMATCH exit **1**;
restored exit **0**. Pins integer div/mod across **all four sign combinations**
(Fortran truncates toward zero; floor division passes every positive-only test),
exponentiation, `index`/`len_trim`/`adjustl`/`adjustr`, and character comparison in
conditional context as the executable proof that #761 is routing-only. The three
gap rows are #761's refused forms kept visible by name, never deleted.
##### Formatted output: `Iw.m` zero-pad and `Aw` truncate FIXED (ffc `f11ff2f`)

Silent wrong bytes in ordinary formatted output, found by probing edit
descriptors against gfortran:

| program | ffc before | gfortran |
|---|---|---|
| `print "(I5.3)", 42` | `"   42"` | `"  042"` |
| `print "(I6.3)", 0` | `"     0"` | `"   000"` |
| `print "(I4.4)", 7` | `"     7"` | `"0007"` |
| `print "(A3)", "ABCDEF"` | `"ABCDEF"` | `"ABC"` |
| `print "(A2,I4.2)","XY",3` | `"XY   3"` | `"XY  03"` |

The width was parsed and discarded - for `I` the source said so in a comment
(`! Iw.m: ignore the minimum-digit count m`). printf already carries the Fortran
semantics in its **precision** field, so the fix is `%w.md` for `Iw.m` and
`%w.Ns` for `Aw`. Padding direction was **verified against gfortran, not taken
from the standard's prose** (the prose says A is left-justified; gfortran
right-justifies `(A5)`/"AB" to `"   AB"`, which this path already matched), so
only the missing precision changed. `I0`/`Iw` without `m` keep plain `%d`/`%wd`.

Oracle `tools/test_format_edit_descriptor_parity.py`: `runs=24 match=24
refused=0 mismatch=0 ref_fail=0`, per-row ref+ffc digests
(`perf/fmted/report.tsv`); falsified -> 23 MISMATCH, exit **1**; restored exit
**0**. `docs/SUPPORT_CONTRACT.md` gained a "Formatted output: field width and
zero-padding" section **in the same commit**, since this is semantic.

Background verification, complete and clean (`logs/w41-verify.log`, suite then gate
in one sequential job to respect one-`fo`-per-tree):
- **Suite w41-fmt: total=500 pass=489 fail=11**, fail-name set **identical** to
  `e339_base_failnames.txt` (new-fail-name list empty), wall 249 s at load1 7.79,
  binaries mean **2.1 MB** (`perf/suite-w41-fmt.txt`).
- **Rejection gate rc=0, 705 ACCEPTED / 213 REJECTED** - exactly the contract
  numbers - **0** files moved ACCEPTED->REJECTED, wall **33.22 s** (< 90 s target)
  (`perf/gate-w41.tsv`, `perf/gate-w41.log`).

**Method note worth keeping:** this is the third routing/mapping defect found by
*probing descriptors and shapes against gfortran* rather than reading the plan.
Each was 1-2 lines (`%w.md`, `%w.Ns`, gather route, substring route) and each had
plausible-but-wrong output, so none was detectable by a compile-and-run-ok check.
##### Structured branch targets: two defects pinned (`1492c4c`)

Work on **#455** (named EXIT/CYCLE) attempted, **reverted**, and turned into a
forced fix order. Named EXIT/CYCLE naming an enclosing loop binds to the
innermost loop - `outer: do; do; if (j==1) exit outer` prints `7 7 7`, gfortran
prints nothing. The guard I wrote (`current_loop_label` + refuse mismatches) also
refused innermost-named EXIT/CYCLE, which work today, because fortfront emits **no
DO block name at all** - the field is permanently empty. Reverted; shrinking the
accepted set is not a fix. Chain + fix order in ffc#455.

**ffc#760 filed (new)**: named DO nested >=2 levels under unnamed DOs is refused
outright (`do / do / c: do ... end do c` -> "Unrecognized statement: end program
p", wrong line named). Boundary probed: depth-2 named-in-unnamed OK, all-unnamed
depth-3 OK, all-named OK.

Oracle `tools/test_construct_name_exit_cycle_parity.py`: `runs=22 match=22
fail=0 known_gap=6`, `perf/cnname/report.tsv`, falsified (shared-reference
injection -> 24 MISMATCH, exit 1; restored exit 0). Two generator bugs were mine
and are fixed: packed one-liners over the 132-char free-form limit, and `b: do`
closed by a bare `end do` (both compilers correctly reject).
##### ffc#756 closed on measurement (last turn)

Repro prints **5** (was 0); extent routing works on `main`. Not closed on one
probe - `tools/test_assumed_shape_reduction_parity.py` pins the whole
`read_runtime_dim_extent` family: `runs=22 match=22 refused=0 mismatch=0
ref_fail=0`, 18 distinct digests, `perf/ashred/report.tsv`, falsified (wrong
binary -> 20 MISMATCH, exit 1; restored -> exit 0). PLAN row checked with that
pointer.

#### Parity oracle inventory at close (13 files, every one RUNS>=20 and falsified)

| oracle | runs | what it pins |
|---|---|---|
| `tools/test_char_runtime_surface_parity.py` | 22 | character runtime surface |
| `tools/test_assumed_shape_reduction_parity.py` | 22 | assumed-shape reductions (#756) |
| `tools/test_construct_name_exit_cycle_parity.py` | 22 | EXIT/CYCLE naming (#455 gaps) |
| `tools/test_char_substring_concat_parity.py` | 29 | substring concat (#348) |
| `tools/test_vector_subscript_gather_parity.py` | 36 | vector gather print (#399) |
| `tools/test_format_edit_descriptor_parity.py` | 24 | `Iw.m` zero-pad, `Aw` truncate (`f11ff2f`) |
| `tools/test_logical_print_parity.py` | 36 | comparison/logical list print (#761) |
| `tools/test_allocatable_intent_optional_parity.py` | 24 | allocatable/intent/optional |
| `tools/test_derived_type_access_parity.py` | 23 | derived-type access |
| `tools/test_scalar_semantics_parity.py` | 31 | scalar semantics, sign conventions |
| `tools/test_name_namespace_collision_parity.py` | 29 | **9 KNOWN_OVERACCEPT** (#3021) |
| `tools/test_string_search_back_parity.py` | 26 | **7 KNOWN_WRONG_ANSWER** `back=` (#764) |
| `tools/test_mask_reduction_merge_parity.py` | 28 | **12 KNOWN_REFUSED** `MASK=`/`merge`/nested `abs` (#766) + 6 numeric regression guards |

Gap rows are named with their issue number and are never deleted; each flips to a
pass as its fix lands, and each oracle fails if a currently-passing case drifts.
Every one was falsified by disabling its exemption branch and confirming a non-zero
exit with the expected number of surfaced defects — not by trusting a green run.

##### #764 FIXED: `BACK=` honoured in `index`, `scan`, `verify` (ffc `5b4eddf`)

Wrong answer on valid code, closed by implementation (was blocked item 7 in
the close-state above). `BACK=.true.` was silently dropped by all three
string-search intrinsics - `index("abcabc","abc",back=.true.)` printed `1`
where gfortran prints `4`. The fix resolves BACK (positional 3rd or `BACK=`,
literal or runtime scalar expression) and turns the match block into scan-on:
result overwrites itself, last qualifying position survives at `done`. Self-
correction recorded: the earlier claim "verify honours back correctly" came
from a coincidental oracle row (`verify("aabbc","abc",back=.true.)` - every
character in the set, both directions answer 0); `verify` ignored BACK too
and is fixed with discriminating rows (`verify("aabcc","ab",back=.true.)`
= 5 vs forward 4). `KIND` accepted only as foldable constant 4; unknown
keywords refused by name (`backet=` - a dropped keyword is how this bug hid).
Oracle `tools/test_string_search_back_parity.py`: `runs=39 match=36
known_refused=3 regressions=0` (`perf/ssback/report.tsv`, per-row md5);
falsified - deleting the index condbr surfaces 10 REGRESSION rows, exit 1;
restored exit 0. `SUPPORT_CONTRACT.md` character-intrinsics row updated in
the same commit. Verified at `5b4eddf`: suite **489/11, zero new failure
names** (`perf/suite-w46.txt`, only the documented corpus-XPASS extra);
rejection gate **705 ACCEPTED / 213 REJECTED, 0 ACCEPTED→REJECTED**
(`perf/gate-w46.tsv`). The "rank3/rank4_multiple_arms" names from the prior
close were probe files under `/var/tmp`, not suite members - the live suite
has no such names (`perf/suite-w46.fails`).

#### Background verifiers pending at close

None pending. The last rejection-gate run completed (`logs/gate-w47.log`); its
`FAIL=1` was root-scoped to `read_statement.lf` and shown pre-existing by a rebuilt
HEAD~1 control. `fortfront-lf` re-run at final HEAD after the rebuild: `PASS=219
XFAIL=47 XPASS=0 FAIL=1`, matching the control exactly. The two ffc unit tests that
asserted the old diagnostic text were updated and re-run at final HEAD: **2 passed,
0 failed**. Stale-binary hazard (fo#133) was hit live during this close — an aborted
rebuild left the binary emitting the pre-fix message while the source carried the fix
— and was cleared by rebuild + re-running the message probe.
