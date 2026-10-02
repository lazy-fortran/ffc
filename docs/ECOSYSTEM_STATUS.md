# Fortran ecosystem status

Checkpoint: 2026-10-02. This document records the current delivery and takes
precedence over historical metrics in PLAN.md. Completion of this delivery does
not mean full Fortran support or completion of every item in the plan.

## Repository delivery

| Repository | Main revision | Remote state |
|---|---|---|
| fo | `50de371` | Pushed |
| fortfront | `2b6bc1a` | Pushed |
| ffc | Main, this checkpoint | Committed and pushed with this document |
| liric | `9042548` | No code changes in this delivery |
| standard | `160032a` | No code changes in this delivery |

All implementation, build, and validation work for this delivery lives under
`/var/tmp/lazy-fortran-finish-20261002`. Workers return frozen patches; the
controller commits and promotes them. No review wait is part of this delivery.

## Delivered behavior

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

## Validation

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
baseline after a correct CLI build was 487 passing and 13 failing tests; final
failures will be reported explicitly, including pre-existing failures.

## Failing checks

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

## Remaining work

Compiler lint succeeds with 14 unused imports and 93 compiler warnings in
pre-existing code. No warning is anchored to a newly added or modified line.
The documentation checker reports two unchanged prose findings in
RUNTIME_ABI.md; introduced support and audit prose passes.

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

## Evidence

The evidence root contains frozen worker patches and their SHA256 digests,
`manifest.json`, `suite-evidence.json`, `suite-relink-evidence.json`, the final
pipeline logs, numeric/format/descriptor parity receipts, and conformance
observations. In particular, `logs/ffc-suite-final-frozen.log`,
`evidence/final-test-verdicts.json`, `logs/rejection-final.log`,
`logs/ffc-lint-final.log`, and `logs/conformance-oracle-integrated.log` record
the final compiler result. `evidence/ffc-final-controller.patch` freezes the
integrated source before its checkpoint commit. These artifacts are retained in /var/tmp; Git commits preserve
the implementation, test oracles, and public support documentation.
