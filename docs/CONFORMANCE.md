# Conformance gauntlet

Drive external Fortran test corpora through the full `ffc` pipeline.
The runner compiles each source to a native binary, runs it, compares
standard Fortran output against `gfortran -w` when gfortran accepts the
source, and writes an xfail-style report.

Compilation and expectation classification are separate phases. The first
phase writes one expectation-neutral observation for every selected case. The
second phase maps those raw `PASS`/`FAIL` outcomes through an XFAIL manifest;
it does not select, compile, execute, or invoke an oracle. This separation
makes expectation-only changes cheap to inspect and prevents the manifest from
changing the behavior being measured. SKIP and NOREF manifests remain
operational inputs: they define which cases have runnable behavioral oracles,
and their digests are therefore part of observation provenance.

## No-vendoring rule

The runner never copies source files from external suites into this
repository. Only conformance manifests live here. External checkouts are
referenced by path.

## Suites

| Suite | Source | Extension | Reference |
|---|---|---|---|
| `fortfront-f90` | FortFront standard-mode examples | `.f90` | `gfortran -w` |
| `fortfront-lf` | FortFront lazy-mode examples | `.lf`, `.f90` | ffc only (gfortran cannot compile lazy Fortran) |
| `lfortran` | LFortran integration tests | `.f90` | `gfortran -w` |
| `gfortran-dg` | GCC gfortran.dg testsuite | `.f90` | `gfortran -w` |

### Nested-DO tail regression

`test_session_nested_do_tail_compiler` is a focused accepted-side control-flow
oracle. It compiles the nested-loop source with both `gfortran -w` and ffc,
then compares the complete stdout bytes and exit status. The case protects
against FortFront construct-span regressions in contained procedures: the
FortFront parser must be at least `f3ab76ba`, which fixed the two-word `end do`
terminator scan that previously swallowed statements following a nested nest.
The ffc lowerer must not recover this case with source-name or statement-order
special cases.

## Fetching corpora

`scripts/fetch_corpora.sh` checks out reviewed external corpus revisions at
the default sibling paths: a shallow LFortran clone and a blobless sparse GCC
checkout containing `gcc/testsuite/gfortran.dg`. Existing checkouts must
already match the detached pins; the script never repairs a stale or corrupt
cache. A corpus argument (`lfortran`, `gfortran-dg`) restricts the operation.

```bash
scripts/fetch_corpora.sh               # fetch or verify both pins
scripts/fetch_corpora.sh --verify-only # verify without fetching
```

| Corpus | Pinned revision |
|---|---|
| LFortran | `caf87b660f803148f000046392a5da803f9fc630` |
| GCC gfortran.dg | `395e3d8131c189cd58e8c8061cdc77d1c44e3822` |

Update a pin by changing its full SHA in `scripts/fetch_corpora.sh` and
reviewing the resulting corpus reports. CI cache keys include that SHA and do
not use prefix restore keys, so another revision cannot satisfy the cache.

## Environment variables

Set these to point at local checkouts of the external repositories.
Defaults assume sibling directories under the parent of the ffc checkout.
`scripts/fetch_corpora.sh` honors the same variables when choosing
destinations.

| Variable | Default | Suite |
|---|---|---|
| `FFC_FORTFRONT_DIR` | `../fortfront` | `fortfront-f90`, `fortfront-lf` |
| `FFC_LFORTRAN_DIR` | `../lfortran` | `lfortran` |
| `FFC_GFORTRAN_DG_DIR` | `../gcc/gcc/testsuite/gfortran.dg` | `gfortran-dg` |

A suite whose root directory does not exist prints `SKIP: <suite> not
found at <path>` and exits 0. Optional external suites may stay absent.

## Running the gauntlet

Keep conformance artifacts on the storage volume used by the corpus worktrees:

```bash
export TMPDIR=/mnt/storage/lazy-fortran-artifacts-20260806
mkdir -p "$TMPDIR"
```

```bash
scripts/conformance_gauntlet.sh --suite SUITE [OPTIONS]
```

Options:

| Flag | Description |
|---|---|
| `--suite SUITE` | Required. One of `fortfront-f90`, `fortfront-lf`, `lfortran`, `gfortran-dg` |
| `--ffc PATH` | Path to the `ffc` binary. Auto-discovered from `build/` or `PATH` if omitted. |
| `--report PATH` | JSONL report path. Defaults to `$TMPDIR/ffc_gauntlet_<suite>.jsonl`. |
| `--observations PATH` | Expectation-neutral JSONL observation path. Defaults to `<report stem>.observations.jsonl`. |
| `--file PATH` | Select one suite-relative file. Repeat to select more files. |
| `--files-from PATH` | Read suite-relative files from a list. Repeat to read more lists. |
| `--max-files N` | Only test the first N files. Use for smoke runs. |
| `--timeout N` | Per-file timeout in seconds. Default: 5. |
| `--sample N` | Measure a deterministic random subset of N files. Marks the report sampled and `full_run` false. |
| `--seed S` | Seed for `--sample`. Default: 0. The same seed over the same corpus selects the same files. |
| `--ref-cache DIR` | Reuse successful hermetic gfortran reference observations cached under DIR. The key binds the source closure, compiler and flags, target, runtime ABI, harness policy, corpus, and compile/run environment. |
| `--require-provenance` | Build ffc with `fo`, require clean compiler/dependency/corpus inputs, and record exact tree and file-list identities. |

Smoke run (20 files, auto-discovers ffc):

```bash
scripts/conformance_gauntlet.sh --suite fortfront-f90 --max-files 20
```

Full run (all files, explicit ffc path):

```bash
scripts/conformance_gauntlet.sh --suite fortfront-f90 \
    --ffc "$(find build -name ffc -type f -executable | head -1)"
```

### Reclassifying without another corpus run

The gauntlet always leaves an observation sidecar. Its case records contain
only observed `PASS`, `FAIL`, `SKIP`, or `FLAKY` states. The XFAIL manifest is
not read until that file is complete. A malformed expectation manifest can
therefore fail classification, but it cannot prevent or alter observation;
the diagnostic names the preserved sidecar.

Create another normal expectation view with a different manifest:

```bash
scripts/classify_conformance_observations.sh \
    --suite fortfront-f90 \
    --observations "$TMPDIR/ffc-one.observations.jsonl" \
    --xfail-manifest "$TMPDIR/proposed-xfail.txt" \
    --report "$TMPDIR/ffc-one-proposed.jsonl"
```

Create the all-observed view, with XFAIL handling disabled:

```bash
scripts/classify_conformance_observations.sh \
    --suite fortfront-f90 \
    --observations "$TMPDIR/ffc-one.observations.jsonl" \
    --no-xfail \
    --report "$TMPDIR/ffc-one-no-xfail.jsonl"
```

The classifier exits nonzero when the resulting view contains `FAIL`, `XPASS`,
or `FLAKY`, just as the gauntlet does. It refuses to use the observation path
as its output, validates that there is exactly one raw record per case, and
writes the new view atomically. `--no-xfail` does not read an XFAIL manifest.

### Named files

Run one external regression without scanning the full corpus:

```bash
scripts/conformance_gauntlet.sh --suite fortfront-f90 \
    --file ast_coverage_control_flow.f90 \
    --report "$TMPDIR/ffc-one.jsonl"
```

A list contains one suite-relative path per line. Leading and trailing
whitespace is ignored. Blank lines and lines whose first nonblank character is
`#` are ignored.

```text
# scope regressions
ast_coverage_control_flow.f90
ast_coverage_io_statements.f90
```

```bash
scripts/conformance_gauntlet.sh --suite fortfront-f90 \
    --files-from "$TMPDIR/ffc-scope-files.txt"
```

`--file` and `--files-from` entries accumulate in command order. Duplicate,
unknown, absolute, and parent-traversal paths are errors. Named selection is
applied before `--max-files`, so `--max-files 1` runs the first named entry.
The report contains one file record per selected entry followed by a SUMMARY
whose `total` is the number of entries run.

## Single-command conformance gate

`scripts/conformance_check.sh` is the documented routine contributors run
before pushing and after dependency (fortfront, liric) updates. It builds
ffc, runs every available suite, fails on any FAIL or XPASS, and prints
the promotable XPASS list.

```bash
scripts/conformance_check.sh                    # build + all available suites
scripts/conformance_check.sh --no-build          # skip build, run suites
scripts/conformance_check.sh --suite fortfront-f90  # single suite
scripts/conformance_check.sh --no-build --suite fortfront-f90 \
    --file ast_coverage_control_flow.f90
```

Named selection requires `--suite`; the check forwards all `--file` and
`--files-from` options to that suite.

## Sampled runs and the reference cache

A full four-suite run compiles the whole corpus with two compilers, which is
slow enough that measurement gets skipped and the checked-in snapshot goes
stale (#567). Two mechanisms make a routine progress read cheap without
weakening the exact record.

`--sample N` measures a deterministic random subset. `scripts/conformance_check.sh
--sample N` stratifies that total across the available suites in proportion to
their size, so every suite keeps its own margin:

```bash
scripts/conformance_check.sh --no-build --sample 1700 --seed 1
scripts/conformance_check.sh --sample 1700 --print-sample-plan
```

About 1,700 files gives a 95% margin near plus or minus 2% against the full
corpus, which is ample for deciding whether a change moved the corpus. The
summary reports the margin next to the rate, and the report carries
`"sampled":true` with `full_run` false. A sampled report is an estimate: the
dashboard validator rejects it outright, so `test/conformance/parity_dashboard.tsv`
stays a full, provenance-verified run.

### XFAIL-first sampling gate

Sampling is a safety and progress instrument, not permission to widen the
work queue. Each cycle selects one owned in-scope XFAIL tranche, fixes the
implementation or its independent behavioral oracle, and removes the manifest
entry only after the named case passes with `XFAIL=0`, `XPASS=0`, and `FAIL=0`.
Do not move to another corpus area or increase the sample count while that
tranche is nonzero. XFAIL is an expectation classification; NOREF and SKIP are
operational dispositions rather than behavioral passes. A clean sample with
any of them does not claim full parity.

Compiler jobs remain sequential and bounded. A sampled command that exceeds
the memory or timeout budget is a failed measurement and must be reduced or
fixed before the sample is increased.

`--ref-cache DIR` caches successful gfortran reference output. A cache key
covers every input that can change that output: the main source, recursively
resolved Fortran INCLUDE files, and sibling or extra-source dependency closure,
the suite and corpus revision, the
reference compiler executable hash and version, its target triple and flags
(`-w` and the private `-J` module directory policy), the declared locale and
runtime environment, the conformance harness scripts and policy, and the
reference runtime ABI. A hit replaces only the gfortran compilation and run.
The gauntlet still compiles and runs ffc for every selected case. Failed
gfortran compilation or execution is measured on every invocation and is not
stored for reuse. If a cached reference disagrees with the fresh ffc result,
the runner discards the entry and measures the reference again. The key remains
the cache-validity boundary: agreement with ffc is not proof that an entry from
different inputs is valid. Sampling and reference caching compound.

## Repeated runs and FLAKY records

A single run cannot distinguish a stable result from a case that flips
between runs, so promoting an XPASS on one run can convert a flake into a
manifest entry that the next run breaks (#599). `--repeat N` runs the
selection N times as independent child observations. The raw attempts are
merged first and only the merged observation is classified:

```bash
scripts/conformance_gauntlet.sh --suite lfortran --repeat 5 \
    --files-from suspect_cases.txt --report "$TMPDIR/repeat.jsonl"
scripts/conformance_check.sh --no-build --suite lfortran --repeat 5
```

A file whose complete raw evidence is identical in every attempt keeps its
record. A status, exit-code, NOREF, or other evidence change is recorded as

```json
{"suite":"lfortran","file":"kwargs_01.f90","status":"FLAKY",
 "note":"unstable across 5 attempts","attempts":5,"observed":"PASS|FAIL"}
```

and counted in the SUMMARY `flaky` field, which is present only when the
count is nonzero. FLAKY is a failure: both the gauntlet and the conformance
check exit nonzero, and a FLAKY case is never reported as promotable XPASS.
Never stabilise such a case by raising a timeout; reduce it and fix the
underlying nondeterminism.

The script auto-detects available suites by checking the suite root
directories. If a suite root does not exist, it prints a SKIP message and
continues with the remaining suites. Only `fortfront-f90` and `fortfront-lf`
run out of the box; `lfortran` and `gfortran-dg` require external checkouts
(see below).

The script exits 1 when any suite has a FAIL or XPASS record. XPASS records
indicate manifest drift: the file now passes but is still listed in the xfail
manifest. Promote by removing the entry from the manifest.

## External corpora

`lfortran` and `gfortran-dg` suites require local checkouts of the external
repositories. `scripts/fetch_corpora.sh` handles this:

```bash
scripts/fetch_corpora.sh               # fetch or verify pinned revisions
scripts/fetch_corpora.sh --verify-only # verify exact revisions only
```

Alternatively, set the environment variables to point at existing checkouts:

| Variable | Default | Suite |
|---|---|---|
| `FFC_LFORTRAN_DIR` | `../lfortran` | `lfortran` |
| `FFC_GFORTRAN_DG_DIR` | `../gcc/gcc/testsuite/gfortran.dg` | `gfortran-dg` |

When the conformance check script finds the directory, it includes the suite
automatically. When absent, it prints a SKIP message.

CI restores and verifies separate exact-revision caches, then runs these named
buckets through `conformance_check.sh`:

- `test/conformance/ci_lfortran.txt`: `common_39.f90`
- `test/conformance/ci_gfortran_dg.txt`: `host_assoc_function_3.f90`

The expected selected total is one for each suite. Cache misses fetch and verify the
same detached revisions before saving; cache hits run the identical verifier.

## Current checked-in manifests

The manifest files are the local gate's source of truth. As of this checkout
they contain these normalized entry counts, ignoring comments and blank lines:

| Manifest | Entries |
|---|---:|
| `test/conformance/xfail_fortfront_f90.txt` | 90 |
| `test/conformance/xfail_fortfront_lf.txt` | 51 |
| `test/conformance/noref_fortfront_f90.txt` | 6 |
| `test/conformance/noref_lfortran.txt` | 6 |
| `test/conformance/xfail_lfortran.txt` | 3213 |
| `test/conformance/xfail_gfortran_dg.txt` | 2174 |
| `test/conformance/skip_fortfront_f90.txt` | 2 |
| `test/conformance/skip_lfortran.txt` | 3 |
| `test/conformance/skip_gfortran_dg.txt` | 2300 |
| `test/conformance/fail_owners_lfortran.txt` | 4 |
| `test/conformance/fail_owners_gfortran_dg.txt` | 284 |
| `test/conformance/scopes_fortfront_f90.txt` | 6 |
| `test/conformance/scopes_lfortran.txt` | 302 |
| `test/conformance/scopes_gfortran_dg.txt` | 219 |
| `test/conformance/owner_subsystems.txt` | 135 |

The aggregate manifest inventory is `XFAIL=5527`, `FAIL=288`, `NOREF=12`, and
`SKIP=2305`. `FAIL` counts FAIL-owner rows. These are inventory facts, not fresh
compiler outcomes.

Use `docs/PARITY_PLAN.md` and issue #299 for the latest full-suite pass-rate
snapshot. The seed baselines below are historical starting points, not current
scoreboard values.

## JSONL output

The observation sidecar has one raw record per selected file:

```json
{"suite":"fortfront-f90","file":"example.f90","status":"PASS","ffc_exit":0,"ref_exit":0,"note":"output matches gfortran","epoch_sha256":"...","action":"compile-run","ffc_compile_action":"executed","ffc_compile_exit":0,"ffc_compile_termination":"exit","ffc_compile_signal":0,"ffc_run_action":"executed","ffc_run_exit":0,"ffc_run_termination":"exit","ffc_run_signal":0,"ref_compile_action":"executed","ref_compile_exit":0,"ref_compile_termination":"exit","ref_compile_signal":0,"ref_run_action":"executed","ref_run_exit":0,"ref_run_termination":"exit","ref_run_signal":0,"source_sha256":"...","dependency_closure_sha256":"...","ffc_flags":"default","ref_flags":"-w -J @private-module-dir","compiler_flags_sha256":"...","environment_sha256":"...","target_triple":"x86_64-linux-gnu","runtime_abi_sha256":"...","harness_sha256":"...","toolchain_sha256":"...","phase":"compare","diagnostic_signature_sha256":"...","crash_signature_sha256":"...","ffc_output_sha256":"...","ref_output_sha256":"...","elapsed_ms":31,"ffc_compile_ms":12,"ffc_run_ms":2,"ref_compile_ms":15,"ref_run_ms":2,"peak_rss_kb":28400,"semantic_tags":"procedure","coverage_mode":"none","coverage_sha256":"..."}
```

The classified report keeps that evidence and adds the expectation decision:

```json
{"suite":"fortfront-f90","file":"example.f90","status":"XPASS","ffc_exit":0,"ref_exit":0,"note":"output matches gfortran","observed_status":"PASS","expectation":"xfail"}
```

Fields:

| Field | Type | Description |
|---|---|---|
| `suite` | string | Suite name |
| `file` | string | File basename (suite-relative path) |
| `status` | string | `PASS`, `XFAIL`, `XPASS`, `FAIL`, or `SKIP` |
| `ffc_exit` | int | ffc exit code (0 = built and ran) |
| `ref_exit` | int | gfortran exit code (0 = built and ran) |
| `note` | string | Human-readable explanation |
| `epoch_sha256` | SHA-256 | Immutable execution descriptor shared by each row and its SUMMARY |
| `action` | string | Case mode: `compile-run`, `compile-only`, `reject`, or `exclude` |
| `*_compile_action`, `*_run_action` | string | `executed`, reference `cache-hit`, or `not-run` |
| `*_compile_exit`, `*_run_exit` | int | Separate action exits; `-1` means not run |
| `*_compile_termination`, `*_run_termination` | string | `exit`, `timeout`, `signal`, `exec-error`, or `not-run` |
| `*_compile_signal`, `*_run_signal` | int | Exact terminating/timeout signal, otherwise zero |
| `observed_status` | string | Immutable raw state used by a classified view; absent from the observation sidecar |
| `expectation` | string | `xfail` or `none`; absent from the observation sidecar |
| `warning_expectation` | string | `unchecked` for warning-only gfortran.dg files; omitted otherwise |
| `noref` | boolean | `true` when the case has no behavioral oracle; omitted otherwise |
| `noref_reason` | string | Required with `noref`: an approved manifest category, `reference-rejected`, or `reference-runtime-failure` |
| `source_sha256`, `dependency_closure_sha256` | SHA-256 | Main source and canonical, recursively resolved source/INCLUDE closure compiled from the per-case snapshot |
| `ffc_flags`, `ref_flags`, `compiler_flags_sha256` | string/SHA-256 | Canonical compiler arguments and their joint digest |
| `environment_sha256`, `target_triple` | SHA-256/string | Declared compile/run environment and target |
| `runtime_abi_sha256`, `harness_sha256`, `toolchain_sha256` | SHA-256 | Runtime contract, harness implementation, and exact compiler toolchain |
| `phase` | string | Last decisive phase: compile, run, reference, compare, skip, directive, or complete |
| diagnostic, crash, and output SHA-256 fields | SHA-256 | Normalized diagnostics/crash evidence and byte-exact program outputs |
| elapsed and per-phase `*_ms`, `peak_rss_kb` | int | Nonnegative resource measurements; repeat stability ignores these values |
| `semantic_tags` | string | Deterministic comma-separated feature tags, or `none` |
| `coverage_mode`, `coverage_sha256` | string/SHA-256 | Coverage collector and evidence digest; `none` binds the empty digest |

A final SUMMARY record closes each file. The observation summary has
`report_kind: "observation"`; each derived report changes that to
`"classification"` and records `classification_mode`, the SHA-256 of the
complete observation, and the SHA-256 of the expectation manifest:

```json
{"suite":"fortfront-f90","status":"SUMMARY","pass":15,"xfail":3,"xpass":1,"fail":2,"noref":1,"skip":0,"warning_unchecked":0,"total":21,"schema_version":2,"full_run":true,"provenance_verified":true,"epoch_sha256":"...","ffc_revision":"...","ffc_source_sha256":"...","ffc_binary_sha256":"...","fortfront_revision":"...","fortfront_tree":"...","liric_revision":"...","liric_tree":"...","corpus_revision":"...","corpus_tree":"...","corpus_files_sha256":"...","worktree":"/home/you/ffc","report_kind":"classification","observation_schema_version":2,"reference_compiler":"GNU Fortran ...","reference_cache_enabled":false,"reference_cache_hits":0,"timeout_seconds":5,"skip_manifest_sha256":"...","noref_manifest_sha256":"...","target_triple":"x86_64-linux-gnu","environment_sha256":"...","runtime_abi_sha256":"...","harness_sha256":"...","toolchain_sha256":"...","compiler_flags_sha256":"...","coverage_mode":"none","classification_mode":"manifest","observation_sha256":"...","classification_manifest_sha256":"..."}
```

The revision and tree fields are full Git hashes. `ffc_revision` identifies the
tested compiler commit. `ffc_source_sha256` hashes `src`, `app`, and `fpm.toml`;
`ffc_binary_sha256` identifies the exact executable. With
`--require-provenance`, the runner first builds that executable with `fo`,
requires clean inputs across ffc and every dependency or corpus checkout, and
rejects a binary older than any tracked compiler or dependency input.
`corpus_files_sha256` hashes the exact suite-relative denominator.
`worktree` is the absolute path of the checkout that produced the report.
The observation also binds the reference compiler, timeout, reference-cache
use, and the operational skip/NOREF manifest digests. A classification binds
the byte-exact observation and XFAIL manifest. Reclassification reuses the raw
observation without running either compiler. During observation, an explicit
`--ref-cache` may reuse only the gfortran side of a case under the hermetic key
described above. The runner still compiles and runs ffc for each selected case.
`full_run` is false when a report used `--file`, `--files-from`,
`--max-files`, or `--sample`. A sampled report additionally carries
`sampled`, `sample_size`, `sample_population`, `sample_seed`, and
`sample_margin_pct`, and dashboard validation rejects it as an estimate.
Dashboard generation requires verified provenance and rejects
partial reports, mismatched tree or file-list identities, a stale source
digest, or a different selected compiler binary.

The epoch digest binds the selection, corpus/compiler revisions, input and
tool hashes, flags, target, environment, timeout, cache policy, manifests, and
worktree. Every row must match the SUMMARY epoch. Compile and run exits are
never overloaded: `ffc_exit` and `ref_exit` remain only as strictly validated
projections for older consumers. The action supervisor observes the OS return
code directly, so deliberate exits 124 or 137 remain `exit`; only an expired
deadline is `timeout`, and only termination by a signal is `signal`. A timeout
records SIGTERM (15), or SIGKILL (9) after escalation.

## Comparing two reports

Two clean worktrees at the same commit have been observed to disagree on corpus
results (ffc #642), so a delta measured across worktrees carries unknown error.
Sound A/B measurement is same-worktree before/after: measure the baseline,
apply the change, rebuild, measure again, all in one checkout.

`scripts/compare_conformance_reports.sh BASELINE.jsonl CANDIDATE.jsonl` enforces
that. It exits 0 when no per-file status changed, 1 when some did (the changes
are printed), and 2 when the pair is not comparable at all — different
`worktree` values, a missing `worktree` field, or different suites.

## Disposition states

| State | Meaning | Gate impact |
|---|---|---|
| `PASS` | ffc built and ran; standard files matched gfortran when gfortran accepted the source | None |
| `XFAIL` | Listed in xfail manifest; ffc failed as expected | None |
| `XPASS` | Listed in xfail manifest; ffc passed unexpectedly | Visible in summary; promote the entry |
| `FAIL` | Not in xfail manifest; ffc failed or mismatched | Fails the gate |
| `SKIP` | Listed in a skip manifest because the runner does not model the case | Counted in summary |

The runner exits nonzero if any `FAIL`, `XPASS`, or `FLAKY` record exists.
`XFAIL` does not cause a nonzero exit. XPASS is a stale expectation and must be
promoted before the normal manifest view can pass its gate.

The `noref` summary count is the number of files with no behavioral oracle:
files classified in `test/conformance/noref_<suite>.txt`, plus files where
`gfortran -w` rejected the source after ffc compiled and ran it. These
records are `PASS` unless the file is still listed in the xfail manifest,
in which case they are `XPASS`. See "NOREF manifests" below for the
approved categories.

The `skip` summary count is the number of files listed in
`test/conformance/skip_<suite>.txt`. They are explicit entries, not silent
drops.

The `warning_unchecked` count is the number of warning-only gfortran.dg files
whose compile or run disposition was checked without matching warning text.

## FortFront corpus gate

The fpm test `test_fortfront_corpus_conformance` runs the full
`fortfront-f90` and `fortfront-lf` suites through the gauntlet with
reports under `$TMPDIR`. Passing files need no manifest entry. A new
FortFront example fails the ffc test until ffc supports it or its
basename is added to the matching xfail manifest.

The maintained fpm test rejects both FAIL and XPASS records. An XPASS is a
stale manifest entry and must be promoted before merging.

Current xfail manifests:

- `test/conformance/xfail_fortfront_f90.txt`
- `test/conformance/xfail_fortfront_lf.txt`

### NOREF manifests

`test/conformance/noref_<suite>.txt` classifies the cases that cannot have a
behavioral oracle. Each entry states an approved category and a free-text
reason:

```
undefined_var_segfault.f90 # noref=undefined-runtime-value; reason=reads an undefined variable
```

The runner rejects a manifest entry with a missing delimiter, a missing or
empty reason, a duplicate path, or a category outside this list:

| Category | Meaning | What the runner still enforces |
|---|---|---|
| `undefined-runtime-value` | printed values depend on undefined data | both compilers build and exit zero; stdout is ignored |
| `missing-external-definition` | a referenced definition lives outside this suite invocation | the reference must fail to build a complete executable |
| `compile-only` | the source is not a runnable program unit | `ffc -c` must succeed and the reference must not link |
| `nondeterministic-runtime-value` | defined runtime randomness can choose different control-flow/output branches in independent processor streams | both compilers must build and exit zero; stdout is intentionally not compared |

For the `missing-external-definition` and `compile-only` categories the runner first tries to build the file
with `gfortran -w`. If that produces a runnable executable, the case is a
stable valid executable, the category does not apply, and the record is a
`FAIL` — a valid program can never be hidden behind NOREF. NOREF entries must
not also appear in an xfail or skip manifest.

Every NOREF result record carries `"noref":true` and a `"noref_reason"` field.
Besides approved manifest categories, `reference-rejected` records that
`gfortran -w` rejected the source, and `reference-runtime-failure` records that
gfortran built but did not terminate normally while the ffc program did. The
report validator rejects any other reason, any incompatible exit statuses, and
any `noref_reason` without `noref`.
NOREF cases stay visible in the suite totals and in the `noref` summary count;
they never mask a compiler crash or deterministic wrong output.

## gfortran.dg testsuite

The `gfortran-dg` suite reads `$FFC_GFORTRAN_DG_DIR/*.f90` from a local
GCC checkout. The runner evaluates each file with `ffc -c` (compile), checks
rejection of `dg-error` tests, or executes `dg-do run` files and compares their
output against `gfortran -w`.

### Local checkout

The runner expects a local GCC source checkout, typically acquired via
sparse checkout to minimize disk usage. The directory must contain the
`gfortran.dg` subdirectory with `.f90` test files. Set
`FFC_GFORTRAN_DG_DIR` to point at that directory.

### Directive subset

The runner models these gfortran.dg directives:

- `dg-do compile` (default): compile with `ffc -c`
- `dg-do run`: build, execute, and compare stdout and exit status against `gfortran -w`
- `dg-error`: negative test; `ffc -c` must reject compilation
- `dg-warning` without `dg-error`: follow `dg-do`; warning text remains unchecked
- `dg-additional-sources`: multifile tests (skipped)
- `dg-options` / `dg-add-options`: tests requesting nonempty compiler flags
  (skipped); empty directives continue through the declared compile or run path
- `dg-require`, `dg-skip-if`, `dg-final`, `dg-prune-output`,
  `dg-excess-errors`, `dg-shouldfail`: directive tests (skipped)

Files with unlisted skip reasons are marked FAIL until added to the skip
manifest.

`dg-error` takes precedence when a file also contains `dg-warning`. A
warning-only file records `"warning_expectation":"unchecked"`, increments
`warning_unchecked`, and otherwise uses the normal compile or run disposition.
Successful compilation is not accepted-invalid behavior. This accounting does
not claim warning-text parity.

### Skip manifest

`test/conformance/skip_gfortran_dg.txt` lists files the runner skips.
The runner exits nonzero for files that trigger a skip reason but are not
listed in the manifest.

### Expected-disposition metadata

Every xfail and skip entry names either one implementation issue or one
excluded scope and gives a reason:

```text
basename.f90 # owner=ORG/REPO#123; reason=nonempty text
basename.f90 # scope=OpenMP; reason=nonempty text
```

Allowed scope values are `coarray`, `OpenMP`, `OpenACC`, `GPU`, `vendor`,
`legacy`, `compiler-flags`, and `harness`. The last two cover tests whose result
depends on unmodeled compiler options or DejaGNU behavior rather than the source alone.
Owner syntax and metadata are validated offline during every gauntlet run. The
ordinary runner does not contact GitHub. Use the explicit liveness audit to
require every referenced issue to be open:

```bash
scripts/audit_manifest_owners.sh
```

Duplicate paths and malformed entries fail with the manifest path and line
number. Undefined-output manifests remain plain filename lists because they
describe the comparison oracle, not an expected failure or skip.

`test/conformance/fail_owners_<suite>.txt` uses the same metadata grammar for
current `FAIL` rows. It supplies dashboard ownership without changing gauntlet
dispositions. The explicit owner audit covers xfail, skip, and failure-owner
manifests.

`test/conformance/scopes_<suite>.txt` tags any result, including `PASS`, for the
scoped dashboard view. A scope disposition in an xfail or skip manifest must
have the same entry in the scope registry. `owner_subsystems.txt` maps every
issue owner to one explicit compiler subsystem; missing and stale mappings are
errors.

### Generated parity dashboard

Run all four suites without selectors so the default report paths contain full
reports, then generate the checked-in dashboard:

```bash
scripts/conformance_gauntlet.sh --suite fortfront-f90 \
    --require-provenance \
    --report "$TMPDIR/ffc_parity_fortfront-f90.jsonl"
scripts/conformance_gauntlet.sh --suite fortfront-lf \
    --require-provenance \
    --report "$TMPDIR/ffc_parity_fortfront-lf.jsonl"
scripts/conformance_gauntlet.sh --suite lfortran \
    --require-provenance \
    --report "$TMPDIR/ffc_parity_lfortran.jsonl"
scripts/conformance_gauntlet.sh --suite gfortran-dg \
    --require-provenance \
    --report "$TMPDIR/ffc_parity_gfortran-dg.jsonl"
scripts/generate_parity_dashboard.sh
scripts/generate_parity_dashboard.sh --check
scripts/generate_parity_dashboard.sh \
    --from-snapshot test/conformance/parity_dashboard.tsv --check
```

### Three constraints that are not obvious

Each of these has broken `main` or produced a wrong measurement.

**Generate from a branch whose HEAD equals `main`, as its first commit.** The
snapshot records the ffc revision it was generated at, and the check requires
that revision to be an ancestor of HEAD:

```bash
git merge-base --is-ancestor "$ffc_revision" HEAD \
    || fail "snapshot ffc revision is not an ancestor"
```

Squash-merge creates a commit with no ancestry to the branch, so any commit
sitting above `main` when the snapshot was generated takes the snapshot's
provenance with it and `main` goes red on landing. The PR will be green on its
own branch, which is why this is easy to miss.

**A PR that both changes corpus behaviour and carries a regenerated snapshot
must be merged with a merge commit, not squashed.** The snapshot cannot be
generated at `main`'s HEAD when the change it measures lives on a branch, so
preserving ancestry is the only option that keeps `main` green.

**Regenerate on an idle machine, with a raised compile timeout.**

```bash
FFC_COMPILE_TIMEOUT=60 scripts/conformance_gauntlet.sh --suite ... --require-provenance
```

`benchmark_5000_lines.f90` compiles in about five seconds against the ten-second
default, so it passes when the machine is quiet and times out when it is not.
Schema 2 records it as `ffc_compile_exit: 124` and
`ffc_compile_termination: "timeout"`, rather than collapsing it into a compile
error. It remains a blocking measurement. See #478.

Compiler-performance changes use a separate behavioral and resource oracle:

```bash
scripts/benchmark_large_translation_unit.sh \
    --baseline-dir /path/to/baseline-ffc \
    --candidate-dir /path/to/candidate-ffc \
    --source /path/to/fortfront/examples/f90/benchmark_5000_lines.f90 \
    --report "$TMPDIR/ffc-large-unit.md"
```

Build both worktrees with `fo build` first. The script keeps the fixture at
5,000 lines and replaces padding with calls to its first, middle, and last
contained functions. It checks each executable's stdout byte-for-byte against
the output of a reference executable built by the recorded gfortran binary,
then reports median wall time and peak RSS from alternating runs. A compiler
that changes the program result exits before the script writes a performance
report. The report fingerprints every tracked or untracked, nonignored file
in both source worktrees and both `ffc` build artifacts, so a dirty candidate
remains identifiable.

Note also that `scripts/conformance_check.sh` does **not** pass
`--require-provenance`, so its reports cannot feed the generator; invoke the
gauntlet directly per suite as shown above.

The generator requires Bash 4.3 or newer. It parses each flat JSON object,
rejects unknown or duplicate fields, validates field types and totals, checks
structured NOREF rows, verifies report provenance, and joins every disposition,
scope, owner, and subsystem. It contacts no external service.

A full generation writes the compact
`test/conformance/parity_dashboard.tsv` snapshot and renders
`docs/PARITY_STATUS.md` from it. The snapshot-only check is the fast test gate;
it requires the recorded compiler commit to be an available ancestor and
verifies its source digest. It also compares the recorded dependency trees,
corpus trees and file lists, and all dashboard manifests with the current
inputs. Snapshot validation reconciles suite, all-view, scoped-view, and owner
totals. Full-report validation applies the same identity checks and requires
all reports to agree on the exact compiler binary used. The dashboard reports
suite totals, scoped totals, rates, and subsystem ownership for non-passing
results. The scoped view excludes only `coarray`, `OpenMP`, `OpenACC`, and
`GPU` tags. A scoped passing file is excluded from both numerator and
denominator.

### Seed baseline

Full run against local GCC checkout: `PASS=1173`, `XFAIL=0`,
`XPASS=0`, `FAIL=2395`, `NOREF=0`, `SKIP=2299`, `TOTAL=5867`.
The xfail manifest (`test/conformance/xfail_gfortran_dg.txt`) is seeded
from the FAIL records of this run.

### Checked-in dashboard measurement

The stale checked-in dashboard was measured at GCC revision
`395e3d8131c189cd58e8c8061cdc77d1c44e3822`. It records `PASS=1175`,
`XFAIL=2132`, `XPASS=0`, `FAIL=333`, `NOREF=1`, `SKIP=2298`,
`WARNING_UNCHECKED=75`, and `TOTAL=5938`. The current manifest inventory is
listed above.

## LFortran integration tests

The `lfortran` suite reads `$FFC_LFORTRAN_DIR/integration_tests/*.f90`.
The default root is `../lfortran`. To use another checkout:

```bash
FFC_LFORTRAN_DIR=/path/to/lfortran \
    scripts/conformance_gauntlet.sh --suite lfortran \
    --report "$TMPDIR/ffc_lfortran.jsonl"
```

No lfortran source is copied into this repository. The checked-in files
are:

- `test/conformance/xfail_lfortran.txt`
- `test/conformance/skip_lfortran.txt`

Seed baseline from lfortran commit `5e3229bd6`: `PASS=123`,
`XFAIL=4134`, `XPASS=0`, `FAIL=0`, `NOREF=72`, `SKIP=0`,
`TOTAL=4257`.

The stale checked-in dashboard was measured at LFortran revision
`caf87b660f803148f000046392a5da803f9fc630`. It records `PASS=849`,
`XFAIL=3419`, `XPASS=0`, `FAIL=11`, `NOREF=145`, `SKIP=1`, and `TOTAL=4280`.
The current manifest inventory is listed above.

## Separate compilation

A test program often USEs a module defined in a sibling file in the same
suite directory. The runner compiles those prerequisites first so the
program links. Before building a file, it resolves the modules the file
USEs that no `module` unit in the file itself defines. For each such
module it finds the sibling `.f90`/`.lf` file whose `module <name>`
matches (case-insensitive; `module procedure` and `submodule` are not
definitions), follows module-to-module dependencies transitively, and
appends any submodule files that implement a pulled-in module's
interfaces. The prerequisites compile in dependency order with ffc into
a per-test include directory, emitting their `.fmod` files there; the
main file then builds with `-I <that dir>` plus the prerequisite object
files. The `gfortran -w` reference compiles the same sibling sources, so
its binary links too and the comparison is honest.

Before either compiler runs, the gauntlet copies the selected source and every
recursively resolved Fortran INCLUDE file into private per-case storage. It
preserves their suite-relative layout, records each canonical relative name and
SHA-256 in the dependency closure, and passes the copied source and include
directories to both compilers. The closure hash and compiler input therefore
describe the same bytes even if the corpus checkout changes during the case.
Sibling module and explicit extra sources use the same snapshot rule before
they compile. Missing INCLUDE names remain missing in the private snapshot and
contribute a canonical missing entry to the closure, so a file created later in
the corpus cannot change that observation.

An absolute INCLUDE name or a relative INCLUDE that escapes the suite and its
declared include roots has no portable snapshot layout. A locked run records a
raw setup failure for that case instead of reading an untracked path after
hashing. Such a dependency must first be placed under the corpus or a declared
include root.

A file that defines only modules and no program keeps the single-file
handling. A self-contained file resolves to no prerequisites and builds
exactly as before. When ffc cannot compile a prerequisite module, the
main file's `-I` search finds no `.fmod` and the build fails as it would
without separate compilation; the reference still receives the full
source list. The `gfortran-dg` suite models multifile cases through its
own `dg-additional-sources` directive and does not use this resolution.

Per-case source snapshots and build artifacts live under `TMPDIR`; no foreign
source is added to the repository.

## xfail promotion workflow

When an entry in an xfail manifest starts passing (XPASS), promote it
by removing its line from the manifest. This is a soft signal that the
feature is now supported.

```bash
# Find XPASS entries in the latest report
grep '"status":"XPASS"' "$TMPDIR/ffc_gauntlet_fortfront_f90.jsonl"

# Remove the promoted entry from the manifest
sed -i '/^example\.f90$/d' test/conformance/xfail_fortfront_f90.txt
```

## Smoke test

The fpm test `test_conformance_gauntlet_smoke` exercises the runner
against `fortfront-f90` with a 20-file cap. It runs under 120 seconds.
It requires runner exit 0 and a SUMMARY record with zero `FAIL`
entries.

```bash
LIBRARY_PATH=../liric/build fpm test test_conformance_gauntlet_smoke
```

## Shared helpers

`scripts/lib_conformance.sh` provides shell functions used by the
gauntlet runner:

- `find_ffc`: resolve ffc binary path
- `compile_with_ffc`: compile through ffc, with extra `-I`/object args
- `compile_with_gfortran`: compile with `gfortran -w`, prerequisite sources first
- `run_capture`: run with timeout, capture stdout+stderr
- `compare_outputs`: compare stdout files and exit statuses
- `build_module_index`: map sibling module and submodule names to their files
- `resolve_prerequisites`: order the sibling files a source must compile first

Source this file from other scripts; do not execute it directly.

`scripts/lib_conformance_observation.sh` validates raw observation JSONL and
materializes classified views. `scripts/classify_conformance_observations.sh`
is its command-line entry point; it never loads or runs `ffc`.
