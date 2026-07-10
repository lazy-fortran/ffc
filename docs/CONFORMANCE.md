# Conformance gauntlet

Drive external Fortran test corpora through the full `ffc` pipeline.
The runner compiles each source to a native binary, runs it, compares
standard Fortran output against `gfortran -w` when gfortran accepts the
source, and writes an xfail-style report.

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

```bash
scripts/conformance_gauntlet.sh --suite SUITE [OPTIONS]
```

Options:

| Flag | Description |
|---|---|
| `--suite SUITE` | Required. One of `fortfront-f90`, `fortfront-lf`, `lfortran`, `gfortran-dg` |
| `--ffc PATH` | Path to the `ffc` binary. Auto-discovered from `build/` or `PATH` if omitted. |
| `--report PATH` | JSONL report path. Defaults to `/tmp/ffc_gauntlet_<suite>.jsonl`. |
| `--file PATH` | Select one suite-relative file. Repeat to select more files. |
| `--files-from PATH` | Read suite-relative files from a list. Repeat to read more lists. |
| `--max-files N` | Only test the first N files. Use for smoke runs. |
| `--timeout N` | Per-file timeout in seconds. Default: 5. |

Smoke run (20 files, auto-discovers ffc):

```bash
scripts/conformance_gauntlet.sh --suite fortfront-f90 --max-files 20
```

Full run (all files, explicit ffc path):

```bash
scripts/conformance_gauntlet.sh --suite fortfront-f90 \
    --ffc "$(find build -name ffc -type f -executable | head -1)"
```

### Named files

Run one external regression without scanning the full corpus:

```bash
scripts/conformance_gauntlet.sh --suite fortfront-f90 \
    --file ast_coverage_control_flow.f90 \
    --report /tmp/ffc-one.jsonl
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
    --files-from /tmp/ffc-scope-files.txt
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
| `test/conformance/xfail_fortfront_f90.txt` | 100 |
| `test/conformance/xfail_fortfront_lf.txt` | 59 |
| `test/conformance/undefined_output_fortfront_f90.txt` | 3 |
| `test/conformance/xfail_lfortran.txt` | 3421 |
| `test/conformance/xfail_gfortran_dg.txt` | 2136 |
| `test/conformance/skip_lfortran.txt` | 0 |
| `test/conformance/skip_gfortran_dg.txt` | 2298 |

Use `docs/PARITY_PLAN.md` and issue #299 for the latest full-suite pass-rate
snapshot. The seed baselines below are historical starting points, not current
scoreboard values.

## JSONL output

One record per attempted file:

```json
{"suite":"fortfront-f90","file":"example.f90","status":"PASS","ffc_exit":0,"ref_exit":0,"note":"output matches gfortran"}
```

Fields:

| Field | Type | Description |
|---|---|---|
| `suite` | string | Suite name |
| `file` | string | File basename (suite-relative path) |
| `status` | string | `PASS`, `XFAIL`, `XPASS`, or `FAIL` |
| `ffc_exit` | int | ffc exit code (0 = built and ran) |
| `ref_exit` | int | gfortran exit code (0 = built and ran) |
| `note` | string | Human-readable explanation |
| `warning_expectation` | string | `unchecked` for warning-only gfortran.dg files; omitted otherwise |

A final SUMMARY record closes the file:

```json
{"suite":"fortfront-f90","status":"SUMMARY","pass":15,"xfail":3,"xpass":1,"fail":2,"noref":1,"skip":0,"warning_unchecked":0,"total":21}
```

## Disposition states

| State | Meaning | Gate impact |
|---|---|---|
| `PASS` | ffc built and ran; standard files matched gfortran when gfortran accepted the source | None |
| `XFAIL` | Listed in xfail manifest; ffc failed as expected | None |
| `XPASS` | Listed in xfail manifest; ffc passed unexpectedly | Visible in summary; promote the entry |
| `FAIL` | Not in xfail manifest; ffc failed or mismatched | Fails the gate |
| `SKIP` | Listed in a skip manifest because the runner does not model the case | Counted in summary |

The runner exits nonzero if any `FAIL` record exists. `XFAIL` and
`XPASS` never cause a nonzero exit.

The `noref` summary count is the number of standard Fortran files where
`gfortran -w` rejected the source after ffc compiled and ran it. These
records are `PASS` unless the file is still listed in the xfail manifest,
in which case they are `XPASS`.

The `skip` summary count is the number of files listed in
`test/conformance/skip_<suite>.txt`. They are explicit entries, not silent
drops.

The `warning_unchecked` count is the number of warning-only gfortran.dg files
whose compile or run disposition was checked without matching warning text.

## FortFront corpus gate

The fpm test `test_fortfront_corpus_conformance` runs the full
`fortfront-f90` and `fortfront-lf` suites through the gauntlet with
reports under `/tmp`. Passing files need no manifest entry. A new
FortFront example fails the ffc test until ffc supports it or its
basename is added to the matching xfail manifest.

The maintained fpm test rejects both FAIL and XPASS records. An XPASS is a
stale manifest entry and must be promoted before merging.

Current xfail manifests:

- `test/conformance/xfail_fortfront_f90.txt`
- `test/conformance/xfail_fortfront_lf.txt`

`test/conformance/undefined_output_fortfront_f90.txt` lists programs whose
printed values depend on uninitialized data. Both ffc and gfortran must build
and terminate with exit zero. The runner ignores stdout only for those named
files; compilation, control flow, and termination remain enforced.
Undefined-output entries must not also appear in an xfail or skip manifest.

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

### Seed baseline

Full run against local GCC checkout: `PASS=1173`, `XFAIL=0`,
`XPASS=0`, `FAIL=2395`, `NOREF=0`, `SKIP=2299`, `TOTAL=5867`.
The xfail manifest (`test/conformance/xfail_gfortran_dg.txt`) is seeded
from the FAIL records of this run.

### Current pinned measurement

At GCC revision `395e3d8131c189cd58e8c8061cdc77d1c44e3822`, the post-cleanup
summary is `PASS=1175`, `XFAIL=2132`, `XPASS=0`, `FAIL=333`, `NOREF=1`,
`SKIP=2298`, `WARNING_UNCHECKED=75`, `TOTAL=5938`.

## LFortran integration tests

The `lfortran` suite reads `$FFC_LFORTRAN_DIR/integration_tests/*.f90`.
The default root is `../lfortran`. To use another checkout:

```bash
FFC_LFORTRAN_DIR=/path/to/lfortran \
    scripts/conformance_gauntlet.sh --suite lfortran \
    --report /tmp/ffc_lfortran.jsonl
```

No lfortran source is copied into this repository. The checked-in files
are:

- `test/conformance/xfail_lfortran.txt`
- `test/conformance/skip_lfortran.txt`

Seed baseline from lfortran commit `5e3229bd6`: `PASS=123`,
`XFAIL=4134`, `XPASS=0`, `FAIL=0`, `NOREF=72`, `SKIP=0`,
`TOTAL=4257`.

At LFortran revision `caf87b660f803148f000046392a5da803f9fc630`, the current
summary is `PASS=848`, `XFAIL=3421`, `XPASS=0`, `FAIL=11`, `NOREF=145`,
`SKIP=0`, `TOTAL=4280`.

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

A file that defines only modules and no program keeps the single-file
handling. A self-contained file resolves to no prerequisites and builds
exactly as before. When ffc cannot compile a prerequisite module, the
main file's `-I` search finds no `.fmod` and the build fails as it would
without separate compilation; the reference still receives the full
source list. The `gfortran-dg` suite models multifile cases through its
own `dg-additional-sources` directive and does not use this resolution.

Only build artifacts live under `TMPDIR`; the no-vendoring rule holds,
since prerequisite sources are referenced in place by path.

## xfail promotion workflow

When an entry in an xfail manifest starts passing (XPASS), promote it
by removing its line from the manifest. This is a soft signal that the
feature is now supported.

```bash
# Find XPASS entries in the latest report
grep '"status":"XPASS"' /tmp/ffc_gauntlet_fortfront_f90.jsonl

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
