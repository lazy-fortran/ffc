# Conformance gauntlet

Drive external Fortran test corpora through the full `ffc` pipeline
(compile to a native binary, run it, compare against a `gfortran -w`
reference) and produce an xfail-style report.

## No-vendoring rule

The runner never copies source files from external suites into this
repository. Only manifests (lists of filenames plus expected
disposition) live here. External checkouts are referenced by path.

## Suites

| Suite | Source | Extension | Reference |
|---|---|---|---|
| `fortfront-f90` | FortFront standard-mode examples | `.f90` | `gfortran -w` |
| `fortfront-lf` | FortFront lazy-mode examples | `.lf` | ffc only (gfortran cannot compile lazy Fortran) |
| `lfortran` | LFortran integration tests | `.f90` | `gfortran -w` |
| `gfortran-dg` | GCC gfortran.dg testsuite | `.f90` | `gfortran -w` |

## Environment variables

Set these to point at local checkouts of the external repositories.
Defaults assume sibling directories under the parent of the ffc checkout.

| Variable | Default | Suite |
|---|---|---|
| `FFC_FORTFRONT_DIR` | `../fortfront` | `fortfront-f90`, `fortfront-lf` |
| `FFC_LFORTRAN_DIR` | `../lfortran` | `lfortran` |
| `FFC_GFORTRAN_DG_DIR` | `../gcc/gcc/testsuite/gfortran.dg` | `gfortran-dg` |

A suite whose root directory does not exist prints `SKIP: <suite> not
found at <path>` and exits 0. This keeps CI green when a suite is not
checked out locally.

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

A final SUMMARY record closes the file:

```json
{"suite":"fortfront-f90","status":"SUMMARY","pass":15,"xfail":3,"xpass":1,"fail":2}
```

## Disposition states

| State | Meaning | Gate impact |
|---|---|---|
| `PASS` | ffc built, ran, and matched reference | None |
| `XFAIL` | Listed in xfail manifest; ffc failed as expected | None |
| `XPASS` | Listed in xfail manifest; ffc passed unexpectedly | Visible in summary; promote the entry |
| `FAIL` | Not in xfail manifest; ffc failed or mismatched | Fails the gate |

The runner exits nonzero if any `FAIL` record exists. `XFAIL` and
`XPASS` never cause a nonzero exit.

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

`scripts/lib_conformance.sh` provides shell functions shared between
the gauntlet runner and the FortFront corpus conformance test:

- `find_ffc`: resolve ffc binary path
- `compile_with_ffc`: compile through ffc
- `compile_with_gfortran`: compile with `gfortran -w`
- `run_capture`: run with timeout, capture stdout+stderr
- `compare_outputs`: compare stdout files and exit statuses

Source this file from other scripts; do not execute it directly.
