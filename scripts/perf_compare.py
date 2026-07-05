#!/usr/bin/env python3
"""Compile and runtime performance comparison for ffc vs oracle compilers.

Compiles every benchmark under each toolchain, verifies runtime output against
the gfortran -O2 reference, and reports best-of-N compile and run wall times.

Toolchains:
  gfortran-O0, gfortran-O2      always (if gfortran on PATH)
  ffc-isel, ffc-copy-patch      native liric backends (O0-class direct emission)
  ffc-llvm-O0, ffc-llvm-O2      LLVM liric backend, only with --ffc-llvm <bin>
  lfortran                      only when --lfortran <bin> is given

The ffc-llvm toolchains need a second ffc binary linked against the LLVM liric
build. Build it (from an LLVM-enabled liric at <liric>/build_llvm) by adding the
LLVM link lib to fpm.toml temporarily, `LIBRARY_PATH=<liric>/build_llvm fpm
build`, then reverting fpm.toml, and pass that build's app/ffc via --ffc-llvm.
The shipped ffc links native liric and does not need LLVM.

Usage:
  LIBRARY_PATH=<liric-build> scripts/perf_compare.py \
      --ffc build/fo/bin/ffc [--ffc-llvm <llvm-ffc>] [--lfortran <bin>] \
      [--repeats 3] [--out /tmp/ffc_perf.md]
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH_DIR = os.path.join(REPO, "scripts", "perf_benchmarks")

FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eEdD][-+]?\d+)?")


def norm_output(text):
    """Tokenize into floats (tolerant) and other tokens for comparison."""
    toks = []
    for tok in text.split():
        m = FLOAT_RE.fullmatch(tok)
        if m:
            toks.append(("f", float(tok.replace("d", "e").replace("D", "e"))))
        else:
            toks.append(("s", tok))
    return toks


def outputs_match(a, b, rtol=1e-9):
    ta, tb = norm_output(a), norm_output(b)
    if len(ta) != len(tb):
        return False
    for (ka, va), (kb, vb) in zip(ta, tb):
        if ka != kb:
            return False
        if ka == "f":
            scale = max(1.0, abs(va), abs(vb))
            if abs(va - vb) > rtol * scale:
                return False
        elif va != vb:
            return False
    return True


def best_time(fn, repeats):
    best = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        rc, out = fn()
        dt = time.perf_counter() - t0
        if rc != 0:
            return None, out, rc
        best = dt if best is None else min(best, dt)
    return best, out, 0


def run(cmd, timeout=120):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT"


def make_toolchains(args):
    tc = {}
    if shutil.which("gfortran"):
        tc["gfortran-O0"] = lambda src, exe: ["gfortran", "-O0", "-w", src, "-o", exe]
        tc["gfortran-O2"] = lambda src, exe: ["gfortran", "-O2", "-w", src, "-o", exe]
    tc["ffc-isel"] = lambda src, exe: [args.ffc, src, "-o", exe, "--backend", "isel"]
    tc["ffc-copy-patch"] = lambda src, exe: [
        args.ffc, src, "-o", exe, "--backend", "copy-patch"]
    if args.ffc_llvm:
        tc["ffc-llvm-O0"] = lambda src, exe: [
            args.ffc_llvm, src, "-o", exe, "--backend", "llvm", "-O0"]
        tc["ffc-llvm-O2"] = lambda src, exe: [
            args.ffc_llvm, src, "-o", exe, "--backend", "llvm", "-O2"]
    if args.lfortran:
        tc["lfortran"] = lambda src, exe: [args.lfortran, src, "-o", exe]
    return tc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ffc", required=True)
    ap.add_argument("--ffc-llvm", default=None,
                    help="ffc built against the LLVM liric backend "
                         "(adds ffc-llvm-O0 and ffc-llvm-O2 toolchains)")
    ap.add_argument("--lfortran", default=None)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="/tmp/ffc_perf.md")
    args = ap.parse_args()

    benches = sorted(
        os.path.join(BENCH_DIR, f)
        for f in os.listdir(BENCH_DIR)
        if f.endswith(".f90"))
    toolchains = make_toolchains(args)
    order = ["gfortran-O0", "gfortran-O2", "ffc-isel", "ffc-copy-patch",
             "ffc-llvm-O0", "ffc-llvm-O2", "lfortran"]
    order = [t for t in order if t in toolchains]

    tmp = tempfile.mkdtemp(prefix="ffc_perf_")
    rows = []
    for src in benches:
        name = os.path.basename(src)[:-4]
        ref_out = None
        cells = {}
        for tc in order:
            exe = os.path.join(tmp, f"{name}.{tc}")
            ctime, cout, _ = best_time(
                lambda: run(toolchains[tc](src, exe)), args.repeats)
            if ctime is None:
                cells[tc] = ("compile-fail", None, None, cout.strip()[:60])
                continue
            rtime, rout, _ = best_time(lambda: run([exe]), args.repeats)
            if rtime is None:
                cells[tc] = ("run-fail", ctime, None, rout.strip()[:60])
                continue
            cells[tc] = ("ok", ctime, rtime, rout)
            if tc == "gfortran-O2":
                ref_out = rout
        # correctness vs gfortran-O2 reference
        for tc in order:
            st = cells[tc]
            if st[0] == "ok" and ref_out is not None:
                ok = outputs_match(st[3], ref_out)
                cells[tc] = (st[0], st[1], st[2], st[3], ok)
        rows.append((name, cells))

    # markdown report
    lines = ["# ffc performance comparison", ""]
    lines.append(f"ffc: `{args.ffc}`  repeats: {args.repeats}  "
                 f"reference: gfortran-O2 output")
    lines.append("")
    lines.append("Compile / run are best-of-N wall time in milliseconds. "
                 "OK = runtime output matches gfortran-O2.")
    lines.append("")
    header = "| benchmark | metric | " + " | ".join(order) + " |"
    sep = "|" + "---|" * (len(order) + 2)
    lines.append(header)
    lines.append(sep)
    for name, cells in rows:
        def fmt_compile(tc):
            st = cells[tc]
            if st[0] in ("compile-fail",):
                return "FAIL"
            return f"{st[1]*1000:.1f}" if st[1] is not None else "-"

        def fmt_run(tc):
            st = cells[tc]
            if st[0] == "compile-fail":
                return "-"
            if st[0] == "run-fail":
                return "RUNFAIL"
            rt = f"{st[2]*1000:.1f}" if st[2] is not None else "-"
            ok = ""
            if len(st) >= 5:
                ok = " ok" if st[4] else " **X**"
            return rt + ok
        lines.append(f"| {name} | compile ms | "
                     + " | ".join(fmt_compile(t) for t in order) + " |")
        lines.append(f"| {name} | run ms | "
                     + " | ".join(fmt_run(t) for t in order) + " |")

    if not args.lfortran:
        lines.append("")
        lines.append("_lfortran column omitted: no lfortran binary provided "
                     "(pass --lfortran <bin>)._")

    report = "\n".join(lines) + "\n"
    with open(args.out, "w") as f:
        f.write(report)
    print(report)
    print(f"[written to {args.out}]", file=sys.stderr)


if __name__ == "__main__":
    main()
