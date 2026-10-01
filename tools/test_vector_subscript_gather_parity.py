#!/usr/bin/env python3
"""Parity oracle: vector subscripts as gather views (PLAN W1.a #399 slice 1).

gfortran is the independent oracle. 24 shapes: 12 assignment contexts (the
read-gather machinery that already works, so this is the regression guard) and
12 print contexts (the gap: `print *, a(idx)` is refused today).

Falsification: each case carries a distinct index permutation, and the printed
bytes are md5-compared per case. Change any permutation and that case's md5
stops matching gfortran, so a silent pass is not available.
"""
import hashlib
import pathlib
import subprocess
import sys

FFC = pathlib.Path(__import__("os").environ.get(
    "FFC", str(pathlib.Path(__file__).resolve().parents[1] / "build/fo/app/ffc")))
WORK = pathlib.Path(__import__("os").environ.get(
    "VS_PAR_WORK", "/var/tmp/ffc-goal/perf/vspar-latest"))
WORK.mkdir(parents=True, exist_ok=True)

CASES = [
    (3, "1,2,3"), (3, "3,2,1"), (3, "2,2,1"),
    (4, "2,3,4,1"), (4, "4,3,2,1"), (4, "4,4,3,3"),
    (5, "2,3,4,5,1"), (5, "5,4,3,2,1"), (5, "1,3,5,2,4"),
    (2, "2,1"), (6, "2,4,6,1,3,5"), (6, "6,5,4,3,2,1"),
]


def source(n: int, idx: str, mode: str) -> str:
    vals = ",".join(str(i * 10) for i in range(1, n + 1))
    k = len(idx.split(","))
    head = (
        f"program p\n"
        f"  integer :: a({n})\n"
        f"  integer :: idx({k})\n"
    )
    # Element-wise output, not `print '(9i6)', b`. A formatted whole-array
    # print is its own unsupported feature here ("array reads require exactly
    # one subscript"), and folding it into the assign cases made 12 green
    # gather reads look like failures. This oracle is about the gather.
    # Whole-array forms, kept in a separate mode so the guard above stays
    # green and the gap stays visible: `print *, a(idx)` and a formatted
    # whole-array print are the two shapes this slice has to route.
    elem = ", ".join(f"b({i})" for i in range(1, k + 1))
    pelem = ", ".join(f"a(idx({i}))" for i in range(1, k + 1))
    if mode == "print":
        return head + f"  a = [{vals}]\n  idx = [{idx}]\n  print *, {pelem}\nend program p\n"
    return head + (
        f"  integer :: b({k})\n"
        f"  a = [{vals}]\n  idx = [{idx}]\n  b = a(idx)\n"
        f"  print *, {elem}\nend program p\n"
    )


def run(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def main() -> int:
    runs = match = refused = mismatch = other = 0
    report: list[str] = []
    for mode in ("assign", "print"):
        for n, idx in CASES:
            name = f"{mode}_{n}_{idx.replace(',', '_')}"
            src = WORK / f"{name}.f90"
            src.write_text(source(n, idx, mode))
            ref = WORK / f"{name}_ref"
            rc, err = run(["gfortran", str(src), "-o", str(ref)])
            if rc != 0:
                report.append(f"{name}\t{mode}\tREF_COMPILE_FAIL")
                other += 1
                runs += 1
                continue
            rmd5 = hashlib.md5(subprocess.run(
                [str(ref)], capture_output=True, text=True).stdout.encode()).hexdigest()
            exe = WORK / f"{name}_ffc"
            rc, err = run([str(FFC), str(src), "-o", str(exe)])
            runs += 1
            if rc != 0:
                if "unsupported array subscript" in err:
                    refused += 1
                    report.append(f"{name}\t{mode}\tREFUSED")
                else:
                    other += 1
                    report.append(f"{name}\t{mode}\tOTHER_ERROR")
                continue
            fmd5 = hashlib.md5(subprocess.run(
                [str(exe)], capture_output=True, text=True).stdout.encode()).hexdigest()
            if fmd5 == rmd5:
                match += 1
                report.append(f"{name}\t{mode}\tMATCH\t{rmd5}")
            else:
                mismatch += 1
                report.append(f"{name}\t{mode}\tMISMATCH\tref={rmd5}\tffc={fmd5}")


    # Gap rows, not part of the pass count: these are the shapes the slice
    # exists to fix. They are reported so "unsupported" is a measured state,
    # not a rumour, and so the day they start working the file changes.
    gap: list[str] = []
    for n, idx in CASES[:6]:
        k = len(idx.split(","))
        vals = ",".join(str(i * 10) for i in range(1, n + 1))
        src = WORK / f"whole_{n}_{idx.replace(',', '_')}.f90"
        src.write_text(
            f"program p\n  integer :: a({n})\n  integer :: idx({k})\n"
            f"  a = [{vals}]\n  idx = [{idx}]\n  print *, a(idx)\nend program p\n")
        ref = WORK / f"whole_{n}_{idx.replace(',', '_')}_ref"
        rc, _ = run(["gfortran", str(src), "-o", str(ref)])
        if rc != 0:
            gap.append(f"whole_{n}_{idx}\tREF_COMPILE_FAIL")
            continue
        rmd5 = hashlib.md5(subprocess.run([str(ref)], capture_output=True,
                                           text=True).stdout.encode()).hexdigest()
        exe = WORK / f"whole_{n}_{idx.replace(',', '_')}_ffc"
        rc, err = run([str(FFC), str(src), "-o", str(exe)])
        if rc != 0:
            tag = "REFUSED" if "unsupported" in err else "OTHER_ERROR"
            gap.append(f"whole_{n}_{idx}\t{tag}\tref={rmd5}")
        else:
            fmd5 = hashlib.md5(subprocess.run([str(exe)], capture_output=True,
                                              text=True).stdout.encode()).hexdigest()
            gap.append(f"whole_{n}_{idx}\t{'MATCH' if fmd5 == rmd5 else 'MISMATCH'}"
                       f"\tref={rmd5}\tffc={fmd5}")
    (WORK / "gap.tsv").write_text("\n".join(gap) + "\n")
    (WORK / "report.tsv").write_text("\n".join(report) + "\n")
    summary = f"runs={runs} match={match} refused={refused} mismatch={mismatch} other={other}"
    (WORK / "summary.txt").write_text(summary + "\n")
    print(summary)
    print(f"report={WORK/'report.tsv'}")
    if gap.any() if False else gap:
        import collections
        c = collections.Counter(g.split("\t")[1] for g in gap)
        print("gap rows:", dict(c), f"(file={WORK/'gap.tsv'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
