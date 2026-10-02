#!/usr/bin/env python3
"""Byte-exact parity for string-search intrinsics; pins the ignored `back=` (#764).

`index(s, sub, back=.true.)` is lowered as a forward search: the third argument is
dropped, so ffc reports the leftmost match where Fortran requires the rightmost.
Without `back` the compilers agree, which is exactly why this hides - it misreports
only when the caller explicitly asks for the rightmost match, and it returns a
plausible integer with no diagnostic. This is a wrong answer on **valid** code, a
more severe class than the over-acceptance rows in
test_name_namespace_collision_parity.py, so it gets its own status label.

`KNOWN_WRONG_ANSWER` rows are named and counted, never deleted and never counted as
passes; each flips to MATCH when #764 is honoured or made into an honest refusal
(a refusal is a strict improvement over a silent wrong index, and this oracle treats
a refusal as progress, not failure).

Verified NOT defects and kept in the green set rather than assumed: `len(trim(s))`
looks wrong (`0` vs `8`) only when `s` is undefined - every row here assigns its
characters first, because an undefined variable is not evidence about a compiler.

Run:  python3 tools/test_string_search_back_parity.py
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FFC = ROOT / "build" / "fo" / "app" / "ffc"
WORK = Path("/var/tmp/ffc-goal/perf/ssback")
REPORT = Path("/var/tmp/ffc-goal/perf/ssback/report.tsv")

KNOWN_WRONG_ANSWER = [
    "index_back_literal_named", "index_back_literal_positional",
    "index_back_multi", "index_back_dotted_arg", "scan_back",
    "index_back_named", "index_back_positional",
]

# Character variables are deliberately given the EXACT content length. Padding is
# not cosmetic here: a `character(len=8) :: t` holding "ab" is the eight-character
# string "ab      ", and `index("abcabc", t)` is correctly 0 because that padded
# needle never occurs. The first version of this file used padded dummies and so
# scored the #764 defect as MATCH - the test neutralised the very bug it was
# written to catch. Literal needles, or exactly-sized variables, are mandatory.
DECL = 'character(len=6) :: s\n  character(len=2) :: t\n  s="abcabc"\n  t="ab"'

CASES = [
    # forward search - correct today, must stay correct
    ("index_fwd_basic", 'print *, index("abcabc","abc")'),
    ("index_fwd_absent", 'print *, index("abcabc","zzz")'),
    ("index_fwd_first", 'print *, index("zabzab","ab")'),
    ("index_var", "print *, index(s,t)"),
    ("index_at_end", 'print *, index("aaaab","ab")'),
    ("index_single_char", 'print *, index("abc","c")'),
    ("index_whole", 'print *, index("abc","abc")'),
    ("index_empty_needle", 'print *, index("abc","")'),
    # back=.true. - wrong today (#764)
    ("index_back_named", "print *, index(s,t,back=.true.)"),
    ("index_back_positional", "print *, index(s,t,.true.)"),
    ("index_back_literal_named", 'print *, index("abcabc","abc",back=.true.)'),
    ("index_back_literal_positional", 'print *, index("abcabc","abc",.true.)'),
    ("index_back_multi", 'print *, index("zabzab","ab",back=.true.)'),
    ("index_back_absent", 'print *, index("abcabc","zzz",back=.true.)'),
    ("index_back_dotted_arg", 'print *, index("abcabc","bc",.true.)'),
    # other search intrinsics - correct today
    ("scan_first", 'print *, scan("aabbc","ab")'),
    ("scan_absent", 'print *, scan("xyz","ab")'),
    ("scan_back", 'print *, scan("aabbc","ab",back=.true.)'),
    ("verify_first", 'print *, verify("aabc","ab")'),
    ("verify_back", 'print *, verify("aabcc","abc",back=.true.)'),
    ("index_len_interact", "print *, index(s,t), len(t)"),
    ("len_trim_assigned", 'print *, len(trim("ab      "))'),
    ("len_trim_blank", 'print *, len(trim("     "))'),
    ("adjustl_len", 'print *, len(adjustl("  ab"))'),
    ("repeat_len", 'print *, len(repeat("ab",3))'),
    ("achar_iachar", 'print *, achar(iachar("A")+1)'),
]


def run(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return p.returncode, p.stdout


def main() -> int:
    if not FFC.exists():
        print(f"SKIP: ffc not built at {FFC}", file=sys.stderr)
        return 0
    WORK.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    runs = match = wrong = refused = 0
    for name, stmt in CASES:
        src = WORK / f"{name}.f90"
        src.write_text("program p\n  implicit none\n  " + DECL + "\n    " + stmt +
                       "\nend program p\n")
        grc, _ = run(["gfortran", str(src), "-o", str(WORK / f"{name}_r")])
        if grc != 0:
            lines.append(f"{name}\tBAD_CASE\tgfortran_refuses")
            continue
        frc, ferr = run([str(FFC), str(src), "-o", str(WORK / f"{name}_f")])
        if frc != 0:
            runs += 1
            if name in KNOWN_WRONG_ANSWER:
                refused += 1
                lines.append(f"{name}\tKNOWN_REFUSED\tprogress_over_wrong_answer")
                continue
            lines.append(f"{name}\tUNEXPECTED_REFUSE\t{ferr.strip()[:36]}")
            continue
        _, gout = run([str(WORK / f"{name}_r")])
        _, fout = run([str(WORK / f"{name}_f")])
        gm = hashlib.md5(gout.encode()).hexdigest()
        fm = hashlib.md5(fout.encode()).hexdigest()
        runs += 1
        if gm == fm:
            match += 1
            lines.append(f"{name}\tMATCH\t{gm}")
        elif name in KNOWN_WRONG_ANSWER:
            wrong += 1
            lines.append(f"{name}\tKNOWN_WRONG_ANSWER\tg={gout.strip()!r}"
                         f"\tffc={fout.strip()!r}")
        else:
            lines.append(f"{name}\tREGRESSION\tg={gout.strip()!r}\tffc={fout.strip()!r}")
    REPORT.write_text("\n".join(lines) + "\n")
    print(f"runs={runs} match={match} known_wrong={wrong} known_refused={refused} "
          f"regressions={runs - match - wrong - refused}")
    print(f"report={REPORT}")
    unexpected = [l for l in lines if l.split("\t")[1] in ("REGRESSION",
                "UNEXPECTED_REFUSE", "BAD_CASE")]
    return 1 if unexpected else 0


if __name__ == "__main__":
    sys.exit(main())
