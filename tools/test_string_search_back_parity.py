#!/usr/bin/env python3
"""Byte-exact parity for string-search intrinsics; #764 `back=` honoured.

`index(s, sub, back=.true.)` and `scan(s, set, back=.true.)` used to be
lowered as forward searches - the third argument was dropped, so ffc
reported the leftmost match where Fortran requires the rightmost. Fixed in
this file's green set: BACK is now resolved (positional or keyword, literal
or runtime expression) and turns the match block into scan-on - the result
overwrites itself and the LAST match survives. `verify` ignored BACK too
(the earlier claim that it honoured BACK came from a coincidental row where
every character was in the set, making forward and backward agree at 0);
`verify_back_real` below is the discriminating case for it.

A wrong answer on valid code is more severe than the over-acceptance rows in
test_name_namespace_collision_parity.py, so the pre-fix rows carried their
own status label; with the fix landed the wrong-answer list is empty and any
mismatch on these rows is now a REGRESSION that fails the oracle.
KNOWN_REFUSED pins the honest refusals kept by name: KIND=8 (results are
INTEGER(4) only) and misspelled keywords - a dropped keyword is exactly how
#764 hid, so refusing it loudly is part of the fix.

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

KNOWN_WRONG_ANSWER = []  # empty since the #764 fix; any wrong answer is a REGRESSION

KNOWN_REFUSED = [
    "index_kind8_refused", "index_bad_keyword_refused", "scan_kind8_refused",
]

# Character variables are deliberately given the EXACT content length. Padding is
# not cosmetic here: a `character(len=8) :: t` holding "ab" is the eight-character
# string "ab      ", and `index("abcabc", t)` is correctly 0 because that padded
# needle never occurs. The first version of this file used padded dummies and so
# scored the #764 defect as MATCH - the test neutralised the very bug it was
# written to catch. Literal needles, or exactly-sized variables, are mandatory.
DECL = ('character(len=6) :: s\n  character(len=2) :: t\n  logical :: bt\n'
        '  s="abcabc"\n  t="ab"\n  bt=.true.')

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
    # back=.true. - fixed (#764), pinned against regressions
    ("index_back_named", "print *, index(s,t,back=.true.)"),
    ("index_back_positional", "print *, index(s,t,.true.)"),
    ("index_back_literal_named", 'print *, index("abcabc","abc",back=.true.)'),
    ("index_back_literal_positional", 'print *, index("abcabc","abc",.true.)'),
    ("index_back_multi", 'print *, index("zabzab","ab",back=.true.)'),
    ("index_back_absent", 'print *, index("abcabc","zzz",back=.true.)'),
    ("index_back_dotted_arg", 'print *, index("abcabc","bc",.true.)'),
    ("index_back_false_pos", 'print *, index("abcabc","abc",.false.)'),
    ("index_back_var", "print *, index(s,t,back=bt)"),
    ("index_back_computed", "print *, index(s,t,back=len(s)>1)"),
    ("index_back_overlap", 'print *, index("aaaa","aa",back=.true.)'),
    ("index_pos_kind4", 'print *, index("abcabc","abc",.true.,4)'),
    # other search intrinsics - correct today
    ("scan_first", 'print *, scan("aabbc","ab")'),
    ("scan_absent", 'print *, scan("xyz","ab")'),
    ("scan_back", 'print *, scan("aabbc","ab",back=.true.)'),
    ("scan_back_rightmost", 'print *, scan("ccbbaa","ab",back=.true.)'),
    ("scan_back_absent", 'print *, scan("xyz","ab",back=.true.)'),
    ("scan_back_var", "print *, scan(\"aabbc\",\"ab\",bt)"),
    ("verify_first", 'print *, verify("aabc","ab")'),
    ("verify_back", 'print *, verify("aabcc","abc",back=.true.)'),
    # discriminating verify rows: the old verify_back agreed by coincidence
    # (every char in the set -> both directions answer 0). These differ.
    ("verify_back_real", 'print *, verify("aabcc","ab",back=.true.)'),
    ("verify_back_var", "print *, verify(\"aabcc\",\"ab\",bt)"),
    # honest refusals kept by name (part of the #764 fix)
    ("index_kind8_refused", 'print *, index(s,t,kind=8)'),
    ("index_bad_keyword_refused", 'print *, index(s,t,backet=.true.)'),
    ("scan_kind8_refused", 'print *, scan("abc","ab",kind=8)'),
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
        frc_pre, ferr = run([str(FFC), str(src), "-o", str(WORK / f"{name}_f")])
        if grc != 0:
            runs += 1
            if name in KNOWN_REFUSED:
                if frc_pre != 0:
                    refused += 1
                    lines.append(f"{name}\tKNOWN_REFUSED\tboth_refuse")
                else:
                    lines.append(f"{name}\tREGRESSION\tffc_accepts_invalid_ref_refuses")
                continue
            lines.append(f"{name}\tBAD_CASE\tgfortran_refuses")
            continue
        frc, ferr = frc_pre, ferr
        if frc != 0:
            runs += 1
            if name in KNOWN_REFUSED:
                refused += 1
                lines.append(f"{name}\tKNOWN_REFUSED\trefused_by_name")
                continue
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
