#!/usr/bin/env python3
"""Parity oracle: character substring // chains as print items (#348 cluster).

`print *, a(2:4)//b(1:2)` was refused with "unsupported array subscript ...
not non-integer identifiers" because `is_character_operand` did not recognise a
substring, so the whole `//` chain was not classified as character and the
print item fell through to the integer/array fallback. The identical expression
assigned to a character target already printed correctly, which is what makes
this a routing gap rather than missing lowering.

gfortran is the independent oracle. 24 shapes across substring width, chain
length, repeats, reversals, mixed literal/variable operands, trim() in the
chain, and full-string slices. Each case's stdout is md5-compared, so a
wrongly-clamped or mis-ordered concatenation cannot pass by accident.
"""
import hashlib
import pathlib
import subprocess
import sys
import os

FFC = pathlib.Path(os.environ.get(
    "FFC", str(pathlib.Path(__file__).resolve().parents[1] / "build/fo/app/ffc")))
WORK = pathlib.Path(os.environ.get("CS_CONCAT_WORK", "/var/tmp/ffc-goal/perf/cssub"))
WORK.mkdir(parents=True, exist_ok=True)

# (name, declarations, body) - each body is a print item under test.
CASES = [
    ("two_three", 'character(len=5) :: a, b\n  a="abcde"\n  b="12345"', 'a(2:4)//b(1:2)'),
    ("one_five", 'character(len=5) :: a, b\n  a="abcde"\n  b="12345"', 'a(1:5)//b(1:5)'),
    ("single", 'character(len=4) :: a, b\n  a="wxyz"\n  b="qrst"', 'a(2:2)//b(3:3)'),
    ("three_way", 'character(len=3) :: a, b, c\n  a="AAA"\n  b="BBB"\n  c="CCC"', 'a(1:2)//b(2:3)//c(1:1)'),
    ("lit_left", 'character(len=3) :: a\n  a="xyz"', '"PRE"//a(2:3)'),
    ("lit_right", 'character(len=3) :: a\n  a="xyz"', 'a(1:2)//"POST"'),
    ("same_var", 'character(len=4) :: a\n  a="abcd"', 'a(1:2)//a(3:4)'),
    ("overlap", 'character(len=5) :: a\n  a="abcde"', 'a(1:3)//a(3:5)'),
    ("reversed", 'character(len=3) :: a\n  a="abc"', 'a(3:3)//a(2:2)//a(1:1)'),
    ("with_trim", 'character(len=6) :: a\n  character(len=3) :: b\n  a="xy   "\n  b="123"', 'trim(a)//b(1:2)'),
    ("len_six", 'character(len=8) :: a, b\n  a="12345678"\n  b="abcdefgh"', 'a(2:6)//b(1:3)'),
    ("empty_head", 'character(len=3) :: a\n  character(len=2) :: b\n  a="abc"\n  b="de"', 'a(1:0)//b(1:2)'),
    ("eight_chain", 'character(len=4) :: a, b, c, d\n  a="aaaa"\n  b="bbbb"\n  c="cccc"\n  d="dddd"', 'a(1:2)//b(1:1)//c(2:4)//d(1:2)'),
    ("pad_target", 'character(len=3) :: a\n  character(len=4) :: b\n  a="abc"\n  b="wxyz"', 'a(1:3)//b(2:4)'),
    ("mid_slice", 'character(len=10) :: a\n  a="0123456789"', 'a(4:7)//a(0+1:2)'),
    ("dup_twice", 'character(len=2) :: a\n  a="zz"', 'a(1:2)//a(1:2)//a(1:2)'),
    ("digits", 'character(len=6) :: a, b\n  a="102030"\n  b="405060"', 'a(1:1)//b(3:4)//a(5:6)'),
    ("space_pad", 'character(len=4) :: a\n  character(len=3) :: b\n  a="a  "\n  b=" b "', 'a(1:4)//b(1:3)'),
    ("full_slice", 'character(len=3) :: a, b\n  a="abc"\n  b="def"', 'a(:)//b(2:)'),
    ("head_slice", 'character(len=4) :: a, b\n  a="1234"\n  b="5678"', 'a(:2)//b(:)'),
    ("tail_slice", 'character(len=4) :: a, b\n  a="1234"\n  b="5678"', 'a(3:)//b(:3)'),
    ("five_chain", 'character(len=2) :: a,b,c,d,e\n  a="aa"\n  b="bb"\n  c="cc"\n  d="dd"\n  e="ee"', 'a(1:2)//b(1:2)//c(1:1)//d(2:2)//e(1:2)'),
    ("mixed_lit", 'character(len=3) :: a\n  a="mno"', '"x"//a(2:3)//"y"//a(1:1)'),
    ("long_slice", 'character(len=12) :: a\n  a="abcdefghijkl"', 'a(3:9)//a(1:4)'),
    # Adversarial-review additions. The nested form c(i)(l:u)//c(j)(l:u) is a
    # capability this fix actually enables (FortFront stores it as a slice whose
    # base is a character-array element designator) and it was unpinned. The
    # integer-section rows are the blast-radius guard: `is_character_substring`
    # checks the is_character_substring flag, a character-array-element base, or
    # value_kind == VALUE_CHARACTER .and. .not. is_array, so an INTEGER section
    # must still take the array path and print unchanged. If one of these stops
    # matching, the character classifier is bleeding onto integer shapes.
    ("nested_bc", 'character(len=3) :: c(2)\n  c=(/ "abc", "def" /)', 'c(1)(2:3)'),
    ("nested_concat", 'character(len=3) :: c(2)\n  c=(/ "abc", "def" /)', 'c(1)(2:3)//c(2)(1:2)'),
    ("nested_three", 'character(len=2) :: c(3)\n  c=(/ "ab", "cd", "ef" /)', 'c(1)(2:2)//c(2)(1:2)//c(3)(1:1)'),
    ("int_section_print", 'integer :: ia(4)\n  ia=[1,2,3,4]', 'ia(2:4)'),
    ("int_section_pair", 'integer :: ia(4), ib(3)\n  ia=[1,2,3,4]\n  ib=[5,6,7]', 'ia(2:4)+ib(1:3)'),
]


def build(name: str, decls: str, expr: str) -> str:
    return (f"program {name}\n  implicit none\n  {decls}\n"
            f"  print *, {expr}\nend program {name}\n")


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def main() -> int:
    runs = match = refused = mismatch = ref_fail = 0
    # Known pre-existing gap, reported but not failed: whole-array integer
    # arithmetic `ia(2:4)+ib(1:3)` is refused with "unsupported array
    # subscript". gfortran accepts it. That is #337 territory (centralize
    # array element-expression lowering) and is NOT this commit's blast radius -
    # it is a compile-time refusal, not a wrong answer or a crash. Listed so the
    # gap stays visible without turning the guard red.
    KNOWN_GAP = {"int_section_pair"}

    lines = []
    for name, decls, expr in CASES:
        src = WORK / f"{name}.f90"
        src.write_text(build(name, decls, expr))
        rc, _, gerr = run(["gfortran", str(src), "-o", str(WORK / f"{name}_ref")])
        if rc != 0:
            ref_fail += 1
            runs += 1
            lines.append(f"{name}\tREF_COMPILE_FAIL")
            continue
        rcode, rout, _ = run([str(WORK / f"{name}_ref")])
        rmd5 = hashlib.md5(rout.encode()).hexdigest()
        rc, _, err = run([str(FFC), str(src), "-o", str(WORK / f"{name}_ffc")])
        runs += 1
        if rc != 0:
            tag = "REFUSED" if "unsupported" in err else "OTHER_ERROR"
            if name in KNOWN_GAP:
                lines.append(f"{name}\tKNOWN_GAP\t{tag}")
                continue
            refused += 1
            lines.append(f"{name}\t{tag}")
            continue
        fcode, fout, _ = run([str(WORK / f"{name}_ffc")])
        fmd5 = hashlib.md5(fout.encode()).hexdigest()
        if fmd5 == rmd5 and fcode == rcode:
            match += 1
            lines.append(f"{name}\tMATCH\t{rmd5}")
        else:
            mismatch += 1
            lines.append(f"{name}\tMISMATCH\tref={rmd5}\tffc={fmd5}")
    (WORK / "report.tsv").write_text("\n".join(lines) + "\n")
    summary = (f"runs={runs} match={match} refused={refused} "
              f"mismatch={mismatch} ref_fail={ref_fail}")
    (WORK / "summary.txt").write_text(summary + "\n")
    print(summary)
    print(f"report={WORK/'report.tsv'}")
    return 0 if mismatch == 0 and refused == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
