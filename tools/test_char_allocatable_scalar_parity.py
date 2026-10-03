#!/usr/bin/env python3
"""Byte-exact guard: fixed-length allocatable CHARACTER scalars.

Pins the #348 migration slice: `character(len=N), allocatable :: s`
now takes the canonical deferred character descriptor with the declared
width recorded, so allocate(character(len=N) :: s), assignment, print,
trim, len, concat, comparison, deallocate/reallocate all behave like
gfortran. Refusals pinned: type-spec length mismatch and allocatable
character arrays (still separate bucket work).

Run:  python3 tools/test_char_allocatable_scalar_parity.py
Exit: 0 when every run row is byte-exact vs gfortran.
"""
from __future__ import annotations
import hashlib, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FFC = ROOT / "build" / "fo" / "bin" / "ffc"
WORK = Path("/var/tmp/ffc-goal/perf/chscalar")
ENV = dict(os.environ)
ENV["LD_LIBRARY_PATH"] = str(ROOT/"build"/"fo"/"lib") + os.pathsep + ENV.get("LD_LIBRARY_PATH","")

def P(b): return f"program p\n  implicit none\n  {b}\nend program p\n"

CASES = [
 ("alloc_assign_print", P("character(len=3), allocatable :: s\n  allocate(character(len=3) :: s)\n  s = 'abc'\n  print '(a)', s\n")),
 ("trim_short",         P("character(len=5), allocatable :: s\n  allocate(character(len=5) :: s)\n  s = 'ab'\n  print '(a)', trim(s)\n")),
 ("len_after_alloc",   P("character(len=4), allocatable :: s\n  allocate(character(len=4) :: s)\n  print '(i0)', len(s)\n")),
 ("compare_eq",        P("character(len=2), allocatable :: s\n  allocate(character(len=2) :: s)\n  s = 'ab'\n  if (s == 'ab') print '(a)', 'EQ'\n")),
 ("compare_pad",       P("character(len=4), allocatable :: s\n  allocate(character(len=4) :: s)\n  s = 'ab'\n  if (s == 'ab  ') print '(a)', 'PAD'\n")),
 ("concat",            P("character(len=2), allocatable :: a, b\n  allocate(character(len=2) :: a)\n  allocate(character(len=2) :: b)\n  a = 'ab'\n  b = 'cd'\n  print '(a)', a//b\n")),
 ("index_fn",          P("character(len=5), allocatable :: s\n  allocate(character(len=5) :: s)\n  s = 'abcde'\n  print '(i0)', index(s, 'cd')\n")),
 ("reassign_shorter",  P("character(len=4), allocatable :: s\n  allocate(character(len=4) :: s)\n  s = 'abcd'\n  s = 'xy'\n  print '(a)', trim(s)\n")),
 ("write_two",         P("character(len=2), allocatable :: s\n  integer :: i\n  allocate(character(len=2) :: s)\n  s = 'q'\n  do i = 1, 2\n    write(*,'(a)') trim(s)\n  end do\n")),
 ("len_eq_cond",     P("character(len=3), allocatable :: s\n  allocate(character(len=3) :: s)\n  s = 'abc'\n  if (len(trim(s)) == 3) print '(i0)', 1\n")),
 ("move_alloc",      P("character(len=3), allocatable :: a, b\n  allocate(character(len=3) :: a)\n  a = 'xyz'\n  call take(a)\ncontains\n  subroutine take(t)\n    character(len=3), intent(in) :: t\n    print '(a)', t\n  end subroutine\n")),
 ("print_star",      P("character(len=2), allocatable :: s\n  allocate(character(len=2) :: s)\n  s = 'ok'\n  print *, s\n")),
 ("cmp_ne",          P("character(len=2), allocatable :: s\n  allocate(character(len=2) :: s)\n  s = 'ab'\n  if (s /= 'ba') print '(a)', 'NE'\n")),
 ("three_scalars",   P("character(len=2), allocatable :: a, b, c\n  allocate(character(len=2) :: a)\n  allocate(character(len=2) :: b)\n  allocate(character(len=2) :: c)\n  a = '1'\n  b = '2'\n  c = a//b\n  print '(a)', c\n")),
 ("realloc_same_len",P("character(len=2), allocatable :: s\n  allocate(character(len=2) :: s)\n  s = 'ab'\n  deallocate(s)\n  allocate(character(len=2) :: s)\n  s = 'cd'\n  print '(a)', s\n")),
 ("concat_len",      P("character(len=2), allocatable :: a\n  allocate(character(len=2) :: a)\n  a = 'ab'\n  print '(i0)', len(a//'cd')\n")),
 ("trim_empty",      P("character(len=3), allocatable :: s\n  allocate(character(len=3) :: s)\n  s = ''\n  print '(a)', '['//trim(s)//']'\n")),
 ("len_ge",            P("character(len=3), allocatable :: s\n  allocate(character(len=3) :: s)\n  s = 'ab'\n  if (len(s) >= 3) print '(i0)', len(trim(s))\n")),
 ("realloc_after_reassign", P("character(len=2), allocatable :: s\n  allocate(character(len=2) :: s)\n  s = 'ab'\n  s = 'cd'\n  deallocate(s)\n  allocate(character(len=2) :: s)\n  s = 'ef'\n  print '(a)', trim(s)//'z'\n")),
 ("write_item_pad",  P("character(len=4), allocatable :: s\n  allocate(character(len=4) :: s)\n  s = 'ab'\n  write(*,'(a,a)') '<', trim(s)//'>'\n")),
 ("loop_alloc_free", P("character(len=2), allocatable :: s\n  integer :: i\n  do i = 1, 3\n    allocate(character(len=2) :: s)\n    s = 'a'\n    print '(i1,a)', i, trim(s)\n    deallocate(s)\n  end do\n")),
]

REFUSE_CASES = [
 ("refuse_substring_fmt",    P("character(len=4), allocatable :: s\n  allocate(character(len=4) :: s)\n  s = 'abcd'\n  print '(a)', s(2:3)\n")),
 ("refuse_allocatable_func", P("character(len=3) :: r\n  r = mk()\n  print '(a)', r\ncontains\n  function mk()\n    character(len=3), allocatable :: mk\n    allocate(character(len=3) :: mk)\n    mk = 'fn'\n  end function\n")),
 ("refuse_len_mismatch", P("character(len=2), allocatable :: s\n  allocate(character(len=3) :: s)\n  s = 'abc'\n  print '(a)', s\n")),
 ("refuse_char_array",   P("character(len=2), allocatable :: a(:)\n  allocate(character(len=2) :: a(2))\n  a = 'ab'\n  print '(a)', a(1)\n")),
]

def build_run(is_ffc, src, exe):
    argv=[str(FFC),str(src),"-o",str(exe)] if is_ffc else ["gfortran","-w",str(src),"-o",str(exe)]
    out=subprocess.run(argv,capture_output=True,text=True,env=ENV)
    if out.returncode!=0: return None,(out.stderr or out.stdout).strip().splitlines()[0] if (out.stderr or out.stdout).strip() else "compile-fail"
    r=subprocess.run([str(exe)],capture_output=True,text=True,timeout=30,env=ENV)
    return hashlib.md5(r.stdout.encode()).hexdigest(), r.stdout

def main()->int:
    WORK.mkdir(parents=True,exist_ok=True)
    rows=[]; mismatch=refused=okref=0
    for name,body in CASES:
        src=WORK/f"{name}.f90"; src.write_text(body)
        g,go=build_run(False,src,WORK/f"{name}.gf")
        if g is None: rows.append((name,"REF_FAIL","",go)); continue
        f,fo=build_run(True,src,WORK/f"{name}.fc")
        if f is None: rows.append((name,"REFUSED",g,fo)); refused+=1; continue
        ok=f==g
        if not ok: mismatch+=1
        rows.append((name,"MATCH" if ok else "MISMATCH",g,f))
    for name,body in REFUSE_CASES:
        src=WORK/f"{name}.f90"; src.write_text(body)
        out=subprocess.run([str(FFC),str(src),"-o",str(WORK/f"{name}.fc")],capture_output=True,text=True,env=ENV)
        if out.returncode==0:
            rows.append((name,"SHOULD_REFUSE","","compiled"))
            mismatch+=1
        else:
            rows.append((name,"REFUSE_OK","","")); okref+=1
    runs=sum(1 for r in rows if r[1] in("MATCH","MISMATCH","REFUSED","REF_FAIL"))
    print(f"runs={runs-sum(1 for r in rows if r[1]=='REF_FAIL')} match={sum(1 for r in rows if r[1]=='MATCH')} mismatch={mismatch} refused={refused} pinned_refusals={okref} report={WORK/'report.tsv'}")
    with open(WORK/"report.tsv","w") as fh:
        for r in rows: fh.write("\t".join(r)+"\n")
    for r in rows:
        if r[1] not in ("MATCH","REFUSE_OK"): print(f"  {r[1]} {r[0]} ref={r[2]} ffc={r[3][:90]}")
    return 1 if mismatch or refused else 0

if __name__=="__main__": sys.exit(main())
