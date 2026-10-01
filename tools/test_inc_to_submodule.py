#!/usr/bin/env python3
"""Oracle for the interface-ordering rule in inc_to_submodule.

A submodule sees its parent's interface bodies in declaration order, so a
callee's interface must precede its caller's. File order does not guarantee
that - `character_tail.inc` calls `emit_or_add1` 44 lines before defining it,
which host association tolerates inside the includer and gfortran rejects the
moment the include becomes a named submodule.
"""
import importlib.util, pathlib, sys

spec = importlib.util.spec_from_file_location(
    "inc_to_submodule", pathlib.Path(__file__).parent / "inc_to_submodule.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

fails = 0


def check(label, got, want):
    global fails
    ok = got == want
    if not ok:
        fails += 1
    print(f"{'PASS' if ok else 'FAIL'} {label}: got={got} want={want}")


lines = ["subroutine caller(x)", "  call emit_or_add1(x)", "end subroutine caller",
         "subroutine emit_or_add1(y)", "  y = 1", "end subroutine emit_or_add1"]
procs = [{"name": "caller", "start": 0, "end": 3},
         {"name": "emit_or_add1", "start": 3, "end": 6}]
got = m.order_by_dependency(procs, lines)
check("callee precedes caller despite file order",
      got and [p["name"] for p in got], ["emit_or_add1", "caller"])

lines2 = ["subroutine a(x)", "  call b(x)", "end subroutine a",
          "subroutine b(y)", "  call a(y)", "end subroutine b"]
check("cycle refuses instead of emitting an unusable permutation",
      m.order_by_dependency([{"name": "a", "start": 0, "end": 3},
                             {"name": "b", "start": 3, "end": 6}], lines2), None)

indep = ["subroutine a()", "end subroutine a", "subroutine b()", "end subroutine b"]
check("independent procedures keep file order",
      [p["name"] for p in m.order_by_dependency(
          [{"name": "a", "start": 0, "end": 2},
           {"name": "b", "start": 2, "end": 4}], indep)], ["a", "b"])

print(f"{'OK' if not fails else f'{fails} FAILED'}")
sys.exit(1 if fails else 0)
