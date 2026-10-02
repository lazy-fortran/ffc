#!/usr/bin/env python3
"""Turn one `.inc` text include of the lowering root into a real submodule.

The lowering root `src/session_program_lowering.f90` pastes hundreds of
procedures in with `include 'foo.inc'`. Those procedures have no independent
existence: nothing states their interface, and a mistake in a dummy's type is
only visible after text substitution into a 7k-line file. This tool performs
the mechanical half of the migration - extract the procedures, state their
interfaces from their own dummy declarations, and emit

  * `src/<stem>.f90`          submodule (session_program_lowering_impl) <stem>
  * an `interface` block      to insert into the parent before `contains`

and rewrite the parent to drop the `include`. What it deliberately does not do
is invent anything: a dummy whose declaration it cannot find inside the body is
a hard error, so the operator writes it rather than the tool guessing.

Usage:
    python3 tools/inc_to_submodule.py src/foo.inc [--apply]
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path("src/session_program_lowering.f90")
MODULE = "session_program_lowering_impl"

# A procedure header at the canonical four-space indent, with the optional
# prefixes Fortran allows before the keyword.
HEADER = re.compile(
    r"^[ \t]{0,8}(?:(?P<prefix>recursive|pure|elemental) )*"
    r"(?P<rettype>(?:(?!(?:subroutine|function)\b)"
    r"[A-Za-z_][A-Za-z0-9_]*(?:\([^)\n]*\))?\s+)*)"
    r"(?P<kind>subroutine|function)\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
    r"\s*(?P<args>\([^)]*\)?)?"
    r"(?P<result>\s+result\s*\(\s*[A-Za-z_][A-Za-z0-9_]*\s*\))?"
)
# `end subroutine foo` is not a header; without this the parser invents a
# procedure per terminator and emits a duplicate interface for it.
END_HEADER = re.compile(r"^\s*end\s+(subroutine|function|procedure)\b", re.I)
PAREN = r"(?:[^()]|\([^()]*\))*"
DECL = re.compile(
    r"^[ \t]{0,8}(?P<spec>(?:type|class)\(" + PAREN + r"\)|"
    r"(?:integer|logical|real|complex|character)(?:\(" + PAREN + r"\))?)"
    r"(?P<attrs>[^:]*)"
    r"::\s*(?P<names>[A-Za-z_][A-Za-z0-9_]*(?:\s*\(" + PAREN + r"\))?"
    r"(?:\s*,\s*[A-Za-z_][A-Za-z0-9_]*(?:\s*\(" + PAREN + r"\))?)*)\s*$"
)


def join_continuations(lines: list[str]) -> list[tuple[int, str]]:
    """Return (start_index, logical_line) with `&` continuations folded."""
    out: list[tuple[int, str]] = []
    i = 0
    while i < len(lines):
        text = lines[i]
        start = i
        while text.rstrip().endswith("&") and i + 1 < len(lines):
            i += 1
            tail = lines[i].strip()
            text = text.rstrip()[:-1] + " " + tail
        out.append((start, text))
        i += 1
    return out


def parse_args(raw: str) -> list[str]:
    raw = raw.strip()
    if raw.startswith("("):
        raw = raw[1:]
    if raw.endswith(")"):
        raw = raw[:-1]
    out: list[str] = []
    depth = 0
    cur = ""
    for ch in raw:  # split on top-level commas only
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return [re.sub(r"\s*\(.*\)$", "", a) for a in out]


DECLSECTION = re.compile(
    r"^\s*(?:use\b|implicit\b|private\b|public\b|integer\b|real\b|"
    r"double\b|complex\b|character\b|logical\b|type\s*\(|type\s+|"
    r"class\s*\(|procedure\b|dimension\b|pointer\b|allocatable\b|"
    r"contiguous\b|optional\b|target\b|value\b|parameter\b|external\b|"
    r"intent\b|save\b|enum\b|common\b|equivalence\b|bind\s*\()",
    re.I)


def declarations(body: list[str]) -> dict[str, str]:
    """Map dummy name -> declaration line from the body's DECLARATION
    SECTION only. Scanning executables lets an internal procedure's dummy
    (`type(x), intent(in) :: context`) overwrite the outer procedure's
    (`intent(inout) :: context`), producing a corrupt interface body."""
    found: dict[str, str] = {}
    skipped_header = False
    for _, line in join_continuations(body):
        stripped = line.strip()
        if not stripped or stripped.startswith("!"):
            continue
        if not skipped_header:
            skipped_header = True  # the first logical line is the header
            continue
        if not DECLSECTION.match(line):
            break  # first executable statement: declaration section is over
        m = DECL.match(line.rstrip())
        if not m:
            continue
        for name in re.findall(r"[A-Za-z_][A-Za-z0-9_]*(?:\s*\([^)]*\))?",
                               m.group("names")):
            name = re.sub(r"\s*\(.*\)$", "", name.strip())
            if name and name not in found:
                found[name] = line.rstrip()
    return found


def indent_decl(line: str, extra: str = "    ") -> str:
    """Re-indent a body declaration into an interface block."""
    return extra + line.strip()


def migrate(path: pathlib.Path, apply: bool) -> int:
    text = path.read_text()
    lines = text.split("\n")
    stem = path.stem
    # A submodule whose name is a bare Fortran statement keyword (inquire,
    # open, read, ...) breaks the build: the submodule ordering/resolution
    # misparses `submodule (M) inquire` and the parent's .smod is never
    # generated before the submodule compiles (observed on inquire.inc -
    # clean build failed with "session_program_lowering_impl.smod has not
    # been generated"). Keep the qualified name for such stems.
    KEYWORD_STEMS = {
        "inquire", "open", "close", "read", "write", "print", "rewind",
        "backspace", "endfile", "allocate", "deallocate", "associate",
        "select", "block", "critical", "sync", "call", "do", "if", "then",
        "else", "elseif", "endif", "where", "elsewhere", "endwhere",
        "cycle", "exit", "return", "stop", "goto", "assign", "format",
        "namelist", "equivalence", "data", "common", "parameter", "save",
        "use", "include", "interface", "procedure", "module", "function",
        "subroutine", "program", "contains", "public", "private",
    }
    if stem.startswith("session_program_lowering_"):
        short = stem[len("session_program_lowering_"):]
        if short not in KEYWORD_STEMS:
            stem = short

    procedures: list[dict] = []
    heads = [(i, HEADER.match(l)) for i, l in enumerate(lines)]
    heads = [(i, m) for i, m in heads
             if m and not END_HEADER.match(lines[i])]
    if not heads:
        print(f"{path}: no procedures found", file=sys.stderr)
        return 2

    for n, (start, m) in enumerate(heads):
        end_next = heads[n + 1][0] if n + 1 < len(heads) else len(lines)
        # Fold the header itself if it continues.
        header_text = lines[start]
        j = start
        while header_text.rstrip().endswith("&") and j + 1 < len(lines):
            j += 1
            header_text = header_text.rstrip()[:-1] + " " + lines[j].strip()
        # Re-match with continuations folded (args may span lines).
        hm = HEADER.match(header_text)
        if not hm:
            print(f"{path}:{start+1}: unparsed header: {header_text.strip()}",
                  file=sys.stderr)
            return 2
        body = lines[start:end_next]
        decls = declarations(body)
        args = parse_args(hm.group("args") or "")
        missing = [a for a in args if a not in decls]
        if missing:
            print(f"{path}:{start+1}: {hm.group('name')}: no declaration found "
                  f"for dummies {missing} - state them by hand", file=sys.stderr)
            return 3
        procedures.append({
            "name": hm.group("name"),
            "kind": hm.group("kind"),
            "prefix": hm.group("prefix"),
            "rettype": (hm.group("rettype") or "").strip(),
            "args": args,
            "decls": decls,
            "result": hm.group("result"),
            "start": start,
            "end": end_next,
        })

    # The `include` directive does not necessarily live in the root module:
    # the hub `top.inc` includes `submodules.inc`, and `functions.inc`
    # includes `functions_tail.inc`. Find where it actually is; the interface
    # still goes into the root module's declaration section, because that is
    # the scope the text lands in either way.
    includer = None
    for cand in ([ROOT] + sorted(pathlib.Path("src").glob("*.inc"))
                 + sorted(f for f in pathlib.Path("src").glob("*.f90")
                         if f != ROOT)):
        if cand.exists() and f"include '{path.name}'" in cand.read_text():
            includer = cand
            break
    if includer is None:
        print(f"{path}: no `include` directive found in src - dead text, "
              "delete it instead of migrating it", file=sys.stderr)
        return 10

    # Two failure modes a text include makes invisible until the build fails
    # deep inside a 7k-line includer. Refuse both here, at the file that has
    # the problem, with the reason.
    for i, l in enumerate(lines):
        m = re.search(r"include\s+'([^']+)'", l, re.I)
        if m and m.group(1) != path.name:
            print(f"{path}: includes {m.group(1)} - migrate that child first, "
                  "or the captured region duplicates its procedures",
                  file=sys.stderr)
            return 6
    fname = stem if stem.startswith("session_program_lowering_") \
        else f"session_program_lowering_{stem}"
    out = pathlib.Path("src") / f"{fname}.f90"
    for p_ in procedures:
        for other in sorted(pathlib.Path("src").glob("*.f90")):
            # Never collide with this tool's own destination: a stray left by
            # an aborted run would otherwise look like a duplicate definition
            # in a sibling include.
            if other == path or other.name == ROOT.name or other == out:
                continue
            if re.search(rf"^    (?:recursive |pure |elemental |)*"
                         rf"(?:[A-Za-z_][A-Za-z0-9_]* )?"
                         rf"(?:subroutine|function)\s+{p_['name']}\b",
                         other.read_text(), re.M):
                print(f"{path}: {p_['name']} is also defined in {other.name} "
                      f"(duplicate sibling include) - split them apart first",
                    file=sys.stderr)
                return 7

    # Some includes do not contain whole procedures: they carry a *fragment*
    # of a procedure whose header lives in the includer or in an earlier
    # include, so the body uses locals (a `logical(c_bool) :: c_false_local`)
    # that no captured region can declare. Procedure headers and terminators
    # are balanced inside a real unit; if they are not, this file is a
    # fragment and the mechanical move cannot be correct.
    # A file whose captured region starts with executable code is the tail of
    # a procedure whose header sits in an earlier include - `save.inc` uses
    # `c_false_local`, a local declared inside a procedure of `common.inc`.
    # Comments and blanks before the first header are fine; statements are not.
    lead = lines[:procedures[0]["start"]]
    frag = [l for l in lead
            if l.strip() and not l.strip().startswith("!")]
    if frag:
        print(f"{path}: {len(lead) - len(frag)} executable statement(s) before "
              "the first procedure header - this include continues a procedure "
              "that started in another file, so it cannot move as a unit",
              file=sys.stderr)
        return 9

    region = "\n".join(lines[procedures[0]["start"]:procedures[-1]["end"]])
    # `end subroutine foo` must not count as a header: `end` fits the
    # optional return-type slot, which would report a false imbalance.
    n_head = len(re.findall(r"^[ \t]*(?!end\b)(?:(?:recursive|pure|elemental) )"
                            r"*(?:[A-Za-z_][A-Za-z0-9_]*(?:\([^)\n]*\))? )*"
                            r"(?:subroutine|function)\s+[A-Za-z_]", region, re.M))
    n_end = len(re.findall(r"^[ \t]*end\s+(?:subroutine|function)\b", region, re.M))
    if n_head != n_end:
        print(f"{path}: {n_head} procedure headers but {n_end} terminators in "
              "the captured region - this include holds a fragment of a "
              "procedure that starts outside it, so it cannot move as a unit",
              file=sys.stderr)
        return 8

    # Bodies are emitted as one contiguous slice of the original file, so the
    # slice bounds are taken before any reordering. The interface block, in
    # contrast, is order-sensitive: inside an `interface` body a callee's
    # interface must precede its caller's, or gfortran reports the symbol as
    # having no implicit type. That is not a hypothetical - `character_tail.inc`
    # calls `emit_or_add1` 44 lines before defining it, legal today through
    # host association and broken the moment it becomes a submodule.
    body_first = min(p["start"] for p in procedures)
    body_last = max(p["end"] for p in procedures)
    iface_procs = order_by_dependency(procedures, lines)
    if iface_procs is None:
        # Cycles are fine: every folded procedure gets a parent-module
        # interface, and submodule bodies see all parent interfaces
        # regardless of order. Keep source order and mark callees recursive.
        iface_procs = list(procedures)
        print(f"{path}: cyclic call graph - keeping source order "
              "(parent interfaces make bodies order-independent)",
              file=sys.stderr)

    iface: list[str] = ["    interface"]
    for p in iface_procs:
        pre = f"{p['prefix']} " if p["prefix"] else ""
        ret = f"{p['rettype']} " if p["rettype"] else ""
        args_txt = ", ".join(p["args"])
        res = ""
        if p.get("result"):
            # A function that names its result must keep that name in the
            # interface too: `function f(x) result(r)` and `function f(x)`
            # are different bindings, and dropping the clause is a mismatch
            # gfortran reports against the body.
            res = " result(" + re.search(r"\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)",
                                          p["result"]).group(1) + ")"
        head = f"        {pre}{ret}module {p['kind']} {p['name']}({args_txt}){res}"
        if len(head) > 90:
            prefix_part = f"        {pre}{ret}module {p['kind']} {p['name']}("
            cont = " " * len(prefix_part)
            iface.append(prefix_part + p["args"][0] + ", &")
            for a in p["args"][1:-1]:
                iface.append(cont + a + ", &")
            iface.append(cont + p["args"][-1] + ")" + res)
        else:
            iface.append(head)
        seen_decl: set[str] = set()
        for a in p["args"]:
            d = indent_decl(p["decls"][a], " " * 12)
            # One declaration line may name several dummies (`src_ptr,
            # dest_ptr`); emitting it once per dummy would duplicate it.
            if d in seen_decl:
                continue
            seen_decl.add(d)
            iface.append(d)
        if p.get("result"):
            # A function that names its result takes its return type from a
            # declaration of that name in the body, and that declaration must
            # come after the dummies: `character(len=len(s)) :: t` references
            # `s`, so hoisting it above `s`'s own declaration is an error.
            rname = re.search(r"\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)",
                              p["result"]).group(1)
            if rname in p["decls"]:
                d = indent_decl(p["decls"][rname], " " * 12)
                if d not in seen_decl:
                    iface.append(d)
        elif p["kind"] == "function" and p["rettype"] == "":
            # No `result(...)` clause: the function's type comes from a
            # declaration of its own name.
            if p["name"] in p["decls"]:
                d = indent_decl(p["decls"][p["name"]], " " * 12)
                if d not in seen_decl:
                    iface.append(d)
        iface.append(f"        end {p['kind']} {p['name']}")
    iface.append("    end interface")

    sub = [
        f"submodule ({MODULE}) {stem}",
        f"    !! `{stem}` procedures, moved out of `{path.name}` so that this",
        "    !! unit has a name, a checked interface, and can be compiled, edited",
        "    !! and pointed at on its own instead of only inside its includer.",
        "    implicit none",
        "",
        "contains",
        "",
    ]
    sub.extend(lines[body_first:body_last])
    sub.append("")
    sub.append(f"end submodule {stem}")

    print(f"{path.name}: {len(procedures)} procedures -> submodule {stem}")
    for p in procedures:
        print(f"  {p['kind']} {p['name']}({', '.join(p['args'])})")
    if not apply:
        print("--- interface (dry run) ---")
        print("\n".join(iface))
        return 0

    out.write_text("\n".join(sub) + "\n")
    inc_text = includer.read_text()
    n = len(re.findall(rf"^\s*include\s+'{re.escape(path.name)}'\s*$",
                       inc_text, re.M))
    if n != 1:
        print(f"{includer}: expected exactly one include of {path.name}, "
              f"found {n}", file=sys.stderr)
        return 4
    def drop_include(text):
        # Line filter, not re.sub: CPython 3.14.7 re.sub matched zero
        # replacements here while re.search found the line (reproducible in
        # memory); the filter is immune and equally exact.
        want = f"include '{path.name}'"
        return "".join(l for l in text.splitlines(keepends=True)
                       if l.strip() != want)
    if includer != ROOT:
        includer.write_text(drop_include(inc_text))
        root_text = drop_include(ROOT.read_text())
    else:
        root_text = drop_include(inc_text)
    nl = "\n" + "contains" + "\n"
    if root_text.count(nl) != 1:
        print(f"{ROOT}: cannot locate `contains`", file=sys.stderr)
        return 5
    root_text = root_text.replace(nl, "\n" + "\n".join(iface) + nl)
    ROOT.write_text(root_text)
    print(f"wrote src/session_program_lowering_{stem}.f90 and updated {ROOT}")
    return 0


def order_by_dependency(procedures: list[dict], lines: list[str]):
    """Sort procedure headers so a callee precedes its caller.

    Returns None when the set is cyclic, which the caller turns into a refusal:
    a cycle means the interface block cannot satisfy gfortran by ordering alone
    and the fix is explicit interfaces, not a different permutation.
    """
    names = {p["name"] for p in procedures}
    deps: dict[str, set[str]] = {}
    for p in procedures:
        body = "\n".join(lines[p["start"]:p["end"]])
        deps[p["name"]] = {o for o in names if o != p["name"]
                           and re.search(r"\b" + re.escape(o) + r"\b", body)}
    ordered: list[dict] = []
    done: set[str] = set()
    rest = list(procedures)
    while rest:
        ready = [p for p in rest if deps[p["name"]] <= done]
        if not ready:
            return None
        # Stable within a level so the emitted block still reads in file order.
        ordered.extend(ready)
        done.update(p["name"] for p in ready)
        rest = [p for p in rest if p["name"] not in done]
    return ordered


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("inc", type=pathlib.Path)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    return migrate(a.inc, a.apply)


if __name__ == "__main__":
    sys.exit(main())
