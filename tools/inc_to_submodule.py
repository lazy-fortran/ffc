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
    r"^    (?:(?P<prefix>recursive|pure|elemental) )*"
    r"(?P<rettype>[A-Za-z_][A-Za-z0-9_]* )?"
    r"(?P<kind>subroutine|function)\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
    r"\s*(?P<args>\([^)]*\)?)?"
    r"(?P<result>\s+result\s*\(\s*[A-Za-z_][A-Za-z0-9_]*\s*\))?"
)
# `end subroutine foo` is not a header; without this the parser invents a
# procedure per terminator and emits a duplicate interface for it.
END_HEADER = re.compile(r"^\s*end\s+(subroutine|function|procedure)\b", re.I)
PAREN = r"(?:[^()]|\([^()]*\))*"
DECL = re.compile(
    r"^        (?P<spec>(?:type|class)\(" + PAREN + r"\)|"
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


def declarations(body: list[str]) -> dict[str, str]:
    """Map dummy name -> full declaration line, from the body's own text."""
    found: dict[str, str] = {}
    for _, line in join_continuations(body):
        m = DECL.match(line.rstrip())
        if not m:
            continue
        for name in re.findall(r"[A-Za-z_][A-Za-z0-9_]*(?:\s*\([^)]*\))?",
                               m.group("names")):
            name = re.sub(r"\s*\(.*\)$", "", name.strip())
            if name:
                found[name] = line.rstrip()
    return found


def indent_decl(line: str, extra: str = "    ") -> str:
    """Re-indent a body declaration into an interface block."""
    return extra + line.strip()


def migrate(path: pathlib.Path, apply: bool) -> int:
    text = path.read_text()
    lines = text.split("\n")
    stem = path.stem
    if stem.startswith("session_program_lowering_"):
        stem = stem[len("session_program_lowering_"):]

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

    iface: list[str] = ["    interface"]
    for p in procedures:
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
            iface.append(cont + p["args"][-1] + ")")
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
    first = procedures[0]["start"]
    sub.extend(lines[first:procedures[-1]["end"]])
    sub.append("")
    sub.append(f"end submodule {stem}")

    print(f"{path.name}: {len(procedures)} procedures -> submodule {stem}")
    for p in procedures:
        print(f"  {p['kind']} {p['name']}({', '.join(p['args'])})")
    if not apply:
        print("--- interface (dry run) ---")
        print("\n".join(iface))
        return 0

    out = pathlib.Path("src") / f"session_program_lowering_{stem}.f90"
    out.write_text("\n".join(sub) + "\n")
    root_text = ROOT.read_text()
    inc_line = f"    include '{path.name}'\n"
    if root_text.count(inc_line) != 1:
        print(f"{ROOT}: expected exactly one include of {path.name}",
              file=sys.stderr)
        return 4
    root_text = root_text.replace(inc_line, "")
    nl = "\n" + "contains" + "\n"
    if root_text.count(nl) != 1:
        print(f"{ROOT}: cannot locate `contains`", file=sys.stderr)
        return 5
    root_text = root_text.replace(nl, "\n" + "\n".join(iface) + nl)
    ROOT.write_text(root_text)
    print(f"wrote src/session_program_lowering_{stem}.f90 and updated {ROOT}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("inc", type=pathlib.Path)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    return migrate(a.inc, a.apply)


if __name__ == "__main__":
    sys.exit(main())
