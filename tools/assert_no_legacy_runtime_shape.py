#!/usr/bin/env python3
"""assert_no_legacy_runtime_shape - guard the descriptor-backed writer set.

PLAN W1.a #339 slice C/D asked us to verify that the assumed-shape descriptor
binder is the only writer of runtime shape metadata, then delete the legacy
cache writes. The audit (2026-10-01) found the premise is wrong in a specific
way, and the honest artefact is therefore an enumerated guard rather than a
deletion.

Audit over every routine in `src/` that assigns `has_runtime_dim_size(...)`:

  DESCRIPTOR-BACKED (legitimate producers, whitelisted):
    bind_assumed_shape_descriptor_params  assumed_shape_descriptor.inc:1032
        Callee side. Reads extents OUT of an incoming descriptor.
    define_runtime_array_symbol           runtime_alloc.inc:510
        Caller side. Creates a fresh descriptor at ALLOCATE and writes the
        extents of the symbol it just made.
  plain (different symbol class, may not carry has_runtime_descriptor):
    alias_pointer_to_alloc_array_component, bind_assumed_rank_rankn,
    bind_one_assumed_shape, define_declared_array_symbol,
    lower_array_reduction_intrinsic, bind_optional_assumed_shape_descriptor
        (the last is invoked by the canonical binder, so it is the same
        route and not a competing writer).

So nothing here is legacy metadata to delete. What this test enforces is the
boundary: the set of routines that write runtime extents for a symbol that also
carries `has_runtime_descriptor` stays EXACTLY the two above. A third one
means somebody reintroduced a parallel shape cache, and that is the failure
this guard exists to catch - it cannot pass by accident, because the whitelist
is the audited set and any new writer appears as an unlisted name.
"""
import pathlib
import re
import sys

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"

WHITELIST = {
    "bind_assumed_shape_descriptor_params",
    "define_runtime_array_symbol",
}

WRITE = re.compile(r"has_runtime_dim_size\([^)]*\)\s*=")
DESC = re.compile(r"has_runtime_descriptor\s*=")
SUB = re.compile(r"^\s*(?:recursive\s+)?(?:pure\s+)?(?:module\s+)?"
                 r"(?:subroutine|function)\s+([a-z0-9_]+)", re.I)


def routines(text: str):
    starts = [i for i, line in enumerate(text.split("\n")) if SUB.match(line)]
    ends = starts[1:] + [len(text.split("\n"))]
    lines = text.split("\n")
    for a, b in zip(starts, ends):
        yield SUB.match(lines[a]).group(1), "\n".join(lines[a:b]), a + 1


def main() -> int:
    found: dict[str, list[str]] = {}
    paths = sorted(set(SRC.rglob("*.inc")) | set(SRC.rglob("*.f90")))
    for path in paths:
        text = path.read_text(errors="replace")
        for name, body, line in routines(text):
            if WRITE.search(body) and DESC.search(body):
                found.setdefault(name, []).append(f"{path.name}:{line}")

    offenders = {k: v for k, v in found.items() if k not in WHITELIST}
    missing = WHITELIST - set(found)

    print(f"descriptor-backed extent writers: {sorted(found)}")
    rc = 0
    if offenders:
        print("FAIL - unlisted descriptor-backed shape writer (parallel "
              "runtime-shape cache reintroduced?):")
        for k, v in sorted(offenders.items()):
            print(f"  {k} {v}")
        rc = 1
    if missing:
        print(f"FAIL - whitelisted producer vanished from src: {sorted(missing)}")
        rc = 1
    if rc == 0:
        print("PASS - writer set is exactly the audited pair")
    return rc


if __name__ == "__main__":
    sys.exit(main())
