# Migration Guide: Legacy Backend to LIRIC

The project direction is LIRIC, not the old MLIR/HLFIR experiment.

## Target State

```text
FortFront compiler API
        |
        v
ffc direct lowering
        |
        v
LIRIC session C API
        |
        v
native executable
```

## What To Move

Move any useful scalar lowering behavior from the bootstrap reference path into
`session_program_lowering` and its included control/loop files.

Use these modules:

- `liric_session_bindings` for session and instruction emission.
- `liric_session_control_bindings` for blocks and branches.
- `liric_session_io_bindings` for the direct-session `printf` ABI used by
  minimal integer `print`.
- `session_lowering_ops` for opcode and predicate mapping.

## What To Leave Alone

The old MLIR/HLFIR source tree is not the active backend. Do not expand it.
Reference it only when extracting behavior that should be reimplemented against
the LIRIC session API.

The bootstrap text-IR path is temporary. It can stay until the direct session
path covers the same executable subset.

## Migration Checklist

- Add the feature to the direct session lowerer.
- Add or update a `test_session_*` executable test.
- Keep CLI behavior on the direct session path.
- Document ABI/runtime decisions before broadening the language surface.
- If LIRIC lacks a required primitive, file the issue in LIRIC and keep the
  `ffc` issue blocked on that upstream item.
