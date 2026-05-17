# Type Conversion

The active compiler path maps FortFront semantic types to LIRIC types in `ffc`.
FortFront should expose typed AST and semantic queries; it should not know about
LIRIC type handles or executable ABI.

## Current Direct-Session Mapping

| Fortran surface type | LIRIC lowering status |
| --- | --- |
| `integer` | Supported as signed i32 in the MVP path |
| `logical` | Pending explicit representation decision |
| `real` | Pending direct-session lowering |
| `character` | Pending storage and length-passing ABI |
| arrays | Pending descriptor design |
| procedures | Pending call ABI |

## Rules

- Keep type conversion in `ffc`, close to lowering.
- Add explicit tests for each ABI decision.
- Do not add backend-specific type logic to FortFront.
- Do not grow the old MLIR type-conversion code.

## Next Work

The next useful step is a small `ffc` type mapper for scalar FortFront semantic
types used by `session_program_lowering`. It should start with i32 integers,
then add logicals, reals, and characters with documented ABI choices.
