# Character Descriptor ABI

`character_descriptor_t` is the canonical scalar character descriptor for new
runtime interfaces. Current lowering paths still use the representations in
`RUNTIME_ABI.md`; migrations must preserve this layout and ownership contract.

## Layout

The descriptor is a 32-byte, 8-byte-aligned `bind(C)` record on supported
64-bit targets:

| Offset | Size | Field | Meaning |
|---:|---:|---|---|
| 0 | 8 | `data` | First character byte; null only for zero-capacity or null descriptors |
| 8 | 8 | `length` | Current Fortran character length in bytes |
| 16 | 8 | `capacity` | Bytes available at `data` without reallocation |
| 24 | 4 | `storage_class` | Storage and ownership class |
| 28 | 4 | padding | C layout tail padding |

Only default character kind is supported, so one Fortran character occupies
one byte. `length` and `capacity` are signed 64-bit values with
`0 <= length <= capacity`. Borrowed descriptors have `capacity == length`.

## Storage classes

| Value | Name | Lifetime | Release behavior |
|---:|---|---|---|
| 0 | `CHARACTER_STORAGE_NULL` | No storage | Return a null pointer |
| 1 | `CHARACTER_STORAGE_STATIC` | Static or global owner | Return a null pointer |
| 2 | `CHARACTER_STORAGE_STACK` | Owning stack frame | Return a null pointer |
| 3 | `CHARACTER_STORAGE_OWNED` | Descriptor owns heap storage | Return `data` to the caller |

`release_character_descriptor` resets every field to the null state. It
returns the former data pointer only for `CHARACTER_STORAGE_OWNED`. The caller
passes that pointer to the runtime deallocator exactly once. Borrowed storage
is never freed through this descriptor.

The null state has a null data pointer, zero length, zero capacity, and storage
class zero. Failed initialization also leaves this state.

## Initialization errors

| Code | Name | Condition |
|---:|---|---|
| 0 | `CHARACTER_DESCRIPTOR_OK` | Descriptor initialized |
| 1 | `CHARACTER_DESCRIPTOR_NEGATIVE_LENGTH` | `length < 0` |
| 2 | `CHARACTER_DESCRIPTOR_INVALID_CAPACITY` | `capacity < 0` or `capacity < length` |
| 3 | `CHARACTER_DESCRIPTOR_INVALID_STORAGE` | Borrowed initialization received another class |
| 4 | `CHARACTER_DESCRIPTOR_NULL_DATA` | Positive capacity has a null data pointer |

## Assignment and result lifetime

A borrowed descriptor may be rebound without releasing its former data. An
owned descriptor must release its pointer before replacement unless ownership
is transferred intact to the replacement descriptor.

Character assignment sets `length` to the assigned Fortran length. It may
reuse an owned allocation when the new length does not exceed `capacity`.
Otherwise it allocates new storage, releases the former owned pointer, and
installs class 3.

A function result that escapes its callee uses class 3. Returning the
descriptor transfers ownership to the caller. Static literals and stack
temporaries use classes 1 and 2 and must not escape their valid lifetime.

## Scope

This contract does not change current lowering, character array layout,
nondefault character kinds, coarray storage, or the ABI of existing compiled
procedures.
