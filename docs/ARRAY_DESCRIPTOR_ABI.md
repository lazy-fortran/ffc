# Array Descriptor ABI

`array_descriptor_t` is the canonical array descriptor for new runtime
interfaces. Existing lowering paths still use the ad hoc representations in
`RUNTIME_ABI.md`; each migration issue moves one path onto this layout and
removes its old representation rather than running both.

## Layout

The descriptor is a 200-byte, 8-byte-aligned `bind(C)` record on supported
64-bit targets:

| Offset | Size | Field | Meaning |
|---:|---:|---|---|
| 0 | 8 | `base` | Address of the element whose subscripts are the lower bounds |
| 8 | 8 | `element_size` | Element storage size in bytes |
| 16 | 4 | `element_type` | Element type code |
| 20 | 4 | `rank` | Number of dimensions, 1 to 7 |
| 24 | 4 | `flags` | Allocation, association, ownership, contiguity bits |
| 28 | 4 | `reserved` | Zero; reserved for future ABI use |
| 32 | 168 | `dim(7)` | Per-dimension metadata, seven entries of 24 bytes |

Each `dim` entry is an `array_dimension_t`:

| Offset in entry | Size | Field | Meaning |
|---:|---:|---|---|
| 0 | 8 | `lower_bound` | Fortran lower bound of the dimension |
| 8 | 8 | `extent` | Number of elements in the dimension, `>= 0` |
| 16 | 8 | `stride_bytes` | Signed byte distance between consecutive elements |

Entry `d` is at descriptor offset `32 + 24*(d-1)`. Entries beyond `rank` are
not part of the value; they hold the null-state defaults.

The descriptor is rank-agnostic and element-kind-agnostic. All extents,
bounds, and strides are signed 64-bit values, so a stride may be negative and
`base` need not be the lowest address in the array.

## Addressing

The address of element `(i(1), ..., i(rank))` is

```
address = base + sum over d of (i(d) - lower_bound(d)) * stride_bytes(d)
```

The subscript `i(d)` is valid when
`lower_bound(d) <= i(d) <= lower_bound(d) + extent(d) - 1`. Any other
subscript is an `ARRAY_DESCRIPTOR_INVALID_INDEX` error from the checked
helpers.

Fortran column-major layout is a property of the strides, not of the
addressing rule. A contiguous array has

```
stride_bytes(1) = element_size
stride_bytes(d) = stride_bytes(d-1) * extent(d-1)   for d > 1
```

so the leftmost subscript varies fastest. `set_contiguous_array_descriptor`
computes exactly these strides. `set_strided_array_descriptor` accepts
arbitrary strides for view construction and sets the contiguity flag only when
the given strides match the column-major sequence above.

Bounds are carried, not normalized. A descriptor with `lower_bound = -1` and
`extent = 2` addresses subscripts -1 and 0, and `base` is the address of
element -1. Rebinding to a dummy with different declared bounds changes
`lower_bound` and leaves `base`, `extent`, and `stride_bytes` untouched.

## Flags

| Bit | Value | Name | Meaning |
|---:|---:|---|---|
| 0 | 1 | `ARRAY_FLAG_ALLOCATED` | Descriptor has storage; `allocated` is true |
| 1 | 2 | `ARRAY_FLAG_ASSOCIATED` | Descriptor designates an object; `associated` is true |
| 2 | 4 | `ARRAY_FLAG_OWNS_DATA` | Descriptor owns the `base` allocation |
| 3 | 8 | `ARRAY_FLAG_CONTIGUOUS` | Strides are column-major contiguous |

A null descriptor has `flags == 0`, a null `base`, zero rank, zero element
size, and element type zero. Failed initialization also leaves this state.

## Element type codes

| Value | Name |
|---:|---|
| 0 | `ARRAY_ELEMENT_NONE` |
| 1 | `ARRAY_ELEMENT_INTEGER` |
| 2 | `ARRAY_ELEMENT_REAL` |
| 3 | `ARRAY_ELEMENT_LOGICAL` |
| 4 | `ARRAY_ELEMENT_COMPLEX` |
| 5 | `ARRAY_ELEMENT_CHARACTER` |
| 6 | `ARRAY_ELEMENT_DERIVED` |

The code names the type only. The kind lives in `element_size`, so
`real(real64)` is code 2 with element size 8.

## Polymorphic array dummies

A `class(t)` array dummy may be associated with an actual whose dynamic element
type extends `t`, so its elements are wider than `t`'s own layout. No extra
field is needed for this: `element_size` and the per-dimension `stride_bytes`
already describe the actual's concrete elements, because the caller builds the
descriptor from the actual, not from the dummy's declared type.

The callee therefore must not stride by its declared type's size. At entry a
`class(t)` array dummy reads `element_size` from the descriptor and uses it as
its element stride for the whole call; a `type(t)` dummy is monomorphic and
keeps its compile-time stride. The declared type still governs which components
are nameable — a `class(t)` dummy sees only `t`'s prefix of each element — so
the declared and dynamic types stay distinct exactly as for a scalar.

## Ownership and lifetime

Exactly one descriptor owns any given allocation. `ARRAY_FLAG_OWNS_DATA`
marks that descriptor. `release_array_descriptor` returns the base pointer
only for an owning descriptor and then resets every field to the null state,
so the pointer reaches the runtime deallocator exactly once. For a borrowed
descriptor it returns a null pointer and still resets the descriptor, so
dropping a view never frees storage.

- An allocatable array's descriptor owns its allocation and is the entity's
  only representation (#336): `allocate` installs the shape and the owning
  flag, `deallocate` frees the base pointer and returns the descriptor to the
  unallocated state, and `move_alloc` copies the whole record and clears the
  source. Finalization order is unchanged: elements are finalized before the
  base pointer is released.
- A pointer array's descriptor never owns storage acquired by pointer
  assignment. Ownership stays with the target's own descriptor. A pointer
  that acquired storage through `allocate` does own it and is the descriptor
  that releases it.
- A dummy argument descriptor is borrowed for the duration of the call.
  A callee never releases a descriptor it received.

## View lifetime and aliasing

A section view is a descriptor whose `base` points into another descriptor's
allocation, built through `set_strided_array_descriptor`, which never sets
`ARRAY_FLAG_OWNS_DATA`. A view is therefore valid only while its parent
allocation is valid, and never extends that allocation's lifetime.

A view and its parent alias the same storage. Writes through either are
immediately visible through the other, so any operation whose result depends
on the order of overlapping element reads and writes must copy to a temporary
first, exactly as before this ABI. Constructing a view copies metadata only
and never moves elements.

## Initialization errors

| Code | Name | Condition |
|---:|---|---|
| 0 | `ARRAY_DESCRIPTOR_OK` | Descriptor initialized |
| 1 | `ARRAY_DESCRIPTOR_INVALID_RANK` | `rank < 1`, `rank > 7`, or short metadata arrays |
| 2 | `ARRAY_DESCRIPTOR_INVALID_EXTENT` | A negative extent |
| 3 | `ARRAY_DESCRIPTOR_INVALID_ELEMENT_SIZE` | `element_size <= 0` |
| 4 | `ARRAY_DESCRIPTOR_INVALID_ELEMENT_TYPE` | Element type outside 1 to 6 |
| 5 | `ARRAY_DESCRIPTOR_NULL_DATA` | A null base with a positive element count |
| 6 | `ARRAY_DESCRIPTOR_INVALID_OWNERSHIP` | Ownership requested for a null base |
| 7 | `ARRAY_DESCRIPTOR_INVALID_INDEX` | A subscript outside its dimension's bounds |

A zero-sized array is valid and may carry a null base. It is allocated and
associated, and every subscript is out of bounds.

## Scope

Migrated onto this contract so far: assumed-shape dummy arguments (#334),
runtime-sized automatic arrays (#335), and allocatable arrays (#336). Pointer
arrays and section views migrate in their own issues, as does retiring the
last of the legacy runtime-shape metadata.

Allocatable **components** of a derived type keep their own inline
`{data, extent}` record for now; that representation is not part of this
contract yet. Coarray codimensions are outside this descriptor.
