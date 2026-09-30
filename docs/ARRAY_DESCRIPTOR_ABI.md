# Array Descriptor ABI

`array_descriptor_t` is the canonical array descriptor for all migrated array
runtime interfaces. Some older lowering paths still use ad hoc
representations; each migration issue moves one path onto this layout and
removes its old representation rather than running both.

## Migration gate

The descriptor is an architecture boundary, not an adapter target. New code
must use this layout for storage, sections, assumed-shape dummies, pointers,
allocatables, character arrays, and polymorphic views. The remaining migration
work is tracked by ffc #337, #338, #339, #348, and #643. A migration is not
complete until its old convention is deleted, its ownership/view lifetime is
tested, and both positive and negative behavioral cases pass.

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

### Allocatable arrays of derived elements (#643)

`type(t), allocatable :: a(:)` and `a(:,:)` use this same descriptor without an
element-type-specific side record. `element_size` is the complete concrete
derived instance size (including inline component descriptors), and each
dimension's `stride_bytes` is derived from that size. Element component
addressing therefore computes the descriptor-relative linear index first and
then applies the concrete byte stride; it must not assume one four-byte slot.
`size`, `lbound`, and `ubound` load extents and bounds from the descriptor after
allocation, so dynamic shapes cannot fall back to declaration-time metadata.
The descriptor owns the contiguous allocation and deallocation clears it;
deep-copy assignment, finalization, `SOURCE=`/`MOLD=`, polymorphic extension
sizes, and non-unit-bound element addressing remain separate conformance gates.

The bounded direct-session owner path currently uses this canonical descriptor
for standalone intrinsic integer, real, and logical allocatables of rank one
through rank three, including allocatable dummies. Runtime allocation and
deallocation, extent inquiries, element addressing, and supported whole-owner
copy all read the descriptor's dimension records. Rank-four owners and
derived allocatable components remain outside that path; their separate
inline component descriptor is documented by the support contract.

### Rank-3 derived allocatable owners, and the `RANK`/`SHAPE` gap (#643)

A rank-3 derived-type allocatable **owner** is supported on this descriptor:

```fortran
type :: box_t
  integer :: id
end type box_t
type(box_t), allocatable :: a(:, :, :)
allocate (a(2, 2, 2))
a(1, 1, 1)%id = 5
print *, a(1, 1, 1)%id
deallocate (a)
```

allocates, stores through, reads back and deallocates, matching gfortran
byte-exactly (`test_session_derived_alloc_array_compiler`). An earlier version
of that test demanded this declaration be **refused**, because the first
descriptor slice exposed only ranks one and two; that was a slice boundary and
not a language rule, and the diagnostic string it expected was never present in
`src`, so the case had been red since it was written. It is replaced by the
positive comparison above, which is the stronger assertion.

`RANK(a)` and `SHAPE(a)` are **supported** as of this commit:

- `RANK(a)` lowers to the declared rank as an `i32` immediate — F2018
  16.9.122 makes RANK a compile-time value for every non-assumed-rank
  array, so it is correct whether or not the array is currently allocated.
  Pinned on integer rank-1/2/3, static rank-4, and rank-3 derived
  allocatables (`test_session_derived_alloc_array_compiler`,
  `test_rank3_rank_intrinsic`, converted from refusal to byte-exact).
- `s = shape(a)` where `s` is a rank-1 integer array of length
  `RANK(a)` reads each extent **from the descriptor**
  (`emit_alloc_desc_load_extent`) and stores it into `s(i)`. Until now it
  read the static `array_dim_sizes` slots, which an allocatable never
  writes: `s = shape(a)` **silently stored zeros** where gfortran stored
  the extents (verified `ffc [0 0]` vs `gfortran [2 3]` at the parent
  commit — a wrong-output defect, not a refusal). Fixed and falsified:
  reverting the descriptor read brings the zeros back and the oracle
  fails; `7 5 7 2 4 6 2 3` matches gfortran 20/20 including a
  reallocated `(5,7)` array and a derived rank-3 `(2,4,6)` owner.

STILL OPEN within #643, stated rather than smoothed: `print *, shape(a)`
directly as a print item is refused (`unsupported scalar intrinsic`) —
array-valued inquiry results have no print materialization path yet.

Whole-array **broadcast into an allocatable component**, `h%items = 5`, is
**supported** as of the follow-up commit: a scalar rhs does not reallocate
(F2018 7.2.3.3) — it broadcasts over the CURRENT shape, so the extent is
read from the component descriptor at runtime and a counted loop stores the
value at the component element stride (`
lower_alloc_rank1_component_scalar_broadcast`, reusing the section path's
`store_alloc_component_element_runtime` idiom). Byte-exact vs gfortran over
20 runs (integer, real, scalar-identifier rhs, allocated or not); the
previous refusal reappears when the dispatch is disabled (falsification).
Constructor and compile-time-sized whole-array rhs forms keep their
reallocating path unchanged.

### Logical elements and store width

A `LOGICAL` element occupies a **four-byte slot**, written and read as four
bytes. Writing it from a comparison, which arrives as `i1`, must still write the
full slot: inferring store width from the value produced a one-byte store at a
four-byte stride, left the upper three bytes uninitialised, and made a `.false.`
element read back stack garbage and print `.true.`, nondeterministically from a
byte-identical binary. Section printing had a matching gap - the section print
dispatch had no logical case, so `print *, l(1:6:1)` emitted `0 1 0 1` while
`print *, l` and `print *, l(2)` printed `F T F T` correctly. Both are pinned by
`test/test_logical_array_parity.sh`, twenty runs per shape against gfortran, and
the width pairing with `liric` is stated in `docs/RUNTIME_ABI.md`.

### Assumed-rank `RANK (1)` / `RANK (2)` / `RANK (3)` / `RANK (4)` boundary

The supported genuine assumed-rank slice uses this same descriptor without a
second ABI. For a contained `REAL :: x(..)` dummy called with a rank-1,
rank-2, rank-3, or rank-4 whole array, the caller allocates a borrowed stack
`array_descriptor_t`, fills `base`, `element_size`,
`element_type=ARRAY_ELEMENT_REAL`, the actual rank, the allocated/associated
flags, and each active dimension's `lower_bound=1`, `extent`, and
`stride_bytes`, then passes the descriptor address through the dummy's single
visible pointer parameter. The callee does not infer rank from source or
append hidden extents. Inside exactly one matching `RANK (1)`, `RANK (2)`, or
`RANK (3)`, or `RANK (4)` arm it loads every active extent from dimensions 1
through 4; rank 1 retains descriptor byte-stride addressing, while rank 2,
rank 3, and rank 4 compute column-major linear indices with the descriptor
element-size stride. Rank 3 uses `i1 + (i2-1)*extent(1) +
(i3-1)*extent(1)*extent(2)` and rank 4 adds
`(i4-1)*extent(1)*extent(2)*extent(3)`.

This boundary is borrowed: the callee never releases or changes descriptor
ownership. Only a whole rank-1, rank-2, rank-3, or rank-4 REAL actual with one
matching rank arm is admitted. Scalar actuals, rank five and higher,
dynamic-shape forms, sections and aliases (including pointers), global or
owning storage, `RANK DEFAULT`, `RANK (*)`, unsupported or non-matching rank
arms, and unsupported element kinds are outside the boundary and must be
refused before emitting a descriptor call.

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
runtime-sized automatic arrays (#335), allocatable arrays (#336), and the
rank-1 intrinsic section views documented in `RUNTIME_ABI.md`. Pointer arrays,
higher-rank and derived-type section views, forwarding a whole pointer or
already-strided assumed-shape dummy to another assumed-shape dummy, and the
side-effectful section-bound expressions, and the last legacy runtime-shape
metadata migrate in their own issues.

The declaration-shape classifier is now isolated in the typed
`session_program_lowering_array_shape.f90` descendant. It preserves the
assumed-shape/assumed-rank/assumed-size classification boundary without
changing descriptor bytes or hidden arguments;
`test_session_array_shape_module_compiler` compares the emitted rank-2 shape
and element observations with gfortran.

Allocatable **components** of a derived type keep an inline component-owned
descriptor `{data, extent1[, extent2]}` rather than the canonical standalone
descriptor. Intrinsic integer, real, and logical rank-one/rank-two components
use it for allocation, `allocated`, `size`, element access, and deallocation;
whole-component assignment, rank-two aliases/actual arguments, unsupported
kinds, and higher ranks remain outside this contract. Coarray codimensions are
outside this descriptor.

### Cross-unit USE of derived types (#337/#338)

A program-unit spec `use M` + `type(T) :: x`, with `T` defined in a
separately compiled module, resolves at declaration-collection time: the
collection pass imports `M.fmod` (tolerating a missing artefact, which the
body walk reports with full context) before collecting declarations, so
`T` is registered when the declaration is resolved. The import is
idempotent per module name; a renames-free repeat writes nothing, while a
USE carrying renames always runs to its collision check, so the legal
repeat stays valid and the ambiguous rename stays ambiguous. Cross-unit
`type(T)` dummies keep the working bare by-reference address ABI. A
cross-unit `class(T)` dummy is refused with
`unsupported call argument: cross-unit CLASS dummy argument ... awaits
the unified class descriptor ABI`: the caller's scalar class descriptor
box is not consumed by the separately compiled callee, and passing the
bare address silently loses the actual. Type-bound dispatch through
recorded vtables is unaffected.
