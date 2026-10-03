# Exploration: fixed-length allocatable CHARACTER storage (#348)

Probed 2025-10 at ffc 899dc14 (allocatable_oob_01, allocate_29,
read_78, string_42 refuse signatures).

## Findings

- `character(len=:), allocatable` scalars: fully descriptor-backed
  (`define_deferred_character_symbol`, canonical {data,len,capacity,
  storage} ABI); `allocate(character(len=<expr>) :: s)` lowers via
  `lower_allocate_char_type_spec` (allocatable.f90:725).
- `character(len=2), allocatable :: s` does NOT reach that path:
  the allocate guard demands `is_deferred_character`. Debug shows the
  symbol at allocate time is `value_kind=CHARACTER, is_allocatable=.false.,
  rank=0, character_length=2`.
- Root cause: fixed allocatable scalar chars declare through
  `define_character_symbol` (static slot, session_program_lowering_
  character.f90 ~:641); `is_allocatable` is set on symbols only for
  parameter re-declares (:568), deferred upgrades (:611), arrays
  (allocatable.f90 declare path sets it for rank>0 slots).
- Relaxing only the allocate guard is unsafe: the fixed slot has no
  owned descriptor, so malloc/store would bind to the wrong storage.

## Required migration (next slice)

1. In the character declaration path, route
   `allocatable + fixed-length scalar CHARACTER` to the deferred
   descriptor factory, recording `character_length` as the declared
   width (same trick already used for fixed dummies at :575-:586 and
   for fixed-with-deferred-ABI results at :556-:562).
2. `allocate(character(len=N) :: s)` then works via the existing
   deferred guard; add the spec-length vs declared-length equality
   check (pure digit parser needed: ffc self-host refuses internal
   READ with literal or variable UNIT in compiler sources).
3. Assignment/print/len/trim reuse the deferred machinery unchanged.
4. Oracle: >=20 rows (allocate+assign+print+trim+len+compare+deallocate
   +reallocate longer), falsification by neutering the routing gate.

## Self-host constraints re-confirmed

- Compiler sources must not use `read(str, '(i10)')` or even
  `read(str, fmt_var)`; parse digits manually.
- `print '(A,L1,...)'` debug prints are viable in compiler sources.

## Negative results (2nd probe, same day)

- Instrumenting the character.f90 existing_index branch (:611) and the
  allocatable.f90 rank-0 char declare tail (:191) produced NO output
  for `character(len=2), allocatable :: s`: neither handler owns this
  declaration in the current lowering order.
- The allocate-time guard symbol has `character_length=2,
  is_allocatable=.false., is_deferred=.false.` and `sum`-style
  debugging shows the slot exists before allocate.
- Next probe should start from the generic declaration dispatcher
  (who calls define_declared_symbol / which handler claims
  allocatable-character scalars) with a single print at each
  candidate owner, or dump the declaration_node attributes as seen by
  fortfront (`is_allocatable` on the node itself was never confirmed
  true at any instrumented site).

## RESOLVED (same day, 57135ef)

Owner found: `lower_declaration_entities` @ session_program_lowering
.f90:20605 routed allocatable non-CHARACTER scalars to
`lower_scalar_allocatable_declaration` and excluded CHARACTER; allocatable
CHARACTER fell into `define_character_symbol` (static slot). Fix:
route allocatable+non-dummy CHARACTER there too; new
`declare_character_allocatable` mints/upgrades the deferred descriptor
(idempotent via `upgrade_existing_deferred_character_symbol` for
pre-created slots: function results, module hosts), records the
declared width, sets is_allocatable. INTENT-carrying dummies keep the
caller-binding paths. Length-mismatch in type-spec allocate refused via
self-host-safe digit parser. string_42 + allocatable-function results
byte-exact; oracle runs=22 match=22 + 3 pinned refusals.
