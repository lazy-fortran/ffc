# Fortran 2023 delta audit

This inventory records the additions listed in the Introduction of
[J3/24-007](https://j3-fortran.org/doc/year/24/24-007.pdf), pages xiii–xiv,
and their producer or consumer gaps in the current compiler. Clause references
in the tables refer to that public Fortran 2023 interpretation draft. The
[WG5 N2212 explanation](https://wg5-fortran.org/N2201-N2250/N2212.pdf)
helps identify related additions; it explicitly describes itself as an
unapproved explanatory paper. Annex B.1 of J3/24-007 lists deleted Fortran 90
features and is not the F2018-to-F2023 addition inventory.

The reader is a compiler maintainer selecting one implementable feature slice.
A supported row claims only its stated subset. An issue row stays unsupported
until its producer contract and consumer behavior pass their tests. Coarray,
image and team additions are mandatory future work in [PLAN.md](../PLAN.md). This
audit completes scope enumeration and issue splitting; it makes no F2023
compliance or completion-percentage claim.

## Evidence boundary

The observations use ffc base `d61de6b1e247117221ed15b5b606010ad86f95b8`
plus the structured-branch worktree patch SHA256
`214ac2c020b063fd41141e6244ecb62111fd459d15e2c73bd48e7e2aea29697d`,
FortFront `bd566f2f0a59ec2cb0415896f1fba179f7f743ca`, fo 0.3.2 and
GNU Fortran 16.2.1 (20260810), measured on 2026-10-02. The probes were compiled
through `fo exec --no-build ffc`; source parsing/semantic producers and native
lowering consumers were inspected at their owners. Every gap issue carries
its reproducer or source evidence and an independent expected behavior.

Gfortran still lacks several valid additions and fails at runtime on some
unallocated deferred-character output probes. Those reference failures are
recorded as unavailable evidence, not as proof that the programs are invalid.
For example, the real BOZ assignment result is checked against the equivalent
`REAL(boz)` representation operation, and conditional FALSE probes have an
independent gfortran execution returning nine. Processor-dependent kind
numbers and clock values are checked by their contract, not byte equality.

The shared intrinsic registry omits the new trigonometric families,
SELECTED_LOGICAL_KIND, SPLIT, TOKENIZE, F_C_STRING and C_F_STRPOINTER.
[fortfront#3035](https://github.com/lazy-fortran/fortfront/issues/3035)
owns their standard identities and signatures. Its independent registry
client returns false for seven sampled new names. This metadata prerequisite
applies to the corresponding consumer issues below; it does not erase the
observed constant SELECTED_LOGICAL_KIND folding subset.

## Source form

| Addition | Disposition | Evidence and boundary | J3 clause |
| --- | --- | --- | --- |
| Free-form line budget: 10,000 characters | issue [fortfront#3022](https://github.com/lazy-fortran/fortfront/issues/3022) | 9,999 characters execute correctly; 10,001 characters are accepted without the required source diagnostic. | 6.3.2.1; 4.2 |
| Logical-statement budget: 1,000,000 characters | issue [fortfront#3022](https://github.com/lazy-fortran/fortfront/issues/3022) | Lexer has no corresponding logical-statement budget accounting; the issue covers boundary acceptance and diagnostics. | 6.3.2.6; 4.2 |
| Removal of the continuation-count limit | supported (stated subset) | A 300-continuation parameter expression prints 300 and matches gfortran; source-size limits still apply. | 6.3.2 |

## Data declaration

| Addition | Disposition | Evidence and boundary | J3 clause |
| --- | --- | --- | --- |
| Arrays/allocatables with coarray components | issue [ffc#799](https://github.com/lazy-fortran/ffc/issues/799), [ffc#800](https://github.com/lazy-fortran/ffc/issues/800) | Single-image foundation followed by full component/multi-image runtime work. | 7.5; 9.7 |
| Existing ENUM,BIND(C) terminology | supported (stated subset) | The former enumeration construct is called an interoperable enumeration; existing anonymous integer-enumerator support keeps its published restrictions. | 7.6.1 |
| Named interoperable enum types | issue [fortfront#3025](https://github.com/lazy-fortran/fortfront/issues/3025), [ffc#794](https://github.com/lazy-fortran/ffc/issues/794) | Parser consumes ENUM headers without a distinct named enum identity; scalar enum declarations are refused downstream. | 7.6.1 |
| Noninteroperable enumeration types | issue [fortfront#3026](https://github.com/lazy-fortran/fortfront/issues/3026), [ffc#795](https://github.com/lazy-fortran/ffc/issues/795) | Dedicated ordered identity, ordinal construction and enum operations are absent. | 7.6.2 |
| TYPEOF | issue [fortfront#3023](https://github.com/lazy-fortran/fortfront/issues/3023) | Declaration metadata does not retain the referenced entity type/parameters. | 7.3.2.1; 8.3 |
| CLASSOF | issue [fortfront#3024](https://github.com/lazy-fortran/fortfront/issues/3024) | Polymorphic selector declarations lose their intended identity. | 7.3.2.1; 8.3 |
| PUBLIC NAMELIST with PRIVATE objects/components | issue [ffc#767](https://github.com/lazy-fortran/ffc/issues/767) | ffc retains the previous privacy refusal; NAMELIST WRITE is separately tracked by existing #460. | 8.9 |
| Array-valued declaration bounds/DIMENSION | issue [fortfront#3028](https://github.com/lazy-fortran/fortfront/issues/3028) | Bounds vectors are not resolved into ordered dimension metadata. | 8.5.8 |
| RANK declaration clause | issue [fortfront#3027](https://github.com/lazy-fortran/fortfront/issues/3027) | The probe does not acquire rank-two deferred-shape metadata. | 8.5.17 |

## Data usage and computation

| Addition | Disposition | Evidence and boundary | J3 clause |
| --- | --- | --- | --- |
| BOZ integer initialization and assignment | supported (stated subset) | Default integer parameter Z'7' and assignment Z'9' print 7 and 9; this row claims those scalar forms only. | 7.7; 10.2.1 |
| BOZ real initialization and assignment | issue [ffc#768](https://github.com/lazy-fortran/ffc/issues/768) | Z'3F800000' and Z'40000000' become integer magnitudes instead of real(4) bit values 1 and 2. | 7.7; 10.2.1 |
| BOZ typed INTEGER/REAL array constructors | issue [ffc#769](https://github.com/lazy-fortran/ffc/issues/769) | The reject pass explicitly refuses every BOZ array element, even with an appropriate type-spec. | 7.7; 7.8 |
| BOZ interoperable enum constructors | issue [fortfront#3025](https://github.com/lazy-fortran/fortfront/issues/3025), [ffc#794](https://github.com/lazy-fortran/ffc/issues/794) | Requires the new enum identity and constructor typing; the integer bit value is contextual. | 7.6.1; 7.7 |
| Deferred allocatable ERRMSG | issue [ffc#770](https://github.com/lazy-fortran/ffc/issues/770) | A controlled double-allocation error reports success and leaves ERRMSG unallocated; no passing reference execution is available here. | 9.7 |
| Array-valued ALLOCATE bounds | issue [ffc#774](https://github.com/lazy-fortran/ffc/issues/774) | Runtime bound vectors are consumed as scalar expressions and refused. | 9.7.1 |
| Array-valued pointer bounds/remapping | issue [ffc#775](https://github.com/lazy-fortran/ffc/issues/775) | Pointer assignment does not expand vector bounds into descriptor dimensions. | 10.2.2 |
| @ multiple subscripts and multiple triplets | issue [fortfront#3029](https://github.com/lazy-fortran/fortfront/issues/3029), [ffc#793](https://github.com/lazy-fortran/ffc/issues/793) | Grouped dimensions are lost or miscounted; these are distinct from older vector gather/scatter subscripts. | 9.5.3.2 |
| Conditional expressions | issue [fortfront#3030](https://github.com/lazy-fortran/fortfront/issues/3030), [ffc#791](https://github.com/lazy-fortran/ffc/issues/791) | FALSE and runtime-false probes print 0 where gfortran prints 9; branch structure and selective evaluation are required. | 10.1.2.3 |

## Input/output

| Addition | Disposition | Evidence and boundary | J3 clause |
| --- | --- | --- | --- |
| AT character editing | issue [ffc#776](https://github.com/lazy-fortran/ffc/issues/776) | Current compound format parser refuses AT. | 13.7.4 |
| LEADING_ZERO on OPEN and WRITE | issue [fortfront#3034](https://github.com/lazy-fortran/fortfront/issues/3034), [ffc#777](https://github.com/lazy-fortran/ffc/issues/777) | WRITE parsing rejects the specifier; connection and statement policy need consumer transport. | 12.5.6.12; 12.6 |
| LZP/LZS/LZ controls | issue [ffc#777](https://github.com/lazy-fortran/ffc/issues/777) | Real format lowering has no leading-zero policy state. | 13.8 |
| Deferred allocatable IOMSG | issue [ffc#771](https://github.com/lazy-fortran/ffc/issues/771) | Message storage requires a declared fixed length. | 12.6 |
| Deferred scalar internal WRITE unit | issue [ffc#772](https://github.com/lazy-fortran/ffc/issues/772) | The deferred-length record destination is refused. | 12.6 |

## Execution control

| Addition | Disposition | Evidence and boundary | J3 clause |
| --- | --- | --- | --- |
| DO CONCURRENT REDUCE locality | issue [fortfront#3033](https://github.com/lazy-fortran/fortfront/issues/3033) | A scalar serial sum matches; the parser drops REDUCE metadata and accepts a forbidden VOLATILE variable, so full support is not claimed. | 11.1.7 |
| NOTIFY WAIT | issue [ffc#801](https://github.com/lazy-fortran/ffc/issues/801) | Notification synchronization requires the multi-image runtime. | 11.7 |
| NOTIFY= image selector | issue [ffc#801](https://github.com/lazy-fortran/ffc/issues/801) | Coindexed notification selectors require producer and runtime support. | 9.6 |
| ISO_FORTRAN_ENV NOTIFY_TYPE | issue [ffc#801](https://github.com/lazy-fortran/ffc/issues/801) | Notification state requires the standard image runtime. | 16.10 |

## Intrinsic procedures

| Addition | Disposition | Evidence and boundary | J3 clause |
| --- | --- | --- | --- |
| ACOSD/ASIND/ATAND/ATAN2D/COSD/SIND/TAND | issue [ffc#778](https://github.com/lazy-fortran/ffc/issues/778) | Typed real lowering refuses the degree-angle family. | 16.9 |
| ACOSPI/ASINPI/ATANPI/ATAN2PI/COSPI/SINPI/TANPI | issue [ffc#779](https://github.com/lazy-fortran/ffc/issues/779) | Typed real lowering refuses the half-revolution family. | 16.9 |
| SELECTED_LOGICAL_KIND constant inputs | supported (stated subset) | Foldable arguments use the documented default-only logical kind policy; a logical selected with BITS=1 prints T. | 16.9.182 |
| SELECTED_LOGICAL_KIND runtime inputs | issue [ffc#780](https://github.com/lazy-fortran/ffc/issues/780) | A runtime scalar integer argument is refused. | 16.9.182 |
| SPLIT | issue [ffc#781](https://github.com/lazy-fortran/ffc/issues/781) | Subroutine dispatch misroutes the character arguments to integer lowering. | 16.9.196 |
| SYSTEM_CLOCK selection and kind rules | issue [ffc#783](https://github.com/lazy-fortran/ffc/issues/783) | One default-kind seconds clock is present; multiple clocks are permitted rather than required. Long integer and valid real-rate calls need coverage and documented policy. | 16.9.202; 4.3.3 |
| TOKENIZE | issue [ffc#782](https://github.com/lazy-fortran/ffc/issues/782) | Neither token-owner nor integer-position output overload is implemented. | 16.9.210 |
| Deferred character intrinsic outputs | issue [ffc#773](https://github.com/lazy-fortran/ffc/issues/773) | GET_ENVIRONMENT_VARIABLE/GET_COMMAND output allocation is missing. | 16.9 |
| Collective error results can differ by image | issue [ffc#802](https://github.com/lazy-fortran/ffc/issues/802) | Image-local error behavior follows basic multi-image collective support. | 16.9 collective subroutines |

## Intrinsic modules

| Addition | Disposition | Evidence and boundary | J3 clause |
| --- | --- | --- | --- |
| ISO_FORTRAN_ENV LOGICAL8/16/32/64 | issue [ffc#784](https://github.com/lazy-fortran/ffc/issues/784) | Imports are permitted upstream, but consumer constant folding does not resolve their values. | 16.10 |
| ISO_FORTRAN_ENV REAL16 | issue [ffc#784](https://github.com/lazy-fortran/ffc/issues/784) | 16-bit real availability must use its prescribed positive/negative kind constant, not the existing 16-byte REAL128 kind. | 16.10 |
| IEEE_ARITHMETIC mode getter/setter purity | issue [fortfront#3032](https://github.com/lazy-fortran/fortfront/issues/3032), [ffc#790](https://github.com/lazy-fortran/ffc/issues/790) | GET/SET_ROUNDING_MODE and GET/SET_UNDERFLOW_MODE need pure/simple metadata and native procedure support. | 15.8; 17.11.8/10/42/44 |
| IEEE_EXCEPTIONS mode/status getter/setter purity | issue [fortfront#3032](https://github.com/lazy-fortran/fortfront/issues/3032), [ffc#797](https://github.com/lazy-fortran/ffc/issues/797) | GET/SET_MODES and GET/SET_STATUS need pure/simple metadata and opaque runtime state. | 15.8; 17.11.7/9/41/43 |
| C_F_STRPOINTER C_PTR overload | issue [ffc#786](https://github.com/lazy-fortran/ffc/issues/786) | A C_LOC character TARGET probe produces length 3 with blank data instead of length 2 with ab; borrowed target storage must be preserved. | 18.2.3.5 |
| C_F_STRPOINTER C_CHAR array overload | issue [ffc#796](https://github.com/lazy-fortran/ffc/issues/796) | The required rank-one length-one TARGET character array is refused. | 18.2.3.5 |
| F_C_STRING | issue [ffc#785](https://github.com/lazy-fortran/ffc/issues/785) | The character result intrinsic is absent. | 18.2.3.9 |
| C_F_POINTER optional LOWER | issue [ffc#787](https://github.com/lazy-fortran/ffc/issues/787) | A fourth argument is explicitly refused. | 18.2.3.3 |

## IEEE 60559:2020

| Addition | Disposition | Evidence and boundary | J3 clause |
| --- | --- | --- | --- |
| IEEE_MAX/MAX_MAG/MIN/MIN_MAG | issue [ffc#788](https://github.com/lazy-fortran/ffc/issues/788) | New propagating-NaN extrema are absent from real intrinsic lowering. | 17.11.17/18/21/22 |
| IEEE_MAX_NUM/MAX_NUM_MAG/MIN_NUM/MIN_NUM_MAG | issue [ffc#789](https://github.com/lazy-fortran/ffc/issues/789) | Revised numeric extrema require explicit signed-zero/NaN/exception semantics; current calls are refused. | 17.11.19/20/23/24 |

## Program units and procedures

| Addition | Disposition | Evidence and boundary | J3 clause |
| --- | --- | --- | --- |
| SIMPLE procedures | issue [fortfront#3032](https://github.com/lazy-fortran/fortfront/issues/3032) | The prefix can be silently ignored: a prohibited USE-associated variable read compiles and prints 7. | 15.8 |
| Conditional actual arguments | issue [fortfront#3031](https://github.com/lazy-fortran/fortfront/issues/3031), [ffc#792](https://github.com/lazy-fortran/ffc/issues/792) | A selected variable actual for INOUT is misclassified as a constant; selected address and .NIL. absence need distinct transport. | 15.5.1 R1526; 15.5.2.4 |

## Separate language contracts

Unsigned integers and `uint()` are outside this F2023 inventory. `ASYNC=`,
`DEPENDENCY` and Synthesis projection features do not add independent F2023
rows. Standard `ASYNCHRONOUS` and asynchronous I/O belong to the earlier
language baseline. The normative `@` multiple-subscript feature is included
above and does not authorize other projection syntax.

Fortran Synthesis remains outside the present denominator until its standard
contract is accepted: [standard#756](https://github.com/lazy-fortran/standard/issues/756),
then [fortfront#2976](https://github.com/lazy-fortran/fortfront/issues/2976),
[ffc#632](https://github.com/lazy-fortran/ffc/issues/632) and
[fo#120](https://github.com/lazy-fortran/fo/issues/120). This inventory neither
reopens the retired language umbrellas nor treats proposed extensions as
already supported syntax.

## Atomic issue index

Frontend issues define parsing, semantic identity and compiler-query transport.
Consumer issues define native lowering and runtime behavior. Their dependency
links freeze that order so one implementation does not guess the other side.

FortFront: [3022](https://github.com/lazy-fortran/fortfront/issues/3022), [3023](https://github.com/lazy-fortran/fortfront/issues/3023), [3024](https://github.com/lazy-fortran/fortfront/issues/3024), [3025](https://github.com/lazy-fortran/fortfront/issues/3025), [3026](https://github.com/lazy-fortran/fortfront/issues/3026), [3027](https://github.com/lazy-fortran/fortfront/issues/3027), [3028](https://github.com/lazy-fortran/fortfront/issues/3028), [3029](https://github.com/lazy-fortran/fortfront/issues/3029), [3030](https://github.com/lazy-fortran/fortfront/issues/3030), [3031](https://github.com/lazy-fortran/fortfront/issues/3031), [3032](https://github.com/lazy-fortran/fortfront/issues/3032), [3033](https://github.com/lazy-fortran/fortfront/issues/3033), [3034](https://github.com/lazy-fortran/fortfront/issues/3034), [3035](https://github.com/lazy-fortran/fortfront/issues/3035).

ffc: [767](https://github.com/lazy-fortran/ffc/issues/767), [768](https://github.com/lazy-fortran/ffc/issues/768), [769](https://github.com/lazy-fortran/ffc/issues/769), [770](https://github.com/lazy-fortran/ffc/issues/770), [771](https://github.com/lazy-fortran/ffc/issues/771), [772](https://github.com/lazy-fortran/ffc/issues/772), [773](https://github.com/lazy-fortran/ffc/issues/773), [774](https://github.com/lazy-fortran/ffc/issues/774), [775](https://github.com/lazy-fortran/ffc/issues/775), [776](https://github.com/lazy-fortran/ffc/issues/776), [777](https://github.com/lazy-fortran/ffc/issues/777), [778](https://github.com/lazy-fortran/ffc/issues/778), [779](https://github.com/lazy-fortran/ffc/issues/779), [780](https://github.com/lazy-fortran/ffc/issues/780), [781](https://github.com/lazy-fortran/ffc/issues/781), [782](https://github.com/lazy-fortran/ffc/issues/782), [783](https://github.com/lazy-fortran/ffc/issues/783), [784](https://github.com/lazy-fortran/ffc/issues/784), [785](https://github.com/lazy-fortran/ffc/issues/785), [786](https://github.com/lazy-fortran/ffc/issues/786), [787](https://github.com/lazy-fortran/ffc/issues/787), [788](https://github.com/lazy-fortran/ffc/issues/788), [789](https://github.com/lazy-fortran/ffc/issues/789), [790](https://github.com/lazy-fortran/ffc/issues/790), [791](https://github.com/lazy-fortran/ffc/issues/791), [792](https://github.com/lazy-fortran/ffc/issues/792), [793](https://github.com/lazy-fortran/ffc/issues/793), [794](https://github.com/lazy-fortran/ffc/issues/794), [795](https://github.com/lazy-fortran/ffc/issues/795), [796](https://github.com/lazy-fortran/ffc/issues/796), [797](https://github.com/lazy-fortran/ffc/issues/797).

Expanded ISO-parallel ownership: [fortfront#3036](https://github.com/lazy-fortran/fortfront/issues/3036),
[ffc#799](https://github.com/lazy-fortran/ffc/issues/799),
[ffc#800](https://github.com/lazy-fortran/ffc/issues/800),
[ffc#801](https://github.com/lazy-fortran/ffc/issues/801) and
[ffc#802](https://github.com/lazy-fortran/ffc/issues/802). These are required
future work; no parallel implementation is claimed by this planning update.

The tables contain 51 entries: 4 constrained supported entries and 47 issue-backed required entries. Family rows enumerate every procedure name in the Introduction. These counts describe this inventory and do not measure total language coverage.
