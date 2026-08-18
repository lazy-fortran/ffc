module session_program_lowering_types
    use, intrinsic :: iso_c_binding, only: c_int32_t, c_int64_t
    use liric_session_bindings, only: lr_operand_desc_t, liric_session_t
    use fortfront_compiler, only: ast_arena_t
    use session_symbol_table, only: session_symbol_table_t
    implicit none
    private

    ! Module-variable classification (#263): MODVAR_OK means a scalar module
    ! variable that can be lowered to a global; MODVAR_UNSUPPORTED keeps the
    ! unit xfail with a clean diagnostic.
    integer, parameter, public :: MODVAR_OK = 0
    integer, parameter, public :: MODVAR_UNSUPPORTED = 1
    ! SCALAR_REAL_NONE is the scalar-kind engine's "not a real scalar" answer
    ! (#447). It is deliberately not a VALUE_* code: the engine never guesses a
    ! width when a symbol or construct does not resolve.
    integer, parameter, public :: SCALAR_REAL_NONE = 0
    integer, parameter, public :: VALUE_I32 = 1
    integer, parameter, public :: VALUE_F64 = 2
    integer, parameter, public :: VALUE_LOGICAL = 3
    ! VALUE_F32 is f32 (single precision, kind 4). Bare 'real' and real(4)
    ! lower as f32; real(8) and double precision stay as VALUE_F64.
    integer, parameter, public :: VALUE_F32 = 10
    ! VALUE_I64 is integer(8) / integer(int64). Arithmetic uses i64 ops;
    ! list-directed output uses %20ld (gfortran field width 21 including sep).
    integer, parameter, public :: VALUE_I64 = 11
    ! VALUE_I8  is integer(1)/integer(int8).  Field width 5 (gfortran); %4d.
    integer, parameter, public :: VALUE_I8 = 12
    ! VALUE_I16 is integer(2)/integer(int16). Field width 7 (gfortran); %6d.
    integer, parameter, public :: VALUE_I16 = 13
    ! VALUE_C4 is complex(4): two f32 components.  `address` = re ptr, `element_address` = im ptr.
    integer, parameter, public :: VALUE_C4 = 14
    ! VALUE_C8 is complex(8): two f64 components.  Same layout as VALUE_C4.
    integer, parameter, public :: VALUE_C8 = 15
    integer, parameter, public :: VALUE_CHARACTER = 4
    integer, parameter, public :: VALUE_DERIVED = 5
    integer, parameter, public :: VALUE_DEFERRED_CHARACTER_RESULT = 6
    integer, parameter, public :: VALUE_SUBROUTINE = 7
    integer, parameter, public :: VALUE_C_PTR = 8
    integer, parameter, public :: VALUE_CLASS_STAR = 9
    ! VALUE_PROC_PTR is a procedure pointer; stored as a ptr alloca slot holding
    ! the callee function address.  Lowering for #245 slice B3d.
    integer, parameter, public :: VALUE_PROC_PTR = 16
    ! VALUE_ARRAY_RESULT is a fixed-size rank-1 array function result. Like the
    ! derived and complex results it returns via a leading sret pointer: the
    ! caller passes the destination array's base address as param 0 and the
    ! callee binds the result symbol's element storage onto that pointer. The
    ! element scalar kind comes from the result variable's body declaration.
    integer, parameter, public :: VALUE_ARRAY_RESULT = 17
    ! VALUE_ALLOC_ARRAY_RESULT is an allocatable rank-1/2 array function
    ! result. It returns via a leading sret pointer to a 40-byte allocatable
    ! descriptor (data ptr + bounds): the caller passes a zeroed temporary
    ! descriptor as param 0, the callee allocates into it, and the caller then
    ! moves that descriptor into the destination allocatable.
    integer, parameter, public :: VALUE_ALLOC_ARRAY_RESULT = 18
    ! VALUE_DATA_PTR_RESULT is a scalar data POINTER function result (#407).
    ! The function returns the target address itself, so no pointee storage is
    ! copied: the LIRIC function returns ptr and the caller binds that address
    ! into the left-hand pointer of a pointer assignment. A disassociated
    ! result returns a null pointer, which the caller's ASSOCIATED tests at
    ! run time.
    integer, parameter, public :: VALUE_DATA_PTR_RESULT = 19
    ! Runtime type ids carried in a class(*) descriptor's type slot. Intrinsic
    ! ids are fixed and disjoint from derived-type ids (a derived type's id is
    ! its 1-based table index, always small).
    integer, parameter, public :: TYPE_ID_INTEGER = 1000001
    integer, parameter, public :: TYPE_ID_REAL = 1000002
    integer, parameter, public :: TYPE_ID_LOGICAL = 1000003
    ! Coarse intrinsic type classes used by the comparison operand
    ! type-mismatch check. Numeric groups integer/real/complex together.
    integer, parameter, public :: CMP_CLASS_UNKNOWN = 0
    integer, parameter, public :: CMP_CLASS_NUMERIC = 1
    integer, parameter, public :: CMP_CLASS_CHAR = 2
    integer, parameter, public :: CMP_CLASS_LOGICAL = 3
    ! Exact intrinsic type of an array-constructor value. Unlike CMP_CLASS_*,
    ! integer and real are distinct: they pick different element types.
    integer, parameter, public :: CTOR_TYPE_UNKNOWN = 0
    integer, parameter, public :: CTOR_TYPE_INTEGER = 1
    integer, parameter, public :: CTOR_TYPE_REAL = 2
    integer, parameter, public :: CTOR_TYPE_LOGICAL = 3
    integer, parameter, public :: CTOR_TYPE_CHAR = 4
    integer, parameter, public :: I32_INTRINSIC_NONE = 0
    integer, parameter, public :: I32_INTRINSIC_ABS = 1
    integer, parameter, public :: I32_INTRINSIC_MIN = 2
    integer, parameter, public :: I32_INTRINSIC_MAX = 3
    integer, parameter, public :: I32_INTRINSIC_MOD = 4
    integer, parameter, public :: I32_INTRINSIC_IAND = 5
    integer, parameter, public :: I32_INTRINSIC_IOR = 6
    integer, parameter, public :: I32_INTRINSIC_IEOR = 7
    integer, parameter, public :: I32_INTRINSIC_NOT = 8
    integer, parameter, public :: I32_INTRINSIC_ISHFT = 9
    integer, parameter, public :: I32_INTRINSIC_ISHFTC = 10
    integer, parameter, public :: I32_INTRINSIC_SIGN = 11
    integer, parameter, public :: I32_INTRINSIC_INT = 12
    integer, parameter, public :: I32_INTRINSIC_NINT = 13
    integer, parameter, public :: I32_INTRINSIC_FLOOR = 14
    integer, parameter, public :: I32_INTRINSIC_CEILING = 15
    integer, parameter, public :: I32_INTRINSIC_MATMUL = 16
    integer, parameter, public :: I32_INTRINSIC_TRANSPOSE = 17
    integer, parameter, public :: I32_INTRINSIC_DOT_PRODUCT = 18
    integer, parameter, public :: I32_INTRINSIC_RESHAPE = 19
    integer, parameter, public :: I32_INTRINSIC_SELECTED_INT_KIND = 20
    integer, parameter, public :: I32_INTRINSIC_SELECTED_REAL_KIND = 21
    integer, parameter, public :: I32_INTRINSIC_MODULO = 22
    integer, parameter, public :: I32_INTRINSIC_DIM = 23
    integer, parameter, public :: I32_INTRINSIC_IABS = 24
    integer, parameter, public :: I32_INTRINSIC_IBITS = 25
    integer, parameter, public :: I32_INTRINSIC_IBSET = 26
    integer, parameter, public :: I32_INTRINSIC_IBCLR = 27
    integer, parameter, public :: I32_INTRINSIC_BIT_SIZE = 28
    integer, parameter, public :: F64_INTRINSIC_NONE = 0
    integer, parameter, public :: F64_INTRINSIC_ABS = 1
    integer, parameter, public :: F64_INTRINSIC_MIN = 2
    integer, parameter, public :: F64_INTRINSIC_MAX = 3
    integer, parameter, public :: F64_INTRINSIC_REAL = 4
    integer, parameter, public :: F64_INTRINSIC_SIGN = 5
    integer, parameter, public :: F64_INTRINSIC_SQRT = 6
    integer, parameter, public :: F64_INTRINSIC_EXP = 7
    integer, parameter, public :: F64_INTRINSIC_LOG = 8
    integer, parameter, public :: F64_INTRINSIC_SIN = 9
    integer, parameter, public :: F64_INTRINSIC_COS = 10
    integer, parameter, public :: F64_INTRINSIC_TAN = 11
    integer, parameter, public :: F64_INTRINSIC_ATAN = 12
    integer, parameter, public :: F64_INTRINSIC_ATAN2 = 13
    integer, parameter, public :: F64_INTRINSIC_ASIN = 14
    integer, parameter, public :: F64_INTRINSIC_ACOS = 15
    integer, parameter, public :: F64_INTRINSIC_SINH = 16
    integer, parameter, public :: F64_INTRINSIC_COSH = 17
    integer, parameter, public :: F64_INTRINSIC_TANH = 18
    integer, parameter, public :: F64_INTRINSIC_ASINH = 19
    integer, parameter, public :: F64_INTRINSIC_ACOSH = 20
    integer, parameter, public :: F64_INTRINSIC_ATANH = 21
    integer, parameter, public :: F64_INTRINSIC_LOG10 = 22
    integer, parameter, public :: F64_INTRINSIC_ERF = 23
    integer, parameter, public :: F64_INTRINSIC_ERFC = 24
    integer, parameter, public :: F64_INTRINSIC_GAMMA = 25
    integer, parameter, public :: F64_INTRINSIC_LOG_GAMMA = 26
    integer, parameter, public :: F64_INTRINSIC_HYPOT = 27
    integer, parameter, public :: F64_INTRINSIC_MOD = 28
    integer, parameter, public :: F64_INTRINSIC_MODULO = 29
    integer, parameter, public :: F64_INTRINSIC_AINT = 30
    integer, parameter, public :: F64_INTRINSIC_ANINT = 31
    character(len=18), parameter, public :: I32_INTRINSIC_NAMES(28) = &
        [character(len=18) :: 'abs', 'min', 'max', 'mod', &
        'iand', 'ior', 'ieor', 'not', 'ishft', &
        'ishftc', 'sign', 'int', 'nint', 'floor', &
        'ceiling', 'matmul', 'transpose', &
        'dot_product', 'reshape', 'selected_int_kind', &
        'selected_real_kind', 'modulo', 'dim', 'iabs', &
        'ibits', 'ibset', 'ibclr', 'bit_size']
    integer, parameter, public :: I32_INTRINSIC_IDS(28) = &
        [I32_INTRINSIC_ABS, I32_INTRINSIC_MIN, I32_INTRINSIC_MAX, &
        I32_INTRINSIC_MOD, I32_INTRINSIC_IAND, I32_INTRINSIC_IOR, &
        I32_INTRINSIC_IEOR, I32_INTRINSIC_NOT, I32_INTRINSIC_ISHFT, &
        I32_INTRINSIC_ISHFTC, I32_INTRINSIC_SIGN, I32_INTRINSIC_INT, &
        I32_INTRINSIC_NINT, I32_INTRINSIC_FLOOR, &
        I32_INTRINSIC_CEILING, I32_INTRINSIC_MATMUL, &
        I32_INTRINSIC_TRANSPOSE, I32_INTRINSIC_DOT_PRODUCT, &
        I32_INTRINSIC_RESHAPE, I32_INTRINSIC_SELECTED_INT_KIND, &
        I32_INTRINSIC_SELECTED_REAL_KIND, I32_INTRINSIC_MODULO, &
        I32_INTRINSIC_DIM, I32_INTRINSIC_IABS, I32_INTRINSIC_IBITS, &
        I32_INTRINSIC_IBSET, I32_INTRINSIC_IBCLR, &
        I32_INTRINSIC_BIT_SIZE]
    character(len=9), parameter, public :: F64_INTRINSIC_NAMES(31) = &
        [character(len=9) :: 'abs', 'min', 'max', 'real', &
        'sign', 'sqrt', 'exp', 'log', 'sin', 'cos', &
        'tan', 'atan', 'atan2', 'asin', 'acos', 'sinh', &
        'cosh', 'tanh', 'asinh', 'acosh', 'atanh', &
        'log10', 'erf', 'erfc', 'gamma', 'log_gamma', &
        'hypot', 'mod', 'modulo', 'aint', 'anint']
    integer, parameter, public :: F64_INTRINSIC_IDS(31) = &
        [F64_INTRINSIC_ABS, F64_INTRINSIC_MIN, F64_INTRINSIC_MAX, &
        F64_INTRINSIC_REAL, F64_INTRINSIC_SIGN, F64_INTRINSIC_SQRT, &
        F64_INTRINSIC_EXP, F64_INTRINSIC_LOG, F64_INTRINSIC_SIN, &
        F64_INTRINSIC_COS, F64_INTRINSIC_TAN, F64_INTRINSIC_ATAN, &
        F64_INTRINSIC_ATAN2, F64_INTRINSIC_ASIN, &
        F64_INTRINSIC_ACOS, F64_INTRINSIC_SINH, &
        F64_INTRINSIC_COSH, F64_INTRINSIC_TANH, &
        F64_INTRINSIC_ASINH, F64_INTRINSIC_ACOSH, &
        F64_INTRINSIC_ATANH, F64_INTRINSIC_LOG10, &
        F64_INTRINSIC_ERF, F64_INTRINSIC_ERFC, &
        F64_INTRINSIC_GAMMA, F64_INTRINSIC_LOG_GAMMA, &
        F64_INTRINSIC_HYPOT, F64_INTRINSIC_MOD, &
        F64_INTRINSIC_MODULO, F64_INTRINSIC_AINT, &
        F64_INTRINSIC_ANINT]

    ! COMMON-block slot (#1578, #1900): one shared global per variable in a
    ! COMMON block, keyed by block name and position. has_init/init_text carry
    ! a BLOCK DATA literal initialiser folded later into the global.
    integer, parameter, public :: COMMON_MAX_SLOTS = 64
    integer, parameter, public :: EQUIV_MAX_MEMBERS = 32
    ! Highest array rank the direct session lowers for fixed-size and dummy
    ! arrays. Per-dimension lower bounds and extents are stored inline in
    ! symbol_t, so this caps those fixed arrays (Fortran 2003 max rank).
    integer, parameter, public :: ARRAY_MAX_RANK = 7
    ! Byte size of an allocatable array descriptor: data pointer plus the
    ! lower/upper bound pair of each supported dimension, all i64.
    ! Allocatable arrays are described by the canonical array descriptor
    ! (docs/ARRAY_DESCRIPTOR_ABI.md), so their stack or global slot is that
    ! record's size (#336).
    integer, parameter, public :: ALLOC_DESCRIPTOR_BYTES = 200
    type, public :: common_slot_t
        character(len=:), allocatable :: block_name
        character(len=:), allocatable :: var_name
        integer :: value_kind = VALUE_I32
        integer :: byte_offset = 0
        logical :: has_init = .false.
        character(len=:), allocatable :: init_text
        logical :: is_array = .false.
        integer :: array_size = 0
        integer, allocatable :: array_init_indices(:)
        character(len=:), allocatable :: array_init_values(:)
    end type common_slot_t

    ! One member of an EQUIVALENCE group (#370). `byte_offset` is the storage
    ! offset of the designated element inside the member itself (nonzero only
    ! for an array-element designator); `start_offset` is where the member's
    ! own storage begins inside the group global, once every member's
    ! designated element has been aligned on the shared association point.
    type, public :: equiv_member_t
        character(len=:), allocatable :: var_name
        integer :: value_kind = VALUE_I32
        logical :: is_array = .false.
        integer :: array_size = 0
        integer :: byte_offset = 0
        integer :: start_offset = 0
    end type equiv_member_t

    type, public :: symbol_t
        ! `name` is display data (diagnostics, mangling), not identity.
        ! Identity is the FortFront binding below, when this symbol came
        ! from a declaration FortFront could resolve (#327).
        character(len=64) :: name = ''
        ! FortFront binding identity: the declaration node, the entity
        ! within it (integer :: a, b, c share one node), and the scope
        ! node that owns it, as reported by declaration_binding_t. Stays
        ! .false./0 for symbols ffc synthesises without a FortFront
        ! declaration (inferred lazy-Fortran locals, DO variables,
        ! function-result and ABI temporaries).
        logical :: has_binding = .false.
        integer :: binding_declaration_index = 0
        integer :: binding_entity_index = 0
        integer :: binding_scope_index = 0
        integer :: value_kind = VALUE_I32
        type(lr_operand_desc_t) :: value
        type(lr_operand_desc_t) :: address
        logical :: is_parameter = .false.
        ! A dummy argument of the current procedure. Its storage belongs to the
        ! caller, so scope exit never finalizes it (#403) and a data-pointer
        ! result may legitimately return it (#407).
        logical :: is_dummy_argument = .false.
        ! Temporary by-value override used while inlining a statement function;
        ! it is resolved by its synthetic name until the body expression is
        ! restored (#332).
        logical :: is_statement_function_argument = .false.
        logical :: is_reference = .false.
        logical :: has_address = .false.
        ! Bound to a COMMON slot global (session_program_lowering_common.inc):
        ! its own program-unit declaration, reached before or after the
        ! COMMON statement in source order, must not reallocate storage.
        logical :: is_common_bound = .false.
        ! Bound to a saved-local global whose DATA values were folded into the
        ! static bytes (#466). The body walk consumes the matching DATA values
        ! without storing them, so the initializer applies once rather than on
        ! every call.
        logical :: is_static_data_initialized = .false.
        integer :: character_length = 0
        logical :: has_character_value = .false.
        logical :: is_array = .false.
        ! An assumed-rank dummy is descriptor-backed until SELECT RANK binds a
        ! supported concrete arm. It must not be treated as a static array
        ! before that construct has established the rank.
        logical :: is_assumed_rank = .false.
        integer :: array_rank = 0
        integer :: array_size = 0
        integer :: array_lower_bound = 1
        integer, dimension(ARRAY_MAX_RANK) :: array_dim_sizes = 0
        integer, dimension(ARRAY_MAX_RANK) :: array_dim_lowers = 0
        ! Non-contiguous rank-1 pointer/view stride in bytes. A zero value
        ! means ordinary contiguous element addressing; pointer descriptors
        ! replace it with a runtime stride at dummy entry.
        integer :: array_stride_bytes = 0
        logical :: has_runtime_array_stride = .false.
        type(lr_operand_desc_t) :: runtime_array_stride_bytes
        ! Runtime extents for descriptor-backed dummies whose actual has no
        ! compile-time-foldable shape, or for the genuine assumed-rank
        ! rank(1)/rank(2)/rank(3)/rank(4) SELECT RANK slice. array_dim_sizes stays the
        ! 0 sentinel; each active dimension's count lives in this i32 operand.
        logical, dimension(4) :: has_runtime_dim_size = .false.
        type(lr_operand_desc_t), dimension(4) :: runtime_dim_size
        ! Element stride, in i32 slots, of a polymorphic array dummy (#422).
        ! A class(t) array dummy may receive an actual whose dynamic element
        ! type is an extension of t, so its elements are wider than the
        ! declared type's layout. The concrete element size travels in the
        ! canonical array descriptor's element_size field and is loaded here at
        ! procedure entry; element addressing scales by this operand instead of
        ! the declared type's compile-time slot count.
        logical :: has_runtime_element_slots = .false.
        type(lr_operand_desc_t) :: runtime_element_slots
        ! A rank-1 local automatic array whose element count is only known at
        ! runtime (integer :: a(n) with n a dummy/host value). Its storage is
        ! a dynamic alloca reached through element_address; the compile-time
        ! lower bound stays in array_lower_bound/array_dim_lowers(1) and the
        ! runtime element count lives in runtime_dim_size(1). array_size stays
        ! the 0 sentinel. Whole-array print/assign and size()/sum() walk a
        ! genuine LIRIC loop over runtime_dim_size(1) instead of unrolling.
        logical :: is_runtime_array = .false.
        ! True for an assumed-shape dummy bound through the canonical descriptor.
        ! Runtime reductions use this to keep intrinsic-specific rank boundaries
        ! precise when the descriptor ABI itself admits a higher rank.
        logical :: is_assumed_shape_dummy = .false.
        ! Canonical array descriptor backing a runtime-sized automatic array
        ! (#335). It is the stored shape of record: base address, element size
        ! and type, rank, flags, and per-dimension lower bound, extent, and byte
        ! stride, laid out by docs/ARRAY_DESCRIPTOR_ABI.md.
        logical :: has_runtime_descriptor = .false.
        type(lr_operand_desc_t) :: runtime_descriptor_address
        logical :: is_derived = .false.
        integer :: derived_type_index = 0
        ! Polymorphism (#417). is_polymorphic marks an entity declared class(t):
        ! derived_type_index stays the DECLARED type, while the dynamic type is
        ! a runtime value living in the scalar class descriptor the caller
        ! built. dynamic_type_address is the address of that descriptor's
        ! dynamic_type field; class_descriptor_address is the descriptor base.
        logical :: is_polymorphic = .false.
        ! A class(t), pointer declaration has unresolved runtime dispatch but
        ! does not use the scalar class descriptor lowering above.
        logical :: is_class_pointer = .false.
        logical :: has_dynamic_type_address = .false.
        type(lr_operand_desc_t) :: dynamic_type_address
        type(lr_operand_desc_t) :: class_descriptor_address
        type(lr_operand_desc_t) :: element_address
        ! A rank-1 array function result bound to the sret buffer (param 0). The
        ! body array declaration rebinds its shape/element kind onto this symbol's
        ! element_address instead of allocating fresh storage (array results).
        logical :: is_array_result = .false.
        logical :: has_i32_constant = .false.
        integer(c_int64_t) :: i32_constant = 0_c_int64_t
        logical :: is_transient_i32_constant = .false.
        logical :: is_deferred_character = .false.
        ! A scalar automatic CHARACTER whose declared length is known only
        ! at runtime (currently character(len=len(dummy))). It shares the
        ! {data, length} storage layout with a deferred-length character,
        ! but its captured length is immutable across assignments: values
        ! are blank-padded or truncated to that width.
        logical :: is_runtime_fixed_character = .false.
        type(lr_operand_desc_t) :: deferred_data
        type(lr_operand_desc_t) :: deferred_length
        ! Canonical character descriptor ownership slots, see
        ! docs/CHARACTER_DESCRIPTOR_ABI.md: capacity at offset 16 and
        ! storage_class at offset 24. Only a local deferred-length allocatable
        ! scalar owns the full 32-byte record; a dummy descriptor keeps the
        ! 16-byte {data, length} prefix, which the 32-byte record extends
        ! compatibly, so passing it to a character(len=*) dummy is unchanged.
        logical :: has_character_ownership = .false.
        type(lr_operand_desc_t) :: deferred_capacity
        type(lr_operand_desc_t) :: deferred_storage
        logical :: is_allocatable = .false.
        type(lr_operand_desc_t) :: allocatable_descriptor_address
        ! Compile-time element count of a rank-1 allocatable when the most
        ! recent allocate/constructor used a constant size; 0 when unknown.
        ! Drives compile-time-unrolled whole-array print without a runtime loop.
        integer :: allocatable_static_size = 0
        ! Scalar POINTER/TARGET (#245, slice B3a). A target lives in memory at
        ! `address`; a pointer carries the current target's `address` once
        ! associated. `is_associated` tracks straight-line association at compile
        ! time for `associated`/`nullify`.
        logical :: is_pointer = .false.
        logical :: is_target = .false.
        logical :: is_associated = .false.
        ! A pointer whose association state is only known at run time: its
        ! address came from a data-pointer function result (#407). ASSOCIATED
        ! compares that address against null instead of folding is_associated.
        logical :: has_runtime_association = .false.
        ! Procedure pointer (#245 B3d): `address` holds the ptr alloca slot;
        ! after assignment `value` holds the loaded callee ptr operand.
        logical :: is_proc_pointer = .false.
        ! File I/O (#247 B5c). When this symbol holds a Fortran unit number that
        ! was opened via OPEN, file_ptr_address is the alloca'd ptr holding the
        ! FILE* handle. is_file_unit is set to .true. at that point. The
        ! form flag lets WRITE distinguish list-directed from unformatted
        ! binary transfer for a unit opened in this lowering session.
        logical :: is_file_unit = .false.
        logical :: is_unformatted = .false.
        ! Logical expressions use i32 semantic values internally. Keep the
        ! declared storage width separately so unformatted transfer preserves
        ! LOGICAL(KIND=...) byte size without changing expression lowering.
        integer :: logical_kind_bytes = 0
        type(lr_operand_desc_t) :: file_ptr_address
        ! Straight-line constant integer assigned to this scalar, tracked only to
        ! link a unit number used by name (unit = 10) with WRITE/READ/REWIND that
        ! reference it by number. Kept separate from has_i32_constant so it does
        ! not affect array-bound or kind-inquiry constant folding.
        logical :: has_unit_const = .false.
        integer :: unit_const = 0
        ! Compile-time text of a character named constant (PARAMETER),
        ! kept so a later constant's initializer can fold a reference to
        ! this one (z_pad = x_pad // y_pad) at compile time.
        character(len=:), allocatable :: character_constant_text
        ! Remote binding this symbol was imported from by a USE rename
        ! (module name plus the name used inside that module). Repeating the
        ! same remote binding under the same local name is valid; a second,
        ! different remote binding under that local name is ambiguous
        ! (F2018 19.5.2).
        character(len=:), allocatable :: use_rename_module
        character(len=:), allocatable :: use_rename_remote
    end type symbol_t

    ! One resolved declaration record. The binding triple is the identity;
    ! names and source spelling are metadata retained for diagnostics and
    ! later declaration lowering. A record may exist before its procedure has
    ! an active LIRIC function, so it deliberately carries no storage operand.
    type, public :: declaration_record_t
        integer :: declaration_node_index = 0
        integer :: declaration_entity_index = 0
        integer :: scope_node_index = 0
        character(len=64) :: name = ''
        character(len=64) :: type_name = ''
        integer :: type_kind = 0
        integer :: kind_value = 0
        integer :: value_kind = VALUE_I32
        integer :: rank = 0
        logical :: is_array = .false.
        logical :: is_parameter = .false.
    end type declaration_record_t

    type, public :: array_section_info_t
        character(len=64) :: source_name = ''
        integer :: source_index = 0
        integer :: source_rank = 0
        integer :: result_rank = 0
        integer :: kept_dims(ARRAY_MAX_RANK) = 0
        logical :: keep_dim(ARRAY_MAX_RANK) = .false.
        integer(c_int64_t) :: source_lowers(ARRAY_MAX_RANK) = 0_c_int64_t
        integer(c_int64_t) :: source_sizes(ARRAY_MAX_RANK) = 0_c_int64_t
        integer(c_int64_t) :: section_lowers(ARRAY_MAX_RANK) = 0_c_int64_t
        integer(c_int64_t) :: section_uppers(ARRAY_MAX_RANK) = 0_c_int64_t
        integer(c_int64_t) :: section_strides(ARRAY_MAX_RANK) = 1_c_int64_t
        integer(c_int64_t) :: section_extents(ARRAY_MAX_RANK) = 0_c_int64_t
        integer(c_int64_t) :: scalar_indices(ARRAY_MAX_RANK) = 0_c_int64_t
        logical :: has_runtime_bounds = .false.
    end type array_section_info_t

    ! One operand of an all/any/count comparison mask (or a bare mask), used
    ! by the general scalar-result reduction path. mode selects the storage:
    ! 0 scalar broadcast, 1 whole stored array, 2 array section, 3 array
    ! constructor of scalar elements, 4 nested whole-array expression. extent
    ! is the element count for an array-shaped operand and -1 for a scalar.
    type, public :: reduction_operand_t
        integer :: mode = 0
        integer :: sym = 0
        integer :: scalar_idx = 0
        integer, allocatable :: flat(:)
        type(array_section_info_t) :: info
        integer :: vk = VALUE_I32
        integer :: extent = -1
        logical :: apply_abs = .false.
        logical :: apply_complex_abs = .false.
        integer :: component = 0
    end type reduction_operand_t

    type, public :: derived_type_info_t
        character(len=64) :: name = ''
        logical :: layout_registered = .false.
        integer :: component_count = 0
        character(len=64), allocatable :: component_names(:)
        logical, allocatable :: component_has_default(:)
        integer(c_int64_t), allocatable :: component_default_value(:)
        integer, allocatable :: component_array_size(:)
        ! Scalar value kind of each component (VALUE_I32, VALUE_F64, VALUE_F32,
        ! VALUE_LOGICAL, VALUE_C_PTR, or VALUE_DERIVED for a nested derived
        ! component). Drives slot width and typed load/store.
        integer, allocatable :: component_value_kind(:)
        ! Derived type index of a nested derived component (0 for scalars and
        ! intrinsic arrays). A nested component occupies the inner type's slots
        ! inline; component_array_size holds that slot count.
        integer, allocatable :: component_type_index(:)
        ! True for a scalar allocatable component (integer/real/logical). Such
        ! a component stores an 8-byte data pointer (two i32 slots) inline;
        ! its value lives in a separately malloc'd slot reached by loading the
        ! pointer. Null pointer marks it unallocated.
        logical, allocatable :: component_is_allocatable(:)
        ! True for a scalar data-pointer component (integer/real/logical).
        ! Such a component stores an 8-byte target address (two i32 slots)
        ! inline; a null address marks it disassociated. `=>` writes the
        ! address, intrinsic assignment of the parent copies it, so the
        ! association travels with the object and never the pointee storage.
        logical, allocatable :: component_is_pointer(:)
        ! True for a bounded-rank allocatable array component
        ! (integer/real/logical). Such a component stores an inline descriptor
        ! with an 8-byte data pointer followed by one i64 extent per dimension.
        ! A null data pointer marks it unallocated.
        logical, allocatable :: component_is_alloc_array(:)
        ! Declared rank of an allocatable array component (1 through 3); zero for
        ! scalar and non-allocatable components. This is separate from
        ! component_dim1, which describes fixed-size component storage.
        integer, allocatable :: component_alloc_rank(:)
        ! Declared character length of a VALUE_CHARACTER component (0 for
        ! every other kind). Scalar character components keep their historical
        ! slot count in component_array_size; fixed character arrays keep their
        ! element count there and use component_dim1 as the fixed-array marker.
        integer, allocatable :: component_char_length(:)
        ! Extent of the first dimension of a fixed-size component when needed
        ! for addressing (0 for scalars). Column-major element addressing of
        ! comp(i,j) needs this stride; fixed character arrays also use a
        ! positive value as their array marker.
        integer, allocatable :: component_dim1(:)
        ! Name of the type this one extends (empty when it extends nothing).
        ! Lets a polymorphic dummy check accept an extension of its declared
        ! type (#369).
        character(len=64) :: parent_name = ''
        ! Name of the scalar FINAL procedure declared for this type (empty
        ! when the type declares none). It runs once when an owned scalar
        ! value of the type reaches the end of its lifetime (#403).
        character(len=64) :: final_proc_name = ''
        integer :: binding_count = 0
        character(len=64), allocatable :: binding_method_names(:)
        character(len=64), allocatable :: binding_target_names(:)
        ! Space-joined target names for a type-bound generic. The first name
        ! mirrors binding_target_names; the complete set is used for overload
        ! selection and is carried through .fmod metadata.
        character(len=1024), allocatable :: binding_specific_names(:)
        ! Empty unless the binding declared pass(name); names the dummy that
        ! receives the passed object.
        character(len=64), allocatable :: binding_pass_names(:)
        ! True when the binding passes the object (default PASS or PASS(name));
        ! false identifies an explicit NOPASS binding, whose call has no
        ! implicit receiver argument.
        logical, allocatable :: binding_pass_args(:)
        ! True when the layout and bindings came from an imported .fmod. The
        ! defining module owns the per-type vtable global; importing units use
        ! the binding metadata for direct calls but must not emit a duplicate
        ! definition at link time.
        logical :: is_imported = .false.
        ! Non-empty when the type declares the same binding name twice. Such a
        ! type has no single occupant for that vtable slot, so it is reported
        ! when the vtables are emitted rather than when it is collected: a
        ! collection error would be swallowed by the "skip a type ffc cannot
        ! lower" path and reappear as an unrelated diagnostic (#420).
        character(len=64) :: duplicate_binding_name = ''
    end type derived_type_info_t

    type, public :: module_exports_t
        character(len=64) :: module_name = ''
        integer, allocatable :: derived_type_indices(:)
        integer :: derived_type_count = 0
        integer, allocatable :: parameter_indices(:)
        integer :: parameter_count = 0
        ! Non-parameter variable declarations exported from the module (#249 B7a).
        integer, allocatable :: variable_indices(:)
        integer :: variable_count = 0
        ! enum_node definitions exported from the module so a module procedure
        ! can host-associate the enumerators (#1826).
        integer, allocatable :: enum_indices(:)
        integer :: enum_count = 0
    end type module_exports_t

    integer, parameter, public :: MAX_PROC_ARGS = 16
    integer, parameter, public :: MAX_GENERIC_SPECIFICS = 8

    ! A generic interface: maps a generic name to up to MAX_GENERIC_SPECIFICS
    ! specific procedure names (#249 B7c). At a call site the first specific
    ! whose first-argument kind matches the actual is selected.
    type, public :: generic_interface_t
        character(len=64) :: generic_name = ''
        integer :: specific_count = 0
        character(len=64) :: specific_names(MAX_GENERIC_SPECIFICS) = ''
        integer :: specific_arg_counts(MAX_GENERIC_SPECIFICS) = 0
        ! Per-specific argument kinds: position 1 is the first
        ! dummy, up to MAX_PROC_ARGS. Unused trailing entries stay VALUE_I32.
        integer :: specific_arg_kinds(MAX_PROC_ARGS, MAX_GENERIC_SPECIFICS) = &
            VALUE_I32
        ! Per-specific argument ranks: 0 for scalar, >0 for array rank.
        ! Populated from declaration metadata so dispatch distinguishes
        ! same-kind specifics that differ in rank (refs #303).
        integer :: specific_arg_ranks(MAX_PROC_ARGS, MAX_GENERIC_SPECIFICS) = 0
        ! Return value kind of each specific.
        integer :: return_kinds(MAX_GENERIC_SPECIFICS) = VALUE_I32
    end type generic_interface_t

    ! A user-defined operator or assignment overload. interface operator(+)
    ! / operator(.dot.) / assignment(=) maps an operator token to specific
    ! procedures; a binary-op or assignment whose operand kinds match dispatches
    ! to the matching specific instead of the builtin path. Dispatch keys on the
    ! pair of operand value kinds so distinct overloads of the same token (e.g.
    ! integer .myop. integer vs real .myop. real) stay separate.
    type, public :: operator_interface_t
        character(len=64) :: operator_name = ''
        logical :: is_assignment = .false.
        integer :: specific_count = 0
        character(len=64) :: specific_names(MAX_GENERIC_SPECIFICS) = ''
        integer :: first_arg_kinds(MAX_GENERIC_SPECIFICS) = VALUE_I32
        integer :: second_arg_kinds(MAX_GENERIC_SPECIFICS) = VALUE_I32
        integer :: return_kinds(MAX_GENERIC_SPECIFICS) = VALUE_I32
    end type operator_interface_t

    ! INTENT contract of a dummy argument as carried in a .fmod (#397).
    integer, parameter, public :: ARG_INTENT_NONE = 0
    integer, parameter, public :: ARG_INTENT_IN = 1
    integer, parameter, public :: ARG_INTENT_OUT = 2
    integer, parameter, public :: ARG_INTENT_INOUT = 3

    ! A Lazy procedure whose untyped dummies FortFront could not resolve to one
    ! concrete type, so it monomorphized instead: the written name is not a
    ! callable body, and each concrete signature lives in its own typed copy
    ! (#437).
    type, public :: lazy_specialization_t
        character(len=64) :: procedure_name = ''
    end type lazy_specialization_t

    type, public :: external_procedure_t
        character(len=64) :: fortran_name = ''
        character(len=64) :: c_name = ''
        integer :: return_value_kind = VALUE_I32
        integer :: arg_value_kinds(MAX_PROC_ARGS) = VALUE_I32
        ! Declared dummy names, when the signature carried them, so a call
        ! may associate its actuals by keyword (#408). Empty when unknown.
        character(len=64) :: arg_names(MAX_PROC_ARGS) = ''
        integer :: arg_count = 0
        ! A bind(c) external passes arguments by value; a separately
        ! compiled module procedure resolved from a .fmod passes them by
        ! reference (Fortran ABI) and targets its mangled name (#284).
        logical :: by_reference = .false.
        ! A BIND(C) interface uses the platform C ABI even when its dummies
        ! are not VALUE.  Keep this separate from by_reference: the latter is
        ! the .fmod/module-procedure ABI selector, while this flag identifies
        ! the call contract itself.
        logical :: is_bind_c = .false.
        ! Per-dummy contracts the .fmod carries: OPTIONAL dummies may be
        ! omitted at a call site, a VALUE dummy receives a copy, and an
        ! INTENT(OUT)/INTENT(INOUT) dummy requires a definable actual (#397).
        logical :: arg_is_optional(MAX_PROC_ARGS) = .false.
        logical :: arg_is_value(MAX_PROC_ARGS) = .false.
        integer :: arg_intents(MAX_PROC_ARGS) = ARG_INTENT_NONE
        ! Rank of each dummy (0 for a scalar) and, for an explicit-shape array
        ! dummy, its total element count. A generic resolves an imported
        ! specific by these ranks, and an array actual takes the base-address
        ! ABI because the artefact said the dummy is an array (#415).
        integer :: arg_ranks(MAX_PROC_ARGS) = 0
        integer :: arg_extents(MAX_PROC_ARGS) = 0
        ! An opaque imported dummy has no scalar kind claim. Its actual is
        ! accepted only by a dedicated ABI lowering path.
        logical :: arg_is_opaque(MAX_PROC_ARGS) = .false.
        ! A separately compiled class(t) dummy receives a scalar class
        ! descriptor; type(t) passed-object dummies receive the raw object
        ! address. The declared type spelling is retained for class dummies
        ! so inherited bindings use the parent's descriptor type.
        logical :: arg_is_class(MAX_PROC_ARGS) = .false.
        character(len=64) :: arg_class_types(MAX_PROC_ARGS) = ''
    end type external_procedure_t

    integer, parameter, public :: MAX_NAMELIST_MEMBERS = 32

    ! Namelist input scratch sizes and the IOSTAT values reported for a group
    ! that ends before it closes and for malformed namelist input (#436).
    integer, parameter, public :: NAMELIST_NAME_BUFFER = 63
    integer, parameter, public :: NAMELIST_VALUE_BUFFER = 255
    integer, parameter, public :: NAMELIST_IOSTAT_END = -1
    integer, parameter, public :: NAMELIST_IOSTAT_BAD = 5010

    ! A NAMELIST group: maps a group name to its ordered member names so a
    ! WRITE(unit, nml=group) can emit the group banner plus each member's
    ! current value (#247 namelist I/O).
    type, public :: namelist_group_t
        character(len=64) :: group_name = ''
        integer :: member_count = 0
        character(len=64) :: member_names(MAX_NAMELIST_MEMBERS) = ''
    end type namelist_group_t

    ! A statement function definition: name, ordered scalar dummy names, and the
    ! arena index of the defining expression. Calls inline the body.
    integer, parameter, public :: MAX_STMT_FN_ARGS = 8
    type, public :: statement_function_t
        character(len=64) :: name = ''
        integer :: arg_count = 0
        character(len=64) :: arg_names(MAX_STMT_FN_ARGS) = ''
        integer :: body_expr_index = 0
    end type statement_function_t

    type, public :: lowering_context_t
        type(liric_session_t) :: session
        ! Non-owning, read-only-by-contract view of the translation unit AST.
        ! The frontend result owns the arena and outlives every lowering
        ! context. Child procedure contexts share this view; they must never
        ! assign or deallocate the target. Keeping ownership outside the
        ! context prevents an O(procedures * AST size) deep-copy cost.
        type(ast_arena_t), pointer :: arena => null()
        integer :: root_index = 0
        ! Lazy Fortran dialect defaults are active for this compilation unit
        ! (#438). The driver sets this from the frontend input mode, so the
        ! policy is explicit in the lowering context instead of being guessed
        ! from the source text: a kind-less `real` is real(8), and a dummy
        ! argument without an explicit INTENT is INTENT(IN).
        logical :: lazy_mode = .false.
        type(symbol_t), allocatable :: symbols(:)
        integer :: symbol_count = 0
        ! Maps a FortFront (declaration, scope) binding identity onto a
        ! slot in `symbols` (#327). Populated as declarations materialize;
        ! consulted before any name comparison at a reference site.
        type(session_symbol_table_t) :: binding_table
        ! Resolved declaration metadata collected before executable lowering
        ! (#457). This registry has no LIRIC operands: those are materialized
        ! only inside the active procedure, while every later path can still
        ! consult one binding-keyed declaration record.
        type(declaration_record_t), allocatable :: declaration_records(:)
        integer :: declaration_record_count = 0
        logical :: declaration_collection_complete = .false.
        ! Symbols at index <= block_scope_floor belong to an enclosing scope. A
        ! declaration inside a BLOCK whose name matches such a symbol creates a
        ! fresh shadowing slot instead of reusing the outer storage (#280).
        integer :: block_scope_floor = 0
        ! Number of explicit BLOCK/ASSOCIATE storage scopes currently active.
        ! Construct-local declarations are the identities that must be popped.
        integer :: storage_scope_depth = 0
        ! Arena index of the declaration whose specification expressions
        ! are being lowered (#329), or 0 outside a declaration. A
        ! specification expression - an array bound, a character length,
        ! a kind selector - is evaluated in the scope where its
        ! declaration appears, so this node is the anchor FortFront
        ! resolves its names against. Character lengths in particular
        ! reach lowering as text with no arena node of their own, and
        ! text alone cannot distinguish a BLOCK-shadowed constant from
        ! the host one it hides.
        integer :: current_declaration_index = 0
        type(derived_type_info_t), allocatable :: derived_types(:)
        integer :: derived_type_count = 0
        ! Local spellings imported through USE renames map to one canonical
        ! derived-type record. Keeping aliases out of derived_types preserves
        ! type identity for inheritance, nested components, and descriptors.
        character(len=64), allocatable :: derived_type_alias_names(:)
        integer, allocatable :: derived_type_alias_indices(:)
        integer :: derived_type_alias_count = 0
        ! Parameterized derived types (#411). A PDT definition registers its
        ! name here as a template instead of a concrete layout; every distinct
        ! tuple of constant actual type parameters instantiates one concrete
        ! derived type in derived_types, named base(v1,v2,...).
        character(len=64), allocatable :: pdt_template_names(:)
        integer :: pdt_template_count = 0
        type(module_exports_t), allocatable :: module_exports(:)
        integer :: module_export_count = 0
        integer(c_int32_t) :: current_block_id = 0_c_int32_t
        integer(c_int32_t) :: i32_print_format_id = -1_c_int32_t
        integer(c_int32_t) :: i64_print_format_id = -1_c_int32_t
        integer(c_int32_t) :: i8_print_format_id = -1_c_int32_t
        integer(c_int32_t) :: i16_print_format_id = -1_c_int32_t
        integer(c_int32_t) :: str_print_format_id = -1_c_int32_t
        integer :: string_literal_count = 0
        ! Per-unit token inserted into counter-named .ffc.* content globals
        ! (string literals, char temporaries, user formats) so a separately
        ! compiled module object does not collide with the main or a sibling
        ! module object at link time (#284). Empty for a main executable.
        character(len=:), allocatable :: unit_symbol_prefix
        logical :: has_command_args = .false.
        ! A module-only compilation unit (bare module or a container of
        ! modules with no main program) compiled to an object emits no main;
        ! its module nodes carry no executable body to run (#284).
        logical :: unit_is_module_only = .false.
        type(lr_operand_desc_t) :: argc_value
        type(lr_operand_desc_t) :: argv_value
        character(len=64), allocatable :: function_names(:)
        integer, allocatable :: function_value_kinds(:)
        integer, allocatable :: function_param_counts(:)
        integer, allocatable :: function_node_indices(:)
        integer :: function_count = 0
        ! USE-rename procedure aliases (#274): a call to local name
        ! proc_alias_locals(k) resolves to the real procedure
        ! proc_alias_targets(k) before mangling.
        character(len=64), allocatable :: proc_alias_locals(:)
        character(len=64), allocatable :: proc_alias_targets(:)
        integer :: proc_alias_count = 0
        type(external_procedure_t), allocatable :: external_procedures(:)
        integer :: external_procedure_count = 0
        ! Generic interface table (#249 B7c).
        type(generic_interface_t), allocatable :: generics(:)
        integer :: generic_count = 0
        ! Operator/assignment overload table.
        type(operator_interface_t), allocatable :: operators(:)
        integer :: operator_count = 0
        logical :: in_internal_function = .false.
        logical :: in_internal_subroutine = .false.
        ! Alternate-return ABI (#353). A subroutine with `*` dummies takes one
        ! hidden trailing i32-by-reference selector parameter at 0-based
        ! position altret_param_index; `return n` stores n into it. The caller
        ! allocates the slot, zeroes it, and branches on the loaded value.
        ! altret_slot_count is the number of `*` dummies (0 means none).
        integer :: altret_slot_count = 0
        integer :: altret_param_index = -1
        ! Name of the contained procedure currently being lowered. Lets an
        ! assumed-shape dummy a(:) recover its extent from the caller's actual.
        character(len=:), allocatable :: current_proc_name
        ! AST node for the procedure currently being lowered. Unlike the text
        ! name, this remains unambiguous when a contained procedure shadows a
        ! USE-associated procedure with the same spelling (#330).
        integer :: current_proc_node_index = 0
        ! BIND(C) bodies use the C ABI for their emitted signature. This is
        ! deliberately per-procedure: ordinary Fortran/module procedures in
        ! the same translation unit keep the pointer-based ABI.
        logical :: current_proc_bind_c = .false.
        integer :: current_function_result_index = 0
        ! Character result temporaries created while lowering the current
        ! statement. A character-returning call writes its result through a
        ! return descriptor the caller stack-allocates; whoever consumes that
        ! result reads the bytes but does not own them, so the temporary needs
        ! an owner of its own. The statement is that owner: each temporary is
        ! registered here as it is created and released once the statement
        ! that produced it has finished with it. A consumer that takes
        ! ownership instead - a deferred destination adopting the returned
        ! descriptor - deregisters it, so a block is released exactly once.
        type(lr_operand_desc_t), allocatable :: char_temp_data(:)
        type(lr_operand_desc_t), allocatable :: char_temp_storage(:)
        integer :: char_temp_count = 0
        ! Whole-array expressions may contain an array-valued function call.
        ! Materialise each such call once per consuming statement, then let the
        ! elementwise evaluator load its cached array elements rather than
        ! re-invoking the function for every element.
        integer, allocatable :: array_expression_cache_nodes(:)
        integer, allocatable :: array_expression_cache_symbols(:)
        integer :: array_expression_cache_count = 0
        ! FORALL assignment snapshot (#673). A FORALL RHS observes the target
        ! array as it existed before the construct, while stores update the
        ! original array. The active flag is scoped to RHS lowering; the
        ! snapshot address is stack storage owned by the current procedure.
        logical :: forall_snapshot_reads = .false.
        logical :: forall_snapshot_writes = .false.
        integer :: forall_snapshot_symbol = 0
        type(lr_operand_desc_t) :: forall_snapshot_address
        integer :: forall_body_statement_index = 0
        logical :: current_block_terminated = .false.
        integer(c_int32_t) :: current_loop_exit_block = 0_c_int32_t
        integer(c_int32_t) :: current_loop_latch_block = 0_c_int32_t
        logical :: in_loop = .false.
        logical :: current_block_exited_loop = .false.
        ! Values captured at explicit EXIT edges. The loop's common exit block
        ! merges these with the normal condition-false edge so an EXIT from a
        ! branch preserves the values computed on that path.
        integer :: loop_exit_count = 0
        integer(c_int32_t), allocatable :: loop_exit_blocks(:)
        type(lr_operand_desc_t), allocatable :: loop_exit_values(:,:)
        ! GOTO label table for a labeled program body (#270). Each labeled
        ! statement owns a LIRIC block; a `goto N` branches to label_blocks(k)
        ! where label_names(k) == 'N'. Active only while in_labeled_body.
        logical :: in_labeled_body = .false.
        character(len=16), allocatable :: label_names(:)
        integer(c_int32_t), allocatable :: label_blocks(:)
        integer :: label_count = 0
        ! Search paths (-I) for .fmod module artefacts resolved on USE.
        character(len=:), allocatable :: include_paths(:)
        integer :: include_path_count = 0
        ! File I/O unit table (#247 B5c). unit_table_id is the LIRIC global vreg
        ! for the [256 x ptr] table; -1 means not yet created.
        integer(c_int32_t) :: unit_table_id = -1_c_int32_t
        ! Next unit number assigned by newunit=. Start at 10 to avoid 0/1/6.
        integer(c_int32_t) :: next_file_unit = 10_c_int32_t
        ! NAMELIST group table (#247 namelist I/O).
        type(namelist_group_t), allocatable :: namelist_groups(:)
        integer :: namelist_group_count = 0
        ! Statement functions (f(x) = x*x + 1). Each entry records the name, its
        ! scalar dummy names, and the arena index of the defining expression.
        ! A call to such a name inlines the body with actuals bound to dummies.
        type(statement_function_t), allocatable :: statement_functions(:)
        integer :: statement_function_count = 0
        type(lazy_specialization_t), allocatable :: lazy_specializations(:)
        integer :: lazy_specialization_count = 0
        ! OPEN(unit=6, sign='plus') reconfigures the preconnected stdout
        ! connection's sign mode (#280): PRINT and WRITE(*,...) share that
        ! connection, so a forced-plus mode applies to their F editing too.
        logical :: stdout_force_plus_sign = .false.
        ! Explicitly declared variable names (from declaration_node). Used
        ! by the inferred-symbol seeding pass to avoid overwriting an
        ! explicit declaration with an inferred one (#262).
        character(len=64), allocatable :: explicit_decl_names(:)
        integer :: explicit_decl_count = 0
    end type lowering_context_t

    type, public :: branch_result_t
        type(symbol_t), allocatable :: symbols(:)
        integer :: symbol_count = 0
        integer(c_int32_t) :: predecessor_block_id = 0_c_int32_t
        logical :: terminated = .false.
    end type branch_result_t

end module session_program_lowering_types
