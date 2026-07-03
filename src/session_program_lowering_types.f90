module session_program_lowering_types
    use, intrinsic :: iso_c_binding, only: c_int32_t, c_int64_t, c_ptr
    use liric_session_bindings, only: lr_operand_desc_t, liric_session_t
    use fortfront_compiler, only: ast_arena_t
    implicit none
    private

    ! Module-variable classification (#263): MODVAR_OK means a scalar module
    ! variable that can be lowered to a global; MODVAR_UNSUPPORTED keeps the
    ! unit xfail with a clean diagnostic.
    integer, parameter, public :: MODVAR_OK = 0
    integer, parameter, public :: MODVAR_UNSUPPORTED = 1
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
        character(len=18), parameter, public :: I32_INTRINSIC_NAMES(23) = &
            [character(len=18) :: 'abs', 'min', 'max', 'mod', &
            'iand', 'ior', 'ieor', 'not', 'ishft', &
            'ishftc', 'sign', 'int', 'nint', 'floor', &
            'ceiling', 'matmul', 'transpose', &
            'dot_product', 'reshape', 'selected_int_kind', &
            'selected_real_kind', 'modulo', 'dim']
        integer, parameter, public :: I32_INTRINSIC_IDS(23) = &
            [I32_INTRINSIC_ABS, I32_INTRINSIC_MIN, I32_INTRINSIC_MAX, &
            I32_INTRINSIC_MOD, I32_INTRINSIC_IAND, I32_INTRINSIC_IOR, &
            I32_INTRINSIC_IEOR, I32_INTRINSIC_NOT, I32_INTRINSIC_ISHFT, &
            I32_INTRINSIC_ISHFTC, I32_INTRINSIC_SIGN, I32_INTRINSIC_INT, &
            I32_INTRINSIC_NINT, I32_INTRINSIC_FLOOR, &
            I32_INTRINSIC_CEILING, I32_INTRINSIC_MATMUL, &
            I32_INTRINSIC_TRANSPOSE, I32_INTRINSIC_DOT_PRODUCT, &
            I32_INTRINSIC_RESHAPE, I32_INTRINSIC_SELECTED_INT_KIND, &
            I32_INTRINSIC_SELECTED_REAL_KIND, I32_INTRINSIC_MODULO, &
            I32_INTRINSIC_DIM]
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
        type, public :: common_slot_t
            character(len=:), allocatable :: block_name
            character(len=:), allocatable :: var_name
            integer :: value_kind = VALUE_I32
            integer :: pos_in_block = 0
            logical :: has_init = .false.
            character(len=:), allocatable :: init_text
            logical :: is_array = .false.
            integer :: array_size = 0
            integer, allocatable :: array_init_indices(:)
            character(len=:), allocatable :: array_init_values(:)
        end type common_slot_t

        type, public :: symbol_t
            character(len=64) :: name = ''
            integer :: value_kind = VALUE_I32
            type(lr_operand_desc_t) :: value
            type(lr_operand_desc_t) :: address
            logical :: is_parameter = .false.
            logical :: is_reference = .false.
            logical :: has_address = .false.
            ! Bound to a COMMON slot global (session_program_lowering_common.inc):
            ! its own program-unit declaration, reached before or after the
            ! COMMON statement in source order, must not reallocate storage.
            logical :: is_common_bound = .false.
            integer :: character_length = 0
            logical :: has_character_value = .false.
            logical :: is_array = .false.
            integer :: array_rank = 0
            integer :: array_size = 0
            integer :: array_lower_bound = 1
            integer, dimension(ARRAY_MAX_RANK) :: array_dim_sizes = 0
            integer, dimension(ARRAY_MAX_RANK) :: array_dim_lowers = 0
            ! Runtime extent of a rank-1 assumed-shape dummy whose actual has no
            ! compile-time-foldable shape (an allocatable actual): the hidden i64
            ! extent argument ABI. array_dim_sizes(1) stays the 0 sentinel; the
            ! per-dimension count lives in this i32 operand instead.
            logical, dimension(2) :: has_runtime_dim_size = .false.
            type(lr_operand_desc_t), dimension(2) :: runtime_dim_size
            logical :: is_derived = .false.
            integer :: derived_type_index = 0
            type(lr_operand_desc_t) :: element_address
            ! A rank-1 array function result bound to the sret buffer (param 0). The
            ! body array declaration rebinds its shape/element kind onto this symbol's
            ! element_address instead of allocating fresh storage (array results).
            logical :: is_array_result = .false.
            logical :: has_i32_constant = .false.
            integer(c_int64_t) :: i32_constant = 0_c_int64_t
            logical :: is_deferred_character = .false.
            type(lr_operand_desc_t) :: deferred_data
            type(lr_operand_desc_t) :: deferred_length
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
            ! Procedure pointer (#245 B3d): `address` holds the ptr alloca slot;
            ! after assignment `value` holds the loaded callee ptr operand.
            logical :: is_proc_pointer = .false.
            ! File I/O (#247 B5c). When this symbol holds a Fortran unit number that
            ! was opened via OPEN, file_ptr_address is the alloca'd ptr holding the
            ! FILE* handle. is_file_unit is set to .true. at that point.
            logical :: is_file_unit = .false.
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
        end type symbol_t

        type, public :: array_section_info_t
            character(len=64) :: source_name = ''
            integer :: source_index = 0
            integer :: source_rank = 0
            integer :: result_rank = 0
            integer :: kept_dims(2) = 0
            logical :: keep_dim(2) = .false.
            integer(c_int64_t) :: source_lowers(2) = 0_c_int64_t
            integer(c_int64_t) :: source_sizes(2) = 0_c_int64_t
            integer(c_int64_t) :: section_lowers(2) = 0_c_int64_t
            integer(c_int64_t) :: section_uppers(2) = 0_c_int64_t
            integer(c_int64_t) :: section_strides(2) = 1_c_int64_t
            integer(c_int64_t) :: section_extents(2) = 0_c_int64_t
            integer(c_int64_t) :: scalar_indices(2) = 0_c_int64_t
        end type array_section_info_t

        type, public :: derived_type_info_t
            character(len=64) :: name = ''
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
            integer :: binding_count = 0
            character(len=64), allocatable :: binding_method_names(:)
            character(len=64), allocatable :: binding_target_names(:)
            ! Empty unless the binding declared pass(name); names the dummy that
            ! receives the passed object.
            character(len=64), allocatable :: binding_pass_names(:)
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
            ! Value kind of the first dummy of each specific (used for dispatch).
            integer :: first_arg_kinds(MAX_GENERIC_SPECIFICS) = VALUE_I32
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

        type, public :: external_procedure_t
            character(len=64) :: fortran_name = ''
            character(len=64) :: c_name = ''
            integer :: return_value_kind = VALUE_I32
            integer :: arg_value_kinds(MAX_PROC_ARGS) = VALUE_I32
            integer :: arg_count = 0
            ! A bind(c) external passes arguments by value; a separately
            ! compiled module procedure resolved from a .fmod passes them by
            ! reference (Fortran ABI) and targets its mangled name (#284).
            logical :: by_reference = .false.
        end type external_procedure_t

        integer, parameter, public :: MAX_NAMELIST_MEMBERS = 32

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
            type(ast_arena_t) :: arena
            integer :: root_index = 0
            type(symbol_t), allocatable :: symbols(:)
            integer :: symbol_count = 0
            ! Symbols at index <= block_scope_floor belong to an enclosing scope. A
            ! declaration inside a BLOCK whose name matches such a symbol creates a
            ! fresh shadowing slot instead of reusing the outer storage (#280).
            integer :: block_scope_floor = 0
            type(derived_type_info_t), allocatable :: derived_types(:)
            integer :: derived_type_count = 0
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
                    ! Name of the contained procedure currently being lowered. Lets an
                    ! assumed-shape dummy a(:) recover its extent from the caller's actual.
                    character(len=:), allocatable :: current_proc_name
                    integer :: current_function_result_index = 0
                    logical :: current_block_terminated = .false.
                    integer(c_int32_t) :: current_loop_exit_block = 0_c_int32_t
                    integer(c_int32_t) :: current_loop_latch_block = 0_c_int32_t
                    logical :: in_counted_do = .false.
                    logical :: current_block_exited_loop = .false.
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
                end type lowering_context_t

                type, public :: branch_result_t
                    type(symbol_t), allocatable :: symbols(:)
                    integer :: symbol_count = 0
                    integer(c_int32_t) :: predecessor_block_id = 0_c_int32_t
                    logical :: terminated = .false.
                end type branch_result_t

            end module session_program_lowering_types
