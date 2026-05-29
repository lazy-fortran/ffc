module session_program_lowering_types
    use, intrinsic :: iso_c_binding, only: c_int32_t, c_int64_t, c_ptr
    use liric_session_bindings, only: lr_operand_desc_t, liric_session_t
    use fortfront, only: ast_arena_t
    implicit none
    private

    integer, parameter, public :: VALUE_I32 = 1
    integer, parameter, public :: VALUE_F64 = 2
    integer, parameter, public :: VALUE_LOGICAL = 3
    integer, parameter, public :: VALUE_CHARACTER = 4
    integer, parameter, public :: VALUE_DERIVED = 5
    integer, parameter, public :: VALUE_DEFERRED_CHARACTER_RESULT = 6
    integer, parameter, public :: VALUE_SUBROUTINE = 7
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
    character(len=8), parameter, public :: I32_INTRINSIC_NAMES(15) = &
                                   [character(len=8) :: 'abs', 'min', 'max', 'mod', &
                                    'iand', 'ior', 'ieor', 'not', 'ishft', &
                                    'ishftc', 'sign', 'int', 'nint', 'floor', &
                                    'ceiling']
    integer, parameter, public :: I32_INTRINSIC_IDS(15) = &
                          [I32_INTRINSIC_ABS, I32_INTRINSIC_MIN, I32_INTRINSIC_MAX, &
                           I32_INTRINSIC_MOD, I32_INTRINSIC_IAND, I32_INTRINSIC_IOR, &
                           I32_INTRINSIC_IEOR, I32_INTRINSIC_NOT, I32_INTRINSIC_ISHFT, &
                           I32_INTRINSIC_ISHFTC, I32_INTRINSIC_SIGN, I32_INTRINSIC_INT, &
                           I32_INTRINSIC_NINT, I32_INTRINSIC_FLOOR, &
                           I32_INTRINSIC_CEILING]
    character(len=8), parameter, public :: F64_INTRINSIC_NAMES(13) = &
                                   [character(len=8) :: 'abs', 'min', 'max', 'real', &
                                    'sign', 'sqrt', 'exp', 'log', 'sin', 'cos', &
                                    'tan', 'atan', 'atan2']
    integer, parameter, public :: F64_INTRINSIC_IDS(13) = &
                          [F64_INTRINSIC_ABS, F64_INTRINSIC_MIN, F64_INTRINSIC_MAX, &
                           F64_INTRINSIC_REAL, F64_INTRINSIC_SIGN, F64_INTRINSIC_SQRT, &
                           F64_INTRINSIC_EXP, F64_INTRINSIC_LOG, F64_INTRINSIC_SIN, &
                           F64_INTRINSIC_COS, F64_INTRINSIC_TAN, F64_INTRINSIC_ATAN, &
                           F64_INTRINSIC_ATAN2]

    type, public :: symbol_t
        character(len=64) :: name = ''
        integer :: value_kind = VALUE_I32
        type(lr_operand_desc_t) :: value
        type(lr_operand_desc_t) :: address
        logical :: is_parameter = .false.
        logical :: is_reference = .false.
        logical :: has_address = .false.
        integer :: character_length = 0
        logical :: has_character_value = .false.
        logical :: is_array = .false.
        logical :: is_derived = .false.
        integer :: derived_type_index = 0
        integer :: array_size = 0
        integer :: array_lower_bound = 1
        type(lr_operand_desc_t) :: element_address
        logical :: has_i32_constant = .false.
        integer(c_int64_t) :: i32_constant = 0_c_int64_t
        logical :: is_deferred_character = .false.
        type(lr_operand_desc_t) :: deferred_data
        type(lr_operand_desc_t) :: deferred_length
        logical :: is_allocatable = .false.
        type(lr_operand_desc_t) :: allocatable_descriptor_address
    end type symbol_t

    type, public :: derived_type_info_t
        character(len=64) :: name = ''
        integer :: component_count = 0
        character(len=64), allocatable :: component_names(:)
        logical, allocatable :: component_has_default(:)
        integer(c_int64_t), allocatable :: component_default_value(:)
        integer, allocatable :: component_array_size(:)
        integer :: binding_count = 0
        character(len=64), allocatable :: binding_method_names(:)
        character(len=64), allocatable :: binding_target_names(:)
    end type derived_type_info_t

    type, public :: module_exports_t
        character(len=64) :: module_name = ''
        integer, allocatable :: derived_type_indices(:)
        integer :: derived_type_count = 0
        integer, allocatable :: parameter_indices(:)
        integer :: parameter_count = 0
    end type module_exports_t

    integer, parameter, public :: MAX_PROC_ARGS = 16

    type, public :: external_procedure_t
        character(len=64) :: fortran_name = ''
        character(len=64) :: c_name = ''
        integer :: return_value_kind = VALUE_I32
        integer :: arg_value_kinds(MAX_PROC_ARGS) = VALUE_I32
        integer :: arg_count = 0
    end type external_procedure_t

    type, public :: lowering_context_t
        type(liric_session_t) :: session
        type(ast_arena_t) :: arena
        type(symbol_t), allocatable :: symbols(:)
        integer :: symbol_count = 0
        type(derived_type_info_t), allocatable :: derived_types(:)
        integer :: derived_type_count = 0
        type(module_exports_t), allocatable :: module_exports(:)
        integer :: module_export_count = 0
        integer(c_int32_t) :: current_block_id = 0_c_int32_t
        integer(c_int32_t) :: i32_print_format_id = -1_c_int32_t
        integer(c_int32_t) :: str_print_format_id = -1_c_int32_t
        integer :: string_literal_count = 0
        logical :: has_command_args = .false.
        type(lr_operand_desc_t) :: argc_value
        type(lr_operand_desc_t) :: argv_value
        character(len=64), allocatable :: function_names(:)
        integer, allocatable :: function_value_kinds(:)
        integer, allocatable :: function_param_counts(:)
        integer :: function_count = 0
        type(external_procedure_t), allocatable :: external_procedures(:)
        integer :: external_procedure_count = 0
        logical :: in_internal_function = .false.
        logical :: in_internal_subroutine = .false.
        integer :: current_function_result_index = 0
        logical :: current_block_terminated = .false.
        integer(c_int32_t) :: current_loop_exit_block = 0_c_int32_t
        integer(c_int32_t) :: current_loop_latch_block = 0_c_int32_t
        logical :: in_counted_do = .false.
        logical :: current_block_exited_loop = .false.
    end type lowering_context_t

    type, public :: branch_result_t
        type(symbol_t), allocatable :: symbols(:)
        integer :: symbol_count = 0
        integer(c_int32_t) :: predecessor_block_id = 0_c_int32_t
        logical :: terminated = .false.
    end type branch_result_t

end module session_program_lowering_types
