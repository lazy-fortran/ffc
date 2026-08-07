module session_program_lowering_impl
    use, intrinsic :: iso_c_binding, only: c_associated, c_char, c_double, &
        c_float, c_int, c_int8_t, c_int32_t, c_int64_t, c_loc, c_null_ptr, c_ptr
    use session_lowering_diagnostics, only: unsupported_feature_error
    use session_program_lowering_reject_text, only: normalized_base_type, &
        base_type_root, implicit_base_type, starts_with_word
    use ast_nodes_bounds, only: array_slice_node, array_bounds_node, &
        range_expression_node
    use ast_nodes_core, only: component_access_node, array_literal_node, &
        pointer_assignment_node, literal_node, &
        identifier_node, binary_op_node
    use ast_nodes_transfer, only: nullify_node, entry_node
    use ast_nodes_data, only: derived_type_node, type_binding_node, &
        block_data_node
    use ast_nodes_legacy, only: common_block_node, enum_node
    use ast_nodes_io, only: open_statement_node, close_statement_node, &
        rewind_statement_node, io_implied_do_node, inquire_statement_node, &
        io_specifier_t
    use ast_nodes_misc, only: use_statement_node, interface_block_node, &
        module_procedure_node, &
        visibility_statement_node, data_statement_node, &
        complex_literal_node, comment_node, &
        namelist_statement_node, statement_function_node, &
        end_statement_node, intrinsic_statement_node
    use ast_nodes_conditional, only: select_type_node, select_rank_node, &
        rank_block_node
    use ast_nodes_associate, only: associate_node, association_t
    use ast_nodes_control, only: block_construct_node, where_stmt_node, &
        elsewhere_clause_t, goto_node, pause_node, &
        continue_node
    use fortfront, only: assignment_node, ast_arena_t, &
        call_or_subscript_node, case_block_node, &
        case_range_node, &
        case_default_node, declaration_node, do_loop_node, &
        do_while_node, cycle_node, exit_node, function_def_node, &
        if_node, &
        parameter_declaration_node, &
        print_statement_node, program_node, read_statement_node, &
        module_node, &
        return_node, select_case_node, stop_node, &
        error_stop_node, &
        subroutine_def_node, write_statement_node, &
        allocate_statement_node, deallocate_statement_node, &
        where_node, forall_node, &
        get_subroutine_call_arg_indices, &
        get_subroutine_call_name, is_subroutine_call_statement, &
        is_binary_op, get_binary_op_info, &
        is_literal, get_literal_info, &
        is_identifier, get_identifier_name, &
        is_module_node, is_program_node, &
        declaration_binding_t, resolve_name_at_node, get_scope_bindings, &
        resolve_identifier_binding, BINDING_DECLARATION, &
        BINDING_DUMMY_ARGUMENT, BINDING_FUNCTION_RESULT, &
        BINDING_NAMED_CONSTANT, ASSOCIATION_DIRECT, ASSOCIATION_HOST, &
        ASSOCIATION_USE, &
        BINDING_ASSOCIATE_NAME, &
        get_alternate_return_label, get_return_selector, &
        is_alternate_return_dummy
    use ffc_runtime_link, only: ffc_runtime_link_input
    use ffc_polymorphic_descriptor, only: &
        POLYMORPHIC_DESCRIPTOR_DATA_OFFSET, &
        POLYMORPHIC_DESCRIPTOR_DECLARED_TYPE_OFFSET, &
        POLYMORPHIC_DESCRIPTOR_DYNAMIC_TYPE_OFFSET, &
        POLYMORPHIC_DESCRIPTOR_OWNERSHIP_OFFSET, &
        POLYMORPHIC_DESCRIPTOR_SIZE, POLYMORPHIC_OWNERSHIP_BORROWED, &
        POLYMORPHIC_OWNERSHIP_NONE, POLYMORPHIC_OWNERSHIP_OWNED, &
        POLYMORPHIC_TYPE_ID_NONE
    use liric_session_bindings, only: destroy, begin_i32_main, &
        liric_session_t, &
        begin_i32_function, begin_i64_function, begin_void_subroutine, &
        begin_typed_function, &
        begin_ptr_function, &
        emit_ret_i32_operand, emit_ret_i64_operand, emit_ret_ptr_operand, &
        emit_ret_void, &
        finish_function, finish_and_emit_exe, &
        finish_and_emit_exe_objects, emit_object_no_active_function, &
        finish_and_emit_object, emit_void_call, &
        emit_i32_call, emit_i64_call, emit_ptr_call, &
        emit_c_i32_call, emit_c_i64_call, emit_c_f32_call, emit_c_f64_call, &
        emit_c_void_call, emit_c_aggregate_call, &
        emit_i32_indirect_call, &
        emit_f64_indirect_call, &
        emit_void_indirect_call, &
        liric_session_create, lr_session_config_t, &
        i32_immediate, i32_vreg, f32_vreg, f64_vreg, global_operand, &
        lr_operand_desc_t, &
        lr_type_i32_s, lr_type_i8_s, lr_type_i16_s, lr_type_ptr_s, lr_type_i64_s, &
        lr_type_f32_s, lr_type_f64_s, lr_type_void_s, lr_type_struct_s, &
        lr_type_array_s, &
        lr_session_global, lr_session_intern, lr_session_param, &
        lr_session_emit, lr_inst_desc_t, lr_error_t, &
        clear_liric_error, status_ok, to_c_chars, &
        LR_OP_ADD, LR_OP_SREM, LR_OP_SDIV, LR_OP_SUB, &
        LR_OP_MUL, LR_OP_FADD, LR_OP_FMUL, LR_OP_FDIV, &
        LR_OP_AND, LR_OP_OR, LR_OP_XOR, &
        LR_OP_SHL, LR_OP_LSHR, LR_OP_KIND_IMM_I64, LR_OP_KIND_VREG, &
        LR_OP_KIND_GLOBAL, c_false, c_true
    use liric_session_memory_bindings, only: reserve_i32_vreg, i64_immediate, &
        ptr_vreg, &
        emit_i32_binary, emit_i32_binary_into, &
        emit_i32_copy_to, emit_real_copy_to, emit_i32_alloca, &
        emit_ptr_alloca, &
        emit_i32_load, emit_i32_store, &
        emit_i64_load, emit_ptr_load, &
        emit_i64_store, &
        emit_alloca_typed, emit_load_typed, emit_store_typed, &
        emit_i64_binary, emit_i64_alloca, &
        emit_alloca_bytes, emit_malloc, emit_calloc, &
        emit_free, emit_ptr_store, &
        emit_memcpy, emit_strnlen, emit_i64_load_at, &
        emit_i64_store_at, &
        emit_i32_array_alloca, &
        emit_i32_array_element_addr, &
        emit_ptr_array_alloca, &
        emit_ptr_array_element_addr, &
        emit_f32_array_alloca, &
        emit_f32_array_element_addr, &
        emit_f64_array_alloca, &
        emit_f64_array_element_addr, &
        emit_i64_array_alloca, &
        emit_i64_array_element_addr, &
        emit_i8_array_alloca, &
        emit_i8_array_element_addr, &
        emit_i16_array_alloca, &
        emit_i16_array_element_addr, &
        emit_ptr_offset, emit_ptr_offset_dyn, &
        emit_complex_value_load, emit_c_complex_call, &
        ptr_param, &
        i8_immediate, emit_i8_alloca, &
        emit_i8_load, emit_i8_store, emit_i8_binary, &
        i16_immediate, emit_i16_alloca, &
        emit_i16_load, emit_i16_store, emit_i16_binary
    use liric_session_control_bindings, only: create_liric_block, &
        emit_liric_br, &
        emit_liric_condbr, &
        emit_liric_f32_fcmp, &
        emit_liric_f64_fcmp, &
        emit_liric_i32_icmp, &
        emit_liric_i64_icmp, &
        emit_liric_i32_phi, &
        emit_liric_phi, &
        emit_liric_phi_n, &
        LR_FCMP_OGT, &
        LR_FCMP_OGE, &
        LR_FCMP_OLT, &
        LR_FCMP_OLE, &
        LR_FCMP_OEQ, &
        LR_FCMP_ONE, &
        LR_FCMP_UNO, &
        LR_CMP_SGE, &
        LR_CMP_SGT, &
        LR_CMP_SLE, &
        LR_CMP_SLT, &
        LR_CMP_NE, &
        LR_CMP_EQ, &
        set_liric_block
    use liric_session_format_bindings, only: LR_OP_FSUB, &
        prepare_liric_print_runtime, &
        create_printf_format_global, &
        printf_format_ptr, &
        create_type_info_global, create_pointer_table_global, &
        create_i64_table_global
    use liric_session_real_print_bindings, only: synthesize_real8_printer, &
        synthesize_real4_printer, &
        synthesize_get_arg_helper, &
        emit_get_arg_call, emit_snprintf, &
        emit_sscanf, emit_scanf, &
        emit_fscanf, emit_fscanf_count, &
        emit_fprintf, emit_dprintf, &
        emit_getchar, emit_exit
    use liric_session_complex_print_bindings, only: synthesize_complex4_printer, &
        synthesize_complex8_printer, &
        emit_complex4_print_call, &
        emit_complex8_print_call
    use liric_session_io_bindings, only: emit_liric_f32_binary, &
        emit_liric_i32_to_f32, &
        emit_liric_f32_to_i32, &
        emit_liric_f32_to_f64, &
        emit_liric_f64_to_f32, &
        emit_liric_print_f32, &
        emit_liric_print_f32_value, &
        emit_liric_f64_binary, &
        emit_liric_i32_to_f64, &
        emit_liric_i1_to_i32, &
        emit_liric_f64_to_i32, &
        emit_liric_f64_to_i64, &
        emit_liric_char_byte_zext, &
        emit_liric_i32_to_i64, &
        emit_liric_i32_to_i8, &
        emit_liric_i32_to_i16, &
        emit_liric_i64_to_i32, &
        emit_liric_store_char_byte, &
        emit_liric_print_f64, &
        emit_liric_print_f64_value, &
        emit_liric_print_i32, &
        emit_liric_print_i32_value, &
        emit_liric_print_i64, &
        emit_liric_print_i64_value, &
        emit_liric_print_newline, &
        emit_liric_print_space, &
        emit_liric_print_string_operand, &
        emit_liric_print_string_operand_value, &
        emit_liric_print_string, &
        emit_liric_print_string_value, &
        emit_liric_write_string_operand, &
        liric_f32_immediate, &
        liric_f64_immediate, &
        materialize_liric_string, &
        emit_liric_i8_to_i32, &
        emit_liric_i16_to_i32, &
        emit_liric_print_i8, &
        emit_liric_print_i8_value, &
        emit_liric_print_i16, &
        emit_liric_print_i16_value
    use liric_session_procedure_bindings, only: begin_liric_f32_function, &
        emit_liric_f32_alloca, &
        emit_liric_f32_call, &
        emit_liric_f32_load, &
        emit_liric_f32_store, &
        begin_liric_f64_function, &
        emit_liric_f64_alloca, &
        emit_liric_f64_call, &
        emit_liric_f64_load, &
        emit_liric_f64_store
    use liric_session_timing_bindings, only: emit_cpu_time_value, &
        emit_system_clock_value, emit_random_number_value, &
        emit_random_seed_size, emit_random_seed_put, emit_random_seed_get, &
        emit_random_seed_default
    use session_lowering_ops, only: integer_compare_predicate, &
        integer_opcode, parse_i32_literal
    use session_array_expr_types, only: array_expr_plan_t, &
                                        array_expr_plans_conform, &
                                        ARRAY_EXPR_MAX_RANK
    use ffc_strings, only: set_empty
    use ast_arena_source_text, only: get_source_text
    use fortfront_compiler, only: query_io_statement, io_statement_query_t, &
        IO_STATEMENT_FORMAT, IO_STATEMENT_WRITE, IO_STATEMENT_READ
    use ast_base, only: LITERAL_STRING
    use ffc_fortfront_queries, only: node_exists, get_node_type_at, &
        get_type_for_node, mono_type_t, &
        TINT, TREAL, TCHAR, TLOGICAL, TARRAY, TCOMPLEX, TDOUBLE, TDERIVED, &
        get_program_body_info, get_module_body_info, &
        get_function_body_info, get_subroutine_body_info, &
        get_select_case_info, get_case_block_info, &
        get_case_default_body, get_case_range_info, &
        get_select_type_info, get_type_guard_info, &
        is_derived_type_node, is_declaration_node, &
        get_derived_type_name, get_derived_type_components, &
        get_declaration_var_name, get_declaration_type_name, &
        get_declaration_has_initializer, &
        get_declaration_initializer_index, &
        get_node_stmt_label, get_goto_label, &
        goto_is_computed, get_goto_label_list, &
        get_goto_selector_index
    use fortfront_utils, only: get_node_as_function_def, &
        get_node_as_program, &
        get_node_as_subroutine_def, &
        get_parent
    use ast_nodes_data, only: mixed_construct_container_node, &
        multi_unit_container_node, submodule_node
    use fortfront, only: get_node_line, get_node_column
    use session_program_lowering_types, only: lowering_context_t, &
        branch_result_t, symbol_t, declaration_record_t, &
        array_section_info_t, &
        reduction_operand_t, &
        derived_type_info_t, &
        module_exports_t, &
        external_procedure_t, &
        lazy_specialization_t, &
        generic_interface_t, &
        operator_interface_t, &
        MAX_PROC_ARGS, &
        ARG_INTENT_NONE, ARG_INTENT_IN, ARG_INTENT_OUT, ARG_INTENT_INOUT, &
        MAX_GENERIC_SPECIFICS, &
        MODVAR_OK, MODVAR_UNSUPPORTED, &
        common_slot_t, COMMON_MAX_SLOTS, &
        EQUIV_MAX_MEMBERS, equiv_member_t, &
        ARRAY_MAX_RANK, &
        ALLOC_DESCRIPTOR_BYTES, &
        namelist_group_t, &
        statement_function_t, &
        MAX_STMT_FN_ARGS, &
        MAX_NAMELIST_MEMBERS, &
        NAMELIST_NAME_BUFFER, NAMELIST_VALUE_BUFFER, &
        NAMELIST_IOSTAT_END, NAMELIST_IOSTAT_BAD, &
        SCALAR_REAL_NONE, VALUE_I8, VALUE_I16, VALUE_I32, VALUE_I64, &
        VALUE_F32, VALUE_F64, &
        VALUE_C4, VALUE_C8, &
        VALUE_LOGICAL, VALUE_CHARACTER, &
        VALUE_DERIVED, &
        VALUE_DEFERRED_CHARACTER_RESULT, &
        VALUE_SUBROUTINE, VALUE_C_PTR, &
        VALUE_CLASS_STAR, VALUE_PROC_PTR, &
        VALUE_ARRAY_RESULT, &
        VALUE_ALLOC_ARRAY_RESULT, VALUE_DATA_PTR_RESULT, &
        TYPE_ID_INTEGER, TYPE_ID_REAL, &
        TYPE_ID_LOGICAL, &
        CMP_CLASS_UNKNOWN, CMP_CLASS_NUMERIC, &
        CMP_CLASS_CHAR, CMP_CLASS_LOGICAL, &
        CTOR_TYPE_UNKNOWN, CTOR_TYPE_INTEGER, &
        CTOR_TYPE_REAL, CTOR_TYPE_LOGICAL, CTOR_TYPE_CHAR, &
        I32_INTRINSIC_NONE, &
        I32_INTRINSIC_ABS, I32_INTRINSIC_MIN, &
        I32_INTRINSIC_MAX, I32_INTRINSIC_MOD, &
        I32_INTRINSIC_IAND, I32_INTRINSIC_IOR, &
        I32_INTRINSIC_IEOR, I32_INTRINSIC_NOT, &
        I32_INTRINSIC_ISHFT, &
        I32_INTRINSIC_ISHFTC, &
        I32_INTRINSIC_SIGN, &
        I32_INTRINSIC_INT, I32_INTRINSIC_NINT, &
        I32_INTRINSIC_FLOOR, &
        I32_INTRINSIC_CEILING, &
        I32_INTRINSIC_MATMUL, &
        I32_INTRINSIC_TRANSPOSE, &
        I32_INTRINSIC_DOT_PRODUCT, &
        I32_INTRINSIC_RESHAPE, &
        I32_INTRINSIC_SELECTED_INT_KIND, &
        I32_INTRINSIC_SELECTED_REAL_KIND, &
        I32_INTRINSIC_MODULO, &
        I32_INTRINSIC_DIM, &
        I32_INTRINSIC_IABS, &
        I32_INTRINSIC_IBITS, &
        I32_INTRINSIC_IBSET, &
        I32_INTRINSIC_IBCLR, &
        I32_INTRINSIC_BIT_SIZE, &
        F64_INTRINSIC_SIGN, &
        F64_INTRINSIC_SQRT, F64_INTRINSIC_EXP, &
        F64_INTRINSIC_LOG, F64_INTRINSIC_SIN, &
        F64_INTRINSIC_COS, F64_INTRINSIC_TAN, &
        F64_INTRINSIC_ATAN, F64_INTRINSIC_ATAN2, &
        F64_INTRINSIC_ASIN, F64_INTRINSIC_ACOS, &
        F64_INTRINSIC_SINH, F64_INTRINSIC_COSH, &
        F64_INTRINSIC_TANH, F64_INTRINSIC_ASINH, &
        F64_INTRINSIC_ACOSH, F64_INTRINSIC_ATANH, &
        F64_INTRINSIC_LOG10, F64_INTRINSIC_ERF, &
        F64_INTRINSIC_ERFC, F64_INTRINSIC_GAMMA, &
        F64_INTRINSIC_LOG_GAMMA, &
        F64_INTRINSIC_HYPOT, &
        F64_INTRINSIC_AINT, &
        F64_INTRINSIC_ANINT, &
        F64_INTRINSIC_NONE, F64_INTRINSIC_ABS, &
        F64_INTRINSIC_MIN, F64_INTRINSIC_MAX, &
        F64_INTRINSIC_REAL, I32_INTRINSIC_NAMES, &
        I32_INTRINSIC_IDS, F64_INTRINSIC_NAMES, &
        F64_INTRINSIC_IDS
    use ffc_module_artefact, only: module_info_t, fmod_parameter_t, &
        fmod_component_t, fmod_derived_type_t, &
        fmod_variable_t, fmod_procedure_t, fmod_generic_t, &
        write_fmod, read_fmod
    use session_program_lowering_fmod, only: integer_token, scalar_kind_token, &
        value_kind_of_token
    use session_program_lowering_scalar_kind, only: real_value_kind_of, &
        wider_real_kind, real_kind_from_kind_number
    use ffc_array_descriptor, only: ARRAY_DESCRIPTOR_BYTES, &
        ARRAY_DESCRIPTOR_ELEMENT_SIZE_OFFSET, &
        ARRAY_DESCRIPTOR_ELEMENT_TYPE_OFFSET, ARRAY_DESCRIPTOR_RANK_OFFSET, &
        ARRAY_DESCRIPTOR_FLAGS_OFFSET, ARRAY_DESCRIPTOR_RESERVED_OFFSET, &
        ARRAY_DESCRIPTOR_DIM_OFFSET, ARRAY_DIMENSION_BYTES, &
        ARRAY_DIMENSION_LOWER_OFFSET, ARRAY_DIMENSION_EXTENT_OFFSET, &
        ARRAY_DIMENSION_STRIDE_OFFSET, ARRAY_FLAG_ALLOCATED, &
        ARRAY_FLAG_ASSOCIATED, ARRAY_FLAG_CONTIGUOUS, ARRAY_FLAG_OWNS_DATA, &
        ARRAY_ELEMENT_INTEGER, &
        ARRAY_ELEMENT_REAL, ARRAY_ELEMENT_LOGICAL, ARRAY_ELEMENT_COMPLEX, &
        ARRAY_ELEMENT_CHARACTER, ARRAY_ELEMENT_DERIVED
    use ffc_character_descriptor, only: CHARACTER_STORAGE_STATIC, &
        CHARACTER_STORAGE_STACK, CHARACTER_STORAGE_OWNED
    implicit none
    private
    public :: lower_program_to_liric_exe
    public :: lower_program_to_liric_object

    ! Procedures shared with descendant implementation units. GCC 14 gives a
    ! private ancestor procedure local linkage even when a submodule calls it.
    ! Keep this implementation API explicit; session_program_lowering is the
    ! public facade and exports only the two compiler entry points above.
    public :: alloc_array_result_call_info, array_access_value_kind
    public :: bind_c_name, call_argument_kinds, call_argument_ranks
    public :: callee_dummy_is_array, callee_dummy_value_kind
    public :: char_expr_operands, collect_param_names
    public :: component_element_access_kind, component_slot_width
    public :: declaration_declares_name, declaration_index_for_name
    public :: declaration_is_assumed_rank, declaration_is_assumed_shape
    public :: declaration_named, declaration_value_kind
    public :: derived_component_access_kind, dim_is_assumed_shape
    public :: dim_is_assumed_size, dummy_explicit_element_count
    public :: emit_array_literal_print_items, emit_array_section_print_items
    public :: emit_io_implied_do_print_items, emit_whole_array_print_items
    public :: eval_i32_constant, external_procedure_index, f64_intrinsic_id
    public :: find_derived_type, find_module_in_arena, find_symbol_compat
    public :: flatten_constructor_elements, fold_scoped_i32_name
    public :: generic_call_return_kind, grow_symbols
    public :: interface_body_procedure_name, intrinsic_real_conversion_args
    public :: is_alloc_array_result_call, is_char_expr_call
    public :: is_character_concat, is_character_operand
    public :: is_character_substring, is_complex_component_extract
    public :: is_complex_valued, is_contained_deferred_char_function
    public :: is_contained_f32_function, is_contained_f64_function
    public :: is_contained_function_reference
    public :: is_declared_array_element_ref, is_equivalence_text
    public :: kind_of_literal, lower_f32_expression, lower_f64_expression
    public :: lower_i32_expression, lower_logical_expression
    public :: lower_print_expression_value, lower_print_logical_value
    public :: lowercase_text, module_procedure_mangled
    public :: module_symbol_is_private, param_at_is_character
    public :: param_at_value_kind, parameter_name, parse_equivalence_group
    public :: parse_i32_constant, procedure_has_nested_contains
    public :: reduction_arg_extent, reduction_expression_has_kind
    public :: reduction_expression_is_abs_call, resolve_symbol_at_node
    public :: same_name, split_csv, transfer_operand_kind
    public :: type_name_value_kind, use_only_wants, value_kind_number
    public :: alloc_desc_dim_offset, emit_alloc_desc_header
    public :: emit_alloc_desc_flags, emit_alloc_desc_set_dim
    public :: emit_alloc_desc_load_lower, emit_alloc_desc_load_extent
    public :: emit_alloc_desc_load_upper, emit_alloc_desc_allocate_shape
    public :: emit_alloc_desc_clear
    public :: allocatable_elem_size, assumed_shape_element_type
    public :: store_descriptor_i32
    public :: check_constant_initialization_exprs, check_scope_const_inits
    public :: check_async_specifiers, bare_name_const_reason
    public :: const_expr_reason, call_const_reason
    public :: identifier_const_reason, shape_inquiry_reason
    public :: arena_has_function_def_named
    ! Helpers referenced by descendants extracted after the initial GCC-14
    ! visibility pass above. Keep these explicit so gfortran-14 emits
    ! externally linkable definitions instead of local symbols.
    public :: assign_i32_to_symbol, char_length_operands, define_symbol
    public :: expression_value_kind, file_unit_pseudo_name, identifier_name
    public :: io_control_value, is_character_array_element
    public :: lower_i16_expression, lower_i64_expression
    public :: lower_i8_expression, ptr_plus_i32
    public :: reject_constant_substring_overrun, runtime_local_array_kind_ok

    ! Storage classes of the canonical character descriptor, widened to the
    ! i64 immediate width the lowering emits with. They are derived from
    ! ffc_character_descriptor so the emitted descriptors and the host-side
    ! descriptor helpers cannot drift apart.
    integer(c_int64_t), parameter :: LOWERING_CHARACTER_STORAGE_STATIC = &
        int(CHARACTER_STORAGE_STATIC, c_int64_t)
    integer(c_int64_t), parameter :: LOWERING_CHARACTER_STORAGE_STACK = &
        int(CHARACTER_STORAGE_STACK, c_int64_t)
    integer(c_int64_t), parameter :: LOWERING_CHARACTER_STORAGE_OWNED = &
        int(CHARACTER_STORAGE_OWNED, c_int64_t)

    ! Passed as emit_concat_copies' dest_len when the destination has no
    ! declared width to clamp to, which is every deferred-length destination.
    integer, parameter :: CONCAT_NO_CLAMP = -1

    ! Reduction kinds for dim-wise whole-array reductions (sum/product/count/
    ! any/all along one dimension). See lower_dim_reduction_assignment.
    ! An argument-less call still goes through prepare_reference_args so that
    ! callee diagnostics (a dummy procedure with no body in this unit) apply
    ! whether or not actual arguments are present (#576).
    integer, parameter :: NO_ACTUAL_ARGS(0) = [integer ::]

    integer, parameter :: DIM_REDUCE_SUM = 1
    integer, parameter :: DIM_REDUCE_PROD = 2
    integer, parameter :: DIM_REDUCE_COUNT = 3
    integer, parameter :: DIM_REDUCE_ANY = 4
    integer, parameter :: DIM_REDUCE_ALL = 5
    integer, parameter :: DIM_REDUCE_NORM2 = 6
    integer, parameter :: DIM_REDUCE_MIN = 7
    integer, parameter :: DIM_REDUCE_MAX = 8

    ! Pull-based cursor over a DATA value list. A scalar value yields once; an
    ! implied-do value (e.g. (i*1.0, i=1,2)) unrolls its inner expression,
    ! binding the control variable before each is handed to the consumer.
    type :: data_value_cursor_t
        integer :: vpos = 0 ! index into value_indices
        integer :: count = 0 ! values remaining in active implied-do
        integer :: var_sym = 0 ! control symbol of active implied-do
        integer :: cur = 0 ! current control value
        integer :: step = 1
        integer :: inner = 0 ! single inner value expression index
    end type data_value_cursor_t

    ! Body emitter for the reusable counted-loop scaffold. lower_counted_loop
    ! owns the header/body/latch/exit blocks and the induction phi; the emitter
    ! fills the body block. FORALL passes a recursive emitter (inner loops plus
    ! the optionally masked assignment); a plain DO lowers its own statements.
    abstract interface
        subroutine counted_loop_body_i(context, terminated, error_msg)
            import :: lowering_context_t
            type(lowering_context_t), intent(inout) :: context
            logical, intent(out) :: terminated
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine counted_loop_body_i
    end interface
    interface
        module subroutine lower_print(arena, node, context, error_msg)
            type(ast_arena_t), intent(in) :: arena
            type(print_statement_node), intent(in) :: node
            type(lowering_context_t), intent(inout) :: context
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine lower_print
        module subroutine lower_formatted_print(arena, node, context, error_msg)
            type(ast_arena_t), intent(in) :: arena
            type(print_statement_node), intent(in) :: node
            type(lowering_context_t), intent(inout) :: context
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine lower_formatted_print
        module subroutine parse_single_edit_descriptor(spec, kind_char, printf_fmt, &
                error_msg)
            character(len=*), intent(in) :: spec
            character, intent(out) :: kind_char
            character(len=:), allocatable, intent(out) :: printf_fmt
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine parse_single_edit_descriptor
        module subroutine lower_compound_formatted_print(arena, node, context, &
                format_body, error_msg)
            type(ast_arena_t), intent(in) :: arena
            type(print_statement_node), intent(in) :: node
            type(lowering_context_t), intent(inout) :: context
            character(len=*), intent(in) :: format_body
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine lower_compound_formatted_print
        recursive module subroutine lower_next_compound_descriptor(arena, node, &
                context, format_body, pos, item_index, exhausted, error_msg)
            type(ast_arena_t), intent(in) :: arena
            type(print_statement_node), intent(in) :: node
            type(lowering_context_t), intent(inout) :: context
            character(len=*), intent(in) :: format_body
            integer, intent(inout) :: pos
            integer, intent(inout) :: item_index
            logical, intent(out) :: exhausted
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine lower_next_compound_descriptor
        recursive module subroutine lower_compound_group(arena, node, context, &
                format_body, pos, repeat_count, item_index, exhausted, error_msg)
            type(ast_arena_t), intent(in) :: arena
            type(print_statement_node), intent(in) :: node
            type(lowering_context_t), intent(inout) :: context
            character(len=*), intent(in) :: format_body
            integer, intent(inout) :: pos
            integer, intent(in) :: repeat_count
            integer, intent(inout) :: item_index
            logical, intent(out) :: exhausted
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine lower_compound_group
        module subroutine find_group_close(text, open_pos, close_pos, error_msg)
            character(len=*), intent(in) :: text
            integer, intent(in) :: open_pos
            integer, intent(out) :: close_pos
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine find_group_close
        module subroutine skip_dot_modifier(format_body, pos)
            character(len=*), intent(in) :: format_body
            integer, intent(inout) :: pos
        end subroutine skip_dot_modifier
        module subroutine repeat_data_descriptor(arena, node, context, kind_char, &
                printf_fmt, buffer_size, repeat_count, item_index, exhausted, &
                error_msg)
            type(ast_arena_t), intent(in) :: arena
            type(print_statement_node), intent(in) :: node
            type(lowering_context_t), intent(inout) :: context
            character, intent(in) :: kind_char
            character(len=*), intent(in) :: printf_fmt
            integer, intent(in) :: buffer_size
            integer, intent(in) :: repeat_count
            integer, intent(inout) :: item_index
            logical, intent(out) :: exhausted
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine repeat_data_descriptor
        module subroutine lower_format_string_literal(context, format_body, pos, &
                error_msg)
            type(lowering_context_t), intent(inout) :: context
            character(len=*), intent(in) :: format_body
            integer, intent(inout) :: pos
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine lower_format_string_literal
        module function ffc_unit_global_name(context, kind_tag, counter) result(name)
            type(lowering_context_t), intent(in) :: context
            character(len=*), intent(in) :: kind_tag
            integer, intent(in) :: counter
            character(len=:), allocatable :: name
        end function ffc_unit_global_name
        module subroutine lower_compound_logical_descriptor(arena, node, context, &
                width, item_index, exhausted, error_msg)
            type(ast_arena_t), intent(in) :: arena
            type(print_statement_node), intent(in) :: node
            type(lowering_context_t), intent(inout) :: context
            integer, intent(in) :: width
            integer, intent(inout) :: item_index
            logical, intent(out) :: exhausted
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine lower_compound_logical_descriptor
        module subroutine lower_compound_data_descriptor(arena, node, context, &
                kind_char, printf_fmt, buffer_size, item_index, exhausted, &
                error_msg)
            type(ast_arena_t), intent(in) :: arena
            type(print_statement_node), intent(in) :: node
            type(lowering_context_t), intent(inout) :: context
            character, intent(in) :: kind_char
            character(len=*), intent(in) :: printf_fmt
            integer, intent(in) :: buffer_size
            integer, intent(inout) :: item_index
            logical, intent(out) :: exhausted
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine lower_compound_data_descriptor
        module subroutine repeat_e_en_descriptor(arena, node, context, mode, width, &
                precision, repeat_count, item_index, exhausted, error_msg)
            type(ast_arena_t), intent(in) :: arena
            type(print_statement_node), intent(in) :: node
            type(lowering_context_t), intent(inout) :: context
            integer, intent(in) :: mode, width, precision, repeat_count
            integer, intent(inout) :: item_index
            logical, intent(out) :: exhausted
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine repeat_e_en_descriptor
        module subroutine lower_formatted_real_item(arena, node_index, context, &
                fmt_id, buffer_size, error_msg)
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: node_index
            type(lowering_context_t), intent(inout) :: context
            integer(c_int32_t), intent(in) :: fmt_id
            integer, intent(in) :: buffer_size
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine lower_formatted_real_item
        module subroutine lower_formatted_e_en_real_item(arena, node_index, context, &
                mode, width, precision, error_msg)
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: node_index, mode, width, precision
            type(lowering_context_t), intent(inout) :: context
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine lower_formatted_e_en_real_item
        module subroutine read_decimal_value(digits, value, error_msg)
            character(len=*), intent(in) :: digits
            integer, intent(out) :: value
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine read_decimal_value
        module subroutine parse_decimal_digits(text, pos, digits)
            character(len=*), intent(in) :: text
            integer, intent(inout) :: pos
            character(len=:), allocatable, intent(out) :: digits
        end subroutine parse_decimal_digits
        module function is_decimal_digit(ch)
            logical :: is_decimal_digit
            character, intent(in) :: ch
        end function is_decimal_digit
        module subroutine skip_format_separators(text, pos)
            character(len=*), intent(in) :: text
            integer, intent(inout) :: pos
        end subroutine skip_format_separators
        module subroutine normalize_format_body(spec, body)
            character(len=*), intent(in) :: spec
            character(len=:), allocatable, intent(out) :: body
        end subroutine normalize_format_body
        module subroutine collapse_doubled_quote(text, quote)
            character(len=:), allocatable, intent(inout) :: text
            character, intent(in) :: quote
        end subroutine collapse_doubled_quote
        module subroutine lower_formatted_int_item(arena, node_index, context, &
                fmt_id, error_msg)
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: node_index
            type(lowering_context_t), intent(inout) :: context
            integer(c_int32_t), intent(in) :: fmt_id
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine lower_formatted_int_item
        module subroutine lower_formatted_char_item(arena, node_index, context, &
                fmt_id, error_msg)
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: node_index
            type(lowering_context_t), intent(inout) :: context
            integer(c_int32_t), intent(in) :: fmt_id
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine lower_formatted_char_item
        module function char_print_item(arena, node_index, context) result( &
                is_char)
            logical :: is_char
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: node_index
            type(lowering_context_t), intent(in) :: context
        end function char_print_item
    end interface
    interface
        module function iso_c_binding_kind_value(name, value) result(found)
            logical :: found
            character(len=*), intent(in) :: name
            integer(c_int64_t), intent(out) :: value
        end function iso_c_binding_kind_value
        module function iso_fortran_env_kind_value(name, value) result(found)
            logical :: found
            character(len=*), intent(in) :: name
            integer(c_int64_t), intent(out) :: value
        end function iso_fortran_env_kind_value
        module subroutine integer_literal_kind_number(text, k, error_msg)
            character(len=*), intent(in) :: text
            integer(c_int64_t), intent(out) :: k
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine integer_literal_kind_number
        module subroutine eval_min_max_constant(arena, node, context, want_max, &
                                                constant_value, error_msg)
            type(ast_arena_t), intent(in) :: arena
            type(call_or_subscript_node), intent(in) :: node
            type(lowering_context_t), intent(in) :: context
            logical, intent(in) :: want_max
            integer(c_int64_t), intent(out) :: constant_value
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine eval_min_max_constant
        module subroutine eval_int_constant(arena, node, context, constant_value, &
                                            error_msg)
            type(ast_arena_t), intent(in) :: arena
            type(call_or_subscript_node), intent(in) :: node
            type(lowering_context_t), intent(in) :: context
            integer(c_int64_t), intent(out) :: constant_value
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine eval_int_constant
        module subroutine constant_arg_kind_number(arena, context, arg_index, k, &
                                                   error_msg)
            type(ast_arena_t), intent(in) :: arena
            type(lowering_context_t), intent(in) :: context
            integer, intent(in) :: arg_index
            integer(c_int64_t), intent(out) :: k
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine constant_arg_kind_number
        module subroutine eval_huge_constant(arena, node, context, constant_value, &
                                             error_msg)
            type(ast_arena_t), intent(in) :: arena
            type(call_or_subscript_node), intent(in) :: node
            type(lowering_context_t), intent(in) :: context
            integer(c_int64_t), intent(out) :: constant_value
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine eval_huge_constant
        module subroutine eval_precision_constant(arena, node, context, &
                                                   constant_value, error_msg)
            type(ast_arena_t), intent(in) :: arena
            type(call_or_subscript_node), intent(in) :: node
            type(lowering_context_t), intent(in) :: context
            integer(c_int64_t), intent(out) :: constant_value
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine eval_precision_constant
        module subroutine eval_bit_size_constant(arena, node, context, &
                                                  constant_value, error_msg)
            type(ast_arena_t), intent(in) :: arena
            type(call_or_subscript_node), intent(in) :: node
            type(lowering_context_t), intent(in) :: context
            integer(c_int64_t), intent(out) :: constant_value
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine eval_bit_size_constant
        module subroutine eval_range_constant(arena, node, context, constant_value, &
                                              error_msg)
            type(ast_arena_t), intent(in) :: arena
            type(call_or_subscript_node), intent(in) :: node
            type(lowering_context_t), intent(in) :: context
            integer(c_int64_t), intent(out) :: constant_value
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine eval_range_constant
        module subroutine eval_selected_char_kind_constant(arena, node, &
                                                           constant_value, error_msg)
            type(ast_arena_t), intent(in) :: arena
            type(call_or_subscript_node), intent(in) :: node
            integer(c_int64_t), intent(out) :: constant_value
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine eval_selected_char_kind_constant
        module subroutine eval_one_arg_i32_intrinsic(arena, node, context, name, &
                                                     constant_value, error_msg)
            type(ast_arena_t), intent(in) :: arena
            type(call_or_subscript_node), intent(in) :: node
            type(lowering_context_t), intent(in) :: context
            character(len=*), intent(in) :: name
            integer(c_int64_t), intent(out) :: constant_value
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine eval_one_arg_i32_intrinsic
        module subroutine eval_two_arg_i32_intrinsic(arena, node, context, name, &
                                                     constant_value, error_msg)
            type(ast_arena_t), intent(in) :: arena
            type(call_or_subscript_node), intent(in) :: node
            type(lowering_context_t), intent(in) :: context
            character(len=*), intent(in) :: name
            integer(c_int64_t), intent(out) :: constant_value
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine eval_two_arg_i32_intrinsic
        module subroutine eval_ibits_constant(arena, node, context, constant_value, &
                                              error_msg)
            type(ast_arena_t), intent(in) :: arena
            type(call_or_subscript_node), intent(in) :: node
            type(lowering_context_t), intent(in) :: context
            integer(c_int64_t), intent(out) :: constant_value
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine eval_ibits_constant
        module subroutine eval_merge_constant(arena, node, context, constant_value, &
                                              error_msg)
            type(ast_arena_t), intent(in) :: arena
            type(call_or_subscript_node), intent(in) :: node
            type(lowering_context_t), intent(in) :: context
            integer(c_int64_t), intent(out) :: constant_value
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine eval_merge_constant
        module subroutine eval_digits_constant(arena, node, context, constant_value, &
                                               error_msg)
            type(ast_arena_t), intent(in) :: arena
            type(call_or_subscript_node), intent(in) :: node
            type(lowering_context_t), intent(in) :: context
            integer(c_int64_t), intent(out) :: constant_value
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine eval_digits_constant
        module subroutine eval_radix_constant(arena, node, constant_value, error_msg)
            type(ast_arena_t), intent(in) :: arena
            type(call_or_subscript_node), intent(in) :: node
            integer(c_int64_t), intent(out) :: constant_value
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine eval_radix_constant
        module subroutine eval_exponent_range_constant(arena, node, context, &
                                                       want_max, constant_value, &
                                                       error_msg)
            type(ast_arena_t), intent(in) :: arena
            type(call_or_subscript_node), intent(in) :: node
            type(lowering_context_t), intent(in) :: context
            logical, intent(in) :: want_max
            integer(c_int64_t), intent(out) :: constant_value
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine eval_exponent_range_constant
        module subroutine eval_selected_logical_kind_constant(arena, node, context, &
                                                              constant_value, &
                                                              error_msg)
            type(ast_arena_t), intent(in) :: arena
            type(call_or_subscript_node), intent(in) :: node
            type(lowering_context_t), intent(in) :: context
            integer(c_int64_t), intent(out) :: constant_value
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine eval_selected_logical_kind_constant
        module function find_integer_parameter_array_decl(arena, name) result(decl)
            type(ast_arena_t), intent(in) :: arena
            character(len=*), intent(in) :: name
            type(declaration_node), pointer :: decl
        end function find_integer_parameter_array_decl
        recursive module subroutine fold_i32_array_literal(arena, context, &
                                                           node_index, values, &
                                                           error_msg)
            type(ast_arena_t), intent(in) :: arena
            type(lowering_context_t), intent(in) :: context
            integer, intent(in) :: node_index
            integer(c_int64_t), allocatable, intent(out) :: values(:)
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine fold_i32_array_literal
        recursive module subroutine fold_i32_array_literal_by_name(arena, context, &
                                                                   name, values, &
                                                                   error_msg)
            type(ast_arena_t), intent(in) :: arena
            type(lowering_context_t), intent(in) :: context
            character(len=*), intent(in) :: name
            integer(c_int64_t), allocatable, intent(out) :: values(:)
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine fold_i32_array_literal_by_name
        module subroutine fold_param_array_element_from_arena(arena, node, context, &
                                                              constant_value, &
                                                              error_msg)
            type(ast_arena_t), intent(in) :: arena
            type(call_or_subscript_node), intent(in) :: node
            type(lowering_context_t), intent(in) :: context
            integer(c_int64_t), intent(out) :: constant_value
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine fold_param_array_element_from_arena
        module subroutine eval_product_constant(arena, node, context, &
                                                constant_value, error_msg)
            type(ast_arena_t), intent(in) :: arena
            type(call_or_subscript_node), intent(in) :: node
            type(lowering_context_t), intent(in) :: context
            integer(c_int64_t), intent(out) :: constant_value
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine eval_product_constant
        module subroutine eval_array_reduction_constant(arena, node, context, name, &
                                                        constant_value, error_msg)
            type(ast_arena_t), intent(in) :: arena
            type(call_or_subscript_node), intent(in) :: node
            type(lowering_context_t), intent(in) :: context
            character(len=*), intent(in) :: name
            integer(c_int64_t), intent(out) :: constant_value
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine eval_array_reduction_constant
        module subroutine eval_dot_product_constant(arena, node, context, &
                                                    constant_value, error_msg)
            type(ast_arena_t), intent(in) :: arena
            type(call_or_subscript_node), intent(in) :: node
            type(lowering_context_t), intent(in) :: context
            integer(c_int64_t), intent(out) :: constant_value
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine eval_dot_product_constant
        module function character_length_literal_reason(expr) result(reason)
            character(len=*), intent(in) :: expr
            character(len=:), allocatable :: reason
        end function character_length_literal_reason
        module function strip_outer_parentheses(expr) result(text)
            character(len=*), intent(in) :: expr
            character(len=:), allocatable :: text
        end function strip_outer_parentheses
        recursive module function scalar_real_expr_kind(arena, node_index, context) &
            result(vk)
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: node_index
            type(lowering_context_t), intent(in) :: context
            integer :: vk
        end function scalar_real_expr_kind
        recursive module function scalar_real_call_kind(arena, node, context) &
            result(vk)
            type(ast_arena_t), intent(in) :: arena
            type(call_or_subscript_node), intent(in) :: node
            type(lowering_context_t), intent(in) :: context
            integer :: vk
        end function scalar_real_call_kind
        recursive module function scalar_real_intrinsic_kind(arena, node, context) &
            result(vk)
            type(ast_arena_t), intent(in) :: arena
            type(call_or_subscript_node), intent(in) :: node
            type(lowering_context_t), intent(in) :: context
            integer :: vk
        end function scalar_real_intrinsic_kind
        module function real_intrinsic_is_f64(arena, node, context) result(is_f64)
            logical :: is_f64
            type(ast_arena_t), intent(in) :: arena
            type(call_or_subscript_node), intent(in) :: node
            type(lowering_context_t), intent(in) :: context
        end function real_intrinsic_is_f64
        module function real_conversion_intrinsic(name) result(is_conversion)
            logical :: is_conversion
            character(len=*), intent(in) :: name
        end function real_conversion_intrinsic
        module function real_conversion_kind_is_f64(arena, kind_index) result(is_f64)
            logical :: is_f64
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: kind_index
        end function real_conversion_kind_is_f64
        module function is_real_array_reduction(arena, node, context, vk) result(ok)
            logical :: ok
            type(ast_arena_t), intent(in) :: arena
            type(call_or_subscript_node), intent(in) :: node
            type(lowering_context_t), intent(in) :: context
            integer, intent(in) :: vk
        end function is_real_array_reduction
        module function is_real_inquiry_intrinsic(name) result(ok)
            logical :: ok
            character(len=*), intent(in) :: name
        end function is_real_inquiry_intrinsic
        module function inquiry_arg_real_kind(arena, node, context) result(kind_num)
            integer :: kind_num
            type(ast_arena_t), intent(in) :: arena
            type(call_or_subscript_node), intent(in) :: node
            type(lowering_context_t), intent(in) :: context
        end function inquiry_arg_real_kind
        module function is_real_dot_product(arena, node, context, vk) result(ok)
            logical :: ok
            type(ast_arena_t), intent(in) :: arena
            type(call_or_subscript_node), intent(in) :: node
            type(lowering_context_t), intent(in) :: context
            integer, intent(in) :: vk
        end function is_real_dot_product
        module subroutine real_opcode(source_op, line, column, opcode, error_msg)
            character(len=*), intent(in) :: source_op
            integer, intent(in) :: line, column
            integer, intent(out) :: opcode
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine real_opcode
        module function is_real_literal(arena, node_index)
            logical :: is_real_literal
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: node_index
        end function is_real_literal
        module function is_boz_literal_text(text) result(is_boz)
            logical :: is_boz
            character(len=*), intent(in) :: text
        end function is_boz_literal_text
        module function node_is_boz_literal(arena, node_index) result(is_boz)
            logical :: is_boz
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: node_index
        end function node_is_boz_literal
        module function is_boz_designator(c)
            logical :: is_boz_designator
            character(len=1), intent(in) :: c
        end function is_boz_designator
        module function boz_bits_i32(v) result(bits)
            integer(c_int32_t) :: bits
            integer(c_int64_t), intent(in) :: v
        end function boz_bits_i32
        module subroutine lower_boz_real_bits(arena, node_index, context, want_f64, &
                value, handled, error_msg)
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: node_index
            type(lowering_context_t), intent(in) :: context
            logical, intent(in) :: want_f64
            type(lr_operand_desc_t), intent(out) :: value
            logical, intent(out) :: handled
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine lower_boz_real_bits
        module function is_character_literal(arena, node_index)
            logical :: is_character_literal
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: node_index
        end function is_character_literal
        module function is_logical_literal(arena, node_index)
            logical :: is_logical_literal
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: node_index
        end function is_logical_literal
        module function starts_with_quote(text)
            logical :: starts_with_quote
            character(len=*), intent(in) :: text
        end function starts_with_quote
        module subroutine strip_literal_quotes(text, value)
            character(len=*), intent(in) :: text
            character(len=:), allocatable, intent(out) :: value
        end subroutine strip_literal_quotes
        module function logical_i32_value(text) result(value)
            integer(c_int64_t) :: value
            character(len=*), intent(in) :: text
        end function logical_i32_value
        recursive module function literal_is_f64(text, context, reference_index)
            logical :: literal_is_f64
            character(len=*), intent(in) :: text
            type(lowering_context_t), intent(in) :: context
            integer, intent(in), optional :: reference_index
        end function literal_is_f64
        module function named_kind_suffix_is_f64(suffix, context, reference_index) &
                result(is_f64)
            logical :: is_f64
            character(len=*), intent(in) :: suffix
            type(lowering_context_t), intent(in) :: context
            integer, intent(in), optional :: reference_index
        end function named_kind_suffix_is_f64
        module subroutine parse_f64_literal(text, context, value, error_msg, &
                reference_index)
            character(len=*), intent(in) :: text
            type(lowering_context_t), intent(in) :: context
            real(c_double), intent(out) :: value
            character(len=:), allocatable, intent(out) :: error_msg
            integer, intent(in), optional :: reference_index
        end subroutine parse_f64_literal
        module subroutine emit_module_fmod_artifacts(arena, context, output_path, &
                error_msg)
            type(ast_arena_t), intent(in) :: arena
            type(lowering_context_t), intent(in) :: context
            character(len=*), intent(in) :: output_path
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine emit_module_fmod_artifacts
        module function path_dirname(path) result(dir)
            character(len=*), intent(in) :: path
            character(len=:), allocatable :: dir
        end function path_dirname
        module subroutine build_module_info(arena, context, export, info, error_msg)
            type(ast_arena_t), intent(in) :: arena
            type(lowering_context_t), intent(in) :: context
            type(module_exports_t), intent(in) :: export
            type(module_info_t), intent(out) :: info
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine build_module_info
        module subroutine build_fmod_generics(arena, module_name, procs, generics, &
                error_msg)
            type(ast_arena_t), intent(in) :: arena
            character(len=*), intent(in) :: module_name
            type(fmod_procedure_t), allocatable, intent(in) :: procs(:)
            type(fmod_generic_t), allocatable, intent(out) :: generics(:)
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine build_fmod_generics
        module function generic_block_name(arena, node_index) result(name)
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: node_index
            character(len=:), allocatable :: name
        end function generic_block_name
        module subroutine fmod_generic_specifics(arena, node_index, procs, &
                module_name, specifics)
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: node_index
            type(fmod_procedure_t), allocatable, intent(in) :: procs(:)
            character(len=*), intent(in) :: module_name
            character(len=:), allocatable, intent(out) :: specifics
        end subroutine fmod_generic_specifics
        module subroutine append_generic_specific(procs, name, specifics)
            type(fmod_procedure_t), allocatable, intent(in) :: procs(:)
            character(len=*), intent(in) :: name
            character(len=:), allocatable, intent(inout) :: specifics
        end subroutine append_generic_specific
        module subroutine grow_fmod_generics(arr, n)
            type(fmod_generic_t), allocatable, intent(inout) :: arr(:)
            integer, intent(in) :: n
        end subroutine grow_fmod_generics
        module subroutine build_fmod_procedures(arena, context, module_name, procs, &
                error_msg)
            type(ast_arena_t), intent(in) :: arena
            type(lowering_context_t), intent(in) :: context
            character(len=*), intent(in) :: module_name
            type(fmod_procedure_t), allocatable, intent(out) :: procs(:)
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine build_fmod_procedures
        module subroutine record_fmod_interface_procedures(arena, context, &
                node_index, mod_node, procs, count)
            type(ast_arena_t), intent(in) :: arena
            type(lowering_context_t), intent(in) :: context
            integer, intent(in) :: node_index
            type(module_node), intent(in) :: mod_node
            type(fmod_procedure_t), allocatable, intent(inout) :: procs(:)
            integer, intent(inout) :: count
        end subroutine record_fmod_interface_procedures
        module function procedure_is_deferred_module_body(arena, node_index) &
                result(is_module_body)
            logical :: is_module_body
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: node_index
        end function procedure_is_deferred_module_body
        module function prefix_has_module(prefix_keywords) result(has_module)
            logical :: has_module
            character(len=16), allocatable, intent(in) :: prefix_keywords(:)
        end function prefix_has_module
        module subroutine record_fmod_procedure(arena, context, node_index, mod_node, &
                deferred_body, procs, count, external_binding)
            type(ast_arena_t), intent(in) :: arena
            type(lowering_context_t), intent(in) :: context
            integer, intent(in) :: node_index
            type(module_node), intent(in) :: mod_node
            logical, intent(in) :: deferred_body
            type(fmod_procedure_t), allocatable, intent(inout) :: procs(:)
            integer, intent(inout) :: count
            logical, intent(in), optional :: external_binding
        end subroutine record_fmod_procedure
        module subroutine fmod_procedure_result(arena, node_index, kind_text, &
                result_name, result_kind)
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: node_index
            character(len=*), intent(in) :: kind_text
            character(len=:), allocatable, intent(out) :: result_name
            character(len=:), allocatable, intent(out) :: result_kind
        end subroutine fmod_procedure_result
        module subroutine fmod_procedure_dummy_attributes(arena, node_index, &
                intents, optionals, values)
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: node_index
            character(len=:), allocatable, intent(out) :: intents
            character(len=:), allocatable, intent(out) :: optionals
            character(len=:), allocatable, intent(out) :: values
        end subroutine fmod_procedure_dummy_attributes
        module function flag_token(flag) result(token)
            logical, intent(in) :: flag
            character(len=:), allocatable :: token
        end function flag_token
        module subroutine param_at_attributes(arena, param_indices, body_indices, &
                pos, intent_token, is_optional, is_value)
            type(ast_arena_t), intent(in) :: arena
            integer, allocatable, intent(in) :: param_indices(:)
            integer, allocatable, intent(in) :: body_indices(:)
            integer, intent(in) :: pos
            character(len=:), allocatable, intent(out) :: intent_token
            logical, intent(out) :: is_optional
            logical, intent(out) :: is_value
        end subroutine param_at_attributes
        module function fmod_procedure_arg_names(arena, node_index) result(tokens)
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: node_index
            character(len=:), allocatable :: tokens
        end function fmod_procedure_arg_names
        module subroutine grow_fmod_procs(arr, n)
            type(fmod_procedure_t), allocatable, intent(inout) :: arr(:)
            integer, intent(in) :: n
        end subroutine grow_fmod_procs
        module function get_module_node_ptr(arena, module_index) result(mod_node)
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: module_index
            type(module_node), pointer :: mod_node
        end function get_module_node_ptr
        module function procedure_fortran_name(arena, node_index) result(name)
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: node_index
            character(len=:), allocatable :: name
        end function procedure_fortran_name
        module function fmod_procedure_external_name(arena, node_index) result(name)
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: node_index
            character(len=:), allocatable :: name
        end function fmod_procedure_external_name
        module subroutine fmod_procedure_signature(arena, context, node_index, &
                mod_node, kind_text, nargs, arg_tokens, rank_tokens, extent_tokens, &
                allow_runtime_array)
            type(ast_arena_t), intent(in) :: arena
            type(lowering_context_t), intent(in) :: context
            integer, intent(in) :: node_index
            type(module_node), intent(in), optional :: mod_node
            character(len=:), allocatable, intent(out) :: kind_text
            integer, intent(out) :: nargs
            character(len=:), allocatable, intent(out) :: arg_tokens
            character(len=:), allocatable, intent(out) :: rank_tokens
            character(len=:), allocatable, intent(out) :: extent_tokens
            logical, intent(in), optional :: allow_runtime_array
        end subroutine fmod_procedure_signature
        module function fmod_function_result_value_kind(arena, fn_node) &
                result(value_kind)
            integer :: value_kind
            type(ast_arena_t), intent(in) :: arena
            type(function_def_node), intent(in) :: fn_node
        end function fmod_function_result_value_kind
        module function params_all_supported(arena, context, param_indices, &
                body_indices, nargs, arg_tokens, rank_tokens, extent_tokens, &
                allow_runtime_array) result(ok)
            logical :: ok
            type(ast_arena_t), intent(in) :: arena
            type(lowering_context_t), intent(in) :: context
            integer, allocatable, intent(in) :: param_indices(:)
            integer, allocatable, intent(in) :: body_indices(:)
            integer, intent(out) :: nargs
            character(len=:), allocatable, intent(out) :: arg_tokens
            character(len=:), allocatable, intent(out) :: rank_tokens
            character(len=:), allocatable, intent(out) :: extent_tokens
            logical, intent(in), optional :: allow_runtime_array
        end function params_all_supported
        module subroutine param_at_array_shape(arena, context, param_indices, &
                body_indices, pos, value_kind, rank, extent, allow_runtime_extent)
            type(ast_arena_t), intent(in) :: arena
            type(lowering_context_t), intent(in) :: context
            integer, allocatable, intent(in) :: param_indices(:)
            integer, allocatable, intent(in) :: body_indices(:)
            integer, intent(in) :: pos
            integer, intent(out) :: value_kind
            integer, intent(out) :: rank
            integer, intent(out) :: extent
            logical, intent(in), optional :: allow_runtime_extent
        end subroutine param_at_array_shape
        module subroutine build_fmod_variable(arena, node_index, var, error_msg)
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: node_index
            type(fmod_variable_t), intent(out) :: var
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine build_fmod_variable
        module function fmod_variable_kind_token(type_name) result(token)
            character(len=*), intent(in) :: type_name
            character(len=:), allocatable :: token
        end function fmod_variable_kind_token
        module subroutine build_fmod_parameter(arena, node_index, param, error_msg)
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: node_index
            type(fmod_parameter_t), intent(out) :: param
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine build_fmod_parameter
        module function fmod_kind_string(type_name) result(kind_text)
            character(len=*), intent(in) :: type_name
            character(len=:), allocatable :: kind_text
        end function fmod_kind_string
        module subroutine build_fmod_derived_type(arena, context, node_index, dtype, &
                error_msg)
            type(ast_arena_t), intent(in) :: arena
            type(lowering_context_t), intent(in) :: context
            integer, intent(in) :: node_index
            type(fmod_derived_type_t), intent(out) :: dtype
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine build_fmod_derived_type
        module function module_reexports_type(arena, module_index, type_name) &
                result(reexports)
            logical :: reexports
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: module_index
            character(len=*), intent(in) :: type_name
        end function module_reexports_type
        module subroutine build_fmod_derived_type_from_context(context, type_index, &
                dtype, error_msg)
            type(lowering_context_t), intent(in) :: context
            integer, intent(in) :: type_index
            type(fmod_derived_type_t), intent(out) :: dtype
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine build_fmod_derived_type_from_context
        module function fmod_component_kind_token(context, type_index, comp_index) &
                result(token)
            type(lowering_context_t), intent(in) :: context
            integer, intent(in) :: type_index
            integer, intent(in) :: comp_index
            character(len=:), allocatable :: token
        end function fmod_component_kind_token
        module function fmod_component_value_kind(token) result(value_kind)
            integer :: value_kind
            character(len=*), intent(in) :: token
        end function fmod_component_value_kind
    end interface
    interface
        module subroutine check_generic_ambiguity(arena, context, generic_idx, error_msg)
            type(ast_arena_t), intent(in) :: arena
            type(lowering_context_t), intent(in) :: context
            integer, intent(in) :: generic_idx
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_generic_ambiguity
        module function specifics_indistinguishable(arena, name_a, name_b) result(same)
            logical :: same
            type(ast_arena_t), intent(in) :: arena
            character(len=*), intent(in) :: name_a, name_b
        end function specifics_indistinguishable
        module subroutine dummy_signature(arena, proc_name, pos, known, base_name, kind_value, rank, is_proc, is_any)
            type(ast_arena_t), intent(in) :: arena
            character(len=*), intent(in) :: proc_name
            integer, intent(in) :: pos
            logical, intent(out) :: known
            character(len=:), allocatable, intent(out) :: base_name
            integer, intent(out) :: kind_value
            integer, intent(out) :: rank
            logical, intent(out) :: is_proc
            logical, intent(out) :: is_any
        end subroutine dummy_signature
        module subroutine refine_dummy_signature(arena, proc_name, pos, kind_value, rank)
            type(ast_arena_t), intent(in) :: arena
            character(len=*), intent(in) :: proc_name
            integer, intent(in) :: pos
            integer, intent(inout) :: kind_value
            integer, intent(inout) :: rank
        end subroutine refine_dummy_signature
        module subroutine dummy_signature_at(arena, param_indices, body_indices, pos, known, base_name, kind_value, rank, is_proc, is_any)
            type(ast_arena_t), intent(in) :: arena
            integer, allocatable, intent(in) :: param_indices(:)
            integer, allocatable, intent(in) :: body_indices(:)
            integer, intent(in) :: pos
            logical, intent(out) :: known
            character(len=:), allocatable, intent(out) :: base_name
            integer, intent(out) :: kind_value
            integer, intent(out) :: rank
            logical, intent(out) :: is_proc
            logical, intent(out) :: is_any
        end subroutine dummy_signature_at
        module function dummy_declared_rank(arena, body_indices, param_name) result(rank)
            integer :: rank
            type(ast_arena_t), intent(in) :: arena
            integer, allocatable, intent(in) :: body_indices(:)
            character(len=*), intent(in) :: param_name
        end function dummy_declared_rank
        module subroutine dummy_decl_signature(arena, body_indices, param_name, found, base_name, kind_value, rank, is_proc, is_any, unresolved)
            type(ast_arena_t), intent(in) :: arena
            integer, allocatable, intent(in) :: body_indices(:)
            character(len=*), intent(in) :: param_name
            logical, intent(out) :: found
            character(len=:), allocatable, intent(out) :: base_name
            integer, intent(out) :: kind_value
            integer, intent(out) :: rank
            logical, intent(out) :: is_proc
            logical, intent(out) :: is_any
            logical, intent(out) :: unresolved
        end subroutine dummy_decl_signature
        module function arena_proc_param_count(arena, proc_name) result(count)
            integer :: count
            type(ast_arena_t), intent(in) :: arena
            character(len=*), intent(in) :: proc_name
        end function arena_proc_param_count
        module subroutine identifier_name_at(arena, idx, name)
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: idx
            character(len=:), allocatable, intent(out) :: name
        end subroutine identifier_name_at
        module subroutine check_array_constructor_compatibility(arena, error_msg, lazy_mode)
            type(ast_arena_t), intent(in) :: arena
            character(len=:), allocatable, intent(out) :: error_msg
            logical, intent(in), optional :: lazy_mode
        end subroutine check_array_constructor_compatibility
        module subroutine check_array_constructor_type_specs(arena, error_msg)
            type(ast_arena_t), intent(in) :: arena
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_array_constructor_type_specs
        module subroutine check_one_array_constructor(arena, node_index, error_msg)
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: node_index
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_one_array_constructor
        recursive module subroutine check_array_ctor_elements(arena, elems, spec_class, spec_text, line, col, error_msg)
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: elems(:)
            integer, intent(in) :: spec_class, line, col
            character(len=*), intent(in) :: spec_text
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_array_ctor_elements
        module function array_ctor_typespec_class(type_spec) result(cls)
            integer :: cls
            character(len=*), intent(in) :: type_spec
        end function array_ctor_typespec_class
        module function array_ctor_literal_class(arena, node_index) result(cls)
            integer :: cls
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: node_index
        end function array_ctor_literal_class
        module function cmp_class_name(cls) result(name)
            integer, intent(in) :: cls
            character(len=:), allocatable :: name
        end function cmp_class_name
        module subroutine check_gcc_calling_convention_assignments(arena, error_msg)
            type(ast_arena_t), intent(in) :: arena
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_gcc_calling_convention_assignments
        module subroutine parse_gcc_calling_convention_comment(text, name, convention)
            character(len=*), intent(in) :: text
            character(len=:), allocatable, intent(out) :: name, convention
        end subroutine parse_gcc_calling_convention_comment
        module subroutine add_gcc_calling_convention(names, conventions, attr_count, name, convention)
            character(len=64), intent(inout) :: names(:)
            character(len=16), intent(inout) :: conventions(:)
            integer, intent(inout) :: attr_count
            character(len=*), intent(in) :: name, convention
        end subroutine add_gcc_calling_convention
        module function gcc_calling_convention_for_name(names, conventions, attr_count, name) result(convention)
            character(len=64), intent(in) :: names(:)
            character(len=16), intent(in) :: conventions(:)
            integer, intent(in) :: attr_count
            character(len=*), intent(in) :: name
            character(len=:), allocatable :: convention
        end function gcc_calling_convention_for_name
        module function leading_identifier(text) result(name)
            character(len=*), intent(in) :: text
            character(len=:), allocatable :: name
        end function leading_identifier
        module function is_fortran_identifier_char(ch) result(ok)
            logical :: ok
            character(len=1), intent(in) :: ch
        end function is_fortran_identifier_char
        module subroutine check_boz_in_array_constructors(arena, error_msg)
            type(ast_arena_t), intent(in) :: arena
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_boz_in_array_constructors
        recursive module subroutine check_boz_ctor_elements(arena, elems, error_msg)
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: elems(:)
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_boz_ctor_elements
        module subroutine check_boz_literal_contexts(arena, error_msg)
            type(ast_arena_t), intent(in) :: arena
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_boz_literal_contexts
        module function boz_argument_intrinsic(name) result(ok)
            logical :: ok
            character(len=*), intent(in) :: name
        end function boz_argument_intrinsic
        module function boz_assignment_target_typed(arena, target_index) result(ok)
            logical :: ok
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: target_index
        end function boz_assignment_target_typed
        module function boz_compatible_type(type_name) result(ok)
            logical :: ok
            character(len=*), intent(in) :: type_name
        end function boz_compatible_type
        module subroutine declared_type_name_of(arena, name, type_name)
            type(ast_arena_t), intent(in) :: arena
            character(len=*), intent(in) :: name
            character(len=:), allocatable, intent(out) :: type_name
        end subroutine declared_type_name_of
        module subroutine check_assumed_size_dimension_order(arena, error_msg)
            type(ast_arena_t), intent(in) :: arena
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_assumed_size_dimension_order
        module subroutine check_dims_assumed_size_order(arena, dimension_indices, line, column, error_msg)
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: dimension_indices(:)
            integer, intent(in) :: line, column
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_dims_assumed_size_order
        module subroutine check_intrinsic_external_conflict(arena, error_msg)
            type(ast_arena_t), intent(in) :: arena
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_intrinsic_external_conflict
        module subroutine check_intrinsic_external_scope(arena, scope_indices, error_msg)
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: scope_indices(:)
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_intrinsic_external_scope
        module subroutine find_bare_external_name(arena, scope_indices, target_name, found_name)
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: scope_indices(:)
            character(len=*), intent(in) :: target_name
            character(len=:), allocatable, intent(out) :: found_name
        end subroutine find_bare_external_name
        module subroutine check_function_result_save(arena, error_msg)
            type(ast_arena_t), intent(in) :: arena
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_function_result_save
        module subroutine check_result_save_in_body(arena, result_name, body_indices, error_msg)
            type(ast_arena_t), intent(in) :: arena
            character(len=*), intent(in) :: result_name
            integer, intent(in) :: body_indices(:)
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_result_save_in_body
        module subroutine check_duplicate_contained_procedures(arena, error_msg)
            type(ast_arena_t), intent(in) :: arena
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_duplicate_contained_procedures
        module subroutine check_procedure_names_unique(arena, indices, error_msg)
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: indices(:)
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_procedure_names_unique
        module subroutine procedure_def_name(arena, node_index, name)
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: node_index
            character(len=:), allocatable, intent(out) :: name
        end subroutine procedure_def_name
        module subroutine check_bit_intrinsic_arg_ranges(arena, error_msg)
            type(ast_arena_t), intent(in) :: arena
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_bit_intrinsic_arg_ranges
        module subroutine check_bit_intrinsic_call(arena, name, arg_indices, line, col, error_msg)
            type(ast_arena_t), intent(in) :: arena
            character(len=*), intent(in) :: name
            integer, intent(in) :: arg_indices(:)
            integer, intent(in) :: line, col
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_bit_intrinsic_call
        recursive module subroutine try_const_int64(arena, node_index, value, ok)
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: node_index
            integer(c_int64_t), intent(out) :: value
            logical, intent(out) :: ok
        end subroutine try_const_int64
        module function is_integer_text(text) result(is_int)
            logical :: is_int
            character(len=*), intent(in) :: text
        end function is_integer_text
        module subroutine check_scope_nonconstant_bounds(arena, error_msg)
            type(ast_arena_t), intent(in) :: arena
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_scope_nonconstant_bounds
        module subroutine check_decls_nonconstant_bounds(arena, indices, error_msg)
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: indices(:)
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_decls_nonconstant_bounds
        recursive module function expr_has_illegal_call(arena, idx) result(found)
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: idx
            logical :: found
        end function expr_has_illegal_call
        module function arena_has_function_def_named(arena, name) result(found)
            logical :: found
            type(ast_arena_t), intent(in) :: arena
            character(len=*), intent(in) :: name
        end function arena_has_function_def_named
        module subroutine check_derived_type_names_not_intrinsic(arena, error_msg)
            type(ast_arena_t), intent(in) :: arena
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_derived_type_names_not_intrinsic
        module subroutine check_intrinsic_type_stmt_source(arena, error_msg)
            type(ast_arena_t), intent(in) :: arena
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_intrinsic_type_stmt_source
        module subroutine intrinsic_type_stmt_name(line, name, column)
            character(len=*), intent(in) :: line
            character(len=:), allocatable, intent(out) :: name
            integer, intent(out) :: column
        end subroutine intrinsic_type_stmt_name
        module function derived_type_name_is_intrinsic(name) result(is_intrinsic)
            logical :: is_intrinsic
            character(len=*), intent(in) :: name
        end function derived_type_name_is_intrinsic
        module subroutine check_scope_class_declarations(arena, error_msg)
            type(ast_arena_t), intent(in) :: arena
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_scope_class_declarations
        module subroutine check_decls_class(arena, indices, error_msg)
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: indices(:)
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_decls_class
        module subroutine check_automatic_storage_association(arena, error_msg)
            type(ast_arena_t), intent(in) :: arena
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_automatic_storage_association
        module subroutine check_proc_automatic_assoc(arena, param_indices, body_indices, error_msg)
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: param_indices(:)
            integer, intent(in) :: body_indices(:)
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_proc_automatic_assoc
        module function decl_bound_refs_param(arena, decl, param_indices) result(refs)
            logical :: refs
            type(ast_arena_t), intent(in) :: arena
            type(declaration_node), intent(in) :: decl
            integer, intent(in) :: param_indices(:)
        end function decl_bound_refs_param
        recursive module function expr_refs_name(arena, idx, name) result(found)
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: idx
            character(len=*), intent(in) :: name
            logical :: found
        end function expr_refs_name
        module function name_in_common(arena, body_indices, name) result(found)
            logical :: found
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: body_indices(:)
            character(len=*), intent(in) :: name
        end function name_in_common
        module function name_in_equivalence(arena, body_indices, name) result(found)
            logical :: found
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: body_indices(:)
            character(len=*), intent(in) :: name
        end function name_in_equivalence
        module subroutine check_data_source_forms(arena, error_msg)
            type(ast_arena_t), intent(in) :: arena
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_data_source_forms
        module function count_data_statement_nodes(arena) result(total)
            integer :: total
            type(ast_arena_t), intent(in) :: arena
        end function count_data_statement_nodes
        module function strip_data_source_comment(line) result(code)
            character(len=*), intent(in) :: line
            character(len=:), allocatable :: code
        end function strip_data_source_comment
        module function is_data_statement_line(line) result(is_data)
            logical :: is_data
            character(len=*), intent(in) :: line
        end function is_data_statement_line
        module function is_old_style_init_line(line) result(is_old)
            logical :: is_old
            character(len=*), intent(in) :: line
        end function is_old_style_init_line
        module function starts_with_type_keyword(low) result(is_type)
            logical :: is_type
            character(len=*), intent(in) :: low
        end function starts_with_type_keyword
        module subroutine check_format_specifications(arena, error_msg)
            type(ast_arena_t), intent(in) :: arena
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_format_specifications
        module subroutine check_format_tag(arena, query, error_msg)
            type(ast_arena_t), intent(in) :: arena
            type(io_statement_query_t), intent(in) :: query
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_format_tag
        module subroutine check_asynchronous_specifier(query, error_msg)
            type(io_statement_query_t), intent(in) :: query
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_asynchronous_specifier
        module function is_character_intrinsic_name(name) result(is_intrinsic)
            logical :: is_intrinsic
            character(len=*), intent(in) :: name
        end function is_character_intrinsic_name
        module subroutine declared_type_of_name(arena, name, type_name, found)
            type(ast_arena_t), intent(in) :: arena
            character(len=*), intent(in) :: name
            character(len=:), allocatable, intent(out) :: type_name
            logical, intent(out) :: found
        end subroutine declared_type_of_name
        module subroutine check_format_text(text, line, column, error_msg)
            character(len=*), intent(in) :: text
            integer, intent(in) :: line
            integer, intent(in) :: column
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_format_text
        module subroutine check_zero_repeat(text, run_start, run_end, location, error_msg)
            character(len=*), intent(in) :: text
            integer, intent(in) :: run_start
            integer, intent(in) :: run_end
            character(len=*), intent(in) :: location
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_zero_repeat
        module subroutine check_concatenated_format_source(arena, error_msg)
            type(ast_arena_t), intent(in) :: arena
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_concatenated_format_source
        module function ends_with_continuation(text)
            logical :: ends_with_continuation
            character(len=*), intent(in) :: text
        end function ends_with_continuation
        module subroutine append_continuation_line(logical_line, line)
            character(len=:), allocatable, intent(inout) :: logical_line
            character(len=*), intent(in) :: line
        end subroutine append_continuation_line
        module subroutine check_transfer_line_format(line, line_no, error_msg)
            character(len=*), intent(in) :: line
            integer, intent(in) :: line_no
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_transfer_line_format
        module subroutine check_asynchronous_source(text, line_no, error_msg)
            character(len=*), intent(in) :: text
            integer, intent(in) :: line_no
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_asynchronous_source
        module subroutine strip_line_comment(line, stripped)
            character(len=*), intent(in) :: line
            character(len=:), allocatable, intent(out) :: stripped
        end subroutine strip_line_comment
        module subroutine format_expression_text(text, format_text, split_pos)
            character(len=*), intent(in) :: text
            character(len=:), allocatable, intent(out) :: format_text
            integer, intent(out) :: split_pos
        end subroutine format_expression_text
        module subroutine concatenated_literal_text(text, joined, literal_only)
            character(len=*), intent(in) :: text
            character(len=:), allocatable, intent(out) :: joined
            logical, intent(out) :: literal_only
        end subroutine concatenated_literal_text
        module subroutine check_private_component_access(arena, error_msg)
            type(ast_arena_t), intent(in) :: arena
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_private_component_access
        module function private_component_message(arena, node_index, comp_name, type_name) result(message)
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: node_index
            character(len=*), intent(in) :: comp_name, type_name
            character(len=:), allocatable :: message
        end function private_component_message
        module function component_is_private(arena, type_name, comp_name, module_idx)
            logical :: component_is_private
            type(ast_arena_t), intent(in) :: arena
            character(len=*), intent(in) :: type_name, comp_name
            integer, intent(out) :: module_idx
        end function component_is_private
        module subroutine first_private_component(arena, type_name, comp_name, module_idx)
            type(ast_arena_t), intent(in) :: arena
            character(len=*), intent(in) :: type_name
            character(len=:), allocatable, intent(out) :: comp_name
            integer, intent(out) :: module_idx
        end subroutine first_private_component
        module function derived_type_node_index(arena, type_name) result(type_idx)
            integer :: type_idx
            type(ast_arena_t), intent(in) :: arena
            character(len=*), intent(in) :: type_name
        end function derived_type_node_index
        module function derived_component_decl(arena, type_name, comp_name, type_idx) result(decl)
            integer :: decl
            type(ast_arena_t), intent(in) :: arena
            character(len=*), intent(in) :: type_name, comp_name
            integer, intent(out) :: type_idx
        end function derived_component_decl
        module function enclosing_module_index(arena, node_index) result(module_idx)
            integer :: module_idx
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: node_index
        end function enclosing_module_index
        module function node_within(arena, node_index, ancestor) result(within)
            logical :: within
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: node_index, ancestor
        end function node_within
        recursive module subroutine designator_type_name(arena, idx, type_name)
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: idx
            character(len=:), allocatable, intent(out) :: type_name
        end subroutine designator_type_name
        module subroutine declared_derived_type_name(arena, decl, type_name)
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: decl
            character(len=:), allocatable, intent(out) :: type_name
        end subroutine declared_derived_type_name
        module subroutine check_data_allocatable_components(arena, error_msg)
            type(ast_arena_t), intent(in) :: arena
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_data_allocatable_components
        module subroutine check_alloc_pointer_targets(arena, error_msg)
            type(ast_arena_t), intent(in) :: arena
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_alloc_pointer_targets
        module subroutine check_constant_expression_overflow(arena, error_msg)
            type(ast_arena_t), intent(in) :: arena
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_constant_expression_overflow
        module subroutine check_pointer_target_contracts(arena, error_msg)
            type(ast_arena_t), intent(in) :: arena
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_pointer_target_contracts
        module subroutine check_present_argument_subobject(arena, error_msg)
            type(ast_arena_t), intent(in) :: arena
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_present_argument_subobject
        module subroutine check_abstract_interface_pointer(arena, error_msg)
            type(ast_arena_t), intent(in) :: arena
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_abstract_interface_pointer
        module function pointer_declaration_line(arena, name) result(line_no)
            type(ast_arena_t), intent(in) :: arena
            character(len=*), intent(in) :: name
            integer :: line_no
        end function pointer_declaration_line
        module subroutine check_proc_pointer_targets(arena, error_msg)
            type(ast_arena_t), intent(in) :: arena
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_proc_pointer_targets
        module function name_is_proc_pointer(arena, name) result(is_proc_ptr)
            type(ast_arena_t), intent(in) :: arena
            character(len=*), intent(in) :: name
            logical :: is_proc_ptr
        end function name_is_proc_pointer
        module function name_is_data_object(arena, name) result(is_data)
            type(ast_arena_t), intent(in) :: arena
            character(len=*), intent(in) :: name
            logical :: is_data
        end function name_is_data_object
        module function name_is_visible_procedure(arena, name) result(is_visible)
            type(ast_arena_t), intent(in) :: arena
            character(len=*), intent(in) :: name
            logical :: is_visible
        end function name_is_visible_procedure
        module subroutine enclosing_procedure_of(arena, node_index, name, is_recursive)
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: node_index
            character(len=:), allocatable, intent(out) :: name
            logical, intent(out) :: is_recursive
        end subroutine enclosing_procedure_of
        module subroutine check_pointer_intent_actuals(arena, error_msg)
            type(ast_arena_t), intent(in) :: arena
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_pointer_intent_actuals
        module subroutine dummy_pointer_intent(arena, proc_name, position, &
                                               is_pointer, intent_text)
            type(ast_arena_t), intent(in) :: arena
            character(len=*), intent(in) :: proc_name
            integer, intent(in) :: position
            logical, intent(out) :: is_pointer
            character(len=:), allocatable, intent(out) :: intent_text
        end subroutine dummy_pointer_intent
        module subroutine param_node_name(arena, node_index, name)
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: node_index
            character(len=:), allocatable, intent(out) :: name
        end subroutine param_node_name
        module subroutine body_declaration_attributes(arena, body_indices, name, &
                                                       found, is_pointer, intent_text)
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: body_indices(:)
            character(len=*), intent(in) :: name
            logical, intent(out) :: found, is_pointer
            character(len=:), allocatable, intent(out) :: intent_text
        end subroutine body_declaration_attributes
        module subroutine declared_pointer_intent(arena, name, found, is_pointer, &
                                                  intent_text)
            type(ast_arena_t), intent(in) :: arena
            character(len=*), intent(in) :: name
            logical, intent(out) :: found, is_pointer
            character(len=:), allocatable, intent(out) :: intent_text
        end subroutine declared_pointer_intent
        module subroutine check_pointer_source_forms(arena, error_msg)
            type(ast_arena_t), intent(in) :: arena
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_pointer_source_forms
        module function pointer_statement_line(arena, name) result(line_no)
            type(ast_arena_t), intent(in) :: arena
            character(len=*), intent(in) :: name
            integer :: line_no
        end function pointer_statement_line
        module subroutine check_present_source_line(code, line_no, error_msg)
            character(len=*), intent(in) :: code
            integer, intent(in) :: line_no
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_present_source_line
        module function is_plain_identifier(text) result(plain)
            character(len=*), intent(in) :: text
            logical :: plain
        end function is_plain_identifier
        module subroutine get_pointer_source_lines(arena, source)
            type(ast_arena_t), intent(in) :: arena
            character(len=:), allocatable, intent(out) :: source
        end subroutine get_pointer_source_lines
        module subroutine check_cray_pointer_line(code, line_no, error_msg)
            character(len=*), intent(in) :: code
            integer, intent(in) :: line_no
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_cray_pointer_line
        module subroutine check_associated_target_line(code, line_no, error_msg)
            character(len=*), intent(in) :: code
            integer, intent(in) :: line_no
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_associated_target_line
        module subroutine check_parenthesised_actual_line(arena, code, line_no, &
                                                          error_msg)
            type(ast_arena_t), intent(in) :: arena
            character(len=*), intent(in) :: code
            integer, intent(in) :: line_no
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_parenthesised_actual_line
        module subroutine find_call_paren(low, name, from, open_pos)
            character(len=*), intent(in) :: low
            character(len=*), intent(in) :: name
            integer, intent(in) :: from
            integer, intent(out) :: open_pos
        end subroutine find_call_paren
        module subroutine matching_paren(text, open_pos, close_pos)
            character(len=*), intent(in) :: text
            integer, intent(in) :: open_pos
            integer, intent(out) :: close_pos
        end subroutine matching_paren
        module subroutine top_level_args(text, starts, ends, n_args)
            character(len=*), intent(in) :: text
            integer, intent(out) :: starts(:), ends(:), n_args
        end subroutine top_level_args
        module function is_parenthesised(text) result(wrapped)
            character(len=*), intent(in) :: text
            logical :: wrapped
        end function is_parenthesised
        module subroutine check_declaration_conflicts(arena, error_msg)
            type(ast_arena_t), intent(in) :: arena
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_declaration_conflicts
        module subroutine check_scope_decl_conflicts(arena, indices, error_msg)
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: indices(:)
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_scope_decl_conflicts
        module subroutine check_procedure_type_conflicts(arena, indices, &
                                                         error_msg)
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: indices(:)
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_procedure_type_conflicts
        module function procedure_decl_has_interface(decl) result(has_iface)
            logical :: has_iface
            type(declaration_node), intent(in) :: decl
        end function procedure_decl_has_interface
        module subroutine typed_declaration_position(arena, indices, name, &
                                                     line, column)
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: indices(:)
            character(len=*), intent(in) :: name
            integer, intent(out) :: line, column
        end subroutine typed_declaration_position
        module function proc_decl_name_count(decl) result(count)
            integer :: count
            type(declaration_node), intent(in) :: decl
        end function proc_decl_name_count
        module function proc_decl_name_at(decl, k) result(name)
            character(len=:), allocatable :: name
            type(declaration_node), intent(in) :: decl
            integer, intent(in) :: k
        end function proc_decl_name_at
        module subroutine check_null_mold_assignments(arena, indices, error_msg)
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: indices(:)
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_null_mold_assignments
        module subroutine null_mold_name(arena, idx, name)
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: idx
            character(len=:), allocatable, intent(out) :: name
        end subroutine null_mold_name
        module subroutine declared_type_and_rank(arena, indices, name, &
                                                 type_name, is_array, found)
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: indices(:)
            character(len=*), intent(in) :: name
            character(len=:), allocatable, intent(out) :: type_name
            logical, intent(out) :: is_array
            logical, intent(out) :: found
        end subroutine declared_type_and_rank
        module function base_type_name(text) result(base)
            character(len=:), allocatable :: base
            character(len=*), intent(in) :: text
        end function base_type_name
        module subroutine check_polymorphic_entity_attributes(arena, error_msg)
            type(ast_arena_t), intent(in) :: arena
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_polymorphic_entity_attributes
        module subroutine check_deferred_length_attributes(arena, error_msg)
            type(ast_arena_t), intent(in) :: arena
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_deferred_length_attributes
        module subroutine check_pointer_shape_specs(arena, error_msg)
            type(ast_arena_t), intent(in) :: arena
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_pointer_shape_specs
        module subroutine check_alloc_definition_contexts(arena, error_msg)
            type(ast_arena_t), intent(in) :: arena
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_alloc_definition_contexts
        module subroutine check_scope_alloc_targets(arena, body_indices, error_msg)
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: body_indices(:)
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_scope_alloc_targets
        module subroutine check_definable_target(arena, body_indices, target_index, &
                                                 context_name, error_msg)
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: body_indices(:)
            integer, intent(in) :: target_index
            character(len=*), intent(in) :: context_name
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_definable_target
        module subroutine check_argument_definition_contexts(arena, error_msg)
            type(ast_arena_t), intent(in) :: arena
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_argument_definition_contexts
        module function prefix_has_pure(prefix_keywords) result(is_pure)
            logical :: is_pure
            character(len=16), allocatable, intent(in) :: prefix_keywords(:)
        end function prefix_has_pure
        module subroutine check_scope_call_arguments(arena, body_indices, in_pure, &
                                                     error_msg)
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: body_indices(:)
            logical, intent(in) :: in_pure
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_scope_call_arguments
        module subroutine check_call_actual_attributes(arena, body_indices, &
                                                       callee_name, arg_indices, &
                                                       in_pure, error_msg)
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: body_indices(:)
            character(len=*), intent(in) :: callee_name
            integer, intent(in) :: arg_indices(:)
            logical, intent(in) :: in_pure
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_call_actual_attributes
        module subroutine check_one_actual(arena, body_indices, actual_index, &
                                           dummy_name, dummy_alloc, dummy_ptr, &
                                           dummy_definable, dummy_ptr_intent_in, &
                                           in_pure, error_msg)
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: body_indices(:)
            integer, intent(in) :: actual_index
            character(len=*), intent(in) :: dummy_name
            logical, intent(in) :: dummy_alloc, dummy_ptr, dummy_definable
            logical, intent(in) :: dummy_ptr_intent_in
            logical, intent(in) :: in_pure
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_one_actual
        module function decl_display_name(decl) result(name)
            type(declaration_node), intent(in) :: decl
            character(len=:), allocatable :: name
        end function decl_display_name
        module function decl_names_include_dummy(arena, decl) result(is_dummy)
            logical :: is_dummy
            type(ast_arena_t), intent(in) :: arena
            type(declaration_node), intent(in) :: decl
        end function decl_names_include_dummy
        module function decl_names_have_pointer_attr(arena, decl) result(has_attr)
            logical :: has_attr
            type(ast_arena_t), intent(in) :: arena
            type(declaration_node), intent(in) :: decl
        end function decl_names_have_pointer_attr
        module function name_has_pointer_attr_stmt(arena, name) result(has_attr)
            logical :: has_attr
            type(ast_arena_t), intent(in) :: arena
            character(len=*), intent(in) :: name
        end function name_has_pointer_attr_stmt
        module function decl_declares_name(decl, name) result(declares)
            logical :: declares
            type(declaration_node), intent(in) :: decl
            character(len=*), intent(in) :: name
        end function decl_declares_name
        module function name_is_dummy_anywhere(arena, name) result(is_dummy)
            logical :: is_dummy
            type(ast_arena_t), intent(in) :: arena
            character(len=*), intent(in) :: name
        end function name_is_dummy_anywhere
        module function params_contain_name(arena, param_indices, name) &
                result(found)
            logical :: found
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: param_indices(:)
            character(len=*), intent(in) :: name
        end function params_contain_name
        module subroutine param_name_at(arena, param_index, name)
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: param_index
            character(len=:), allocatable, intent(out) :: name
        end subroutine param_name_at
        module subroutine procedure_signature(arena, name, param_indices, &
                                               body_indices, found)
            type(ast_arena_t), intent(in) :: arena
            character(len=*), intent(in) :: name
            integer, allocatable, intent(out) :: param_indices(:)
            integer, allocatable, intent(out) :: body_indices(:)
            logical, intent(out) :: found
        end subroutine procedure_signature
        module subroutine scope_decl_for_name(arena, indices, name, decl_index)
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: indices(:)
            character(len=*), intent(in) :: name
            integer, intent(out) :: decl_index
        end subroutine scope_decl_for_name
        module subroutine module_decl_for_name(arena, name, decl_index)
            type(ast_arena_t), intent(in) :: arena
            character(len=*), intent(in) :: name
            integer, intent(out) :: decl_index
        end subroutine module_decl_for_name
        module subroutine target_base_name(arena, node_index, root_name)
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: node_index
            character(len=:), allocatable, intent(out) :: root_name
        end subroutine target_base_name
        recursive module subroutine check_data_object_components(arena, idx, error_msg)
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: idx
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_data_object_components
    end interface
    interface
        module subroutine check_generic_interface_forms(arena, error_msg)
        type(ast_arena_t), intent(in) :: arena
        character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_generic_interface_forms
        module subroutine generic_source_lines(source, lines, count)
        character(len=*), intent(in) :: source
        character(len=512), allocatable, intent(out) :: lines(:)
        integer, intent(out) :: count
        end subroutine generic_source_lines
        module function stmt_starts(line, kw) result(yes)
        character(len=*), intent(in) :: line
        character(len=*), intent(in) :: kw
        logical :: yes
        end function stmt_starts
        module function identifier_after(text, from) result(name)
        character(len=:), allocatable :: name
        character(len=*), intent(in) :: text
        integer, intent(in) :: from
        end function identifier_after
        module subroutine append_owned_name(tab, owners, n, name, owner)
        character(len=64), intent(inout) :: tab(:)
        integer, intent(inout) :: owners(:)
        integer, intent(inout) :: n
        character(len=*), intent(in) :: name
        integer, intent(in) :: owner
        end subroutine append_owned_name
        module function owned_name_present(tab, owners, n, name, owner) result(yes)
        character(len=64), intent(in) :: tab(:)
        integer, intent(in) :: owners(:)
        integer, intent(in) :: n
        character(len=*), intent(in) :: name
        integer, intent(in) :: owner
        logical :: yes
        end function owned_name_present
        module subroutine interface_regions(lines, count, depth, iface_name)
        character(len=512), intent(in) :: lines(:)
        integer, intent(in) :: count
        integer, allocatable, intent(out) :: depth(:)
        character(len=64), allocatable, intent(out) :: iface_name(:)
        end subroutine interface_regions
        module function is_plain_generic_name(name) result(yes)
        character(len=*), intent(in) :: name
        logical :: yes
        end function is_plain_generic_name
        module subroutine check_generic_binding_syntax(lines, count, error_msg)
        character(len=512), intent(in) :: lines(:)
        integer, intent(in) :: count
        character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_generic_binding_syntax
        module subroutine interface_body_header(text, name, signature, ok)
        character(len=*), intent(in) :: text
        character(len=:), allocatable, intent(out) :: name
        character(len=:), allocatable, intent(out) :: signature
        logical, intent(out) :: ok
        end subroutine interface_body_header
        module subroutine check_implicit_interface_ambiguity(lines, count, error_msg)
        character(len=512), intent(in) :: lines(:)
        integer, intent(in) :: count
        character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_implicit_interface_ambiguity
        module subroutine append_name_list(tab, owners, n, rest, owner)
        character(len=64), intent(inout) :: tab(:)
        integer, intent(inout) :: owners(:)
        integer, intent(inout) :: n
        character(len=*), intent(in) :: rest
        integer, intent(in) :: owner
        end subroutine append_name_list
        module subroutine program_unit_ids(lines, count, ids)
        character(len=512), intent(in) :: lines(:)
        integer, intent(in) :: count
        integer, allocatable, intent(out) :: ids(:)
        end subroutine program_unit_ids
        module subroutine collect_procedure_tables(lines, count, defs, def_owner, ndef, &
        bodies, body_owner, nbody, gens, &
        gen_owner, ngen, exts, ext_owner, next)
        character(len=512), intent(in) :: lines(:)
        integer, intent(in) :: count
        character(len=64), intent(inout) :: defs(:), bodies(:), gens(:), exts(:)
        integer, intent(inout) :: def_owner(:), body_owner(:), gen_owner(:), &
        ext_owner(:)
        integer, intent(out) :: ndef, nbody, ngen, next
        end subroutine collect_procedure_tables
        module subroutine check_module_procedure_targets(lines, count, error_msg)
        character(len=512), intent(in) :: lines(:)
        integer, intent(in) :: count
        character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_module_procedure_targets
        module subroutine generic_block_specifics(lines, count, header_line, specs, nspec)
        character(len=512), intent(in) :: lines(:)
        integer, intent(in) :: count
        integer, intent(in) :: header_line
        character(len=64), intent(inout) :: specs(:)
        integer, intent(out) :: nspec
        integer :: owners(size(specs))
        end subroutine generic_block_specifics
        module subroutine check_generic_name_collisions(lines, count, error_msg)
        character(len=512), intent(in) :: lines(:)
        integer, intent(in) :: count
        character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_generic_name_collisions
        module subroutine module_export_table(lines, count, mods, nmod, names, owner, nname)
        character(len=512), intent(in) :: lines(:)
        integer, intent(in) :: count
        character(len=64), intent(inout) :: mods(:)
        integer, intent(out) :: nmod
        character(len=64), intent(inout) :: names(:)
        integer, intent(inout) :: owner(:)
        integer, intent(out) :: nname
        end subroutine module_export_table
        module function used_module_index(text, mods, nmod) result(idx)
        character(len=*), intent(in) :: text
        character(len=64), intent(in) :: mods(:)
        integer, intent(in) :: nmod
        integer :: idx
        end function used_module_index
        module subroutine check_use_shadows_program_unit(lines, count, error_msg)
        character(len=512), intent(in) :: lines(:)
        integer, intent(in) :: count
        character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_use_shadows_program_unit
        module function use_makes_name_accessible(text, name) result(yes)
        character(len=*), intent(in) :: text
        character(len=*), intent(in) :: name
        logical :: yes
        end function use_makes_name_accessible
        module function line_references_name(line, name) result(yes)
        character(len=*), intent(in) :: line
        character(len=*), intent(in) :: name
        logical :: yes
        end function line_references_name
        module function preceded_by_component_selector(line, pos) result(yes)
        character(len=*), intent(in) :: line
        integer, intent(in) :: pos
        logical :: yes
        end function preceded_by_component_selector
        module subroutine check_ambiguous_use_association(lines, count, error_msg)
        character(len=512), intent(in) :: lines(:)
        integer, intent(in) :: count
        character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_ambiguous_use_association
        module function generic_extends_own_name(lines, count, mods, midx, name) result(yes)
        character(len=512), intent(in) :: lines(:)
        integer, intent(in) :: count
        character(len=64), intent(in) :: mods(:)
        integer, intent(in) :: midx
        character(len=*), intent(in) :: name
        logical :: yes
        end function generic_extends_own_name
        module subroutine ambiguous_reference_line(lines, first, last, name, found_line)
        character(len=512), intent(in) :: lines(:)
        integer, intent(in) :: first, last
        character(len=*), intent(in) :: name
        integer, intent(out) :: found_line
        end subroutine ambiguous_reference_line
        module subroutine scoping_regions(lines, count, depth, region)
        character(len=512), intent(in) :: lines(:)
        integer, intent(in) :: count
        integer, intent(in) :: depth(:)
        integer, allocatable, intent(out) :: region(:)
        end subroutine scoping_regions
        module subroutine check_typebound_generic_inheritance(arena, lines, count, error_msg)
        type(ast_arena_t), intent(in) :: arena
        character(len=512), intent(in) :: lines(:)
        integer, intent(in) :: count
        character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_typebound_generic_inheritance
        module function bindings_indistinguishable(arena, name_a, name_b, tname, tparent, nt) result(same)
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: name_a, name_b
        character(len=64), intent(in) :: tname(:), tparent(:)
        integer, intent(in) :: nt
        logical :: same
        end function bindings_indistinguishable
        module function base_types_related(base_a, base_b, tname, tparent, nt) result(related)
        character(len=*), intent(in) :: base_a, base_b
        character(len=64), intent(in) :: tname(:), tparent(:)
        integer, intent(in) :: nt
        logical :: related
        end function base_types_related
        module function declared_type_index(base, tname, nt) result(idx)
        character(len=*), intent(in) :: base
        character(len=64), intent(in) :: tname(:)
        integer, intent(in) :: nt
        integer :: idx
        end function declared_type_index
        module subroutine collect_type_generics(lines, count, tname, tparent, nt, gspec, &
        gtarget, gtype, gline, ng)
        character(len=512), intent(in) :: lines(:)
        integer, intent(in) :: count
        character(len=64), intent(inout) :: tname(:), tparent(:)
        integer, intent(out) :: nt
        character(len=64), intent(inout) :: gspec(:), gtarget(:)
        integer, intent(inout) :: gtype(:), gline(:)
        integer, intent(out) :: ng
        end subroutine collect_type_generics
        module function squeeze_blanks(text) result(packed)
        character(len=:), allocatable :: packed
        character(len=*), intent(in) :: text
        end function squeeze_blanks
        recursive module function type_extends_type(tname, tparent, nt, child, ancestor) result(yes)
        character(len=64), intent(in) :: tname(:), tparent(:)
        integer, intent(in) :: nt, child, ancestor
        logical :: yes
        end function type_extends_type
        module subroutine check_intrinsic_assignment_redefinition(arena, lines, count, &
        error_msg)
        type(ast_arena_t), intent(in) :: arena
        character(len=512), intent(in) :: lines(:)
        integer, intent(in) :: count
        character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_intrinsic_assignment_redefinition
        module function intrinsic_assignment_defined(base_l, base_r) result(yes)
        character(len=*), intent(in) :: base_l, base_r
        logical :: yes
        end function intrinsic_assignment_defined
        module function is_numeric_base_type(base) result(yes)
        character(len=*), intent(in) :: base
        logical :: yes
        end function is_numeric_base_type
        module subroutine check_generic_call_resolves(arena, lines, count, error_msg)
        type(ast_arena_t), intent(in) :: arena
        character(len=512), intent(in) :: lines(:)
        integer, intent(in) :: count
        character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_generic_call_resolves
        module subroutine generic_union_specifics(lines, count, name, specs, nspec)
        character(len=512), intent(in) :: lines(:)
        integer, intent(in) :: count
        character(len=*), intent(in) :: name
        character(len=64), intent(inout) :: specs(:)
        integer, intent(out) :: nspec
        end subroutine generic_union_specifics
        module subroutine literal_actual_types(text, actuals, nactual)
        character(len=*), intent(in) :: text
        character(len=64), intent(inout) :: actuals(:)
        integer, intent(out) :: nactual
        end subroutine literal_actual_types
        module function literal_base_type(item) result(base)
        character(len=:), allocatable :: base
        character(len=*), intent(in) :: item
        end function literal_base_type
        module subroutine generic_call_matches(arena, specs, nspec, actuals, nactual, &
        matched, resolvable)
        type(ast_arena_t), intent(in) :: arena
        character(len=64), intent(in) :: specs(:)
        integer, intent(in) :: nspec
        character(len=64), intent(in) :: actuals(:)
        integer, intent(in) :: nactual
        logical, intent(out) :: matched, resolvable
        end subroutine generic_call_matches
    end interface
    interface
        module subroutine check_result_and_entry_rules(arena, error_msg)
            type(ast_arena_t), intent(in) :: arena
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_result_and_entry_rules
    end interface
    interface
        module subroutine check_constant_initialization_exprs(arena, error_msg)
            type(ast_arena_t), intent(in) :: arena
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_constant_initialization_exprs
        module subroutine check_scope_const_inits(arena, indices, error_msg)
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: indices(:)
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_scope_const_inits
        module subroutine check_async_specifiers(arena, indices, specifiers, line, &
                                                 col, error_msg)
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: indices(:)
            type(io_specifier_t), intent(in) :: specifiers(:)
            integer, intent(in) :: line, col
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_async_specifiers
        module subroutine bare_name_const_reason(arena, indices, text, reason)
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: indices(:)
            character(len=*), intent(in) :: text
            character(len=:), allocatable, intent(out) :: reason
        end subroutine bare_name_const_reason
        recursive module subroutine const_expr_reason(arena, indices, idx, &
                                                      loop_names, reason)
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: indices(:)
            integer, intent(in) :: idx
            character(len=*), intent(in) :: loop_names
            character(len=:), allocatable, intent(out) :: reason
        end subroutine const_expr_reason
        recursive module subroutine call_const_reason(arena, indices, nd, &
                                                      loop_names, reason)
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: indices(:)
            type(call_or_subscript_node), intent(in) :: nd
            character(len=*), intent(in) :: loop_names
            character(len=:), allocatable, intent(out) :: reason
        end subroutine call_const_reason
        module subroutine identifier_const_reason(arena, indices, name, &
                                                  loop_names, reason)
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: indices(:)
            character(len=*), intent(in) :: name
            character(len=*), intent(in) :: loop_names
            character(len=:), allocatable, intent(out) :: reason
        end subroutine identifier_const_reason
        module subroutine shape_inquiry_reason(arena, indices, arg_index, reason)
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: indices(:)
            integer, intent(in) :: arg_index
            character(len=:), allocatable, intent(out) :: reason
        end subroutine shape_inquiry_reason
        module function declaration_declares_name(decl, lname) result(declares)
            type(declaration_node), intent(in) :: decl
            character(len=*), intent(in) :: lname
            logical :: declares
        end function declaration_declares_name
    end interface
    interface
        module subroutine check_storage_association_restrictions(arena, error_msg)
            type(ast_arena_t), intent(in) :: arena
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_storage_association_restrictions
        module subroutine storage_source_lines(arena, lines, line_count, found)
            type(ast_arena_t), intent(in) :: arena
            character(len=256), allocatable, intent(out) :: lines(:)
            integer, intent(out) :: line_count
            logical, intent(out) :: found
        end subroutine storage_source_lines
        module subroutine append_storage_name(names, count, name)
            character(len=*), intent(inout) :: names(:)
            integer, intent(inout) :: count
            character(len=*), intent(in) :: name
        end subroutine append_storage_name
        module function storage_name_listed(names, count, name) result(listed)
            logical :: listed
            character(len=*), intent(in) :: names(:)
            integer, intent(in) :: count
            character(len=*), intent(in) :: name
        end function storage_name_listed
    end interface
    interface
        module subroutine check_array_shape_expressions(arena, error_msg)
            type(ast_arena_t), intent(in) :: arena
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_array_shape_expressions
    end interface
    interface
        module subroutine check_purity_attribute_restrictions(arena, error_msg)
            type(ast_arena_t), intent(in) :: arena
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_purity_attribute_restrictions
    end interface
    interface
        module subroutine check_comparison_operand_types(arena, bin_op, left_idx, &
                                                          right_idx, line, col, &
                                                          context, error_msg)
            type(ast_arena_t), intent(in) :: arena
            character(len=*), intent(in) :: bin_op
            integer, intent(in) :: left_idx, right_idx, line, col
            type(lowering_context_t), intent(in) :: context
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine check_comparison_operand_types
        module function is_relational_operator(op) result(is_rel)
            logical :: is_rel
            character(len=*), intent(in) :: op
        end function is_relational_operator
        recursive module function comparison_operand_class(arena, node_index, &
                                                            context) result(cls)
            integer :: cls
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: node_index
            type(lowering_context_t), intent(in) :: context
        end function comparison_operand_class
        module function comparison_value_kind_class(value_kind) result(cls)
            integer :: cls
            integer, intent(in) :: value_kind
        end function comparison_value_kind_class
        module function is_hollerith_literal(arena, node_index) result(is_holl)
            logical :: is_holl
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: node_index
        end function is_hollerith_literal
    end interface
    interface
        module subroutine lower_enum_block(arena, node_index, context, error_msg, &
                                           handled)
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: node_index
            type(lowering_context_t), intent(inout) :: context
            character(len=:), allocatable, intent(out) :: error_msg
            logical, intent(out) :: handled
        end subroutine lower_enum_block
        module function enumerator_value(node, idx) result(value)
            type(enum_node), intent(in) :: node
            integer, intent(in) :: idx
            integer(c_int64_t) :: value
        end function enumerator_value
        module subroutine bind_enum_constant(context, name, value, error_msg)
            type(lowering_context_t), intent(inout) :: context
            character(len=*), intent(in) :: name
            integer(c_int64_t), intent(in) :: value
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine bind_enum_constant
    end interface
    interface
        module function alloc_desc_dim_offset(dim, field) result(offset)
            integer, intent(in) :: dim
            integer(c_int64_t), intent(in) :: field
            integer(c_int64_t) :: offset
        end function alloc_desc_dim_offset
        module subroutine emit_alloc_desc_header(context, descriptor, value_kind, &
                                                 rank, error_msg, element_bytes)
            type(lowering_context_t), intent(inout) :: context
            type(lr_operand_desc_t), intent(in) :: descriptor
            integer, intent(in) :: value_kind
            integer, intent(in) :: rank
            character(len=:), allocatable, intent(out) :: error_msg
            integer(c_int64_t), intent(in), optional :: element_bytes
        end subroutine emit_alloc_desc_header
        module subroutine emit_alloc_desc_flags(context, descriptor, &
                                                allocated_state, error_msg)
            type(lowering_context_t), intent(inout) :: context
            type(lr_operand_desc_t), intent(in) :: descriptor
            logical, intent(in) :: allocated_state
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine emit_alloc_desc_flags
        module subroutine emit_alloc_desc_set_dim(context, descriptor, dim, &
                                                  lower_i64, extent_i64, &
                                                  stride_i64, error_msg)
            type(lowering_context_t), intent(inout) :: context
            type(lr_operand_desc_t), intent(in) :: descriptor
            integer, intent(in) :: dim
            type(lr_operand_desc_t), intent(in) :: lower_i64
            type(lr_operand_desc_t), intent(in) :: extent_i64
            type(lr_operand_desc_t), intent(in) :: stride_i64
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine emit_alloc_desc_set_dim
        module subroutine emit_alloc_desc_load_lower(context, descriptor, dim, &
                                                     lower_i64, error_msg)
            type(lowering_context_t), intent(inout) :: context
            type(lr_operand_desc_t), intent(in) :: descriptor
            integer, intent(in) :: dim
            type(lr_operand_desc_t), intent(out) :: lower_i64
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine emit_alloc_desc_load_lower
        module subroutine emit_alloc_desc_load_extent(context, descriptor, dim, &
                                                      extent_i64, error_msg)
            type(lowering_context_t), intent(inout) :: context
            type(lr_operand_desc_t), intent(in) :: descriptor
            integer, intent(in) :: dim
            type(lr_operand_desc_t), intent(out) :: extent_i64
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine emit_alloc_desc_load_extent
        module subroutine emit_alloc_desc_load_upper(context, descriptor, dim, &
                                                     upper_i64, error_msg)
            type(lowering_context_t), intent(inout) :: context
            type(lr_operand_desc_t), intent(in) :: descriptor
            integer, intent(in) :: dim
            type(lr_operand_desc_t), intent(out) :: upper_i64
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine emit_alloc_desc_load_upper
        module subroutine emit_alloc_desc_allocate_shape(context, descriptor, &
                                                         value_kind, rank, &
                                                         extents_i64, error_msg, &
                                                         lowers_i64, element_bytes)
            type(lowering_context_t), intent(inout) :: context
            type(lr_operand_desc_t), intent(in) :: descriptor
            integer, intent(in) :: value_kind
            integer, intent(in) :: rank
            type(lr_operand_desc_t), intent(in) :: extents_i64(:)
            character(len=:), allocatable, intent(out) :: error_msg
            type(lr_operand_desc_t), intent(in), optional :: lowers_i64(:)
            integer(c_int64_t), intent(in), optional :: element_bytes
        end subroutine emit_alloc_desc_allocate_shape
        module subroutine emit_alloc_desc_clear(context, descriptor, error_msg)
            type(lowering_context_t), intent(inout) :: context
            type(lr_operand_desc_t), intent(in) :: descriptor
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine emit_alloc_desc_clear
    end interface
    ! OPEN/CLOSE and file-unit WRITE lowering lives in a descendant so the
    ! compiler's I/O service has an explicit, independently buildable seam.
    public :: parse_open_spec, spec_lower, lower_open, open_status_operand
    public :: open_file_operands, unit_string_operand, allocate_newunit
    public :: mark_unit_symbol, set_unit_form, store_io_status
    public :: store_iostat_value, store_iomsg_text, runtime_iostat_operand
    public :: lower_close, unit_number_operand, unit_spec_text
    public :: load_unit_file_ptr, is_file_unit_write, unit_is_unformatted
    public :: lower_write_file, lower_file_write_unformatted
    public :: lower_file_write_unformatted_item, logical_write_kind_bytes
    public :: store_write_iostat_success, file_write_value_kind
    public :: lower_file_write_item, lower_file_write_newline
    public :: lower_file_write_formatted, fortran_fmt_to_c, read_fmt_int

    interface
        module subroutine parse_open_spec(spec, unit_str, newunit_var, file_path, &
                                          file_quoted, status_str, status_quoted, &
                                          form_str, access_str, sign_str, &
                                          iostat_var, iomsg_var, error_msg)
            character(len=*), intent(in) :: spec
            character(len=:), allocatable, intent(out) :: unit_str
            character(len=:), allocatable, intent(out) :: newunit_var
            character(len=:), allocatable, intent(out) :: file_path
            logical, intent(out) :: file_quoted
            character(len=:), allocatable, intent(out) :: status_str
            logical, intent(out) :: status_quoted
            character(len=:), allocatable, intent(out) :: form_str
            character(len=:), allocatable, intent(out) :: access_str
            character(len=:), allocatable, intent(out) :: sign_str
            character(len=:), allocatable, intent(out) :: iostat_var
            character(len=:), allocatable, intent(out) :: iomsg_var
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine parse_open_spec
        module function spec_lower(s) result(t)
            character(len=*), intent(in) :: s
            character(len=len(s)) :: t
        end function spec_lower
        module subroutine lower_open(arena, node, context, error_msg)
            type(ast_arena_t), intent(in) :: arena
            type(open_statement_node), intent(in) :: node
            type(lowering_context_t), intent(inout) :: context
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine lower_open
        module subroutine open_status_operand(context, status_text, status_quoted, &
                                             status_op, error_msg)
            type(lowering_context_t), intent(inout) :: context
            character(len=*), intent(in) :: status_text
            logical, intent(in) :: status_quoted
            type(lr_operand_desc_t), intent(out) :: status_op
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine open_status_operand
        module subroutine open_file_operands(context, fpath, file_quoted, data_op, &
                                             len_op, error_msg)
            type(lowering_context_t), intent(inout) :: context
            character(len=*), intent(in) :: fpath
            logical, intent(in) :: file_quoted
            type(lr_operand_desc_t), intent(out) :: data_op
            type(lr_operand_desc_t), intent(out) :: len_op
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine open_file_operands
        module subroutine unit_string_operand(context, tag, text, ptr_op, error_msg)
            type(lowering_context_t), intent(inout) :: context
            character(len=*), intent(in) :: tag
            character(len=*), intent(in) :: text
            type(lr_operand_desc_t), intent(out) :: ptr_op
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine unit_string_operand
        module subroutine allocate_newunit(context, nuvar, unit_op, error_msg)
            type(lowering_context_t), intent(inout) :: context
            character(len=*), intent(in) :: nuvar
            type(lr_operand_desc_t), intent(out) :: unit_op
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine allocate_newunit
        module subroutine mark_unit_symbol(context, pseudo_name, error_msg)
            type(lowering_context_t), intent(inout) :: context
            character(len=*), intent(in) :: pseudo_name
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine mark_unit_symbol
        module subroutine set_unit_form(context, name, unformatted, error_msg)
            type(lowering_context_t), intent(inout) :: context
            character(len=*), intent(in) :: name
            logical, intent(in) :: unformatted
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine set_unit_form
        module subroutine store_io_status(context, iostat_name, iomsg_name, status_op, &
                                          error_msg)
            type(lowering_context_t), intent(inout) :: context
            character(len=*), intent(in) :: iostat_name
            character(len=*), intent(in) :: iomsg_name
            type(lr_operand_desc_t), intent(in) :: status_op
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine store_io_status
        module subroutine store_iostat_value(context, name, status_op, error_msg)
            type(lowering_context_t), intent(inout) :: context
            character(len=*), intent(in) :: name
            type(lr_operand_desc_t), intent(in) :: status_op
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine store_iostat_value
        module subroutine store_iomsg_text(context, name, error_msg)
            type(lowering_context_t), intent(inout) :: context
            character(len=*), intent(in) :: name
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine store_iomsg_text
        module function runtime_iostat_operand(context, error_msg) result(status_op)
            type(lowering_context_t), intent(inout) :: context
            character(len=:), allocatable, intent(out) :: error_msg
            type(lr_operand_desc_t) :: status_op
        end function runtime_iostat_operand
        module subroutine lower_close(node, context, error_msg)
            type(close_statement_node), intent(in) :: node
            type(lowering_context_t), intent(inout) :: context
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine lower_close
        module subroutine unit_number_operand(unit_spec, context, unit_op, error_msg)
            character(len=*), intent(in) :: unit_spec
            type(lowering_context_t), intent(inout) :: context
            type(lr_operand_desc_t), intent(out) :: unit_op
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine unit_number_operand
        module function unit_spec_text(unit_spec) result(plain)
            character(len=*), intent(in) :: unit_spec
            character(len=:), allocatable :: plain
        end function unit_spec_text
        module subroutine load_unit_file_ptr(unit_spec, context, fp, error_msg)
            character(len=*), intent(in) :: unit_spec
            type(lowering_context_t), intent(inout) :: context
            type(lr_operand_desc_t), intent(out) :: fp
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine load_unit_file_ptr
        module function is_file_unit_write(node, context) result(result_value)
            type(write_statement_node), intent(in) :: node
            type(lowering_context_t), intent(inout) :: context
            logical :: result_value
        end function is_file_unit_write
        module function unit_is_unformatted(node, context) result(result_value)
            type(write_statement_node), intent(in) :: node
            type(lowering_context_t), intent(inout) :: context
            logical :: result_value
        end function unit_is_unformatted
        module subroutine lower_write_file(arena, node, context, error_msg)
            type(ast_arena_t), intent(in) :: arena
            type(write_statement_node), intent(in) :: node
            type(lowering_context_t), intent(inout) :: context
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine lower_write_file
        module subroutine lower_file_write_unformatted(arena, node, fp, context, &
                                                       error_msg)
            type(ast_arena_t), intent(in) :: arena
            type(write_statement_node), intent(in) :: node
            type(lr_operand_desc_t), intent(in) :: fp
            type(lowering_context_t), intent(inout) :: context
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine lower_file_write_unformatted
        module subroutine lower_file_write_unformatted_item(arena, node_index, fp, &
                                                            context, error_msg)
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: node_index
            type(lr_operand_desc_t), intent(in) :: fp
            type(lowering_context_t), intent(inout) :: context
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine lower_file_write_unformatted_item
        module function logical_write_kind_bytes(arena, node_index, context) result(bytes)
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: node_index
            type(lowering_context_t), intent(in) :: context
            integer :: bytes
        end function logical_write_kind_bytes
        module subroutine store_write_iostat_success(node, context, error_msg)
            type(write_statement_node), intent(in) :: node
            type(lowering_context_t), intent(inout) :: context
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine store_write_iostat_success
        module function file_write_value_kind(arena, node_index, context) &
            result(result_value)
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: node_index
            type(lowering_context_t), intent(inout) :: context
            integer :: result_value
        end function file_write_value_kind
        module subroutine lower_file_write_item(arena, node_index, fp, context, &
                                                error_msg)
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: node_index
            type(lr_operand_desc_t), intent(in) :: fp
            type(lowering_context_t), intent(inout) :: context
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine lower_file_write_item
        module subroutine lower_file_write_newline(fp, context, error_msg)
            type(lr_operand_desc_t), intent(in) :: fp
            type(lowering_context_t), intent(inout) :: context
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine lower_file_write_newline
        module subroutine lower_file_write_formatted(arena, node, fp, context, &
                                                      error_msg)
            type(ast_arena_t), intent(in) :: arena
            type(write_statement_node), intent(in) :: node
            type(lr_operand_desc_t), intent(in) :: fp
            type(lowering_context_t), intent(inout) :: context
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine lower_file_write_formatted
        module function fortran_fmt_to_c(fort_fmt, c_fmt, error_msg) &
            result(result_value)
            character(len=*), intent(in) :: fort_fmt
            character(len=:), allocatable, intent(out) :: c_fmt
            character(len=:), allocatable, intent(out) :: error_msg
            logical :: result_value
        end function fortran_fmt_to_c
        module subroutine read_fmt_int(s, pos, val)
            character(len=*), intent(in) :: s
            integer, intent(inout) :: pos
            integer, intent(out) :: val
        end subroutine read_fmt_int
    end interface
    ! Character-array element substring resolution lives in a typed descendant
    ! so the expression/descriptor contract is explicit instead of being
    ! hidden in the textual character-family include.
    interface
        module function is_character_substring(arena, node_index, context) &
                result(is_substring)
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: node_index
            type(lowering_context_t), intent(in) :: context
            logical :: is_substring
        end function is_character_substring
        module subroutine substring_operands(arena, node_index, context, &
                                              data_ptr, length, error_msg)
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: node_index
            type(lowering_context_t), intent(inout) :: context
            type(lr_operand_desc_t), intent(out) :: data_ptr
            type(lr_operand_desc_t), intent(out) :: length
            character(len=:), allocatable, intent(out) :: error_msg
        end subroutine substring_operands
        module function actual_is_character(arena, node_index, context) &
                result(is_character)
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: node_index
            type(lowering_context_t), intent(in) :: context
            logical :: is_character
        end function actual_is_character
    end interface

    interface
        module function dim_is_assumed_shape(arena, dim_index) result(is_assumed)
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: dim_index
            logical :: is_assumed
        end function dim_is_assumed_shape
        module function declaration_is_assumed_shape(node, context) result(is_assumed)
            type(declaration_node), intent(in) :: node
            type(lowering_context_t), intent(in) :: context
            logical :: is_assumed
        end function declaration_is_assumed_shape
        module function declaration_is_runtime_rank1(node, context) result(is_runtime)
            type(declaration_node), intent(in) :: node
            type(lowering_context_t), intent(in) :: context
            logical :: is_runtime
        end function declaration_is_runtime_rank1
        recursive module function bound_expr_references_variable(arena, idx, context) &
                result(has_var)
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: idx
            type(lowering_context_t), intent(in) :: context
            logical :: has_var
        end function bound_expr_references_variable
        module function bound_identifier_references_variable(arena, idx, context) &
                result(has_var)
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: idx
            type(lowering_context_t), intent(in) :: context
            logical :: has_var
        end function bound_identifier_references_variable
        module function declaration_bound_is_variable_driven(node, context) &
                result(is_var)
            type(declaration_node), intent(in) :: node
            type(lowering_context_t), intent(in) :: context
            logical :: is_var
        end function declaration_bound_is_variable_driven
        module function declaration_is_runtime_local_array(node, context, value_kind) &
                result(is_local)
            type(declaration_node), intent(in) :: node
            type(lowering_context_t), intent(in) :: context
            integer, intent(in) :: value_kind
            logical :: is_local
        end function declaration_is_runtime_local_array
        module function declaration_rebinds_runtime_array_result(node, context) &
                result(is_rebind)
            type(declaration_node), intent(in) :: node
            type(lowering_context_t), intent(in) :: context
            logical :: is_rebind
        end function declaration_rebinds_runtime_array_result
        module function declaration_is_assumed_rank(node, context) result(is_rank)
            type(declaration_node), intent(in) :: node
            type(lowering_context_t), intent(in) :: context
            logical :: is_rank
        end function declaration_is_assumed_rank
        module function dim_is_assumed_size(arena, dim_index) result(is_assumed)
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: dim_index
            logical :: is_assumed
        end function dim_is_assumed_size
        module function declaration_is_assumed_size(node, context) result(is_assumed)
            type(declaration_node), intent(in) :: node
            type(lowering_context_t), intent(in) :: context
            logical :: is_assumed
        end function declaration_is_assumed_size
    end interface
contains
    include 'session_program_lowering_top.inc'
    subroutine lower_declaration(node_in, node_index, context, error_msg)
        !! Lower a declaration, then record the FortFront binding identity of
        !! every name it declares (#327). Registration happens after lowering
        !! so it sees the symbol slot the declaration actually produced,
        !! whichever of the many declaration paths below created it.
        type(declaration_node), intent(in) :: node_in
        ! Arena index of this declaration, the unique key for SAVE-local globals.
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: enclosing_declaration_index

        ! Publish this declaration as the scope anchor for the specification
        ! expressions it contains (#329). A declaration nests only through a
        ! derived-type definition, whose components lower on their own path,
        ! so saving and restoring the previous value is enough to keep a
        ! stale anchor from leaking into an unrelated declaration.
        enclosing_declaration_index = context%current_declaration_index
        context%current_declaration_index = node_index
        call lower_declaration_entities(node_in, node_index, context, error_msg)
        context%current_declaration_index = enclosing_declaration_index
        if (len_trim(error_msg) > 0) return
        call register_declaration_bindings(context, node_in, node_index)
    end subroutine lower_declaration

    subroutine register_declaration_bindings(context, node, node_index)
        !! Bind each name this declaration introduces to its lowering symbol
        !! by FortFront binding identity. Nothing is registered unless
        !! FortFront agrees that, at the declaration's own site, the name
        !! denotes this very declaration: ffc must never invent an identity.
        type(lowering_context_t), intent(inout) :: context
        type(declaration_node), intent(in) :: node
        integer, intent(in) :: node_index
        integer :: i

        if (node_index <= 0) return
        if (node%is_multi_declaration .and. allocated(node%var_names)) then
            do i = 1, size(node%var_names)
                call register_declaration_binding(context, &
                    trim(node%var_names(i)), node_index)
            end do
            return
        end if
        if (allocated(node%var_name)) &
            call register_declaration_binding(context, trim(node%var_name), &
            node_index)
    end subroutine register_declaration_bindings

    subroutine register_declaration_binding(context, name, node_index)
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: name
        integer, intent(in) :: node_index
        type(declaration_binding_t) :: binding
        character(len=:), allocatable :: resolve_error
        integer :: symbol_index, binding_index

        if (len_trim(name) == 0) return
        ! The declaration has just run, so the newest same-named slot is the
        ! one it produced. This is the last place text is used as a key.
        symbol_index = find_symbol_compat(context, name)
        if (symbol_index <= 0) return
        call resolve_name_at_node(context%arena, node_index, name, binding, &
            resolve_error)
        if (len_trim(resolve_error) > 0) return
        if (.not. binding%found) return
        if (binding%declaration_node_index /= node_index) return
        if (binding%scope_node_index <= 0) return
        if (context%declaration_collection_complete) then
            if (find_declaration_record(context, binding) <= 0) return
        end if
        binding_index = find_symbol_for_binding(context, binding)
        if (binding_index > 0) then
            symbol_index = binding_index
        else if (context%symbols(symbol_index)%has_binding) then
            return
        end if
        call attach_symbol_binding(context, symbol_index, binding)
    end subroutine register_declaration_binding

    subroutine attach_symbol_binding(context, symbol_index, binding)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        type(declaration_binding_t), intent(in) :: binding

        if (symbol_index <= 0 .or. symbol_index > context%symbol_count) return
        if (.not. binding%found) return
        context%symbols(symbol_index)%has_binding = .true.
        context%symbols(symbol_index)%binding_declaration_index = &
            binding%declaration_node_index
        context%symbols(symbol_index)%binding_entity_index = &
            binding%declaration_entity_index
        context%symbols(symbol_index)%binding_scope_index = &
            binding%scope_node_index
        call context%binding_table%insert_binding( &
            binding%declaration_node_index, binding%declaration_entity_index, &
            binding%scope_node_index, symbol_index)
    end subroutine attach_symbol_binding

    subroutine push_storage_scope(context, saved_symbol_count, saved_floor)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(out) :: saved_symbol_count
        integer, intent(out) :: saved_floor

        saved_symbol_count = context%symbol_count
        saved_floor = context%block_scope_floor
        context%block_scope_floor = saved_symbol_count
        context%storage_scope_depth = context%storage_scope_depth + 1
    end subroutine push_storage_scope

    subroutine pop_storage_scope(context, saved_symbol_count, saved_floor)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: saved_symbol_count
        integer, intent(in) :: saved_floor

        call context%binding_table%drop_bindings_from(saved_symbol_count + 1)
        context%symbol_count = saved_symbol_count
        context%block_scope_floor = saved_floor
        context%storage_scope_depth = max(0, context%storage_scope_depth - 1)
    end subroutine pop_storage_scope

    subroutine lower_declaration_entities(node_in, node_index, context, error_msg)
        type(declaration_node), intent(in) :: node_in
        ! Arena index of this declaration, the unique key for SAVE-local globals.
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        ! A DIMENSION statement (is_array, no type) and the variable's typed
        ! declaration arrive as two separate nodes. Merge the pending shape into
        ! a local working copy so an otherwise-scalar typed declaration lowers as
        ! the intended array; skip the bare DIMENSION statement itself.
        type(declaration_node) :: node
        integer :: array_lower_bound
        integer :: array_size
        integer :: derived_type_index
        integer :: i
        integer :: value_kind
        integer :: pointer_symbol
        logical :: complex_dummy_pointer

        if (declaration_is_bare_dimension(node_in)) then
            call set_empty(error_msg)
            return
        end if
        ! A bare EXTERNAL statement (external :: bar) names a procedure, not a
        ! variable, and carries no type. The call site resolves the procedure
        ! from the function table, so the statement itself defines no storage.
        if (declaration_is_bare_external(node_in)) then
            call set_empty(error_msg)
            return
        end if
        node = node_in
        if (.not. node%is_array .and. allocated(node%var_name)) &
            call apply_pending_dimension(context, node)
        derived_type_index = declaration_derived_type_index(context, node)
        if (derived_type_index > 0) then
            call lower_derived_type_declaration(node, context, derived_type_index, &
                error_msg)
            return
        end if
        ! Procedure pointer: procedure(iface), pointer :: fp (#245 B3d).
        ! Detected by type_name starting with "procedure" and is_pointer.
        if (node%is_pointer .and. declaration_names_procedure(node_in)) then
            call lower_proc_pointer_declaration(node, context, error_msg)
            return
        end if
        ! A PROCEDURE declaration statement (procedure(iface) :: p) names an
        ! external procedure with an explicit interface, not storage. Calls
        ! resolve through the procedure table, so it defines no operand (#364).
        if (declaration_names_procedure(node_in)) then
            call set_empty(error_msg)
            return
        end if
        if ((node%is_pointer .or. node%is_target) .and. node%is_array) then
            ! TARGET is permitted on an assumed-shape dummy. It uses the same
            ! caller-supplied descriptor as an ordinary assumed-shape dummy;
            ! only the target property is retained for pointer association.
            if (node%is_target .and. declaration_is_assumed_shape(node, context)) then
                call declaration_value_kind(node, value_kind, error_msg, context, &
                    node_index)
                if (len_trim(error_msg) > 0) return
                call lower_assumed_shape_declaration(node, context, value_kind, &
                    error_msg)
                if (len_trim(error_msg) > 0) return
                if (node%is_multi_declaration .and. allocated(node%var_names)) then
                    do i = 1, size(node%var_names)
                        derived_type_index = find_symbol_compat(context, &
                            node%var_names(i))
                        if (derived_type_index > 0) &
                            context%symbols(derived_type_index)%is_target = .true.
                    end do
                else if (allocated(node%var_name)) then
                    derived_type_index = find_symbol_compat(context, node%var_name)
                    if (derived_type_index > 0) &
                        context%symbols(derived_type_index)%is_target = .true.
                end if
                return
            end if
            call declaration_value_kind(node, value_kind, error_msg, context, &
                node_index)
            if (len_trim(error_msg) > 0) return
            call lower_pointer_target_array(node, context, value_kind, error_msg)
            return
        end if
        if (node%is_pointer .or. node%is_target) then
            call declaration_value_kind(node, value_kind, error_msg, context, &
                node_index)
            if (len_trim(error_msg) > 0) return
            complex_dummy_pointer = .false.
            if ((value_kind == VALUE_C4 .or. value_kind == VALUE_C8) .and. &
                allocated(node%var_name)) then
                pointer_symbol = find_symbol_compat(context, node%var_name)
                if (pointer_symbol > 0) then
                    complex_dummy_pointer = &
                        context%symbols(pointer_symbol)%is_dummy_argument
                end if
            end if
            if (value_kind /= VALUE_I32 .and. value_kind /= VALUE_F32 .and. &
                value_kind /= VALUE_F64 .and. value_kind /= VALUE_LOGICAL .and. &
                value_kind /= VALUE_CHARACTER .and. .not. complex_dummy_pointer) then
                call unsupported_feature_error('pointer/target declaration', &
                    node%line, node%column, &
                    'direct LIRIC session supports scalar integer, real, logical, '// &
                    'and fixed-length character pointer/target only (#452)', &
                    error_msg)
                return
            end if
            call lower_scalar_pointer_target(node, context, value_kind, error_msg)
            return
        end if
        if (node%is_array) then
            ! A saved array lives in one static global emitted by the pre-pass,
            ! so bind it there instead of allocating call-local storage (#466).
            if (node%is_save .and. .not. node%is_allocatable) then
                call lower_saved_array_declaration(context%arena, node, &
                                                   node_index, context, error_msg)
                return
            end if
            call declaration_value_kind(node, value_kind, error_msg, context, &
                node_index)
            if (len_trim(error_msg) > 0) return
            if (value_kind == VALUE_CHARACTER) then
                call lower_character_array_declaration(node, context, error_msg)
                return
            end if
            if (value_kind /= VALUE_I32 .and. value_kind /= VALUE_F32 .and. &
                value_kind /= VALUE_F64 .and. value_kind /= VALUE_LOGICAL .and. &
                value_kind /= VALUE_I64 .and. value_kind /= VALUE_I8 .and. &
                value_kind /= VALUE_I16 .and. value_kind /= VALUE_C4 .and. &
                value_kind /= VALUE_C8) then
                call unsupported_feature_error('array declaration', node%line, &
                    node%column, &
                    'ffc direct-session lowering only '// &
                    'supports integer, real, and '// &
                    'logical arrays', &
                    error_msg)
                return
            end if
            if (node%is_allocatable) then
                call lower_allocatable_declaration(node, context, error_msg, &
                    value_kind)
                return
            end if
            ! Complex arrays store real and imaginary parts in two parallel
            ! fixed-size arrays (define_declared_array_symbol); that layout has
            ! no defined ABI for a caller-supplied actual, so dummies, assumed-
            ! shape, and assumed-rank complex arrays stay unsupported here.
            if ((value_kind == VALUE_C4 .or. value_kind == VALUE_C8) .and. &
                (declaration_is_assumed_rank(node, context) .or. &
                declaration_is_assumed_shape(node, context))) then
                call unsupported_feature_error('complex array declaration', &
                    node%line, node%column, &
                    'direct LIRIC session supports complex arrays as '// &
                    'fixed-size local declarations only, not assumed-shape '// &
                    'or assumed-rank dummies', error_msg)
                return
            end if
            ! Assumed-rank dummy arr(..): no static rank, so bind it to the
            ! parameter base and take its rank from the caller's actual; a later
            ! select rank dispatches on that resolved rank (#273).
            if (declaration_is_assumed_rank(node, context)) then
                call lower_assumed_rank_declaration(node, context, value_kind, &
                    error_msg)
                return
            end if
            ! Assumed-shape dummy a(:): no compile-time bound on the colon
            ! dimensions, so bind it to the parameter base and take its extent
            ! from the caller's actual instead of folding the declaration.
            if (declaration_is_assumed_shape(node, context)) then
                call lower_assumed_shape_declaration(node, context, value_kind, &
                    error_msg)
                return
            end if
            ! A named constant a(*) = [...] takes its extent from the array
            ! constructor initializer rather than a caller's actual (only
            ! valid for a dummy argument or a PARAMETER); size it from the
            ! initializer and fall through to the normal fixed-size path.
            if (declaration_is_assumed_size(node, context) .and. &
                declaration_is_parameter_assumed_size_array(node, context)) then
                call get_parameter_assumed_size_extent(node, context, &
                    array_size, error_msg)
                if (len_trim(error_msg) > 0) return
                array_lower_bound = 1
                call define_declared_array_symbol(context, node, node%var_name, &
                    array_lower_bound, array_size, value_kind, error_msg, &
                    is_assumed_size=.true.)
                return
            end if
            ! Assumed-size dummy a(n1, ..., *): the trailing asterisk carries
            ! no compile-time extent, so fold only the leading dimensions and
            ! bind the parameter base without a whole-array size.
            if (declaration_is_assumed_size(node, context)) then
                call lower_assumed_size_declaration(node, context, value_kind, &
                    error_msg)
                return
            end if
            ! A rank-1 local automatic array sized by a runtime expression
            ! (integer :: a(n) with n a dummy/host value): allocate dynamic
            ! storage and record the runtime element count. Fresh locals only;
            ! adjustable-array dummies keep their parameter-base binding.
            if (declaration_is_runtime_local_array(node, context, value_kind)) then
                if (node%is_multi_declaration .and. allocated(node%var_names)) then
                    do i = 1, size(node%var_names)
                        call define_runtime_array_symbol(context, node, &
                            node%var_names(i), value_kind, error_msg)
                        if (len_trim(error_msg) > 0) return
                    end do
                else
                    call define_runtime_array_symbol(context, node, &
                        node%var_name, value_kind, error_msg)
                end if
                return
            end if
            ! A runtime-sized array function result (dimension(n) with dummy n):
            ! its extent does not fold at compile time. The result symbol is
            ! pre-bound to the sret buffer, so skip bound folding and let
            ! define_declared_array_symbol bind the runtime view onto param 0.
            if (declaration_rebinds_runtime_array_result(node, context)) then
                array_lower_bound = 1
                array_size = 0
            else
                call get_array_bounds(node, context, array_lower_bound, &
                    array_size, error_msg)
                if (len_trim(error_msg) > 0) return
            end if
            if (node%is_multi_declaration .and. allocated(node%var_names)) then
                do i = 1, size(node%var_names)
                    call define_declared_array_symbol( &
                        context, node, node%var_names(i), array_lower_bound, &
                        array_size, value_kind, error_msg)
                    if (len_trim(error_msg) > 0) return
                end do
            else if (allocated(node%var_name)) then
                call define_declared_array_symbol(context, node, node%var_name, &
                    array_lower_bound, array_size, &
                    value_kind, error_msg)
            else
                error_msg = 'array declaration did not expose a variable name'
            end if
            return
        end if
        call declaration_value_kind(node, value_kind, error_msg, context, node_index)
        if (len_trim(error_msg) > 0) return
        if (node%is_allocatable .and. value_kind /= VALUE_CHARACTER .and. &
            value_kind /= VALUE_CLASS_STAR) then
            call lower_scalar_allocatable_declaration(node, context, value_kind, &
                error_msg)
            return
        end if
        ! A true PARAMETER constant never carries an INTENT. Lazy-mode
        ! standardization restates a dummy (integer, intent(in) :: a) as a
        ! declaration that also sets is_parameter; routing that to the constant
        ! path demands an initializer the dummy has not got. An intent marks it
        ! as the already-bound dummy, so take the normal scalar path, which
        ! benignly refreshes the dummy's kind (#2812).
        if (node%is_parameter .and. .not. node%has_intent) then
            call lower_constant_declaration(node, context, value_kind, error_msg)
            return
        end if
        ! SAVE gives a scalar local static storage that persists across calls,
        ! so it is backed by a global with a once-applied initializer (#1541).
        if (node%is_save) then
            call lower_saved_scalar_declaration(node, node_index, context, &
                value_kind, error_msg)
            return
        end if
        if (node%is_multi_declaration .and. allocated(node%var_names)) then
            do i = 1, size(node%var_names)
                call define_declared_symbol(context, node, node%var_names(i), &
                    value_kind, error_msg)
                if (len_trim(error_msg) > 0) return
            end do
        else if (allocated(node%var_name)) then
            call define_declared_symbol(context, node, node%var_name, &
                value_kind, error_msg)
            if (len_trim(error_msg) > 0) return
            if (node%has_initializer .and. node%initializer_index > 0) then
                if (value_kind == VALUE_CHARACTER) then
                    call lower_character_initializer(context, node%var_name, &
                        node%initializer_index, &
                        error_msg)
                else
                    call lower_scalar_initializer(context, node%var_name, &
                        value_kind, &
                        node%initializer_index, &
                        error_msg)
                end if
            end if
        else
            error_msg = 'scalar declaration did not expose a variable name'
        end if
    end subroutine lower_declaration_entities

    ! A bare DIMENSION statement: it carries the array shape (is_array with
    ! dimension_indices) but names no type, so the variable's type comes from a
    ! separate typed declaration. Lowering the statement on its own has no type
    ! to define, so it is skipped and its shape merged into the typed node.
    logical function declaration_is_bare_dimension(node) result(is_bare)
        type(declaration_node), intent(in) :: node

        is_bare = .false.
        if (.not. node%is_array) return
        if (.not. allocated(node%type_name)) then
            is_bare = .true.
            return
        end if
        is_bare = len_trim(node%type_name) == 0
    end function declaration_is_bare_dimension

    logical function declaration_is_bare_external(node) result(is_bare)
        ! An EXTERNAL statement names a procedure even when it carries the
        ! procedure's result type (character, external :: f). It contributes no
        ! storage; calls resolve through the procedure table.
        type(declaration_node), intent(in) :: node

        is_bare = .false.
        is_bare = node%is_external
    end function declaration_is_bare_external

    logical function declaration_names_procedure(node) result(is_procedure)
        ! True for a PROCEDURE declaration statement, whose type-spec is the
        ! referenced interface name (procedure(iface) :: p) rather than an
        ! intrinsic or derived type.
        type(declaration_node), intent(in) :: node
        integer :: span

        is_procedure = .false.
        if (.not. allocated(node%type_name)) return
        span = min(9, len_trim(node%type_name))
        if (span < 9) return
        is_procedure = lowercase_text(trim(adjustl(node%type_name(1:span)))) == &
                       'procedure'
    end function declaration_names_procedure

    subroutine apply_pending_dimension(context, node)
        ! Give a typed scalar declaration the array shape declared for the same
        ! name by a separate DIMENSION statement (attr_dim_02).
        type(lowering_context_t), intent(in) :: context
        type(declaration_node), intent(inout) :: node
        integer :: n

        do n = 1, context%arena%size
            if (.not. node_exists(context%arena, n)) cycle
            select type (other => context%arena%entries(n)%node)
                type is (declaration_node)
                if (.not. declaration_is_bare_dimension(other)) cycle
                if (.not. allocated(other%var_name)) cycle
                if (.not. same_name(trim(other%var_name), &
                    trim(node%var_name))) cycle
                if (.not. allocated(other%dimension_indices)) return
                node%is_array = .true.
                node%dimension_indices = other%dimension_indices
                return
            end select
        end do
    end subroutine apply_pending_dimension

    subroutine lower_scalar_initializer(context, name, value_kind, init_index, &
            error_msg)
        !! Apply a scalar declaration initializer (integer :: x = 2) by lowering
        !! the initializer expression and storing it into the variable, mirroring
        !! a plain assignment. Without this the variable keeps its zero default.
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: name
        integer, intent(in) :: value_kind, init_index
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: value
        integer :: symbol_index

        call set_empty(error_msg)
        symbol_index = find_symbol_compat(context, name)
        if (symbol_index <= 0) return
        select case (value_kind)
        case (VALUE_F32)
            call lower_f32_expression(context%arena, init_index, context, value, &
                error_msg)
        case (VALUE_F64)
            call lower_f64_expression(context%arena, init_index, context, value, &
                error_msg)
        case (VALUE_LOGICAL)
            call lower_logical_expression(context%arena, init_index, context, &
                value, error_msg)
        case (VALUE_I32)
            call lower_i32_expression(context%arena, init_index, context, value, &
                error_msg)
        case (VALUE_I8)
            call lower_i8_expression(context%arena, init_index, context, value, &
                error_msg)
        case (VALUE_I16)
            call lower_i16_expression(context%arena, init_index, context, value, &
                error_msg)
        case (VALUE_I64)
            call lower_i64_expression(context%arena, init_index, context, value, &
                error_msg)
        case (VALUE_C4)
            ! Complex initializers write re/im into the symbol's two slots
            ! directly, so reuse the assignment helper and skip the scalar
            ! value-store path below.
            call lower_c4_assignment(context%arena, init_index, symbol_index, &
                context, error_msg)
            return
        case (VALUE_C8)
            call lower_c8_assignment(context%arena, init_index, symbol_index, &
                context, error_msg)
            return
        case default
            return
        end select
        if (len_trim(error_msg) > 0) return
        context%symbols(symbol_index)%value = value
        if (context%symbols(symbol_index)%has_address .and. &
            context%symbols(symbol_index)%is_reference) then
            call store_reference_value(context, symbol_index, value, error_msg)
        end if
    end subroutine lower_scalar_initializer

    subroutine lower_character_initializer(context, name, init_index, error_msg)
        !! Apply a fixed-length character declaration initializer
        !! (character(len=N) :: s = "...") by folding the literal, padding to
        !! the declared width, and materialising it into the symbol's storage.
        !! Without this the symbol keeps a null value pointer and any read
        !! (trim, len_trim, print, concat) dereferences garbage.
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: name
        integer, intent(in) :: init_index
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: literal_text
        character(len=64) :: string_name
        logical :: fold_ok
        integer :: symbol_index
        type(lr_operand_desc_t) :: value

        call set_empty(error_msg)
        symbol_index = find_symbol_compat(context, name)
        if (symbol_index <= 0) return
        if (context%symbols(symbol_index)%is_runtime_fixed_character) then
            call lower_runtime_fixed_char_assignment(context%arena, init_index, &
                context, symbol_index, error_msg)
            return
        end if
        call concat_character_literals(context%arena, init_index, literal_text, &
            fold_ok)
        if (.not. fold_ok) then
            call unsupported_feature_error('character initializer', 0, 0, &
                'only character-literal initializers are supported by direct '// &
                'LIRIC session', error_msg)
            return
        end if
        call normalize_character_literal( &
            literal_text, context%symbols(symbol_index)%character_length)
        context%string_literal_count = context%string_literal_count + 1
        string_name = ffc_unit_global_name( &
            context, 'char.', context%string_literal_count)
        call materialize_liric_string(context%session, trim(string_name), &
            literal_text, &
            value, &
            error_msg)
        if (len_trim(error_msg) > 0) return
        if ((context%symbols(symbol_index)%is_target .or. &
            context%symbols(symbol_index)%is_pointer) .and. &
            context%symbols(symbol_index)%has_address) then
            if (.not. emit_memcpy(context%session, &
                context%symbols(symbol_index)%address, value, &
                i64_immediate(context%session, int( &
                context%symbols(symbol_index)%character_length + 1, &
                c_int64_t)), error_msg)) return
            context%symbols(symbol_index)%value = &
                context%symbols(symbol_index)%address
        else
            context%symbols(symbol_index)%value = value
        end if
        context%symbols(symbol_index)%has_character_value = .true.
    end subroutine lower_character_initializer

    include 'session_program_lowering_data.inc'
    include 'session_program_lowering_declarations.inc'
    include 'session_program_lowering_inferred.inc'
    include 'session_program_lowering_lazy_monomorph.inc'
    subroutine define_declared_symbol(context, node, name, value_kind, error_msg)
        type(lowering_context_t), intent(inout) :: context
        type(declaration_node), intent(in) :: node
        character(len=*), intent(in) :: name
        integer, intent(in) :: value_kind
        character(len=:), allocatable, intent(out) :: error_msg
        if (value_kind == VALUE_CHARACTER) then
            call define_declared_character_symbol(context, node, name, error_msg)
        else if (value_kind == VALUE_CLASS_STAR) then
            call define_declared_class_star_symbol(context, name, error_msg)
        else
            call define_symbol(context, name, value_kind, error_msg)
        end if
        if (len_trim(error_msg) == 0 .and. value_kind == VALUE_LOGICAL) then
            call set_declared_logical_kind(context, name, node%resolved_kind_value)
        end if
    end subroutine define_declared_symbol

    subroutine set_declared_logical_kind(context, name, kind_value)
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: name
        integer, intent(in) :: kind_value
        integer :: index

        index = find_symbol_compat(context, name)
        if (index <= 0) return
        select case (kind_value)
        case (1, 2, 4, 8)
            context%symbols(index)%logical_kind_bytes = kind_value
        case default
            ! The default logical representation in this lowering is i32.
            context%symbols(index)%logical_kind_bytes = 4
        end select
    end subroutine set_declared_logical_kind

    subroutine define_declared_class_star_symbol(context, name, error_msg)
        ! A class(*) dummy: the parameter pointer addresses a 16-byte
        ! {void* data; i64 type_id} descriptor (#141). data is at offset 0 and
        ! the runtime type id at offset 8. A local class(*) allocatable gets the
        ! same descriptor layout in a stack slot, allocated empty and populated
        ! by a typed allocate (#273).
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: name
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: index

        index = find_symbol_compat(context, name)
        if (index <= 0) then
            call define_local_class_star_symbol(context, name, error_msg)
            return
        end if
        if (.not. context%symbols(index)%is_parameter) then
            call define_local_class_star_symbol(context, name, error_msg)
            return
        end if
        context%symbols(index)%value_kind = VALUE_CLASS_STAR
        if (.not. emit_ptr_offset(context%session, &
            context%symbols(index)%address, 0_c_int64_t, &
            context%symbols(index)%deferred_data, error_msg)) return
        if (.not. emit_ptr_offset(context%session, &
            context%symbols(index)%address, 8_c_int64_t, &
            context%symbols(index)%deferred_length, error_msg)) return
        call set_empty(error_msg)
    end subroutine define_declared_class_star_symbol

    subroutine define_local_class_star_symbol(context, name, error_msg)
        ! A local class(*) allocatable: stack-allocate the 16-byte
        ! {void* data; i64 type_id} descriptor, zero both slots (unallocated),
        ! and expose the slot addresses through deferred_data/deferred_length so
        ! select type and a typed allocate share the dummy descriptor path (#273).
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: name
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: index

        if (find_symbol_compat(context, name) > 0) then
            error_msg = 'duplicate class(*) declaration: '//trim(name)
            return
        end if
        call grow_symbols(context)
        index = context%symbol_count + 1
        context%symbols(index)%name = trim(name)
        context%symbols(index)%value_kind = VALUE_CLASS_STAR
        context%symbols(index)%is_allocatable = .true.
        context%symbol_count = index
        if (.not. emit_alloca_bytes(context%session, &
            i64_immediate(context%session, 16_c_int64_t), &
            context%symbols(index)%address, error_msg)) return
        if (.not. emit_ptr_offset(context%session, &
            context%symbols(index)%address, 0_c_int64_t, &
            context%symbols(index)%deferred_data, error_msg)) return
        if (.not. emit_ptr_offset(context%session, &
            context%symbols(index)%address, 8_c_int64_t, &
            context%symbols(index)%deferred_length, error_msg)) return
        if (.not. emit_ptr_store(context%session, null_ptr_operand(context), &
            context%symbols(index)%deferred_data, error_msg)) return
        if (.not. emit_i64_store(context%session, &
            i64_immediate(context%session, 0_c_int64_t), &
            context%symbols(index)%deferred_length, error_msg)) return
        call set_empty(error_msg)
    end subroutine define_local_class_star_symbol
    subroutine lower_parameter_declaration(node, context, error_msg)
        type(parameter_declaration_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: symbol_index, value_kind, character_length, declared_index
        character(len=:), allocatable :: declared_type
        if (.not. allocated(node%name)) then
            error_msg = 'parameter declaration did not expose a name'
            return
        end if
        if (node%is_array) then
            call unsupported_feature_error('array parameter declaration', &
                node%line, node%column, &
                'array parameters are not supported '// &
                'by direct LIRIC session', error_msg)
            return
        end if
        if (allocated(node%type_name)) then
            call type_name_value_kind(node%type_name, node%line, node%column, &
                value_kind, error_msg, context)
            if (len_trim(error_msg) > 0) return
        else
            value_kind = VALUE_I32
        end if
        symbol_index = find_symbol_compat(context, node%name)
        if (symbol_index <= 0) then
            error_msg = 'parameter declaration did not match a dummy argument: '// &
                trim(node%name)
            return
        end if
        if (.not. context%symbols(symbol_index)%is_parameter) then
            error_msg = 'parameter declaration did not match a dummy argument: '// &
                trim(node%name)
            return
        end if
        if (value_kind == VALUE_CHARACTER) then
            ! Both a fixed-length dummy (character(len=N)) and an
            ! assumed-length one (character(len=*)) read their data pointer
            ! from the caller's {data, length} descriptor; a fixed-length
            ! dummy keeps its own declared width N rather than the caller's
            ! runtime length.
            call parse_character_length(node%type_name, character_length, error_msg)
            if (len_trim(error_msg) > 0) return
            if (character_length > 0) then
                call bind_fixed_character_parameter_symbol(context, &
                    symbol_index, character_length, error_msg)
            else
                call bind_character_parameter_symbol(context, symbol_index, &
                    error_msg)
            end if
        else if (value_kind == VALUE_DERIVED) then
            if (allocated(node%type_name)) then
                if (is_class_derived_type_spec(node%type_name)) then
                    call extracted_derived_type_name(node%type_name, declared_type)
                    declared_index = find_derived_type(context, declared_type)
                    call bind_class_dummy_descriptor(context, symbol_index, &
                        declared_index, error_msg)
                else
                    call update_parameter_symbol(context, symbol_index, &
                        value_kind, error_msg)
                end if
            else
                call update_parameter_symbol(context, symbol_index, value_kind, &
                    error_msg)
            end if
        else
            call update_parameter_symbol(context, symbol_index, value_kind, &
                error_msg)
        end if
        if (len_trim(error_msg) > 0) return
        call set_empty(error_msg)
    end subroutine lower_parameter_declaration
    ! Array lowering is included here; keep this unit invalidated when the include changes.
    ! Typed TRANSPOSE also shares this include's complex and logical paths.
    ! Parameter TRANSPOSE initialization uses the same typed array stores.
    ! Typed integer MIN/MAX calls use the i64 comparison wrapper.
    ! Legacy MIN aliases reuse the typed scalar min/max engines.
    include 'session_program_lowering_arrays.inc'
    include 'session_program_lowering_array_elements.inc'
    include 'session_program_lowering_vector_subscript.inc'
    include 'session_program_lowering_char_arrays.inc'
    include 'session_program_lowering_allocatable.inc'
    include 'session_program_lowering_runtime_alloc.inc'
    include 'session_program_lowering_alloc_array_result.inc'
    include 'session_program_lowering_reduction_expr.inc'
    include 'session_program_lowering_io_implied_do.inc'
    include 'session_program_lowering_scalar_allocatable.inc'
    include 'session_program_lowering_internal_write.inc'
    include 'session_program_lowering_internal_read.inc'
    subroutine define_symbol(context, name, value_kind, error_msg)
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: name
        integer, intent(in) :: value_kind
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: existing_index
        type(declaration_binding_t) :: binding
        character(len=:), allocatable :: binding_error
        logical :: has_current_binding
        integer :: text_index, binding_index

        has_current_binding = .false.
        if (context%current_declaration_index > 0) then
            call resolve_name_at_node(context%arena, &
                context%current_declaration_index, name, binding, binding_error)
            if (len_trim(binding_error) == 0 .and. binding%found) &
                has_current_binding = .true.
        end if
        text_index = find_symbol_compat(context, name)
        existing_index = text_index
        if (has_current_binding) then
            binding_index = find_symbol_for_binding(context, binding)
            if (binding_index > 0) then
                if (context%symbols(binding_index)%is_parameter) then
                    call update_parameter_symbol(context, binding_index, value_kind, &
                        error_msg)
                    return
                end if
                if (context%symbols(binding_index)%value_kind == value_kind) then
                    call set_empty(error_msg)
                    return
                end if
                existing_index = binding_index
            end if
        end if
        ! A match in an enclosing scope must not be reused: a BLOCK-local
        ! declaration shadows it with a fresh slot so the outer storage is left
        ! intact (#280). Only same-scope matches take the benign-redeclare path.
        if (existing_index > 0 .and. existing_index > context%block_scope_floor) then
            if (context%symbols(existing_index)%is_parameter) then
                call update_parameter_symbol(context, existing_index, value_kind, &
                    error_msg)
                return
            end if
            ! A re-declaration of an existing same-kind symbol is benign (e.g. a
            ! function result variable pre-defined from the result clause and
            ! then declared in the body). True conflicts are caught by FortFront.
            if (context%symbols(existing_index)%value_kind == value_kind) then
                call set_empty(error_msg)
                return
            end if
        end if
        if (value_kind == VALUE_I32) then
            call define_i32_symbol(context, name, error_msg)
        else if (value_kind == VALUE_I8) then
            call define_i8_symbol(context, name, error_msg)
        else if (value_kind == VALUE_I16) then
            call define_i16_symbol(context, name, error_msg)
        else if (value_kind == VALUE_I64) then
            call define_i64_symbol(context, name, error_msg)
        else if (value_kind == VALUE_F32) then
            call define_f32_symbol(context, name, error_msg)
        else if (value_kind == VALUE_F64) then
            call define_f64_symbol(context, name, error_msg)
        else if (value_kind == VALUE_LOGICAL) then
            call define_logical_symbol(context, name, error_msg)
        else if (value_kind == VALUE_CHARACTER) then
            call define_character_symbol(context, name, 1, error_msg)
        else if (value_kind == VALUE_C_PTR) then
            call define_c_ptr_symbol(context, name, error_msg)
        else if (value_kind == VALUE_C4) then
            call define_c4_symbol(context, name, error_msg)
        else if (value_kind == VALUE_C8) then
            call define_c8_symbol(context, name, error_msg)
        else
            error_msg = 'unknown scalar value kind for direct LIRIC session'
        end if
    end subroutine define_symbol
    subroutine update_parameter_symbol(context, index, value_kind, error_msg)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: index
        integer, intent(in) :: value_kind
        character(len=:), allocatable, intent(out) :: error_msg
        logical :: direct_value
        if (index <= 0 .or. index > context%symbol_count) then
            error_msg = 'parameter index is outside the symbol table'
            return
        end if
        if (.not. context%symbols(index)%is_parameter) then
            error_msg = 'symbol is not a parameter: '//trim(context%symbols(index)%name)
            return
        end if
        ! A BIND(C) VALUE dummy already carries its ABI value in the incoming
        ! parameter vreg. The specification-part redeclaration only supplies
        ! its resolved kind; replacing that vreg with a zero immediate would
        ! silently discard the C argument (#584).
        direct_value = context%current_proc_bind_c .and. &
            context%symbols(index)%is_dummy_argument .and. &
            .not. context%symbols(index)%has_address .and. &
            .not. context%symbols(index)%is_reference
        if (direct_value) then
            call set_empty(error_msg)
            return
        end if
        context%symbols(index)%value_kind = value_kind
        if (value_kind == VALUE_F32) then
            context%symbols(index)%value = liric_f32_immediate(context%session, &
                0.0_c_float)
        else if (value_kind == VALUE_F64) then
            context%symbols(index)%value = liric_f64_immediate(context%session, &
                0.0_c_double)
        else if (value_kind == VALUE_LOGICAL .or. value_kind == VALUE_I32) then
            context%symbols(index)%value = i32_immediate(context%session, 0_c_int64_t)
        else if (value_kind == VALUE_I64) then
            context%symbols(index)%value = i64_immediate(context%session, 0_c_int64_t)
        else if (value_kind == VALUE_I16) then
            context%symbols(index)%value = i16_immediate(context%session, 0_c_int64_t)
        else if (value_kind == VALUE_I8) then
            context%symbols(index)%value = i8_immediate(context%session, 0_c_int64_t)
        else if (value_kind == VALUE_C_PTR) then
            context%symbols(index)%value = null_ptr_operand(context)
        else if (value_kind == VALUE_C4) then
            if (.not. context%symbols(index)%has_address) then
                if (.not. emit_alloca_bytes(context%session, &
                    i64_immediate(context%session, 8_c_int64_t), &
                    context%symbols(index)%address, error_msg)) return
                if (.not. emit_ptr_offset(context%session, &
                    context%symbols(index)%address, 4_c_int64_t, &
                    context%symbols(index)%element_address, error_msg)) return
                context%symbols(index)%has_address = .true.
            else if (.not. emit_ptr_offset(context%session, &
                    context%symbols(index)%address, 4_c_int64_t, &
                    context%symbols(index)%element_address, error_msg)) then
                return
            end if
            context%symbols(index)%value = &
                liric_f32_immediate(context%session, 0.0_c_float)
        else if (value_kind == VALUE_C8) then
            if (.not. context%symbols(index)%has_address) then
                if (.not. emit_alloca_bytes(context%session, &
                    i64_immediate(context%session, 16_c_int64_t), &
                    context%symbols(index)%address, error_msg)) return
                if (.not. emit_ptr_offset(context%session, &
                    context%symbols(index)%address, 8_c_int64_t, &
                    context%symbols(index)%element_address, error_msg)) return
                context%symbols(index)%has_address = .true.
            else if (.not. emit_ptr_offset(context%session, &
                    context%symbols(index)%address, 8_c_int64_t, &
                    context%symbols(index)%element_address, error_msg)) then
                return
            end if
            context%symbols(index)%value = &
                liric_f64_immediate(context%session, 0.0_c_double)
        else if (value_kind == VALUE_DERIVED) then
            ! A class(t) dummy is initially registered from the procedure
            ! signature, before its specification-part declaration binds the
            ! class descriptor. Keep that signature update harmless; the
            ! declaration path above installs the descriptor once the declared
            ! type is available.
            call set_empty(error_msg)
        else
            error_msg = 'unsupported parameter declaration value kind'
            return
        end if
        call set_empty(error_msg)
    end subroutine update_parameter_symbol
    subroutine define_i32_symbol(context, name, error_msg)
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: name
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: index
        if (find_symbol_same_scope(context, name) > 0) then
            error_msg = 'duplicate integer declaration: '//trim(name)
            return
        end if
        call grow_symbols(context)
        index = context%symbol_count + 1
        context%symbols(index)%name = trim(name)
        context%symbols(index)%value_kind = VALUE_I32
        context%symbols(index)%value = i32_immediate(context%session, 0_c_int64_t)
        context%symbol_count = index
        call set_empty(error_msg)
    end subroutine define_i32_symbol

    subroutine define_i64_symbol(context, name, error_msg)
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: name
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: index
        if (find_symbol_same_scope(context, name) > 0) then
            error_msg = 'duplicate integer(8) declaration: '//trim(name)
            return
        end if
        call grow_symbols(context)
        index = context%symbol_count + 1
        context%symbols(index)%name = trim(name)
        context%symbols(index)%value_kind = VALUE_I64
        context%symbols(index)%value = i64_immediate(context%session, 0_c_int64_t)
        context%symbol_count = index
        call set_empty(error_msg)
    end subroutine define_i64_symbol

    subroutine define_i8_symbol(context, name, error_msg)
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: name
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: index
        if (find_symbol_same_scope(context, name) > 0) then
            error_msg = 'duplicate integer(1) declaration: '//trim(name)
            return
        end if
        call grow_symbols(context)
        index = context%symbol_count + 1
        context%symbols(index)%name = trim(name)
        context%symbols(index)%value_kind = VALUE_I8
        context%symbols(index)%value = i8_immediate(context%session, 0_c_int64_t)
        context%symbol_count = index
        call set_empty(error_msg)
    end subroutine define_i8_symbol

    subroutine define_i16_symbol(context, name, error_msg)
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: name
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: index
        if (find_symbol_same_scope(context, name) > 0) then
            error_msg = 'duplicate integer(2) declaration: '//trim(name)
            return
        end if
        call grow_symbols(context)
        index = context%symbol_count + 1
        context%symbols(index)%name = trim(name)
        context%symbols(index)%value_kind = VALUE_I16
        context%symbols(index)%value = i16_immediate(context%session, 0_c_int64_t)
        context%symbol_count = index
        call set_empty(error_msg)
    end subroutine define_i16_symbol

    subroutine define_f32_symbol(context, name, error_msg)
        use, intrinsic :: iso_c_binding, only: c_float
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: name
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: index
        if (find_symbol_same_scope(context, name) > 0) then
            error_msg = 'duplicate real declaration: '//trim(name)
            return
        end if
        call grow_symbols(context)
        index = context%symbol_count + 1
        context%symbols(index)%name = trim(name)
        context%symbols(index)%value_kind = VALUE_F32
        context%symbols(index)%value = liric_f32_immediate(context%session, &
            0.0_c_float)
        context%symbol_count = index
        call set_empty(error_msg)
    end subroutine define_f32_symbol
    subroutine define_f64_symbol(context, name, error_msg)
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: name
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: index
        if (find_symbol_same_scope(context, name) > 0) then
            error_msg = 'duplicate real declaration: '//trim(name)
            return
        end if
        call grow_symbols(context)
        index = context%symbol_count + 1
        context%symbols(index)%name = trim(name)
        context%symbols(index)%value_kind = VALUE_F64
        context%symbols(index)%value = liric_f64_immediate(context%session, &
            0.0_c_double)
        context%symbol_count = index
        call set_empty(error_msg)
    end subroutine define_f64_symbol
    subroutine define_c4_symbol(context, name, error_msg)
        use, intrinsic :: iso_c_binding, only: c_float
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: name
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: index
        if (find_symbol_same_scope(context, name) > 0) then
            error_msg = 'duplicate complex declaration: '//trim(name)
            return
        end if
        call grow_symbols(context)
        index = context%symbol_count + 1
        context%symbols(index)%name = trim(name)
        context%symbols(index)%value_kind = VALUE_C4
        ! Alloca re and im slots; re in address, im in element_address.
        if (.not. emit_alloca_bytes(context%session, &
            i64_immediate(context%session, 8_c_int64_t), &
            context%symbols(index)%address, error_msg)) return
        if (.not. emit_ptr_offset(context%session, &
            context%symbols(index)%address, 4_c_int64_t, &
            context%symbols(index)%element_address, error_msg)) return
        context%symbols(index)%has_address = .true.
        context%symbols(index)%value = liric_f32_immediate(context%session, 0.0_c_float)
        context%symbol_count = index
        call set_empty(error_msg)
    end subroutine define_c4_symbol
    subroutine define_c8_symbol(context, name, error_msg)
        use, intrinsic :: iso_c_binding, only: c_double
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: name
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: index
        if (find_symbol_same_scope(context, name) > 0) then
            error_msg = 'duplicate complex declaration: '//trim(name)
            return
        end if
        call grow_symbols(context)
        index = context%symbol_count + 1
        context%symbols(index)%name = trim(name)
        context%symbols(index)%value_kind = VALUE_C8
        if (.not. emit_alloca_bytes(context%session, &
            i64_immediate(context%session, 16_c_int64_t), &
            context%symbols(index)%address, error_msg)) return
        if (.not. emit_ptr_offset(context%session, &
            context%symbols(index)%address, 8_c_int64_t, &
            context%symbols(index)%element_address, error_msg)) return
        context%symbols(index)%has_address = .true.
        context%symbols(index)%value = liric_f64_immediate(context%session, 0.0_c_double)
        context%symbol_count = index
        call set_empty(error_msg)
    end subroutine define_c8_symbol
    subroutine define_logical_symbol(context, name, error_msg)
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: name
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: index
        if (find_symbol_same_scope(context, name) > 0) then
            error_msg = 'duplicate logical declaration: '//trim(name)
            return
        end if
        call grow_symbols(context)
        index = context%symbol_count + 1
        context%symbols(index)%name = trim(name)
        context%symbols(index)%value_kind = VALUE_LOGICAL
        context%symbols(index)%value = i32_immediate(context%session, 0_c_int64_t)
        context%symbol_count = index
        call set_empty(error_msg)
    end subroutine define_logical_symbol
    subroutine lower_assignment(arena, node, context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(assignment_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: value
        character(len=:), allocatable :: name
        integer :: symbol_index
        logical :: handled

        call try_lower_overloaded_assignment(arena, node, context, handled, &
            error_msg)
        if (handled .or. len_trim(error_msg) > 0) return
        ! A spec-section assignment that defines a statement function (the
        ! explicit-program path leaves it as an assignment) emits no code; the
        ! body is inlined at each call site instead.
        if (is_statement_function_definition(arena, node, context)) then
            call set_empty(error_msg)
            return
        end if
        select type (target => arena%entries(node%target_index)%node)
            type is (call_or_subscript_node)
            if (target%base_expr_index > 0) then
                call lower_derived_component_element_assignment(arena, node, &
                    target, context, error_msg)
                return
            end if
            call lower_vector_subscript_assignment(arena, node, target, context, &
                handled, error_msg)
            if (handled .or. len_trim(error_msg) > 0) return
            if (is_call_or_subscript_array_section(arena, target, context)) then
                call lower_call_or_subscript_section_assignment(arena, node, target, &
                    context, error_msg)
                return
            end if
            if (target%is_array_access) then
                call lower_array_element_assignment(arena, node, target, context, &
                    value, error_msg)
                return
            end if
            if (is_declared_array_element_ref(target, context)) then
                ! a(i,j) = ... where FortFront left is_array_access unset (array
                ! element write in a program that also defines a module, mirroring
                ! the read-side fallback).
                call lower_array_element_assignment(arena, node, target, context, &
                    value, error_msg)
                return
            end if
            if (allocated(target%name) .and. allocated(target%arg_indices)) then
                symbol_index = find_symbol_compat(context, target%name)
                if (symbol_index > 0) then
                    if (context%symbols(symbol_index)%is_allocatable) then
                        call lower_array_element_assignment(arena, node, target, &
                            context, value, error_msg)
                        return
                    end if
                end if
            end if
            type is (component_access_node)
            call try_lower_complex_component_write(arena, node, target, context, &
                handled, error_msg)
            if (handled .or. len_trim(error_msg) > 0) return
            call lower_derived_component_assignment(arena, node, target, context, &
                value, error_msg)
            return
            type is (array_slice_node)
            if (is_character_substring(arena, node%target_index, context)) then
                call lower_character_substring_assignment(arena, node, context, &
                    error_msg)
                return
            end if
            call lower_array_section_assignment(arena, node, target, context, &
                error_msg)
            return
        end select
        call identifier_name(arena, node%target_index, name, error_msg)
        if (len_trim(error_msg) > 0) return
        symbol_index = resolve_symbol_at_node(context, node%target_index, name)
        if (symbol_index <= 0) symbol_index = find_symbol_compat(context, name)
        if (symbol_index <= 0) then
            error_msg = 'assignment target was not declared: '//trim(name)
            return
        end if
        if (context%symbols(symbol_index)%is_allocatable .and. &
            context%symbols(symbol_index)%array_rank > 0) then
            if (context%symbols(symbol_index)%is_derived) then
                call lower_derived_array_whole_assignment(arena, node, context, &
                    symbol_index, handled, error_msg)
                if (handled .or. len_trim(error_msg) > 0) return
            end if
            if (context%symbols(symbol_index)%value_kind == VALUE_CHARACTER) then
                call lower_allocatable_character_array_assignment(arena, node, &
                    symbol_index, context, handled, error_msg)
                if (handled .or. len_trim(error_msg) > 0) return
            end if
            if (is_alloc_array_result_call(arena, node%value_index, context)) then
                call lower_alloc_array_result_assignment(arena, node%value_index, &
                    symbol_index, context, error_msg)
                return
            end if
            if (node_exists(arena, node%value_index)) then
                select type (rhs => arena%entries(node%value_index)%node)
                    type is (array_literal_node)
                    call lower_allocatable_constructor_assignment(arena, rhs, &
                        symbol_index, context, error_msg)
                    return
                end select
            end if
            if (is_scalar_broadcast_to_allocatable(arena, node%value_index, &
                context)) then
                call lower_allocatable_scalar_broadcast(arena, node%value_index, &
                    symbol_index, context, error_msg)
                return
            end if
            if (allocatable_assignment_rhs_is_matmul(arena, node)) then
                call lower_allocatable_matmul_assignment(arena, node, symbol_index, &
                    context, error_msg)
                return
            end if
            call lower_allocatable_elementwise_assignment(arena, node, &
                symbol_index, context, error_msg)
            return
        end if
        if (context%symbols(symbol_index)%is_array) then
            if (is_array_result_call(arena, node%value_index, context)) then
                call lower_array_result_call(arena, node%value_index, &
                    symbol_index, context, error_msg)
                return
            end if
            if (node_exists(arena, node%value_index)) then
                select type (val => arena%entries(node%value_index)%node)
                    type is (array_literal_node)
                    ! A runtime-sized automatic array has no compile-time extent
                    ! to match the constructor against; its whole-array path
                    ! stores the elements into the entry-sized storage.
                    if (context%symbols(symbol_index)%is_runtime_array) then
                        call lower_array_whole_assignment(arena, node, &
                            symbol_index, context, error_msg)
                        return
                    end if
                    call lower_array_constructor_assignment(arena, val, &
                        symbol_index, context, error_msg)
                    return
                class default
                    call lower_array_whole_assignment(arena, node, symbol_index, &
                        context, error_msg)
                    return
                end select
            end if
            if (is_identifier(arena, node%target_index)) then
                call unsupported_feature_error('array assignment target', &
                    get_node_line(arena, node%target_index), &
                    get_node_column(arena, node%target_index), &
                    'whole-array assignment is not supported', error_msg)
            else
                call unsupported_feature_error('array assignment target', &
                    node%line, node%column, &
                    'whole-array assignment is not '// &
                    'supported', error_msg)
            end if
            return
        end if
        if (context%symbols(symbol_index)%is_derived) then
            ! Assignment to an unallocated scalar allocatable derived target
            ! auto-allocates it first (F2018 10.2.1.3), giving the whole-derived
            ! copy a heap instance to write into.
            if (context%symbols(symbol_index)%is_allocatable .and. &
                context%symbols(symbol_index)%array_rank == 0 .and. &
                .not. context%symbols(symbol_index)%has_address) then
                call lower_allocate_scalar_derived(symbol_index, context, error_msg)
                if (len_trim(error_msg) > 0) return
            end if
            if (is_derived_result_call(arena, node%value_index, context)) then
                call lower_derived_result_call(arena, node%value_index, &
                    symbol_index, context, error_msg)
                return
            end if
            call lower_derived_whole_assignment(arena, node, context, &
                symbol_index, handled, error_msg)
            if (handled .or. len_trim(error_msg) > 0) return
            call lower_derived_whole_assignment_diagnostic(arena, node, &
                context, symbol_index, &
                error_msg)
            return
        end if
        if (context%symbols(symbol_index)%value_kind == VALUE_I8) then
            call lower_i8_expression(arena, node%value_index, context, value, &
                error_msg)
        else if (context%symbols(symbol_index)%value_kind == VALUE_I16) then
            call lower_i16_expression(arena, node%value_index, context, value, &
                error_msg)
        else if (context%symbols(symbol_index)%value_kind == VALUE_I64) then
            call lower_i64_expression(arena, node%value_index, context, value, &
                error_msg)
        else if (context%symbols(symbol_index)%value_kind == VALUE_F32) then
            call lower_f32_expression(arena, node%value_index, context, value, &
                error_msg)
        else if (context%symbols(symbol_index)%value_kind == VALUE_F64) then
            call lower_f64_expression(arena, node%value_index, context, value, &
                error_msg)
        else if (context%symbols(symbol_index)%value_kind == VALUE_LOGICAL) then
            call lower_logical_expression(arena, node%value_index, context, &
                value, error_msg)
        else if (context%symbols(symbol_index)%value_kind == VALUE_CHARACTER) then
            call lower_character_assignment(arena, node, context, symbol_index, &
                error_msg)
            return
        else if (context%symbols(symbol_index)%value_kind == VALUE_C_PTR) then
            call lower_c_ptr_expression(arena, node%value_index, context, value, &
                error_msg)
        else if (context%symbols(symbol_index)%value_kind == VALUE_C4) then
            call lower_c4_assignment(arena, node%value_index, symbol_index, &
                context, error_msg)
            return
        else if (context%symbols(symbol_index)%value_kind == VALUE_C8) then
            call lower_c8_assignment(arena, node%value_index, symbol_index, &
                context, error_msg)
            return
        else if (scalar_real_expr_kind(arena, node%value_index, context) /= &
                 SCALAR_REAL_NONE) then
            ! Integer target with a real rhs: assignment converts by
            ! truncation toward zero (F2018 10.2.1.3), unlike subscript or
            ! bound positions where a real is rejected.
            block
                type(lr_operand_desc_t) :: real_value, wide_value
                if (scalar_real_expr_kind(arena, node%value_index, context) &
                    == VALUE_F64) then
                    call lower_f64_expression(arena, node%value_index, context, &
                        wide_value, error_msg)
                else
                    call lower_f32_expression(arena, node%value_index, context, &
                        real_value, error_msg)
                    if (len_trim(error_msg) > 0) return
                    if (.not. emit_liric_f32_to_f64(context%session, real_value, &
                        wide_value, error_msg)) return
                end if
                if (len_trim(error_msg) > 0) return
                if (.not. emit_liric_f64_to_i32(context%session, wide_value, &
                    value, error_msg)) return
            end block
        else
            call lower_i32_expression(arena, node%value_index, context, value, &
                error_msg)
        end if
        if (len_trim(error_msg) > 0) return
        context%symbols(symbol_index)%value = value
        if (context%symbols(symbol_index)%has_address .and. &
            context%symbols(symbol_index)%is_reference) then
            call store_reference_value(context, symbol_index, value, error_msg)
            if (len_trim(error_msg) > 0) return
        end if
        call track_assigned_i32_constant(arena, node%value_index, context, &
            symbol_index)
        call set_empty(error_msg)
    end subroutine lower_assignment

    subroutine track_assigned_i32_constant(arena, value_index, context, symbol_index)
        ! Straight-line constant tracking for plain I32 scalars, scoped to unit
        ! linking: a unit number assigned to a variable (unit = 10) lets
        ! WRITE/READ/REWIND that refer to it by number vs. by name resolve to the
        ! same connection. A non-constant RHS clears the tracked value.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: value_index
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        integer(c_int64_t) :: constant_value
        character(len=:), allocatable :: eval_err

        if (context%symbols(symbol_index)%value_kind /= VALUE_I32) return
        if (context%symbols(symbol_index)%is_parameter) return
        call eval_i32_constant(arena, value_index, context, constant_value, eval_err)
        if (allocated(eval_err)) then
            if (len_trim(eval_err) > 0) then
                context%symbols(symbol_index)%has_unit_const = .false.
                return
            end if
        end if
        context%symbols(symbol_index)%has_unit_const = .true.
        context%symbols(symbol_index)%unit_const = int(constant_value)
    end subroutine track_assigned_i32_constant

    subroutine try_lower_overloaded_assignment(arena, node, context, handled, &
            error_msg)
        ! An interface assignment(=) maps `lhs = rhs` to a subroutine call
        ! specific(lhs, rhs) when the operand kinds match a registered overload.
        ! The LHS is the intent(out)/(inout) first dummy, so it is passed by
        ! reference exactly like any 2-argument subroutine call argument.
        type(ast_arena_t), intent(in) :: arena
        type(assignment_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        logical, intent(out) :: handled
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: slot, left_kind, right_kind
        character(len=:), allocatable :: specific
        type(lr_operand_desc_t), allocatable :: args(:)
        integer, allocatable :: copyback_indices(:)

        handled = .false.
        call set_empty(error_msg)
        if (context%operator_count == 0) return
        left_kind = operand_overload_kind(arena, node%target_index, context)
        right_kind = operand_overload_kind(arena, node%value_index, context)
        slot = find_operator(context, '=', left_kind, right_kind, .true.)
        if (slot == 0) return
        specific = operator_specific_name(context, slot)
        call prepare_reference_args(arena, [node%target_index, node%value_index], &
            context, VALUE_I32, specific, args, &
            copyback_indices, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_void_call(context%session, &
            call_emit_name(arena, specific, context), args, error_msg)) return
        call copy_back_reference_args(context, args, copyback_indices, error_msg)
        handled = len_trim(error_msg) == 0
    end subroutine try_lower_overloaded_assignment
    include 'session_program_lowering_arguments.inc'
    include 'session_program_lowering_assumed_shape_extent.inc'
    include 'session_program_lowering_assumed_shape_descriptor.inc'
    include 'session_program_lowering_character.inc'
    include 'session_program_lowering_deferred_char.inc'
    subroutine lower_function_return(node, context, error_msg)
        type(return_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: idx
        type(lr_operand_desc_t) :: result_value
        idx = context%current_function_result_index
        if (idx <= 0) then
            error_msg = 'function return without a tracked result symbol'
            return
        end if
        select case (context%symbols(idx)%value_kind)
        case (VALUE_I32, VALUE_LOGICAL, VALUE_F32, VALUE_F64)
            call load_function_result_value(context, idx, result_value, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_ret_i32_operand(context%session, result_value, &
                error_msg)) return
            context%current_block_terminated = .true.
        case (VALUE_I64)
            call load_function_result_value(context, idx, result_value, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_ret_i64_operand(context%session, result_value, &
                error_msg)) return
            context%current_block_terminated = .true.
        case default
            call unsupported_feature_error('return from non-scalar function', &
                node%line, node%column, &
                'only integer, logical and real '// &
                'function returns are supported', &
                error_msg)
        end select
    end subroutine lower_function_return
    subroutine lower_stop(arena, node, context, value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(stop_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        if (node%stop_code_index <= 0) then
            value = i32_immediate(context%session, 0_c_int64_t)
            call set_empty(error_msg)
        else
            call lower_i32_expression(arena, node%stop_code_index, context, &
                value, error_msg)
        end if
    end subroutine lower_stop

    subroutine emit_stop_banner(node, code_value, context, error_msg)
        ! gfortran writes a STOP banner to stderr (fd 2): "STOP <message>" for a
        ! character message, "STOP <n>" for any integer stop code. Bare stop
        ! prints nothing. Match that via dprintf(2, ...).
        type(stop_node), intent(in) :: node
        type(lr_operand_desc_t), intent(in) :: code_value
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: fa(3), fmtop, msgop
        character(len=:), allocatable :: msg_text
        integer(c_int32_t) :: fmt_gid, msg_gid
        character(len=64) :: gname

        call set_empty(error_msg)
        if (allocated(node%stop_message)) then
            call strip_literal_quotes(node%stop_message, msg_text)
            context%string_literal_count = context%string_literal_count + 1
            gname = ffc_unit_global_name( &
                context, 'stop.msg.', context%string_literal_count)
            call create_printf_format_global(context%session, trim(gname), &
                msg_text, msg_gid, error_msg)
            if (len_trim(error_msg) > 0) return
            msgop = printf_format_ptr(context%session, msg_gid)
            context%string_literal_count = context%string_literal_count + 1
            gname = ffc_unit_global_name( &
                context, 'stop.fmt.', context%string_literal_count)
            call create_printf_format_global(context%session, trim(gname), &
                'STOP %s'//achar(10), fmt_gid, &
                error_msg)
            if (len_trim(error_msg) > 0) return
            fmtop = printf_format_ptr(context%session, fmt_gid)
            fa(1) = i32_immediate(context%session, 2_c_int64_t)
            fa(2) = fmtop
            fa(3) = msgop
            if (.not. emit_dprintf(context%session, fa, error_msg)) return
        else if (node%stop_code_index > 0) then
            context%string_literal_count = context%string_literal_count + 1
            gname = ffc_unit_global_name( &
                context, 'stop.fmt.', context%string_literal_count)
            call create_printf_format_global(context%session, trim(gname), &
                'STOP %d'//achar(10), fmt_gid, &
                error_msg)
            if (len_trim(error_msg) > 0) return
            fmtop = printf_format_ptr(context%session, fmt_gid)
            fa(1) = i32_immediate(context%session, 2_c_int64_t)
            fa(2) = fmtop
            fa(3) = code_value
            if (.not. emit_dprintf(context%session, fa, error_msg)) return
        end if
        call set_empty(error_msg)
    end subroutine emit_stop_banner

    subroutine lower_error_stop(arena, node, context, value, error_msg)
        ! ERROR STOP terminates with an error code: the given integer code, or 1
        ! when none is supplied. A character-valued stop expression is a
        ! message, not an integer termination code, and uses the default code.
        type(ast_arena_t), intent(in) :: arena
        type(error_stop_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        if (node%error_code_index <= 0) then
            value = i32_immediate(context%session, 1_c_int64_t)
            call set_empty(error_msg)
        else if (is_character_operand(arena, node%error_code_index, context)) then
            value = i32_immediate(context%session, 1_c_int64_t)
            call set_empty(error_msg)
        else
            call lower_i32_expression(arena, node%error_code_index, context, &
                value, error_msg)
        end if
    end subroutine lower_error_stop

    subroutine emit_error_stop_banner(arena, node, code_value, context, error_msg)
        ! gfortran writes "ERROR STOP <message>" / "ERROR STOP <n>" / "ERROR STOP"
        ! to stderr. Mirror emit_stop_banner with the ERROR STOP prefix.
        type(ast_arena_t), intent(in) :: arena
        type(error_stop_node), intent(in) :: node
        type(lr_operand_desc_t), intent(in) :: code_value
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: fa(3), fmtop, msgop, dynamic_msg
        type(lr_operand_desc_t) :: dynamic_length
        character(len=:), allocatable :: msg_text
        logical :: literal_message
        integer(c_int32_t) :: fmt_gid, msg_gid
        character(len=64) :: gname

        call set_empty(error_msg)
        literal_message = .false.
        if (allocated(node%error_message)) then
            literal_message = len_trim(node%error_message) > 0
        end if
        if (literal_message) then
            call strip_literal_quotes(node%error_message, msg_text)
            context%string_literal_count = context%string_literal_count + 1
            gname = ffc_unit_global_name( &
                context, 'estop.msg.', context%string_literal_count)
            call create_printf_format_global(context%session, trim(gname), &
                msg_text, msg_gid, error_msg)
            if (len_trim(error_msg) > 0) return
            msgop = printf_format_ptr(context%session, msg_gid)
            context%string_literal_count = context%string_literal_count + 1
            gname = ffc_unit_global_name( &
                context, 'estop.fmt.', context%string_literal_count)
            call create_printf_format_global(context%session, trim(gname), &
                'ERROR STOP %s'//achar(10), fmt_gid, &
                error_msg)
            if (len_trim(error_msg) > 0) return
            fmtop = printf_format_ptr(context%session, fmt_gid)
            fa(1) = i32_immediate(context%session, 2_c_int64_t)
            fa(2) = fmtop
            fa(3) = msgop
            if (.not. emit_dprintf(context%session, fa, error_msg)) return
        else if (node%error_code_index > 0 .and. &
                 is_character_operand(arena, node%error_code_index, context)) then
            call char_expr_operands(arena, node%error_code_index, context, &
                                    dynamic_msg, dynamic_length, error_msg)
            if (len_trim(error_msg) > 0) return
            context%string_literal_count = context%string_literal_count + 1
            gname = ffc_unit_global_name( &
                context, 'estop.fmt.', context%string_literal_count)
            call create_printf_format_global(context%session, trim(gname), &
                'ERROR STOP %s'//achar(10), fmt_gid, &
                error_msg)
            if (len_trim(error_msg) > 0) return
            fmtop = printf_format_ptr(context%session, fmt_gid)
            ! The runtime maps Fortran unit 0 to stderr; descriptor 2 is the
            ! POSIX file descriptor used by the variadic dprintf path.
            fa(1) = i32_immediate(context%session, 0_c_int64_t)
            if (.not. emit_liric_write_string_operand(context%session, fa(1), &
                    fmtop, dynamic_msg, error_msg)) return
        else if (node%error_code_index > 0) then
            context%string_literal_count = context%string_literal_count + 1
            gname = ffc_unit_global_name( &
                context, 'estop.fmt.', context%string_literal_count)
            call create_printf_format_global(context%session, trim(gname), &
                'ERROR STOP %d'//achar(10), fmt_gid, &
                error_msg)
            if (len_trim(error_msg) > 0) return
            fmtop = printf_format_ptr(context%session, fmt_gid)
            fa(1) = i32_immediate(context%session, 2_c_int64_t)
            fa(2) = fmtop
            fa(3) = code_value
            if (.not. emit_dprintf(context%session, fa, error_msg)) return
        end if
        call set_empty(error_msg)
    end subroutine emit_error_stop_banner
    include 'session_program_lowering_write_ops.inc'
    include 'session_program_lowering_io_typecheck.inc'
    include 'session_program_lowering_inquire.inc'
    include 'session_program_lowering_read_ops.inc'
    include 'session_program_lowering_read_al.inc'
    include 'session_program_lowering_print_expr.inc'
    include 'session_program_lowering_expr_lowering.inc'
    include 'session_program_lowering_complex.inc'
    include 'session_program_lowering_complex_arrays.inc'
    include 'session_program_lowering_integer.inc'
    include 'session_program_lowering_intrinsics.inc'
    include 'session_program_lowering_intrinsics_extra.inc'
    include 'session_program_lowering_logical_reduction.inc'
    include 'session_program_lowering_transfer.inc'
    include 'session_program_lowering_c_ptr.inc'
    include 'session_program_lowering_pointer.inc'
    include 'session_program_lowering_proc_dummy.inc'
    include 'session_program_lowering_statement_function.inc'
    subroutine lower_subroutine_call(arena, node_index, context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: name, call_name
        integer, allocatable :: arg_indices(:)
        integer, allocatable :: alt_labels(:)
        type(lr_operand_desc_t), allocatable :: args(:)
        integer, allocatable :: copyback_indices(:)
        integer :: call_arg_count
        integer :: call_arg_kinds(MAX_PROC_ARGS)
        integer :: call_arg_ranks(MAX_PROC_ARGS)
        integer :: i
        call get_subroutine_call_name(arena, node_index, name, error_msg)
        if (len_trim(error_msg) > 0) return
        ! Indirect subroutine call through a procedure pointer (B3d).
        if (is_proc_pointer_call(context, name)) then
            call lower_void_proc_ptr_call(arena, node_index, context, error_msg)
            return
        end if
        call get_subroutine_call_arg_indices(arena, node_index, arg_indices, &
            error_msg)
        if (len_trim(error_msg) > 0) return
        ! `*label` actual arguments are alternate-return specifiers, not passed
        ! arguments: split them off before argument analysis (#353).
        call split_alt_return_args(arena, arg_indices, alt_labels)
        ! Resolve generic -> specific (#249 B7c).
        call_arg_count = 0
        call_arg_kinds = VALUE_I32
        call_arg_ranks = 0
        if (allocated(arg_indices)) then
            call_arg_count = min(size(arg_indices), MAX_PROC_ARGS)
            do i = 1, call_arg_count
                call_arg_kinds(i) = expression_value_kind(arena, arg_indices(i), &
                    context, VALUE_I32)
                call_arg_ranks(i) = expression_value_rank(arena, arg_indices(i), &
                    context)
            end do
        end if
        call reject_monomorphized_call(context, trim(name), &
                                       get_node_line(arena, node_index), &
                                       get_node_column(arena, node_index), &
                                       error_msg)
        if (len_trim(error_msg) > 0) return
        call_name = degeneric_call_name(context, name, call_arg_count, call_arg_kinds, &
            call_arg_ranks)
        if (same_name(call_name, 'get_command_argument')) then
            call lower_get_command_argument(arena, arg_indices, context, error_msg)
            return
        end if
        if (same_name(call_name, 'c_f_strpointer')) then
            call lower_c_f_strpointer(arena, arg_indices, context, error_msg)
            return
        end if
        if (same_name(call_name, 'c_f_pointer')) then
            call lower_c_f_pointer(arena, arg_indices, context, error_msg)
            return
        end if
        if (same_name(call_name, 'move_alloc')) then
            call lower_move_alloc(arena, arg_indices, context, error_msg)
            return
        end if
        if (same_name(call_name, 'cpu_time')) then
            call lower_cpu_time(arena, arg_indices, context, error_msg)
            return
        end if
        if (same_name(call_name, 'system_clock')) then
            call lower_system_clock(arena, arg_indices, context, error_msg)
            return
        end if
        if (same_name(call_name, 'random_number')) then
            call lower_random_number(arena, arg_indices, context, error_msg)
            return
        end if
        if (same_name(call_name, 'random_seed')) then
            call lower_random_seed(arena, arg_indices, context, error_msg)
            return
        end if
        if (is_method_subroutine_call(call_name)) then
            call lower_method_subroutine_call(arena, call_name, arg_indices, &
                context, error_msg)
            return
        end if
        if (external_procedure_index(context, call_name) > 0) then
            if (context%external_procedures( &
                external_procedure_index(context, call_name))%by_reference) then
                call lower_module_proc_void_call(arena, node_index, &
                    external_procedure_index(context, call_name), context, &
                    error_msg)
            else
                call lower_external_void_call(arena, node_index, &
                    external_procedure_index(context, call_name), context, &
                    error_msg)
            end if
            return
        end if
        if (same_name(call_name, 'exit')) then
            call lower_exit_intrinsic(arena, arg_indices, context, error_msg)
            return
        end if
        call prepare_reference_args(arena, arg_indices, context, VALUE_I32, &
            call_name, args, copyback_indices, error_msg)
        if (len_trim(error_msg) > 0) return
        if (size(alt_labels) > 0) then
            call emit_alt_return_call(arena, node_index, &
                call_emit_name(arena, call_name, context), args, alt_labels, &
                copyback_indices, context, error_msg)
            return
        end if
        if (.not. emit_call_with_optional_padding(context, &
            call_emit_name(arena, call_name, context), args, error_msg)) &
            return
        call copy_back_reference_args(context, args, copyback_indices, error_msg)
    end subroutine lower_subroutine_call

    subroutine lower_exit_intrinsic(arena, arg_indices, context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, allocatable, intent(in) :: arg_indices(:)
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: status
        call set_empty(error_msg)
        if (allocated(arg_indices)) then
            if (size(arg_indices) > 0) then
                call lower_i32_expression(arena, arg_indices(1), context, &
                    status, error_msg)
                if (len_trim(error_msg) > 0) return
            else
                status = i32_immediate(context%session, 0_c_int64_t)
            end if
        else
            status = i32_immediate(context%session, 0_c_int64_t)
        end if
        if (.not. emit_void_call(context%session, 'exit', [status], error_msg)) &
            return
    end subroutine lower_exit_intrinsic

    subroutine lower_cpu_time(arena, arg_indices, context, error_msg)
        ! cpu_time(t): t = processor time in seconds (real). The intent(out)
        ! argument must be a declared real scalar.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: arg_indices(:)
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: value
        type(lr_operand_desc_t) :: wide_value
        integer :: symbol_index

        call intrinsic_out_scalar(arena, arg_indices, context, 'cpu_time', &
            VALUE_F32, symbol_index, error_msg, allow_f64=.true.)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_cpu_time_value(context%session, value, error_msg)) return
        if (context%symbols(symbol_index)%value_kind == VALUE_F64) then
            if (.not. emit_liric_f32_to_f64(context%session, value, wide_value, &
                error_msg)) return
            value = wide_value
        end if
        call store_intrinsic_scalar_result(context, symbol_index, value, error_msg)
    end subroutine lower_cpu_time

    subroutine lower_system_clock(arena, arg_indices, context, error_msg)
        ! system_clock(count): count = an integer tick counter. Only the count
        ! argument is supported; count_rate and count_max are not.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: arg_indices(:)
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: value
        integer :: symbol_index

        call intrinsic_out_scalar(arena, arg_indices, context, 'system_clock', &
            VALUE_I32, symbol_index, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_system_clock_value(context%session, value, error_msg)) return
        call store_intrinsic_scalar_result(context, symbol_index, value, error_msg)
    end subroutine lower_system_clock

    subroutine lower_random_number(arena, arg_indices, context, error_msg)
        ! random_number(harvest): harvest = a pseudo-random real in [0,1).
        ! Only a default-real scalar argument is supported; an array harvest
        ! or a real(8) harvest is diagnosed by intrinsic_out_scalar.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: arg_indices(:)
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: value
        integer :: symbol_index

        call intrinsic_out_scalar(arena, arg_indices, context, 'random_number', &
            VALUE_F32, symbol_index, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_random_number_value(context%session, value, error_msg)) return
        call store_intrinsic_scalar_result(context, symbol_index, value, error_msg)
    end subroutine lower_random_number

    subroutine lower_random_seed(arena, arg_indices, context, error_msg)
        ! random_seed([size=n] | [put=seed] | [get=seed]): control the
        ! generator RANDOM_NUMBER draws from. The runtime owns the seed state
        ! (_ffc_random_seed_*), so the size and the PUT/GET semantics live in
        ! one place instead of being duplicated in emitted code. Exactly one
        ! keyword per call is supported, which covers every corpus use.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: arg_indices(:)
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: value, address
        character(len=:), allocatable :: kw_name
        integer :: kw_value, symbol_index

        call set_empty(error_msg)
        if (size(arg_indices) == 0) then
            if (.not. emit_random_seed_default(context%session, error_msg)) return
            return
        end if
        if (size(arg_indices) /= 1) then
            error_msg = 'random_seed supports one of size=, put= or get= per call'
            return
        end if
        call reshape_keyword_arg(arena, arg_indices(1), kw_name, kw_value)
        if (kw_name == 'size') then
            call intrinsic_out_scalar(arena, [kw_value], context, 'random_seed', &
                VALUE_I32, symbol_index, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_random_seed_size(context%session, value, error_msg)) return
            call store_intrinsic_scalar_result(context, symbol_index, value, &
                error_msg)
            return
        end if
        if (kw_name /= 'put') then
            if (kw_name /= 'get') then
                error_msg = 'random_seed argument must be size=, put= or get='
                return
            end if
        end if
        call random_seed_array_address(arena, kw_value, context, address, error_msg)
        if (len_trim(error_msg) > 0) return
        if (kw_name == 'put') then
            if (.not. emit_random_seed_put(context%session, address, error_msg)) &
                return
        else
            if (.not. emit_random_seed_get(context%session, address, error_msg)) &
                return
        end if
    end subroutine lower_random_seed

    subroutine random_seed_array_address(arena, node_index, context, address, &
            error_msg)
        ! Base address of the integer seed array a PUT/GET actual names. An
        ! allocatable's storage is behind its descriptor, so load the data
        ! pointer; an explicit-shape array is its own storage.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: address
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: id_name
        integer :: symbol_index

        if (.not. is_identifier(arena, node_index)) then
            error_msg = 'random_seed put=/get= argument must be an integer array'
            return
        end if
        call get_identifier_name(arena, node_index, id_name, error_msg)
        if (len_trim(error_msg) > 0) return
        symbol_index = resolve_symbol_at_node(context, node_index, id_name)
        if (symbol_index <= 0) then
            error_msg = 'random_seed argument is not declared: '//trim(id_name)
            return
        end if
        if (.not. context%symbols(symbol_index)%is_array) then
            error_msg = 'random_seed put=/get= argument must be an array: '// &
                        trim(id_name)
            return
        end if
        if (context%symbols(symbol_index)%value_kind /= VALUE_I32) then
            error_msg = 'random_seed put=/get= argument must be a default '// &
                        'integer array: '//trim(id_name)
            return
        end if
        if (context%symbols(symbol_index)%is_allocatable) then
            if (.not. emit_ptr_load(context%session, &
                context%symbols(symbol_index)%allocatable_descriptor_address, &
                address, error_msg)) return
            call set_empty(error_msg)
            return
        end if
        address = context%symbols(symbol_index)%element_address
        call set_empty(error_msg)
    end subroutine random_seed_array_address

    subroutine intrinsic_out_scalar(arena, arg_indices, context, name, kind, &
            symbol_index, error_msg, allow_f64)
        ! Resolve the single intent(out) scalar argument of a timing intrinsic
        ! and verify its declared kind.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: arg_indices(:)
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: name
        integer, intent(in) :: kind
        integer, intent(out) :: symbol_index
        character(len=:), allocatable, intent(out) :: error_msg
        logical, intent(in), optional :: allow_f64
        character(len=:), allocatable :: var_name
        logical :: accepts_f64

        symbol_index = 0
        accepts_f64 = .false.
        if (present(allow_f64)) accepts_f64 = allow_f64
        if (size(arg_indices) /= 1) then
            error_msg = trim(name)//' requires exactly one scalar argument'
            return
        end if
        if (.not. is_identifier(arena, arg_indices(1))) then
            error_msg = trim(name)//' argument must be a scalar variable'
            return
        end if
        call get_identifier_name(arena, arg_indices(1), var_name, error_msg)
        if (len_trim(error_msg) > 0) return
        symbol_index = find_symbol_compat(context, var_name)
        if (symbol_index <= 0) then
            error_msg = trim(name)//' argument is not declared: '//trim(var_name)
            return
        end if
        if (context%symbols(symbol_index)%value_kind /= kind .and. &
            (.not. accepts_f64 .or. &
             context%symbols(symbol_index)%value_kind /= VALUE_F64)) then
            error_msg = trim(name)//' argument has the wrong type: '//trim(var_name)
            return
        end if
        call set_empty(error_msg)
    end subroutine intrinsic_out_scalar

    subroutine store_intrinsic_scalar_result(context, symbol_index, value, &
            error_msg)
        ! Write a freshly computed scalar into its symbol, persisting through the
        ! backing address when the variable lives in memory.
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        type(lr_operand_desc_t), intent(in) :: value
        character(len=:), allocatable, intent(out) :: error_msg

        context%symbols(symbol_index)%value = value
        if (context%symbols(symbol_index)%has_address) then
            call store_reference_value(context, symbol_index, value, error_msg)
            if (len_trim(error_msg) > 0) return
        end if
        call set_empty(error_msg)
    end subroutine store_intrinsic_scalar_result

    logical function emit_call_with_optional_padding(context, name, args, &
            error_msg) result(ok)
        ! Emit a void call to a contained subroutine, padding omitted trailing
        ! optional dummies with null pointers up to the callee's declared
        ! parameter count.
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: name
        type(lr_operand_desc_t), intent(in) :: args(:)
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t), allocatable :: padded(:)
        type(lr_operand_desc_t) :: nullptr
        integer :: pcount
        integer :: j

        ok = .false.
        pcount = proc_param_count(context, name)
        ! A module procedure is not a nested procedure of this scope, so its
        ! dummy count only resolves through the arena. Without it an omitted
        ! trailing optional would be left unpadded and the callee would read an
        ! uninitialized parameter slot instead of a null reference.
        if (pcount < 0) pcount = arena_proc_param_count(context%arena, &
                                                       unmangled_proc_name(name))
        if (pcount <= size(args)) then
            ok = emit_void_call(context%session, name, args, error_msg)
            return
        end if

        nullptr%kind = LR_OP_KIND_IMM_I64
        nullptr%payload = 0_c_int64_t
        nullptr%typ = lr_type_ptr_s(context%session%handle)
        nullptr%global_offset = 0_c_int64_t
        allocate (padded(pcount))
        if (size(args) > 0) padded(1:size(args)) = args
        do j = size(args) + 1, pcount
            padded(j) = nullptr
        end do
        ok = emit_void_call(context%session, name, padded, error_msg)
    end function emit_call_with_optional_padding

    function unmangled_proc_name(name) result(source_name)
        ! Source name behind a module procedure's emitted symbol
        ! `__<module>_MOD_<name>`; any other name is returned unchanged.
        character(len=*), intent(in) :: name
        character(len=:), allocatable :: source_name
        integer :: marker

        source_name = trim(name)
        if (len(source_name) < 8) return
        if (source_name(1:2) /= '__') return
        marker = index(source_name, '_MOD_')
        if (marker <= 0) return
        source_name = source_name(marker + 5:)
    end function unmangled_proc_name
    recursive subroutine lower_i1_condition(arena, node_index, context, &
            value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: lhs
        type(lr_operand_desc_t) :: rhs
        integer(c_int) :: pred
        character(len=:), allocatable :: bin_op
        integer :: bin_left, bin_right, bin_line, bin_col
        integer :: compare_kind
        if (.not. node_exists(arena, node_index)) then
            error_msg = 'condition index does not reference an AST node'
            return
        end if
        if (is_binary_op(arena, node_index)) then
            call get_binary_op_info(arena, node_index, bin_op, bin_left, &
                bin_right, bin_line, bin_col, error_msg)
            if (len_trim(error_msg) > 0) return
            if (trim(adjustl(lowercase_text(bin_op))) == '.not.') then
                ! Unary .not. is parsed as a binary op with a virtual operand;
                ! the real condition is the right operand. Invert it.
                call lower_i1_condition(arena, bin_right, context, lhs, &
                    error_msg)
                if (len_trim(error_msg) > 0) return
                rhs = lhs
                rhs%kind = LR_OP_KIND_IMM_I64
                rhs%payload = 0_c_int64_t
                if (.not. emit_liric_i32_icmp(context%session, LR_CMP_EQ, lhs, &
                    rhs, value, error_msg)) return
                call set_empty(error_msg)
                return
            end if
            block
                integer :: op_slot
                if (overloaded_operator_slot(arena, node_index, context, &
                                              op_slot)) then
                    if (operator_return_kind(context, op_slot) == &
                        VALUE_LOGICAL) then
                        call lower_overloaded_operator(arena, node_index, &
                                                       op_slot, context, lhs, &
                                                       error_msg)
                        if (len_trim(error_msg) > 0) return
                        rhs = i32_immediate(context%session, 0_c_int64_t)
                        if (.not. emit_liric_i32_icmp(context%session, &
                            LR_CMP_NE, lhs, rhs, value, error_msg)) return
                        return
                    end if
                end if
            end block
            if (is_logical_connective(bin_op)) then
                ! .and./.or./.eqv./.neqv. combine two i1 sub-conditions. Both
                ! operands are lowered (Fortran does not short-circuit), then
                ! combined with the matching bitwise op on the i1 values.
                call lower_logical_connective(arena, bin_op, bin_left, &
                    bin_right, context, value, error_msg)
                return
            end if
            call check_comparison_operand_types(arena, bin_op, bin_left, &
                bin_right, bin_line, bin_col, context, error_msg)
            if (len_trim(error_msg) > 0) return
            ! A comparison with a character operand lowers through Fortran's
            ! blank-padded lexical ordering.
            if (is_character_operand(arena, bin_left, context) .or. &
                is_character_operand(arena, bin_right, context)) then
                call lower_character_condition(arena, bin_op, bin_left, &
                    bin_right, context, value, error_msg)
                return
            end if
            ! A comparison whose operands are real (including libm intrinsic
            ! calls such as sin(x) > cos(y)) lowers through the float compare
            ! path at the widest operand kind (F2018 10.1.5.2.1), so comparing
            ! a real(8) against a default real literal compares in f64 rather
            ! than narrowing the wide operand.
            compare_kind = wider_real_kind( &
                scalar_real_expr_kind(arena, bin_left, context), &
                scalar_real_expr_kind(arena, bin_right, context))
            if (compare_kind == VALUE_F32) then
                call lower_f32_expression(arena, bin_left, context, lhs, error_msg)
                if (len_trim(error_msg) > 0) return
                call lower_f32_expression(arena, bin_right, context, rhs, error_msg)
                if (len_trim(error_msg) > 0) return
                call real_compare_predicate(bin_op, pred, error_msg)
                if (len_trim(error_msg) > 0) return
                if (.not. emit_liric_f32_fcmp(context%session, pred, lhs, rhs, &
                    value, error_msg)) return
                return
            end if
            if (compare_kind == VALUE_F64) then
                ! lower_f64_operand promotes an integer or f32 operand into the
                ! f64 comparison rather than narrowing the wide side.
                call lower_f64_operand(arena, bin_left, context, lhs, error_msg)
                if (len_trim(error_msg) > 0) return
                call lower_f64_operand(arena, bin_right, context, rhs, error_msg)
                if (len_trim(error_msg) > 0) return
                call real_compare_predicate(bin_op, pred, error_msg)
                if (len_trim(error_msg) > 0) return
                if (.not. emit_liric_f64_fcmp(context%session, pred, lhs, rhs, &
                    value, error_msg)) return
                return
            end if
            ! A comparison with a non-default-width integer operand
            ! (integer(1)/(2)/(8)) lowers each side through its matching
            ! iN path so the icmp compares same-width operands; emit_liric_i32_icmp
            ! is width-agnostic (the LIRIC IR infers width from the operands).
            if (is_i64_binary_op(arena, node_index, context)) then
                call lower_i64_expression(arena, bin_left, context, lhs, error_msg)
                if (len_trim(error_msg) > 0) return
                call lower_i64_expression(arena, bin_right, context, rhs, error_msg)
                if (len_trim(error_msg) > 0) return
                call integer_compare_predicate(bin_op, pred, error_msg)
                if (len_trim(error_msg) > 0) return
                if (.not. emit_liric_i32_icmp(context%session, pred, lhs, rhs, &
                    value, error_msg)) return
                return
            end if
            if (is_i16_binary_op(arena, node_index, context)) then
                call lower_i16_expression(arena, bin_left, context, lhs, error_msg)
                if (len_trim(error_msg) > 0) return
                call lower_i16_expression(arena, bin_right, context, rhs, error_msg)
                if (len_trim(error_msg) > 0) return
                call integer_compare_predicate(bin_op, pred, error_msg)
                if (len_trim(error_msg) > 0) return
                if (.not. emit_liric_i32_icmp(context%session, pred, lhs, rhs, &
                    value, error_msg)) return
                return
            end if
            if (is_i8_binary_op(arena, node_index, context)) then
                call lower_i8_expression(arena, bin_left, context, lhs, error_msg)
                if (len_trim(error_msg) > 0) return
                call lower_i8_expression(arena, bin_right, context, rhs, error_msg)
                if (len_trim(error_msg) > 0) return
                call integer_compare_predicate(bin_op, pred, error_msg)
                if (len_trim(error_msg) > 0) return
                if (.not. emit_liric_i32_icmp(context%session, pred, lhs, rhs, &
                    value, error_msg)) return
                return
            end if
            call lower_i32_expression(arena, bin_left, context, lhs, error_msg)
            if (len_trim(error_msg) > 0) return
            call lower_i32_expression(arena, bin_right, context, rhs, error_msg)
            if (len_trim(error_msg) > 0) return
            call integer_compare_predicate(bin_op, pred, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_liric_i32_icmp(context%session, pred, lhs, rhs, &
                value, error_msg)) return
            return
        end if
        if (is_literal(arena, node_index)) then
            call lower_logical_expression(arena, node_index, context, lhs, &
                error_msg)
            if (len_trim(error_msg) > 0) return
            rhs = i32_immediate(context%session, 0_c_int64_t)
            if (.not. emit_liric_i32_icmp(context%session, LR_CMP_NE, lhs, &
                rhs, value, error_msg)) return
            return
        end if
        if (is_identifier(arena, node_index)) then
            call lower_logical_expression(arena, node_index, context, lhs, &
                error_msg)
            if (len_trim(error_msg) > 0) return
            rhs = i32_immediate(context%session, 0_c_int64_t)
            if (.not. emit_liric_i32_icmp(context%session, LR_CMP_NE, lhs, &
                rhs, value, error_msg)) return
            return
        end if
        select type (node => arena%entries(node_index)%node)
            type is (component_access_node)
            ! Scalar logical component used directly as a condition: x%flag.
            call lower_logical_expression(arena, node_index, context, lhs, &
                error_msg)
            if (len_trim(error_msg) > 0) return
            rhs = i32_immediate(context%session, 0_c_int64_t)
            if (.not. emit_liric_i32_icmp(context%session, LR_CMP_NE, lhs, &
                rhs, value, error_msg)) return
            type is (call_or_subscript_node)
            if (node%base_expr_index > 0) then
                ! Logical array-component element used as a condition: x%flag(i).
                call lower_logical_expression(arena, node_index, context, lhs, &
                    error_msg)
                if (len_trim(error_msg) > 0) return
                rhs = i32_immediate(context%session, 0_c_int64_t)
                if (.not. emit_liric_i32_icmp(context%session, LR_CMP_NE, lhs, &
                    rhs, value, error_msg)) return
            else if (is_present_call(arena, node_index)) then
                call lower_present_condition(arena, node_index, context, value, &
                    error_msg)
            else if ((node%is_array_access .and. &
                    array_access_value_kind(node, context) == VALUE_LOGICAL) &
                    .or. is_allocatable_element_ref(node, context)) then
                ! Logical array element used directly as a condition: flags(i).
                call lower_i32_array_element(arena, node, context, lhs, error_msg)
                if (len_trim(error_msg) > 0) return
                rhs = i32_immediate(context%session, 0_c_int64_t)
                if (.not. emit_liric_i32_icmp(context%session, LR_CMP_NE, lhs, &
                    rhs, value, error_msg)) return
            else if (allocated(node%name)) then
                ! present() aside, every other named call (allocated(), the
                ! ISO_C_BINDING associated forms, and a contained logical
                ! function) shares lower_logical_call's dispatch.
                call lower_logical_call(arena, node, context, lhs, error_msg)
                if (len_trim(error_msg) > 0) return
                rhs = i32_immediate(context%session, 0_c_int64_t)
                if (.not. emit_liric_i32_icmp(context%session, LR_CMP_NE, lhs, &
                    rhs, value, error_msg)) return
            else
                error_msg = 'direct LIRIC session IF condition supports '// &
                    'comparisons, logicals, and present()'
            end if
        class default
            error_msg = 'direct LIRIC session IF requires an integer '// &
                'comparison or logical expression'
        end select
    end subroutine lower_i1_condition

    logical function is_logical_connective(op)
        character(len=*), intent(in) :: op
        select case (trim(adjustl(lowercase_text(op))))
        case ('.and.', 'and', '.or.', 'or', '.eqv.', 'eqv', '.neqv.', 'neqv', &
              '.xor.', 'xor')
            is_logical_connective = .true.
        case default
            is_logical_connective = .false.
        end select
    end function is_logical_connective

    subroutine lower_logical_connective(arena, op, left_index, right_index, &
            context, value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: op
        integer, intent(in) :: left_index, right_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: lhs, rhs
        type(lr_operand_desc_t) :: xor_value
        integer(c_int) :: opcode

        call lower_i1_condition(arena, left_index, context, lhs, error_msg)
        if (len_trim(error_msg) > 0) return
        call lower_i1_condition(arena, right_index, context, rhs, error_msg)
        if (len_trim(error_msg) > 0) return
        select case (trim(adjustl(lowercase_text(op))))
        case ('.and.', 'and')
            opcode = LR_OP_AND
        case ('.or.', 'or')
            opcode = LR_OP_OR
        case ('.neqv.', 'neqv')
            opcode = LR_OP_XOR
        case ('.xor.', 'xor')
            opcode = LR_OP_XOR
        case ('.eqv.', 'eqv')
            ! a .eqv. b is .not. (a .neqv. b): xor then invert the i1.
            if (.not. emit_i32_binary(context%session, LR_OP_XOR, lhs, rhs, &
                xor_value, error_msg)) return
            if (.not. emit_liric_i32_icmp(context%session, LR_CMP_EQ, xor_value, &
                i32_immediate(context%session, 0_c_int64_t), value, &
                error_msg)) return
            return
        case default
            error_msg = 'unsupported logical connective: '//trim(op)
            return
        end select
        if (.not. emit_i32_binary(context%session, opcode, lhs, rhs, value, &
            error_msg)) return
    end subroutine lower_logical_connective

    subroutine identifier_name(arena, node_index, name, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        character(len=:), allocatable, intent(out) :: name
        character(len=:), allocatable, intent(out) :: error_msg
        if (.not. node_exists(arena, node_index)) then
            error_msg = 'identifier index does not reference an AST node'
            call set_empty(name)
            return
        end if
        if (is_identifier(arena, node_index)) then
            call get_identifier_name(arena, node_index, name, error_msg)
            return
        end if
        select type (node => arena%entries(node_index)%node)
            type is (call_or_subscript_node)
            if (allocated(node%arg_indices)) then
                error_msg = 'expected scalar assignment target'
                call set_empty(name)
                return
            end if
            name = node%name
            call set_empty(error_msg)
            type is (component_access_node)
            call unsupported_feature_error('derived type component assignment target', &
                node%line, node%column, &
                'direct LIRIC session does not '// &
                'support assigning to components', &
                error_msg)
            call set_empty(name)
        class default
            error_msg = 'expected identifier assignment target'
            call set_empty(name)
        end select
    end subroutine identifier_name
    subroutine grow_symbols(context)
        type(lowering_context_t), intent(inout) :: context
        type(symbol_t), allocatable :: tmp(:)
        integer :: old_size

        if (context%symbol_count < size(context%symbols)) return
        old_size = size(context%symbols)
        call move_alloc(context%symbols, tmp)
        allocate(context%symbols(2 * old_size))
        context%symbols(1:old_size) = tmp
    end subroutine grow_symbols

    subroutine grow_derived_types(context)
        type(lowering_context_t), intent(inout) :: context
        type(derived_type_info_t), allocatable :: tmp(:)
        integer :: old_size

        if (context%derived_type_count < size(context%derived_types)) return
        old_size = size(context%derived_types)
        call move_alloc(context%derived_types, tmp)
        allocate(context%derived_types(2 * old_size))
        context%derived_types(1:old_size) = tmp
    end subroutine grow_derived_types

    subroutine grow_module_exports(context)
        type(lowering_context_t), intent(inout) :: context
        type(module_exports_t), allocatable :: tmp(:)
        integer :: old_size

        if (context%module_export_count < size(context%module_exports)) return
        old_size = size(context%module_exports)
        call move_alloc(context%module_exports, tmp)
        allocate(context%module_exports(2 * old_size))
        context%module_exports(1:old_size) = tmp
    end subroutine grow_module_exports

    subroutine grow_function_names(context)
        type(lowering_context_t), intent(inout) :: context
        character(len=64), allocatable :: tmp_names(:)
        integer, allocatable :: tmp_kinds(:)
        integer, allocatable :: tmp_counts(:)
        integer, allocatable :: tmp_indices(:)
        integer :: old_size, new_size

        if (context%function_count < size(context%function_names)) return
        old_size = size(context%function_names)
        new_size = 2 * old_size
        call move_alloc(context%function_names, tmp_names)
        call move_alloc(context%function_value_kinds, tmp_kinds)
        call move_alloc(context%function_param_counts, tmp_counts)
        if (allocated(context%function_node_indices)) &
            call move_alloc(context%function_node_indices, tmp_indices)
        allocate(context%function_names(new_size))
        allocate(context%function_value_kinds(new_size))
        allocate(context%function_param_counts(new_size))
        allocate(context%function_node_indices(new_size))
        context%function_param_counts = 0
        context%function_node_indices = 0
        context%function_names(1:old_size) = tmp_names
        context%function_value_kinds(1:old_size) = tmp_kinds
        context%function_param_counts(1:old_size) = tmp_counts
        if (allocated(tmp_indices)) &
            context%function_node_indices(1:size(tmp_indices)) = tmp_indices
    end subroutine grow_function_names

    integer function find_symbol_compat(context, name) result(index)
        !! Compatibility lookup for symbols synthesized without a FortFront
        !! declaration (for example inferred locals, DO variables, and ABI
        !! temporaries). Reference sites with an AST node must use
        !! resolve_symbol_at_node so declaration identity is consulted first.
        !! Keep this adapter private while the remaining synthetic-symbol
        !! paths migrate away from textual lookup (#332).
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: name
        integer :: i
        index = 0
        ! Search newest-first so a BLOCK-local declaration shadows an
        ! identically named outer symbol (variable_usage_shadowed_block). Names
        ! are unique outside nested scopes, so this matches forward search there.
        do i = context%symbol_count, 1, -1
            ! Fortran identifiers are case-insensitive. FortFront lowercases
            ! declared names but can preserve the source case at use sites, so a
            ! case-folded comparison is required to match them.
            if (same_name(trim(context%symbols(i)%name), trim(name))) then
                index = i
                return
            end if
        end do
    end function find_symbol_compat

    integer function find_symbol_same_scope(context, name) result(index)
        ! Like find_symbol_compat, but ignores symbols belonging to an enclosing
        ! scope (index <= block_scope_floor). A declaration that re-uses a name
        ! from an outer BLOCK scope is a legal shadow, not a duplicate (#280).
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: name
        index = find_symbol_compat(context, name)
        if (index > 0 .and. index <= context%block_scope_floor) index = 0
    end function find_symbol_same_scope

    integer function resolve_symbol_at_node(context, node_index, name) &
            result(index)
        !! Resolve a name reference to a lowering symbol (#327).
        !!
        !! FortFront owns name resolution: it maps the reference to a
        !! declaration binding, and the binding table maps that identity to a
        !! symbol slot. Local bindings must resolve to an exact symbol identity.
        !! Host- and USE-associated symbols may additionally use the private
        !! compatibility adapter because module and .fmod import paths can
        !! materialize legacy storage without carrying the local binding table
        !! entry. The same exception covers inferred lazy-Fortran symbols that
        !! have no collected declaration record. Collected direct/local
        !! bindings never fall through to a flat name scan (#332).
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: node_index
        character(len=*), intent(in) :: name
        type(declaration_binding_t) :: binding
        character(len=:), allocatable :: resolve_error
        integer :: bound, compat

        index = 0
        if (node_index <= 0) then
            index = find_symbol_compat(context, name)
            return
        end if
        call resolve_name_at_node(context%arena, node_index, name, binding, &
            resolve_error)
        if (len_trim(resolve_error) == 0 .and. binding%found) then
            bound = find_symbol_for_binding(context, binding)
            if (symbol_slot_holds_binding(context, bound, binding)) then
                index = bound
                return
            end if
            compat = find_symbol_compat(context, name)
            if (compat > 0) then
                if (context%symbols(compat)%is_statement_function_argument) then
                    index = compat
                    return
                end if
            end if
            if (binding%association /= ASSOCIATION_HOST .and. &
                binding%association /= ASSOCIATION_USE .and. &
                find_declaration_record(context, binding) > 0) return
        end if
        if (len_trim(resolve_error) > 0) return
        index = find_symbol_compat(context, name)
    end function resolve_symbol_at_node

    integer function find_symbol_for_binding(context, binding) result(index)
        !! Find a symbol by FortFront identity, without consulting its spelling.
        type(lowering_context_t), intent(in) :: context
        type(declaration_binding_t), intent(in) :: binding
        integer :: i

        index = context%binding_table%find_binding( &
            binding%declaration_node_index, binding%declaration_entity_index, &
            binding%scope_node_index)
        if (symbol_slot_holds_binding(context, index, binding)) return
        index = 0
        do i = context%symbol_count, 1, -1
            if (.not. context%symbols(i)%has_binding) cycle
            if (context%symbols(i)%binding_declaration_index /= &
                binding%declaration_node_index) cycle
            if (context%symbols(i)%binding_entity_index /= &
                binding%declaration_entity_index) cycle
            if (context%symbols(i)%binding_scope_index /= &
                binding%scope_node_index) cycle
            index = i
            return
        end do
    end function find_symbol_for_binding

    logical function symbol_slot_holds_binding(context, slot, binding) &
            result(holds)
        !! True when `slot` is a live symbol that still carries this exact
        !! binding identity. The lowering context reuses slot numbers once a
        !! BLOCK or a procedure body pops its locals, so the identity stored
        !! on the symbol itself — not the table entry — is the authority.
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: slot
        type(declaration_binding_t), intent(in) :: binding

        holds = .false.
        if (slot <= 0) return
        if (slot > context%symbol_count) return
        if (.not. context%symbols(slot)%has_binding) return
        if (context%symbols(slot)%binding_declaration_index /= &
            binding%declaration_node_index) return
        if (context%symbols(slot)%binding_entity_index /= &
            binding%declaration_entity_index) return
        if (context%symbols(slot)%binding_scope_index /= &
            binding%scope_node_index) return
        holds = .true.
    end function symbol_slot_holds_binding

    subroutine fold_named_constant_at_node(context, node_index, name, value, &
            found, error_msg)
        !! A named constant that FortFront resolves at this reference but that
        !! has no lowering symbol — a host-associated PARAMETER, whose
        !! declaration was lowered into a different context. Its value comes
        !! from the bound declaration's own initializer, reached through the
        !! binding rather than by searching the arena for the spelling: an
        !! identically named constant in a module that is not USEd stays
        !! invisible and keeps producing the undeclared-name diagnostic.
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: node_index
        character(len=*), intent(in) :: name
        integer(c_int64_t), intent(out) :: value
        logical, intent(out) :: found
        character(len=:), allocatable, intent(out) :: error_msg
        type(declaration_binding_t) :: binding
        character(len=:), allocatable :: resolve_error
        character(len=:), allocatable :: fold_error

        value = 0_c_int64_t
        found = .false.
        call set_empty(error_msg)
        if (node_index <= 0) return
        call resolve_name_at_node(context%arena, node_index, name, binding, &
            resolve_error)
        if (len_trim(resolve_error) > 0) return
        if (.not. binding%found) return
        if (binding%binding_kind /= BINDING_NAMED_CONSTANT) return
        call fold_i32_binding(context%arena, context, binding, value, fold_error)
        if (len_trim(fold_error) > 0) return
        found = .true.
    end subroutine fold_named_constant_at_node

    subroutine unwrap_intrinsic_arg(arena, arg_index, keyword, value_index)
        !! FortFront represents intrinsic keyword actuals as assignments.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: arg_index
        character(len=:), allocatable, intent(out) :: keyword
        integer, intent(out) :: value_index
        character(len=:), allocatable :: id_name, id_error

        keyword = ''
        value_index = arg_index
        if (.not. node_exists(arena, arg_index)) return
        select type (arg => arena%entries(arg_index)%node)
        type is (assignment_node)
            if (arg%value_index > 0) value_index = arg%value_index
            if (.not. node_exists(arena, arg%target_index)) return
            if (.not. is_identifier(arena, arg%target_index)) return
            call get_identifier_name(arena, arg%target_index, id_name, id_error)
            if (len_trim(id_error) > 0) return
            keyword = lowercase_text(trim(id_name))
        end select
    end subroutine unwrap_intrinsic_arg

    subroutine intrinsic_real_conversion_args(arena, node, value_index, kind_index)
        !! Locate the value and KIND actuals, including reordered keywords.
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        integer, intent(out) :: value_index
        integer, intent(out) :: kind_index
        character(len=:), allocatable :: keyword
        integer :: i, actual_index, positional_count

        value_index = 0
        kind_index = 0
        positional_count = 0
        if (.not. allocated(node%arg_indices)) return
        do i = 1, size(node%arg_indices)
            call unwrap_intrinsic_arg(arena, node%arg_indices(i), keyword, &
                                       actual_index)
            if (same_name(keyword, 'kind')) then
                kind_index = actual_index
            else if (len_trim(keyword) == 0) then
                positional_count = positional_count + 1
                if (positional_count == 1) then
                    value_index = actual_index
                else if (positional_count == 2) then
                    kind_index = actual_index
                end if
            else if (same_name(keyword, 'a') .and. value_index == 0) then
                value_index = actual_index
            end if
        end do
    end subroutine intrinsic_real_conversion_args

    logical function same_name(lhs, rhs)
        character(len=*), intent(in) :: lhs
        character(len=*), intent(in) :: rhs

        same_name = lowercase_text(lhs) == lowercase_text(rhs)
    end function same_name

    function lowercase_text(text) result(lowered)
        character(len=*), intent(in) :: text
        character(len=len_trim(text)) :: lowered
        integer :: code
        integer :: i

        do i = 1, len(lowered)
            code = iachar(text(i:i))
            if (code >= iachar('A') .and. code <= iachar('Z')) then
                lowered(i:i) = achar(code + 32)
            else
                lowered(i:i) = text(i:i)
            end if
        end do
    end function lowercase_text

    include 'session_program_lowering_select.inc'
end module session_program_lowering_impl
