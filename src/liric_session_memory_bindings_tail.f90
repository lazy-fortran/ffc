submodule (liric_session_memory_bindings) liric_session_memory_bindings_tail
contains

    module procedure emit_binary
        type(lr_operand_desc_t), target :: operands(2)
        type(lr_inst_desc_t) :: inst

        operands = [lhs, rhs]

        inst%op = opcode
        inst%typ = lhs%typ
        inst%dest = 0_c_int32_t
        inst%operands = c_loc(operands)
        inst%num_operands = 2_c_int32_t
        inst%indices = c_null_ptr
        inst%num_indices = 0_c_int32_t
        inst%align = 0_c_int32_t
        inst%icmp_pred = 0_c_int
        inst%fcmp_pred = 0_c_int
        inst%call_external_abi = c_false
        inst%call_vararg = c_false
        inst%call_fixed_args = 0_c_int32_t

        call clear_liric_error(error)
        vreg = lr_session_emit(handle, inst, error)
    end procedure emit_binary

    module procedure emit_binary_with_dest

        type(lr_operand_desc_t), target :: operands(2)
        type(lr_inst_desc_t) :: inst

        operands = [lhs, rhs]

        inst%op = opcode
        inst%typ = lhs%typ
        inst%dest = dest_vreg
        inst%operands = c_loc(operands)
        inst%num_operands = 2_c_int32_t
        inst%indices = c_null_ptr
        inst%num_indices = 0_c_int32_t
        inst%align = 0_c_int32_t
        inst%icmp_pred = 0_c_int
        inst%fcmp_pred = 0_c_int
        inst%call_external_abi = c_false
        inst%call_vararg = c_false
        inst%call_fixed_args = 0_c_int32_t

        call clear_liric_error(error)
        vreg = lr_session_emit(handle, inst, error)
    end procedure emit_binary_with_dest

    module procedure emit_alloca_i32

        vreg = emit_alloca_typed(handle, lr_type_i32_s(handle), error)
    end procedure emit_alloca_i32

    module procedure emit_alloca_i64

        vreg = emit_alloca_typed(handle, lr_type_i64_s(handle), error)
    end procedure emit_alloca_i64

    module procedure emit_load_i32

        vreg = emit_load_typed(handle, lr_type_i32_s(handle), address, error)
    end procedure emit_load_i32

    module procedure emit_store_i32

        vreg = emit_store_typed(handle, value, address, error)
    end procedure emit_store_i32

    module procedure emit_load_i64

        vreg = emit_load_typed(handle, lr_type_i64_s(handle), address, error)
    end procedure emit_load_i64

    module procedure emit_load_ptr

        vreg = emit_load_typed(handle, lr_type_ptr_s(handle), address, error)
    end procedure emit_load_ptr

    module procedure emit_store_i64

        vreg = emit_store_typed(handle, value, address, error)
    end procedure emit_store_i64

    module procedure emit_binary_i64
        type(lr_operand_desc_t), target :: operands(2)
        type(lr_inst_desc_t) :: inst

        operands = [lhs, rhs]

        inst%op = opcode
        inst%typ = lr_type_i64_s(handle)
        inst%dest = 0_c_int32_t
        inst%operands = c_loc(operands)
        inst%num_operands = 2_c_int32_t
        inst%indices = c_null_ptr
        inst%num_indices = 0_c_int32_t
        inst%align = 0_c_int32_t
        inst%icmp_pred = 0_c_int
        inst%fcmp_pred = 0_c_int
        inst%call_external_abi = c_false
        inst%call_vararg = c_false
        inst%call_fixed_args = 0_c_int32_t

        call clear_liric_error(error)
        vreg = lr_session_emit(handle, inst, error)
    end procedure emit_binary_i64

    module procedure emit_alloca_i64_bytes
        type(lr_operand_desc_t), target :: operands(1)
        type(lr_inst_desc_t) :: inst

        operands(1) = size

        inst%op = LR_OP_ALLOCA
        inst%typ = lr_type_ptr_s(handle)
        inst%dest = 0_c_int32_t
        inst%operands = c_loc(operands)
        inst%num_operands = 1_c_int32_t
        inst%indices = c_null_ptr
        inst%num_indices = 0_c_int32_t
        inst%align = 0_c_int32_t
        inst%icmp_pred = 0_c_int
        inst%fcmp_pred = 0_c_int
        inst%call_external_abi = c_false
        inst%call_vararg = c_false
        inst%call_fixed_args = 0_c_int32_t

        call clear_liric_error(error)
        vreg = lr_session_emit(handle, inst, error)
    end procedure emit_alloca_i64_bytes

    module procedure emit_store_ptr

        vreg = emit_store_typed(handle, value, address, error)
    end procedure emit_store_ptr

    module function global_operand(session, id, typ) result(operand)
        type(liric_session_t), intent(in) :: session
        integer(c_int32_t), intent(in) :: id
        type(c_ptr), intent(in) :: typ
        type(lr_operand_desc_t) :: operand

        operand%kind = LR_OP_KIND_GLOBAL
        operand%payload = int(id, c_int64_t)
        operand%typ = typ
        operand%global_offset = 0_c_int64_t
    end function global_operand

    module procedure emit_memcpy_call
        type(lr_operand_desc_t), target :: operands(9)
        type(lr_inst_desc_t) :: inst
        character(kind=c_char), allocatable :: c_name(:)
        integer(c_int32_t) :: symbol_id
        integer :: i

        call to_c_chars('memcpy', c_name)
        symbol_id = lr_session_intern(handle, c_name)
        if (symbol_id < 0_c_int32_t .or. size(args) > 8) then
            call clear_liric_error(error)
            error%code = 1_c_int
            return
        end if

        operands(1) = global_operand_from_id(handle, symbol_id)
        do i = 1, size(args)
            operands(i + 1) = args(i)
        end do

        inst%op = LR_OP_CALL
        inst%typ = lr_type_void_s(handle)
        inst%dest = 0_c_int32_t
        inst%operands = c_loc(operands)
        inst%num_operands = int(size(args) + 1, c_int32_t)
        inst%indices = c_null_ptr
        inst%num_indices = 0_c_int32_t
        inst%align = 0_c_int32_t
        inst%icmp_pred = 0_c_int
        inst%fcmp_pred = 0_c_int
        inst%call_external_abi = c_false
        inst%call_vararg = c_false
        inst%call_fixed_args = 0_c_int32_t

        call clear_liric_error(error)
        vreg = lr_session_emit(handle, inst, error)
    end procedure emit_memcpy_call

    module procedure emit_malloc_call
        type(lr_operand_desc_t), target :: operands(3)
        type(lr_inst_desc_t) :: inst
        character(kind=c_char), allocatable :: c_name(:)
        integer(c_int32_t) :: symbol_id

        ! _ffc_alloc(count, elem_size) rather than malloc(bytes): the
        ! runtime validates the size and tracks the allocation so a double
        ! release is reported instead of corrupting the heap (#428). A
        ! caller with only a byte count passes it as the count with an
        ! element size of 1.
        call to_c_chars('_ffc_alloc', c_name)
        symbol_id = lr_session_intern(handle, c_name)
        if (symbol_id < 0_c_int32_t .or. size(args) /= 2) then
            call clear_liric_error(error)
            error%code = 1_c_int
            return
        end if

        operands(1) = ptr_global_operand(handle, symbol_id)
        operands(2) = args(1)
        operands(3) = args(2)

        inst%op = LR_OP_CALL
        inst%typ = lr_type_ptr_s(handle)
        inst%dest = 0_c_int32_t
        inst%operands = c_loc(operands)
        inst%num_operands = 3_c_int32_t
        inst%indices = c_null_ptr
        inst%num_indices = 0_c_int32_t
        inst%align = 0_c_int32_t
        inst%icmp_pred = 0_c_int
        inst%fcmp_pred = 0_c_int
        inst%call_external_abi = c_true
        inst%call_vararg = c_false
        inst%call_fixed_args = 2_c_int32_t

        call clear_liric_error(error)
        vreg = lr_session_emit(handle, inst, error)
    end procedure emit_malloc_call

    module procedure emit_strnlen_call
        type(lr_operand_desc_t), target :: operands(3)
        type(lr_inst_desc_t) :: inst
        character(kind=c_char), allocatable :: c_name(:)
        integer(c_int32_t) :: symbol_id

        call to_c_chars('strnlen', c_name)
        symbol_id = lr_session_intern(handle, c_name)
        if (symbol_id < 0_c_int32_t .or. size(args) /= 2) then
            call clear_liric_error(error)
            error%code = 1_c_int
            return
        end if

        operands(1) = ptr_global_operand(handle, symbol_id)
        operands(2) = args(1)
        operands(3) = args(2)

        inst%op = LR_OP_CALL
        inst%typ = lr_type_i64_s(handle)
        inst%dest = 0_c_int32_t
        inst%operands = c_loc(operands)
        inst%num_operands = 3_c_int32_t
        inst%indices = c_null_ptr
        inst%num_indices = 0_c_int32_t
        inst%align = 0_c_int32_t
        inst%icmp_pred = 0_c_int
        inst%fcmp_pred = 0_c_int
        inst%call_external_abi = c_true
        inst%call_vararg = c_false
        inst%call_fixed_args = 2_c_int32_t

        call clear_liric_error(error)
        vreg = lr_session_emit(handle, inst, error)
    end procedure emit_strnlen_call

    module procedure emit_calloc_call
        type(lr_operand_desc_t), target :: operands(3)
        type(lr_inst_desc_t) :: inst
        character(kind=c_char), allocatable :: c_name(:)
        integer(c_int32_t) :: symbol_id

        call to_c_chars('_ffc_calloc', c_name)
        symbol_id = lr_session_intern(handle, c_name)
        if (symbol_id < 0_c_int32_t .or. size(args) /= 2) then
            call clear_liric_error(error)
            error%code = 1_c_int
            return
        end if

        operands(1) = ptr_global_operand(handle, symbol_id)
        operands(2) = args(1)
        operands(3) = args(2)

        inst%op = LR_OP_CALL
        inst%typ = lr_type_ptr_s(handle)
        inst%dest = 0_c_int32_t
        inst%operands = c_loc(operands)
        inst%num_operands = 3_c_int32_t
        inst%indices = c_null_ptr
        inst%num_indices = 0_c_int32_t
        inst%align = 0_c_int32_t
        inst%icmp_pred = 0_c_int
        inst%fcmp_pred = 0_c_int
        inst%call_external_abi = c_true
        inst%call_vararg = c_false
        inst%call_fixed_args = 2_c_int32_t

        call clear_liric_error(error)
        vreg = lr_session_emit(handle, inst, error)
    end procedure emit_calloc_call

    module procedure global_operand_from_id

        operand%kind = LR_OP_KIND_GLOBAL
        operand%payload = int(symbol_id, c_int64_t)
        operand%typ = lr_type_i32_s(handle)
        operand%global_offset = 0_c_int64_t
    end procedure global_operand_from_id

    module procedure ptr_global_operand

        operand%kind = LR_OP_KIND_GLOBAL
        operand%payload = int(symbol_id, c_int64_t)
        operand%typ = lr_type_ptr_s(handle)
        operand%global_offset = 0_c_int64_t
    end procedure ptr_global_operand

    module procedure emit_alloca_typed
        type(lr_inst_desc_t) :: inst

        inst%op = LR_OP_ALLOCA
        inst%typ = typ
        inst%dest = 0_c_int32_t
        inst%operands = c_null_ptr
        inst%num_operands = 0_c_int32_t
        inst%indices = c_null_ptr
        inst%num_indices = 0_c_int32_t
        inst%align = 0_c_int32_t
        inst%icmp_pred = 0_c_int
        inst%fcmp_pred = 0_c_int
        inst%call_external_abi = c_false
        inst%call_vararg = c_false
        inst%call_fixed_args = 0_c_int32_t

        call clear_liric_error(error)
        vreg = lr_session_emit(handle, inst, error)
    end procedure emit_alloca_typed

    module procedure emit_load_typed
        type(lr_operand_desc_t), target :: operands(1)
        type(lr_inst_desc_t) :: inst

        operands(1) = address

        inst%op = LR_OP_LOAD
        inst%typ = typ
        inst%dest = 0_c_int32_t
        inst%operands = c_loc(operands)
        inst%num_operands = 1_c_int32_t
        inst%indices = c_null_ptr
        inst%num_indices = 0_c_int32_t
        inst%align = 0_c_int32_t
        inst%icmp_pred = 0_c_int
        inst%fcmp_pred = 0_c_int
        inst%call_external_abi = c_false
        inst%call_vararg = c_false
        inst%call_fixed_args = 0_c_int32_t

        call clear_liric_error(error)
        vreg = lr_session_emit(handle, inst, error)
    end procedure emit_load_typed

    module procedure emit_complex_value_load

        type(c_ptr), target :: fields(2)
        type(c_ptr) :: aggregate_type
        type(lr_error_t) :: error
        integer(c_int32_t) :: vreg

        emit_complex_value_load = .false.
        if (.not. require_open_session(session, error_msg)) return
        if (is_double) then
            fields(1) = lr_type_f64_s(session%handle)
        else
            fields(1) = lr_type_f32_s(session%handle)
        end if
        fields(2) = fields(1)
        aggregate_type = lr_type_struct_s(session%handle, c_loc(fields), &
                                          2_c_int32_t, c_false)
        if (.not. c_associated(aggregate_type)) then
            error_msg = 'LIRIC did not return a complex aggregate type'
            return
        end if
        call clear_liric_error(error)
        vreg = emit_load_typed(session%handle, aggregate_type, address, error)
        if (.not. status_ok(error%code, error, error_msg)) return
        value%kind = LR_OP_KIND_VREG
        value%payload = int(vreg, c_int64_t)
        value%typ = aggregate_type
        value%global_offset = 0_c_int64_t
        call set_empty(error_msg)
        emit_complex_value_load = .true.
    end procedure emit_complex_value_load

    module procedure emit_c_complex_call
        type(c_ptr), target :: fields(2)
        type(c_ptr) :: aggregate_type, element_type
        type(lr_operand_desc_t) :: aggregate_value

        emit_c_complex_call = .false.
        if (.not. require_open_session(session, error_msg)) return
        if (is_double) then
            element_type = lr_type_f64_s(session%handle)
        else
            element_type = lr_type_f32_s(session%handle)
        end if
        fields = [element_type, element_type]
        aggregate_type = lr_type_struct_s(session%handle, c_loc(fields), &
                                          2_c_int32_t, c_false)
        if (.not. c_associated(aggregate_type)) then
            error_msg = 'LIRIC did not return a complex aggregate type'
            return
        end if
        if (.not. emit_c_aggregate_call(session, name, args, aggregate_type, &
                                        aggregate_value, error_msg)) return
        if (.not. emit_extract_value_typed(session, aggregate_value, element_type, &
                                           0_c_int32_t, re_value, error_msg)) return
        if (.not. emit_extract_value_typed(session, aggregate_value, element_type, &
                                           1_c_int32_t, im_value, error_msg)) return
        call set_empty(error_msg)
        emit_c_complex_call = .true.
    end procedure emit_c_complex_call

    module procedure emit_extract_value_typed
        type(lr_operand_desc_t), target :: operands(1)
        integer(c_int32_t), target :: indices(1)
        type(lr_inst_desc_t) :: inst
        type(lr_error_t) :: error
        integer(c_int32_t) :: vreg

        emit_extract_value_typed = .false.
        if (.not. require_open_session(session, error_msg)) return
        operands(1) = aggregate
        indices(1) = index
        inst%op = 45_c_int
        inst%typ = element_type
        inst%dest = 0_c_int32_t
        inst%operands = c_loc(operands)
        inst%num_operands = 1_c_int32_t
        inst%indices = c_loc(indices)
        inst%num_indices = 1_c_int32_t
        inst%align = 0_c_int32_t
        inst%icmp_pred = 0_c_int
        inst%fcmp_pred = 0_c_int
        inst%call_external_abi = c_false
        inst%call_vararg = c_false
        inst%call_fixed_args = 0_c_int32_t
        call clear_liric_error(error)
        vreg = lr_session_emit(session%handle, inst, error)
        if (.not. status_ok(error%code, error, error_msg)) return
        value%kind = LR_OP_KIND_VREG
        value%payload = int(vreg, c_int64_t)
        value%typ = element_type
        value%global_offset = 0_c_int64_t
        call set_empty(error_msg)
        emit_extract_value_typed = .true.
    end procedure emit_extract_value_typed

    module procedure emit_store_typed
        type(lr_operand_desc_t), target :: operands(2)
        type(lr_inst_desc_t) :: inst

        operands = [value, address]

        inst%op = LR_OP_STORE
        inst%typ = c_null_ptr
        inst%dest = 0_c_int32_t
        inst%operands = c_loc(operands)
        inst%num_operands = 2_c_int32_t
        inst%indices = c_null_ptr
        inst%num_indices = 0_c_int32_t
        inst%align = 0_c_int32_t
        inst%icmp_pred = 0_c_int
        inst%fcmp_pred = 0_c_int
        inst%call_external_abi = c_false
        inst%call_vararg = c_false
        inst%call_fixed_args = 0_c_int32_t

        call clear_liric_error(error)
        vreg = lr_session_emit(handle, inst, error)
    end procedure emit_store_typed

end submodule liric_session_memory_bindings_tail
