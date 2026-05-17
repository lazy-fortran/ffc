    logical function emit_i32_array_alloca(this, array_size, address, error_msg)
        class(liric_session_t), intent(inout) :: this
        integer, intent(in) :: array_size
        type(lr_operand_desc_t), intent(out) :: address
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_error_t) :: error
        type(c_ptr) :: array_type
        integer(c_int32_t) :: vreg
        type(lr_inst_desc_t) :: inst

        emit_i32_array_alloca = .false.
        if (.not. require_open_session(this, error_msg)) return

        array_type = lr_type_array_s(this%handle, &
                                     lr_type_i32_s(this%handle), &
                                     int(array_size, c_int64_t))
        if (.not. c_associated(array_type)) then
            error_msg = 'LIRIC did not return an integer array type'
            return
        end if
        inst%op = LR_OP_ALLOCA
        inst%typ = array_type
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
        vreg = lr_session_emit(this%handle, inst, error)
        if (.not. status_ok(error%code, error, error_msg)) then
            error_msg = 'array_alloca: '//error_msg
            return
        end if

        address = this%ptr_vreg(vreg)
        call set_empty(error_msg)
        emit_i32_array_alloca = .true.
    end function emit_i32_array_alloca

    logical function emit_i32_array_element_addr(this, array_size, base_ptr, &
                                                 index_0based, element_addr, &
                                                 error_msg)
        class(liric_session_t), intent(inout) :: this
        integer, intent(in) :: array_size
        type(lr_operand_desc_t), intent(in) :: base_ptr
        type(lr_operand_desc_t), intent(in) :: index_0based
        type(lr_operand_desc_t), intent(out) :: element_addr
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_error_t) :: error
        type(c_ptr) :: array_type
        type(lr_operand_desc_t), target :: operands(3)
        type(lr_operand_desc_t) :: zero64
        type(lr_inst_desc_t) :: inst
        integer(c_int32_t) :: vreg

        emit_i32_array_element_addr = .false.
        if (.not. require_open_session(this, error_msg)) return

        array_type = lr_type_array_s(this%handle, &
                                     lr_type_i32_s(this%handle), &
                                     int(array_size, c_int64_t))

        ! First GEP index walks across the [N x i32] aggregate; second
        ! selects the element within it. Match what lr_emit_gep does in C.
        zero64%kind = LR_OP_KIND_IMM_I64
        zero64%payload = 0_c_int64_t
        zero64%typ = lr_type_i64_s(this%handle)
        zero64%global_offset = 0_c_int64_t

        operands(1) = base_ptr
        operands(2) = zero64
        operands(3) = index_0based

        inst%op = LR_OP_GEP
        inst%typ = array_type
        inst%dest = 0_c_int32_t
        inst%operands = c_loc(operands)
        inst%num_operands = 3_c_int32_t
        inst%indices = c_null_ptr
        inst%num_indices = 0_c_int32_t
        inst%align = 0_c_int32_t
        inst%icmp_pred = 0_c_int
        inst%fcmp_pred = 0_c_int
        inst%call_external_abi = c_false
        inst%call_vararg = c_false
        inst%call_fixed_args = 0_c_int32_t

        call clear_liric_error(error)
        vreg = lr_session_emit(this%handle, inst, error)
        if (.not. status_ok(error%code, error, error_msg)) then
            error_msg = 'gep: '//error_msg
            return
        end if

        element_addr = this%ptr_vreg(vreg)
        call set_empty(error_msg)
        emit_i32_array_element_addr = .true.
    end function emit_i32_array_element_addr

    function emit_ret_i32(handle, value, error) result(vreg)
        type(c_ptr), intent(in) :: handle
        type(lr_operand_desc_t), intent(in) :: value
        type(lr_error_t), intent(inout) :: error
        integer(c_int32_t) :: vreg
        type(lr_operand_desc_t), target :: operands(1)
        type(lr_inst_desc_t) :: inst

        operands(1) = value

        inst%op = LR_OP_RET
        inst%typ = value%typ
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
    end function emit_ret_i32

    function emit_ret_void_inst(handle, error) result(vreg)
        type(c_ptr), intent(in) :: handle
        type(lr_error_t), intent(inout) :: error
        integer(c_int32_t) :: vreg
        type(lr_inst_desc_t) :: inst

        inst%op = LR_OP_RET_VOID
        inst%typ = lr_type_void_s(handle)
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
    end function emit_ret_void_inst

    function emit_binary(handle, opcode, lhs, rhs, error, dest_vreg) result(vreg)
        type(c_ptr), intent(in) :: handle
        integer(c_int), intent(in) :: opcode
        type(lr_operand_desc_t), intent(in) :: lhs
        type(lr_operand_desc_t), intent(in) :: rhs
        type(lr_error_t), intent(inout) :: error
        integer(c_int32_t), intent(in), optional :: dest_vreg
        integer(c_int32_t) :: vreg
        type(lr_operand_desc_t), target :: operands(2)
        type(lr_inst_desc_t) :: inst

        operands = [lhs, rhs]

        inst%op = opcode
        inst%typ = lhs%typ
        inst%dest = 0_c_int32_t
        if (present(dest_vreg)) inst%dest = dest_vreg
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
    end function emit_binary

    function emit_alloca_i32(handle, error) result(vreg)
        type(c_ptr), intent(in) :: handle
        type(lr_error_t), intent(inout) :: error
        integer(c_int32_t) :: vreg
        type(lr_inst_desc_t) :: inst

        inst%op = LR_OP_ALLOCA
        inst%typ = lr_type_i32_s(handle)
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
    end function emit_alloca_i32
