    ! Fixed-size array alloca/GEP for narrow integer element kinds:
    ! integer(8)/(1)/(2). Mirrors the f64/f32 array helpers above, one
    ! element type per block.

    logical function emit_i64_array_alloca(session, array_size, address, &
            error_msg)
        type(liric_session_t), intent(inout) :: session
        integer, intent(in) :: array_size
        type(lr_operand_desc_t), intent(out) :: address
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_error_t) :: error
        type(c_ptr) :: array_type
        integer(c_int32_t) :: vreg
        type(lr_inst_desc_t) :: inst

        emit_i64_array_alloca = .false.
        if (.not. require_open_session(session, error_msg)) return

        array_type = lr_type_array_s(session%handle, &
            lr_type_i64_s(session%handle), &
            int(array_size, c_int64_t))
        if (.not. c_associated(array_type)) then
            error_msg = 'LIRIC did not return an i64 array type'
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
        vreg = lr_session_emit(session%handle, inst, error)
        if (.not. status_ok(error%code, error, error_msg)) then
            error_msg = 'i64_array_alloca: '//error_msg
            return
        end if

        address = ptr_vreg(session, vreg)
        call set_empty(error_msg)
        emit_i64_array_alloca = .true.
    end function emit_i64_array_alloca

    logical function emit_i64_array_element_addr(session, array_size, &
            base_ptr, index_0based, &
            element_addr, error_msg)
        type(liric_session_t), intent(inout) :: session
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

        emit_i64_array_element_addr = .false.
        if (.not. require_open_session(session, error_msg)) return

        array_type = lr_type_array_s(session%handle, &
            lr_type_i64_s(session%handle), &
            int(array_size, c_int64_t))

        zero64%kind = LR_OP_KIND_IMM_I64
        zero64%payload = 0_c_int64_t
        zero64%typ = lr_type_i64_s(session%handle)
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
        vreg = lr_session_emit(session%handle, inst, error)
        if (.not. status_ok(error%code, error, error_msg)) then
            error_msg = 'i64_gep: '//error_msg
            return
        end if

        element_addr = ptr_vreg(session, vreg)
        call set_empty(error_msg)
        emit_i64_array_element_addr = .true.
    end function emit_i64_array_element_addr

    logical function emit_i8_array_alloca(session, array_size, address, &
            error_msg)
        use liric_session_bindings, only: lr_type_i8_s
        type(liric_session_t), intent(inout) :: session
        integer, intent(in) :: array_size
        type(lr_operand_desc_t), intent(out) :: address
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_error_t) :: error
        type(c_ptr) :: array_type
        integer(c_int32_t) :: vreg
        type(lr_inst_desc_t) :: inst

        emit_i8_array_alloca = .false.
        if (.not. require_open_session(session, error_msg)) return

        array_type = lr_type_array_s(session%handle, &
            lr_type_i8_s(session%handle), &
            int(array_size, c_int64_t))
        if (.not. c_associated(array_type)) then
            error_msg = 'LIRIC did not return an i8 array type'
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
        vreg = lr_session_emit(session%handle, inst, error)
        if (.not. status_ok(error%code, error, error_msg)) then
            error_msg = 'i8_array_alloca: '//error_msg
            return
        end if

        address = ptr_vreg(session, vreg)
        call set_empty(error_msg)
        emit_i8_array_alloca = .true.
    end function emit_i8_array_alloca

    logical function emit_i8_array_element_addr(session, array_size, &
            base_ptr, index_0based, &
            element_addr, error_msg)
        use liric_session_bindings, only: lr_type_i8_s
        type(liric_session_t), intent(inout) :: session
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

        emit_i8_array_element_addr = .false.
        if (.not. require_open_session(session, error_msg)) return

        array_type = lr_type_array_s(session%handle, &
            lr_type_i8_s(session%handle), &
            int(array_size, c_int64_t))

        zero64%kind = LR_OP_KIND_IMM_I64
        zero64%payload = 0_c_int64_t
        zero64%typ = lr_type_i64_s(session%handle)
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
        vreg = lr_session_emit(session%handle, inst, error)
        if (.not. status_ok(error%code, error, error_msg)) then
            error_msg = 'i8_gep: '//error_msg
            return
        end if

        element_addr = ptr_vreg(session, vreg)
        call set_empty(error_msg)
        emit_i8_array_element_addr = .true.
    end function emit_i8_array_element_addr

    logical function emit_i16_array_alloca(session, array_size, address, &
            error_msg)
        use liric_session_bindings, only: lr_type_i16_s
        type(liric_session_t), intent(inout) :: session
        integer, intent(in) :: array_size
        type(lr_operand_desc_t), intent(out) :: address
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_error_t) :: error
        type(c_ptr) :: array_type
        integer(c_int32_t) :: vreg
        type(lr_inst_desc_t) :: inst

        emit_i16_array_alloca = .false.
        if (.not. require_open_session(session, error_msg)) return

        array_type = lr_type_array_s(session%handle, &
            lr_type_i16_s(session%handle), &
            int(array_size, c_int64_t))
        if (.not. c_associated(array_type)) then
            error_msg = 'LIRIC did not return an i16 array type'
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
        vreg = lr_session_emit(session%handle, inst, error)
        if (.not. status_ok(error%code, error, error_msg)) then
            error_msg = 'i16_array_alloca: '//error_msg
            return
        end if

        address = ptr_vreg(session, vreg)
        call set_empty(error_msg)
        emit_i16_array_alloca = .true.
    end function emit_i16_array_alloca

    logical function emit_i16_array_element_addr(session, array_size, &
            base_ptr, index_0based, &
            element_addr, error_msg)
        use liric_session_bindings, only: lr_type_i16_s
        type(liric_session_t), intent(inout) :: session
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

        emit_i16_array_element_addr = .false.
        if (.not. require_open_session(session, error_msg)) return

        array_type = lr_type_array_s(session%handle, &
            lr_type_i16_s(session%handle), &
            int(array_size, c_int64_t))

        zero64%kind = LR_OP_KIND_IMM_I64
        zero64%payload = 0_c_int64_t
        zero64%typ = lr_type_i64_s(session%handle)
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
        vreg = lr_session_emit(session%handle, inst, error)
        if (.not. status_ok(error%code, error, error_msg)) then
            error_msg = 'i16_gep: '//error_msg
            return
        end if

        element_addr = ptr_vreg(session, vreg)
        call set_empty(error_msg)
        emit_i16_array_element_addr = .true.
    end function emit_i16_array_element_addr
