    ! -- i8 (integer(1)) helpers -----------------------------------------

    function i8_vreg_op(session, vreg) result(operand)
        use liric_session_bindings, only: lr_type_i8_s
        type(liric_session_t), intent(in) :: session
        integer(c_int32_t), intent(in) :: vreg
        type(lr_operand_desc_t) :: operand

        operand%kind = LR_OP_KIND_VREG
        operand%payload = int(vreg, c_int64_t)
        operand%typ = lr_type_i8_s(session%handle)
        operand%global_offset = 0_c_int64_t
    end function i8_vreg_op

    function i8_immediate(session, value) result(operand)
        use liric_session_bindings, only: lr_type_i8_s
        type(liric_session_t), intent(in) :: session
        integer(c_int64_t), intent(in) :: value
        type(lr_operand_desc_t) :: operand

        operand%kind = LR_OP_KIND_IMM_I64
        operand%payload = value
        operand%typ = lr_type_i8_s(session%handle)
        operand%global_offset = 0_c_int64_t
    end function i8_immediate

    logical function emit_i8_alloca(session, address, error_msg)
        use liric_session_bindings, only: lr_type_i8_s
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(out) :: address
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_error_t) :: error
        integer(c_int32_t) :: vreg

        emit_i8_alloca = .false.
        if (.not. require_open_session(session, error_msg)) return
        vreg = emit_alloca_typed(session%handle, lr_type_i8_s(session%handle), error)
        if (.not. status_ok(error%code, error, error_msg)) return
        address = ptr_vreg(session, vreg)
        call set_empty(error_msg)
        emit_i8_alloca = .true.
    end function emit_i8_alloca

    logical function emit_i8_load(session, address, value, error_msg)
        use liric_session_bindings, only: lr_type_i8_s
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: address
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_error_t) :: error
        integer(c_int32_t) :: vreg

        emit_i8_load = .false.
        if (.not. require_open_session(session, error_msg)) return
        vreg = emit_load_typed(session%handle, lr_type_i8_s(session%handle), &
                               address, error)
        if (.not. status_ok(error%code, error, error_msg)) return
        value = i8_vreg_op(session, vreg)
        call set_empty(error_msg)
        emit_i8_load = .true.
    end function emit_i8_load

    logical function emit_i8_store(session, value, address, error_msg)
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: value
        type(lr_operand_desc_t), intent(in) :: address
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_error_t) :: error
        integer(c_int32_t) :: unused_vreg

        emit_i8_store = .false.
        if (.not. require_open_session(session, error_msg)) return
        unused_vreg = emit_store_typed(session%handle, value, address, error)
        if (.not. status_ok(error%code, error, error_msg)) return
        call set_empty(error_msg)
        emit_i8_store = .true.
    end function emit_i8_store

    logical function emit_i8_binary(session, opcode, lhs, rhs, result, error_msg)
        use liric_session_bindings, only: lr_type_i8_s, lr_session_emit
        type(liric_session_t), intent(inout) :: session
        integer(c_int), intent(in) :: opcode
        type(lr_operand_desc_t), intent(in) :: lhs
        type(lr_operand_desc_t), intent(in) :: rhs
        type(lr_operand_desc_t), intent(out) :: result
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_error_t) :: error
        type(lr_operand_desc_t), target :: operands(2)
        type(lr_inst_desc_t) :: inst
        integer(c_int32_t) :: vreg

        emit_i8_binary = .false.
        if (.not. require_open_session(session, error_msg)) return
        operands = [lhs, rhs]
        inst%op = opcode
        inst%typ = lr_type_i8_s(session%handle)
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
        vreg = lr_session_emit(session%handle, inst, error)
        if (.not. status_ok(error%code, error, error_msg)) return
        result = i8_vreg_op(session, vreg)
        call set_empty(error_msg)
        emit_i8_binary = .true.
    end function emit_i8_binary

    ! -- i16 (integer(2)) helpers ----------------------------------------

    function i16_vreg_op(session, vreg) result(operand)
        use liric_session_bindings, only: lr_type_i16_s
        type(liric_session_t), intent(in) :: session
        integer(c_int32_t), intent(in) :: vreg
        type(lr_operand_desc_t) :: operand

        operand%kind = LR_OP_KIND_VREG
        operand%payload = int(vreg, c_int64_t)
        operand%typ = lr_type_i16_s(session%handle)
        operand%global_offset = 0_c_int64_t
    end function i16_vreg_op

    function i16_immediate(session, value) result(operand)
        use liric_session_bindings, only: lr_type_i16_s
        type(liric_session_t), intent(in) :: session
        integer(c_int64_t), intent(in) :: value
        type(lr_operand_desc_t) :: operand

        operand%kind = LR_OP_KIND_IMM_I64
        operand%payload = value
        operand%typ = lr_type_i16_s(session%handle)
        operand%global_offset = 0_c_int64_t
    end function i16_immediate

    logical function emit_i16_alloca(session, address, error_msg)
        use liric_session_bindings, only: lr_type_i16_s
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(out) :: address
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_error_t) :: error
        integer(c_int32_t) :: vreg

        emit_i16_alloca = .false.
        if (.not. require_open_session(session, error_msg)) return
        vreg = emit_alloca_typed(session%handle, lr_type_i16_s(session%handle), error)
        if (.not. status_ok(error%code, error, error_msg)) return
        address = ptr_vreg(session, vreg)
        call set_empty(error_msg)
        emit_i16_alloca = .true.
    end function emit_i16_alloca

    logical function emit_i16_load(session, address, value, error_msg)
        use liric_session_bindings, only: lr_type_i16_s
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: address
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_error_t) :: error
        integer(c_int32_t) :: vreg

        emit_i16_load = .false.
        if (.not. require_open_session(session, error_msg)) return
        vreg = emit_load_typed(session%handle, lr_type_i16_s(session%handle), &
                               address, error)
        if (.not. status_ok(error%code, error, error_msg)) return
        value = i16_vreg_op(session, vreg)
        call set_empty(error_msg)
        emit_i16_load = .true.
    end function emit_i16_load

    logical function emit_i16_store(session, value, address, error_msg)
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: value
        type(lr_operand_desc_t), intent(in) :: address
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_error_t) :: error
        integer(c_int32_t) :: unused_vreg

        emit_i16_store = .false.
        if (.not. require_open_session(session, error_msg)) return
        unused_vreg = emit_store_typed(session%handle, value, address, error)
        if (.not. status_ok(error%code, error, error_msg)) return
        call set_empty(error_msg)
        emit_i16_store = .true.
    end function emit_i16_store

    logical function emit_i16_binary(session, opcode, lhs, rhs, result, error_msg)
        use liric_session_bindings, only: lr_type_i16_s, lr_session_emit
        type(liric_session_t), intent(inout) :: session
        integer(c_int), intent(in) :: opcode
        type(lr_operand_desc_t), intent(in) :: lhs
        type(lr_operand_desc_t), intent(in) :: rhs
        type(lr_operand_desc_t), intent(out) :: result
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_error_t) :: error
        type(lr_operand_desc_t), target :: operands(2)
        type(lr_inst_desc_t) :: inst
        integer(c_int32_t) :: vreg

        emit_i16_binary = .false.
        if (.not. require_open_session(session, error_msg)) return
        operands = [lhs, rhs]
        inst%op = opcode
        inst%typ = lr_type_i16_s(session%handle)
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
        vreg = lr_session_emit(session%handle, inst, error)
        if (.not. status_ok(error%code, error, error_msg)) return
        result = i16_vreg_op(session, vreg)
        call set_empty(error_msg)
        emit_i16_binary = .true.
    end function emit_i16_binary
