module liric_session_memory_bindings
    use, intrinsic :: iso_c_binding, only: c_associated, c_bool, c_char, c_int, &
                                              c_int32_t, c_int64_t
    use, intrinsic :: iso_c_binding, only: c_loc, c_null_char, c_null_ptr, c_ptr
    use liric_session_common, only: require_open_session, status_ok, &
                                    liric_session_error_message, &
                                    clear_liric_error, to_c_chars, set_empty, &
                                    lr_error_t, lr_inst_desc_t
    use liric_session_bindings, only: liric_session_t, lr_operand_desc_t, &
                                      i32_vreg, i32_immediate
    implicit none
    private

    integer(c_int), parameter :: LR_OP_ALLOCA = 26_c_int
    integer(c_int), parameter :: LR_OP_GEP = 29_c_int
    integer(c_int), parameter :: LR_OP_LOAD = 27_c_int
    integer(c_int), parameter :: LR_OP_STORE = 28_c_int
    integer(c_int), parameter :: LR_OP_ADD = 5_c_int
    integer(c_int), parameter :: LR_OP_CALL = 30_c_int
    integer(c_int), parameter :: LR_OP_KIND_VREG = 0_c_int
    integer(c_int), parameter :: LR_OP_KIND_IMM_I64 = 1_c_int
    integer(c_int), parameter :: LR_OP_KIND_GLOBAL = 4_c_int
    integer(c_int), parameter :: LR_OK = 0_c_int
    logical(c_bool), parameter :: c_false = .false.
    logical(c_bool), parameter :: c_true = .true.

    public :: i64_immediate, ptr_param, &
              reserve_i32_vreg, ptr_vreg, i64_vreg
    public :: emit_i32_binary, emit_i32_binary_into, emit_i32_copy_to
    public :: emit_i32_alloca, emit_i64_alloca
    public :: emit_i32_load, emit_i64_load, emit_ptr_load
    public :: emit_i32_store, emit_i64_store
    public :: emit_i64_load_at, emit_i64_store_at
    public :: emit_ptr_offset, emit_ptr_offset_dyn
    public :: emit_alloca_bytes, emit_malloc, emit_free, emit_ptr_store, emit_memcpy
    public :: emit_i32_array_alloca, emit_i32_array_element_addr
    public :: emit_i64_binary
    public :: emit_alloca_typed, emit_load_typed, emit_store_typed

    interface
        function lr_session_vreg(handle) result(vreg) bind(c)
            import :: c_int32_t, c_ptr
            type(c_ptr), value :: handle
            integer(c_int32_t) :: vreg
        end function lr_session_vreg

        function lr_session_emit(handle, inst, err) result(vreg) bind(c)
            import :: c_int32_t, c_ptr, lr_error_t, lr_inst_desc_t
            type(c_ptr), value :: handle
            type(lr_inst_desc_t), intent(in) :: inst
            type(lr_error_t), intent(inout) :: err
            integer(c_int32_t) :: vreg
        end function lr_session_emit

        function lr_type_i32_s(handle) result(typ) bind(c)
            import :: c_ptr
            type(c_ptr), value :: handle
            type(c_ptr) :: typ
        end function lr_type_i32_s

        function lr_type_i64_s(handle) result(typ) bind(c)
            import :: c_ptr
            type(c_ptr), value :: handle
            type(c_ptr) :: typ
        end function lr_type_i64_s

        function lr_type_ptr_s(handle) result(typ) bind(c)
            import :: c_ptr
            type(c_ptr), value :: handle
            type(c_ptr) :: typ
        end function lr_type_ptr_s

        function lr_type_array_s(handle, elem, count) result(typ) bind(c)
            import :: c_int64_t, c_ptr
            type(c_ptr), value :: handle
            type(c_ptr), value :: elem
            integer(c_int64_t), value :: count
            type(c_ptr) :: typ
        end function lr_type_array_s

        function lr_session_param(handle, index) result(vreg) bind(c)
            import :: c_int32_t, c_ptr
            type(c_ptr), value :: handle
            integer(c_int32_t), value :: index
            integer(c_int32_t) :: vreg
        end function lr_session_param

        function lr_session_intern(handle, name) result(symbol_id) bind(c)
            import :: c_char, c_int32_t, c_ptr
            type(c_ptr), value :: handle
            character(kind=c_char), intent(in) :: name(*)
            integer(c_int32_t) :: symbol_id
        end function lr_session_intern

        function lr_type_void_s(handle) result(typ) bind(c)
            import :: c_ptr
            type(c_ptr), value :: handle
            type(c_ptr) :: typ
        end function lr_type_void_s
    end interface

contains

    function i32_param(session, index) result(operand)
        type(liric_session_t), intent(in) :: session
        integer, intent(in) :: index
        type(lr_operand_desc_t) :: operand
        integer(c_int32_t) :: vreg

        vreg = lr_session_param(session%handle, int(index, c_int32_t))
        operand = i32_vreg(session, vreg)
    end function i32_param

    function ptr_param(session, index) result(operand)
        type(liric_session_t), intent(in) :: session
        integer, intent(in) :: index
        type(lr_operand_desc_t) :: operand
        integer(c_int32_t) :: vreg

        vreg = lr_session_param(session%handle, int(index, c_int32_t))
        operand = ptr_vreg(session, vreg)
    end function ptr_param

    function reserve_i32_vreg(session) result(vreg)
        type(liric_session_t), intent(in) :: session
        integer(c_int32_t) :: vreg

        vreg = lr_session_vreg(session%handle)
    end function reserve_i32_vreg

    function ptr_vreg(session, vreg) result(operand)
        type(liric_session_t), intent(in) :: session
        integer(c_int32_t), intent(in) :: vreg
        type(lr_operand_desc_t) :: operand

        operand%kind = LR_OP_KIND_VREG
        operand%payload = int(vreg, c_int64_t)
        operand%typ = lr_type_ptr_s(session%handle)
        operand%global_offset = 0_c_int64_t
    end function ptr_vreg

    function i64_vreg(session, vreg) result(operand)
        type(liric_session_t), intent(in) :: session
        integer(c_int32_t), intent(in) :: vreg
        type(lr_operand_desc_t) :: operand

        operand%kind = LR_OP_KIND_VREG
        operand%payload = int(vreg, c_int64_t)
        operand%typ = lr_type_i64_s(session%handle)
        operand%global_offset = 0_c_int64_t
    end function i64_vreg

    function i64_immediate(session, value) result(operand)
        type(liric_session_t), intent(in) :: session
        integer(c_int64_t), intent(in) :: value
        type(lr_operand_desc_t) :: operand

        operand%kind = LR_OP_KIND_IMM_I64
        operand%payload = value
        operand%typ = lr_type_i64_s(session%handle)
        operand%global_offset = 0_c_int64_t
    end function i64_immediate

    logical function emit_i32_binary(session, opcode, lhs, rhs, result, &
                                     error_msg)
        type(liric_session_t), intent(inout) :: session
        integer(c_int), intent(in) :: opcode
        type(lr_operand_desc_t), intent(in) :: lhs
        type(lr_operand_desc_t), intent(in) :: rhs
        type(lr_operand_desc_t), intent(out) :: result
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_error_t) :: error
        integer(c_int32_t) :: vreg

        emit_i32_binary = .false.
        if (.not. require_open_session(session, error_msg)) return

        vreg = emit_binary(session%handle, opcode, lhs, rhs, error)
        if (.not. status_ok(error%code, error, error_msg)) return

        result = i32_vreg(session, vreg)
        call set_empty(error_msg)
        emit_i32_binary = .true.
    end function emit_i32_binary

    logical function emit_i32_binary_into(session, opcode, lhs, rhs, &
                                          dest_vreg, result, error_msg)
        type(liric_session_t), intent(inout) :: session
        integer(c_int), intent(in) :: opcode
        type(lr_operand_desc_t), intent(in) :: lhs
        type(lr_operand_desc_t), intent(in) :: rhs
        integer(c_int32_t), intent(in) :: dest_vreg
        type(lr_operand_desc_t), intent(out) :: result
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_error_t) :: error
        integer(c_int32_t) :: vreg

        emit_i32_binary_into = .false.
        if (.not. require_open_session(session, error_msg)) return
        if (dest_vreg <= 0_c_int32_t) then
            error_msg = 'explicit LIRIC destination vreg must be positive'
            return
        end if

        vreg = emit_binary_with_dest(session%handle, opcode, lhs, rhs, &
                                     error, dest_vreg)
        if (.not. status_ok(error%code, error, error_msg)) return
        if (vreg /= dest_vreg) then
            error_msg = 'LIRIC did not honor explicit binary destination vreg'
            return
        end if

        result = i32_vreg(session, vreg)
        call set_empty(error_msg)
        emit_i32_binary_into = .true.
    end function emit_i32_binary_into

    logical function emit_i32_copy_to(session, value, dest_vreg, result, &
                                      error_msg)
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: value
        integer(c_int32_t), intent(in) :: dest_vreg
        type(lr_operand_desc_t), intent(out) :: result
        character(len=:), allocatable, intent(out) :: error_msg

        emit_i32_copy_to = emit_i32_binary_into(session, &
                           LR_OP_ADD, value, i32_immediate(session, 0_c_int64_t), &
                           dest_vreg, result, error_msg)
    end function emit_i32_copy_to

    logical function emit_i32_alloca(session, address, error_msg)
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(out) :: address
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_error_t) :: error
        integer(c_int32_t) :: vreg

        emit_i32_alloca = .false.
        if (.not. require_open_session(session, error_msg)) return

        vreg = emit_alloca_i32(session%handle, error)
        if (.not. status_ok(error%code, error, error_msg)) return

        address = ptr_vreg(session, vreg)
        call set_empty(error_msg)
        emit_i32_alloca = .true.
    end function emit_i32_alloca

    logical function emit_i64_alloca(session, address, error_msg)
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(out) :: address
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_error_t) :: error
        integer(c_int32_t) :: vreg

        emit_i64_alloca = .false.
        if (.not. require_open_session(session, error_msg)) return

        vreg = emit_alloca_i64(session%handle, error)
        if (.not. status_ok(error%code, error, error_msg)) return

        address = ptr_vreg(session, vreg)
        call set_empty(error_msg)
        emit_i64_alloca = .true.
    end function emit_i64_alloca

    logical function emit_i32_load(session, address, value, error_msg)
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: address
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_error_t) :: error
        integer(c_int32_t) :: vreg

        emit_i32_load = .false.
        if (.not. require_open_session(session, error_msg)) return

        vreg = emit_load_i32(session%handle, address, error)
        if (.not. status_ok(error%code, error, error_msg)) return

        value = i32_vreg(session, vreg)
        call set_empty(error_msg)
        emit_i32_load = .true.
    end function emit_i32_load

    logical function emit_i64_load(session, address, value, error_msg)
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: address
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_error_t) :: error
        integer(c_int32_t) :: vreg

        emit_i64_load = .false.
        if (.not. require_open_session(session, error_msg)) return

        vreg = emit_load_i64(session%handle, address, error)
        if (.not. status_ok(error%code, error, error_msg)) return

        value = i64_vreg(session, vreg)
        call set_empty(error_msg)
        emit_i64_load = .true.
    end function emit_i64_load

    logical function emit_ptr_load(session, address, value, error_msg)
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: address
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_error_t) :: error
        integer(c_int32_t) :: vreg

        emit_ptr_load = .false.
        if (.not. require_open_session(session, error_msg)) return

        vreg = emit_load_ptr(session%handle, address, error)
        if (.not. status_ok(error%code, error, error_msg)) return

        value = ptr_vreg(session, vreg)
        call set_empty(error_msg)
        emit_ptr_load = .true.
    end function emit_ptr_load

    logical function emit_i32_store(session, value, address, error_msg)
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: value
        type(lr_operand_desc_t), intent(in) :: address
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_error_t) :: error
        integer(c_int32_t) :: unused_vreg

        emit_i32_store = .false.
        if (.not. require_open_session(session, error_msg)) return

        unused_vreg = emit_store_i32(session%handle, value, address, error)
        if (.not. status_ok(error%code, error, error_msg)) return

        call set_empty(error_msg)
        emit_i32_store = .true.
    end function emit_i32_store

    logical function emit_i64_store(session, value, address, error_msg)
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: value
        type(lr_operand_desc_t), intent(in) :: address
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_error_t) :: error
        integer(c_int32_t) :: unused_vreg

        emit_i64_store = .false.
        if (.not. require_open_session(session, error_msg)) return

        unused_vreg = emit_store_i64(session%handle, value, address, error)
        if (.not. status_ok(error%code, error, error_msg)) return

        call set_empty(error_msg)
        emit_i64_store = .true.
    end function emit_i64_store

    logical function emit_i64_load_at(session, base, offset, result, &
                                      error_msg)
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: base
        integer(c_int64_t), intent(in) :: offset
        type(lr_operand_desc_t), intent(out) :: result
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_error_t) :: error
        type(lr_operand_desc_t), target :: operands(2)
        type(lr_operand_desc_t) :: offset_op
        type(lr_inst_desc_t) :: gep_inst
        type(lr_inst_desc_t) :: load_inst
        integer(c_int32_t) :: vreg

        emit_i64_load_at = .false.
        if (.not. require_open_session(session, error_msg)) return

        offset_op%kind = LR_OP_KIND_IMM_I64
        offset_op%payload = offset
        offset_op%typ = lr_type_i64_s(session%handle)
        offset_op%global_offset = 0_c_int64_t

        ! GEP: gep_result = base + offset (byte offset for ptr type)
        operands(1) = base
        operands(2) = offset_op

        gep_inst%op = LR_OP_GEP
        gep_inst%typ = lr_type_ptr_s(session%handle)
        gep_inst%dest = 0_c_int32_t
        gep_inst%operands = c_loc(operands)
        gep_inst%num_operands = 2_c_int32_t
        gep_inst%indices = c_null_ptr
        gep_inst%num_indices = 0_c_int32_t
        gep_inst%align = 0_c_int32_t
        gep_inst%icmp_pred = 0_c_int
        gep_inst%fcmp_pred = 0_c_int
        gep_inst%call_external_abi = c_false
        gep_inst%call_vararg = c_false
        gep_inst%call_fixed_args = 0_c_int32_t

        call clear_liric_error(error)
        vreg = lr_session_emit(session%handle, gep_inst, error)
        if (.not. status_ok(error%code, error, error_msg)) return

        ! Load i64 from the computed address
        operands(1) = ptr_vreg(session, vreg)

        load_inst%op = LR_OP_LOAD
        load_inst%typ = lr_type_i64_s(session%handle)
        load_inst%dest = 0_c_int32_t
        load_inst%operands = c_loc(operands)
        load_inst%num_operands = 1_c_int32_t
        load_inst%indices = c_null_ptr
        load_inst%num_indices = 0_c_int32_t
        load_inst%align = 0_c_int32_t
        load_inst%icmp_pred = 0_c_int
        load_inst%fcmp_pred = 0_c_int
        load_inst%call_external_abi = c_false
        load_inst%call_vararg = c_false
        load_inst%call_fixed_args = 0_c_int32_t

        call clear_liric_error(error)
        vreg = lr_session_emit(session%handle, load_inst, error)
        if (.not. status_ok(error%code, error, error_msg)) return

        result = i64_vreg(session, vreg)
        call set_empty(error_msg)
        emit_i64_load_at = .true.
    end function emit_i64_load_at

    logical function emit_ptr_offset(session, base, offset, result, error_msg)
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: base
        integer(c_int64_t), intent(in) :: offset
        type(lr_operand_desc_t), intent(out) :: result
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_error_t) :: error
        type(lr_operand_desc_t), target :: operands(2)
        type(lr_operand_desc_t) :: offset_op
        type(lr_inst_desc_t) :: inst
        integer(c_int32_t) :: vreg

        emit_ptr_offset = .false.
        if (.not. require_open_session(session, error_msg)) return

        offset_op%kind = LR_OP_KIND_IMM_I64
        offset_op%payload = offset
        offset_op%typ = lr_type_i64_s(session%handle)
        offset_op%global_offset = 0_c_int64_t

        operands(1) = base
        operands(2) = offset_op

        inst%op = LR_OP_GEP
        inst%typ = lr_type_ptr_s(session%handle)
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

        result = ptr_vreg(session, vreg)
        call set_empty(error_msg)
        emit_ptr_offset = .true.
    end function emit_ptr_offset

    logical function emit_ptr_offset_dyn(session, base, offset, result, error_msg)
        ! GEP a raw pointer by a runtime i64 byte offset: result = base + offset.
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: base
        type(lr_operand_desc_t), intent(in) :: offset
        type(lr_operand_desc_t), intent(out) :: result
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_error_t) :: error
        type(lr_operand_desc_t), target :: operands(2)
        type(lr_inst_desc_t) :: inst
        integer(c_int32_t) :: vreg

        emit_ptr_offset_dyn = .false.
        if (.not. require_open_session(session, error_msg)) return

        operands(1) = base
        operands(2) = offset

        inst%op = LR_OP_GEP
        inst%typ = lr_type_ptr_s(session%handle)
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

        result = ptr_vreg(session, vreg)
        call set_empty(error_msg)
        emit_ptr_offset_dyn = .true.
    end function emit_ptr_offset_dyn

    logical function emit_i64_store_at(session, value, base, offset, &
                                       error_msg)
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: value
        type(lr_operand_desc_t), intent(in) :: base
        integer(c_int64_t), intent(in) :: offset
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_error_t) :: error
        type(lr_operand_desc_t), target :: operands(2)
        type(lr_operand_desc_t) :: offset_op
        type(lr_operand_desc_t) :: addr
        type(lr_inst_desc_t) :: gep_inst
        type(lr_inst_desc_t) :: store_inst
        integer(c_int32_t) :: vreg

        emit_i64_store_at = .false.
        if (.not. require_open_session(session, error_msg)) return

        offset_op%kind = LR_OP_KIND_IMM_I64
        offset_op%payload = offset
        offset_op%typ = lr_type_i64_s(session%handle)
        offset_op%global_offset = 0_c_int64_t

        ! GEP: addr = base + offset
        operands(1) = base
        operands(2) = offset_op

        gep_inst%op = LR_OP_GEP
        gep_inst%typ = lr_type_ptr_s(session%handle)
        gep_inst%dest = 0_c_int32_t
        gep_inst%operands = c_loc(operands)
        gep_inst%num_operands = 2_c_int32_t
        gep_inst%indices = c_null_ptr
        gep_inst%num_indices = 0_c_int32_t
        gep_inst%align = 0_c_int32_t
        gep_inst%icmp_pred = 0_c_int
        gep_inst%fcmp_pred = 0_c_int
        gep_inst%call_external_abi = c_false
        gep_inst%call_vararg = c_false
        gep_inst%call_fixed_args = 0_c_int32_t

        call clear_liric_error(error)
        vreg = lr_session_emit(session%handle, gep_inst, error)
        if (.not. status_ok(error%code, error, error_msg)) return

        addr = ptr_vreg(session, vreg)

        ! Store value at addr
        operands(1) = value
        operands(2) = addr

        store_inst%op = LR_OP_STORE
        store_inst%typ = c_null_ptr
        store_inst%dest = 0_c_int32_t
        store_inst%operands = c_loc(operands)
        store_inst%num_operands = 2_c_int32_t
        store_inst%indices = c_null_ptr
        store_inst%num_indices = 0_c_int32_t
        store_inst%align = 0_c_int32_t
        store_inst%icmp_pred = 0_c_int
        store_inst%fcmp_pred = 0_c_int
        store_inst%call_external_abi = c_false
        store_inst%call_vararg = c_false
        store_inst%call_fixed_args = 0_c_int32_t

        call clear_liric_error(error)
        vreg = lr_session_emit(session%handle, store_inst, error)
        if (.not. status_ok(error%code, error, error_msg)) return

        call set_empty(error_msg)
        emit_i64_store_at = .true.
    end function emit_i64_store_at

    logical function emit_i64_binary(session, opcode, lhs, rhs, result, &
                                     error_msg)
        type(liric_session_t), intent(inout) :: session
        integer(c_int), intent(in) :: opcode
        type(lr_operand_desc_t), intent(in) :: lhs
        type(lr_operand_desc_t), intent(in) :: rhs
        type(lr_operand_desc_t), intent(out) :: result
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_error_t) :: error
        integer(c_int32_t) :: vreg

        emit_i64_binary = .false.
        if (.not. require_open_session(session, error_msg)) return

        vreg = emit_binary_i64(session%handle, opcode, lhs, rhs, error)
        if (.not. status_ok(error%code, error, error_msg)) return

        result = i64_vreg(session, vreg)
        call set_empty(error_msg)
        emit_i64_binary = .true.
    end function emit_i64_binary

    logical function emit_alloca_bytes(session, size, result, error_msg)
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: size
        type(lr_operand_desc_t), intent(out) :: result
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_error_t) :: error
        integer(c_int32_t) :: vreg

        emit_alloca_bytes = .false.
        if (.not. require_open_session(session, error_msg)) return

        vreg = emit_alloca_i64_bytes(session%handle, size, error)
        if (.not. status_ok(error%code, error, error_msg)) return

        result = ptr_vreg(session, vreg)
        call set_empty(error_msg)
        emit_alloca_bytes = .true.
    end function emit_alloca_bytes

    logical function emit_malloc(session, size, result, error_msg)
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: size
        type(lr_operand_desc_t), intent(out) :: result
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_error_t) :: error
        type(lr_operand_desc_t) :: args(1)
        integer(c_int32_t) :: vreg

        emit_malloc = .false.
        if (.not. require_open_session(session, error_msg)) return

        args(1) = size
        vreg = emit_malloc_call(session%handle, args, error)
        if (.not. status_ok(error%code, error, error_msg)) return

        result = ptr_vreg(session, vreg)
        call set_empty(error_msg)
        emit_malloc = .true.
    end function emit_malloc

    logical function emit_free(session, ptr, error_msg)
        ! free(ptr). free(NULL) is a no-op, so callers need not null-check.
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: ptr
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_error_t) :: error
        integer(c_int32_t) :: vreg

        emit_free = .false.
        if (.not. require_open_session(session, error_msg)) return

        vreg = emit_free_call(session%handle, ptr, error)
        if (.not. status_ok(error%code, error, error_msg)) return

        call set_empty(error_msg)
        emit_free = .true.
    end function emit_free

    function emit_free_call(handle, ptr, error) result(vreg)
        type(c_ptr), intent(in) :: handle
        type(lr_operand_desc_t), intent(in) :: ptr
        type(lr_error_t), intent(inout) :: error
        integer(c_int32_t) :: vreg
        type(lr_operand_desc_t), target :: operands(2)
        type(lr_inst_desc_t) :: inst
        character(kind=c_char), allocatable :: c_name(:)
        integer(c_int32_t) :: symbol_id

        call to_c_chars('free', c_name)
        symbol_id = lr_session_intern(handle, c_name)
        if (symbol_id < 0_c_int32_t) then
            call clear_liric_error(error)
            error%code = 1_c_int
            return
        end if

        operands(1) = ptr_global_operand(handle, symbol_id)
        operands(2) = ptr

        inst%op = LR_OP_CALL
        inst%typ = c_null_ptr
        inst%dest = 0_c_int32_t
        inst%operands = c_loc(operands)
        inst%num_operands = 2_c_int32_t
        inst%indices = c_null_ptr
        inst%num_indices = 0_c_int32_t
        inst%align = 0_c_int32_t
        inst%icmp_pred = 0_c_int
        inst%fcmp_pred = 0_c_int
        inst%call_external_abi = c_true
        inst%call_vararg = c_false
        inst%call_fixed_args = 1_c_int32_t

        call clear_liric_error(error)
        vreg = lr_session_emit(handle, inst, error)
    end function emit_free_call

    logical function emit_ptr_store(session, value, address, error_msg)
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: value
        type(lr_operand_desc_t), intent(in) :: address
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_error_t) :: error
        integer(c_int32_t) :: unused_vreg

        emit_ptr_store = .false.
        if (.not. require_open_session(session, error_msg)) return

        unused_vreg = emit_store_ptr(session%handle, value, address, error)
        if (.not. status_ok(error%code, error, error_msg)) return

        call set_empty(error_msg)
        emit_ptr_store = .true.
    end function emit_ptr_store

    logical function emit_memcpy(session, dest, src, n_bytes, error_msg)
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: dest
        type(lr_operand_desc_t), intent(in) :: src
        type(lr_operand_desc_t), intent(in) :: n_bytes
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_error_t) :: error
        type(lr_operand_desc_t) :: args(3)
        integer(c_int32_t) :: unused_vreg

        emit_memcpy = .false.
        if (.not. require_open_session(session, error_msg)) return

        args(1) = dest
        args(2) = src
        args(3) = n_bytes

        unused_vreg = emit_memcpy_call(session%handle, args, error)
        if (.not. status_ok(error%code, error, error_msg)) return

        call set_empty(error_msg)
        emit_memcpy = .true.
    end function emit_memcpy

    logical function emit_i32_array_alloca(session, array_size, address, &
                                           error_msg)
        type(liric_session_t), intent(inout) :: session
        integer, intent(in) :: array_size
        type(lr_operand_desc_t), intent(out) :: address
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_error_t) :: error
        type(c_ptr) :: array_type
        integer(c_int32_t) :: vreg
        type(lr_inst_desc_t) :: inst

        emit_i32_array_alloca = .false.
        if (.not. require_open_session(session, error_msg)) return

        array_type = lr_type_array_s(session%handle, &
                                     lr_type_i32_s(session%handle), &
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
        vreg = lr_session_emit(session%handle, inst, error)
        if (.not. status_ok(error%code, error, error_msg)) then
            error_msg = 'array_alloca: '//error_msg
            return
        end if

        address = ptr_vreg(session, vreg)
        call set_empty(error_msg)
        emit_i32_array_alloca = .true.
    end function emit_i32_array_alloca

    logical function emit_i32_array_element_addr(session, array_size, &
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

        emit_i32_array_element_addr = .false.
        if (.not. require_open_session(session, error_msg)) return

        array_type = lr_type_array_s(session%handle, &
                                     lr_type_i32_s(session%handle), &
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
            error_msg = 'gep: '//error_msg
            return
        end if

        element_addr = ptr_vreg(session, vreg)
        call set_empty(error_msg)
        emit_i32_array_element_addr = .true.
    end function emit_i32_array_element_addr
    include 'liric_session_memory_bindings_tail.inc'

end module liric_session_memory_bindings
