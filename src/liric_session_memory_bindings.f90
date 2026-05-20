module liric_session_memory_bindings
    use, intrinsic :: iso_c_binding, only: c_associated, c_bool, c_char, c_int, &
                                              c_int32_t, c_int64_t
    use, intrinsic :: iso_c_binding, only: c_loc, c_null_char, c_null_ptr, c_ptr
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

    type, bind(c), public :: lr_error_t
        integer(c_int) :: code = LR_OK
        character(kind=c_char) :: msg(256)
    end type lr_error_t

    type, bind(c), public :: lr_inst_desc_t
        integer(c_int) :: op = 0
        type(c_ptr) :: typ = c_null_ptr
        integer(c_int32_t) :: dest = 0_c_int32_t
        type(c_ptr) :: operands = c_null_ptr
        integer(c_int32_t) :: num_operands = 0_c_int32_t
        type(c_ptr) :: indices = c_null_ptr
        integer(c_int32_t) :: num_indices = 0_c_int32_t
        integer(c_int32_t) :: align = 0_c_int32_t
        integer(c_int) :: icmp_pred = 0_c_int
        integer(c_int) :: fcmp_pred = 0_c_int
        logical(c_bool) :: call_external_abi = .false.
        logical(c_bool) :: call_vararg = .false.
        integer(c_int32_t) :: call_fixed_args = 0_c_int32_t
    end type lr_inst_desc_t

    public :: i64_immediate, ptr_param, &
              reserve_i32_vreg, ptr_vreg, i64_vreg
    public :: emit_i32_binary, emit_i32_binary_into, emit_i32_copy_to
    public :: emit_i32_alloca, emit_i64_alloca
    public :: emit_i32_load, emit_i64_load, emit_i32_store, emit_i64_store
    public :: emit_alloca_bytes, emit_ptr_store, emit_memcpy
    public :: emit_i32_array_alloca, emit_i32_array_element_addr
    public :: emit_i64_binary

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

    function emit_binary(handle, opcode, lhs, rhs, error) result(vreg)
        type(c_ptr), intent(in) :: handle
        integer(c_int), intent(in) :: opcode
        type(lr_operand_desc_t), intent(in) :: lhs
        type(lr_operand_desc_t), intent(in) :: rhs
        type(lr_error_t), intent(inout) :: error
        integer(c_int32_t) :: vreg
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
    end function emit_binary

    function emit_binary_with_dest(handle, opcode, lhs, rhs, error, &
                                   dest_vreg) result(vreg)
        type(c_ptr), intent(in) :: handle
        integer(c_int), intent(in) :: opcode
        type(lr_operand_desc_t), intent(in) :: lhs
        type(lr_operand_desc_t), intent(in) :: rhs
        type(lr_error_t), intent(inout) :: error
        integer(c_int32_t), intent(in) :: dest_vreg
        integer(c_int32_t) :: vreg
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
    end function emit_binary_with_dest

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

    function emit_alloca_i64(handle, error) result(vreg)
        type(c_ptr), intent(in) :: handle
        type(lr_error_t), intent(inout) :: error
        integer(c_int32_t) :: vreg
        type(lr_inst_desc_t) :: inst

        inst%op = LR_OP_ALLOCA
        inst%typ = lr_type_i64_s(handle)
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
    end function emit_alloca_i64

    function emit_load_i32(handle, address, error) result(vreg)
        type(c_ptr), intent(in) :: handle
        type(lr_operand_desc_t), intent(in) :: address
        type(lr_error_t), intent(inout) :: error
        integer(c_int32_t) :: vreg
        type(lr_operand_desc_t), target :: operands(1)
        type(lr_inst_desc_t) :: inst

        operands(1) = address

        inst%op = LR_OP_LOAD
        inst%typ = lr_type_i32_s(handle)
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
    end function emit_load_i32

    function emit_store_i32(handle, value, address, error) result(vreg)
        type(c_ptr), intent(in) :: handle
        type(lr_operand_desc_t), intent(in) :: value
        type(lr_operand_desc_t), intent(in) :: address
        type(lr_error_t), intent(inout) :: error
        integer(c_int32_t) :: vreg
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
    end function emit_store_i32

    function emit_load_i64(handle, address, error) result(vreg)
        type(c_ptr), intent(in) :: handle
        type(lr_operand_desc_t), intent(in) :: address
        type(lr_error_t), intent(inout) :: error
        integer(c_int32_t) :: vreg
        type(lr_operand_desc_t), target :: operands(1)
        type(lr_inst_desc_t) :: inst

        operands(1) = address

        inst%op = LR_OP_LOAD
        inst%typ = lr_type_i64_s(handle)
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
    end function emit_load_i64

    function emit_store_i64(handle, value, address, error) result(vreg)
        type(c_ptr), intent(in) :: handle
        type(lr_operand_desc_t), intent(in) :: value
        type(lr_operand_desc_t), intent(in) :: address
        type(lr_error_t), intent(inout) :: error
        integer(c_int32_t) :: vreg
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
    end function emit_store_i64

    function emit_binary_i64(handle, opcode, lhs, rhs, error) result(vreg)
        type(c_ptr), intent(in) :: handle
        integer(c_int), intent(in) :: opcode
        type(lr_operand_desc_t), intent(in) :: lhs
        type(lr_operand_desc_t), intent(in) :: rhs
        type(lr_error_t), intent(inout) :: error
        integer(c_int32_t) :: vreg
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
    end function emit_binary_i64

    function emit_alloca_i64_bytes(handle, size, error) result(vreg)
        type(c_ptr), intent(in) :: handle
        type(lr_operand_desc_t), intent(in) :: size
        type(lr_error_t), intent(inout) :: error
        integer(c_int32_t) :: vreg
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
    end function emit_alloca_i64_bytes

    function emit_store_ptr(handle, value, address, error) result(vreg)
        type(c_ptr), intent(in) :: handle
        type(lr_operand_desc_t), intent(in) :: value
        type(lr_operand_desc_t), intent(in) :: address
        type(lr_error_t), intent(inout) :: error
        integer(c_int32_t) :: vreg
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
    end function emit_store_ptr

    function global_operand(session, id, typ) result(operand)
        type(liric_session_t), intent(in) :: session
        integer(c_int32_t), intent(in) :: id
        type(c_ptr), intent(in) :: typ
        type(lr_operand_desc_t) :: operand

        operand%kind = LR_OP_KIND_GLOBAL
        operand%payload = int(id, c_int64_t)
        operand%typ = typ
        operand%global_offset = 0_c_int64_t
    end function global_operand

    function emit_memcpy_call(handle, args, error) result(vreg)
        type(c_ptr), intent(in) :: handle
        type(lr_operand_desc_t), intent(in) :: args(:)
        type(lr_error_t), intent(inout) :: error
        integer(c_int32_t) :: vreg
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
    end function emit_memcpy_call

    function global_operand_from_id(handle, symbol_id) result(operand)
        type(c_ptr), intent(in) :: handle
        integer(c_int32_t), intent(in) :: symbol_id
        type(lr_operand_desc_t) :: operand

        operand%kind = LR_OP_KIND_GLOBAL
        operand%payload = int(symbol_id, c_int64_t)
        operand%typ = lr_type_i32_s(handle)
        operand%global_offset = 0_c_int64_t
    end function global_operand_from_id

    logical function require_open_session(session, error_msg)
        type(liric_session_t), intent(in) :: session
        character(len=:), allocatable, intent(out) :: error_msg

        require_open_session = c_associated(session%handle)
        if (require_open_session) then
            call set_empty(error_msg)
        else
            error_msg = 'LIRIC session handle is not open'
        end if
    end function require_open_session

    logical function status_ok(status, error, error_msg)
        integer(c_int), intent(in) :: status
        type(lr_error_t), intent(in) :: error
        character(len=:), allocatable, intent(out) :: error_msg

        status_ok = status == LR_OK
        if (status_ok) then
            call set_empty(error_msg)
        else
            error_msg = liric_session_error_message(error)
        end if
    end function status_ok

    function liric_session_error_message(error) result(message)
        type(lr_error_t), intent(in) :: error
        character(len=:), allocatable :: message
        integer :: i
        integer :: message_len

        message_len = 0
        do i = 1, size(error%msg)
            if (error%msg(i) == c_null_char) exit
            message_len = message_len + 1
        end do

        if (message_len == 0) then
            allocate (character(len=32) :: message)
            write (message, '(A,I0)') 'LIRIC error code ', error%code
            return
        end if

        allocate (character(len=message_len) :: message)
        do i = 1, message_len
            message(i:i) = error%msg(i)
        end do
    end function liric_session_error_message

    subroutine clear_liric_error(error)
        type(lr_error_t), intent(out) :: error

        error%code = LR_OK
        error%msg = c_null_char
    end subroutine clear_liric_error

    subroutine to_c_chars(text, chars)
        character(len=*), intent(in) :: text
        character(kind=c_char), allocatable, intent(out) :: chars(:)
        integer :: i

        allocate (chars(len(text) + 1))
        do i = 1, len(text)
            chars(i) = text(i:i)
        end do
        chars(len(text) + 1) = c_null_char
    end subroutine to_c_chars

    subroutine set_empty(value)
        character(len=:), allocatable, intent(out) :: value

        allocate (character(len=0) :: value)
    end subroutine set_empty

end module liric_session_memory_bindings
