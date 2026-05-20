module liric_session_procedure_bindings
    use, intrinsic :: iso_c_binding, only: c_associated, c_bool, c_char
    use, intrinsic :: iso_c_binding, only: c_int, c_int32_t, c_int64_t
    use, intrinsic :: iso_c_binding, only: c_loc, c_null_char, c_null_ptr, c_ptr
    use liric_session_bindings, only: liric_session_error_message, &
                                      liric_session_t, lr_error_t, &
                                      lr_inst_desc_t, lr_operand_desc_t, LR_OK, &
                                      LR_OP_KIND_GLOBAL, LR_OP_KIND_VREG
    use liric_session_memory_bindings, only: ptr_param
    implicit none
    private

    integer(c_int), parameter :: LR_OP_CALL = 30_c_int
    integer(c_int), parameter :: LR_OP_ALLOCA = 26_c_int
    integer(c_int), parameter :: LR_OP_LOAD = 27_c_int
    integer(c_int), parameter :: LR_OP_STORE = 28_c_int
    logical(c_bool), parameter :: c_false = .false.

    public :: begin_liric_f64_function
    public :: emit_liric_f64_alloca
    public :: emit_liric_f64_call
    public :: emit_liric_f64_load
    public :: emit_liric_f64_store

    interface
        function lr_type_f64_s(handle) result(typ) bind(c)
            import :: c_ptr
            type(c_ptr), value :: handle
            type(c_ptr) :: typ
        end function lr_type_f64_s

        function lr_type_ptr_s(handle) result(typ) bind(c)
            import :: c_ptr
            type(c_ptr), value :: handle
            type(c_ptr) :: typ
        end function lr_type_ptr_s

        function lr_session_func_begin(handle, name, ret, params, n, &
                                       vararg, err) result(status) bind(c)
            import :: c_bool, c_char, c_int, c_int32_t, c_ptr, lr_error_t
            type(c_ptr), value :: handle
            character(kind=c_char), intent(in) :: name(*)
            type(c_ptr), value :: ret
            type(c_ptr), value :: params
            integer(c_int32_t), value :: n
            logical(c_bool), value :: vararg
            type(lr_error_t), intent(inout) :: err
            integer(c_int) :: status
        end function lr_session_func_begin

        function lr_session_block(handle) result(block_id) bind(c)
            import :: c_int32_t, c_ptr
            type(c_ptr), value :: handle
            integer(c_int32_t) :: block_id
        end function lr_session_block

        function lr_session_set_block(handle, block_id, err) result(status) &
            bind(c)
            import :: c_int, c_int32_t, c_ptr, lr_error_t
            type(c_ptr), value :: handle
            integer(c_int32_t), value :: block_id
            type(lr_error_t), intent(inout) :: err
            integer(c_int) :: status
        end function lr_session_set_block

        function lr_session_intern(handle, name) result(symbol_id) bind(c)
            import :: c_char, c_int32_t, c_ptr
            type(c_ptr), value :: handle
            character(kind=c_char), intent(in) :: name(*)
            integer(c_int32_t) :: symbol_id
        end function lr_session_intern

        function lr_session_emit(handle, inst, err) result(vreg) bind(c)
            import :: c_int32_t, c_ptr, lr_error_t, lr_inst_desc_t
            type(c_ptr), value :: handle
            type(lr_inst_desc_t), intent(in) :: inst
            type(lr_error_t), intent(inout) :: err
            integer(c_int32_t) :: vreg
        end function lr_session_emit
    end interface

contains

    logical function begin_liric_f64_function(session, name, param_count, &
                                              error_msg)
        type(liric_session_t), intent(inout) :: session
        character(len=*), intent(in) :: name
        integer, intent(in) :: param_count
        character(len=:), allocatable, intent(out) :: error_msg
        character(kind=c_char), allocatable :: c_name(:)
        type(c_ptr), allocatable, target :: params(:)
        type(c_ptr) :: params_ptr
        type(c_ptr) :: param_type
        type(lr_error_t) :: error
        integer(c_int32_t) :: block_id
        integer(c_int) :: status
        integer :: i

        begin_liric_f64_function = .false.
        if (.not. require_open_session(session, error_msg)) return

        param_type = lr_type_ptr_s(session%handle)
        if (.not. c_associated(param_type)) then
            error_msg = 'LIRIC did not return a pointer type'
            return
        end if

        params_ptr = c_null_ptr
        if (param_count > 0) then
            allocate (params(param_count))
            do i = 1, param_count
                params(i) = param_type
            end do
            params_ptr = c_loc(params)
        end if

        call clear_liric_error(error)
        call to_c_chars(trim(name), c_name)
        status = lr_session_func_begin(session%handle, c_name, &
                                       lr_type_f64_s(session%handle), &
                                       params_ptr, int(param_count, c_int32_t), &
                                       c_false, error)
        if (.not. status_ok(status, error, error_msg)) return

        block_id = lr_session_block(session%handle)
        status = lr_session_set_block(session%handle, block_id, error)
        if (.not. status_ok(status, error, error_msg)) return

        call set_empty(error_msg)
        begin_liric_f64_function = .true.
    end function begin_liric_f64_function

    logical function emit_liric_f64_alloca(session, address, error_msg)
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(out) :: address
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_error_t) :: error
        integer(c_int32_t) :: vreg

        emit_liric_f64_alloca = .false.
        if (.not. require_open_session(session, error_msg)) return

        vreg = emit_alloca_typed(session%handle, lr_type_f64_s(session%handle), &
                                 error)
        if (.not. status_ok(error%code, error, error_msg)) return

        address = ptr_vreg(session%handle, vreg)
        call set_empty(error_msg)
        emit_liric_f64_alloca = .true.
    end function emit_liric_f64_alloca

    logical function emit_liric_f64_load(session, address, value, error_msg)
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: address
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_error_t) :: error
        integer(c_int32_t) :: vreg

        emit_liric_f64_load = .false.
        if (.not. require_open_session(session, error_msg)) return

        vreg = emit_load_typed(session%handle, lr_type_f64_s(session%handle), &
                               address, error)
        if (.not. status_ok(error%code, error, error_msg)) return

        value = f64_vreg(session%handle, vreg)
        call set_empty(error_msg)
        emit_liric_f64_load = .true.
    end function emit_liric_f64_load

    logical function emit_liric_f64_store(session, value, address, error_msg)
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: value
        type(lr_operand_desc_t), intent(in) :: address
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_error_t) :: error
        integer(c_int32_t) :: unused_vreg

        emit_liric_f64_store = .false.
        if (.not. require_open_session(session, error_msg)) return

        unused_vreg = emit_store_typed(session%handle, value, address, error)
        if (.not. status_ok(error%code, error, error_msg)) return

        call set_empty(error_msg)
        emit_liric_f64_store = .true.
    end function emit_liric_f64_store

    logical function emit_liric_f64_call(session, name, args, result, error_msg)
        type(liric_session_t), intent(inout) :: session
        character(len=*), intent(in) :: name
        type(lr_operand_desc_t), intent(in) :: args(:)
        type(lr_operand_desc_t), intent(out) :: result
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_error_t) :: error
        integer(c_int32_t) :: vreg

        emit_liric_f64_call = .false.
        if (.not. require_open_session(session, error_msg)) return

        vreg = emit_call_f64_function(session%handle, name, args, error)
        if (.not. status_ok(error%code, error, error_msg)) return

        result = f64_vreg(session%handle, vreg)
        call set_empty(error_msg)
        emit_liric_f64_call = .true.
    end function emit_liric_f64_call

    function emit_alloca_typed(handle, typ, error) result(vreg)
        type(c_ptr), intent(in) :: handle
        type(c_ptr), intent(in) :: typ
        type(lr_error_t), intent(inout) :: error
        integer(c_int32_t) :: vreg
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
    end function emit_alloca_typed

    function emit_load_typed(handle, typ, address, error) result(vreg)
        type(c_ptr), intent(in) :: handle
        type(c_ptr), intent(in) :: typ
        type(lr_operand_desc_t), intent(in) :: address
        type(lr_error_t), intent(inout) :: error
        integer(c_int32_t) :: vreg
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
    end function emit_load_typed

    function emit_store_typed(handle, value, address, error) result(vreg)
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
    end function emit_store_typed

    function emit_call_f64_function(handle, name, args, error) result(vreg)
        type(c_ptr), intent(in) :: handle
        character(len=*), intent(in) :: name
        type(lr_operand_desc_t), intent(in) :: args(:)
        type(lr_error_t), intent(inout) :: error
        integer(c_int32_t) :: vreg
        type(lr_operand_desc_t), target :: operands(9)
        type(lr_inst_desc_t) :: inst
        character(kind=c_char), allocatable :: c_name(:)
        integer(c_int32_t) :: symbol_id
        integer :: i

        call to_c_chars(trim(name), c_name)
        symbol_id = lr_session_intern(handle, c_name)
        if (symbol_id < 0_c_int32_t .or. size(args) > 8) then
            call clear_liric_error(error)
            error%code = 1_c_int
            return
        end if

        operands(1) = global_operand(handle, symbol_id)
        do i = 1, size(args)
            operands(i + 1) = args(i)
        end do

        inst%op = LR_OP_CALL
        inst%typ = lr_type_f64_s(handle)
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
    end function emit_call_f64_function

    function f64_vreg(handle, vreg) result(operand)
        type(c_ptr), intent(in) :: handle
        integer(c_int32_t), intent(in) :: vreg
        type(lr_operand_desc_t) :: operand

        operand%kind = LR_OP_KIND_VREG
        operand%payload = int(vreg, c_int64_t)
        operand%typ = lr_type_f64_s(handle)
        operand%global_offset = 0_c_int64_t
    end function f64_vreg

    function ptr_vreg(handle, vreg) result(operand)
        type(c_ptr), intent(in) :: handle
        integer(c_int32_t), intent(in) :: vreg
        type(lr_operand_desc_t) :: operand

        operand%kind = LR_OP_KIND_VREG
        operand%payload = int(vreg, c_int64_t)
        operand%typ = lr_type_ptr_s(handle)
        operand%global_offset = 0_c_int64_t
    end function ptr_vreg

    function global_operand(handle, id) result(operand)
        type(c_ptr), intent(in) :: handle
        integer(c_int32_t), intent(in) :: id
        type(lr_operand_desc_t) :: operand

        operand%kind = LR_OP_KIND_GLOBAL
        operand%payload = int(id, c_int64_t)
        operand%typ = lr_type_ptr_s(handle)
        operand%global_offset = 0_c_int64_t
    end function global_operand

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

end module liric_session_procedure_bindings
