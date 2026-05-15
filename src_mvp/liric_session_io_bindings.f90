module liric_session_io_bindings
    use, intrinsic :: iso_c_binding, only: c_associated, c_bool, c_char
    use, intrinsic :: iso_c_binding, only: c_int, c_int32_t, c_int64_t
    use, intrinsic :: iso_c_binding, only: c_loc, c_null_char, c_null_ptr
    use, intrinsic :: iso_c_binding, only: c_ptr, c_size_t
    use liric_session_bindings, only: liric_session_error_message, &
                                      liric_session_t, lr_error_t, &
                                      lr_inst_desc_t, lr_operand_desc_t, LR_OK, &
                                      LR_OP_KIND_VREG
    implicit none
    private

    integer(c_int), parameter :: LR_OP_CALL = 30_c_int
    integer(c_int), parameter :: LR_OP_BITCAST = 36_c_int
    integer(c_int), parameter :: LR_OP_KIND_GLOBAL = 4_c_int
    logical(c_bool), parameter :: c_false = .false.
    logical(c_bool), parameter :: c_true = .true.

    public :: emit_liric_print_i32
    public :: prepare_liric_i32_print

    interface
        function lr_type_i8_s(handle) result(typ) bind(c)
            import :: c_ptr
            type(c_ptr), value :: handle
            type(c_ptr) :: typ
        end function lr_type_i8_s

        function lr_type_i32_s(handle) result(typ) bind(c)
            import :: c_ptr
            type(c_ptr), value :: handle
            type(c_ptr) :: typ
        end function lr_type_i32_s

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

        function lr_session_declare(handle, name, ret, params, n, vararg, &
                                    err) result(status) bind(c)
            import :: c_bool, c_char, c_int, c_int32_t, c_ptr, lr_error_t
            type(c_ptr), value :: handle
            character(kind=c_char), intent(in) :: name(*)
            type(c_ptr), value :: ret
            type(c_ptr), value :: params
            integer(c_int32_t), value :: n
            logical(c_bool), value :: vararg
            type(lr_error_t), intent(inout) :: err
            integer(c_int) :: status
        end function lr_session_declare

        function lr_session_global(handle, name, typ, is_const, init, &
                                   init_size) result(global_id) bind(c)
            import :: c_bool, c_char, c_int32_t, c_ptr, c_size_t
            type(c_ptr), value :: handle
            character(kind=c_char), intent(in) :: name(*)
            type(c_ptr), value :: typ
            logical(c_bool), value :: is_const
            type(c_ptr), value :: init
            integer(c_size_t), value :: init_size
            integer(c_int32_t) :: global_id
        end function lr_session_global

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

    logical function prepare_liric_i32_print(session, format_global_id, error_msg)
        type(liric_session_t), intent(inout) :: session
        integer(c_int32_t), intent(out) :: format_global_id
        character(len=:), allocatable, intent(out) :: error_msg

        prepare_liric_i32_print = .false.
        if (.not. declare_printf_i32(session, error_msg)) return
        call create_i32_format_global(session, '.ffc.fmt.i32', &
                                      format_global_id, error_msg)
        if (len_trim(error_msg) > 0) return

        call set_empty(error_msg)
        prepare_liric_i32_print = .true.
    end function prepare_liric_i32_print

    logical function declare_printf_i32(session, error_msg)
        type(liric_session_t), intent(inout) :: session
        character(len=:), allocatable, intent(out) :: error_msg
        character(kind=c_char), allocatable :: c_name(:)
        type(c_ptr), target :: params(1)
        type(lr_error_t) :: error
        integer(c_int) :: status

        declare_printf_i32 = .false.
        if (.not. require_open_session(session, error_msg)) return

        params(1) = lr_type_ptr_s(session%handle)
        call clear_liric_error(error)
        call to_c_chars('printf', c_name)
        status = lr_session_declare(session%handle, c_name, &
                                    lr_type_i32_s(session%handle), &
                                    c_loc(params), 1_c_int32_t, c_true, &
                                    error)
        if (.not. status_ok(status, error, error_msg)) return

        call set_empty(error_msg)
        declare_printf_i32 = .true.
    end function declare_printf_i32

    logical function emit_liric_print_i32(session, format_global_id, value, &
                                          error_msg)
        type(liric_session_t), intent(inout) :: session
        integer(c_int32_t), intent(in) :: format_global_id
        type(lr_operand_desc_t), intent(in) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: callee
        type(lr_operand_desc_t) :: format_ptr
        type(lr_error_t) :: error
        integer(c_int32_t) :: unused_vreg

        emit_liric_print_i32 = .false.
        if (.not. require_open_session(session, error_msg)) return

        call make_printf_operand(session, callee, error_msg)
        if (len_trim(error_msg) > 0) return

        call materialize_global_pointer(session, format_global_id, format_ptr, &
                                        error_msg)
        if (len_trim(error_msg) > 0) return

        unused_vreg = emit_call_i32(session%handle, callee, format_ptr, value, &
                                    error)
        if (.not. status_ok(error%code, error, error_msg)) return

        call set_empty(error_msg)
        emit_liric_print_i32 = .true.
    end function emit_liric_print_i32

    subroutine make_printf_operand(session, operand, error_msg)
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(out) :: operand
        character(len=:), allocatable, intent(out) :: error_msg
        character(kind=c_char), allocatable :: c_name(:)
        integer(c_int32_t) :: symbol_id

        call to_c_chars('printf', c_name)
        symbol_id = lr_session_intern(session%handle, c_name)
        if (symbol_id < 0_c_int32_t) then
            error_msg = 'LIRIC could not intern printf'
            return
        end if

        operand = global_operand(session, symbol_id, lr_type_ptr_s(session%handle))
        call set_empty(error_msg)
    end subroutine make_printf_operand

    subroutine create_i32_format_global(session, name, global_id, error_msg)
        type(liric_session_t), intent(inout) :: session
        character(len=*), intent(in) :: name
        integer(c_int32_t), intent(out) :: global_id
        character(len=:), allocatable, intent(out) :: error_msg
        character(kind=c_char), allocatable :: c_name(:)
        character(kind=c_char), target :: format_text(4)
        type(c_ptr) :: array_type

        call to_c_chars(name, c_name)
        format_text(1) = '%'
        format_text(2) = 'd'
        format_text(3) = achar(10, kind=c_char)
        format_text(4) = c_null_char

        array_type = lr_type_array_s(session%handle, lr_type_i8_s(session%handle), &
                                     4_c_int64_t)
        if (.not. c_associated(array_type)) then
            error_msg = 'LIRIC did not return a format string array type'
            return
        end if

        global_id = lr_session_global(session%handle, c_name, array_type, &
                                      c_true, c_loc(format_text(1)), &
                                      4_c_size_t)
        if (global_id < 0_c_int32_t) then
            error_msg = 'LIRIC could not create printf format string'
            return
        end if

        global_id = lr_session_intern(session%handle, c_name)
        if (global_id < 0_c_int32_t) then
            error_msg = 'LIRIC could not intern printf format string'
            return
        end if

        call set_empty(error_msg)
    end subroutine create_i32_format_global

    subroutine materialize_global_pointer(session, global_id, operand, error_msg)
        type(liric_session_t), intent(inout) :: session
        integer(c_int32_t), intent(in) :: global_id
        type(lr_operand_desc_t), intent(out) :: operand
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: global_ref
        type(lr_error_t) :: error
        integer(c_int32_t) :: vreg

        global_ref = global_operand(session, global_id, &
                                    lr_type_ptr_s(session%handle))
        vreg = emit_bitcast_ptr(session%handle, global_ref, error)
        if (.not. status_ok(error%code, error, error_msg)) return

        operand%kind = LR_OP_KIND_VREG
        operand%payload = int(vreg, c_int64_t)
        operand%typ = lr_type_ptr_s(session%handle)
        operand%global_offset = 0_c_int64_t
        call set_empty(error_msg)
    end subroutine materialize_global_pointer

    function emit_call_i32(handle, callee, format_ptr, value, error) result(vreg)
        type(c_ptr), intent(in) :: handle
        type(lr_operand_desc_t), intent(in) :: callee
        type(lr_operand_desc_t), intent(in) :: format_ptr
        type(lr_operand_desc_t), intent(in) :: value
        type(lr_error_t), intent(inout) :: error
        integer(c_int32_t) :: vreg
        type(lr_operand_desc_t), target :: operands(3)
        type(lr_inst_desc_t) :: inst

        operands = [callee, format_ptr, value]

        inst%op = LR_OP_CALL
        inst%typ = lr_type_i32_s(handle)
        inst%dest = 0_c_int32_t
        inst%operands = c_loc(operands)
        inst%num_operands = 3_c_int32_t
        inst%indices = c_null_ptr
        inst%num_indices = 0_c_int32_t
        inst%align = 0_c_int32_t
        inst%icmp_pred = 0_c_int
        inst%fcmp_pred = 0_c_int
        inst%call_external_abi = c_true
        inst%call_vararg = c_true
        inst%call_fixed_args = 1_c_int32_t

        call clear_liric_error(error)
        vreg = lr_session_emit(handle, inst, error)
    end function emit_call_i32

    function emit_bitcast_ptr(handle, source, error) result(vreg)
        type(c_ptr), intent(in) :: handle
        type(lr_operand_desc_t), intent(in) :: source
        type(lr_error_t), intent(inout) :: error
        integer(c_int32_t) :: vreg
        type(lr_operand_desc_t), target :: operands(1)
        type(lr_inst_desc_t) :: inst

        operands(1) = source

        inst%op = LR_OP_BITCAST
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
    end function emit_bitcast_ptr

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

end module liric_session_io_bindings
