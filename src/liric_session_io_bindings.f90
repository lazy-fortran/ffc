module liric_session_io_bindings
    use, intrinsic :: iso_c_binding, only: c_associated, c_bool, c_char
    use, intrinsic :: iso_c_binding, only: c_double, c_int, c_int32_t, &
                                                                              c_int64_t
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
    integer(c_int), parameter :: LR_OP_SITOFP = 39_c_int
    integer(c_int), parameter :: LR_OP_FADD = 18_c_int
    integer(c_int), parameter, public :: LR_OP_FSUB = 19_c_int
    integer(c_int), parameter :: LR_OP_FMUL = 20_c_int
    integer(c_int), parameter :: LR_OP_FDIV = 21_c_int
    integer(c_int), parameter :: LR_OP_KIND_IMM_F64 = 2_c_int
    integer(c_int), parameter :: LR_OP_KIND_GLOBAL = 4_c_int
    logical(c_bool), parameter :: c_false = .false.
    logical(c_bool), parameter :: c_true = .true.

    public :: emit_liric_f64_binary
    public :: emit_liric_i32_to_f64
    public :: emit_liric_print_f64
    public :: emit_liric_print_f64_value
    public :: emit_liric_print_i32
    public :: emit_liric_print_i32_value
    public :: emit_liric_print_newline
    public :: emit_liric_print_space
    public :: emit_liric_print_string
    public :: emit_liric_print_string_operand
    public :: emit_liric_print_string_operand_value
    public :: emit_liric_print_string_value
    public :: liric_f64_immediate
    public :: materialize_liric_string
    public :: prepare_liric_print_runtime

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

    logical function prepare_liric_print_runtime(session, i32_format_id, &
                                                 f64_format_id, str_format_id, &
                                                 error_msg)
        type(liric_session_t), intent(inout) :: session
        integer(c_int32_t), intent(out) :: i32_format_id
        integer(c_int32_t), intent(out) :: f64_format_id
        integer(c_int32_t), intent(out) :: str_format_id
        character(len=:), allocatable, intent(out) :: error_msg

        prepare_liric_print_runtime = .false.
        if (.not. declare_printf_i32(session, error_msg)) return
        call create_i32_format_global(session, '.ffc.fmt.i32', &
                                      i32_format_id, error_msg)
        if (len_trim(error_msg) > 0) return
        call create_i32_format_global_no_newline(session, &
                                                 '.ffc.fmt.i32.vn', &
                                                 i32_format_id, error_msg)
        if (len_trim(error_msg) > 0) return
        call create_f64_format_global(session, '.ffc.fmt.f64', &
                                      f64_format_id, error_msg)
        if (len_trim(error_msg) > 0) return
        call create_f64_format_global_no_newline(session, &
                                                 '.ffc.fmt.f64.vn', &
                                                 f64_format_id, error_msg)
        if (len_trim(error_msg) > 0) return
        call create_str_format_global(session, '.ffc.fmt.str', &
                                      str_format_id, error_msg)
        if (len_trim(error_msg) > 0) return
        call create_str_format_global_no_newline(session, &
                                                 '.ffc.fmt.str.vn', &
                                                 str_format_id, error_msg)
        if (len_trim(error_msg) > 0) return

        call set_empty(error_msg)
        prepare_liric_print_runtime = .true.
    end function prepare_liric_print_runtime

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

    logical function emit_liric_print_f64(session, format_global_id, value, &
                                          error_msg)
        type(liric_session_t), intent(inout) :: session
        integer(c_int32_t), intent(in) :: format_global_id
        type(lr_operand_desc_t), intent(in) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: callee
        type(lr_operand_desc_t) :: format_ptr
        type(lr_error_t) :: error
        integer(c_int32_t) :: unused_vreg

        emit_liric_print_f64 = .false.
        if (.not. require_open_session(session, error_msg)) return

        call make_printf_operand(session, callee, error_msg)
        if (len_trim(error_msg) > 0) return

        call materialize_global_pointer(session, format_global_id, format_ptr, &
                                        error_msg)
        if (len_trim(error_msg) > 0) return

        unused_vreg = emit_call_f64(session%handle, callee, format_ptr, value, &
                                    error)
        if (.not. status_ok(error%code, error, error_msg)) return

        call set_empty(error_msg)
        emit_liric_print_f64 = .true.
    end function emit_liric_print_f64

    logical function emit_liric_print_string(session, format_global_id, &
                                             string_global_name, text, error_msg)
        type(liric_session_t), intent(inout) :: session
        integer(c_int32_t), intent(in) :: format_global_id
        character(len=*), intent(in) :: string_global_name
        character(len=*), intent(in) :: text
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: string_ptr

        emit_liric_print_string = .false.
        call materialize_liric_string(session, string_global_name, text, &
                                      string_ptr, error_msg)
        if (len_trim(error_msg) > 0) return

        emit_liric_print_string = emit_liric_print_string_operand( &
                                  session, format_global_id, string_ptr, error_msg)
    end function emit_liric_print_string

  logical function emit_liric_print_string_operand(session, format_global_id, &
                                                      string_ptr, error_msg)
        type(liric_session_t), intent(inout) :: session
        integer(c_int32_t), intent(in) :: format_global_id
        type(lr_operand_desc_t), intent(in) :: string_ptr
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: callee
        type(lr_operand_desc_t) :: format_ptr
        type(lr_error_t) :: error
        integer(c_int32_t) :: unused_vreg

        emit_liric_print_string_operand = .false.
        if (.not. require_open_session(session, error_msg)) return

        call make_printf_operand(session, callee, error_msg)
        if (len_trim(error_msg) > 0) return

        call materialize_global_pointer(session, format_global_id, format_ptr, &
                                        error_msg)
        if (len_trim(error_msg) > 0) return

        unused_vreg = emit_call_ptr(session%handle, callee, format_ptr, &
                                    string_ptr, error)
        if (.not. status_ok(error%code, error, error_msg)) return

        call set_empty(error_msg)
        emit_liric_print_string_operand = .true.
    end function emit_liric_print_string_operand

    logical function emit_liric_print_i32_value(session, format_global_id, &
                                                value, error_msg)
        type(liric_session_t), intent(inout) :: session
        integer(c_int32_t), intent(in) :: format_global_id
        type(lr_operand_desc_t), intent(in) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: callee
        type(lr_operand_desc_t) :: format_ptr
        type(lr_error_t) :: error
        integer(c_int32_t) :: unused_vreg

        emit_liric_print_i32_value = .false.
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
        emit_liric_print_i32_value = .true.
    end function emit_liric_print_i32_value

    logical function emit_liric_print_f64_value(session, format_global_id, &
                                                value, error_msg)
        type(liric_session_t), intent(inout) :: session
        integer(c_int32_t), intent(in) :: format_global_id
        type(lr_operand_desc_t), intent(in) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: callee
        type(lr_operand_desc_t) :: format_ptr
        type(lr_error_t) :: error
        integer(c_int32_t) :: unused_vreg

        emit_liric_print_f64_value = .false.
        if (.not. require_open_session(session, error_msg)) return

        call make_printf_operand(session, callee, error_msg)
        if (len_trim(error_msg) > 0) return

        call materialize_global_pointer(session, format_global_id, format_ptr, &
                                        error_msg)
        if (len_trim(error_msg) > 0) return

        unused_vreg = emit_call_f64(session%handle, callee, format_ptr, value, &
                                    error)
        if (.not. status_ok(error%code, error, error_msg)) return

        call set_empty(error_msg)
        emit_liric_print_f64_value = .true.
    end function emit_liric_print_f64_value

    logical function emit_liric_print_string_value(session, format_global_id, &
                                                   string_global_name, text, &
                                                   error_msg)
        type(liric_session_t), intent(inout) :: session
        integer(c_int32_t), intent(in) :: format_global_id
        character(len=*), intent(in) :: string_global_name
        character(len=*), intent(in) :: text
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: string_ptr

        emit_liric_print_string_value = .false.
        call materialize_liric_string(session, string_global_name, text, &
                                      string_ptr, error_msg)
        if (len_trim(error_msg) > 0) return

        emit_liric_print_string_value = emit_liric_print_string_operand_value( &
                                          session, format_global_id, string_ptr, &
                                          error_msg)
    end function emit_liric_print_string_value

    logical function emit_liric_print_string_operand_value(session, &
                                                           format_global_id, &
                                                           string_ptr, &
                                                           error_msg)
        type(liric_session_t), intent(inout) :: session
        integer(c_int32_t), intent(in) :: format_global_id
        type(lr_operand_desc_t), intent(in) :: string_ptr
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: callee
        type(lr_operand_desc_t) :: format_ptr
        type(lr_error_t) :: error
        integer(c_int32_t) :: unused_vreg

        emit_liric_print_string_operand_value = .false.
        if (.not. require_open_session(session, error_msg)) return

        call make_printf_operand(session, callee, error_msg)
        if (len_trim(error_msg) > 0) return

        call materialize_global_pointer(session, format_global_id, format_ptr, &
                                        error_msg)
        if (len_trim(error_msg) > 0) return

        unused_vreg = emit_call_ptr(session%handle, callee, format_ptr, &
                                    string_ptr, error)
        if (.not. status_ok(error%code, error, error_msg)) return

        call set_empty(error_msg)
        emit_liric_print_string_operand_value = .true.
    end function emit_liric_print_string_operand_value

    logical function emit_liric_print_newline(session, error_msg)
        type(liric_session_t), intent(inout) :: session
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: callee
        type(lr_error_t) :: error
        integer(c_int32_t) :: unused_vreg

        emit_liric_print_newline = .false.
        if (.not. require_open_session(session, error_msg)) return

        call make_printf_operand(session, callee, error_msg)
        if (len_trim(error_msg) > 0) return

        unused_vreg = emit_call_newline(session%handle, callee, error)
        if (.not. status_ok(error%code, error, error_msg)) return

        call set_empty(error_msg)
        emit_liric_print_newline = .true.
    end function emit_liric_print_newline

    logical function emit_liric_print_space(session, error_msg)
        type(liric_session_t), intent(inout) :: session
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: callee
        type(lr_error_t) :: error
        integer(c_int32_t) :: unused_vreg

        emit_liric_print_space = .false.
        if (.not. require_open_session(session, error_msg)) return

        call make_printf_operand(session, callee, error_msg)
        if (len_trim(error_msg) > 0) return

        unused_vreg = emit_call_space(session%handle, callee, error)
        if (.not. status_ok(error%code, error, error_msg)) return

        call set_empty(error_msg)
        emit_liric_print_space = .true.
    end function emit_liric_print_space

    subroutine materialize_liric_string(session, string_global_name, text, &
                                        operand, error_msg)
        type(liric_session_t), intent(inout) :: session
        character(len=*), intent(in) :: string_global_name
        character(len=*), intent(in) :: text
        type(lr_operand_desc_t), intent(out) :: operand
        character(len=:), allocatable, intent(out) :: error_msg

        call make_string_literal_operand(session, string_global_name, text, &
                                         operand, error_msg)
    end subroutine materialize_liric_string

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

    subroutine create_i32_format_global_no_newline(session, name, global_id, &
                                                   error_msg)
        type(liric_session_t), intent(inout) :: session
        character(len=*), intent(in) :: name
        integer(c_int32_t), intent(out) :: global_id
        character(len=:), allocatable, intent(out) :: error_msg
        character(kind=c_char), allocatable :: c_name(:)
        character(kind=c_char), target :: format_text(3)
        type(c_ptr) :: array_type

        call to_c_chars(name, c_name)
        format_text(1) = '%'
        format_text(2) = 'd'
        format_text(3) = c_null_char

        array_type = lr_type_array_s(session%handle, lr_type_i8_s(session%handle), &
                                      3_c_int64_t)
        if (.not. c_associated(array_type)) then
            error_msg = 'LIRIC did not return a format string array type'
            return
        end if

        global_id = lr_session_global(session%handle, c_name, array_type, &
                                      c_true, c_loc(format_text(1)), &
                                      3_c_size_t)
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
    end subroutine create_i32_format_global_no_newline

subroutine create_f64_format_global(session, name, global_id, error_msg)
        type(liric_session_t), intent(inout) :: session
        character(len=*), intent(in) :: name
        integer(c_int32_t), intent(out) :: global_id
        character(len=:), allocatable, intent(out) :: error_msg
        character(kind=c_char), allocatable :: c_name(:)
        character(kind=c_char), target :: format_text(4)
        type(c_ptr) :: array_type

        call to_c_chars(name, c_name)
        format_text(1) = '%'
        format_text(2) = 'f'
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
    end subroutine create_f64_format_global

    subroutine create_f64_format_global_no_newline(session, name, global_id, &
                                                   error_msg)
        type(liric_session_t), intent(inout) :: session
        character(len=*), intent(in) :: name
        integer(c_int32_t), intent(out) :: global_id
        character(len=:), allocatable, intent(out) :: error_msg
        character(kind=c_char), allocatable :: c_name(:)
        character(kind=c_char), target :: format_text(3)
        type(c_ptr) :: array_type

        call to_c_chars(name, c_name)
        format_text(1) = '%'
        format_text(2) = 'f'
        format_text(3) = c_null_char

        array_type = lr_type_array_s(session%handle, lr_type_i8_s(session%handle), &
                                      3_c_int64_t)
        if (.not. c_associated(array_type)) then
            error_msg = 'LIRIC did not return a format string array type'
            return
        end if

        global_id = lr_session_global(session%handle, c_name, array_type, &
                                      c_true, c_loc(format_text(1)), &
                                      3_c_size_t)
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
    end subroutine create_f64_format_global_no_newline

subroutine create_str_format_global(session, name, global_id, error_msg)
        type(liric_session_t), intent(inout) :: session
        character(len=*), intent(in) :: name
        integer(c_int32_t), intent(out) :: global_id
        character(len=:), allocatable, intent(out) :: error_msg
        character(kind=c_char), allocatable :: c_name(:)
        character(kind=c_char), target :: format_text(4)
        type(c_ptr) :: array_type

        call to_c_chars(name, c_name)
        format_text(1) = '%'
        format_text(2) = 's'
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
    end subroutine create_str_format_global

    subroutine create_str_format_global_no_newline(session, name, global_id, &
                                                   error_msg)
        type(liric_session_t), intent(inout) :: session
        character(len=*), intent(in) :: name
        integer(c_int32_t), intent(out) :: global_id
        character(len=:), allocatable, intent(out) :: error_msg
        character(kind=c_char), allocatable :: c_name(:)
        character(kind=c_char), target :: format_text(3)
        type(c_ptr) :: array_type

        call to_c_chars(name, c_name)
        format_text(1) = '%'
        format_text(2) = 's'
        format_text(3) = c_null_char

        array_type = lr_type_array_s(session%handle, lr_type_i8_s(session%handle), &
                                      3_c_int64_t)
        if (.not. c_associated(array_type)) then
            error_msg = 'LIRIC did not return a format string array type'
            return
        end if

        global_id = lr_session_global(session%handle, c_name, array_type, &
                                      c_true, c_loc(format_text(1)), &
                                      3_c_size_t)
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
    end subroutine create_str_format_global_no_newline

subroutine make_string_literal_operand(session, name, text, operand, error_msg)
        type(liric_session_t), intent(inout) :: session
        character(len=*), intent(in) :: name
        character(len=*), intent(in) :: text
        type(lr_operand_desc_t), intent(out) :: operand
        character(len=:), allocatable, intent(out) :: error_msg
        character(kind=c_char), allocatable :: c_name(:)
        character(kind=c_char), allocatable, target :: text_chars(:)
        type(c_ptr) :: array_type
        integer(c_int32_t) :: global_id
        integer :: i

        call to_c_chars(name, c_name)
        allocate (text_chars(len(text) + 1))
        do i = 1, len(text)
            text_chars(i) = text(i:i)
        end do
        text_chars(len(text) + 1) = c_null_char

        array_type = lr_type_array_s(session%handle, lr_type_i8_s(session%handle), &
                                      int(len(text) + 1, c_int64_t))
        if (.not. c_associated(array_type)) then
            error_msg = 'LIRIC did not return a string literal array type'
            return
        end if

        global_id = lr_session_global(session%handle, c_name, array_type, &
                                      c_true, c_loc(text_chars(1)), &
                                      int(len(text) + 1, c_size_t))
        if (global_id < 0_c_int32_t) then
            error_msg = 'LIRIC could not create string literal'
            return
        end if

        global_id = lr_session_intern(session%handle, c_name)
        if (global_id < 0_c_int32_t) then
            error_msg = 'LIRIC could not intern string literal'
            return
        end if

        call materialize_global_pointer(session, global_id, operand, error_msg)
    end subroutine make_string_literal_operand

    subroutine make_string_literal_operand_from_chars(handle, name, text_chars, &
                                                      total_size, operand, error)
        type(c_ptr), intent(in) :: handle
        character(len=*), intent(in) :: name
        character(kind=c_char), intent(in), target :: text_chars(:)
        integer(c_int64_t), intent(in) :: total_size
        type(lr_operand_desc_t), intent(out) :: operand
        type(lr_error_t), intent(inout) :: error
        character(kind=c_char), allocatable :: c_name(:)
        type(c_ptr) :: array_type
        integer(c_int32_t) :: global_id

        call to_c_chars(name, c_name)

        array_type = lr_type_array_s(handle, lr_type_i8_s(handle), total_size)
        if (.not. c_associated(array_type)) then
            error%code = 1_c_int
            error%msg = c_null_char
            return
        end if

        global_id = lr_session_global(handle, c_name, array_type, c_true, &
                                      c_loc(text_chars(1)), &
                                      int(total_size, c_size_t))
        if (global_id < 0_c_int32_t) then
            error%code = 1_c_int
            error%msg = c_null_char
            return
        end if

        global_id = lr_session_intern(handle, c_name)
        if (global_id < 0_c_int32_t) then
            error%code = 1_c_int
            error%msg = c_null_char
            return
        end if

        operand%kind = LR_OP_KIND_GLOBAL
        operand%payload = int(global_id, c_int64_t)
        operand%typ = lr_type_ptr_s(handle)
        operand%global_offset = 0_c_int64_t
    end subroutine make_string_literal_operand_from_chars

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

    function emit_call_f64(handle, callee, format_ptr, value, error) result(vreg)
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
    end function emit_call_f64

    function emit_call_ptr(handle, callee, format_ptr, value, error) result(vreg)
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
    end function emit_call_ptr

    function emit_call_newline(handle, callee, error) result(vreg)
        type(c_ptr), intent(in) :: handle
        type(lr_operand_desc_t), intent(in) :: callee
        type(lr_error_t), intent(inout) :: error
        integer(c_int32_t) :: vreg
        type(lr_operand_desc_t), target :: operands(2)
        type(lr_inst_desc_t) :: inst
        character(kind=c_char), target :: newline_text(2)

        newline_text(1) = achar(10, kind=c_char)
        newline_text(2) = c_null_char

        call make_string_literal_operand_from_chars( &
            handle, '.ffc.str.nl', newline_text, 2_c_int64_t, &
            operands(2), error)
        if (.not. c_associated(operands(2)%typ)) return

        operands(1) = callee

        inst%op = LR_OP_CALL
        inst%typ = lr_type_i32_s(handle)
        inst%dest = 0_c_int32_t
        inst%operands = c_loc(operands)
        inst%num_operands = 2_c_int32_t
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
    end function emit_call_newline

    function emit_call_space(handle, callee, error) result(vreg)
        type(c_ptr), intent(in) :: handle
        type(lr_operand_desc_t), intent(in) :: callee
        type(lr_error_t), intent(inout) :: error
        integer(c_int32_t) :: vreg
        type(lr_operand_desc_t), target :: operands(2)
        type(lr_inst_desc_t) :: inst
        character(kind=c_char), target :: space_text(2)

        space_text(1) = achar(32, kind=c_char)
        space_text(2) = c_null_char

        call make_string_literal_operand_from_chars( &
            handle, '.ffc.str.sp', space_text, 2_c_int64_t, &
            operands(2), error)
        if (.not. c_associated(operands(2)%typ)) return

        operands(1) = callee

        inst%op = LR_OP_CALL
        inst%typ = lr_type_i32_s(handle)
        inst%dest = 0_c_int32_t
        inst%operands = c_loc(operands)
        inst%num_operands = 2_c_int32_t
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
    end function emit_call_space

    function liric_f64_immediate(session, value) result(operand)
        type(liric_session_t), intent(in) :: session
        real(c_double), intent(in) :: value
        type(lr_operand_desc_t) :: operand

        operand%kind = LR_OP_KIND_IMM_F64
        operand%payload = transfer(value, operand%payload)
        operand%typ = lr_type_f64_s(session%handle)
        operand%global_offset = 0_c_int64_t
    end function liric_f64_immediate

    logical function emit_liric_f64_binary(session, opcode, lhs, rhs, result, &
                                           error_msg)
        type(liric_session_t), intent(inout) :: session
        integer(c_int), intent(in) :: opcode
        type(lr_operand_desc_t), intent(in) :: lhs
        type(lr_operand_desc_t), intent(in) :: rhs
        type(lr_operand_desc_t), intent(out) :: result
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_error_t) :: error
        integer(c_int32_t) :: vreg

        emit_liric_f64_binary = .false.
        if (.not. require_open_session(session, error_msg)) return

        vreg = emit_binary_f64(session%handle, opcode, lhs, rhs, error)
        if (.not. status_ok(error%code, error, error_msg)) return

        result%kind = LR_OP_KIND_VREG
        result%payload = int(vreg, c_int64_t)
        result%typ = lr_type_f64_s(session%handle)
        result%global_offset = 0_c_int64_t
        call set_empty(error_msg)
        emit_liric_f64_binary = .true.
    end function emit_liric_f64_binary

    logical function emit_liric_i32_to_f64(session, source, result, error_msg)
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: source
        type(lr_operand_desc_t), intent(out) :: result
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_error_t) :: error
        integer(c_int32_t) :: vreg

        emit_liric_i32_to_f64 = .false.
        if (.not. require_open_session(session, error_msg)) return

        vreg = emit_cast_f64(session%handle, LR_OP_SITOFP, source, error)
        if (.not. status_ok(error%code, error, error_msg)) return

        result%kind = LR_OP_KIND_VREG
        result%payload = int(vreg, c_int64_t)
        result%typ = lr_type_f64_s(session%handle)
        result%global_offset = 0_c_int64_t
        call set_empty(error_msg)
        emit_liric_i32_to_f64 = .true.
    end function emit_liric_i32_to_f64

    function emit_binary_f64(handle, opcode, lhs, rhs, error) result(vreg)
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
        inst%typ = lr_type_f64_s(handle)
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
    end function emit_binary_f64

    function emit_cast_f64(handle, opcode, source, error) result(vreg)
        type(c_ptr), intent(in) :: handle
        integer(c_int), intent(in) :: opcode
        type(lr_operand_desc_t), intent(in) :: source
        type(lr_error_t), intent(inout) :: error
        integer(c_int32_t) :: vreg
        type(lr_operand_desc_t), target :: operands(1)
        type(lr_inst_desc_t) :: inst

        operands(1) = source

        inst%op = opcode
        inst%typ = lr_type_f64_s(handle)
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
    end function emit_cast_f64

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
