module liric_session_format_bindings
    use, intrinsic :: iso_c_binding, only: c_associated, c_bool, c_char
    use, intrinsic :: iso_c_binding, only: c_int, c_int32_t, c_int64_t
    use, intrinsic :: iso_c_binding, only: c_loc, c_null_char, c_null_ptr, c_ptr, &
                                              c_size_t
    use liric_session_bindings, only: liric_session_t, lr_error_t, &
                                      lr_operand_desc_t, lr_inst_desc_t, &
                                      LR_OP_KIND_GLOBAL
    implicit none
    private

    integer(c_int), parameter, public :: LR_OP_FSUB = 19_c_int
    logical(c_bool), parameter :: c_true = .true.
    logical(c_bool), parameter :: c_false = .false.
    integer(c_int), parameter :: LR_OK = 0_c_int

    public :: prepare_liric_print_runtime
    public :: create_printf_format_global
    public :: printf_format_ptr
    public :: create_type_info_global

    interface
        function lr_type_i32_s(handle) result(typ) bind(c)
            import :: c_ptr
            type(c_ptr), value :: handle
            type(c_ptr) :: typ
        end function lr_type_i32_s

        function lr_type_i8_s(handle) result(typ) bind(c)
            import :: c_ptr
            type(c_ptr), value :: handle
            type(c_ptr) :: typ
        end function lr_type_i8_s

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
    end interface

contains

    logical function prepare_liric_print_runtime(session, i32_format_id, &
                                                 str_format_id, error_msg)
        type(liric_session_t), intent(inout) :: session
        integer(c_int32_t), intent(out) :: i32_format_id
        integer(c_int32_t), intent(out) :: str_format_id
        character(len=:), allocatable, intent(out) :: error_msg

        prepare_liric_print_runtime = .false.
        if (.not. declare_printf_i32(session, error_msg)) return
        call create_i32_format_global_no_newline(session, &
                                                 '.ffc.fmt.i32', &
                                                 i32_format_id, error_msg)
        if (len_trim(error_msg) > 0) return
        call create_str_format_global_no_newline(session, &
                                                 '.ffc.fmt.str', &
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

    subroutine create_i32_format_global_no_newline(session, name, global_id, &
                                                   error_msg)
        type(liric_session_t), intent(inout) :: session
        character(len=*), intent(in) :: name
        integer(c_int32_t), intent(out) :: global_id
        character(len=:), allocatable, intent(out) :: error_msg
        character(kind=c_char), allocatable :: c_name(:)
        character(kind=c_char), target :: format_text(5)
        type(c_ptr) :: array_type

        call to_c_chars(name, c_name)
        ! gfortran list-directed default integer field is I12 including its
        ! one leading blank. That blank is emitted as the list separator, so
        ! the field itself is I11.
        format_text(1) = '%'
        format_text(2) = '1'
        format_text(3) = '1'
        format_text(4) = 'd'
        format_text(5) = c_null_char

        array_type = lr_type_array_s(session%handle, lr_type_i8_s(session%handle), &
                                      5_c_int64_t)
        if (.not. c_associated(array_type)) then
            error_msg = 'LIRIC did not return a format string array type'
            return
        end if

        global_id = lr_session_global(session%handle, c_name, array_type, &
                                      c_true, c_loc(format_text(1)), &
                                      5_c_size_t)
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

    function printf_format_ptr(session, global_id) result(operand)
        ! Build a pointer operand referencing an interned format-string global.
        type(liric_session_t), intent(in) :: session
        integer(c_int32_t), intent(in) :: global_id
        type(lr_operand_desc_t) :: operand

        operand%kind = LR_OP_KIND_GLOBAL
        operand%payload = int(global_id, c_int64_t)
        operand%typ = lr_type_ptr_s(session%handle)
        operand%global_offset = 0_c_int64_t
    end function printf_format_ptr

    subroutine create_printf_format_global(session, name, text, global_id, &
                                           error_msg)
        ! Build an interned const [len(text)+1 x i8] global holding text plus a
        ! null terminator, for use as a printf format string.
        type(liric_session_t), intent(inout) :: session
        character(len=*), intent(in) :: name
        character(len=*), intent(in) :: text
        integer(c_int32_t), intent(out) :: global_id
        character(len=:), allocatable, intent(out) :: error_msg
        character(kind=c_char), allocatable :: c_name(:)
        character(kind=c_char), allocatable, target :: bytes(:)
        type(c_ptr) :: array_type
        integer :: i, n

        n = len(text) + 1
        allocate (bytes(n))
        do i = 1, len(text)
            bytes(i) = text(i:i)
        end do
        bytes(n) = c_null_char

        call to_c_chars(name, c_name)
        array_type = lr_type_array_s(session%handle, lr_type_i8_s(session%handle), &
                                     int(n, c_int64_t))
        if (.not. c_associated(array_type)) then
            error_msg = 'LIRIC did not return a format string array type'
            return
        end if

        global_id = lr_session_global(session%handle, c_name, array_type, &
                                      c_true, c_loc(bytes), int(n, c_size_t))
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
    end subroutine create_printf_format_global

    subroutine create_type_info_global(session, name, type_id, size_bytes, &
                                        error_msg)
        ! Emit a compile-time ffc_type_info_t constant {i64 id; i64 size_bytes}
        ! as a 16-byte const global under the given (already-mangled) symbol.
        ! Not interned, so it keeps its symbol name for cross-unit comparison.
        use, intrinsic :: iso_c_binding, only: c_int64_t
        type(liric_session_t), intent(inout) :: session
        character(len=*), intent(in) :: name
        integer(c_int64_t), intent(in) :: type_id
        integer(c_int64_t), intent(in) :: size_bytes
        character(len=:), allocatable, intent(out) :: error_msg
        character(kind=c_char), allocatable :: c_name(:)
        character(kind=c_char), allocatable, target :: bytes(:)
        type(c_ptr) :: array_type
        integer :: k
        integer(c_int32_t) :: global_id

        allocate (bytes(16))
        do k = 0, 7
            bytes(1 + k) = char(int(iand(ishft(type_id, -8 * k), &
                255_c_int64_t)), kind=c_char)
            bytes(9 + k) = char(int(iand(ishft(size_bytes, -8 * k), &
                255_c_int64_t)), kind=c_char)
        end do

        call to_c_chars(name, c_name)
        array_type = lr_type_array_s(session%handle, lr_type_i8_s(session%handle), &
                                     16_c_int64_t)
        if (.not. c_associated(array_type)) then
            error_msg = 'LIRIC did not return a type-info array type'
            return
        end if
        global_id = lr_session_global(session%handle, c_name, array_type, &
                                      c_true, c_loc(bytes), 16_c_size_t)
        if (global_id < 0_c_int32_t) then
            error_msg = 'LIRIC could not create type-info global '//trim(name)
            return
        end if
        call set_empty(error_msg)
    end subroutine create_type_info_global

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

end module liric_session_format_bindings
