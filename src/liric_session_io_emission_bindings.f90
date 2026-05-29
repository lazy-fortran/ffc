module liric_session_io_emission_bindings
    use, intrinsic :: iso_c_binding, only: c_associated, c_bool, c_char
    use, intrinsic :: iso_c_binding, only: c_double, c_int, c_int32_t, &
                                              c_int64_t
    use, intrinsic :: iso_c_binding, only: c_loc, c_null_char, c_null_ptr, c_ptr, &
                                              c_size_t
    use liric_session_bindings, only: liric_session_t, lr_error_t, &
                                      lr_inst_desc_t, lr_operand_desc_t, LR_OK, &
                                      LR_OP_KIND_VREG, LR_OP_KIND_GLOBAL, &
                                      LR_OP_CALL, LR_OP_BITCAST, LR_OP_SITOFP, &
                                      LR_OP_FPTOSI, LR_OP_ZEXT, &
                                      LR_OP_GEP, LR_OP_LOAD, LR_OP_KIND_IMM_I64, &
                                      LR_OP_FADD, LR_OP_FMUL, LR_OP_FDIV, &
                                      liric_session_error_message, &
                                      lr_type_i32_s, lr_type_f64_s, lr_type_i64_s, &
                                      lr_type_ptr_s, lr_type_i8_s, &
                                      lr_type_array_s, lr_session_emit, &
                                      lr_session_intern, lr_session_global, &
                                      global_operand, &
                                      status_ok, clear_liric_error, &
                                      set_empty, require_open_session, &
                                      to_c_chars, i32_vreg
    use liric_session_memory_bindings, only: ptr_vreg, i64_vreg
    use liric_session_format_bindings, only: LR_OP_FSUB
    implicit none
    private

    integer(c_int), parameter :: LR_OP_KIND_IMM_F64 = 2_c_int
    logical(c_bool), parameter :: c_false = .false.
    logical(c_bool), parameter :: c_true = .true.

    public :: emit_liric_f64_binary
    public :: emit_liric_i32_to_f64
    public :: emit_liric_f64_to_i32
    public :: emit_liric_char_byte_zext
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

contains

    function global_operand_session(session, id, typ) result(operand)
        type(liric_session_t), intent(in) :: session
        integer(c_int32_t), intent(in) :: id
        type(c_ptr), intent(in) :: typ
        type(lr_operand_desc_t) :: operand

        operand%kind = LR_OP_KIND_GLOBAL
        operand%payload = int(id, c_int64_t)
        operand%typ = typ
        operand%global_offset = 0_c_int64_t
    end function global_operand_session

    logical function emit_liric_print_i32(session, format_global_id, value, &
                                          error_msg)
        type(liric_session_t), intent(inout) :: session
        integer(c_int32_t), intent(in) :: format_global_id
        type(lr_operand_desc_t), intent(in) :: value
        character(len=:), allocatable, intent(out) :: error_msg

        emit_liric_print_i32 = .false.
        if (.not. emit_liric_print_i32_value(session, format_global_id, value, &
                                             error_msg)) return
        if (.not. emit_liric_print_newline(session, error_msg)) return
        call set_empty(error_msg)
        emit_liric_print_i32 = .true.
    end function emit_liric_print_i32

    logical function emit_liric_print_f64(session, format_global_id, value, &
                                          error_msg)
        type(liric_session_t), intent(inout) :: session
        integer(c_int32_t), intent(in) :: format_global_id
        type(lr_operand_desc_t), intent(in) :: value
        character(len=:), allocatable, intent(out) :: error_msg

        emit_liric_print_f64 = .false.
        if (.not. emit_liric_print_f64_value(session, format_global_id, value, &
                                             error_msg)) return
        if (.not. emit_liric_print_newline(session, error_msg)) return
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

        emit_liric_print_string = .false.
        if (.not. emit_liric_print_string_value(session, format_global_id, &
                                                string_global_name, text, &
                                                error_msg)) return
        if (.not. emit_liric_print_newline(session, error_msg)) return
        call set_empty(error_msg)
        emit_liric_print_string = .true.
    end function emit_liric_print_string

  logical function emit_liric_print_string_operand(session, format_global_id, &
                                                       string_ptr, error_msg)
        type(liric_session_t), intent(inout) :: session
        integer(c_int32_t), intent(in) :: format_global_id
        type(lr_operand_desc_t), intent(in) :: string_ptr
        character(len=:), allocatable, intent(out) :: error_msg

        emit_liric_print_string_operand = .false.
        if (.not. emit_liric_print_string_operand_value(session, format_global_id, &
                                                        string_ptr, error_msg)) return
        if (.not. emit_liric_print_newline(session, error_msg)) return
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

        operand = global_operand_session(session, symbol_id, lr_type_ptr_s(session%handle))
        call set_empty(error_msg)
   end subroutine make_printf_operand

    subroutine make_string_literal_operand(session, name, text, operand, &
                                           error_msg)
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
                                                      total_size, operand, &
                                                      error)
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

    subroutine materialize_global_pointer(session, global_id, operand, &
                                          error_msg)
        type(liric_session_t), intent(inout) :: session
        integer(c_int32_t), intent(in) :: global_id
        type(lr_operand_desc_t), intent(out) :: operand
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: global_ref
        type(lr_error_t) :: error
        integer(c_int32_t) :: vreg

      global_ref = global_operand_session(session, global_id, &
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

    logical function emit_liric_f64_to_i32(session, source, result, error_msg)
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: source
        type(lr_operand_desc_t), intent(out) :: result
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_error_t) :: error
        integer(c_int32_t) :: vreg

        emit_liric_f64_to_i32 = .false.
        if (.not. require_open_session(session, error_msg)) return

        vreg = emit_cast_i32(session%handle, LR_OP_FPTOSI, source, error)
        if (.not. status_ok(error%code, error, error_msg)) return

        result = i32_vreg(session, vreg)
        call set_empty(error_msg)
        emit_liric_f64_to_i32 = .true.
    end function emit_liric_f64_to_i32

    function emit_cast_i32(handle, opcode, source, error) result(vreg)
        type(c_ptr), intent(in) :: handle
        integer(c_int), intent(in) :: opcode
        type(lr_operand_desc_t), intent(in) :: source
        type(lr_error_t), intent(inout) :: error
        integer(c_int32_t) :: vreg
        type(lr_operand_desc_t), target :: operands(1)
        type(lr_inst_desc_t) :: inst

        operands(1) = source

        inst%op = opcode
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
    end function emit_cast_i32

    logical function emit_liric_char_byte_zext(session, base, index_op, result, &
                                               error_msg)
        ! Load the byte at base[index] and zero-extend it to i32.
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: base
        type(lr_operand_desc_t), intent(in) :: index_op
        type(lr_operand_desc_t), intent(out) :: result
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_error_t) :: error
        type(lr_operand_desc_t), target :: operands(2)
        type(lr_inst_desc_t) :: inst
        integer(c_int32_t) :: vreg

        emit_liric_char_byte_zext = .false.
        if (.not. require_open_session(session, error_msg)) return

        ! Widen the i32 index to i64 for pointer arithmetic (offset is bytes).
        operands(1) = index_op
        inst%op = LR_OP_ZEXT
        inst%typ = lr_type_i64_s(session%handle)
        inst%operands = c_loc(operands)
        inst%num_operands = 1_c_int32_t
        call clear_liric_error(error)
        vreg = lr_session_emit(session%handle, inst, error)
        if (.not. status_ok(error%code, error, error_msg)) return

        ! getelementptr i8, ptr base, i64 index (element type drives scaling).
        operands(1) = base
        operands(2) = i64_vreg(session, vreg)
        inst%op = LR_OP_GEP
        inst%typ = lr_type_i8_s(session%handle)
        inst%operands = c_loc(operands)
        inst%num_operands = 2_c_int32_t
        call clear_liric_error(error)
        vreg = lr_session_emit(session%handle, inst, error)
        if (.not. status_ok(error%code, error, error_msg)) return

        operands(1) = ptr_vreg(session, vreg)
        inst%op = LR_OP_LOAD
        inst%typ = lr_type_i8_s(session%handle)
        inst%operands = c_loc(operands)
        inst%num_operands = 1_c_int32_t
        call clear_liric_error(error)
        vreg = lr_session_emit(session%handle, inst, error)
        if (.not. status_ok(error%code, error, error_msg)) return

        operands(1) = i8_vreg(session, vreg)
        inst%op = LR_OP_ZEXT
        inst%typ = lr_type_i32_s(session%handle)
        inst%operands = c_loc(operands)
        inst%num_operands = 1_c_int32_t
        call clear_liric_error(error)
        vreg = lr_session_emit(session%handle, inst, error)
        if (.not. status_ok(error%code, error, error_msg)) return

        result = i32_vreg(session, vreg)
        call set_empty(error_msg)
        emit_liric_char_byte_zext = .true.
    end function emit_liric_char_byte_zext

    function i8_vreg(session, vreg) result(operand)
        type(liric_session_t), intent(in) :: session
        integer(c_int32_t), intent(in) :: vreg
        type(lr_operand_desc_t) :: operand

        operand%kind = LR_OP_KIND_VREG
        operand%payload = int(vreg, c_int64_t)
        operand%typ = lr_type_i8_s(session%handle)
        operand%global_offset = 0_c_int64_t
    end function i8_vreg

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

end module liric_session_io_emission_bindings
