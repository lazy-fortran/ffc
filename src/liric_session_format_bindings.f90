module liric_session_format_bindings
    use, intrinsic :: iso_c_binding, only: c_associated, c_bool, c_char
    use, intrinsic :: iso_c_binding, only: c_double, c_int, c_int32_t, c_int64_t
    use, intrinsic :: iso_c_binding, only: c_loc, c_null_char, c_null_ptr, c_ptr, &
        c_size_t
    use liric_session_common, only: require_open_session, status_ok, &
        clear_liric_error, to_c_chars, set_empty, &
        liric_session_error_message, lr_error_t, &
        lr_operand_desc_t, lr_inst_desc_t, &
        liric_session_t, LR_OP_KIND_GLOBAL, &
        LR_OP_KIND_VREG, LR_OP_KIND_IMM_I64
    use liric_session_bindings, only: LR_OP_CALL, LR_OP_RET, LR_OP_FDIV, &
        LR_OP_SDIV, LR_OP_MUL, LR_OP_SUB, &
        LR_OP_ADD, LR_OP_SITOFP, LR_OP_FPTOSI, &
        lr_type_f64_s, lr_type_i64_s, &
        lr_session_param, &
        lr_session_emit, i32_immediate, i32_vreg
    use liric_session_memory_bindings, only: emit_alloca_bytes, &
        emit_i32_binary, i64_immediate, &
        ptr_vreg
    use liric_session_control_bindings, only: create_liric_block, &
        set_liric_block, emit_liric_br, &
        emit_liric_condbr, &
        emit_liric_i32_icmp, LR_CMP_EQ, &
        LR_CMP_NE, LR_CMP_SLT
    implicit none
    private

    integer(c_int), parameter, public :: LR_OP_FSUB = 19_c_int
    logical(c_bool), parameter :: c_true = .true.
    logical(c_bool), parameter :: c_false = .false.
    integer(c_int), parameter :: LR_OK = 0_c_int
    integer(c_int), parameter :: LR_OP_KIND_IMM_F64 = 2_c_int
    character(len=*), parameter :: E_EN_FORMAT_HELPER = '.ffc.fmt_e_en'

    public :: prepare_liric_print_runtime
    public :: create_printf_format_global
    public :: printf_format_ptr
    public :: create_type_info_global
    public :: create_pointer_table_global
    public :: create_i64_table_global
    public :: create_i8_format_global_no_newline
    public :: create_i16_format_global_no_newline
    public :: emit_e_en_format_call

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

        subroutine lr_session_global_reloc(handle, global_id, offset, sym) bind(c)
            import :: c_char, c_int32_t, c_ptr, c_size_t
            type(c_ptr), value :: handle
            integer(c_int32_t), value :: global_id
            integer(c_size_t), value :: offset
            character(kind=c_char), intent(in) :: sym(*)
        end subroutine lr_session_global_reloc

        function lr_session_intern(handle, name) result(symbol_id) bind(c)
            import :: c_char, c_int32_t, c_ptr
            type(c_ptr), value :: handle
            character(kind=c_char), intent(in) :: name(*)
            integer(c_int32_t) :: symbol_id
        end function lr_session_intern

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

        function lr_session_func_end(handle, out_addr, err) result(status) bind(c)
            import :: c_int, c_ptr, lr_error_t
            type(c_ptr), value :: handle
            type(c_ptr), intent(out) :: out_addr
            type(lr_error_t), intent(inout) :: err
            integer(c_int) :: status
        end function lr_session_func_end
    end interface

    interface
        module function synthesize_e_en_format_helper(session, error_msg)
            type(liric_session_t), intent(inout) :: session
            character(len=:), allocatable, intent(out) :: error_msg
            logical :: synthesize_e_en_format_helper
        end function synthesize_e_en_format_helper

        module subroutine begin_e_en_helper(session, params, c_name, status, error)
            type(liric_session_t), intent(inout) :: session
            type(c_ptr), target, intent(out) :: params(6)
            character(kind=c_char), allocatable, intent(out) :: c_name(:)
            integer(c_int), intent(out) :: status
            type(lr_error_t), intent(out) :: error
        end subroutine begin_e_en_helper

        module subroutine e_en_params(session, x, mode, digits, width, buf, &
                                      exp_digits)
            type(liric_session_t), intent(in) :: session
            type(lr_operand_desc_t), intent(out) :: x, mode, digits, width, buf
            type(lr_operand_desc_t), intent(out) :: exp_digits
        end subroutine e_en_params

        module subroutine create_e_en_blocks(session, entry, finite, nonfinite, &
                                             eblk, enblk, ezero, enonzero)
            type(liric_session_t), intent(inout) :: session
            integer(c_int32_t), intent(out) :: entry, finite, nonfinite, eblk, enblk
            integer(c_int32_t), intent(out) :: ezero, enonzero
        end subroutine create_e_en_blocks

        module function emit_e_en_head(session, x, g_norm, tmp, p, finite, &
                                               nonfinite, error_msg)
            type(liric_session_t), intent(inout) :: session
            type(lr_operand_desc_t), intent(in) :: x, g_norm
            type(lr_operand_desc_t), intent(out) :: tmp, p
            integer(c_int32_t), intent(in) :: finite, nonfinite
            character(len=:), allocatable, intent(out) :: error_msg
            logical :: emit_e_en_head
        end function emit_e_en_head

        module function parse_e_en_exponent(session, p, ex, error_msg)
            type(liric_session_t), intent(inout) :: session
            type(lr_operand_desc_t), intent(in) :: p
            type(lr_operand_desc_t), intent(out) :: ex
            character(len=:), allocatable, intent(out) :: error_msg
            logical :: parse_e_en_exponent
        end function parse_e_en_exponent

        module function emit_e_field(session, x, ex, digits, width, buf, &
                                             g_field, exp_digits, error_msg)
            type(liric_session_t), intent(inout) :: session
            type(lr_operand_desc_t), intent(in) :: x, ex, digits, width, buf, g_field
            type(lr_operand_desc_t), intent(in) :: exp_digits
            character(len=:), allocatable, intent(out) :: error_msg
            logical :: emit_e_field
        end function emit_e_field

        module function emit_en_field(session, x, ex, digits, width, buf, &
                                              g_field, exp_digits, error_msg)
            type(liric_session_t), intent(inout) :: session
            type(lr_operand_desc_t), intent(in) :: x, ex, digits, width, buf, g_field
            type(lr_operand_desc_t), intent(in) :: exp_digits
            character(len=:), allocatable, intent(out) :: error_msg
            logical :: emit_en_field
        end function emit_en_field

        module function emit_scaled_field(session, x, e_exp, digits, width, &
                                                  buf, g_field, exp_digits, error_msg)
            type(liric_session_t), intent(inout) :: session
            type(lr_operand_desc_t), intent(in) :: x, e_exp, digits, width, buf
            type(lr_operand_desc_t), intent(in) :: g_field, exp_digits
            character(len=:), allocatable, intent(out) :: error_msg
            logical :: emit_scaled_field
        end function emit_scaled_field

        module function scaled_mantissa(session, x, e_exp, mant, error_msg)
            type(liric_session_t), intent(inout) :: session
            type(lr_operand_desc_t), intent(in) :: x, e_exp
            type(lr_operand_desc_t), intent(out) :: mant
            character(len=:), allocatable, intent(out) :: error_msg
            logical :: scaled_mantissa
        end function scaled_mantissa

        module function snprintf_scaled(session, buf, g_field, field_width, &
                                                digits, mant, sign_ch, abs_exp, &
                                                exp_digits, error_msg)
            type(liric_session_t), intent(inout) :: session
            type(lr_operand_desc_t), intent(in) :: buf, g_field, field_width, digits
            type(lr_operand_desc_t), intent(in) :: mant, abs_exp, exp_digits
            character, intent(in) :: sign_ch
            character(len=:), allocatable, intent(out) :: error_msg
            logical :: snprintf_scaled
        end function snprintf_scaled

        module function copy_nonfinite_field(session, tmp, width, buf, &
                                                     error_msg)
            type(liric_session_t), intent(inout) :: session
            type(lr_operand_desc_t), intent(in) :: tmp, width, buf
            character(len=:), allocatable, intent(out) :: error_msg
            logical :: copy_nonfinite_field
        end function copy_nonfinite_field

        module function return_zero(session, error_msg)
            type(liric_session_t), intent(inout) :: session
            character(len=:), allocatable, intent(out) :: error_msg
            logical :: return_zero
        end function return_zero

        module function emit_e_en_format_call(session, value, mode, digits, &
                                                      width, buf, error_msg, &
                                                      exp_digits)
            type(liric_session_t), intent(inout) :: session
            type(lr_operand_desc_t), intent(in) :: value
            integer, intent(in) :: mode, digits, width
            type(lr_operand_desc_t), intent(in) :: buf
            character(len=:), allocatable, intent(out) :: error_msg
            integer, intent(in), optional :: exp_digits
            logical :: emit_e_en_format_call
        end function emit_e_en_format_call

        module function declare_e_en_libc(session, error_msg)
            type(liric_session_t), intent(inout) :: session
            character(len=:), allocatable, intent(out) :: error_msg
            logical :: declare_e_en_libc
        end function declare_e_en_libc

        module function declare_c_func(session, name, ret, params_ptr, n, &
                                               vararg, error_msg)
            type(liric_session_t), intent(inout) :: session
            character(len=*), intent(in) :: name
            type(c_ptr), intent(in) :: ret, params_ptr
            integer(c_int32_t), intent(in) :: n
            logical(c_bool), intent(in) :: vararg
            character(len=:), allocatable, intent(out) :: error_msg
            logical :: declare_c_func
        end function declare_c_func

        module function create_cstring_operand(session, name, text, operand, &
                                                       error_msg)
            type(liric_session_t), intent(inout) :: session
            character(len=*), intent(in) :: name, text
            type(lr_operand_desc_t), intent(out) :: operand
            character(len=:), allocatable, intent(out) :: error_msg
            logical :: create_cstring_operand
        end function create_cstring_operand

        module function typed_param(session, index, typ) result(operand)
            type(liric_session_t), intent(in) :: session
            integer(c_int32_t), intent(in) :: index
            type(c_ptr), intent(in) :: typ
            type(lr_operand_desc_t) :: operand
        end function typed_param

        module subroutine null_ptr_operand(session, operand)
            type(liric_session_t), intent(in) :: session
            type(lr_operand_desc_t), intent(out) :: operand
        end subroutine null_ptr_operand

        module function find_char(session, text, ch, result, error_msg)
            type(liric_session_t), intent(inout) :: session
            type(lr_operand_desc_t), intent(in) :: text
            character, intent(in) :: ch
            type(lr_operand_desc_t), intent(out) :: result
            character(len=:), allocatable, intent(out) :: error_msg
            logical :: find_char
        end function find_char

        module function emit_c_call(session, name, args, ret_typ, fixed_args, &
                                            vararg, error_msg)
            type(liric_session_t), intent(inout) :: session
            character(len=*), intent(in) :: name
            type(lr_operand_desc_t), intent(in) :: args(:)
            type(c_ptr), intent(in) :: ret_typ
            integer(c_int32_t), intent(in) :: fixed_args
            logical(c_bool), intent(in) :: vararg
            character(len=:), allocatable, intent(out) :: error_msg
            logical :: emit_c_call
        end function emit_c_call

        module function emit_c_call_vreg(session, name, args, ret_typ, &
                                                 fixed_args, vararg, vreg, error_msg)
            type(liric_session_t), intent(inout) :: session
            character(len=*), intent(in) :: name
            type(lr_operand_desc_t), intent(in) :: args(:)
            type(c_ptr), intent(in) :: ret_typ
            integer(c_int32_t), intent(in) :: fixed_args
            logical(c_bool), intent(in) :: vararg
            integer(c_int32_t), intent(out) :: vreg
            character(len=:), allocatable, intent(out) :: error_msg
            logical :: emit_c_call_vreg
        end function emit_c_call_vreg

        module function gep_byte(session, base, offset, result, error_msg)
            type(liric_session_t), intent(inout) :: session
            type(lr_operand_desc_t), intent(in) :: base
            integer(c_int64_t), intent(in) :: offset
            type(lr_operand_desc_t), intent(out) :: result
            character(len=:), allocatable, intent(out) :: error_msg
            logical :: gep_byte
        end function gep_byte

        module function cast_i32_to_f64(session, source, result, error_msg)
            type(liric_session_t), intent(inout) :: session
            type(lr_operand_desc_t), intent(in) :: source
            type(lr_operand_desc_t), intent(out) :: result
            character(len=:), allocatable, intent(out) :: error_msg
            logical :: cast_i32_to_f64
        end function cast_i32_to_f64

        module function cast_f64_to_i32(session, source, result, error_msg)
            type(liric_session_t), intent(inout) :: session
            type(lr_operand_desc_t), intent(in) :: source
            type(lr_operand_desc_t), intent(out) :: result
            character(len=:), allocatable, intent(out) :: error_msg
            logical :: cast_f64_to_i32
        end function cast_f64_to_i32

        module function emit_cast(session, op, src, dst_typ, result, &
                                          error_msg)
            type(liric_session_t), intent(inout) :: session
            integer(c_int), intent(in) :: op
            type(lr_operand_desc_t), intent(in) :: src
            type(c_ptr), intent(in) :: dst_typ
            type(lr_operand_desc_t), intent(out) :: result
            character(len=:), allocatable, intent(out) :: error_msg
            logical :: emit_cast
        end function emit_cast

        module function emit_f64_binary(session, opcode, lhs, rhs, result, &
                                                error_msg)
            type(liric_session_t), intent(inout) :: session
            integer(c_int), intent(in) :: opcode
            type(lr_operand_desc_t), intent(in) :: lhs, rhs
            type(lr_operand_desc_t), intent(out) :: result
            character(len=:), allocatable, intent(out) :: error_msg
            logical :: emit_f64_binary
        end function emit_f64_binary

        module function call_f64_unary(session, name, arg, result, error_msg)
            type(liric_session_t), intent(inout) :: session
            character(len=*), intent(in) :: name
            type(lr_operand_desc_t), intent(in) :: arg
            type(lr_operand_desc_t), intent(out) :: result
            character(len=:), allocatable, intent(out) :: error_msg
            logical :: call_f64_unary
        end function call_f64_unary

        module function f64_immediate(session, value) result(operand)
            type(liric_session_t), intent(in) :: session
            real(c_double), intent(in) :: value
            type(lr_operand_desc_t) :: operand
        end function f64_immediate

        module function emit_ret_i32_local(session, value, error_msg)
            type(liric_session_t), intent(inout) :: session
            type(lr_operand_desc_t), intent(in) :: value
            character(len=:), allocatable, intent(out) :: error_msg
            logical :: emit_ret_i32_local
        end function emit_ret_i32_local
    end interface

contains

    logical function prepare_liric_print_runtime(session, i32_format_id, &
            str_format_id, error_msg, &
            i64_format_id, i8_format_id, &
            i16_format_id)
        type(liric_session_t), intent(inout) :: session
        integer(c_int32_t), intent(out) :: i32_format_id
        integer(c_int32_t), intent(out) :: str_format_id
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int32_t), intent(out), optional :: i64_format_id
        integer(c_int32_t), intent(out), optional :: i8_format_id
        integer(c_int32_t), intent(out), optional :: i16_format_id
        integer(c_int32_t) :: local_i64_format_id
        integer(c_int32_t) :: local_i8_format_id
        integer(c_int32_t) :: local_i16_format_id

        prepare_liric_print_runtime = .false.
        if (.not. declare_printf_i32(session, error_msg)) return
        call create_i32_format_global_no_newline(session, &
            '.ffc.fmt.i32', &
            i32_format_id, error_msg)
        if (len_trim(error_msg) > 0) return
        call create_i64_format_global_no_newline(session, &
            '.ffc.fmt.i64', &
            local_i64_format_id, error_msg)
        if (len_trim(error_msg) > 0) return
        if (present(i64_format_id)) i64_format_id = local_i64_format_id
        call create_i8_format_global_no_newline(session, &
            '.ffc.fmt.i8', &
            local_i8_format_id, error_msg)
        if (len_trim(error_msg) > 0) return
        if (present(i8_format_id)) i8_format_id = local_i8_format_id
        call create_i16_format_global_no_newline(session, &
            '.ffc.fmt.i16', &
            local_i16_format_id, error_msg)
        if (len_trim(error_msg) > 0) return
        if (present(i16_format_id)) i16_format_id = local_i16_format_id
        call create_str_format_global_no_newline(session, &
            '.ffc.fmt.str', &
            str_format_id, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. synthesize_e_en_format_helper(session, error_msg)) return

        call set_empty(error_msg)
        prepare_liric_print_runtime = .true.
    end function prepare_liric_print_runtime


    logical function declare_printf_i32(session, error_msg)
        type(liric_session_t), intent(inout) :: session
        character(len=:), allocatable, intent(out) :: error_msg
        character(kind=c_char), allocatable :: c_name(:)
        type(c_ptr), target :: params(3)
        type(lr_error_t) :: error
        integer(c_int) :: status

        declare_printf_i32 = .false.
        if (.not. require_open_session(session, error_msg)) return

        ! The scalar output entry point, declared once per session. It
        ! replaced the variadic printf this used to declare (#423): a
        ! fixed-arity signature is the same on every target.
        params(1) = lr_type_i32_s(session%handle)
        params(2) = lr_type_ptr_s(session%handle)
        params(3) = lr_type_i32_s(session%handle)
        call clear_liric_error(error)
        call to_c_chars('_ffc_write_i32', c_name)
        status = lr_session_declare(session%handle, c_name, &
            lr_type_i32_s(session%handle), &
            c_loc(params), 3_c_int32_t, c_false, &
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

        call create_printf_format_global(session, name, '%11d', global_id, &
            error_msg)
    end subroutine create_i32_format_global_no_newline

    subroutine create_str_format_global_no_newline(session, name, global_id, &
            error_msg)
        type(liric_session_t), intent(inout) :: session
        character(len=*), intent(in) :: name
        integer(c_int32_t), intent(out) :: global_id
        character(len=:), allocatable, intent(out) :: error_msg

        call create_printf_format_global(session, name, '%s', global_id, &
            error_msg)
    end subroutine create_str_format_global_no_newline

    subroutine create_i8_format_global_no_newline(session, name, global_id, &
            error_msg)
        type(liric_session_t), intent(inout) :: session
        character(len=*), intent(in) :: name
        integer(c_int32_t), intent(out) :: global_id
        character(len=:), allocatable, intent(out) :: error_msg

        call create_printf_format_global(session, name, '%4d', global_id, &
            error_msg)
    end subroutine create_i8_format_global_no_newline

    subroutine create_i16_format_global_no_newline(session, name, global_id, &
            error_msg)
        type(liric_session_t), intent(inout) :: session
        character(len=*), intent(in) :: name
        integer(c_int32_t), intent(out) :: global_id
        character(len=:), allocatable, intent(out) :: error_msg

        call create_printf_format_global(session, name, '%6d', global_id, &
            error_msg)
    end subroutine create_i16_format_global_no_newline

    subroutine create_i64_format_global_no_newline(session, name, global_id, &
            error_msg)
        type(liric_session_t), intent(inout) :: session
        character(len=*), intent(in) :: name
        integer(c_int32_t), intent(out) :: global_id
        character(len=:), allocatable, intent(out) :: error_msg

        call create_printf_format_global(session, name, '%20ld', global_id, &
            error_msg)
    end subroutine create_i64_format_global_no_newline

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

    subroutine create_pointer_table_global(session, name, symbols, error_msg)
        ! Emit a const global holding size(symbols) 8-byte pointer slots. Each
        ! slot whose symbol name is non-blank carries a data relocation to that
        ! symbol; a blank name leaves the slot a null pointer. This backs the
        ! per-type vtables and the link-unit vtable table of
        ! docs/RUNTIME_ABI.md, both of which are arrays of code or data
        ! addresses that only the linker (or the JIT's global materialiser) can
        ! fill in.
        type(liric_session_t), intent(inout) :: session
        character(len=*), intent(in) :: name
        character(len=*), intent(in) :: symbols(:)
        character(len=:), allocatable, intent(out) :: error_msg
        character(kind=c_char), allocatable :: c_name(:), c_sym(:)
        character(kind=c_char), allocatable, target :: bytes(:)
        type(c_ptr) :: array_type
        integer :: slot
        integer(c_size_t) :: nbytes
        integer(c_int32_t) :: global_id

        call set_empty(error_msg)
        if (size(symbols) <= 0) return
        nbytes = int(size(symbols), c_size_t) * 8_c_size_t
        allocate (bytes(nbytes))
        bytes = char(0, kind=c_char)

        call to_c_chars(name, c_name)
        array_type = lr_type_array_s(session%handle, lr_type_i8_s(session%handle), &
            int(nbytes, c_int64_t))
        if (.not. c_associated(array_type)) then
            error_msg = 'LIRIC did not return a pointer-table array type'
            return
        end if
        global_id = lr_session_global(session%handle, c_name, array_type, &
            c_true, c_loc(bytes), nbytes)
        if (global_id < 0_c_int32_t) then
            error_msg = 'LIRIC could not create pointer table global '//trim(name)
            return
        end if
        do slot = 1, size(symbols)
            if (len_trim(symbols(slot)) == 0) cycle
            call to_c_chars(trim(symbols(slot)), c_sym)
            call lr_session_global_reloc(session%handle, global_id, &
                int(slot - 1, c_size_t) * 8_c_size_t, c_sym)
        end do
    end subroutine create_pointer_table_global

    subroutine create_i64_table_global(session, name, values, error_msg)
        ! Emit a const global holding size(values) little-endian i64 entries.
        ! This backs the link-unit type size table of docs/RUNTIME_ABI.md, which
        ! a dense type id indexes to get the exact storage size of that type.
        type(liric_session_t), intent(inout) :: session
        character(len=*), intent(in) :: name
        integer(c_int64_t), intent(in) :: values(:)
        character(len=:), allocatable, intent(out) :: error_msg
        character(kind=c_char), allocatable :: c_name(:)
        character(kind=c_char), allocatable, target :: bytes(:)
        type(c_ptr) :: array_type
        integer :: slot, k
        integer(c_size_t) :: nbytes
        integer(c_int32_t) :: global_id

        call set_empty(error_msg)
        if (size(values) <= 0) return
        nbytes = int(size(values), c_size_t) * 8_c_size_t
        allocate (bytes(nbytes))
        do slot = 1, size(values)
            do k = 0, 7
                bytes((slot - 1) * 8 + 1 + k) = char(int(iand( &
                    ishft(values(slot), -8 * k), 255_c_int64_t)), kind=c_char)
            end do
        end do

        call to_c_chars(name, c_name)
        array_type = lr_type_array_s(session%handle, lr_type_i8_s(session%handle), &
            int(nbytes, c_int64_t))
        if (.not. c_associated(array_type)) then
            error_msg = 'LIRIC did not return an i64-table array type'
            return
        end if
        global_id = lr_session_global(session%handle, c_name, array_type, &
            c_true, c_loc(bytes), nbytes)
        if (global_id < 0_c_int32_t) then
            error_msg = 'LIRIC could not create i64 table global '//trim(name)
        end if
    end subroutine create_i64_table_global

end module liric_session_format_bindings
