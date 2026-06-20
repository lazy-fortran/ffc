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

    logical function synthesize_e_en_format_helper(session, error_msg)
        type(liric_session_t), intent(inout) :: session
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: g_norm, g_field
        type(lr_operand_desc_t) :: x, mode, digits, width, buf, tmp, p, pe1, ex
        type(lr_operand_desc_t) :: cond, nullptr, zptr
        integer(c_int32_t) :: entry, finite, nonfinite, eblk, enblk
        integer(c_int32_t) :: ezero, enonzero
        type(c_ptr), target :: params(5)
        type(c_ptr) :: out_addr
        character(kind=c_char), allocatable :: c_name(:)
        type(lr_error_t) :: error
        integer(c_int) :: status
        synthesize_e_en_format_helper = .false.
        if (.not. require_open_session(session, error_msg)) return
        if (.not. declare_e_en_libc(session, error_msg)) return
        if (.not. create_cstring_operand(session, '.ffc.een.norm', '%.17E', &
                                         g_norm, error_msg)) return
        if (.not. create_cstring_operand(session, '.ffc.een.field', &
                                         '%*.*fE%c%02d', g_field, error_msg)) &
            return
        call begin_e_en_helper(session, params, c_name, status, error)
        if (.not. status_ok(status, error, error_msg)) return
        call e_en_params(session, x, mode, digits, width, buf)
        call create_e_en_blocks(session, entry, finite, nonfinite, eblk, enblk, &
                                ezero, enonzero)
        if (.not. set_liric_block(session, entry, error_msg)) return
        if (.not. emit_e_en_head(session, x, g_norm, tmp, p, finite, &
                                 nonfinite, error_msg)) return
        if (.not. set_liric_block(session, finite, error_msg)) return
        if (.not. parse_e_en_exponent(session, p, ex, error_msg)) return
        if (.not. emit_liric_i32_icmp(session, LR_CMP_EQ, mode, &
                                      i32_immediate(session, 0_c_int64_t), &
                                      cond, error_msg)) return
        if (.not. emit_liric_condbr(session, cond, eblk, enblk, error_msg)) &
            return
        if (.not. set_liric_block(session, eblk, error_msg)) return
        if (.not. find_char(session, tmp, '0', zptr, error_msg)) return
        call null_ptr_operand(session, nullptr)
        if (.not. emit_liric_i32_icmp(session, LR_CMP_EQ, zptr, tmp, cond, &
                                      error_msg)) return
        if (.not. emit_liric_condbr(session, cond, ezero, enonzero, error_msg)) &
            return
        if (.not. set_liric_block(session, ezero, error_msg)) return
        if (.not. emit_scaled_field(session, x, ex, digits, width, buf, g_field, &
                                    error_msg)) return
        if (.not. set_liric_block(session, enonzero, error_msg)) return
        if (.not. emit_e_field(session, x, ex, digits, width, buf, g_field, &
                               error_msg)) return
        if (.not. set_liric_block(session, enblk, error_msg)) return
        if (.not. emit_en_field(session, x, ex, digits, width, buf, g_field, &
                                error_msg)) return
        if (.not. set_liric_block(session, nonfinite, error_msg)) return
        if (.not. copy_nonfinite_field(session, tmp, width, buf, error_msg)) return
        call clear_liric_error(error)
        out_addr = c_null_ptr
        status = lr_session_func_end(session%handle, out_addr, error)
        if (.not. status_ok(status, error, error_msg)) return
        call set_empty(error_msg)
        synthesize_e_en_format_helper = .true.
    end function synthesize_e_en_format_helper
    subroutine begin_e_en_helper(session, params, c_name, status, error)
        type(liric_session_t), intent(inout) :: session
        type(c_ptr), target, intent(out) :: params(5)
        character(kind=c_char), allocatable, intent(out) :: c_name(:)
        integer(c_int), intent(out) :: status
        type(lr_error_t), intent(out) :: error
        params(1) = lr_type_f64_s(session%handle)
        params(2) = lr_type_i32_s(session%handle)
        params(3) = lr_type_i32_s(session%handle)
        params(4) = lr_type_i32_s(session%handle)
        params(5) = lr_type_ptr_s(session%handle)
        call clear_liric_error(error)
        call to_c_chars(E_EN_FORMAT_HELPER, c_name)
        status = lr_session_func_begin(session%handle, c_name, &
                                       lr_type_i32_s(session%handle), &
                                       c_loc(params), 5_c_int32_t, c_false, &
                                       error)
    end subroutine begin_e_en_helper
    subroutine e_en_params(session, x, mode, digits, width, buf)
        type(liric_session_t), intent(in) :: session
        type(lr_operand_desc_t), intent(out) :: x, mode, digits, width, buf
        x = typed_param(session, 0_c_int32_t, lr_type_f64_s(session%handle))
        mode = typed_param(session, 1_c_int32_t, lr_type_i32_s(session%handle))
        digits = typed_param(session, 2_c_int32_t, lr_type_i32_s(session%handle))
        width = typed_param(session, 3_c_int32_t, lr_type_i32_s(session%handle))
        buf = typed_param(session, 4_c_int32_t, lr_type_ptr_s(session%handle))
    end subroutine e_en_params
    subroutine create_e_en_blocks(session, entry, finite, nonfinite, eblk, enblk, &
                                  ezero, enonzero)
        type(liric_session_t), intent(inout) :: session
        integer(c_int32_t), intent(out) :: entry, finite, nonfinite, eblk, enblk
        integer(c_int32_t), intent(out) :: ezero, enonzero
        entry = create_liric_block(session)
        finite = create_liric_block(session)
        nonfinite = create_liric_block(session)
        eblk = create_liric_block(session)
        enblk = create_liric_block(session)
        ezero = create_liric_block(session)
        enonzero = create_liric_block(session)
    end subroutine create_e_en_blocks
    logical function emit_e_en_head(session, x, g_norm, tmp, p, finite, &
                                    nonfinite, error_msg)
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: x, g_norm
        type(lr_operand_desc_t), intent(out) :: tmp, p
        integer(c_int32_t), intent(in) :: finite, nonfinite
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: args(4), cond, nullptr
        emit_e_en_head = .false.
        if (.not. emit_alloca_bytes(session, i64_immediate(session, 96_c_int64_t), &
                                    tmp, error_msg)) return
        args(1) = tmp
        args(2) = i64_immediate(session, 96_c_int64_t)
        args(3) = g_norm
        args(4) = x
        if (.not. emit_c_call(session, 'snprintf', args, &
                              lr_type_i32_s(session%handle), 3_c_int32_t, &
                              c_true, error_msg)) return
        if (.not. find_char(session, tmp, 'E', p, error_msg)) return
        call null_ptr_operand(session, nullptr)
        if (.not. emit_liric_i32_icmp(session, LR_CMP_NE, p, nullptr, cond, &
                                      error_msg)) return
        if (.not. emit_liric_condbr(session, cond, finite, nonfinite, &
                                    error_msg)) return
        emit_e_en_head = .true.
    end function emit_e_en_head
    logical function parse_e_en_exponent(session, p, ex, error_msg)
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: p
        type(lr_operand_desc_t), intent(out) :: ex
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: pe1, args(1)
        integer(c_int32_t) :: vreg
        parse_e_en_exponent = .false.
        if (.not. gep_byte(session, p, 1_c_int64_t, pe1, error_msg)) return
        args(1) = pe1
        if (.not. emit_c_call_vreg(session, 'atoi', args, &
                                   lr_type_i32_s(session%handle), 1_c_int32_t, &
                                   c_false, vreg, error_msg)) return
        ex = i32_vreg(session, vreg)
        parse_e_en_exponent = .true.
    end function parse_e_en_exponent
    logical function emit_e_field(session, x, ex, digits, width, buf, g_field, &
                                  error_msg)
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: x, ex, digits, width, buf, g_field
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: e_exp
        emit_e_field = .false.
        if (.not. emit_i32_binary(session, LR_OP_ADD, ex, &
                                  i32_immediate(session, 1_c_int64_t), e_exp, &
                                  error_msg)) return
        emit_e_field = emit_scaled_field(session, x, e_exp, digits, width, buf, &
                                         g_field, error_msg)
    end function emit_e_field
    logical function emit_en_field(session, x, ex, digits, width, buf, g_field, &
                                   error_msg)
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: x, ex, digits, width, buf, g_field
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: ex_f64, third, div, floored, quotient, e_exp
        emit_en_field = .false.
        if (.not. cast_i32_to_f64(session, ex, ex_f64, error_msg)) return
        third = f64_immediate(session, 3.0_c_double)
        if (.not. emit_f64_binary(session, LR_OP_FDIV, ex_f64, third, div, &
                                  error_msg)) return
        if (.not. call_f64_unary(session, 'floor', div, floored, error_msg)) return
        if (.not. cast_f64_to_i32(session, floored, quotient, error_msg)) return
        if (.not. emit_i32_binary(session, LR_OP_MUL, quotient, &
                                  i32_immediate(session, 3_c_int64_t), e_exp, &
                                  error_msg)) return
        emit_en_field = emit_scaled_field(session, x, e_exp, digits, width, buf, &
                                          g_field, error_msg)
    end function emit_en_field
    logical function emit_scaled_field(session, x, e_exp, digits, width, buf, &
                                       g_field, error_msg)
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: x, e_exp, digits, width, buf
        type(lr_operand_desc_t), intent(in) :: g_field
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: abs_exp, exp_f64, scale, mant, field_width
        type(lr_operand_desc_t) :: args(8), cond
        integer(c_int32_t) :: neg, pos
        emit_scaled_field = .false.
        if (.not. scaled_mantissa(session, x, e_exp, mant, error_msg)) return
        if (.not. emit_i32_binary(session, LR_OP_SUB, width, &
                                  i32_immediate(session, 4_c_int64_t), &
                                  field_width, error_msg)) return
        if (.not. emit_liric_i32_icmp(session, LR_CMP_SLT, e_exp, &
                                      i32_immediate(session, 0_c_int64_t), &
                                      cond, error_msg)) return
        neg = create_liric_block(session)
        pos = create_liric_block(session)
        if (.not. emit_liric_condbr(session, cond, neg, pos, error_msg)) return
        if (.not. set_liric_block(session, neg, error_msg)) return
        if (.not. emit_i32_binary(session, LR_OP_SUB, &
                                  i32_immediate(session, 0_c_int64_t), e_exp, &
                                  abs_exp, error_msg)) return
        if (.not. snprintf_scaled(session, buf, g_field, field_width, digits, &
                                  mant, '-', abs_exp, error_msg)) return
        if (.not. set_liric_block(session, pos, error_msg)) return
        if (.not. snprintf_scaled(session, buf, g_field, field_width, digits, &
                                  mant, '+', e_exp, error_msg)) return
        emit_scaled_field = .true.
    end function emit_scaled_field
    logical function scaled_mantissa(session, x, e_exp, mant, error_msg)
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: x, e_exp
        type(lr_operand_desc_t), intent(out) :: mant
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: exp_f64, scale, args(2)
        integer(c_int32_t) :: vreg
        scaled_mantissa = .false.
        if (.not. cast_i32_to_f64(session, e_exp, exp_f64, error_msg)) return
        args(1) = f64_immediate(session, 10.0_c_double)
        args(2) = exp_f64
        if (.not. emit_c_call_vreg(session, 'pow', args, &
                                   lr_type_f64_s(session%handle), 2_c_int32_t, &
                                   c_false, vreg, error_msg)) return
        scale%kind = LR_OP_KIND_VREG
        scale%payload = int(vreg, c_int64_t)
        scale%typ = lr_type_f64_s(session%handle)
        scale%global_offset = 0_c_int64_t
        if (.not. emit_f64_binary(session, LR_OP_FDIV, x, scale, mant, &
                                  error_msg)) return
        scaled_mantissa = .true.
    end function scaled_mantissa
    logical function snprintf_scaled(session, buf, g_field, field_width, digits, &
                                     mant, sign_ch, abs_exp, error_msg)
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: buf, g_field, field_width, digits
        type(lr_operand_desc_t), intent(in) :: mant, abs_exp
        character, intent(in) :: sign_ch
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: args(8)
        snprintf_scaled = .false.
        args(1) = buf
        args(2) = i64_immediate(session, 256_c_int64_t)
        args(3) = g_field
        args(4) = field_width
        args(5) = digits
        args(6) = mant
        args(7) = i32_immediate(session, int(iachar(sign_ch), c_int64_t))
        args(8) = abs_exp
        if (.not. emit_c_call(session, 'snprintf', args, &
                              lr_type_i32_s(session%handle), 3_c_int32_t, &
                              c_true, error_msg)) return
        if (.not. return_zero(session, error_msg)) return
        snprintf_scaled = .true.
    end function snprintf_scaled
    logical function copy_nonfinite_field(session, tmp, width, buf, error_msg)
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: tmp, width, buf
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: args(5), g_fmt
        copy_nonfinite_field = .false.
        if (.not. create_cstring_operand(session, '.ffc.een.nonfinite', '%*s', &
                                         g_fmt, error_msg)) return
        args(1) = buf
        args(2) = i64_immediate(session, 256_c_int64_t)
        args(3) = g_fmt
        args(4) = width
        args(5) = tmp
        if (.not. emit_c_call(session, 'snprintf', args, &
                              lr_type_i32_s(session%handle), 3_c_int32_t, &
                              c_true, error_msg)) return
        if (.not. return_zero(session, error_msg)) return
        copy_nonfinite_field = .true.
    end function copy_nonfinite_field
    logical function return_zero(session, error_msg)
        type(liric_session_t), intent(inout) :: session
        character(len=:), allocatable, intent(out) :: error_msg
        return_zero = emit_ret_i32_local(session, &
                                         i32_immediate(session, 0_c_int64_t), &
                                         error_msg)
    end function return_zero
    logical function emit_e_en_format_call(session, value, mode, digits, width, &
                                           buf, error_msg)
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: value
        integer, intent(in) :: mode, digits, width
        type(lr_operand_desc_t), intent(in) :: buf
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: args(5)
        args(1) = value
        args(2) = i32_immediate(session, int(mode, c_int64_t))
        args(3) = i32_immediate(session, int(digits, c_int64_t))
        args(4) = i32_immediate(session, int(width, c_int64_t))
        args(5) = buf
        emit_e_en_format_call = emit_c_call(session, E_EN_FORMAT_HELPER, args, &
                                            lr_type_i32_s(session%handle), &
                                            5_c_int32_t, c_false, error_msg)
    end function emit_e_en_format_call
    logical function declare_e_en_libc(session, error_msg)
        type(liric_session_t), intent(inout) :: session
        character(len=:), allocatable, intent(out) :: error_msg
        type(c_ptr), target :: p3(3), p2(2), p1(1), f1(1)
        declare_e_en_libc = .false.
        p3(1) = lr_type_ptr_s(session%handle)
        p3(2) = lr_type_i64_s(session%handle)
        p3(3) = lr_type_ptr_s(session%handle)
        if (.not. declare_c_func(session, 'snprintf', &
                                 lr_type_i32_s(session%handle), c_loc(p3), &
                                 3_c_int32_t, c_true, error_msg)) return
        p2(1) = lr_type_ptr_s(session%handle)
        p2(2) = lr_type_i32_s(session%handle)
        if (.not. declare_c_func(session, 'strchr', &
                                 lr_type_ptr_s(session%handle), c_loc(p2), &
                                 2_c_int32_t, c_false, error_msg)) return
        p1(1) = lr_type_ptr_s(session%handle)
        if (.not. declare_c_func(session, 'atoi', lr_type_i32_s(session%handle), &
                                 c_loc(p1), 1_c_int32_t, c_false, &
                                 error_msg)) return
        f1(1) = lr_type_f64_s(session%handle)
        if (.not. declare_c_func(session, 'floor', lr_type_f64_s(session%handle), &
                                 c_loc(f1), 1_c_int32_t, c_false, error_msg)) &
            return
        p2(1) = lr_type_f64_s(session%handle)
        p2(2) = lr_type_f64_s(session%handle)
        if (.not. declare_c_func(session, 'pow', lr_type_f64_s(session%handle), &
                                 c_loc(p2), 2_c_int32_t, c_false, error_msg)) &
            return
        declare_e_en_libc = .true.
    end function declare_e_en_libc
    logical function declare_c_func(session, name, ret, params_ptr, n, vararg, &
                                    error_msg)
        type(liric_session_t), intent(inout) :: session
        character(len=*), intent(in) :: name
        type(c_ptr), intent(in) :: ret, params_ptr
        integer(c_int32_t), intent(in) :: n
        logical(c_bool), intent(in) :: vararg
        character(len=:), allocatable, intent(out) :: error_msg
        character(kind=c_char), allocatable :: c_name(:)
        type(lr_error_t) :: error
        integer(c_int) :: status
        declare_c_func = .false.
        call clear_liric_error(error)
        call to_c_chars(name, c_name)
        status = lr_session_declare(session%handle, c_name, ret, params_ptr, n, &
                                    vararg, error)
        if (.not. status_ok(status, error, error_msg)) return
        declare_c_func = .true.
    end function declare_c_func
    logical function create_cstring_operand(session, name, text, operand, &
                                            error_msg)
        type(liric_session_t), intent(inout) :: session
        character(len=*), intent(in) :: name, text
        type(lr_operand_desc_t), intent(out) :: operand
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int32_t) :: global_id
        create_cstring_operand = .false.
        call create_printf_format_global(session, name, text, global_id, &
                                         error_msg)
        if (len_trim(error_msg) > 0) return
        operand = printf_format_ptr(session, global_id)
        create_cstring_operand = .true.
    end function create_cstring_operand
    function typed_param(session, index, typ) result(operand)
        type(liric_session_t), intent(in) :: session
        integer(c_int32_t), intent(in) :: index
        type(c_ptr), intent(in) :: typ
        type(lr_operand_desc_t) :: operand
        operand%kind = LR_OP_KIND_VREG
        operand%payload = int(lr_session_param(session%handle, index), c_int64_t)
        operand%typ = typ
        operand%global_offset = 0_c_int64_t
    end function typed_param
    subroutine null_ptr_operand(session, operand)
        type(liric_session_t), intent(in) :: session
        type(lr_operand_desc_t), intent(out) :: operand
        operand%kind = LR_OP_KIND_IMM_I64
        operand%payload = 0_c_int64_t
        operand%typ = lr_type_ptr_s(session%handle)
        operand%global_offset = 0_c_int64_t
    end subroutine null_ptr_operand
    logical function find_char(session, text, ch, result, error_msg)
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: text
        character, intent(in) :: ch
        type(lr_operand_desc_t), intent(out) :: result
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: args(2)
        integer(c_int32_t) :: vreg
        find_char = .false.
        args(1) = text
        args(2) = i32_immediate(session, int(iachar(ch), c_int64_t))
        if (.not. emit_c_call_vreg(session, 'strchr', args, &
                                   lr_type_ptr_s(session%handle), 2_c_int32_t, &
                                   c_false, vreg, error_msg)) return
        result = ptr_vreg(session, vreg)
        find_char = .true.
    end function find_char
    logical function emit_c_call(session, name, args, ret_typ, fixed_args, &
                                 vararg, error_msg)
        type(liric_session_t), intent(inout) :: session
        character(len=*), intent(in) :: name
        type(lr_operand_desc_t), intent(in) :: args(:)
        type(c_ptr), intent(in) :: ret_typ
        integer(c_int32_t), intent(in) :: fixed_args
        logical(c_bool), intent(in) :: vararg
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int32_t) :: vreg
        emit_c_call = emit_c_call_vreg(session, name, args, ret_typ, fixed_args, &
                                       vararg, vreg, error_msg)
    end function emit_c_call
    logical function emit_c_call_vreg(session, name, args, ret_typ, fixed_args, &
                                      vararg, vreg, error_msg)
        type(liric_session_t), intent(inout) :: session
        character(len=*), intent(in) :: name
        type(lr_operand_desc_t), intent(in) :: args(:)
        type(c_ptr), intent(in) :: ret_typ
        integer(c_int32_t), intent(in) :: fixed_args
        logical(c_bool), intent(in) :: vararg
        integer(c_int32_t), intent(out) :: vreg
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t), allocatable, target :: operands(:)
        type(lr_operand_desc_t) :: callee
        type(lr_inst_desc_t) :: inst
        type(lr_error_t) :: error
        character(kind=c_char), allocatable :: c_name(:)
        integer(c_int32_t) :: sym_id
        integer :: nargs
        emit_c_call_vreg = .false.
        call to_c_chars(name, c_name)
        sym_id = lr_session_intern(session%handle, c_name)
        if (sym_id < 0_c_int32_t) then
            error_msg = 'emit_c_call: could not intern '//trim(name)
            return
        end if
        callee%kind = LR_OP_KIND_GLOBAL
        callee%payload = int(sym_id, c_int64_t)
        callee%typ = lr_type_ptr_s(session%handle)
        callee%global_offset = 0_c_int64_t
        nargs = size(args)
        allocate (operands(nargs + 1))
        operands(1) = callee
        operands(2:nargs + 1) = args
        inst%op = LR_OP_CALL
        inst%typ = ret_typ
        inst%dest = 0_c_int32_t
        inst%operands = c_loc(operands)
        inst%num_operands = int(nargs + 1, c_int32_t)
        inst%indices = c_null_ptr
        inst%num_indices = 0_c_int32_t
        inst%align = 0_c_int32_t
        inst%icmp_pred = 0_c_int
        inst%fcmp_pred = 0_c_int
        inst%call_external_abi = c_true
        inst%call_vararg = vararg
        inst%call_fixed_args = fixed_args
        call clear_liric_error(error)
        vreg = lr_session_emit(session%handle, inst, error)
        if (.not. status_ok(error%code, error, error_msg)) return
        call set_empty(error_msg)
        emit_c_call_vreg = .true.
    end function emit_c_call_vreg
    logical function gep_byte(session, base, offset, result, error_msg)
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: base
        integer(c_int64_t), intent(in) :: offset
        type(lr_operand_desc_t), intent(out) :: result
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t), target :: operands(2)
        type(lr_inst_desc_t) :: inst
        type(lr_error_t) :: error
        integer(c_int32_t) :: vreg
        gep_byte = .false.
        operands(1) = base
        operands(2) = i64_immediate(session, offset)
        inst%op = 29_c_int
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
        result = ptr_vreg(session, vreg)
        gep_byte = .true.
    end function gep_byte
    logical function cast_i32_to_f64(session, source, result, error_msg)
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: source
        type(lr_operand_desc_t), intent(out) :: result
        character(len=:), allocatable, intent(out) :: error_msg
        cast_i32_to_f64 = emit_cast(session, LR_OP_SITOFP, source, &
                                    lr_type_f64_s(session%handle), result, &
                                    error_msg)
    end function cast_i32_to_f64
    logical function cast_f64_to_i32(session, source, result, error_msg)
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: source
        type(lr_operand_desc_t), intent(out) :: result
        character(len=:), allocatable, intent(out) :: error_msg
        cast_f64_to_i32 = emit_cast(session, LR_OP_FPTOSI, source, &
                                    lr_type_i32_s(session%handle), result, &
                                    error_msg)
    end function cast_f64_to_i32
    logical function emit_cast(session, op, src, dst_typ, result, error_msg)
        type(liric_session_t), intent(inout) :: session
        integer(c_int), intent(in) :: op
        type(lr_operand_desc_t), intent(in) :: src
        type(c_ptr), intent(in) :: dst_typ
        type(lr_operand_desc_t), intent(out) :: result
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t), target :: operands(1)
        type(lr_inst_desc_t) :: inst
        type(lr_error_t) :: error
        integer(c_int32_t) :: vreg
        emit_cast = .false.
        operands(1) = src
        inst%op = op
        inst%typ = dst_typ
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
        vreg = lr_session_emit(session%handle, inst, error)
        if (.not. status_ok(error%code, error, error_msg)) return
        result%kind = LR_OP_KIND_VREG
        result%payload = int(vreg, c_int64_t)
        result%typ = dst_typ
        result%global_offset = 0_c_int64_t
        emit_cast = .true.
    end function emit_cast
    logical function emit_f64_binary(session, opcode, lhs, rhs, result, error_msg)
        type(liric_session_t), intent(inout) :: session
        integer(c_int), intent(in) :: opcode
        type(lr_operand_desc_t), intent(in) :: lhs, rhs
        type(lr_operand_desc_t), intent(out) :: result
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t), target :: operands(2)
        type(lr_inst_desc_t) :: inst
        type(lr_error_t) :: error
        integer(c_int32_t) :: vreg
        emit_f64_binary = .false.
        operands(1) = lhs
        operands(2) = rhs
        inst%op = opcode
        inst%typ = lr_type_f64_s(session%handle)
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
        result%kind = LR_OP_KIND_VREG
        result%payload = int(vreg, c_int64_t)
        result%typ = lr_type_f64_s(session%handle)
        result%global_offset = 0_c_int64_t
        emit_f64_binary = .true.
    end function emit_f64_binary
    logical function call_f64_unary(session, name, arg, result, error_msg)
        type(liric_session_t), intent(inout) :: session
        character(len=*), intent(in) :: name
        type(lr_operand_desc_t), intent(in) :: arg
        type(lr_operand_desc_t), intent(out) :: result
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: args(1)
        integer(c_int32_t) :: vreg
        call_f64_unary = .false.
        args(1) = arg
        if (.not. emit_c_call_vreg(session, name, args, &
                                   lr_type_f64_s(session%handle), 1_c_int32_t, &
                                   c_false, vreg, error_msg)) return
        result%kind = LR_OP_KIND_VREG
        result%payload = int(vreg, c_int64_t)
        result%typ = lr_type_f64_s(session%handle)
        result%global_offset = 0_c_int64_t
        call_f64_unary = .true.
    end function call_f64_unary
    function f64_immediate(session, value) result(operand)
        type(liric_session_t), intent(in) :: session
        real(c_double), intent(in) :: value
        type(lr_operand_desc_t) :: operand
        operand%kind = LR_OP_KIND_IMM_F64
        operand%payload = transfer(value, operand%payload)
        operand%typ = lr_type_f64_s(session%handle)
        operand%global_offset = 0_c_int64_t
    end function f64_immediate
    logical function emit_ret_i32_local(session, value, error_msg)
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t), target :: operands(1)
        type(lr_inst_desc_t) :: inst
        type(lr_error_t) :: error
        integer(c_int32_t) :: vreg
        emit_ret_i32_local = .false.
        operands(1) = value
        inst%op = LR_OP_RET
        inst%typ = lr_type_i32_s(session%handle)
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
        vreg = lr_session_emit(session%handle, inst, error)
        if (.not. status_ok(error%code, error, error_msg)) return
        emit_ret_i32_local = .true.
    end function emit_ret_i32_local

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

end module liric_session_format_bindings
