submodule (liric_session_format_bindings) liric_session_format_e_en
contains
    ! Synthesized .ffc.fmt_e_en runtime helper and its emit utilities for the
    ! Fortran E and EN edit descriptors. printf cannot produce the 0.dddE+nn
    ! (E) or engineering (EN) forms, so the helper rebuilds the field from a
    ! normalized %.*E decomposition. Implemented in a separate module subprogram.

    module procedure synthesize_e_en_format_helper
        type(lr_operand_desc_t) :: g_norm, g_field
        type(lr_operand_desc_t) :: x, mode, digits, width, buf, tmp, p, pe1, ex
        type(lr_operand_desc_t) :: exp_digits
        type(lr_operand_desc_t) :: cond, nullptr, zptr
        integer(c_int32_t) :: entry, finite, nonfinite, eblk, enblk
        integer(c_int32_t) :: ezero, enonzero
        type(c_ptr), target :: params(6)
        type(c_ptr) :: out_addr
        character(kind=c_char), allocatable :: c_name(:)
        type(lr_error_t) :: error
        integer(c_int) :: status
        synthesize_e_en_format_helper = .false.
        if (.not. require_open_session(session, error_msg)) return
        if (.not. declare_e_en_libc(session, error_msg)) return
        if (.not. create_cstring_operand(session, '.ffc.een.norm', '%.17E', &
                                         g_norm, error_msg)) return
        ! The "#" flag keeps the decimal point when the fractional digit count
        ! is zero (gfortran's Ew.0/ENw.0 print "3.E+00"; plain %.0f drops it).
        if (.not. create_cstring_operand(session, '.ffc.een.field', &
                                         '%#*.*fE%c%0*d', g_field, error_msg)) &
            return
        call begin_e_en_helper(session, params, c_name, status, error)
        if (.not. status_ok(status, error, error_msg)) return
        call e_en_params(session, x, mode, digits, width, buf, exp_digits)
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
                                    exp_digits, error_msg)) return
        if (.not. set_liric_block(session, enonzero, error_msg)) return
        if (.not. emit_e_field(session, x, ex, digits, width, buf, g_field, &
                               exp_digits, error_msg)) return
        if (.not. set_liric_block(session, enblk, error_msg)) return
        if (.not. emit_en_field(session, x, ex, digits, width, buf, g_field, &
                                exp_digits, error_msg)) return
        if (.not. set_liric_block(session, nonfinite, error_msg)) return
        if (.not. copy_nonfinite_field(session, tmp, width, buf, error_msg)) return
        call clear_liric_error(error)
        out_addr = c_null_ptr
        status = lr_session_func_end(session%handle, out_addr, error)
        if (.not. status_ok(status, error, error_msg)) return
        call set_empty(error_msg)
        synthesize_e_en_format_helper = .true.
    end procedure synthesize_e_en_format_helper
    module procedure begin_e_en_helper
        params(1) = lr_type_f64_s(session%handle)
        params(2) = lr_type_i32_s(session%handle)
        params(3) = lr_type_i32_s(session%handle)
        params(4) = lr_type_i32_s(session%handle)
        params(5) = lr_type_ptr_s(session%handle)
        params(6) = lr_type_i32_s(session%handle)
        call clear_liric_error(error)
        call to_c_chars(E_EN_FORMAT_HELPER, c_name)
        status = lr_session_func_begin(session%handle, c_name, &
                                       lr_type_i32_s(session%handle), &
                                       c_loc(params), 6_c_int32_t, c_false, &
                                       error)
    end procedure begin_e_en_helper
    module procedure e_en_params
        x = typed_param(session, 0_c_int32_t, lr_type_f64_s(session%handle))
        mode = typed_param(session, 1_c_int32_t, lr_type_i32_s(session%handle))
        digits = typed_param(session, 2_c_int32_t, lr_type_i32_s(session%handle))
        width = typed_param(session, 3_c_int32_t, lr_type_i32_s(session%handle))
        buf = typed_param(session, 4_c_int32_t, lr_type_ptr_s(session%handle))
        exp_digits = typed_param(session, 5_c_int32_t, &
                                 lr_type_i32_s(session%handle))
    end procedure e_en_params
    module procedure create_e_en_blocks
        entry = create_liric_block(session)
        finite = create_liric_block(session)
        nonfinite = create_liric_block(session)
        eblk = create_liric_block(session)
        enblk = create_liric_block(session)
        ezero = create_liric_block(session)
        enonzero = create_liric_block(session)
    end procedure create_e_en_blocks
    module procedure emit_e_en_head
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
    end procedure emit_e_en_head
    module procedure parse_e_en_exponent
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
    end procedure parse_e_en_exponent
    module procedure emit_e_field
        type(lr_operand_desc_t) :: e_exp
        emit_e_field = .false.
        if (.not. emit_i32_binary(session, LR_OP_ADD, ex, &
                                  i32_immediate(session, 1_c_int64_t), e_exp, &
                                  error_msg)) return
        emit_e_field = emit_scaled_field(session, x, e_exp, digits, width, buf, &
                                         g_field, exp_digits, error_msg)
    end procedure emit_e_field
    module procedure emit_en_field
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
                                          g_field, exp_digits, error_msg)
    end procedure emit_en_field
    module procedure emit_scaled_field
        type(lr_operand_desc_t) :: abs_exp, exp_f64, scale, mant, field_width
        type(lr_operand_desc_t) :: exp_room
        type(lr_operand_desc_t) :: args(8), cond
        integer(c_int32_t) :: neg, pos
        emit_scaled_field = .false.
        if (.not. scaled_mantissa(session, x, e_exp, mant, error_msg)) return
        ! The exponent occupies "E", the sign, and exp_digits digits, so the
        ! mantissa field takes the remaining width.
        if (.not. emit_i32_binary(session, LR_OP_ADD, exp_digits, &
                                  i32_immediate(session, 2_c_int64_t), &
                                  exp_room, error_msg)) return
        if (.not. emit_i32_binary(session, LR_OP_SUB, width, exp_room, &
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
                                  mant, '-', abs_exp, exp_digits, error_msg)) &
            return
        if (.not. set_liric_block(session, pos, error_msg)) return
        if (.not. snprintf_scaled(session, buf, g_field, field_width, digits, &
                                  mant, '+', e_exp, exp_digits, error_msg)) return
        emit_scaled_field = .true.
    end procedure emit_scaled_field
    module procedure scaled_mantissa
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
    end procedure scaled_mantissa
    module procedure snprintf_scaled
        type(lr_operand_desc_t) :: args(9)
        snprintf_scaled = .false.
        args(1) = buf
        args(2) = i64_immediate(session, 256_c_int64_t)
        args(3) = g_field
        args(4) = field_width
        args(5) = digits
        args(6) = mant
        args(7) = i32_immediate(session, int(iachar(sign_ch), c_int64_t))
        args(8) = exp_digits
        args(9) = abs_exp
        if (.not. emit_c_call(session, 'snprintf', args, &
                              lr_type_i32_s(session%handle), 3_c_int32_t, &
                              c_true, error_msg)) return
        if (.not. return_zero(session, error_msg)) return
        snprintf_scaled = .true.
    end procedure snprintf_scaled
    module procedure copy_nonfinite_field
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
    end procedure copy_nonfinite_field
    module procedure return_zero
        return_zero = emit_ret_i32_local(session, &
                                         i32_immediate(session, 0_c_int64_t), &
                                         error_msg)
    end procedure return_zero
    module procedure emit_e_en_format_call
        ! exp_digits is the Ew.dEe exponent digit count; it defaults to the
        ! standard two-digit exponent when the descriptor omits Ee.
        integer :: exp_width
        type(lr_operand_desc_t) :: args(6)
        args(1) = value
        args(2) = i32_immediate(session, int(mode, c_int64_t))
        args(3) = i32_immediate(session, int(digits, c_int64_t))
        args(4) = i32_immediate(session, int(width, c_int64_t))
        args(5) = buf
        exp_width = 2
        if (present(exp_digits)) exp_width = exp_digits
        args(6) = i32_immediate(session, int(exp_width, c_int64_t))
        emit_e_en_format_call = emit_c_call(session, E_EN_FORMAT_HELPER, args, &
                                            lr_type_i32_s(session%handle), &
                                            6_c_int32_t, c_false, error_msg)
    end procedure emit_e_en_format_call
    module procedure declare_e_en_libc
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
    end procedure declare_e_en_libc
    module procedure declare_c_func
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
    end procedure declare_c_func
    module procedure create_cstring_operand
        integer(c_int32_t) :: global_id
        create_cstring_operand = .false.
        call create_printf_format_global(session, name, text, global_id, &
                                         error_msg)
        if (len_trim(error_msg) > 0) return
        operand = printf_format_ptr(session, global_id)
        create_cstring_operand = .true.
    end procedure create_cstring_operand
    module procedure typed_param
        operand%kind = LR_OP_KIND_VREG
        operand%payload = int(lr_session_param(session%handle, index), c_int64_t)
        operand%typ = typ
        operand%global_offset = 0_c_int64_t
    end procedure typed_param
    module procedure null_ptr_operand
        operand%kind = LR_OP_KIND_IMM_I64
        operand%payload = 0_c_int64_t
        operand%typ = lr_type_ptr_s(session%handle)
        operand%global_offset = 0_c_int64_t
    end procedure null_ptr_operand
    module procedure find_char
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
    end procedure find_char
    module procedure emit_c_call
        integer(c_int32_t) :: vreg
        emit_c_call = emit_c_call_vreg(session, name, args, ret_typ, fixed_args, &
                                       vararg, vreg, error_msg)
    end procedure emit_c_call
    module procedure emit_c_call_vreg
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
    end procedure emit_c_call_vreg
    module procedure gep_byte
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
    end procedure gep_byte
    module procedure cast_i32_to_f64
        cast_i32_to_f64 = emit_cast(session, LR_OP_SITOFP, source, &
                                    lr_type_f64_s(session%handle), result, &
                                    error_msg)
    end procedure cast_i32_to_f64
    module procedure cast_f64_to_i32
        cast_f64_to_i32 = emit_cast(session, LR_OP_FPTOSI, source, &
                                    lr_type_i32_s(session%handle), result, &
                                    error_msg)
    end procedure cast_f64_to_i32
    module procedure emit_cast
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
    end procedure emit_cast
    module procedure emit_f64_binary
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
    end procedure emit_f64_binary
    module procedure call_f64_unary
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
    end procedure call_f64_unary
    module procedure f64_immediate
        operand%kind = LR_OP_KIND_IMM_F64
        operand%payload = transfer(value, operand%payload)
        operand%typ = lr_type_f64_s(session%handle)
        operand%global_offset = 0_c_int64_t
    end procedure f64_immediate
    module procedure emit_ret_i32_local
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
    end procedure emit_ret_i32_local

end submodule liric_session_format_e_en
