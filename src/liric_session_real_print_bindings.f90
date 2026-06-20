module liric_session_real_print_bindings
    ! Synthesized runtime helpers emitted once into the module:
    !  * .ffc.print_real8 - gfortran list-directed real(8) output.
    !  * .ffc.get_arg      - copy command-line argument argv[i] into a
    !                        fixed-length character variable, blank-padded.
    ! The real(8) printer is documented below; the argument copier reuses the
    ! same low-level emission helpers.
    !
    ! gfortran prints 17 significant digits for real(8). For a decimal
    ! exponent ex in [-1, 16] it uses fixed (F) notation with 16 - ex
    ! digits after the point, right-justified in a 20-wide field followed by
    ! five trailing blanks. Otherwise it uses exponential notation (one digit
    ! before the point, 16 after, an uppercase E, a sign, and a three-digit
    ! exponent), right-justified in a 25-wide field. The list-directed
    ! leading blank is emitted by the print statement, not here.
    use, intrinsic :: iso_c_binding, only: c_associated, c_bool, c_char, c_double, &
                                           c_int, c_int32_t, c_int64_t, c_loc, &
                                           c_null_char, c_null_ptr, c_ptr, c_size_t
    use liric_session_bindings, only: liric_session_t, lr_error_t, &
                                      lr_operand_desc_t, lr_inst_desc_t, LR_OK, &
                                      LR_OP_CALL, LR_OP_SUB, LR_OP_STORE, &
                                      LR_OP_GEP, LR_OP_LOAD, LR_OP_ZEXT, &
                                      LR_OP_ADD, LR_OP_TRUNC, LR_OP_FPEXT, &
                                      LR_OP_KIND_VREG, &
                                      LR_OP_KIND_IMM_I64, &
                                      LR_OP_KIND_GLOBAL, lr_type_i32_s, &
                                      lr_type_i64_s, lr_type_f32_s, lr_type_f64_s, &
                                      lr_type_i8_s, &
                                      lr_type_ptr_s, lr_type_array_s, &
                                      lr_session_emit, lr_session_intern, &
                                      lr_session_global, lr_session_vreg, &
                                      lr_session_param, status_ok, clear_liric_error, &
                                      set_empty, require_open_session, to_c_chars, &
                                      i32_vreg, i32_immediate
    use liric_session_memory_bindings, only: emit_alloca_bytes, emit_memcpy, &
                                             emit_i32_binary, emit_i64_binary, &
                                             ptr_vreg, i64_vreg, i64_immediate
    use liric_session_control_bindings, only: create_liric_block, set_liric_block, &
                                              emit_liric_br, emit_liric_condbr, &
                                              emit_liric_i32_icmp, LR_CMP_SGE, &
                                              LR_CMP_SLE, LR_CMP_SLT, &
                                              LR_CMP_NE, LR_CMP_EQ
    implicit none
    private

    logical(c_bool), parameter :: c_false = .false.
    logical(c_bool), parameter :: c_true = .true.

    character(len=*), parameter :: REAL8_PRINTER = '.ffc.print_real8'
    character(len=*), parameter :: REAL4_PRINTER = '.ffc.print_real4'

    public :: synthesize_real8_printer
    public :: emit_real8_print_call
    public :: synthesize_real4_printer
    public :: emit_real4_print_call
    public :: synthesize_get_arg_helper
    public :: emit_get_arg_call
    public :: emit_snprintf
    public :: emit_sscanf
    public :: emit_scanf
    public :: emit_fscanf
    public :: emit_fprintf
    public :: emit_dprintf

    interface
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

        function lr_type_void_s(handle) result(typ) bind(c)
            import :: c_ptr
            type(c_ptr), value :: handle
            type(c_ptr) :: typ
        end function lr_type_void_s
    end interface

contains

    logical function synthesize_real8_printer(session, error_msg)
        type(liric_session_t), intent(inout) :: session
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: g_fe, g_ff, g_feo, g_padf, g_pade
        type(lr_operand_desc_t) :: g_inf, g_ninf, g_nan
        type(lr_operand_desc_t) :: x, tmp, num, p, ex
        integer(c_int32_t) :: finblk, nfblk
        integer(c_int32_t) :: chkhi, fblk, eblk, eneg, epos, eprint, exitb
        integer(c_int32_t) :: entry_block, param_vreg
        type(lr_error_t) :: error
        type(c_ptr) :: out_addr
        integer(c_int) :: status

        synthesize_real8_printer = .false.
        if (.not. require_open_session(session, error_msg)) return

        if (.not. declare_libc(session, error_msg)) return
        if (.not. create_cstring(session, '.ffc.r8.fe', '%.16e', g_fe, &
                                 error_msg)) return
        if (.not. create_cstring(session, '.ffc.r8.ff', '%#.*f', g_ff, &
                                 error_msg)) return
        if (.not. create_cstring(session, '.ffc.r8.feo', '%sE%c%03d', g_feo, &
                                 error_msg)) return
        if (.not. create_cstring(session, '.ffc.r8.padf', '%20s     ', g_padf, &
                                 error_msg)) return
        if (.not. create_cstring(session, '.ffc.r8.pade', '%25s', g_pade, &
                                 error_msg)) return
        if (.not. create_cstring(session, '.ffc.r8.inf', 'Infinity', g_inf, &
                                 error_msg)) return
        if (.not. create_cstring(session, '.ffc.r8.ninf', '-Infinity', g_ninf, &
                                 error_msg)) return
        if (.not. create_cstring(session, '.ffc.r8.nan', 'NaN', g_nan, &
                                 error_msg)) return

        if (.not. begin_real8_printer(session, param_vreg, entry_block, &
                                      error_msg)) return
        x%kind = LR_OP_KIND_VREG
        x%payload = int(param_vreg, c_int64_t)
        x%typ = lr_type_f64_s(session%handle)
        x%global_offset = 0_c_int64_t

        finblk = create_liric_block(session)
        nfblk = create_liric_block(session)
        chkhi = create_liric_block(session)
        fblk = create_liric_block(session)
        eblk = create_liric_block(session)
        eneg = create_liric_block(session)
        epos = create_liric_block(session)
        eprint = create_liric_block(session)
        exitb = create_liric_block(session)

        if (.not. set_liric_block(session, entry_block, error_msg)) return
        if (.not. build_head(session, x, g_fe, tmp, num, p, finblk, nfblk, &
                             error_msg)) return

        if (.not. set_liric_block(session, finblk, error_msg)) return
        if (.not. build_finite(session, tmp, p, ex, chkhi, eblk, error_msg)) return

        if (.not. set_liric_block(session, chkhi, error_msg)) return
        if (.not. branch_on_high(session, ex, fblk, eblk, error_msg)) return

        if (.not. set_liric_block(session, fblk, error_msg)) return
        if (.not. build_fixed(session, x, ex, num, g_ff, g_padf, exitb, &
                              error_msg)) return

        if (.not. build_exponential(session, ex, tmp, num, g_feo, g_pade, &
                                    eblk, eneg, epos, eprint, exitb, error_msg)) &
            return

        if (.not. set_liric_block(session, nfblk, error_msg)) return
        if (.not. build_nonfinite(session, tmp, g_inf, g_ninf, g_nan, g_pade, &
                                  exitb, error_msg)) return

        if (.not. set_liric_block(session, exitb, error_msg)) return
        if (.not. emit_ret_void_local(session, error_msg)) return

        call clear_liric_error(error)
        out_addr = c_null_ptr
        status = lr_session_func_end(session%handle, out_addr, error)
        if (.not. status_ok(status, error, error_msg)) return

        call set_empty(error_msg)
        synthesize_real8_printer = .true.
    end function synthesize_real8_printer

    logical function build_head(session, x, g_fe, tmp, num, p, finblk, nfblk, &
                                error_msg)
        ! tmp = "%.16e" of x; p = strchr(tmp,'e'). A finite value has an 'e';
        ! inf/nan do not, so a null p routes to the non-finite handler.
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: x, g_fe
        type(lr_operand_desc_t), intent(out) :: tmp, num, p
        integer(c_int32_t), intent(in) :: finblk, nfblk
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: args(4), nullptr, cond
        integer(c_int32_t) :: vreg

        build_head = .false.
        if (.not. emit_alloca_bytes(session, i64_immediate(session, 64_c_int64_t), &
                                    tmp, error_msg)) return
        if (.not. emit_alloca_bytes(session, i64_immediate(session, 48_c_int64_t), &
                                    num, error_msg)) return

        args(1) = tmp
        args(2) = i64_immediate(session, 64_c_int64_t)
        args(3) = g_fe
        args(4) = x
        if (.not. emit_call(session, 'snprintf', args, lr_type_i32_s(session%handle), &
                            3_c_int32_t, c_true, vreg, error_msg)) return

        args(1) = tmp
        args(2) = i32_immediate(session, int(iachar('e'), c_int64_t))
        if (.not. emit_call(session, 'strchr', args(1:2), &
                            lr_type_ptr_s(session%handle), 2_c_int32_t, c_false, &
                            vreg, error_msg)) return
        p = ptr_vreg(session, vreg)

        nullptr%kind = LR_OP_KIND_IMM_I64
        nullptr%payload = 0_c_int64_t
        nullptr%typ = lr_type_ptr_s(session%handle)
        nullptr%global_offset = 0_c_int64_t
        if (.not. emit_liric_i32_icmp(session, LR_CMP_NE, p, nullptr, cond, &
                                      error_msg)) return
        if (.not. emit_liric_condbr(session, cond, finblk, nfblk, error_msg)) return
        build_head = .true.
    end function build_head

    logical function build_finite(session, tmp, p, ex, chkhi, eblk, error_msg)
        ! ex = atoi(p+1); *p = 0 (truncate mantissa); branch to F path
        ! (ex >= -1) or fall through to the exponential path.
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: tmp, p
        type(lr_operand_desc_t), intent(out) :: ex
        integer(c_int32_t), intent(in) :: chkhi, eblk
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: pe1, args(1), cond
        integer(c_int32_t) :: vreg

        build_finite = .false.
        if (.not. gep_byte(session, p, 1_c_int64_t, pe1, error_msg)) return
        args(1) = pe1
        if (.not. emit_call(session, 'atoi', args(1:1), &
                            lr_type_i32_s(session%handle), 1_c_int32_t, c_false, &
                            vreg, error_msg)) return
        ex = i32_vreg(session, vreg)

        if (.not. store_zero_byte(session, p, error_msg)) return

        if (.not. emit_liric_i32_icmp(session, LR_CMP_SGE, ex, &
                i32_immediate(session, -1_c_int64_t), cond, error_msg)) return
        if (.not. emit_liric_condbr(session, cond, chkhi, eblk, error_msg)) return
        build_finite = .true.
    end function build_finite

    logical function build_nonfinite(session, buf, g_inf, g_ninf, g_nan, g_pade, &
                                     exitb, error_msg)
        ! buf holds glibc's "inf"/"-inf"/"nan". Map to gfortran's
        ! "Infinity"/"-Infinity"/"NaN", right-justified in the 25-wide field.
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: buf, g_inf, g_ninf, g_nan, g_pade
        integer(c_int32_t), intent(in) :: exitb
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: b0, b1, cond
        integer(c_int32_t) :: nf_pos, nf_neg, prt_inf, prt_ninf, prt_nan

        build_nonfinite = .false.
        nf_pos = create_liric_block(session)
        nf_neg = create_liric_block(session)
        prt_inf = create_liric_block(session)
        prt_ninf = create_liric_block(session)
        prt_nan = create_liric_block(session)

        if (.not. load_byte(session, buf, 0_c_int64_t, b0, error_msg)) return
        if (.not. emit_liric_i32_icmp(session, LR_CMP_EQ, b0, &
                i32_immediate(session, int(iachar('-'), c_int64_t)), cond, &
                error_msg)) return
        if (.not. emit_liric_condbr(session, cond, nf_neg, nf_pos, error_msg)) return

        ! positive: 'i' -> Infinity, else NaN
        if (.not. set_liric_block(session, nf_pos, error_msg)) return
        if (.not. emit_liric_i32_icmp(session, LR_CMP_EQ, b0, &
                i32_immediate(session, int(iachar('i'), c_int64_t)), cond, &
                error_msg)) return
        if (.not. emit_liric_condbr(session, cond, prt_inf, prt_nan, error_msg)) &
            return

        ! negative: byte 1 'i' -> -Infinity, else NaN
        if (.not. set_liric_block(session, nf_neg, error_msg)) return
        if (.not. load_byte(session, buf, 1_c_int64_t, b1, error_msg)) return
        if (.not. emit_liric_i32_icmp(session, LR_CMP_EQ, b1, &
                i32_immediate(session, int(iachar('i'), c_int64_t)), cond, &
                error_msg)) return
        if (.not. emit_liric_condbr(session, cond, prt_ninf, prt_nan, error_msg)) &
            return

        if (.not. nonfinite_leaf(session, prt_inf, g_pade, g_inf, exitb, &
                                 error_msg)) return
        if (.not. nonfinite_leaf(session, prt_ninf, g_pade, g_ninf, exitb, &
                                 error_msg)) return
        if (.not. nonfinite_leaf(session, prt_nan, g_pade, g_nan, exitb, &
                                 error_msg)) return
        build_nonfinite = .true.
    end function build_nonfinite

    logical function nonfinite_leaf(session, block, g_pade, text, exitb, error_msg)
        type(liric_session_t), intent(inout) :: session
        integer(c_int32_t), intent(in) :: block
        type(lr_operand_desc_t), intent(in) :: g_pade, text
        integer(c_int32_t), intent(in) :: exitb
        character(len=:), allocatable, intent(out) :: error_msg

        nonfinite_leaf = .false.
        if (.not. set_liric_block(session, block, error_msg)) return
        if (.not. emit_print(session, g_pade, text, error_msg)) return
        if (.not. emit_liric_br(session, exitb, error_msg)) return
        nonfinite_leaf = .true.
    end function nonfinite_leaf

    logical function load_byte(session, base, offset, result, error_msg)
        ! result(i32) = zext(load i8 base[offset])
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: base
        integer(c_int64_t), intent(in) :: offset
        type(lr_operand_desc_t), intent(out) :: result
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: addr, byte_val
        type(lr_operand_desc_t), target :: operands(1)
        type(lr_inst_desc_t) :: inst
        type(lr_error_t) :: error
        integer(c_int32_t) :: vreg

        load_byte = .false.
        if (.not. gep_byte(session, base, offset, addr, error_msg)) return

        operands(1) = addr
        inst%op = LR_OP_LOAD
        inst%typ = lr_type_i8_s(session%handle)
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
        byte_val%kind = LR_OP_KIND_VREG
        byte_val%payload = int(vreg, c_int64_t)
        byte_val%typ = lr_type_i8_s(session%handle)
        byte_val%global_offset = 0_c_int64_t

        operands(1) = byte_val
        inst%op = LR_OP_ZEXT
        inst%typ = lr_type_i32_s(session%handle)
        inst%num_operands = 1_c_int32_t
        call clear_liric_error(error)
        vreg = lr_session_emit(session%handle, inst, error)
        if (.not. status_ok(error%code, error, error_msg)) return
        result = i32_vreg(session, vreg)
        call set_empty(error_msg)
        load_byte = .true.
    end function load_byte

    logical function branch_on_high(session, ex, fblk, eblk, error_msg)
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: ex
        integer(c_int32_t), intent(in) :: fblk, eblk
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: cond

        branch_on_high = .false.
        if (.not. emit_liric_i32_icmp(session, LR_CMP_SLE, ex, &
                i32_immediate(session, 16_c_int64_t), cond, error_msg)) return
        if (.not. emit_liric_condbr(session, cond, fblk, eblk, error_msg)) return
        branch_on_high = .true.
    end function branch_on_high

    logical function build_fixed(session, x, ex, num, g_ff, g_padf, exitb, &
                                 error_msg)
        ! num = snprintf("%#.*f", 16 - ex, x); printf(" %20s     ", num)
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: x, ex, num, g_ff, g_padf
        integer(c_int32_t), intent(in) :: exitb
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: dd, args(5)
        integer(c_int32_t) :: vreg

        build_fixed = .false.
        if (.not. emit_i32_binary(session, LR_OP_SUB, &
                i32_immediate(session, 16_c_int64_t), ex, dd, error_msg)) return

        args(1) = num
        args(2) = i64_immediate(session, 48_c_int64_t)
        args(3) = g_ff
        args(4) = dd
        args(5) = x
        if (.not. emit_call(session, 'snprintf', args, lr_type_i32_s(session%handle), &
                            3_c_int32_t, c_true, vreg, error_msg)) return

        if (.not. emit_print(session, g_padf, num, error_msg)) return
        if (.not. emit_liric_br(session, exitb, error_msg)) return
        build_fixed = .true.
    end function build_fixed

    logical function build_exponential(session, ex, tmp, num, g_feo, g_pade, &
                                       eblk, eneg, epos, eprint, exitb, error_msg)
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: ex, tmp, num, g_feo, g_pade
        integer(c_int32_t), intent(in) :: eblk, eneg, epos, eprint, exitb
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: cond, negex

        build_exponential = .false.
        if (.not. set_liric_block(session, eblk, error_msg)) return
        if (.not. emit_liric_i32_icmp(session, LR_CMP_SLT, ex, &
                i32_immediate(session, 0_c_int64_t), cond, error_msg)) return
        if (.not. emit_liric_condbr(session, cond, eneg, epos, error_msg)) return

        if (.not. set_liric_block(session, eneg, error_msg)) return
        if (.not. emit_i32_binary(session, LR_OP_SUB, &
                i32_immediate(session, 0_c_int64_t), ex, negex, error_msg)) return
        if (.not. emit_exp_format(session, num, g_feo, tmp, iachar('-'), negex, &
                                  error_msg)) return
        if (.not. emit_liric_br(session, eprint, error_msg)) return

        if (.not. set_liric_block(session, epos, error_msg)) return
        if (.not. emit_exp_format(session, num, g_feo, tmp, iachar('+'), ex, &
                                  error_msg)) return
        if (.not. emit_liric_br(session, eprint, error_msg)) return

        if (.not. set_liric_block(session, eprint, error_msg)) return
        if (.not. emit_print(session, g_pade, num, error_msg)) return
        if (.not. emit_liric_br(session, exitb, error_msg)) return
        build_exponential = .true.
    end function build_exponential

    logical function emit_exp_format(session, num, g_feo, tmp, sign_char, expval, &
                                     error_msg)
        ! num = snprintf("%sE%c%03d", tmp, sign_char, expval)
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: num, g_feo, tmp, expval
        integer, intent(in) :: sign_char
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: args(6)
        integer(c_int32_t) :: vreg

        args(1) = num
        args(2) = i64_immediate(session, 48_c_int64_t)
        args(3) = g_feo
        args(4) = tmp
        args(5) = i32_immediate(session, int(sign_char, c_int64_t))
        args(6) = expval
        emit_exp_format = emit_call(session, 'snprintf', args, &
                                    lr_type_i32_s(session%handle), 3_c_int32_t, &
                                    c_true, vreg, error_msg)
    end function emit_exp_format

    logical function emit_print(session, fmt, str, error_msg)
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: fmt, str
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: args(2)
        integer(c_int32_t) :: vreg

        args(1) = fmt
        args(2) = str
        emit_print = emit_call(session, 'printf', args, &
                               lr_type_i32_s(session%handle), 1_c_int32_t, c_true, &
                               vreg, error_msg)
    end function emit_print

    logical function emit_real8_print_call(session, value, error_msg)
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: args(1)
        integer(c_int32_t) :: vreg

        args(1) = value
        emit_real8_print_call = emit_call(session, REAL8_PRINTER, args, &
                                          lr_type_void_s(session%handle), &
                                          1_c_int32_t, c_false, vreg, error_msg)
    end function emit_real8_print_call

    logical function synthesize_real4_printer(session, error_msg)
        ! Synthesizes .ffc.print_real4: void(f32). Implements gfortran
        ! list-directed real(4) output (9 significant digits):
        !   fixed (F) when ex in [-1, 8]: (8-ex) digits, %12s    field
        !   exponential otherwise: 1+8 digits, %16s field, %02d exponent
        type(liric_session_t), intent(inout) :: session
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: g_fe, g_ff, g_feo, g_padf, g_pade
        type(lr_operand_desc_t) :: g_inf, g_ninf, g_nan
        type(lr_operand_desc_t) :: x, xf64, tmp, num, p, ex
        integer(c_int32_t) :: finblk, nfblk
        integer(c_int32_t) :: chkhi, fblk, eblk, eneg, epos, eprint, exitb
        integer(c_int32_t) :: entry_block, param_vreg
        type(lr_error_t) :: error
        type(c_ptr) :: out_addr
        integer(c_int) :: status

        synthesize_real4_printer = .false.
        if (.not. require_open_session(session, error_msg)) return

        if (.not. declare_libc(session, error_msg)) return
        if (.not. create_cstring(session, '.ffc.r4.fe', '%.8e', g_fe, &
                                 error_msg)) return
        if (.not. create_cstring(session, '.ffc.r4.ff', '%#.*f', g_ff, &
                                 error_msg)) return
        if (.not. create_cstring(session, '.ffc.r4.feo', '%sE%c%02d', g_feo, &
                                 error_msg)) return
        if (.not. create_cstring(session, '.ffc.r4.padf', '%12s    ', g_padf, &
                                 error_msg)) return
        if (.not. create_cstring(session, '.ffc.r4.pade', '%16s', g_pade, &
                                 error_msg)) return
        if (.not. create_cstring(session, '.ffc.r4.inf', 'Infinity', g_inf, &
                                 error_msg)) return
        if (.not. create_cstring(session, '.ffc.r4.ninf', '-Infinity', g_ninf, &
                                 error_msg)) return
        if (.not. create_cstring(session, '.ffc.r4.nan', 'NaN', g_nan, &
                                 error_msg)) return

        if (.not. begin_real4_printer(session, param_vreg, entry_block, &
                                      error_msg)) return
        ! The f32 parameter must be extended to f64 before passing to
        ! snprintf because printf/snprintf is variadic and f32 is not
        ! automatically promoted by the LIRIC external ABI.
        x%kind = LR_OP_KIND_VREG
        x%payload = int(param_vreg, c_int64_t)
        x%typ = lr_type_f32_s(session%handle)
        x%global_offset = 0_c_int64_t

        finblk = create_liric_block(session)
        nfblk = create_liric_block(session)
        chkhi = create_liric_block(session)
        fblk = create_liric_block(session)
        eblk = create_liric_block(session)
        eneg = create_liric_block(session)
        epos = create_liric_block(session)
        eprint = create_liric_block(session)
        exitb = create_liric_block(session)

        if (.not. set_liric_block(session, entry_block, error_msg)) return

        ! Extend f32 to f64 so snprintf("%.8e") gets a double argument.
        if (.not. fpext_to_f64(session, x, xf64, error_msg)) return

        if (.not. build_head_r4(session, xf64, g_fe, tmp, num, p, finblk, &
                                 nfblk, error_msg)) return

        if (.not. set_liric_block(session, finblk, error_msg)) return
        if (.not. build_finite_r4(session, tmp, p, ex, chkhi, eblk, &
                                  error_msg)) return

        if (.not. set_liric_block(session, chkhi, error_msg)) return
        if (.not. branch_on_high_r4(session, ex, fblk, eblk, error_msg)) return

        if (.not. set_liric_block(session, fblk, error_msg)) return
        if (.not. build_fixed_r4(session, xf64, ex, num, g_ff, g_padf, exitb, &
                                  error_msg)) return

        if (.not. build_exponential_r4(session, ex, tmp, num, g_feo, g_pade, &
                                       eblk, eneg, epos, eprint, exitb, &
                                       error_msg)) return

        if (.not. set_liric_block(session, nfblk, error_msg)) return
        if (.not. build_nonfinite(session, tmp, g_inf, g_ninf, g_nan, g_pade, &
                                  exitb, error_msg)) return

        if (.not. set_liric_block(session, exitb, error_msg)) return
        if (.not. emit_ret_void_local(session, error_msg)) return

        call clear_liric_error(error)
        out_addr = c_null_ptr
        status = lr_session_func_end(session%handle, out_addr, error)
        if (.not. status_ok(status, error, error_msg)) return

        call set_empty(error_msg)
        synthesize_real4_printer = .true.
    end function synthesize_real4_printer

    logical function emit_real4_print_call(session, value, error_msg)
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: args(1)
        integer(c_int32_t) :: vreg

        args(1) = value
        emit_real4_print_call = emit_call(session, REAL4_PRINTER, args, &
                                          lr_type_void_s(session%handle), &
                                          1_c_int32_t, c_false, vreg, error_msg)
    end function emit_real4_print_call

    ! --- real4 printer helpers ---

    logical function fpext_to_f64(session, src, result, error_msg)
        ! Emit an FPEXT f32 -> f64 instruction.
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: src
        type(lr_operand_desc_t), intent(out) :: result
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t), target :: operands(1)
        type(lr_inst_desc_t) :: inst
        type(lr_error_t) :: error
        integer(c_int32_t) :: vreg

        fpext_to_f64 = .false.
        operands(1) = src
        inst%op = LR_OP_FPEXT
        inst%typ = lr_type_f64_s(session%handle)
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
        result%typ = lr_type_f64_s(session%handle)
        result%global_offset = 0_c_int64_t
        call set_empty(error_msg)
        fpext_to_f64 = .true.
    end function fpext_to_f64

    logical function begin_real4_printer(session, param_vreg, entry_block, &
                                         error_msg)
        type(liric_session_t), intent(inout) :: session
        integer(c_int32_t), intent(out) :: param_vreg
        integer(c_int32_t), intent(out) :: entry_block
        character(len=:), allocatable, intent(out) :: error_msg
        character(kind=c_char), allocatable :: c_name(:)
        type(c_ptr), target :: params(1)
        type(lr_error_t) :: error
        integer(c_int) :: status

        begin_real4_printer = .false.
        params(1) = lr_type_f32_s(session%handle)
        call clear_liric_error(error)
        call to_c_chars(REAL4_PRINTER, c_name)
        status = lr_session_func_begin(session%handle, c_name, &
                                       lr_type_void_s(session%handle), &
                                       c_loc(params), 1_c_int32_t, c_false, error)
        if (.not. status_ok(status, error, error_msg)) return

        param_vreg = lr_session_param(session%handle, 0_c_int32_t)
        entry_block = create_liric_block(session)
        call set_empty(error_msg)
        begin_real4_printer = .true.
    end function begin_real4_printer

    logical function build_head_r4(session, x, g_fe, tmp, num, p, finblk, &
                                   nfblk, error_msg)
        ! Same as build_head but with the r4 format string (%.8e).
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: x, g_fe
        type(lr_operand_desc_t), intent(out) :: tmp, num, p
        integer(c_int32_t), intent(in) :: finblk, nfblk
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: args(4), nullptr, cond
        integer(c_int32_t) :: vreg

        build_head_r4 = .false.
        if (.not. emit_alloca_bytes(session, i64_immediate(session, 36_c_int64_t), &
                                    tmp, error_msg)) return
        if (.not. emit_alloca_bytes(session, i64_immediate(session, 32_c_int64_t), &
                                    num, error_msg)) return

        args(1) = tmp
        args(2) = i64_immediate(session, 36_c_int64_t)
        args(3) = g_fe
        args(4) = x
        if (.not. emit_call(session, 'snprintf', args, lr_type_i32_s(session%handle), &
                            3_c_int32_t, c_true, vreg, error_msg)) return

        args(1) = tmp
        args(2) = i32_immediate(session, int(iachar('e'), c_int64_t))
        if (.not. emit_call(session, 'strchr', args(1:2), &
                            lr_type_ptr_s(session%handle), 2_c_int32_t, c_false, &
                            vreg, error_msg)) return
        p = ptr_vreg(session, vreg)

        nullptr%kind = LR_OP_KIND_IMM_I64
        nullptr%payload = 0_c_int64_t
        nullptr%typ = lr_type_ptr_s(session%handle)
        nullptr%global_offset = 0_c_int64_t
        if (.not. emit_liric_i32_icmp(session, LR_CMP_NE, p, nullptr, cond, &
                                      error_msg)) return
        if (.not. emit_liric_condbr(session, cond, finblk, nfblk, error_msg)) return
        build_head_r4 = .true.
    end function build_head_r4

    logical function build_finite_r4(session, tmp, p, ex, chkhi, eblk, error_msg)
        ! Same as build_finite; threshold check is done in branch_on_high_r4.
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: tmp, p
        type(lr_operand_desc_t), intent(out) :: ex
        integer(c_int32_t), intent(in) :: chkhi, eblk
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: pe1, args(1), cond
        integer(c_int32_t) :: vreg

        build_finite_r4 = .false.
        if (.not. gep_byte(session, p, 1_c_int64_t, pe1, error_msg)) return
        args(1) = pe1
        if (.not. emit_call(session, 'atoi', args(1:1), &
                            lr_type_i32_s(session%handle), 1_c_int32_t, c_false, &
                            vreg, error_msg)) return
        ex = i32_vreg(session, vreg)

        if (.not. store_zero_byte(session, p, error_msg)) return

        if (.not. emit_liric_i32_icmp(session, LR_CMP_SGE, ex, &
                i32_immediate(session, -1_c_int64_t), cond, error_msg)) return
        if (.not. emit_liric_condbr(session, cond, chkhi, eblk, error_msg)) return
        build_finite_r4 = .true.
    end function build_finite_r4

    logical function branch_on_high_r4(session, ex, fblk, eblk, error_msg)
        ! Fixed form for ex <= 8; otherwise exponential.
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: ex
        integer(c_int32_t), intent(in) :: fblk, eblk
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: cond

        branch_on_high_r4 = .false.
        if (.not. emit_liric_i32_icmp(session, LR_CMP_SLE, ex, &
                i32_immediate(session, 8_c_int64_t), cond, error_msg)) return
        if (.not. emit_liric_condbr(session, cond, fblk, eblk, error_msg)) return
        branch_on_high_r4 = .true.
    end function branch_on_high_r4

    logical function build_fixed_r4(session, x, ex, num, g_ff, g_padf, exitb, &
                                    error_msg)
        ! num = snprintf("%#.*f", 8 - ex, x); printf("%12s    ", num)
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: x, ex, num, g_ff, g_padf
        integer(c_int32_t), intent(in) :: exitb
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: dd, args(5)
        integer(c_int32_t) :: vreg

        build_fixed_r4 = .false.
        if (.not. emit_i32_binary(session, LR_OP_SUB, &
                i32_immediate(session, 8_c_int64_t), ex, dd, error_msg)) return

        args(1) = num
        args(2) = i64_immediate(session, 32_c_int64_t)
        args(3) = g_ff
        args(4) = dd
        args(5) = x
        if (.not. emit_call(session, 'snprintf', args, lr_type_i32_s(session%handle), &
                            3_c_int32_t, c_true, vreg, error_msg)) return

        if (.not. emit_print(session, g_padf, num, error_msg)) return
        if (.not. emit_liric_br(session, exitb, error_msg)) return
        build_fixed_r4 = .true.
    end function build_fixed_r4

    logical function build_exponential_r4(session, ex, tmp, num, g_feo, g_pade, &
                                          eblk, eneg, epos, eprint, exitb, error_msg)
        ! Two-digit exponent: %02d.
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: ex, tmp, num, g_feo, g_pade
        integer(c_int32_t), intent(in) :: eblk, eneg, epos, eprint, exitb
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: cond, negex

        build_exponential_r4 = .false.
        if (.not. set_liric_block(session, eblk, error_msg)) return
        if (.not. emit_liric_i32_icmp(session, LR_CMP_SLT, ex, &
                i32_immediate(session, 0_c_int64_t), cond, error_msg)) return
        if (.not. emit_liric_condbr(session, cond, eneg, epos, error_msg)) return

        if (.not. set_liric_block(session, eneg, error_msg)) return
        if (.not. emit_i32_binary(session, LR_OP_SUB, &
                i32_immediate(session, 0_c_int64_t), ex, negex, error_msg)) return
        if (.not. emit_exp_format_r4(session, num, g_feo, tmp, iachar('-'), &
                                     negex, error_msg)) return
        if (.not. emit_liric_br(session, eprint, error_msg)) return

        if (.not. set_liric_block(session, epos, error_msg)) return
        if (.not. emit_exp_format_r4(session, num, g_feo, tmp, iachar('+'), &
                                     ex, error_msg)) return
        if (.not. emit_liric_br(session, eprint, error_msg)) return

        if (.not. set_liric_block(session, eprint, error_msg)) return
        if (.not. emit_print(session, g_pade, num, error_msg)) return
        if (.not. emit_liric_br(session, exitb, error_msg)) return
        build_exponential_r4 = .true.
    end function build_exponential_r4

    logical function emit_exp_format_r4(session, num, g_feo, tmp, sign_char, &
                                        expval, error_msg)
        ! num = snprintf(num, 32, "%sE%c%02d", tmp, sign_char, expval)
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: num, g_feo, tmp, expval
        integer, intent(in) :: sign_char
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: args(6)
        integer(c_int32_t) :: vreg

        args(1) = num
        args(2) = i64_immediate(session, 32_c_int64_t)
        args(3) = g_feo
        args(4) = tmp
        args(5) = i32_immediate(session, int(sign_char, c_int64_t))
        args(6) = expval
        emit_exp_format_r4 = emit_call(session, 'snprintf', args, &
                                       lr_type_i32_s(session%handle), 3_c_int32_t, &
                                       c_true, vreg, error_msg)
    end function emit_exp_format_r4

    include 'liric_session_real_print_bindings_tail.inc'

end module liric_session_real_print_bindings
