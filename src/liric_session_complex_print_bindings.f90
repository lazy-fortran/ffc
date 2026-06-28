module liric_session_complex_print_bindings
    ! Synthesized runtime helpers for complex list-directed output:
    !   .ffc.print_c4  void(f32 re, f32 im)  -- gfortran complex(4) output
    !   .ffc.print_c8  void(f64 re, f64 im)  -- gfortran complex(8) output
    !
    ! gfortran complex(4) list-directed field width: 35 chars. The complex
    ! value "(re,im)" is right-justified inside that field; each component
    ! uses the same F/E format as standalone real(4) but WITHOUT the
    ! "%12s    " padding. Leading spaces fill to 35.
    !
    ! gfortran complex(8): field 53 chars; each component as real(8) without
    ! "%20s     " padding.
    !
    ! Implementation: synthesize .ffc.fmt_r4(f64 x, ptr buf) -> i32 (length)
    ! and .ffc.fmt_r8(f64 x, ptr buf) -> i32, then .ffc.print_c4 and
    ! .ffc.print_c8 call these for re and im, compute padding, and printf.
    use, intrinsic :: iso_c_binding, only: c_bool, c_char, c_int, c_int32_t, &
        c_int64_t, c_loc, c_null_char, &
        c_null_ptr, c_ptr, c_size_t
    use liric_session_bindings, only: liric_session_t, lr_error_t, &
        lr_inst_desc_t, lr_operand_desc_t, LR_OK, &
        LR_OP_CALL, LR_OP_TRUNC, LR_OP_FPEXT, &
        LR_OP_ADD, LR_OP_SUB, &
        LR_OP_STORE, LR_OP_GEP, LR_OP_RET, &
        LR_OP_RET_VOID, &
        LR_OP_KIND_VREG, LR_OP_KIND_IMM_I64, &
        LR_OP_KIND_GLOBAL, &
        lr_type_i32_s, lr_type_f32_s, lr_type_f64_s, &
        lr_type_i64_s, lr_type_ptr_s, lr_type_i8_s, &
        lr_type_array_s, lr_type_void_s, &
        lr_session_emit, lr_session_intern, &
        lr_session_global, lr_session_param, &
        status_ok, clear_liric_error, &
        set_empty, require_open_session, to_c_chars, &
        i32_vreg, i32_immediate
    use liric_session_memory_bindings, only: ptr_vreg, i64_vreg, i64_immediate, &
        emit_alloca_bytes, emit_i32_binary
    use liric_session_control_bindings, only: create_liric_block, set_liric_block, &
        emit_liric_br, emit_liric_condbr, &
        emit_liric_i32_icmp, &
        LR_CMP_NE, LR_CMP_SGE, &
        LR_CMP_SLE, LR_CMP_SLT
    implicit none
    private

    logical(c_bool), parameter :: c_false = .false.
    logical(c_bool), parameter :: c_true = .true.
    character(len=*), parameter :: FMT_R4_HELPER = '.ffc.fmt_r4'
    character(len=*), parameter :: FMT_R8_HELPER = '.ffc.fmt_r8'
    character(len=*), parameter :: COMPLEX4_PRINTER = '.ffc.print_c4'
    character(len=*), parameter :: COMPLEX8_PRINTER = '.ffc.print_c8'

    public :: synthesize_complex4_printer
    public :: synthesize_complex8_printer
    public :: emit_complex4_print_call
    public :: emit_complex8_print_call
    public :: emit_extern_i32_call

    interface
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
    end interface

contains

    ! -------------------------------------------------------------------------
    ! Shared low-level helpers

    logical function emit_extern_i32_call(session, name, args, result, error_msg)
        ! Call an external C function returning int via the external ABI and
        ! capture its result. emit_i32_call uses the internal ABI, whose return
        ! register is not valid for libc calls (e.g. feof); this one is.
        type(liric_session_t), intent(inout) :: session
        character(len=*), intent(in) :: name
        type(lr_operand_desc_t), intent(in) :: args(:)
        type(lr_operand_desc_t), intent(out) :: result
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int32_t) :: vreg

        emit_extern_i32_call = cx_emit_call(session, name, args, &
            lr_type_i32_s(session%handle), &
            0_c_int32_t, c_false, &
            vreg, error_msg)
        if (.not. emit_extern_i32_call) return
        result = i32_vreg(session, vreg)
    end function emit_extern_i32_call

    logical function cx_emit_call(session, name, args, ret_typ, fixed_args, &
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

        cx_emit_call = .false.
        call to_c_chars(name, c_name)
        sym_id = lr_session_intern(session%handle, c_name)
        if (sym_id < 0_c_int32_t) then
            error_msg = 'cx_emit_call: could not intern '//trim(name)
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
        cx_emit_call = .true.
    end function cx_emit_call

    logical function cx_emit_cast(session, op, src, dst_typ, result, error_msg)
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

        cx_emit_cast = .false.
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
        call set_empty(error_msg)
        cx_emit_cast = .true.
    end function cx_emit_cast

    logical function cx_gep_byte(session, base, offset, result, error_msg)
        ! getelementptr i8, base, i64(offset) — byte stride
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: base
        integer(c_int64_t), intent(in) :: offset
        type(lr_operand_desc_t), intent(out) :: result
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t), target :: operands(2)
        type(lr_inst_desc_t) :: inst
        type(lr_error_t) :: error
        integer(c_int32_t) :: vreg

        cx_gep_byte = .false.
        operands(1) = base
        operands(2)%kind = LR_OP_KIND_IMM_I64
        operands(2)%payload = offset
        operands(2)%typ = lr_type_i64_s(session%handle)
        operands(2)%global_offset = 0_c_int64_t
        inst%op = LR_OP_GEP
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
        call set_empty(error_msg)
        cx_gep_byte = .true.
    end function cx_gep_byte

    logical function cx_store_zero_byte(session, addr, error_msg)
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: addr
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t), target :: operands(2)
        type(lr_inst_desc_t) :: inst
        type(lr_error_t) :: error
        integer(c_int32_t) :: vreg

        cx_store_zero_byte = .false.
        operands(1)%kind = LR_OP_KIND_IMM_I64
        operands(1)%payload = 0_c_int64_t
        operands(1)%typ = lr_type_i8_s(session%handle)
        operands(1)%global_offset = 0_c_int64_t
        operands(2) = addr
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
        vreg = lr_session_emit(session%handle, inst, error)
        if (.not. status_ok(error%code, error, error_msg)) return
        call set_empty(error_msg)
        cx_store_zero_byte = .true.
    end function cx_store_zero_byte

    logical function cx_ret_i32(session, value, error_msg)
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t), target :: operands(1)
        type(lr_inst_desc_t) :: inst
        type(lr_error_t) :: error
        integer(c_int32_t) :: vreg

        cx_ret_i32 = .false.
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
        call set_empty(error_msg)
        cx_ret_i32 = .true.
    end function cx_ret_i32

    logical function cx_ret_void(session, error_msg)
        type(liric_session_t), intent(inout) :: session
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_inst_desc_t) :: inst
        type(lr_error_t) :: error
        integer(c_int32_t) :: vreg

        cx_ret_void = .false.
        inst%op = LR_OP_RET_VOID
        inst%typ = c_null_ptr
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
        if (.not. status_ok(error%code, error, error_msg)) return
        call set_empty(error_msg)
        cx_ret_void = .true.
    end function cx_ret_void

    logical function cx_create_cstring(session, name, text, operand, error_msg)
        type(liric_session_t), intent(inout) :: session
        character(len=*), intent(in) :: name, text
        type(lr_operand_desc_t), intent(out) :: operand
        character(len=:), allocatable, intent(out) :: error_msg
        character(kind=c_char), allocatable, target :: bytes(:)
        character(kind=c_char), allocatable :: c_name(:)
        type(c_ptr) :: array_type
        integer(c_int32_t) :: global_id
        integer :: i, n

        cx_create_cstring = .false.
        n = len(text) + 1
        allocate (bytes(n))
        do i = 1, len(text)
            bytes(i) = text(i:i)
        end do
        bytes(n) = c_null_char
        array_type = lr_type_array_s(session%handle, lr_type_i8_s(session%handle), &
            int(n, c_int64_t))
        call to_c_chars(name, c_name)
        global_id = lr_session_global(session%handle, c_name, array_type, c_true, &
            c_loc(bytes), int(n, c_size_t))
        if (global_id < 0_c_int32_t) then
            error_msg = 'cx: cannot create string global '//trim(name)
            return
        end if
        global_id = lr_session_intern(session%handle, c_name)
        if (global_id < 0_c_int32_t) then
            error_msg = 'cx: cannot intern string global '//trim(name)
            return
        end if
        operand%kind = LR_OP_KIND_GLOBAL
        operand%payload = int(global_id, c_int64_t)
        operand%typ = lr_type_ptr_s(session%handle)
        operand%global_offset = 0_c_int64_t
        call set_empty(error_msg)
        cx_create_cstring = .true.
    end function cx_create_cstring

    logical function cx_declare_one(session, name, ret, params_ptr, n, vararg, &
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

        cx_declare_one = .false.
        call clear_liric_error(error)
        call to_c_chars(name, c_name)
        status = lr_session_declare(session%handle, c_name, ret, params_ptr, n, &
            vararg, error)
        if (.not. status_ok(status, error, error_msg)) return
        cx_declare_one = .true.
    end function cx_declare_one

    logical function cx_declare_libc(session, error_msg)
        type(liric_session_t), intent(inout) :: session
        character(len=:), allocatable, intent(out) :: error_msg
        type(c_ptr), target :: p3(3), p2(2), p1(1)

        cx_declare_libc = .false.
        p3(1) = lr_type_ptr_s(session%handle)
        p3(2) = lr_type_i64_s(session%handle)
        p3(3) = lr_type_ptr_s(session%handle)
        if (.not. cx_declare_one(session, 'snprintf', lr_type_i32_s(session%handle), &
            c_loc(p3), 3_c_int32_t, c_true, error_msg)) return
        p1(1) = lr_type_ptr_s(session%handle)
        if (.not. cx_declare_one(session, 'printf', lr_type_i32_s(session%handle), &
            c_loc(p1), 1_c_int32_t, c_true, error_msg)) return
        if (.not. cx_declare_one(session, 'strlen', lr_type_i64_s(session%handle), &
            c_loc(p1), 1_c_int32_t, c_false, error_msg)) return
        if (.not. cx_declare_one(session, 'atoi', lr_type_i32_s(session%handle), &
            c_loc(p1), 1_c_int32_t, c_false, error_msg)) return
        p2(1) = lr_type_ptr_s(session%handle)
        p2(2) = lr_type_i32_s(session%handle)
        if (.not. cx_declare_one(session, 'strchr', lr_type_ptr_s(session%handle), &
            c_loc(p2), 2_c_int32_t, c_false, error_msg)) return
        cx_declare_libc = .true.
    end function cx_declare_libc

    function cx_typed_param(session, index, typ) result(operand)
        type(liric_session_t), intent(in) :: session
        integer(c_int32_t), intent(in) :: index
        type(c_ptr), intent(in) :: typ
        type(lr_operand_desc_t) :: operand

        operand%kind = LR_OP_KIND_VREG
        operand%payload = int(lr_session_param(session%handle, index), c_int64_t)
        operand%typ = typ
        operand%global_offset = 0_c_int64_t
    end function cx_typed_param

    ! -------------------------------------------------------------------------
    ! .ffc.fmt_r4(f64 x, ptr buf) -> i32
    ! Writes a real(4)-formatted C string to buf; returns its length.
    ! Same F/E decision as standalone real(4): exponent in [-1, 8] -> fixed,
    ! otherwise exponential. No leading-field padding is added.

    logical function synthesize_fmt_r4_helper(session, error_msg)
        type(liric_session_t), intent(inout) :: session
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: g_fe, g_ff, g_feo
        type(lr_operand_desc_t) :: x, buf, tmp, num, p, ex, prec, negex
        type(lr_operand_desc_t) :: pe1, cond, slen, nullptr, args(6)
        integer(c_int32_t) :: entry_blk, finblk, nfblk, chkhi
        integer(c_int32_t) :: fblk, eblk, eneg, epos, eprint, exitb
        type(c_ptr), target :: params(2)
        type(c_ptr) :: out_addr
        character(kind=c_char), allocatable :: c_name(:)
        type(lr_error_t) :: error
        integer(c_int32_t) :: vreg
        integer(c_int) :: status

        synthesize_fmt_r4_helper = .false.
        if (.not. require_open_session(session, error_msg)) return

        ! gfortran complex(4) exponential form carries 9 fraction digits
        ! (d.dddddddddE+nn, 9 significant). %.9e matches; %.8e drops one.
        if (.not. cx_create_cstring(session, '.ffc.c4.fe', '%.9e', g_fe, &
            error_msg)) return
        if (.not. cx_create_cstring(session, '.ffc.c4.ff', '%#.*f', g_ff, &
            error_msg)) return
        if (.not. cx_create_cstring(session, '.ffc.c4.feo', '%sE%c%02d', g_feo, &
            error_msg)) return

        params(1) = lr_type_f64_s(session%handle)
        params(2) = lr_type_ptr_s(session%handle)
        call clear_liric_error(error)
        call to_c_chars(FMT_R4_HELPER, c_name)
        status = lr_session_func_begin(session%handle, c_name, &
            lr_type_i32_s(session%handle), &
            c_loc(params), 2_c_int32_t, c_false, error)
        if (.not. status_ok(status, error, error_msg)) return

        x = cx_typed_param(session, 0_c_int32_t, lr_type_f64_s(session%handle))
        buf = cx_typed_param(session, 1_c_int32_t, lr_type_ptr_s(session%handle))

        entry_blk = create_liric_block(session)
        finblk = create_liric_block(session)
        nfblk = create_liric_block(session)
        chkhi = create_liric_block(session)
        fblk = create_liric_block(session)
        eblk = create_liric_block(session)
        eneg = create_liric_block(session)
        epos = create_liric_block(session)
        eprint = create_liric_block(session)
        exitb = create_liric_block(session)

        ! entry: tmp = alloca 36; num = alloca 32; snprintf(tmp, 36, "%.8e", x)
        if (.not. set_liric_block(session, entry_blk, error_msg)) return
        if (.not. emit_alloca_bytes(session, i64_immediate(session, 36_c_int64_t), &
            tmp, error_msg)) return
        if (.not. emit_alloca_bytes(session, i64_immediate(session, 32_c_int64_t), &
            num, error_msg)) return
        args(1) = tmp
        args(2) = i64_immediate(session, 36_c_int64_t)
        args(3) = g_fe
        args(4) = x
        if (.not. cx_emit_call(session, 'snprintf', args(1:4), &
            lr_type_i32_s(session%handle), 3_c_int32_t, &
            c_true, vreg, error_msg)) return

        ! p = strchr(tmp, 'e')
        args(1) = tmp
        args(2) = i32_immediate(session, int(iachar('e'), c_int64_t))
        if (.not. cx_emit_call(session, 'strchr', args(1:2), &
            lr_type_ptr_s(session%handle), 2_c_int32_t, &
            c_false, vreg, error_msg)) return
        p = ptr_vreg(session, vreg)

        nullptr%kind = LR_OP_KIND_IMM_I64
        nullptr%payload = 0_c_int64_t
        nullptr%typ = lr_type_ptr_s(session%handle)
        nullptr%global_offset = 0_c_int64_t
        if (.not. emit_liric_i32_icmp(session, LR_CMP_NE, p, nullptr, cond, &
            error_msg)) return
        if (.not. emit_liric_condbr(session, cond, finblk, nfblk, error_msg)) return

        ! finblk: pe1 = p+1; ex = atoi(pe1); *p = '\0'
        if (.not. set_liric_block(session, finblk, error_msg)) return
        if (.not. cx_gep_byte(session, p, 1_c_int64_t, pe1, error_msg)) return
        args(1) = pe1
        if (.not. cx_emit_call(session, 'atoi', args(1:1), &
            lr_type_i32_s(session%handle), 1_c_int32_t, &
            c_false, vreg, error_msg)) return
        ex = i32_vreg(session, vreg)
        if (.not. cx_store_zero_byte(session, p, error_msg)) return

        if (.not. emit_liric_i32_icmp(session, LR_CMP_SGE, ex, &
            i32_immediate(session, -1_c_int64_t), cond, error_msg)) return
        if (.not. emit_liric_condbr(session, cond, chkhi, eblk, error_msg)) return

        if (.not. set_liric_block(session, chkhi, error_msg)) return
        if (.not. emit_liric_i32_icmp(session, LR_CMP_SLE, ex, &
            i32_immediate(session, 8_c_int64_t), cond, error_msg)) return
        if (.not. emit_liric_condbr(session, cond, fblk, eblk, error_msg)) return

        ! fblk: buf = snprintf("%#.*f", 8-ex, x)
        if (.not. set_liric_block(session, fblk, error_msg)) return
        if (.not. emit_i32_binary(session, LR_OP_SUB, &
            i32_immediate(session, 8_c_int64_t), ex, prec, error_msg)) return
        args(1) = buf
        args(2) = i64_immediate(session, 32_c_int64_t)
        args(3) = g_ff
        args(4) = prec
        args(5) = x
        if (.not. cx_emit_call(session, 'snprintf', args(1:5), &
            lr_type_i32_s(session%handle), 3_c_int32_t, &
            c_true, vreg, error_msg)) return
        if (.not. emit_liric_br(session, exitb, error_msg)) return

        ! eblk: exponential. eneg: negative exponent; epos: positive.
        if (.not. set_liric_block(session, eblk, error_msg)) return
        if (.not. emit_liric_i32_icmp(session, LR_CMP_SLT, ex, &
            i32_immediate(session, 0_c_int64_t), cond, error_msg)) return
        if (.not. emit_liric_condbr(session, cond, eneg, epos, error_msg)) return

        if (.not. set_liric_block(session, eneg, error_msg)) return
        if (.not. emit_i32_binary(session, LR_OP_SUB, &
            i32_immediate(session, 0_c_int64_t), ex, negex, error_msg)) return
        args(1) = num
        args(2) = i64_immediate(session, 32_c_int64_t)
        args(3) = g_feo
        args(4) = tmp
        args(5) = i32_immediate(session, int(iachar('-'), c_int64_t))
        args(6) = negex
        if (.not. cx_emit_call(session, 'snprintf', args, &
            lr_type_i32_s(session%handle), 3_c_int32_t, &
            c_true, vreg, error_msg)) return
        if (.not. emit_liric_br(session, eprint, error_msg)) return

        if (.not. set_liric_block(session, epos, error_msg)) return
        args(5) = i32_immediate(session, int(iachar('+'), c_int64_t))
        args(6) = ex
        if (.not. cx_emit_call(session, 'snprintf', args, &
            lr_type_i32_s(session%handle), 3_c_int32_t, &
            c_true, vreg, error_msg)) return
        if (.not. emit_liric_br(session, eprint, error_msg)) return

        ! eprint: copy num -> buf
        if (.not. set_liric_block(session, eprint, error_msg)) return
        args(1) = buf
        args(2) = i64_immediate(session, 32_c_int64_t)
        args(3) = num
        if (.not. cx_emit_call(session, 'snprintf', args(1:3), &
            lr_type_i32_s(session%handle), 3_c_int32_t, &
            c_false, vreg, error_msg)) return
        if (.not. emit_liric_br(session, exitb, error_msg)) return

        ! nfblk: non-finite (Inf/NaN). Copy tmp -> buf.
        if (.not. set_liric_block(session, nfblk, error_msg)) return
        args(1) = buf
        args(2) = i64_immediate(session, 32_c_int64_t)
        args(3) = tmp
        if (.not. cx_emit_call(session, 'snprintf', args(1:3), &
            lr_type_i32_s(session%handle), 3_c_int32_t, &
            c_false, vreg, error_msg)) return
        if (.not. emit_liric_br(session, exitb, error_msg)) return

        ! exitb: return (i32) strlen(buf)
        if (.not. set_liric_block(session, exitb, error_msg)) return
        args(1) = buf
        if (.not. cx_emit_call(session, 'strlen', args(1:1), &
            lr_type_i64_s(session%handle), 1_c_int32_t, &
            c_false, vreg, error_msg)) return
        if (.not. cx_emit_cast(session, LR_OP_TRUNC, i64_vreg(session, vreg), &
            lr_type_i32_s(session%handle), slen, error_msg)) return
        if (.not. cx_ret_i32(session, slen, error_msg)) return

        call clear_liric_error(error)
        out_addr = c_null_ptr
        status = lr_session_func_end(session%handle, out_addr, error)
        if (.not. status_ok(status, error, error_msg)) return
        call set_empty(error_msg)
        synthesize_fmt_r4_helper = .true.
    end function synthesize_fmt_r4_helper

    ! -------------------------------------------------------------------------
    ! .ffc.fmt_r8(f64 x, ptr buf) -> i32: same but for real(8).
    ! Exponent range for fixed notation: [-1, 16]; format uses "%.16e".

    logical function synthesize_fmt_r8_helper(session, error_msg)
        type(liric_session_t), intent(inout) :: session
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: g_fe, g_ff, g_feo
        type(lr_operand_desc_t) :: x, buf, tmp, num, p, ex, prec, negex
        type(lr_operand_desc_t) :: pe1, cond, slen, nullptr, args(6)
        integer(c_int32_t) :: entry_blk, finblk, nfblk, chkhi
        integer(c_int32_t) :: fblk, eblk, eneg, epos, eprint, exitb
        type(c_ptr), target :: params(2)
        type(c_ptr) :: out_addr
        character(kind=c_char), allocatable :: c_name(:)
        type(lr_error_t) :: error
        integer(c_int32_t) :: vreg
        integer(c_int) :: status

        synthesize_fmt_r8_helper = .false.
        if (.not. require_open_session(session, error_msg)) return

        if (.not. cx_create_cstring(session, '.ffc.c8.fe', '%.17e', g_fe, &
            error_msg)) return
        if (.not. cx_create_cstring(session, '.ffc.c8.ff', '%#.*f', g_ff, &
            error_msg)) return
        if (.not. cx_create_cstring(session, '.ffc.c8.feo', '%sE%c%03d', g_feo, &
            error_msg)) return

        params(1) = lr_type_f64_s(session%handle)
        params(2) = lr_type_ptr_s(session%handle)
        call clear_liric_error(error)
        call to_c_chars(FMT_R8_HELPER, c_name)
        status = lr_session_func_begin(session%handle, c_name, &
            lr_type_i32_s(session%handle), &
            c_loc(params), 2_c_int32_t, c_false, error)
        if (.not. status_ok(status, error, error_msg)) return

        x = cx_typed_param(session, 0_c_int32_t, lr_type_f64_s(session%handle))
        buf = cx_typed_param(session, 1_c_int32_t, lr_type_ptr_s(session%handle))

        entry_blk = create_liric_block(session)
        finblk = create_liric_block(session)
        nfblk = create_liric_block(session)
        chkhi = create_liric_block(session)
        fblk = create_liric_block(session)
        eblk = create_liric_block(session)
        eneg = create_liric_block(session)
        epos = create_liric_block(session)
        eprint = create_liric_block(session)
        exitb = create_liric_block(session)

        if (.not. set_liric_block(session, entry_blk, error_msg)) return
        if (.not. emit_alloca_bytes(session, i64_immediate(session, 48_c_int64_t), &
            tmp, error_msg)) return
        if (.not. emit_alloca_bytes(session, i64_immediate(session, 48_c_int64_t), &
            num, error_msg)) return
        args(1) = tmp
        args(2) = i64_immediate(session, 48_c_int64_t)
        args(3) = g_fe
        args(4) = x
        if (.not. cx_emit_call(session, 'snprintf', args(1:4), &
            lr_type_i32_s(session%handle), 3_c_int32_t, &
            c_true, vreg, error_msg)) return

        args(1) = tmp
        args(2) = i32_immediate(session, int(iachar('e'), c_int64_t))
        if (.not. cx_emit_call(session, 'strchr', args(1:2), &
            lr_type_ptr_s(session%handle), 2_c_int32_t, &
            c_false, vreg, error_msg)) return
        p = ptr_vreg(session, vreg)

        nullptr%kind = LR_OP_KIND_IMM_I64
        nullptr%payload = 0_c_int64_t
        nullptr%typ = lr_type_ptr_s(session%handle)
        nullptr%global_offset = 0_c_int64_t
        if (.not. emit_liric_i32_icmp(session, LR_CMP_NE, p, nullptr, cond, &
            error_msg)) return
        if (.not. emit_liric_condbr(session, cond, finblk, nfblk, error_msg)) return

        if (.not. set_liric_block(session, finblk, error_msg)) return
        if (.not. cx_gep_byte(session, p, 1_c_int64_t, pe1, error_msg)) return
        args(1) = pe1
        if (.not. cx_emit_call(session, 'atoi', args(1:1), &
            lr_type_i32_s(session%handle), 1_c_int32_t, &
            c_false, vreg, error_msg)) return
        ex = i32_vreg(session, vreg)
        if (.not. cx_store_zero_byte(session, p, error_msg)) return

        if (.not. emit_liric_i32_icmp(session, LR_CMP_SGE, ex, &
            i32_immediate(session, -1_c_int64_t), cond, error_msg)) return
        if (.not. emit_liric_condbr(session, cond, chkhi, eblk, error_msg)) return

        if (.not. set_liric_block(session, chkhi, error_msg)) return
        if (.not. emit_liric_i32_icmp(session, LR_CMP_SLE, ex, &
            i32_immediate(session, 16_c_int64_t), cond, error_msg)) return
        if (.not. emit_liric_condbr(session, cond, fblk, eblk, error_msg)) return

        if (.not. set_liric_block(session, fblk, error_msg)) return
        if (.not. emit_i32_binary(session, LR_OP_SUB, &
            i32_immediate(session, 16_c_int64_t), ex, prec, error_msg)) return
        args(1) = buf
        args(2) = i64_immediate(session, 48_c_int64_t)
        args(3) = g_ff
        args(4) = prec
        args(5) = x
        if (.not. cx_emit_call(session, 'snprintf', args(1:5), &
            lr_type_i32_s(session%handle), 3_c_int32_t, &
            c_true, vreg, error_msg)) return
        if (.not. emit_liric_br(session, exitb, error_msg)) return

        if (.not. set_liric_block(session, eblk, error_msg)) return
        if (.not. emit_liric_i32_icmp(session, LR_CMP_SLT, ex, &
            i32_immediate(session, 0_c_int64_t), cond, error_msg)) return
        if (.not. emit_liric_condbr(session, cond, eneg, epos, error_msg)) return

        if (.not. set_liric_block(session, eneg, error_msg)) return
        if (.not. emit_i32_binary(session, LR_OP_SUB, &
            i32_immediate(session, 0_c_int64_t), ex, negex, error_msg)) return
        args(1) = num
        args(2) = i64_immediate(session, 48_c_int64_t)
        args(3) = g_feo
        args(4) = tmp
        args(5) = i32_immediate(session, int(iachar('-'), c_int64_t))
        args(6) = negex
        if (.not. cx_emit_call(session, 'snprintf', args, &
            lr_type_i32_s(session%handle), 3_c_int32_t, &
            c_true, vreg, error_msg)) return
        if (.not. emit_liric_br(session, eprint, error_msg)) return

        if (.not. set_liric_block(session, epos, error_msg)) return
        args(5) = i32_immediate(session, int(iachar('+'), c_int64_t))
        args(6) = ex
        if (.not. cx_emit_call(session, 'snprintf', args, &
            lr_type_i32_s(session%handle), 3_c_int32_t, &
            c_true, vreg, error_msg)) return
        if (.not. emit_liric_br(session, eprint, error_msg)) return

        if (.not. set_liric_block(session, eprint, error_msg)) return
        args(1) = buf
        args(2) = i64_immediate(session, 48_c_int64_t)
        args(3) = num
        if (.not. cx_emit_call(session, 'snprintf', args(1:3), &
            lr_type_i32_s(session%handle), 3_c_int32_t, &
            c_false, vreg, error_msg)) return
        if (.not. emit_liric_br(session, exitb, error_msg)) return

        if (.not. set_liric_block(session, nfblk, error_msg)) return
        args(1) = buf
        args(2) = i64_immediate(session, 48_c_int64_t)
        args(3) = tmp
        if (.not. cx_emit_call(session, 'snprintf', args(1:3), &
            lr_type_i32_s(session%handle), 3_c_int32_t, &
            c_false, vreg, error_msg)) return
        if (.not. emit_liric_br(session, exitb, error_msg)) return

        if (.not. set_liric_block(session, exitb, error_msg)) return
        args(1) = buf
        if (.not. cx_emit_call(session, 'strlen', args(1:1), &
            lr_type_i64_s(session%handle), 1_c_int32_t, &
            c_false, vreg, error_msg)) return
        if (.not. cx_emit_cast(session, LR_OP_TRUNC, i64_vreg(session, vreg), &
            lr_type_i32_s(session%handle), slen, error_msg)) return
        if (.not. cx_ret_i32(session, slen, error_msg)) return

        call clear_liric_error(error)
        out_addr = c_null_ptr
        status = lr_session_func_end(session%handle, out_addr, error)
        if (.not. status_ok(status, error, error_msg)) return
        call set_empty(error_msg)
        synthesize_fmt_r8_helper = .true.
    end function synthesize_fmt_r8_helper

    ! -------------------------------------------------------------------------
    ! .ffc.print_c4(f32 re, f32 im): gfortran complex(4) list-directed output.
    ! Field width = 35. Each component: up to 16 chars -> pad = 32 - W1 - W2.

    logical function synthesize_complex4_printer(session, error_msg)
        type(liric_session_t), intent(inout) :: session
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: re, im, re_f64, im_f64, re_buf, im_buf
        type(lr_operand_desc_t) :: w1, w2, wsum, pad, empty, g_pf
        type(lr_operand_desc_t) :: args(5)
        type(c_ptr), target :: params(2)
        type(c_ptr) :: out_addr
        character(kind=c_char), allocatable :: c_name(:)
        type(lr_error_t) :: error
        integer(c_int32_t) :: vreg, entry_blk
        integer(c_int) :: status

        synthesize_complex4_printer = .false.
        if (.not. require_open_session(session, error_msg)) return

        if (.not. cx_declare_libc(session, error_msg)) return
        if (.not. synthesize_fmt_r4_helper(session, error_msg)) return

        if (.not. cx_create_cstring(session, '.ffc.c4.pf', '%*s(%s,%s)', g_pf, &
            error_msg)) return
        if (.not. cx_create_cstring(session, '.ffc.c4.es', '', empty, &
            error_msg)) return

        params(1) = lr_type_f32_s(session%handle)
        params(2) = lr_type_f32_s(session%handle)
        call clear_liric_error(error)
        call to_c_chars(COMPLEX4_PRINTER, c_name)
        status = lr_session_func_begin(session%handle, c_name, &
            lr_type_void_s(session%handle), &
            c_loc(params), 2_c_int32_t, c_false, error)
        if (.not. status_ok(status, error, error_msg)) return

        re = cx_typed_param(session, 0_c_int32_t, lr_type_f32_s(session%handle))
        im = cx_typed_param(session, 1_c_int32_t, lr_type_f32_s(session%handle))

        entry_blk = create_liric_block(session)
        if (.not. set_liric_block(session, entry_blk, error_msg)) return

        ! Extend f32 params to f64 for fmt_r4 (takes f64)
        if (.not. cx_emit_cast(session, LR_OP_FPEXT, re, &
            lr_type_f64_s(session%handle), re_f64, error_msg)) return
        if (.not. cx_emit_cast(session, LR_OP_FPEXT, im, &
            lr_type_f64_s(session%handle), im_f64, error_msg)) return

        if (.not. emit_alloca_bytes(session, i64_immediate(session, 36_c_int64_t), &
            re_buf, error_msg)) return
        if (.not. emit_alloca_bytes(session, i64_immediate(session, 36_c_int64_t), &
            im_buf, error_msg)) return

        ! w1 = .ffc.fmt_r4(re_f64, re_buf)
        args(1) = re_f64
        args(2) = re_buf
        if (.not. cx_emit_call(session, FMT_R4_HELPER, args(1:2), &
            lr_type_i32_s(session%handle), 2_c_int32_t, &
            c_false, vreg, error_msg)) return
        w1 = i32_vreg(session, vreg)

        ! w2 = .ffc.fmt_r4(im_f64, im_buf)
        args(1) = im_f64
        args(2) = im_buf
        if (.not. cx_emit_call(session, FMT_R4_HELPER, args(1:2), &
            lr_type_i32_s(session%handle), 2_c_int32_t, &
            c_false, vreg, error_msg)) return
        w2 = i32_vreg(session, vreg)

        ! pad = 32 - w1 - w2  (field 35, parens/comma occupy 3 chars)
        if (.not. emit_i32_binary(session, LR_OP_ADD, w1, w2, wsum, &
            error_msg)) return
        if (.not. emit_i32_binary(session, LR_OP_SUB, &
            i32_immediate(session, 32_c_int64_t), wsum, pad, error_msg)) return

        ! printf("%*s(%s,%s)", pad, "", re_buf, im_buf)
        args(1) = g_pf
        args(2) = pad
        args(3) = empty
        args(4) = re_buf
        args(5) = im_buf
        if (.not. cx_emit_call(session, 'printf', args, lr_type_i32_s(session%handle), &
            1_c_int32_t, c_true, vreg, error_msg)) return

        if (.not. cx_ret_void(session, error_msg)) return

        call clear_liric_error(error)
        out_addr = c_null_ptr
        status = lr_session_func_end(session%handle, out_addr, error)
        if (.not. status_ok(status, error, error_msg)) return
        call set_empty(error_msg)
        synthesize_complex4_printer = .true.
    end function synthesize_complex4_printer

    ! -------------------------------------------------------------------------
    ! .ffc.print_c8(f64 re, f64 im): gfortran complex(8) list-directed output.
    ! Field width = 53. pad = 50 - w1 - w2.

    logical function synthesize_complex8_printer(session, error_msg)
        type(liric_session_t), intent(inout) :: session
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: re, im, re_buf, im_buf
        type(lr_operand_desc_t) :: w1, w2, wsum, pad, empty, g_pf
        type(lr_operand_desc_t) :: args(5)
        type(c_ptr), target :: params(2)
        type(c_ptr) :: out_addr
        character(kind=c_char), allocatable :: c_name(:)
        type(lr_error_t) :: error
        integer(c_int32_t) :: vreg, entry_blk
        integer(c_int) :: status

        synthesize_complex8_printer = .false.
        if (.not. require_open_session(session, error_msg)) return

        if (.not. cx_declare_libc(session, error_msg)) return
        if (.not. synthesize_fmt_r8_helper(session, error_msg)) return

        if (.not. cx_create_cstring(session, '.ffc.c8.pf', '%*s(%s,%s)', g_pf, &
            error_msg)) return
        if (.not. cx_create_cstring(session, '.ffc.c8.es', '', empty, &
            error_msg)) return

        params(1) = lr_type_f64_s(session%handle)
        params(2) = lr_type_f64_s(session%handle)
        call clear_liric_error(error)
        call to_c_chars(COMPLEX8_PRINTER, c_name)
        status = lr_session_func_begin(session%handle, c_name, &
            lr_type_void_s(session%handle), &
            c_loc(params), 2_c_int32_t, c_false, error)
        if (.not. status_ok(status, error, error_msg)) return

        re = cx_typed_param(session, 0_c_int32_t, lr_type_f64_s(session%handle))
        im = cx_typed_param(session, 1_c_int32_t, lr_type_f64_s(session%handle))

        entry_blk = create_liric_block(session)
        if (.not. set_liric_block(session, entry_blk, error_msg)) return

        if (.not. emit_alloca_bytes(session, i64_immediate(session, 48_c_int64_t), &
            re_buf, error_msg)) return
        if (.not. emit_alloca_bytes(session, i64_immediate(session, 48_c_int64_t), &
            im_buf, error_msg)) return

        args(1) = re
        args(2) = re_buf
        if (.not. cx_emit_call(session, FMT_R8_HELPER, args(1:2), &
            lr_type_i32_s(session%handle), 2_c_int32_t, &
            c_false, vreg, error_msg)) return
        w1 = i32_vreg(session, vreg)

        args(1) = im
        args(2) = im_buf
        if (.not. cx_emit_call(session, FMT_R8_HELPER, args(1:2), &
            lr_type_i32_s(session%handle), 2_c_int32_t, &
            c_false, vreg, error_msg)) return
        w2 = i32_vreg(session, vreg)

        ! pad = 50 - w1 - w2  (field 53, parens/comma = 3)
        if (.not. emit_i32_binary(session, LR_OP_ADD, w1, w2, wsum, &
            error_msg)) return
        if (.not. emit_i32_binary(session, LR_OP_SUB, &
            i32_immediate(session, 50_c_int64_t), wsum, pad, error_msg)) return

        args(1) = g_pf
        args(2) = pad
        args(3) = empty
        args(4) = re_buf
        args(5) = im_buf
        if (.not. cx_emit_call(session, 'printf', args, lr_type_i32_s(session%handle), &
            1_c_int32_t, c_true, vreg, error_msg)) return

        if (.not. cx_ret_void(session, error_msg)) return

        call clear_liric_error(error)
        out_addr = c_null_ptr
        status = lr_session_func_end(session%handle, out_addr, error)
        if (.not. status_ok(status, error, error_msg)) return
        call set_empty(error_msg)
        synthesize_complex8_printer = .true.
    end function synthesize_complex8_printer

    ! -------------------------------------------------------------------------
    ! Emit calls to the complex printers from the main program.

    logical function emit_complex4_print_call(session, re, im, error_msg)
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: re, im
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: args(2)
        integer(c_int32_t) :: vreg

        args(1) = re
        args(2) = im
        emit_complex4_print_call = cx_emit_call(session, COMPLEX4_PRINTER, args, &
            lr_type_void_s(session%handle), &
            2_c_int32_t, c_false, vreg, error_msg)
    end function emit_complex4_print_call

    logical function emit_complex8_print_call(session, re, im, error_msg)
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: re, im
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: args(2)
        integer(c_int32_t) :: vreg

        args(1) = re
        args(2) = im
        emit_complex8_print_call = cx_emit_call(session, COMPLEX8_PRINTER, args, &
            lr_type_void_s(session%handle), &
            2_c_int32_t, c_false, vreg, error_msg)
    end function emit_complex8_print_call

end module liric_session_complex_print_bindings
