    logical function synthesize_get_arg_helper(session, error_msg)
        ! void .ffc.get_arg(i32 index, ptr argv, ptr dest, i64 destlen):
        !   snprintf(tmp[destlen+1], "%-*.*s", destlen, destlen, argv[index]);
        !   memcpy(dest, tmp, destlen)
        ! "%-*.*s" left-justifies, pads with blanks to destlen, and truncates
        ! to destlen, reproducing Fortran character assignment into dest.
        ! snprintf is already declared by synthesize_real8_printer.
        type(liric_session_t), intent(inout) :: session
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: g_fmt
        type(lr_operand_desc_t) :: index_op, argv_op, dest_op, destlen_op
        type(lr_operand_desc_t) :: idx64, slot, src, tmpsize, tmp, dl32, args(5)
        type(c_ptr), target :: params(4)
        type(c_ptr) :: out_addr
        character(kind=c_char), allocatable :: c_name(:)
        type(lr_error_t) :: error
        integer(c_int32_t) :: block_id
        integer(c_int) :: status

        synthesize_get_arg_helper = .false.
        if (.not. require_open_session(session, error_msg)) return
        if (.not. create_cstring(session, '.ffc.argfmt', '%-*.*s', g_fmt, &
                                 error_msg)) return

        params(1) = lr_type_i32_s(session%handle)
        params(2) = lr_type_ptr_s(session%handle)
        params(3) = lr_type_ptr_s(session%handle)
        params(4) = lr_type_i64_s(session%handle)
        call clear_liric_error(error)
        call to_c_chars('.ffc.get_arg', c_name)
        status = lr_session_func_begin(session%handle, c_name, &
                                       lr_type_void_s(session%handle), &
                                       c_loc(params), 4_c_int32_t, c_false, error)
        if (.not. status_ok(status, error, error_msg)) return
        index_op = typed_param(session, 0, lr_type_i32_s(session%handle))
        argv_op = typed_param(session, 1, lr_type_ptr_s(session%handle))
        dest_op = typed_param(session, 2, lr_type_ptr_s(session%handle))
        destlen_op = typed_param(session, 3, lr_type_i64_s(session%handle))

        block_id = create_liric_block(session)
        if (.not. set_liric_block(session, block_id, error_msg)) return

        if (.not. emit_cast(session, LR_OP_ZEXT, index_op, &
                lr_type_i64_s(session%handle), idx64, error_msg)) return
        if (.not. gep_index(session, argv_op, idx64, &
                lr_type_ptr_s(session%handle), slot, error_msg)) return
        if (.not. load_typed(session, slot, lr_type_ptr_s(session%handle), src, &
                             error_msg)) return
        if (.not. emit_i64_binary(session, LR_OP_ADD, destlen_op, &
                i64_immediate(session, 1_c_int64_t), tmpsize, error_msg)) return
        if (.not. emit_alloca_bytes(session, tmpsize, tmp, error_msg)) return
        if (.not. emit_cast(session, LR_OP_TRUNC, destlen_op, &
                lr_type_i32_s(session%handle), dl32, error_msg)) return

        args(1) = tmp
        args(2) = tmpsize
        args(3) = g_fmt
        args(4) = dl32
        args(5) = dl32
        ! note: src is the 6th vararg; emit_call handles a trailing single arg
        if (.not. emit_call6(session, 'snprintf', args, src, error_msg)) return
        ! Copy destlen padded bytes plus the null terminator snprintf wrote, so
        ! dest (sized destlen+1) is a valid null-terminated print buffer.
        if (.not. emit_memcpy(session, dest_op, tmp, tmpsize, error_msg)) return
        if (.not. emit_ret_void_local(session, error_msg)) return

        call clear_liric_error(error)
        out_addr = c_null_ptr
        status = lr_session_func_end(session%handle, out_addr, error)
        if (.not. status_ok(status, error, error_msg)) return
        call set_empty(error_msg)
        synthesize_get_arg_helper = .true.
    end function synthesize_get_arg_helper

    logical function emit_get_arg_call(session, index_op, argv_op, dest_op, &
                                       destlen_op, error_msg)
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: index_op, argv_op, dest_op
        type(lr_operand_desc_t), intent(in) :: destlen_op
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: args(4)
        integer(c_int32_t) :: vreg

        args(1) = index_op
        args(2) = argv_op
        args(3) = dest_op
        args(4) = destlen_op
        emit_get_arg_call = emit_call(session, '.ffc.get_arg', args, &
                                      lr_type_void_s(session%handle), &
                                      4_c_int32_t, c_false, vreg, error_msg)
    end function emit_get_arg_call

    function typed_param(session, index, typ) result(operand)
        type(liric_session_t), intent(in) :: session
        integer, intent(in) :: index
        type(c_ptr), intent(in) :: typ
        type(lr_operand_desc_t) :: operand

        operand%kind = LR_OP_KIND_VREG
        operand%payload = int(lr_session_param(session%handle, &
                                               int(index, c_int32_t)), c_int64_t)
        operand%typ = typ
        operand%global_offset = 0_c_int64_t
    end function typed_param

    logical function emit_call6(session, callee_name, head_args, tail_arg, error_msg)
        ! Variadic call with five leading args plus one trailing arg.
        type(liric_session_t), intent(inout) :: session
        character(len=*), intent(in) :: callee_name
        type(lr_operand_desc_t), intent(in) :: head_args(5)
        type(lr_operand_desc_t), intent(in) :: tail_arg
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: args(6)
        integer(c_int32_t) :: vreg

        args(1:5) = head_args
        args(6) = tail_arg
        emit_call6 = emit_call(session, callee_name, args, &
                               lr_type_i32_s(session%handle), 3_c_int32_t, &
                               c_true, vreg, error_msg)
    end function emit_call6

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
        call set_empty(error_msg)
        emit_cast = .true.
    end function emit_cast

    logical function gep_index(session, base, index_op, elem_typ, result, error_msg)
        ! getelementptr elem_typ, base, index_op (element-typed stride).
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: base, index_op
        type(c_ptr), intent(in) :: elem_typ
        type(lr_operand_desc_t), intent(out) :: result
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t), target :: operands(2)
        type(lr_inst_desc_t) :: inst
        type(lr_error_t) :: error
        integer(c_int32_t) :: vreg

        gep_index = .false.
        operands(1) = base
        operands(2) = index_op
        inst%op = LR_OP_GEP
        inst%typ = elem_typ
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
        gep_index = .true.
    end function gep_index

    logical function load_typed(session, addr, typ, result, error_msg)
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: addr
        type(c_ptr), intent(in) :: typ
        type(lr_operand_desc_t), intent(out) :: result
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t), target :: operands(1)
        type(lr_inst_desc_t) :: inst
        type(lr_error_t) :: error
        integer(c_int32_t) :: vreg

        load_typed = .false.
        operands(1) = addr
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
        vreg = lr_session_emit(session%handle, inst, error)
        if (.not. status_ok(error%code, error, error_msg)) return
        result%kind = LR_OP_KIND_VREG
        result%payload = int(vreg, c_int64_t)
        result%typ = typ
        result%global_offset = 0_c_int64_t
        call set_empty(error_msg)
        load_typed = .true.
    end function load_typed

    logical function emit_snprintf(session, args, error_msg)
        ! Variadic snprintf(dest, size, fmt, ...) call. args holds dest, size,
        ! fmt, then the value arguments. snprintf is declared by
        ! synthesize_real8_printer (run during runtime preparation).
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: args(:)
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int32_t) :: vreg

        emit_snprintf = emit_call(session, 'snprintf', args, &
                                  lr_type_i32_s(session%handle), 3_c_int32_t, &
                                  c_true, vreg, error_msg)
    end function emit_snprintf

    logical function emit_sscanf(session, args, error_msg)
        ! Variadic sscanf(buf, fmt, ...) call. args holds buf, fmt, then the
        ! destination pointer arguments.
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: args(:)
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int32_t) :: vreg

        emit_sscanf = emit_call(session, 'sscanf', args, &
                                lr_type_i32_s(session%handle), 2_c_int32_t, &
                                c_true, vreg, error_msg)
    end function emit_sscanf

    logical function emit_scanf(session, args, error_msg)
        ! Variadic scanf(fmt, ...) call for list-directed stdin reads.
        ! args(1) is the format-string pointer; remaining args are destination
        ! pointers. Only the format is fixed; all destination pointers are vararg.
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: args(:)
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int32_t) :: vreg

        emit_scanf = emit_call(session, 'scanf', args, &
                               lr_type_i32_s(session%handle), 1_c_int32_t, &
                               c_true, vreg, error_msg)
    end function emit_scanf

    logical function emit_fprintf(session, args, error_msg)
        ! Variadic fprintf(fp, fmt, ...) call (#247 B5c file I/O).
        ! args(1) is the FILE* pointer, args(2) the format-string pointer;
        ! remaining args are the values to print.
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: args(:)
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int32_t) :: vreg

        emit_fprintf = emit_call(session, 'fprintf', args, &
                                 lr_type_i32_s(session%handle), 2_c_int32_t, &
                                 c_true, vreg, error_msg)
    end function emit_fprintf

    logical function emit_call(session, callee_name, args, ret_typ, fixed_args, &
                               vararg, vreg, error_msg)
        type(liric_session_t), intent(inout) :: session
        character(len=*), intent(in) :: callee_name
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
        integer :: nargs

        emit_call = .false.
        nargs = size(args)
        if (.not. make_callee(session, callee_name, callee, error_msg)) return

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
        emit_call = .true.
    end function emit_call

    logical function gep_byte(session, base, offset, result, error_msg)
        ! getelementptr i8, ptr base, i64 offset (byte stride).
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: base
        integer(c_int64_t), intent(in) :: offset
        type(lr_operand_desc_t), intent(out) :: result
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: offset_op
        type(lr_operand_desc_t), target :: operands(2)
        type(lr_inst_desc_t) :: inst
        type(lr_error_t) :: error
        integer(c_int32_t) :: vreg

        gep_byte = .false.
        offset_op%kind = LR_OP_KIND_IMM_I64
        offset_op%payload = offset
        offset_op%typ = lr_type_i64_s(session%handle)
        offset_op%global_offset = 0_c_int64_t
        operands(1) = base
        operands(2) = offset_op

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
        gep_byte = .true.
    end function gep_byte

    logical function store_zero_byte(session, addr, error_msg)
        ! Store an i8 0 at addr to null-terminate the mantissa string.
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: addr
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: val
        type(lr_operand_desc_t), target :: operands(2)
        type(lr_inst_desc_t) :: inst
        type(lr_error_t) :: error
        integer(c_int32_t) :: vreg

        store_zero_byte = .false.
        val%kind = LR_OP_KIND_IMM_I64
        val%payload = 0_c_int64_t
        val%typ = lr_type_i8_s(session%handle)
        val%global_offset = 0_c_int64_t
        operands(1) = val
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
        store_zero_byte = .true.
    end function store_zero_byte

    logical function make_callee(session, name, operand, error_msg)
        type(liric_session_t), intent(inout) :: session
        character(len=*), intent(in) :: name
        type(lr_operand_desc_t), intent(out) :: operand
        character(len=:), allocatable, intent(out) :: error_msg
        character(kind=c_char), allocatable :: c_name(:)
        integer(c_int32_t) :: symbol_id

        make_callee = .false.
        call to_c_chars(name, c_name)
        symbol_id = lr_session_intern(session%handle, c_name)
        if (symbol_id < 0_c_int32_t) then
            error_msg = 'LIRIC could not intern '//trim(name)
            return
        end if
        operand%kind = LR_OP_KIND_GLOBAL
        operand%payload = int(symbol_id, c_int64_t)
        operand%typ = lr_type_ptr_s(session%handle)
        operand%global_offset = 0_c_int64_t
        call set_empty(error_msg)
        make_callee = .true.
    end function make_callee

    logical function declare_libc(session, error_msg)
        type(liric_session_t), intent(inout) :: session
        character(len=:), allocatable, intent(out) :: error_msg
        type(c_ptr), target :: p3(3), p2(2), p1(1)

        declare_libc = .false.
        p3(1) = lr_type_ptr_s(session%handle)
        p3(2) = lr_type_i64_s(session%handle)
        p3(3) = lr_type_ptr_s(session%handle)
        if (.not. declare_one(session, 'snprintf', lr_type_i32_s(session%handle), &
                              c_loc(p3), 3_c_int32_t, c_true, error_msg)) return
        p1(1) = lr_type_ptr_s(session%handle)
        if (.not. declare_one(session, 'atoi', lr_type_i32_s(session%handle), &
                              c_loc(p1), 1_c_int32_t, c_false, error_msg)) return
        p2(1) = lr_type_ptr_s(session%handle)
        p2(2) = lr_type_i32_s(session%handle)
        if (.not. declare_one(session, 'strchr', lr_type_ptr_s(session%handle), &
                              c_loc(p2), 2_c_int32_t, c_false, error_msg)) return
        declare_libc = .true.
    end function declare_libc

    logical function declare_one(session, name, ret, params, n, vararg, error_msg)
        type(liric_session_t), intent(inout) :: session
        character(len=*), intent(in) :: name
        type(c_ptr), intent(in) :: ret
        type(c_ptr), intent(in) :: params
        integer(c_int32_t), intent(in) :: n
        logical(c_bool), intent(in) :: vararg
        character(len=:), allocatable, intent(out) :: error_msg
        character(kind=c_char), allocatable :: c_name(:)
        type(lr_error_t) :: error
        integer(c_int) :: status

        declare_one = .false.
        call clear_liric_error(error)
        call to_c_chars(name, c_name)
        status = lr_session_declare(session%handle, c_name, ret, params, n, &
                                    vararg, error)
        if (.not. status_ok(status, error, error_msg)) return
        declare_one = .true.
    end function declare_one

    logical function create_cstring(session, name, text, operand, error_msg)
        type(liric_session_t), intent(inout) :: session
        character(len=*), intent(in) :: name
        character(len=*), intent(in) :: text
        type(lr_operand_desc_t), intent(out) :: operand
        character(len=:), allocatable, intent(out) :: error_msg
        character(kind=c_char), allocatable, target :: bytes(:)
        character(kind=c_char), allocatable :: c_name(:)
        type(c_ptr) :: array_type
        integer(c_int32_t) :: global_id
        integer :: i, n

        create_cstring = .false.
        n = len(text) + 1
        allocate (bytes(n))
        do i = 1, len(text)
            bytes(i) = text(i:i)
        end do
        bytes(n) = c_null_char

        array_type = lr_type_array_s(session%handle, lr_type_i8_s(session%handle), &
                                     int(n, c_int64_t))
        if (.not. c_associated(array_type)) then
            error_msg = 'LIRIC did not return a format string array type'
            return
        end if

        call to_c_chars(name, c_name)
        global_id = lr_session_global(session%handle, c_name, array_type, c_true, &
                                      c_loc(bytes), int(n, c_size_t))
        if (global_id < 0_c_int32_t) then
            error_msg = 'LIRIC could not create string global '//trim(name)
            return
        end if
        global_id = lr_session_intern(session%handle, c_name)
        if (global_id < 0_c_int32_t) then
            error_msg = 'LIRIC could not intern string global '//trim(name)
            return
        end if

        operand%kind = LR_OP_KIND_GLOBAL
        operand%payload = int(global_id, c_int64_t)
        operand%typ = lr_type_ptr_s(session%handle)
        operand%global_offset = 0_c_int64_t
        call set_empty(error_msg)
        create_cstring = .true.
    end function create_cstring

    logical function begin_real8_printer(session, param_vreg, entry_block, &
                                         error_msg)
        type(liric_session_t), intent(inout) :: session
        integer(c_int32_t), intent(out) :: param_vreg
        integer(c_int32_t), intent(out) :: entry_block
        character(len=:), allocatable, intent(out) :: error_msg
        character(kind=c_char), allocatable :: c_name(:)
        type(c_ptr), target :: params(1)
        type(lr_error_t) :: error
        integer(c_int) :: status

        begin_real8_printer = .false.
        params(1) = lr_type_f64_s(session%handle)
        call clear_liric_error(error)
        call to_c_chars(REAL8_PRINTER, c_name)
        status = lr_session_func_begin(session%handle, c_name, &
                                       lr_type_void_s(session%handle), &
                                       c_loc(params), 1_c_int32_t, c_false, error)
        if (.not. status_ok(status, error, error_msg)) return

        param_vreg = lr_session_param(session%handle, 0_c_int32_t)
        entry_block = create_liric_block(session)
        call set_empty(error_msg)
        begin_real8_printer = .true.
    end function begin_real8_printer

    logical function emit_ret_void_local(session, error_msg)
        type(liric_session_t), intent(inout) :: session
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_inst_desc_t) :: inst
        type(lr_error_t) :: error
        integer(c_int32_t) :: vreg
        integer(c_int), parameter :: LR_OP_RET_VOID = 1_c_int

        emit_ret_void_local = .false.
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
        emit_ret_void_local = .true.
    end function emit_ret_void_local
