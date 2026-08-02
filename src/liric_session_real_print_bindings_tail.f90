submodule (liric_session_real_print_bindings) liric_session_real_print_bindings_tail
contains

    module procedure synthesize_get_arg_helper
        ! void .ffc.get_arg(i32 index, ptr argv, ptr dest, i64 destlen):
        !   snprintf(tmp[destlen+1], "%-*.*s", destlen, destlen, argv[index]);
        !   memcpy(dest, tmp, destlen)
        ! "%-*.*s" left-justifies, pads with blanks to destlen, and truncates
        ! to destlen, reproducing Fortran character assignment into dest.
        ! snprintf is already declared by synthesize_real8_printer.
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
    end procedure synthesize_get_arg_helper

    module procedure emit_get_arg_call
        type(lr_operand_desc_t) :: args(4)
        integer(c_int32_t) :: vreg

        args(1) = index_op
        args(2) = argv_op
        args(3) = dest_op
        args(4) = destlen_op
        emit_get_arg_call = emit_call(session, '.ffc.get_arg', args, &
                                      lr_type_void_s(session%handle), &
                                      4_c_int32_t, c_false, vreg, error_msg)
    end procedure emit_get_arg_call

    module procedure typed_param

        operand%kind = LR_OP_KIND_VREG
        operand%payload = int(lr_session_param(session%handle, &
                                               int(index, c_int32_t)), c_int64_t)
        operand%typ = typ
        operand%global_offset = 0_c_int64_t
    end procedure typed_param

    module procedure emit_call6
        ! Variadic call with five leading args plus one trailing arg.
        type(lr_operand_desc_t) :: args(6)
        integer(c_int32_t) :: vreg

        args(1:5) = head_args
        args(6) = tail_arg
        emit_call6 = emit_call(session, callee_name, args, &
                               lr_type_i32_s(session%handle), 3_c_int32_t, &
                               c_true, vreg, error_msg)
    end procedure emit_call6

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
        call set_empty(error_msg)
        emit_cast = .true.
    end procedure emit_cast

    module procedure gep_index
        ! getelementptr elem_typ, base, index_op (element-typed stride).
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
    end procedure gep_index

    module procedure load_typed
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
    end procedure load_typed

    module procedure emit_snprintf
        ! Variadic snprintf(dest, size, fmt, ...) call. args holds dest, size,
        ! fmt, then the value arguments. snprintf is declared by
        ! synthesize_real8_printer (run during runtime preparation).
        integer(c_int32_t) :: vreg

        emit_snprintf = emit_call(session, 'snprintf', args, &
                                  lr_type_i32_s(session%handle), 3_c_int32_t, &
                                  c_true, vreg, error_msg)
    end procedure emit_snprintf

    module procedure emit_sscanf
        ! Variadic sscanf(buf, fmt, ...) call. args holds buf, fmt, then the
        ! destination pointer arguments.
        integer(c_int32_t) :: vreg

        emit_sscanf = emit_call(session, 'sscanf', args, &
                                lr_type_i32_s(session%handle), 2_c_int32_t, &
                                c_true, vreg, error_msg)
    end procedure emit_sscanf

    module procedure emit_scanf
        ! Variadic scanf(fmt, ...) call for list-directed stdin reads.
        ! args(1) is the format-string pointer; remaining args are destination
        ! pointers. Only the format is fixed; all destination pointers are vararg.
        integer(c_int32_t) :: vreg

        emit_scanf = emit_call(session, 'scanf', args, &
                               lr_type_i32_s(session%handle), 1_c_int32_t, &
                               c_true, vreg, error_msg)
    end procedure emit_scanf

    module procedure emit_fscanf
        ! Variadic fscanf(fp, fmt, ...) call for list-directed file-unit reads.
        ! args(1) is the FILE* pointer, args(2) the format-string pointer;
        ! remaining args are destination pointers. Two fixed args, rest vararg.
        integer(c_int32_t) :: vreg

        emit_fscanf = emit_call(session, 'fscanf', args, &
                                lr_type_i32_s(session%handle), 2_c_int32_t, &
                                c_true, vreg, error_msg)
    end procedure emit_fscanf

    module procedure emit_fscanf_count
        ! Variadic fscanf(fp, fmt, ...) that also yields the conversion count.
        ! A caller uses the count to distinguish a successful conversion from a
        ! matching failure when it must report a READ status.
        integer(c_int32_t) :: vreg

        emit_fscanf_count = emit_call(session, 'fscanf', args, &
                                      lr_type_i32_s(session%handle), &
                                      2_c_int32_t, c_true, vreg, error_msg)
        if (.not. emit_fscanf_count) return
        result = i32_vreg(session, vreg)
    end procedure emit_fscanf_count

    module procedure emit_fprintf
        ! Variadic fprintf(fp, fmt, ...) call (#247 B5c file I/O).
        ! args(1) is the FILE* pointer, args(2) the format-string pointer;
        ! remaining args are the values to print.
        integer(c_int32_t) :: vreg

        emit_fprintf = emit_call(session, 'fprintf', args, &
                                 lr_type_i32_s(session%handle), 2_c_int32_t, &
                                 c_true, vreg, error_msg)
    end procedure emit_fprintf

    module procedure emit_dprintf
        ! Variadic dprintf(int fd, fmt, ...) call. args(1) is the file
        ! descriptor, args(2) the format-string pointer; the rest are values.
        ! Used for STOP banners, which gfortran writes to stderr (fd 2).
        integer(c_int32_t) :: vreg

        emit_dprintf = emit_call(session, 'dprintf', args, &
                                 lr_type_i32_s(session%handle), 2_c_int32_t, &
                                 c_true, vreg, error_msg)
    end procedure emit_dprintf

    module procedure emit_getchar
        ! getchar() -> int: read one byte from stdin, or EOF (-1). Used by PAUSE
        ! to detect whether a resume line is available (#280).
        use, intrinsic :: iso_c_binding, only: c_int64_t
        type(lr_operand_desc_t) :: args(0)
        integer(c_int32_t) :: vreg

        emit_getchar = .false.
        if (.not. emit_call(session, 'getchar', args, &
                            lr_type_i32_s(session%handle), 0_c_int32_t, &
                            c_false, vreg, error_msg)) return
        result%kind = LR_OP_KIND_VREG
        result%payload = int(vreg, c_int64_t)
        result%typ = lr_type_i32_s(session%handle)
        result%global_offset = 0_c_int64_t
        emit_getchar = .true.
    end procedure emit_getchar

    module procedure emit_exit
        ! exit(int) terminates the process, flushing stdio buffers. PAUSE calls
        ! it with 0 when stdin signals end-of-input, matching gfortran (#280).
        type(lr_operand_desc_t) :: args(1)
        integer(c_int32_t) :: vreg

        args(1) = code
        emit_exit = emit_call(session, 'exit', args, &
                              lr_type_void_s(session%handle), 1_c_int32_t, &
                              c_false, vreg, error_msg)
    end procedure emit_exit

    module procedure emit_call
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
    end procedure emit_call

    module procedure gep_byte
        ! getelementptr i8, ptr base, i64 offset (byte stride).
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
    end procedure gep_byte

    module procedure store_zero_byte
        ! Store an i8 0 at addr to null-terminate the mantissa string.
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
    end procedure store_zero_byte

    module procedure make_callee
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
    end procedure make_callee

    module procedure declare_libc
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
    end procedure declare_libc

    module procedure declare_one
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
    end procedure declare_one

    module procedure create_cstring
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
    end procedure create_cstring

    module procedure begin_real8_printer
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
    end procedure begin_real8_printer

    module procedure emit_ret_void_local
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
    end procedure emit_ret_void_local

end submodule liric_session_real_print_bindings_tail
