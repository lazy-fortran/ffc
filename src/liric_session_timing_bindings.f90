module liric_session_timing_bindings
    ! Runtime support for the CPU_TIME and SYSTEM_CLOCK intrinsic subroutines
    ! (#2820). Both lower to libc calls through the external ABI:
    !   cpu_time(t)        t = (real) clock() / CLOCKS_PER_SEC
    !   system_clock(n)    n = (int)  time(NULL)
    ! CLOCKS_PER_SEC is 1_000_000 on glibc. time(NULL) returns seconds since
    ! the epoch, always positive, matching gfortran's "real then integer" field
    ! structure. The actual magnitudes are nondeterministic; the conformance
    ! gauntlet compares such programs structurally (see lib_conformance.sh).
    use, intrinsic :: iso_c_binding, only: c_bool, c_char, c_float, c_int, &
                                           c_int32_t, c_int64_t, c_loc, &
                                           c_null_ptr, c_ptr
    use liric_session_bindings, only: liric_session_t, lr_error_t, &
                                      lr_inst_desc_t, lr_operand_desc_t, &
                                      LR_OP_CALL, LR_OP_TRUNC, LR_OP_SITOFP, &
                                      LR_OP_FMUL, LR_OP_KIND_VREG, &
                                      LR_OP_KIND_GLOBAL, LR_OP_KIND_IMM_I64, &
                                      lr_type_i32_s, lr_type_f32_s, &
                                      lr_type_i64_s, lr_type_ptr_s, &
                                      lr_session_emit, lr_session_intern, &
                                      status_ok, clear_liric_error, set_empty, &
                                      require_open_session, to_c_chars
    use liric_session_io_bindings, only: liric_f32_immediate, &
                                         emit_liric_f32_binary
    implicit none
    private

    logical(c_bool), parameter :: c_false = .false.

    public :: emit_cpu_time_value
    public :: emit_system_clock_value

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
    end interface

contains

    logical function emit_cpu_time_value(session, result, error_msg)
        ! result (f32) = (real) clock() * (1 / CLOCKS_PER_SEC).
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(out) :: result
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: ticks_i64, ticks_f32, scale

        emit_cpu_time_value = .false.
        if (.not. require_open_session(session, error_msg)) return
        if (.not. declare_extern(session, 'clock', &
                                 lr_type_i64_s(session%handle), error_msg)) return
        if (.not. call_extern_i64(session, 'clock', ticks_i64, error_msg)) return
        if (.not. emit_cast(session, LR_OP_SITOFP, ticks_i64, &
                            lr_type_f32_s(session%handle), ticks_f32, &
                            error_msg)) return
        scale = liric_f32_immediate(session, 1.0e-6_c_float)
        if (.not. emit_liric_f32_binary(session, LR_OP_FMUL, ticks_f32, scale, &
                                        result, error_msg)) return
        call set_empty(error_msg)
        emit_cpu_time_value = .true.
    end function emit_cpu_time_value

    logical function emit_system_clock_value(session, result, error_msg)
        ! result (i32) = (int) time(NULL); a positive seconds-since-epoch count.
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(out) :: result
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: secs_i64, args(1)

        emit_system_clock_value = .false.
        if (.not. require_open_session(session, error_msg)) return
        if (.not. declare_extern_ptr_arg(session, 'time', &
                                         lr_type_i64_s(session%handle), &
                                         error_msg)) return
        args(1) = i64_null_ptr(session)
        if (.not. call_extern(session, 'time', args, &
                              lr_type_i64_s(session%handle), 1_c_int32_t, &
                              secs_i64, error_msg)) return
        if (.not. emit_cast(session, LR_OP_TRUNC, secs_i64, &
                            lr_type_i32_s(session%handle), result, &
                            error_msg)) return
        call set_empty(error_msg)
        emit_system_clock_value = .true.
    end function emit_system_clock_value

    function i64_null_ptr(session) result(operand)
        ! A null pointer immediate for the time(NULL) argument.
        type(liric_session_t), intent(in) :: session
        type(lr_operand_desc_t) :: operand

        operand%kind = LR_OP_KIND_IMM_I64
        operand%payload = 0_c_int64_t
        operand%typ = lr_type_ptr_s(session%handle)
        operand%global_offset = 0_c_int64_t
    end function i64_null_ptr

    logical function declare_extern(session, name, ret_typ, error_msg)
        ! Declare an external libc function taking no arguments.
        type(liric_session_t), intent(inout) :: session
        character(len=*), intent(in) :: name
        type(c_ptr), intent(in) :: ret_typ
        character(len=:), allocatable, intent(out) :: error_msg
        character(kind=c_char), allocatable :: c_name(:)
        type(lr_error_t) :: error
        integer(c_int) :: status

        declare_extern = .false.
        call clear_liric_error(error)
        call to_c_chars(name, c_name)
        status = lr_session_declare(session%handle, c_name, ret_typ, c_null_ptr, &
                                    0_c_int32_t, c_false, error)
        if (.not. status_ok(status, error, error_msg)) return
        declare_extern = .true.
    end function declare_extern

    logical function declare_extern_ptr_arg(session, name, ret_typ, error_msg)
        ! Declare an external libc function taking one pointer argument.
        type(liric_session_t), intent(inout) :: session
        character(len=*), intent(in) :: name
        type(c_ptr), intent(in) :: ret_typ
        character(len=:), allocatable, intent(out) :: error_msg
        character(kind=c_char), allocatable :: c_name(:)
        type(c_ptr), target :: params(1)
        type(lr_error_t) :: error
        integer(c_int) :: status

        declare_extern_ptr_arg = .false.
        params(1) = lr_type_ptr_s(session%handle)
        call clear_liric_error(error)
        call to_c_chars(name, c_name)
        status = lr_session_declare(session%handle, c_name, ret_typ, &
                                    c_loc(params), 1_c_int32_t, c_false, error)
        if (.not. status_ok(status, error, error_msg)) return
        declare_extern_ptr_arg = .true.
    end function declare_extern_ptr_arg

    logical function call_extern_i64(session, name, result, error_msg)
        ! Call a no-argument external function returning i64.
        type(liric_session_t), intent(inout) :: session
        character(len=*), intent(in) :: name
        type(lr_operand_desc_t), intent(out) :: result
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: args(0)

        call_extern_i64 = call_extern(session, name, args, &
                                      lr_type_i64_s(session%handle), &
                                      0_c_int32_t, result, error_msg)
    end function call_extern_i64

    logical function call_extern(session, name, args, ret_typ, fixed_args, &
                                 result, error_msg)
        ! Emit a fixed-arg external-ABI call and capture its result vreg.
        type(liric_session_t), intent(inout) :: session
        character(len=*), intent(in) :: name
        type(lr_operand_desc_t), intent(in) :: args(:)
        type(c_ptr), intent(in) :: ret_typ
        integer(c_int32_t), intent(in) :: fixed_args
        type(lr_operand_desc_t), intent(out) :: result
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t), allocatable, target :: operands(:)
        type(lr_operand_desc_t) :: callee
        type(lr_inst_desc_t) :: inst
        type(lr_error_t) :: error
        character(kind=c_char), allocatable :: c_name(:)
        integer(c_int32_t) :: sym_id, vreg
        integer :: nargs

        call_extern = .false.
        call to_c_chars(name, c_name)
        sym_id = lr_session_intern(session%handle, c_name)
        if (sym_id < 0_c_int32_t) then
            error_msg = 'call_extern: could not intern '//trim(name)
            return
        end if
        callee%kind = LR_OP_KIND_GLOBAL
        callee%payload = int(sym_id, c_int64_t)
        callee%typ = lr_type_ptr_s(session%handle)
        callee%global_offset = 0_c_int64_t

        nargs = size(args)
        allocate (operands(nargs + 1))
        operands(1) = callee
        if (nargs > 0) operands(2:nargs + 1) = args

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
        inst%call_external_abi = .true.
        inst%call_vararg = c_false
        inst%call_fixed_args = fixed_args
        call clear_liric_error(error)
        vreg = lr_session_emit(session%handle, inst, error)
        if (.not. status_ok(error%code, error, error_msg)) return
        result%kind = LR_OP_KIND_VREG
        result%payload = int(vreg, c_int64_t)
        result%typ = ret_typ
        result%global_offset = 0_c_int64_t
        call set_empty(error_msg)
        call_extern = .true.
    end function call_extern

    logical function emit_cast(session, op, src, dst_typ, result, error_msg)
        ! Emit a unary cast (SITOFP, TRUNC, ...) and capture its result.
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

end module liric_session_timing_bindings
