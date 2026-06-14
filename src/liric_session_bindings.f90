module liric_session_bindings
    use, intrinsic :: iso_c_binding, only: c_associated, c_bool, c_char
    use, intrinsic :: iso_c_binding, only: c_int, c_int32_t, c_int64_t
    use, intrinsic :: iso_c_binding, only: c_loc, c_null_char, c_null_ptr, c_ptr, &
                                              c_size_t
    use liric_session_common, only: lr_session_config_t, lr_error_t, &
                                    lr_operand_desc_t, lr_inst_desc_t, &
                                    liric_session_t, require_open_session, &
                                    status_ok, liric_session_error_message, &
                                    clear_liric_error, to_c_chars, set_empty
    implicit none
    private

    integer(c_int), parameter, public :: LR_MODE_DIRECT = 0_c_int
    integer(c_int), parameter, public :: LR_SESSION_BACKEND_DEFAULT = 0_c_int
    integer(c_int), parameter, public :: LR_OK = 0_c_int
    logical(c_bool), parameter, public :: c_false = .false.

    integer(c_int), parameter, public :: LR_OP_RET = 0_c_int
    integer(c_int), parameter, public :: LR_OP_RET_VOID = 1_c_int
    integer(c_int), parameter, public :: LR_OP_ADD = 5_c_int
    integer(c_int), parameter, public :: LR_OP_SUB = 6_c_int
    integer(c_int), parameter, public :: LR_OP_MUL = 7_c_int
    integer(c_int), parameter, public :: LR_OP_SDIV = 8_c_int
    integer(c_int), parameter, public :: LR_OP_SREM = 9_c_int
    integer(c_int), parameter, public :: LR_OP_AND = 12_c_int
    integer(c_int), parameter, public :: LR_OP_OR = 13_c_int
    integer(c_int), parameter, public :: LR_OP_XOR = 14_c_int
    integer(c_int), parameter, public :: LR_OP_SHL = 15_c_int
    integer(c_int), parameter, public :: LR_OP_LSHR = 16_c_int
    integer(c_int), parameter, public :: LR_OP_ALLOCA = 26_c_int
    integer(c_int), parameter, public :: LR_OP_LOAD = 27_c_int
    integer(c_int), parameter, public :: LR_OP_STORE = 28_c_int
    integer(c_int), parameter, public :: LR_OP_GEP = 29_c_int
    integer(c_int), parameter, public :: LR_OP_CALL = 30_c_int
    integer(c_int), parameter, public :: LR_OP_BITCAST = 36_c_int
    integer(c_int), parameter, public :: LR_OP_SEXT = 33_c_int
    integer(c_int), parameter, public :: LR_OP_ZEXT = 34_c_int
    integer(c_int), parameter, public :: LR_OP_TRUNC = 35_c_int
    integer(c_int), parameter, public :: LR_OP_SITOFP = 39_c_int
    integer(c_int), parameter, public :: LR_OP_FPTOSI = 41_c_int
    integer(c_int), parameter, public :: LR_OP_FPEXT = 43_c_int
    integer(c_int), parameter, public :: LR_OP_FPTRUNC = 44_c_int
    integer(c_int), parameter, public :: LR_OP_FADD = 18_c_int
    integer(c_int), parameter, public :: LR_OP_FMUL = 20_c_int
    integer(c_int), parameter, public :: LR_OP_FDIV = 21_c_int

    integer(c_int), parameter, public :: LR_OP_KIND_VREG = 0_c_int
    integer(c_int), parameter, public :: LR_OP_KIND_IMM_I64 = 1_c_int
    integer(c_int), parameter, public :: LR_OP_KIND_BLOCK = 3_c_int
    integer(c_int), parameter, public :: LR_OP_KIND_GLOBAL = 4_c_int

    public :: lr_session_config_t, lr_error_t, lr_operand_desc_t, &
              lr_inst_desc_t, liric_session_t
    public :: liric_session_create
    public :: liric_session_error_message
    public :: i32_vreg, i32_immediate
    public :: global_operand
    public :: lr_type_i32_s, lr_type_i64_s, lr_type_f32_s, lr_type_f64_s, &
              lr_type_ptr_s, lr_type_i8_s, lr_type_i16_s, lr_type_array_s, lr_type_void_s
    public :: lr_session_emit, lr_session_vreg, lr_session_param, &
              lr_session_intern, lr_session_global
    public :: status_ok, clear_liric_error, set_empty, require_open_session, &
              to_c_chars, liric_session_error_message
    public :: destroy, is_open, begin_i32_main, begin_i32_function, &
              begin_void_subroutine, begin_ptr_function, &
              emit_ret_i32_main_exe, &
              emit_ret_i32_operand, emit_ret_void, finish_function, &
              finish_and_emit_object, finish_and_emit_exe, &
              emit_i32_call, emit_ptr_call, emit_void_call, &
              emit_i32_indirect_call, emit_void_indirect_call

    interface
        function lr_session_create(cfg, err) result(handle) bind(c)
            import :: c_ptr, lr_error_t, lr_session_config_t
            type(lr_session_config_t), intent(in) :: cfg
            type(lr_error_t), intent(inout) :: err
            type(c_ptr) :: handle
        end function lr_session_create

        subroutine lr_session_destroy(handle) bind(c)
            import :: c_ptr
            type(c_ptr), value :: handle
        end subroutine lr_session_destroy

        function lr_type_i32_s(handle) result(typ) bind(c)
            import :: c_ptr
            type(c_ptr), value :: handle
            type(c_ptr) :: typ
        end function lr_type_i32_s

        function lr_type_i64_s(handle) result(typ) bind(c)
            import :: c_ptr
            type(c_ptr), value :: handle
            type(c_ptr) :: typ
        end function lr_type_i64_s

        function lr_type_void_s(handle) result(typ) bind(c)
            import :: c_ptr
            type(c_ptr), value :: handle
            type(c_ptr) :: typ
        end function lr_type_void_s

        function lr_type_f32_s(handle) result(typ) bind(c)
            import :: c_ptr
            type(c_ptr), value :: handle
            type(c_ptr) :: typ
        end function lr_type_f32_s

        function lr_type_f64_s(handle) result(typ) bind(c)
            import :: c_ptr
            type(c_ptr), value :: handle
            type(c_ptr) :: typ
        end function lr_type_f64_s

        function lr_type_i8_s(handle) result(typ) bind(c)
            import :: c_ptr
            type(c_ptr), value :: handle
            type(c_ptr) :: typ
        end function lr_type_i8_s

        function lr_type_i16_s(handle) result(typ) bind(c)
            import :: c_ptr
            type(c_ptr), value :: handle
            type(c_ptr) :: typ
        end function lr_type_i16_s

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

        function lr_session_vreg(handle) result(vreg) bind(c)
            import :: c_int32_t, c_ptr
            type(c_ptr), value :: handle
            integer(c_int32_t) :: vreg
        end function lr_session_vreg

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

        function lr_session_func_end(handle, out_addr, err) result(status) &
            bind(c)
            import :: c_int, c_ptr, lr_error_t
            type(c_ptr), value :: handle
            type(c_ptr), intent(out) :: out_addr
            type(lr_error_t), intent(inout) :: err
            integer(c_int) :: status
        end function lr_session_func_end

        function lr_session_param(handle, index) result(vreg) bind(c)
            import :: c_int32_t, c_ptr
            type(c_ptr), value :: handle
            integer(c_int32_t), value :: index
            integer(c_int32_t) :: vreg
        end function lr_session_param

        function lr_session_block(handle) result(block_id) bind(c)
            import :: c_int32_t, c_ptr
            type(c_ptr), value :: handle
            integer(c_int32_t) :: block_id
        end function lr_session_block

        function lr_session_set_block(handle, block_id, err) result(status) &
            bind(c)
            import :: c_int, c_int32_t, c_ptr, lr_error_t
            type(c_ptr), value :: handle
            integer(c_int32_t), value :: block_id
            type(lr_error_t), intent(inout) :: err
            integer(c_int) :: status
        end function lr_session_set_block

        function lr_session_emit(handle, inst, err) result(vreg) bind(c)
            import :: c_int32_t, c_ptr, lr_error_t, lr_inst_desc_t
            type(c_ptr), value :: handle
            type(lr_inst_desc_t), intent(in) :: inst
            type(lr_error_t), intent(inout) :: err
            integer(c_int32_t) :: vreg
        end function lr_session_emit

        function lr_session_emit_exe(handle, path, err) result(status) &
            bind(c)
            import :: c_char, c_int, c_ptr, lr_error_t
            type(c_ptr), value :: handle
            character(kind=c_char), intent(in) :: path(*)
            type(lr_error_t), intent(inout) :: err
            integer(c_int) :: status
        end function lr_session_emit_exe

        function lr_session_emit_object(handle, path, err) result(status) &
            bind(c)
            import :: c_char, c_int, c_ptr, lr_error_t
            type(c_ptr), value :: handle
            character(kind=c_char), intent(in) :: path(*)
            type(lr_error_t), intent(inout) :: err
            integer(c_int) :: status
        end function lr_session_emit_object

        function lr_session_intern(handle, name) result(symbol_id) bind(c)
            import :: c_char, c_int32_t, c_ptr
            type(c_ptr), value :: handle
            character(kind=c_char), intent(in) :: name(*)
            integer(c_int32_t) :: symbol_id
        end function lr_session_intern

        function lr_session_global(handle, name, typ, is_const, init, &
                                   init_size) result(global_id) bind(c)
            import :: c_bool, c_char, c_int32_t, c_ptr, c_size_t, c_int
            type(c_ptr), value :: handle
            character(kind=c_char), intent(in) :: name(*)
            type(c_ptr), value :: typ
            logical(c_bool), value :: is_const
            type(c_ptr), value :: init
            integer(c_size_t), value :: init_size
            integer(c_int32_t) :: global_id
        end function lr_session_global
    end interface

contains

    subroutine liric_session_create(session, error_msg, config)
        type(liric_session_t), intent(out) :: session
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_session_config_t), intent(in), optional :: config
        type(lr_session_config_t) :: local_config
        type(lr_error_t) :: error

        call clear_liric_error(error)
        local_config = lr_session_config_t()
        if (present(config)) local_config = config

        session%handle = lr_session_create(local_config, error)
        if (c_associated(session%handle)) then
            call set_empty(error_msg)
        else
            error_msg = liric_session_error_message(error)
        end if
    end subroutine liric_session_create

    subroutine destroy(session)
        type(liric_session_t), intent(inout) :: session

        if (c_associated(session%handle)) then
            call lr_session_destroy(session%handle)
            session%handle = c_null_ptr
        end if
    end subroutine destroy

    logical function is_open(session)
        type(liric_session_t), intent(in) :: session

        is_open = c_associated(session%handle)
    end function is_open

    logical function emit_ret_i32_main_exe(session, return_code, path, &
                                           error_msg)
        type(liric_session_t), intent(inout) :: session
        integer, intent(in) :: return_code
        character(len=*), intent(in) :: path
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: value

        if (.not. begin_i32_main(session, error_msg)) then
            emit_ret_i32_main_exe = .false.
            return
        end if

        value = i32_immediate(session, int(return_code, c_int64_t))
        if (.not. emit_ret_i32_operand(session, value, error_msg)) then
            emit_ret_i32_main_exe = .false.
            return
        end if

        emit_ret_i32_main_exe = finish_and_emit_exe(session, path, error_msg)
    end function emit_ret_i32_main_exe

    logical function begin_i32_main(session, error_msg, argc_vreg, argv_vreg)
        ! main is emitted as `i32 main(i32 argc, ptr argv)`. The C runtime
        ! always passes argc/argv, so declaring them is ABI-safe even when the
        ! program ignores them. Their vregs are returned for the command-line
        ! argument intrinsics.
        type(liric_session_t), intent(inout) :: session
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int32_t), intent(out), optional :: argc_vreg
        integer(c_int32_t), intent(out), optional :: argv_vreg
        character(kind=c_char), allocatable :: c_name(:)
        type(c_ptr) :: i32_type
        type(c_ptr), target :: params(2)
        type(lr_error_t) :: error
        integer(c_int32_t) :: block_id
        integer(c_int) :: status

        begin_i32_main = .false.
        if (.not. require_open_session(session, error_msg)) return

        i32_type = session_i32_type(session, error_msg)
        if (len_trim(error_msg) > 0) return
        params(1) = i32_type
        params(2) = lr_type_ptr_s(session%handle)

        call clear_liric_error(error)
        call to_c_chars('main', c_name)
        status = lr_session_func_begin(session%handle, c_name, i32_type, &
                                       c_loc(params), 2_c_int32_t, c_false, &
                                       error)
        if (.not. status_ok(status, error, error_msg)) return

        if (present(argc_vreg)) argc_vreg = lr_session_param(session%handle, &
                                                             0_c_int32_t)
        if (present(argv_vreg)) argv_vreg = lr_session_param(session%handle, &
                                                             1_c_int32_t)

        block_id = lr_session_block(session%handle)
        status = lr_session_set_block(session%handle, block_id, error)
        if (.not. status_ok(status, error, error_msg)) return

        call set_empty(error_msg)
        begin_i32_main = .true.
    end function begin_i32_main

    logical function begin_i32_function(session, name, param_count, &
                                        error_msg)
        type(liric_session_t), intent(inout) :: session
        character(len=*), intent(in) :: name
        integer, intent(in) :: param_count
        character(len=:), allocatable, intent(out) :: error_msg
        character(kind=c_char), allocatable :: c_name(:)
        type(c_ptr), allocatable, target :: params(:)
        type(c_ptr) :: param_type
        type(c_ptr) :: i32_type
        type(c_ptr) :: params_ptr
        type(lr_error_t) :: error
        integer(c_int32_t) :: block_id
        integer(c_int) :: status
        integer :: i

        begin_i32_function = .false.
        if (.not. require_open_session(session, error_msg)) return

        i32_type = session_i32_type(session, error_msg)
        if (len_trim(error_msg) > 0) return

        param_type = lr_type_ptr_s(session%handle)
        if (.not. c_associated(param_type)) then
            error_msg = 'LIRIC did not return a pointer type'
            return
        end if

        params_ptr = c_null_ptr
        if (param_count > 0) then
            allocate (params(param_count))
            do i = 1, param_count
                params(i) = param_type
            end do
            params_ptr = c_loc(params)
        end if

        call clear_liric_error(error)
        call to_c_chars(trim(name), c_name)
        status = lr_session_func_begin(session%handle, c_name, i32_type, &
                                       params_ptr, int(param_count, c_int32_t), &
                                       c_false, error)
        if (.not. status_ok(status, error, error_msg)) return

        block_id = lr_session_block(session%handle)
        status = lr_session_set_block(session%handle, block_id, error)
        if (.not. status_ok(status, error, error_msg)) return

        call set_empty(error_msg)
        begin_i32_function = .true.
    end function begin_i32_function

    logical function begin_void_subroutine(session, name, param_count, &
                                           error_msg)
        type(liric_session_t), intent(inout) :: session
        character(len=*), intent(in) :: name
        integer, intent(in) :: param_count
        character(len=:), allocatable, intent(out) :: error_msg
        character(kind=c_char), allocatable :: c_name(:)
        type(c_ptr), allocatable, target :: params(:)
        type(c_ptr) :: param_type
        type(c_ptr) :: i32_type
        type(c_ptr) :: params_ptr
        type(c_ptr) :: void_type
        type(lr_error_t) :: error
        integer(c_int32_t) :: block_id
        integer(c_int) :: status
        integer :: i

        begin_void_subroutine = .false.
        if (.not. require_open_session(session, error_msg)) return

        i32_type = session_i32_type(session, error_msg)
        if (len_trim(error_msg) > 0) return
        void_type = lr_type_void_s(session%handle)
        if (.not. c_associated(void_type)) then
            error_msg = 'LIRIC did not return a void type'
            return
        end if

        param_type = lr_type_ptr_s(session%handle)
        if (.not. c_associated(param_type)) then
            error_msg = 'LIRIC did not return a pointer type'
            return
        end if

        params_ptr = c_null_ptr
        if (param_count > 0) then
            allocate (params(param_count))
            do i = 1, param_count
                params(i) = param_type
            end do
            params_ptr = c_loc(params)
        end if

        call clear_liric_error(error)
        call to_c_chars(trim(name), c_name)
        status = lr_session_func_begin(session%handle, c_name, void_type, &
                                       params_ptr, int(param_count, c_int32_t), &
                                       c_false, error)
        if (.not. status_ok(status, error, error_msg)) return

        block_id = lr_session_block(session%handle)
        status = lr_session_set_block(session%handle, block_id, error)
        if (.not. status_ok(status, error, error_msg)) return

        call set_empty(error_msg)
        begin_void_subroutine = .true.
    end function begin_void_subroutine

    logical function begin_ptr_function(session, name, param_count, &
                                        error_msg)
        type(liric_session_t), intent(inout) :: session
        character(len=*), intent(in) :: name
        integer, intent(in) :: param_count
        character(len=:), allocatable, intent(out) :: error_msg
        character(kind=c_char), allocatable :: c_name(:)
        type(c_ptr), allocatable, target :: params(:)
        type(c_ptr) :: param_type
        type(c_ptr) :: params_ptr
        type(lr_error_t) :: error
        integer(c_int32_t) :: block_id
        integer(c_int) :: status
        integer :: i

        begin_ptr_function = .false.
        if (.not. require_open_session(session, error_msg)) return

        param_type = lr_type_ptr_s(session%handle)
        if (.not. c_associated(param_type)) then
            error_msg = 'LIRIC did not return a pointer type'
            return
        end if

        params_ptr = c_null_ptr
        if (param_count > 0) then
            allocate (params(param_count))
            do i = 1, param_count
                params(i) = param_type
            end do
            params_ptr = c_loc(params)
        end if

        call clear_liric_error(error)
        call to_c_chars(trim(name), c_name)
        status = lr_session_func_begin(session%handle, c_name, &
                                       lr_type_ptr_s(session%handle), &
                                       params_ptr, int(param_count, c_int32_t), &
                                       c_false, error)
        if (.not. status_ok(status, error, error_msg)) return

        block_id = lr_session_block(session%handle)
        status = lr_session_set_block(session%handle, block_id, error)
        if (.not. status_ok(status, error, error_msg)) return

        call set_empty(error_msg)
        begin_ptr_function = .true.
    end function begin_ptr_function

    logical function emit_ret_i32_operand(session, value, error_msg)
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_error_t) :: error
        integer(c_int32_t) :: unused_vreg

        emit_ret_i32_operand = .false.
        if (.not. require_open_session(session, error_msg)) return

        unused_vreg = emit_ret_i32(session%handle, value, error)
        if (.not. status_ok(error%code, error, error_msg)) return

        call set_empty(error_msg)
        emit_ret_i32_operand = .true.
    end function emit_ret_i32_operand

    logical function emit_ret_void(session, error_msg)
        type(liric_session_t), intent(inout) :: session
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_error_t) :: error
        integer(c_int32_t) :: unused_vreg

        emit_ret_void = .false.
        if (.not. require_open_session(session, error_msg)) return

        unused_vreg = emit_ret_void_inst(session%handle, error)
        if (.not. status_ok(error%code, error, error_msg)) return

        call set_empty(error_msg)
        emit_ret_void = .true.
    end function emit_ret_void

    logical function finish_and_emit_exe(session, path, error_msg)
        type(liric_session_t), intent(inout) :: session
        character(len=*), intent(in) :: path
        character(len=:), allocatable, intent(out) :: error_msg
        character(kind=c_char), allocatable :: c_path(:)
        type(c_ptr) :: out_addr
        type(lr_error_t) :: error
        integer(c_int) :: status

        finish_and_emit_exe = .false.
        if (.not. require_open_session(session, error_msg)) return

        call clear_liric_error(error)
        out_addr = c_null_ptr
        status = lr_session_func_end(session%handle, out_addr, error)
        if (.not. status_ok(status, error, error_msg)) return

        call to_c_chars(path, c_path)
        status = lr_session_emit_exe(session%handle, c_path, error)
        if (.not. status_ok(status, error, error_msg)) return

        call set_empty(error_msg)
        finish_and_emit_exe = .true.
    end function finish_and_emit_exe

    logical function finish_and_emit_object(session, path, error_msg)
        type(liric_session_t), intent(inout) :: session
        character(len=*), intent(in) :: path
        character(len=:), allocatable, intent(out) :: error_msg
        character(kind=c_char), allocatable :: c_path(:)
        type(c_ptr) :: out_addr
        type(lr_error_t) :: error
        integer(c_int) :: status

        finish_and_emit_object = .false.
        if (.not. require_open_session(session, error_msg)) return

        call clear_liric_error(error)
        out_addr = c_null_ptr
        status = lr_session_func_end(session%handle, out_addr, error)
        if (.not. status_ok(status, error, error_msg)) return

        call to_c_chars(path, c_path)
        status = lr_session_emit_object(session%handle, c_path, error)
        if (.not. status_ok(status, error, error_msg)) return

        call set_empty(error_msg)
        finish_and_emit_object = .true.
    end function finish_and_emit_object

    logical function finish_function(session, error_msg)
        type(liric_session_t), intent(inout) :: session
        character(len=:), allocatable, intent(out) :: error_msg
        type(c_ptr) :: out_addr
        type(lr_error_t) :: error
        integer(c_int) :: status

        finish_function = .false.
        if (.not. require_open_session(session, error_msg)) return

        call clear_liric_error(error)
        out_addr = c_null_ptr
        status = lr_session_func_end(session%handle, out_addr, error)
        if (.not. status_ok(status, error, error_msg)) return

        call set_empty(error_msg)
        finish_function = .true.
    end function finish_function

    logical function emit_i32_call(session, name, args, result, error_msg)
        type(liric_session_t), intent(inout) :: session
        character(len=*), intent(in) :: name
        type(lr_operand_desc_t), intent(in) :: args(:)
        type(lr_operand_desc_t), intent(out) :: result
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_error_t) :: error
        integer(c_int32_t) :: vreg

        emit_i32_call = .false.
        if (.not. require_open_session(session, error_msg)) return

        vreg = emit_call_i32(session%handle, name, args, error)
        if (.not. status_ok(error%code, error, error_msg)) return

        result = i32_vreg(session, vreg)
        call set_empty(error_msg)
        emit_i32_call = .true.
    end function emit_i32_call

    logical function emit_void_call(session, name, args, error_msg)
        type(liric_session_t), intent(inout) :: session
        character(len=*), intent(in) :: name
        type(lr_operand_desc_t), intent(in) :: args(:)
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_error_t) :: error
        integer(c_int32_t) :: unused_vreg

        emit_void_call = .false.
        if (.not. require_open_session(session, error_msg)) return

        unused_vreg = emit_call_void(session%handle, name, args, error)
        if (.not. status_ok(error%code, error, error_msg)) return

        call set_empty(error_msg)
        emit_void_call = .true.
    end function emit_void_call

    logical function emit_ptr_call(session, name, args, result, error_msg)
        type(liric_session_t), intent(inout) :: session
        character(len=*), intent(in) :: name
        type(lr_operand_desc_t), intent(in) :: args(:)
        type(lr_operand_desc_t), intent(out) :: result
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_error_t) :: error
        integer(c_int32_t) :: vreg
        type(lr_operand_desc_t), target :: operands(9)
        type(lr_inst_desc_t) :: inst
        character(kind=c_char), allocatable :: c_name(:)
        integer(c_int32_t) :: symbol_id
        integer :: i

        emit_ptr_call = .false.
        if (.not. require_open_session(session, error_msg)) return
        call to_c_chars(trim(name), c_name)
        symbol_id = lr_session_intern(session%handle, c_name)
        if (symbol_id < 0_c_int32_t .or. size(args) > 8) then
            error_msg = 'could not intern symbol for ptr call: '//trim(name)
            return
        end if
        operands(1) = global_operand(session%handle, symbol_id)
        do i = 1, size(args)
            operands(i + 1) = args(i)
        end do
        inst%op = LR_OP_CALL
        inst%typ = lr_type_ptr_s(session%handle)
        inst%dest = 0_c_int32_t
        inst%operands = c_loc(operands)
        inst%num_operands = int(size(args) + 1, c_int32_t)
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
        ! Inline ptr_vreg: cannot use liric_session_memory_bindings here
        ! (circular dependency); build the ptr operand directly.
        result%kind = LR_OP_KIND_VREG
        result%payload = int(vreg, c_int64_t)
        result%typ = lr_type_ptr_s(session%handle)
        result%global_offset = 0_c_int64_t
        call set_empty(error_msg)
        emit_ptr_call = .true.
    end function emit_ptr_call

    function emit_ret_i32(handle, value, error) result(vreg)
        type(c_ptr), intent(in) :: handle
        type(lr_operand_desc_t), intent(in) :: value
        type(lr_error_t), intent(inout) :: error
        integer(c_int32_t) :: vreg
        type(lr_operand_desc_t), target :: operands(1)
        type(lr_inst_desc_t) :: inst

        operands(1) = value

        inst%op = LR_OP_RET
        inst%typ = value%typ
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
    end function emit_ret_i32

    function emit_ret_void_inst(handle, error) result(vreg)
        type(c_ptr), intent(in) :: handle
        type(lr_error_t), intent(inout) :: error
        integer(c_int32_t) :: vreg
        type(lr_inst_desc_t) :: inst

        inst%op = LR_OP_RET_VOID
        inst%typ = lr_type_void_s(handle)
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
        vreg = lr_session_emit(handle, inst, error)
    end function emit_ret_void_inst

    function emit_binary(handle, opcode, lhs, rhs, error) result(vreg)
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
        inst%typ = lhs%typ
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
    end function emit_binary

    function emit_call_i32(handle, name, args, error) result(vreg)
        type(c_ptr), intent(in) :: handle
        character(len=*), intent(in) :: name
        type(lr_operand_desc_t), intent(in) :: args(:)
        type(lr_error_t), intent(inout) :: error
        integer(c_int32_t) :: vreg
        type(lr_operand_desc_t), target :: operands(9)
        type(lr_inst_desc_t) :: inst
        character(kind=c_char), allocatable :: c_name(:)
        integer(c_int32_t) :: symbol_id
        integer :: i

        call to_c_chars(trim(name), c_name)
        symbol_id = lr_session_intern(handle, c_name)
        if (symbol_id < 0_c_int32_t .or. size(args) > 8) then
            call clear_liric_error(error)
            error%code = 1_c_int
            return
        end if

        operands(1) = global_operand(handle, symbol_id)
        do i = 1, size(args)
            operands(i + 1) = args(i)
        end do

        inst%op = LR_OP_CALL
        inst%typ = lr_type_i32_s(handle)
        inst%dest = 0_c_int32_t
        inst%operands = c_loc(operands)
        inst%num_operands = int(size(args) + 1, c_int32_t)
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
    end function emit_call_i32

    function emit_call_void(handle, name, args, error) result(vreg)
        type(c_ptr), intent(in) :: handle
        character(len=*), intent(in) :: name
        type(lr_operand_desc_t), intent(in) :: args(:)
        type(lr_error_t), intent(inout) :: error
        integer(c_int32_t) :: vreg
        type(lr_operand_desc_t), target :: operands(9)
        type(lr_inst_desc_t) :: inst
        character(kind=c_char), allocatable :: c_name(:)
        integer(c_int32_t) :: symbol_id
        integer :: i

        call to_c_chars(trim(name), c_name)
        symbol_id = lr_session_intern(handle, c_name)
        if (symbol_id < 0_c_int32_t .or. size(args) > 8) then
            call clear_liric_error(error)
            error%code = 1_c_int
            return
        end if

        operands(1) = global_operand(handle, symbol_id)
        do i = 1, size(args)
            operands(i + 1) = args(i)
        end do

        inst%op = LR_OP_CALL
        inst%typ = lr_type_void_s(handle)
        inst%dest = 0_c_int32_t
        inst%operands = c_loc(operands)
        inst%num_operands = int(size(args) + 1, c_int32_t)
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
    end function emit_call_void

    logical function emit_i32_indirect_call(session, callee_ptr, args, result, &
                                            error_msg)
        ! Indirect i32 call: callee_ptr is a loaded function pointer (ptr vreg).
        ! First operand of LR_OP_CALL is the ptr vreg, not a global symbol.
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: callee_ptr
        type(lr_operand_desc_t), intent(in) :: args(:)
        type(lr_operand_desc_t), intent(out) :: result
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t), target :: operands(9)
        type(lr_inst_desc_t) :: inst
        type(lr_error_t) :: error
        integer(c_int32_t) :: vreg
        integer :: i

        emit_i32_indirect_call = .false.
        if (.not. require_open_session(session, error_msg)) return
        if (size(args) > 8) then
            error_msg = 'indirect i32 call has too many arguments'
            return
        end if
        operands(1) = callee_ptr
        do i = 1, size(args)
            operands(i + 1) = args(i)
        end do
        inst%op = LR_OP_CALL
        inst%typ = lr_type_i32_s(session%handle)
        inst%dest = 0_c_int32_t
        inst%operands = c_loc(operands)
        inst%num_operands = int(size(args) + 1, c_int32_t)
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
        result = i32_vreg(session, vreg)
        call set_empty(error_msg)
        emit_i32_indirect_call = .true.
    end function emit_i32_indirect_call

    logical function emit_void_indirect_call(session, callee_ptr, args, error_msg)
        ! Indirect void call: callee_ptr is a loaded function pointer (ptr vreg).
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: callee_ptr
        type(lr_operand_desc_t), intent(in) :: args(:)
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t), target :: operands(9)
        type(lr_inst_desc_t) :: inst
        type(lr_error_t) :: error
        integer(c_int32_t) :: unused_vreg
        integer :: i

        emit_void_indirect_call = .false.
        if (.not. require_open_session(session, error_msg)) return
        if (size(args) > 8) then
            error_msg = 'indirect void call has too many arguments'
            return
        end if
        operands(1) = callee_ptr
        do i = 1, size(args)
            operands(i + 1) = args(i)
        end do
        inst%op = LR_OP_CALL
        inst%typ = lr_type_void_s(session%handle)
        inst%dest = 0_c_int32_t
        inst%operands = c_loc(operands)
        inst%num_operands = int(size(args) + 1, c_int32_t)
        inst%indices = c_null_ptr
        inst%num_indices = 0_c_int32_t
        inst%align = 0_c_int32_t
        inst%icmp_pred = 0_c_int
        inst%fcmp_pred = 0_c_int
        inst%call_external_abi = c_false
        inst%call_vararg = c_false
        inst%call_fixed_args = 0_c_int32_t
        call clear_liric_error(error)
        unused_vreg = lr_session_emit(session%handle, inst, error)
        if (.not. status_ok(error%code, error, error_msg)) return
        call set_empty(error_msg)
        emit_void_indirect_call = .true.
    end function emit_void_indirect_call

    function global_operand(handle, symbol_id) result(operand)
        type(c_ptr), intent(in) :: handle
        integer(c_int32_t), intent(in) :: symbol_id
        type(lr_operand_desc_t) :: operand

        operand%kind = LR_OP_KIND_GLOBAL
        operand%payload = int(symbol_id, c_int64_t)
        operand%typ = lr_type_i32_s(handle)
        operand%global_offset = 0_c_int64_t
    end function global_operand

    function session_i32_type(session, error_msg) result(i32_type)
        type(liric_session_t), intent(in) :: session
        character(len=:), allocatable, intent(out) :: error_msg
        type(c_ptr) :: i32_type

        i32_type = lr_type_i32_s(session%handle)
        if (c_associated(i32_type)) then
            call set_empty(error_msg)
        else
            error_msg = 'LIRIC did not return an i32 type'
        end if
    end function session_i32_type

    function i32_immediate(session, value) result(operand)
        type(liric_session_t), intent(in) :: session
        integer(c_int64_t), intent(in) :: value
        type(lr_operand_desc_t) :: operand

        operand%kind = LR_OP_KIND_IMM_I64
        operand%payload = value
        operand%typ = lr_type_i32_s(session%handle)
        operand%global_offset = 0_c_int64_t
    end function i32_immediate

    function i32_vreg(session, vreg) result(operand)
        type(liric_session_t), intent(in) :: session
        integer(c_int32_t), intent(in) :: vreg
        type(lr_operand_desc_t) :: operand

        operand%kind = LR_OP_KIND_VREG
        operand%payload = int(vreg, c_int64_t)
        operand%typ = lr_type_i32_s(session%handle)
        operand%global_offset = 0_c_int64_t
    end function i32_vreg

end module liric_session_bindings
