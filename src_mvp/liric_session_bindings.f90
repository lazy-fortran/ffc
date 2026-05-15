module liric_session_bindings
    use, intrinsic :: iso_c_binding, only: c_associated, c_bool, c_char
    use, intrinsic :: iso_c_binding, only: c_int, c_int32_t, c_int64_t
    use, intrinsic :: iso_c_binding, only: c_loc, c_null_char, c_null_ptr, c_ptr
    implicit none
    private

    integer(c_int), parameter, public :: LR_MODE_DIRECT = 0_c_int
    integer(c_int), parameter, public :: LR_MODE_IR = 1_c_int
    integer(c_int), parameter, public :: LR_SESSION_BACKEND_DEFAULT = 0_c_int
    integer(c_int), parameter, public :: LR_SESSION_BACKEND_ISEL = 1_c_int
    integer(c_int), parameter, public :: LR_SESSION_BACKEND_COPY_PATCH = 2_c_int
    integer(c_int), parameter, public :: LR_SESSION_BACKEND_LLVM = 3_c_int
    integer(c_int), parameter, public :: LR_OK = 0_c_int
    logical(c_bool), parameter :: c_false = .false.

    integer(c_int), parameter, public :: LR_OP_RET = 0_c_int
    integer(c_int), parameter, public :: LR_OP_RET_VOID = 1_c_int
    integer(c_int), parameter, public :: LR_OP_ADD = 5_c_int
    integer(c_int), parameter, public :: LR_OP_SUB = 6_c_int
    integer(c_int), parameter, public :: LR_OP_MUL = 7_c_int
    integer(c_int), parameter, public :: LR_OP_SDIV = 8_c_int
    integer(c_int), parameter, public :: LR_OP_CALL = 30_c_int

    integer(c_int), parameter, public :: LR_OP_KIND_VREG = 0_c_int
    integer(c_int), parameter, public :: LR_OP_KIND_IMM_I64 = 1_c_int
    integer(c_int), parameter, public :: LR_OP_KIND_BLOCK = 3_c_int
    integer(c_int), parameter, public :: LR_OP_KIND_GLOBAL = 4_c_int

    type, bind(c), public :: lr_session_config_t
        integer(c_int) :: mode = LR_MODE_DIRECT
        type(c_ptr) :: target = c_null_ptr
        integer(c_int) :: backend = LR_SESSION_BACKEND_DEFAULT
    end type lr_session_config_t

    type, bind(c), public :: lr_error_t
        integer(c_int) :: code = LR_OK
        character(kind=c_char) :: msg(256)
    end type lr_error_t

    type, bind(c), public :: lr_operand_desc_t
        integer(c_int) :: kind = LR_OP_KIND_IMM_I64
        integer(c_int64_t) :: payload = 0_c_int64_t
        type(c_ptr) :: typ = c_null_ptr
        integer(c_int64_t) :: global_offset = 0_c_int64_t
    end type lr_operand_desc_t

    type, bind(c), public :: lr_inst_desc_t
        integer(c_int) :: op = LR_OP_RET
        type(c_ptr) :: typ = c_null_ptr
        integer(c_int32_t) :: dest = 0_c_int32_t
        type(c_ptr) :: operands = c_null_ptr
        integer(c_int32_t) :: num_operands = 0_c_int32_t
        type(c_ptr) :: indices = c_null_ptr
        integer(c_int32_t) :: num_indices = 0_c_int32_t
        integer(c_int32_t) :: align = 0_c_int32_t
        integer(c_int) :: icmp_pred = 0_c_int
        integer(c_int) :: fcmp_pred = 0_c_int
        logical(c_bool) :: call_external_abi = .false.
        logical(c_bool) :: call_vararg = .false.
        integer(c_int32_t) :: call_fixed_args = 0_c_int32_t
    end type lr_inst_desc_t

    type, public :: liric_session_t
        type(c_ptr) :: handle = c_null_ptr
    contains
        procedure :: destroy => liric_session_destroy
        procedure :: is_open => liric_session_is_open
        procedure :: begin_i32_main
        procedure :: begin_i32_function
        procedure :: begin_void_subroutine
        procedure :: emit_i32_binary
        procedure :: emit_i32_call
        procedure :: emit_ret_i32_main_exe
        procedure :: emit_ret_i32_operand
        procedure :: emit_ret_void
        procedure :: emit_void_call
        procedure :: finish_function
        procedure :: finish_and_emit_object
        procedure :: finish_and_emit_exe
        procedure :: i32_param
        procedure :: i32_immediate
        procedure :: i32_vreg
    end type liric_session_t

    public :: liric_session_create
    public :: liric_session_error_message

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

        function lr_type_void_s(handle) result(typ) bind(c)
            import :: c_ptr
            type(c_ptr), value :: handle
            type(c_ptr) :: typ
        end function lr_type_void_s

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

        function lr_session_intern(handle, name) result(symbol_id) bind(c)
            import :: c_char, c_int32_t, c_ptr
            type(c_ptr), value :: handle
            character(kind=c_char), intent(in) :: name(*)
            integer(c_int32_t) :: symbol_id
        end function lr_session_intern

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

    subroutine liric_session_destroy(this)
        class(liric_session_t), intent(inout) :: this

        if (c_associated(this%handle)) then
            call lr_session_destroy(this%handle)
            this%handle = c_null_ptr
        end if
    end subroutine liric_session_destroy

    logical function liric_session_is_open(this)
        class(liric_session_t), intent(in) :: this

        liric_session_is_open = c_associated(this%handle)
    end function liric_session_is_open

    logical function emit_ret_i32_main_exe(this, return_code, path, error_msg)
        class(liric_session_t), intent(inout) :: this
        integer, intent(in) :: return_code
        character(len=*), intent(in) :: path
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: value

        if (.not. this%begin_i32_main(error_msg)) then
            emit_ret_i32_main_exe = .false.
            return
        end if

        value = this%i32_immediate(int(return_code, c_int64_t))
        if (.not. this%emit_ret_i32_operand(value, error_msg)) then
            emit_ret_i32_main_exe = .false.
            return
        end if

        emit_ret_i32_main_exe = this%finish_and_emit_exe(path, error_msg)
    end function emit_ret_i32_main_exe

    logical function begin_i32_main(this, error_msg)
        class(liric_session_t), intent(inout) :: this
        character(len=:), allocatable, intent(out) :: error_msg
        character(kind=c_char), allocatable :: c_name(:)
        type(c_ptr) :: i32_type
        type(lr_error_t) :: error
        integer(c_int32_t) :: block_id
        integer(c_int) :: status

        begin_i32_main = .false.
        if (.not. require_open_session(this, error_msg)) return

        i32_type = session_i32_type(this, error_msg)
        if (len_trim(error_msg) > 0) return

        call clear_liric_error(error)
        call to_c_chars('main', c_name)
        status = lr_session_func_begin(this%handle, c_name, i32_type, &
                                       c_null_ptr, 0_c_int32_t, c_false, &
                                       error)
        if (.not. status_ok(status, error, error_msg)) return

        block_id = lr_session_block(this%handle)
        status = lr_session_set_block(this%handle, block_id, error)
        if (.not. status_ok(status, error, error_msg)) return

        call set_empty(error_msg)
        begin_i32_main = .true.
    end function begin_i32_main

    logical function begin_i32_function(this, name, param_count, error_msg)
        class(liric_session_t), intent(inout) :: this
        character(len=*), intent(in) :: name
        integer, intent(in) :: param_count
        character(len=:), allocatable, intent(out) :: error_msg
        character(kind=c_char), allocatable :: c_name(:)
        type(c_ptr), allocatable, target :: params(:)
        type(c_ptr) :: i32_type
        type(c_ptr) :: params_ptr
        type(lr_error_t) :: error
        integer(c_int32_t) :: block_id
        integer(c_int) :: status
        integer :: i

        begin_i32_function = .false.
        if (.not. require_open_session(this, error_msg)) return

        i32_type = session_i32_type(this, error_msg)
        if (len_trim(error_msg) > 0) return

        params_ptr = c_null_ptr
        if (param_count > 0) then
            allocate (params(param_count))
            do i = 1, param_count
                params(i) = i32_type
            end do
            params_ptr = c_loc(params)
        end if

        call clear_liric_error(error)
        call to_c_chars(trim(name), c_name)
        status = lr_session_func_begin(this%handle, c_name, i32_type, &
                                       params_ptr, int(param_count, c_int32_t), &
                                       c_false, error)
        if (.not. status_ok(status, error, error_msg)) return

        block_id = lr_session_block(this%handle)
        status = lr_session_set_block(this%handle, block_id, error)
        if (.not. status_ok(status, error, error_msg)) return

        call set_empty(error_msg)
        begin_i32_function = .true.
    end function begin_i32_function

    logical function begin_void_subroutine(this, name, param_count, error_msg)
        class(liric_session_t), intent(inout) :: this
        character(len=*), intent(in) :: name
        integer, intent(in) :: param_count
        character(len=:), allocatable, intent(out) :: error_msg
        character(kind=c_char), allocatable :: c_name(:)
        type(c_ptr), allocatable, target :: params(:)
        type(c_ptr) :: i32_type
        type(c_ptr) :: params_ptr
        type(c_ptr) :: void_type
        type(lr_error_t) :: error
        integer(c_int32_t) :: block_id
        integer(c_int) :: status
        integer :: i

        begin_void_subroutine = .false.
        if (.not. require_open_session(this, error_msg)) return

        i32_type = session_i32_type(this, error_msg)
        if (len_trim(error_msg) > 0) return
        void_type = lr_type_void_s(this%handle)
        if (.not. c_associated(void_type)) then
            error_msg = 'LIRIC did not return a void type'
            return
        end if

        params_ptr = c_null_ptr
        if (param_count > 0) then
            allocate (params(param_count))
            do i = 1, param_count
                params(i) = i32_type
            end do
            params_ptr = c_loc(params)
        end if

        call clear_liric_error(error)
        call to_c_chars(trim(name), c_name)
        status = lr_session_func_begin(this%handle, c_name, void_type, &
                                       params_ptr, int(param_count, c_int32_t), &
                                       c_false, error)
        if (.not. status_ok(status, error, error_msg)) return

        block_id = lr_session_block(this%handle)
        status = lr_session_set_block(this%handle, block_id, error)
        if (.not. status_ok(status, error, error_msg)) return

        call set_empty(error_msg)
        begin_void_subroutine = .true.
    end function begin_void_subroutine

    logical function emit_ret_i32_operand(this, value, error_msg)
        class(liric_session_t), intent(inout) :: this
        type(lr_operand_desc_t), intent(in) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_error_t) :: error
        integer(c_int32_t) :: unused_vreg

        emit_ret_i32_operand = .false.
        if (.not. require_open_session(this, error_msg)) return

        unused_vreg = emit_ret_i32(this%handle, value, error)
        if (.not. status_ok(error%code, error, error_msg)) return

        call set_empty(error_msg)
        emit_ret_i32_operand = .true.
    end function emit_ret_i32_operand

    logical function emit_ret_void(this, error_msg)
        class(liric_session_t), intent(inout) :: this
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_error_t) :: error
        integer(c_int32_t) :: unused_vreg

        emit_ret_void = .false.
        if (.not. require_open_session(this, error_msg)) return

        unused_vreg = emit_ret_void_inst(this%handle, error)
        if (.not. status_ok(error%code, error, error_msg)) return

        call set_empty(error_msg)
        emit_ret_void = .true.
    end function emit_ret_void

    logical function finish_and_emit_exe(this, path, error_msg)
        class(liric_session_t), intent(inout) :: this
        character(len=*), intent(in) :: path
        character(len=:), allocatable, intent(out) :: error_msg
        character(kind=c_char), allocatable :: c_path(:)
        type(c_ptr) :: out_addr
        type(lr_error_t) :: error
        integer(c_int) :: status

        finish_and_emit_exe = .false.
        if (.not. require_open_session(this, error_msg)) return

        call clear_liric_error(error)
        out_addr = c_null_ptr
        status = lr_session_func_end(this%handle, out_addr, error)
        if (.not. status_ok(status, error, error_msg)) return

        call to_c_chars(path, c_path)
        status = lr_session_emit_exe(this%handle, c_path, error)
        if (.not. status_ok(status, error, error_msg)) return

        call set_empty(error_msg)
        finish_and_emit_exe = .true.
    end function finish_and_emit_exe

    logical function finish_and_emit_object(this, path, error_msg)
        class(liric_session_t), intent(inout) :: this
        character(len=*), intent(in) :: path
        character(len=:), allocatable, intent(out) :: error_msg
        character(kind=c_char), allocatable :: c_path(:)
        type(c_ptr) :: out_addr
        type(lr_error_t) :: error
        integer(c_int) :: status

        finish_and_emit_object = .false.
        if (.not. require_open_session(this, error_msg)) return

        call clear_liric_error(error)
        out_addr = c_null_ptr
        status = lr_session_func_end(this%handle, out_addr, error)
        if (.not. status_ok(status, error, error_msg)) return

        call to_c_chars(path, c_path)
        status = lr_session_emit_object(this%handle, c_path, error)
        if (.not. status_ok(status, error, error_msg)) return

        call set_empty(error_msg)
        finish_and_emit_object = .true.
    end function finish_and_emit_object

    logical function finish_function(this, error_msg)
        class(liric_session_t), intent(inout) :: this
        character(len=:), allocatable, intent(out) :: error_msg
        type(c_ptr) :: out_addr
        type(lr_error_t) :: error
        integer(c_int) :: status

        finish_function = .false.
        if (.not. require_open_session(this, error_msg)) return

        call clear_liric_error(error)
        out_addr = c_null_ptr
        status = lr_session_func_end(this%handle, out_addr, error)
        if (.not. status_ok(status, error, error_msg)) return

        call set_empty(error_msg)
        finish_function = .true.
    end function finish_function

    function i32_immediate(this, value) result(operand)
        class(liric_session_t), intent(in) :: this
        integer(c_int64_t), intent(in) :: value
        type(lr_operand_desc_t) :: operand

        operand%kind = LR_OP_KIND_IMM_I64
        operand%payload = value
        operand%typ = lr_type_i32_s(this%handle)
        operand%global_offset = 0_c_int64_t
    end function i32_immediate

    function i32_vreg(this, vreg) result(operand)
        class(liric_session_t), intent(in) :: this
        integer(c_int32_t), intent(in) :: vreg
        type(lr_operand_desc_t) :: operand

        operand%kind = LR_OP_KIND_VREG
        operand%payload = int(vreg, c_int64_t)
        operand%typ = lr_type_i32_s(this%handle)
        operand%global_offset = 0_c_int64_t
    end function i32_vreg

    function i32_param(this, index) result(operand)
        class(liric_session_t), intent(in) :: this
        integer, intent(in) :: index
        type(lr_operand_desc_t) :: operand
        integer(c_int32_t) :: vreg

        vreg = lr_session_param(this%handle, int(index, c_int32_t))
        operand = this%i32_vreg(vreg)
    end function i32_param

    logical function emit_i32_binary(this, opcode, lhs, rhs, result, error_msg)
        class(liric_session_t), intent(inout) :: this
        integer(c_int), intent(in) :: opcode
        type(lr_operand_desc_t), intent(in) :: lhs
        type(lr_operand_desc_t), intent(in) :: rhs
        type(lr_operand_desc_t), intent(out) :: result
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_error_t) :: error
        integer(c_int32_t) :: vreg

        emit_i32_binary = .false.
        if (.not. require_open_session(this, error_msg)) return

        vreg = emit_binary(this%handle, opcode, lhs, rhs, error)
        if (.not. status_ok(error%code, error, error_msg)) return

        result = this%i32_vreg(vreg)
        call set_empty(error_msg)
        emit_i32_binary = .true.
    end function emit_i32_binary

    logical function emit_i32_call(this, name, args, result, error_msg)
        class(liric_session_t), intent(inout) :: this
        character(len=*), intent(in) :: name
        type(lr_operand_desc_t), intent(in) :: args(:)
        type(lr_operand_desc_t), intent(out) :: result
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_error_t) :: error
        integer(c_int32_t) :: vreg

        emit_i32_call = .false.
        if (.not. require_open_session(this, error_msg)) return

        vreg = emit_call_i32(this%handle, name, args, error)
        if (.not. status_ok(error%code, error, error_msg)) return

        result = this%i32_vreg(vreg)
        call set_empty(error_msg)
        emit_i32_call = .true.
    end function emit_i32_call

    logical function emit_void_call(this, name, args, error_msg)
        class(liric_session_t), intent(inout) :: this
        character(len=*), intent(in) :: name
        type(lr_operand_desc_t), intent(in) :: args(:)
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_error_t) :: error
        integer(c_int32_t) :: unused_vreg

        emit_void_call = .false.
        if (.not. require_open_session(this, error_msg)) return

        unused_vreg = emit_call_void(this%handle, name, args, error)
        if (.not. status_ok(error%code, error, error_msg)) return

        call set_empty(error_msg)
        emit_void_call = .true.
    end function emit_void_call

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

    function global_operand(handle, symbol_id) result(operand)
        type(c_ptr), intent(in) :: handle
        integer(c_int32_t), intent(in) :: symbol_id
        type(lr_operand_desc_t) :: operand

        operand%kind = LR_OP_KIND_GLOBAL
        operand%payload = int(symbol_id, c_int64_t)
        operand%typ = lr_type_i32_s(handle)
        operand%global_offset = 0_c_int64_t
    end function global_operand

    logical function require_open_session(this, error_msg)
        class(liric_session_t), intent(in) :: this
        character(len=:), allocatable, intent(out) :: error_msg

        require_open_session = c_associated(this%handle)
        if (require_open_session) then
            call set_empty(error_msg)
        else
            error_msg = 'LIRIC session handle is not open'
        end if
    end function require_open_session

    function session_i32_type(this, error_msg) result(i32_type)
        class(liric_session_t), intent(in) :: this
        character(len=:), allocatable, intent(out) :: error_msg
        type(c_ptr) :: i32_type

        i32_type = lr_type_i32_s(this%handle)
        if (c_associated(i32_type)) then
            call set_empty(error_msg)
        else
            error_msg = 'LIRIC did not return an i32 type'
        end if
    end function session_i32_type

    logical function status_ok(status, error, error_msg)
        integer(c_int), intent(in) :: status
        type(lr_error_t), intent(in) :: error
        character(len=:), allocatable, intent(out) :: error_msg

        status_ok = status == LR_OK
        if (status_ok) then
            call set_empty(error_msg)
        else
            error_msg = liric_session_error_message(error)
        end if
    end function status_ok

    function liric_session_error_message(error) result(message)
        type(lr_error_t), intent(in) :: error
        character(len=:), allocatable :: message
        integer :: i
        integer :: message_len

        message_len = 0
        do i = 1, size(error%msg)
            if (error%msg(i) == c_null_char) exit
            message_len = message_len + 1
        end do

        if (message_len == 0) then
            allocate (character(len=32) :: message)
            write (message, '(A,I0)') 'LIRIC error code ', error%code
            return
        end if

        allocate (character(len=message_len) :: message)
        do i = 1, message_len
            message(i:i) = error%msg(i)
        end do
    end function liric_session_error_message

    subroutine clear_liric_error(error)
        type(lr_error_t), intent(out) :: error

        error%code = LR_OK
        error%msg = c_null_char
    end subroutine clear_liric_error

    subroutine to_c_chars(text, chars)
        character(len=*), intent(in) :: text
        character(kind=c_char), allocatable, intent(out) :: chars(:)
        integer :: i

        allocate (chars(len(text) + 1))
        do i = 1, len(text)
            chars(i) = text(i:i)
        end do
        chars(len(text) + 1) = c_null_char
    end subroutine to_c_chars

    subroutine set_empty(value)
        character(len=:), allocatable, intent(out) :: value

        allocate (character(len=0) :: value)
    end subroutine set_empty

end module liric_session_bindings
