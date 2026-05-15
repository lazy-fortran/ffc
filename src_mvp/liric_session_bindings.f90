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

    integer(c_int), parameter, public :: LR_OP_KIND_VREG = 0_c_int
    integer(c_int), parameter, public :: LR_OP_KIND_IMM_I64 = 1_c_int
    integer(c_int), parameter, public :: LR_OP_KIND_BLOCK = 3_c_int

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
        procedure :: emit_ret_i32_main_exe
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
        character(kind=c_char), allocatable :: c_name(:)
        character(kind=c_char), allocatable :: c_path(:)
        type(c_ptr) :: i32_type
        type(c_ptr) :: out_addr
        type(lr_error_t) :: error
        integer(c_int32_t) :: block_id
        integer(c_int32_t) :: unused_vreg
        integer(c_int) :: status

        emit_ret_i32_main_exe = .false.
        if (.not. c_associated(this%handle)) then
            error_msg = 'LIRIC session handle is not open'
            return
        end if

        call clear_liric_error(error)
        i32_type = lr_type_i32_s(this%handle)
        if (.not. c_associated(i32_type)) then
            error_msg = 'LIRIC did not return an i32 type'
            return
        end if

        call to_c_chars('main', c_name)
        status = lr_session_func_begin(this%handle, c_name, i32_type, &
                                       c_null_ptr, 0_c_int32_t, c_false, &
                                       error)
        if (.not. status_ok(status, error, error_msg)) return

        block_id = lr_session_block(this%handle)
        status = lr_session_set_block(this%handle, block_id, error)
        if (.not. status_ok(status, error, error_msg)) return

        unused_vreg = emit_ret_i32(this%handle, int(return_code, c_int64_t), &
                                   i32_type, error)
        if (.not. status_ok(error%code, error, error_msg)) return

        out_addr = c_null_ptr
        status = lr_session_func_end(this%handle, out_addr, error)
        if (.not. status_ok(status, error, error_msg)) return

        call to_c_chars(path, c_path)
        status = lr_session_emit_exe(this%handle, c_path, error)
        if (.not. status_ok(status, error, error_msg)) return

        call set_empty(error_msg)
        emit_ret_i32_main_exe = .true.
    end function emit_ret_i32_main_exe

    function emit_ret_i32(handle, return_code, i32_type, error) result(vreg)
        type(c_ptr), intent(in) :: handle
        integer(c_int64_t), intent(in) :: return_code
        type(c_ptr), intent(in) :: i32_type
        type(lr_error_t), intent(inout) :: error
        integer(c_int32_t) :: vreg
        type(lr_operand_desc_t), target :: operands(1)
        type(lr_inst_desc_t) :: inst

        operands(1)%kind = LR_OP_KIND_IMM_I64
        operands(1)%payload = return_code
        operands(1)%typ = i32_type
        operands(1)%global_offset = 0_c_int64_t

        inst%op = LR_OP_RET
        inst%typ = i32_type
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
