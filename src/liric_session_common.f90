module liric_session_common
    use, intrinsic :: iso_c_binding, only: c_associated, c_bool, c_char
    use, intrinsic :: iso_c_binding, only: c_int, c_int32_t, c_int64_t
    use, intrinsic :: iso_c_binding, only: c_loc, c_null_char, c_null_ptr, c_ptr, &
                                              c_size_t
    implicit none
    private

    public :: lr_session_config_t, lr_error_t, lr_operand_desc_t, &
              lr_inst_desc_t, liric_session_t
    public :: LR_OK, LR_OP_KIND_VREG, LR_OP_KIND_IMM_I64, LR_OP_KIND_GLOBAL, &
              LR_OP_KIND_BLOCK
    public :: require_open_session, status_ok, liric_session_error_message, &
              clear_liric_error, to_c_chars, set_empty

    integer(c_int), parameter :: LR_OK = 0_c_int
    integer(c_int), parameter :: LR_OP_RET = 0_c_int
    integer(c_int), parameter :: LR_OP_KIND_VREG = 0_c_int
    integer(c_int), parameter :: LR_OP_KIND_IMM_I64 = 1_c_int
    integer(c_int), parameter :: LR_OP_KIND_GLOBAL = 4_c_int
    integer(c_int), parameter :: LR_OP_KIND_BLOCK = 3_c_int

    type, bind(c), public :: lr_session_config_t
        integer(c_int) :: mode = 0_c_int
        type(c_ptr) :: target = c_null_ptr
        integer(c_int) :: backend = 0_c_int
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
     end type liric_session_t

contains

    logical function require_open_session(session, error_msg)
        type(liric_session_t), intent(in) :: session
        character(len=:), allocatable, intent(out) :: error_msg

        require_open_session = c_associated(session%handle)
        if (require_open_session) then
            call set_empty(error_msg)
        else
            error_msg = 'LIRIC session handle is not open'
        end if
    end function require_open_session

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

end module liric_session_common
