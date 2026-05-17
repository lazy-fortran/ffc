module error_handling
    use logger, only: log_error, log_warn
    implicit none
    private

    public :: error_result_t, success_result_t
    public :: make_error, make_success, has_error
    public :: report_error, report_warning

    ! Result type for operations that can fail
    type :: error_result_t
        logical :: success = .false.
        character(len=:), allocatable :: message
    contains
        procedure :: is_success => error_result_is_success
        procedure :: get_message => error_result_get_message
        procedure :: clear => error_result_clear
    end type error_result_t

    type :: success_result_t
        logical :: success = .true.
        character(len=:), allocatable :: message
    end type success_result_t

contains

    ! Create error result with message
    function make_error(message) result(error)
        character(len=*), intent(in) :: message
        type(error_result_t) :: error
        error%success = .false.
        error%message = message
        call log_error(message)
    end function make_error

    ! Create success result 
    function make_success(message) result(success)
        character(len=*), intent(in), optional :: message
        type(error_result_t) :: success
        success%success = .true.
        if (present(message)) then
            success%message = message
        else
            success%message = "Operation completed successfully"
        end if
    end function make_success

    ! Check if result has error
    logical function has_error(result)
        type(error_result_t), intent(in) :: result
        has_error = .not. result%success
    end function has_error

    ! Report error to both logger and output parameter
    subroutine report_error(message, error_msg)
        character(len=*), intent(in) :: message
        character(len=*), intent(out) :: error_msg
        call log_error(message)
        error_msg = message
    end subroutine report_error

    ! Report warning to both logger and output parameter
    subroutine report_warning(message, error_msg)
        character(len=*), intent(in) :: message
        character(len=*), intent(out) :: error_msg
        call log_warn(message)
        error_msg = message
    end subroutine report_warning

    ! Error result methods
    logical function error_result_is_success(this)
        class(error_result_t), intent(in) :: this
        error_result_is_success = this%success
    end function error_result_is_success

    function error_result_get_message(this) result(message)
        class(error_result_t), intent(in) :: this
        character(len=:), allocatable :: message
        if (allocated(this%message)) then
            message = this%message
        else
            message = ""
        end if
    end function error_result_get_message

    subroutine error_result_clear(this)
        class(error_result_t), intent(inout) :: this
        this%success = .true.
        if (allocated(this%message)) deallocate(this%message)
    end subroutine error_result_clear

end module error_handling