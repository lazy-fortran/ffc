module logger
    use iso_fortran_env, only: error_unit, output_unit
    implicit none
    private

    public :: log_level_t, logger_t, global_logger
    public :: LOG_ERROR, LOG_WARN, LOG_INFO, LOG_DEBUG
    public :: log_error, log_warn, log_info, log_debug

    ! Log levels
    integer, parameter :: LOG_ERROR = 1
    integer, parameter :: LOG_WARN  = 2
    integer, parameter :: LOG_INFO  = 3
    integer, parameter :: LOG_DEBUG = 4

    type :: log_level_t
        integer :: level = LOG_INFO
        character(len=:), allocatable :: prefix
    contains
        procedure :: set_level => log_level_set
        procedure :: should_log => log_level_should_log
    end type log_level_t

    type :: logger_t
        private
        type(log_level_t) :: level
        logical :: enabled = .true.
        integer :: output_unit = output_unit
        integer :: error_unit_val = error_unit
    contains
        procedure :: set_level => logger_set_level
        procedure :: set_enabled => logger_set_enabled
        procedure :: log => logger_log
        procedure :: error => logger_error
        procedure :: warn => logger_warn
        procedure :: info => logger_info
        procedure :: debug => logger_debug
    end type logger_t

    ! Global logger instance
    type(logger_t) :: global_logger

contains

    subroutine log_level_set(this, level)
        class(log_level_t), intent(inout) :: this
        integer, intent(in) :: level
        this%level = level
        select case(level)
        case(LOG_ERROR)
            this%prefix = "ERROR"
        case(LOG_WARN)
            this%prefix = "WARN"
        case(LOG_INFO)
            this%prefix = "INFO"
        case(LOG_DEBUG)
            this%prefix = "DEBUG"
        case default
            this%prefix = "UNKNOWN"
        end select
    end subroutine log_level_set

    logical function log_level_should_log(this, level) result(should)
        class(log_level_t), intent(in) :: this
        integer, intent(in) :: level
        should = level <= this%level
    end function log_level_should_log

    subroutine logger_set_level(this, level)
        class(logger_t), intent(inout) :: this
        integer, intent(in) :: level
        call this%level%set_level(level)
    end subroutine logger_set_level

    subroutine logger_set_enabled(this, enabled)
        class(logger_t), intent(inout) :: this
        logical, intent(in) :: enabled
        this%enabled = enabled
    end subroutine logger_set_enabled

    subroutine logger_log(this, level, message)
        class(logger_t), intent(in) :: this
        integer, intent(in) :: level
        character(len=*), intent(in) :: message
        integer :: unit_to_use

        if (.not. this%enabled .or. .not. this%level%should_log(level)) return

        unit_to_use = this%output_unit
        if (level == LOG_ERROR .or. level == LOG_WARN) unit_to_use = this%error_unit_val

        write(unit_to_use, '("[", A, "] ", A)') this%level%prefix, message
    end subroutine logger_log

    subroutine logger_error(this, message)
        class(logger_t), intent(in) :: this
        character(len=*), intent(in) :: message
        call this%log(LOG_ERROR, message)
    end subroutine logger_error

    subroutine logger_warn(this, message)
        class(logger_t), intent(in) :: this
        character(len=*), intent(in) :: message
        call this%log(LOG_WARN, message)
    end subroutine logger_warn

    subroutine logger_info(this, message)
        class(logger_t), intent(in) :: this
        character(len=*), intent(in) :: message
        call this%log(LOG_INFO, message)
    end subroutine logger_info

    subroutine logger_debug(this, message)
        class(logger_t), intent(in) :: this
        character(len=*), intent(in) :: message
        call this%log(LOG_DEBUG, message)
    end subroutine logger_debug

    ! Convenience functions for global logger
    subroutine log_error(message)
        character(len=*), intent(in) :: message
        call global_logger%error(message)
    end subroutine log_error

    subroutine log_warn(message)
        character(len=*), intent(in) :: message
        call global_logger%warn(message)
    end subroutine log_warn

    subroutine log_info(message)
        character(len=*), intent(in) :: message
        call global_logger%info(message)
    end subroutine log_info

    subroutine log_debug(message)
        character(len=*), intent(in) :: message
        call global_logger%debug(message)
    end subroutine log_debug

end module logger