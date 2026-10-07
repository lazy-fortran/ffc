! fo: dispatcher
module ffc_case_test_session_cli_backend
    implicit none
    private
    public :: case_test_session_cli_backend
    interface
        subroutine case_test_session_cli_backend()
        end subroutine case_test_session_cli_backend
    end interface
end module ffc_case_test_session_cli_backend

subroutine case_test_session_cli_backend()
    use ffc_cli_options, only: cli_options_t, parse_arguments, CLI_PATH_LEN
    implicit none
    save

    call check_implicit_backend()
    call check_named('isel', 1)
    call check_named('copy-patch', 2)
    call check_named('llvm', 3)
    call check_rejected_default()
    call check_unknown()
    call check_missing_value()

    print *, 'PASS: CLI parses --backend selection'

contains

    subroutine check_implicit_backend()
        type(cli_options_t) :: opts
        character(len=CLI_PATH_LEN) :: argv(1)

        argv(1) = 'test.f90'
        call parse_arguments(argv, opts)
        if (opts%error) then
            print *, 'FAIL [implicit backend]: ', opts%error_message
            stop 1
        end if
        if (opts%backend /= 2) then
            print *, 'FAIL [implicit backend]: backend=', opts%backend
            stop 1
        end if
    end subroutine check_implicit_backend

    subroutine check_rejected_default()
        type(cli_options_t) :: opts
        character(len=CLI_PATH_LEN) :: argv(3)

        argv = [character(len=CLI_PATH_LEN) :: 'test.f90', '--backend', 'default']
        call parse_arguments(argv, opts)
        if (.not. opts%error) then
            print *, 'FAIL [default]: obsolete backend was accepted'
            stop 1
        end if
    end subroutine check_rejected_default

    subroutine check_named(name, expected)
        character(len=*), intent(in) :: name
        integer, intent(in) :: expected
        type(cli_options_t) :: opts
        character(len=CLI_PATH_LEN) :: argv(3)

        argv(1) = 'test.f90'
        argv(2) = '--backend'
        argv(3) = name
        call parse_arguments(argv, opts)
        if (opts%error) then
            print *, 'FAIL ['//name//']: ', opts%error_message
            stop 1
        end if
        if (opts%backend /= expected) then
            print *, 'FAIL ['//name//']: backend=', opts%backend
            stop 1
        end if
    end subroutine check_named

    subroutine check_unknown()
        type(cli_options_t) :: opts
        character(len=CLI_PATH_LEN) :: argv(3)

        argv(1) = 'test.f90'
        argv(2) = '--backend'
        argv(3) = 'nonsense'
        call parse_arguments(argv, opts)
        if (.not. opts%error) then
            print *, 'FAIL [unknown]: expected an error'
            stop 1
        end if
    end subroutine check_unknown

    subroutine check_missing_value()
        type(cli_options_t) :: opts
        character(len=CLI_PATH_LEN) :: argv(2)

        argv(1) = 'test.f90'
        argv(2) = '--backend'
        call parse_arguments(argv, opts)
        if (.not. opts%error) then
            print *, 'FAIL [missing]: expected an error'
            stop 1
        end if
    end subroutine check_missing_value

end subroutine case_test_session_cli_backend
