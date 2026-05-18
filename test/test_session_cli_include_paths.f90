program test_session_cli_include_paths
    use ffc_cli_options, only: cli_options_t, parse_arguments, CLI_PATH_LEN
    implicit none

    call check_single_path()
    call check_repeated_paths()
    call check_missing_value()

    print *, 'PASS: CLI accepts and stores -I include paths'

contains

    subroutine check_single_path()
        type(cli_options_t) :: opts
        character(len=CLI_PATH_LEN) :: argv(3)

        argv(1) = 'test.f90'
        argv(2) = '-I'
        argv(3) = '/tmp/x'
        call parse_arguments(argv, opts)
        if (opts%error) then
            print *, 'FAIL [single]: ', opts%error_message
            stop 1
        end if
        if (size(opts%include_paths) /= 1) then
            print *, 'FAIL [single]: include_paths size ', size(opts%include_paths)
            stop 1
        end if
        if (trim(opts%include_paths(1)) /= '/tmp/x') then
            print *, 'FAIL [single]: include_paths(1)=', trim(opts%include_paths(1))
            stop 1
        end if
    end subroutine check_single_path

    subroutine check_repeated_paths()
        type(cli_options_t) :: opts
        character(len=CLI_PATH_LEN) :: argv(5)

        argv(1) = 'test.f90'
        argv(2) = '-I'
        argv(3) = '/tmp/a'
        argv(4) = '-I'
        argv(5) = '/tmp/b'
        call parse_arguments(argv, opts)
        if (opts%error) then
            print *, 'FAIL [repeated]: ', opts%error_message
            stop 1
        end if
        if (size(opts%include_paths) /= 2) then
            print *, 'FAIL [repeated]: include_paths size ', size(opts%include_paths)
            stop 1
        end if
        if (trim(opts%include_paths(1)) /= '/tmp/a' .or. &
            trim(opts%include_paths(2)) /= '/tmp/b') then
            print *, 'FAIL [repeated]: paths mismatch'
            stop 1
        end if
    end subroutine check_repeated_paths

    subroutine check_missing_value()
        type(cli_options_t) :: opts
        character(len=CLI_PATH_LEN) :: argv(2)

        argv(1) = 'test.f90'
        argv(2) = '-I'
        call parse_arguments(argv, opts)
        if (.not. opts%error) then
            print *, 'FAIL [missing]: -I without value should error'
            stop 1
        end if
        if (index(opts%error_message, 'Missing value for -I') == 0) then
            print *, 'FAIL [missing]: message was: ', opts%error_message
            stop 1
        end if
    end subroutine check_missing_value

end program test_session_cli_include_paths
