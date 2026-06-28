program test_cli_version_help
    use ffc_cli_options, only: cli_options_t, parse_arguments, CLI_PATH_LEN
    implicit none

    call check_version()
    call check_help()
    call check_help_short()
    call check_normal_parse()

    print *, 'PASS'
    stop 0

contains

    subroutine check_version()
        type(cli_options_t) :: opts
        character(len=CLI_PATH_LEN) :: argv(1)

        argv(1) = '--version'
        call parse_arguments(argv, opts)
        if (opts%error) then
            print *, 'FAIL [version]: ', opts%error_message
            stop 1
        end if
        if (.not. opts%show_version) then
            print *, 'FAIL [version]: show_version not set'
            stop 1
        end if
        if (opts%show_help) then
            print *, 'FAIL [version]: show_help should be false'
            stop 1
        end if
    end subroutine check_version

    subroutine check_help()
        type(cli_options_t) :: opts
        character(len=CLI_PATH_LEN) :: argv(1)

        argv(1) = '--help'
        call parse_arguments(argv, opts)
        if (opts%error) then
            print *, 'FAIL [help]: ', opts%error_message
            stop 1
        end if
        if (.not. opts%show_help) then
            print *, 'FAIL [help]: show_help not set'
            stop 1
        end if
        if (opts%show_version) then
            print *, 'FAIL [help]: show_version should be false'
            stop 1
        end if
    end subroutine check_help

    subroutine check_help_short()
        type(cli_options_t) :: opts
        character(len=CLI_PATH_LEN) :: argv(1)

        argv(1) = '-h'
        call parse_arguments(argv, opts)
        if (opts%error) then
            print *, 'FAIL [-h]: ', opts%error_message
            stop 1
        end if
        if (.not. opts%show_help) then
            print *, 'FAIL [-h]: show_help not set'
            stop 1
        end if
        if (opts%show_version) then
            print *, 'FAIL [-h]: show_version should be false'
            stop 1
        end if
    end subroutine check_help_short

    subroutine check_normal_parse()
        type(cli_options_t) :: opts
        character(len=CLI_PATH_LEN) :: argv(2)

        argv(1) = '-c'
        argv(2) = 'foo.f90'
        call parse_arguments(argv, opts)
        if (opts%error) then
            print *, 'FAIL [normal]: ', opts%error_message
            stop 1
        end if
        if (opts%show_version) then
            print *, 'FAIL [normal]: show_version should be false'
            stop 1
        end if
        if (opts%show_help) then
            print *, 'FAIL [normal]: show_help should be false'
            stop 1
        end if
        if (.not. opts%emit_object) then
            print *, 'FAIL [normal]: emit_object should be true'
            stop 1
        end if
        if (trim(opts%input_file) /= 'foo.f90') then
            print *, 'FAIL [normal]: input_file should be foo.f90, got ', &
                trim(opts%input_file)
            stop 1
        end if
    end subroutine check_normal_parse

end program test_cli_version_help
