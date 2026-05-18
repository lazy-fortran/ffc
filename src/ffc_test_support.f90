module ffc_test_support
    ! Behavioural test helpers shared across test/test_*.f90.  Each helper
    ! drives FortFront + the direct-session lowerer end to end so individual
    ! test programs only describe the source they want compiled and the
    ! expected behaviour of the resulting executable.
    use fortfront, only: compiler_frontend_options_t, &
                         compiler_frontend_result_t, &
                         compile_frontend_from_string, INPUT_MODE_STANDARD
    use session_program_lowering, only: lower_program_to_liric_exe
    implicit none
    private

    public :: expect_exit_status
    public :: expect_output
    public :: expect_error_contains

contains

    logical function expect_exit_status(source, expected, exe_path) result(ok)
        character(len=*), intent(in) :: source
        integer, intent(in) :: expected
        character(len=*), intent(in) :: exe_path
        character(len=:), allocatable :: error_msg
        integer :: exit_stat, cmd_stat

        ok = .false.
        call compile_to_exe(source, exe_path, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL: ', trim(error_msg)
            call execute_command_line('rm -f '//exe_path)
            return
        end if

        call execute_command_line(exe_path, exitstat=exit_stat, cmdstat=cmd_stat)
        call execute_command_line('rm -f '//exe_path)
        if (cmd_stat /= 0) then
            print *, 'FAIL: emitted executable did not run: ', exe_path
            return
        end if
        if (exit_stat /= expected) then
            print *, 'FAIL: exit status ', exit_stat, ' expected ', expected
            return
        end if
        ok = .true.
    end function expect_exit_status

    logical function expect_output(source, expected, exe_path) result(ok)
        character(len=*), intent(in) :: source
        character(len=*), intent(in) :: expected
        character(len=*), intent(in) :: exe_path
        character(len=:), allocatable :: error_msg
        character(len=:), allocatable :: actual
        integer :: cmd_stat
        character(len=:), allocatable :: stdout_path

        ok = .false.
        call compile_to_exe(source, exe_path, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL: ', trim(error_msg)
            call execute_command_line('rm -f '//exe_path)
            return
        end if

        stdout_path = exe_path//'.out'
        call execute_command_line(exe_path//' > '//stdout_path, &
                                  cmdstat=cmd_stat)
        if (cmd_stat /= 0) then
            print *, 'FAIL: emitted executable did not run: ', exe_path
            call execute_command_line('rm -f '//exe_path//' '//stdout_path)
            return
        end if

        call read_file(stdout_path, actual)
        call execute_command_line('rm -f '//exe_path//' '//stdout_path)
        if (.not. allocated(actual)) then
            print *, 'FAIL: could not read output of ', exe_path
            return
        end if
        if (actual /= expected) then
            print *, 'FAIL: output mismatch'
            print *, '  expected: ', expected
            print *, '  actual:   ', actual
            return
        end if
        ok = .true.
    end function expect_output

    logical function expect_error_contains(source, fragment, exe_path) result(ok)
        character(len=*), intent(in) :: source
        character(len=*), intent(in) :: fragment
        character(len=*), intent(in) :: exe_path
        character(len=:), allocatable :: error_msg

        ok = .false.
        call compile_to_exe(source, exe_path, error_msg)
        call execute_command_line('rm -f '//exe_path)
        if (len_trim(error_msg) == 0) then
            print *, 'FAIL: unsupported source lowered without diagnostic'
            return
        end if
        if (index(error_msg, fragment) == 0) then
            print *, 'FAIL: diagnostic did not contain "', fragment, '"'
            print *, '  actual: ', trim(error_msg)
            return
        end if
        ok = .true.
    end function expect_error_contains

    subroutine compile_to_exe(source, exe_path, error_msg)
        character(len=*), intent(in) :: source
        character(len=*), intent(in) :: exe_path
        character(len=:), allocatable, intent(out) :: error_msg
        type(compiler_frontend_options_t) :: options
        type(compiler_frontend_result_t) :: frontend_result

        options = compiler_frontend_options_t()
        options%run_semantics = .true.
        options%input_mode = INPUT_MODE_STANDARD

        call compile_frontend_from_string(source, frontend_result, options)
        if (.not. frontend_result%success()) then
            error_msg = 'FortFront rejected source: '// &
                        trim(frontend_result%diagnostic_text)
            return
        end if

        call execute_command_line('rm -f '//exe_path)
        call lower_program_to_liric_exe(frontend_result%arena, &
                                        frontend_result%root_index, exe_path, &
                                        error_msg)
    end subroutine compile_to_exe

    subroutine read_file(path, contents)
        character(len=*), intent(in) :: path
        character(len=:), allocatable, intent(out) :: contents
        integer :: unit, io_stat
        character(len=4096) :: buffer
        integer :: bytes

        open (newunit=unit, file=path, status='old', action='read', &
              access='stream', form='unformatted', iostat=io_stat)
        if (io_stat /= 0) return
        inquire (unit=unit, size=bytes)
        if (bytes <= 0) then
            allocate (character(len=0) :: contents)
            close (unit)
            return
        end if
        if (bytes > len(buffer)) bytes = len(buffer)
        read (unit, iostat=io_stat) buffer(1:bytes)
        close (unit)
        if (io_stat /= 0) return
        contents = buffer(1:bytes)
    end subroutine read_file

end module ffc_test_support
