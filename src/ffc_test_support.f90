module ffc_test_support
    ! Behavioural test helpers shared across test/test_*.f90.  Each helper
    ! drives FortFront + the direct-session lowerer end to end so individual
    ! test programs only describe the source they want compiled and the
    ! expected behaviour of the resulting executable.
    use fortfront_compiler, only: compiler_frontend_options_t, &
        compiler_frontend_result_t, &
        compile_frontend_from_string, INPUT_MODE_STANDARD
    use session_program_lowering, only: lower_program_to_liric_exe, &
        lower_program_to_liric_object
    implicit none
    private

    public :: expect_exit_status
    public :: expect_output
    public :: expect_error_contains
    public :: expect_cli_error_contains
    public :: expect_cli_no_error
    public :: expect_object_exists
    public :: expect_no_error
    public :: expect_exe_has_symbol
    public :: expect_output_with_stdin
    public :: expect_stderr_and_exit
    public :: expect_eof_stderr_and_exit

contains

    logical function expect_stderr_and_exit(source, expected, expected_exit, &
            exe_path) result(ok)
        ! Compiles source, runs it capturing combined stdout+stderr, and checks
        ! both that combined output and the exit status. Used for STOP banners,
        ! which gfortran writes to stderr.
        character(len=*), intent(in) :: source
        character(len=*), intent(in) :: expected
        integer, intent(in) :: expected_exit
        character(len=*), intent(in) :: exe_path
        character(len=:), allocatable :: error_msg, actual, out_path
        integer :: cmd_stat, exit_stat

        ok = .false.
        call compile_to_exe(source, exe_path, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL: ', trim(error_msg)
            call execute_command_line('rm -f '//exe_path)
            return
        end if

        out_path = exe_path//'.out'
        call execute_command_line(exe_path//' > '//out_path//' 2>&1', &
            exitstat=exit_stat, cmdstat=cmd_stat)
        if (cmd_stat /= 0) then
            print *, 'FAIL: emitted executable did not run: ', exe_path
            call execute_command_line('rm -f '//exe_path//' '//out_path)
            return
        end if
        if (exit_stat /= expected_exit) then
            print *, 'FAIL: exit status ', exit_stat, ' expected ', expected_exit
            call execute_command_line('rm -f '//exe_path//' '//out_path)
            return
        end if

        call read_file(out_path, actual)
        call execute_command_line('rm -f '//exe_path//' '//out_path)
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
    end function expect_stderr_and_exit

    logical function expect_eof_stderr_and_exit(source, expected, expected_exit, &
            exe_path) result(ok)
        ! Like expect_stderr_and_exit but redirects stdin from /dev/null, so a
        ! PAUSE (which reads stdin) sees end-of-input and terminates exactly as
        ! the conformance gauntlet runs it.
        character(len=*), intent(in) :: source
        character(len=*), intent(in) :: expected
        integer, intent(in) :: expected_exit
        character(len=*), intent(in) :: exe_path
        character(len=:), allocatable :: error_msg, actual, out_path
        integer :: cmd_stat, exit_stat

        ok = .false.
        call compile_to_exe(source, exe_path, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL: ', trim(error_msg)
            call execute_command_line('rm -f '//exe_path)
            return
        end if

        out_path = exe_path//'.out'
        call execute_command_line(exe_path//' > '//out_path//' 2>&1 < /dev/null', &
            exitstat=exit_stat, cmdstat=cmd_stat)
        if (cmd_stat /= 0) then
            print *, 'FAIL: emitted executable did not run: ', exe_path
            call execute_command_line('rm -f '//exe_path//' '//out_path)
            return
        end if
        if (exit_stat /= expected_exit) then
            print *, 'FAIL: exit status ', exit_stat, ' expected ', expected_exit
            call execute_command_line('rm -f '//exe_path//' '//out_path)
            return
        end if

        call read_file(out_path, actual)
        call execute_command_line('rm -f '//exe_path//' '//out_path)
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
    end function expect_eof_stderr_and_exit

    logical function expect_exe_has_symbol(source, object_path, symbol) result(ok)
        ! Compiles source to an object file and checks it defines the named
        ! symbol (via nm), used to verify bind(c, name="...") emission.
        character(len=*), intent(in) :: source
        character(len=*), intent(in) :: object_path
        character(len=*), intent(in) :: symbol
        character(len=:), allocatable :: error_msg
        type(compiler_frontend_options_t) :: options
        type(compiler_frontend_result_t) :: frontend_result
        integer :: stat, cmd_stat

        ok = .false.
        options = compiler_frontend_options_t()
        options%run_semantics = .true.
        options%input_mode = INPUT_MODE_STANDARD
        call compile_frontend_from_string(source, frontend_result, options)
        if (.not. frontend_result%success()) then
            print *, 'FAIL: FortFront rejected source: ', &
                trim(frontend_result%diagnostic_text)
            return
        end if
        call execute_command_line('rm -f '//object_path)
        call lower_program_to_liric_object(frontend_result%arena, &
            frontend_result%root_index, &
            object_path, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL: object lowering failed: ', trim(error_msg)
            call execute_command_line('rm -f '//object_path)
            return
        end if

        call execute_command_line('nm '//object_path//' | grep -qw '//symbol, &
            exitstat=stat, cmdstat=cmd_stat)
        call execute_command_line('rm -f '//object_path)
        if (cmd_stat /= 0) then
            print *, 'FAIL: could not run nm on ', object_path
            return
        end if
        if (stat /= 0) then
            print *, 'FAIL: symbol not found in object: ', symbol
            return
        end if
        ok = .true.
    end function expect_exe_has_symbol

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
        integer :: exit_stat
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
            exitstat=exit_stat, cmdstat=cmd_stat)
        if (cmd_stat /= 0) then
            print *, 'FAIL: emitted executable did not run: ', exe_path
            call execute_command_line('rm -f '//exe_path//' '//stdout_path)
            return
        end if
        if (exit_stat /= 0) then
            print *, 'FAIL: emitted executable exited with status ', exit_stat
            return
        end if

        call read_file(stdout_path, actual)
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
        call execute_command_line('rm -f '//exe_path//' '//stdout_path)
        ok = .true.
    end function expect_output

    logical function expect_output_with_stdin(source, stdin_input, expected, exe_path) result(ok)
        ! Like expect_output but pipes stdin_input to the executable via echo.
        character(len=*), intent(in) :: source
        character(len=*), intent(in) :: stdin_input
        character(len=*), intent(in) :: expected
        character(len=*), intent(in) :: exe_path
        character(len=:), allocatable :: error_msg
        character(len=:), allocatable :: actual
        character(len=:), allocatable :: stdout_path
        integer :: cmd_stat
        integer :: exit_stat

        ok = .false.
        call compile_to_exe(source, exe_path, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL: ', trim(error_msg)
            call execute_command_line('rm -f '//exe_path)
            return
        end if

        stdout_path = exe_path//'.out'
        call execute_command_line("echo '"//stdin_input//"' | "//exe_path// &
            " > "//stdout_path, &
            exitstat=exit_stat, cmdstat=cmd_stat)
        if (cmd_stat /= 0) then
            print *, 'FAIL: emitted executable did not run: ', exe_path
            call execute_command_line('rm -f '//exe_path//' '//stdout_path)
            return
        end if
        if (exit_stat /= 0) then
            print *, 'FAIL: emitted executable exited with status ', exit_stat
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
    end function expect_output_with_stdin

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

    logical function expect_cli_error_contains(source, fragment, stem) result(ok)
        ! Tests the CLI binary (ffc) directly for unsupported-source diagnostics.
        ! Each migrated test that calls this helper uses a unique stem so the
        ! temporary files do not collide across test programs.
        character(len=*), intent(in) :: source
        character(len=*), intent(in) :: fragment
        character(len=*), intent(in) :: stem
        character(len=:), allocatable :: output
        character(len=:), allocatable :: command
        character(len=:), allocatable :: source_path
        character(len=:), allocatable :: output_path
        character(len=:), allocatable :: exe_path
        integer :: exit_stat
        integer :: cmd_stat

        ok = .false.
        source_path = stem//'.f90'
        output_path = stem//'.out'
        exe_path = stem//'.exe'
        call execute_command_line('rm -f '//source_path//' '//output_path//' '// &
            exe_path)
        if (.not. write_source_file(source_path, source)) return

        command = "sh -c 'exe=$(ls -t build/*/app/ffc build/fo/bin/ffc "// &
            "2>/dev/null | head -n 1); "// &
            "test -n ""$exe"" && ""$exe"" "// &
            source_path//' -o '//exe_path//' > '//output_path//" 2>&1'"
        call execute_command_line(command, exitstat=exit_stat, cmdstat=cmd_stat)
        output = read_text_file(output_path)
        call execute_command_line('rm -f '//source_path//' '//output_path//' '// &
            exe_path)

        if (cmd_stat /= 0) then
            print *, 'FAIL: ffc CLI command could not be executed'
            return
        end if
        if (exit_stat == 0) then
            print *, 'FAIL: unsupported source CLI exited successfully'
            return
        end if
        if (index(output, fragment) == 0) then
            print *, 'FAIL: expected CLI diagnostic substring "', fragment, '"'
            print *, '  got: ', trim(output)
            return
        end if
        if (index(output, 'line ') == 0 .or. index(output, 'column ') == 0) then
            print *, 'FAIL: expected CLI line/column diagnostic, got: ', &
                trim(output)
            return
        end if

        ok = .true.
    end function expect_cli_error_contains

    logical function expect_cli_no_error(source, stem) result(ok)
        ! Invokes the CLI binary and checks that compilation succeeds.
        character(len=*), intent(in) :: source
        character(len=*), intent(in) :: stem
        character(len=:), allocatable :: output
        character(len=:), allocatable :: command
        character(len=:), allocatable :: source_path
        character(len=:), allocatable :: output_path
        character(len=:), allocatable :: exe_path
        integer :: exit_stat
        integer :: cmd_stat

        ok = .false.
        source_path = stem//'.f90'
        output_path = stem//'.out'
        exe_path = stem//'.exe'
        call execute_command_line('rm -f '//source_path//' '//output_path//' '// &
            exe_path)
        if (.not. write_source_file(source_path, source)) return

        command = "sh -c 'exe=$(ls -t build/*/app/ffc build/fo/bin/ffc "// &
            "2>/dev/null | head -n 1); "// &
            "test -n ""$exe"" && ""$exe"" "// &
            source_path//' -o '//exe_path//' > '//output_path//" 2>&1'"
        call execute_command_line(command, exitstat=exit_stat, cmdstat=cmd_stat)
        output = read_text_file(output_path)
        call execute_command_line('rm -f '//source_path//' '//output_path//' '// &
            exe_path)

        if (cmd_stat /= 0) then
            print *, 'FAIL: ffc CLI command could not be executed'
            return
        end if
        if (exit_stat /= 0) then
            print *, 'FAIL: expected CLI success, got exit code ', exit_stat
            print *, '  output: ', trim(output)
            return
        end if
        ok = .true.
    end function expect_cli_no_error

    logical function expect_object_exists(source, object_path) result(ok)
        ! Compiles source and checks that a non-empty object file was emitted.
        character(len=*), intent(in) :: source
        character(len=*), intent(in) :: object_path
        character(len=:), allocatable :: error_msg
        integer :: object_size
        logical :: object_exists

        ok = .false.
        call compile_to_exe(source, object_path, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL: object lowering failed: ', trim(error_msg)
            call execute_command_line('rm -f '//object_path)
            return
        end if

        inquire (file=object_path, exist=object_exists, size=object_size)
        call execute_command_line('rm -f '//object_path)

        if (.not. object_exists .or. object_size <= 0) then
            print *, 'FAIL: expected non-empty object file at ', object_path
            return
        end if

        ok = .true.
    end function expect_object_exists

    logical function expect_no_error(source, exe_path) result(ok)
        ! Compiles source and checks that lowering succeeds without errors.
        character(len=*), intent(in) :: source
        character(len=*), intent(in) :: exe_path
        character(len=:), allocatable :: error_msg

        ok = .false.
        call compile_to_exe(source, exe_path, error_msg)
        call execute_command_line('rm -f '//exe_path)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL: expected no error, got: ', trim(error_msg)
            return
        end if
        ok = .true.
    end function expect_no_error

    logical function write_source_file(path, source)
        character(len=*), intent(in) :: path
        character(len=*), intent(in) :: source
        integer :: unit
        integer :: io_stat

        write_source_file = .false.
        open (newunit=unit, file=path, status='replace', action='write', &
            iostat=io_stat)
        if (io_stat /= 0) then
            print *, 'FAIL: could not create source file ', path
            return
        end if

        write (unit, '(A)', iostat=io_stat) source
        close (unit)
        if (io_stat /= 0) then
            print *, 'FAIL: could not write source file ', path
            return
        end if

        write_source_file = .true.
    end function write_source_file

    function read_text_file(path) result(text)
        character(len=*), intent(in) :: path
        character(len=:), allocatable :: text
        character(len=512) :: line
        integer :: unit
        integer :: io_stat

        text = ''
        open (newunit=unit, file=path, status='old', action='read', &
            iostat=io_stat)
        if (io_stat /= 0) return

        do
            read (unit, '(A)', iostat=io_stat) line
            if (io_stat /= 0) exit
            text = text//trim(line)//new_line('a')
        end do
        close (unit)
    end function read_text_file

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
