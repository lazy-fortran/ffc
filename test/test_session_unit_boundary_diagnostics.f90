program test_session_unit_boundary_diagnostics
    use fortfront, only: compiler_frontend_options_t, &
                         compiler_frontend_result_t, &
                         compile_frontend_from_string, INPUT_MODE_STANDARD
    use session_program_lowering, only: lower_program_to_liric_exe
    implicit none

    logical :: all_passed

    print *, '=== direct session program unit boundary diagnostic test ==='

    all_passed = .true.
    if (.not. test_interface_block_diagnostic()) all_passed = .false.
    if (.not. test_cli_interface_block_diagnostic()) all_passed = .false.
    if (.not. test_use_intrinsic_module_no_error()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: program unit boundary diagnostics are targeted'

contains

    logical function test_use_intrinsic_module_no_error()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  use iso_fortran_env'//new_line('a')// &
                                       'end program main'
        character(len=:), allocatable :: error_msg

        test_use_intrinsic_module_no_error = .false.
        call compile_and_lower(source, '/tmp/ffc_session_use_intrinsic_test', &
                               error_msg)
        call execute_command_line('rm -f /tmp/ffc_session_use_intrinsic_test')

        ! USE of intrinsic module should be silently ignored (no error)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL: unexpected error for intrinsic module use: ', &
                trim(error_msg)
            return
        end if

        test_use_intrinsic_module_no_error = .true.
    end function test_use_intrinsic_module_no_error

    logical function test_interface_block_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  interface'//new_line('a')// &
                                       '  end interface'//new_line('a')// &
                                       'end program main'

        test_interface_block_diagnostic = expect_error_contains( &
                                          source, &
                                          'unsupported interface block', &
                                          '/tmp/ffc_session_interface_block_test')
    end function test_interface_block_diagnostic

    logical function test_cli_interface_block_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  interface'//new_line('a')// &
                                       '  end interface'//new_line('a')// &
                                       'end program main'

        test_cli_interface_block_diagnostic = expect_cli_error_contains( &
                                              source, &
                                              'unsupported interface block', &
                                              '/tmp/ffc_cli_interface_block_test')
    end function test_cli_interface_block_diagnostic

    logical function expect_error_contains(source, expected, exe_path)
        character(len=*), intent(in) :: source
        character(len=*), intent(in) :: expected
        character(len=*), intent(in) :: exe_path
        character(len=:), allocatable :: error_msg

        expect_error_contains = .false.
        call compile_and_lower(source, exe_path, error_msg)
        call execute_command_line('rm -f '//exe_path)

        if (len_trim(error_msg) == 0) then
            print *, 'FAIL: unsupported source lowered without error'
            return
        end if
        if (index(error_msg, expected) <= 0) then
            print *, 'FAIL: expected diagnostic substring ', expected
            print *, '  got ', trim(error_msg)
            return
        end if
        if (index(error_msg, 'line ') <= 0 .or. &
            index(error_msg, 'column ') <= 0) then
            print *, 'FAIL: expected line/column diagnostic, got ', &
                trim(error_msg)
            return
        end if

        expect_error_contains = .true.
    end function expect_error_contains

    logical function expect_cli_error_contains(source, expected, stem)
        character(len=*), intent(in) :: source
        character(len=*), intent(in) :: expected
        character(len=*), intent(in) :: stem
        character(len=:), allocatable :: command
        character(len=:), allocatable :: exe_path
        character(len=:), allocatable :: output
        character(len=:), allocatable :: output_path
        character(len=:), allocatable :: source_path
        integer :: cmd_stat
        integer :: exit_stat

        expect_cli_error_contains = .false.
        source_path = stem//'.f90'
        output_path = stem//'.out'
        exe_path = stem//'.exe'
        call execute_command_line('rm -f '//source_path//' '//output_path//' '// &
                                  exe_path)
        if (.not. write_source_file(source_path, source)) return

        command = "sh -c 'exe=$(ls -t build/*/app/ffc 2>/dev/null | "// &
                  "head -n 1); "// &
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
        if (index(output, expected) <= 0) then
            print *, 'FAIL: expected CLI diagnostic substring ', expected
            print *, '  got ', trim(output)
            return
        end if
        if (index(output, 'line ') <= 0 .or. index(output, 'column ') <= 0) then
            print *, 'FAIL: expected CLI line/column diagnostic, got ', &
                trim(output)
            return
        end if

        expect_cli_error_contains = .true.
    end function expect_cli_error_contains

    logical function write_source_file(path, source)
        character(len=*), intent(in) :: path
        character(len=*), intent(in) :: source
        integer :: io_stat
        integer :: unit

        write_source_file = .false.
        open (newunit=unit, file=path, status='replace', action='write', &
              iostat=io_stat)
        if (io_stat /= 0) return

        write (unit, '(A)', iostat=io_stat) source
        close (unit)
        if (io_stat /= 0) return

        write_source_file = .true.
    end function write_source_file

    function read_text_file(path) result(text)
        character(len=*), intent(in) :: path
        character(len=:), allocatable :: text
        character(len=512) :: line
        integer :: io_stat
        integer :: unit

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

    subroutine compile_and_lower(source, exe_path, error_msg)
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

        call lower_program_to_liric_exe(frontend_result%arena, &
                                        frontend_result%root_index, exe_path, &
                                        error_msg)
    end subroutine compile_and_lower
end program test_session_unit_boundary_diagnostics
