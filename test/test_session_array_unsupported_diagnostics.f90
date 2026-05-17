program test_session_array_unsupported_diagnostics
    use fortfront, only: compiler_frontend_options_t, &
                         compiler_frontend_result_t, &
                         compile_frontend_from_string, INPUT_MODE_STANDARD
    use session_program_lowering, only: lower_program_to_liric_exe
    implicit none

    logical :: all_passed

    print *, '=== direct session array unsupported diagnostic test ==='

    all_passed = .true.
    if (.not. test_array_declaration_diagnostic()) all_passed = .false.
    if (.not. test_noninteger_array_declaration_diagnostic()) &
        all_passed = .false.
    if (.not. test_array_assignment_target_diagnostic()) all_passed = .false.
    if (.not. test_array_expression_diagnostic()) all_passed = .false.
    if (.not. test_whole_array_expression_diagnostic()) all_passed = .false.
    if (.not. test_array_rhs_assignment_diagnostic()) all_passed = .false.
    if (.not. test_whole_array_assignment_target_diagnostic()) &
        all_passed = .false.
    if (.not. test_whole_array_argument_diagnostic()) all_passed = .false.
    if (.not. test_cli_array_declaration_diagnostic()) all_passed = .false.
    if (.not. test_cli_noninteger_array_declaration_diagnostic()) &
        all_passed = .false.
    if (.not. test_cli_array_assignment_target_diagnostic()) all_passed = .false.
    if (.not. test_cli_array_expression_diagnostic()) all_passed = .false.
    if (.not. test_cli_whole_array_expression_diagnostic()) &
        all_passed = .false.
    if (.not. test_cli_array_rhs_assignment_diagnostic()) all_passed = .false.
    if (.not. test_cli_whole_array_assignment_target_diagnostic()) &
        all_passed = .false.
    if (.not. test_cli_whole_array_argument_diagnostic()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: unsupported direct-session array features emit diagnostics'

contains

    logical function test_array_declaration_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: values(2, 2)'//new_line('a')// &
                                       'end program main'

        test_array_declaration_diagnostic = expect_error_contains( &
                                            source, 'unsupported array declaration', &
                                            '/tmp/ffc_session_array_diagnostic_test')
    end function test_array_declaration_diagnostic

    logical function test_noninteger_array_declaration_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  real :: values(3)'//new_line('a')// &
                                       'end program main'

        test_noninteger_array_declaration_diagnostic = expect_error_contains( &
                                                    source, 'supports integer arrays', &
                                                     '/tmp/ffc_session_real_array_test')
    end function test_noninteger_array_declaration_diagnostic

    logical function test_array_assignment_target_diagnostic()
        character(len=*), parameter :: expected = &
                                       'unsupported array assignment target'
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: values(3)'//new_line('a')// &
                                       '  values(1, 1) = 1'//new_line('a')// &
                                       'end program main'

        test_array_assignment_target_diagnostic = &
            expect_error_contains(source, expected, &
                                  '/tmp/ffc_session_array_target_test')
    end function test_array_assignment_target_diagnostic

    logical function test_array_expression_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: values(3)'//new_line('a')// &
                                       '  print *, values(1, 1)'//new_line('a')// &
                                       'end program main'

        test_array_expression_diagnostic = expect_error_contains( &
                                           source, 'array expression', &
                                           '/tmp/ffc_session_array_expr_test')
    end function test_array_expression_diagnostic

    logical function test_whole_array_expression_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: values(3)'//new_line('a')// &
                                       '  print *, values'//new_line('a')// &
                                       'end program main'

        test_whole_array_expression_diagnostic = expect_error_contains( &
                                               source, 'unsupported array expression', &
                                               '/tmp/ffc_session_whole_array_expr_test')
    end function test_whole_array_expression_diagnostic

    logical function test_array_rhs_assignment_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: values(3)'//new_line('a')// &
                                       '  integer :: x'//new_line('a')// &
                                       '  x = values'//new_line('a')// &
                                       'end program main'

        test_array_rhs_assignment_diagnostic = expect_error_contains( &
                                               source, 'unsupported array expression', &
                                           '/tmp/ffc_session_array_rhs_assignment_test')
    end function test_array_rhs_assignment_diagnostic

    logical function test_whole_array_assignment_target_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: values(3)'//new_line('a')// &
                                       '  values = 4'//new_line('a')// &
                                       'end program main'

        test_whole_array_assignment_target_diagnostic = expect_error_contains( &
                                        source, 'unsupported array assignment target', &
                                         '/tmp/ffc_session_whole_array_assignment_test')
    end function test_whole_array_assignment_target_diagnostic

    logical function test_whole_array_argument_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: values(1)'//new_line('a')// &
                                       '  values(1) = 1'//new_line('a')// &
                                       '  call bump(values)'//new_line('a')// &
                                       'contains'//new_line('a')// &
                                       '  subroutine bump(x)'//new_line('a')// &
                                       '    integer :: x'//new_line('a')// &
                                       '    x = x + 4'//new_line('a')// &
                                       '  end subroutine bump'//new_line('a')// &
                                       'end program main'

        test_whole_array_argument_diagnostic = expect_error_contains( &
                                             source, 'unsupported array argument', &
                                             '/tmp/ffc_session_array_arg_test')
    end function test_whole_array_argument_diagnostic

    logical function test_cli_array_declaration_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: values(2, 2)'//new_line('a')// &
                                       'end program main'

        test_cli_array_declaration_diagnostic = expect_cli_error_contains( &
                                              source, 'unsupported array declaration', &
                                                '/tmp/ffc_cli_array_diagnostic_test')
    end function test_cli_array_declaration_diagnostic

    logical function test_cli_noninteger_array_declaration_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  real :: values(3)'//new_line('a')// &
                                       'end program main'

        test_cli_noninteger_array_declaration_diagnostic = &
            expect_cli_error_contains(source, 'supports integer arrays', &
                                      '/tmp/ffc_cli_real_array_test')
    end function test_cli_noninteger_array_declaration_diagnostic

    logical function test_cli_array_assignment_target_diagnostic()
        character(len=*), parameter :: expected = &
                                       'unsupported array assignment target'
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: values(3)'//new_line('a')// &
                                       '  values(1, 1) = 1'//new_line('a')// &
                                       'end program main'

        test_cli_array_assignment_target_diagnostic = &
            expect_cli_error_contains(source, expected, &
                                      '/tmp/ffc_cli_array_target_test')
    end function test_cli_array_assignment_target_diagnostic

    logical function test_cli_array_expression_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: values(3)'//new_line('a')// &
                                       '  print *, values(1, 1)'//new_line('a')// &
                                       'end program main'

        test_cli_array_expression_diagnostic = expect_cli_error_contains( &
                                               source, 'array expression', &
                                               '/tmp/ffc_cli_array_expr_test')
    end function test_cli_array_expression_diagnostic

    logical function test_cli_whole_array_expression_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: values(3)'//new_line('a')// &
                                       '  print *, values'//new_line('a')// &
                                       'end program main'

        test_cli_whole_array_expression_diagnostic = expect_cli_error_contains( &
                                               source, 'unsupported array expression', &
                                                   '/tmp/ffc_cli_whole_array_expr_test')
    end function test_cli_whole_array_expression_diagnostic

    logical function test_cli_array_rhs_assignment_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: values(3)'//new_line('a')// &
                                       '  integer :: x'//new_line('a')// &
                                       '  x = values'//new_line('a')// &
                                       'end program main'

        test_cli_array_rhs_assignment_diagnostic = expect_cli_error_contains( &
                                               source, 'unsupported array expression', &
                                               '/tmp/ffc_cli_array_rhs_assignment_test')
    end function test_cli_array_rhs_assignment_diagnostic

    logical function test_cli_whole_array_assignment_target_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: values(3)'//new_line('a')// &
                                       '  values = 4'//new_line('a')// &
                                       'end program main'

        test_cli_whole_array_assignment_target_diagnostic = &
            expect_cli_error_contains(source, &
                                      'unsupported array assignment target', &
                                      '/tmp/ffc_cli_whole_array_assignment_test')
    end function test_cli_whole_array_assignment_target_diagnostic

    logical function test_cli_whole_array_argument_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: values(1)'//new_line('a')// &
                                       '  values(1) = 1'//new_line('a')// &
                                       '  call bump(values)'//new_line('a')// &
                                       'contains'//new_line('a')// &
                                       '  subroutine bump(x)'//new_line('a')// &
                                       '    integer :: x'//new_line('a')// &
                                       '    x = x + 4'//new_line('a')// &
                                       '  end subroutine bump'//new_line('a')// &
                                       'end program main'

        test_cli_whole_array_argument_diagnostic = expect_cli_error_contains( &
                                             source, 'unsupported array argument', &
                                             '/tmp/ffc_cli_array_arg_test')
    end function test_cli_whole_array_argument_diagnostic

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
        character(len=:), allocatable :: output
        character(len=:), allocatable :: command
        character(len=:), allocatable :: source_path
        character(len=:), allocatable :: output_path
        character(len=:), allocatable :: exe_path
        integer :: exit_stat
        integer :: cmd_stat

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
end program test_session_array_unsupported_diagnostics
