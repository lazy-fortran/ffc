program test_session_non_integer_procedure_compiler
    use fortfront, only: compiler_frontend_options_t, &
                         compiler_frontend_result_t, &
                         compile_frontend_from_string, INPUT_MODE_STANDARD
    use session_program_lowering, only: lower_program_to_liric_exe
    implicit none

    logical :: all_passed

    print *, '=== direct session non-integer procedure compiler test ==='

    all_passed = .true.
    if (.not. test_real_subroutine()) all_passed = .false.
    if (.not. test_logical_subroutine()) all_passed = .false.
    if (.not. test_real_function()) all_passed = .false.
    if (.not. test_mixed_real_logical_function()) all_passed = .false.
    if (.not. test_logical_function()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: non-integer contained procedures lower through direct LIRIC'

contains

    logical function test_real_subroutine()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  real :: x'//new_line('a')// &
                                       '  x = 1.5'//new_line('a')// &
                                       '  call bump(x)'//new_line('a')// &
                                       '  print *, x'//new_line('a')// &
                                       'contains'//new_line('a')// &
                                       '  subroutine bump(value)'//new_line('a')// &
                                   '    real, intent(inout) :: value'//new_line('a')// &
                                       '    value = value + 1.0'//new_line('a')// &
                                       '  end subroutine bump'//new_line('a')// &
                                       'end program main'

        test_real_subroutine = expect_output(source, '2.500000', &
                                             '/tmp/ffc_session_real_sub_test')
    end function test_real_subroutine

    logical function test_logical_subroutine()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  logical :: flag'//new_line('a')// &
                                       '  flag = .false.'//new_line('a')// &
                                       '  call enable(flag)'//new_line('a')// &
                                       '  if (flag) then'//new_line('a')// &
                                       '    print *, 1'//new_line('a')// &
                                       '  else'//new_line('a')// &
                                       '    print *, 0'//new_line('a')// &
                                       '  end if'//new_line('a')// &
                                       'contains'//new_line('a')// &
                                       '  subroutine enable(value)'//new_line('a')// &
                                '    logical, intent(inout) :: value'//new_line('a')// &
                                       '    value = .true.'//new_line('a')// &
                                       '  end subroutine enable'//new_line('a')// &
                                       'end program main'

        test_logical_subroutine = expect_output(source, '1', &
                                                '/tmp/ffc_session_logical_sub_test')
    end function test_logical_subroutine

    logical function test_real_function()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  real :: x'//new_line('a')// &
                                       '  x = add(1.5, 3.0)'//new_line('a')// &
                                       '  print *, x'//new_line('a')// &
                                       'contains'//new_line('a')// &
                                       '  real function add(a, b)'//new_line('a')// &
                                       '    real, intent(in) :: a'//new_line('a')// &
                                       '    real, intent(in) :: b'//new_line('a')// &
                                       '    add = a + b'//new_line('a')// &
                                       '  end function add'//new_line('a')// &
                                       'end program main'

        test_real_function = expect_output(source, '4.500000', &
                                           '/tmp/ffc_session_real_fn_test')
    end function test_real_function

    logical function test_mixed_real_logical_function()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  real :: x'//new_line('a')// &
                                       '  logical :: enabled'//new_line('a')// &
                                       '  x = 1.25'//new_line('a')// &
                                       '  enabled = .true.'//new_line('a')// &
                                       '  print *, choose(x, enabled)'// &
                                       new_line('a')// &
                                       'contains'//new_line('a')// &
                                       '  real function choose(value, flag)'// &
                                       new_line('a')// &
                                       '    real, intent(in) :: value'// &
                                       new_line('a')// &
                                       '    logical, intent(in) :: flag'// &
                                       new_line('a')// &
                                       '    if (flag) then'//new_line('a')// &
                                       '      choose = value + 0.75'//new_line('a')// &
                                       '    else'//new_line('a')// &
                                       '      choose = value'//new_line('a')// &
                                       '    end if'//new_line('a')// &
                                       '  end function choose'//new_line('a')// &
                                       'end program main'

        test_mixed_real_logical_function = expect_output( &
                                           source, '2.000000', &
                                           '/tmp/ffc_session_mixed_fn_test')
    end function test_mixed_real_logical_function

    logical function test_logical_function()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  logical :: flag'//new_line('a')// &
                                       '  logical :: ok'//new_line('a')// &
                                       '  flag = .false.'//new_line('a')// &
                                       '  ok = enabled(flag)'//new_line('a')// &
                                       '  if (ok) then'//new_line('a')// &
                                       '    print *, 1'//new_line('a')// &
                                       '  else'//new_line('a')// &
                                       '    print *, 0'//new_line('a')// &
                                       '  end if'//new_line('a')// &
                                       'contains'//new_line('a')// &
                                   '  logical function enabled(flag)'//new_line('a')// &
                                    '    logical, intent(in) :: flag'//new_line('a')// &
                                       '    if (flag) then'//new_line('a')// &
                                       '      enabled = .false.'//new_line('a')// &
                                       '    else'//new_line('a')// &
                                       '      enabled = .true.'//new_line('a')// &
                                       '    end if'//new_line('a')// &
                                       '  end function enabled'//new_line('a')// &
                                       'end program main'

        test_logical_function = expect_output(source, '1', &
                                              '/tmp/ffc_session_logical_fn_test')
    end function test_logical_function

    logical function expect_output(source, expected, stem)
        character(len=*), intent(in) :: source
        character(len=*), intent(in) :: expected
        character(len=*), intent(in) :: stem
        character(len=64) :: output_line
        character(len=:), allocatable :: error_msg
        character(len=:), allocatable :: exe_path
        character(len=:), allocatable :: out_path
        integer :: exit_stat
        integer :: io_stat
        integer :: unit

        expect_output = .false.
        exe_path = stem
        out_path = stem//'.out'
        call execute_command_line('rm -f '//exe_path//' '//out_path)
        call compile_and_lower(source, exe_path, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL: direct LIRIC lowering failed: ', trim(error_msg)
            return
        end if

        call execute_command_line(exe_path//' > '//out_path, exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL: executable exit status ', exit_stat
            call execute_command_line('rm -f '//exe_path//' '//out_path)
            return
        end if

        open (newunit=unit, file=out_path, status='old', action='read', &
              iostat=io_stat)
        if (io_stat /= 0) then
            print *, 'FAIL: could not open captured output'
            call execute_command_line('rm -f '//exe_path//' '//out_path)
            return
        end if
        read (unit, '(A)', iostat=io_stat) output_line
        close (unit)
        call execute_command_line('rm -f '//exe_path//' '//out_path)

        if (io_stat /= 0 .or. trim(adjustl(output_line)) /= expected) then
            print *, 'FAIL: expected output ', expected, ', got ', &
                trim(output_line)
            return
        end if

        expect_output = .true.
    end function expect_output

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
end program test_session_non_integer_procedure_compiler
