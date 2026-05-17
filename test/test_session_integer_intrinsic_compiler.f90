program test_session_integer_intrinsic_compiler
    use fortfront, only: compiler_frontend_options_t, &
                         compiler_frontend_result_t, &
                         compile_frontend_from_string, INPUT_MODE_STANDARD
    use session_program_lowering, only: lower_program_to_liric_exe
    implicit none

    logical :: all_passed

    print *, '=== direct session scalar intrinsic compiler test ==='

    all_passed = .true.
    if (.not. test_integer_intrinsic_values()) all_passed = .false.
    if (.not. test_integer_mod_intrinsic()) all_passed = .false.
    if (.not. test_real_intrinsic_values()) all_passed = .false.
    if (.not. test_real_conversion_intrinsic()) all_passed = .false.
    if (.not. test_unsupported_intrinsic_diagnostic()) all_passed = .false.
    if (.not. test_unsupported_real_intrinsic_diagnostic()) all_passed = .false.
    if (.not. test_user_function_shadowing()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: scalar intrinsics lower through direct LIRIC session'

contains

    logical function test_integer_intrinsic_values()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: x'//new_line('a')// &
                                       '  integer :: y'//new_line('a')// &
                                       '  x = abs(0 - 7)'//new_line('a')// &
                                  '  y = min(x, 9, 4) + max(1, 2, 3)'//new_line('a')// &
                                       '  stop y'//new_line('a')// &
                                       'end program main'

        test_integer_intrinsic_values = compile_and_expect_exit( &
                                   source, '/tmp/ffc_session_integer_intrinsic_test', 7)
    end function test_integer_intrinsic_values

    logical function test_integer_mod_intrinsic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: a'//new_line('a')// &
                                       '  integer :: b'//new_line('a')// &
                                       '  a = 17'//new_line('a')// &
                                       '  b = 5'//new_line('a')// &
                                       '  stop mod(a, b) + mod(-7, 3)'// &
                                       new_line('a')// &
                                       'end program main'

        test_integer_mod_intrinsic = compile_and_expect_exit( &
                                     source, &
                                     '/tmp/ffc_session_integer_mod_test', 1)
    end function test_integer_mod_intrinsic

    logical function test_real_intrinsic_values()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  real :: x'//new_line('a')// &
                                       '  x = abs(0.0 - 2.5) '// &
                                       '+ min(3.5, 1.25, 2.0) '// &
                                       '+ max(0.5, 4.25)'//new_line('a')// &
                                       '  print *, x'//new_line('a')// &
                                       'end program main'

        test_real_intrinsic_values = compile_and_expect_output( &
                                     source, '/tmp/ffc_session_real_intrinsic_test', &
                                     '8.000000')
    end function test_real_intrinsic_values

    logical function test_real_conversion_intrinsic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: i'//new_line('a')// &
                                       '  real :: x'//new_line('a')// &
                                       '  i = 4'//new_line('a')// &
                                       '  x = real(i) + 1.5'//new_line('a')// &
                                       '  print *, x'//new_line('a')// &
                                       'end program main'

        test_real_conversion_intrinsic = compile_and_expect_output( &
                                      source, '/tmp/ffc_session_real_conversion_test', &
                                         '5.500000')
    end function test_real_conversion_intrinsic

    logical function test_unsupported_intrinsic_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  stop int(5.5)'//new_line('a')// &
                                       'end program main'
        character(len=:), allocatable :: error_msg

        test_unsupported_intrinsic_diagnostic = .false.
        call compile_and_lower(source, &
                               '/tmp/ffc_session_unsupported_intrinsic_test', &
                               error_msg)
        call execute_command_line( &
            'rm -f /tmp/ffc_session_unsupported_intrinsic_test')

        if (index(error_msg, 'unsupported scalar intrinsic: int') <= 0) then
            print *, 'FAIL: expected unsupported intrinsic diagnostic, got ', &
                trim(error_msg)
            return
        end if

        test_unsupported_intrinsic_diagnostic = .true.
    end function test_unsupported_intrinsic_diagnostic

    logical function test_unsupported_real_intrinsic_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  real :: x'//new_line('a')// &
                                       '  x = sqrt(4.0)'//new_line('a')// &
                                       'end program main'
        character(len=:), allocatable :: error_msg

        test_unsupported_real_intrinsic_diagnostic = .false.
        call compile_and_lower(source, &
                               '/tmp/ffc_session_unsupported_real_intrinsic_test', &
                               error_msg)
        call execute_command_line( &
            'rm -f /tmp/ffc_session_unsupported_real_intrinsic_test')

        if (index(error_msg, 'unsupported scalar intrinsic: sqrt') <= 0) then
            print *, 'FAIL: expected unsupported real intrinsic diagnostic, got ', &
                trim(error_msg)
            return
        end if

        test_unsupported_real_intrinsic_diagnostic = .true.
    end function test_unsupported_real_intrinsic_diagnostic

    logical function test_user_function_shadowing()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: x'//new_line('a')// &
                                       '  x = min(2, 3)'//new_line('a')// &
                                       '  stop x'//new_line('a')// &
                                       'contains'//new_line('a')// &
                                       '  integer function min(a, b)'//new_line('a')// &
                                       '    min = a + b'//new_line('a')// &
                                       '  end function min'//new_line('a')// &
                                       'end program main'

        test_user_function_shadowing = compile_and_expect_exit( &
                                    source, '/tmp/ffc_session_intrinsic_shadow_test', 5)
    end function test_user_function_shadowing

    logical function compile_and_expect_exit(source, exe_path, expected_status)
        character(len=*), intent(in) :: source
        character(len=*), intent(in) :: exe_path
        integer, intent(in) :: expected_status
        character(len=:), allocatable :: error_msg
        integer :: cmd_stat
        integer :: exit_stat

        compile_and_expect_exit = .false.
        call execute_command_line('rm -f '//exe_path)
        call compile_and_lower(source, exe_path, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL: direct LIRIC lowering failed: ', trim(error_msg)
            return
        end if

        call execute_command_line(exe_path, exitstat=exit_stat, &
                                  cmdstat=cmd_stat)
        call execute_command_line('rm -f '//exe_path)
        if (cmd_stat /= 0) then
            print *, 'FAIL: emitted executable did not run'
            return
        end if
        if (exit_stat /= expected_status) then
            print *, 'FAIL: executable exit status ', exit_stat
            return
        end if

        compile_and_expect_exit = .true.
    end function compile_and_expect_exit

    logical function compile_and_expect_output(source, exe_path, expected_output)
        character(len=*), intent(in) :: source
        character(len=*), intent(in) :: exe_path
        character(len=*), intent(in) :: expected_output
        character(len=:), allocatable :: error_msg
        character(len=:), allocatable :: out_path
        character(len=64) :: output_line
        integer :: cmd_stat
        integer :: exit_stat
        integer :: io_stat
        integer :: unit

        compile_and_expect_output = .false.
        out_path = exe_path//'.out'
        call execute_command_line('rm -f '//exe_path//' '//out_path)
        call compile_and_lower(source, exe_path, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL: direct LIRIC lowering failed: ', trim(error_msg)
            return
        end if

        call execute_command_line(exe_path//' > '//out_path, exitstat=exit_stat, &
                                  cmdstat=cmd_stat)
        if (cmd_stat /= 0 .or. exit_stat /= 0) then
            print *, 'FAIL: emitted executable did not run cleanly'
            call execute_command_line('rm -f '//exe_path//' '//out_path)
            return
        end if

        open (newunit=unit, file=out_path, status='old', action='read', &
              iostat=io_stat)
        if (io_stat == 0) read (unit, '(A)', iostat=io_stat) output_line
        if (io_stat == 0) close (unit)
        call execute_command_line('rm -f '//exe_path//' '//out_path)
        if (io_stat /= 0) then
            print *, 'FAIL: could not read captured output'
            return
        end if
        if (trim(adjustl(output_line)) /= expected_output) then
            print *, 'FAIL: expected output ', expected_output, &
                ', got ', trim(output_line)
            return
        end if

        compile_and_expect_output = .true.
    end function compile_and_expect_output

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
end program test_session_integer_intrinsic_compiler
