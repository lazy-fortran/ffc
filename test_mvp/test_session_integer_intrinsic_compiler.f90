program test_session_integer_intrinsic_compiler
    use fortfront, only: compiler_frontend_options_t, &
                         compiler_frontend_result_t, &
                         compile_frontend_from_string, INPUT_MODE_STANDARD
    use session_program_lowering, only: lower_program_to_liric_exe
    implicit none

    logical :: all_passed

    print *, '=== direct session integer intrinsic compiler test ==='

    all_passed = .true.
    if (.not. test_integer_intrinsic_values()) all_passed = .false.
    if (.not. test_unsupported_intrinsic_diagnostic()) all_passed = .false.
    if (.not. test_user_function_shadowing()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: integer intrinsics lower through direct LIRIC session'

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

    logical function test_unsupported_intrinsic_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  stop mod(5, 2)'//new_line('a')// &
                                       'end program main'
        character(len=:), allocatable :: error_msg

        test_unsupported_intrinsic_diagnostic = .false.
        call compile_and_lower(source, &
                               '/tmp/ffc_session_unsupported_intrinsic_test', &
                               error_msg)
        call execute_command_line( &
            'rm -f /tmp/ffc_session_unsupported_intrinsic_test')

        if (index(error_msg, 'unsupported scalar intrinsic: mod') <= 0) then
            print *, 'FAIL: expected unsupported intrinsic diagnostic, got ', &
                trim(error_msg)
            return
        end if

        test_unsupported_intrinsic_diagnostic = .true.
    end function test_unsupported_intrinsic_diagnostic

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
