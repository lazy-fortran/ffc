program test_session_select_case_compiler
    use fortfront, only: compiler_frontend_options_t, &
                         compiler_frontend_result_t, &
                         compile_frontend_from_string, INPUT_MODE_STANDARD
    use session_program_lowering, only: lower_program_to_liric_exe
    implicit none

    logical :: all_passed

    print *, '=== direct session select case compiler test ==='

    all_passed = .true.
    if (.not. test_select_case_one_arm_matches()) all_passed = .false.
    if (.not. test_select_case_one_arm_default()) all_passed = .false.
    if (.not. test_select_case_three_arms_matches_first()) all_passed = .false.
    if (.not. test_select_case_three_arms_matches_middle()) all_passed = .false.
    if (.not. test_select_case_three_arms_matches_default()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: SELECT CASE lowers through direct LIRIC'

contains

    logical function test_select_case_one_arm_matches()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: x'//new_line('a')// &
            '  x = 3'//new_line('a')// &
            '  select case (x)'//new_line('a')// &
            '  case (3)'//new_line('a')// &
            '    stop 11'//new_line('a')// &
            '  case default'//new_line('a')// &
            '    stop 99'//new_line('a')// &
            '  end select'//new_line('a')// &
            'end program main'

        test_select_case_one_arm_matches = expect_exit_status( &
            source, 11, '/tmp/ffc_session_select_match_test')
    end function test_select_case_one_arm_matches

    logical function test_select_case_one_arm_default()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: x'//new_line('a')// &
            '  x = 7'//new_line('a')// &
            '  select case (x)'//new_line('a')// &
            '  case (3)'//new_line('a')// &
            '    stop 11'//new_line('a')// &
            '  case default'//new_line('a')// &
            '    stop 99'//new_line('a')// &
            '  end select'//new_line('a')// &
            'end program main'

        test_select_case_one_arm_default = expect_exit_status( &
            source, 99, '/tmp/ffc_session_select_default_test')
    end function test_select_case_one_arm_default

    logical function test_select_case_three_arms_matches_first()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: x'//new_line('a')// &
            '  x = 1'//new_line('a')// &
            '  select case (x)'//new_line('a')// &
            '  case (1)'//new_line('a')// &
            '    stop 11'//new_line('a')// &
            '  case (2)'//new_line('a')// &
            '    stop 22'//new_line('a')// &
            '  case (3)'//new_line('a')// &
            '    stop 33'//new_line('a')// &
            '  case default'//new_line('a')// &
            '    stop 99'//new_line('a')// &
            '  end select'//new_line('a')// &
            'end program main'

        test_select_case_three_arms_matches_first = expect_exit_status( &
            source, 11, '/tmp/ffc_session_select_first_test')
    end function test_select_case_three_arms_matches_first

    logical function test_select_case_three_arms_matches_middle()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: x'//new_line('a')// &
            '  x = 2'//new_line('a')// &
            '  select case (x)'//new_line('a')// &
            '  case (1)'//new_line('a')// &
            '    stop 11'//new_line('a')// &
            '  case (2)'//new_line('a')// &
            '    stop 22'//new_line('a')// &
            '  case (3)'//new_line('a')// &
            '    stop 33'//new_line('a')// &
            '  case default'//new_line('a')// &
            '    stop 99'//new_line('a')// &
            '  end select'//new_line('a')// &
            'end program main'

        test_select_case_three_arms_matches_middle = expect_exit_status( &
            source, 22, '/tmp/ffc_session_select_middle_test')
    end function test_select_case_three_arms_matches_middle

    logical function test_select_case_three_arms_matches_default()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: x'//new_line('a')// &
            '  x = 7'//new_line('a')// &
            '  select case (x)'//new_line('a')// &
            '  case (1)'//new_line('a')// &
            '    stop 11'//new_line('a')// &
            '  case (2)'//new_line('a')// &
            '    stop 22'//new_line('a')// &
            '  case (3)'//new_line('a')// &
            '    stop 33'//new_line('a')// &
            '  case default'//new_line('a')// &
            '    stop 99'//new_line('a')// &
            '  end select'//new_line('a')// &
            'end program main'

        test_select_case_three_arms_matches_default = expect_exit_status( &
            source, 99, '/tmp/ffc_session_select_three_default_test')
    end function test_select_case_three_arms_matches_default

    logical function expect_exit_status(source, expected, exe_path) result(ok)
        character(len=*), intent(in) :: source
        integer, intent(in) :: expected
        character(len=*), intent(in) :: exe_path
        type(compiler_frontend_options_t) :: options
        type(compiler_frontend_result_t) :: frontend_result
        character(len=:), allocatable :: error_msg
        integer :: exit_stat, cmd_stat

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

        call execute_command_line('rm -f '//exe_path)
        call lower_program_to_liric_exe(frontend_result%arena, &
                                        frontend_result%root_index, &
                                        exe_path, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL: direct LIRIC lowering failed: ', trim(error_msg)
            return
        end if

        call execute_command_line(exe_path, exitstat=exit_stat, cmdstat=cmd_stat)
        call execute_command_line('rm -f '//exe_path)
        if (cmd_stat /= 0) then
            print *, 'FAIL: emitted executable did not run'
            return
        end if
        if (exit_stat /= expected) then
            print *, 'FAIL: exit status ', exit_stat, ' expected ', expected
            return
        end if
        ok = .true.
    end function expect_exit_status

end program test_session_select_case_compiler
