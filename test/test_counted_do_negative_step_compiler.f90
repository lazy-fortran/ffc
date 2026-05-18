program test_counted_do_negative_step_compiler
    use fortfront, only: compiler_frontend_options_t, &
                         compiler_frontend_result_t, &
                         compile_frontend_from_string, INPUT_MODE_STANDARD
    use session_program_lowering, only: lower_program_to_liric_exe
    implicit none

    logical :: all_passed

    print *, '=== counted DO negative literal step compiler test ==='

    all_passed = .true.
    if (.not. test_negative_literal_step_descends()) all_passed = .false.
    if (.not. test_negative_step_skip_two()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: counted DO with negative literal step lowers through direct LIRIC'

contains

    logical function test_negative_literal_step_descends()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  integer :: total'//new_line('a')// &
            '  total = 0'//new_line('a')// &
            '  do i = 10, 1, -1'//new_line('a')// &
            '    total = total + i'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  stop total'//new_line('a')// &
            'end program main'

        test_negative_literal_step_descends = expect_exit_status( &
            source, 55, '/tmp/ffc_do_neg_step_test')
    end function test_negative_literal_step_descends

    logical function test_negative_step_skip_two()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  integer :: total'//new_line('a')// &
            '  total = 0'//new_line('a')// &
            '  do i = 10, 0, -2'//new_line('a')// &
            '    total = total + i'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  stop total'//new_line('a')// &
            'end program main'

        test_negative_step_skip_two = expect_exit_status( &
            source, 30, '/tmp/ffc_do_neg_step_skip2_test')
    end function test_negative_step_skip_two

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

end program test_counted_do_negative_step_compiler
