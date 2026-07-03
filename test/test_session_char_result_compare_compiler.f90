program test_session_char_result_compare_compiler
    use ffc_test_support, only: expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== direct session character function result comparison test ==='

    all_passed = .true.
    if (.not. test_result_equal_passes()) all_passed = .false.
    if (.not. test_result_notequal_stops()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: character function result comparison lowers correctly'

contains

    logical function test_result_equal_passes()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  if (label(1) /= "one") error stop'//new_line('a')// &
            '  if (label(2) /= "two") error stop'//new_line('a')// &
            '  if (label(3) /= "def") error stop'//new_line('a')// &
            'contains'//new_line('a')// &
            '  function label(i) result(s)'//new_line('a')// &
            '    integer, intent(in) :: i'//new_line('a')// &
            '    character(len=3) :: s'//new_line('a')// &
            '    select case (i)'//new_line('a')// &
            '    case (1)'//new_line('a')// &
            '      s = "one"'//new_line('a')// &
            '    case (2)'//new_line('a')// &
            '      s = "two"'//new_line('a')// &
            '    case default'//new_line('a')// &
            '      s = "def"'//new_line('a')// &
            '    end select'//new_line('a')// &
            '  end function label'//new_line('a')// &
            'end program main'

        test_result_equal_passes = expect_exit_status( &
            source, 0, '/tmp/ffc_session_char_res_cmp_ok')
    end function test_result_equal_passes

    logical function test_result_notequal_stops()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  if (tag() /= "zz") error stop 5'//new_line('a')// &
            '  if (tag() == "zz") error stop 5'//new_line('a')// &
            'contains'//new_line('a')// &
            '  function tag() result(s)'//new_line('a')// &
            '    character(len=2) :: s'//new_line('a')// &
            '    s = "zz"'//new_line('a')// &
            '  end function tag'//new_line('a')// &
            'end program main'

        test_result_notequal_stops = expect_exit_status( &
            source, 5, '/tmp/ffc_session_char_res_cmp_stop')
    end function test_result_notequal_stops

end program test_session_char_result_compare_compiler
