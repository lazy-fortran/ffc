program test_session_if_merge_compiler
    use ffc_test_support, only: expect_exit_status, expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session if merge compiler test ==='

    all_passed = .true.
    if (.not. test_integer_if_merge()) all_passed = .false.
    if (.not. test_real_if_merge()) all_passed = .false.
    if (.not. test_logical_if_merge()) all_passed = .false.
    if (.not. test_nested_if_in_do_merge()) all_passed = .false.
    if (.not. test_do_in_if_merge()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: fallthrough IF merges scalar values through direct LIRIC'

contains

    logical function test_integer_if_merge()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       'integer :: x'//new_line('a')// &
                                       'if (2 < 3) then'//new_line('a')// &
                                       '    x = 9'//new_line('a')// &
                                       'else'//new_line('a')// &
                                       '    x = 4'//new_line('a')// &
                                       'end if'//new_line('a')// &
                                       'stop x'//new_line('a')// &
                                       'end program main'

        test_integer_if_merge = expect_exit_status( &
                                source, 9, &
                                '/tmp/ffc_session_if_integer_merge_test')
    end function test_integer_if_merge

    logical function test_real_if_merge()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       'real :: x'//new_line('a')// &
                                       'if (2 < 3) then'//new_line('a')// &
                                       '    x = 4.5'//new_line('a')// &
                                       'else'//new_line('a')// &
                                       '    x = 1.25'//new_line('a')// &
                                       'end if'//new_line('a')// &
                                       'print *, x'//new_line('a')// &
                                       'end program main'

        test_real_if_merge = expect_output( &
                             source, '4.500000'//new_line('a'), &
                             '/tmp/ffc_session_if_real_merge_test')
    end function test_real_if_merge

    logical function test_logical_if_merge()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       'logical :: flag'//new_line('a')// &
                                       'if (2 < 3) then'//new_line('a')// &
                                       '    flag = .true.'//new_line('a')// &
                                       'else'//new_line('a')// &
                                       '    flag = .false.'//new_line('a')// &
                                       'end if'//new_line('a')// &
                                       'if (flag) then'//new_line('a')// &
                                       '    stop 7'//new_line('a')// &
                                       'else'//new_line('a')// &
                                       '    stop 3'//new_line('a')// &
                                       'end if'//new_line('a')// &
                                       'end program main'

        test_logical_if_merge = expect_exit_status( &
                                source, 7, &
                                '/tmp/ffc_session_if_logical_merge_test')
    end function test_logical_if_merge

    logical function test_nested_if_in_do_merge()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       'integer :: i'//new_line('a')// &
                                       'integer :: total'//new_line('a')// &
                                       'logical :: found'//new_line('a')// &
                                       'total = 0'//new_line('a')// &
                                       'found = .false.'//new_line('a')// &
                                       'do i = 1, 3'//new_line('a')// &
                                       '    if (i < 3) then'//new_line('a')// &
                                       '        total = total + i'//new_line('a')// &
                                       '        found = .true.'//new_line('a')// &
                                       '    else'//new_line('a')// &
                                       '        total = total + 4'//new_line('a')// &
                                       '        found = found'//new_line('a')// &
                                       '    end if'//new_line('a')// &
                                       'end do'//new_line('a')// &
                                       'if (found) then'//new_line('a')// &
                                       '    stop total'//new_line('a')// &
                                       'else'//new_line('a')// &
                                       '    stop 1'//new_line('a')// &
                                       'end if'//new_line('a')// &
                                       'end program main'

        test_nested_if_in_do_merge = expect_exit_status( &
                                     source, 7, &
                                     '/tmp/ffc_session_nested_if_do_test')
    end function test_nested_if_in_do_merge

    logical function test_do_in_if_merge()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       'integer :: i'//new_line('a')// &
                                       'integer :: total'//new_line('a')// &
                                       'logical :: flag'//new_line('a')// &
                                       'total = 0'//new_line('a')// &
                                       'flag = .false.'//new_line('a')// &
                                       'if (2 < 3) then'//new_line('a')// &
                                       '    do i = 1, 2'//new_line('a')// &
                                       '        total = total + i'//new_line('a')// &
                                       '        flag = .true.'//new_line('a')// &
                                       '    end do'//new_line('a')// &
                                       'else'//new_line('a')// &
                                       '    total = 9'//new_line('a')// &
                                       '    flag = .false.'//new_line('a')// &
                                       'end if'//new_line('a')// &
                                       'if (flag) then'//new_line('a')// &
                                       '    stop total'//new_line('a')// &
                                       'else'//new_line('a')// &
                                       '    stop 8'//new_line('a')// &
                                       'end if'//new_line('a')// &
                                       'end program main'

        test_do_in_if_merge = expect_exit_status( &
                              source, 3, &
                              '/tmp/ffc_session_do_if_test')
    end function test_do_in_if_merge

end program test_session_if_merge_compiler
