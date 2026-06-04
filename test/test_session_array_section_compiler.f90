program test_session_array_section_compiler
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session array section compiler test ==='

    all_passed = .true.
    if (.not. test_print_section()) all_passed = .false.
    if (.not. test_copy_section()) all_passed = .false.

    if (.not. all_passed) stop 1

    print *, 'PASS: array sections lower through direct LIRIC session'

contains

    logical function test_print_section()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: a(1:4)'//new_line('a')// &
                                       '  integer :: b(0:1, 2:3)'//new_line('a')// &
                                       '  a = [1, 2, 3, 4]'//new_line('a')// &
                                       '  b = [5, 6, 7, 8]'//new_line('a')// &
                                       '  print *, a(2:3)'//new_line('a')// &
                                       '  print *, b(0:1, 3:3)'//new_line('a')// &
                                       'end program main'

        test_print_section = expect_output( &
            source, '           2           3'//new_line('a')// &
                    '           7           8'//new_line('a'), &
            '/tmp/ffc_session_array_section_print_test')
    end function test_print_section

    logical function test_copy_section()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: a(1:4)'//new_line('a')// &
                                       '  integer :: c(2)'//new_line('a')// &
                                       '  integer :: b(0:1, 2:3)'//new_line('a')// &
                                       '  integer :: d(0:1, 3:3)'//new_line('a')// &
                                       '  a = [1, 2, 3, 4]'//new_line('a')// &
                                       '  b = [5, 6, 7, 8]'//new_line('a')// &
                                       '  c = a(2:3)'//new_line('a')// &
                                       '  d = b(0:1, 3:3)'//new_line('a')// &
                                       '  print *, c(1) + c(2)'//new_line('a')// &
                                       '  print *, d(0, 3) + d(1, 3)'//new_line('a')// &
                                       'end program main'

        test_copy_section = expect_output( &
            source, '           5'//new_line('a')// &
                    '          15'//new_line('a'), &
            '/tmp/ffc_session_array_section_copy_test')
    end function test_copy_section
end program test_session_array_section_compiler
