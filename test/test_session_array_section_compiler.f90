program test_session_array_section_compiler
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session array section compiler test ==='

    all_passed = .true.
    if (.not. test_print_section()) all_passed = .false.
    if (.not. test_copy_section()) all_passed = .false.
    if (.not. test_whole_array_copy()) all_passed = .false.
    if (.not. test_elementwise_sections()) all_passed = .false.
    if (.not. test_sum_section()) all_passed = .false.
    if (.not. test_section_after_string()) all_passed = .false.

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

    logical function test_whole_array_copy()
        ! b = a(2:4): whole-array assignment from a rank-1 integer section.
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: a(4), b(3)'//new_line('a')// &
                                       '  a = [1, 2, 3, 4]'//new_line('a')// &
                                       '  b = a(2:4)'//new_line('a')// &
                                       '  print *, b'//new_line('a')// &
                                       'end program main'

        test_whole_array_copy = expect_output( &
            source, '           2           3           4'//new_line('a'), &
            '/tmp/ffc_session_array_section_whole_test')
    end function test_whole_array_copy

    logical function test_elementwise_sections()
        ! c = a(1:3) + d(2:4): elementwise op between two conformable sections.
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: a(4), d(4), c(3)'//new_line('a')// &
                                       '  a = [10, 20, 30, 40]'//new_line('a')// &
                                       '  d = [1, 2, 3, 4]'//new_line('a')// &
                                       '  c = a(1:3) + d(2:4)'//new_line('a')// &
                                       '  print *, c'//new_line('a')// &
                                       'end program main'

        test_elementwise_sections = expect_output( &
            source, '          12          23          34'//new_line('a'), &
            '/tmp/ffc_session_array_section_elem_test')
    end function test_elementwise_sections

    logical function test_sum_section()
        ! sum(a(lo:hi)) reduces over the section extent only.
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: a(6)'//new_line('a')// &
                                       '  a = [1, 2, 3, 4, 5, 6]'//new_line('a')// &
                                       '  print *, sum(a(2:5))'//new_line('a')// &
                                       'end program main'

        test_sum_section = expect_output( &
            source, '          14'//new_line('a'), &
            '/tmp/ffc_session_array_section_sum_test')
    end function test_sum_section

    logical function test_section_after_string()
        ! 'tag', a(lo:hi): a section among other list-directed print items.
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: a(5)'//new_line('a')// &
                                       '  a = [1, 2, 3, 4, 5]'//new_line('a')// &
                                       "  print *, 'vals:', a(2:4)"//new_line('a')// &
                                       'end program main'

        test_section_after_string = expect_output( &
            source, ' vals:           2           3           4'//new_line('a'), &
            '/tmp/ffc_session_array_section_after_string_test')
    end function test_section_after_string
end program test_session_array_section_compiler
