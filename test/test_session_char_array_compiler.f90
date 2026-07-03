program test_session_char_array_compiler
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session character array compiler test ==='

    all_passed = .true.
    if (.not. test_rank1_element_assign_print()) all_passed = .false.
    if (.not. test_rank1_loop_whole_array_print()) all_passed = .false.
    if (.not. test_rank2_multi_item_print()) all_passed = .false.
    if (.not. test_whole_array_after_literal()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: character arrays lower through direct LIRIC session'

contains

    logical function test_rank1_element_assign_print()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=5) :: names(3)'//new_line('a')// &
            '  names(1) = "abc"'//new_line('a')// &
            '  names(2) = "de"'//new_line('a')// &
            '  names(3) = "fghij"'//new_line('a')// &
            '  print *, names(2)'//new_line('a')// &
            'end program main'

        test_rank1_element_assign_print = expect_output( &
            source, ' de   '//new_line('a'), &
            '/tmp/ffc_session_char_array_rank1_test')
    end function test_rank1_element_assign_print

    logical function test_rank1_loop_whole_array_print()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=3) :: v(4)'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  do i = 1, 4'//new_line('a')// &
            '    v(i) = "x"'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  v(2) = "yy"'//new_line('a')// &
            '  print *, v'//new_line('a')// &
            'end program main'

        test_rank1_loop_whole_array_print = expect_output( &
            source, ' x  yy x  x  '//new_line('a'), &
            '/tmp/ffc_session_char_array_whole_test')
    end function test_rank1_loop_whole_array_print

    logical function test_rank2_multi_item_print()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=4) :: g(2,2)'//new_line('a')// &
            '  g(1,1) = "aa"'//new_line('a')// &
            '  g(2,1) = "cc"'//new_line('a')// &
            '  print *, g(2,1), g(1,1)'//new_line('a')// &
            'end program main'

        test_rank2_multi_item_print = expect_output( &
            source, ' cc  aa  '//new_line('a'), &
            '/tmp/ffc_session_char_array_rank2_test')
    end function test_rank2_multi_item_print

    logical function test_whole_array_after_literal()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=3) :: v(3)'//new_line('a')// &
            '  v(1) = "a"'//new_line('a')// &
            '  v(2) = "bb"'//new_line('a')// &
            '  v(3) = "ccc"'//new_line('a')// &
            '  print *, "tag:", v'//new_line('a')// &
            'end program main'

        test_whole_array_after_literal = expect_output( &
            source, ' tag:a  bb ccc'//new_line('a'), &
            '/tmp/ffc_session_char_array_after_literal_test')
    end function test_whole_array_after_literal

end program test_session_char_array_compiler
