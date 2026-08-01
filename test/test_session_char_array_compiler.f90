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
    if (.not. test_elements_are_contiguous_at_declared_stride()) &
        all_passed = .false.
    if (.not. test_element_lengths_are_the_declared_length()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: character arrays lower through direct LIRIC session'

contains

    logical function test_elements_are_contiguous_at_declared_stride()
        ! Printing every element of a rank-1 character array back to back shows
        ! the storage as one run of characters. With a contiguous layout the
        ! elements sit at a stride of the declared length, so the concatenated
        ! output has no gaps and no repetition.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=3) :: c(4)'//new_line('a')// &
            '  c(1) = "aaa"'//new_line('a')// &
            '  c(2) = "bbb"'//new_line('a')// &
            '  c(3) = "ccc"'//new_line('a')// &
            '  c(4) = "ddd"'//new_line('a')// &
            '  print *, "[", c(1), c(2), c(3), c(4), "]"'//new_line('a')// &
            'end program main'

        test_elements_are_contiguous_at_declared_stride = expect_output( &
            source, ' [aaabbbcccddd]'//new_line('a'), &
            '/tmp/ffc_session_char_array_stride_test')
    end function test_elements_are_contiguous_at_declared_stride

    logical function test_element_lengths_are_the_declared_length()
        ! Each element carries the array's declared element length, including
        ! the blank padding of a shorter assigned value.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=5) :: c(3)'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  c(1) = "apple"'//new_line('a')// &
            '  c(2) = "fig"'//new_line('a')// &
            '  c(3) = "peach"'//new_line('a')// &
            '  do i = 1, 3'//new_line('a')// &
            '    print *, i, len(c(i)), "[", c(i), "]"'//new_line('a')// &
            '  end do'//new_line('a')// &
            'end program main'

        test_element_lengths_are_the_declared_length = expect_output( &
            source, &
            '           1           5 [apple]'//new_line('a')// &
            '           2           5 [fig  ]'//new_line('a')// &
            '           3           5 [peach]'//new_line('a'), &
            '/tmp/ffc_session_char_array_len_test')
    end function test_element_lengths_are_the_declared_length

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
