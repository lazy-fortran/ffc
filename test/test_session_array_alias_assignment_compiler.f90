program test_session_array_alias_assignment_compiler
    !! Overlapping array section assignment must behave as if the right-hand
    !! side were fully evaluated before any element of the target is written
    !! (Fortran 2018 clause 10.2.1.3). Expected values are the gfortran output
    !! for the same program.
    use ffc_test_support, only: expect_output
    implicit none

    print *, '=== direct session overlapping array assignment compiler test ==='
    if (.not. test_forward_overlap()) stop 1
    if (.not. test_backward_overlap()) stop 1
    if (.not. test_strided_overlap()) stop 1
    print *, 'PASS: overlapping array section assignment is alias-safe'

contains

    logical function test_forward_overlap()
        !! a(2:5) = a(1:4): target starts above the source, so an ascending
        !! element-by-element copy would read already-overwritten elements.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(6)'//new_line('a')// &
            '  a = [1, 2, 3, 4, 5, 6]'//new_line('a')// &
            '  a(2:5) = a(1:4)'//new_line('a')// &
            '  print *, a(1), a(2), a(3), a(4), a(5), a(6)'//new_line('a')// &
            'end program main'

        test_forward_overlap = expect_output( &
            source, &
            '           1           1           2           3           4'// &
            '           6'//new_line('a'), &
            '/tmp/ffc_session_array_alias_forward_test')
    end function test_forward_overlap

    logical function test_backward_overlap()
        !! a(1:4) = a(2:5): the reverse overlap direction.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(6)'//new_line('a')// &
            '  a = [1, 2, 3, 4, 5, 6]'//new_line('a')// &
            '  a(1:4) = a(2:5)'//new_line('a')// &
            '  print *, a(1), a(2), a(3), a(4), a(5), a(6)'//new_line('a')// &
            'end program main'

        test_backward_overlap = expect_output( &
            source, &
            '           2           3           4           5           5'// &
            '           6'//new_line('a'), &
            '/tmp/ffc_session_array_alias_backward_test')
    end function test_backward_overlap

    logical function test_strided_overlap()
        !! A noncontiguous section assigned from a section of the same array.
        !! a(2:6:2) = a(1:3) writes a(2), a(4), a(6) while reading a(1), a(2),
        !! a(3), so a(2) must be read at its pre-assignment value.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(6)'//new_line('a')// &
            '  a = [1, 2, 3, 4, 5, 6]'//new_line('a')// &
            '  a(2:6:2) = a(1:3)'//new_line('a')// &
            '  print *, a(1), a(2), a(3), a(4), a(5), a(6)'//new_line('a')// &
            'end program main'

        test_strided_overlap = expect_output( &
            source, &
            '           1           1           3           2           5'// &
            '           3'//new_line('a'), &
            '/tmp/ffc_session_array_alias_strided_test')
    end function test_strided_overlap

end program test_session_array_alias_assignment_compiler
