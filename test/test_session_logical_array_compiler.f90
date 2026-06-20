program test_session_logical_array
    ! Logical fixed-size arrays through direct LIRIC lowering (Epic E3, #264):
    ! declaration, element write/read, scalar broadcast, whole-array copy, and
    ! rank-2 element access. Logical occupies an i32 slot, so element storage
    ! mirrors integer arrays while list-directed print formats T/F.
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session logical array compiler test ==='

    all_passed = .true.
    if (.not. test_element_write_and_print()) all_passed = .false.
    if (.not. test_broadcast_and_copy()) all_passed = .false.
    if (.not. test_rank2_elements()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: logical arrays lower through direct LIRIC session'

contains

    logical function test_element_write_and_print()
        ! Per-element assignment of logical literals prints as T/F.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  logical :: flags(3)'//new_line('a')// &
            '  flags(1) = .true.'//new_line('a')// &
            '  flags(2) = .false.'//new_line('a')// &
            '  flags(3) = .true.'//new_line('a')// &
            '  print *, flags(1), flags(2), flags(3)'//new_line('a')// &
            'end program main'

        test_element_write_and_print = expect_output( &
            source, ' T F T'//new_line('a'), '/tmp/ffc_logical_arr_elem')
    end function test_element_write_and_print

    logical function test_broadcast_and_copy()
        ! Whole-array scalar broadcast and array-to-array copy.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  logical :: a(3)'//new_line('a')// &
            '  logical :: b(3)'//new_line('a')// &
            '  a = .true.'//new_line('a')// &
            '  b = a'//new_line('a')// &
            '  print *, b(1), b(2), b(3)'//new_line('a')// &
            'end program main'

        test_broadcast_and_copy = expect_output( &
            source, ' T T T'//new_line('a'), '/tmp/ffc_logical_arr_copy')
    end function test_broadcast_and_copy

    logical function test_rank2_elements()
        ! Rank-2 logical element write and read in nested loops.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  logical :: g(2,2)'//new_line('a')// &
            '  integer :: i, j'//new_line('a')// &
            '  g(1,1) = .true.'//new_line('a')// &
            '  g(1,2) = .false.'//new_line('a')// &
            '  g(2,1) = .false.'//new_line('a')// &
            '  g(2,2) = .true.'//new_line('a')// &
            '  do i = 1, 2'//new_line('a')// &
            '    do j = 1, 2'//new_line('a')// &
            '      print *, g(i,j)'//new_line('a')// &
            '    end do'//new_line('a')// &
            '  end do'//new_line('a')// &
            'end program main'

        test_rank2_elements = expect_output( &
            source, ' T'//new_line('a')//' F'//new_line('a')// &
            ' F'//new_line('a')//' T'//new_line('a'), &
            '/tmp/ffc_logical_arr_rank2')
    end function test_rank2_elements

end program test_session_logical_array
