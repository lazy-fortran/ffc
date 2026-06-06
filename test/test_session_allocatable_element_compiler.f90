program test_session_allocatable_element
    ! Element read/write on an already-allocated 1-D integer allocatable (#244
    ! slice B2a). Fill in a loop, read back, sum, and check the result through
    ! the process exit status.
    use ffc_test_support, only: expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== direct session allocatable element compiler test ==='

    all_passed = .true.
    if (.not. test_write_then_read_single()) all_passed = .false.
    if (.not. test_fill_loop_and_sum()) all_passed = .false.
    if (.not. test_element_expression_write()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: 1-D integer allocatable element access lowers through LIRIC'

contains

    logical function test_write_then_read_single()
        ! a(3) = 7; stop a(3) -> exit status 7.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, allocatable :: a(:)'//new_line('a')// &
            '  allocate(a(5))'//new_line('a')// &
            '  a(3) = 7'//new_line('a')// &
            '  stop a(3)'//new_line('a')// &
            'end program main'

        test_write_then_read_single = expect_exit_status( &
            source, 7, '/tmp/ffc_alloc_elem_single')
    end function test_write_then_read_single

    logical function test_fill_loop_and_sum()
        ! a(i) = i for i = 1..4, then sum a(i): 1+2+3+4 = 10.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, allocatable :: a(:)'//new_line('a')// &
            '  integer :: i, total'//new_line('a')// &
            '  allocate(a(4))'//new_line('a')// &
            '  do i = 1, 4'//new_line('a')// &
            '    a(i) = i'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  total = 0'//new_line('a')// &
            '  do i = 1, 4'//new_line('a')// &
            '    total = total + a(i)'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  stop total'//new_line('a')// &
            'end program main'

        test_fill_loop_and_sum = expect_exit_status( &
            source, 10, '/tmp/ffc_alloc_elem_sum')
    end function test_fill_loop_and_sum

    logical function test_element_expression_write()
        ! a(2) = 4*2 + 1 = 9; stop a(2).
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, allocatable :: a(:)'//new_line('a')// &
            '  allocate(a(3))'//new_line('a')// &
            '  a(2) = 4 * 2 + 1'//new_line('a')// &
            '  stop a(2)'//new_line('a')// &
            'end program main'

        test_element_expression_write = expect_exit_status( &
            source, 9, '/tmp/ffc_alloc_elem_expr')
    end function test_element_expression_write

end program test_session_allocatable_element
