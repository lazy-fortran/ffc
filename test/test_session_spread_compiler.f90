program test_session_spread_compiler
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session spread compiler test ==='

    all_passed = .true.
    if (.not. test_spread_dim2()) all_passed = .false.
    if (.not. test_spread_dim1()) all_passed = .false.
    if (.not. test_spread_real()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: spread lowers through direct LIRIC session'

contains

    ! spread(a, 2, 2) replicates a rank-1 array along a new trailing dimension:
    ! r(i,j) = a(i), so the two columns each repeat 1,2,3.
    logical function test_spread_dim2()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(3)'//new_line('a')// &
            '  integer :: r(3, 2)'//new_line('a')// &
            '  integer :: i, j'//new_line('a')// &
            '  a = [1, 2, 3]'//new_line('a')// &
            '  r = spread(a, 2, 2)'//new_line('a')// &
            '  do j = 1, 2'//new_line('a')// &
            '     do i = 1, 3'//new_line('a')// &
            '        print *, r(i, j)'//new_line('a')// &
            '     end do'//new_line('a')// &
            '  end do'//new_line('a')// &
            'end program main'
        test_spread_dim2 = expect_output( &
            source, '           1'//new_line('a')// &
                    '           2'//new_line('a')// &
                    '           3'//new_line('a')// &
                    '           1'//new_line('a')// &
                    '           2'//new_line('a')// &
                    '           3'//new_line('a'), &
            '/tmp/ffc_session_spread_dim2_test')
    end function test_spread_dim2

    ! spread(a, 1, 2) replicates along a new leading dimension: s(i,j) = a(j),
    ! so each column j holds a(j) repeated down the two rows.
    logical function test_spread_dim1()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(3)'//new_line('a')// &
            '  integer :: s(2, 3)'//new_line('a')// &
            '  integer :: i, j'//new_line('a')// &
            '  a = [1, 2, 3]'//new_line('a')// &
            '  s = spread(a, 1, 2)'//new_line('a')// &
            '  do j = 1, 3'//new_line('a')// &
            '     do i = 1, 2'//new_line('a')// &
            '        print *, s(i, j)'//new_line('a')// &
            '     end do'//new_line('a')// &
            '  end do'//new_line('a')// &
            'end program main'
        test_spread_dim1 = expect_output( &
            source, '           1'//new_line('a')// &
                    '           1'//new_line('a')// &
                    '           2'//new_line('a')// &
                    '           2'//new_line('a')// &
                    '           3'//new_line('a')// &
                    '           3'//new_line('a'), &
            '/tmp/ffc_session_spread_dim1_test')
    end function test_spread_dim1

    ! spread keeps the source element kind for real arrays.
    logical function test_spread_real()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real :: a(2)'//new_line('a')// &
            '  real :: r(2, 2)'//new_line('a')// &
            '  integer :: i, j'//new_line('a')// &
            '  a = [1.5, 2.5]'//new_line('a')// &
            '  r = spread(a, 2, 2)'//new_line('a')// &
            '  do j = 1, 2'//new_line('a')// &
            '     do i = 1, 2'//new_line('a')// &
            '        print *, r(i, j)'//new_line('a')// &
            '     end do'//new_line('a')// &
            '  end do'//new_line('a')// &
            'end program main'
        test_spread_real = expect_output( &
            source, '   1.50000000    '//new_line('a')// &
                    '   2.50000000    '//new_line('a')// &
                    '   1.50000000    '//new_line('a')// &
                    '   2.50000000    '//new_line('a'), &
            '/tmp/ffc_session_spread_real_test')
    end function test_spread_real

end program test_session_spread_compiler
