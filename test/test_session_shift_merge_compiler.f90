program test_session_shift_merge_compiler
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session cshift/eoshift/merge compiler test ==='

    all_passed = .true.
    if (.not. test_cshift_positive()) all_passed = .false.
    if (.not. test_cshift_negative()) all_passed = .false.
    if (.not. test_eoshift_fills_zero()) all_passed = .false.
    if (.not. test_merge_integer()) all_passed = .false.
    if (.not. test_merge_real()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: cshift/eoshift/merge lower through direct LIRIC session'

contains

    ! cshift(a, 1) rotates elements left by one position, wrapping the first
    ! element to the end.
    logical function test_cshift_positive()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(4)'//new_line('a')// &
            '  integer :: r(4)'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  a = [1, 2, 3, 4]'//new_line('a')// &
            '  r = cshift(a, 1)'//new_line('a')// &
            '  do i = 1, 4'//new_line('a')// &
            '     print *, r(i)'//new_line('a')// &
            '  end do'//new_line('a')// &
            'end program main'
        test_cshift_positive = expect_output( &
            source, '           2'//new_line('a')// &
                    '           3'//new_line('a')// &
                    '           4'//new_line('a')// &
                    '           1'//new_line('a'), &
            '/tmp/ffc_session_cshift_pos_test')
    end function test_cshift_positive

    ! A negative shift rotates to the right.
    logical function test_cshift_negative()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(4)'//new_line('a')// &
            '  integer :: r(4)'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  a = [1, 2, 3, 4]'//new_line('a')// &
            '  r = cshift(a, -1)'//new_line('a')// &
            '  do i = 1, 4'//new_line('a')// &
            '     print *, r(i)'//new_line('a')// &
            '  end do'//new_line('a')// &
            'end program main'
        test_cshift_negative = expect_output( &
            source, '           4'//new_line('a')// &
                    '           1'//new_line('a')// &
                    '           2'//new_line('a')// &
                    '           3'//new_line('a'), &
            '/tmp/ffc_session_cshift_neg_test')
    end function test_cshift_negative

    ! eoshift(a, 1) shifts left and fills the vacated tail with zero.
    logical function test_eoshift_fills_zero()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(4)'//new_line('a')// &
            '  integer :: r(4)'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  a = [1, 2, 3, 4]'//new_line('a')// &
            '  r = eoshift(a, 1)'//new_line('a')// &
            '  do i = 1, 4'//new_line('a')// &
            '     print *, r(i)'//new_line('a')// &
            '  end do'//new_line('a')// &
            'end program main'
        test_eoshift_fills_zero = expect_output( &
            source, '           2'//new_line('a')// &
                    '           3'//new_line('a')// &
                    '           4'//new_line('a')// &
                    '           0'//new_line('a'), &
            '/tmp/ffc_session_eoshift_zero_test')
    end function test_eoshift_fills_zero

    ! merge picks from the first array where the mask is .true., else the second.
    logical function test_merge_integer()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(3)'//new_line('a')// &
            '  integer :: b(3)'//new_line('a')// &
            '  integer :: r(3)'//new_line('a')// &
            '  logical :: m(3)'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  a = [1, 2, 3]'//new_line('a')// &
            '  b = [4, 5, 6]'//new_line('a')// &
            '  m = [.true., .false., .true.]'//new_line('a')// &
            '  r = merge(a, b, m)'//new_line('a')// &
            '  do i = 1, 3'//new_line('a')// &
            '     print *, r(i)'//new_line('a')// &
            '  end do'//new_line('a')// &
            'end program main'
        test_merge_integer = expect_output( &
            source, '           1'//new_line('a')// &
                    '           5'//new_line('a')// &
                    '           3'//new_line('a'), &
            '/tmp/ffc_session_merge_int_test')
    end function test_merge_integer

    ! merge over real arrays keeps the result element kind.
    logical function test_merge_real()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real :: a(2)'//new_line('a')// &
            '  real :: b(2)'//new_line('a')// &
            '  real :: r(2)'//new_line('a')// &
            '  logical :: m(2)'//new_line('a')// &
            '  a = [1.5, 2.5]'//new_line('a')// &
            '  b = [3.5, 4.5]'//new_line('a')// &
            '  m = [.false., .true.]'//new_line('a')// &
            '  r = merge(a, b, m)'//new_line('a')// &
            '  print *, r(1)'//new_line('a')// &
            '  print *, r(2)'//new_line('a')// &
            'end program main'
        test_merge_real = expect_output( &
            source, '   3.50000000    '//new_line('a')// &
                    '   2.50000000    '//new_line('a'), &
            '/tmp/ffc_session_merge_real_test')
    end function test_merge_real

end program test_session_shift_merge_compiler
