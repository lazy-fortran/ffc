program test_session_whole_array_compare_compiler
    use ffc_test_support, only: expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== direct session whole-array comparison compiler test ==='

    all_passed = .true.
    if (.not. test_integer_relational_masks()) all_passed = .false.
    if (.not. test_real_relational_masks()) all_passed = .false.
    if (.not. test_not_equal_mask()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: whole-array comparisons lower to logical arrays'

contains

    logical function test_integer_relational_masks()
        ! Exact ties (index 3, both arrays 3) exercise strict "<"/">" at
        ! their boundary: a select_value branch-and-phi, not a materialized
        ! comparison flag stored straight to memory, keeps ties correct.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(4), b(4)'//new_line('a')// &
            '  logical :: gt(4), ge(4), lt(4), le(4), eq(4)'//new_line('a')// &
            '  a = [1, 5, 3, -7]'//new_line('a')// &
            '  b = [2, 5, 3, 9]'//new_line('a')// &
            '  gt = a > b'//new_line('a')// &
            '  ge = a >= b'//new_line('a')// &
            '  lt = a < b'//new_line('a')// &
            '  le = a <= b'//new_line('a')// &
            '  eq = a == b'//new_line('a')// &
            '  if (gt(1) .or. gt(2) .or. gt(3) .or. gt(4)) error stop 1'// &
            new_line('a')// &
            '  if (.not. (ge(2) .and. ge(3))) error stop 2'//new_line('a')// &
            '  if (ge(1) .or. ge(4)) error stop 3'//new_line('a')// &
            '  if (.not. (lt(1) .and. lt(4))) error stop 4'//new_line('a')// &
            '  if (lt(2) .or. lt(3)) error stop 5'//new_line('a')// &
            '  if (.not. (le(1) .and. le(2) .and. le(3) .and. le(4))) &'// &
            new_line('a')// &
            '    error stop 6'//new_line('a')// &
            '  if (.not. (eq(2) .and. eq(3))) error stop 7'//new_line('a')// &
            '  if (eq(1) .or. eq(4)) error stop 8'//new_line('a')// &
            'end program main'

        test_integer_relational_masks = expect_exit_status( &
            source, 0, '/tmp/ffc_whole_array_compare_int')
    end function test_integer_relational_masks

    logical function test_real_relational_masks()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real(8) :: x(3), y(3)'//new_line('a')// &
            '  logical :: gt(3), lt(3)'//new_line('a')// &
            '  x = [1.0d0, 5.0d0, 3.0d0]'//new_line('a')// &
            '  y = [2.0d0, 2.0d0, 3.0d0]'//new_line('a')// &
            '  gt = x > y'//new_line('a')// &
            '  lt = x < y'//new_line('a')// &
            '  if (.not. gt(2)) error stop 1'//new_line('a')// &
            '  if (gt(1) .or. gt(3)) error stop 2'//new_line('a')// &
            '  if (.not. lt(1)) error stop 3'//new_line('a')// &
            '  if (lt(2) .or. lt(3)) error stop 4'//new_line('a')// &
            'end program main'

        test_real_relational_masks = expect_exit_status( &
            source, 0, '/tmp/ffc_whole_array_compare_real')
    end function test_real_relational_masks

    logical function test_not_equal_mask()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(3), b(3)'//new_line('a')// &
            '  logical :: ne(3)'//new_line('a')// &
            '  a = [3, 5, 3]'//new_line('a')// &
            '  b = [3, 3, 5]'//new_line('a')// &
            '  ne = a /= b'//new_line('a')// &
            '  if (ne(1)) error stop 1'//new_line('a')// &
            '  if (.not. (ne(2) .and. ne(3))) error stop 2'//new_line('a')// &
            'end program main'

        test_not_equal_mask = expect_exit_status( &
            source, 0, '/tmp/ffc_whole_array_compare_ne')
    end function test_not_equal_mask

end program test_session_whole_array_compare_compiler
