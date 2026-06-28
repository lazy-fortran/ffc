program test_session_array_intrinsics_compiler
    use ffc_test_support, only: expect_output, expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== direct session array intrinsic compiler test ==='

    all_passed = .true.
    if (.not. test_rank2_array_intrinsics()) all_passed = .false.
    if (.not. test_lbound_default_lower()) all_passed = .false.
    if (.not. test_lbound_nondefault_lower()) all_passed = .false.
    if (.not. test_ubound_default()) all_passed = .false.
    if (.not. test_ubound_nondefault()) all_passed = .false.
    if (.not. test_count_true_elements()) all_passed = .false.
    if (.not. test_any_returns_nonzero()) all_passed = .false.
    if (.not. test_any_all_false_returns_zero()) all_passed = .false.
    if (.not. test_all_all_true()) all_passed = .false.
    if (.not. test_all_mixed_returns_zero()) all_passed = .false.
    if (.not. test_maxloc_integer_dim()) all_passed = .false.
    if (.not. test_minloc_integer_dim()) all_passed = .false.
    if (.not. test_maxloc_real_first_tie()) all_passed = .false.
    if (.not. test_minloc_with_mask()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: array intrinsics lower through direct LIRIC'

contains

    logical function test_rank2_array_intrinsics()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(1:2, 1:2)'//new_line('a')// &
            '  integer :: r(1:4)'//new_line('a')// &
            '  integer :: w(1:4)'//new_line('a')// &
            '  integer :: s'//new_line('a')// &
            '  integer :: dims(2)'//new_line('a')// &
            '  a = [1, 2, 3, 4]'//new_line('a')// &
            '  w = [1, 1, 1, 1]'//new_line('a')// &
            '  dims = shape(a)'//new_line('a')// &
            '  print *, dims(1)'//new_line('a')// &
            '  print *, dims(2)'//new_line('a')// &
            '  print *, size(a)'//new_line('a')// &
            '  print *, sum(a)'//new_line('a')// &
            '  print *, maxval(a)'//new_line('a')// &
            '  print *, minval(a)'//new_line('a')// &
            '  r = reshape(a, [4])'//new_line('a')// &
            '  s = dot_product(r, w)'//new_line('a')// &
            '  print *, r(1)'//new_line('a')// &
            '  print *, r(4)'//new_line('a')// &
            '  print *, s'//new_line('a')// &
            'end program main'

        test_rank2_array_intrinsics = expect_output( &
            source, '           2'//new_line('a')// &
            '           2'//new_line('a')// &
            '           4'//new_line('a')// &
            '          10'//new_line('a')// &
            '           4'//new_line('a')// &
            '           1'//new_line('a')// &
            '           1'//new_line('a')// &
            '           4'//new_line('a')// &
            '          10'//new_line('a'), &
            '/tmp/ffc_session_array_intrinsics_test')
    end function test_rank2_array_intrinsics

    logical function test_lbound_default_lower()
        ! lbound(a, 1) on a(5) is 1 (default lower bound).
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(5)'//new_line('a')// &
            '  stop lbound(a, 1)'//new_line('a')// &
            'end program main'
        test_lbound_default_lower = expect_exit_status( &
            source, 1, '/tmp/ffc_session_lbound_default_test')
    end function test_lbound_default_lower

    logical function test_lbound_nondefault_lower()
        ! lbound(a, 1) on a(3:7) is 3.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(3:7)'//new_line('a')// &
            '  stop lbound(a, 1)'//new_line('a')// &
            'end program main'
        test_lbound_nondefault_lower = expect_exit_status( &
            source, 3, '/tmp/ffc_session_lbound_nondefault_test')
    end function test_lbound_nondefault_lower

    logical function test_ubound_default()
        ! ubound(a, 1) on a(5) is 5.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(5)'//new_line('a')// &
            '  stop ubound(a, 1)'//new_line('a')// &
            'end program main'
        test_ubound_default = expect_exit_status( &
            source, 5, '/tmp/ffc_session_ubound_default_test')
    end function test_ubound_default

    logical function test_ubound_nondefault()
        ! ubound(a, 1) on a(3:7) is 7.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(3:7)'//new_line('a')// &
            '  stop ubound(a, 1)'//new_line('a')// &
            'end program main'
        test_ubound_nondefault = expect_exit_status( &
            source, 7, '/tmp/ffc_session_ubound_nondefault_test')
    end function test_ubound_nondefault

    logical function test_count_true_elements()
        ! count(mask) with 3 nonzero elements in a 4-element integer array.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: mask(4)'//new_line('a')// &
            '  mask = [1, 0, 1, 1]'//new_line('a')// &
            '  stop count(mask)'//new_line('a')// &
            'end program main'
        test_count_true_elements = expect_exit_status( &
            source, 3, '/tmp/ffc_session_count_test')
    end function test_count_true_elements

    logical function test_any_returns_nonzero()
        ! any(mask) returns nonzero when at least one element is nonzero.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: mask(3)'//new_line('a')// &
            '  mask = [0, 1, 0]'//new_line('a')// &
            '  stop any(mask)'//new_line('a')// &
            'end program main'
        ! stop exits with 1 (any returns nonzero OR of elements).
        test_any_returns_nonzero = expect_exit_status( &
            source, 1, '/tmp/ffc_session_any_nonzero_test')
    end function test_any_returns_nonzero

    logical function test_any_all_false_returns_zero()
        ! any(mask) returns 0 when all elements are 0.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: mask(3)'//new_line('a')// &
            '  mask = [0, 0, 0]'//new_line('a')// &
            '  stop any(mask)'//new_line('a')// &
            'end program main'
        test_any_all_false_returns_zero = expect_exit_status( &
            source, 0, '/tmp/ffc_session_any_false_test')
    end function test_any_all_false_returns_zero

    logical function test_all_all_true()
        ! all(mask) returns 1 when every element is nonzero.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: mask(3)'//new_line('a')// &
            '  mask = [1, 1, 1]'//new_line('a')// &
            '  stop all(mask)'//new_line('a')// &
            'end program main'
        test_all_all_true = expect_exit_status( &
            source, 1, '/tmp/ffc_session_all_true_test')
    end function test_all_all_true

    logical function test_all_mixed_returns_zero()
        ! all(mask) returns 0 when any element is 0.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: mask(3)'//new_line('a')// &
            '  mask = [1, 0, 1]'//new_line('a')// &
            '  stop all(mask)'//new_line('a')// &
            'end program main'
        test_all_mixed_returns_zero = expect_exit_status( &
            source, 0, '/tmp/ffc_session_all_mixed_test')
    end function test_all_mixed_returns_zero

    logical function test_maxloc_integer_dim()
        ! maxloc(a, dim=1) on a rank-1 integer array returns the 1-based index
        ! of the largest element (a(4)=9 here).
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(5)'//new_line('a')// &
            '  a = [3, 7, 2, 9, 4]'//new_line('a')// &
            '  stop maxloc(a, dim=1)'//new_line('a')// &
            'end program main'
        test_maxloc_integer_dim = expect_exit_status( &
            source, 4, '/tmp/ffc_session_maxloc_int_test')
    end function test_maxloc_integer_dim

    logical function test_minloc_integer_dim()
        ! minloc(a, 1) returns the index of the smallest element (a(3)=2).
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(5)'//new_line('a')// &
            '  a = [3, 7, 2, 9, 4]'//new_line('a')// &
            '  stop minloc(a, 1)'//new_line('a')// &
            'end program main'
        test_minloc_integer_dim = expect_exit_status( &
            source, 3, '/tmp/ffc_session_minloc_int_test')
    end function test_minloc_integer_dim

    logical function test_maxloc_real_first_tie()
        ! On a tie, maxloc returns the first occurrence (both a(2),a(4)=9.0).
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real :: a(5)'//new_line('a')// &
            '  a = [3.0, 9.0, 2.0, 9.0, 4.0]'//new_line('a')// &
            '  stop maxloc(a, dim=1)'//new_line('a')// &
            'end program main'
        test_maxloc_real_first_tie = expect_exit_status( &
            source, 2, '/tmp/ffc_session_maxloc_tie_test')
    end function test_maxloc_real_first_tie

    logical function test_minloc_with_mask()
        ! minloc with a logical mask considers only masked elements; the
        ! smallest masked value is a(5)=4 among indices 1,3,5 (3,2,4).
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real :: a(5)'//new_line('a')// &
            '  logical :: m(5)'//new_line('a')// &
            '  a = [3.0, 7.0, 2.0, 9.0, 4.0]'//new_line('a')// &
            '  m = [.true., .false., .false., .false., .true.]'//new_line('a')// &
            '  stop minloc(a, dim=1, mask=m)'//new_line('a')// &
            'end program main'
        ! Among masked indices 1 (3.0) and 5 (4.0), the minimum is index 1.
        test_minloc_with_mask = expect_exit_status( &
            source, 1, '/tmp/ffc_session_minloc_mask_test')
    end function test_minloc_with_mask

end program test_session_array_intrinsics_compiler
