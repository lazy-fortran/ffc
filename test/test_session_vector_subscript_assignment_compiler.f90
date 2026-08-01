program test_session_vector_subscript_assignment_compiler
    ! Behavioural coverage for assignment to a vector subscript, a(v) = rhs.
    ! Fortran requires the RHS and the index vector to be evaluated before any
    ! element of the target is redefined, so a self-referential scatter such as
    ! a(v) = a(w) must observe the original values of a.
    use ffc_test_support, only: expect_output, expect_error_contains
    implicit none

    logical :: all_passed

    print *, '=== direct session vector subscript assignment test ==='

    all_passed = .true.
    if (.not. test_reordered_scatter()) all_passed = .false.
    if (.not. test_repeated_indices()) all_passed = .false.
    if (.not. test_self_referential()) all_passed = .false.
    if (.not. test_scalar_rhs()) all_passed = .false.
    if (.not. test_real_scatter()) all_passed = .false.
    if (.not. test_literal_vector()) all_passed = .false.
    if (.not. test_out_of_bounds_rejected()) all_passed = .false.
    if (.not. test_nonconformable_rejected()) all_passed = .false.
    if (.not. test_noninteger_vector_rejected()) all_passed = .false.

    if (.not. all_passed) stop 1

    print *, 'PASS: vector subscript assignment lowers as an alias-safe scatter'

contains

    logical function test_reordered_scatter()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(5), v(3)'//new_line('a')// &
            '  a = [1, 2, 3, 4, 5]'//new_line('a')// &
            '  v = [5, 1, 3]'//new_line('a')// &
            '  a(v) = [10, 20, 30]'//new_line('a')// &
            '  print *, a'//new_line('a')// &
            'end program main'

        test_reordered_scatter = expect_output( &
            source, &
            '          20           2          30           4          10'// &
            new_line('a'), &
            '/tmp/ffc_session_vector_subscript_reorder_test')
    end function test_reordered_scatter

    logical function test_repeated_indices()
        ! Repeated indices are not standard-conforming for a variable target,
        ! but gfortran scatters in array element order so the last write wins.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(4), v(3)'//new_line('a')// &
            '  a = 0'//new_line('a')// &
            '  v = [2, 2, 4]'//new_line('a')// &
            '  a(v) = [7, 8, 9]'//new_line('a')// &
            '  print *, a'//new_line('a')// &
            'end program main'

        test_repeated_indices = expect_output( &
            source, &
            '           0           8           0           9'//new_line('a'), &
            '/tmp/ffc_session_vector_subscript_repeat_test')
    end function test_repeated_indices

    logical function test_self_referential()
        ! a(v) = a(w) must read the original a before any element is stored.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(4), v(4), w(4)'//new_line('a')// &
            '  a = [1, 2, 3, 4]'//new_line('a')// &
            '  v = [1, 2, 3, 4]'//new_line('a')// &
            '  w = [2, 1, 4, 3]'//new_line('a')// &
            '  a(v) = a(w)'//new_line('a')// &
            '  print *, a'//new_line('a')// &
            'end program main'

        test_self_referential = expect_output( &
            source, &
            '           2           1           4           3'//new_line('a'), &
            '/tmp/ffc_session_vector_subscript_self_test')
    end function test_self_referential

    logical function test_scalar_rhs()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(5), v(2)'//new_line('a')// &
            '  a = [1, 2, 3, 4, 5]'//new_line('a')// &
            '  v = [4, 2]'//new_line('a')// &
            '  a(v) = 99'//new_line('a')// &
            '  print *, a'//new_line('a')// &
            'end program main'

        test_scalar_rhs = expect_output( &
            source, &
            '           1          99           3          99           5'// &
            new_line('a'), &
            '/tmp/ffc_session_vector_subscript_scalar_test')
    end function test_scalar_rhs

    logical function test_real_scatter()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real(8) :: a(3)'//new_line('a')// &
            '  integer :: v(2)'//new_line('a')// &
            '  a = [1.0d0, 2.0d0, 3.0d0]'//new_line('a')// &
            '  v = [3, 1]'//new_line('a')// &
            '  a(v) = [7.5d0, 8.5d0]'//new_line('a')// &
            '  print *, a(1) + a(2) + a(3)'//new_line('a')// &
            'end program main'

        test_real_scatter = expect_output( &
            source, '   18.000000000000000     '//new_line('a'), &
            '/tmp/ffc_session_vector_subscript_real_test')
    end function test_real_scatter

    logical function test_literal_vector()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(4)'//new_line('a')// &
            '  a = [1, 2, 3, 4]'//new_line('a')// &
            '  a([4, 2]) = [70, 80]'//new_line('a')// &
            '  print *, a'//new_line('a')// &
            'end program main'

        test_literal_vector = expect_output( &
            source, &
            '           1          80           3          70'//new_line('a'), &
            '/tmp/ffc_session_vector_subscript_literal_test')
    end function test_literal_vector

    logical function test_out_of_bounds_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(4)'//new_line('a')// &
            '  a = 0'//new_line('a')// &
            '  a([1, 9]) = [1, 2]'//new_line('a')// &
            '  print *, a'//new_line('a')// &
            'end program main'

        test_out_of_bounds_rejected = expect_error_contains( &
            source, 'out of bounds', &
            '/tmp/ffc_session_vector_subscript_bounds_test')
    end function test_out_of_bounds_rejected

    logical function test_nonconformable_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(5), v(3)'//new_line('a')// &
            '  a = 0'//new_line('a')// &
            '  v = [1, 2, 3]'//new_line('a')// &
            '  a(v) = [1, 2]'//new_line('a')// &
            '  print *, a'//new_line('a')// &
            'end program main'

        test_nonconformable_rejected = expect_error_contains( &
            source, 'nonconformable', &
            '/tmp/ffc_session_vector_subscript_shape_test')
    end function test_nonconformable_rejected

    logical function test_noninteger_vector_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(5)'//new_line('a')// &
            '  real(8) :: v(2)'//new_line('a')// &
            '  a = 0'//new_line('a')// &
            '  v = [1.0d0, 2.0d0]'//new_line('a')// &
            '  a(v) = [1, 2]'//new_line('a')// &
            '  print *, a'//new_line('a')// &
            'end program main'

        test_noninteger_vector_rejected = expect_error_contains( &
            source, 'integer', &
            '/tmp/ffc_session_vector_subscript_kind_test')
    end function test_noninteger_vector_rejected

end program test_session_vector_subscript_assignment_compiler
