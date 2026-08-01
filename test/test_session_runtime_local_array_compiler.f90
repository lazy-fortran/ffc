program test_session_runtime_local_array
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session runtime-sized local array compiler test ==='

    all_passed = .true.
    if (.not. test_integer_automatic_array()) all_passed = .false.
    if (.not. test_real_broadcast_and_copy()) all_passed = .false.
    if (.not. test_lower_bound_array()) all_passed = .false.
    if (.not. test_entry_extent_survives_bound_mutation()) all_passed = .false.
    if (.not. test_rank2_nonunit_lower_bounds()) all_passed = .false.
    if (.not. test_rank2_unit_bounds_column_major()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: runtime-sized local automatic arrays lower end to end'

contains

    logical function test_integer_automatic_array()
        ! integer :: b(m) with m a dummy value only known at run time. Storage is
        ! a dynamic alloca; element write, size(), sum(), and whole-array print
        ! all walk the runtime element count.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'integer :: n'//new_line('a')// &
            'n = 4'//new_line('a')// &
            'call go(n)'//new_line('a')// &
            'contains'//new_line('a')// &
            'subroutine go(m)'//new_line('a')// &
            'integer, intent(in) :: m'//new_line('a')// &
            'integer :: b(m)'//new_line('a')// &
            'integer :: j'//new_line('a')// &
            'do j = 1, m'//new_line('a')// &
            'b(j) = j * 10'//new_line('a')// &
            'end do'//new_line('a')// &
            'print *, size(b)'//new_line('a')// &
            'print *, sum(b)'//new_line('a')// &
            'print *, b'//new_line('a')// &
            'end subroutine go'//new_line('a')// &
            'end program main'
        character(len=*), parameter :: expected = &
            '           4'//new_line('a')// &
            '         100'//new_line('a')// &
            '          10          20          30          40'//new_line('a')

        test_integer_automatic_array = expect_output(source, expected, &
            '/tmp/ffc_session_runtime_local_array_i')
    end function test_integer_automatic_array

    logical function test_real_broadcast_and_copy()
        ! real :: a(m), b(m): whole-array scalar broadcast (a = 2.5) and a
        ! whole-array copy (b = a), both driven by a runtime loop.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'integer :: n'//new_line('a')// &
            'n = 3'//new_line('a')// &
            'call go(n)'//new_line('a')// &
            'contains'//new_line('a')// &
            'subroutine go(m)'//new_line('a')// &
            'integer, intent(in) :: m'//new_line('a')// &
            'real :: a(m)'//new_line('a')// &
            'real :: b(m)'//new_line('a')// &
            'a = 2.5'//new_line('a')// &
            'b = a'//new_line('a')// &
            'print *, a'//new_line('a')// &
            'print *, b'//new_line('a')// &
            'end subroutine go'//new_line('a')// &
            'end program main'
        character(len=*), parameter :: expected = &
            '   2.50000000       2.50000000       2.50000000    '//new_line('a')// &
            '   2.50000000       2.50000000       2.50000000    '//new_line('a')

        test_real_broadcast_and_copy = expect_output(source, expected, &
            '/tmp/ffc_session_runtime_local_array_r')
    end function test_real_broadcast_and_copy

    logical function test_lower_bound_array()
        ! real(8) :: d(0:m): a runtime upper bound with a constant lower bound.
        ! lbound/ubound/size honour the lower bound; element access indexes from
        ! the declared lower bound.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'integer :: n'//new_line('a')// &
            'n = 5'//new_line('a')// &
            'call go(n)'//new_line('a')// &
            'contains'//new_line('a')// &
            'subroutine go(m)'//new_line('a')// &
            'integer, intent(in) :: m'//new_line('a')// &
            'real(8) :: d(0:m)'//new_line('a')// &
            'integer :: j'//new_line('a')// &
            'do j = 0, m'//new_line('a')// &
            'd(j) = j * 1.0d0'//new_line('a')// &
            'end do'//new_line('a')// &
            'print *, size(d), lbound(d, 1), ubound(d, 1)'//new_line('a')// &
            'print *, d(0), d(5)'//new_line('a')// &
            'end subroutine go'//new_line('a')// &
            'end program main'
        character(len=*), parameter :: expected = &
            '           6           0           5'//new_line('a')// &
            '   0.0000000000000000        5.0000000000000000     '//new_line('a')

        test_lower_bound_array = expect_output(source, expected, &
            '/tmp/ffc_session_runtime_local_array_lb')
    end function test_lower_bound_array

    logical function test_entry_extent_survives_bound_mutation()
        ! arrays_06_size / arrays_07_size shape: integer :: keep(x) with x an
        ! intent(inout) dummy. The extent is fixed on entry, so assigning x = 1
        ! after the declaration must not resize keep. The array-constructor
        ! assignment keep = [1,2] fills the runtime-sized storage.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'integer :: y = 2'//new_line('a')// &
            'call temp(y)'//new_line('a')// &
            'contains'//new_line('a')// &
            'subroutine temp(x)'//new_line('a')// &
            'integer, intent(inout) :: x'//new_line('a')// &
            'integer :: keep(x)'//new_line('a')// &
            'keep = [1, 2]'//new_line('a')// &
            'x = 1'//new_line('a')// &
            'print *, size(keep)'//new_line('a')// &
            'print *, keep'//new_line('a')// &
            'if (size(keep) /= 2) error stop'//new_line('a')// &
            'if (keep(1) /= 1) error stop'//new_line('a')// &
            'if (keep(2) /= 2) error stop'//new_line('a')// &
            'end subroutine temp'//new_line('a')// &
            'end program main'
        character(len=*), parameter :: expected = &
            '           2'//new_line('a')// &
            '           1           2'//new_line('a')

        test_entry_extent_survives_bound_mutation = expect_output(source, &
            expected, '/tmp/ffc_session_runtime_local_array_entry')
    end function test_entry_extent_survives_bound_mutation

    logical function test_rank2_nonunit_lower_bounds()
        ! A rank-2 automatic array with runtime upper bounds and non-unit lower
        ! bounds. Its canonical descriptor carries lower bound, extent, and
        ! column-major byte stride per dimension, so SIZE, LBOUND, UBOUND, and
        ! boundary element access all agree with gfortran.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'implicit none'//new_line('a')// &
            'integer :: n, m'//new_line('a')// &
            'n = 3'//new_line('a')// &
            'm = 4'//new_line('a')// &
            'call work(n, m)'//new_line('a')// &
            'contains'//new_line('a')// &
            'subroutine work(nn, mm)'//new_line('a')// &
            'integer, intent(in) :: nn, mm'//new_line('a')// &
            'integer :: a(0:nn, 2:mm)'//new_line('a')// &
            'a(0,2) = 2'//new_line('a')// &
            'a(nn,mm) = 34'//new_line('a')// &
            'a(1,3) = 13'//new_line('a')// &
            'print *, size(a), size(a,1), size(a,2)'//new_line('a')// &
            'print *, lbound(a,1), ubound(a,1), lbound(a,2), ubound(a,2)'// &
            new_line('a')// &
            'print *, a(0,2), a(nn,mm), a(1,3)'//new_line('a')// &
            'end subroutine work'//new_line('a')// &
            'end program main'
        character(len=*), parameter :: expected = &
            '          12           4           3'//new_line('a')// &
            '           0           3           2           4'//new_line('a')// &
            '           2          34          13'//new_line('a')

        test_rank2_nonunit_lower_bounds = expect_output(source, expected, &
            '/tmp/ffc_session_runtime_local_array_rank2_bounds')
    end function test_rank2_nonunit_lower_bounds

    logical function test_rank2_unit_bounds_column_major()
        ! Column-major identity for a rank-2 runtime automatic array: the
        ! descriptor stride for dimension 2 is the element size times the
        ! dimension-1 extent, so a(i,j) and a(i+1,j) are adjacent.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'implicit none'//new_line('a')// &
            'integer :: n'//new_line('a')// &
            'n = 3'//new_line('a')// &
            'call work(n)'//new_line('a')// &
            'contains'//new_line('a')// &
            'subroutine work(k)'//new_line('a')// &
            'integer, intent(in) :: k'//new_line('a')// &
            'integer :: a(k, 2)'//new_line('a')// &
            'integer :: i'//new_line('a')// &
            'do i = 1, k'//new_line('a')// &
            'a(i,1) = i'//new_line('a')// &
            'end do'//new_line('a')// &
            'a(1,2) = 71'//new_line('a')// &
            'a(k,2) = 79'//new_line('a')// &
            'print *, size(a), size(a,1), size(a,2)'//new_line('a')// &
            'print *, a(1,1), a(2,1), a(3,1)'//new_line('a')// &
            'print *, a(1,2), a(k,2)'//new_line('a')// &
            'end subroutine work'//new_line('a')// &
            'end program main'
        character(len=*), parameter :: expected = &
            '           6           3           2'//new_line('a')// &
            '           1           2           3'//new_line('a')// &
            '          71          79'//new_line('a')

        test_rank2_unit_bounds_column_major = expect_output(source, expected, &
            '/tmp/ffc_session_runtime_local_array_rank2_cm')
    end function test_rank2_unit_bounds_column_major

end program test_session_runtime_local_array
