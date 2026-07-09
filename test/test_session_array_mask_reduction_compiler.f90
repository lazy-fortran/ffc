program test_session_array_mask_reduction_compiler
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session array mask reduction test ==='

    all_passed = .true.
    if (.not. test_array_vs_scalar()) all_passed = .false.
    if (.not. test_array_vs_array()) all_passed = .false.
    if (.not. test_array_vs_constructor()) all_passed = .false.
    if (.not. test_real_and_allocatable()) all_passed = .false.
    if (.not. test_elemental_abs_mask()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: any/all over whole-array comparisons lower correctly'

contains

    logical function test_array_vs_scalar()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '    integer :: a(3)'//new_line('a')// &
            '    a = [1, 1, 1]'//new_line('a')// &
            '    if (any(a /= 1)) error stop'//new_line('a')// &
            '    if (.not. all(a == 1)) error stop'//new_line('a')// &
            '    print *, "ok"'//new_line('a')// &
            'end program main'

        test_array_vs_scalar = expect_output( &
            source, ' ok'//new_line('a'), '/tmp/ffc_mask_scalar')
    end function test_array_vs_scalar

    logical function test_array_vs_array()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '    integer :: a(3), b(3)'//new_line('a')// &
            '    a = [1, 2, 3]'//new_line('a')// &
            '    b = [1, 2, 3]'//new_line('a')// &
            '    if (any(a /= b)) error stop'//new_line('a')// &
            '    b = [1, 9, 3]'//new_line('a')// &
            '    if (all(a == b)) error stop'//new_line('a')// &
            '    print *, "ok"'//new_line('a')// &
            'end program main'

        test_array_vs_array = expect_output( &
            source, ' ok'//new_line('a'), '/tmp/ffc_mask_array')
    end function test_array_vs_array

    logical function test_array_vs_constructor()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '    integer :: a(4)'//new_line('a')// &
            '    a = [1, 2, 3, 4]'//new_line('a')// &
            '    if (any(a /= [1, 2, 3, 4])) error stop'//new_line('a')// &
            '    print *, "ok"'//new_line('a')// &
            'end program main'

        test_array_vs_constructor = expect_output( &
            source, ' ok'//new_line('a'), '/tmp/ffc_mask_ctor')
    end function test_array_vs_constructor

    logical function test_real_and_allocatable()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '    real :: x(3)'//new_line('a')// &
            '    integer, allocatable :: k(:)'//new_line('a')// &
            '    x = [1.0, 2.0, 3.0]'//new_line('a')// &
            '    if (any(x > 3.5)) error stop'//new_line('a')// &
            '    allocate(k(3))'//new_line('a')// &
            '    k = 5'//new_line('a')// &
            '    if (any(k /= 5)) error stop'//new_line('a')// &
            '    print *, "ok"'//new_line('a')// &
            'end program main'

        test_real_and_allocatable = expect_output( &
            source, ' ok'//new_line('a'), '/tmp/ffc_mask_real_alloc')
    end function test_real_and_allocatable

    logical function test_elemental_abs_mask()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '    real :: samples(5)'//new_line('a')// &
            '    real(8) :: wide(2)'//new_line('a')// &
            '    real :: tolerance'//new_line('a')// &
            '    samples = [1.0, -2.0, 3.0, -4.0, 5.0]'//new_line('a')// &
            '    wide = [-1.0d0, 2.0d0]'//new_line('a')// &
            '    tolerance = 0.1'//new_line('a')// &
            '    if (.not. any(abs(wide) > 1.5d0)) error stop'//new_line('a')// &
            '    if (any(abs(samples) &'//new_line('a')// &
            '        > tolerance)) then'//new_line('a')// &
            '        print *, "values exceed tolerance"'//new_line('a')// &
            '    end if'//new_line('a')// &
            'end program main'

        test_elemental_abs_mask = expect_output( &
            source, ' values exceed tolerance'//new_line('a'), &
            '/tmp/ffc_mask_elemental_abs')
    end function test_elemental_abs_mask

end program test_session_array_mask_reduction_compiler
