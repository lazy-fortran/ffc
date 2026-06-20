program test_session_mask_reduction_compiler
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session mask reduction compiler test ==='

    all_passed = .true.
    if (.not. test_count_integer_comparison()) all_passed = .false.
    if (.not. test_any_integer_comparison()) all_passed = .false.
    if (.not. test_all_integer_comparison()) all_passed = .false.
    if (.not. test_count_real_comparison()) all_passed = .false.
    if (.not. test_any_scalar_on_left()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: count/any/all over comparison masks lower correctly'

contains

    logical function test_count_integer_comparison()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(5) = [1, 2, 3, 4, 5]'//new_line('a')// &
            '  print *, count(a > 2)'//new_line('a')// &
            'end program main'

        test_count_integer_comparison = expect_output( &
            source, '           3'//new_line('a'), '/tmp/ffc_count_cmp')
    end function test_count_integer_comparison

    logical function test_any_integer_comparison()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(5) = [1, 2, 3, 4, 5]'//new_line('a')// &
            '  print *, any(a == 0)'//new_line('a')// &
            'end program main'

        test_any_integer_comparison = expect_output( &
            source, ' F'//new_line('a'), '/tmp/ffc_any_cmp')
    end function test_any_integer_comparison

    logical function test_all_integer_comparison()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(5) = [1, 2, 3, 4, 5]'//new_line('a')// &
            '  print *, all(a >= 0)'//new_line('a')// &
            'end program main'

        test_all_integer_comparison = expect_output( &
            source, ' T'//new_line('a'), '/tmp/ffc_all_cmp')
    end function test_all_integer_comparison

    logical function test_count_real_comparison()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real :: a(4) = [1.0, -2.0, 3.0, 0.0]'//new_line('a')// &
            '  print *, count(a > 0.0)'//new_line('a')// &
            'end program main'

        test_count_real_comparison = expect_output( &
            source, '           2'//new_line('a'), '/tmp/ffc_count_real_cmp')
    end function test_count_real_comparison

    logical function test_any_scalar_on_left()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(5) = [1, 2, 3, 4, 5]'//new_line('a')// &
            '  print *, any(3 < a)'//new_line('a')// &
            'end program main'

        test_any_scalar_on_left = expect_output( &
            source, ' T'//new_line('a'), '/tmp/ffc_any_scalar_left')
    end function test_any_scalar_on_left

end program test_session_mask_reduction_compiler
