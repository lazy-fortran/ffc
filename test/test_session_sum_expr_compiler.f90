program test_session_sum_expr_compiler
    ! sum() over a general array-valued expression argument: a binary-op
    ! combination of arrays/sections (sum(a + b)), and a bare call to a
    ! contained function returning an allocatable array (sum(f())). Only the
    ! plain-identifier and bare-section arguments were previously supported.
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session sum() expression argument compiler test ==='

    all_passed = .true.
    if (.not. test_real_binary_expr()) all_passed = .false.
    if (.not. test_integer_section_expr()) all_passed = .false.
    if (.not. test_alloc_function_result()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: sum() lowers general array expression arguments'

contains

    logical function test_real_binary_expr()
        ! sum(a + b) inside do concurrent: element-wise array addition.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  real :: a(3), b(3), c(3)'//new_line('a')// &
            '  a = [1.0, 2.0, 3.0]'//new_line('a')// &
            '  b = [4.0, 5.0, 6.0]'//new_line('a')// &
            '  do concurrent (i = 1:3)'//new_line('a')// &
            '    c(i) = sum(a + b)'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  print *, c(1), c(2), c(3)'//new_line('a')// &
            'end program main'

        test_real_binary_expr = expect_output( &
            source, &
            '   21.0000000       21.0000000       21.0000000    '// &
            new_line('a'), &
            '/tmp/ffc_sum_expr_real')
    end function test_real_binary_expr

    logical function test_integer_section_expr()
        ! sum(c(1:4) + d(1:4)) inside do concurrent: section addition.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  integer :: c(4), d(4), ires(2)'//new_line('a')// &
            '  c = [1, 2, 3, 4]'//new_line('a')// &
            '  d = [5, 6, 7, 8]'//new_line('a')// &
            '  do concurrent (i = 1:2)'//new_line('a')// &
            '    ires(i) = sum(c(1:4) + d(1:4))'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  print *, ires(1), ires(2)'//new_line('a')// &
            'end program main'

        test_integer_section_expr = expect_output( &
            source, '          36          36'//new_line('a'), &
            '/tmp/ffc_sum_expr_int')
    end function test_integer_section_expr

    logical function test_alloc_function_result()
        ! sum(f()) where f is a contained function returning an allocatable
        ! array: the result is materialised into a temporary descriptor and
        ! reduced over its compile-time extent.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  real :: x(1)'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  do concurrent (i = 1:1)'//new_line('a')// &
            '    x(i) = sum(f())'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  print *, x(1)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  pure function f() result(v)'//new_line('a')// &
            '    real, allocatable :: v(:)'//new_line('a')// &
            '    allocate(v(2))'//new_line('a')// &
            '    v = 1.0'//new_line('a')// &
            '  end function'//new_line('a')// &
            'end program main'

        test_alloc_function_result = expect_output( &
            source, '   2.00000000    '//new_line('a'), &
            '/tmp/ffc_sum_expr_alloc')
    end function test_alloc_function_result

end program test_session_sum_expr_compiler
