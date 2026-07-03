program test_session_allocatable_elementwise_compiler
    use ffc_test_support, only: expect_output
    implicit none

    print *, '=== direct session allocatable elementwise assignment compiler test ==='
    if (.not. test_allocatable_sum_expression()) stop 1
    if (.not. test_allocatable_self_referencing_expression()) stop 1
    print *, 'PASS: allocatable whole-array assignment from an expression '// &
        'lowers through direct LIRIC'

contains

    logical function test_allocatable_sum_expression()
        ! a = b + c: a, b, c are rank-1 allocatables, all already allocated to
        ! the same constant extent.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, allocatable :: a(:), b(:), c(:)'//new_line('a')// &
            '  allocate(a(3))'//new_line('a')// &
            '  allocate(b(3))'//new_line('a')// &
            '  allocate(c(3))'//new_line('a')// &
            '  b = [1, 2, 3]'//new_line('a')// &
            '  c = [4, 5, 6]'//new_line('a')// &
            '  a = b + c'//new_line('a')// &
            '  print *, a(1), a(2), a(3)'//new_line('a')// &
            'end program main'

        test_allocatable_sum_expression = expect_output( &
            source, '           5           7           9'//new_line('a'), &
            '/tmp/ffc_alloc_ew_sum_test')
    end function test_allocatable_sum_expression

    logical function test_allocatable_self_referencing_expression()
        ! a = a * 3: the target allocatable also appears as a source operand.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, allocatable :: a(:)'//new_line('a')// &
            '  allocate(a(3))'//new_line('a')// &
            '  a = [1, 2, 3]'//new_line('a')// &
            '  a = a * 3'//new_line('a')// &
            '  print *, a(1), a(2), a(3)'//new_line('a')// &
            'end program main'

        test_allocatable_self_referencing_expression = expect_output( &
            source, '           3           6           9'//new_line('a'), &
            '/tmp/ffc_alloc_ew_self_test')
    end function test_allocatable_self_referencing_expression

end program test_session_allocatable_elementwise_compiler
