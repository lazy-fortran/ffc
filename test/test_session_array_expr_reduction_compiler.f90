program test_session_array_expr_reduction_compiler
    !! Reductions over an array-valued *expression* argument, folded through
    !! the shared identity/combine pair. Expected values are the gfortran
    !! output for the same programs.
    use ffc_test_support, only: expect_output
    implicit none

    print *, '=== direct session array-expression reduction compiler test ==='
    if (.not. test_integer_expression_reductions()) stop 1
    if (.not. test_real_expression_reductions()) stop 1
    if (.not. test_mask_expression_reductions()) stop 1
    print *, 'PASS: reductions fold array expressions through one iterator'

contains

    logical function test_integer_expression_reductions()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(4), b(4)'//new_line('a')// &
            '  a = [1, 2, 3, 4]'//new_line('a')// &
            '  b = [4, 3, 2, 1]'//new_line('a')// &
            '  print *, sum(a*b), product(a+b), maxval(a*b), minval(a+b)'// &
            new_line('a')// &
            'end program main'

        test_integer_expression_reductions = expect_output( &
            source, &
            '          20         625           6           5'//new_line('a'), &
            '/tmp/ffc_session_array_expr_reduction_int_test')
    end function test_integer_expression_reductions

    logical function test_real_expression_reductions()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real :: x(4), y(4)'//new_line('a')// &
            '  x = [1.0, 2.0, 3.0, 4.0]'//new_line('a')// &
            '  y = [4.0, 3.0, 2.0, 1.0]'//new_line('a')// &
            '  print *, sum(x*y)'//new_line('a')// &
            '  print *, product(x+y)'//new_line('a')// &
            '  print *, maxval(x*y)'//new_line('a')// &
            '  print *, minval(x+y)'//new_line('a')// &
            'end program main'

        test_real_expression_reductions = expect_output( &
            source, &
            '   20.0000000    '//new_line('a')// &
            '   625.000000    '//new_line('a')// &
            '   6.00000000    '//new_line('a')// &
            '   5.00000000    '//new_line('a'), &
            '/tmp/ffc_session_array_expr_reduction_real_test')
    end function test_real_expression_reductions

    logical function test_mask_expression_reductions()
        !! COUNT, ANY, and ALL over a relational array expression.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(4), b(4)'//new_line('a')// &
            '  a = [1, 2, 3, 4]'//new_line('a')// &
            '  b = [4, 3, 2, 1]'//new_line('a')// &
            '  print *, count(a > b)'//new_line('a')// &
            '  if (any(a > b)) print *, ''any yes'''//new_line('a')// &
            '  if (.not. all(a > b)) print *, ''all no'''//new_line('a')// &
            'end program main'

        test_mask_expression_reductions = expect_output( &
            source, &
            '           2'//new_line('a')// &
            ' any yes'//new_line('a')// &
            ' all no'//new_line('a'), &
            '/tmp/ffc_session_array_expr_reduction_mask_test')
    end function test_mask_expression_reductions

end program test_session_array_expr_reduction_compiler
