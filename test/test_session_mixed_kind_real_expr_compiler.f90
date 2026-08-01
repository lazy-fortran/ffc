program test_session_mixed_kind_real_expr_compiler
    use ffc_test_support, only: expect_output
    implicit none

    ! Mixed-kind arithmetic takes the widest operand kind (F2018 10.1.5.2.1),
    ! so a binary operation that combines a default-real (f32) literal with a
    ! real(8) operand is an f64 expression. Nested inside a wider f64 operand
    ! context ("2.0*x + 1"), the f32 sub-expression classification used to claim
    ! the whole "2.0*x" product for the f32 path and the lowering rejected the
    ! real(8) identifier inside it. Covers lfortran expr_11.f90.
    !
    ! Expected values were produced by gfortran on the same source.
    character(len=*), parameter :: source = &
        'program main'//new_line('a')// &
        '    real*8 :: x'//new_line('a')// &
        '    real(8) :: y'//new_line('a')// &
        '    double precision :: z'//new_line('a')// &
        '    x = 1.5d0'//new_line('a')// &
        '    x = (2.0*x+1.0)/(x*(x+1))'//new_line('a')// &
        "    print '(f0.10)', x"//new_line('a')// &
        '    y = 3.0d0'//new_line('a')// &
        '    y = 2.0*y+1'//new_line('a')// &
        "    print '(f0.10)', y"//new_line('a')// &
        '    z = 0.5d0'//new_line('a')// &
        '    z = 1.0 + 2.0*z*3.0 - 2'//new_line('a')// &
        "    print '(f0.10)', z"//new_line('a')// &
        'end program main'

    character(len=*), parameter :: expected = &
        '1.0666666667'//new_line('a')// &
        '7.0000000000'//new_line('a')// &
        '2.0000000000'//new_line('a')

    print *, '=== mixed-kind real expression compiler test ==='

    if (.not. expect_output(source, expected, &
        '/tmp/ffc_session_mixed_kind_real_expr_test')) stop 1

    print *, 'PASS: mixed f32/f64 arithmetic lowers at real(8) precision'
end program test_session_mixed_kind_real_expr_compiler
