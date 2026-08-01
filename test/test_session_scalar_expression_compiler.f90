program test_session_scalar_expression_compiler
    !! Behavioral oracle for the typed scalar expression engine (#447).
    !!
    !! Every expected value below is what gfortran prints for the same program;
    !! the oracle is the standard's mixed-kind rule (F2018 10.1.5.2.1), not the
    !! shape of ffc's own lowering.
    use ffc_test_support, only: expect_output, expect_exit_status
    implicit none
    logical :: ok

    print *, '=== scalar expression engine test ==='

    ok = .true.
    if (.not. test_wide_comparison_is_not_narrowed()) ok = .false.
    if (.not. test_mixed_kind_binary_takes_widest()) ok = .false.
    if (.not. test_nested_intrinsic_keeps_wide_kind()) ok = .false.
    if (.not. test_kind_selector_beats_argument_kind()) ok = .false.
    if (.not. test_single_precision_stays_single()) ok = .false.
    if (.not. test_integer_operand_promotes_not_narrows()) ok = .false.
    if (.not. ok) stop 1

    print *, 'PASS: scalar expressions lower through one typed kind engine'

contains

    logical function test_wide_comparison_is_not_narrowed() result(res)
        !! A real(8) compared against a default-real operand must compare at
        !! the widest operand kind (F2018 10.1.5.2.1). Narrowing the real(8)
        !! side to f32 rounds away the difference and flips the result.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real :: a'//new_line('a')// &
            '  real(8) :: x'//new_line('a')// &
            '  a = 0.5'//new_line('a')// &
            '  x = 0.5d0 + 1.0d-9'//new_line('a')// &
            '  if (x > a) then'//new_line('a')// &
            '    stop 3'//new_line('a')// &
            '  else'//new_line('a')// &
            '    stop 7'//new_line('a')// &
            '  end if'//new_line('a')// &
            'end program main'

        ! x is strictly greater than a in f64, so gfortran stops with 3.
        ! Narrowing x to f32 makes both sides exactly 0.5 and yields 7.
        res = expect_exit_status(source, 3, '/tmp/ffc_scalar_expr_cmp_a')
    end function test_wide_comparison_is_not_narrowed

    logical function test_mixed_kind_binary_takes_widest() result(res)
        !! 2.0*x with x real(8) is an f64 expression: the default-real literal
        !! converts up, the real(8) operand is never rounded down.
        res = expect_output( &
            'program main'//new_line('a')// &
            '  real(8) :: x'//new_line('a')// &
            '  x = 0.1d0'//new_line('a')// &
            '  print *, 2.0*x'//new_line('a')// &
            'end program main', &
            '  0.20000000000000001     '//new_line('a'), '/tmp/ffc_scalar_expr_mixed')
    end function test_mixed_kind_binary_takes_widest

    logical function test_nested_intrinsic_keeps_wide_kind() result(res)
        !! A kind-preserving elemental intrinsic returns the widest kind among
        !! its arguments, and nesting must not lose that width.
        res = expect_output( &
            'program main'//new_line('a')// &
            '  real(8) :: x'//new_line('a')// &
            '  x = 2.0d0'//new_line('a')// &
            '  print *, sqrt(sqrt(x*1.0))'//new_line('a')// &
            'end program main', &
            '   1.1892071150027210     '//new_line('a'), '/tmp/ffc_scalar_expr_nested')
    end function test_nested_intrinsic_keeps_wide_kind

    logical function test_kind_selector_beats_argument_kind() result(res)
        !! REAL(A, KIND) takes its result kind from the selector, not from A.
        res = expect_output( &
            'program main'//new_line('a')// &
            '  real(8) :: x'//new_line('a')// &
            '  real :: r'//new_line('a')// &
            '  x = 0.1d0'//new_line('a')// &
            '  r = real(x, kind=4)'//new_line('a')// &
            '  print *, r'//new_line('a')// &
            'end program main', &
            '  0.100000001    '//new_line('a'), '/tmp/ffc_scalar_expr_kindsel')
    end function test_kind_selector_beats_argument_kind

    logical function test_single_precision_stays_single() result(res)
        !! No implicit widening: an all-f32 expression stays f32.
        res = expect_output( &
            'program main'//new_line('a')// &
            '  real :: a'//new_line('a')// &
            '  real :: b'//new_line('a')// &
            '  a = 1.0'//new_line('a')// &
            '  b = 3.0'//new_line('a')// &
            '  print *, a/b'//new_line('a')// &
            'end program main', &
            '  0.333333343    '//new_line('a'), '/tmp/ffc_scalar_expr_f32')
    end function test_single_precision_stays_single

    logical function test_integer_operand_promotes_not_narrows() result(res)
        !! An integer operand in a real(8) comparison promotes to f64; the real
        !! operand is never narrowed to reach the integer.
        res = expect_exit_status( &
            'program main'//new_line('a')// &
            '  integer :: n'//new_line('a')// &
            '  real(8) :: x'//new_line('a')// &
            '  n = 1'//new_line('a')// &
            '  x = 1.0d0 + 1.0d-12'//new_line('a')// &
            '  if (x > n) then'//new_line('a')// &
            '    stop 3'//new_line('a')// &
            '  else'//new_line('a')// &
            '    stop 7'//new_line('a')// &
            '  end if'//new_line('a')// &
            'end program main', 3, '/tmp/ffc_scalar_expr_intcmp')
    end function test_integer_operand_promotes_not_narrows

end program test_session_scalar_expression_compiler
