program test_session_whole_array_div_pow_compiler
    use ffc_test_support, only: expect_output
    implicit none

    print *, '=== direct session whole-array division/power compiler test ==='
    if (.not. test_integer_whole_array_division()) stop 1
    if (.not. test_integer_whole_array_power()) stop 1
    if (.not. test_real_whole_array_power()) stop 1
    if (.not. test_array_constructor_operand()) stop 1
    if (.not. test_scalar_call_broadcast()) stop 1
    print *, 'PASS: whole-array division, power, constructor operands, and '// &
        'scalar-call broadcast lower through direct LIRIC'

contains

    logical function test_integer_whole_array_division()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(3), b(3), c(3)'//new_line('a')// &
            '  b = [6, 8, 9]'//new_line('a')// &
            '  c = [2, 4, 3]'//new_line('a')// &
            '  a = b / c'//new_line('a')// &
            '  print *, a(1), a(2), a(3)'//new_line('a')// &
            'end program main'

        test_integer_whole_array_division = expect_output( &
            source, '           3           2           3'//new_line('a'), &
            '/tmp/ffc_wa_intdiv_test')
    end function test_integer_whole_array_division

    logical function test_integer_whole_array_power()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(3), b(3)'//new_line('a')// &
            '  b = [2, 3, 4]'//new_line('a')// &
            '  a = b ** 2'//new_line('a')// &
            '  print *, a(1), a(2), a(3)'//new_line('a')// &
            'end program main'

        test_integer_whole_array_power = expect_output( &
            source, '           4           9          16'//new_line('a'), &
            '/tmp/ffc_wa_intpow_test')
    end function test_integer_whole_array_power

    logical function test_real_whole_array_power()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real :: a(2), b(2)'//new_line('a')// &
            '  b = [2.0, 3.0]'//new_line('a')// &
            '  a = b ** 2'//new_line('a')// &
            '  print *, a(1), a(2)'//new_line('a')// &
            'end program main'

        test_real_whole_array_power = expect_output( &
            source, '   4.00000000       9.00000000    '//new_line('a'), &
            '/tmp/ffc_wa_realpow_test')
    end function test_real_whole_array_power

    logical function test_array_constructor_operand()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(3), b(3)'//new_line('a')// &
            '  b = [1, 2, 3]'//new_line('a')// &
            '  a = b + [10, 20, 30]'//new_line('a')// &
            '  print *, a(1), a(2), a(3)'//new_line('a')// &
            'end program main'

        test_array_constructor_operand = expect_output( &
            source, '          11          22          33'//new_line('a'), &
            '/tmp/ffc_wa_ctor_operand_test')
    end function test_array_constructor_operand

    logical function test_scalar_call_broadcast()
        ! real(i) is a scalar conversion call, not a literal or identifier;
        ! the whole array should broadcast its single result.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real :: a(3)'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  i = 4'//new_line('a')// &
            '  a = real(i)'//new_line('a')// &
            '  print *, a(1), a(2), a(3)'//new_line('a')// &
            'end program main'

        test_scalar_call_broadcast = expect_output( &
            source, '   4.00000000       4.00000000       4.00000000    ' &
            //new_line('a'), '/tmp/ffc_wa_scalar_call_test')
    end function test_scalar_call_broadcast

end program test_session_whole_array_div_pow_compiler
