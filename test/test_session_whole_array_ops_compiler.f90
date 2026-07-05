program test_session_whole_array_ops_compiler
    use ffc_test_support, only: expect_output
    implicit none

    print *, '=== direct session whole-array unary minus/division compiler test ==='
    if (.not. test_integer_rank1_array_division()) stop 1
    if (.not. test_integer_rank1_scalar_division()) stop 1
    if (.not. test_integer_rank1_unary_minus()) stop 1
    if (.not. test_real_rank1_array_division()) stop 1
    if (.not. test_real_rank1_scalar_division()) stop 1
    if (.not. test_real_rank1_unary_minus()) stop 1
    if (.not. test_rank2_array_division_and_unary_minus()) stop 1
    print *, 'PASS: whole-array division and unary minus lower through '// &
        'direct LIRIC'

contains

    logical function test_integer_rank1_array_division()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(3), b(3), c(3)'//new_line('a')// &
            '  a = [10, 20, 30]'//new_line('a')// &
            '  b = [2, 4, 5]'//new_line('a')// &
            '  c = a / b'//new_line('a')// &
            '  print *, c(1), c(2), c(3)'//new_line('a')// &
            'end program main'

        test_integer_rank1_array_division = expect_output( &
            source, '           5           5           6'//new_line('a'), &
            '/tmp/ffc_wa_ops_int_div_test')
    end function test_integer_rank1_array_division

    logical function test_integer_rank1_scalar_division()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(3), c(3)'//new_line('a')// &
            '  a = [10, 20, 30]'//new_line('a')// &
            '  c = a / 2'//new_line('a')// &
            '  print *, c(1), c(2), c(3)'//new_line('a')// &
            'end program main'

        test_integer_rank1_scalar_division = expect_output( &
            source, '           5          10          15'//new_line('a'), &
            '/tmp/ffc_wa_ops_int_scalar_div_test')
    end function test_integer_rank1_scalar_division

    logical function test_integer_rank1_unary_minus()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(3), c(3)'//new_line('a')// &
            '  a = [10, -20, 30]'//new_line('a')// &
            '  c = -a'//new_line('a')// &
            '  print *, c(1), c(2), c(3)'//new_line('a')// &
            'end program main'

        test_integer_rank1_unary_minus = expect_output( &
            source, '         -10          20         -30'//new_line('a'), &
            '/tmp/ffc_wa_ops_int_neg_test')
    end function test_integer_rank1_unary_minus

    logical function test_real_rank1_array_division()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real :: a(3), b(3), c(3)'//new_line('a')// &
            '  a = [10.0, 20.0, 30.0]'//new_line('a')// &
            '  b = [2.0, 4.0, 5.0]'//new_line('a')// &
            '  c = a / b'//new_line('a')// &
            '  print *, c(1), c(2), c(3)'//new_line('a')// &
            'end program main'

        test_real_rank1_array_division = expect_output( &
            source, '   5.00000000       5.00000000       6.00000000    ' &
            //new_line('a'), '/tmp/ffc_wa_ops_real_div_test')
    end function test_real_rank1_array_division

    logical function test_real_rank1_scalar_division()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real :: a(3), c(3)'//new_line('a')// &
            '  a = [10.0, 20.0, 30.0]'//new_line('a')// &
            '  c = a / 2.0'//new_line('a')// &
            '  print *, c(1), c(2), c(3)'//new_line('a')// &
            'end program main'

        test_real_rank1_scalar_division = expect_output( &
            source, '   5.00000000       10.0000000       15.0000000    ' &
            //new_line('a'), '/tmp/ffc_wa_ops_real_scalar_div_test')
    end function test_real_rank1_scalar_division

    logical function test_real_rank1_unary_minus()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real :: a(3), c(3)'//new_line('a')// &
            '  a = [10.0, -20.0, 30.0]'//new_line('a')// &
            '  c = -a'//new_line('a')// &
            '  print *, c(1), c(2), c(3)'//new_line('a')// &
            'end program main'

        test_real_rank1_unary_minus = expect_output( &
            source, '  -10.0000000       20.0000000      -30.0000000    ' &
            //new_line('a'), '/tmp/ffc_wa_ops_real_neg_test')
    end function test_real_rank1_unary_minus

    logical function test_rank2_array_division_and_unary_minus()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(2, 2), b(2, 2), c(2, 2)'//new_line('a')// &
            '  a = reshape([10, 20, 30, 40], [2, 2])'//new_line('a')// &
            '  b = reshape([2, 4, 5, 8], [2, 2])'//new_line('a')// &
            '  c = a / b'//new_line('a')// &
            '  print *, c(1, 1), c(2, 1), c(1, 2), c(2, 2)'//new_line('a')// &
            '  c = -a'//new_line('a')// &
            '  print *, c(1, 1), c(2, 1), c(1, 2), c(2, 2)'//new_line('a')// &
            'end program main'

        test_rank2_array_division_and_unary_minus = expect_output( &
            source, '           5           5           6           5'// &
            new_line('a')//'         -10         -20         -30         -40'// &
            new_line('a'), '/tmp/ffc_wa_ops_rank2_test')
    end function test_rank2_array_division_and_unary_minus

end program test_session_whole_array_ops_compiler
