program test_session_whole_array_not_compiler
    use ffc_test_support, only: expect_output
    implicit none

    print *, '=== direct session whole-array .not./division/scalar-broadcast '// &
        'compiler test ==='
    if (.not. test_rank1_logical_not()) stop 1
    if (.not. test_rank2_logical_not()) stop 1
    if (.not. test_not_of_comparison_mask()) stop 1
    if (.not. test_real_division_scalar_broadcast()) stop 1
    print *, 'PASS: whole-array .not., elementwise division, and scalar '// &
        'broadcast lower through direct LIRIC'

contains

    logical function test_rank1_logical_not()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  logical :: a(3), b(3)'//new_line('a')// &
            '  b = [.true., .false., .true.]'//new_line('a')// &
            '  a = .not. b'//new_line('a')// &
            '  print *, a(1), a(2), a(3)'//new_line('a')// &
            'end program main'

        test_rank1_logical_not = expect_output( &
            source, ' F T F'//new_line('a'), &
            '/tmp/ffc_wa_not_rank1_test')
    end function test_rank1_logical_not

    logical function test_rank2_logical_not()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  logical :: a(2, 2), b(2, 2)'//new_line('a')// &
            '  b = reshape([.true., .false., .true., .false.], [2, 2])'// &
            new_line('a')// &
            '  a = .not. b'//new_line('a')// &
            '  print *, a(1, 1), a(2, 1), a(1, 2), a(2, 2)'//new_line('a')// &
            'end program main'

        test_rank2_logical_not = expect_output( &
            source, ' F T F T'//new_line('a'), &
            '/tmp/ffc_wa_not_rank2_test')
    end function test_rank2_logical_not

    logical function test_not_of_comparison_mask()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  logical :: a(4)'//new_line('a')// &
            '  integer :: b(4), c(4)'//new_line('a')// &
            '  b = [1, 2, 3, 4]'//new_line('a')// &
            '  c = [4, 2, 1, 4]'//new_line('a')// &
            '  a = .not. (b > c)'//new_line('a')// &
            '  print *, a(1), a(2), a(3), a(4)'//new_line('a')// &
            'end program main'

        test_not_of_comparison_mask = expect_output( &
            source, ' T T F T'//new_line('a'), &
            '/tmp/ffc_wa_not_compare_test')
    end function test_not_of_comparison_mask

    logical function test_real_division_scalar_broadcast()
        ! Scalar-broadcast on the left of an elementwise divide (10.0 / b),
        ! mixing an array operand with a scalar literal.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real :: a(3), b(3)'//new_line('a')// &
            '  b = [2.0, 4.0, 5.0]'//new_line('a')// &
            '  a = 10.0 / b'//new_line('a')// &
            '  print *, a(1), a(2), a(3)'//new_line('a')// &
            'end program main'

        test_real_division_scalar_broadcast = expect_output( &
            source, '   5.00000000       2.50000000       2.00000000    ' &
                //new_line('a'), '/tmp/ffc_wa_div_scalar_broadcast_test')
    end function test_real_division_scalar_broadcast

end program test_session_whole_array_not_compiler
