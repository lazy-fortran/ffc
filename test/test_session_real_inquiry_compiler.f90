program test_session_real_inquiry_compiler
    use ffc_test_support, only: expect_exit_status, expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session real inquiry intrinsics compiler test ==='

    all_passed = .true.
    if (.not. test_tiny_f32_expression()) all_passed = .false.
    if (.not. test_huge_f32_comparison()) all_passed = .false.
    if (.not. test_epsilon_f64_expression()) all_passed = .false.
    if (.not. test_tiny_f64_variable_arg()) all_passed = .false.
    if (.not. test_epsilon_f32_print()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: tiny/huge/epsilon real inquiry intrinsics lower through '// &
        'direct LIRIC session'

contains

    logical function test_tiny_f32_expression()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real :: x'//new_line('a')// &
            '  x = 2.0 * tiny(1.0)'//new_line('a')// &
            '  if (x < tiny(1.0)) error stop'//new_line('a')// &
            '  if (x > 3.0e-38) error stop'//new_line('a')// &
            'end program main'

        test_tiny_f32_expression = expect_exit_status( &
            source, 0, '/tmp/ffc_session_tiny_f32_expr_test')
    end function test_tiny_f32_expression

    logical function test_huge_f32_comparison()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real :: x'//new_line('a')// &
            '  x = 1.0'//new_line('a')// &
            '  if (x > huge(x)) error stop'//new_line('a')// &
            '  if (.not. (huge(1.0) > 1.0e38)) error stop'//new_line('a')// &
            'end program main'

        test_huge_f32_comparison = expect_exit_status( &
            source, 0, '/tmp/ffc_session_huge_f32_cmp_test')
    end function test_huge_f32_comparison

    logical function test_epsilon_f64_expression()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real(8) :: e'//new_line('a')// &
            '  e = epsilon(1.0d0)'//new_line('a')// &
            '  if (e > 1.0d-15) error stop'//new_line('a')// &
            '  if (e < 1.0d-17) error stop'//new_line('a')// &
            'end program main'

        test_epsilon_f64_expression = expect_exit_status( &
            source, 0, '/tmp/ffc_session_epsilon_f64_expr_test')
    end function test_epsilon_f64_expression

    logical function test_tiny_f64_variable_arg()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real(8) :: b, t'//new_line('a')// &
            '  b = 1.0d0'//new_line('a')// &
            '  t = tiny(b)'//new_line('a')// &
            '  if (t > 1.0d-300) error stop'//new_line('a')// &
            '  if (t <= 0.0d0) error stop'//new_line('a')// &
            'end program main'

        test_tiny_f64_variable_arg = expect_exit_status( &
            source, 0, '/tmp/ffc_session_tiny_f64_var_test')
    end function test_tiny_f64_variable_arg

    logical function test_epsilon_f32_print()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  print *, epsilon(1.0)'//new_line('a')// &
            'end program main'

        test_epsilon_f32_print = expect_output( &
            source, '   1.19209290E-07'//new_line('a'), &
            '/tmp/ffc_session_epsilon_f32_print_test')
    end function test_epsilon_f32_print

end program test_session_real_inquiry_compiler
