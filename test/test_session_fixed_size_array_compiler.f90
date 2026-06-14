program test_session_fixed_size_array_compiler
    use ffc_test_support, only: expect_exit_status, expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session fixed-size array compiler test ==='

    all_passed = .true.
    if (.not. test_simple_array()) all_passed = .false.
    if (.not. test_array_parameter_bound()) all_passed = .false.
    if (.not. test_array_explicit_bounds()) all_passed = .false.
    if (.not. test_array_negative_lower_bound()) all_passed = .false.
    if (.not. test_array_with_loop()) all_passed = .false.
    if (.not. test_array_variable_subscript()) all_passed = .false.
    if (.not. test_array_element_argument()) all_passed = .false.
    if (.not. test_array_stop_code()) all_passed = .false.
    if (.not. test_array_in_contained_subroutine()) all_passed = .false.
    if (.not. test_array_in_contained_function()) all_passed = .false.
    if (.not. test_array_constructor_literal()) all_passed = .false.
    if (.not. test_array_constructor_runtime()) all_passed = .false.
    if (.not. test_array_of_derived_assign_and_read()) all_passed = .false.
    if (.not. test_array_inline_literal_initializer()) all_passed = .false.
    if (.not. test_array_inline_scalar_initializer()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: fixed-size arrays lower through direct LIRIC'

contains

    logical function test_array_constructor_literal()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: a(3)'//new_line('a')// &
                                       '  a = [1, 2, 3]'//new_line('a')// &
                                       '  stop a(2)'//new_line('a')// &
                                       'end program main'

        test_array_constructor_literal = expect_exit_status( &
            source, 2, '/tmp/ffc_session_array_ctor_lit_test')
    end function test_array_constructor_literal

    logical function test_array_of_derived_assign_and_read()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  type :: point_t'//new_line('a')// &
                                       '    integer :: x'//new_line('a')// &
                                       '    integer :: y'//new_line('a')// &
                                       '  end type point_t'//new_line('a')// &
                                       '  type(point_t) :: ps(3)'//new_line('a')// &
                                       '  ps(1)%x = 7'//new_line('a')// &
                                       '  ps(2)%y = 8'//new_line('a')// &
                                       '  stop ps(1)%x + ps(2)%y'//new_line('a')// &
                                       'end program main'

        test_array_of_derived_assign_and_read = expect_exit_status( &
            source, 15, '/tmp/ffc_session_array_of_derived_test')
    end function test_array_of_derived_assign_and_read

    logical function test_array_constructor_runtime()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: a(3)'//new_line('a')// &
                                       '  integer :: n'//new_line('a')// &
                                       '  n = 5'//new_line('a')// &
                                       '  a = [n, n * 2, n + 1]'//new_line('a')// &
                                       '  stop a(1) + a(2) + a(3)'//new_line('a')// &
                                       'end program main'

        ! [5, 10, 6] -> 21
        test_array_constructor_runtime = expect_exit_status( &
            source, 21, '/tmp/ffc_session_array_ctor_rt_test')
    end function test_array_constructor_runtime

    logical function test_simple_array()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: a(3)'//new_line('a')// &
                                       '  a(1) = 4'//new_line('a')// &
                                       '  a(2) = 5'//new_line('a')// &
                                       '  print *, a(1) + a(2)'//new_line('a')// &
                                       'end program main'

        test_simple_array = expect_output(source, '           9'//new_line('a'), &
                                          '/tmp/ffc_session_array_simple_test')
    end function test_simple_array

    logical function test_array_parameter_bound()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer, parameter :: n = 3'// &
                                       new_line('a')// &
                                       '  integer :: a(n)'//new_line('a')// &
                                       '  a(1) = 2'//new_line('a')// &
                                       '  a(n) = 7'//new_line('a')// &
                                       '  print *, a(1) + a(n)'//new_line('a')// &
                                       'end program main'

        test_array_parameter_bound = expect_output( &
                                     source, '           9'//new_line('a'), &
                                     '/tmp/ffc_session_array_param_bound_test')
    end function test_array_parameter_bound

    logical function test_array_explicit_bounds()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: a(2:4)'//new_line('a')// &
                                       '  a(2) = 4'//new_line('a')// &
                                       '  a(4) = 5'//new_line('a')// &
                                       '  print *, a(2) + a(4)'//new_line('a')// &
                                       'end program main'

        test_array_explicit_bounds = expect_output( &
                                     source, '           9'//new_line('a'), &
                                     '/tmp/ffc_session_array_explicit_bounds_test')
    end function test_array_explicit_bounds

    logical function test_array_negative_lower_bound()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: values(-1:1)'//new_line('a')// &
                                       '  values(-1) = 4'//new_line('a')// &
                                       '  values(0) = 5'//new_line('a')// &
                                       '  values(1) = 6'//new_line('a')// &
                                       '  print *, values(-1) + values(0) '// &
                                       '+ values(1)'// &
                                       new_line('a')// &
                                       'end program main'

        test_array_negative_lower_bound = expect_output( &
                                          source, '          15'//new_line('a'), &
                                          '/tmp/ffc_session_array_neg_lower_test')
    end function test_array_negative_lower_bound

    logical function test_array_with_loop()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: a(5)'//new_line('a')// &
                                       '  integer :: i'//new_line('a')// &
                                       '  integer :: sum'//new_line('a')// &
                                       '  sum = 0'//new_line('a')// &
                                       '  do i = 1, 5'//new_line('a')// &
                                       '    a(i) = i'//new_line('a')// &
                                       '    sum = sum + a(i)'//new_line('a')// &
                                       '  end do'//new_line('a')// &
                                       '  print *, sum'//new_line('a')// &
                                       'end program main'

        test_array_with_loop = expect_output(source, '          15'//new_line('a'), &
                                             '/tmp/ffc_session_array_loop_test')
    end function test_array_with_loop

    logical function test_array_variable_subscript()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: a(4)'//new_line('a')// &
                                       '  integer :: idx'//new_line('a')// &
                                       '  a(1) = 10'//new_line('a')// &
                                       '  a(2) = 20'//new_line('a')// &
                                       '  a(3) = 30'//new_line('a')// &
                                       '  idx = 2'//new_line('a')// &
                                       '  print *, a(idx)'//new_line('a')// &
                                       'end program main'

        test_array_variable_subscript = expect_output(source, '          20'//new_line('a'), &
                                                      '/tmp/ffc_session_array_var_test')
    end function test_array_variable_subscript

    logical function test_array_element_argument()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: a(1)'//new_line('a')// &
                                       '  a(1) = 1'//new_line('a')// &
                                       '  call bump(a(1))'//new_line('a')// &
                                       '  print *, a(1)'//new_line('a')// &
                                       'contains'//new_line('a')// &
                                       '  subroutine bump(x)'//new_line('a')// &
                                       '    integer :: x'//new_line('a')// &
                                       '    x = x + 4'//new_line('a')// &
                                       '  end subroutine bump'//new_line('a')// &
                                       'end program main'

        test_array_element_argument = expect_output( &
                                      source, '           5'//new_line('a'), &
                                      '/tmp/ffc_session_array_element_arg_test')
    end function test_array_element_argument

    logical function test_array_stop_code()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: a(2)'//new_line('a')// &
                                       '  a(1) = 7'//new_line('a')// &
                                       '  stop a(1)'//new_line('a')// &
                                       'end program main'

        test_array_stop_code = expect_exit_status(source, 7, &
                                                  '/tmp/ffc_session_array_stop_test')
    end function test_array_stop_code

    logical function test_array_in_contained_subroutine()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  call fill()'//new_line('a')// &
                                       'contains'//new_line('a')// &
                                       '  subroutine fill()'//new_line('a')// &
                                       '    integer :: a(2)'//new_line('a')// &
                                       '    a(1) = 4'//new_line('a')// &
                                       '    a(2) = 5'//new_line('a')// &
                                       '    print *, a(1) + a(2)'//new_line('a')// &
                                       '  end subroutine fill'//new_line('a')// &
                                       'end program main'

        test_array_in_contained_subroutine = expect_output( &
                                             source, '           9'//new_line('a'), &
                                             '/tmp/ffc_session_array_subroutine_test')
    end function test_array_in_contained_subroutine

    logical function test_array_in_contained_function()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  print *, fill()'//new_line('a')// &
                                       'contains'//new_line('a')// &
                                       '  integer function fill()'//new_line('a')// &
                                       '    integer :: a(2)'//new_line('a')// &
                                       '    a(1) = 4'//new_line('a')// &
                                       '    a(2) = 5'//new_line('a')// &
                                       '    fill = a(1) + a(2)'//new_line('a')// &
                                       '  end function fill'//new_line('a')// &
                                       'end program main'

        test_array_in_contained_function = expect_output( &
                                           source, '           9'//new_line('a'), &
                                           '/tmp/ffc_session_array_function_test')
    end function test_array_in_contained_function

    logical function test_array_inline_literal_initializer()
        ! Inline array-literal initializer must populate storage, so a later
        ! strided section assignment leaves the untouched elements intact.
        character(len=*), parameter :: source = &
                                       'integer :: arr(10) = '// &
                                       '[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]'// &
                                       new_line('a')// &
                                       'arr(1:9:2) = 0'//new_line('a')// &
                                       'print *, arr'//new_line('a')

        test_array_inline_literal_initializer = expect_output( &
            source, &
            '           0           2           0           4           0'// &
            '           6           0           8           0          10'// &
            new_line('a'), &
            '/tmp/ffc_session_array_inline_literal_test')
    end function test_array_inline_literal_initializer

    logical function test_array_inline_scalar_initializer()
        ! Inline scalar initializer broadcasts to every element.
        character(len=*), parameter :: source = &
                                       'integer :: a(4) = 7'//new_line('a')// &
                                       'print *, a(1) + a(4)'//new_line('a')

        test_array_inline_scalar_initializer = expect_output( &
            source, '          14'//new_line('a'), &
            '/tmp/ffc_session_array_inline_scalar_test')
    end function test_array_inline_scalar_initializer

end program test_session_fixed_size_array_compiler
