program test_session_multi_declaration_compiler
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== multi-declaration compiler test ==='

    all_passed = .true.
    if (.not. test_multi_scalar()) all_passed = .false.
    if (.not. test_multi_array_distinct_shapes()) all_passed = .false.
    if (.not. test_array_dummy_arguments()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: multi-declarations lower through direct LIRIC session'

contains

    logical function test_multi_scalar()
        ! integer :: a, b, c on one line declares three distinct scalars.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer :: a, b, c'//new_line('a')// &
            '  a = 10'//new_line('a')// &
            '  b = 20'//new_line('a')// &
            '  c = 30'//new_line('a')// &
            '  print *, a, b, c'//new_line('a')// &
            'end program main'

        test_multi_scalar = expect_output( &
            source, '          10          20          30'//new_line('a'), &
            '/tmp/ffc_multi_scalar_test')
    end function test_multi_scalar

    logical function test_multi_array_distinct_shapes()
        ! real :: x(2), y(3) declares two arrays with distinct extents on one
        ! line; each entity keeps its own shape.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  real :: x(2), y(3)'//new_line('a')// &
            '  x = 1.0'//new_line('a')// &
            '  y = 2.0'//new_line('a')// &
            '  x(2) = 5.0'//new_line('a')// &
            '  y(3) = 9.0'//new_line('a')// &
            '  print *, x(1), x(2), y(1), y(3)'//new_line('a')// &
            'end program main'

        test_multi_array_distinct_shapes = expect_output( &
            source, &
            '   1.00000000       5.00000000       2.00000000       9.00000000    '// &
            new_line('a'), &
            '/tmp/ffc_multi_array_test')
    end function test_multi_array_distinct_shapes

    logical function test_array_dummy_arguments()
        ! Multi-name array dummy arguments (p(2), q(3)) bind onto the caller's
        ! storage; writes through the dummy reach the actual.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  real :: x(2), y(3)'//new_line('a')// &
            '  x = 1.0'//new_line('a')// &
            '  y = 2.0'//new_line('a')// &
            '  call fill(x, y)'//new_line('a')// &
            '  print *, x(1), y(3)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine fill(p, q)'//new_line('a')// &
            '    real, intent(inout) :: p(2), q(3)'//new_line('a')// &
            '    p(1) = 7.0'//new_line('a')// &
            '    q(3) = 8.0'//new_line('a')// &
            '  end subroutine fill'//new_line('a')// &
            'end program main'

        test_array_dummy_arguments = expect_output( &
            source, '   7.00000000       8.00000000    '//new_line('a'), &
            '/tmp/ffc_array_dummy_test')
    end function test_array_dummy_arguments

end program test_session_multi_declaration_compiler
