program test_session_complex_component_compiler
    use ffc_test_support, only: expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== direct session complex %re/%im component compiler test ==='

    all_passed = .true.
    if (.not. test_scalar_component_write_read()) all_passed = .false.
    if (.not. test_array_element_component()) all_passed = .false.
    if (.not. test_c8_scalar_component()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: complex %re/%im component read/write lowers through '// &
        'direct LIRIC session'

contains

    ! Scalar complex(4): write both components, read them back through both the
    ! %re/%im component syntax and real()/aimag(). Assertions error-stop on
    ! mismatch, so exit status 0 confirms correct lowering.
    logical function test_scalar_component_write_read()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  complex :: x'//new_line('a')// &
            '  x%re = 1'//new_line('a')// &
            '  x%im = 2'//new_line('a')// &
            '  if (abs(x%re - 1.0) > 1e-5) error stop'//new_line('a')// &
            '  if (abs(x%im - 2.0) > 1e-5) error stop'//new_line('a')// &
            '  if (abs(real(x) - 1.0) > 1e-5) error stop'//new_line('a')// &
            '  if (abs(aimag(x) - 2.0) > 1e-5) error stop'//new_line('a')// &
            'end program main'

        test_scalar_component_write_read = expect_exit_status( &
            source, 0, '/tmp/ffc_session_complex_scalar_component_test')
    end function test_scalar_component_write_read

    ! Fixed-size complex(4) array: element %re/%im write and read, plus
    ! real()/aimag() of an array element.
    logical function test_array_element_component()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  complex :: x(3)'//new_line('a')// &
            '  x(1)%re = 1'//new_line('a')// &
            '  x(2)%re = 2'//new_line('a')// &
            '  x(1)%im = 4'//new_line('a')// &
            '  x(2)%im = 5'//new_line('a')// &
            '  if (abs(x(1)%re - 1.0) > 1e-5) error stop'//new_line('a')// &
            '  if (abs(x(2)%im - 5.0) > 1e-5) error stop'//new_line('a')// &
            '  if (abs(real(x(1)) - 1.0) > 1e-5) error stop'//new_line('a')// &
            '  if (abs(aimag(x(2)) - 5.0) > 1e-5) error stop'//new_line('a')// &
            'end program main'

        test_array_element_component = expect_exit_status( &
            source, 0, '/tmp/ffc_session_complex_array_component_test')
    end function test_array_element_component

    ! Scalar complex(8): double-precision component write/read.
    logical function test_c8_scalar_component()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  complex(8) :: z'//new_line('a')// &
            '  z%re = 3.0d0'//new_line('a')// &
            '  z%im = -4.0d0'//new_line('a')// &
            '  if (abs(z%re - 3.0d0) > 1e-12) error stop'//new_line('a')// &
            '  if (abs(z%im + 4.0d0) > 1e-12) error stop'//new_line('a')// &
            'end program main'

        test_c8_scalar_component = expect_exit_status( &
            source, 0, '/tmp/ffc_session_complex_c8_component_test')
    end function test_c8_scalar_component

end program test_session_complex_component_compiler
