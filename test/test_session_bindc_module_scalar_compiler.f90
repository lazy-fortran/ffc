program test_session_bindc_module_scalar_compiler
    use ffc_test_support, only: expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== direct session bind(c) module-scalar compiler test ==='

    all_passed = .true.
    if (.not. test_bindc_real_scalar()) all_passed = .false.
    if (.not. test_bindc_integer_scalar()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: bind(c) module scalars keep their initializer (#296)'

contains

    logical function test_bindc_real_scalar()
        ! #296: a bind(c) module scalar reached through host association must
        ! read its initializer, not the default 0. The IF condition also
        ! exercises the host-associated read that previously hard-failed.
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '  use iso_c_binding'//new_line('a')// &
            '  real(c_double), bind(c) :: v = 42.0d0'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  if (v > 0.0d0) stop int(v)'//new_line('a')// &
            '  stop 1'//new_line('a')// &
            'end program main'

        test_bindc_real_scalar = expect_exit_status( &
            source, 42, '/tmp/ffc_session_bindc_real')
    end function test_bindc_real_scalar

    logical function test_bindc_integer_scalar()
        ! The bind(c) misparse also corrupted the type (an integer printed as
        ! real); check the integer value survives.
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '  use iso_c_binding'//new_line('a')// &
            '  integer(c_int), bind(c) :: n = 7'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  stop n'//new_line('a')// &
            'end program main'

        test_bindc_integer_scalar = expect_exit_status( &
            source, 7, '/tmp/ffc_session_bindc_int')
    end function test_bindc_integer_scalar

end program test_session_bindc_module_scalar_compiler
