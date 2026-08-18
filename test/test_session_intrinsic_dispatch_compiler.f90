program test_session_intrinsic_dispatch_compiler
    !! Independent GNU-output oracle for scalar intrinsic result kinds (#453).
    use ffc_test_support, only: expect_error_contains, expect_exit_status, &
        expect_output
    implicit none
    logical :: ok

    ok = .true.
    if (.not. test_legacy_integer_conversions()) ok = .false.
    if (.not. test_legacy_conversion_arity()) ok = .false.
    if (.not. test_legacy_conversion_type()) ok = .false.
    if (.not. test_legacy_conversion_shadowing()) ok = .false.
    if (.not. test_real_kind8_of_complex4()) ok = .false.
    if (.not. test_real_kind4_of_complex8()) ok = .false.
    if (.not. test_real_no_kind_keeps_component_width()) ok = .false.
    if (.not. test_aimag_keeps_component_width()) ok = .false.
    if (.not. expect_output( &
        'program main'//new_line('a')// &
        '  complex :: z'//new_line('a')// &
        '  real(8) :: r'//new_line('a')// &
        '  z = (3.5, 4.0)'//new_line('a')// &
        '  r = real(z, kind=8)'//new_line('a')// &
        '  print *, r'//new_line('a')// &
        'end program main', &
        '   3.5000000000000000     '//new_line('a'), &
        '/tmp/ffc_probe_real_k8')) ok = .false.
    if (.not. expect_output( &
        'program main'//new_line('a')// &
        '  complex(8) :: z'//new_line('a')// &
        '  real :: r'//new_line('a')// &
        '  z = (3.5d0, 4.0d0)'//new_line('a')// &
        '  r = real(z, kind=4)'//new_line('a')// &
        '  print *, r'//new_line('a')// &
        'end program main', &
        '   3.50000000    '//new_line('a'), &
        '/tmp/ffc_probe_real_k4')) ok = .false.
    if (.not. expect_output( &
        'program main'//new_line('a')// &
        '  complex(8) :: z'//new_line('a')// &
        '  real(8) :: r'//new_line('a')// &
        '  z = (3.5d0, 4.0d0)'//new_line('a')// &
        '  r = real(z)'//new_line('a')// &
        '  print *, r'//new_line('a')// &
        'end program main', &
        '   3.5000000000000000     '//new_line('a'), &
        '/tmp/ffc_probe_real_default')) ok = .false.
    if (.not. expect_output( &
        'program main'//new_line('a')// &
        '  complex :: z'//new_line('a')// &
        '  real :: r'//new_line('a')// &
        '  z = (3.5, 4.25)'//new_line('a')// &
        '  r = aimag(z)'//new_line('a')// &
        '  print *, r'//new_line('a')// &
        'end program main', &
        '   4.25000000    '//new_line('a'), &
        '/tmp/ffc_probe_aimag')) ok = .false.

    if (.not. ok) stop 1
    print *, 'PASS: typed scalar conversion and complex component kinds'

contains

    logical function test_legacy_integer_conversions()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer(1) :: i1 = 1_1'//new_line('a')// &
            '  integer(2) :: i2 = 2_2'//new_line('a')// &
            '  integer(4) :: i4 = 4_4'//new_line('a')// &
            '  integer(8) :: i8 = 8_8'//new_line('a')// &
            '  if (float(i1) /= 1.0) stop 1'//new_line('a')// &
            '  if (float(i2) /= 2.0) stop 2'//new_line('a')// &
            '  if (float(i4) /= 4.0) stop 3'//new_line('a')// &
            '  if (float(i8) /= 8.0) stop 4'//new_line('a')// &
            '  if (dfloat(i1) /= 1.0d0) stop 5'//new_line('a')// &
            '  if (dfloat(i2) /= 2.0d0) stop 6'//new_line('a')// &
            '  if (dfloat(i4) /= 4.0d0) stop 7'//new_line('a')// &
            '  if (dfloat(i8) /= 8.0d0) stop 8'//new_line('a')// &
            '  if (kind(float(i8)) /= kind(1.0)) stop 9'//new_line('a')// &
            '  if (kind(dfloat(i4)) /= kind(1.0_8)) stop 10'//new_line('a')// &
            '  if (dfloat(i4*i2) /= 8.0d0) stop 11'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_legacy_integer_conversions = expect_exit_status( &
            source, 0, '/tmp/ffc_probe_legacy_integer_conversions')
    end function test_legacy_integer_conversions

    logical function test_legacy_conversion_arity()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  real :: x'//new_line('a')// &
            '  x = float(i, 4)'//new_line('a')// &
            'end program main'

        test_legacy_conversion_arity = expect_error_contains( &
            source, 'invalid argument count', &
            '/tmp/ffc_probe_legacy_conversion_arity')
    end function test_legacy_conversion_arity

    logical function test_legacy_conversion_type()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real :: x'//new_line('a')// &
            '  x = dfloat(1.5)'//new_line('a')// &
            'end program main'

        test_legacy_conversion_type = expect_error_contains( &
            source, 'argument must be INTEGER', &
            '/tmp/ffc_probe_legacy_conversion_type')
    end function test_legacy_conversion_type

    logical function test_legacy_conversion_shadowing()
        test_legacy_conversion_shadowing = expect_output( &
            'program main'//new_line('a')// &
            '  real :: x'//new_line('a')// &
            '  x = float(3)'//new_line('a')// &
            '  print *, x'//new_line('a')// &
            'contains'//new_line('a')// &
            '  real function float(i)'//new_line('a')// &
            '    integer :: i'//new_line('a')// &
            '    float = 7.0'//new_line('a')// &
            '  end function float'//new_line('a')// &
            'end program main', &
            '   7.00000000    '//new_line('a'), &
            '/tmp/ffc_probe_legacy_conversion_shadowing')
    end function test_legacy_conversion_shadowing

    logical function test_real_kind8_of_complex4()
        !! REAL(z, KIND=8) on a complex(4) widens the extracted component.
        test_real_kind8_of_complex4 = expect_output( &
            'program main'//new_line('a')// &
            '  complex :: z'//new_line('a')// &
            '  real(8) :: r'//new_line('a')// &
            '  z = (3.5, 4.0)'//new_line('a')// &
            '  r = real(z, kind=8)'//new_line('a')// &
            '  print *, r'//new_line('a')// &
            'end program main', &
            '   3.5000000000000000     '//new_line('a'), &
            '/tmp/ffc_probe_real_k8')
    end function test_real_kind8_of_complex4

    logical function test_real_kind4_of_complex8()
        !! REAL(z, KIND=4) narrows the extracted component.
        test_real_kind4_of_complex8 = expect_output( &
            'program main'//new_line('a')// &
            '  complex(8) :: z'//new_line('a')// &
            '  real :: r'//new_line('a')// &
            '  z = (3.5d0, 4.0d0)'//new_line('a')// &
            '  r = real(z, kind=4)'//new_line('a')// &
            '  print *, r'//new_line('a')// &
            'end program main', &
            '   3.50000000    '//new_line('a'), &
            '/tmp/ffc_probe_real_k4')
    end function test_real_kind4_of_complex8

    logical function test_real_no_kind_keeps_component_width()
        !! Without KIND, REAL(z) retains the complex component width.
        test_real_no_kind_keeps_component_width = expect_output( &
            'program main'//new_line('a')// &
            '  complex(8) :: z'//new_line('a')// &
            '  real(8) :: r'//new_line('a')// &
            '  z = (3.5d0, 4.0d0)'//new_line('a')// &
            '  r = real(z)'//new_line('a')// &
            '  print *, r'//new_line('a')// &
            'end program main', &
            '   3.5000000000000000     '//new_line('a'), &
            '/tmp/ffc_probe_real_default')
    end function test_real_no_kind_keeps_component_width

    logical function test_aimag_keeps_component_width()
        !! AIMAG has no KIND selector and keeps the component width.
        test_aimag_keeps_component_width = expect_output( &
            'program main'//new_line('a')// &
            '  complex :: z'//new_line('a')// &
            '  real :: r'//new_line('a')// &
            '  z = (3.5, 4.25)'//new_line('a')// &
            '  r = aimag(z)'//new_line('a')// &
            '  print *, r'//new_line('a')// &
            'end program main', &
            '   4.25000000    '//new_line('a'), &
            '/tmp/ffc_probe_aimag')
    end function test_aimag_keeps_component_width

end program test_session_intrinsic_dispatch_compiler
