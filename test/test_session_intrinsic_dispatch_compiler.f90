program test_session_intrinsic_dispatch_compiler
    !! Behavioral oracle for scalar intrinsic dispatch (#453).
    !!
    !! The case the issue calls out is REAL(A, KIND) on a complex operand: the
    !! KIND selector fixes the result width, and the complex operand's own
    !! component width does not. Expected values are gfortran's.
    use ffc_test_support, only: expect_output
    implicit none
    logical :: ok

    print *, '=== intrinsic dispatch test ==='

    ok = .true.
    if (.not. test_real_kind8_of_complex4()) ok = .false.
    if (.not. test_real_kind4_of_complex8()) ok = .false.
    if (.not. test_real_no_kind_keeps_component_width()) ok = .false.
    if (.not. test_aimag_keeps_component_width()) ok = .false.
    if (.not. ok) stop 1

    print *, 'PASS: scalar intrinsics dispatch on their declared result kind'

contains

    logical function test_real_kind8_of_complex4() result(res)
        !! REAL(z, KIND=8) on a complex(4) is real(8): the f32 component is
        !! extracted and widened, not reinterpreted and not rejected.
        res = expect_output( &
            'program main'//new_line('a')// &
            '  complex :: z'//new_line('a')// &
            '  real(8) :: r'//new_line('a')// &
            '  z = (3.5, 4.0)'//new_line('a')// &
            '  r = real(z, kind=8)'//new_line('a')// &
            '  print *, r'//new_line('a')// &
            'end program main', &
            '   3.5000000000000000     '//new_line('a'), &
            '/tmp/ffc_intr_real_k8_c4')
    end function test_real_kind8_of_complex4

    logical function test_real_kind4_of_complex8() result(res)
        !! The mirror: REAL(z, KIND=4) on a complex(8) is real(4).
        res = expect_output( &
            'program main'//new_line('a')// &
            '  complex(8) :: z'//new_line('a')// &
            '  real :: r'//new_line('a')// &
            '  z = (3.5d0, 4.0d0)'//new_line('a')// &
            '  r = real(z, kind=4)'//new_line('a')// &
            '  print *, r'//new_line('a')// &
            'end program main', &
            '   3.50000000    '//new_line('a'), &
            '/tmp/ffc_intr_real_k4_c8')
    end function test_real_kind4_of_complex8

    logical function test_real_no_kind_keeps_component_width() result(res)
        !! Without a KIND selector, REAL(A) on a complex keeps the argument's
        !! component kind (F2018 16.9.160) -- the one no-KIND case that is not
        !! default real.
        res = expect_output( &
            'program main'//new_line('a')// &
            '  complex(8) :: z'//new_line('a')// &
            '  real(8) :: r'//new_line('a')// &
            '  z = (3.5d0, 4.0d0)'//new_line('a')// &
            '  r = real(z)'//new_line('a')// &
            '  print *, r'//new_line('a')// &
            'end program main', &
            '   3.5000000000000000     '//new_line('a'), &
            '/tmp/ffc_intr_real_nokind')
    end function test_real_no_kind_keeps_component_width

    logical function test_aimag_keeps_component_width() result(res)
        !! AIMAG takes no KIND selector, so its result kind is the operand's
        !! component kind and nothing else.
        res = expect_output( &
            'program main'//new_line('a')// &
            '  complex :: z'//new_line('a')// &
            '  real :: r'//new_line('a')// &
            '  z = (3.5, 4.25)'//new_line('a')// &
            '  r = aimag(z)'//new_line('a')// &
            '  print *, r'//new_line('a')// &
            'end program main', &
            '   4.25000000    '//new_line('a'), &
            '/tmp/ffc_intr_aimag')
    end function test_aimag_keeps_component_width

end program test_session_intrinsic_dispatch_compiler
