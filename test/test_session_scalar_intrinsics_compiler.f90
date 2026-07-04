program test_session_scalar_intrinsics_compiler
    use ffc_test_support, only: expect_exit_status, expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session scalar intrinsics compiler test ==='

    all_passed = .true.
    if (.not. test_iabs_negative()) all_passed = .false.
    if (.not. test_iabs_positive()) all_passed = .false.
    if (.not. test_sign_intrinsic()) all_passed = .false.
    if (.not. test_dim_intrinsic()) all_passed = .false.
    if (.not. test_modulo_intrinsic()) all_passed = .false.
    if (.not. test_ibits_intrinsic()) all_passed = .false.
    if (.not. test_ibset_intrinsic()) all_passed = .false.
    if (.not. test_ibclr_intrinsic()) all_passed = .false.
    if (.not. test_btest_intrinsic()) all_passed = .false.
    if (.not. test_bit_size_intrinsic()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: scalar intrinsics (iabs, dim, modulo, bit ops) lower '// &
        'through direct LIRIC session'

contains

    logical function test_iabs_negative()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: x'//new_line('a')// &
            '  x = iabs(-99)'//new_line('a')// &
            '  stop x'//new_line('a')// &
            'end program main'

        test_iabs_negative = expect_exit_status( &
            source, 99, '/tmp/ffc_session_iabs_negative_test')
    end function test_iabs_negative

    logical function test_iabs_positive()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: x'//new_line('a')// &
            '  x = iabs(42)'//new_line('a')// &
            '  stop x'//new_line('a')// &
            'end program main'

        test_iabs_positive = expect_exit_status( &
            source, 42, '/tmp/ffc_session_iabs_positive_test')
    end function test_iabs_positive

    logical function test_sign_intrinsic()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: x'//new_line('a')// &
            '  x = sign(-3, 7)'//new_line('a')// &
            '  stop x'//new_line('a')// &
            'end program main'

        test_sign_intrinsic = expect_exit_status( &
            source, 3, '/tmp/ffc_session_sign_scalar_test')
    end function test_sign_intrinsic

    logical function test_dim_intrinsic()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: x'//new_line('a')// &
            '  x = dim(2, 9)'//new_line('a')// &
            '  stop x'//new_line('a')// &
            'end program main'

        test_dim_intrinsic = expect_exit_status( &
            source, 0, '/tmp/ffc_session_dim_scalar_test')
    end function test_dim_intrinsic

    logical function test_modulo_intrinsic()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: x'//new_line('a')// &
            '  x = modulo(-7, 3)'//new_line('a')// &
            '  stop x'//new_line('a')// &
            'end program main'

        test_modulo_intrinsic = expect_exit_status( &
            source, 2, '/tmp/ffc_session_modulo_scalar_test')
    end function test_modulo_intrinsic

    logical function test_ibits_intrinsic()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: x'//new_line('a')// &
            '  x = ibits(10, 2, 2)'//new_line('a')// &
            '  stop x'//new_line('a')// &
            'end program main'

        test_ibits_intrinsic = expect_exit_status( &
            source, 2, '/tmp/ffc_session_ibits_test')
    end function test_ibits_intrinsic

    logical function test_ibset_intrinsic()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: x'//new_line('a')// &
            '  x = ibset(1, 2)'//new_line('a')// &
            '  stop x'//new_line('a')// &
            'end program main'

        test_ibset_intrinsic = expect_exit_status( &
            source, 5, '/tmp/ffc_session_ibset_test')
    end function test_ibset_intrinsic

    logical function test_ibclr_intrinsic()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: x'//new_line('a')// &
            '  x = ibclr(3, 0)'//new_line('a')// &
            '  stop x'//new_line('a')// &
            'end program main'

        test_ibclr_intrinsic = expect_exit_status( &
            source, 2, '/tmp/ffc_session_ibclr_test')
    end function test_ibclr_intrinsic

    logical function test_btest_intrinsic()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  logical :: ok'//new_line('a')// &
            '  ok = btest(5, 0) .and. .not. btest(5, 1)'//new_line('a')// &
            '  print *, ok'//new_line('a')// &
            'end program main'

        test_btest_intrinsic = expect_output( &
            source, ' T'//new_line('a'), '/tmp/ffc_session_btest_test')
    end function test_btest_intrinsic

    logical function test_bit_size_intrinsic()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: x'//new_line('a')// &
            '  x = bit_size(0)'//new_line('a')// &
            '  print *, x'//new_line('a')// &
            'end program main'

        test_bit_size_intrinsic = expect_output( &
            source, '          32'//new_line('a'), &
            '/tmp/ffc_session_bit_size_test')
    end function test_bit_size_intrinsic

end program test_session_scalar_intrinsics_compiler
