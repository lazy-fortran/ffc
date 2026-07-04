program test_session_transfer_compiler
    ! TRANSFER(source, mold) for scalar intrinsic types of matching byte size:
    ! integer(4)<->real(4) and integer(8)<->real(8), reinterpreting source's
    ! bit pattern as mold's type.
    use ffc_test_support, only: expect_output
    implicit none
    logical :: all_passed

    all_passed = .true.
    print *, '=== TRANSFER scalar bit-reinterpret compiler test ==='

    if (.not. test_int_to_real()) all_passed = .false.
    if (.not. test_real_to_int()) all_passed = .false.
    if (.not. test_int64_to_real64()) all_passed = .false.
    if (.not. test_same_kind_identity()) all_passed = .false.

    if (all_passed) then
        print *, 'PASS: TRANSFER reinterprets scalar bit patterns through direct '// &
            'LIRIC session'
    else
        print *, 'FAIL: TRANSFER scalar bit-reinterpret test failed'
    end if
    if (.not. all_passed) stop 1

contains

    logical function test_int_to_real()
        ! integer(4) 21432 reinterpreted as real(4): gfortran prints
        ! 3.00326287E-41 for this bit pattern.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer :: x'//new_line('a')// &
            '  x = 21432'//new_line('a')// &
            '  print *, transfer(x, 1.0)'//new_line('a')// &
            'end program main'
        test_int_to_real = expect_output( &
            source, '   3.00326287E-41'//new_line('a'), &
            '/tmp/ffc_transfer_i2r_test')
    end function test_int_to_real

    logical function test_real_to_int()
        ! real(4) 1.0 reinterpreted as integer(4): IEEE-754 single precision
        ! 0x3F800000 = 1065353216.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  real :: r'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  r = 1.0'//new_line('a')// &
            '  i = transfer(r, i)'//new_line('a')// &
            '  print *, i'//new_line('a')// &
            'end program main'
        test_real_to_int = expect_output( &
            source, '  1065353216'//new_line('a'), &
            '/tmp/ffc_transfer_r2i_test')
    end function test_real_to_int

    logical function test_int64_to_real64()
        ! real(8) 1.0d0 reinterpreted as integer(8): IEEE-754 double precision
        ! 0x3FF0000000000000 = 4607182418800017408.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  real(8) :: d'//new_line('a')// &
            '  integer(8) :: n'//new_line('a')// &
            '  d = 1.0d0'//new_line('a')// &
            '  n = transfer(d, n)'//new_line('a')// &
            '  print *, n'//new_line('a')// &
            'end program main'
        test_int64_to_real64 = expect_output( &
            source, '  4607182418800017408'//new_line('a'), &
            '/tmp/ffc_transfer_i642r64_test')
    end function test_int64_to_real64

    logical function test_same_kind_identity()
        ! transfer between matching source and mold kinds is the identity.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer :: i, j'//new_line('a')// &
            '  i = 42'//new_line('a')// &
            '  j = transfer(i, j)'//new_line('a')// &
            '  print *, j'//new_line('a')// &
            'end program main'
        test_same_kind_identity = expect_output( &
            source, '          42'//new_line('a'), &
            '/tmp/ffc_transfer_identity_test')
    end function test_same_kind_identity

end program test_session_transfer_compiler
