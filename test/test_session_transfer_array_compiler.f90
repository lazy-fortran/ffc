program test_session_transfer_array_compiler
    ! Array-valued TRANSFER(source, mold [, size]) for intrinsic types whose
    ! elements share a byte size (integer(4)<->real(4), integer(8)<->real(8)).
    ! The bit pattern of each source element is reinterpreted as the mold's
    ! type and stored into the whole-array target; SIZE, when present, fixes
    ! the number of result elements.
    use ffc_test_support, only: expect_output, expect_error_contains
    implicit none
    logical :: all_passed

    all_passed = .true.
    print *, '=== TRANSFER array-valued compiler test ==='

    if (.not. test_array_to_array()) all_passed = .false.
    if (.not. test_scalar_to_array()) all_passed = .false.
    if (.not. test_array_to_scalar()) all_passed = .false.
    if (.not. test_explicit_size()) all_passed = .false.
    if (.not. test_real64_array()) all_passed = .false.
    if (.not. test_negative_size_rejected()) all_passed = .false.

    if (all_passed) then
        print *, 'PASS: array-valued TRANSFER lowers through direct LIRIC session'
    else
        print *, 'FAIL: array-valued TRANSFER test failed'
    end if
    if (.not. all_passed) stop 1

contains

    logical function test_array_to_array()
        ! real(4) 1.0 / 2.0 as integer(4): 0x3F800000 / 0x40000000.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  real :: r(2)'//new_line('a')// &
            '  integer :: a(2)'//new_line('a')// &
            '  r(1) = 1.0'//new_line('a')// &
            '  r(2) = 2.0'//new_line('a')// &
            '  a = transfer(r, a)'//new_line('a')// &
            '  print *, a(1)'//new_line('a')// &
            '  print *, a(2)'//new_line('a')// &
            'end program main'
        test_array_to_array = expect_output( &
            source, '  1065353216'//new_line('a')//'  1073741824'//new_line('a'), &
            '/tmp/ffc_transfer_arr2arr_test')
    end function test_array_to_array

    logical function test_scalar_to_array()
        ! A scalar real(4) source fills a one-element integer(4) result.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  real :: r'//new_line('a')// &
            '  integer :: a(1)'//new_line('a')// &
            '  r = 1.0'//new_line('a')// &
            '  a = transfer(r, a)'//new_line('a')// &
            '  print *, a(1)'//new_line('a')// &
            'end program main'
        test_scalar_to_array = expect_output( &
            source, '  1065353216'//new_line('a'), &
            '/tmp/ffc_transfer_scal2arr_test')
    end function test_scalar_to_array

    logical function test_array_to_scalar()
        ! A scalar mold takes the leading element's bits.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  real :: r(2)'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  r(1) = 2.0'//new_line('a')// &
            '  r(2) = 1.0'//new_line('a')// &
            '  i = transfer(r, i)'//new_line('a')// &
            '  print *, i'//new_line('a')// &
            'end program main'
        test_array_to_scalar = expect_output( &
            source, '  1073741824'//new_line('a'), &
            '/tmp/ffc_transfer_arr2scal_test')
    end function test_array_to_scalar

    logical function test_explicit_size()
        ! transfer(r, 0.0, 2) with an explicit SIZE yields a rank-1 result.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer :: n(3)'//new_line('a')// &
            '  real :: r(2)'//new_line('a')// &
            '  n(1) = 1065353216'//new_line('a')// &
            '  n(2) = 1073741824'//new_line('a')// &
            '  n(3) = 0'//new_line('a')// &
            '  r = transfer(n, 0.0, 2)'//new_line('a')// &
            '  print *, r(1)'//new_line('a')// &
            '  print *, r(2)'//new_line('a')// &
            'end program main'
        test_explicit_size = expect_output( &
            source, '   1.00000000    '//new_line('a')// &
            '   2.00000000    '//new_line('a'), &
            '/tmp/ffc_transfer_size_test')
    end function test_explicit_size

    logical function test_real64_array()
        ! integer(8) 4607182418800017408 is real(8) 1.0.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer(8) :: n(2)'//new_line('a')// &
            '  real(8) :: d(2)'//new_line('a')// &
            '  n(1) = 4607182418800017408_8'//new_line('a')// &
            '  n(2) = 4611686018427387904_8'//new_line('a')// &
            '  d = transfer(n, d)'//new_line('a')// &
            '  print *, d(1)'//new_line('a')// &
            '  print *, d(2)'//new_line('a')// &
            'end program main'
        test_real64_array = expect_output( &
            source, '   1.0000000000000000     '//new_line('a')// &
            '   2.0000000000000000     '//new_line('a'), &
            '/tmp/ffc_transfer_r64_test')
    end function test_real64_array

    logical function test_negative_size_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  real :: r(2)'//new_line('a')// &
            '  integer :: a(2)'//new_line('a')// &
            '  r(1) = 1.0'//new_line('a')// &
            '  r(2) = 2.0'//new_line('a')// &
            '  a = transfer(r, a, -1)'//new_line('a')// &
            '  print *, a(1)'//new_line('a')// &
            'end program main'
        test_negative_size_rejected = expect_error_contains( &
            source, 'transfer size', '/tmp/ffc_transfer_negsize_test')
    end function test_negative_size_rejected

end program test_session_transfer_array_compiler
