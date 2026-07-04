program test_session_equivalence_compiler
    ! #280 (issue_1745): EQUIVALENCE overlays the storage of its members. A write
    ! through one member is observable bit-for-bit through another.
    use ffc_test_support, only: expect_output
    implicit none
    logical :: all_passed

    all_passed = .true.
    print *, '=== EQUIVALENCE storage overlay compiler test ==='

    if (.not. test_int_real_overlay()) all_passed = .false.
    if (.not. test_real_write_seen_as_int()) all_passed = .false.

    if (all_passed) then
        print *, 'PASS: EQUIVALENCE overlays storage through direct LIRIC session'
    else
        print *, 'FAIL: EQUIVALENCE storage overlay test failed'
    end if
    if (.not. all_passed) stop 1

contains

    logical function test_int_real_overlay()
        ! Writing the integer member, then reading the real member, must yield
        ! the reinterpreted bit pattern, not zero.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  real :: r'//new_line('a')// &
            '  equivalence (i, r)'//new_line('a')// &
            '  i = 42'//new_line('a')// &
            '  print *, i'//new_line('a')// &
            '  print *, r'//new_line('a')// &
            'end program main'
        test_int_real_overlay = expect_output( &
            source, '          42'//new_line('a')// &
            '   5.88545355E-44'//new_line('a'), '/tmp/ffc_equiv_ir_test')
    end function test_int_real_overlay

    logical function test_real_write_seen_as_int()
        ! Writing the real member, then reading the integer member, exposes the
        ! same shared storage as the bit pattern of 1.0.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  real :: r'//new_line('a')// &
            '  equivalence (i, r)'//new_line('a')// &
            '  r = 1.0'//new_line('a')// &
            '  print *, i'//new_line('a')// &
            'end program main'
        ! 1.0 as IEEE-754 single precision is 0x3F800000 = 1065353216.
        test_real_write_seen_as_int = expect_output( &
            source, '  1065353216'//new_line('a'), '/tmp/ffc_equiv_ri_test')
    end function test_real_write_seen_as_int

end program test_session_equivalence_compiler
