program test_session_equivalence_compiler
    ! #280 (issue_1745): EQUIVALENCE overlays the storage of its members. A write
    ! through one member is observable bit-for-bit through another.
    use ffc_test_support, only: expect_output, expect_error_contains
    implicit none
    logical :: all_passed

    all_passed = .true.
    print *, '=== EQUIVALENCE storage overlay compiler test ==='

    if (.not. test_int_real_overlay()) all_passed = .false.
    if (.not. test_real_write_seen_as_int()) all_passed = .false.
    if (.not. test_module_named_constant_subscript()) all_passed = .false.
    if (.not. test_local_named_constant_subscript()) all_passed = .false.
    if (.not. test_nonconstant_subscript_rejected()) all_passed = .false.

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

    logical function test_module_named_constant_subscript()
        ! #370: an EQUIVALENCE array-element subscript is a constant
        ! expression. A USE-associated named constant must fold before the
        ! byte offset is computed, so r overlays a(3), not a(1).
        character(len=*), parameter :: source = &
            'module cst'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer, parameter :: np = 3'//new_line('a')// &
            'end module cst'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use cst, only: np'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer :: a(5)'//new_line('a')// &
            '  real :: r'//new_line('a')// &
            '  equivalence (a(np), r)'//new_line('a')// &
            '  a(1) = 0'//new_line('a')// &
            '  a(3) = 0'//new_line('a')// &
            '  r = 1.0'//new_line('a')// &
            '  print *, a(1)'//new_line('a')// &
            '  print *, a(3)'//new_line('a')// &
            'end program main'
        ! 1.0 as IEEE-754 single precision is 0x3F800000 = 1065353216, and it
        ! lands on element 3 only when np folds to 3.
        test_module_named_constant_subscript = expect_output( &
            source, '           0'//new_line('a')// &
            '  1065353216'//new_line('a'), '/tmp/ffc_equiv_modparam_test')
    end function test_module_named_constant_subscript

    logical function test_local_named_constant_subscript()
        ! The same fold for a named constant declared in the program itself.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer, parameter :: np = 2'//new_line('a')// &
            '  integer :: a(4)'//new_line('a')// &
            '  real :: r'//new_line('a')// &
            '  equivalence (a(np), r)'//new_line('a')// &
            '  a(2) = 0'//new_line('a')// &
            '  r = 1.0'//new_line('a')// &
            '  print *, a(2)'//new_line('a')// &
            'end program main'
        test_local_named_constant_subscript = expect_output( &
            source, '  1065353216'//new_line('a'), &
            '/tmp/ffc_equiv_localparam_test')
    end function test_local_named_constant_subscript

    logical function test_nonconstant_subscript_rejected()
        ! A nonconstant subscript is not a constant expression: diagnose it
        ! rather than picking an arbitrary offset.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer :: n'//new_line('a')// &
            '  integer :: a(4)'//new_line('a')// &
            '  real :: r'//new_line('a')// &
            '  equivalence (a(n), r)'//new_line('a')// &
            '  n = 1'//new_line('a')// &
            '  r = 1.0'//new_line('a')// &
            '  print *, a(1)'//new_line('a')// &
            'end program main'
        test_nonconstant_subscript_rejected = expect_error_contains( &
            source, 'EQUIVALENCE subscript', '/tmp/ffc_equiv_nonconst_test')
    end function test_nonconstant_subscript_rejected

end program test_session_equivalence_compiler
