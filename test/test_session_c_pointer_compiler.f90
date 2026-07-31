program test_session_c_pointer_compiler
    use ffc_test_support, only: expect_error_contains, expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== ISO C pointer round-trip test ==='

    all_passed = .true.
    if (.not. test_scalar_round_trip_aliases_storage()) all_passed = .false.
    if (.not. test_array_round_trip_with_shape()) all_passed = .false.
    if (.not. test_two_argument_c_associated_true()) all_passed = .false.
    if (.not. test_two_argument_c_associated_false()) all_passed = .false.
    if (.not. test_shape_rank_mismatch_rejected()) all_passed = .false.
    if (.not. test_negative_extent_rejected()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: ISO C pointer round-trips lower correctly'

contains

    logical function test_scalar_round_trip_aliases_storage()
        ! c_f_pointer(c_loc(x), p) makes p alias x's storage.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  use, intrinsic :: iso_c_binding'//new_line('a')// &
            '  integer, target :: x'//new_line('a')// &
            '  integer, pointer :: p'//new_line('a')// &
            '  type(c_ptr) :: cp'//new_line('a')// &
            '  x = 11'//new_line('a')// &
            '  cp = c_loc(x)'//new_line('a')// &
            '  call c_f_pointer(cp, p)'//new_line('a')// &
            '  stop p'//new_line('a')// &
            'end program main'

        test_scalar_round_trip_aliases_storage = expect_exit_status( &
            source, 11, '/tmp/ffc_session_c_pointer_scalar')
    end function test_scalar_round_trip_aliases_storage

    logical function test_array_round_trip_with_shape()
        ! A rank-1 round trip with SHAPE preserves values and extent.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  use, intrinsic :: iso_c_binding'//new_line('a')// &
            '  integer, dimension(3), target :: a'//new_line('a')// &
            '  integer, pointer :: p(:)'//new_line('a')// &
            '  type(c_ptr) :: cp'//new_line('a')// &
            '  a(1) = 4'//new_line('a')// &
            '  a(2) = 5'//new_line('a')// &
            '  a(3) = 6'//new_line('a')// &
            '  cp = c_loc(a)'//new_line('a')// &
            '  call c_f_pointer(cp, p, [3])'//new_line('a')// &
            '  stop p(2) + size(p)'//new_line('a')// &
            'end program main'

        test_array_round_trip_with_shape = expect_exit_status( &
            source, 8, '/tmp/ffc_session_c_pointer_array')
    end function test_array_round_trip_with_shape

    logical function test_two_argument_c_associated_true()
        ! c_associated(p, q) is true when both refer to the same target.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  use, intrinsic :: iso_c_binding'//new_line('a')// &
            '  integer, target :: x'//new_line('a')// &
            '  type(c_ptr) :: cp, cq'//new_line('a')// &
            '  x = 1'//new_line('a')// &
            '  cp = c_loc(x)'//new_line('a')// &
            '  cq = c_loc(x)'//new_line('a')// &
            '  if (c_associated(cp, cq)) then'//new_line('a')// &
            '    stop 5'//new_line('a')// &
            '  else'//new_line('a')// &
            '    stop 6'//new_line('a')// &
            '  end if'//new_line('a')// &
            'end program main'

        test_two_argument_c_associated_true = expect_exit_status( &
            source, 5, '/tmp/ffc_session_c_pointer_assoc2_true')
    end function test_two_argument_c_associated_true

    logical function test_two_argument_c_associated_false()
        ! c_associated(p, q) is false for distinct targets.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  use, intrinsic :: iso_c_binding'//new_line('a')// &
            '  integer, target :: x, y'//new_line('a')// &
            '  type(c_ptr) :: cp, cq'//new_line('a')// &
            '  x = 1'//new_line('a')// &
            '  y = 2'//new_line('a')// &
            '  cp = c_loc(x)'//new_line('a')// &
            '  cq = c_loc(y)'//new_line('a')// &
            '  if (c_associated(cp, cq)) then'//new_line('a')// &
            '    stop 5'//new_line('a')// &
            '  else'//new_line('a')// &
            '    stop 6'//new_line('a')// &
            '  end if'//new_line('a')// &
            'end program main'

        test_two_argument_c_associated_false = expect_exit_status( &
            source, 6, '/tmp/ffc_session_c_pointer_assoc2_false')
    end function test_two_argument_c_associated_false

    logical function test_shape_rank_mismatch_rejected()
        ! SHAPE size must equal the rank of FPTR.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  use, intrinsic :: iso_c_binding'//new_line('a')// &
            '  integer, dimension(4), target :: a'//new_line('a')// &
            '  integer, pointer :: p(:)'//new_line('a')// &
            '  type(c_ptr) :: cp'//new_line('a')// &
            '  a(1) = 1'//new_line('a')// &
            '  cp = c_loc(a)'//new_line('a')// &
            '  call c_f_pointer(cp, p, [2, 2])'//new_line('a')// &
            '  stop p(1)'//new_line('a')// &
            'end program main'

        test_shape_rank_mismatch_rejected = expect_error_contains( &
            source, 'SHAPE', '/tmp/ffc_session_c_pointer_shape_rank')
    end function test_shape_rank_mismatch_rejected

    logical function test_negative_extent_rejected()
        ! A negative extent in SHAPE is invalid.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  use, intrinsic :: iso_c_binding'//new_line('a')// &
            '  integer, dimension(4), target :: a'//new_line('a')// &
            '  integer, pointer :: p(:)'//new_line('a')// &
            '  type(c_ptr) :: cp'//new_line('a')// &
            '  a(1) = 1'//new_line('a')// &
            '  cp = c_loc(a)'//new_line('a')// &
            '  call c_f_pointer(cp, p, [-1])'//new_line('a')// &
            '  stop p(1)'//new_line('a')// &
            'end program main'

        test_negative_extent_rejected = expect_error_contains( &
            source, 'extent', '/tmp/ffc_session_c_pointer_negative_extent')
    end function test_negative_extent_rejected

end program test_session_c_pointer_compiler
