program test_session_reject_bit_intrinsic_range_compiler
    use ffc_test_support, only: expect_error_contains, expect_no_error
    implicit none

    logical :: all_passed

    print *, '=== bit-intrinsic nonnegative-argument rejection compiler test ==='

    all_passed = .true.
    if (.not. test_btest_negative_pos_rejected()) all_passed = .false.
    if (.not. test_ibits_negative_len_rejected()) all_passed = .false.
    if (.not. test_mvbits_negative_frompos_rejected()) all_passed = .false.
    if (.not. test_valid_bit_intrinsics_accepted()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: negative bit-position/length constants rejected, ' // &
        'nonnegative bit-intrinsic calls still accepted'

contains

    logical function test_btest_negative_pos_rejected()
        ! BTEST requires POS >= 0 (gfortran: "must be nonnegative").
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: i = 42'//new_line('a')// &
            '  logical :: l'//new_line('a')// &
            '  l = btest(i, -1)'//new_line('a')// &
            '  print *, l'//new_line('a')// &
            'end program main'

        test_btest_negative_pos_rejected = expect_error_contains( &
            source, 'must be nonnegative', '/tmp/ffc_session_btest_neg_reject')
    end function test_btest_negative_pos_rejected

    logical function test_ibits_negative_len_rejected()
        ! IBITS requires POS >= 0 and LEN >= 0.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: i = 42, j'//new_line('a')// &
            '  j = ibits(i, 1, -1)'//new_line('a')// &
            '  print *, j'//new_line('a')// &
            'end program main'

        test_ibits_negative_len_rejected = expect_error_contains( &
            source, 'must be nonnegative', '/tmp/ffc_session_ibits_neg_reject')
    end function test_ibits_negative_len_rejected

    logical function test_mvbits_negative_frompos_rejected()
        ! MVBITS requires FROMPOS >= 0, LEN >= 0, and TOPOS >= 0.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: n = 42'//new_line('a')// &
            '  call mvbits(n, -1, 2, n, 3)'//new_line('a')// &
            '  print *, n'//new_line('a')// &
            'end program main'

        test_mvbits_negative_frompos_rejected = expect_error_contains( &
            source, 'must be nonnegative', '/tmp/ffc_session_mvbits_neg_reject')
    end function test_mvbits_negative_frompos_rejected

    logical function test_valid_bit_intrinsics_accepted()
        ! Nonnegative bit positions and lengths are valid; the check must not
        ! reject them. Lowering must succeed without a nonnegative diagnostic.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: i = 42, j'//new_line('a')// &
            '  logical :: l'//new_line('a')// &
            '  l = btest(i, 3)'//new_line('a')// &
            '  j = ibits(i, 1, 2)'//new_line('a')// &
            '  print *, l, j'//new_line('a')// &
            'end program main'

        test_valid_bit_intrinsics_accepted = expect_no_error( &
            source, '/tmp/ffc_session_bit_intrinsic_accept')
    end function test_valid_bit_intrinsics_accepted

end program test_session_reject_bit_intrinsic_range_compiler
