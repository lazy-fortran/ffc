program test_session_pointer_array_rank2
    use ffc_test_support, only: expect_error_contains, expect_exit_status, expect_output
    implicit none

    print *, '=== direct session rank-2/complex pointer/target array compiler test ==='

    ! p => t aliases a rank-2 target array's storage: a write through
    ! p(u, v) mutates t(u, v).
    if (.not. expect_exit_status( &
        'program main'//new_line('a')// &
        'integer, target :: t(2, 3)'//new_line('a')// &
        'integer, pointer :: p(:, :)'//new_line('a')// &
        'p => t'//new_line('a')// &
        'p(2, 3) = 99'//new_line('a')// &
        'stop t(2, 3)'//new_line('a')// &
        'end program main', 99, &
        '/tmp/ffc_session_pointer_array_rank2_alias')) stop 1

    ! lbound/ubound on a rank-2 real pointer array report the aliased target's
    ! per-dimension bounds.
    if (.not. expect_exit_status( &
        'program main'//new_line('a')// &
        'real, target :: r(2, 3)'//new_line('a')// &
        'real, pointer :: q(:, :)'//new_line('a')// &
        'q => r'//new_line('a')// &
        'if (lbound(q, 1) /= 1 .or. ubound(q, 1) /= 2) stop 1'//new_line('a')// &
        'if (lbound(q, 2) /= 1 .or. ubound(q, 2) /= 3) stop 2'//new_line('a')// &
        'stop 0'//new_line('a')// &
        'end program main', 0, &
        '/tmp/ffc_session_pointer_array_rank2_bounds')) stop 2

    ! A contiguous rank-2 section keeps the complete first dimension.  The
    ! pointer gets the section shape and writes through it reach the target in
    ! column-major order.
    if (.not. expect_exit_status( &
        'program main'//new_line('a')// &
        'integer, target :: t(2, 4)'//new_line('a')// &
        'integer, pointer :: p(:, :)'//new_line('a')// &
        't = reshape([11, 12, 21, 22, 31, 32, 41, 42], [2, 4])'//new_line('a')// &
        'p => t(:, 2:3)'//new_line('a')// &
        'if (size(p) /= 4) stop 1'//new_line('a')// &
        'if (size(p, 1) /= 2 .or. size(p, 2) /= 2) stop 2'//new_line('a')// &
        'if (lbound(p, 1) /= 1 .or. lbound(p, 2) /= 1) stop 3'//new_line('a')// &
        'if (ubound(p, 1) /= 2 .or. ubound(p, 2) /= 2) stop 4'//new_line('a')// &
        'if (p(1, 1) /= 21 .or. p(2, 2) /= 32) stop 5'//new_line('a')// &
        'p(2, 1) = 99'//new_line('a')// &
        'if (t(2, 2) /= 99) stop 6'//new_line('a')// &
        'stop 0'//new_line('a')// &
        'end program main', 0, &
        '/tmp/ffc_session_pointer_array_rank2_section')) stop 3

    ! A rank-2 section with a non-unit stride is not contiguous and must remain
    ! a diagnosed unsupported association rather than an incorrect descriptor.
    if (.not. expect_error_contains( &
        'program main'//new_line('a')// &
        'integer, target :: t(2, 4)'//new_line('a')// &
        'integer, pointer :: p(:, :)'//new_line('a')// &
        'p => t(:, 1:4:2)'//new_line('a')// &
        'end program main', 'rank-2 pointer sections must be contiguous', &
        '/tmp/ffc_session_pointer_array_rank2_noncontiguous')) stop 4

    ! Rank-3 whole-array association adopts all three target dimensions.
    if (.not. expect_exit_status( &
        'program main'//new_line('a')// &
        'integer, target :: t(2, 3, 4)'//new_line('a')// &
        'integer, pointer :: p(:, :, :)'//new_line('a')// &
        'p => t'//new_line('a')// &
        'if (size(p) /= 24) stop 1'//new_line('a')// &
        'if (ubound(p, 1) /= 2 .or. ubound(p, 2) /= 3 .or. '// &
            'ubound(p, 3) /= 4) stop 2'//new_line('a')// &
        'p(2, 3, 4) = 77'//new_line('a')// &
        'stop t(2, 3, 4)'//new_line('a')// &
        'end program main', 77, &
        '/tmp/ffc_session_pointer_array_rank3_alias')) stop 5

    ! A contiguous rank-3 section may shorten a trailing dimension while
    ! retaining all dimensions in the descriptor view.
    if (.not. expect_exit_status( &
        'program main'//new_line('a')// &
        'integer, target :: t(2, 3, 4)'//new_line('a')// &
        'integer, pointer :: p(:, :, :)'//new_line('a')// &
        't = reshape([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, '// &
            '13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24], '// &
            '[2, 3, 4])'//new_line('a')// &
        'p => t(:, :, 2:3)'//new_line('a')// &
        'if (size(p) /= 12) stop 1'//new_line('a')// &
        'if (size(p, 1) /= 2 .or. size(p, 2) /= 3 .or. '// &
            'size(p, 3) /= 2) stop 2'//new_line('a')// &
        'if (p(2, 3, 1) /= 12) stop 3'//new_line('a')// &
        'p(1, 1, 2) = 99'//new_line('a')// &
        'stop t(1, 1, 3)'//new_line('a')// &
        'end program main', 99, &
        '/tmp/ffc_session_pointer_array_rank3_section')) stop 6

    ! A rank-3 section with a non-unit stride remains unsupported.
    if (.not. expect_error_contains( &
        'program main'//new_line('a')// &
        'integer, target :: t(2, 3, 4)'//new_line('a')// &
        'integer, pointer :: p(:, :, :)'//new_line('a')// &
        'p => t(:, :, 1:4:2)'//new_line('a')// &
        'end program main', 'rank-3 pointer sections must be contiguous', &
        '/tmp/ffc_session_pointer_array_rank3_noncontiguous')) stop 7

    ! A rank-1 complex(4) target array: p => t aliases the re/im storage, and
    ! element writes through the pointer flow back to the target.
    if (.not. expect_output( &
        'program main'//new_line('a')// &
        'complex, target :: t(2)'//new_line('a')// &
        'complex, pointer :: p(:)'//new_line('a')// &
        't(1) = (1.0, 2.0)'//new_line('a')// &
        't(2) = (3.0, 4.0)'//new_line('a')// &
        'p => t'//new_line('a')// &
        'p(2) = (9.0, 8.0)'//new_line('a')// &
        'print *, t(1)'//new_line('a')// &
        'print *, t(2)'//new_line('a')// &
        'end program main', &
        '             (1.00000000,2.00000000)'//new_line('a')// &
        '             (9.00000000,8.00000000)'//new_line('a'), &
        '/tmp/ffc_session_pointer_array_complex')) stop 8

    print *, 'PASS: rank-2/rank-3 and complex pointer/target array, => , element access'
end program test_session_pointer_array_rank2
