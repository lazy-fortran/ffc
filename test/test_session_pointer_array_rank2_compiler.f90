program test_session_pointer_array_rank2
    use ffc_test_support, only: expect_exit_status, expect_output
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
        '/tmp/ffc_session_pointer_array_complex')) stop 3

    print *, 'PASS: rank-2 and complex pointer/target array, => , element access'
end program test_session_pointer_array_rank2
