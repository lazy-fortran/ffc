program test_session_inferred_logical_compiler
    use ffc_test_support, only: expect_exit_status
    use fortfront_compiler, only: INPUT_MODE_LAZY
    implicit none

    print *, '=== direct session inferred logical compiler test ==='

    ! Variable with no explicit declaration, assigned a logical literal.
    ! FortFront infers logical type; ffc seeds the symbol from inferred_type.
    if (.not. expect_exit_status( &
        'program main'//new_line('a')// &
        'flag = .true.'//new_line('a')// &
        'if (flag) stop 1'//new_line('a')// &
        'stop 0'//new_line('a')// &
        'end program main', 1, &
        '/tmp/ffc_session_inferred_logical')) stop 1

    ! Lazy Fortran accepts bare true/false literals. They must enter the same
    ! typed logical lowering path as their standard .true./.false. spellings.
    if (.not. expect_exit_status( &
        'program main'//new_line('a')// &
        'flag = true'//new_line('a')// &
        'if (.not. flag) stop 1'//new_line('a')// &
        'stop 0'//new_line('a')// &
        'end program main', 0, '/tmp/ffc_session_lazy_true', INPUT_MODE_LAZY)) &
        stop 1
    if (.not. expect_exit_status( &
        'program main'//new_line('a')// &
        'flag = false'//new_line('a')// &
        'if (flag) stop 1'//new_line('a')// &
        'stop 0'//new_line('a')// &
        'end program main', 0, '/tmp/ffc_session_lazy_false', INPUT_MODE_LAZY)) &
        stop 1

    print *, 'PASS: inferred logical variables lower through direct LIRIC session'
end program test_session_inferred_logical_compiler
