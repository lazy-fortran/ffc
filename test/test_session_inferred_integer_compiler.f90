program test_session_inferred_integer_compiler
    use ffc_test_support, only: expect_exit_status
    implicit none

    print *, '=== direct session inferred integer compiler test ==='

    ! Variable with no explicit declaration, but assigned an integer literal.
    ! FortFront infers integer type; ffc seeds the symbol from inferred_type.
    if (.not. expect_exit_status( &
        'program main'//new_line('a')// &
        'x = 42'//new_line('a')// &
        'stop x'//new_line('a')// &
        'end program main', 42, &
        '/tmp/ffc_session_inferred_int')) stop 1

    ! Inferred integer with arithmetic
    if (.not. expect_exit_status( &
        'program main'//new_line('a')// &
        'a = 10'//new_line('a')// &
        'b = 20'//new_line('a')// &
        'c = a + b'//new_line('a')// &
        'stop c'//new_line('a')// &
        'end program main', 30, &
        '/tmp/ffc_session_inferred_int_arith')) stop 1

    print *, 'PASS: inferred integer variables lower through direct LIRIC session'
end program test_session_inferred_integer_compiler
