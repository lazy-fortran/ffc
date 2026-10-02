! fo: dispatcher
module ffc_case_test_session_inferred_real_compiler
    implicit none
    private
    public :: case_test_session_inferred_real_compiler
    interface
        subroutine case_test_session_inferred_real_compiler()
        end subroutine case_test_session_inferred_real_compiler
    end interface
end module ffc_case_test_session_inferred_real_compiler

subroutine case_test_session_inferred_real_compiler()
    use ffc_test_support, only: expect_exit_status
    implicit none
    save

    print *, '=== direct session inferred real compiler test ==='

    ! Variable with no explicit declaration, assigned a real literal.
    ! FortFront infers real type; ffc seeds the symbol from inferred_type.
    if (.not. expect_exit_status( &
        'program main'//new_line('a')// &
        'x = 3.14'//new_line('a')// &
        'y = 2.0'//new_line('a')// &
        'z = x + y'//new_line('a')// &
        'stop int(z)'//new_line('a')// &
        'end program main', 5, &
        '/tmp/ffc_session_inferred_real')) stop 1

    print *, 'PASS: inferred real variables lower through direct LIRIC session'
end subroutine case_test_session_inferred_real_compiler
