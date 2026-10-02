! fo: dispatcher
module ffc_case_test_session_external_dummy_callback_compiler
    implicit none
    private
    public :: case_test_session_external_dummy_callback_compiler
    interface
        subroutine case_test_session_external_dummy_callback_compiler()
        end subroutine case_test_session_external_dummy_callback_compiler
    end interface
end module ffc_case_test_session_external_dummy_callback_compiler

subroutine case_test_session_external_dummy_callback_compiler()
    use ffc_test_support, only: expect_output_matches_gfortran
    implicit none
    save

    print *, '=== EXTERNAL-statement dummy procedure callback ==='

    ! An EXTERNAL statement naming a dummy declares a procedure dummy: the
    ! caller passes a callable address and every call through the dummy is
    ! indirect. Before this boundary, ffc emitted a direct call to the dummy
    ! name and the link died on `undefined reference to f`. gfortran is the
    ! independent oracle for status and stdout.
    if (.not. expect_output_matches_gfortran( &
        'program p'//new_line('a')// &
        '    implicit none'//new_line('a')// &
        '    external mysub, applier'//new_line('a')// &
        '    call applier(mysub)'//new_line('a')// &
        'end program p'//new_line('a')// &
        'subroutine applier(f)'//new_line('a')// &
        '    implicit none'//new_line('a')// &
        '    external :: f'//new_line('a')// &
        '    call f(7)'//new_line('a')// &
        'end subroutine applier'//new_line('a')// &
        'subroutine mysub(n)'//new_line('a')// &
        '    implicit none'//new_line('a')// &
        '    integer :: n'//new_line('a')// &
        '    print *, n * 3'//new_line('a')// &
        'end subroutine mysub', &
        'external_dummy_callback')) stop 1

    print *, 'PASS: EXTERNAL dummy callback matches gfortran'
end subroutine case_test_session_external_dummy_callback_compiler
