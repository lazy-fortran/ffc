program test_session_reject_round2_compiler
    use ffc_test_support, only: expect_error_contains, expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== round-2 negative-accept rejection compiler test ==='

    all_passed = .true.
    if (.not. test_intrinsic_external_conflict_rejected()) all_passed = .false.
    if (.not. test_function_result_save_rejected()) all_passed = .false.
    if (.not. test_duplicate_contained_procedure_rejected()) all_passed = .false.
    if (.not. test_bare_external_still_accepted()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: round-2 negative-accept checks reject the invalid forms '// &
        'and still accept the valid bare EXTERNAL form'

contains

    logical function test_intrinsic_external_conflict_rejected()
        ! A name shall not appear in both an EXTERNAL and an INTRINSIC
        ! statement in the same scoping unit (gfortran.dg intrinsic_external_1).
        character(len=*), parameter :: source = &
            'program u'//new_line('a')// &
            '  intrinsic :: nint'//new_line('a')// &
            '  external :: nint'//new_line('a')// &
            'end program u'

        test_intrinsic_external_conflict_rejected = expect_error_contains( &
            source, 'EXTERNAL attribute conflicts with INTRINSIC attribute', &
            '/tmp/ffc_session_intrinsic_external_reject')
    end function test_intrinsic_external_conflict_rejected

    logical function test_function_result_save_rejected()
        ! A function RESULT variable never carries the SAVE attribute
        ! (gfortran.dg save_result, PR20856).
        character(len=*), parameter :: source = &
            'function x() result(y)'//new_line('a')// &
            '  real, save :: y'//new_line('a')// &
            '  y = 1'//new_line('a')// &
            'end function x'

        test_function_result_save_rejected = expect_error_contains( &
            source, 'RESULT attribute conflicts with SAVE attribute', &
            '/tmp/ffc_session_result_save_reject')
    end function test_function_result_save_rejected

    logical function test_duplicate_contained_procedure_rejected()
        ! Two contained procedures sharing one name in the same CONTAINS
        ! section are not distinguishable at the call site (gfortran.dg
        ! import13).
        character(len=*), parameter :: source = &
            'program foo'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  call bah()'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine bah()'//new_line('a')// &
            '    print *, 1'//new_line('a')// &
            '  end subroutine bah'//new_line('a')// &
            '  subroutine bah()'//new_line('a')// &
            '    print *, 2'//new_line('a')// &
            '  end subroutine bah'//new_line('a')// &
            'end program foo'

        test_duplicate_contained_procedure_rejected = expect_error_contains( &
            source, 'already defined', &
            '/tmp/ffc_session_duplicate_procedure_reject')
    end function test_duplicate_contained_procedure_rejected

    logical function test_bare_external_still_accepted()
        ! The bare EXTERNAL skip stays valid on its own: an EXTERNAL name
        ! never confused with an INTRINSIC name must still compile and run
        ! (guards against the new intrinsic/external check over-triggering).
        character(len=*), parameter :: source = &
            'program p'//new_line('a')// &
            '    implicit none'//new_line('a')// &
            '    integer :: n'//new_line('a')// &
            '    external :: bar'//new_line('a')// &
            '    n = 41'//new_line('a')// &
            '    call bar(n)'//new_line('a')// &
            '    if (n /= 42) error stop'//new_line('a')// &
            'end program'//new_line('a')// &
            'subroutine bar(n)'//new_line('a')// &
            '    implicit none'//new_line('a')// &
            '    integer, intent(inout) :: n'//new_line('a')// &
            '    n = n + 1'//new_line('a')// &
            'end subroutine'

        test_bare_external_still_accepted = expect_exit_status( &
            source, 0, '/tmp/ffc_session_bare_external_accept')
    end function test_bare_external_still_accepted

end program test_session_reject_round2_compiler
