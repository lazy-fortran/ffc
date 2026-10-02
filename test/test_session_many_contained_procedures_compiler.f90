! fo: dispatcher
module ffc_case_test_session_many_contained_procedures_compiler
    implicit none
    private
    public :: case_test_session_many_contained_procedures_compiler
    interface
        subroutine case_test_session_many_contained_procedures_compiler()
        end subroutine case_test_session_many_contained_procedures_compiler
    end interface
end module ffc_case_test_session_many_contained_procedures_compiler

subroutine case_test_session_many_contained_procedures_compiler()
    use ffc_test_support, only: expect_exit_status
    implicit none
    save

    print *, '=== direct session many contained procedures compiler test ==='

    ! Regression for #290: a program (or module) with more than eight contained
    ! procedures grows the function-name table while copying it into a nested
    ! procedure context. The old capacity helper re-allocated an already
    ! allocated array and aborted with "Attempting to allocate already
    ! allocated variable 'context'". Ten contained subroutines force the grow.
    if (.not. expect_exit_status( &
        'program main'//new_line('a')// &
        '  implicit none'//new_line('a')// &
        '  call s10()'//new_line('a')// &
        'contains'//new_line('a')// &
        '  subroutine s1()'//new_line('a')//'  end subroutine'//new_line('a')// &
        '  subroutine s2()'//new_line('a')//'  end subroutine'//new_line('a')// &
        '  subroutine s3()'//new_line('a')//'  end subroutine'//new_line('a')// &
        '  subroutine s4()'//new_line('a')//'  end subroutine'//new_line('a')// &
        '  subroutine s5()'//new_line('a')//'  end subroutine'//new_line('a')// &
        '  subroutine s6()'//new_line('a')//'  end subroutine'//new_line('a')// &
        '  subroutine s7()'//new_line('a')//'  end subroutine'//new_line('a')// &
        '  subroutine s8()'//new_line('a')//'  end subroutine'//new_line('a')// &
        '  subroutine s9()'//new_line('a')//'  end subroutine'//new_line('a')// &
        '  subroutine s10()'//new_line('a')// &
        '    call s1()'//new_line('a')// &
        '    stop 42'//new_line('a')// &
        '  end subroutine'//new_line('a')// &
        'end program main', 42, &
        '/tmp/ffc_session_many_contained_procedures_test')) stop 1

    print *, 'PASS: many contained procedures lower without a capacity crash'

end subroutine case_test_session_many_contained_procedures_compiler
