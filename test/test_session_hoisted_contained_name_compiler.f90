program test_session_hoisted_contained_name_compiler
    use ffc_test_support, only: expect_output
    implicit none

    print *, '=== hoisted contained procedure name clash ==='

    ! FortFront hoists the last CONTAINS procedure of a sibling out to the
    ! top-level unit list. When it shares a name with a genuine top-level
    ! procedure the flat function table saw two same-named units and rejected
    ! the second with "duplicate contained function". The hoisted duplicate is
    ! now dropped, so the unit compiles and runs.
    if (.not. expect_output( &
        'subroutine outer()'//new_line('a')// &
        'contains'//new_line('a')// &
        '    subroutine first_helper()'//new_line('a')// &
        '    end subroutine'//new_line('a')// &
        '    real function dup_name()'//new_line('a')// &
        '        dup_name = 1.0'//new_line('a')// &
        '    end function'//new_line('a')// &
        'end subroutine'//new_line('a')// &
        'real function dup_name()'//new_line('a')// &
        '    dup_name = 2.0'//new_line('a')// &
        'end function'//new_line('a')// &
        'program p'//new_line('a')// &
        "    print *, 'ok'"//new_line('a')// &
        'end program', ' ok'//new_line('a'), &
        '/tmp/ffc_session_hoisted_contained_name_test')) stop 1

    print *, 'PASS: hoisted contained duplicate name compiles and runs'
end program test_session_hoisted_contained_name_compiler
