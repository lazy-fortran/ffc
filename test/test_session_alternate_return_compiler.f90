program test_session_alternate_return_compiler
    use ffc_test_support, only: expect_output, expect_error_contains
    implicit none

    print *, '=== direct session alternate-return compiler test ==='

    ! Two alternate-return slots are distinct positional specifiers: RETURN 1
    ! branches to the first caller label, RETURN 2 to the second.
    if (.not. expect_output( &
        'program main'//new_line('a')// &
        '    implicit none'//new_line('a')// &
        '    call s(1, *100, *200)'//new_line('a')// &
        '    goto 900'//new_line('a')// &
        '100 print *, 1'//new_line('a')// &
        '    call s(2, *910, *200)'//new_line('a')// &
        '    goto 900'//new_line('a')// &
        '200 print *, 2'//new_line('a')// &
        '    goto 900'//new_line('a')// &
        '910 continue'//new_line('a')// &
        '900 continue'//new_line('a')// &
        'contains'//new_line('a')// &
        '    subroutine s(k, *, *)'//new_line('a')// &
        '        integer, intent(in) :: k'//new_line('a')// &
        '        if (k == 1) return 1'//new_line('a')// &
        '        return 2'//new_line('a')// &
        '    end subroutine s'//new_line('a')// &
        'end program main', &
        '           1'//new_line('a')//'           2'//new_line('a'), &
        '/tmp/ffc_session_altret_two_slots_test')) stop 1
    print *, 'PASS: two alternate returns branch to distinct labels'

    ! A plain RETURN in a subroutine with alternate-return slots continues at
    ! the statement after the CALL.
    if (.not. expect_output( &
        'program main'//new_line('a')// &
        '    implicit none'//new_line('a')// &
        '    call s(0, *100, *200)'//new_line('a')// &
        '    print *, 7'//new_line('a')// &
        '    goto 900'//new_line('a')// &
        '100 print *, 1'//new_line('a')// &
        '    goto 900'//new_line('a')// &
        '200 print *, 2'//new_line('a')// &
        '900 continue'//new_line('a')// &
        'contains'//new_line('a')// &
        '    subroutine s(k, *, *)'//new_line('a')// &
        '        integer, intent(in) :: k'//new_line('a')// &
        '        if (k == 1) return 1'//new_line('a')// &
        '        if (k == 2) return 2'//new_line('a')// &
        '        return'//new_line('a')// &
        '    end subroutine s'//new_line('a')// &
        'end program main', &
        '           7'//new_line('a'), &
        '/tmp/ffc_session_altret_plain_return_test')) stop 1
    print *, 'PASS: plain return falls through to the statement after the call'

    ! RETURN 3 with only two alternate-return slots is a diagnostic.
    if (.not. expect_error_contains( &
        'program main'//new_line('a')// &
        '    implicit none'//new_line('a')// &
        '    call s(1, *100, *200)'//new_line('a')// &
        '    goto 900'//new_line('a')// &
        '100 print *, 1'//new_line('a')// &
        '    goto 900'//new_line('a')// &
        '200 print *, 2'//new_line('a')// &
        '900 continue'//new_line('a')// &
        'contains'//new_line('a')// &
        '    subroutine s(k, *, *)'//new_line('a')// &
        '        integer, intent(in) :: k'//new_line('a')// &
        '        return 3'//new_line('a')// &
        '    end subroutine s'//new_line('a')// &
        'end program main', &
        'alternate return', &
        '/tmp/ffc_session_altret_out_of_range_test')) stop 1
    print *, 'PASS: out-of-range alternate return is rejected'

    print *, 'PASS: alternate returns lower through direct LIRIC session'
end program test_session_alternate_return_compiler
