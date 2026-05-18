program test_session_if_empty_body_diagnostic
    ! FortFront's standardiser silently drops some body statements (observed
    ! for derived-type component assignments inside if branches: the
    ! resulting if_node has size-0 then_body_indices and else_body_indices).
    ! ffc used to lower these as no-ops which let the program run and
    ! returned the wrong answer.  This test fences in the new targeted
    ! diagnostic so the regression cannot come back as silently-wrong code.
    use ffc_test_support, only: expect_error_contains
    implicit none

    character(len=*), parameter :: source = &
        'program main'//new_line('a')// &
        '  type :: point_t'//new_line('a')// &
        '    integer :: x'//new_line('a')// &
        '    integer :: y'//new_line('a')// &
        '  end type point_t'//new_line('a')// &
        '  type(point_t) :: p'//new_line('a')// &
        '  integer :: flag'//new_line('a')// &
        '  flag = 1'//new_line('a')// &
        '  if (flag == 1) then'//new_line('a')// &
        '    p%x = 7'//new_line('a')// &
        '  else'//new_line('a')// &
        '    p%x = 9'//new_line('a')// &
        '  end if'//new_line('a')// &
        '  stop p%x'//new_line('a')// &
        'end program main'

    print *, '=== if empty body diagnostic test ==='

    if (.not. expect_error_contains(source, &
        'both branches empty after frontend standardisation', &
        '/tmp/ffc_session_if_empty_body_test')) stop 1

    print *, 'PASS: if with frontend-dropped bodies is rejected loudly'
end program test_session_if_empty_body_diagnostic
