program test_session_if_merge_derived_compiler
    ! Regression test for IF-merge across derived-type component
    ! assignments.  Until fortfront 0f9d4a9e (May 2026) the parser
    ! dropped "p%x = 7" inside if bodies, leaving the if_node with empty
    ! then_body_indices / else_body_indices; ffc then lowered the if as
    ! a no-op and returned a silently-wrong runtime value.  With the
    ! upstream parser fix the body assignments survive, the merge runs
    ! on shared derived storage (alloca), and stop reads the correct
    ! component.
    use ffc_test_support, only: expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== derived-type IF merge compiler test ==='

    all_passed = .true.
    if (.not. test_then_branch_writes_component()) all_passed = .false.
    if (.not. test_else_branch_writes_component()) all_passed = .false.
    if (.not. all_passed) stop 1
    print *, 'PASS: derived-type IF merge lowers through direct LIRIC'

contains

    logical function test_then_branch_writes_component()
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

        test_then_branch_writes_component = expect_exit_status( &
            source, 7, '/tmp/ffc_session_if_derived_then_test')
    end function test_then_branch_writes_component

    logical function test_else_branch_writes_component()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: point_t'//new_line('a')// &
            '    integer :: x'//new_line('a')// &
            '    integer :: y'//new_line('a')// &
            '  end type point_t'//new_line('a')// &
            '  type(point_t) :: p'//new_line('a')// &
            '  integer :: flag'//new_line('a')// &
            '  flag = 0'//new_line('a')// &
            '  if (flag == 1) then'//new_line('a')// &
            '    p%x = 7'//new_line('a')// &
            '  else'//new_line('a')// &
            '    p%x = 9'//new_line('a')// &
            '  end if'//new_line('a')// &
            '  stop p%x'//new_line('a')// &
            'end program main'

        test_else_branch_writes_component = expect_exit_status( &
            source, 9, '/tmp/ffc_session_if_derived_else_test')
    end function test_else_branch_writes_component

end program test_session_if_merge_derived_compiler
