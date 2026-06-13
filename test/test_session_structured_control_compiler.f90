program test_session_structured_control_compiler
    use ffc_test_support, only: expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== structured control-flow construct compiler test ==='

    all_passed = .true.
    if (.not. test_block_local_declaration()) all_passed = .false.
    if (.not. test_block_outer_visible()) all_passed = .false.
    if (.not. test_do_concurrent_array_fill()) all_passed = .false.
    if (.not. test_where_stmt_true()) all_passed = .false.
    if (.not. test_where_stmt_false()) all_passed = .false.
    if (.not. test_where_construct_true_branch()) all_passed = .false.
    if (.not. test_where_construct_elsewhere_branch()) all_passed = .false.
    if (.not. test_forall_array_fill()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: BLOCK/DO CONCURRENT/WHERE/FORALL lower through direct LIRIC'

contains

    logical function test_block_local_declaration()
        ! B8b: BLOCK with a local declaration.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: x'//new_line('a')// &
            '  x = 5'//new_line('a')// &
            '  block'//new_line('a')// &
            '    integer :: y'//new_line('a')// &
            '    y = x + 3'//new_line('a')// &
            '    x = y * 2'//new_line('a')// &
            '  end block'//new_line('a')// &
            '  stop x'//new_line('a')// &
            'end program main'
        test_block_local_declaration = expect_exit_status( &
            source, 16, '/tmp/ffc_block_local_test')
    end function test_block_local_declaration

    logical function test_block_outer_visible()
        ! BLOCK local symbol does not leak to outer scope.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: r'//new_line('a')// &
            '  r = 7'//new_line('a')// &
            '  block'//new_line('a')// &
            '    integer :: tmp'//new_line('a')// &
            '    tmp = r + 1'//new_line('a')// &
            '    r = tmp'//new_line('a')// &
            '  end block'//new_line('a')// &
            '  stop r'//new_line('a')// &
            'end program main'
        test_block_outer_visible = expect_exit_status( &
            source, 8, '/tmp/ffc_block_outer_test')
    end function test_block_outer_visible

    logical function test_do_concurrent_array_fill()
        ! B8d: DO CONCURRENT over a 1-D array.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(4), i, s'//new_line('a')// &
            '  do concurrent (i = 1:4)'//new_line('a')// &
            '    a(i) = i * i'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  s = a(1) + a(2) + a(3) + a(4)'//new_line('a')// &
            '  stop s'//new_line('a')// &
            'end program main'
        ! 1^2 + 2^2 + 3^2 + 4^2 = 1 + 4 + 9 + 16 = 30
        test_do_concurrent_array_fill = expect_exit_status( &
            source, 30, '/tmp/ffc_do_concurrent_test')
    end function test_do_concurrent_array_fill

    logical function test_where_stmt_true()
        ! B8e: WHERE single-statement, mask true.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: x, flag'//new_line('a')// &
            '  x = 5'//new_line('a')// &
            '  flag = 1'//new_line('a')// &
            '  where (flag > 0) x = 99'//new_line('a')// &
            '  stop x'//new_line('a')// &
            'end program main'
        test_where_stmt_true = expect_exit_status( &
            source, 99, '/tmp/ffc_where_stmt_true_test')
    end function test_where_stmt_true

    logical function test_where_stmt_false()
        ! WHERE single-statement, mask false: assignment skipped.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: x, flag'//new_line('a')// &
            '  x = 5'//new_line('a')// &
            '  flag = 0'//new_line('a')// &
            '  where (flag > 0) x = 99'//new_line('a')// &
            '  stop x'//new_line('a')// &
            'end program main'
        test_where_stmt_false = expect_exit_status( &
            source, 5, '/tmp/ffc_where_stmt_false_test')
    end function test_where_stmt_false

    logical function test_where_construct_true_branch()
        ! WHERE construct: true branch executes when mask holds.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: x, y, flag'//new_line('a')// &
            '  x = 5'//new_line('a')// &
            '  y = 0'//new_line('a')// &
            '  flag = 1'//new_line('a')// &
            '  where (flag > 0)'//new_line('a')// &
            '    x = 10'//new_line('a')// &
            '    y = 20'//new_line('a')// &
            '  elsewhere'//new_line('a')// &
            '    x = 99'//new_line('a')// &
            '    y = 99'//new_line('a')// &
            '  end where'//new_line('a')// &
            '  stop x + y'//new_line('a')// &
            'end program main'
        test_where_construct_true_branch = expect_exit_status( &
            source, 30, '/tmp/ffc_where_true_test')
    end function test_where_construct_true_branch

    logical function test_where_construct_elsewhere_branch()
        ! WHERE construct: ELSEWHERE executes when mask is false.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: x, flag'//new_line('a')// &
            '  x = 5'//new_line('a')// &
            '  flag = 0'//new_line('a')// &
            '  where (flag > 0)'//new_line('a')// &
            '    x = 99'//new_line('a')// &
            '  elsewhere'//new_line('a')// &
            '    x = 42'//new_line('a')// &
            '  end where'//new_line('a')// &
            '  stop x'//new_line('a')// &
            'end program main'
        test_where_construct_elsewhere_branch = expect_exit_status( &
            source, 42, '/tmp/ffc_where_else_test')
    end function test_where_construct_elsewhere_branch

    logical function test_forall_array_fill()
        ! B8f: FORALL masked assignment over a 1-D array.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(4), i, s'//new_line('a')// &
            '  forall (i = 1:4) a(i) = i * 3'//new_line('a')// &
            '  s = a(1) + a(2) + a(3) + a(4)'//new_line('a')// &
            '  stop s'//new_line('a')// &
            'end program main'
        ! 3 + 6 + 9 + 12 = 30
        test_forall_array_fill = expect_exit_status( &
            source, 30, '/tmp/ffc_forall_test')
    end function test_forall_array_fill

end program test_session_structured_control_compiler
