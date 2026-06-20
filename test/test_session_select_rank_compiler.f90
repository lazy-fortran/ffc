program test_session_select_rank_compiler
    use ffc_test_support, only: expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== direct session select rank compiler test ==='

    all_passed = .true.
    if (.not. test_select_rank_one_matches()) all_passed = .false.
    if (.not. test_select_rank_two_matches()) all_passed = .false.
    if (.not. test_select_rank_default()) all_passed = .false.
    if (.not. test_select_rank_scalar()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: SELECT RANK lowers through direct LIRIC'

contains

    logical function test_select_rank_one_matches()
        ! A rank-1 selector picks the rank(1) arm; the terminating stop in that
        ! arm reflects the choice.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(5)'//new_line('a')// &
            '  select rank (a)'//new_line('a')// &
            '  rank (1)'//new_line('a')// &
            '    stop 11'//new_line('a')// &
            '  rank (2)'//new_line('a')// &
            '    stop 22'//new_line('a')// &
            '  rank default'//new_line('a')// &
            '    stop 99'//new_line('a')// &
            '  end select'//new_line('a')// &
            'end program main'

        test_select_rank_one_matches = expect_exit_status( &
            source, 11, '/tmp/ffc_session_select_rank_one_test')
    end function test_select_rank_one_matches

    logical function test_select_rank_two_matches()
        ! A rank-2 selector picks the rank(2) arm.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(3, 4)'//new_line('a')// &
            '  select rank (a)'//new_line('a')// &
            '  rank (1)'//new_line('a')// &
            '    stop 11'//new_line('a')// &
            '  rank (2)'//new_line('a')// &
            '    stop 22'//new_line('a')// &
            '  rank default'//new_line('a')// &
            '    stop 99'//new_line('a')// &
            '  end select'//new_line('a')// &
            'end program main'

        test_select_rank_two_matches = expect_exit_status( &
            source, 22, '/tmp/ffc_session_select_rank_two_test')
    end function test_select_rank_two_matches

    logical function test_select_rank_default()
        ! No arm matches a rank-2 selector, so the default arm runs.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(3, 4)'//new_line('a')// &
            '  select rank (a)'//new_line('a')// &
            '  rank (1)'//new_line('a')// &
            '    stop 11'//new_line('a')// &
            '  rank default'//new_line('a')// &
            '    stop 99'//new_line('a')// &
            '  end select'//new_line('a')// &
            'end program main'

        test_select_rank_default = expect_exit_status( &
            source, 99, '/tmp/ffc_session_select_rank_default_test')
    end function test_select_rank_default

    logical function test_select_rank_scalar()
        ! A scalar selector has rank 0 and picks the rank(0) arm.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a'//new_line('a')// &
            '  a = 1'//new_line('a')// &
            '  select rank (a)'//new_line('a')// &
            '  rank (0)'//new_line('a')// &
            '    stop 7'//new_line('a')// &
            '  rank default'//new_line('a')// &
            '    stop 99'//new_line('a')// &
            '  end select'//new_line('a')// &
            'end program main'

        test_select_rank_scalar = expect_exit_status( &
            source, 7, '/tmp/ffc_session_select_rank_scalar_test')
    end function test_select_rank_scalar

end program test_session_select_rank_compiler
