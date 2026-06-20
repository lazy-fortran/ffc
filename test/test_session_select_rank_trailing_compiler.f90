program test_session_select_rank_trailing_compiler
    ! Issue #2811 / #273: an assumed-rank dummy arr(..) drives a select rank
    ! whose chosen arm sets a scalar a trailing print reads back. ffc resolves
    ! the dummy's rank from the caller's actual, so the construct dispatches at
    ! compile time and the trailing statement runs as a sibling.
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed
    character(len=*), parameter :: nl = new_line('a')

    print *, '=== select rank trailing-statement compiler test ==='

    all_passed = .true.
    if (.not. test_rank_one_actual_then_print()) all_passed = .false.
    if (.not. test_rank_two_actual_then_print()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: select rank dispatches on the actual rank into the print'

contains

    logical function test_rank_one_actual_then_print()
        ! A rank-1 actual selects the rank (1) arm (i = 1); the trailing print
        ! reads it.
        character(len=*), parameter :: source = &
            'program main'//nl// &
            '  integer :: a1(3)'//nl// &
            '  a1 = 0'//nl// &
            '  call probe(a1)'//nl// &
            'contains'//nl// &
            '  subroutine probe(arr)'//nl// &
            '    integer :: arr(..)'//nl// &
            '    integer :: i'//nl// &
            '    i = 0'//nl// &
            '    select rank (arr)'//nl// &
            '    rank (1)'//nl// &
            '      i = 1'//nl// &
            '    rank default'//nl// &
            '      i = 2'//nl// &
            '    end select'//nl// &
            '    print *, i'//nl// &
            '  end subroutine probe'//nl// &
            'end program main'

        test_rank_one_actual_then_print = expect_output( &
            source, '           1'//nl, '/tmp/ffc_select_rank_trailing_one')
    end function test_rank_one_actual_then_print

    logical function test_rank_two_actual_then_print()
        ! A rank-2 actual matches no rank (1) arm, so the rank default arm sets
        ! i = 2; the trailing print reads it.
        character(len=*), parameter :: source = &
            'program main'//nl// &
            '  integer :: a2(2, 2)'//nl// &
            '  a2 = 0'//nl// &
            '  call probe(a2)'//nl// &
            'contains'//nl// &
            '  subroutine probe(arr)'//nl// &
            '    integer :: arr(..)'//nl// &
            '    integer :: i'//nl// &
            '    i = 0'//nl// &
            '    select rank (arr)'//nl// &
            '    rank (1)'//nl// &
            '      i = 1'//nl// &
            '    rank default'//nl// &
            '      i = 2'//nl// &
            '    end select'//nl// &
            '    print *, i'//nl// &
            '  end subroutine probe'//nl// &
            'end program main'

        test_rank_two_actual_then_print = expect_output( &
            source, '           2'//nl, '/tmp/ffc_select_rank_trailing_two')
    end function test_rank_two_actual_then_print

end program test_session_select_rank_trailing_compiler
