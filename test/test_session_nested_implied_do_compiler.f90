program test_session_nested_implied_do_compiler
    ! Nested implied-do array constructors, arr = [((expr, j=1,m), i=1,n)],
    ! fold at compile time into a flat array. The inner loop fills consecutive
    ! elements with a shared running slot. Outputs match gfortran list-directed
    ! formatting.
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session nested implied-do compiler test ==='

    all_passed = .true.
    if (.not. test_two_level()) all_passed = .false.
    if (.not. test_three_by_four()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: nested implied-do constructors lower through direct LIRIC'

contains

    logical function test_two_level()
        ! [((i+j, j=1,2), i=1,3)] -> i=1: 2,3  i=2: 3,4  i=3: 4,5
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: i, j'//new_line('a')// &
            '  integer :: a(6)'//new_line('a')// &
            '  a = [((i + j, j = 1, 2), i = 1, 3)]'//new_line('a')// &
            '  print *, a'//new_line('a')// &
            'end program main'

        test_two_level = expect_output( &
            source, &
            '           2           3           3           4           4'// &
            '           5'//new_line('a'), '/tmp/ffc_nested_id_2lvl')
    end function test_two_level

    logical function test_three_by_four()
        ! [(((i - 1) * 4 + j, j = 1, 4), i = 1, 3)] -> 1..12 in row-major order
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: i, j'//new_line('a')// &
            '  integer, dimension(12) :: matrix_flat'//new_line('a')// &
            '  matrix_flat = [(((i - 1) * 4 + j, j = 1, 4), i = 1, 3)]'// &
            new_line('a')// &
            '  print *, matrix_flat'//new_line('a')// &
            'end program main'

        test_three_by_four = expect_output( &
            source, &
            '           1           2           3           4           5'// &
            '           6           7           8           9          10'// &
            '          11          12'//new_line('a'), &
            '/tmp/ffc_nested_id_3x4')
    end function test_three_by_four

end program test_session_nested_implied_do_compiler
