program test_counted_do_compiler
    use ffc_test_support, only: expect_exit_status
    implicit none

    print *, '=== counted do compiler test ==='

    if (.not. expect_exit_status( &
        'program main'//new_line('a')// &
        '  integer :: i'//new_line('a')// &
        '  integer :: n'//new_line('a')// &
        '  integer :: total'//new_line('a')// &
        '  n = 1 + 2'//new_line('a')// &
        '  total = 0'//new_line('a')// &
        '  do i = 1, n'//new_line('a')// &
        '    total = total + i'//new_line('a')// &
        '  end do'//new_line('a')// &
        '  stop total'//new_line('a')// &
        'end program main', 6, &
        '/tmp/ffc_counted_do_test')) stop 1

    if (.not. test_runtime_positive_step()) stop 1
    if (.not. test_runtime_negative_step()) stop 1

    print *, 'PASS: runtime counted DO loop lowers through direct LIRIC session'

contains

    logical function test_runtime_positive_step()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  integer :: s'//new_line('a')// &
            '  integer :: total'//new_line('a')// &
            '  s = 2'//new_line('a')// &
            '  total = 0'//new_line('a')// &
            '  do i = 1, 10, s'//new_line('a')// &
            '    total = total + i'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  stop total'//new_line('a')// &
            'end program main'

        ! 1 + 3 + 5 + 7 + 9 = 25
        test_runtime_positive_step = expect_exit_status( &
            source, 25, &
            '/tmp/ffc_counted_do_runtime_pos_test')
    end function test_runtime_positive_step

    logical function test_runtime_negative_step()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  integer :: s'//new_line('a')// &
            '  integer :: total'//new_line('a')// &
            '  s = -1'//new_line('a')// &
            '  total = 0'//new_line('a')// &
            '  do i = 5, 1, s'//new_line('a')// &
            '    total = total + i'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  stop total'//new_line('a')// &
            'end program main'

        ! 5 + 4 + 3 + 2 + 1 = 15
        test_runtime_negative_step = expect_exit_status( &
            source, 15, &
            '/tmp/ffc_counted_do_runtime_neg_test')
    end function test_runtime_negative_step

end program test_counted_do_compiler
