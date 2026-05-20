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

    print *, 'PASS: runtime counted DO loop lowers through direct LIRIC session'
end program test_counted_do_compiler
