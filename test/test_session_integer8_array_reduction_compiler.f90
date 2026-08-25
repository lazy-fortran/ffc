program test_session_integer8_array_reduction_compiler
    use ffc_test_support, only: expect_exit_status
    implicit none

    character(len=*), parameter :: source = &
        'program main'//new_line('a')// &
        '  integer(8) :: a(4)'//new_line('a')// &
        '  a = [1_8, 4_8, 2_8, 3_8]'//new_line('a')// &
        '  if (maxval(a) /= 4_8) error stop 1'//new_line('a')// &
        '  if (minval(a) /= 1_8) error stop 2'//new_line('a')// &
        'end program main'

    if (.not. expect_exit_status(source, 0, &
        '/tmp/ffc_session_integer8_reduction_test')) stop 1
    print *, 'PASS: integer(8) array maxval/minval lower through direct LIRIC'
end program test_session_integer8_array_reduction_compiler
