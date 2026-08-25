program test_session_logical_reduction_expression_compiler
    use ffc_test_support, only: expect_output
    implicit none

    character(len=*), parameter :: source = &
        'program main'//new_line('a')// &
        '  real :: x(10, 10)'//new_line('a')// &
        '  x = 6.0'//new_line('a')// &
        '  if (count(abs(x - 6.0) > 1e-6) /= 0) error stop 1'//new_line('a')// &
        '  if (any(abs(x - 6.0) > 1e-6)) error stop 2'//new_line('a')// &
        '  print *, "ok"'//new_line('a')// &
        'end program main'

    if (.not. expect_output(source, ' ok'//new_line('a'), &
            '/tmp/ffc_session_logical_reduction_expression_test')) stop 1
end program test_session_logical_reduction_expression_compiler
