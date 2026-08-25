program test_session_implicit_dimension_data_compiler
    use ffc_test_support, only: expect_output
    implicit none

    character(len=*), parameter :: source = &
        'subroutine gamo'//new_line('a')// &
        '  implicit double precision(a-h,o-z)'//new_line('a')// &
        '  dimension g(2)'//new_line('a')// &
        '  data g /1.5d0, 2.5d0/'//new_line('a')// &
        '  if (abs(g(1)-1.5d0) > 1.0d-12) stop 2'//new_line('a')// &
        '  if (abs(g(2)-2.5d0) > 1.0d-12) stop 3'//new_line('a')// &
        "  print *, 'PASS'"//new_line('a')// &
        'end subroutine gamo'//new_line('a')// &
        'program main'//new_line('a')// &
        '  call gamo'//new_line('a')// &
        'end program main'

    if (.not. expect_output(source, ' PASS'//new_line('a'), &
                            '/tmp/ffc_implicit_dimension_data')) stop 1
end program test_session_implicit_dimension_data_compiler
