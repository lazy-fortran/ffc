program test_session_open_status_variable_compiler
    ! A dynamic STATUS= value must be evaluated at runtime. In particular,
    ! NEWUNIT with STATUS='scratch' is valid even when the compiler cannot
    ! know the value at parse time (#628).
    use ffc_test_support, only: expect_exit_status
    implicit none

    character(len=*), parameter :: source = &
        'program main'//new_line('a')// &
        '  character(len=8) :: mode'//new_line('a')// &
        '  integer :: u, ios'//new_line('a')// &
        '  mode = ''scratch'''//new_line('a')// &
        '  open(newunit=u, status=mode, iostat=ios)'//new_line('a')// &
        '  if (ios /= 0) error stop 1'//new_line('a')// &
        '  write(u,*) 42'//new_line('a')// &
        '  close(u)'//new_line('a')// &
        'end program main'

    print *, '=== direct session OPEN(STATUS=variable) compiler test ==='
    if (.not. expect_exit_status(source, 0, &
                                 '/tmp/ffc_628_status_variable')) stop 1
    print *, 'PASS: dynamic STATUS= values reach the OPEN runtime'
end program test_session_open_status_variable_compiler
