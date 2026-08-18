program test_session_open_status_variable_compiler
    ! A dynamic STATUS= value must be evaluated at runtime. FILE= makes the
    ! runtime compare the mixed-case value rather than taking the scratch
    ! temporary-file path; REPLACE must also truncate an existing file.
    use ffc_test_support, only: expect_exit_status
    implicit none

    character(len=*), parameter :: source = &
        'program main'//new_line('a')// &
        '  character(len=8) :: mode'//new_line('a')// &
        '  integer :: u, ios, bytes'//new_line('a')// &
        '  open(unit=41, file=''/tmp/ffc_628_status_variable.dat'', '// &
        'status=''replace'', iostat=ios)'//new_line('a')// &
        '  if (ios /= 0) error stop 1'//new_line('a')// &
        '  write(41,*) 42'//new_line('a')// &
        '  close(41)'//new_line('a')// &
        '  mode = ''RePlAcE'''//new_line('a')// &
        '  open(newunit=u, file=''/tmp/ffc_628_status_variable.dat'', '// &
        'status=mode, iostat=ios)'//new_line('a')// &
        '  if (ios /= 0) error stop 1'//new_line('a')// &
        '  close(u)'//new_line('a')// &
        '  inquire(file=''/tmp/ffc_628_status_variable.dat'', size=bytes)'// &
        new_line('a')// &
        '  if (bytes /= 0) error stop 1'//new_line('a')// &
        'end program main'

    print *, '=== direct session OPEN(STATUS=variable) compiler test ==='
    if (.not. expect_exit_status(source, 0, &
                                 '/tmp/ffc_628_status_variable')) stop 1
    print *, 'PASS: dynamic STATUS= values reach the OPEN runtime'
end program test_session_open_status_variable_compiler
