program test_session_close_status_compiler
    ! CLOSE must apply STATUS= and report its own result through IOSTAT=/IOMSG=.
    ! The helper runs the same source through ffc and gfortran, so the file
    ! deletion and invalid-unit status are independently observed.
    use ffc_test_support, only: expect_output_matches_gfortran
    implicit none

    character(len=*), parameter :: source = &
        'program main'//new_line('a')// &
        '  character(len=8) :: mode'//new_line('a')// &
        '  character(len=40) :: msg'//new_line('a')// &
        '  integer :: u, ios'//new_line('a')// &
        '  logical :: exists'//new_line('a')// &
        '  mode = ''DeLeTe'''//new_line('a')// &
        '  open(newunit=u, file=''/tmp/ffc_close_status_628.dat'', '// &
        'status=''replace'', iostat=ios, iomsg=msg)'//new_line('a')// &
        '  if (ios /= 0) error stop 1'//new_line('a')// &
        '  write(u,*) 42'//new_line('a')// &
        '  close(u, status=mode, iostat=ios, iomsg=msg)'//new_line('a')// &
        '  if (ios /= 0) error stop 2'//new_line('a')// &
        '  inquire(file=''/tmp/ffc_close_status_628.dat'', exist=exists)'// &
        new_line('a')// &
        '  if (exists) error stop 3'//new_line('a')// &
        '  open(newunit=u, file=''/tmp/ffc_close_keep_628.dat'', '// &
        'status=''replace'')'//new_line('a')// &
        '  close(u)'//new_line('a')// &
        '  inquire(file=''/tmp/ffc_close_keep_628.dat'', exist=exists)'// &
        new_line('a')// &
        '  if (.not. exists) error stop 5'//new_line('a')// &
        '  open(newunit=u, file=''/tmp/ffc_close_keep_628.dat'', '// &
        'status=''old'')'//new_line('a')// &
        '  close(u, status=''delete'')'//new_line('a')// &
        '  inquire(file=''/tmp/ffc_close_keep_628.dat'', exist=exists)'// &
        new_line('a')// &
        '  if (exists) error stop 6'//new_line('a')// &
        '  close(17, status=''whatever'', iostat=ios, iomsg=msg)'// &
        new_line('a')// &
        '  if (ios == 0 .or. len_trim(msg) == 0) error stop 4'//new_line('a')// &
        '  print *, ''close-status-ok'''//new_line('a')// &
        'end program main'

    print *, '=== direct session CLOSE(STATUS/IOSTAT/IOMSG) test ==='
    if (.not. expect_output_matches_gfortran(source, 'close_status_628')) stop 1
    print *, 'PASS: CLOSE status and diagnostics match gfortran'
end program test_session_close_status_compiler
