program test_sync
    implicit none
    integer :: i
    character(len=512) :: line

    call execute_command_line('find /home/ert/code/lazy-fortran/fortfront/examples/lf/ -name "*.lf" -type f | sort > /tmp/lf_paths.txt 2>&1', &
                              exitstat=i)
    print '(A,I0)', 'find exit: ', i

    ! Check file size
    inquire(file='/tmp/lf_paths.txt', size=i)
    print '(A,I0)', 'file size: ', i

    open(newunit=i, file='/tmp/lf_paths.txt', status='old', action='read')
    read(i, '(A)', iostat=i) line
    print '(A)', 'first line: ', trim(line)
    close(i)
end program test_sync
