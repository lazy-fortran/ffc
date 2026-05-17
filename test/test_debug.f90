program test_debug
    implicit none
    integer :: i
    character(len=512) :: line
    character(len=:), allocatable :: file_content

    call execute_command_line('find /home/ert/code/lazy-fortran/fortfront/examples/lf/ -name "*.lf" -type f | sort > /tmp/lf_paths.txt 2>&1', &
                              exitstat=i)
    print '(A,I0)', 'find exit: ', i

    open(newunit=i, file='/tmp/lf_paths.txt', status='old', action='read')
    do
        read(i, '(A)', iostat=i) line
        if (i /= 0) exit
        print '(A)', trim(line)
    end do
    close(i)
end program test_debug
