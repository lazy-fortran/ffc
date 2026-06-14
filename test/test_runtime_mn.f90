program test_runtime_mn
    use ffc_test_support, only: expect_exit_status
    implicit none
    logical :: ok
    
    ! Use runtime M and N (passed from command-line or computed)
    ! Actually allocate(a(m,n)) where m,n are variables
    ok = expect_exit_status(&
        'program main'//new_line('a')// &
        '  integer, allocatable :: a(:,:)'//new_line('a')// &
        '  integer :: m, n'//new_line('a')// &
        '  m = 2'//new_line('a')// &
        '  n = 3'//new_line('a')// &
        '  allocate(a(m,n))'//new_line('a')// &
        '  a(1,1) = 4'//new_line('a')// &
        '  a(1,2) = 5'//new_line('a')// &
        '  stop a(1,1)'//new_line('a')// &
        'end program main', 4, '/tmp/ffc_runtime_mn')
    if (ok) then
        print *, 'PASS: runtime M,N - a(1,1) not overwritten by a(1,2)'
    else
        print *, 'FAIL: runtime M,N test failed'
        stop 1
    end if

end program test_runtime_mn
