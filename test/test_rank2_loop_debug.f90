program test_rank2_loop_debug
    use ffc_test_support, only: expect_exit_status
    implicit none
    
    ! First: manual writes, no loops, sum all 6 elements
    logical :: ok
    
    ! test: a(1,1)=4,a(2,1)=7,a(1,2)=5,a(2,2)=8,a(1,3)=6,a(2,3)=9 -> sum=39
    ok = expect_exit_status(&
        'program main'//new_line('a')// &
        '  integer, allocatable :: a(:,:)'//new_line('a')// &
        '  integer :: total'//new_line('a')// &
        '  allocate(a(2,3))'//new_line('a')// &
        '  a(1,1) = 4'//new_line('a')// &
        '  a(2,1) = 7'//new_line('a')// &
        '  a(1,2) = 5'//new_line('a')// &
        '  a(2,2) = 8'//new_line('a')// &
        '  a(1,3) = 6'//new_line('a')// &
        '  a(2,3) = 9'//new_line('a')// &
        '  total = a(1,1)+a(2,1)+a(1,2)+a(2,2)+a(1,3)+a(2,3)'//new_line('a')// &
        '  stop total'//new_line('a')// &
        'end program main', 39, '/tmp/ffc_rank2_manual')
    if (ok) then
        print *, 'PASS: manual writes sum=39'
    else
        print *, 'FAIL: manual writes'
        stop 1
    end if
    
end program test_rank2_loop_debug
