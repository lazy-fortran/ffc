program test_minimal_ext1
    use ffc_test_support, only: expect_exit_status
    implicit none
    logical :: ok
    
    ! Test 1: a(1,1)=4, a(1,2)=5, stop a(1,1) - should be 4 if extent1=2, 5 if extent1=0
    ok = expect_exit_status(&
        'program main'//new_line('a')// &
        '  integer, allocatable :: a(:,:)'//new_line('a')// &
        '  allocate(a(2,3))'//new_line('a')// &
        '  a(1,1) = 4'//new_line('a')// &
        '  a(1,2) = 5'//new_line('a')// &
        '  stop a(1,1)'//new_line('a')// &
        'end program main', 4, '/tmp/ffc_ext1_test')
    if (ok) then
        print *, 'PASS: extent1 correctly computed as 2'
    else
        print *, 'FAIL: extent1 may be 0 (a(1,2) overwrote a(1,1))'
        stop 1
    end if
    
    ! Test 2: a(2,1)=7, a(2,2)=8, stop a(2,1) - should be 7 if extent1=2, 8 if extent1=0
    ok = expect_exit_status(&
        'program main'//new_line('a')// &
        '  integer, allocatable :: a(:,:)'//new_line('a')// &
        '  allocate(a(2,3))'//new_line('a')// &
        '  a(2,1) = 7'//new_line('a')// &
        '  a(2,2) = 8'//new_line('a')// &
        '  stop a(2,1)'//new_line('a')// &
        'end program main', 7, '/tmp/ffc_ext1_test2')
    if (ok) then
        print *, 'PASS: a(2,1) not overwritten by a(2,2)'
    else
        print *, 'FAIL: a(2,2) overwrote a(2,1), extent1 may be 0'
    end if
    
end program test_minimal_ext1
