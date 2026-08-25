program test_session_array_element_rank34_compiler
    use ffc_test_support, only: expect_output_matches_gfortran
    implicit none

    character(len=*), parameter :: source = &
        'program main'//new_line('a')// &
        '  integer :: a(2,3,2), b(2,2,2,2)'//new_line('a')// &
        '  integer :: i, j, k, l'//new_line('a')// &
        '  a = 0'//new_line('a')// &
        '  b = 0'//new_line('a')// &
        '  a(2,1,2) = 123'//new_line('a')// &
        '  a(1,3,1) = 231'//new_line('a')// &
        '  b(2,1,2,2) = 412'//new_line('a')// &
        '  b(1,2,1,1) = 121'//new_line('a')// &
        '  print *, a(2,1,2), a(1,3,1)'//new_line('a')// &
        '  print *, b(2,1,2,2), b(1,2,1,1)'//new_line('a')// &
        '  if (a(2,1,2) /= 123 .or. a(1,3,1) /= 231) error stop 1'// &
        new_line('a')// &
        '  if (b(2,1,2,2) /= 412 .or. b(1,2,1,1) /= 121) error stop 2'// &
        new_line('a')// &
        '  do l = 1, 2'//new_line('a')// &
        '    do k = 1, 2'//new_line('a')// &
        '      do j = 1, 3'//new_line('a')// &
        '        do i = 1, 2'//new_line('a')// &
        '          a(i,j,k) = 100*k + 10*j + i'//new_line('a')// &
        '          b(i,mod(j-1,2)+1,k,l) = 1000*l + 100*k + 10*j + i'// &
        new_line('a')// &
        '        end do'//new_line('a')// &
        '      end do'//new_line('a')// &
        '    end do'//new_line('a')// &
        '  end do'//new_line('a')// &
        '  print *, a(2,3,2), b(2,2,2,2)'//new_line('a')// &
        'end program main'

    print *, '=== direct session rank-3/rank-4 array element test ==='
    if (.not. expect_output_matches_gfortran(source, 'array_element_rank34')) stop 1
    print *, 'PASS: rank-3/rank-4 array element reads and writes match gfortran'

end program test_session_array_element_rank34_compiler
