program test_session_shift_rank34_compiler
    use ffc_test_support, only: expect_output_matches_gfortran
    implicit none

    character(len=*), parameter :: source = &
        'program main'//new_line('a')// &
        '  integer :: a(2,3,2), r(2,3,2)'//new_line('a')// &
        '  integer :: b(2,2,3,2), q(2,2,3,2), t(2,2,3,2)'//new_line('a')// &
        '  integer :: i, j, k, l'//new_line('a')// &
        '  do k = 1, 2'//new_line('a')// &
        '     do j = 1, 3'//new_line('a')// &
        '        do i = 1, 2'//new_line('a')// &
        '           a(i,j,k) = 100*k + 10*j + i'//new_line('a')// &
        '        end do'//new_line('a')// &
        '     end do'//new_line('a')// &
        '  end do'//new_line('a')// &
        '  r = cshift(a, 1, 2)'//new_line('a')// &
        '  if (r(1,1,1) /= 121 .or. r(2,1,1) /= 122) stop 11'//new_line('a')// &
        '  if (r(1,2,1) /= 131 .or. r(2,2,1) /= 132) stop 12'//new_line('a')// &
        '  if (r(1,3,1) /= 111 .or. r(2,3,1) /= 112) stop 13'//new_line('a')// &
        '  if (r(1,1,2) /= 221 .or. r(2,1,2) /= 222) stop 14'//new_line('a')// &
        '  if (r(1,2,2) /= 231 .or. r(2,2,2) /= 232) stop 15'//new_line('a')// &
        '  if (r(1,3,2) /= 211 .or. r(2,3,2) /= 212) stop 16'//new_line('a')// &
        '  do l = 1, 2'//new_line('a')// &
        '     do k = 1, 3'//new_line('a')// &
        '        do j = 1, 2'//new_line('a')// &
        '           do i = 1, 2'//new_line('a')// &
        '              b(i,j,k,l) = 1000*l + 100*k + 10*j + i'//new_line('a')// &
        '           end do'//new_line('a')// &
        '        end do'//new_line('a')// &
        '     end do'//new_line('a')// &
        '  end do'//new_line('a')// &
        '  q = eoshift(b, 1, 99, 3)'//new_line('a')// &
        '  t = cshift(b, -1, 4)'//new_line('a')// &
        '  if (q(1,1,1,1) /= 1211 .or. q(2,2,1,2) /= 2222) stop 21'//new_line('a')// &
        '  if (q(1,1,2,1) /= 1311 .or. q(2,2,2,2) /= 2322) stop 22'//new_line('a')// &
        '  if (q(1,1,3,1) /= 99 .or. q(2,2,3,2) /= 99) stop 23'//new_line('a')// &
        '  if (t(1,1,1,1) /= 2111 .or. t(2,2,3,1) /= 2322) stop 24'//new_line('a')// &
        '  if (t(1,1,1,2) /= 1111 .or. t(2,2,3,2) /= 1322) stop 25'//new_line('a')// &
        '  print *, r'//new_line('a')// &
        '  print *, q'//new_line('a')// &
        '  print *, t'//new_line('a')// &
        'end program main'

    print *, '=== direct session rank-3/rank-4 cshift/eoshift test ==='
    if (.not. expect_output_matches_gfortran(source, 'shift_rank34')) stop 1
    print *, 'PASS: rank-3/rank-4 shifts match independent checks and gfortran'

end program test_session_shift_rank34_compiler
