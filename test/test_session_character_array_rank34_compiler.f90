program test_session_character_array_rank34_compiler
    use ffc_test_support, only: expect_output_matches_gfortran
    implicit none

    logical :: all_passed

    print *, '=== direct session fixed character rank-3/rank-4 test ==='
    all_passed = test_rank3() .and. test_rank4()
    if (.not. all_passed) stop 1
    print *, 'PASS: fixed character rank-3/rank-4 elements match independent values and gfortran'

contains

    logical function test_rank3()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=4) :: a(2,3,2)'//new_line('a')// &
            '  integer :: i, j, k'//new_line('a')// &
            '  do k = 1, 2'//new_line('a')// &
            '    do j = 1, 3'//new_line('a')// &
            '      do i = 1, 2'//new_line('a')// &
            '        a(i,j,k) = achar(ichar("A") + 10*i + j + k)'//new_line('a')// &
            '      end do'//new_line('a')// &
            '    end do'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  a(2,3,1) = "xy"'//new_line('a')// &
            '  if (a(1,1,1) /= "M   ") error stop 1'//new_line('a')// &
            '  if (a(2,3,1) /= "xy  ") error stop 2'//new_line('a')// &
            '  if (a(1,3,2) /= "P   ") error stop 3'//new_line('a')// &
            '  print *, "[", a(1,1,1), a(2,3,1), a(1,3,2), "]"'//new_line('a')// &
            'end program main'

        test_rank3 = expect_output_matches_gfortran(source, &
            'character_array_rank34_rank3')
    end function test_rank3

    logical function test_rank4()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=3) :: a(2,2,2,2)'//new_line('a')// &
            '  integer :: i, j, k, l'//new_line('a')// &
            '  do l = 1, 2'//new_line('a')// &
            '    do k = 1, 2'//new_line('a')// &
            '      do j = 1, 2'//new_line('a')// &
            '        do i = 1, 2'//new_line('a')// &
            '          a(i,j,k,l) = "abc"'//new_line('a')// &
            '        end do'//new_line('a')// &
            '      end do'//new_line('a')// &
            '    end do'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  a(2,1,2,1) = "longer"'//new_line('a')// &
            '  a(1,2,1,2) = "q"'//new_line('a')// &
            '  if (a(2,1,2,1) /= "lon") error stop 4'//new_line('a')// &
            '  if (a(1,2,1,2) /= "q  ") error stop 5'//new_line('a')// &
            '  if (a(2,2,2,2) /= "abc") error stop 6'//new_line('a')// &
            '  print *, "[", a(2,1,2,1), a(1,2,1,2), a(2,2,2,2), "]"'//new_line('a')// &
            'end program main'

        test_rank4 = expect_output_matches_gfortran(source, &
            'character_array_rank34_rank4')
    end function test_rank4

end program test_session_character_array_rank34_compiler
