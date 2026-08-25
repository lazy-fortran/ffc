program test_session_reshape_rank4_compiler
    use ffc_test_support, only: expect_output, expect_output_matches_gfortran
    implicit none

    character(len=*), parameter :: source = &
        'program main'//new_line('a')// &
        '  integer :: a(2, 3, 2, 2)'//new_line('a')// &
        '  integer :: i, j, k, l'//new_line('a')// &
        '  a = reshape([1, 2, 3, 4, 5], [2, 3, 2, 2], '// &
        'pad=[9, 8, 7], order=[2, 4, 1, 3])'//new_line('a')// &
        '  do l = 1, 2'//new_line('a')// &
        '     do k = 1, 2'//new_line('a')// &
        '        do j = 1, 3'//new_line('a')// &
        '           do i = 1, 2'//new_line('a')// &
        '              print *, a(i, j, k, l)'//new_line('a')// &
        '           end do'//new_line('a')// &
        '        end do'//new_line('a')// &
        '     end do'//new_line('a')// &
        '  end do'//new_line('a')// &
        'end program main'

    character(len=*), parameter :: expected = &
        '           1'//new_line('a')// &
        '           8'//new_line('a')// &
        '           2'//new_line('a')// &
        '           7'//new_line('a')// &
        '           3'//new_line('a')// &
        '           9'//new_line('a')// &
        '           8'//new_line('a')// &
        '           8'//new_line('a')// &
        '           7'//new_line('a')// &
        '           7'//new_line('a')// &
        '           9'//new_line('a')// &
        '           9'//new_line('a')// &
        '           4'//new_line('a')// &
        '           8'//new_line('a')// &
        '           5'//new_line('a')// &
        '           7'//new_line('a')// &
        '           9'//new_line('a')// &
        '           9'//new_line('a')// &
        '           8'//new_line('a')// &
        '           8'//new_line('a')// &
        '           7'//new_line('a')// &
        '           7'//new_line('a')// &
        '           9'//new_line('a')// &
        '           9'//new_line('a')

    print *, '=== direct session reshape rank-4 compiler test ==='
    if (.not. expect_output(source, expected, &
            '/tmp/ffc_session_reshape_rank4_expected')) then
        print *, 'FAIL: independent rank-4 expected-value oracle'
        stop 1
    end if
    if (.not. expect_output_matches_gfortran(source, 'reshape_rank4')) then
        print *, 'FAIL: rank-4 output differs from gfortran'
        stop 1
    end if
    print *, 'PASS: rank-4 RESHAPE preserves order and cyclic padding'

end program test_session_reshape_rank4_compiler
