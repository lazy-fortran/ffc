program test_session_real_loop_accumulation_compiler
    use ffc_test_support, only: expect_output
    implicit none

    print *, '=== real loop accumulation compiler test ==='

    if (.not. test_real8_sum()) stop 1
    if (.not. test_real4_sum()) stop 1
    if (.not. test_nested_real8_sum()) stop 1

    print *, 'PASS: real scalars accumulate across counted DO iterations'

contains

    logical function test_real8_sum()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real(8) :: s'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  s = 0.0d0'//new_line('a')// &
            '  do i = 1, 5'//new_line('a')// &
            '    s = s + real(i, 8)'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  print *, s'//new_line('a')// &
            'end program main'

        ! 1.0 + 2.0 + 3.0 + 4.0 + 5.0 = 15.0, not the last term 5.0 (#300).
        test_real8_sum = expect_output(source, &
            '   15.000000000000000     '//new_line('a'), &
            '/tmp/ffc_real8_loop_accum_test')
    end function test_real8_sum

    logical function test_real4_sum()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real(4) :: s'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  s = 0.0'//new_line('a')// &
            '  do i = 1, 5'//new_line('a')// &
            '    s = s + real(i, 4)'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  print *, s'//new_line('a')// &
            'end program main'

        test_real4_sum = expect_output(source, &
            '   15.0000000    '//new_line('a'), &
            '/tmp/ffc_real4_loop_accum_test')
    end function test_real4_sum

    logical function test_nested_real8_sum()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real(8) :: s'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  integer :: j'//new_line('a')// &
            '  s = 0.0d0'//new_line('a')// &
            '  do i = 1, 3'//new_line('a')// &
            '    do j = 1, 2'//new_line('a')// &
            '      s = s + real(i, 8)'//new_line('a')// &
            '    end do'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  print *, s'//new_line('a')// &
            'end program main'

        ! Inner loop adds real(i) twice per i: 2*(1+2+3) = 12.0.
        test_nested_real8_sum = expect_output(source, &
            '   12.000000000000000     '//new_line('a'), &
            '/tmp/ffc_nested_real8_loop_accum_test')
    end function test_nested_real8_sum

end program test_session_real_loop_accumulation_compiler
