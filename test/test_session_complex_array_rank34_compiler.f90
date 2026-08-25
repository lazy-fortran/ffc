program test_session_complex_array_rank34_compiler
    use ffc_test_support, only: expect_output_matches_gfortran
    implicit none

    logical :: all_passed

    print *, '=== direct session fixed complex rank-3/rank-4 test ==='
    all_passed = .true.
    if (.not. test_rank3()) all_passed = .false.
    if (.not. test_rank4()) all_passed = .false.
    if (.not. all_passed) stop 1
    print *, 'PASS: fixed complex rank-3/rank-4 elements and assignment '// &
        'match independent values and gfortran'

contains

    logical function test_rank3()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  complex(4) :: a(2,2,2), b(2,2,2)'//new_line('a')// &
            '  a = (1.0, -1.0)'//new_line('a')// &
            '  a(2,1,2) = (4.0, 5.0)'//new_line('a')// &
            '  b = a'//new_line('a')// &
            '  b(1,2,1) = b(2,1,2) + (2.0, -1.0)'//new_line('a')// &
            '  if (real(a(1,1,1)) /= 1.0 .or. aimag(a(1,1,1)) /= -1.0) '// &
                'error stop 1'//new_line('a')// &
            '  if (real(a(2,1,2)) /= 4.0 .or. aimag(a(2,1,2)) /= 5.0) '// &
                'error stop 2'//new_line('a')// &
            '  if (real(b(1,1,1)) /= 1.0 .or. aimag(b(1,1,1)) /= -1.0) '// &
                'error stop 3'//new_line('a')// &
            '  if (real(b(1,2,1)) /= 6.0 .or. aimag(b(1,2,1)) /= 4.0) '// &
                'error stop 4'//new_line('a')// &
            '  print *, a(2,1,2), b(1,2,1)'//new_line('a')// &
            'end program main'

        test_rank3 = expect_output_matches_gfortran(source, &
            'complex_array_rank34_rank3')
    end function test_rank3

    logical function test_rank4()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  complex(8) :: a(2,2,2,2), b(2,2,2,2)'//new_line('a')// &
            '  a = (1.0d0, 2.0d0)'//new_line('a')// &
            '  a(2,1,2,1) = (5.0d0, -4.0d0)'//new_line('a')// &
            '  b = a'//new_line('a')// &
            '  b(1,2,1,2) = b(2,1,2,1) + (3.0d0, 6.0d0)'//new_line('a')// &
            '  if (real(a(1,1,1,1)) /= 1.0d0 .or. aimag(a(1,1,1,1)) /= 2.0d0) '// &
                'error stop 5'//new_line('a')// &
            '  if (real(a(2,1,2,1)) /= 5.0d0 .or. aimag(a(2,1,2,1)) /= -4.0d0) '// &
                'error stop 6'//new_line('a')// &
            '  if (real(b(1,1,1,1)) /= 1.0d0 .or. aimag(b(1,1,1,1)) /= 2.0d0) '// &
                'error stop 7'//new_line('a')// &
            '  if (real(b(1,2,1,2)) /= 8.0d0 .or. aimag(b(1,2,1,2)) /= 2.0d0) '// &
                'error stop 8'//new_line('a')// &
            '  print *, a(2,1,2,1), b(1,2,1,2)'//new_line('a')// &
            'end program main'

        test_rank4 = expect_output_matches_gfortran(source, &
            'complex_array_rank34_rank4')
    end function test_rank4

end program test_session_complex_array_rank34_compiler
