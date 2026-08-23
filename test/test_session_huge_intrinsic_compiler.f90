program test_session_huge_intrinsic_compiler
    use ffc_test_support, only: expect_error_contains, expect_no_error, &
        expect_output_matches_gfortran
    implicit none

    logical :: all_passed

    all_passed = .true.
    if (.not. test_integer_huge_matches_gfortran()) all_passed = .false.
    if (.not. test_repeat_huge_count_compiles()) all_passed = .false.
    if (.not. test_real_huge_count_is_rejected()) all_passed = .false.
    if (.not. test_wide_huge_count_is_rejected()) all_passed = .false.
    if (.not. test_huge_arity_is_rejected()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: bounded integer HUGE lowering'

contains

    logical function test_integer_huge_matches_gfortran()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  print *, huge(1_4)'//new_line('a')// &
            '  print *, len(repeat("z", huge(1_1)))'//new_line('a')// &
            'end program main'

        test_integer_huge_matches_gfortran = expect_output_matches_gfortran( &
            source, 'huge_integer_values')
    end function test_integer_huge_matches_gfortran

    logical function test_repeat_huge_count_compiles()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character, parameter :: z = "z"'//new_line('a')// &
            '  print *, repeat(z, huge(1_4))'//new_line('a')// &
            'end program main'

        test_repeat_huge_count_compiles = expect_no_error( &
            source, '/tmp/ffc_huge_repeat_compile')
    end function test_repeat_huge_count_compiles

    logical function test_real_huge_count_is_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  print *, repeat("z", huge(1.0))'//new_line('a')// &
            'end program main'

        test_real_huge_count_is_rejected = expect_error_contains( &
            source, 'HUGE argument must be INTEGER', &
            '/tmp/ffc_huge_repeat_real_count')
    end function test_real_huge_count_is_rejected

    logical function test_wide_huge_count_is_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  print *, repeat("z", huge(1_8))'//new_line('a')// &
            'end program main'

        test_wide_huge_count_is_rejected = expect_error_contains( &
            source, 'HUGE result exceeds INTEGER(4)', &
            '/tmp/ffc_huge_repeat_wide_count')
    end function test_wide_huge_count_is_rejected

    logical function test_huge_arity_is_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  print *, huge()'//new_line('a')// &
            'end program main'

        test_huge_arity_is_rejected = expect_error_contains( &
            source, 'huge requires one argument', &
            '/tmp/ffc_huge_wrong_arity')
    end function test_huge_arity_is_rejected

end program test_session_huge_intrinsic_compiler
