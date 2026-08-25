program test_session_associate_expression_rank4_compiler
    use ffc_test_support, only: expect_exit_status, &
        expect_output_matches_gfortran
    implicit none

    logical :: all_passed

    print *, '=== direct session rank-4 associate expression test ==='
    all_passed = .true.
    if (.not. test_rank4_expected()) all_passed = .false.
    if (.not. test_rank4_matches_gfortran()) all_passed = .false.
    if (.not. all_passed) stop 1
    print *, 'PASS: rank-4 associate expressions read and write through storage'

contains

    logical function test_rank4_expected()
        ! The expression selector is materialised once.  Reads use its four
        ! dimensions, and writes update that materialised value without
        ! changing either source operand.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'integer :: a(2,2,2,2), b(2,2,2,2)'//new_line('a')// &
            'a = 1'//new_line('a')// &
            'b = 10'//new_line('a')// &
            'b(2,1,1,2) = 20'//new_line('a')// &
            'associate (x => a + b)'//new_line('a')// &
            '    if (size(x,1) /= 2 .or. size(x,2) /= 2 .or. '// &
            'size(x,3) /= 2 .or. size(x,4) /= 2) stop 1'//new_line('a')// &
            '    if (x(1,1,1,1) /= 11 .or. x(2,1,1,2) /= 21) stop 2'// &
            new_line('a')// &
            '    x(2,1,1,2) = 77'//new_line('a')// &
            '    if (x(2,1,1,2) /= 77) stop 3'//new_line('a')// &
            'end associate'//new_line('a')// &
            'if (a(2,1,1,2) /= 1 .or. b(2,1,1,2) /= 20) stop 4'// &
            new_line('a')// &
            'end program main'

        test_rank4_expected = expect_exit_status( &
            source, 0, '/tmp/ffc_session_associate_expression_rank4_expected')
    end function test_rank4_expected

    logical function test_rank4_matches_gfortran()
        ! Keep the oracle case within the expression-selector behavior that
        ! gfortran accepts: shape and value observations only.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'integer :: a(2,2,2,2), b(2,2,2,2)'//new_line('a')// &
            'a = 1'//new_line('a')// &
            'b = 10'//new_line('a')// &
            'b(2,1,1,2) = 20'//new_line('a')// &
            'associate (x => a + b)'//new_line('a')// &
            '    print *, size(x), x(1,1,1,1), x(2,1,1,2), x(1,2,2,2)'// &
            new_line('a')// &
            'end associate'//new_line('a')// &
            'end program main'

        test_rank4_matches_gfortran = expect_output_matches_gfortran( &
            source, 'associate_expression_rank4')
    end function test_rank4_matches_gfortran

end program test_session_associate_expression_rank4_compiler
