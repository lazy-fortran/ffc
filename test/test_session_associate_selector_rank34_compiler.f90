program test_session_associate_selector_rank34_compiler
    use ffc_test_support, only: expect_exit_status, &
        expect_output_matches_gfortran
    implicit none

    logical :: all_passed

    print *, '=== direct session rank-3/rank-4 associate selector test ==='
    all_passed = .true.
    if (.not. test_rank3_expected()) all_passed = .false.
    if (.not. test_rank4_expected()) all_passed = .false.
    if (.not. test_rank34_matches_gfortran()) all_passed = .false.
    if (.not. all_passed) stop 1
    print *, 'PASS: rank-3 and rank-4 associate selectors read and write through views'

contains

    logical function test_rank3_expected()
        ! Independent checks cover shape, reads, writes through x, and the
        ! source array after the associate scope ends.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'integer :: a(2,2,2)'//new_line('a')// &
            'a = reshape([1,2,3,4,5,6,7,8], [2,2,2])'//new_line('a')// &
            'associate (x => a(:,:,:))'//new_line('a')// &
            '    if (size(x,1) /= 2 .or. size(x,2) /= 2 .or. size(x,3) /= 2) stop 1'// &
            new_line('a')// &
            '    if (lbound(x,1) /= 1 .or. lbound(x,2) /= 1 .or. lbound(x,3) /= 1) stop 2'// &
            new_line('a')// &
            '    if (x(1,2,1) /= 3 .or. x(2,1,2) /= 6) stop 3'//new_line('a')// &
            '    x(2,1,2) = 99'//new_line('a')// &
            'end associate'//new_line('a')// &
            'if (a(2,1,2) /= 99) stop 4'//new_line('a')// &
            'if (a(1,1,1) /= 1 .or. a(1,2,1) /= 3 .or. a(1,1,2) /= 5 .or. '// &
            'a(2,2,2) /= 8) stop 5'//new_line('a')// &
            'end program main'

        test_rank3_expected = expect_exit_status( &
            source, 0, '/tmp/ffc_session_associate_selector_rank3_expected')
    end function test_rank3_expected

    logical function test_rank4_expected()
        ! The rank-4 view uses the same borrowed flat storage with four shape
        ! entries, including a write that must reach the original array.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'integer :: a(2,2,2,2)'//new_line('a')// &
            'a = 1'//new_line('a')// &
            'a(2,1,1,2) = 10'//new_line('a')// &
            'associate (x => a(:,:,:,:))'//new_line('a')// &
            '    if (size(x,1) /= 2 .or. size(x,2) /= 2 .or. size(x,3) /= 2 .or. '// &
            'size(x,4) /= 2) stop 1'//new_line('a')// &
            '    if (x(1,1,1,1) /= 1 .or. x(2,1,1,2) /= 10) stop 2'//new_line('a')// &
            '    x(2,1,1,2) = 77'//new_line('a')// &
            'end associate'//new_line('a')// &
            'if (a(2,1,1,2) /= 77 .or. a(1,1,1,1) /= 1 .or. '// &
            'a(2,2,2,2) /= 1) stop 3'//new_line('a')// &
            'end program main'

        test_rank4_expected = expect_exit_status( &
            source, 0, '/tmp/ffc_session_associate_selector_rank4_expected')
    end function test_rank4_expected

    logical function test_rank34_matches_gfortran()
        ! gfortran independently checks the observable reads and writes for
        ! both ranks, while the two tests above retain exact expected values.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'integer :: a(2,2,2), b(2,2,2,2)'//new_line('a')// &
            'a = 1'//new_line('a')// &
            'b = 1'//new_line('a')// &
            new_line('a')// &
            'associate (x => a(:,:,:))'//new_line('a')// &
            '    x(1,2,2) = 31'//new_line('a')// &
            '    print *, size(x), x(1,1,1), x(1,2,2), x(2,2,2)'//new_line('a')// &
            'end associate'//new_line('a')// &
            'associate (y => b(:,:,:,:))'//new_line('a')// &
            '    y(2,2,2,2) = 41'//new_line('a')// &
            '    print *, size(y), y(1,1,1,1), y(2,2,2,2), y(1,2,1,2)'//new_line('a')// &
            'end associate'//new_line('a')// &
            'end program main'

        test_rank34_matches_gfortran = expect_output_matches_gfortran( &
            source, 'associate_selector_rank34')
    end function test_rank34_matches_gfortran

end program test_session_associate_selector_rank34_compiler
