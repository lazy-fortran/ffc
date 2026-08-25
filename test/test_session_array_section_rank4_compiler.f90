program test_session_array_section_rank4_compiler
    use ffc_test_support, only: expect_error_contains, expect_output, &
                                expect_output_matches_gfortran
    implicit none

    logical :: all_passed

    print *, '=== direct session fixed rank-4 array section assignment test ==='

    all_passed = .true.
    if (.not. test_rank4_scalar_broadcast_expected()) all_passed = .false.
    if (.not. test_rank4_conformable_copy_matches_gfortran()) all_passed = .false.
    if (.not. test_rank5_section_rejected()) all_passed = .false.

    if (.not. all_passed) stop 1

    print *, 'PASS: fixed rank-4 array section assignment lowers correctly'

contains

    logical function test_rank4_scalar_broadcast_expected()
        ! The exact expected values independently check selected and untouched
        ! rank-4 elements after scalar broadcast.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(2, 2, 2, 2)'//new_line('a')// &
            '  a = 1'//new_line('a')// &
            '  a(:, :, :, 2:2) = 7'//new_line('a')// &
            '  print *, sum(a), a(1, 1, 1, 1), a(1, 1, 1, 2)'//new_line('a')// &
            'end program main'

        test_rank4_scalar_broadcast_expected = expect_output( &
            source, '          64           1           7'//new_line('a'), &
            '/tmp/ffc_session_array_section_rank4_broadcast_test')
    end function test_rank4_scalar_broadcast_expected

    logical function test_rank4_conformable_copy_matches_gfortran()
        ! Compare a rank-4 section copy against gfortran's independent oracle.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(2, 2, 2, 2), b(2, 2, 2, 2)'//new_line('a')// &
            '  a = 0'//new_line('a')// &
            '  a(:, :, :, 1:1) = 11'//new_line('a')// &
            '  a(:, :, :, 2:2) = 21'//new_line('a')// &
            '  b = 0'//new_line('a')// &
            '  b(:, :, :, 1:1) = a(:, :, :, 2:2)'//new_line('a')// &
            '  print *, sum(b), b(1, 1, 1, 1), b(2, 2, 2, 1), b(1, 1, 1, 2)'// &
            new_line('a')// &
            'end program main'

        test_rank4_conformable_copy_matches_gfortran = &
            expect_output_matches_gfortran(source, 'array_section_fixed_rank4')
    end function test_rank4_conformable_copy_matches_gfortran

    logical function test_rank5_section_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(1, 1, 1, 1, 1)'//new_line('a')// &
            '  a(:, :, :, :, :) = 1'//new_line('a')// &
            'end program main'

        test_rank5_section_rejected = expect_error_contains( &
            source, 'most rank-4 array sections', &
            '/tmp/ffc_session_array_section_rank5_test')
    end function test_rank5_section_rejected

end program test_session_array_section_rank4_compiler
