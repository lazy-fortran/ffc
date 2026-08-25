program test_session_array_section_copy_rank4_compiler
    use ffc_test_support, only: expect_output, expect_output_matches_gfortran
    implicit none

    logical :: all_passed

    print *, '=== direct session rank-4 array-section copy test ==='

    all_passed = test_rank4_copy_expected()
    if (.not. test_rank4_copy_matches_gfortran()) all_passed = .false.
    if (.not. all_passed) stop 1

    print *, 'PASS: rank-4 conformable section copy and scalar broadcast'

contains

    logical function test_rank4_copy_expected()
        ! The expected values are calculated from the two source slices:
        ! eight 31s, eight 47s, and eight broadcast 7s.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(2, 2, 2, 3), b(2, 2, 2, 3)'//new_line('a')// &
            '  a = 0'//new_line('a')// &
            '  b = 0'//new_line('a')// &
            '  b(:, :, :, 2:2) = 31'//new_line('a')// &
            '  b(:, :, :, 3:3) = 47'//new_line('a')// &
            '  a(:, :, :, 1:2) = b(:, :, :, 2:3)'//new_line('a')// &
            '  a(:, :, :, 3:3) = 7'//new_line('a')// &
            '  print *, sum(a), a(1, 1, 1, 1), a(1, 1, 1, 2), '// &
            'a(1, 1, 1, 3), a(2, 2, 2, 3)'//new_line('a')// &
            'end program main'

        test_rank4_copy_expected = expect_output(source, &
            '         680          31          47           7           7'// &
            new_line('a'), '/tmp/ffc_session_array_section_copy_rank4_expected')
    end function test_rank4_copy_expected

    logical function test_rank4_copy_matches_gfortran()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(2, 2, 2, 3), b(2, 2, 2, 3)'//new_line('a')// &
            '  a = 0'//new_line('a')// &
            '  b = 0'//new_line('a')// &
            '  b(:, :, :, 2:2) = 31'//new_line('a')// &
            '  b(:, :, :, 3:3) = 47'//new_line('a')// &
            '  a(:, :, :, 1:2) = b(:, :, :, 2:3)'//new_line('a')// &
            '  a(:, :, :, 3:3) = 7'//new_line('a')// &
            '  print *, sum(a), a(1, 1, 1, 1), a(1, 1, 1, 2), '// &
            'a(1, 1, 1, 3), a(2, 2, 2, 3)'//new_line('a')// &
            'end program main'

        test_rank4_copy_matches_gfortran = expect_output_matches_gfortran( &
            source, 'array_section_copy_fixed_rank4')
    end function test_rank4_copy_matches_gfortran

end program test_session_array_section_copy_rank4_compiler
