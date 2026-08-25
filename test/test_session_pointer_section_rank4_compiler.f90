program test_session_pointer_section_rank4_compiler
    use ffc_test_support, only: expect_error_contains, expect_exit_status, &
        expect_output_matches_gfortran
    implicit none

    logical :: all_passed

    print *, '=== direct session rank-4 pointer-section compiler test ==='

    all_passed = .true.
    if (.not. test_rank4_alias_read_write()) all_passed = .false.
    if (.not. test_rank4_alias_matches_gfortran()) all_passed = .false.
    if (.not. test_rank4_noncontiguous_rejected()) all_passed = .false.
    if (.not. test_rank_changing_view_rejected()) all_passed = .false.

    if (.not. all_passed) stop 1

    print *, 'PASS: rank-4 pointer-section alias/read-write and refusals'

contains

    logical function test_rank4_alias_read_write()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, target :: t(2, 2, 2, 2)'//new_line('a')// &
            '  integer, pointer :: p(:, :, :, :)'//new_line('a')// &
            '  t = reshape([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, '// &
                '13, 14, 15, 16], [2, 2, 2, 2])'//new_line('a')// &
            '  p => t(:, :, :, 2:2)'//new_line('a')// &
            '  if (size(p) /= 8) stop 1'//new_line('a')// &
            '  if (size(p, 1) /= 2 .or. size(p, 2) /= 2 .or. '// &
                'size(p, 3) /= 2 .or. size(p, 4) /= 1) stop 2'//new_line('a')// &
            '  if (p(1, 1, 1, 1) /= 9 .or. p(2, 2, 2, 1) /= 16) stop 3'// &
                new_line('a')// &
            '  p(2, 1, 2, 1) = 707'//new_line('a')// &
            '  if (t(2, 1, 2, 2) /= 707) stop 4'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_rank4_alias_read_write = expect_exit_status( &
            source, 0, '/tmp/ffc_session_pointer_section_rank4_alias')
    end function test_rank4_alias_read_write

    logical function test_rank4_alias_matches_gfortran()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, target :: t(2, 2, 2, 2)'//new_line('a')// &
            '  integer, pointer :: p(:, :, :, :)'//new_line('a')// &
            '  t = reshape([3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, '// &
                '39, 42, 45, 48], [2, 2, 2, 2])'//new_line('a')// &
            '  p => t(:, :, :, 2:2)'//new_line('a')// &
            '  p(1, 2, 1, 1) = 999'//new_line('a')// &
            '  print *, sum(p), p(1, 1, 1, 1), p(1, 2, 1, 1), t(1, 2, 1, 2)'// &
                new_line('a')// &
            'end program main'

        test_rank4_alias_matches_gfortran = expect_output_matches_gfortran( &
            source, 'pointer_section_rank4')
    end function test_rank4_alias_matches_gfortran

    logical function test_rank4_noncontiguous_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, target :: t(2, 2, 2, 4)'//new_line('a')// &
            '  integer, pointer :: p(:, :, :, :)'//new_line('a')// &
            '  p => t(:, :, :, 1:4:2)'//new_line('a')// &
            'end program main'

        test_rank4_noncontiguous_rejected = expect_error_contains( &
            source, 'rank-4 pointer sections must be contiguous', &
            '/tmp/ffc_session_pointer_section_rank4_noncontiguous')
    end function test_rank4_noncontiguous_rejected

    logical function test_rank_changing_view_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, target :: t(2, 2, 2, 4)'//new_line('a')// &
            '  integer, pointer :: p(:, :, :)'//new_line('a')// &
            '  p => t(:, :, :, 2:3)'//new_line('a')// &
            'end program main'

        test_rank_changing_view_rejected = expect_error_contains( &
            source, 'pointer and section ranks do not match', &
            '/tmp/ffc_session_pointer_section_rank4_rank_change')
    end function test_rank_changing_view_rejected

end program test_session_pointer_section_rank4_compiler
