program test_session_case_overlap_compiler
    use ffc_test_support, only: expect_error_contains, expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== direct session SELECT CASE overlap compiler test ==='

    all_passed = .true.
    if (.not. test_overlapping_range_and_label()) all_passed = .false.
    if (.not. test_overlapping_two_ranges()) all_passed = .false.
    if (.not. test_overlapping_open_ended()) all_passed = .false.
    if (.not. test_non_overlapping_still_compiles()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: overlapping SELECT CASE labels are rejected'

contains

    logical function test_overlapping_range_and_label()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  i = 3'//new_line('a')// &
            '  select case (i)'//new_line('a')// &
            '  case (1:10)'//new_line('a')// &
            '    stop 1'//new_line('a')// &
            '  case (5)'//new_line('a')// &
            '    stop 2'//new_line('a')// &
            '  end select'//new_line('a')// &
            'end program main'

        test_overlapping_range_and_label = expect_error_contains( &
            source, 'overlaps', '/tmp/ffc_case_overlap_range_label')
    end function test_overlapping_range_and_label

    logical function test_overlapping_two_ranges()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  i = 3'//new_line('a')// &
            '  select case (i)'//new_line('a')// &
            '  case (20:30)'//new_line('a')// &
            '    stop 1'//new_line('a')// &
            '  case (25:40)'//new_line('a')// &
            '    stop 2'//new_line('a')// &
            '  end select'//new_line('a')// &
            'end program main'

        test_overlapping_two_ranges = expect_error_contains( &
            source, 'overlaps', '/tmp/ffc_case_overlap_two_ranges')
    end function test_overlapping_two_ranges

    logical function test_overlapping_open_ended()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  i = 3'//new_line('a')// &
            '  select case (i)'//new_line('a')// &
            '  case (30)'//new_line('a')// &
            '    stop 1'//new_line('a')// &
            '  case (25:)'//new_line('a')// &
            '    stop 2'//new_line('a')// &
            '  end select'//new_line('a')// &
            'end program main'

        test_overlapping_open_ended = expect_error_contains( &
            source, 'overlaps', '/tmp/ffc_case_overlap_open_ended')
    end function test_overlapping_open_ended

    logical function test_non_overlapping_still_compiles()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  i = 7'//new_line('a')// &
            '  select case (i)'//new_line('a')// &
            '  case (1:5)'//new_line('a')// &
            '    stop 11'//new_line('a')// &
            '  case (6:10)'//new_line('a')// &
            '    stop 22'//new_line('a')// &
            '  case default'//new_line('a')// &
            '    stop 33'//new_line('a')// &
            '  end select'//new_line('a')// &
            'end program main'

        test_non_overlapping_still_compiles = expect_exit_status( &
            source, 22, '/tmp/ffc_case_overlap_valid')
    end function test_non_overlapping_still_compiles

end program test_session_case_overlap_compiler
