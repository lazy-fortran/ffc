program test_session_array_ctor_length_compiler
    use ffc_test_support, only: expect_error_contains, expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== direct session array-constructor length-mismatch test ==='

    all_passed = .true.
    if (.not. test_assignment_too_few_rejected()) all_passed = .false.
    if (.not. test_assignment_too_many_rejected()) all_passed = .false.
    if (.not. test_initializer_too_few_rejected()) all_passed = .false.
    if (.not. test_matching_length_still_runs()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: array-constructor length mismatches are rejected'

contains

    logical function test_assignment_too_few_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(4)'//new_line('a')// &
            '  a = [1, 2, 3]'//new_line('a')// &
            '  print *, a'//new_line('a')// &
            'end program main'

        test_assignment_too_few_rejected = expect_error_contains( &
            source, 'element count does not match', &
            '/tmp/ffc_ac_too_few')
    end function test_assignment_too_few_rejected

    logical function test_assignment_too_many_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(2)'//new_line('a')// &
            '  a = [1, 2, 3, 4]'//new_line('a')// &
            '  print *, a'//new_line('a')// &
            'end program main'

        test_assignment_too_many_rejected = expect_error_contains( &
            source, 'element count does not match', &
            '/tmp/ffc_ac_too_many')
    end function test_assignment_too_many_rejected

    logical function test_initializer_too_few_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(3) = [1, 2]'//new_line('a')// &
            '  print *, a'//new_line('a')// &
            'end program main'

        test_initializer_too_few_rejected = expect_error_contains( &
            source, 'element count does not match', &
            '/tmp/ffc_ac_init_too_few')
    end function test_initializer_too_few_rejected

    logical function test_matching_length_still_runs()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(4)'//new_line('a')// &
            '  a = [1, 2, 3, 4]'//new_line('a')// &
            '  if (sum(a) == 10) stop 10'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_matching_length_still_runs = expect_exit_status( &
            source, 10, '/tmp/ffc_ac_valid')
    end function test_matching_length_still_runs

end program test_session_array_ctor_length_compiler
