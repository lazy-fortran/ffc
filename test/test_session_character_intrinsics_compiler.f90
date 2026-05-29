program test_session_character_intrinsics_compiler
    use ffc_test_support, only: expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== direct session character intrinsics compiler test ==='

    all_passed = .true.
    if (.not. test_len_of_deferred_after_assignment()) all_passed = .false.
    if (.not. test_len_trim_of_padded_fixed_length()) all_passed = .false.
    if (.not. test_len_of_unallocated_is_zero()) all_passed = .false.
    if (.not. test_len_of_fixed_length()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: len/len_trim lower through direct LIRIC session'

contains

    logical function test_len_of_deferred_after_assignment()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=:), allocatable :: s'//new_line('a')// &
            '  s = "hello"'//new_line('a')// &
            '  stop len(s)'//new_line('a')// &
            'end program main'

        test_len_of_deferred_after_assignment = expect_exit_status( &
            source, 5, '/tmp/ffc_session_len_deferred_test')
    end function test_len_of_deferred_after_assignment

    logical function test_len_trim_of_padded_fixed_length()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=10) :: s'//new_line('a')// &
            '  s = "hi"'//new_line('a')// &
            '  stop len_trim(s)'//new_line('a')// &
            'end program main'

        test_len_trim_of_padded_fixed_length = expect_exit_status( &
            source, 2, '/tmp/ffc_session_len_trim_padded_test')
    end function test_len_trim_of_padded_fixed_length

    logical function test_len_of_unallocated_is_zero()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=:), allocatable :: s'//new_line('a')// &
            '  stop len(s)'//new_line('a')// &
            'end program main'

        test_len_of_unallocated_is_zero = expect_exit_status( &
            source, 0, '/tmp/ffc_session_len_unalloc_test')
    end function test_len_of_unallocated_is_zero

    logical function test_len_of_fixed_length()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=7) :: s'//new_line('a')// &
            '  s = "ab"'//new_line('a')// &
            '  stop len(s)'//new_line('a')// &
            'end program main'

        test_len_of_fixed_length = expect_exit_status( &
            source, 7, '/tmp/ffc_session_len_fixed_test')
    end function test_len_of_fixed_length

end program test_session_character_intrinsics_compiler
