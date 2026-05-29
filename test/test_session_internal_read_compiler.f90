program test_session_internal_read_compiler
    ! Internal read: read (buf, fmt) value parses an integer from a character
    ! variable. The parsed value is returned as the process exit status.
    use ffc_test_support, only: expect_exit_status, expect_error_contains
    implicit none

    logical :: all_passed

    print *, '=== direct session internal read compiler test ==='

    all_passed = .true.
    if (.not. test_internal_read_from_padded_string()) all_passed = .false.
    if (.not. test_internal_read_from_minimal_string()) all_passed = .false.
    if (.not. test_internal_read_real_rejected()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: internal read lowers through direct LIRIC session'

contains

    logical function test_internal_read_from_padded_string()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=20) :: buf'//new_line('a')// &
            '  integer :: v'//new_line('a')// &
            "  buf = '   42'"//new_line('a')// &
            "  read (buf, '(I5)') v"//new_line('a')// &
            '  stop v'//new_line('a')// &
            'end program main'

        test_internal_read_from_padded_string = expect_exit_status( &
            source, 42, '/tmp/ffc_iread_i5_test')
    end function test_internal_read_from_padded_string

    logical function test_internal_read_from_minimal_string()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=20) :: buf'//new_line('a')// &
            '  integer :: v'//new_line('a')// &
            "  buf = '42'"//new_line('a')// &
            "  read (buf, '(I0)') v"//new_line('a')// &
            '  stop v'//new_line('a')// &
            'end program main'

        test_internal_read_from_minimal_string = expect_exit_status( &
            source, 42, '/tmp/ffc_iread_i0_test')
    end function test_internal_read_from_minimal_string

    logical function test_internal_read_real_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=20) :: buf'//new_line('a')// &
            '  real :: x'//new_line('a')// &
            "  buf = '1.5'"//new_line('a')// &
            "  read (buf, '(F5.2)') x"//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_internal_read_real_rejected = expect_error_contains( &
            source, 'internal read', '/tmp/ffc_iread_real_test')
    end function test_internal_read_real_rejected

end program test_session_internal_read_compiler
