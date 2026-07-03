program test_session_io_char_spec_compiler
    use ffc_test_support, only: expect_error_contains, expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== direct session I/O character-specifier type test ==='

    all_passed = .true.
    if (.not. test_write_advance_real_rejected()) all_passed = .false.
    if (.not. test_open_status_integer_rejected()) all_passed = .false.
    if (.not. test_open_access_logical_rejected()) all_passed = .false.
    if (.not. test_close_status_integer_rejected()) all_passed = .false.
    if (.not. test_open_access_real_variable_rejected()) all_passed = .false.
    if (.not. test_valid_advance_still_runs()) all_passed = .false.
    if (.not. test_valid_open_specs_still_run()) all_passed = .false.
    if (.not. test_valid_char_variable_spec_runs()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: non-character I/O specifier values are rejected'

contains

    logical function test_write_advance_real_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            "  write(*,'(a)',advance=5.) 'x'"//new_line('a')// &
            'end program main'

        test_write_advance_real_rejected = expect_error_contains( &
            source, 'must be of type CHARACTER', '/tmp/ffc_io_advance_real')
    end function test_write_advance_real_rejected

    logical function test_open_status_integer_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: u'//new_line('a')// &
            '  open(newunit=u, status=1)'//new_line('a')// &
            'end program main'

        test_open_status_integer_rejected = expect_error_contains( &
            source, 'must be of type CHARACTER', '/tmp/ffc_io_status_int')
    end function test_open_status_integer_rejected

    logical function test_open_access_logical_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: u'//new_line('a')// &
            '  open(newunit=u, access=.false.)'//new_line('a')// &
            'end program main'

        test_open_access_logical_rejected = expect_error_contains( &
            source, 'must be of type CHARACTER', '/tmp/ffc_io_access_logical')
    end function test_open_access_logical_rejected

    logical function test_close_status_integer_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: u'//new_line('a')// &
            "  open(newunit=u, status='scratch')"//new_line('a')// &
            '  close(u, status=1)'//new_line('a')// &
            'end program main'

        test_close_status_integer_rejected = expect_error_contains( &
            source, 'must be of type CHARACTER', '/tmp/ffc_io_close_status_int')
    end function test_close_status_integer_rejected

    logical function test_open_access_real_variable_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: u'//new_line('a')// &
            '  real :: a'//new_line('a')// &
            '  a = 6.0'//new_line('a')// &
            '  open(newunit=u, access=a)'//new_line('a')// &
            'end program main'

        test_open_access_real_variable_rejected = expect_error_contains( &
            source, 'must be of type CHARACTER', '/tmp/ffc_io_access_realvar')
    end function test_open_access_real_variable_rejected

    logical function test_valid_advance_still_runs()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            "  write(*,'(a)',advance='no') 'x'"//new_line('a')// &
            "  write(*,'(a)') 'y'"//new_line('a')// &
            '  stop 3'//new_line('a')// &
            'end program main'

        test_valid_advance_still_runs = expect_exit_status( &
            source, 3, '/tmp/ffc_io_advance_valid')
    end function test_valid_advance_still_runs

    logical function test_valid_open_specs_still_run()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: u'//new_line('a')// &
            "  open(newunit=u, status='scratch', action='readwrite', "// &
            "form='formatted')"//new_line('a')// &
            '  close(u)'//new_line('a')// &
            '  stop 4'//new_line('a')// &
            'end program main'

        test_valid_open_specs_still_run = expect_exit_status( &
            source, 4, '/tmp/ffc_io_open_valid')
    end function test_valid_open_specs_still_run

    logical function test_valid_char_variable_spec_runs()
        ! A character variable is a valid ACCESS= value and must not be flagged.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: u'//new_line('a')// &
            '  character(len=10) :: acc'//new_line('a')// &
            "  acc = 'sequential'"//new_line('a')// &
            "  open(newunit=u, status='scratch', access=acc)"//new_line('a')// &
            '  close(u)'//new_line('a')// &
            '  stop 5'//new_line('a')// &
            'end program main'

        test_valid_char_variable_spec_runs = expect_exit_status( &
            source, 5, '/tmp/ffc_io_charvar_valid')
    end function test_valid_char_variable_spec_runs

end program test_session_io_char_spec_compiler
