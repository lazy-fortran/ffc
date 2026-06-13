program test_session_open_close_file_compiler
    ! OPEN / WRITE(unit,*) / CLOSE round-trip (#247 B5c).
    use ffc_test_support, only: expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== direct session OPEN/CLOSE/file-write compiler test ==='

    all_passed = .true.
    if (.not. test_open_write_close_newunit()) all_passed = .false.
    if (.not. test_open_write_close_literal_unit()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: OPEN/WRITE/CLOSE lowers through direct LIRIC session'

contains

    logical function test_open_write_close_newunit()
        ! OPEN with newunit=, WRITE to that unit, CLOSE, then check via stdout.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: u'//new_line('a')// &
            '  open(newunit=u, file='//achar(39)//'/tmp/ffc_open_close_newunit.txt'//achar(39)// &
            ', status='//achar(39)//'replace'//achar(39)//')'//new_line('a')// &
            '  write(u, *) 42'//new_line('a')// &
            '  close(u)'//new_line('a')// &
            'end program main'

        test_open_write_close_newunit = expect_exit_status(source, 0, &
            '/tmp/ffc_open_close_newunit')
    end function test_open_write_close_newunit

    logical function test_open_write_close_literal_unit()
        ! OPEN with unit=<literal>, WRITE, CLOSE.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  open(unit=20, file='//achar(39)//'/tmp/ffc_open_close_literal.txt'//achar(39)// &
            ', status='//achar(39)//'replace'//achar(39)//')'//new_line('a')// &
            '  write(20, *) 99'//new_line('a')// &
            '  close(20)'//new_line('a')// &
            'end program main'

        test_open_write_close_literal_unit = expect_exit_status(source, 0, &
            '/tmp/ffc_open_close_literal')
    end function test_open_write_close_literal_unit

end program test_session_open_close_file_compiler
