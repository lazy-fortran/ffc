program test_session_iostat_compiler
    ! iostat= on file-unit READ and WRITE: 0 on success, -1 at end-of-file,
    ! matching gfortran's iostat_end.
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session iostat compiler test ==='

    all_passed = .true.
    if (.not. test_read_iostat_success_and_eof()) all_passed = .false.
    if (.not. test_write_iostat_success()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: iostat lowers through direct LIRIC session'

contains

    logical function test_read_iostat_success_and_eof()
        ! First read succeeds (iostat 0); the second hits EOF (iostat -1).
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: u, v, ios'//new_line('a')// &
            '  open(newunit=u, status='//achar(39)//'scratch'//achar(39)//')'// &
            new_line('a')// &
            '  write(u, *) 7'//new_line('a')// &
            '  rewind(u)'//new_line('a')// &
            '  read(u, *, iostat=ios) v'//new_line('a')// &
            '  print *, v, ios'//new_line('a')// &
            '  read(u, *, iostat=ios) v'//new_line('a')// &
            '  print *, ios'//new_line('a')// &
            '  close(u)'//new_line('a')// &
            'end program main'

        test_read_iostat_success_and_eof = expect_output(source, &
            repeat(' ', 11)//'7'//repeat(' ', 11)//'0'//new_line('a')// &
            repeat(' ', 10)//'-1'//new_line('a'), &
            '/tmp/ffc_iostat_read')
    end function test_read_iostat_success_and_eof

    logical function test_write_iostat_success()
        ! A WRITE to a connected unit succeeds: iostat is set to 0 even when the
        ! variable held a nonzero value before.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: u, ios'//new_line('a')// &
            '  open(newunit=u, status='//achar(39)//'scratch'//achar(39)//')'// &
            new_line('a')// &
            '  ios = 99'//new_line('a')// &
            '  write(u, *, iostat=ios) 42'//new_line('a')// &
            '  print *, ios'//new_line('a')// &
            '  close(u)'//new_line('a')// &
            'end program main'

        test_write_iostat_success = expect_output(source, &
            repeat(' ', 11)//'0'//new_line('a'), '/tmp/ffc_iostat_write')
    end function test_write_iostat_success

end program test_session_iostat_compiler
