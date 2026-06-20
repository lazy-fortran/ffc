program test_session_scratch_unit_compiler
    ! OPEN with status='scratch' (no file=) round-trips through a tmpfile()
    ! handle: WRITE, REWIND, READ back, and a redundant CLOSE must not abort.
    use ffc_test_support, only: expect_output, expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== direct session scratch-unit compiler test ==='

    all_passed = .true.
    if (.not. test_scratch_roundtrip()) all_passed = .false.
    if (.not. test_double_close_is_noop()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: scratch units lower through direct LIRIC session'

contains

    logical function test_scratch_roundtrip()
        ! Write to a scratch unit, rewind, read back, print the value.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: u, v'//new_line('a')// &
            '  open(newunit=u, status='//achar(39)//'scratch'//achar(39)//')'// &
            new_line('a')// &
            '  write(u, *) 123'//new_line('a')// &
            '  rewind(u)'//new_line('a')// &
            '  read(u, *) v'//new_line('a')// &
            '  print *, v'//new_line('a')// &
            '  close(u)'//new_line('a')// &
            'end program main'

        test_scratch_roundtrip = expect_output(source, &
            repeat(' ', 9)//'123'//new_line('a'), '/tmp/ffc_scratch_roundtrip')
    end function test_scratch_roundtrip

    logical function test_double_close_is_noop()
        ! Closing the same scratch unit twice must be a no-op, not an abort.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: u'//new_line('a')// &
            '  u = 17'//new_line('a')// &
            '  open(unit=u, status='//achar(39)//'scratch'//achar(39)//')'// &
            new_line('a')// &
            '  close(u)'//new_line('a')// &
            '  close(u)'//new_line('a')// &
            'end program main'

        test_double_close_is_noop = expect_exit_status(source, 0, &
            '/tmp/ffc_scratch_double_close')
    end function test_double_close_is_noop

end program test_session_scratch_unit_compiler
