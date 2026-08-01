! Issue #427: IOSTAT= and IOMSG= report stable, documented values.
!
! Behavioral oracle. Before #427 a WRITE's IOSTAT= was assigned a literal 0
! whatever happened, OPEN had no IOSTAT= or IOMSG= at all, and IOMSG= was
! never written by any statement. The status classes and the message text
! now come from one place in the runtime, so every statement reports the
! same value for the same condition.
!
!   1. OPEN on a file that cannot be opened reports a nonzero status and a
!      message; the same OPEN on a usable file reports 0 and blanks. Both
!      fail on the pre-#427 compiler, which ignored the specifiers.
!   2. READ reports 0 with a blank message, then -1 with "End of file",
!      distinguishing end of file from an error.
!   3. IOMSG= obeys Fortran character assignment: truncated to the
!      destination length, blank padded when shorter.
!   4. A noninteger IOSTAT= and a noncharacter IOMSG= are still rejected.
program test_session_iostat_iomsg_compiler
    use ffc_test_support, only: expect_output, expect_error_contains
    implicit none

    character(len=*), parameter :: Q = achar(39)
    logical :: all_passed

    print *, '=== IOSTAT and IOMSG tests (#427) ==='

    all_passed = .true.
    if (.not. test_open_failure_and_success()) all_passed = .false.
    if (.not. test_read_end_of_file()) all_passed = .false.
    if (.not. test_iomsg_is_truncated()) all_passed = .false.
    if (.not. test_noninteger_iostat_is_rejected()) all_passed = .false.
    if (.not. test_noncharacter_iomsg_is_rejected()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: IOSTAT and IOMSG are stable'

contains

    ! A missing file is an error with a message; a usable file is not.
    logical function test_open_failure_and_success() result(ok)
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: ios'//new_line('a')// &
            '  character(len=20) :: msg'//new_line('a')// &
            '  open(unit=11, file='//Q//'/nonexistent-427/x.dat'//Q// &
            ', status='//Q//'old'//Q//', iostat=ios, iomsg=msg)'// &
            new_line('a')// &
            '  print *, ios'//new_line('a')// &
            '  print *, msg'//new_line('a')// &
            '  open(unit=12, file='//Q//'/tmp/ffc_427_ok.dat'//Q// &
            ', status='//Q//'replace'//Q//', iostat=ios, iomsg=msg)'// &
            new_line('a')// &
            '  print *, ios'//new_line('a')// &
            '  print *, msg'//new_line('a')// &
            '  close(12)'//new_line('a')// &
            'end program main'

        ok = expect_output(source, &
            '        5004'//new_line('a')// &
            ' Cannot open file    '//new_line('a')// &
            '           0'//new_line('a')// &
            '                     '//new_line('a'), &
            '/tmp/ffc_iostat_open')
    end function test_open_failure_and_success

    ! End of file is its own class, not a generic error and not success.
    logical function test_read_end_of_file() result(ok)
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: u, ios, n'//new_line('a')// &
            '  character(len=15) :: msg'//new_line('a')// &
            '  open(newunit=u, file='//Q//'/tmp/ffc_427_eof.dat'//Q// &
            ', status='//Q//'replace'//Q//')'//new_line('a')// &
            '  write(u, *) 7'//new_line('a')// &
            '  rewind(u)'//new_line('a')// &
            '  read(u, *, iostat=ios, iomsg=msg) n'//new_line('a')// &
            '  print *, ios, n'//new_line('a')// &
            '  print *, msg'//new_line('a')// &
            '  read(u, *, iostat=ios, iomsg=msg) n'//new_line('a')// &
            '  print *, ios'//new_line('a')// &
            '  print *, msg'//new_line('a')// &
            '  close(u)'//new_line('a')// &
            'end program main'

        ok = expect_output(source, &
            '           0           7'//new_line('a')// &
            '                '//new_line('a')// &
            '          -1'//new_line('a')// &
            ' End of file    '//new_line('a'), &
            '/tmp/ffc_iostat_eof')
    end function test_read_end_of_file

    ! Fortran character assignment, not a C string: too long is truncated.
    logical function test_iomsg_is_truncated() result(ok)
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: ios'//new_line('a')// &
            '  character(len=6) :: msg'//new_line('a')// &
            '  open(unit=13, file='//Q//'/nonexistent-427/y.dat'//Q// &
            ', status='//Q//'old'//Q//', iostat=ios, iomsg=msg)'// &
            new_line('a')// &
            '  print *, msg'//new_line('a')// &
            'end program main'

        ok = expect_output(source, ' Cannot'//new_line('a'), &
                           '/tmp/ffc_iostat_trunc')
    end function test_iomsg_is_truncated

    logical function test_noninteger_iostat_is_rejected() result(ok)
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real :: ios'//new_line('a')// &
            '  open(unit=14, file='//Q//'/tmp/ffc_427_r.dat'//Q// &
            ', iostat=ios)'//new_line('a')// &
            'end program main'

        ok = expect_error_contains(source, 'default integer', &
                                   '/tmp/ffc_iostat_badtype')
    end function test_noninteger_iostat_is_rejected

    logical function test_noncharacter_iomsg_is_rejected() result(ok)
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: ios'//new_line('a')// &
            '  integer :: msg'//new_line('a')// &
            '  open(unit=15, file='//Q//'/tmp/ffc_427_m.dat'//Q// &
            ', iostat=ios, iomsg=msg)'//new_line('a')// &
            'end program main'

        ! FortFront rejects this before lowering sees it; the compiler's own
        ! check on the IOMSG target is the backstop behind that.
        ok = expect_error_contains(source, 'type CHARACTER', &
                                   '/tmp/ffc_iomsg_badtype')
    end function test_noncharacter_iomsg_is_rejected

end program test_session_iostat_iomsg_compiler
