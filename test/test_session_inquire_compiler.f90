program test_session_inquire_compiler
    ! Minimal INQUIRE: exist=, opened=, and iostat= on file= and unit=.
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session inquire compiler test ==='

    all_passed = .true.
    if (.not. test_inquire_file_exist()) all_passed = .false.
    if (.not. test_inquire_two_files_exist()) all_passed = .false.
    if (.not. test_inquire_unit_opened()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: inquire lowers through direct LIRIC session'

contains

    ! Two file= exist= inquiries in the same program: a regression guard. A
    ! guarded fopen()/fclose() probes existence instead of access(2); two
    ! back-to-back access(2) calls compared against 0 collapsed onto one
    ! shared comparison result in the LIRIC backend.
    logical function test_inquire_file_exist()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  logical :: e1, e2'//new_line('a')// &
            '  integer :: u, ios'//new_line('a')// &
            "  open(newunit=u, file='/tmp/ffc_inquire_exist.dat', status='replace')"// &
            new_line('a')// &
            '  close(u)'//new_line('a')// &
            "  inquire(file='/tmp/ffc_inquire_exist.dat', exist=e1, iostat=ios)"// &
            new_line('a')// &
            "  inquire(file='/tmp/ffc_inquire_missing.dat', exist=e2)"// &
            new_line('a')// &
            '  print *, e1, e2, ios'//new_line('a')// &
            'end program main'

        test_inquire_file_exist = expect_output( &
            source, ' T F           0'//new_line('a'), &
            '/tmp/ffc_session_inquire_exist')
    end function test_inquire_file_exist

    logical function test_inquire_two_files_exist()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  logical :: e1, e2'//new_line('a')// &
            '  integer :: u1, u2'//new_line('a')// &
            "  open(newunit=u1, file='/tmp/ffc_inquire_a.dat', status='replace')"// &
            new_line('a')// &
            '  close(u1)'//new_line('a')// &
            "  open(newunit=u2, file='/tmp/ffc_inquire_b.dat', status='replace')"// &
            new_line('a')// &
            '  close(u2)'//new_line('a')// &
            "  inquire(file='/tmp/ffc_inquire_a.dat', exist=e1)"//new_line('a')// &
            "  inquire(file='/tmp/ffc_inquire_missing2.dat', exist=e2)"// &
            new_line('a')// &
            '  print *, e1, e2'//new_line('a')// &
            'end program main'

        test_inquire_two_files_exist = expect_output( &
            source, ' T F'//new_line('a'), '/tmp/ffc_session_inquire_two_files')
    end function test_inquire_two_files_exist

    logical function test_inquire_unit_opened()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  logical :: before, after'//new_line('a')// &
            '  integer :: u'//new_line('a')// &
            '  u = 33'//new_line('a')// &
            '  inquire(unit=u, opened=before)'//new_line('a')// &
            "  open(u, file='/tmp/ffc_inquire_unit.dat', status='replace')"// &
            new_line('a')// &
            '  inquire(unit=u, opened=after)'//new_line('a')// &
            '  close(u)'//new_line('a')// &
            '  print *, before, after'//new_line('a')// &
            'end program main'

        test_inquire_unit_opened = expect_output( &
            source, ' F T'//new_line('a'), '/tmp/ffc_session_inquire_unit')
    end function test_inquire_unit_opened

end program test_session_inquire_compiler
