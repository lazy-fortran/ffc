program test_session_file_unit_io
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session file-unit I/O compiler test ==='

    all_passed = .true.
    if (.not. test_newunit_roundtrip()) all_passed = .false.
    if (.not. test_variable_unit_literal_write()) all_passed = .false.
    if (.not. test_implicit_unit_connection()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: file-unit write/rewind/read round-trips match gfortran'

contains

    ! OPEN(newunit=) then write two records, rewind, read them back. Exercises
    ! list-directed file write and read for integer and real scalars.
    logical function test_newunit_roundtrip()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: u, n'//new_line('a')// &
            '  real :: x'//new_line('a')// &
            '  open(newunit=u, file=''/tmp/ffc_fui_a.dat'', status=''replace'')'// &
            new_line('a')// &
            '  write(u, *) 7'//new_line('a')// &
            '  write(u, *) 2.5'//new_line('a')// &
            '  rewind(u)'//new_line('a')// &
            '  read(u, *) n'//new_line('a')// &
            '  read(u, *) x'//new_line('a')// &
            '  close(u)'//new_line('a')// &
            '  print *, n'//new_line('a')// &
            '  print *, x'//new_line('a')// &
            'end program main'

        test_newunit_roundtrip = expect_output( &
            source, &
            '           7'//new_line('a')// &
            '   2.50000000    '//new_line('a'), &
            '/tmp/ffc_session_file_unit_newunit')
    end function test_newunit_roundtrip

    ! OPEN by a variable unit number, WRITE by the literal number, rewind, read
    ! back by the variable. The literal and the variable must resolve to the
    ! same connection.
    logical function test_variable_unit_literal_write()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: u, n'//new_line('a')// &
            '  u = 17'//new_line('a')// &
            '  open(unit=u, file=''/tmp/ffc_fui_b.dat'', status=''unknown'')'// &
            new_line('a')// &
            '  write(17, ''(I0)'') 42'//new_line('a')// &
            '  rewind(u)'//new_line('a')// &
            '  read(u, *) n'//new_line('a')// &
            '  close(u)'//new_line('a')// &
            '  print *, n'//new_line('a')// &
            'end program main'

        test_variable_unit_literal_write = expect_output( &
            source, '          42'//new_line('a'), &
            '/tmp/ffc_session_file_unit_varlit')
    end function test_variable_unit_literal_write

    ! A numeric unit used without OPEN is implicitly preconnected to fort.<N>.
    logical function test_implicit_unit_connection()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: n'//new_line('a')// &
            '  write(21, *) 99'//new_line('a')// &
            '  rewind(21)'//new_line('a')// &
            '  read(21, *) n'//new_line('a')// &
            '  print *, n'//new_line('a')// &
            'end program main'

        test_implicit_unit_connection = expect_output( &
            source, '          99'//new_line('a'), &
            '/tmp/ffc_session_file_unit_implicit')
    end function test_implicit_unit_connection

end program test_session_file_unit_io
