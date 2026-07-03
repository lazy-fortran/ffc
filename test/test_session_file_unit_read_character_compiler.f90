program test_session_file_unit_read_character_compiler
    ! List-directed READ of a fixed-length character scalar from an opened
    ! file unit, and OPEN with no status= preserving an existing file's
    ! content (gfortran's default STATUS='UNKNOWN') instead of truncating it.
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session file-unit character read compiler test ==='

    all_passed = .true.
    if (.not. test_read_character_token()) all_passed = .false.
    if (.not. test_open_no_status_preserves_content()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: file-unit character read lowers through direct LIRIC session'

contains

    logical function test_read_character_token()
        character(len=*), parameter :: data_path = '/tmp/ffc_furc_a.dat'
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: u, n'//new_line('a')// &
            '  character(len=10) :: name'//new_line('a')// &
            "  open(newunit=u, file='"//data_path//"')"//new_line('a')// &
            '  read(u, *) n, name'//new_line('a')// &
            '  close(u)'//new_line('a')// &
            "  print '(I0,A,A,A)', n, ' [', trim(name), ']'"//new_line('a')// &
            'end program main'
        integer :: data_unit

        ! Data file exists on disk before ffc's generated program runs; the
        ! file-unit character WRITE path is a separate, unimplemented feature.
        open (newunit=data_unit, file=data_path, status='replace')
        write (data_unit, '(A)') '7 Alice'
        close (data_unit)

        test_read_character_token = expect_output( &
            source, '7 [Alice]'//new_line('a'), '/tmp/ffc_session_furc_token')
    end function test_read_character_token

    ! OPEN a unit twice on the same file: the second OPEN (no status=) must
    ! see the first connection's written content rather than truncating it
    ! via fopen's "w+" mode (gfortran's default STATUS='UNKNOWN' preserves an
    ! existing file).
    logical function test_open_no_status_preserves_content()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: u1, u2, n'//new_line('a')// &
            "  open(newunit=u1, file='/tmp/ffc_furc_preserve.dat', "// &
            "status='replace')"//new_line('a')// &
            '  write(u1, *) 99'//new_line('a')// &
            '  close(u1)'//new_line('a')// &
            "  open(newunit=u2, file='/tmp/ffc_furc_preserve.dat')"// &
            new_line('a')// &
            '  read(u2, *) n'//new_line('a')// &
            '  close(u2)'//new_line('a')// &
            '  print *, n'//new_line('a')// &
            'end program main'

        test_open_no_status_preserves_content = expect_output( &
            source, '          99'//new_line('a'), &
            '/tmp/ffc_session_furc_preserve')
    end function test_open_no_status_preserves_content

end program test_session_file_unit_read_character_compiler
