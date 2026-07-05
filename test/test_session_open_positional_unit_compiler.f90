program test_session_open_positional_unit_compiler
    ! OPEN with a positional unit argument (no unit= keyword) plus READ end=
    ! label transfer on end-of-file (#247 file I/O).
    use ffc_test_support, only: expect_output, expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== OPEN positional unit / READ end= compiler test ==='

    all_passed = .true.
    if (.not. test_positional_unit_roundtrip()) all_passed = .false.
    if (.not. test_close_with_trailing_status()) all_passed = .false.
    if (.not. test_read_end_label_on_eof()) all_passed = .false.
    if (.not. test_stdout_sign_plus()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: positional-unit OPEN and READ end= lower correctly'

contains

    logical function test_positional_unit_roundtrip()
        ! open(u, file=...) without a unit= keyword: the first positional
        ! argument is the unit. Write a value, rewind, read it back.
        character(len=*), parameter :: q = achar(39)
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: u, x'//new_line('a')// &
            '  u = 17'//new_line('a')// &
            '  open(u, file='//q//'/tmp/ffc_pos_unit.txt'//q// &
            ', status='//q//'replace'//q//')'//new_line('a')// &
            '  write(u, *) 314'//new_line('a')// &
            '  rewind(u)'//new_line('a')// &
            '  read(u, *) x'//new_line('a')// &
            '  close(u)'//new_line('a')// &
            '  print *, x'//new_line('a')// &
            'end program main'

        test_positional_unit_roundtrip = expect_output(source, &
            '         314'//new_line('a'), '/tmp/ffc_pos_unit')
    end function test_positional_unit_roundtrip

    logical function test_close_with_trailing_status()
        ! close(u, status="delete") carries a trailing keyword; the unit is the
        ! first argument and must still resolve.
        character(len=*), parameter :: q = achar(39)
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: u'//new_line('a')// &
            '  open(newunit=u, file='//q//'/tmp/ffc_close_status.txt'//q// &
            ', status='//q//'replace'//q//')'//new_line('a')// &
            '  write(u, *) 7'//new_line('a')// &
            '  close(u, status='//q//'delete'//q//')'//new_line('a')// &
            '  print *, '//q//'ok'//q//new_line('a')// &
            'end program main'

        test_close_with_trailing_status = expect_output(source, &
            ' ok'//new_line('a'), '/tmp/ffc_close_status')
    end function test_close_with_trailing_status

    logical function test_read_end_label_on_eof()
        ! read(u, *, end=10) jumps to label 10 when the unit is at end-of-file.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: u, x'//new_line('a')// &
            '  open(newunit=u, status='//achar(39)//'scratch'//achar(39)//')'// &
            new_line('a')// &
            '  write(u, *) 5'//new_line('a')// &
            '  rewind(u)'//new_line('a')// &
            '  read(u, *, end=10) x'//new_line('a')// &
            '  if (x /= 5) error stop'//new_line('a')// &
            '  read(u, *, end=10) x'//new_line('a')// &
            '  error stop'//new_line('a')// &
            '10 continue'//new_line('a')// &
            '  close(u)'//new_line('a')// &
            'end program main'

        test_read_end_label_on_eof = expect_exit_status(source, 0, &
            '/tmp/ffc_read_end_label')
    end function test_read_end_label_on_eof

    logical function test_stdout_sign_plus()
        ! open(6, sign='plus') reconfigures the preconnected stdout connection
        ! (#280): a subsequent F-edited PRINT on a non-negative value gets a
        ! forced leading '+', matching gfortran.
        character(len=*), parameter :: q = achar(39)
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  open(6, sign='//q//'plus'//q//')'//new_line('a')// &
            "  print '(F5.1)', 1.0"//new_line('a')// &
            'end program main'

        test_stdout_sign_plus = expect_output(source, ' +1.0'//new_line('a'), &
            '/tmp/ffc_open_sign_plus')
    end function test_stdout_sign_plus

end program test_session_open_positional_unit_compiler
