! Issue #396: file-unit state lives in the runtime, not in emitted globals.
!
! Behavioral oracle. Two halves, because the change has two consequences.
!
! The runtime half drives `_ffc_unit_*` directly from C and pins the status
! codes IOSTAT= will report: opening a connected unit, operating on a unit
! number outside the supported range, closing a unit that was never
! connected, and NEWUNIT= reuse after CLOSE. These are contract values, so
! the test names the numbers rather than only checking "nonzero".
!
! The compiler half proves the state really moved. Before #396 a unit's FILE*
! lived in a stack slot in the function that opened it, so a unit opened in a
! procedure was gone once that procedure returned: the cross-scope program
! below prints 0 on the pre-#396 compiler and 64 here. The computed-unit
! program is a regression guard rather than a discriminator; it passed before
! and must keep passing now that the unit number is read at run time.
program test_session_unit_runtime_compiler
    use ffc_runtime_link, only: ffc_runtime_link_input
    use ffc_test_support, only: expect_output
    implicit none

    character(len=*), parameter :: WORK = '/tmp/ffc_unit_runtime_396'
    logical :: all_passed

    print *, '=== runtime file-unit state tests (#396) ==='

    call run_quiet('rm -rf '//WORK//' && mkdir -p '//WORK)

    all_passed = .true.
    if (.not. test_runtime_status_codes()) all_passed = .false.
    if (.not. test_runtime_computed_unit_number()) all_passed = .false.
    if (.not. test_unit_survives_the_opening_scope()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: file-unit state is owned by the runtime'

contains

    subroutine run_quiet(command)
        character(len=*), intent(in) :: command
        integer :: exit_stat, cmd_stat

        call execute_command_line(command, exitstat=exit_stat, &
                                  cmdstat=cmd_stat)
    end subroutine run_quiet

    integer function status_of(command) result(status)
        character(len=*), intent(in) :: command
        character(len=*), parameter :: rc_file = WORK//'/rc'
        integer :: unit, ios, exit_stat, cmd_stat

        status = -1
        call execute_command_line('{ '//command//' ; } > '//WORK// &
                                  '/out 2>&1; echo $? > '//rc_file, &
                                  exitstat=exit_stat, cmdstat=cmd_stat)
        if (cmd_stat /= 0) return
        open (newunit=unit, file=rc_file, status='old', iostat=ios)
        if (ios /= 0) return
        read (unit, *, iostat=ios) status
        close (unit)
        if (ios /= 0) status = -1
    end function status_of

    ! Drives the runtime entry points from C. Each failing check exits with
    ! its own number, so a failure says which contract broke.
    logical function test_runtime_status_codes() result(ok)
        character(len=*), parameter :: driver = WORK//'/unit_driver.c'
        character(len=*), parameter :: exe = WORK//'/unit_driver'
        character(len=:), allocatable :: link_input, error_msg
        integer :: unit, ios, status

        ok = .false.
        call ffc_runtime_link_input(link_input, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL: runtime link input: ', trim(error_msg)
            return
        end if

        open (newunit=unit, file=driver, status='replace', iostat=ios)
        if (ios /= 0) then
            print *, 'FAIL: cannot write the runtime driver'
            return
        end if
        write (unit, '(a)') '#include <stdio.h>'
        write (unit, '(a)') 'int _ffc_unit_open(int, const char *,'
        write (unit, '(a)') '                   const char *);'
        write (unit, '(a)') 'int _ffc_unit_close(int);'
        write (unit, '(a)') 'int _ffc_unit_rewind(int);'
        write (unit, '(a)') 'int _ffc_unit_is_open(int);'
        write (unit, '(a)') 'int _ffc_unit_newunit(void);'
        write (unit, '(a)') 'int _ffc_unit_status(void);'
        write (unit, '(a)') 'FILE *_ffc_unit_file(int);'
        write (unit, '(a)') 'int main(void) {'
        write (unit, '(a)') '    int first, second;'
        write (unit, '(a)') '    const char *p = "'//WORK//'/a.dat";'
        write (unit, '(a)') '    if (_ffc_unit_is_open(11)) return 1;'
        write (unit, '(a)') '    if (_ffc_unit_open(11, p, "replace"))'
        write (unit, '(a)') '        return 2;'
        write (unit, '(a)') '    if (!_ffc_unit_is_open(11)) return 3;'
        write (unit, '(a)') '    /* a second OPEN on a connected unit */'
        write (unit, '(a)') '    if (_ffc_unit_open(11, p, "replace")'
        write (unit, '(a)') '        != 5003) return 4;'
        write (unit, '(a)') '    if (_ffc_unit_status() != 5003) return 5;'
        write (unit, '(a)') '    if (_ffc_unit_close(11)) return 6;'
        write (unit, '(a)') '    if (_ffc_unit_is_open(11)) return 7;'
        write (unit, '(a)') '    /* CLOSE of an unconnected unit succeeds */'
        write (unit, '(a)') '    if (_ffc_unit_close(11)) return 8;'
        write (unit, '(a)') '    /* an out-of-range unit is rejected */'
        write (unit, '(a)') '    if (_ffc_unit_close(999999) != 5001)'
        write (unit, '(a)') '        return 9;'
        write (unit, '(a)') '    if (_ffc_unit_file(999999) != NULL)'
        write (unit, '(a)') '        return 10;'
        write (unit, '(a)') '    if (_ffc_unit_status() != 5001) return 11;'
        write (unit, '(a)') '    if (_ffc_unit_rewind(999999) != 5001)'
        write (unit, '(a)') '        return 12;'
        write (unit, '(a)') '    /* NEWUNIT hands out a free unit, and the'
        write (unit, '(a)') '       same one again once it is released */'
        write (unit, '(a)') '    first = _ffc_unit_newunit();'
        write (unit, '(a)') '    if (first < 0) return 13;'
        write (unit, '(a)') '    if (_ffc_unit_open(first, p, "replace"))'
        write (unit, '(a)') '        return 14;'
        write (unit, '(a)') '    second = _ffc_unit_newunit();'
        write (unit, '(a)') '    if (second == first) return 15;'
        write (unit, '(a)') '    if (_ffc_unit_close(first)) return 16;'
        write (unit, '(a)') '    if (_ffc_unit_newunit() != first) return 17;'
        write (unit, '(a)') '    return 0;'
        write (unit, '(a)') '}'
        close (unit)

        status = status_of('cc -o '//exe//' '//driver//' '//link_input)
        if (status /= 0) then
            print *, 'FAIL: the runtime unit driver does not build'
            return
        end if
        status = status_of(exe)
        if (status /= 0) then
            print *, 'FAIL: runtime unit contract check ', status, ' failed'
            return
        end if
        ok = .true.
    end function test_runtime_status_codes

    ! The unit number is only known at run time. Regression guard for the
    ! switch from a compile-time-resolved slot to a run-time unit number.
    logical function test_runtime_computed_unit_number() result(ok)
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: u, n'//new_line('a')// &
            '  u = 17 + command_argument_count()'//new_line('a')// &
            '  open(unit=u, file=''/tmp/ffc_unit_396_a.dat'', '// &
            'status=''replace'')'//new_line('a')// &
            '  write(u, *) 31'//new_line('a')// &
            '  rewind(u)'//new_line('a')// &
            '  read(u, *) n'//new_line('a')// &
            '  close(u)'//new_line('a')// &
            '  print *, n'//new_line('a')// &
            'end program main'

        ok = expect_output(source, '          31'//new_line('a'), &
                           WORK//'/computed_unit')
    end function test_runtime_computed_unit_number

    ! A unit opened inside a procedure is still connected after that
    ! procedure returns, because the connection belongs to the process.
    logical function test_unit_survives_the_opening_scope() result(ok)
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: n'//new_line('a')// &
            '  call fill()'//new_line('a')// &
            '  rewind(23)'//new_line('a')// &
            '  read(23, *) n'//new_line('a')// &
            '  close(23)'//new_line('a')// &
            '  print *, n'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine fill()'//new_line('a')// &
            '    open(unit=23, file=''/tmp/ffc_unit_396_b.dat'', '// &
            'status=''replace'')'//new_line('a')// &
            '    write(23, *) 64'//new_line('a')// &
            '  end subroutine fill'//new_line('a')// &
            'end program main'

        ok = expect_output(source, '          64'//new_line('a'), &
                           WORK//'/cross_scope')
    end function test_unit_survives_the_opening_scope

end program test_session_unit_runtime_compiler
