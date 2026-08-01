! Issue #423: scalar formatted output goes through the runtime ABI.
!
! Behavioral oracle for the migration itself. Byte stability of the output
! is already covered by test_session_formatted_print_compiler and the
! conformance corpora; what those cannot see is which convention produced
! the bytes. That is what this pins.
!
!   1. The lowering really changed. An emitted program's symbols must name
!      `_ffc_write_*` and must not name `printf`: before #423 the compiler
!      emitted a direct variadic `printf` for every scalar output item.
!      Both halves fail on the pre-#423 compiler, and the `printf` half
!      would fail again if a second, parallel convention were reintroduced.
!   2. The runtime owns the status. A unit number outside the supported
!      range is reported with the documented code rather than written to.
program test_session_print_runtime_compiler
    use ffc_runtime_link, only: ffc_runtime_link_input
    use ffc_test_support, only: compile_to_exe
    implicit none

    character(len=*), parameter :: WORK = '/tmp/ffc_print_runtime_423'
    logical :: all_passed

    print *, '=== runtime scalar output tests (#423) ==='

    call run_quiet('rm -rf '//WORK//' && mkdir -p '//WORK)

    all_passed = .true.
    if (.not. test_lowering_calls_the_runtime()) all_passed = .false.
    if (.not. test_bad_unit_status()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: scalar output is owned by the runtime'

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

    subroutine read_text_file(path, text, ok)
        character(len=*), intent(in) :: path
        character(len=:), allocatable, intent(out) :: text
        logical, intent(out) :: ok
        integer :: unit, ios
        integer(kind=8) :: nbytes

        ok = .false.
        open (newunit=unit, file=path, access='stream', form='unformatted', &
              status='old', action='read', iostat=ios)
        if (ios /= 0) then
            text = ''
            return
        end if
        inquire (unit=unit, size=nbytes)
        if (nbytes <= 0) then
            close (unit)
            text = ''
            return
        end if
        allocate (character(len=nbytes) :: text)
        read (unit, iostat=ios) text
        close (unit)
        ok = ios == 0
    end subroutine read_text_file

    logical function test_lowering_calls_the_runtime() result(ok)
        character(len=*), parameter :: exe = WORK//'/symbols'
        character(len=*), parameter :: nm_out = WORK//'/symbols.nm'
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: n'//new_line('a')// &
            '  n = 3'//new_line('a')// &
            '  print *, n'//new_line('a')// &
            'end program main'
        character(len=:), allocatable :: error_msg, symbols
        logical :: read_ok
        integer :: status

        ok = .false.
        call compile_to_exe(source, exe, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL: compiling the probe program: ', trim(error_msg)
            return
        end if
        status = status_of('nm '//exe//' > '//nm_out)
        if (status /= 0) then
            print *, 'FAIL: cannot list the emitted symbols'
            return
        end if
        call read_text_file(nm_out, symbols, read_ok)
        if (.not. read_ok) then
            print *, 'FAIL: cannot read the emitted symbol list'
            return
        end if
        if (index(symbols, '_ffc_write_i32') == 0) then
            print *, 'FAIL: scalar output does not call the runtime'
            return
        end if
        if (index(symbols, '_ffc_write_text') == 0) then
            print *, 'FAIL: the record terminator does not call the runtime'
            return
        end if
        ! A leftover direct printf call would mean two conventions running
        ! in parallel. The symbol may still be *declared* by a session that
        ! also prepares the not-yet-migrated complex path, so this looks for
        ! a call site rather than a symbol-table entry.
        status = status_of('objdump -d '//exe// &
                           ' | grep -q "call.*<printf@plt>"')
        if (status == 0) then
            print *, 'FAIL: the emitted program still calls printf directly'
            return
        end if
        ok = .true.
    end function test_lowering_calls_the_runtime

    logical function test_bad_unit_status() result(ok)
        character(len=*), parameter :: driver = WORK//'/write_driver.c'
        character(len=*), parameter :: exe = WORK//'/write_driver'
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
            print *, 'FAIL: cannot write the output driver'
            return
        end if
        write (unit, '(a)') 'int _ffc_write_i32(int, const char *, int);'
        write (unit, '(a)') 'int _ffc_write_f64(int, const char *, double);'
        write (unit, '(a)') 'int _ffc_write_text(int, const char *);'
        write (unit, '(a)') 'int _ffc_unit_status(void);'
        write (unit, '(a)') 'int main(void) {'
        write (unit, '(a)') '    /* out of range: reported, not written */'
        write (unit, '(a)') '    if (_ffc_write_i32(999999, "%d", 1)'
        write (unit, '(a)') '        != 5001) return 1;'
        write (unit, '(a)') '    if (_ffc_unit_status() != 5001) return 2;'
        write (unit, '(a)') '    if (_ffc_write_f64(999999, "%f", 1.0)'
        write (unit, '(a)') '        != 5001) return 3;'
        write (unit, '(a)') '    if (_ffc_write_text(999999, "x") != 5001)'
        write (unit, '(a)') '        return 4;'
        write (unit, '(a)') '    /* the preconnected unit still works */'
        write (unit, '(a)') '    if (_ffc_write_text(6, "") != 0) return 5;'
        write (unit, '(a)') '    if (_ffc_unit_status() != 0) return 6;'
        write (unit, '(a)') '    return 0;'
        write (unit, '(a)') '}'
        close (unit)

        status = status_of('cc -o '//exe//' '//driver//' '//link_input)
        if (status /= 0) then
            print *, 'FAIL: the output driver does not build'
            return
        end if
        status = status_of(exe)
        if (status /= 0) then
            print *, 'FAIL: output status check ', status, ' failed'
            return
        end if
        ok = .true.
    end function test_bad_unit_status

end program test_session_print_runtime_compiler
