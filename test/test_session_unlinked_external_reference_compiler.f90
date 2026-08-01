program test_session_unlinked_external_reference_compiler
    ! #585: gfortran.dg/pr118640.f90 shape. A procedure-only unit whose
    ! specification part declares an interface for a name that is NOT one of
    ! its dummy arguments references an external procedure the linker resolves.
    ! Such a unit compiles on its own (dg-do compile), links and runs once the
    ! definition is supplied, and still fails to link when it is not. The case
    ! therefore belongs to the linked class, not to a noref manifest.
    implicit none

    logical :: all_passed

    print *, '=== unlinked external reference tests ==='

    all_passed = .true.
    if (.not. test_compile_only_unit_accepts_undefined_external()) &
        all_passed = .false.
    if (.not. test_external_call_links_and_runs()) all_passed = .false.
    if (.not. test_missing_definition_fails_to_link()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: unlinked external references'

contains

    function caller_source() result(src)
        ! An external subroutine that calls an interfaced external function
        ! from inside a PRESENT guard, as pr118640.f90 does.
        character(len=:), allocatable :: src

        src = 'subroutine baz(x, y)'//new_line('a')// &
              '    integer, intent(in) :: y'//new_line('a')// &
              '    integer, pointer, optional :: x'//new_line('a')// &
              '    interface'//new_line('a')// &
              '        function qux(x)'//new_line('a')// &
              '            integer x'//new_line('a')// &
              '            integer qux'//new_line('a')// &
              '        end function'//new_line('a')// &
              '    end interface'//new_line('a')// &
              '    if (present(x)) then'//new_line('a')// &
              '        x = qux(y)'//new_line('a')// &
              '    end if'//new_line('a')// &
              'end subroutine baz'
    end function caller_source

    function driver_source() result(src)
        character(len=:), allocatable :: src

        src = 'program main'//new_line('a')// &
              '    implicit none'//new_line('a')// &
              '    integer, pointer :: p'//new_line('a')// &
              '    interface'//new_line('a')// &
              '        subroutine baz(x, y)'//new_line('a')// &
              '            integer, intent(in) :: y'//new_line('a')// &
              '            integer, pointer, optional :: x'//new_line('a')// &
              '        end subroutine baz'//new_line('a')// &
              '    end interface'//new_line('a')// &
              '    allocate (p)'//new_line('a')// &
              '    p = 0'//new_line('a')// &
              '    call baz(p, 41)'//new_line('a')// &
              '    stop p'//new_line('a')// &
              'end program main'
    end function driver_source

    logical function test_compile_only_unit_accepts_undefined_external() &
        result(ok)
        character(len=*), parameter :: src = '/tmp/ffc_ext585_caller.f90'
        character(len=*), parameter :: obj = '/tmp/ffc_ext585_caller.o'
        integer :: exit_stat, cmd_stat

        ok = .false.
        if (.not. write_file(src, caller_source())) return
        call execute_command_line('rm -f '//obj)
        exit_stat = run_ffc('-c '//src//' -o '//obj, cmd_stat)
        if (cmd_stat /= 0 .or. exit_stat /= 0) then
            print *, 'FAIL: compile-only unit rejected, code ', exit_stat
            return
        end if
        ok = .true.
    end function test_compile_only_unit_accepts_undefined_external

    logical function test_external_call_links_and_runs() result(ok)
        character(len=*), parameter :: caller = '/tmp/ffc_ext585_caller.f90'
        character(len=*), parameter :: defsrc = '/tmp/ffc_ext585_def.f90'
        character(len=*), parameter :: drvsrc = '/tmp/ffc_ext585_drv.f90'
        character(len=*), parameter :: cobj = '/tmp/ffc_ext585_caller.o'
        character(len=*), parameter :: dobj = '/tmp/ffc_ext585_def.o'
        character(len=*), parameter :: exe = '/tmp/ffc_ext585_drv'
        integer :: exit_stat, cmd_stat

        ok = .false.
        if (.not. write_file(caller, caller_source())) return
        if (.not. write_file(defsrc, &
            'function qux(x)'//new_line('a')// &
            '    integer x'//new_line('a')// &
            '    integer qux'//new_line('a')// &
            '    qux = x + 1'//new_line('a')// &
            'end function qux')) return
        if (.not. write_file(drvsrc, driver_source())) return
        call execute_command_line('rm -f '//cobj//' '//dobj//' '//exe)
        exit_stat = run_ffc('-c '//caller//' -o '//cobj, cmd_stat)
        if (cmd_stat /= 0 .or. exit_stat /= 0) then
            print *, 'FAIL: caller object build failed, code ', exit_stat
            return
        end if
        exit_stat = run_ffc('-c '//defsrc//' -o '//dobj, cmd_stat)
        if (cmd_stat /= 0 .or. exit_stat /= 0) then
            print *, 'FAIL: definition object build failed, code ', exit_stat
            return
        end if
        exit_stat = run_ffc(drvsrc//' '//cobj//' '//dobj//' -o '//exe, cmd_stat)
        if (cmd_stat /= 0 .or. exit_stat /= 0) then
            print *, 'FAIL: link with the external definition failed, code ', &
                exit_stat
            return
        end if
        call execute_command_line(exe//' > /dev/null 2>&1', &
                                  exitstat=exit_stat, cmdstat=cmd_stat)
        if (cmd_stat /= 0) then
            print *, 'FAIL: could not run the linked driver'
            return
        end if
        if (exit_stat /= 42) then
            print *, 'FAIL: expected exit 42 from qux(41), got ', exit_stat
            return
        end if
        ok = .true.
    end function test_external_call_links_and_runs

    logical function test_missing_definition_fails_to_link() result(ok)
        ! Negative control: leaving the external undefined must still fail the
        ! link, so the accepting compile above is not silently dropping a call.
        character(len=*), parameter :: cobj = '/tmp/ffc_ext585_caller.o'
        character(len=*), parameter :: drvsrc = '/tmp/ffc_ext585_drv.f90'
        character(len=*), parameter :: exe = '/tmp/ffc_ext585_bad'
        integer :: exit_stat, cmd_stat

        ok = .false.
        if (.not. write_file(drvsrc, driver_source())) return
        call execute_command_line('rm -f '//exe)
        exit_stat = run_ffc(drvsrc//' '//cobj//' -o '//exe, cmd_stat)
        if (cmd_stat /= 0) then
            print *, 'FAIL: could not run the link attempt'
            return
        end if
        if (exit_stat == 0) then
            print *, 'FAIL: link succeeded without a definition for qux'
            return
        end if
        call execute_command_line('rm -f /tmp/ffc_ext585_*')
        ok = .true.
    end function test_missing_definition_fails_to_link

    integer function run_ffc(args, cmd_stat) result(exit_stat)
        character(len=*), intent(in) :: args
        integer, intent(out) :: cmd_stat

        exit_stat = 0
        call execute_command_line( &
            "sh -c 'exe=$(ls -t build/*/app/ffc build/fo/bin/ffc 2>/dev/null | "// &
            'head -n 1); test -n "$exe" || exit 90; exec "$exe" '//args//"'"// &
            ' > /dev/null 2>&1', exitstat=exit_stat, cmdstat=cmd_stat)
    end function run_ffc

    logical function write_file(path, contents) result(ok)
        character(len=*), intent(in) :: path
        character(len=*), intent(in) :: contents
        integer :: unit, io_stat

        ok = .false.
        open (newunit=unit, file=path, status='replace', action='write', &
              iostat=io_stat)
        if (io_stat /= 0) then
            print *, 'FAIL: could not write ', path
            return
        end if
        write (unit, '(A)') contents
        close (unit)
        ok = .true.
    end function write_file

end program test_session_unlinked_external_reference_compiler
