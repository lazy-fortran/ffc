program test_session_program_units_compiler
    use ffc_test_support, only: expect_exit_status, expect_exe_has_symbol
    implicit none

    ! #275: a source file whose only units are modules (no main program) is a
    ! valid translation unit. Each module's procedures are emitted under their
    ! mangled symbols, and the unit as a whole lowers to a no-op main so it both
    ! compiles to an object with -c and links to a runnable (empty) executable,
    ! matching gfortran's own module-only object. Previously ffc rejected it with
    ! "modules as USE targets, not as top-level program units".
    character(len=*), parameter :: two_modules = &
        'module m_a'//new_line('a')// &
        '  implicit none'//new_line('a')// &
        '  integer, parameter :: base = 10'//new_line('a')// &
        'contains'//new_line('a')// &
        '  integer function bump(n)'//new_line('a')// &
        '    integer, intent(in) :: n'//new_line('a')// &
        '    bump = n + base'//new_line('a')// &
        '  end function bump'//new_line('a')// &
        'end module m_a'//new_line('a')// &
        'module m_b'//new_line('a')// &
        '  implicit none'//new_line('a')// &
        'contains'//new_line('a')// &
        '  integer function tripled(n)'//new_line('a')// &
        '    integer, intent(in) :: n'//new_line('a')// &
        '    tripled = 3 * n'//new_line('a')// &
        '  end function tripled'//new_line('a')// &
        'end module m_b'
    logical :: all_passed

    print *, '=== top-level program units compiler test ==='

    all_passed = .true.
    if (.not. test_multi_module_emits_first_procedure()) all_passed = .false.
    if (.not. test_multi_module_emits_second_procedure()) all_passed = .false.
    if (.not. test_multi_module_runs_as_noop_exe()) all_passed = .false.
    if (.not. test_multi_module_object_links_via_constant()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: a program-less file of several modules lowers (#275)'

contains

    logical function test_multi_module_emits_first_procedure() result(ok)
        ok = expect_exe_has_symbol(two_modules, &
            '/tmp/ffc_program_units_a.o', '__m_a_MOD_bump')
    end function test_multi_module_emits_first_procedure

    logical function test_multi_module_emits_second_procedure() result(ok)
        ok = expect_exe_has_symbol(two_modules, &
            '/tmp/ffc_program_units_b.o', '__m_b_MOD_tripled')
    end function test_multi_module_emits_second_procedure

    logical function test_multi_module_runs_as_noop_exe() result(ok)
        ok = expect_exit_status(two_modules, 0, '/tmp/ffc_program_units_exe')
    end function test_multi_module_runs_as_noop_exe

    logical function test_multi_module_object_links_via_constant() result(ok)
        ! End-to-end: the -c object of a program-less multi-module file links
        ! into a separately-compiled driver that USEs one of its modules.
        character(len=*), parameter :: mods_src = '/tmp/ffc_pu_mods.f90'
        character(len=*), parameter :: drv_src = '/tmp/ffc_pu_drv.f90'
        character(len=*), parameter :: mods_obj = '/tmp/ffc_pu_mods.o'
        character(len=*), parameter :: drv_exe = '/tmp/ffc_pu_drv'
        integer :: exit_stat, cmd_stat

        ok = .false.
        if (.not. write_file(mods_src, two_modules)) return
        if (.not. write_file(drv_src, &
            'program drv'//new_line('a')// &
            '  use m_a, only: base'//new_line('a')// &
            '  stop base'//new_line('a')// &
            'end program drv')) return

        call execute_command_line('rm -f '//mods_obj//' /tmp/m_a.fmod '// &
            '/tmp/m_b.fmod '//drv_exe)
        call execute_command_line( &
            "sh -c 'exe=$(ls -t build/*/app/ffc build/fo/bin/ffc 2>/dev/null "// &
            "| head -n 1); "// &
            'test -n "$exe" || exit 90; '// &
            '"$exe" -c '//mods_src//' -o '//mods_obj//' || exit 91; '// &
            '"$exe" '//drv_src//' '//mods_obj//' -o '//drv_exe// &
            " || exit 92'", exitstat=exit_stat, cmdstat=cmd_stat)
        if (cmd_stat /= 0) then
            print *, 'FAIL: could not run the ffc compile pipeline'
            return
        end if
        if (exit_stat /= 0) then
            print *, 'FAIL: multi-module separate-compile pipeline failed, code ', &
                exit_stat
            return
        end if

        call execute_command_line(drv_exe, exitstat=exit_stat, cmdstat=cmd_stat)
        call execute_command_line('rm -f '//mods_src//' '//drv_src//' '// &
            mods_obj//' /tmp/m_a.fmod /tmp/m_b.fmod '//drv_exe)
        if (cmd_stat /= 0) then
            print *, 'FAIL: could not run the linked driver'
            return
        end if
        if (exit_stat /= 10) then
            print *, 'FAIL: expected exit 10 from the driver, got ', exit_stat
            return
        end if
        ok = .true.
    end function test_multi_module_object_links_via_constant

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
        write (unit, '(A)', iostat=io_stat) contents
        close (unit)
        ok = io_stat == 0
    end function write_file

end program test_session_program_units_compiler
