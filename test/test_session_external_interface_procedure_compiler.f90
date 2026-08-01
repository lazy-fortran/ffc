program test_session_external_interface_procedure
    ! #582: an interface block in the specification part of a top-level
    ! (external) procedure declares an EXTERNAL procedure, not a dummy
    ! procedure of that scope. Only a name that also appears in the enclosing
    ! procedure's dummy-argument list is a dummy procedure and needs its body
    ! in the lowering unit. Rejecting the external case blocked
    ! gfortran.dg/intent_out_4.f90 with "procedure body unavailable".
    implicit none

    logical :: all_passed

    print *, '=== external interface in procedure spec tests ==='

    all_passed = .true.
    if (.not. test_external_interface_links()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: external interface in a procedure specification part'

contains

    logical function test_external_interface_links() result(ok)
        character(len=*), parameter :: lib_src = '/tmp/ffc_ext582_lib.f90'
        character(len=*), parameter :: drv_src = '/tmp/ffc_ext582_drv.f90'
        character(len=*), parameter :: lib_obj = '/tmp/ffc_ext582_lib.o'
        character(len=*), parameter :: drv_exe = '/tmp/ffc_ext582_drv'
        integer :: exit_stat, cmd_stat

        ok = .false.
        if (.not. write_file(lib_src, &
            'function compute() result(res)'//new_line('a')// &
            '    implicit none'//new_line('a')// &
            '    integer :: res'//new_line('a')// &
            '    interface'//new_line('a')// &
            '        subroutine foo(a)'//new_line('a')// &
            '            integer, intent(inout) :: a'//new_line('a')// &
            '        end subroutine foo'//new_line('a')// &
            '    end interface'//new_line('a')// &
            '    res = 20'//new_line('a')// &
            '    call foo(res)'//new_line('a')// &
            'end function compute')) return
        if (.not. write_file(drv_src, &
            'subroutine foo(a)'//new_line('a')// &
            '    implicit none'//new_line('a')// &
            '    integer, intent(inout) :: a'//new_line('a')// &
            '    a = a + 22'//new_line('a')// &
            'end subroutine foo'//new_line('a')// &
            'program main'//new_line('a')// &
            '    implicit none'//new_line('a')// &
            '    interface'//new_line('a')// &
            '        function compute() result(res)'//new_line('a')// &
            '            integer :: res'//new_line('a')// &
            '        end function compute'//new_line('a')// &
            '    end interface'//new_line('a')// &
            '    stop compute()'//new_line('a')// &
            'end program main')) return

        call execute_command_line('rm -f '//lib_obj//' '//drv_exe)
        call execute_command_line( &
            "sh -c 'exe=$(ls -t build/*/app/ffc build/fo/bin/ffc 2>/dev/null | "// &
            'head -n 1); test -n "$exe" || exit 90; '// &
            '"$exe" -c '//lib_src//' -o '//lib_obj//' || exit 91; '// &
            '"$exe" '//drv_src//' '//lib_obj//' -o '//drv_exe//" || exit 92'", &
            exitstat=exit_stat, cmdstat=cmd_stat)
        if (cmd_stat /= 0 .or. exit_stat /= 0) then
            print *, 'FAIL: external-interface unit did not build, code ', &
                exit_stat
            return
        end if
        call execute_command_line(drv_exe//' > /dev/null 2>&1', &
            exitstat=exit_stat, cmdstat=cmd_stat)
        if (cmd_stat /= 0) then
            print *, 'FAIL: could not run the linked driver'
            return
        end if
        if (exit_stat /= 42) then
            print *, 'FAIL: expected exit 42 from compute(), got ', exit_stat
            return
        end if
        call execute_command_line('rm -f '//lib_src//' '//drv_src//' '//lib_obj// &
            ' '//drv_exe)
        ok = .true.
    end function test_external_interface_links

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

end program test_session_external_interface_procedure
