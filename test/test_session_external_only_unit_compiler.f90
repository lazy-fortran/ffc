program test_session_external_only_unit_compiler
    ! #416: a translation unit holding only top-level (external) procedures
    ! compiles to an object that defines no main, keeps every procedure it
    ! declares, and links against a separately compiled driver that calls the
    ! external function and subroutine through an explicit interface.
    use ffc_test_support, only: expect_error_contains
    implicit none

    logical :: all_passed

    print *, '=== procedure-only translation unit tests ==='

    all_passed = .true.
    if (.not. test_procedure_only_object_links_with_driver()) all_passed = .false.
    if (.not. test_default_driver_falls_back_to_object()) all_passed = .false.
    if (.not. test_executable_without_main_is_rejected()) all_passed = .false.
    if (.not. test_multiple_procedures_keep_their_order()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: procedure-only translation units'

contains

    logical function test_procedure_only_object_links_with_driver() result(ok)
        character(len=*), parameter :: lib_src = '/tmp/ffc_ext416_lib.f90'
        character(len=*), parameter :: drv_src = '/tmp/ffc_ext416_drv.f90'
        character(len=*), parameter :: lib_obj = '/tmp/ffc_ext416_lib.o'
        character(len=*), parameter :: drv_exe = '/tmp/ffc_ext416_drv'
        character(len=*), parameter :: out_file = '/tmp/ffc_ext416_out.txt'
        integer :: exit_stat, cmd_stat

        ok = .false.
        if (.not. write_file(lib_src, &
            'integer function twice(x)'//new_line('a')// &
            '    integer, intent(in) :: x'//new_line('a')// &
            '    twice = 2 * x'//new_line('a')// &
            'end function twice'//new_line('a')// &
            'subroutine shout()'//new_line('a')// &
            "    print *, 'EXTERNAL_SHOUT'"//new_line('a')// &
            'end subroutine shout')) return
        if (.not. write_file(drv_src, &
            'program main'//new_line('a')// &
            '    implicit none'//new_line('a')// &
            '    interface'//new_line('a')// &
            '        integer function twice(x)'//new_line('a')// &
            '            integer, intent(in) :: x'//new_line('a')// &
            '        end function twice'//new_line('a')// &
            '        subroutine shout()'//new_line('a')// &
            '        end subroutine shout'//new_line('a')// &
            '    end interface'//new_line('a')// &
            '    if (twice(21) /= 42) error stop'//new_line('a')// &
            '    call shout()'//new_line('a')// &
            "    print *, 'DRIVER_OK'"//new_line('a')// &
            'end program main')) return

        call execute_command_line('rm -f '//lib_obj//' '//drv_exe//' '//out_file)
        call execute_command_line( &
            "sh -c 'exe=$(ls -t build/*/app/ffc build/fo/bin/ffc 2>/dev/null | "// &
            'head -n 1); test -n "$exe" || exit 90; '// &
            '"$exe" -c '//lib_src//' -o '//lib_obj//' || exit 91; '// &
            '"$exe" '//drv_src//' '//lib_obj//' -o '//drv_exe//" || exit 92'", &
            exitstat=exit_stat, cmdstat=cmd_stat)
        if (cmd_stat /= 0 .or. exit_stat /= 0) then
            print *, 'FAIL: procedure-only compile/link pipeline failed, code ', &
                exit_stat
            return
        end if
        ! A procedure-only object must not define main; otherwise the link above
        ! would already have failed on a duplicate main.
        call execute_command_line('nm '//lib_obj//' | grep -q " T main"', &
            exitstat=exit_stat, cmdstat=cmd_stat)
        if (cmd_stat == 0 .and. exit_stat == 0) then
            print *, 'FAIL: procedure-only object defines main'
            return
        end if
        call execute_command_line(drv_exe//' > '//out_file//' 2>&1', &
            exitstat=exit_stat, cmdstat=cmd_stat)
        if (cmd_stat /= 0 .or. exit_stat /= 0) then
            print *, 'FAIL: linked driver did not run cleanly, code ', exit_stat
            return
        end if
        if (.not. file_contains(out_file, 'EXTERNAL_SHOUT')) then
            print *, 'FAIL: external subroutine output missing'
            return
        end if
        if (.not. file_contains(out_file, 'DRIVER_OK')) then
            print *, 'FAIL: driver output missing'
            return
        end if
        call execute_command_line('rm -f '//lib_src//' '//drv_src//' '//lib_obj// &
            ' '//drv_exe//' '//out_file)
        ok = .true.
    end function test_procedure_only_object_links_with_driver

    logical function test_default_driver_falls_back_to_object() result(ok)
        ! A compile-only source from the gfortran dg suite is commonly passed
        ! to a driver without an explicit -c.  ffc must turn only its
        ! procedure-only no-main result into object emission, matching gfortran.
        character(len=*), parameter :: src = '/tmp/ffc_ext416_default.f90'
        character(len=*), parameter :: ffc_obj = '/tmp/ffc_ext416_default.o'
        character(len=*), parameter :: gcc_obj = '/tmp/ffc_ext416_gfortran.o'
        integer :: exit_stat, cmd_stat

        ok = .false.
        if (.not. write_file(src, &
            'integer function twice(x)'//new_line('a')// &
            '    integer, intent(in) :: x'//new_line('a')// &
            '    twice = 2 * x'//new_line('a')// &
            'end function twice')) return
        call execute_command_line('rm -f '//ffc_obj//' '//gcc_obj)
        call execute_command_line( &
            "sh -c 'exe=build/fo/bin/ffc; test -x $exe || exit 90; "// &
            '"$exe" '//src//' -o '//ffc_obj//' || exit 91; '// &
            'gfortran -c '//src//' -o '//gcc_obj//"'", &
            exitstat=exit_stat, cmdstat=cmd_stat)
        if (cmd_stat /= 0 .or. exit_stat /= 0) then
            print *, 'FAIL: default driver procedure-only fallback failed, code ', &
                exit_stat
            return
        end if
        call execute_command_line('test -s '//ffc_obj//' && test -s '//gcc_obj, &
            exitstat=exit_stat, cmdstat=cmd_stat)
        if (cmd_stat /= 0 .or. exit_stat /= 0) then
            print *, 'FAIL: fallback did not produce and gfortran did not control an object'
            return
        end if
        call execute_command_line('nm '//ffc_obj//' | grep -q " T main"', &
            exitstat=exit_stat, cmdstat=cmd_stat)
        if (cmd_stat == 0 .and. exit_stat == 0) then
            print *, 'FAIL: fallback object defines main'
            return
        end if
        call execute_command_line('rm -f '//src//' '//ffc_obj//' '//gcc_obj)
        ok = .true.
    end function test_default_driver_falls_back_to_object

    logical function test_executable_without_main_is_rejected() result(ok)
        ! An executable build request needs a main program unit; a
        ! procedure-only root must say so instead of emitting a do-nothing exe.
        ok = expect_error_contains( &
            'integer function twice(x)'//new_line('a')// &
            '    integer, intent(in) :: x'//new_line('a')// &
            '    twice = 2 * x'//new_line('a')// &
            'end function twice', &
            'no main program unit', &
            '/tmp/ffc_ext416_nomain')
    end function test_executable_without_main_is_rejected

    logical function test_multiple_procedures_keep_their_order() result(ok)
        ! Every procedure of a procedure-only unit is exported, including one
        ! defined after the procedure that calls it, so the order of the unit
        ! does not decide what links.
        character(len=*), parameter :: lib_src = '/tmp/ffc_ext416b_lib.f90'
        character(len=*), parameter :: drv_src = '/tmp/ffc_ext416b_drv.f90'
        character(len=*), parameter :: lib_obj = '/tmp/ffc_ext416b_lib.o'
        character(len=*), parameter :: drv_exe = '/tmp/ffc_ext416b_drv'
        integer :: exit_stat, cmd_stat

        ok = .false.
        if (.not. write_file(lib_src, &
            'subroutine first()'//new_line('a')// &
            "    print *, 'FIRST'"//new_line('a')// &
            'end subroutine first'//new_line('a')// &
            'integer function inner(x)'//new_line('a')// &
            '    integer, intent(in) :: x'//new_line('a')// &
            '    inner = x + 1'//new_line('a')// &
            'end function inner'//new_line('a')// &
            'integer function outer(x)'//new_line('a')// &
            '    integer, intent(in) :: x'//new_line('a')// &
            '    integer :: inner'//new_line('a')// &
            '    outer = 10 * inner(x)'//new_line('a')// &
            'end function outer')) return
        if (.not. write_file(drv_src, &
            'program main'//new_line('a')// &
            '    implicit none'//new_line('a')// &
            '    interface'//new_line('a')// &
            '        subroutine first()'//new_line('a')// &
            '        end subroutine first'//new_line('a')// &
            '        integer function outer(x)'//new_line('a')// &
            '            integer, intent(in) :: x'//new_line('a')// &
            '        end function outer'//new_line('a')// &
            '    end interface'//new_line('a')// &
            '    call first()'//new_line('a')// &
            '    stop outer(3)'//new_line('a')// &
            'end program main')) return

        call execute_command_line('rm -f '//lib_obj//' '//drv_exe)
        call execute_command_line( &
            "sh -c 'exe=$(ls -t build/*/app/ffc build/fo/bin/ffc 2>/dev/null | "// &
            'head -n 1); test -n "$exe" || exit 90; '// &
            '"$exe" -c '//lib_src//' -o '//lib_obj//' || exit 91; '// &
            '"$exe" '//drv_src//' '//lib_obj//' -o '//drv_exe//" || exit 92'", &
            exitstat=exit_stat, cmdstat=cmd_stat)
        if (cmd_stat /= 0 .or. exit_stat /= 0) then
            print *, 'FAIL: multi-procedure unit did not build, code ', exit_stat
            return
        end if
        call execute_command_line(drv_exe//' > /dev/null 2>&1', &
            exitstat=exit_stat, cmdstat=cmd_stat)
        if (cmd_stat /= 0) then
            print *, 'FAIL: could not run the linked driver'
            return
        end if
        if (exit_stat /= 40) then
            print *, 'FAIL: expected exit 40 from outer(3), got ', exit_stat
            return
        end if
        call execute_command_line('rm -f '//lib_src//' '//drv_src//' '//lib_obj// &
            ' '//drv_exe)
        ok = .true.
    end function test_multiple_procedures_keep_their_order

    logical function file_contains(path, fragment) result(found)
        character(len=*), intent(in) :: path
        character(len=*), intent(in) :: fragment
        integer :: unit, io_stat
        character(len=512) :: line

        found = .false.
        open (newunit=unit, file=path, status='old', action='read', &
            iostat=io_stat)
        if (io_stat /= 0) return
        do
            read (unit, '(A)', iostat=io_stat) line
            if (io_stat /= 0) exit
            if (index(line, fragment) > 0) then
                found = .true.
                exit
            end if
        end do
        close (unit)
    end function file_contains

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

end program test_session_external_only_unit_compiler
