program test_session_separate_compilation_compiler
    implicit none

    logical :: all_passed

    print *, '=== two-file separate compilation tests ==='

    all_passed = .true.
    if (.not. test_two_file_module_and_main_links_and_runs()) all_passed = .false.
    if (.not. test_module_print_and_real_state()) all_passed = .false.
    if (.not. test_two_module_container_object_links()) all_passed = .false.
    if (.not. test_procedures_only_module_links()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: two-file separate compilation'

contains

    logical function test_module_print_and_real_state() result(ok)
        ! #284/#165: a separately compiled module keeps its own string literals
        ! (its print is not clobbered by the main's colliding .ffc.str globals),
        ! and a real module variable plus real-argument subroutines round-trip
        ! through the .fmod so the value survives the call boundary.
        character(len=*), parameter :: m_src = '/tmp/ffc_sep2_m.f90'
        character(len=*), parameter :: main_src = '/tmp/ffc_sep2_main.f90'
        character(len=*), parameter :: m_obj = '/tmp/ffc_sep2_m.o'
        character(len=*), parameter :: main_exe = '/tmp/ffc_sep2_main'
        character(len=*), parameter :: out_file = '/tmp/ffc_sep2_out.txt'

        ok = .false.
        if (.not. write_file(m_src, &
            'module ffc_sep_state'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  real :: stored'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine store(x)'//new_line('a')// &
            '    real, intent(in) :: x'//new_line('a')// &
            '    stored = x'//new_line('a')// &
            "    print *, 'MODULE_STORE'"//new_line('a')// &
            '  end subroutine store'//new_line('a')// &
            '  subroutine fetch(x)'//new_line('a')// &
            '    real, intent(out) :: x'//new_line('a')// &
            '    x = stored'//new_line('a')// &
            '  end subroutine fetch'//new_line('a')// &
            'end module ffc_sep_state')) return
        if (.not. write_file(main_src, &
            'program main'//new_line('a')// &
            '  use ffc_sep_state'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  real :: a, b'//new_line('a')// &
            '  a = 3.0'//new_line('a')// &
            '  call store(a)'//new_line('a')// &
            "  print *, 'MAIN_MID'"//new_line('a')// &
            '  call fetch(b)'//new_line('a')// &
            '  if (b == 3.0) then'//new_line('a')// &
            "    print *, 'MATCH'"//new_line('a')// &
            '  else'//new_line('a')// &
            "    print *, 'NOMATCH'"//new_line('a')// &
            '  end if'//new_line('a')// &
            'end program main')) return

        if (.not. build_and_run(m_src, main_src, m_obj, main_exe, out_file, &
            '/tmp/ffc_sep_state.fmod')) return
        if (.not. file_contains(out_file, 'MODULE_STORE')) then
            print *, 'FAIL: module print string was clobbered (string collision)'
            return
        end if
        if (.not. file_contains(out_file, 'MAIN_MID')) then
            print *, 'FAIL: main print missing'
            return
        end if
        if (.not. file_contains(out_file, 'MATCH')) then
            print *, 'FAIL: real module state did not round-trip'
            return
        end if
        if (file_contains(out_file, 'NOMATCH')) then
            print *, 'FAIL: real value mismatch across separate compilation'
            return
        end if
        ok = .true.
    end function test_module_print_and_real_state

    logical function test_two_module_container_object_links() result(ok)
        ! #284: a single source defining two modules and no program compiles to
        ! one object (with a .fmod per module beside it); a program using one of
        ! them links and runs.
        character(len=*), parameter :: m_src = '/tmp/ffc_sep3_m.f90'
        character(len=*), parameter :: main_src = '/tmp/ffc_sep3_main.f90'
        character(len=*), parameter :: m_obj = '/tmp/ffc_sep3_m.o'
        character(len=*), parameter :: main_exe = '/tmp/ffc_sep3_main'
        character(len=*), parameter :: out_file = '/tmp/ffc_sep3_out.txt'

        ok = .false.
        if (.not. write_file(m_src, &
            'module ffc_box_a'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer, parameter :: box_a = 11'//new_line('a')// &
            'end module ffc_box_a'//new_line('a')// &
            'module ffc_box_b'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer, parameter :: box_b = 22'//new_line('a')// &
            'end module ffc_box_b')) return
        if (.not. write_file(main_src, &
            'program main'//new_line('a')// &
            '  use ffc_box_b'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  stop box_b'//new_line('a')// &
            'end program main')) return

        if (.not. build_and_run(m_src, main_src, m_obj, main_exe, out_file, &
            '/tmp/ffc_box_a.fmod /tmp/ffc_box_b.fmod')) return
        ok = .true.
    end function test_two_module_container_object_links

    logical function test_procedures_only_module_links() result(ok)
        ! A module whose only exports are contained procedures (no module-level
        ! declaration part) must still emit a .fmod so a separately compiled
        ! program that USEs it resolves the module and links the procedure.
        character(len=*), parameter :: m_src = '/tmp/ffc_sep4_m.f90'
        character(len=*), parameter :: main_src = '/tmp/ffc_sep4_main.f90'
        character(len=*), parameter :: m_obj = '/tmp/ffc_sep4_m.o'
        character(len=*), parameter :: main_exe = '/tmp/ffc_sep4_main'
        character(len=*), parameter :: out_file = '/tmp/ffc_sep4_out.txt'

        ok = .false.
        if (.not. write_file(m_src, &
            'module ffc_only_procs'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            'contains'//new_line('a')// &
            '  integer function twice(x) result(r)'//new_line('a')// &
            '    integer, intent(in) :: x'//new_line('a')// &
            '    r = 2 * x'//new_line('a')// &
            '  end function twice'//new_line('a')// &
            'end module ffc_only_procs')) return
        if (.not. write_file(main_src, &
            'program main'//new_line('a')// &
            '  use ffc_only_procs, only: twice'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  if (twice(21) /= 42) error stop'//new_line('a')// &
            "  print *, 'OK'"//new_line('a')// &
            'end program main')) return

        if (.not. build_and_run(m_src, main_src, m_obj, main_exe, out_file, &
            '/tmp/ffc_only_procs.fmod')) return
        if (.not. file_contains(out_file, 'OK')) then
            print *, 'FAIL: procedures-only module did not round-trip'
            return
        end if
        ok = .true.
    end function test_procedures_only_module_links

    logical function build_and_run(m_src, main_src, m_obj, main_exe, out_file, &
            fmods) result(ok)
        ! Compile the module source to an object (emitting its .fmod), compile
        ! and link the program against it, then run capturing stdout to out_file.
        character(len=*), intent(in) :: m_src, main_src, m_obj, main_exe
        character(len=*), intent(in) :: out_file, fmods
        integer :: exit_stat, cmd_stat

        ok = .false.
        call execute_command_line('rm -f '//m_obj//' '//main_exe//' '// &
            out_file//' '//fmods)
        call execute_command_line( &
            "sh -c 'exe=$(ls -t build/*/app/ffc build/fo/bin/ffc 2>/dev/null | "// &
            'head -n 1); test -n "$exe" || exit 90; '// &
            '"$exe" -c '//m_src//' -o '//m_obj//' || exit 91; '// &
            '"$exe" '//main_src//' '//m_obj//' -o '//main_exe//" || exit 92'", &
            exitstat=exit_stat, cmdstat=cmd_stat)
        if (cmd_stat /= 0 .or. exit_stat /= 0) then
            print *, 'FAIL: separate compile pipeline failed, code ', exit_stat
            return
        end if
        call execute_command_line(main_exe//' > '//out_file//' 2>&1', &
            exitstat=exit_stat, cmdstat=cmd_stat)
        if (cmd_stat /= 0) then
            print *, 'FAIL: could not run the linked program'
            return
        end if
        call execute_command_line('rm -f '//m_src//' '//main_src//' '//m_obj// &
            ' '//main_exe//' '//fmods)
        ok = .true.
    end function build_and_run

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

    logical function test_two_file_module_and_main_links_and_runs() result(ok)
        character(len=*), parameter :: m_src = '/tmp/ffc_sep_m.f90'
        character(len=*), parameter :: main_src = '/tmp/ffc_sep_main.f90'
        character(len=*), parameter :: m_obj = '/tmp/ffc_sep_m.o'
        character(len=*), parameter :: main_exe = '/tmp/ffc_sep_main'
        integer :: exit_stat, cmd_stat

        ok = .false.
        if (.not. write_file(m_src, &
            'module ffc_sep_mod'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer, parameter :: answer = 42'//new_line('a')// &
            'end module ffc_sep_mod')) return
        if (.not. write_file(main_src, &
            'program main'//new_line('a')// &
            '  use ffc_sep_mod'//new_line('a')// &
            '  stop answer'//new_line('a')// &
            'end program main')) return

        call execute_command_line('rm -f '//m_obj//' /tmp/ffc_sep_mod.fmod '// &
            main_exe)

        ! Compile the module (emits the object and its .fmod), then compile the
        ! program naming the object; the .fmod is found beside it.
        call execute_command_line( &
            "sh -c 'exe=$(ls -t build/*/app/ffc build/fo/bin/ffc 2>/dev/null | head -n 1); "// &
            'test -n "$exe" || exit 90; '// &
            '"$exe" -c '//m_src//' -o '//m_obj//' || exit 91; '// &
            '"$exe" '//main_src//' '//m_obj//' -o '//main_exe//" || exit 92'", &
            exitstat=exit_stat, cmdstat=cmd_stat)
        if (cmd_stat /= 0) then
            print *, 'FAIL: could not run the ffc compile pipeline'
            return
        end if
        if (exit_stat /= 0) then
            print *, 'FAIL: two-file compile pipeline failed, code ', exit_stat
            return
        end if

        call execute_command_line(main_exe, exitstat=exit_stat, cmdstat=cmd_stat)
        call execute_command_line('rm -f '//m_src//' '//main_src//' '//m_obj// &
            ' /tmp/ffc_sep_mod.fmod '//main_exe)
        if (cmd_stat /= 0) then
            print *, 'FAIL: could not run the linked program'
            return
        end if
        if (exit_stat /= 42) then
            print *, 'FAIL: expected exit 42 from the linked program, got ', &
                exit_stat
            return
        end if
        ok = .true.
    end function test_two_file_module_and_main_links_and_runs

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

end program test_session_separate_compilation_compiler
