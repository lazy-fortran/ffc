program test_session_separate_compilation_compiler
    implicit none

    logical :: all_passed

    print *, '=== two-file separate compilation tests ==='

    all_passed = .true.
    if (.not. test_two_file_module_and_main_links_and_runs()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: two-file separate compilation'

contains

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
            "sh -c 'exe=$(ls -t build/*/app/ffc 2>/dev/null | head -n 1); "// &
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
