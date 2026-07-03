program test_session_separate_generic_compiler
    implicit none

    logical :: all_passed

    print *, '=== separate-compilation generic interface tests ==='

    all_passed = .true.
    if (.not. test_use_associated_generic_resolves()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: separate-compilation generic interface'

contains

    logical function test_use_associated_generic_resolves() result(ok)
        ! A module exports a named generic interface over an integer-argument and
        ! a real-argument subroutine. A separately compiled program USEs only the
        ! generic name and calls it with each type; the call must resolve to the
        ! matching specific across the .fmod and link against the module object.
        character(len=*), parameter :: m_src = '/tmp/ffc_gen_m.f90'
        character(len=*), parameter :: main_src = '/tmp/ffc_gen_main.f90'
        character(len=*), parameter :: m_obj = '/tmp/ffc_gen_m.o'
        character(len=*), parameter :: main_exe = '/tmp/ffc_gen_main'
        character(len=*), parameter :: out_file = '/tmp/ffc_gen_out.txt'
        integer :: exit_stat, cmd_stat

        ok = .false.
        if (.not. write_file(m_src, &
            'module ffc_gen_mod'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  interface bump'//new_line('a')// &
            '    module procedure bump_i'//new_line('a')// &
            '    module procedure bump_r'//new_line('a')// &
            '  end interface'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine bump_i(a)'//new_line('a')// &
            '    integer, intent(inout) :: a'//new_line('a')// &
            '    a = a + 1'//new_line('a')// &
            '  end subroutine bump_i'//new_line('a')// &
            '  subroutine bump_r(a)'//new_line('a')// &
            '    real, intent(inout) :: a'//new_line('a')// &
            '    a = a + 1'//new_line('a')// &
            '  end subroutine bump_r'//new_line('a')// &
            'end module ffc_gen_mod')) return
        if (.not. write_file(main_src, &
            'program main'//new_line('a')// &
            '  use ffc_gen_mod, only: bump'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  real :: r'//new_line('a')// &
            '  i = 5'//new_line('a')// &
            '  call bump(i)'//new_line('a')// &
            '  if (i /= 6) error stop'//new_line('a')// &
            '  r = 6.0'//new_line('a')// &
            '  call bump(r)'//new_line('a')// &
            '  if (r /= 7.0) error stop'//new_line('a')// &
            "  print *, 'OK'"//new_line('a')// &
            'end program main')) return

        call execute_command_line('rm -f '//m_obj//' /tmp/ffc_gen_mod.fmod '// &
            main_exe//' '//out_file)
        call execute_command_line( &
            "sh -c 'exe=$(ls -t build/*/app/ffc build/fo/bin/ffc 2>/dev/null | "// &
            'head -n 1); test -n "$exe" || exit 90; '// &
            '"$exe" -c '//m_src//' -o '//m_obj//' || exit 91; '// &
            '"$exe" '//main_src//' '//m_obj//' -o '//main_exe//" || exit 92'", &
            exitstat=exit_stat, cmdstat=cmd_stat)
        if (cmd_stat /= 0 .or. exit_stat /= 0) then
            print *, 'FAIL: generic separate-compile pipeline failed, code ', &
                exit_stat
            return
        end if
        call execute_command_line(main_exe//' > '//out_file//' 2>&1', &
            exitstat=exit_stat, cmdstat=cmd_stat)
        if (cmd_stat /= 0 .or. exit_stat /= 0) then
            print *, 'FAIL: linked generic program did not run cleanly, code ', &
                exit_stat
            return
        end if
        if (.not. file_contains(out_file, 'OK')) then
            print *, 'FAIL: generic calls did not resolve to the right specifics'
            return
        end if
        call execute_command_line('rm -f '//m_src//' '//main_src//' '//m_obj// &
            ' /tmp/ffc_gen_mod.fmod '//main_exe//' '//out_file)
        ok = .true.
    end function test_use_associated_generic_resolves

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

end program test_session_separate_generic_compiler
