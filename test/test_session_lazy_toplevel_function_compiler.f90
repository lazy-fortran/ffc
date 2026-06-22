program test_session_lazy_toplevel_function_compiler
    !! A lazy (.lf) program with a bare top-level function called from top-level
    !! statements must lower and run. Lazy-mode standardization wraps the
    !! function and the caller in a synthetic __MULTI_UNIT__ program and restates
    !! each dummy as (integer, intent(in) :: a) carrying a spurious is_parameter
    !! flag. Routing that to the constant path demanded an initializer the dummy
    !! has not got and aborted the whole unit; an intent now marks it as the
    !! already-bound dummy so it takes the normal scalar path (#2812).
    implicit none

    logical :: ok

    print *, '=== direct session lazy top-level function compiler test ==='

    ok = .true.
    ! The issue_2812 corpus case: a function with inferred argument and result
    ! types, called with first-assignment inference at top level.
    if (.not. lazy_runs( &
            'function add(a, b)'//new_line('a')// &
            '    add = a + b'//new_line('a')// &
            'end function'//new_line('a')// &
            ''//new_line('a')// &
            'x = add(5, 3)'//new_line('a')// &
            'print *, x'//new_line('a'), &
            '           8', 'ffc_lazy_toplevel_add')) ok = .false.

    ! A top-level subroutine through the same standardization path.
    if (.not. lazy_runs( &
            'subroutine show(n)'//new_line('a')// &
            '    print *, n'//new_line('a')// &
            'end subroutine'//new_line('a')// &
            ''//new_line('a')// &
            'call show(7)'//new_line('a'), &
            '           7', 'ffc_lazy_toplevel_show')) ok = .false.

    ! A top-level subroutine with an explicit local declaration plus bare main
    ! statements parses to a mixed_construct_container root rather than a
    ! synthetic multi-unit program. The container must lower like a program:
    ! the subroutine becomes a contained procedure, the bare statements the
    ! main body (#266, #275). Mirrors monomorphization_scale_subroutine.lf.
    if (.not. lazy_runs( &
            'subroutine bump(v)'//new_line('a')// &
            '    integer :: v'//new_line('a')// &
            '    v = v + 1'//new_line('a')// &
            '    print *, v'//new_line('a')// &
            'end subroutine'//new_line('a')// &
            ''//new_line('a')// &
            'integer :: k'//new_line('a')// &
            'k = 41'//new_line('a')// &
            'call bump(k)'//new_line('a'), &
            '          42', 'ffc_mixed_container_subroutine')) ok = .false.

    ! A declaration-only unit (no executable statements) parses to a
    ! mixed_construct_container whose body holds only specification nodes. It
    ! must lower to a runnable empty main rather than a blanket "unsupported
    ! program unit" rejection (#266, #275). Mirrors the header-only corpus case.
    if (.not. lazy_runs_empty( &
            'integer :: x'//new_line('a')// &
            'real :: y'//new_line('a'), &
            'ffc_mixed_container_decls_only')) ok = .false.

    if (.not. ok) stop 1
    print *, 'PASS: top-level lazy function and subroutine lower and run'

contains

    ! Write a lazy fragment to a .lf file, drive it through the ffc CLI (which
    ! retries under lazy inference and standardization), run the executable, and
    ! compare stdout. Driving the real CLI exercises the standard-then-lazy
    ! fallback the corpus gauntlet uses.
    logical function lazy_runs(source, expected, stem) result(ok)
        character(len=*), intent(in) :: source
        character(len=*), intent(in) :: expected
        character(len=*), intent(in) :: stem
        character(len=:), allocatable :: src_path, exe_path, out_path, command
        character(len=:), allocatable :: actual
        integer :: unit, exit_stat, cmd_stat

        ok = .false.
        src_path = '/tmp/'//stem//'.lf'
        exe_path = '/tmp/'//stem//'.exe'
        out_path = '/tmp/'//stem//'.out'
        call execute_command_line('rm -f '//src_path//' '//exe_path//' '//out_path)

        open (newunit=unit, file=src_path, status='replace', action='write')
        write (unit, '(A)', advance='no') source
        close (unit)

        command = "sh -c 'exe=$(ls -t build/*/app/ffc build/fo/bin/ffc 2>/dev/null | head -n 1); "// &
                  "test -n ""$exe"" && ""$exe"" "//src_path//' -o '//exe_path//"'"
        call execute_command_line(command, exitstat=exit_stat, cmdstat=cmd_stat)
        if (cmd_stat /= 0 .or. exit_stat /= 0) then
            print *, 'FAIL: ffc rejected lazy source ', stem, ' exit=', exit_stat
            return
        end if

        call execute_command_line(exe_path//' > '//out_path, &
                                  exitstat=exit_stat, cmdstat=cmd_stat)
        if (cmd_stat /= 0 .or. exit_stat /= 0) then
            print *, 'FAIL: lazy executable did not run cleanly: ', stem
            return
        end if

        actual = read_first_line(out_path)
        call execute_command_line('rm -f '//src_path//' '//exe_path//' '//out_path)
        if (trim(actual) /= trim(expected)) then
            print *, 'FAIL: ', stem, ' expected [', trim(expected), &
                '] got [', trim(actual), ']'
            return
        end if
        ok = .true.
    end function lazy_runs

    ! Compile and run a lazy fragment that produces no stdout, asserting only a
    ! clean compile and a zero exit. Used for declaration-only units that lower
    ! to an empty main.
    logical function lazy_runs_empty(source, stem) result(ok)
        character(len=*), intent(in) :: source
        character(len=*), intent(in) :: stem
        character(len=:), allocatable :: src_path, exe_path, command
        integer :: unit, exit_stat, cmd_stat

        ok = .false.
        src_path = '/tmp/'//stem//'.lf'
        exe_path = '/tmp/'//stem//'.exe'
        call execute_command_line('rm -f '//src_path//' '//exe_path)

        open (newunit=unit, file=src_path, status='replace', action='write')
        write (unit, '(A)', advance='no') source
        close (unit)

        command = "sh -c 'exe=$(ls -t build/*/app/ffc build/fo/bin/ffc 2>/dev/null | head -n 1); "// &
                  "test -n ""$exe"" && ""$exe"" "//src_path//' -o '//exe_path//"'"
        call execute_command_line(command, exitstat=exit_stat, cmdstat=cmd_stat)
        if (cmd_stat /= 0 .or. exit_stat /= 0) then
            print *, 'FAIL: ffc rejected declaration-only source ', stem, &
                ' exit=', exit_stat
            return
        end if

        call execute_command_line(exe_path, exitstat=exit_stat, cmdstat=cmd_stat)
        call execute_command_line('rm -f '//src_path//' '//exe_path)
        if (cmd_stat /= 0 .or. exit_stat /= 0) then
            print *, 'FAIL: declaration-only executable did not run cleanly: ', stem
            return
        end if
        ok = .true.
    end function lazy_runs_empty

    function read_first_line(path) result(line)
        character(len=*), intent(in) :: path
        character(len=:), allocatable :: line
        character(len=256) :: buffer
        integer :: unit, io_stat

        line = ''
        open (newunit=unit, file=path, status='old', action='read', iostat=io_stat)
        if (io_stat /= 0) return
        read (unit, '(A)', iostat=io_stat) buffer
        if (io_stat == 0) line = trim(buffer)
        close (unit)
    end function read_first_line

end program test_session_lazy_toplevel_function_compiler
