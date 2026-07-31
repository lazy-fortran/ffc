program test_session_include_compiler
    ! Behavioural coverage for Fortran INCLUDE line expansion (F2018 6.4.2).
    ! Every case drives the ffc CLI on real files on disk and checks observable
    ! program output or the emitted diagnostic.
    implicit none

    character(len=*), parameter :: WORK = '/tmp/ffc_include_test'
    logical :: ok

    print *, '=== direct session include compiler test ==='

    ok = .true.
    if (.not. check_source_relative()) ok = .false.
    if (.not. check_include_path_flag()) ok = .false.
    if (.not. check_nested_include()) ok = .false.
    if (.not. check_missing_include()) ok = .false.
    if (.not. check_include_cycle()) ok = .false.
    if (.not. ok) stop 1

    print *, 'PASS: include lines expand and report missing files and cycles'

contains

    logical function check_source_relative() result(ok)
        character(len=:), allocatable :: output

        ok = .false.
        call reset_work()
        call write_file(WORK//'/value.inc', 'integer, parameter :: v = 42')
        call write_file(WORK//'/main.f90', &
                        'program main'//new_line('a')// &
                        'implicit none'//new_line('a')// &
                        "include 'value.inc'"//new_line('a')// &
                        "print '(I0)', v"//new_line('a')// &
                        'end program main')
        if (.not. compile_ok(WORK//'/main.f90', '')) return
        call run_exe(output)
        if (first_line(output) /= '42') then
            print *, 'FAIL: source-relative include output was ', &
                first_line(output)
            return
        end if
        ok = .true.
    end function check_source_relative

    logical function check_include_path_flag() result(ok)
        character(len=:), allocatable :: output

        ok = .false.
        call reset_work()
        call execute_command_line('mkdir -p '//WORK//'/inc')
        call write_file(WORK//'/inc/value.inc', 'integer, parameter :: v = 42')
        call write_file(WORK//'/main.f90', &
                        'program main'//new_line('a')// &
                        'implicit none'//new_line('a')// &
                        "include 'value.inc'"//new_line('a')// &
                        "print '(I0)', v"//new_line('a')// &
                        'end program main')
        if (.not. compile_ok(WORK//'/main.f90', '-I '//WORK//'/inc')) return
        call run_exe(output)
        if (first_line(output) /= '42') then
            print *, 'FAIL: -I include output was ', first_line(output)
            return
        end if
        ok = .true.
    end function check_include_path_flag

    logical function check_nested_include() result(ok)
        character(len=:), allocatable :: output

        ok = .false.
        call reset_work()
        call write_file(WORK//'/inner.inc', 'integer, parameter :: v = 42')
        call write_file(WORK//'/outer.inc', &
                        "include 'inner.inc'"//new_line('a')// &
                        'integer, parameter :: w = v')
        call write_file(WORK//'/main.f90', &
                        'program main'//new_line('a')// &
                        'implicit none'//new_line('a')// &
                        "include 'outer.inc'"//new_line('a')// &
                        "print '(I0)', w"//new_line('a')// &
                        'end program main')
        if (.not. compile_ok(WORK//'/main.f90', '')) return
        call run_exe(output)
        if (first_line(output) /= '42') then
            print *, 'FAIL: nested include output was ', first_line(output)
            return
        end if
        ok = .true.
    end function check_nested_include

    logical function check_missing_include() result(ok)
        ok = .false.
        call reset_work()
        call write_file(WORK//'/main.f90', &
                        'program main'//new_line('a')// &
                        'implicit none'//new_line('a')// &
                        "include 'absent.inc'"//new_line('a')// &
                        'end program main')
        ok = compile_fails_with(WORK//'/main.f90', '', &
                                'include file not found')
    end function check_missing_include

    logical function check_include_cycle() result(ok)
        ok = .false.
        call reset_work()
        call write_file(WORK//'/loop.inc', "include 'loop.inc'")
        call write_file(WORK//'/main.f90', &
                        'program main'//new_line('a')// &
                        'implicit none'//new_line('a')// &
                        "include 'loop.inc'"//new_line('a')// &
                        'end program main')
        ok = compile_fails_with(WORK//'/main.f90', '', 'include cycle')
    end function check_include_cycle

    logical function compile_ok(source_path, extra_args) result(ok)
        character(len=*), intent(in) :: source_path
        character(len=*), intent(in) :: extra_args
        character(len=:), allocatable :: stderr_text
        integer :: exit_stat

        call run_ffc(source_path, extra_args, exit_stat, stderr_text)
        ok = exit_stat == 0
        if (.not. ok) then
            print *, 'FAIL: ffc rejected ', source_path
            print *, '  stderr: ', trim(stderr_text)
        end if
    end function compile_ok

    logical function compile_fails_with(source_path, extra_args, fragment) &
        result(ok)
        character(len=*), intent(in) :: source_path
        character(len=*), intent(in) :: extra_args
        character(len=*), intent(in) :: fragment
        character(len=:), allocatable :: stderr_text
        integer :: exit_stat

        ok = .false.
        call run_ffc(source_path, extra_args, exit_stat, stderr_text)
        if (exit_stat == 0) then
            print *, 'FAIL: expected failure for ', source_path
            return
        end if
        if (index(stderr_text, fragment) == 0) then
            print *, 'FAIL: diagnostic missing "'//fragment//'"'
            print *, '  stderr: ', trim(stderr_text)
            return
        end if
        ok = .true.
    end function compile_fails_with

    subroutine run_ffc(source_path, extra_args, exit_stat, stderr_text)
        character(len=*), intent(in) :: source_path
        character(len=*), intent(in) :: extra_args
        integer, intent(out) :: exit_stat
        character(len=:), allocatable, intent(out) :: stderr_text
        character(len=:), allocatable :: command
        integer :: cmd_stat

        command = "sh -c 'exe=$(ls -t build/*/app/ffc build/fo/bin/ffc "// &
                  '2>/dev/null | head -n 1); test -n "$exe" && "$exe" '// &
                  trim(extra_args)//' '//source_path//' -o '//WORK// &
                  '/prog > '//WORK//'/stdout 2> '//WORK//"/stderr'"
        exit_stat = -1
        call execute_command_line(command, exitstat=exit_stat, cmdstat=cmd_stat)
        stderr_text = read_text(WORK//'/stderr')
    end subroutine run_ffc

    subroutine run_exe(output)
        character(len=:), allocatable, intent(out) :: output
        integer :: exit_stat, cmd_stat

        call execute_command_line(WORK//'/prog > '//WORK//'/run.out', &
                                  exitstat=exit_stat, cmdstat=cmd_stat)
        output = read_text(WORK//'/run.out')
        if (exit_stat /= 0) print *, 'FAIL: program exited with ', exit_stat
    end subroutine run_exe

    subroutine reset_work()
        call execute_command_line('rm -rf '//WORK//' && mkdir -p '//WORK)
    end subroutine reset_work

    subroutine write_file(path, contents)
        character(len=*), intent(in) :: path
        character(len=*), intent(in) :: contents
        integer :: unit, io_stat

        open (newunit=unit, file=path, status='replace', action='write', &
              iostat=io_stat)
        if (io_stat /= 0) then
            print *, 'FAIL: cannot write ', path
            return
        end if
        write (unit, '(A)') contents
        close (unit)
    end subroutine write_file

    function first_line(text) result(line)
        character(len=*), intent(in) :: text
        character(len=:), allocatable :: line
        integer :: nl

        nl = index(text, new_line('a'))
        if (nl > 0) then
            line = trim(text(:nl - 1))
        else
            line = trim(text)
        end if
    end function first_line

    function read_text(path) result(text)
        character(len=*), intent(in) :: path
        character(len=:), allocatable :: text
        character(len=1024) :: line
        integer :: unit, io_stat

        text = ''
        open (newunit=unit, file=path, status='old', action='read', &
              iostat=io_stat)
        if (io_stat /= 0) return
        do
            read (unit, '(A)', iostat=io_stat) line
            if (io_stat /= 0) exit
            text = text//trim(line)//new_line('a')
        end do
        close (unit)
    end function read_text

end program test_session_include_compiler
