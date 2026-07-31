program test_session_lazy_defaults_compiler
    !! Lazy Fortran default type and intent policy (#438). A .lf source is
    !! compiled with the Lazy dialect defaults: a kind-less `real` is real(8),
    !! and a dummy argument declared without an explicit INTENT is INTENT(IN).
    !! Standard Fortran sources keep the processor default real kind and the
    !! language rule that only an explicit INTENT(IN) protects a dummy.
    implicit none

    logical :: ok

    print *, '=== direct session lazy defaults compiler test ==='

    ok = .true.

    ! Default real is 8 bytes: the lazy source and its explicit standard
    ! equivalent must print the same value.
    if (.not. runs( &
        'real :: y'//new_line('a')// &
        'y = 1.0d0/3.0d0'//new_line('a')// &
        'print *, y'//new_line('a'), 'lf', &
        '  0.33333333333333331', 'ffc_lf_default_real')) ok = .false.
    if (.not. runs( &
        'program p'//new_line('a')// &
        '    implicit none'//new_line('a')// &
        '    real(8) :: y'//new_line('a')// &
        '    y = 1.0d0/3.0d0'//new_line('a')// &
        '    print *, y'//new_line('a')// &
        'end program p'//new_line('a'), 'f90', &
        '  0.33333333333333331', 'ffc_std_explicit_real8')) ok = .false.

    ! A standard source keeps the processor default real kind: a kind-less
    ! `real` stays single precision there.
    if (.not. runs( &
        'program p'//new_line('a')// &
        '    implicit none'//new_line('a')// &
        '    real :: y'//new_line('a')// &
        '    y = 1.0d0/3.0d0'//new_line('a')// &
        '    print *, y'//new_line('a')// &
        'end program p'//new_line('a'), 'f90', &
        '  0.333333343', 'ffc_std_default_real')) ok = .false.

    ! A dummy without an explicit INTENT is readable in lazy mode, and the
    ! explicit INTENT(IN) standard equivalent produces the same output.
    if (.not. runs( &
        'subroutine show(n)'//new_line('a')// &
        '    integer :: n'//new_line('a')// &
        '    print *, n + 1'//new_line('a')// &
        'end subroutine'//new_line('a')// &
        ''//new_line('a')// &
        'call show(41)'//new_line('a'), 'lf', &
        '          42', 'ffc_lf_default_intent_read')) ok = .false.
    if (.not. runs( &
        'program p'//new_line('a')// &
        '    implicit none'//new_line('a')// &
        '    call show(41)'//new_line('a')// &
        'contains'//new_line('a')// &
        '    subroutine show(n)'//new_line('a')// &
        '        integer, intent(in) :: n'//new_line('a')// &
        '        print *, n + 1'//new_line('a')// &
        '    end subroutine show'//new_line('a')// &
        'end program p'//new_line('a'), 'f90', &
        '          42', 'ffc_std_intent_in_read')) ok = .false.

    ! Writing to that intent-less dummy violates the lazy default intent.
    if (.not. rejects( &
        'subroutine bump(x)'//new_line('a')// &
        '    integer :: x'//new_line('a')// &
        '    x = x + 1'//new_line('a')// &
        'end subroutine'//new_line('a')// &
        ''//new_line('a')// &
        'call bump(1)'//new_line('a'), 'lf', &
        'ffc_lf_default_intent_write')) ok = .false.

    ! Explicit INTENT(IN) conflicts with a write in standard mode too.
    if (.not. rejects( &
        'program p'//new_line('a')// &
        '    implicit none'//new_line('a')// &
        '    call bump(1)'//new_line('a')// &
        'contains'//new_line('a')// &
        '    subroutine bump(x)'//new_line('a')// &
        '        integer, intent(in) :: x'//new_line('a')// &
        '        x = x + 1'//new_line('a')// &
        '    end subroutine bump'//new_line('a')// &
        'end program p'//new_line('a'), 'f90', &
        'ffc_std_intent_in_write')) ok = .false.

    ! An explicit INTENT(INOUT) dummy stays writable in lazy mode: the lazy
    ! default never overrides a spelled-out intent.
    if (.not. runs( &
        'subroutine bump(v)'//new_line('a')// &
        '    integer, intent(inout) :: v'//new_line('a')// &
        '    v = v + 1'//new_line('a')// &
        '    print *, v'//new_line('a')// &
        'end subroutine'//new_line('a')// &
        ''//new_line('a')// &
        'integer :: k'//new_line('a')// &
        'k = 41'//new_line('a')// &
        'call bump(k)'//new_line('a'), 'lf', &
        '          42', 'ffc_lf_explicit_inout')) ok = .false.

    if (.not. ok) stop 1
    print *, 'PASS: lazy default real kind and default dummy intent apply'

contains

    ! Compile a source through the real ffc CLI, run it, and compare the first
    ! stdout line. The extension selects the dialect: .lf is lazy Fortran.
    logical function runs(source, ext, expected, stem) result(ok)
        character(len=*), intent(in) :: source
        character(len=*), intent(in) :: ext
        character(len=*), intent(in) :: expected
        character(len=*), intent(in) :: stem
        character(len=:), allocatable :: src_path, exe_path, out_path
        character(len=:), allocatable :: actual
        integer :: exit_stat, cmd_stat

        ok = .false.
        src_path = '/tmp/'//stem//'.'//ext
        exe_path = '/tmp/'//stem//'.exe'
        out_path = '/tmp/'//stem//'.out'
        call execute_command_line('rm -f '//src_path//' '//exe_path//' '//out_path)
        call write_source(src_path, source)
        call compile_source(src_path, exe_path, exit_stat, cmd_stat)
        if (cmd_stat /= 0 .or. exit_stat /= 0) then
            print *, 'FAIL: ffc rejected ', stem, ' exit=', exit_stat
            return
        end if

        call execute_command_line(exe_path//' > '//out_path, &
                                  exitstat=exit_stat, cmdstat=cmd_stat)
        if (cmd_stat /= 0 .or. exit_stat /= 0) then
            print *, 'FAIL: executable did not run cleanly: ', stem
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
    end function runs

    ! The compiler must refuse the source with a non-zero exit status.
    logical function rejects(source, ext, stem) result(ok)
        character(len=*), intent(in) :: source
        character(len=*), intent(in) :: ext
        character(len=*), intent(in) :: stem
        character(len=:), allocatable :: src_path, exe_path
        integer :: exit_stat, cmd_stat

        ok = .false.
        src_path = '/tmp/'//stem//'.'//ext
        exe_path = '/tmp/'//stem//'.exe'
        call execute_command_line('rm -f '//src_path//' '//exe_path)
        call write_source(src_path, source)
        call compile_source(src_path, exe_path, exit_stat, cmd_stat)
        call execute_command_line('rm -f '//src_path//' '//exe_path)
        if (cmd_stat == 0 .and. exit_stat == 0) then
            print *, 'FAIL: ffc accepted ', stem, ' but it must be rejected'
            return
        end if
        ok = .true.
    end function rejects

    subroutine write_source(path, source)
        character(len=*), intent(in) :: path
        character(len=*), intent(in) :: source
        integer :: unit

        open (newunit=unit, file=path, status='replace', action='write')
        write (unit, '(A)', advance='no') source
        close (unit)
    end subroutine write_source

    subroutine compile_source(src_path, exe_path, exit_stat, cmd_stat)
        character(len=*), intent(in) :: src_path
        character(len=*), intent(in) :: exe_path
        integer, intent(out) :: exit_stat
        integer, intent(out) :: cmd_stat
        character(len=:), allocatable :: command

        command = "sh -c 'exe=$(ls -t build/*/app/ffc build/fo/bin/ffc "// &
            "2>/dev/null | head -n 1); test -n ""$exe"" && ""$exe"" "// &
            src_path//' -o '//exe_path//" 2>/dev/null'"
        call execute_command_line(command, exitstat=exit_stat, cmdstat=cmd_stat)
    end subroutine compile_source

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

end program test_session_lazy_defaults_compiler
