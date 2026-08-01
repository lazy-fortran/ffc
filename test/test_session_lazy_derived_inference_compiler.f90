program test_session_lazy_derived_inference_compiler
    !! Lazy Fortran derived inference (#429). A bare `v = outer_t(...)` binding
    !! in a .lf source has no declaration: FortFront resolves the constructor's
    !! concrete derived type and hands ffc one binding-keyed symbol with the
    !! nested layout. ffc must allocate that layout, run the constructor and
    !! default component lowering, and read components back with the value kind
    !! the component actually has -- an integer component used in a real
    !! expression must be converted, not reinterpreted. A constructor whose
    !! type contradicts the inferred binding, or a constructor naming a type
    !! that does not exist, must be a compile-time diagnostic.
    implicit none

    logical :: ok

    print *, '=== direct session lazy derived inference compiler test ==='

    ok = .true.

    ! Nested derived value inferred from the constructor: component chain reads
    ! back through both levels.
    if (.not. lazy_runs( &
        'type :: inner_t'//new_line('a')// &
        '    integer :: k'//new_line('a')// &
        'end type inner_t'//new_line('a')// &
        'type :: outer_t'//new_line('a')// &
        '    type(inner_t) :: part'//new_line('a')// &
        '    integer :: n'//new_line('a')// &
        'end type outer_t'//new_line('a')// &
        'v = outer_t(inner_t(7), 5)'//new_line('a')// &
        'print *, v%part%k, v%n'//new_line('a'), &
        '           7           5', &
        'ffc_lazy_der_nested')) ok = .false.

    ! An integer component chain feeding a second inferred binding. FortFront
    ! infers `m` as the Lazy default real (real(8)), so the integer sum must be
    ! converted to real: reinterpreting the integer bit pattern prints a
    ! denormal instead of 12.
    if (.not. lazy_runs( &
        'type :: inner_t'//new_line('a')// &
        '    integer :: k'//new_line('a')// &
        'end type inner_t'//new_line('a')// &
        'type :: outer_t'//new_line('a')// &
        '    type(inner_t) :: part'//new_line('a')// &
        '    integer :: n'//new_line('a')// &
        'end type outer_t'//new_line('a')// &
        'v = outer_t(inner_t(7), 5)'//new_line('a')// &
        'm = v%part%k + v%n'//new_line('a')// &
        'print *, m'//new_line('a'), &
        '   12.000000000000000', &
        'ffc_lazy_der_mixed')) ok = .false.

    ! A single integer component read in a real context takes the same
    ! conversion path as the mixed sum above.
    if (.not. lazy_runs( &
        'type :: inner_t'//new_line('a')// &
        '    integer :: k'//new_line('a')// &
        'end type inner_t'//new_line('a')// &
        'type :: outer_t'//new_line('a')// &
        '    type(inner_t) :: part'//new_line('a')// &
        'end type outer_t'//new_line('a')// &
        'v = outer_t(inner_t(3))'//new_line('a')// &
        'r = v%part%k'//new_line('a')// &
        'print *, r'//new_line('a'), &
        '   3.0000000000000000', &
        'ffc_lazy_der_scalar')) ok = .false.

    ! Real and character components of a nested inferred value keep their own
    ! kinds.
    if (.not. lazy_runs( &
        'type :: inner_t'//new_line('a')// &
        '    real :: x'//new_line('a')// &
        '    character(len=3) :: s'//new_line('a')// &
        'end type inner_t'//new_line('a')// &
        'type :: outer_t'//new_line('a')// &
        '    type(inner_t) :: part'//new_line('a')// &
        '    integer :: n'//new_line('a')// &
        'end type outer_t'//new_line('a')// &
        'v = outer_t(inner_t(1.5, "abc"), 2)'//new_line('a')// &
        'print *, v%part%s, v%n'//new_line('a'), &
        ' abc           2', &
        'ffc_lazy_der_kinds')) ok = .false.

    ! Negative: a later constructor of a different derived type contradicts the
    ! inferred binding type.
    if (.not. lazy_rejected( &
        'type :: p_t'//new_line('a')// &
        '    integer :: a'//new_line('a')// &
        'end type p_t'//new_line('a')// &
        'type :: q_t'//new_line('a')// &
        '    integer :: a'//new_line('a')// &
        'end type q_t'//new_line('a')// &
        'v = p_t(1)'//new_line('a')// &
        'v = q_t(2)'//new_line('a')// &
        'print *, v%a'//new_line('a'), &
        'ffc_lazy_der_conflict')) ok = .false.

    ! Negative: a constructor naming a type that was never defined cannot
    ! resolve to a derived identity.
    if (.not. lazy_rejected( &
        'type :: p_t'//new_line('a')// &
        '    integer :: a'//new_line('a')// &
        'end type p_t'//new_line('a')// &
        'v = z_t(1)'//new_line('a')// &
        'print *, v%a'//new_line('a'), &
        'ffc_lazy_der_unknown')) ok = .false.

    if (.not. ok) stop 1
    print *, 'PASS: lazy derived inference lowers and runs'

contains

    ! Write a lazy fragment to a .lf file, drive it through the ffc CLI, run the
    ! executable and compare the first stdout line.
    logical function lazy_runs(source, expected, stem) result(ok)
        character(len=*), intent(in) :: source
        character(len=*), intent(in) :: expected
        character(len=*), intent(in) :: stem
        character(len=:), allocatable :: src_path, exe_path, out_path
        character(len=:), allocatable :: actual
        integer :: exit_stat, cmd_stat

        ok = .false.
        src_path = '/tmp/'//stem//'.lf'
        exe_path = '/tmp/'//stem//'.exe'
        out_path = '/tmp/'//stem//'.out'
        call write_source(source, src_path, exe_path, out_path)

        call compile_source(src_path, exe_path, exit_stat, cmd_stat)
        if (cmd_stat /= 0 .or. exit_stat /= 0) then
            print *, 'FAIL: ffc rejected lazy derived source ', stem, &
                ' exit=', exit_stat
            return
        end if

        call execute_command_line(exe_path//' > '//out_path, &
                                  exitstat=exit_stat, cmdstat=cmd_stat)
        if (cmd_stat /= 0 .or. exit_stat /= 0) then
            print *, 'FAIL: lazy derived executable did not run cleanly: ', stem
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

    ! A lazy fragment that must not compile: ffc has to exit non-zero.
    logical function lazy_rejected(source, stem) result(ok)
        character(len=*), intent(in) :: source
        character(len=*), intent(in) :: stem
        character(len=:), allocatable :: src_path, exe_path, out_path
        integer :: exit_stat, cmd_stat

        ok = .false.
        src_path = '/tmp/'//stem//'.lf'
        exe_path = '/tmp/'//stem//'.exe'
        out_path = '/tmp/'//stem//'.out'
        call write_source(source, src_path, exe_path, out_path)

        call compile_source(src_path, exe_path, exit_stat, cmd_stat)
        call execute_command_line('rm -f '//src_path//' '//exe_path//' '//out_path)
        if (cmd_stat == 0 .and. exit_stat == 0) then
            print *, 'FAIL: ffc accepted an invalid lazy derived binding: ', stem
            return
        end if
        ok = .true.
    end function lazy_rejected

    subroutine write_source(source, src_path, exe_path, out_path)
        character(len=*), intent(in) :: source
        character(len=*), intent(in) :: src_path
        character(len=*), intent(in) :: exe_path
        character(len=*), intent(in) :: out_path
        integer :: unit

        call execute_command_line('rm -f '//src_path//' '//exe_path//' '//out_path)
        open (newunit=unit, file=src_path, status='replace', action='write')
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
            src_path//' -o '//exe_path//"'"
        call execute_command_line(command, exitstat=exit_stat, cmdstat=cmd_stat)
    end subroutine compile_source

    function read_first_line(path) result(line)
        character(len=*), intent(in) :: path
        character(len=:), allocatable :: line
        character(len=256) :: buffer
        integer :: unit, io_stat

        line = ''
        open (newunit=unit, file=path, status='old', action='read', iostat=io_stat)
        if (io_stat == 0) then
            read (unit, '(A)', iostat=io_stat) buffer
            if (io_stat == 0) line = trim(buffer)
            close (unit)
        end if
    end function read_first_line

end program test_session_lazy_derived_inference_compiler
