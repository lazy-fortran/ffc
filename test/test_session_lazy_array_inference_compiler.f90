program test_session_lazy_array_inference_compiler
    !! Lazy Fortran array inference (#425). A bare `name = [...]` binding in a
    !! .lf source has no declaration: FortFront infers the element type, rank
    !! and shape and hands ffc one stable descriptor before executable
    !! lowering. ffc must consume that inferred contract for every element type
    !! it supports -- integer, real and character -- and must reject a later
    !! assignment whose shape contradicts the inferred contract rather than
    !! silently storing a wrong number of elements.
    implicit none

    logical :: ok

    print *, '=== direct session lazy array inference compiler test ==='

    ok = .true.

    ! Integer constructor: shape and element type come from the constructor.
    if (.not. lazy_runs( &
        'a = [1, 2, 3]'//new_line('a')// &
        'print *, a'//new_line('a'), &
        '           1           2           3', &
        'ffc_lazy_arr_int')) ok = .false.

    ! Real constructor plus a whole-array expression: the inferred descriptor
    ! of `b` is derived from the array expression, not from a literal.
    if (.not. lazy_runs( &
        'a = [1.0, 2.0, 3.0]'//new_line('a')// &
        'b = a * 2.0'//new_line('a')// &
        'print *, b'//new_line('a'), &
        '   2.00000000       4.00000000       6.00000000', &
        'ffc_lazy_arr_real')) ok = .false.

    ! Integer array expression between two inferred bindings.
    if (.not. lazy_runs( &
        'a = [1, 2, 3]'//new_line('a')// &
        'b = a + [10, 20, 30]'//new_line('a')// &
        'print *, b'//new_line('a'), &
        '          11          22          33', &
        'ffc_lazy_arr_expr')) ok = .false.

    ! Character constructor: FortFront infers character(len=2) :: a(2). The
    ! whole-array constructor assignment must store string slots, not attempt
    ! an integer literal fold.
    if (.not. lazy_runs( &
        'a = ["ab", "cd"]'//new_line('a')// &
        'print *, a'//new_line('a'), &
        ' abcd', &
        'ffc_lazy_arr_char')) ok = .false.

    ! Negative: a later constructor whose element count contradicts the single
    ! inferred descriptor must be a compile-time Lazy diagnostic.
    if (.not. lazy_rejected( &
        'a = [1, 2, 3]'//new_line('a')// &
        'a = [4, 5, 6, 7]'//new_line('a')// &
        'print *, a'//new_line('a'), &
        'ffc_lazy_arr_rank_conflict')) ok = .false.

    if (.not. ok) stop 1
    print *, 'PASS: lazy array inference lowers and runs'

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
            print *, 'FAIL: ffc rejected lazy array source ', stem, &
                ' exit=', exit_stat
            return
        end if

        call execute_command_line(exe_path//' > '//out_path, &
                                  exitstat=exit_stat, cmdstat=cmd_stat)
        if (cmd_stat /= 0 .or. exit_stat /= 0) then
            print *, 'FAIL: lazy array executable did not run cleanly: ', stem
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
            print *, 'FAIL: ffc accepted a conflicting lazy array shape: ', stem
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
        if (io_stat /= 0) return
        read (unit, '(A)', iostat=io_stat) buffer
        if (io_stat == 0) line = trim(buffer)
        close (unit)
    end function read_first_line

end program test_session_lazy_array_inference_compiler
