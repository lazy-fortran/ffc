program test_session_lazy_monomorph_compiler
    !! A Lazy procedure may leave its dummies untyped and take their types from
    !! use. When every call site agrees on one concrete type, FortFront resolves
    !! the dummies in place and one body lowers. When call sites disagree there
    !! is no one type for the dummies, so FortFront monomorphizes: it leaves the
    !! written procedure untyped and adds one typed copy per signature, but the
    !! call sites still name the written procedure (FortFront #2971).
    !!
    !! ffc must not pick a copy for them. Binding one is a guess, and binding
    !! the wrong one returns a wrong value with no diagnostic: before this
    !! change the two-signature program below printed 6 and 0, and swapping the
    !! two calls printed 0.0 and 6.0. The boundary is now diagnosed (#437).
    implicit none

    logical :: ok

    print *, '=== direct session lazy monomorphization compiler test ==='

    ok = .true.

    ! Every call site agrees on one concrete type: FortFront resolves the dummy
    ! and the single body lowers. This is the path monomorphization must not
    ! disturb.
    if (.not. lazy_first_line( &
        'function twice(x)'//new_line('a')// &
        '  twice = 2 * x'//new_line('a')// &
        'end function'//new_line('a')// &
        'program main'//new_line('a')// &
        '  print *, twice(3)'//new_line('a')// &
        '  print *, twice(4)'//new_line('a')// &
        'end program'//new_line('a'), &
        '           6', 'ffc_lazy_mono_same_signature')) ok = .false.

    ! One real call site resolves the dummy to real just as well.
    if (.not. lazy_first_line( &
        'function twice(x)'//new_line('a')// &
        '  twice = 2 * x'//new_line('a')// &
        'end function'//new_line('a')// &
        'program main'//new_line('a')// &
        '  print *, twice(2.5)'//new_line('a')// &
        'end program'//new_line('a'), &
        '   5.0000000000000000', 'ffc_lazy_mono_real_signature')) ok = .false.

    ! Two call sites of different concrete types. The written name denotes no
    ! callable body, so the call is refused rather than bound to whichever copy
    ! happened to be emitted.
    if (.not. lazy_rejected( &
        'function twice(x)'//new_line('a')// &
        '  twice = 2 * x'//new_line('a')// &
        'end function'//new_line('a')// &
        'program main'//new_line('a')// &
        '  print *, twice(3)'//new_line('a')// &
        '  print *, twice(2.5)'//new_line('a')// &
        'end program'//new_line('a'), &
        'monomorphic specializations', &
        'ffc_lazy_mono_conflicting')) ok = .false.

    ! The same program with the calls in the other order is refused the same
    ! way: which call is wrong must not depend on emission order.
    if (.not. lazy_rejected( &
        'function twice(x)'//new_line('a')// &
        '  twice = 2 * x'//new_line('a')// &
        'end function'//new_line('a')// &
        'program main'//new_line('a')// &
        '  print *, twice(2.5)'//new_line('a')// &
        '  print *, twice(3)'//new_line('a')// &
        'end program'//new_line('a'), &
        'monomorphic specializations', &
        'ffc_lazy_mono_conflicting_reversed')) ok = .false.

    if (.not. ok) stop 1
    print *, 'PASS: lazy monomorphization boundary'

contains

    logical function lazy_first_line(source, expected, stem) result(ok)
        ! Compile and run a lazy fragment, comparing its first output line.
        character(len=*), intent(in) :: source
        character(len=*), intent(in) :: expected
        character(len=*), intent(in) :: stem
        character(len=:), allocatable :: dir, src_path, exe_path, out_path
        character(len=:), allocatable :: actual
        integer :: unit, exit_stat, cmd_stat

        ok = .false.
        call scratch_dir(stem, dir)
        src_path = dir//'/'//stem//'.lf'
        exe_path = dir//'/'//stem//'.exe'
        out_path = dir//'/'//stem//'.out'
        open (newunit=unit, file=src_path, status='replace', action='write')
        write (unit, '(A)', advance='no') source
        close (unit)

        call execute_command_line( &
            "sh -c 'exe=$(ls -t build/*/app/ffc build/fo/bin/ffc 2>/dev/null | "// &
            'head -n 1); test -n "$exe" || exit 90; '// &
            '"$exe" '//src_path//' -o '//exe_path//" >"//dir//"/log 2>&1'", &
            exitstat=exit_stat, cmdstat=cmd_stat)
        if (cmd_stat /= 0 .or. exit_stat /= 0) then
            print *, 'FAIL: ffc rejected lazy source ', stem, ' exit=', exit_stat
            call execute_command_line('cat '//dir//'/log')
            return
        end if

        call execute_command_line(exe_path//' > '//out_path, &
                                  exitstat=exit_stat, cmdstat=cmd_stat)
        if (cmd_stat /= 0 .or. exit_stat /= 0) then
            print *, 'FAIL: lazy executable did not run cleanly: ', stem
            return
        end if
        actual = read_first_line(out_path)
        if (trim(actual) /= trim(expected)) then
            print *, 'FAIL: ', stem, ' expected [', trim(expected), &
                '] got [', trim(actual), ']'
            return
        end if
        call execute_command_line('rm -rf '//dir)
        ok = .true.
    end function lazy_first_line

    logical function lazy_rejected(source, fragment, stem) result(ok)
        ! Compiling this lazy fragment must fail with a diagnostic naming the
        ! given fragment. A clean compile would mean ffc bound the call to one
        ! specialization and silently returned a wrong value for the others.
        character(len=*), intent(in) :: source
        character(len=*), intent(in) :: fragment
        character(len=*), intent(in) :: stem
        character(len=:), allocatable :: dir, src_path, exe_path
        integer :: unit, exit_stat, cmd_stat, grep_stat

        ok = .false.
        call scratch_dir(stem, dir)
        src_path = dir//'/'//stem//'.lf'
        exe_path = dir//'/'//stem//'.exe'
        open (newunit=unit, file=src_path, status='replace', action='write')
        write (unit, '(A)', advance='no') source
        close (unit)

        call execute_command_line( &
            "sh -c 'exe=$(ls -t build/*/app/ffc build/fo/bin/ffc 2>/dev/null | "// &
            'head -n 1); test -n "$exe" || exit 90; '// &
            '"$exe" '//src_path//' -o '//exe_path//" >"//dir//"/log 2>&1'", &
            exitstat=exit_stat, cmdstat=cmd_stat)
        if (cmd_stat /= 0) then
            print *, 'FAIL: could not run ffc for ', stem
            return
        end if
        if (exit_stat == 0) then
            print *, 'FAIL: ', stem, ' compiled instead of being diagnosed'
            call execute_command_line('cat '//dir//'/log')
            return
        end if
        call execute_command_line('grep -q "'//fragment//'" '//dir//'/log', &
                                  exitstat=grep_stat, cmdstat=cmd_stat)
        if (cmd_stat /= 0 .or. grep_stat /= 0) then
            print *, 'FAIL: ', stem, ' diagnostic did not mention: ', fragment
            call execute_command_line('cat '//dir//'/log')
            return
        end if
        call execute_command_line('rm -rf '//dir)
        ok = .true.
    end function lazy_rejected

    subroutine scratch_dir(stem, dir)
        ! A scratch directory of this run's own, so concurrent builds of other
        ! worktrees never share it (ffc #547).
        character(len=*), intent(in) :: stem
        character(len=:), allocatable, intent(out) :: dir
        character(len=32) :: stamp
        integer :: values(8)

        call date_and_time(values=values)
        write (stamp, '(I0,A,I0)') values(6)*60000 + values(7)*1000 + &
            values(8), '_', values(5)
        dir = '/tmp/'//stem//'_'//trim(stamp)
        call execute_command_line('rm -rf '//dir//'; mkdir -p '//dir)
    end subroutine scratch_dir

    function read_first_line(path) result(text)
        character(len=*), intent(in) :: path
        character(len=:), allocatable :: text
        character(len=512) :: buffer
        integer :: unit, io_stat

        text = ''
        open (newunit=unit, file=path, status='old', action='read', &
              iostat=io_stat)
        if (io_stat /= 0) return
        read (unit, '(A)', iostat=io_stat) buffer
        if (io_stat == 0) text = trim(buffer)
        close (unit)
    end function read_first_line

end program test_session_lazy_monomorph_compiler
