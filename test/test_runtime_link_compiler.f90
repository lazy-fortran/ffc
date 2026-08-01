! Issue #565: ffc links its runtime into every executable it emits.
!
! Behavioral oracle for the delivery decision. The four issues that move
! compiler-emitted code behind the runtime ABI (#396, #423, #427, #428) all
! depend on an emitted call to a runtime symbol resolving in the produced
! binary, with no environment variable set and no inline fallback.
!
! What is checked, and why each check is here:
!
!   1. Every executable the real lowering path emits defines the runtime
!      entry points. Proved by compiling an ordinary Fortran program through
!      lower_program_to_liric_exe and reading the runtime symbols out of the
!      resulting binary's symbol table. This fails on the pre-#565 compiler,
!      which linked no runtime unless FFC_RUNTIME_ARCHIVE_DIR was set.
!   2. The linked runtime is callable, not merely present: a session that
!      calls `_ffc_runtime_probe` and links the same runtime input runs and
!      exits 42.
!   3. Without that link input the same session produces a binary that dies
!      with an undefined symbol. This is the silent failure mode the design
!      exists to remove, and pinning it keeps check 2 honest.
!   4. Every symbol in FFC_RUNTIME_SYMBOLS is really defined by the embedded
!      runtime source. A lowering that emits a call to a symbol the runtime
!      does not define would otherwise produce a binary that links cleanly
!      and then dies at run time.
!   5. The embedded runtime source is byte-identical to runtime/ffc_runtime.c,
!      so the compiler and the CMake-packaged archives cannot drift apart.
program test_runtime_link_compiler
    use ffc_runtime_link, only: ffc_runtime_link_input, FFC_RUNTIME_SYMBOLS
    use ffc_runtime_source, only: ffc_runtime_source_text
    use liric_session_bindings, only: liric_session_t, liric_session_create, &
        destroy, lr_session_config_t, lr_operand_desc_t, emit_i32_call, &
        emit_ret_i32_operand, begin_i32_main, finish_and_emit_exe, &
        finish_and_emit_exe_objects
    use ffc_test_support, only: compile_to_exe
    implicit none

    character(len=*), parameter :: WORK = '/tmp/ffc_runtime_link_565'
    integer :: failures

    failures = 0
    call run_quiet('rm -rf '//WORK//' && mkdir -p '//WORK)

    print *, '=== runtime link tests (#565) ==='

    call check_embedded_source_matches_file(failures)
    call check_declared_symbols_are_defined(failures)
    call check_probe_runs_with_runtime(failures)
    call check_probe_fails_without_runtime(failures)
    call check_emitted_executable_carries_runtime(failures)

    if (failures /= 0) then
        print *, 'FAIL: ', failures, ' runtime link check(s) failed'
        stop 1
    end if
    print *, 'PASS: runtime link'

contains

    subroutine run_quiet(command)
        character(len=*), intent(in) :: command
        integer :: exit_stat, cmd_stat

        call execute_command_line(command, exitstat=exit_stat, &
                                  cmdstat=cmd_stat)
    end subroutine run_quiet

    ! Runs a command and returns its exit status without aborting the test
    ! when the command fails.
    integer function status_of(command) result(status)
        character(len=*), intent(in) :: command
        character(len=*), parameter :: rc_file = WORK//'/rc'
        integer :: unit, ios, exit_stat, cmd_stat

        status = -1
        call execute_command_line('{ '//command//' ; } > /dev/null 2>&1; '// &
                                  'echo $? > '//rc_file, exitstat=exit_stat, &
                                  cmdstat=cmd_stat)
        if (cmd_stat /= 0) return
        open (newunit=unit, file=rc_file, status='old', iostat=ios)
        if (ios /= 0) return
        read (unit, *, iostat=ios) status
        close (unit)
        if (ios /= 0) status = -1
    end function status_of

    subroutine read_text_file(path, text, ok)
        character(len=*), intent(in) :: path
        character(len=:), allocatable, intent(out) :: text
        logical, intent(out) :: ok
        integer :: unit, ios
        integer(kind=8) :: nbytes

        ok = .false.
        open (newunit=unit, file=path, access='stream', form='unformatted', &
              status='old', action='read', iostat=ios)
        if (ios /= 0) then
            text = ''
            return
        end if
        inquire (unit=unit, size=nbytes)
        if (nbytes <= 0) then
            close (unit)
            text = ''
            return
        end if
        allocate (character(len=nbytes) :: text)
        read (unit, iostat=ios) text
        close (unit)
        ok = ios == 0
    end subroutine read_text_file

    ! The generated embedding must reproduce runtime/ffc_runtime.c exactly,
    ! so the linked runtime and the archived runtime are the same code.
    subroutine check_embedded_source_matches_file(nfail)
        integer, intent(inout) :: nfail
        character(len=:), allocatable :: embedded, on_disk
        logical :: ok

        call ffc_runtime_source_text(embedded)
        call read_text_file('runtime/ffc_runtime.c', on_disk, ok)
        if (.not. ok) then
            print *, 'FAIL: cannot read runtime/ffc_runtime.c'
            nfail = nfail + 1
            return
        end if
        if (embedded /= on_disk) then
            print *, 'FAIL: the embedded runtime source has drifted from ', &
                'runtime/ffc_runtime.c; run scripts/generate_runtime_source.sh'
            nfail = nfail + 1
        end if
    end subroutine check_embedded_source_matches_file

    ! Guards the lowerer: a symbol it is allowed to call must exist.
    subroutine check_declared_symbols_are_defined(nfail)
        integer, intent(inout) :: nfail
        character(len=:), allocatable :: link_input, error_msg, symbols
        character(len=*), parameter :: obj = WORK//'/runtime.o'
        character(len=*), parameter :: nm_out = WORK//'/runtime.nm'
        logical :: ok
        integer :: i, status

        call ffc_runtime_link_input(link_input, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL: runtime link input: ', trim(error_msg)
            nfail = nfail + 1
            return
        end if

        status = status_of('cc -c -o '//obj//' '//link_input)
        if (status /= 0) then
            print *, 'FAIL: the embedded runtime source does not compile'
            nfail = nfail + 1
            return
        end if
        status = status_of('nm --defined-only -g '//obj//' > '//nm_out)
        if (status /= 0) then
            print *, 'FAIL: cannot list the runtime object symbols'
            nfail = nfail + 1
            return
        end if
        call read_text_file(nm_out, symbols, ok)
        if (.not. ok) then
            print *, 'FAIL: cannot read the runtime object symbol list'
            nfail = nfail + 1
            return
        end if
        do i = 1, size(FFC_RUNTIME_SYMBOLS)
            if (index(symbols, ' '//trim(FFC_RUNTIME_SYMBOLS(i))// &
                      new_line('a')) == 0) then
                print *, 'FAIL: the runtime does not define declared ', &
                    'symbol ', trim(FFC_RUNTIME_SYMBOLS(i))
                nfail = nfail + 1
            end if
        end do
    end subroutine check_declared_symbols_are_defined

    ! Emits a session that calls the runtime probe. `with_runtime` selects
    ! whether the runtime link input is supplied, which is the only
    ! difference between the two binaries.
    subroutine emit_probe_exe(with_runtime, exe, emitted)
        logical, intent(in) :: with_runtime
        character(len=*), intent(in) :: exe
        logical, intent(out) :: emitted
        type(liric_session_t) :: session
        type(lr_session_config_t) :: config
        type(lr_operand_desc_t) :: args(0)
        type(lr_operand_desc_t) :: probe_result
        character(len=:), allocatable :: error_msg, link_input
        character(len=4096) :: inputs(1)

        emitted = .false.
        config = lr_session_config_t()
        call liric_session_create(session, error_msg, config)
        if (len_trim(error_msg) > 0) return
        if (.not. begin_i32_main(session, error_msg)) then
            call destroy(session)
            return
        end if
        if (.not. emit_i32_call(session, '_ffc_runtime_probe', args, &
                                probe_result, error_msg)) then
            call destroy(session)
            return
        end if
        if (.not. emit_ret_i32_operand(session, probe_result, error_msg)) then
            call destroy(session)
            return
        end if
        if (with_runtime) then
            call ffc_runtime_link_input(link_input, error_msg)
            if (len_trim(error_msg) > 0) then
                call destroy(session)
                return
            end if
            inputs(1) = link_input
            emitted = finish_and_emit_exe_objects(session, exe, inputs, &
                                                  error_msg)
        else
            emitted = finish_and_emit_exe(session, exe, error_msg)
        end if
        call destroy(session)
    end subroutine emit_probe_exe

    subroutine check_probe_runs_with_runtime(nfail)
        integer, intent(inout) :: nfail
        character(len=*), parameter :: exe = WORK//'/probe_linked'
        logical :: emitted
        integer :: status

        call emit_probe_exe(.true., exe, emitted)
        if (.not. emitted) then
            print *, 'FAIL: could not emit the probe executable'
            nfail = nfail + 1
            return
        end if
        status = status_of(exe)
        if (status /= 42) then
            print *, 'FAIL: linked runtime probe returned ', status, &
                ' instead of 42'
            nfail = nfail + 1
        end if
    end subroutine check_probe_runs_with_runtime

    ! Pins the failure mode the linked runtime removes: without it the call
    ! still links, and the binary dies at run time instead.
    subroutine check_probe_fails_without_runtime(nfail)
        integer, intent(inout) :: nfail
        character(len=*), parameter :: exe = WORK//'/probe_unlinked'
        logical :: emitted
        integer :: status

        call emit_probe_exe(.false., exe, emitted)
        if (.not. emitted) return
        status = status_of(exe)
        if (status == 42) then
            print *, 'FAIL: the probe resolved without linking the runtime, ', &
                'so this test cannot tell the two apart'
            nfail = nfail + 1
        end if
    end subroutine check_probe_fails_without_runtime

    ! The end-to-end guarantee: an ordinary compile links the runtime.
    subroutine check_emitted_executable_carries_runtime(nfail)
        integer, intent(inout) :: nfail
        character(len=*), parameter :: exe = WORK//'/hello'
        character(len=*), parameter :: nm_out = WORK//'/hello.nm'
        character(len=:), allocatable :: error_msg, symbols
        logical :: ok
        integer :: i, status

        call compile_to_exe( &
            'program main'//new_line('a')// &
            '    integer :: value'//new_line('a')// &
            '    value = 7'//new_line('a')// &
            '    stop value'//new_line('a')// &
            'end program main'//new_line('a'), exe, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL: compiling a program failed: ', trim(error_msg)
            nfail = nfail + 1
            return
        end if

        status = status_of(exe)
        if (status /= 7) then
            print *, 'FAIL: the compiled program exited ', status, &
                ' instead of 7'
            nfail = nfail + 1
        end if

        status = status_of('nm --defined-only '//exe//' > '//nm_out)
        if (status /= 0) then
            print *, 'FAIL: cannot list the emitted executable symbols'
            nfail = nfail + 1
            return
        end if
        call read_text_file(nm_out, symbols, ok)
        if (.not. ok) then
            print *, 'FAIL: cannot read the emitted executable symbol list'
            nfail = nfail + 1
            return
        end if
        do i = 1, size(FFC_RUNTIME_SYMBOLS)
            if (index(symbols, ' '//trim(FFC_RUNTIME_SYMBOLS(i))// &
                      new_line('a')) == 0) then
                print *, 'FAIL: the emitted executable does not carry ', &
                    'runtime symbol ', trim(FFC_RUNTIME_SYMBOLS(i))
                nfail = nfail + 1
            end if
        end do
    end subroutine check_emitted_executable_carries_runtime

end program test_runtime_link_compiler
