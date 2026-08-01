! Issue #376: select and load the matching runtime archive.
!
! Behavioral oracle. For every backend that has an archive, this builds the
! archives, installs the matching one into a session, emits an executable that
! calls the packaged `_ffc_runtime_probe`, runs it, and requires exit status
! 42. That checks the archive was really loaded and resolved, not merely that
! a file was read.
!
! It also pins the selection contract: `default` selects the copy-patch
! artifact, a missing archive and a backend-mismatched archive each report the
! exact backend, and a blank artifact directory is an error.
!
! #565 removed the FFC_RUNTIME_ARCHIVE_DIR opt-in and the inline-runtime path
! it used to fall back to, so the directory is now an explicit argument and
! every failure to install a requested archive is loud.
program test_session_runtime_archive_compiler
    use liric_session_bindings, only: liric_session_t, liric_session_create, &
        destroy, lr_session_config_t, lr_operand_desc_t, emit_i32_call, &
        emit_ret_i32_operand, begin_i32_main, finish_and_emit_exe
    use liric_session_runtime_bindings, only: install_runtime_archive, &
        runtime_archive_name, runtime_archive_path, &
        effective_archive_backend, &
        LR_SESSION_BACKEND_DEFAULT, LR_SESSION_BACKEND_ISEL, &
        LR_SESSION_BACKEND_COPY_PATCH
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none

    character(len=*), parameter :: build_dir = '/tmp/ffc_runtime_376_build'
    character(len=*), parameter :: artifact_root = build_dir//'/artifacts'
    integer :: failures

    failures = 0

    print *, '=== runtime archive selection tests (#376) ==='

    call check_names(failures)
    call build_archives(failures)
    if (failures == 0) then
        call check_probe_runs(LR_SESSION_BACKEND_ISEL, 'isel', failures)
        call check_probe_runs(LR_SESSION_BACKEND_COPY_PATCH, 'copy-patch', &
                              failures)
        call check_probe_runs(LR_SESSION_BACKEND_DEFAULT, 'default', failures)
        call check_missing_archive(failures)
        call check_backend_mismatch(failures)
    end if
    call check_blank_directory_is_an_error(failures)

    if (failures /= 0) then
        print *, 'FAIL: ', failures, ' runtime archive check(s) failed'
        stop 1
    end if
    print *, 'PASS: runtime archive selection'

contains

    subroutine run(cmd, exit_stat)
        character(len=*), intent(in) :: cmd
        integer, intent(out) :: exit_stat
        integer :: cmd_stat

        call execute_command_line(cmd, exitstat=exit_stat, cmdstat=cmd_stat)
        if (cmd_stat /= 0) exit_stat = -1
    end subroutine run

    ! `default` must resolve to the copy-patch artifact, not one of its own.
    subroutine check_names(nfail)
        integer, intent(inout) :: nfail

        if (runtime_archive_name(LR_SESSION_BACKEND_ISEL) /= &
            'ffc-runtime-v2-isel.lrarch') then
            print *, 'FAIL: wrong isel archive name'
            nfail = nfail + 1
        end if
        if (runtime_archive_name(LR_SESSION_BACKEND_COPY_PATCH) /= &
            'ffc-runtime-v2-copy-patch.lrarch') then
            print *, 'FAIL: wrong copy-patch archive name'
            nfail = nfail + 1
        end if
        if (runtime_archive_name(LR_SESSION_BACKEND_DEFAULT) /= &
            'ffc-runtime-v2-copy-patch.lrarch') then
            print *, 'FAIL: default backend does not alias copy-patch'
            nfail = nfail + 1
        end if
        if (effective_archive_backend(LR_SESSION_BACKEND_DEFAULT) /= &
            LR_SESSION_BACKEND_COPY_PATCH) then
            print *, 'FAIL: default backend does not resolve to copy-patch'
            nfail = nfail + 1
        end if
        if (runtime_archive_path('/root', 'host', LR_SESSION_BACKEND_ISEL) /= &
            '/root/host/ffc-runtime-v2-isel.lrarch') then
            print *, 'FAIL: archive path is not target-qualified'
            nfail = nfail + 1
        end if
    end subroutine check_names

    subroutine build_archives(nfail)
        integer, intent(inout) :: nfail
        integer :: stat

        call run('rm -rf '//build_dir, stat)
        call run('cmake -S runtime -B '//build_dir// &
                 ' > /tmp/ffc_runtime_376_cfg.log 2>&1 && cmake --build '// &
                 build_dir//' -j 3 > /tmp/ffc_runtime_376_build.log 2>&1', &
                 stat)
        if (stat /= 0) then
            print *, 'FAIL: could not build runtime archives'
            call run('cat /tmp/ffc_runtime_376_cfg.log '// &
                     '/tmp/ffc_runtime_376_build.log', stat)
            nfail = nfail + 1
        end if
    end subroutine build_archives

    ! Emits `program main; return _ffc_runtime_probe(); end` through a session
    ! with the archive installed, then requires the executable to exit 42.
    subroutine check_probe_runs(backend, label, nfail)
        integer(c_int), intent(in) :: backend
        character(len=*), intent(in) :: label
        integer, intent(inout) :: nfail
        type(liric_session_t) :: session
        type(lr_session_config_t) :: config
        type(lr_operand_desc_t) :: args(0)
        type(lr_operand_desc_t) :: probe_result
        character(len=:), allocatable :: error_msg
        character(len=*), parameter :: exe = '/tmp/ffc_runtime_376_probe'
        integer :: stat

        config = lr_session_config_t()
        config%backend = backend

        call liric_session_create(session, error_msg, config)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL: ', label, ' session create: ', trim(error_msg)
            nfail = nfail + 1
            return
        end if

        if (.not. install_runtime_archive(session, artifact_root, backend, &
                                          'host', error_msg)) then
            print *, 'FAIL: ', label, ' archive install: ', trim(error_msg)
            call destroy(session)
            nfail = nfail + 1
            return
        end if

        if (.not. begin_i32_main(session, error_msg)) then
            print *, 'FAIL: ', label, ' begin main: ', trim(error_msg)
            call destroy(session)
            nfail = nfail + 1
            return
        end if
        if (.not. emit_i32_call(session, '_ffc_runtime_probe', args, &
                                probe_result, error_msg)) then
            print *, 'FAIL: ', label, ' probe call: ', trim(error_msg)
            call destroy(session)
            nfail = nfail + 1
            return
        end if
        if (.not. emit_ret_i32_operand(session, probe_result, error_msg)) then
            print *, 'FAIL: ', label, ' return: ', trim(error_msg)
            call destroy(session)
            nfail = nfail + 1
            return
        end if

        call run('rm -f '//exe, stat)
        if (.not. finish_and_emit_exe(session, exe, error_msg)) then
            print *, 'FAIL: ', label, ' emit exe: ', trim(error_msg)
            call destroy(session)
            nfail = nfail + 1
            return
        end if
        call destroy(session)

        call run(exe, stat)
        if (stat /= 42) then
            print *, 'FAIL: ', label, ' probe returned ', stat, ' expected 42'
            nfail = nfail + 1
        end if
        call run('rm -f '//exe, stat)
    end subroutine check_probe_runs

    ! A missing archive must name the backend it was looking for.
    subroutine check_missing_archive(nfail)
        integer, intent(inout) :: nfail
        type(liric_session_t) :: session
        type(lr_session_config_t) :: config
        character(len=:), allocatable :: error_msg

        config = lr_session_config_t()
        config%backend = LR_SESSION_BACKEND_ISEL
        call liric_session_create(session, error_msg, config)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL: session create for missing-archive check'
            nfail = nfail + 1
            return
        end if
        if (install_runtime_archive(session, '/tmp/ffc_runtime_376_absent', &
                                    LR_SESSION_BACKEND_ISEL, 'host', &
                                    error_msg)) then
            print *, 'FAIL: missing archive was accepted'
            nfail = nfail + 1
        else if (index(error_msg, 'ffc-runtime-v2-isel.lrarch') == 0) then
            print *, 'FAIL: missing-archive error does not name the ', &
                'backend artifact: ', trim(error_msg)
            nfail = nfail + 1
        end if
        call destroy(session)
    end subroutine check_missing_archive

    ! A target/backend-mismatched archive must be rejected, naming both the
    ! recorded and the requested backend, rather than silently accepted.
    subroutine check_backend_mismatch(nfail)
        integer, intent(inout) :: nfail
        type(liric_session_t) :: session
        type(lr_session_config_t) :: config
        character(len=:), allocatable :: error_msg
        character(len=*), parameter :: bad_root = '/tmp/ffc_runtime_376_bad'
        integer :: stat

        ! Put the copy-patch archive where the isel archive belongs.
        call run('rm -rf '//bad_root//' && mkdir -p '//bad_root//'/host', stat)
        call run('cp '//artifact_root//'/host/ffc-runtime-v2-copy-patch.lrarch '// &
                 bad_root//'/host/ffc-runtime-v2-isel.lrarch', stat)
        if (stat /= 0) then
            print *, 'FAIL: could not stage mismatched archive'
            nfail = nfail + 1
            return
        end if

        config = lr_session_config_t()
        config%backend = LR_SESSION_BACKEND_ISEL
        call liric_session_create(session, error_msg, config)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL: session create for mismatch check'
            nfail = nfail + 1
            return
        end if
        if (install_runtime_archive(session, bad_root, &
                                    LR_SESSION_BACKEND_ISEL, 'host', &
                                    error_msg)) then
            print *, 'FAIL: backend-mismatched archive was accepted'
            nfail = nfail + 1
        else
            if (index(error_msg, 'copy-patch') == 0 .or. &
                index(error_msg, 'isel') == 0) then
                print *, 'FAIL: mismatch error does not name both backends: ', &
                    trim(error_msg)
                nfail = nfail + 1
            end if
        end if
        call destroy(session)
        call run('rm -rf '//bad_root, stat)
    end subroutine check_backend_mismatch

    ! A blank artifact directory is a caller error, not a licence to skip
    ! installation. The opt-out that used to live here is gone (#565).
    subroutine check_blank_directory_is_an_error(nfail)
        integer, intent(inout) :: nfail
        type(liric_session_t) :: session
        type(lr_session_config_t) :: config
        character(len=:), allocatable :: error_msg

        config = lr_session_config_t()
        call liric_session_create(session, error_msg, config)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL: session create for blank-directory check'
            nfail = nfail + 1
            return
        end if
        if (install_runtime_archive(session, '   ', &
                                    LR_SESSION_BACKEND_DEFAULT, 'host', &
                                    error_msg)) then
            print *, 'FAIL: a blank archive directory was accepted'
            nfail = nfail + 1
        else if (len_trim(error_msg) == 0) then
            print *, 'FAIL: a blank archive directory failed without a reason'
            nfail = nfail + 1
        end if
        call destroy(session)
    end subroutine check_blank_directory_is_an_error

end program test_session_runtime_archive_compiler
