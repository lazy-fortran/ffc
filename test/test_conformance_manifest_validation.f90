program test_conformance_manifest_validation
    implicit none

    character(len=*), parameter :: SCRIPT = &
        'scripts/conformance_gauntlet.sh'
    character(len=*), parameter :: FIXTURE_DIR = &
        '/tmp/ffc_manifest_validation_fixtures'
    character(len=*), parameter :: XFAIL_MANIFEST = &
        '/tmp/ffc_manifest_validation_xfail.txt'
    character(len=*), parameter :: SKIP_MANIFEST = &
        '/tmp/ffc_manifest_validation_skip.txt'
    character(len=*), parameter :: REPORT = &
        '/tmp/ffc_manifest_validation.jsonl'
    character(len=*), parameter :: FAILURE_LOG = &
        '/tmp/ffc_manifest_validation_failure.log'
    character(len=*), parameter :: AUDIT_MANIFEST = &
        '/tmp/ffc_manifest_validation_audit.txt'
    character(len=*), parameter :: FAKE_BIN_DIR = &
        '/tmp/ffc_manifest_validation_bin'
    character(len=64), parameter :: DIAGNOSTICS(7) = [character(len=64) :: &
        'reason is required', &
        'owner or scope is required', &
        'owner and scope cannot both appear', &
        'malformed owner', &
        'duplicate path', &
        'unknown scope', &
        'malformed delimiter']
    integer :: case_number
    logical :: passed

    passed = .true.
    call write_fixture()
    call write_valid_manifests()
    if (.not. valid_manifest_runs()) passed = .false.

    do case_number = 1, size(DIAGNOSTICS)
        call write_invalid_manifest(case_number)
        if (.not. invalid_manifest_fails(case_number, &
            trim(DIAGNOSTICS(case_number)))) passed = .false.
    end do
    call write_fake_gh()
    if (.not. audit_owner('378', .true., '')) passed = .false.
    if (.not. audit_owner('445', .false., 'is CLOSED')) passed = .false.
    if (.not. audit_owner('999999', .false., 'is MISSING')) passed = .false.

    if (.not. passed) stop 1
    print *, 'PASS: expected-disposition manifest validation'

contains

    subroutine write_fixture()
        integer :: unit

        call execute_command_line('rm -rf '//FIXTURE_DIR)
        call execute_command_line('mkdir -p '//FIXTURE_DIR)
        open(newunit=unit, file=FIXTURE_DIR//'/owned_failure.f90', &
            status='replace', action='write')
        write(unit, '(A)') 'subroutine owned_failure'
        write(unit, '(A)') '! { dg-error "synthetic accepted case" }'
        write(unit, '(A)') 'end subroutine owned_failure'
        close(unit)
    end subroutine write_fixture

    subroutine write_valid_manifests()
        integer :: unit

        open(newunit=unit, file=XFAIL_MANIFEST, status='replace', &
            action='write')
        write(unit, '(A)') 'owned_failure.f90 # owner=lazy-fortran/ffc#445;'// &
            ' reason=closed owners are offline-valid'
        write(unit, '(A)') 'coarray.f90 # scope=coarray; reason=excluded'
        write(unit, '(A)') 'openmp.f90 # scope=OpenMP; reason=excluded'
        write(unit, '(A)') 'openacc.f90 # scope=OpenACC; reason=excluded'
        write(unit, '(A)') 'vendor.f90 # scope=vendor; reason=excluded'
        write(unit, '(A)') 'legacy.f90 # scope=legacy; reason=excluded'
        write(unit, '(A)') &
            'flags.f90 # scope=compiler-flags; reason=excluded'
        close(unit)

        open(newunit=unit, file=SKIP_MANIFEST, status='replace', &
            action='write')
        write(unit, '(A)') 'gpu.f90 # scope=GPU; reason=excluded'
        write(unit, '(A)') 'harness.f90 # scope=harness; reason=excluded'
        close(unit)
    end subroutine write_valid_manifests

    logical function valid_manifest_runs() result(ok)
        character(len=:), allocatable :: command
        integer :: exit_stat

        command = environment_prefix()//' bash '//SCRIPT// &
            ' --suite gfortran-dg --file owned_failure.f90'// &
            ' --report '//REPORT//' > '//FAILURE_LOG//' 2>&1'
        call execute_command_line(command, exitstat=exit_stat)
        ok = exit_stat == 0 .and. file_contains(REPORT, &
            '"file":"owned_failure.f90","status":"XFAIL"')
        if (.not. ok) print *, 'FAIL: valid structured manifests were rejected'
    end function valid_manifest_runs

    subroutine write_invalid_manifest(case_number)
        integer, intent(in) :: case_number
        integer :: unit

        open(newunit=unit, file=XFAIL_MANIFEST, status='replace', &
            action='write')
        select case (case_number)
        case (1)
            write(unit, '(A)') &
                'owned_failure.f90 # owner=lazy-fortran/ffc#378'
        case (2)
            write(unit, '(A)') 'owned_failure.f90 # reason=missing owner'
        case (3)
            write(unit, '(A)') 'owned_failure.f90 # '// &
                'owner=lazy-fortran/ffc#378; scope=coarray; reason=both'
        case (4)
            write(unit, '(A)') &
                'owned_failure.f90 # owner=issue-378; reason=malformed'
        case (5)
            write(unit, '(A)') 'owned_failure.f90 # '// &
                'owner=lazy-fortran/ffc#378; reason=first'
            write(unit, '(A)') 'owned_failure.f90 # '// &
                'owner=lazy-fortran/ffc#378; reason=second'
        case (6)
            write(unit, '(A)') &
                'owned_failure.f90 # scope=threads; reason=unknown'
        case (7)
            write(unit, '(A)') 'owned_failure.f90# '// &
                'owner=lazy-fortran/ffc#378; reason=no separator'
        end select
        close(unit)
    end subroutine write_invalid_manifest

    logical function invalid_manifest_fails(case_number, diagnostic) result(ok)
        integer, intent(in) :: case_number
        character(len=*), intent(in) :: diagnostic
        character(len=:), allocatable :: command
        character(len=32) :: line_text
        integer :: exit_stat

        write(line_text, '(I0)') merge(2, 1, case_number == 5)
        command = environment_prefix()//' bash '//SCRIPT// &
            ' --suite gfortran-dg --file owned_failure.f90'// &
            ' --report '//REPORT//' > '//FAILURE_LOG//' 2>&1'
        call execute_command_line(command, exitstat=exit_stat)
        ok = exit_stat /= 0 .and. &
            file_contains(FAILURE_LOG, XFAIL_MANIFEST//':'//trim(line_text)) &
            .and. file_contains(FAILURE_LOG, diagnostic)
        if (.not. ok) print *, 'FAIL: missing rejection: ', trim(diagnostic)
    end function invalid_manifest_fails

    function environment_prefix() result(prefix)
        character(len=:), allocatable :: prefix

        prefix = 'FFC_GFORTRAN_DG_DIR='//FIXTURE_DIR// &
            ' FFC_XFAIL_MANIFEST='//XFAIL_MANIFEST// &
            ' FFC_SKIP_MANIFEST='//SKIP_MANIFEST
    end function environment_prefix

    subroutine write_fake_gh()
        integer :: unit

        call execute_command_line('rm -rf '//FAKE_BIN_DIR)
        call execute_command_line('mkdir -p '//FAKE_BIN_DIR)
        open(newunit=unit, file=FAKE_BIN_DIR//'/gh', status='replace', &
            action='write')
        write(unit, '(A)') '#!/usr/bin/env bash'
        write(unit, '(A)') 'case "$3" in'
        write(unit, '(A)') '378) echo OPEN ;;'
        write(unit, '(A)') '445) echo CLOSED ;;'
        write(unit, '(A)') '*) exit 1 ;;'
        write(unit, '(A)') 'esac'
        close(unit)
        call execute_command_line('chmod +x '//FAKE_BIN_DIR//'/gh')
    end subroutine write_fake_gh

    logical function audit_owner(issue_number, should_pass, diagnostic) &
            result(ok)
        character(len=*), intent(in) :: issue_number
        logical, intent(in) :: should_pass
        character(len=*), intent(in) :: diagnostic
        character(len=:), allocatable :: command
        integer :: exit_stat
        integer :: unit

        open(newunit=unit, file=AUDIT_MANIFEST, status='replace', &
            action='write')
        write(unit, '(A)') 'owned_failure.f90 # owner=lazy-fortran/ffc#'// &
            issue_number//'; reason=audit fixture'
        close(unit)
        command = 'PATH='//FAKE_BIN_DIR//'/:$PATH '// &
            'scripts/audit_manifest_owners.sh '//AUDIT_MANIFEST// &
            ' > '//FAILURE_LOG//' 2>&1'
        call execute_command_line(command, exitstat=exit_stat)
        if (should_pass) then
            ok = exit_stat == 0
        else
            ok = exit_stat /= 0 .and. file_contains(FAILURE_LOG, diagnostic)
        end if
        if (.not. ok) print *, 'FAIL: owner audit case ', issue_number
    end function audit_owner

    logical function file_contains(path, needle) result(found)
        character(len=*), intent(in) :: path
        character(len=*), intent(in) :: needle
        character(len=4096) :: line
        integer :: io_stat
        integer :: unit

        found = .false.
        open(newunit=unit, file=path, status='old', action='read', iostat=io_stat)
        if (io_stat /= 0) return
        do
            read(unit, '(A)', iostat=io_stat) line
            if (io_stat /= 0) exit
            if (index(line, needle) > 0) found = .true.
        end do
        close(unit)
    end function file_contains

end program test_conformance_manifest_validation
