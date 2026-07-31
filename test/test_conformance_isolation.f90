program test_conformance_isolation
    ! Regression test for lazy-fortran/ffc#547.
    !
    ! Concurrent conformance runs (parallel `fo test`, several worktrees) must
    ! not be able to observe each other's artifacts, and must never measure
    ! another checkout's compiler.
    use conformance_temp_dir, only: make_temp_root, remove_temp_root
    implicit none

    character(len=:), allocatable :: root
    logical :: passed

    print *, '=== conformance isolation test ==='

    root = make_temp_root('conformance_isolation')
    passed = .true.

    if (.not. temp_roots_are_disjoint()) passed = .false.
    if (.not. ffc_resolves_inside_own_build_tree()) passed = .false.
    if (.not. ffc_resolution_refuses_foreign_binaries()) passed = .false.
    if (.not. concurrent_runs_do_not_share_reports()) passed = .false.

    ! Keep the scratch directory when something failed: it holds the logs.
    if (.not. passed) stop 1
    call remove_temp_root(root)
    print *, 'PASS: concurrent conformance runs are isolated'

contains

    logical function temp_roots_are_disjoint() result(ok)
        character(len=:), allocatable :: left, right

        left = make_temp_root('isolation_probe')
        right = make_temp_root('isolation_probe')
        ok = left /= right
        if (.not. ok) then
            print *, 'FAIL: two scratch roots collided: '//left
        else
            call execute_command_line('printf left > '//left//'/marker')
            call execute_command_line('printf right > '//right//'/marker')
            ok = file_content_is(left//'/marker', 'left') .and. &
                file_content_is(right//'/marker', 'right')
            if (.not. ok) print *, 'FAIL: scratch roots observed each other'
        end if
        call remove_temp_root(left)
        call remove_temp_root(right)
    end function temp_roots_are_disjoint

    logical function ffc_resolves_inside_own_build_tree() result(ok)
        character(len=:), allocatable :: log_path
        character(len=4096) :: resolved
        integer :: exit_stat

        log_path = root//'/find_ffc_own.log'
        ! A relative PROJECT_DIR must still resolve to an absolute path
        ! inside this checkout's own build tree.
        call execute_command_line('bash -c ''unset FFC_BIN; PROJECT_DIR=.; '// &
            'source scripts/lib_conformance.sh; find_ffc'' > '// &
            log_path//' 2>&1', exitstat=exit_stat)
        ok = exit_stat == 0
        if (.not. ok) then
            print *, 'FAIL: find_ffc did not resolve a binary'
            call execute_command_line('cat '//log_path)
            return
        end if
        call read_first_line(log_path, resolved)
        ok = starts_with(trim(resolved), current_directory()//'/build/')
        if (.not. ok) then
            print *, 'FAIL: find_ffc left this checkout: '//trim(resolved)
            return
        end if
        ok = index(trim(resolved), '..') == 0
        if (.not. ok) print *, 'FAIL: find_ffc returned a traversing path'
    end function ffc_resolves_inside_own_build_tree

    logical function ffc_resolution_refuses_foreign_binaries() result(ok)
        character(len=:), allocatable :: sibling, decoy_dir, log_path
        integer :: exit_stat

        ! A sibling worktree with its own freshly built compiler, plus an ffc
        ! on PATH. Neither may be adopted by a checkout that has no build.
        sibling = root//'/sibling-worktree'
        decoy_dir = root//'/decoy-bin'
        log_path = root//'/find_ffc_foreign.log'
        call execute_command_line('mkdir -p '//sibling//'/build/fo/app '// &
            decoy_dir//' '//root//'/empty-checkout')
        call write_stub(sibling//'/build/fo/app/ffc')
        call write_stub(decoy_dir//'/ffc')

        call execute_command_line('PATH='//decoy_dir//':"$PATH" bash -c '// &
            '''unset FFC_BIN; PROJECT_DIR='//root//'/empty-checkout; '// &
            'source scripts/lib_conformance.sh; find_ffc'' > '// &
            log_path//' 2>&1', exitstat=exit_stat)
        ok = exit_stat /= 0
        if (.not. ok) then
            print *, 'FAIL: find_ffc accepted a foreign ffc binary'
            call execute_command_line('cat '//log_path)
            return
        end if
        ok = .not. file_contains(log_path, sibling) .and. &
            .not. file_contains(log_path, decoy_dir//'/ffc')
        if (.not. ok) print *, 'FAIL: find_ffc offered a foreign ffc path'
    end function ffc_resolution_refuses_foreign_binaries

    logical function concurrent_runs_do_not_share_reports() result(ok)
        character(len=:), allocatable :: left, right

        ! Two gauntlet runs started at the same time, each with its own scratch
        ! directory and no explicit --report. Their default report paths must
        ! stay inside their own scratch directory and describe only their own
        ! selection.
        left = root//'/run-left'
        right = root//'/run-right'
        call execute_command_line('mkdir -p '//left//' '//right)
        call execute_command_line( &
            'TMPDIR='//left//' timeout 180 bash scripts/conformance_gauntlet.sh'// &
            ' --suite fortfront-f90 --file ast_coverage_control_flow.f90'// &
            ' > '//left//'/run.log 2>&1 & '// &
            'TMPDIR='//right//' timeout 180 bash scripts/conformance_gauntlet.sh'// &
            ' --suite fortfront-f90 --file ast_coverage_io_statements.f90'// &
            ' > '//right//'/run.log 2>&1 & '//'wait')

        ok = .true.
        call require_own_report(left, 'ast_coverage_control_flow.f90', &
            'ast_coverage_io_statements.f90', ok)
        call require_own_report(right, 'ast_coverage_io_statements.f90', &
            'ast_coverage_control_flow.f90', ok)
        if (.not. ok) print *, 'FAIL: concurrent runs shared report artifacts'
    end function concurrent_runs_do_not_share_reports

    subroutine require_own_report(run_root, own_file, foreign_file, ok)
        character(len=*), intent(in) :: run_root, own_file, foreign_file
        logical, intent(inout) :: ok
        character(len=:), allocatable :: report
        logical :: exists

        report = run_root//'/ffc_gauntlet_fortfront-f90.jsonl'
        inquire (file=report, exist=exists)
        if (.not. exists) then
            print *, 'FAIL: no report inside own scratch directory: '//report
            ok = .false.
            return
        end if
        if (.not. file_contains(report, '"file":"'//own_file//'"')) then
            print *, 'FAIL: report is missing its own case: '//own_file
            ok = .false.
        end if
        if (file_contains(report, '"file":"'//foreign_file//'"')) then
            print *, 'FAIL: report contains the other run''s case: '//foreign_file
            ok = .false.
        end if
    end subroutine require_own_report

    subroutine write_stub(path)
        character(len=*), intent(in) :: path
        integer :: unit

        open (newunit=unit, file=path, status='replace', action='write')
        write (unit, '(A)') '#!/bin/sh'
        write (unit, '(A)') 'exit 0'
        close (unit)
        call execute_command_line('chmod +x '//path)
        call execute_command_line('touch '//path)
    end subroutine write_stub

    function current_directory() result(directory)
        character(len=:), allocatable :: directory
        character(len=4096) :: buffer

        call getcwd_text(buffer)
        directory = trim(buffer)
    end function current_directory

    subroutine getcwd_text(buffer)
        character(len=*), intent(out) :: buffer
        character(len=:), allocatable :: path

        path = root//'/cwd.txt'
        call execute_command_line('pwd -P > '//path)
        call read_first_line(path, buffer)
    end subroutine getcwd_text

    subroutine read_first_line(path, line)
        character(len=*), intent(in) :: path
        character(len=*), intent(out) :: line
        integer :: unit, io_stat

        line = ''
        open (newunit=unit, file=path, status='old', action='read', &
            iostat=io_stat)
        if (io_stat /= 0) return
        read (unit, '(A)', iostat=io_stat) line
        close (unit)
    end subroutine read_first_line

    logical function starts_with(text, prefix) result(match)
        character(len=*), intent(in) :: text, prefix

        match = .false.
        if (len(text) < len(prefix)) return
        match = text(1:len(prefix)) == prefix
    end function starts_with

    logical function file_content_is(path, expected) result(match)
        character(len=*), intent(in) :: path, expected
        character(len=4096) :: line

        call read_first_line(path, line)
        match = trim(line) == expected
    end function file_content_is

    logical function file_contains(path, needle) result(found)
        character(len=*), intent(in) :: path, needle
        character(len=4096) :: line
        integer :: unit, io_stat

        found = .false.
        open (newunit=unit, file=path, status='old', action='read', &
            iostat=io_stat)
        if (io_stat /= 0) return
        do
            read (unit, '(A)', iostat=io_stat) line
            if (io_stat /= 0) exit
            if (index(line, needle) > 0) found = .true.
        end do
        close (unit)
    end function file_contains

end program test_conformance_isolation
