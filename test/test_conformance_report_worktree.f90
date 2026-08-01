program test_conformance_report_worktree
    ! ffc #642: identical commits built in different worktrees disagree on
    ! corpus results, so a report from worktree A must never be compared
    ! against a report from worktree B. The runner therefore records the
    ! worktree that produced a report, and the comparison tool refuses any
    ! cross-worktree pair instead of reporting a meaningless delta.
    use conformance_temp_dir, only: make_temp_root, remove_temp_root
    implicit none

    character(len=*), parameter :: GAUNTLET = 'scripts/conformance_gauntlet.sh'
    character(len=*), parameter :: COMPARE = 'scripts/compare_conformance_reports.sh'
    character(len=:), allocatable :: root, report, base, cand
    logical :: all_passed

    print *, '=== conformance report worktree identity ==='

    root = make_temp_root('report_worktree')
    report = root//'/gauntlet.jsonl'
    base = root//'/base.jsonl'
    cand = root//'/cand.jsonl'
    all_passed = .true.

    ! The runner records the absolute path of the checkout it ran from.
    call check('timeout 300 bash '//GAUNTLET// &
        ' --suite fortfront-f90 --file ast_coverage_control_flow.f90'// &
        ' --report '//report//' >/dev/null 2>&1', 0, &
        'gauntlet single-file run', all_passed)
    call check('grep -q ''"status":"SUMMARY"'' '//report// &
        ' && grep -q "\"worktree\":\"$(readlink -f "$PWD")\"" '//report, 0, &
        'summary records the producing worktree', all_passed)

    ! Same worktree, identical results: comparison succeeds.
    call write_report(base, '/home/x/ffc', 'PASS')
    call write_report(cand, '/home/x/ffc', 'PASS')
    call check('bash '//COMPARE//' '//base//' '//cand//' >/dev/null 2>&1', 0, &
        'same worktree, no delta', all_passed)

    ! Same worktree, changed result: comparison reports the regression.
    call write_report(cand, '/home/x/ffc', 'FAIL')
    call check('bash '//COMPARE//' '//base//' '//cand// &
        ' 2>&1 | grep -q "a.f90: PASS -> FAIL"', 0, &
        'same worktree delta is reported', all_passed)
    call check('bash '//COMPARE//' '//base//' '//cand//' >/dev/null 2>&1', 1, &
        'same worktree delta exits nonzero', all_passed)

    ! Different worktrees: comparison must refuse rather than report a delta.
    call write_report(cand, '/home/y/ffc-wt', 'PASS')
    call check('bash '//COMPARE//' '//base//' '//cand//' >/dev/null 2>&1', 2, &
        'cross-worktree comparison is refused', all_passed)
    call check('bash '//COMPARE//' '//base//' '//cand// &
        ' 2>&1 | grep -q "different worktrees"', 0, &
        'refusal names the cause', all_passed)

    ! A report without worktree provenance cannot be compared either.
    call write_legacy_report(cand, 'PASS')
    call check('bash '//COMPARE//' '//base//' '//cand//' >/dev/null 2>&1', 2, &
        'report without worktree provenance is refused', all_passed)

    if (all_passed) then
        call remove_temp_root(root)
        print *, 'ALL TESTS PASSED'
    else
        print *, 'FAILURES (scratch kept at '//root//')'
        error stop 1
    end if

contains

    subroutine check(command, expected_status, label, ok)
        character(len=*), intent(in) :: command
        integer, intent(in) :: expected_status
        character(len=*), intent(in) :: label
        logical, intent(inout) :: ok
        integer :: exit_stat

        exit_stat = -1
        call execute_command_line(command, exitstat=exit_stat)
        if (exit_stat == expected_status) then
            print *, 'PASS: ', label
        else
            print *, 'FAIL: ', label, ' expected status ', expected_status, &
                ' got ', exit_stat
            ok = .false.
        end if
    end subroutine check

    subroutine write_report(path, worktree, status)
        character(len=*), intent(in) :: path
        character(len=*), intent(in) :: worktree
        character(len=*), intent(in) :: status
        integer :: unit

        open (newunit=unit, file=path, status='replace', action='write')
        write (unit, '(A)') '{"suite":"fortfront-f90","file":"a.f90",'// &
            '"status":"'//status//'"}'
        write (unit, '(A)') '{"suite":"fortfront-f90","status":"SUMMARY",'// &
            '"total":1,"worktree":"'//worktree//'"}'
        close (unit)
    end subroutine write_report

    subroutine write_legacy_report(path, status)
        character(len=*), intent(in) :: path
        character(len=*), intent(in) :: status
        integer :: unit

        open (newunit=unit, file=path, status='replace', action='write')
        write (unit, '(A)') '{"suite":"fortfront-f90","file":"a.f90",'// &
            '"status":"'//status//'"}'
        write (unit, '(A)') '{"suite":"fortfront-f90","status":"SUMMARY",'// &
            '"total":1}'
        close (unit)
    end subroutine write_legacy_report

end program test_conformance_report_worktree
