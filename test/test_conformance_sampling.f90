program test_conformance_sampling
    ! Routine measurement must be cheap enough that nobody skips it (#567).
    ! Two mechanisms make it cheap: a stratified deterministic sample, and a
    ! reference-output cache. Both are only usable if they cannot corrupt the
    ! exact record, so the oracles here are observable behaviour of the
    ! scripts: which files a seed selects, what the summary states about the
    ! margin, that the dashboard validator refuses a sampled report, and that
    ! a cached run produces the same verdicts without re-running gfortran.
    use conformance_temp_dir, only: make_temp_root, remove_temp_root
    implicit none

    character(len=*), parameter :: GAUNTLET = 'scripts/conformance_gauntlet.sh'
    character(len=*), parameter :: CHECK = 'scripts/conformance_check.sh'
    character(len=:), allocatable :: root
    logical :: all_passed

    print *, '=== conformance sampling and cache test ==='

    root = make_temp_root('conformance_sampling')
    all_passed = .true.

    call check_sampled_report(root, all_passed)
    call check_sample_determinism(root, all_passed)
    call check_sampled_report_is_not_a_dashboard_input(root, all_passed)
    call check_stratified_plan(root, all_passed)
    call check_reference_cache(root, all_passed)

    if (.not. all_passed) stop 1
    call remove_temp_root(root)
    print *, 'PASS: sampling and reference caching behave as specified'

contains

    ! A sampled run measures exactly the requested number of files and says so,
    ! with the confidence margin next to the rate.
    subroutine check_sampled_report(work_dir, ok)
        character(len=*), intent(in) :: work_dir
        logical, intent(inout) :: ok
        character(len=:), allocatable :: report

        report = work_dir//'/sampled.jsonl'
        if (.not. run_shell('timeout 300 bash '//GAUNTLET// &
            ' --suite fortfront-f90 --sample 6 --seed 3 --report '// &
            report//' > '//work_dir//'/sampled.log 2>&1')) then
            print *, 'FAIL: sampled gauntlet run did not succeed'
            ok = .false.
            return
        end if
        if (count_records(report) /= 6) then
            print *, 'FAIL: sampled run did not measure 6 files'
            ok = .false.
        end if
        if (.not. file_contains(report, '"sampled":true')) then
            print *, 'FAIL: summary does not mark the run as sampled'
            ok = .false.
        end if
        if (.not. file_contains(report, '"full_run":false')) then
            print *, 'FAIL: sampled summary claims a full run'
            ok = .false.
        end if
        if (.not. file_contains(report, '"sample_population":')) then
            print *, 'FAIL: summary omits the sampled population'
            ok = .false.
        end if
        if (.not. file_contains(report, '"sample_margin_pct":"')) then
            print *, 'FAIL: summary omits the confidence margin'
            ok = .false.
        end if
        if (.not. file_contains(work_dir//'/sampled.log', &
            '(95% CI)')) then
            print *, 'FAIL: run does not print the margin beside the rate'
            ok = .false.
        end if
    end subroutine check_sampled_report

    ! Same seed, same files; a different seed draws a different subset.
    subroutine check_sample_determinism(work_dir, ok)
        character(len=*), intent(in) :: work_dir
        logical, intent(inout) :: ok

        if (.not. run_shell('timeout 300 bash '//GAUNTLET// &
            ' --suite fortfront-f90 --sample 6 --seed 3 --report '// &
            work_dir//'/repeat.jsonl > /dev/null 2>&1')) then
            print *, 'FAIL: repeated sampled run did not succeed'
            ok = .false.
            return
        end if
        if (.not. run_shell('timeout 300 bash '//GAUNTLET// &
            ' --suite fortfront-f90 --sample 6 --seed 9 --report '// &
            work_dir//'/other_seed.jsonl > /dev/null 2>&1')) then
            print *, 'FAIL: second-seed sampled run did not succeed'
            ok = .false.
            return
        end if
        call extract_files(work_dir//'/sampled.jsonl', work_dir//'/files_a.txt')
        call extract_files(work_dir//'/repeat.jsonl', work_dir//'/files_b.txt')
        call extract_files(work_dir//'/other_seed.jsonl', &
            work_dir//'/files_c.txt')
        if (.not. run_shell('cmp -s '//work_dir//'/files_a.txt '// &
            work_dir//'/files_b.txt')) then
            print *, 'FAIL: same seed selected different files'
            ok = .false.
        end if
        if (run_shell('cmp -s '//work_dir//'/files_a.txt '// &
            work_dir//'/files_c.txt')) then
            print *, 'FAIL: a different seed selected the same files'
            ok = .false.
        end if
    end subroutine check_sample_determinism

    ! The checked-in snapshot stays a full run: the report validator that feeds
    ! the parity dashboard must refuse a sampled report outright.
    subroutine check_sampled_report_is_not_a_dashboard_input(work_dir, ok)
        character(len=*), intent(in) :: work_dir
        logical, intent(inout) :: ok

        if (run_shell('awk -v expected_suite=fortfront-f90'// &
            ' -v source='//work_dir//'/sampled.jsonl'// &
            ' -v rows='//work_dir//'/rows.tsv'// &
            ' -v summaries='//work_dir//'/summaries.tsv'// &
            ' -f scripts/validate_parity_report.awk '// &
            work_dir//'/sampled.jsonl > '//work_dir// &
            '/validate.log 2>&1')) then
            print *, 'FAIL: dashboard validator accepted a sampled report'
            ok = .false.
            return
        end if
        if (.not. file_contains(work_dir//'/validate.log', 'sampled report')) &
            then
            print *, 'FAIL: validator rejection does not name sampling'
            ok = .false.
        end if
    end subroutine check_sampled_report_is_not_a_dashboard_input

    ! The requested total is split across the available suites in proportion to
    ! their size, and every suite keeps at least one file of its own.
    subroutine check_stratified_plan(work_dir, ok)
        character(len=*), intent(in) :: work_dir
        logical, intent(inout) :: ok

        if (.not. run_shell('timeout 300 bash '//CHECK// &
            ' --no-build --sample 40 --seed 5 --print-sample-plan > '// &
            work_dir//'/plan.log 2>&1')) then
            print *, 'FAIL: sample plan run did not succeed'
            ok = .false.
            return
        end if
        if (.not. run_shell("awk '/ of .* files/ { "// &
            'total += $2; if ($2 < 1) bad = 1; if ($4 < $2) bad = 1; n++ }'// &
            ' END { exit (n < 1 || total != 40 || bad) }'//"' "// &
            work_dir//'/plan.log')) then
            print *, 'FAIL: stratified plan does not allocate the sample'
            ok = .false.
        end if
    end subroutine check_stratified_plan

    ! A cached reference run reproduces the uncached verdicts exactly while
    ! invoking gfortran far fewer times. The shim counts every gfortran call.
    subroutine check_reference_cache(work_dir, ok)
        character(len=*), intent(in) :: work_dir
        logical, intent(inout) :: ok
        character(len=:), allocatable :: shim_dir, log_path, cache_dir
        integer :: first_calls, second_calls

        shim_dir = work_dir//'/shim'
        log_path = work_dir//'/gfortran_calls.log'
        cache_dir = work_dir//'/refcache'
        if (.not. run_shell('mkdir -p '//shim_dir//' && '// &
            'real_gfortran="$(command -v gfortran)" && '// &
            'printf ''#!/bin/sh\necho call >> %s\nexec %s "$@"\n'' '// &
            log_path//' "$real_gfortran" > '//shim_dir//'/gfortran && '// &
            'chmod +x '//shim_dir//'/gfortran')) then
            print *, 'FAIL: could not install the gfortran shim'
            ok = .false.
            return
        end if

        if (.not. run_shell(': > '//log_path//' && PATH='//shim_dir// &
            ':$PATH timeout 600 bash '//GAUNTLET// &
            ' --suite fortfront-f90 --max-files 12 --ref-cache '//cache_dir// &
            ' --report '//work_dir//'/cache_first.jsonl > /dev/null 2>&1')) then
            print *, 'FAIL: first reference-cache run did not succeed'
            ok = .false.
            return
        end if
        first_calls = count_lines(log_path)

        if (.not. run_shell(': > '//log_path//' && PATH='//shim_dir// &
            ':$PATH timeout 600 bash '//GAUNTLET// &
            ' --suite fortfront-f90 --max-files 12 --ref-cache '//cache_dir// &
            ' --report '//work_dir//'/cache_second.jsonl > /dev/null 2>&1')) &
            then
            print *, 'FAIL: cached reference run did not succeed'
            ok = .false.
            return
        end if
        second_calls = count_lines(log_path)

        call extract_verdicts(work_dir//'/cache_first.jsonl', &
            work_dir//'/verdicts_first.txt')
        call extract_verdicts(work_dir//'/cache_second.jsonl', &
            work_dir//'/verdicts_second.txt')
        if (.not. run_shell('cmp -s '//work_dir//'/verdicts_first.txt '// &
            work_dir//'/verdicts_second.txt')) then
            print *, 'FAIL: cached run changed a verdict'
            ok = .false.
        end if
        if (second_calls >= first_calls) then
            print *, 'FAIL: cached run re-ran gfortran as often as the first'
            ok = .false.
        end if
    end subroutine check_reference_cache

    logical function run_shell(command) result(ok)
        character(len=*), intent(in) :: command
        integer :: exit_stat

        call execute_command_line(command, exitstat=exit_stat)
        ok = exit_stat == 0
    end function run_shell

    subroutine extract_files(report, destination)
        character(len=*), intent(in) :: report
        character(len=*), intent(in) :: destination
        logical :: ok

        ok = run_shell('grep -o ''"file":"[^"]*"'' '//report//' > '// &
            destination//' 2>/dev/null || true')
    end subroutine extract_files

    subroutine extract_verdicts(report, destination)
        character(len=*), intent(in) :: report
        character(len=*), intent(in) :: destination
        logical :: ok

        ok = run_shell('grep -o ''"file":"[^"]*","status":"[^"]*"'' '// &
            report//' > '//destination//' 2>/dev/null || true')
    end subroutine extract_verdicts

    integer function count_lines(path) result(total)
        character(len=*), intent(in) :: path
        integer :: unit, io_stat
        character(len=4096) :: line

        total = 0
        open (newunit=unit, file=path, status='old', action='read', &
              iostat=io_stat)
        if (io_stat /= 0) return
        do
            read (unit, '(A)', iostat=io_stat) line
            if (io_stat /= 0) exit
            total = total + 1
        end do
        close (unit)
    end function count_lines

    integer function count_records(report) result(total)
        character(len=*), intent(in) :: report
        integer :: unit, io_stat
        character(len=4096) :: line

        total = 0
        open (newunit=unit, file=report, status='old', action='read', &
              iostat=io_stat)
        if (io_stat /= 0) return
        do
            read (unit, '(A)', iostat=io_stat) line
            if (io_stat /= 0) exit
            if (index(line, '"status":"SUMMARY"') > 0) cycle
            if (index(line, '"file":') > 0) total = total + 1
        end do
        close (unit)
    end function count_records

    logical function file_contains(path, needle) result(found)
        character(len=*), intent(in) :: path
        character(len=*), intent(in) :: needle
        integer :: unit, io_stat
        character(len=4096) :: line

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

end program test_conformance_sampling
