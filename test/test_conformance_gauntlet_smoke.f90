program test_conformance_gauntlet_smoke
    implicit none

    integer, parameter :: RUN_TIMEOUT_SECONDS = 120
    character(len=*), parameter :: SCRIPT = &
        'scripts/conformance_gauntlet.sh'
    character(len=*), parameter :: ROOT_REPORT = &
        '/tmp/ffc_gauntlet_smoke_root.jsonl'
    character(len=*), parameter :: TMP_CWD_REPORT = &
        '/tmp/ffc_gauntlet_smoke_tmpcwd.jsonl'
    character(len=*), parameter :: NAMED_REPORT = &
        '/tmp/ffc_gauntlet_smoke_named.jsonl'
    character(len=*), parameter :: LIST_REPORT = &
        '/tmp/ffc_gauntlet_smoke_list.jsonl'
    character(len=*), parameter :: LIMITED_REPORT = &
        '/tmp/ffc_gauntlet_smoke_limited.jsonl'
    character(len=*), parameter :: FORWARDED_REPORT = &
        '/tmp/ffc_conformance_fortfront-f90.jsonl'
    character(len=*), parameter :: SELECTION_LIST = &
        '/tmp/ffc_gauntlet_smoke_files.txt'
    character(len=*), parameter :: FORWARD_SELECTION_LIST = &
        '/tmp/ffc_gauntlet_smoke_forward_files.txt'
    character(len=*), parameter :: FAILURE_LOG = &
        '/tmp/ffc_gauntlet_smoke_failure.log'

    logical :: all_passed

    print *, '=== conformance gauntlet smoke test ==='

    all_passed = .true.
    if (.not. run_smoke('timeout 120 bash '//SCRIPT// &
        ' --suite fortfront-f90 --max-files 20 --report '// &
        ROOT_REPORT, ROOT_REPORT, 20)) all_passed = .false.
    if (.not. run_smoke('repo="$PWD"; cd /tmp && timeout 120 bash '// &
        '"$repo/'//SCRIPT//'" --suite fortfront-f90 --max-files 20 '// &
        '--report '//TMP_CWD_REPORT, &
        TMP_CWD_REPORT, 20)) all_passed = .false.

    if (.not. run_smoke('timeout 120 bash '//SCRIPT// &
        ' --suite fortfront-f90 --file ast_coverage_control_flow.f90'// &
        ' --report '//NAMED_REPORT, NAMED_REPORT, 1)) all_passed = .false.
    if (.not. file_contains(NAMED_REPORT, &
        '"file":"ast_coverage_control_flow.f90","status":"XFAIL"')) &
        all_passed = .false.

    call write_selection_list()
    if (.not. run_smoke('timeout 120 bash '//SCRIPT// &
        ' --suite fortfront-f90 --files-from '//SELECTION_LIST// &
        ' --report '//LIST_REPORT, LIST_REPORT, 2)) all_passed = .false.
    if (.not. report_has_file_order(LIST_REPORT, &
        'ast_coverage_control_flow.f90', &
        'ast_coverage_io_statements.f90')) all_passed = .false.
    if (.not. run_smoke('timeout 120 bash '//SCRIPT// &
        ' --suite fortfront-f90 --files-from '//SELECTION_LIST// &
        ' --max-files 1 --report '//LIMITED_REPORT, LIMITED_REPORT, 1)) &
        all_passed = .false.
    if (.not. file_contains(LIMITED_REPORT, &
        '"file":"ast_coverage_control_flow.f90"')) all_passed = .false.

    if (.not. run_smoke('repo="$PWD"; cd /tmp && timeout 120 bash'// &
        ' "$repo/scripts/conformance_check.sh"'// &
        ' --no-build --suite fortfront-f90'// &
        ' --file ast_coverage_control_flow.f90'// &
        ' --files-from ffc_gauntlet_smoke_forward_files.txt', &
        FORWARDED_REPORT, 2)) &
        all_passed = .false.
    if (.not. report_has_file_order(FORWARDED_REPORT, &
        'ast_coverage_control_flow.f90', &
        'ast_coverage_io_statements.f90')) all_passed = .false.

    if (.not. run_failure('bash '//SCRIPT// &
        ' --suite fortfront-f90 --file missing_named_file.f90', &
        'unknown selected file')) all_passed = .false.
    if (.not. run_failure('bash '//SCRIPT// &
        ' --suite fortfront-f90 --file ast_coverage_control_flow.f90'// &
        ' --file ast_coverage_control_flow.f90', &
        'duplicate selected file')) all_passed = .false.
    if (.not. run_failure('bash '//SCRIPT// &
        ' --suite fortfront-f90 --file /tmp/absolute.f90', &
        'suite-relative')) all_passed = .false.
    if (.not. run_failure('bash '//SCRIPT// &
        ' --suite fortfront-f90 --file ../ast_coverage_control_flow.f90', &
        'parent traversal')) all_passed = .false.
    if (.not. run_failure('bash scripts/conformance_check.sh --no-build'// &
        ' --file ast_coverage_control_flow.f90', &
        'require --suite')) all_passed = .false.
    if (.not. run_failure('bash scripts/conformance_check.sh --no-build'// &
        ' --suite fortfront-f90 --file missing_named_file.f90', &
        'unknown selected file')) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: gauntlet smoke test completed with zero FAIL records'

contains

    logical function run_smoke(cmd, report, expected_total) result(ok)
        character(len=*), intent(in) :: cmd
        character(len=*), intent(in) :: report
        integer, intent(in) :: expected_total
        integer :: exit_stat
        logical :: report_exists

        ok = .false.
        call execute_command_line('rm -f '//report)
        call execute_command_line(cmd, exitstat=exit_stat)

        if (exit_stat > 126) then
            print *, 'FAIL: gauntlet timed out or killed'
            return
        end if

        if (exit_stat /= 0) then
            print *, 'FAIL: gauntlet exited with ', exit_stat
            return
        end if

        inquire(file=report, exist=report_exists)
        if (.not. report_exists) then
            print *, 'FAIL: no report file at ', trim(adjustl(report))
            return
        end if

        ok = report_matches(report, expected_total)
    end function run_smoke

    logical function report_matches(report, expected_total) result(ok)
        character(len=*), intent(in) :: report
        integer, intent(in) :: expected_total
        integer :: unit
        integer :: io_stat
        integer :: record_count
        character(len=4096) :: line
        character(len=32) :: total_text
        logical :: has_summary

        ok = .false.
        has_summary = .false.
        record_count = 0
        write(total_text, '(I0)') expected_total
        open(newunit=unit, file=report, status='old', action='read', &
            iostat=io_stat)
        if (io_stat /= 0) then
            print *, 'FAIL: cannot open report ', trim(adjustl(report))
            return
        end if

        do
            read(unit, '(A)', iostat=io_stat) line
            if (io_stat /= 0) exit
            if (is_blank_or_comment(line)) cycle
            if (index(line, '"status":"SUMMARY"') > 0) then
                has_summary = index(line, '"fail":0') > 0 .and. &
                    index(line, '"total":'//trim(total_text)//'}') > 0
            else
                record_count = record_count + 1
            end if
        end do
        close(unit)

        if (.not. has_summary) then
            print *, 'FAIL: invalid SUMMARY record in ', trim(adjustl(report))
            return
        end if

        if (record_count /= expected_total) then
            print *, 'FAIL: wrong file-record count in ', trim(adjustl(report))
            return
        end if

        ok = .true.
    end function report_matches

    logical function run_failure(cmd, expected_diagnostic) result(ok)
        character(len=*), intent(in) :: cmd
        character(len=*), intent(in) :: expected_diagnostic
        integer :: exit_stat

        call execute_command_line(cmd//' > '//FAILURE_LOG//' 2>&1', &
            exitstat=exit_stat)
        ok = exit_stat /= 0 .and. file_contains(FAILURE_LOG, expected_diagnostic)
        if (.not. ok) print *, 'FAIL: missing rejection: ', trim(expected_diagnostic)
    end function run_failure

    logical function file_contains(path, needle) result(found)
        character(len=*), intent(in) :: path
        character(len=*), intent(in) :: needle
        integer :: unit
        integer :: io_stat
        character(len=4096) :: line

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

    logical function report_has_file_order(report, first_file, second_file) &
            result(ok)
        character(len=*), intent(in) :: report
        character(len=*), intent(in) :: first_file
        character(len=*), intent(in) :: second_file
        integer :: unit
        integer :: io_stat
        integer :: record_count
        character(len=4096) :: line

        ok = .false.
        record_count = 0
        open(newunit=unit, file=report, status='old', action='read', &
            iostat=io_stat)
        if (io_stat /= 0) return
        do
            read(unit, '(A)', iostat=io_stat) line
            if (io_stat /= 0) exit
            if (index(line, '"file":') == 0) cycle
            record_count = record_count + 1
            if (record_count == 1 .and. index(line, first_file) == 0) exit
            if (record_count == 2 .and. index(line, second_file) == 0) exit
            if (record_count == 2) ok = .true.
        end do
        close(unit)
        if (.not. ok) print *, 'FAIL: wrong selected-file order'
    end function report_has_file_order

    subroutine write_selection_list()
        integer :: unit

        open(newunit=unit, file=SELECTION_LIST, status='replace', &
            action='write')
        write(unit, '(A)') '# exact named cases'
        write(unit, '(A)') ''
        write(unit, '(A)') 'ast_coverage_control_flow.f90'
        write(unit, '(A)') 'ast_coverage_io_statements.f90'
        close(unit)

        open(newunit=unit, file=FORWARD_SELECTION_LIST, status='replace', &
            action='write')
        write(unit, '(A)') 'ast_coverage_io_statements.f90'
        close(unit)
    end subroutine write_selection_list

    logical function is_blank_or_comment(line)
        character(len=*), intent(in) :: line
        character(len=:), allocatable :: stripped

        stripped = trim(adjustl(line))
        if (len(stripped) == 0) then
            is_blank_or_comment = .true.
        else if (stripped(1:1) == '#') then
            is_blank_or_comment = .true.
        else
            is_blank_or_comment = .false.
        end if
    end function is_blank_or_comment

end program test_conformance_gauntlet_smoke
