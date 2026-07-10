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
    character(len=*), parameter :: DG_FIXTURE_DIR = &
        '/tmp/ffc_gauntlet_dg_fixtures'
    character(len=*), parameter :: DG_REPORT = &
        '/tmp/ffc_gauntlet_dg_warning.jsonl'
    character(len=*), parameter :: DG_NEGATIVE_REPORT = &
        '/tmp/ffc_gauntlet_dg_negative.jsonl'
    character(len=*), parameter :: UNDEFINED_REPORT = &
        '/tmp/ffc_gauntlet_undefined_output.jsonl'
    character(len=*), parameter :: UNDEFINED_NEGATIVE_REPORT = &
        '/tmp/ffc_gauntlet_undefined_negative.jsonl'
    character(len=*), parameter :: UNDEFINED_FIXTURE_ROOT = &
        '/tmp/ffc_gauntlet_undefined_fortfront'
    character(len=*), parameter :: UNDEFINED_MANIFEST = &
        '/tmp/ffc_gauntlet_undefined_manifest.txt'

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

    call run_dg_directive_smoke(all_passed)
    call run_undefined_output_smoke(all_passed)

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
                    index(line, '"total":'//trim(total_text)//',') > 0 .and. &
                    index(line, '"schema_version":1') > 0 .and. &
                    index(line, '"provenance_verified":false') > 0 .and. &
                    index(line, '"ffc_source_sha256":"') > 0 .and. &
                    index(line, '"ffc_binary_sha256":"') > 0
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

    subroutine run_dg_directive_smoke(passed)
        logical, intent(inout) :: passed

        call write_dg_fixtures()
        if (.not. run_smoke('FFC_GFORTRAN_DG_DIR='//DG_FIXTURE_DIR// &
            ' timeout 120 bash '//SCRIPT//' --suite gfortran-dg'// &
            ' --file warning_compile.f90 --file warning_run.f90'// &
            ' --file true_error.f90 --file empty_options.f90'// &
            ' --file blank_options.f90 --file omitted_options.f90'// &
            ' --file empty_add_options.f90 --file spaced_run.f90'// &
            ' --report '//DG_REPORT, DG_REPORT, 8)) &
            passed = .false.
        if (.not. file_contains(DG_REPORT, &
            '"file":"warning_compile.f90","status":"PASS","ffc_exit":0,'// &
            '"ref_exit":-1,"note":"ffc -c succeeded",'// &
            '"warning_expectation":"unchecked"')) passed = .false.
        if (.not. file_contains(DG_REPORT, &
            '"file":"warning_run.f90","status":"PASS","ffc_exit":0,'// &
            '"ref_exit":0,"note":"output matches gfortran",'// &
            '"warning_expectation":"unchecked"')) passed = .false.
        if (.not. file_contains(DG_REPORT, &
            '"file":"true_error.f90","status":"PASS"')) passed = .false.
        if (.not. file_contains(DG_REPORT, &
            '"warning_unchecked":2,"total":8,"schema_version":1')) &
            passed = .false.
        if (count_lines_with(DG_REPORT, 'options.f90","status":"PASS"', &
            'ffc -c succeeded') /= 4) passed = .false.
        if (.not. file_contains(DG_REPORT, &
            '"file":"spaced_run.f90","status":"PASS","ffc_exit":0,'// &
            '"ref_exit":0,"note":"output matches gfortran"')) &
            passed = .false.
        if (.not. run_failure('FFC_GFORTRAN_DG_DIR='//DG_FIXTURE_DIR// &
            ' bash '//SCRIPT//' --suite gfortran-dg'// &
            ' --file nonempty_options.f90', &
            'unlisted skip: flags')) passed = .false.
        if (.not. run_failure('FFC_GFORTRAN_DG_DIR='//DG_FIXTURE_DIR// &
            ' bash '//SCRIPT//' --suite gfortran-dg'// &
            ' --file no_main_accepted.f90 --report '//DG_NEGATIVE_REPORT, &
            'negative test accepted')) passed = .false.
        if (.not. file_contains(DG_NEGATIVE_REPORT, &
            '"file":"no_main_accepted.f90","status":"FAIL",'// &
            '"ffc_exit":0')) &
            passed = .false.
    end subroutine run_dg_directive_smoke

    subroutine write_dg_fixtures()
        integer :: unit

        call execute_command_line('rm -rf '//DG_FIXTURE_DIR)
        call execute_command_line('mkdir -p '//DG_FIXTURE_DIR)
        open(newunit=unit, file=DG_FIXTURE_DIR//'/warning_compile.f90', &
            status='replace', action='write')
        write(unit, '(A)') '! { dg-do compile }'
        write(unit, '(A)') 'program warning_compile'
        write(unit, '(A)') 'integer :: value ! { dg-warning "unchecked" }'
        write(unit, '(A)') 'value = 1'
        write(unit, '(A)') 'end program warning_compile'
        close(unit)

        open(newunit=unit, file=DG_FIXTURE_DIR//'/warning_run.f90', &
            status='replace', action='write')
        write(unit, '(A)') '! { dg-do run }'
        write(unit, '(A)') 'program warning_run'
        write(unit, '(A)') 'integer :: value ! { dg-warning "unchecked" }'
        write(unit, '(A)') 'value = 1'
        write(unit, '(A)') 'if (value /= 1) stop 1'
        write(unit, '(A)') 'end program warning_run'
        close(unit)

        open(newunit=unit, file=DG_FIXTURE_DIR//'/true_error.f90', &
            status='replace', action='write')
        write(unit, '(A)') '! { dg-do compile }'
        write(unit, '(A)') 'program true_error'
        write(unit, '(A)') 'integer :: value'
        write(unit, '(A)') 'value = @'
        write(unit, '(A)') '! { dg-error "invalid expression" }'
        write(unit, '(A)') '! { dg-warning "secondary expectation" }'
        write(unit, '(A)') 'end program true_error'
        close(unit)

        open(newunit=unit, file=DG_FIXTURE_DIR//'/no_main_accepted.f90', &
            status='replace', action='write')
        write(unit, '(A)') 'subroutine no_main_accepted'
        write(unit, '(A)') '! { dg-error "synthetic accepted case" }'
        write(unit, '(A)') 'end subroutine no_main_accepted'
        close(unit)

        call write_options_fixture('empty_options.f90', &
            '! { dg-options "" }')
        call write_options_fixture('blank_options.f90', &
            '! { dg-options " " }')
        call write_options_fixture('omitted_options.f90', &
            '! { dg-options }')
        call write_options_fixture('empty_add_options.f90', &
            '! { dg-add-options "" }')
        call write_options_fixture('nonempty_options.f90', &
            '! { dg-options "-O2" }')

        open(newunit=unit, file=DG_FIXTURE_DIR//'/spaced_run.f90', &
            status='replace', action='write')
        write(unit, '(A)') '! { dg-do  run }'
        write(unit, '(A)') '! { dg-options "" }'
        write(unit, '(A)') 'program spaced_run'
        write(unit, '(A)') 'print *, 42'
        write(unit, '(A)') 'end program spaced_run'
        close(unit)
    end subroutine write_dg_fixtures

    subroutine write_options_fixture(filename, directive)
        character(len=*), intent(in) :: filename
        character(len=*), intent(in) :: directive
        integer :: unit

        open(newunit=unit, file=DG_FIXTURE_DIR//'/'//filename, &
            status='replace', action='write')
        write(unit, '(A)') directive
        write(unit, '(A)') 'subroutine noop_options'
        write(unit, '(A)') 'end subroutine noop_options'
        close(unit)
    end subroutine write_options_fixture

    subroutine run_undefined_output_smoke(passed)
        logical, intent(inout) :: passed

        if (.not. run_smoke('timeout 120 bash '//SCRIPT// &
            ' --suite fortfront-f90'// &
            ' --file issue_104_if_condition_identifiers.f90'// &
            ' --file undefined_var_segfault.f90'// &
            ' --file issue_2349_data_implied_do.f90'// &
            ' --report '//UNDEFINED_REPORT, UNDEFINED_REPORT, 3)) &
            passed = .false.
        if (count_lines_with(UNDEFINED_REPORT, &
            '"status":"PASS"', 'undefined reference output') /= 3) &
            passed = .false.

        call write_undefined_output_fixture()
        if (.not. run_failure('FFC_FORTFRONT_DIR='//UNDEFINED_FIXTURE_ROOT// &
            ' FFC_UNDEFINED_OUTPUT_MANIFEST='//UNDEFINED_MANIFEST// &
            ' bash '//SCRIPT//' --suite fortfront-f90'// &
            ' --file undefined_nonzero.f90 --report '// &
            UNDEFINED_NEGATIVE_REPORT, &
            'undefined-output execution failed')) passed = .false.
        if (.not. file_contains(UNDEFINED_NEGATIVE_REPORT, &
            '"file":"undefined_nonzero.f90","status":"FAIL",'// &
            '"ffc_exit":1,"ref_exit":1')) &
            passed = .false.

        call write_undefined_manifest('ast_coverage_control_flow.f90')
        if (.not. run_failure('FFC_UNDEFINED_OUTPUT_MANIFEST='// &
            UNDEFINED_MANIFEST//' bash '//SCRIPT//' --suite fortfront-f90', &
            'both xfail and undefined-output')) passed = .false.
        call write_undefined_manifest('20231103-1.f90')
        if (.not. run_failure('FFC_UNDEFINED_OUTPUT_MANIFEST='// &
            UNDEFINED_MANIFEST//' bash '//SCRIPT//' --suite gfortran-dg', &
            'both skip and undefined-output')) passed = .false.
    end subroutine run_undefined_output_smoke

    subroutine write_undefined_output_fixture()
        integer :: unit

        call execute_command_line('rm -rf '//UNDEFINED_FIXTURE_ROOT)
        call execute_command_line('mkdir -p '//UNDEFINED_FIXTURE_ROOT// &
            '/examples/f90')
        open(newunit=unit, file=UNDEFINED_FIXTURE_ROOT// &
            '/examples/f90/undefined_nonzero.f90', &
            status='replace', action='write')
        write(unit, '(A)') 'program undefined_nonzero'
        write(unit, '(A)') 'stop 1'
        write(unit, '(A)') 'end program undefined_nonzero'
        close(unit)
        call write_undefined_manifest('undefined_nonzero.f90')
    end subroutine write_undefined_output_fixture

    subroutine write_undefined_manifest(entry)
        character(len=*), intent(in) :: entry
        integer :: unit

        open(newunit=unit, file=UNDEFINED_MANIFEST, status='replace', &
            action='write')
        write(unit, '(A)') entry
        close(unit)
    end subroutine write_undefined_manifest

    integer function count_lines_with(path, first, second) result(count)
        character(len=*), intent(in) :: path
        character(len=*), intent(in) :: first
        character(len=*), intent(in) :: second
        integer :: unit
        integer :: io_stat
        character(len=4096) :: line

        count = 0
        open(newunit=unit, file=path, status='old', action='read', iostat=io_stat)
        if (io_stat /= 0) return
        do
            read(unit, '(A)', iostat=io_stat) line
            if (io_stat /= 0) exit
            if (index(line, first) > 0 .and. index(line, second) > 0) &
                count = count + 1
        end do
        close(unit)
    end function count_lines_with

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
