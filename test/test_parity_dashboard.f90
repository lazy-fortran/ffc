program test_parity_dashboard
    implicit none

    character(len=*), parameter :: ROOT = '/tmp/ffc_parity_dashboard_test'
    character(len=*), parameter :: REPORT_DIR = ROOT//'/reports'
    character(len=*), parameter :: MANIFEST_DIR = ROOT//'/manifests'
    character(len=*), parameter :: OUTPUT_ONE = ROOT//'/one.md'
    character(len=*), parameter :: OUTPUT_TWO = ROOT//'/two.md'
    character(len=*), parameter :: LOG_PATH = ROOT//'/failure.log'
    character(len=64) :: fixture_source_digest
    character(len=64) :: fixture_binary_digest
    character(len=40) :: fixture_ffc_revision
    character(len=1024) :: fixture_binary_path
    logical :: passed

    passed = .true.
    call write_valid_inputs()
    if (.not. generation_succeeds(.false., OUTPUT_ONE)) passed = .false.
    if (.not. dashboard_has_expected_content()) passed = .false.
    if (.not. generation_succeeds(.true., OUTPUT_TWO)) passed = .false.
    if (.not. files_match(OUTPUT_ONE, OUTPUT_TWO)) passed = .false.
    if (.not. freshness_check_works()) passed = .false.
    if (.not. production_snapshot_check_works()) passed = .false.
    if (.not. stale_binary_check_works()) passed = .false.
    if (.not. dirty_git_tree_check_works()) passed = .false.
    if (.not. snapshot_negative_cases_fail()) passed = .false.
    if (.not. negative_cases_fail()) passed = .false.

    if (.not. passed) stop 1
    print *, 'PASS: parity dashboard generation'

contains

    subroutine write_valid_inputs()
        call execute_command_line('rm -rf '//ROOT)
        call execute_command_line('mkdir -p '//REPORT_DIR//' '//MANIFEST_DIR)
        call read_fixture_digests()
        call write_fortfront_f90_report(.false., fixture_ffc_revision)
        call write_fortfront_lf_report()
        call write_lfortran_report(fixture_ffc_revision)
        call write_gfortran_report()
        call write_manifests(.true.)
    end subroutine write_valid_inputs

    subroutine read_fixture_digests()
        integer :: io_stat, unit

        fixture_source_digest = ''
        fixture_binary_digest = ''
        fixture_ffc_revision = ''
        fixture_binary_path = ''
        call execute_command_line('bash -c ''PROJECT_DIR="$PWD"; '// &
            'source scripts/lib_conformance.sh; git rev-parse HEAD; '// &
            'ffc_source_sha256 "$PWD"; binary=$(find_ffc); '// &
            'printf "%s\n" "$binary"; sha256sum "$binary" | cut -d " " -f 1'' > '// &
            ROOT//'/digests', exitstat=io_stat)
        if (io_stat /= 0) then
            return
        end if
        open(newunit=unit, file=ROOT//'/digests', status='old', action='read', &
            iostat=io_stat)
        if (io_stat /= 0) return
        read(unit, '(A)', iostat=io_stat) fixture_ffc_revision
        if (io_stat == 0) read(unit, '(A)', iostat=io_stat) fixture_source_digest
        if (io_stat == 0) read(unit, '(A)', iostat=io_stat) fixture_binary_path
        if (io_stat == 0) read(unit, '(A)', iostat=io_stat) fixture_binary_digest
        close(unit)
    end subroutine read_fixture_digests

    subroutine write_fortfront_f90_report(duplicate, ffc_revision)
        logical, intent(in) :: duplicate
        character(len=*), intent(in) :: ffc_revision
        integer :: unit

        open(newunit=unit, file=REPORT_DIR//'/fortfront-f90.jsonl', &
            status='replace', action='write')
        write(unit, '(A)') file_record('fortfront-f90', 'pass.f90', 'PASS')
        write(unit, '(A)') file_record('fortfront-f90', 'gpu_pass.f90', 'PASS')
        write(unit, '(A)') file_record('fortfront-f90', 'xfail.f90', 'XFAIL')
        if (duplicate) then
            write(unit, '(A)') file_record('fortfront-f90', 'xfail.f90', 'XFAIL')
        end if
        write(unit, '(A)') file_record('fortfront-f90', 'fail.f90', 'FAIL')
        write(unit, '(A)') summary_record('fortfront-f90', 2, 1, 0, 1, &
            0, 0, 4, ffc_revision, repeat('b', 40))
        close(unit)
    end subroutine write_fortfront_f90_report

    subroutine write_fortfront_lf_report()
        integer :: unit

        open(newunit=unit, file=REPORT_DIR//'/fortfront-lf.jsonl', &
            status='replace', action='write')
        write(unit, '(A)') file_record('fortfront-lf', 'lazy.lf', 'PASS')
        write(unit, '(A)') summary_record('fortfront-lf', 1, 0, 0, 0, &
            0, 0, 1, fixture_ffc_revision, repeat('b', 40))
        close(unit)
    end subroutine write_fortfront_lf_report

    subroutine write_lfortran_report(ffc_revision, source_digest, binary_digest)
        character(len=*), intent(in) :: ffc_revision
        character(len=*), intent(in), optional :: source_digest
        character(len=*), intent(in), optional :: binary_digest
        integer :: unit

        open(newunit=unit, file=REPORT_DIR//'/lfortran.jsonl', &
            status='replace', action='write')
        write(unit, '(A)') file_record('lfortran', 'pass.f90', 'PASS')
        write(unit, '(A)') file_record('lfortran', 'coarray.f90', 'XFAIL')
        if (present(binary_digest)) then
            write(unit, '(A)') summary_record('lfortran', 1, 1, 0, 0, 0, 0, &
                2, ffc_revision, repeat('d', 40), &
                source_digest, binary_digest)
        else if (present(source_digest)) then
            write(unit, '(A)') summary_record('lfortran', 1, 1, 0, 0, 0, 0, &
                2, ffc_revision, repeat('d', 40), &
                source_digest)
        else
            write(unit, '(A)') summary_record('lfortran', 1, 1, 0, 0, 0, 0, &
                2, ffc_revision, repeat('d', 40))
        end if
        close(unit)
    end subroutine write_lfortran_report

    subroutine write_gfortran_report()
        integer :: unit

        open(newunit=unit, file=REPORT_DIR//'/gfortran-dg.jsonl', &
            status='replace', action='write')
        write(unit, '(A)') noref_record()
        write(unit, '(A)') file_record('gfortran-dg', 'openmp.f90', 'SKIP')
        write(unit, '(A)') summary_record('gfortran-dg', 1, 0, 0, 0, 1, 1, 2, &
            fixture_ffc_revision, repeat('e', 40))
        close(unit)
    end subroutine write_gfortran_report

    function file_record(suite, file_name, status) result(line)
        character(len=*), intent(in) :: suite, file_name, status
        character(len=:), allocatable :: line

        line = '{"suite":"'//suite//'","file":"'//file_name// &
            '","status":"'//status//'","ffc_exit":0,"ref_exit":0,'// &
            '"note":"fixture"}'
    end function file_record

    function noref_record() result(line)
        character(len=:), allocatable :: line

        line = '{"suite":"gfortran-dg","file":"noref.f90",'// &
            '"status":"PASS","ffc_exit":0,"ref_exit":1,'// &
            '"note":"gfortran rejects; ffc runs (NO-REF)","noref":true}'
    end function noref_record

    function summary_record(suite, pass_count, xfail_count, xpass_count, &
            fail_count, noref_count, skip_count, total_count, ffc_revision, &
            corpus_revision, source_digest, binary_digest) result(line)
        character(len=*), intent(in) :: suite, ffc_revision, corpus_revision
        character(len=*), intent(in), optional :: source_digest
        character(len=*), intent(in), optional :: binary_digest
        integer, intent(in) :: pass_count, xfail_count, xpass_count, fail_count
        integer, intent(in) :: noref_count, skip_count, total_count
        character(len=:), allocatable :: line
        character(len=:), allocatable :: source_value
        character(len=:), allocatable :: binary_value
        character(len=:), allocatable :: corpus_tree_value
        character(len=:), allocatable :: corpus_files_value

        source_value = fixture_source_digest
        if (present(source_digest)) source_value = source_digest
        binary_value = fixture_binary_digest
        if (present(binary_digest)) binary_value = binary_digest
        select case (suite)
        case ('fortfront-f90')
            corpus_tree_value = repeat('4', 40)
            corpus_files_value = repeat('a', 64)
        case ('fortfront-lf')
            corpus_tree_value = repeat('4', 40)
            corpus_files_value = repeat('b', 64)
        case ('lfortran')
            corpus_tree_value = repeat('6', 40)
            corpus_files_value = repeat('c', 64)
        case default
            corpus_tree_value = repeat('7', 40)
            corpus_files_value = repeat('d', 64)
        end select

        line = '{"suite":"'//suite//'","status":"SUMMARY","pass":'// &
            integer_text(pass_count)//',"xfail":'//integer_text(xfail_count)// &
            ',"xpass":'//integer_text(xpass_count)//',"fail":'// &
            integer_text(fail_count)//',"noref":'//integer_text(noref_count)// &
            ',"skip":'//integer_text(skip_count)// &
            ',"warning_unchecked":0,"total":'//integer_text(total_count)// &
            ',"schema_version":1,"full_run":true,"provenance_verified":true'// &
            ',"ffc_revision":"'//ffc_revision// &
            '","ffc_source_sha256":"'//source_value// &
            '","ffc_binary_sha256":"'//binary_value// &
            '","fortfront_revision":"'//repeat('b', 40)// &
            '","fortfront_tree":"'//repeat('4', 40)// &
            '","liric_revision":"'//repeat('c', 40)// &
            '","liric_tree":"'//repeat('5', 40)// &
            '","corpus_revision":"'//corpus_revision// &
            '","corpus_tree":"'//corpus_tree_value// &
            '","corpus_files_sha256":"'//corpus_files_value//'"}'
    end function summary_record

    function integer_text(value) result(text)
        integer, intent(in) :: value
        character(len=:), allocatable :: text
        character(len=32) :: buffer

        write(buffer, '(I0)') value
        text = trim(buffer)
    end function integer_text

    subroutine write_manifests(include_fail_owner)
        logical, intent(in) :: include_fail_owner
        integer :: unit

        call write_manifest('xfail_fortfront_f90.txt', &
            'xfail.f90 # owner=lazy-fortran/ffc#447; reason=fixture')
        call write_manifest('xfail_lfortran.txt', &
            'coarray.f90 # scope=coarray; reason=fixture')
        call write_manifest('skip_gfortran_dg.txt', &
            'openmp.f90 # scope=OpenMP; reason=fixture')
        call write_manifest('scopes_fortfront_f90.txt', &
            'gpu_pass.f90 # scope=GPU; reason=fixture')
        call write_manifest('scopes_lfortran.txt', &
            'coarray.f90 # scope=coarray; reason=fixture')
        call write_manifest('scopes_gfortran_dg.txt', &
            'openmp.f90 # scope=OpenMP; reason=fixture')
        open(newunit=unit, file=MANIFEST_DIR//'/owner_subsystems.txt', &
            status='replace', action='write')
        write(unit, '(A)') 'owner=lazy-fortran/ffc#447 arrays'
        write(unit, '(A)') 'owner=lazy-fortran/ffc#448 modules-scope'
        close(unit)
        open(newunit=unit, file=MANIFEST_DIR//'/fail_owners_fortfront_f90.txt', &
            status='replace', action='write')
        if (include_fail_owner) then
            write(unit, '(A)') &
                'fail.f90 # owner=lazy-fortran/ffc#448; reason=fixture'
        end if
        close(unit)
    end subroutine write_manifests

    subroutine write_manifest(name, line)
        character(len=*), intent(in) :: name, line
        integer :: unit

        open(newunit=unit, file=MANIFEST_DIR//'/'//name, &
            status='replace', action='write')
        write(unit, '(A)') line
        close(unit)
    end subroutine write_manifest

    logical function generation_succeeds(reverse_order, output) result(ok)
        logical, intent(in) :: reverse_order
        character(len=*), intent(in) :: output
        integer :: exit_stat

        call execute_command_line(generator_command(reverse_order, output, &
            .false.)//' > '//LOG_PATH//' 2>&1', exitstat=exit_stat)
        ok = exit_stat == 0
        if (.not. ok) print *, 'FAIL: valid dashboard generation'
    end function generation_succeeds

    function generator_command(reverse_order, output, check) result(command)
        logical, intent(in) :: reverse_order, check
        character(len=*), intent(in) :: output
        character(len=:), allocatable :: command
        character(len=:), allocatable :: reports

        if (reverse_order) then
            reports = report_argument('gfortran-dg')// &
                report_argument('lfortran')//report_argument('fortfront-lf')// &
                report_argument('fortfront-f90')
        else
            reports = report_argument('fortfront-f90')// &
                report_argument('fortfront-lf')//report_argument('lfortran')// &
                report_argument('gfortran-dg')
        end if
        command = revision_environment()// &
            ' timeout 120 bash scripts/generate_parity_dashboard.sh'// &
            reports//' --manifest-dir '//MANIFEST_DIR//' --output '//output
        if (reverse_order) command = 'LC_ALL=de_AT.utf8 POSIXLY_CORRECT=1 '//command
        if (check) command = command//' --check'
    end function generator_command

    function revision_environment() result(environment)
        character(len=:), allocatable :: environment

        environment = 'FFC_DASHBOARD_FORTFRONT_REVISION='//repeat('b', 40)// &
            ' FFC_DASHBOARD_LIRIC_REVISION='//repeat('c', 40)// &
            ' FFC_DASHBOARD_LFORTRAN_REVISION='//repeat('d', 40)// &
            ' FFC_DASHBOARD_GCC_REVISION='//repeat('e', 40)// &
            ' FFC_DASHBOARD_FORTFRONT_TREE='//repeat('4', 40)// &
            ' FFC_DASHBOARD_LIRIC_TREE='//repeat('5', 40)// &
            ' FFC_DASHBOARD_LFORTRAN_TREE='//repeat('6', 40)// &
            ' FFC_DASHBOARD_GCC_TREE='//repeat('7', 40)// &
            ' FFC_DASHBOARD_FORTFRONT_F90_FILES='//repeat('a', 64)// &
            ' FFC_DASHBOARD_FORTFRONT_LF_FILES='//repeat('b', 64)// &
            ' FFC_DASHBOARD_LFORTRAN_FILES='//repeat('c', 64)// &
            ' FFC_DASHBOARD_GFORTRAN_DG_FILES='//repeat('d', 64)
    end function revision_environment

    function report_argument(suite) result(argument)
        character(len=*), intent(in) :: suite
        character(len=:), allocatable :: argument

        argument = ' --report '//suite//'='//REPORT_DIR//'/'//suite//'.jsonl'
    end function report_argument

    logical function dashboard_has_expected_content() result(ok)
        ok = .true.
        call require_content('| All | 9 | 5 | 2 | 0 | 1 | 1 | 1 |', ok)
        call require_content('| Scoped | 6 | 4 | 1 | 0 | 1 | 1 | 0 |', ok)
        call require_content('62.5%', ok)
        call require_content('66.7%', ok)
        call require_content('| fortfront-f90 | XFAIL | arrays |', ok)
        call require_content('| fortfront-f90 | FAIL | modules-scope |', ok)
        call require_content('[lazy-fortran/ffc#448]'// &
            '(https://github.com/lazy-fortran/ffc/issues/448)', ok)
        if (.not. ok) print *, 'FAIL: generated dashboard content'
    end function dashboard_has_expected_content

    subroutine require_content(needle, ok)
        character(len=*), intent(in) :: needle
        logical, intent(inout) :: ok

        if (.not. file_contains(OUTPUT_ONE, needle)) then
            print *, 'FAIL: missing dashboard text: ', needle
            ok = .false.
        end if
    end subroutine require_content

    logical function freshness_check_works() result(ok)
        integer :: exit_stat

        call execute_command_line('printf stale > '//OUTPUT_TWO)
        call execute_command_line(generator_command(.false., OUTPUT_TWO, &
            .true.)//' > '//LOG_PATH//' 2>&1', exitstat=exit_stat)
        ok = exit_stat /= 0 .and. file_contains(LOG_PATH, 'dashboard is stale')
        call execute_command_line('cp '//OUTPUT_ONE//' '//OUTPUT_TWO)
        call execute_command_line(generator_command(.false., OUTPUT_TWO, &
            .true.)//' > '//LOG_PATH//' 2>&1', exitstat=exit_stat)
        ok = ok .and. exit_stat == 0
        if (.not. ok) print *, 'FAIL: dashboard freshness check'
    end function freshness_check_works

    logical function production_snapshot_check_works() result(ok)
        integer :: exit_stat

        call execute_command_line('timeout 120 bash '// &
            'scripts/generate_parity_dashboard.sh --from-snapshot '// &
            'test/conformance/parity_dashboard.tsv --output '// &
            'docs/PARITY_STATUS.md --check > '//LOG_PATH//' 2>&1', &
            exitstat=exit_stat)
        ok = exit_stat == 0
        if (.not. ok) print *, 'FAIL: production parity snapshot check'
    end function production_snapshot_check_works

    logical function stale_binary_check_works() result(ok)
        integer :: exit_stat

        call execute_command_line('cp '//trim(fixture_binary_path)//' '// &
            ROOT//'/stale-ffc')
        call execute_command_line('touch -d 2000-01-01 '//ROOT//'/stale-ffc')
        call execute_command_line('FFC_BIN='//ROOT//'/stale-ffc '// &
            generator_command(.false., OUTPUT_ONE, .false.)//' > '// &
            LOG_PATH//' 2>&1', exitstat=exit_stat)
        ok = exit_stat /= 0 .and. &
            file_contains(LOG_PATH, 'compiler binary predates')
        if (.not. ok) print *, 'FAIL: stale compiler binary check'
    end function stale_binary_check_works

    logical function dirty_git_tree_check_works() result(ok)
        integer :: exit_stat

        call execute_command_line('mkdir -p '//ROOT//'/dirty-repo')
        call execute_command_line('git -C '//ROOT//'/dirty-repo init -q')
        call execute_command_line('printf clean > '//ROOT//'/dirty-repo/input')
        call execute_command_line('git -C '//ROOT//'/dirty-repo add input')
        call execute_command_line('git -C '//ROOT//'/dirty-repo -c user.name=test '// &
            '-c user.email=test@example.invalid commit -qm initial')
        call execute_command_line('printf dirty >> '//ROOT//'/dirty-repo/input')
        call execute_command_line("bash -c 'source scripts/lib_conformance.sh; "// &
            "require_clean_git_tree "//ROOT//"/dirty-repo fixture' > "// &
            LOG_PATH//' 2>&1', exitstat=exit_stat)
        ok = exit_stat /= 0 .and. file_contains(LOG_PATH, 'dirty fixture Git tree')
        if (.not. ok) print *, 'FAIL: dirty Git tree check'
    end function dirty_git_tree_check_works

    logical function snapshot_negative_cases_fail() result(ok)
        ok = .true.
        call copy_production_snapshot()
        call execute_command_line("sed -i '/^revision[[:space:]]ffc/"// &
            "s/[^[:space:]]*$/"//repeat('0', 40)//"/' "//ROOT//'/bad.tsv')
        ok = expect_snapshot_failure('snapshot ffc revision is not an ancestor') .and. ok
        call copy_production_snapshot()
        call execute_command_line("sed -i '/^digest[[:space:]]ffc-source/"// &
            "s/[^[:space:]]*$/"//repeat('9', 64)//"/' "//ROOT//'/bad.tsv')
        ok = expect_snapshot_failure('stale snapshot source digest') .and. ok
        call copy_production_snapshot()
        call execute_command_line("sed -i '/^digest[[:space:]]manifests/"// &
            "s/[^[:space:]]*$/"//repeat('9', 64)//"/' "//ROOT//'/bad.tsv')
        ok = expect_snapshot_failure('stale snapshot manifest digest') .and. ok
        call copy_production_snapshot()
        call execute_command_line("sed -i '/^suite[[:space:]]fortfront-f90/"// &
            "{s/439/440/;s/339/340/;}' "//ROOT//'/bad.tsv')
        ok = expect_snapshot_failure('All totals do not equal suites') .and. ok
        call copy_production_snapshot()
        call execute_command_line("sed -i '/^view[[:space:]]Scoped/"// &
            "{s/10396/10395/;s/2467/2466/;}' "//ROOT//'/bad.tsv')
        ok = expect_snapshot_failure('Scoped totals do not equal') .and. ok
        call copy_production_snapshot()
        call execute_command_line("sed -i '/owner=lazy-fortran.ffc#297/"// &
            "s/2$/3/' "//ROOT//'/bad.tsv')
        ok = expect_snapshot_failure('owner totals do not equal') .and. ok
    end function snapshot_negative_cases_fail

    subroutine copy_production_snapshot()
        call execute_command_line('cp test/conformance/parity_dashboard.tsv '// &
            ROOT//'/bad.tsv')
    end subroutine copy_production_snapshot

    logical function expect_snapshot_failure(diagnostic) result(ok)
        character(len=*), intent(in) :: diagnostic
        integer :: exit_stat

        call execute_command_line('timeout 120 bash '// &
            'scripts/generate_parity_dashboard.sh --from-snapshot '// &
            ROOT//'/bad.tsv --manifest-dir test/conformance --output '// &
            ROOT//'/bad.md > '//LOG_PATH//' 2>&1', exitstat=exit_stat)
        ok = exit_stat /= 0 .and. file_contains(LOG_PATH, diagnostic)
        if (.not. ok) then
            print *, 'FAIL: snapshot diagnostic: ', diagnostic
            call execute_command_line('cat '//LOG_PATH)
        end if
    end function expect_snapshot_failure

    logical function negative_cases_fail() result(ok)
        ok = .true.
        call write_valid_inputs()
        call execute_command_line('rm -f '//REPORT_DIR//'/lfortran.jsonl')
        ok = expect_failure('missing report') .and. ok
        call write_valid_inputs()
        call write_lfortran_report(repeat('f', 40))
        ok = expect_failure('stale revision') .and. ok
        call write_valid_inputs()
        call write_lfortran_report(fixture_ffc_revision, repeat('9', 64))
        ok = expect_failure('stale source digest') .and. ok
        call write_valid_inputs()
        call write_lfortran_report(fixture_ffc_revision, fixture_source_digest, &
            repeat('8', 64))
        ok = expect_failure('stale binary digest') .and. ok
        call write_valid_inputs()
        call write_fortfront_f90_report(.true., fixture_ffc_revision)
        ok = expect_failure('duplicate report row') .and. ok
        call write_valid_inputs()
        call replace_status_with_broken()
        ok = expect_failure('unknown status') .and. ok
        call write_valid_inputs()
        call write_manifests(.false.)
        ok = expect_failure('missing owner') .and. ok
        call write_valid_inputs()
        call write_invalid_fortfront_record('malformed')
        ok = expect_failure('JSON') .and. ok
        call write_valid_inputs()
        call write_invalid_fortfront_record('duplicate-key')
        ok = expect_failure('duplicate JSON key') .and. ok
        call write_valid_inputs()
        call write_invalid_fortfront_record('warning')
        ok = expect_failure('invalid warning expectation') .and. ok
        call write_valid_inputs()
        call write_invalid_fortfront_record('type')
        ok = expect_failure('wrong field type') .and. ok
        call write_valid_inputs()
        call write_invalid_fortfront_record('missing')
        ok = expect_failure('missing field: note') .and. ok
    end function negative_cases_fail

    logical function expect_failure(diagnostic) result(ok)
        character(len=*), intent(in) :: diagnostic
        integer :: exit_stat

        call execute_command_line(generator_command(.false., OUTPUT_ONE, &
            .false.)//' > '//LOG_PATH//' 2>&1', exitstat=exit_stat)
        ok = exit_stat /= 0 .and. file_contains(LOG_PATH, diagnostic)
        if (.not. ok) print *, 'FAIL: missing diagnostic: ', diagnostic
    end function expect_failure

    subroutine replace_status_with_broken()
        integer :: unit

        open(newunit=unit, file=REPORT_DIR//'/fortfront-f90.jsonl', &
            status='replace', action='write')
        write(unit, '(A)') file_record('fortfront-f90', 'pass.f90', 'BROKEN')
        write(unit, '(A)') summary_record('fortfront-f90', 1, 0, 0, 0, &
            0, 0, 1, fixture_ffc_revision, repeat('b', 40))
        close(unit)
    end subroutine replace_status_with_broken

    subroutine write_invalid_fortfront_record(kind)
        character(len=*), intent(in) :: kind
        integer :: unit

        open(newunit=unit, file=REPORT_DIR//'/fortfront-f90.jsonl', &
            status='replace', action='write')
        select case (kind)
        case ('malformed')
            write(unit, '(A)') '{"suite":"fortfront-f90","file":"pass.f90"'
        case ('duplicate-key')
            write(unit, '(A)') '{"suite":"fortfront-f90","file":"pass.f90",'// &
                '"status":"PASS","status":"PASS","ffc_exit":0,'// &
                '"ref_exit":0,"note":"fixture"}'
        case ('warning')
            write(unit, '(A)') '{"suite":"fortfront-f90","file":"pass.f90",'// &
                '"status":"PASS","ffc_exit":0,"ref_exit":0,'// &
                '"note":"fixture","warning_expectation":"checked"}'
        case ('type')
            write(unit, '(A)') '{"suite":"fortfront-f90","file":"pass.f90",'// &
                '"status":"PASS","ffc_exit":"0","ref_exit":0,'// &
                '"note":"fixture"}'
        case ('missing')
            write(unit, '(A)') '{"suite":"fortfront-f90","file":"pass.f90",'// &
                '"status":"PASS","ffc_exit":0,"ref_exit":0}'
        end select
        write(unit, '(A)') summary_record('fortfront-f90', 1, 0, 0, 0, &
            0, 0, 1, fixture_ffc_revision, repeat('b', 40))
        close(unit)
    end subroutine write_invalid_fortfront_record

    logical function files_match(left, right) result(ok)
        character(len=*), intent(in) :: left, right
        integer :: exit_stat

        call execute_command_line('cmp -s '//left//' '//right, exitstat=exit_stat)
        ok = exit_stat == 0
        if (.not. ok) print *, 'FAIL: dashboard output is not deterministic'
    end function files_match

    logical function file_contains(path, needle) result(found)
        character(len=*), intent(in) :: path, needle
        character(len=4096) :: line
        integer :: io_stat, unit

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

end program test_parity_dashboard
