program test_conformance_observation_once
    use conformance_temp_dir, only: make_temp_root, remove_temp_root
    implicit none

    character(len=*), parameter :: GAUNTLET = &
        'scripts/conformance_gauntlet.sh'
    character(len=*), parameter :: CLASSIFIER = &
        'scripts/classify_conformance_observations.sh'
    character(len=*), parameter :: OBSERVATION_TOOL = &
        'scripts/conformance_observation.py'
    character(len=:), allocatable :: root, fortfront, lfortran, compiler
    character(len=:), allocatable :: fake_program
    character(len=:), allocatable :: pass_program
    character(len=:), allocatable :: compile_counter, run_counter
    character(len=:), allocatable :: xfail_manifest, skip_manifest
    character(len=:), allocatable :: noref_manifest, observations
    character(len=:), allocatable :: observation_copy, normal_report
    character(len=:), allocatable :: empty_report, log_file, command, digest
    character(len=:), allocatable :: missing_observations, missing_report
    character(len=:), allocatable :: merged_observations
    character(len=:), allocatable :: repeat_observations, repeat_report
    character(len=:), allocatable :: malformed_observations, malformed_report
    character(len=:), allocatable :: missing_extra_observations
    character(len=:), allocatable :: missing_extra_report
    character(len=:), allocatable :: corpus_digest, repeat_corpus_digest
    integer :: compile_count_before, exit_status
    logical :: passed

    root = make_temp_root('observation_once')
    fortfront = root//'/fortfront'
    lfortran = root//'/lfortran'
    compiler = root//'/instrumented_ffc.sh'
    fake_program = root//'/instrumented_program.sh'
    pass_program = root//'/instrumented_pass_program.sh'
    compile_counter = root//'/compile_count'
    run_counter = root//'/run_count'
    xfail_manifest = root//'/xfail.txt'
    skip_manifest = root//'/skip.txt'
    noref_manifest = root//'/noref.txt'
    observations = root//'/observations.jsonl'
    observation_copy = root//'/observations.before-classification.jsonl'
    normal_report = root//'/normal.jsonl'
    empty_report = root//'/empty.jsonl'
    log_file = root//'/run.log'
    missing_observations = root//'/missing-attempt.observations.jsonl'
    missing_report = root//'/missing-attempt.jsonl'
    merged_observations = root//'/invalid-merged.observations.jsonl'
    repeat_observations = root//'/repeat.observations.jsonl'
    repeat_report = root//'/repeat.jsonl'
    malformed_observations = root//'/malformed.observations.jsonl'
    malformed_report = root//'/malformed.jsonl'
    missing_extra_observations = root//'/missing-extra.observations.jsonl'
    missing_extra_report = root//'/missing-extra.jsonl'
    passed = .true.

    call write_fixture()
    call write_instrumented_program()
    call write_instrumented_pass_program()
    call write_instrumented_compiler()
    call write_integer(compile_counter, 0)
    call write_integer(run_counter, 0)
    call write_expectation_manifest()
    call write_manifest(skip_manifest, '')
    call write_manifest(noref_manifest, '')

    command = environment_prefix()//' timeout 60 bash '//GAUNTLET// &
        ' --suite fortfront-lf --file observed_once.lf'// &
        ' --file observed_pass_once.lf --ffc '//compiler// &
        ' --observations '//observations//' --report '//normal_report// &
        ' > '//log_file//' 2>&1'
    call execute_command_line(command, exitstat=exit_status)
    if (exit_status /= 1) then
        print *, 'FAIL: normal view did not gate its independent XPASS'
        passed = .false.
    end if

    call execute_command_line('cp '//observations//' '//observation_copy, &
        exitstat=exit_status)
    if (exit_status /= 0) passed = .false.
    digest = sha256_of(observations)

    command = 'timeout 60 bash '//CLASSIFIER//' --suite fortfront-lf'// &
        ' --observations '//observations//' --report '//empty_report// &
        ' --no-xfail >> '//log_file//' 2>&1'
    call execute_command_line(command, exitstat=exit_status)
    if (exit_status /= 1) then
        print *, 'FAIL: XFAIL-disabled classification did not expose FAIL'
        passed = .false.
    end if

    call execute_command_line('cmp -s '//observations//' '//observation_copy, &
        exitstat=exit_status)
    if (exit_status /= 0) then
        print *, 'FAIL: classification modified the raw observation'
        passed = .false.
    end if
    if (read_integer(compile_counter) /= 2) then
        print *, 'FAIL: selected cases were not each compiled exactly once'
        passed = .false.
    end if
    if (read_integer(run_counter) /= 2) then
        print *, 'FAIL: selected programs were not each executed exactly once'
        passed = .false.
    end if

    if (count_lines_with(observations, '"file":') /= 2 .or. &
            .not. file_contains(observations, &
                '"file":"observed_once.lf","status":"FAIL"') .or. &
            .not. file_contains(observations, &
                '"file":"observed_pass_once.lf","status":"PASS"') .or. &
            .not. file_contains(observations, &
                '"report_kind":"observation"') .or. &
            file_contains(observations, '"expectation":')) then
        print *, 'FAIL: raw report is not one expectation-neutral observation'
        passed = .false.
    end if
    if (.not. file_contains(normal_report, &
            '"file":"observed_once.lf","status":"XFAIL"') .or. &
            .not. file_contains(normal_report, &
                '"observed_status":"FAIL","expectation":"xfail"') .or. &
            .not. file_contains(normal_report, &
                '"file":"observed_pass_once.lf","status":"XPASS"') .or. &
            .not. file_contains(normal_report, &
                '"observed_status":"PASS","expectation":"xfail"') .or. &
            .not. file_contains(normal_report, &
                '"classification_mode":"manifest"')) then
        print *, 'FAIL: normal manifest view is incorrect'
        passed = .false.
    end if
    if (.not. file_contains(empty_report, &
            '"file":"observed_once.lf","status":"FAIL"') .or. &
            .not. file_contains(empty_report, &
                '"observed_status":"FAIL","expectation":"none"') .or. &
            .not. file_contains(empty_report, &
                '"file":"observed_pass_once.lf","status":"PASS"') .or. &
            .not. file_contains(empty_report, &
                '"observed_status":"PASS","expectation":"none"') .or. &
            .not. file_contains(empty_report, &
                '"classification_mode":"xfail-disabled"')) then
        print *, 'FAIL: XFAIL-disabled view is incorrect'
        passed = .false.
    end if
    if (len_trim(digest) /= 64 .or. &
            .not. file_contains(normal_report, &
                '"observation_sha256":"'//trim(digest)//'"') .or. &
            .not. file_contains(empty_report, &
                '"observation_sha256":"'//trim(digest)//'"')) then
        print *, 'FAIL: classification views do not bind the observation'
        passed = .false.
    end if

    ! The classifier must reject an incomplete JSONL contract without
    ! replacing an already published report.
    call execute_command_line('cp '//observations//' '// &
        malformed_observations, exitstat=exit_status)
    call append_line(malformed_observations, &
        '{"suite":"fortfront-lf","file":"late.lf",'// &
        '"status":"PASS","ffc_exit":0,"ref_exit":-1,'// &
        '"note":"record after summary"}')
    call write_manifest(malformed_report, 'previous report remains')
    command = 'timeout 60 bash '//CLASSIFIER//' --suite fortfront-lf'// &
        ' --observations '//malformed_observations//' --report '// &
        malformed_report//' --no-xfail >> '//log_file//' 2>&1'
    call execute_command_line(command, exitstat=exit_status)
    if (exit_status == 0 .or. .not. file_contains(log_file, &
            'SUMMARY must be final') .or. .not. file_contains( &
            malformed_report, 'previous report remains')) then
        print *, 'FAIL: malformed observation replaced a valid report'
        passed = .false.
    end if

    ! An individually valid attempt that omitted one selected case must not be
    ! merged with the complete attempt.
    command = environment_prefix()//' timeout 60 bash '//GAUNTLET// &
        ' --suite fortfront-lf --file observed_once.lf --ffc '//compiler// &
        ' --observations '//missing_observations//' --report '// &
        missing_report//' >> '//log_file//' 2>&1'
    call execute_command_line(command, exitstat=exit_status)
    if (exit_status /= 0) then
        print *, 'FAIL: valid one-case attempt fixture did not run'
        passed = .false.
    end if
    command = 'python3 '//OBSERVATION_TOOL//' merge --suite fortfront-lf'// &
        ' --output '//merged_observations//' '//observations//' '// &
        missing_observations//' >> '//log_file//' 2>&1'
    call execute_command_line(command, exitstat=exit_status)
    if (exit_status == 0 .or. .not. file_contains(log_file, &
            'case set/order differs')) then
        print *, 'FAIL: repeat merge accepted an omitted case'
        passed = .false.
    end if

    ! The actual --repeat path must derive its identity from complete child
    ! observations rather than the pre-selection parent process.
    command = environment_prefix()//' timeout 60 bash '//GAUNTLET// &
        ' --suite fortfront-lf --file observed_once.lf'// &
        ' --file observed_pass_once.lf --repeat 2 --ffc '//compiler// &
        ' --observations '//repeat_observations//' --report '//repeat_report// &
        ' >> '//log_file//' 2>&1'
    call execute_command_line(command, exitstat=exit_status)
    if (exit_status /= 1) then
        print *, 'FAIL: repeat classification did not preserve XPASS gate'
        passed = .false.
    end if
    corpus_digest = summary_string_field(observations, &
        'corpus_files_sha256')
    repeat_corpus_digest = summary_string_field(repeat_observations, &
        'corpus_files_sha256')
    if (len_trim(corpus_digest) /= 64 .or. &
            repeat_corpus_digest /= corpus_digest .or. &
            corpus_digest == repeat('0', 64) .or. &
            .not. file_contains(repeat_observations, '"attempt_count":2') .or. &
            .not. file_contains(repeat_observations, &
                '"provenance_verified":false') .or. &
            .not. file_contains(repeat_observations, &
                '"reference_compiler":"GNU Fortran')) then
        print *, 'FAIL: repeat observation lost child provenance'
        passed = .false.
    end if

    ! Missing EXTRAFILES input is one explicit raw FAIL, not a side counter
    ! later overwritten by the main source's verdict.
    compile_count_before = read_integer(compile_counter)
    command = environment_prefix()//' timeout 60 bash '//GAUNTLET// &
        ' --suite lfortran --file minpack_01.f90 --ffc '//compiler// &
        ' --observations '//missing_extra_observations//' --report '// &
        missing_extra_report//' >> '//log_file//' 2>&1'
    call execute_command_line(command, exitstat=exit_status)
    if (exit_status /= 1 .or. &
            read_integer(compile_counter) /= compile_count_before .or. &
            count_lines_with(missing_extra_observations, '"file":') /= 1 .or. &
            .not. file_contains(missing_extra_observations, &
                '"file":"minpack_01.f90","status":"FAIL"') .or. &
            .not. file_contains(missing_extra_observations, &
                '"note":"missing extra source minpack_01_func.f90"') .or. &
            .not. file_contains(missing_extra_observations, &
                '"fail":1') .or. .not. file_contains( &
                missing_extra_observations, '"total":1')) then
        print *, 'FAIL: missing extra source lost its raw verdict'
        passed = .false.
    end if

    if (passed) call remove_temp_root(root)
    if (.not. passed) then
        print *, 'FAIL: observation-once scratch kept at ', root
        stop 1
    end if
    print *, 'PASS: one observation supports independent expectation views'

contains

    function environment_prefix() result(prefix)
        character(len=:), allocatable :: prefix

        prefix = 'OBS_COMPILE_COUNTER='//compile_counter// &
            ' OBS_RUN_COUNTER='//run_counter// &
            ' OBS_FAKE_PROGRAM='//fake_program// &
            ' OBS_FAKE_PASS_PROGRAM='//pass_program// &
            ' FFC_FORTFRONT_DIR='//fortfront// &
            ' FFC_LFORTRAN_DIR='//lfortran// &
            ' FFC_XFAIL_MANIFEST='//xfail_manifest// &
            ' FFC_SKIP_MANIFEST='//skip_manifest// &
            ' FFC_NOREF_MANIFEST='//noref_manifest
    end function environment_prefix

    subroutine write_fixture()
        integer :: unit_number

        call execute_command_line('mkdir -p '//fortfront//'/examples/lf')
        call execute_command_line('mkdir -p '//lfortran//'/integration_tests')
        open(newunit=unit_number, &
            file=fortfront//'/examples/lf/observed_once.lf', &
            status='replace', action='write')
        write(unit_number, '(a)') 'print 1'
        close(unit_number)
        open(newunit=unit_number, &
            file=fortfront//'/examples/lf/observed_pass_once.lf', &
            status='replace', action='write')
        write(unit_number, '(a)') 'print 2'
        close(unit_number)
        open(newunit=unit_number, &
            file=lfortran//'/integration_tests/minpack_01.f90', &
            status='replace', action='write')
        write(unit_number, '(a)') 'program minpack_01'
        write(unit_number, '(a)') 'end program minpack_01'
        close(unit_number)
    end subroutine write_fixture

    subroutine write_instrumented_program()
        integer :: unit_number

        open(newunit=unit_number, file=fake_program, status='replace', &
            action='write')
        write(unit_number, '(a)') '#!/usr/bin/env bash'
        write(unit_number, '(a)') 'set -uo pipefail'
        write(unit_number, '(a)') &
            'n=$(sed -n "1p" "$OBS_RUN_COUNTER")'
        write(unit_number, '(a)') &
            'printf "%s\n" "$((n + 1))" > "$OBS_RUN_COUNTER"'
        write(unit_number, '(a)') 'printf "instrumented failure\n"'
        write(unit_number, '(a)') 'exit 1'
        close(unit_number)
        call make_executable(fake_program)
    end subroutine write_instrumented_program

    subroutine write_instrumented_pass_program()
        integer :: unit_number

        open(newunit=unit_number, file=pass_program, status='replace', &
            action='write')
        write(unit_number, '(a)') '#!/usr/bin/env bash'
        write(unit_number, '(a)') 'set -uo pipefail'
        write(unit_number, '(a)') &
            'n=$(sed -n "1p" "$OBS_RUN_COUNTER")'
        write(unit_number, '(a)') &
            'printf "%s\n" "$((n + 1))" > "$OBS_RUN_COUNTER"'
        write(unit_number, '(a)') 'printf "instrumented success\n"'
        write(unit_number, '(a)') 'exit 0'
        close(unit_number)
        call make_executable(pass_program)
    end subroutine write_instrumented_pass_program

    subroutine write_instrumented_compiler()
        integer :: unit_number

        open(newunit=unit_number, file=compiler, status='replace', &
            action='write')
        write(unit_number, '(a)') '#!/usr/bin/env bash'
        write(unit_number, '(a)') 'set -uo pipefail'
        write(unit_number, '(a)') 'source_path="$1"'
        write(unit_number, '(a)') &
            'n=$(sed -n "1p" "$OBS_COMPILE_COUNTER")'
        write(unit_number, '(a)') &
            'printf "%s\n" "$((n + 1))" > "$OBS_COMPILE_COUNTER"'
        write(unit_number, '(a)') 'output=""'
        write(unit_number, '(a)') 'while [ "$#" -gt 0 ]; do'
        write(unit_number, '(a)') '    if [ "$1" = "-o" ]; then'
        write(unit_number, '(a)') '        output="$2"; shift 2'
        write(unit_number, '(a)') '    else'
        write(unit_number, '(a)') '        shift'
        write(unit_number, '(a)') '    fi'
        write(unit_number, '(a)') 'done'
        write(unit_number, '(a)') '[ -n "$output" ] || exit 2'
        write(unit_number, '(a)') &
            'if [ "$(basename "$source_path")" = "observed_pass_once.lf" ]; then'
        write(unit_number, '(a)') &
            '    cp "$OBS_FAKE_PASS_PROGRAM" "$output"'
        write(unit_number, '(a)') 'else'
        write(unit_number, '(a)') '    cp "$OBS_FAKE_PROGRAM" "$output"'
        write(unit_number, '(a)') 'fi'
        write(unit_number, '(a)') 'chmod +x "$output"'
        close(unit_number)
        call make_executable(compiler)
    end subroutine write_instrumented_compiler

    subroutine make_executable(path)
        character(len=*), intent(in) :: path
        integer :: status

        call execute_command_line('chmod +x '//path, exitstat=status)
        if (status /= 0) stop 2
    end subroutine make_executable

    subroutine write_expectation_manifest()
        integer :: unit_number

        open(newunit=unit_number, file=xfail_manifest, status='replace', &
            action='write')
        write(unit_number, '(a)') &
            'observed_once.lf # owner=lazy-fortran/ffc#1; reason=test fixture'
        write(unit_number, '(a)') 'observed_pass_once.lf # '// &
            'owner=lazy-fortran/ffc#1; reason=XPASS test fixture'
        close(unit_number)
    end subroutine write_expectation_manifest

    subroutine write_manifest(path, line)
        character(len=*), intent(in) :: path, line
        integer :: unit_number

        open(newunit=unit_number, file=path, status='replace', action='write')
        if (len_trim(line) > 0) write(unit_number, '(a)') trim(line)
        close(unit_number)
    end subroutine write_manifest

    subroutine append_line(path, line)
        character(len=*), intent(in) :: path, line
        integer :: unit_number

        open(newunit=unit_number, file=path, status='old', action='write', &
            position='append')
        write(unit_number, '(a)') line
        close(unit_number)
    end subroutine append_line

    subroutine write_integer(path, value)
        character(len=*), intent(in) :: path
        integer, intent(in) :: value
        integer :: unit_number

        open(newunit=unit_number, file=path, status='replace', action='write')
        write(unit_number, '(i0)') value
        close(unit_number)
    end subroutine write_integer

    integer function read_integer(path) result(value)
        character(len=*), intent(in) :: path
        integer :: io_status, unit_number

        value = -1
        open(newunit=unit_number, file=path, status='old', action='read', &
            iostat=io_status)
        if (io_status /= 0) return
        read(unit_number, *, iostat=io_status) value
        if (io_status /= 0) value = -1
        close(unit_number)
    end function read_integer

    function sha256_of(path) result(digest_text)
        character(len=*), intent(in) :: path
        character(len=:), allocatable :: digest_text
        character(len=:), allocatable :: digest_file
        character(len=64) :: buffer
        integer :: io_status, status, unit_number

        digest_file = root//'/digest.txt'
        call execute_command_line('sha256sum '//path//' > '//digest_file, &
            exitstat=status)
        if (status /= 0) then
            digest_text = ''
            return
        end if
        open(newunit=unit_number, file=digest_file, status='old', &
            action='read', iostat=io_status)
        if (io_status /= 0) then
            digest_text = ''
            return
        end if
        read(unit_number, '(a64)', iostat=io_status) buffer
        close(unit_number)
        if (io_status == 0) then
            digest_text = buffer
        else
            digest_text = ''
        end if
    end function sha256_of

    logical function file_contains(path, needle) result(found)
        character(len=*), intent(in) :: path, needle
        character(len=4096) :: line
        integer :: io_status, unit_number

        found = .false.
        open(newunit=unit_number, file=path, status='old', action='read', &
            iostat=io_status)
        if (io_status /= 0) return
        do
            read(unit_number, '(a)', iostat=io_status) line
            if (io_status /= 0) exit
            if (index(line, needle) > 0) found = .true.
        end do
        close(unit_number)
    end function file_contains

    integer function count_lines_with(path, needle) result(count)
        character(len=*), intent(in) :: path, needle
        character(len=4096) :: line
        integer :: io_status, unit_number

        count = 0
        open(newunit=unit_number, file=path, status='old', action='read', &
            iostat=io_status)
        if (io_status /= 0) return
        do
            read(unit_number, '(a)', iostat=io_status) line
            if (io_status /= 0) exit
            if (index(line, needle) > 0) count = count + 1
        end do
        close(unit_number)
    end function count_lines_with

    function summary_string_field(path, key) result(value)
        character(len=*), intent(in) :: path, key
        character(len=:), allocatable :: value
        character(len=4096) :: line
        character(len=:), allocatable :: marker, remainder
        integer :: end_quote, io_status, marker_position, unit_number

        value = ''
        marker = '"'//key//'":"'
        open(newunit=unit_number, file=path, status='old', action='read', &
            iostat=io_status)
        if (io_status /= 0) return
        do
            read(unit_number, '(a)', iostat=io_status) line
            if (io_status /= 0) exit
            if (index(line, '"status":"SUMMARY"') == 0) cycle
            marker_position = index(line, marker)
            if (marker_position == 0) exit
            remainder = line(marker_position + len(marker):)
            end_quote = index(remainder, '"')
            if (end_quote > 1) value = remainder(:end_quote - 1)
            exit
        end do
        close(unit_number)
    end function summary_string_field

end program test_conformance_observation_once
