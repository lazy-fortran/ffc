program test_conformance_flake_detection
    ! A single conformance run cannot tell a stable result from a case that
    ! flips between runs (ffc #599). These checks drive the gauntlet with a
    ! compiler wrapper that alternates between compiling and failing, so the
    ! same file is a PASS on one attempt and a FAIL on the next, and require
    ! the merged report to record it as FLAKY instead of taking one result.
    use conformance_temp_dir, only: make_temp_root, remove_temp_root
    implicit none

    character(len=*), parameter :: SCRIPT = 'scripts/conformance_gauntlet.sh'
    character(len=*), parameter :: CASE_FILE = 'api_pipeline_minimal_program.f90'
    character(len=:), allocatable :: root, wrapper, counter
    character(len=:), allocatable :: flaky_report, stable_report
    logical :: all_passed

    print *, '=== conformance flake detection test ==='

    root = make_temp_root('flake_detect')
    wrapper = root//'/alternating_ffc.sh'
    counter = root//'/attempt_counter'
    flaky_report = root//'/flaky.jsonl'
    stable_report = root//'/stable.jsonl'
    all_passed = .true.

    call write_alternating_wrapper(wrapper)

    ! A case that flips PASS/FAIL between attempts must be recorded FLAKY.
    if (run_command(repeated_run(flaky_report, wrapper, 3)) == 0) then
        print *, 'FAIL: repeated run with an unstable case exited 0'
        all_passed = .false.
    end if
    if (.not. file_contains(flaky_report, '"file":"'//CASE_FILE// &
                            '","status":"FLAKY"')) all_passed = .false.
    if (.not. file_contains(flaky_report, '"observed":"PASS|FAIL"')) &
        all_passed = .false.
    if (.not. file_contains(flaky_report, '"flaky":1')) all_passed = .false.

    ! The same case under the real compiler is stable and must not be flagged.
    if (run_command('timeout 300 bash '//SCRIPT//' --suite fortfront-f90'// &
                    ' --file '//CASE_FILE//' --repeat 3 --report '// &
                    stable_report) /= 0) then
        print *, 'FAIL: repeated run of a stable case did not exit 0'
        all_passed = .false.
    end if
    if (.not. file_contains(stable_report, '"file":"'//CASE_FILE// &
                            '","status":"PASS"')) all_passed = .false.
    if (file_contains_quiet(stable_report, '"status":"FLAKY"')) then
        print *, 'FAIL: stable case reported as FLAKY'
        all_passed = .false.
    end if

    ! Negative control: a malformed repeat count is rejected.
    if (run_command('bash '//SCRIPT//' --suite fortfront-f90 --repeat zero'// &
                    ' > /dev/null 2>&1') == 0) then
        print *, 'FAIL: --repeat zero was accepted'
        all_passed = .false.
    end if

    if (all_passed) call remove_temp_root(root)

    if (all_passed) then
        print *, 'PASS: repeated runs record unstable cases as FLAKY'
    else
        print *, 'FAIL: flake detection test failed (scratch kept: ', root, ')'
        stop 1
    end if

contains

    function repeated_run(report, wrapper_path, attempts) result(command)
        character(len=*), intent(in) :: report, wrapper_path
        integer, intent(in) :: attempts
        character(len=:), allocatable :: command
        character(len=16) :: attempt_text

        write (attempt_text, '(i0)') attempts
        command = 'export FLAKE_COUNTER='//counter//'; echo 0 > $FLAKE_COUNTER; '// &
                  'export FLAKE_REAL_FFC=$(PROJECT_DIR=$PWD bash -c '// &
                  '". scripts/lib_conformance.sh; find_ffc"); '// &
                  'test -x "$FLAKE_REAL_FFC" || exit 2; '// &
                  'timeout 300 bash '//SCRIPT//' --suite fortfront-f90 --file '// &
                  CASE_FILE//' --ffc '//wrapper_path//' --repeat '// &
                  trim(attempt_text)//' --report '//report
    end function repeated_run

    subroutine write_alternating_wrapper(path)
        character(len=*), intent(in) :: path
        integer :: unit_number, exit_status

        open (newunit=unit_number, file=path, status='replace', action='write')
        write (unit_number, '(a)') '#!/usr/bin/env bash'
        write (unit_number, '(a)') '# Compiles on even invocations, fails on odd ones.'
        write (unit_number, '(a)') 'n=$(cat "$FLAKE_COUNTER" 2>/dev/null || echo 0)'
        write (unit_number, '(a)') 'echo $((n + 1)) > "$FLAKE_COUNTER"'
        write (unit_number, '(a)') 'if [ $((n % 2)) -eq 1 ]; then'
        write (unit_number, '(a)') '    echo "alternating wrapper: failure" >&2'
        write (unit_number, '(a)') '    exit 1'
        write (unit_number, '(a)') 'fi'
        write (unit_number, '(a)') 'exec "$FLAKE_REAL_FFC" "$@"'
        close (unit_number)
        call execute_command_line('chmod +x '//path, exitstat=exit_status)
        if (exit_status /= 0) then
            print *, 'FAIL: could not make the wrapper executable'
            stop 1
        end if
    end subroutine write_alternating_wrapper

    integer function run_command(command) result(exit_status)
        character(len=*), intent(in) :: command
        integer :: command_status

        call execute_command_line(command, exitstat=exit_status, &
                                  cmdstat=command_status)
        if (command_status /= 0) exit_status = -1
    end function run_command

    logical function file_contains(path, needle) result(found)
        character(len=*), intent(in) :: path, needle

        found = file_contains_quiet(path, needle)
        if (.not. found) then
            print *, 'FAIL: ', trim(path), ' lacks ', trim(needle)
        end if
    end function file_contains

    logical function file_contains_quiet(path, needle) result(found)
        character(len=*), intent(in) :: path, needle
        integer :: unit_number, io_status
        character(len=4096) :: line
        logical :: exists

        found = .false.
        inquire (file=path, exist=exists)
        if (.not. exists) return
        open (newunit=unit_number, file=path, status='old', action='read')
        do
            read (unit_number, '(a)', iostat=io_status) line
            if (io_status /= 0) exit
            if (index(line, needle) > 0) then
                found = .true.
                exit
            end if
        end do
        close (unit_number)
    end function file_contains_quiet

end program test_conformance_flake_detection
