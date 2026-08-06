program test_conformance_include_snapshot
    use conformance_temp_dir, only: make_temp_root, remove_temp_root
    implicit none

    character(len=*), parameter :: GAUNTLET = &
        'scripts/conformance_gauntlet.sh'
    character(len=:), allocatable :: root, fortfront, source, outer_include
    character(len=:), allocatable :: inner_include, compiler, fake_gfortran
    character(len=:), allocatable :: fake_program, fake_bin, seen_source
    character(len=:), allocatable :: xfail_manifest, skip_manifest
    character(len=:), allocatable :: noref_manifest, observations, report
    character(len=:), allocatable :: log_file, expected_manifest, command
    character(len=:), allocatable :: source_sha, closure_sha
    integer :: exit_status
    logical :: passed

    root = make_temp_root('conformance_include_snapshot')
    fortfront = root//'/fortfront'
    source = fortfront//'/examples/f90/include_snapshot.f90'
    outer_include = fortfront//'/examples/f90/nested/outer.inc'
    inner_include = fortfront//'/examples/f90/nested/inner.inc'
    compiler = root//'/instrumented_ffc.sh'
    fake_bin = root//'/bin'
    fake_gfortran = fake_bin//'/gfortran'
    fake_program = root//'/program.sh'
    seen_source = root//'/seen_source.txt'
    xfail_manifest = root//'/xfail.txt'
    skip_manifest = root//'/skip.txt'
    noref_manifest = root//'/noref.txt'
    observations = root//'/observations.jsonl'
    report = root//'/report.jsonl'
    log_file = root//'/run.log'
    expected_manifest = root//'/expected_closure.tsv'
    passed = .true.

    call write_fixture()
    source_sha = sha256_of(source)
    call write_expected_manifest()
    closure_sha = sha256_of(expected_manifest)
    call write_fake_program()
    call write_compiler(compiler, .false.)
    call write_compiler(fake_gfortran, .true.)
    call write_empty(xfail_manifest)
    call write_empty(skip_manifest)
    call write_empty(noref_manifest)

    command = 'PATH='//fake_bin//'\:$PATH'// &
        ' SNAP_ORIGINAL_SOURCE='//source// &
        ' SNAP_ORIGINAL_OUTER='//outer_include// &
        ' SNAP_ORIGINAL_INNER='//inner_include// &
        ' SNAP_FAKE_PROGRAM='//fake_program// &
        ' SNAP_SEEN_SOURCE='//seen_source// &
        ' FFC_FORTFRONT_DIR='//fortfront// &
        ' FFC_XFAIL_MANIFEST='//xfail_manifest// &
        ' FFC_SKIP_MANIFEST='//skip_manifest// &
        ' FFC_NOREF_MANIFEST='//noref_manifest// &
        ' timeout 60 bash '//GAUNTLET// &
        ' --suite fortfront-f90 --file include_snapshot.f90'// &
        ' --ffc '//compiler//' --observations '//observations// &
        ' --report '//report//' > '//log_file//' 2>&1'
    call execute_command_line(command, exitstat=exit_status)

    if (exit_status /= 0) then
        print *, 'FAIL: snapshot gauntlet exited ', exit_status
        passed = .false.
    end if
    if (.not. file_contains(observations, &
        '"file":"include_snapshot.f90","status":"PASS"')) then
        print *, 'FAIL: snapshot fixture did not pass'
        passed = .false.
    end if
    if (.not. file_contains(observations, &
        '"source_sha256":"'//source_sha//'"')) then
        print *, 'FAIL: source digest does not describe the copied input'
        passed = .false.
    end if
    if (.not. file_contains(observations, &
        '"dependency_closure_sha256":"'//closure_sha//'"')) then
        print *, 'FAIL: closure digest omits canonical nested INCLUDE inputs'
        passed = .false.
    end if
    if (.not. file_contains(outer_include, 'poisoned outer') .or. &
        .not. file_contains(inner_include, 'poisoned inner')) then
        print *, 'FAIL: compiler fixture did not mutate the original closure'
        passed = .false.
    end if
    if (.not. file_contains(seen_source, '/source_snapshot_1/suite/'// &
        'include_snapshot.f90') .or. file_contains(seen_source, source)) then
        print *, 'FAIL: compiler did not receive the immutable source snapshot'
        passed = .false.
    end if

    if (passed) call remove_temp_root(root)
    if (.not. passed) then
        print *, 'FAIL: INCLUDE snapshot scratch kept at ', root
        stop 1
    end if
    print *, 'PASS: INCLUDE closure hash and compiler input use one snapshot'

contains

    subroutine write_fixture()
        integer :: unit_number

        call execute_command_line('mkdir -p '//fortfront// &
            '/examples/f90/nested '//fake_bin)
        open(newunit=unit_number, file=source, status='replace', action='write')
        write(unit_number, '(a)') 'program include_snapshot'
        write(unit_number, '(a)') 'integer :: value'
        write(unit_number, '(a)') "include 'nested/outer.inc'"
        write(unit_number, '(a)') 'print *, value'
        write(unit_number, '(a)') 'end program include_snapshot'
        close(unit_number)
        open(newunit=unit_number, file=outer_include, status='replace', &
            action='write')
        write(unit_number, '(a)') "include 'inner.inc'"
        close(unit_number)
        open(newunit=unit_number, file=inner_include, status='replace', &
            action='write')
        write(unit_number, '(a)') 'value = 42'
        close(unit_number)
    end subroutine write_fixture

    subroutine write_expected_manifest()
        integer :: unit_number

        open(newunit=unit_number, file=expected_manifest, status='replace', &
            action='write')
        write(unit_number, '(a)') 'suite:include_snapshot.f90'//achar(9)// &
            sha256_of(source)
        write(unit_number, '(a)') 'suite:nested/inner.inc'//achar(9)// &
            sha256_of(inner_include)
        write(unit_number, '(a)') 'suite:nested/outer.inc'//achar(9)// &
            sha256_of(outer_include)
        close(unit_number)
    end subroutine write_expected_manifest

    subroutine write_fake_program()
        integer :: unit_number

        open(newunit=unit_number, file=fake_program, status='replace', &
            action='write')
        write(unit_number, '(a)') '#!/usr/bin/env bash'
        write(unit_number, '(a)') 'printf "snapshot-pass\n"'
        close(unit_number)
        call make_executable(fake_program)
    end subroutine write_fake_program

    subroutine write_compiler(path, is_reference)
        character(len=*), intent(in) :: path
        logical, intent(in) :: is_reference
        integer :: unit_number

        open(newunit=unit_number, file=path, status='replace', action='write')
        write(unit_number, '(a)') '#!/usr/bin/env bash'
        write(unit_number, '(a)') 'set -uo pipefail'
        if (is_reference) then
            write(unit_number, '(a)') &
                'if [ "${1:-}" = "--version" ]; then echo "GNU Fortran fake"; exit; fi'
            write(unit_number, '(a)') &
                'if [ "${1:-}" = "-dumpmachine" ]; then echo "fake-linux"; exit; fi'
        end if
        write(unit_number, '(a)') 'source_path=""; output=""'
        write(unit_number, '(a)') 'while [ "$#" -gt 0 ]; do'
        write(unit_number, '(a)') '    case "$1" in'
        write(unit_number, '(a)') '        -o) output="$2"; shift 2 ;;'
        write(unit_number, '(a)') '        *.f90) source_path="$1"; shift ;;'
        write(unit_number, '(a)') '        *) shift ;;'
        write(unit_number, '(a)') '    esac'
        write(unit_number, '(a)') 'done'
        write(unit_number, '(a)') '[ -n "$source_path" ] && [ -n "$output" ]'
        write(unit_number, '(a)') &
            'printf "! poisoned outer\n" > "$SNAP_ORIGINAL_OUTER"'
        write(unit_number, '(a)') &
            'printf "! poisoned inner\n" > "$SNAP_ORIGINAL_INNER"'
        write(unit_number, '(a)') 'test "$source_path" != "$SNAP_ORIGINAL_SOURCE"'
        write(unit_number, '(a)') &
            'grep -Fq "include ''nested/outer.inc''" "$source_path"'
        write(unit_number, '(a)') &
            'grep -Fq "include ''inner.inc''" "$(dirname "$source_path")/nested/outer.inc"'
        write(unit_number, '(a)') &
            'grep -Fq "value = 42" "$(dirname "$source_path")/nested/inner.inc"'
        write(unit_number, '(a)') 'printf "%s\n" "$source_path" > "$SNAP_SEEN_SOURCE"'
        write(unit_number, '(a)') 'cp "$SNAP_FAKE_PROGRAM" "$output"'
        write(unit_number, '(a)') 'chmod +x "$output"'
        close(unit_number)
        call make_executable(path)
    end subroutine write_compiler

    subroutine write_empty(path)
        character(len=*), intent(in) :: path
        integer :: unit_number

        open(newunit=unit_number, file=path, status='replace', action='write')
        close(unit_number)
    end subroutine write_empty

    subroutine make_executable(path)
        character(len=*), intent(in) :: path
        integer :: status

        call execute_command_line('chmod +x '//path, exitstat=status)
        if (status /= 0) stop 2
    end subroutine make_executable

    function sha256_of(path) result(digest)
        character(len=*), intent(in) :: path
        character(len=:), allocatable :: digest
        character(len=:), allocatable :: digest_file
        character(len=64) :: buffer
        integer :: io_status, status, unit_number

        digest_file = root//'/digest.txt'
        call execute_command_line('sha256sum '//path//' > '//digest_file, &
            exitstat=status)
        if (status /= 0) then
            digest = ''
            return
        end if
        open(newunit=unit_number, file=digest_file, status='old', &
            action='read', iostat=io_status)
        if (io_status /= 0) then
            digest = ''
            return
        end if
        read(unit_number, '(a64)', iostat=io_status) buffer
        close(unit_number)
        if (io_status == 0) then
            digest = buffer
        else
            digest = ''
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

end program test_conformance_include_snapshot
