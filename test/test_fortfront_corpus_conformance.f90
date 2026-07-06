program test_fortfront_corpus_conformance
    implicit none

    integer, parameter :: RUN_TIMEOUT_SECONDS = 180
    character(len=*), parameter :: SCRIPT = &
        'scripts/conformance_gauntlet.sh'
    character(len=*), parameter :: F90_REPORT = &
        '/tmp/ffc_fortfront_f90_corpus.jsonl'
    character(len=*), parameter :: LF_REPORT = &
        '/tmp/ffc_fortfront_lf_corpus.jsonl'
    character(len=*), parameter :: F90_LOG = &
        '/tmp/ffc_fortfront_f90_corpus.out'
    character(len=*), parameter :: LF_LOG = &
        '/tmp/ffc_fortfront_lf_corpus.out'

    integer :: failed

    print *, '=== fortfront corpus conformance test ==='

    failed = 0
    call run_suite('fortfront-f90', F90_REPORT, F90_LOG, 439, failed)
    call run_suite('fortfront-lf', LF_REPORT, LF_LOG, 264, failed)

    if (failed > 0) stop 1
    print *, 'PASS: full fortfront corpus conforms to ffc xfail manifests'

contains

    subroutine run_suite(suite, report, log_path, expected_total, failed)
        character(len=*), intent(in) :: suite
        character(len=*), intent(in) :: report
        character(len=*), intent(in) :: log_path
        integer, intent(in) :: expected_total
        integer, intent(inout) :: failed
        character(len=:), allocatable :: cmd
        character(len=32) :: timeout_text
        character(len=32) :: expected_text
        character(len=4096) :: summary
        integer :: exit_stat
        logical :: has_summary

        write (timeout_text, '(I0)') RUN_TIMEOUT_SECONDS
        call execute_command_line('rm -f '//report//' '//log_path)
        ! Generous per-file timeout so a single slow compile under full-suite
        ! load is not a false failure (idle compiles are well under a second).
        cmd = 'timeout '//trim(timeout_text)//' bash '//SCRIPT// &
            ' --suite '//suite//' --report '//report// &
            ' --timeout 30 > '//log_path//' 2>&1'
        call execute_command_line(cmd, exitstat=exit_stat)

        if (exit_stat /= 0) then
            failed = failed + 1
            print *, 'FAIL[', suite, ']: runner exit ', exit_stat
            call print_failures(report)
            call execute_command_line('tail -20 '//log_path)
            return
        end if

        call read_summary(report, summary, has_summary)
        if (.not. has_summary) then
            failed = failed + 1
            print *, 'FAIL[', suite, ']: no SUMMARY record in ', report
            return
        end if

        print *, trim(suite)//' summary: '//trim(summary)

        if (index(summary, '"fail":0') == 0) then
            failed = failed + 1
            print *, 'FAIL[', suite, ']: SUMMARY has nonzero FAIL count'
        end if

        if (index(summary, '"xpass":0') == 0) then
            call print_failures(report)
        end if

        write (expected_text, '(A,I0)') '"total":', expected_total
        if (index(summary, trim(expected_text)) == 0) then
            failed = failed + 1
            print *, 'FAIL[', suite, ']: SUMMARY total differs from ', &
                expected_total
        end if
    end subroutine run_suite

    subroutine print_failures(report)
        character(len=*), intent(in) :: report
        integer :: unit, io_stat
        character(len=4096) :: line

        open (newunit=unit, file=report, status='old', action='read', &
            iostat=io_stat)
        if (io_stat /= 0) return
        do
            read (unit, '(A)', iostat=io_stat) line
            if (io_stat /= 0) exit
            if (index(line, '"status":"FAIL"') > 0 .or. &
                index(line, '"status":"XPASS"') > 0) then
                print *, '  REPORT: '//trim(line)
            end if
        end do
        close (unit)
    end subroutine print_failures

    subroutine read_summary(report, summary, found)
        character(len=*), intent(in) :: report
        character(len=*), intent(out) :: summary
        logical, intent(out) :: found
        integer :: unit
        integer :: io_stat
        character(len=4096) :: line

        found = .false.
        summary = ''

        open (newunit=unit, file=report, status='old', action='read', &
            iostat=io_stat)
        if (io_stat /= 0) return

        do
            read (unit, '(A)', iostat=io_stat) line
            if (io_stat /= 0) exit
            if (index(line, '"status":"SUMMARY"') > 0) then
                summary = line
                found = .true.
            end if
        end do
        close (unit)
    end subroutine read_summary

end program test_fortfront_corpus_conformance
