program test_conformance_gauntlet_smoke
    implicit none

    integer, parameter :: RUN_TIMEOUT_SECONDS = 120
    character(len=*), parameter :: SCRIPT = &
        'scripts/conformance_gauntlet.sh'
    character(len=*), parameter :: REPORT = &
        '/tmp/ffc_gauntlet_smoke.jsonl'

    integer :: exit_stat
    integer :: unit
    integer :: io_stat
    character(len=4096) :: line
    character(len=:), allocatable :: cmd
    logical :: has_summary
    logical :: has_zero_fail_summary
    logical :: report_exists

    print *, '=== conformance gauntlet smoke test ==='

    ! Build the command string at runtime
    cmd = 'timeout 120 bash ' // SCRIPT // &
          ' --suite fortfront-f90 --max-files 20' // &
          ' --report ' // REPORT

    call execute_command_line(cmd, exitstat=exit_stat)

    if (exit_stat > 126) then
        print *, 'FAIL: gauntlet timed out or killed'
        stop 1
    end if

    if (exit_stat /= 0) then
        print *, 'FAIL: gauntlet exited with ', exit_stat
        stop 1
    end if

    inquire(file=REPORT, exist=report_exists)
    if (.not. report_exists) then
        print *, 'FAIL: no report file at ', trim(adjustl(REPORT))
        stop 1
    end if

    has_summary = .false.
    has_zero_fail_summary = .false.
    open(newunit=unit, file=REPORT, status='old', action='read', &
         iostat=io_stat)
    if (io_stat /= 0) then
        print *, 'FAIL: cannot open report ', trim(adjustl(REPORT))
        stop 1
    end if

    do
        read(unit, '(A)', iostat=io_stat) line
        if (io_stat /= 0) exit
        if (is_blank_or_comment(line)) cycle
        if (index(line, '"status":"SUMMARY"') > 0) then
            has_summary = .true.
            if (index(line, '"fail":0') > 0) then
                has_zero_fail_summary = .true.
            end if
            exit
        end if
    end do
    close(unit)

    if (.not. has_summary) then
        print *, 'FAIL: no SUMMARY record in report'
        stop 1
    end if

    if (.not. has_zero_fail_summary) then
        print *, 'FAIL: SUMMARY record has nonzero fail count'
        stop 1
    end if

    print *, 'PASS: gauntlet smoke test completed with zero FAIL records'

contains

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
