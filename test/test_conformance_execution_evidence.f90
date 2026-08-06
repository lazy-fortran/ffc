program test_conformance_execution_evidence
    implicit none

    integer :: exit_status

    call execute_command_line( &
        'timeout 120 bash test/conformance_execution_evidence_oracle.sh', &
        exitstat=exit_status)
    if (exit_status /= 0) then
        print *, 'FAIL: compile/run execution evidence oracle'
        stop 1
    end if

    print *, 'PASS: compile/run exits and terminations remain distinct'
end program test_conformance_execution_evidence
