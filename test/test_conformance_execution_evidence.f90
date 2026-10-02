! fo: dispatcher
module ffc_case_test_conformance_execution_evidence
    implicit none
    private
    public :: case_test_conformance_execution_evidence
    interface
        subroutine case_test_conformance_execution_evidence()
        end subroutine case_test_conformance_execution_evidence
    end interface
end module ffc_case_test_conformance_execution_evidence

subroutine case_test_conformance_execution_evidence()
    implicit none
    save

    integer :: exit_status

    call execute_command_line( &
        'timeout 120 bash test/conformance_execution_evidence_oracle.sh', &
        exitstat=exit_status)
    if (exit_status /= 0) then
        print *, 'FAIL: compile/run execution evidence oracle'
        stop 1
    end if

    print *, 'PASS: compile/run exits and terminations remain distinct'
end subroutine case_test_conformance_execution_evidence
