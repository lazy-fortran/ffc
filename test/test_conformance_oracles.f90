! fo: dispatcher
module ffc_case_test_conformance_oracles
    implicit none
    private
    public :: case_test_conformance_oracles
    interface
        subroutine case_test_conformance_oracles()
        end subroutine case_test_conformance_oracles
    end interface
end module ffc_case_test_conformance_oracles

subroutine case_test_conformance_oracles()
    implicit none
    save
    integer :: status

    call execute_command_line('python3 test/test_conformance_oracles.py', &
                              exitstat=status)
    if (status /= 0) stop 1
end subroutine case_test_conformance_oracles
