! fo: dispatcher
module ffc_case_test_conformance_epoch_lock
    implicit none
    private
    public :: case_test_conformance_epoch_lock
    interface
        subroutine case_test_conformance_epoch_lock()
        end subroutine case_test_conformance_epoch_lock
    end interface
end module ffc_case_test_conformance_epoch_lock

subroutine case_test_conformance_epoch_lock()
    implicit none
    save

    integer :: exit_status

    call execute_command_line( &
        'timeout 60 python3 test/conformance_epoch_lock_oracle.py', &
        exitstat=exit_status)
    if (exit_status /= 0) then
        print *, 'FAIL: four-suite provenance lock oracle'
        stop 1
    end if

    print *, 'PASS: four-suite provenance lock rejects mixed execution inputs'
end subroutine case_test_conformance_epoch_lock
