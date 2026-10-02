! fo: dispatcher
module ffc_case_test_conformance_shard_merge
    implicit none
    private
    public :: case_test_conformance_shard_merge
    interface
        subroutine case_test_conformance_shard_merge()
        end subroutine case_test_conformance_shard_merge
    end interface
end module ffc_case_test_conformance_shard_merge

subroutine case_test_conformance_shard_merge()
    implicit none
    save

    integer :: exit_status

    call execute_command_line( &
        'timeout 120 python3 test/conformance_shard_merge_oracle.py', &
        exitstat=exit_status)
    if (exit_status /= 0) then
        print *, 'FAIL: shard-aware observation merge oracle'
        stop 1
    end if

    print *, 'PASS: shard merge reconstructs one full observation epoch'
end subroutine case_test_conformance_shard_merge
