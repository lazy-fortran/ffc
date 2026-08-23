program test_conformance_shard_merge
    implicit none

    integer :: exit_status

    call execute_command_line( &
        'timeout 120 python3 test/conformance_shard_merge_oracle.py', &
        exitstat=exit_status)
    if (exit_status /= 0) then
        print *, 'FAIL: shard-aware observation merge oracle'
        stop 1
    end if

    print *, 'PASS: shard merge reconstructs one full observation epoch'
end program test_conformance_shard_merge
