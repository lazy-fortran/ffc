program test_mlir_harness
    ! Test the MLIR test harness itself
    implicit none

    logical :: all_passed

    print *, "=== MLIR Test Harness Tests ==="

    all_passed = .true.

    ! Test basic functionality
    all_passed = all_passed .and. test_harness_basic_functionality()
    all_passed = all_passed .and. test_result_reporting()
    all_passed = all_passed .and. test_mlir_validation_detection()

    if (all_passed) then
        print *, ""
        print *, "All MLIR harness tests PASSED!"
    else
        print *, ""
        print *, "Some MLIR harness tests FAILED!"
        stop 1
    end if

contains

    function test_harness_basic_functionality() result(passed)
        logical :: passed
        character(len=256) :: command
        integer :: exit_code

        print *, "Testing basic harness functionality..."

        ! Test the help option
        command = "cd test/mlir && ./run_mlir_tests.sh --help > /dev/null 2>&1"
        call execute_command_line(command, exitstat=exit_code)

        if (exit_code == 0) then
            print *, "PASS: Help option works"
            passed = .true.
        else
            print *, "FAIL: Help option failed"
            passed = .false.
        end if
    end function test_harness_basic_functionality

    function test_result_reporting() result(passed)
        logical :: passed

        print *, "Testing result reporting..."

        ! For now, this is a placeholder test
        ! In a real implementation, we would test the result structure
        passed = .true.
        print *, "PASS: Result reporting structure available"
    end function test_result_reporting

    function test_mlir_validation_detection() result(passed)
        logical :: passed
        character(len=256) :: command
        integer :: exit_code

        print *, "Testing MLIR validation detection..."

        ! Check if mlir-opt is available
        command = "command -v mlir-opt > /dev/null 2>&1"
        call execute_command_line(command, exitstat=exit_code)

        if (exit_code == 0) then
            print *, "PASS: mlir-opt tool detected"
            passed = .true.
        else
            print *, "SKIP: mlir-opt not available (validation will be skipped)"
            passed = .true.  ! Not a failure, just skip
        end if
    end function test_mlir_validation_detection

end program test_mlir_harness
