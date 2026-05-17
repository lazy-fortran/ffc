program test_backend_factory
    use iso_fortran_env, only: error_unit
    use backend_interface
    use backend_factory
    implicit none

    logical :: all_tests_passed

    print *, "=== Backend Factory Tests ==="
    print *

    all_tests_passed = .true.

    ! Run all tests
    if (.not. test_create_fortran_backend()) all_tests_passed = .false.
    if (.not. test_create_mlir_backend()) all_tests_passed = .false.
    if (.not. test_unsupported_backend_error()) all_tests_passed = .false.
    if (.not. test_backend_selection_logic()) all_tests_passed = .false.

    print *
    if (all_tests_passed) then
        print *, "All backend factory tests passed!"
        stop 0
    else
        print *, "Some backend factory tests failed!"
        stop 1
    end if

contains

    function test_create_fortran_backend() result(passed)
        logical :: passed
        class(backend_t), allocatable :: backend
        character(len=256) :: error_msg

        passed = .false.

        ! Test creating Fortran backend
        call create_backend("fortran", backend, error_msg)

        if (allocated(backend)) then
            if (backend%get_name() == "Fortran") then
                print *, "PASS: Created Fortran backend successfully"
                passed = .true.
            else
                print *, "FAIL: Backend has wrong name"
            end if
        else
            print *, "FAIL: Failed to create Fortran backend"
            print *, "Error: ", trim(error_msg)
        end if
    end function test_create_fortran_backend

    function test_create_mlir_backend() result(passed)
        logical :: passed
        class(backend_t), allocatable :: backend
        character(len=256) :: error_msg

        passed = .false.

        ! Test creating MLIR backend
        call create_backend("mlir", backend, error_msg)

        if (allocated(backend)) then
            if (backend%get_name() == "MLIR") then
                print *, "PASS: Created MLIR backend successfully"
                passed = .true.
            else
                print *, "FAIL: MLIR backend has wrong name"
            end if
        else
            print *, "FAIL: Failed to create MLIR backend"
            if (len_trim(error_msg) > 0) then
                print *, "Error: ", trim(error_msg)
            end if
        end if
    end function test_create_mlir_backend

    function test_unsupported_backend_error() result(passed)
        logical :: passed
        class(backend_t), allocatable :: backend
        character(len=256) :: error_msg

        passed = .false.

        ! Test creating unsupported backend
        call create_backend("invalid_backend", backend, error_msg)

    if (.not. allocated(backend) .and. index(error_msg, "Unsupported backend") > 0) then
            print *, "PASS: Unsupported backend error handling works"
            passed = .true.
        else
            print *, "FAIL: Unsupported backend should produce error"
            if (len_trim(error_msg) > 0) then
                print *, "Error: ", trim(error_msg)
            end if
        end if
    end function test_unsupported_backend_error

    function test_backend_selection_logic() result(passed)
        logical :: passed
        character(len=32) :: selected_backend
        type(backend_options_t) :: options

        passed = .false.

        ! Test default backend selection
        options%compile_mode = .false.
        selected_backend = select_backend_type(options)
        if (selected_backend == "fortran") then
            print *, "PASS: Default backend selection returns 'fortran'"
            passed = .true.
        else
           print *, "FAIL: Default backend should be 'fortran', got: ", selected_backend
        end if

        ! Test compile mode backend selection
        options%compile_mode = .true.
        selected_backend = select_backend_type(options)
        if (selected_backend == "mlir") then
            print *, "PASS: Compile mode selects 'mlir' backend"
            ! Note: This is expected behavior even though MLIR isn't implemented yet
        else
            print *, "FAIL: Compile mode should select 'mlir', got: ", selected_backend
            passed = .false.
        end if
    end function test_backend_selection_logic

end program test_backend_factory
