program test_unknown_module_lookup
    use ast_core
    use ast_factory
    use backend_interface
    use backend_factory
    use temp_utils
    implicit none

    logical :: all_passed = .true.

    print *, "=== Testing Unknown Module Symbol Lookup ==="
    print *, ""

    all_passed = all_passed .and. test_unknown_module_use_statement()
    all_passed = all_passed .and. test_unknown_module_compilation()

    if (all_passed) then
        print *, ""
        print *, "All unknown module lookup tests passed!"
        stop 0
    else
        print *, ""
        print *, "Some unknown module lookup tests failed!"
        stop 1
    end if

contains

    function test_unknown_module_use_statement() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        type(backend_options_t) :: backend_opts
        character(len=:), allocatable :: mlir_code
        character(len=256) :: error_msg
        integer :: prog_idx, use_idx

        print *, "Testing unknown module use statement handling..."

        passed = .false.

        ! Create AST with use statement for unknown module
        use_idx = push_use_statement(arena, "unknown_library", &
                                     only_list=["some_function"], &
                                     has_only=.true.)
        prog_idx = push_program(arena, "test_unknown", [use_idx])

        ! Configure backend for compile mode
        backend_opts%compile_mode = .true.

        ! Create MLIR backend
        call create_backend("mlir", backend, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, "FAIL: Error creating backend:", trim(error_msg)
            return
        end if

        ! Generate MLIR code
        call backend%generate_code(arena, prog_idx, backend_opts, mlir_code, error_msg)

        ! Check if code contains the placeholder comment (current limitation)
        if (index(mlir_code, "Module symbols require symbol table lookup") > 0) then
            print *, "FAIL: Unknown module generates placeholder comment"
            print *, "  Expected: Proper symbol resolution or error handling"
            print *, "  Got: Placeholder comment indicating unimplemented feature"
            passed = .false.
        else if (index(mlir_code, "func.func private @unknown_library") > 0) then
            print *, "PASS: Unknown module generates proper symbol declarations"
            passed = .true.
        else
            print *, "FAIL: Unknown module handling unclear"
            print *, "  MLIR output: ", trim(mlir_code)
        end if
    end function test_unknown_module_use_statement

    function test_unknown_module_compilation() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        type(backend_options_t) :: backend_opts
        character(len=:), allocatable :: mlir_code
        character(len=256) :: error_msg
        integer :: prog_idx, use_idx

        print *, "Testing MLIR generation with unknown module (wildcard import)..."

        passed = .false.

        ! Create AST with wildcard use statement (no "only" clause)
        use_idx = push_use_statement(arena, "my_custom_library")
        prog_idx = push_program(arena, "test_wildcard", [use_idx])

        ! Configure backend for compile mode
        backend_opts%compile_mode = .true.

        ! Create MLIR backend
        call create_backend("mlir", backend, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, "FAIL: Error creating backend:", trim(error_msg)
            return
        end if

        ! Generate MLIR code
        call backend%generate_code(arena, prog_idx, backend_opts, mlir_code, error_msg)

        ! Check current behavior (should generate placeholder comment)
     if (index(mlir_code, "Import all public symbols from @my_custom_library") > 0) then
            print *, "CURRENT BEHAVIOR: Wildcard import generates placeholder comment"
            print *, "LIMITATION: No actual symbol table lookup implemented"
            passed = .false.  ! This demonstrates the current limitation
        else if (len_trim(mlir_code) > 0) then
            print *, "UNKNOWN BEHAVIOR: Generated some MLIR but not expected comment"
            print *, "MLIR: ", trim(mlir_code)
            passed = .false.
        else
            print *, "FAIL: No MLIR generated at all"
            passed = .false.
        end if
    end function test_unknown_module_compilation

end program test_unknown_module_lookup
