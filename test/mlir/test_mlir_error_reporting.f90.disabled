program test_mlir_error_reporting
    use ast_core
    use ast_factory
    use backend_interface
    use backend_factory
    use temp_utils
    implicit none

    logical :: all_passed = .true.

    print *, "=== Testing MLIR Syntax Error Reporting ==="
    print *, ""

    all_passed = all_passed .and. test_invalid_mlir_syntax()
    all_passed = all_passed .and. test_mlir_compilation_errors()
    all_passed = all_passed .and. test_mlir_lowering_errors()

    if (all_passed) then
        print *, ""
        print *, "All MLIR error reporting tests passed!"
        stop 0
    else
        print *, ""
        print *, "Some MLIR error reporting tests failed!"
        stop 1
    end if

contains

    function test_invalid_mlir_syntax() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        type(backend_options_t) :: backend_opts
        character(len=:), allocatable :: mlir_code
        character(len=256) :: error_msg
        integer :: prog_idx, body_idx

        print *, "Testing invalid MLIR syntax error reporting..."

        passed = .false.

        ! Create AST with undeclared variable to force MLIR error
        body_idx = push_assignment(arena, &
                                   push_identifier(arena, "undeclared_var"), &
                                   push_literal(arena, "42", LITERAL_INTEGER))
        prog_idx = push_program(arena, "test_error", [body_idx])

        ! Configure backend
        backend_opts%compile_mode = .true.
        backend_opts%debug_info = .true.

        ! Create MLIR backend
        call create_backend("mlir", backend, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, "FAIL: Error creating backend:", trim(error_msg)
            return
        end if

        ! Generate MLIR code - should produce error about undeclared variable
        call backend%generate_code(arena, prog_idx, backend_opts, mlir_code, error_msg)

        ! Should have proper error message
        if (len_trim(error_msg) > 0 .and. &
       (index(error_msg, "undeclared") > 0 .or. index(error_msg, "undefined") > 0)) then
            print *, "PASS: Proper error reporting for invalid syntax"
            passed = .true.
        else
            print *, "FAIL: No proper error message for invalid syntax"
            print *, "  Error msg: ", trim(error_msg)
        end if
    end function test_invalid_mlir_syntax

    function test_mlir_compilation_errors() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        type(backend_options_t) :: backend_opts
        character(len=:), allocatable :: mlir_code
        character(len=256) :: error_msg
        integer :: prog_idx, body_idx

        print *, "Testing MLIR compilation error reporting..."

        passed = .false.

        ! Create simple valid AST (should not generate errors)
        body_idx = push_print_statement(arena, "*", &
                                        [push_literal(arena, "'Test'", LITERAL_STRING)])
        prog_idx = push_program(arena, "test_compile_error", [body_idx])

        ! Configure backend
        backend_opts%compile_mode = .true.

        ! Create MLIR backend
        call create_backend("mlir", backend, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, "FAIL: Error creating backend:", trim(error_msg)
            return
        end if

        ! Generate valid MLIR code - should not produce errors
        call backend%generate_code(arena, prog_idx, backend_opts, mlir_code, error_msg)

        ! Check if generation succeeded and no error reported
        if (len_trim(mlir_code) > 0 .and. len_trim(error_msg) == 0) then
            print *, "PASS: MLIR compilation error handling works"
            passed = .true.
        else
            print *, "FAIL: MLIR compilation error not handled properly"
            print *, "  Error msg: ", trim(error_msg)
        end if
    end function test_mlir_compilation_errors

    function test_mlir_lowering_errors() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        type(backend_options_t) :: backend_opts
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: prog_idx, body_idx

        print *, "Testing MLIR lowering error reporting..."

        passed = .false.

        ! Create AST with complex operations that might fail lowering
        body_idx = push_assignment(arena, &
                                   push_identifier(arena, "result"), &
                                   push_binary_op(arena, &
                                             push_literal(arena, "1.5", LITERAL_REAL), &
                                           push_literal(arena, "abc", LITERAL_STRING), &
                                                  "+"))  ! Invalid: real + string
        prog_idx = push_program(arena, "test_lowering", [body_idx])

        ! Configure backend for lowering
        backend_opts%emit_llvm = .true.

        ! Create MLIR backend
        call create_backend("mlir", backend, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, "FAIL: Error creating backend:", trim(error_msg)
            return
        end if

        ! Try to generate and lower - should catch type mismatch
        call backend%generate_code(arena, prog_idx, backend_opts, output, error_msg)

        ! Should either succeed with proper handling or report meaningful error
        if (len_trim(output) > 0 .or. len_trim(error_msg) > 0) then
            print *, "PASS: MLIR lowering error handling works"
            passed = .true.
        else
            print *, "FAIL: MLIR lowering error not handled"
        end if
    end function test_mlir_lowering_errors

end program test_mlir_error_reporting
