program test_module_generation
    use fortfront
    use ast_factory
    use backend_interface
    use backend_factory
    use mlir_backend
    implicit none

    logical :: all_tests_passed = .true.

    print *, "=== Testing MLIR Module Generation ==="

    if (.not. test_simple_module()) all_tests_passed = .false.
    if (.not. test_module_with_function()) all_tests_passed = .false.
    if (.not. test_module_use_statement()) all_tests_passed = .false.

    if (all_tests_passed) then
        print *, "All module generation tests passed!"
    else
        print *, "Some module generation tests failed!"
        stop 1
    end if

contains

    function test_simple_module() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        type(backend_options_t) :: options
        class(backend_t), allocatable :: backend
        character(len=:), allocatable :: mlir_code
        character(len=256) :: error_msg
        integer :: mod_idx

        passed = .false.

        ! Create AST
        arena = create_ast_arena()
        mod_idx = push_module(arena, "test_module")

        ! Create backend
        call create_backend("mlir", backend, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, "FAIL: Error creating backend:", trim(error_msg)
            return
        end if

        ! Generate MLIR in compile mode
        options%compile_mode = .true.
        call backend%generate_code(arena, mod_idx, options, mlir_code, error_msg)

        if (len_trim(error_msg) > 0) then
            print *, "FAIL: Error generating MLIR:", trim(error_msg)
            return
        end if

        ! Check that module generates proper module structure
        if (index(mlir_code, "// ======== Fortran Module: test_module") == 0) then
            print *, "FAIL: Module header not generated"
            print *, "Generated MLIR:"
            print *, trim(mlir_code)
            return
        end if

        ! For empty modules, we expect at least the module markers
        ! In a full implementation, we'd generate symbol tables or other metadata

        print *, "PASS: Simple module generation"
        passed = .true.

    end function test_simple_module

    function test_module_with_function() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        type(backend_options_t) :: options
        class(backend_t), allocatable :: backend
        character(len=:), allocatable :: mlir_code
        character(len=256) :: error_msg
        integer :: mod_idx, func_idx
        integer :: param_idx, body_indices(1), x_ref1, x_ref2, mul_expr

        passed = .false.

        ! Create AST
        arena = create_ast_arena()

        ! Add a function to the module
        ! First create parameter declaration

        param_idx = push_parameter_declaration(arena, "x", "integer", 4)

        ! Create function body - x * x
        x_ref1 = push_identifier(arena, "x")
        x_ref2 = push_identifier(arena, "x")
        mul_expr = push_binary_op(arena, x_ref1, x_ref2, "*")
        body_indices(1) = mul_expr

        ! Create function with parameter and body
     func_idx = push_function_def(arena, "square", [param_idx], "integer", body_indices)

        ! Create module with function
        mod_idx = push_module(arena, "math_module", [func_idx])

        ! Create backend
        call create_backend("mlir", backend, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, "FAIL: Error creating backend:", trim(error_msg)
            return
        end if

        ! Generate MLIR in compile mode
        options%compile_mode = .true.
        call backend%generate_code(arena, mod_idx, options, mlir_code, error_msg)

        if (len_trim(error_msg) > 0) then
            print *, "FAIL: Error generating MLIR:", trim(error_msg)
            return
        end if

        ! Debug output
        print *, "Module with function MLIR:"
        print *, trim(mlir_code)

        ! Check that module contains the function with proper namespace
        if (index(mlir_code, "func.func @math_module.square") == 0 .and. &
            index(mlir_code, "func.func @square") == 0) then
            print *, "FAIL: Module function not properly generated"
            return
        end if

        print *, "PASS: Module with function generation"
        passed = .true.

    end function test_module_with_function

    function test_module_use_statement() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        type(backend_options_t) :: options
        class(backend_t), allocatable :: backend
        character(len=:), allocatable :: mlir_code
        character(len=256) :: error_msg
        integer :: prog_idx, use_idx

        passed = .false.

        ! Create AST
        arena = create_ast_arena()

        ! Add use statement
        use_idx = push_use_statement(arena, "math_module")
        prog_idx = push_program(arena, "test_use", [use_idx])

        ! Create backend
        call create_backend("mlir", backend, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, "FAIL: Error creating backend:", trim(error_msg)
            return
        end if

        ! Generate MLIR in compile mode to get actual symbol imports
        options%compile_mode = .true.
        call backend%generate_code(arena, prog_idx, options, mlir_code, error_msg)

        if (len_trim(error_msg) > 0) then
            print *, "FAIL: Error generating MLIR:", trim(error_msg)
            return
        end if

        ! Check that use statement generates symbol imports
        if (index(mlir_code, "func.func private @math_module.square") == 0) then
            print *, "FAIL: Use statement doesn't generate symbol imports"
            print *, "Generated MLIR:"
            print *, trim(mlir_code)
            return
        end if

        print *, "PASS: Module use statement generation"
        passed = .true.

    end function test_module_use_statement

end program test_module_generation
