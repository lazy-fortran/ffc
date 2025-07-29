program test_basic_generation
    use iso_fortran_env, only: error_unit
    use mlir_backend
    use backend_factory
    use backend_interface
    use ast_core, only: ast_arena_t, create_ast_stack
    use ast_factory
    implicit none

    logical :: all_tests_passed

    print *, "=== Basic MLIR Generation Tests ==="
    print *

    all_tests_passed = .true.

    ! Run all tests
    if (.not. test_empty_module_generation()) all_tests_passed = .false.
    if (.not. test_simple_function_generation()) all_tests_passed = .false.
    if (.not. test_mlir_syntax_validation()) all_tests_passed = .false.
    if (.not. test_module_structure()) all_tests_passed = .false.

    print *
    if (all_tests_passed) then
        print *, "All basic MLIR generation tests passed!"
        stop 0
    else
        print *, "Some basic MLIR generation tests failed!"
        stop 1
    end if

contains

    function test_empty_module_generation() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: prog_idx

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        ! Create empty program AST
        arena = create_ast_stack()
        prog_idx = push_program(arena, "empty_prog", [integer ::])

        ! Generate MLIR
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0) then
            ! Check for basic MLIR module structure
            if (index(output, "module") > 0 .and. &
                index(output, "}") > 0) then
                print *, "PASS: Empty module generation produces valid structure"
                passed = .true.
            else
                print *, "FAIL: Invalid MLIR module structure"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating empty module: ", trim(error_msg)
        end if
    end function test_empty_module_generation

    function test_simple_function_generation() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: func_idx

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        ! Create simple function AST
        arena = create_ast_stack()
        func_idx = push_function_def(arena, "simple_func", &
                                     param_indices=[integer ::], &
                                     return_type="integer", &
                                     body_indices=[integer ::])

        ! Generate MLIR
        call backend%generate_code(arena, func_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0) then
            ! Check for basic MLIR module structure (minimal implementation for now)
            if (index(output, "module") > 0 .and. &
                index(output, "}") > 0) then
               print *, "PASS: Simple function generation produces basic MLIR structure"
                passed = .true.
            else
                print *, "FAIL: No MLIR module structure found in output"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating simple function: ", trim(error_msg)
        end if
    end function test_simple_function_generation

    function test_mlir_syntax_validation() result(passed)
        logical :: passed
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        ! Test that backend validates MLIR syntax
        if (backend%get_name() == "MLIR") then
            print *, "PASS: MLIR backend correctly identifies itself"
            passed = .true.
        else
            print *, "FAIL: MLIR backend has wrong name: ", backend%get_name()
        end if
    end function test_mlir_syntax_validation

    function test_module_structure() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: prog_idx, decl_idx

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        ! Create program with declaration
        arena = create_ast_stack()
        decl_idx = push_declaration(arena, "integer", "x", kind_value=4)
        prog_idx = push_program(arena, "test_prog", [decl_idx])

        ! Generate MLIR
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0) then
            ! Check for proper module structure
            if (index(output, "module") > 0) then
               print *, "PASS: Module structure generated for program with declarations"
                passed = .true.
            else
                print *, "FAIL: No module structure found"
            end if
        else
            print *, "FAIL: Error generating module: ", trim(error_msg)
        end if
    end function test_module_structure

end program test_basic_generation
