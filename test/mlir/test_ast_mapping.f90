program test_ast_mapping
    use iso_fortran_env, only: error_unit
    use mlir_backend
    use backend_factory
    use backend_interface
    use fortfront, only: ast_arena_t, create_ast_arena, LITERAL_INTEGER, LITERAL_REAL, LITERAL_STRING
    use ast_factory
    implicit none

    logical :: all_tests_passed

    print *, "=== AST to FIR Mapping Tests ==="
    print *

    all_tests_passed = .true.

    ! Run all tests
    if (.not. test_variable_declaration()) all_tests_passed = .false.
    if (.not. test_arithmetic_operations()) all_tests_passed = .false.
    if (.not. test_loop_generation()) all_tests_passed = .false.
    if (.not. test_control_flow()) all_tests_passed = .false.
    if (.not. test_function_calls()) all_tests_passed = .false.
    if (.not. test_program_structure()) all_tests_passed = .false.

    print *
    if (all_tests_passed) then
        print *, "All AST to FIR mapping tests passed!"
        stop 0
    else
        print *, "Some AST to FIR mapping tests failed!"
        stop 1
    end if

contains

    function test_variable_declaration() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: decl_idx, prog_idx

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        ! Create variable declaration AST: integer :: x
        arena = create_ast_arena()
        decl_idx = push_declaration(arena, "integer", ["x"], kind_value=4)
        prog_idx = push_program(arena, "test", [decl_idx])

        ! Generate MLIR
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0) then
            ! Check for HLFIR declaration or memref allocation
            if (index(output, "hlfir.declare") > 0 .or. &
                index(output, "memref.alloca") > 0 .or. &
                index(output, "i32") > 0) then
                print *, "PASS: Variable declaration generates HLFIR/memref allocation"
                passed = .true.
            else
                print *, "FAIL: No HLFIR declaration found for variable declaration"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating variable declaration: ", trim(error_msg)
        end if
    end function test_variable_declaration

    function test_arithmetic_operations() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: x_idx, y_idx, z_idx, add_idx, assign_idx, prog_idx

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        ! Create AST: z = x + y
        arena = create_ast_arena()
        x_idx = push_identifier(arena, "x")
        y_idx = push_identifier(arena, "y")
        add_idx = push_binary_op(arena, x_idx, y_idx, "+")
        z_idx = push_identifier(arena, "z")
        assign_idx = push_assignment(arena, z_idx, add_idx)
        prog_idx = push_program(arena, "test", [assign_idx])

        ! Generate MLIR
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0) then
            ! Check for arithmetic operations
            if (index(output, "arith.addi") > 0 .or. &
                index(output, "arith.addf") > 0 .or. &
                index(output, "memref.load") > 0 .or. &
                index(output, "memref.store") > 0) then
                print *, "PASS: Arithmetic operations generate memref/arith dialect"
                passed = .true.
            else
                print *, "FAIL: No arithmetic operations found"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating arithmetic operations: ", trim(error_msg)
        end if
    end function test_arithmetic_operations

    function test_loop_generation() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: start_idx, end_idx, loop_idx, prog_idx

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        ! Create AST: do i=1,10; end do
        arena = create_ast_arena()
        start_idx = push_literal(arena, "1", LITERAL_INTEGER)
        end_idx = push_literal(arena, "10", LITERAL_INTEGER)
      loop_idx = push_do_loop(arena, "i", start_idx, end_idx, body_indices=[integer ::])
        prog_idx = push_program(arena, "test", [loop_idx])

        ! Generate MLIR
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0) then
            ! Check for SCF for loop
            if (index(output, "scf.for") > 0) then
                print *, "PASS: Loop generation produces SCF loop constructs"
                passed = .true.
            else
                print *, "FAIL: No loop constructs found"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating loop: ", trim(error_msg)
        end if
    end function test_loop_generation

    function test_control_flow() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: cond_idx, if_idx, prog_idx, true_idx

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        ! Create AST: if (condition) then; end if
        arena = create_ast_arena()
        true_idx = push_literal(arena, "true", LITERAL_INTEGER)
        if_idx = push_if(arena, true_idx, then_body_indices=[integer ::])
        prog_idx = push_program(arena, "test", [if_idx])

        ! Generate MLIR
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0) then
            ! Check for SCF if
            if (index(output, "scf.if") > 0 .or. &
                index(output, "cf.cond_br") > 0) then
                print *, "PASS: Control flow generates SCF/CF constructs"
                passed = .true.
            else
                print *, "FAIL: No control flow constructs found"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating control flow: ", trim(error_msg)
        end if
    end function test_control_flow

    function test_function_calls() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: call_idx, prog_idx

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        ! Create AST: call subroutine()
        arena = create_ast_arena()
        call_idx = push_subroutine_call(arena, "test_sub", [integer ::])
        prog_idx = push_program(arena, "test", [call_idx])

        ! Generate MLIR
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0) then
            ! Check for func call
            if (index(output, "func.call") > 0) then
                print *, "PASS: Function calls generate func.call operations"
                passed = .true.
            else
                print *, "FAIL: No function call operations found"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating function call: ", trim(error_msg)
        end if
    end function test_function_calls

    function test_program_structure() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: decl_idx, assign_idx, prog_idx, x_idx, val_idx

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        ! Create complete program: program test; integer :: x; x = 42; end program
        arena = create_ast_arena()
        decl_idx = push_declaration(arena, "integer", ["x"], kind_value=4)
        x_idx = push_identifier(arena, "x")
        val_idx = push_literal(arena, "42", LITERAL_INTEGER)
        assign_idx = push_assignment(arena, x_idx, val_idx)
        prog_idx = push_program(arena, "test", [decl_idx, assign_idx])

        ! Generate MLIR
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0) then
            ! Check for complete program structure
            if (index(output, "module") > 0 .and. &
                index(output, "func.func") > 0 .and. &
                index(output, "i32") > 0) then
                print *, "PASS: Complete program generates proper MLIR structure"
                passed = .true.
            else
                print *, "FAIL: Incomplete MLIR program structure"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating complete program: ", trim(error_msg)
        end if
    end function test_program_structure

end program test_ast_mapping
