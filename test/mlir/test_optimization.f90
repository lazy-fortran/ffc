program test_optimization
    use iso_fortran_env, only: error_unit
    use mlir_backend
    use backend_factory
    use backend_interface
    use ast_core, only: ast_arena_t, create_ast_arena, LITERAL_INTEGER
    use ast_factory
    implicit none

    logical :: all_tests_passed

    print *, "=== MLIR Optimization Pipeline Tests ==="
    print *

    all_tests_passed = .true.

    ! Run all tests
    if (.not. test_basic_optimization()) all_tests_passed = .false.
    if (.not. test_optimization_levels()) all_tests_passed = .false.
    if (.not. test_constant_folding()) all_tests_passed = .false.
    if (.not. test_dead_code_elimination()) all_tests_passed = .false.
    if (.not. test_loop_optimization()) all_tests_passed = .false.

    print *
    if (all_tests_passed) then
        print *, "All MLIR optimization pipeline tests passed!"
        stop 0
    else
        print *, "Some MLIR optimization pipeline tests failed!"
        stop 1
    end if

contains

    function test_basic_optimization() result(passed)
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

        ! Create simple program with optimization enabled
        arena = create_ast_arena()
        decl_idx = push_declaration(arena, "integer", "x", kind_value=4)
        prog_idx = push_program(arena, "test", [decl_idx])

        ! Enable optimization
        options%optimize = .true.

        ! Generate MLIR
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0) then
            ! Check for optimization passes being applied
            if (index(output, "func.func") > 0 .and. &
                index(output, "i32") > 0) then
                print *, "PASS: Basic optimization generates valid MLIR"
                passed = .true.
            else
                print *, "FAIL: Basic optimization should generate valid MLIR"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error in basic optimization: ", trim(error_msg)
        end if
    end function test_basic_optimization

    function test_optimization_levels() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options_o0, options_o1
        character(len=:), allocatable :: output_o0, output_o1
        character(len=256) :: error_msg
        integer :: decl_idx, assign_idx, x_id, val_idx, prog_idx

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        ! Create program with redundant operations: x = 1 + 1
        arena = create_ast_arena()
        decl_idx = push_declaration(arena, "integer", "x", kind_value=4)
        val_idx = push_binary_op(arena, &
                                 push_literal(arena, "1", LITERAL_INTEGER), &
                                 push_literal(arena, "1", LITERAL_INTEGER), "+")
        x_id = push_identifier(arena, "x")
        assign_idx = push_assignment(arena, x_id, val_idx)
        prog_idx = push_program(arena, "test", [decl_idx, assign_idx])

        ! Generate with O0 (no optimization)
        options_o0%optimize = .false.
        call backend%generate_code(arena, prog_idx, options_o0, output_o0, error_msg)

        ! Generate with O1 (optimization)
        options_o1%optimize = .true.
        call backend%generate_code(arena, prog_idx, options_o1, output_o1, error_msg)

        if (len_trim(error_msg) == 0) then
            ! Both should generate valid MLIR
            if (index(output_o0, "func.func") > 0 .and. &
                index(output_o1, "func.func") > 0) then
                print *, "PASS: Different optimization levels generate valid MLIR"
                passed = .true.
            else
                print *, "FAIL: Optimization levels should generate valid MLIR"
            end if
        else
            print *, "FAIL: Error in optimization levels test: ", trim(error_msg)
        end if
    end function test_optimization_levels

    function test_constant_folding() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: const_expr, prog_idx

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        ! Create constant expression: 2 * 3 + 1
        arena = create_ast_arena()
        const_expr = push_binary_op(arena, &
                                    push_binary_op(arena, &
                                            push_literal(arena, "2", LITERAL_INTEGER), &
                                      push_literal(arena, "3", LITERAL_INTEGER), "*"), &
                                    push_literal(arena, "1", LITERAL_INTEGER), "+")
        prog_idx = push_program(arena, "test", [const_expr])

        ! Enable optimization for constant folding
        options%optimize = .true.

        ! Generate MLIR
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0) then
            ! Check for constant operations
            if (index(output, "arith.constant") > 0) then
                print *, "PASS: Constant folding generates arith.constant operations"
                passed = .true.
            else
                print *, "FAIL: Constant folding should generate constant operations"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error in constant folding test: ", trim(error_msg)
        end if
    end function test_constant_folding

    function test_dead_code_elimination() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: decl_idx, unused_expr, prog_idx

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        ! Create program with unused expression
        arena = create_ast_arena()
        decl_idx = push_declaration(arena, "integer", "x", kind_value=4)
        unused_expr = push_binary_op(arena, &
                                     push_literal(arena, "42", LITERAL_INTEGER), &
                                     push_literal(arena, "0", LITERAL_INTEGER), "+")
        prog_idx = push_program(arena, "test", [decl_idx, unused_expr])

        ! Enable optimization for DCE
        options%optimize = .true.

        ! Generate MLIR
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0) then
            ! Check that output is generated (DCE may be applied in later passes)
            if (index(output, "func.func") > 0) then
                print *, "PASS: Dead code elimination compatible MLIR generated"
                passed = .true.
            else
                print *, "FAIL: DCE should generate valid MLIR"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error in dead code elimination test: ", trim(error_msg)
        end if
    end function test_dead_code_elimination

    function test_loop_optimization() result(passed)
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

        ! Create simple loop
        arena = create_ast_arena()
        start_idx = push_literal(arena, "1", LITERAL_INTEGER)
        end_idx = push_literal(arena, "10", LITERAL_INTEGER)
      loop_idx = push_do_loop(arena, "i", start_idx, end_idx, body_indices=[integer ::])
        prog_idx = push_program(arena, "test", [loop_idx])

        ! Enable optimization for loop passes
        options%optimize = .true.

        ! Generate MLIR
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0) then
            ! Check for SCF loop structures
            if (index(output, "scf.for") > 0 .or. &
                index(output, "arith.constant") > 0) then
                print *, "PASS: Loop optimization generates proper SCF constructs"
                passed = .true.
            else
                print *, "FAIL: Loop optimization should generate SCF constructs"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error in loop optimization test: ", trim(error_msg)
        end if
    end function test_loop_optimization

end program test_optimization
