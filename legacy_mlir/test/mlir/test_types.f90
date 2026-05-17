program test_types
    use iso_fortran_env, only: error_unit
    use mlir_backend
    use backend_factory
    use backend_interface
    use ast_core, only: ast_arena_t, create_ast_arena, LITERAL_INTEGER, LITERAL_REAL, LITERAL_STRING
    use ast_factory
    implicit none

    logical :: all_tests_passed

    print *, "=== Type System Integration Tests ==="
    print *

    all_tests_passed = .true.

    ! Run all tests
    if (.not. test_integer_types()) all_tests_passed = .false.
    if (.not. test_real_types()) all_tests_passed = .false.
    if (.not. test_array_types()) all_tests_passed = .false.
    if (.not. test_character_types()) all_tests_passed = .false.
    if (.not. test_type_conversion()) all_tests_passed = .false.
    if (.not. test_mixed_type_expressions()) all_tests_passed = .false.

    print *
    if (all_tests_passed) then
        print *, "All type system integration tests passed!"
        stop 0
    else
        print *, "Some type system integration tests failed!"
        stop 1
    end if

contains

    function test_integer_types() result(passed)
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

        ! Test integer(kind=4) :: x
        arena = create_ast_arena()
        decl_idx = push_declaration(arena, "integer", "x", kind_value=4)
        prog_idx = push_program(arena, "test", [decl_idx])

        ! Generate MLIR
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0) then
            ! Check for i32 type
            if (index(output, "i32") > 0) then
                print *, "PASS: Integer type generates i32"
                passed = .true.
            else
                print *, "FAIL: Integer type should generate i32"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating integer type: ", trim(error_msg)
        end if
    end function test_integer_types

    function test_real_types() result(passed)
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

        ! Test real :: x
        arena = create_ast_arena()
        decl_idx = push_declaration(arena, "real", "x")
        prog_idx = push_program(arena, "test", [decl_idx])

        ! Generate MLIR
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0) then
            ! Check for f32 type
            if (index(output, "f32") > 0) then
                print *, "PASS: Real type generates f32"
                passed = .true.
            else
                print *, "FAIL: Real type should generate f32"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating real type: ", trim(error_msg)
        end if
    end function test_real_types

    function test_array_types() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: decl_idx, prog_idx, dim_idx

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        ! Test real, dimension(10) :: arr
        arena = create_ast_arena()
        dim_idx = push_literal(arena, "10", LITERAL_INTEGER)
        decl_idx = push_declaration(arena, "real", "arr", dimension_indices=[dim_idx])
        prog_idx = push_program(arena, "test", [decl_idx])

        ! Generate MLIR
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0) then
            ! Check for memref or array type
            if (index(output, "memref") > 0 .or. &
                index(output, "!fir.array") > 0 .or. &
                index(output, "f32") > 0) then
                print *, "PASS: Array type generates proper memory reference"
                passed = .true.
            else
                print *, "FAIL: Array type should generate memref or fir.array"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating array type: ", trim(error_msg)
        end if
    end function test_array_types

    function test_character_types() result(passed)
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

        ! Test character :: str
        arena = create_ast_arena()
        decl_idx = push_declaration(arena, "character", "str")
        prog_idx = push_program(arena, "test", [decl_idx])

        ! Generate MLIR
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0) then
            ! Check for character type
            if (index(output, "!fir.char") > 0) then
                print *, "PASS: Character type generates !fir.char"
                passed = .true.
            else
                print *, "FAIL: Character type should generate !fir.char"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating character type: ", trim(error_msg)
        end if
    end function test_character_types

    function test_type_conversion() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: int_lit, real_lit, add_idx, prog_idx

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        ! Test mixed arithmetic: 42 + 3.14 (requires type conversion)
        arena = create_ast_arena()
        int_lit = push_literal(arena, "42", LITERAL_INTEGER)
        real_lit = push_literal(arena, "3.14", LITERAL_REAL)
        add_idx = push_binary_op(arena, int_lit, real_lit, "+")
        prog_idx = push_program(arena, "test", [add_idx])

        ! Generate MLIR
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0) then
            ! Check for type conversion operations
            if (index(output, "arith.") > 0 .and. &
                (index(output, "i32") > 0 .or. index(output, "f32") > 0)) then
                print *, "PASS: Type conversion handled in mixed expressions"
                passed = .true.
            else
                print *, "FAIL: Type conversion not properly handled"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating type conversion: ", trim(error_msg)
        end if
    end function test_type_conversion

    function test_mixed_type_expressions() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: x_decl, y_decl, x_id, y_id, add_idx, prog_idx

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        ! Test mixed type variable arithmetic
        arena = create_ast_arena()
        x_decl = push_declaration(arena, "integer", "x", kind_value=4)
        y_decl = push_declaration(arena, "real", "y")
        x_id = push_identifier(arena, "x")
        y_id = push_identifier(arena, "y")
        add_idx = push_binary_op(arena, x_id, y_id, "+")
        prog_idx = push_program(arena, "test", [x_decl, y_decl, add_idx])

        ! Generate MLIR
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0) then
            ! Check for proper type handling
            if (index(output, "i32") > 0 .and. index(output, "f32") > 0) then
                print *, "PASS: Mixed type expressions generate proper MLIR"
                passed = .true.
            else
                print *, "FAIL: Mixed type expressions not properly handled"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating mixed type expressions: ", trim(error_msg)
        end if
    end function test_mixed_type_expressions

end program test_types
