program test_character_strings
    use iso_fortran_env, only: error_unit
    use mlir_backend
    use backend_factory
    use backend_interface
    use fortfront, only: ast_arena_t, create_ast_arena, LITERAL_STRING, LITERAL_INTEGER
    use ast_factory
    use mlir_utils, only: int_to_str
    implicit none

    logical :: all_tests_passed

    print *, "=== Character Strings Tests ==="
    print *

    all_tests_passed = .true.

    ! Run all character string tests
    if (.not. test_character_declaration()) all_tests_passed = .false.
    if (.not. test_character_assignment()) all_tests_passed = .false.
    if (.not. test_string_concatenation()) all_tests_passed = .false.
    if (.not. test_substring_operations()) all_tests_passed = .false.
    if (.not. test_character_intrinsics()) all_tests_passed = .false.

    print *
    if (all_tests_passed) then
        print *, "All character strings tests passed!"
        stop 0
    else
        print *, "Some character strings tests failed!"
        stop 1
    end if

contains

    function test_character_declaration() result(passed)
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

        error_msg = ""

        ! Test: character(len=20) :: name
        arena = create_ast_arena()

        ! Create character declaration
        decl_idx = push_declaration(arena, "character", ["name"], kind_value=20)
        prog_idx = push_program(arena, "test", [decl_idx])

        ! Generate MLIR code
        options%compile_mode = .true.
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0 .and. allocated(output)) then
            ! Check for proper character type generation
            if ((index(output, "memref<") > 0 .and. &
                 index(output, "i8") > 0) .or. &
                (index(output, "!llvm.array") > 0 .and. &
                 index(output, "i8") > 0)) then
                print *, "PASS: Character declaration generates proper MLIR"
                passed = .true.
            else
                print *, "FAIL: Missing proper character declaration generation"
                print *, "Expected: memref or array of i8 for character storage"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating MLIR: ", trim(error_msg)
        end if
    end function test_character_declaration

    function test_character_assignment() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: decl_idx, literal_idx, assign_idx, prog_idx, var_idx

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        error_msg = ""

        ! Test: character(len=20) :: name
        !       name = "Hello World"
        arena = create_ast_arena()

        ! Create character declaration
        decl_idx = push_declaration(arena, "character", ["name"], kind_value=20)

        ! Create string literal
        literal_idx = push_literal(arena, '"Hello World"', LITERAL_STRING)

        ! Create assignment
        var_idx = push_identifier(arena, "name")
        assign_idx = push_assignment(arena, var_idx, literal_idx)

        prog_idx = push_program(arena, "test", [decl_idx, assign_idx])

        ! Generate MLIR code
        options%compile_mode = .true.
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0 .and. allocated(output)) then
            ! Check for proper string assignment generation
            if ((index(output, "memref.copy") > 0 .or. &
                 index(output, "llvm.store") > 0) .and. &
                (index(output, "Hello World") > 0 .or. &
                 index(output, "constant") > 0)) then
                print *, "PASS: Character assignment generates proper MLIR"
                passed = .true.
            else
                print *, "FAIL: Missing proper character assignment generation"
                print *, "Expected: memref.copy or store with string constant"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating MLIR: ", trim(error_msg)
        end if
    end function test_character_assignment

    function test_string_concatenation() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: concat_idx, prog_idx, str1_idx, str2_idx, result_idx, assign_idx

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        error_msg = ""

        ! Test: character(len=20) :: result
        !       result = "Hello" // " World"
        arena = create_ast_arena()

        ! Create string literals
        str1_idx = push_literal(arena, '"Hello"', LITERAL_STRING)
        str2_idx = push_literal(arena, '" World"', LITERAL_STRING)

        ! Create concatenation using // operator
        concat_idx = push_binary_op(arena, str1_idx, str2_idx, "//")

        ! Create assignment
        result_idx = push_identifier(arena, "result")
        assign_idx = push_assignment(arena, result_idx, concat_idx)

        prog_idx = push_program(arena, "test", [assign_idx])

        ! Generate MLIR code
        options%compile_mode = .true.
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0 .and. allocated(output)) then
            ! Check for proper string concatenation generation
            if ((index(output, "concat") > 0 .or. &
                 index(output, "append") > 0) .and. &
                (index(output, "Hello") > 0 .and. &
                 index(output, "World") > 0)) then
                print *, "PASS: String concatenation generates proper MLIR"
                passed = .true.
            else
                print *, "FAIL: Missing proper string concatenation generation"
                print *, "Expected: string concatenation operations"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating MLIR: ", trim(error_msg)
        end if
    end function test_string_concatenation

    function test_substring_operations() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: substr_idx, prog_idx, str_idx, start_idx, end_idx
        integer :: result_idx, assign_idx

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        error_msg = ""

        ! Test: character(len=20) :: str, result
        !       str = "Hello World"
        !       result = str(1:5)  ! substring
        arena = create_ast_arena()

        ! Create string variable
        str_idx = push_identifier(arena, "str")

        ! Create substring indices
        start_idx = push_literal(arena, "1", LITERAL_INTEGER)
        end_idx = push_literal(arena, "5", LITERAL_INTEGER)

        ! Create substring operation using array-like access
        substr_idx = push_call_or_subscript(arena, "str", [start_idx, end_idx])

        ! Create assignment
        result_idx = push_identifier(arena, "result")
        assign_idx = push_assignment(arena, result_idx, substr_idx)

        prog_idx = push_program(arena, "test", [assign_idx])

        ! Generate MLIR code
        options%compile_mode = .true.
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0 .and. allocated(output)) then
            ! Check for proper substring generation
            if ((index(output, "substr") > 0 .or. &
                 index(output, "memref.subview") > 0) .and. &
                (index(output, "1") > 0 .and. &
                 index(output, "5") > 0)) then
                print *, "PASS: Substring operations generate proper MLIR"
                passed = .true.
            else
                print *, "FAIL: Missing proper substring generation"
                print *, "Expected: substring or subview operations"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating MLIR: ", trim(error_msg)
        end if
    end function test_substring_operations

    function test_character_intrinsics() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: len_call_idx, prog_idx, str_idx, result_idx, assign_idx

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        error_msg = ""

        ! Test: character(len=20) :: str
        !       integer :: length
        !       length = len(str)
        arena = create_ast_arena()

        ! Create string variable
        str_idx = push_identifier(arena, "str")

        ! Create len() intrinsic call
        len_call_idx = push_call_or_subscript(arena, "len", [str_idx])

        ! Create assignment
        result_idx = push_identifier(arena, "length")
        assign_idx = push_assignment(arena, result_idx, len_call_idx)

        prog_idx = push_program(arena, "test", [assign_idx])

        ! Generate MLIR code
        options%compile_mode = .true.
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0 .and. allocated(output)) then
            ! Check for proper character intrinsic generation
            if ((index(output, "len") > 0 .or. &
                 index(output, "string_len") > 0) .and. &
                (index(output, "func.call") > 0 .or. &
                 index(output, "call") > 0)) then
                print *, "PASS: Character intrinsics generate proper MLIR"
                passed = .true.
            else
                print *, "FAIL: Missing proper character intrinsic generation"
                print *, "Expected: function call for len() intrinsic"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating MLIR: ", trim(error_msg)
        end if
    end function test_character_intrinsics

end program test_character_strings
