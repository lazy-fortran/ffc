program test_complex_numbers
    use iso_fortran_env, only: error_unit
    use mlir_backend
    use backend_factory
    use backend_interface
    use ast_core, only: ast_arena_t, create_ast_arena, LITERAL_REAL, LITERAL_INTEGER
    use ast_factory
    use mlir_utils, only: int_to_str
    implicit none

    logical :: all_tests_passed

    print *, "=== Complex Numbers Tests ==="
    print *

    all_tests_passed = .true.

    ! Run all complex number tests
    if (.not. test_complex_declaration()) all_tests_passed = .false.
    if (.not. test_complex_assignment()) all_tests_passed = .false.
    if (.not. test_complex_arithmetic()) all_tests_passed = .false.
    if (.not. test_complex_intrinsics()) all_tests_passed = .false.
    if (.not. test_complex_literals()) all_tests_passed = .false.

    print *
    if (all_tests_passed) then
        print *, "All complex numbers tests passed!"
        stop 0
    else
        print *, "Some complex numbers tests failed!"
        stop 1
    end if

contains

    function test_complex_declaration() result(passed)
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

        ! Test: complex :: z
        arena = create_ast_arena()

        ! Create complex declaration
        decl_idx = push_declaration(arena, "complex", "z")
        prog_idx = push_program(arena, "test", [decl_idx])

        ! Generate MLIR code
        options%compile_mode = .true.
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0 .and. allocated(output)) then
            ! Check for proper complex type generation
            if ((index(output, "complex") > 0 .or. &
                 index(output, "!llvm.struct<(f32, f32)>") > 0) .and. &
                (index(output, "memref") > 0 .or. &
                 index(output, "alloca") > 0)) then
                print *, "PASS: Complex declaration generates proper MLIR"
                passed = .true.
            else
                print *, "FAIL: Missing proper complex declaration generation"
                print *, "Expected: complex type with real and imaginary parts"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating MLIR: ", trim(error_msg)
        end if
    end function test_complex_declaration

    function test_complex_assignment() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: decl_idx, real_idx, imag_idx, complex_literal_idx
        integer :: assign_idx, prog_idx, var_idx

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        error_msg = ""

        ! Test: complex :: z
        !       z = (3.0, 4.0)
        arena = create_ast_arena()

        ! Create complex declaration
        decl_idx = push_declaration(arena, "complex", "z")

        ! Create complex literal (3.0, 4.0)
        real_idx = push_literal(arena, "3.0", LITERAL_REAL)
        imag_idx = push_literal(arena, "4.0", LITERAL_REAL)
        complex_literal_idx = push_complex_literal(arena, real_idx, imag_idx)

        ! Create assignment
        var_idx = push_identifier(arena, "z")
        assign_idx = push_assignment(arena, var_idx, complex_literal_idx)

        prog_idx = push_program(arena, "test", [decl_idx, assign_idx])

        ! Generate MLIR code
        options%compile_mode = .true.
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0 .and. allocated(output)) then
            ! Check for proper complex assignment generation
            if ((index(output, "3.0") > 0 .and. &
                 index(output, "4.0") > 0) .and. &
                (index(output, "insertvalue") > 0 .or. &
                 index(output, "store") > 0)) then
                print *, "PASS: Complex assignment generates proper MLIR"
                passed = .true.
            else
                print *, "FAIL: Missing proper complex assignment generation"
               print *, "Expected: complex literal with insertvalue or store operations"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating MLIR: ", trim(error_msg)
        end if
    end function test_complex_assignment

    function test_complex_arithmetic() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: z1_idx, z2_idx, add_idx, prog_idx, result_idx, assign_idx

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        error_msg = ""

        ! Test: complex :: z1, z2, result
        !       result = z1 + z2
        arena = create_ast_arena()

        ! Create complex variables
        z1_idx = push_identifier(arena, "z1")
        z2_idx = push_identifier(arena, "z2")

        ! Create complex addition
        add_idx = push_binary_op(arena, z1_idx, z2_idx, "+")

        ! Create assignment
        result_idx = push_identifier(arena, "result")
        assign_idx = push_assignment(arena, result_idx, add_idx)

        prog_idx = push_program(arena, "test", [assign_idx])

        ! Generate MLIR code
        options%compile_mode = .true.
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0 .and. allocated(output)) then
            ! Check for proper complex arithmetic generation
            if ((index(output, "complex_add") > 0 .or. &
                 index(output, "func.call") > 0) .and. &
                (index(output, "z1") > 0 .and. &
                 index(output, "z2") > 0)) then
                print *, "PASS: Complex arithmetic generates proper MLIR"
                passed = .true.
            else
                print *, "FAIL: Missing proper complex arithmetic generation"
                print *, "Expected: complex arithmetic operations or function calls"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating MLIR: ", trim(error_msg)
        end if
    end function test_complex_arithmetic

    function test_complex_intrinsics() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: z_idx, real_call_idx, imag_call_idx, prog_idx
        integer :: real_result_idx, imag_result_idx, real_assign_idx, imag_assign_idx

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        error_msg = ""

        ! Test: complex :: z
        !       real :: r, i
        !       r = real(z)
        !       i = aimag(z)
        arena = create_ast_arena()

        ! Create complex variable
        z_idx = push_identifier(arena, "z")

        ! Create real() intrinsic call
        real_call_idx = push_call_or_subscript(arena, "real", [z_idx])

        ! Create aimag() intrinsic call
        imag_call_idx = push_call_or_subscript(arena, "aimag", [z_idx])

        ! Create assignments
        real_result_idx = push_identifier(arena, "r")
        imag_result_idx = push_identifier(arena, "i")
        real_assign_idx = push_assignment(arena, real_result_idx, real_call_idx)
        imag_assign_idx = push_assignment(arena, imag_result_idx, imag_call_idx)

        prog_idx = push_program(arena, "test", [real_assign_idx, imag_assign_idx])

        ! Generate MLIR code
        options%compile_mode = .true.
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0 .and. allocated(output)) then
            ! Check for proper complex intrinsic generation
            if ((index(output, "real") > 0 .or. &
                 index(output, "extractvalue") > 0) .and. &
                (index(output, "aimag") > 0 .or. &
                 index(output, "func.call") > 0)) then
                print *, "PASS: Complex intrinsics generate proper MLIR"
                passed = .true.
            else
                print *, "FAIL: Missing proper complex intrinsic generation"
            print *, "Expected: extractvalue for real/imaginary parts or function calls"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating MLIR: ", trim(error_msg)
        end if
    end function test_complex_intrinsics

    function test_complex_literals() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: real_idx, imag_idx, complex_idx, prog_idx

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        error_msg = ""

        ! Test: complex literal (1.5, -2.3)
        arena = create_ast_arena()

        ! Create complex literal
        real_idx = push_literal(arena, "1.5", LITERAL_REAL)
        imag_idx = push_literal(arena, "-2.3", LITERAL_REAL)
        complex_idx = push_complex_literal(arena, real_idx, imag_idx)

        prog_idx = push_program(arena, "test", [complex_idx])

        ! Generate MLIR code
        options%compile_mode = .true.
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0 .and. allocated(output)) then
            ! Check for proper complex literal generation
            if ((index(output, "1.5") > 0 .and. &
                 index(output, "2.3") > 0) .and. &
                (index(output, "undef") > 0 .or. &
                 index(output, "constant") > 0)) then
                print *, "PASS: Complex literals generate proper MLIR"
                passed = .true.
            else
                print *, "FAIL: Missing proper complex literal generation"
                print *, "Expected: complex constant with real and imaginary parts"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating MLIR: ", trim(error_msg)
        end if
    end function test_complex_literals

end program test_complex_numbers
