program test_format_descriptor_parsing
    use iso_fortran_env, only: error_unit
    use backend_interface
    use backend_factory
    use ast_core, only: ast_arena_t, create_ast_arena, LITERAL_STRING, LITERAL_INTEGER
    use ast_factory
    implicit none

    logical :: all_tests_passed

    print *, "=== Format Descriptor Parsing Tests ==="
    print *

    all_tests_passed = .true.

    ! Run all tests following TDD RED phase - these should fail initially
    if (.not. test_simple_format_descriptors()) all_tests_passed = .false.
    if (.not. test_complex_format_descriptors()) all_tests_passed = .false.
    if (.not. test_format_string_parsing()) all_tests_passed = .false.
    if (.not. test_runtime_format_expressions()) all_tests_passed = .false.

    print *
    if (all_tests_passed) then
        print *, "All format descriptor parsing tests passed!"
        stop 0
    else
        print *, "Some format descriptor parsing tests failed!"
        stop 1
    end if

contains

    ! RED PHASE: Test simple format descriptors like I5, F10.2, A20
    function test_simple_format_descriptors() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg, factory_error
        integer :: write_idx, int_idx, real_idx, str_idx, prog_idx

        passed = .false.
        error_msg = ""
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        ! Create AST arena for: write(*,'(I5,F10.2,A20)') n, x, name
        arena = create_ast_arena()

        ! Create variables to write
        int_idx = push_identifier(arena, "n", 1, 1)
        real_idx = push_identifier(arena, "x", 1, 1)
        str_idx = push_identifier(arena, "name", 1, 1)

        ! Create write statement with format descriptor
write_idx = push_write_statement_with_format(arena, "*", [int_idx, real_idx, str_idx], &
                                                     "(I5,F10.2,A20)", 1, 1)

        ! Create program
        prog_idx = push_program(arena, "test_prog", [write_idx], 1, 1)

        ! Set backend options
        options%compile_mode = .false.
        options%debug_info = .false.

        ! Generate MLIR output
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) > 0) then
            print *, "FAIL: Code generation failed: ", trim(error_msg)
            return
        end if

        ! Check for HLFIR-compliant format descriptor handling
        ! Should generate format parsing operations and typed I/O calls
      if (index(output, "fir.call @_FortranAioBeginExternalFormattedOutput") > 0 .and. &
            index(output, "fir.call @_FortranAioOutputInteger") > 0 .and. &
            index(output, "fir.call @_FortranAioOutputReal") > 0 .and. &
            index(output, "fir.call @_FortranAioOutputCharacter") > 0) then
            print *, "PASS: Simple format descriptors generate HLFIR-compliant I/O"
            passed = .true.
        else
            print *, "FAIL: Missing HLFIR format descriptor handling"
            print *, "Expected: Formatted I/O with typed output calls"
            print *, "Generated MLIR:"
            print *, trim(output)
        end if
    end function test_simple_format_descriptors

    ! RED PHASE: Test complex format descriptors with repetition, nesting
    function test_complex_format_descriptors() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg, factory_error
        integer :: write_idx, arr_idx, prog_idx

        passed = .false.
        error_msg = ""
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        ! Create AST arena for: write(*,'(3(I5,1X))') arr(1:3)
        arena = create_ast_arena()

        ! Create array subscript expression
        arr_idx = push_array_section(arena, "arr", 1, 3, 1, 1)

        ! Create write statement with repeated format descriptor
        write_idx = push_write_statement_with_format(arena, "*", [arr_idx], &
                                                     "(3(I5,1X))", 1, 1)

        ! Create program
        prog_idx = push_program(arena, "test_prog", [write_idx], 1, 1)

        ! Set backend options
        options%compile_mode = .false.
        options%debug_info = .false.

        ! Generate MLIR output
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) > 0) then
            print *, "FAIL: Code generation failed: ", trim(error_msg)
            return
        end if

        ! Check for HLFIR format repetition handling
        if (index(output, "fir.do_loop") > 0 .and. &
            index(output, "fir.call @_FortranAioOutputInteger") > 0) then
            print *, "PASS: Complex format descriptors handle repetition"
            passed = .true.
        else
            print *, "FAIL: Missing HLFIR format repetition handling"
            print *, "Expected: Loop for repeated format descriptors"
            print *, "Generated MLIR:"
            print *, trim(output)
        end if
    end function test_complex_format_descriptors

    ! RED PHASE: Test format string literal parsing
    function test_format_string_parsing() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg, factory_error
        integer :: write_idx, val_idx, prog_idx

        passed = .false.
        error_msg = ""
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        ! Create AST arena for: write(*,'(E15.7E3)') scientific_val
        arena = create_ast_arena()

        ! Create variable
        val_idx = push_identifier(arena, "scientific_val", 1, 1)

        ! Create write statement with scientific format
        write_idx = push_write_statement_with_format(arena, "*", [val_idx], &
                                                     "(E15.7E3)", 1, 1)

        ! Create program
        prog_idx = push_program(arena, "test_prog", [write_idx], 1, 1)

        ! Set backend options
        options%compile_mode = .false.
        options%debug_info = .false.

        ! Generate MLIR output
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) > 0) then
            print *, "FAIL: Code generation failed: ", trim(error_msg)
            return
        end if

        ! Check for format string parsing and storage
        if (index(output, "fir.address_of(@.str.fmt") > 0 .or. &
            index(output, "fir.string_lit") > 0) then
            print *, "PASS: Format strings are parsed and stored"
            passed = .true.
        else
            print *, "FAIL: Missing format string storage"
            print *, "Expected: Format string literal in MLIR"
            print *, "Generated MLIR:"
            print *, trim(output)
        end if
    end function test_format_string_parsing

    ! RED PHASE: Test runtime format expressions (format in variable)
    function test_runtime_format_expressions() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg, factory_error
        integer :: write_idx, val_idx, fmt_idx, prog_idx

        passed = .false.
        error_msg = ""
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        ! Create AST arena for: write(*,fmt) value
        arena = create_ast_arena()

        ! Create variables
        val_idx = push_identifier(arena, "value", 1, 1)
        fmt_idx = push_identifier(arena, "fmt", 1, 1)

        ! Create write statement with runtime format
        write_idx = push_write_statement_with_runtime_format(arena, "*", [val_idx], &
                                                             fmt_idx, 1, 1)

        ! Create program
        prog_idx = push_program(arena, "test_prog", [write_idx], 1, 1)

        ! Set backend options
        options%compile_mode = .false.
        options%debug_info = .false.

        ! Generate MLIR output
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) > 0) then
            print *, "FAIL: Code generation failed: ", trim(error_msg)
            return
        end if

        ! Check for runtime format handling
        if (index(output, "fir.call @_FortranAioBeginExternalFormattedOutput") > 0) then
            print *, "PASS: Runtime format expressions handled correctly"
            passed = .true.
        else
            print *, "FAIL: Missing runtime format expression handling"
            print *, "Expected: Formatted output with runtime format"
            print *, "Generated MLIR:"
            print *, trim(output)
        end if
    end function test_runtime_format_expressions

    ! Helper functions removed - now provided by ast_factory module

end program test_format_descriptor_parsing
