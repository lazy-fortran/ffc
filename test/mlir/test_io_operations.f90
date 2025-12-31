program test_io_operations
    use iso_fortran_env, only: error_unit
    use backend_interface
    use backend_factory
    use fortfront, only: ast_arena_t, create_ast_arena, LITERAL_INTEGER, LITERAL_STRING
    use ast_factory
    implicit none

    logical :: all_tests_passed

    print *, "=== I/O Operations Tests ==="
    print *

    all_tests_passed = .true.

    ! Run all tests
    if (.not. test_print_statement()) all_tests_passed = .false.
    if (.not. test_write_statement()) all_tests_passed = .false.
    if (.not. test_read_statement()) all_tests_passed = .false.
    if (.not. test_format_descriptors()) all_tests_passed = .false.
    if (.not. test_file_io_runtime()) all_tests_passed = .false.

    print *
    if (all_tests_passed) then
        print *, "All I/O operations tests passed!"
        stop 0
    else
        print *, "Some I/O operations tests failed!"
        stop 1
    end if

contains

    function test_print_statement() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg, factory_error
        integer :: print_idx, str_idx, prog_idx

        passed = .false.
        error_msg = ""
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        ! Test print statement: print *, "Hello World"
        arena = create_ast_arena()

        str_idx = push_literal(arena, "Hello World", LITERAL_STRING)
        print_idx = push_print_statement(arena, "*", [str_idx])
        prog_idx = push_program(arena, "test", [print_idx])

        ! Generate MLIR code
        options%compile_mode = .false.  ! Changed to false to see MLIR output
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0 .and. allocated(output)) then
            ! Check for HLFIR-compliant FIR runtime call in MLIR output
            if (index(output, "fir.call @_FortranAio") > 0) then
                print *, "PASS: Print statement generates HLFIR-compliant FIR runtime call"
                passed = .true.
            else
                print *, "FAIL: Missing HLFIR-compliant FIR runtime call (@_FortranAio*)"
                print *, "Expected: fir.call @_FortranAio* functions"
                print *, "Output:"
                print *, output
                ! Check for legacy calls that need to be updated
                if (index(output, "call @fortran_print") > 0 .or. index(output, "call @printf") > 0) then
                    print *, "Note: Found legacy runtime calls - need HLFIR update"
                end if
            end if
        else
            print *, "FAIL: Error generating MLIR: ", trim(error_msg)
        end if
    end function test_print_statement

    function test_write_statement() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg, factory_error
        integer :: write_idx, str_idx, prog_idx

        passed = .false.
        error_msg = ""
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        ! Test write statement: write(10, *) "Hello World"
        arena = create_ast_arena()

        str_idx = push_literal(arena, "Hello World", LITERAL_STRING)
        write_idx = push_write_statement(arena, "10", [str_idx])
        prog_idx = push_program(arena, "test", [write_idx])

        ! Generate MLIR code
        options%compile_mode = .false.  ! Changed to false to see MLIR output
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0 .and. allocated(output)) then
            ! Check for HLFIR-compliant FIR runtime call in MLIR output
            if (index(output, "fir.call @_FortranAio") > 0) then
                print *, "PASS: Write statement generates HLFIR-compliant FIR runtime call"
                passed = .true.
            else
                print *, "FAIL: Missing HLFIR-compliant FIR runtime call (@_FortranAio*)"
                print *, "Expected: fir.call @_FortranAio* functions"
                print *, "Output:"
                print *, output
                ! Check for legacy calls that need to be updated
                if (index(output, "call @fortran_write") > 0) then
                    print *, "Note: Found legacy runtime calls - need HLFIR update"
                end if
            end if
        else
            print *, "FAIL: Error generating MLIR: ", trim(error_msg)
        end if
    end function test_write_statement

    function test_read_statement() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg, factory_error
        integer :: read_idx, var_idx, prog_idx

        passed = .false.
        error_msg = ""
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        ! Test read statement: read(5, *) x
        arena = create_ast_arena()

        var_idx = push_identifier(arena, "x")
        read_idx = push_read_statement(arena, "5", [var_idx])
        prog_idx = push_program(arena, "test", [read_idx])

        ! Generate MLIR code
        options%compile_mode = .false.
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0 .and. allocated(output)) then
            ! Check for HLFIR-compliant FIR runtime call in MLIR output
            if (index(output, "fir.call @_FortranAio") > 0) then
                print *, "PASS: Read statement generates HLFIR-compliant FIR runtime call"
                passed = .true.
            else
                print *, "FAIL: Missing HLFIR-compliant FIR runtime call (@_FortranAio*)"
                print *, "Expected: fir.call @_FortranAio* functions"
                print *, "Output:"
                print *, output
                ! Check for legacy calls that need to be updated
                if (index(output, "call @fortran_read") > 0) then
                    print *, "Note: Found legacy runtime calls - need HLFIR update"
                end if
            end if
        else
            print *, "FAIL: Error generating MLIR: ", trim(error_msg)
        end if
    end function test_read_statement

    function test_format_descriptors() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg, factory_error
        integer :: print_idx, fmt_idx, var_idx, prog_idx

        passed = .false.
        error_msg = ""
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        ! Test formatted print: print '(I0)', 42
        arena = create_ast_arena()

        fmt_idx = push_literal(arena, "(I0)", LITERAL_STRING)
        var_idx = push_literal(arena, "42", LITERAL_INTEGER)
        print_idx = push_print_statement(arena, "(I0)", [var_idx])
        prog_idx = push_program(arena, "test", [print_idx])

        ! Generate MLIR code
        options%compile_mode = .false.
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0 .and. allocated(output)) then
            ! Check for HLFIR-compliant FIR runtime call with format support
            if (index(output, "fir.call @_FortranAio") > 0) then
                print *, "PASS: Format descriptors generate HLFIR-compliant FIR runtime calls"
                passed = .true.
            else
                print *, "FAIL: Missing HLFIR-compliant FIR runtime call (@_FortranAio*)"
                print *, "Expected: fir.call @_FortranAio* functions with format support"
                print *, "Output:"
                print *, output
                ! Check for legacy calls that need to be updated
                if (index(output, "call @fortran_print_formatted") > 0) then
                    print *, "Note: Found legacy runtime calls - need HLFIR update"
                end if
            end if
        else
            print *, "FAIL: Error generating MLIR: ", trim(error_msg)
        end if
    end function test_format_descriptors

    function test_file_io_runtime() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg, factory_error
        integer :: write_idx, str_idx, prog_idx

        passed = .false.
        error_msg = ""
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        ! Test file write: write(20, *) "File output"
        arena = create_ast_arena()

        str_idx = push_literal(arena, "File output", LITERAL_STRING)
        write_idx = push_write_statement(arena, "20", [str_idx])
        prog_idx = push_program(arena, "test", [write_idx])

        ! Generate MLIR code
        options%compile_mode = .false.
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0 .and. allocated(output)) then
            ! Check for HLFIR-compliant FIR runtime call in MLIR output
            if (index(output, "fir.call @_FortranAio") > 0) then
                print *, "PASS: File I/O generates HLFIR-compliant FIR runtime call"
                passed = .true.
            else
                print *, "FAIL: Missing HLFIR-compliant FIR runtime call (@_FortranAio*)"
                print *, "Expected: fir.call @_FortranAio* functions for file I/O"
                print *, "Output:"
                print *, output
                ! Check for legacy calls that need to be updated
                if (index(output, "call @fortran_write") > 0) then
                    print *, "Note: Found legacy runtime calls - need HLFIR update"
                end if
            end if
        else
            print *, "FAIL: Error generating MLIR: ", trim(error_msg)
        end if
    end function test_file_io_runtime

end program test_io_operations
