program test_io_proper_implementation
    use iso_fortran_env, only: error_unit
    use mlir_backend
    use backend_factory
    use backend_interface
    use ast_core, only: ast_arena_t, create_ast_stack, LITERAL_INTEGER, LITERAL_STRING
    use ast_factory
    implicit none

    logical :: all_tests_passed

    print *, "=== Proper I/O Implementation Tests ==="
    print *

    all_tests_passed = .true.

    ! Run all comprehensive I/O tests
    if (.not. test_file_unit_management()) all_tests_passed = .false.
    if (.not. test_memory_managed_io()) all_tests_passed = .false.
    if (.not. test_type_safe_io()) all_tests_passed = .false.
    if (.not. test_error_handling()) all_tests_passed = .false.
    if (.not. test_format_parsing()) all_tests_passed = .false.
    if (.not. test_io_status_variables()) all_tests_passed = .false.

    print *
    if (all_tests_passed) then
        print *, "All proper I/O implementation tests passed!"
        stop 0
    else
        print *, "Some proper I/O implementation tests failed!"
        stop 1
    end if

contains

    function test_file_unit_management() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: open_idx, write_idx, close_idx, str_idx, prog_idx

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        error_msg = ""

        ! Test proper file unit management:
        ! open(unit=10, file="test.txt", status="new")
        ! write(10, *) "Hello"
        ! close(10)
        arena = create_ast_stack()

        ! TODO: Need AST factory support for open/close statements
        ! For now, test that file units are properly managed in MLIR
        str_idx = push_literal(arena, "Hello", LITERAL_STRING)
        write_idx = push_write_statement(arena, "10", [str_idx])
        prog_idx = push_program(arena, "test", [write_idx])

        ! Generate MLIR code
        options%compile_mode = .true.
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0 .and. allocated(output)) then
            ! Check for proper file unit management in MLIR
            if (index(output, "memref.alloca") > 0 .and. &  ! File descriptor allocation
                index(output, "call @fortran_unit_open") > 0 .and. &  ! Proper file opening
                index(output, "call @fortran_unit_write") > 0 .and. &  ! Unit-specific write
                index(output, "call @fortran_unit_close") > 0) then   ! Proper cleanup
                print *, "PASS: File unit management generates proper MLIR operations"
                passed = .true.
            else
                print *, "FAIL: Missing proper file unit management"
                print *, "Expected: memref.alloca, @fortran_unit_open, @fortran_unit_write, @fortran_unit_close"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating MLIR: ", trim(error_msg)
        end if
    end function test_file_unit_management

    function test_memory_managed_io() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: write_idx, str_idx, prog_idx

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        error_msg = ""

        ! Test memory management for I/O operations
        arena = create_ast_stack()

      str_idx = push_literal(arena, "Test string for memory management", LITERAL_STRING)
        write_idx = push_write_statement(arena, "10", [str_idx])
        prog_idx = push_program(arena, "test", [write_idx])

        ! Generate MLIR code
        options%compile_mode = .true.
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0 .and. allocated(output)) then
            ! Check for proper memory management
            if (index(output, "memref.alloca") > 0 .and. &  ! Buffer allocation
                index(output, "memref.store") > 0 .and. &   ! Store data to buffer
                index(output, "memref.load") > 0 .and. &    ! Load from buffer
                index(output, "memref.dealloca") > 0) then  ! Cleanup
                print *, "PASS: I/O memory management generates proper MLIR operations"
                passed = .true.
            else
                print *, "FAIL: Missing proper memory management for I/O"
          print *, "Expected: memref.alloca, memref.store, memref.load, memref.dealloca"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating MLIR: ", trim(error_msg)
        end if
    end function test_memory_managed_io

    function test_type_safe_io() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: write_idx, int_idx, real_idx, prog_idx

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        error_msg = ""

        ! Test type-safe I/O with proper type conversion
        arena = create_ast_stack()

        int_idx = push_literal(arena, "42", LITERAL_INTEGER)
        write_idx = push_write_statement(arena, "10", [int_idx])
        prog_idx = push_program(arena, "test", [write_idx])

        ! Generate MLIR code
        options%compile_mode = .true.
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0 .and. allocated(output)) then
            ! Check for type-safe I/O operations
            if (index(output, "arith.extsi") > 0 .or. &     ! Type conversion
                index(output, "arith.sitofp") > 0 .or. &   ! Integer to float
                index(output, "call @fortran_io_i32") > 0 .or. &  ! Type-specific I/O
                index(output, ": i32") > 0) then           ! Type information preserved
                print *, "PASS: Type-safe I/O generates proper type conversions"
                passed = .true.
            else
                print *, "FAIL: Missing type-safe I/O operations"
             print *, "Expected: Type conversion operations and type-specific I/O calls"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating MLIR: ", trim(error_msg)
        end if
    end function test_type_safe_io

    function test_error_handling() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: write_idx, str_idx, prog_idx

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        error_msg = ""

        ! Test error handling and status checking
        arena = create_ast_stack()

        str_idx = push_literal(arena, "Test", LITERAL_STRING)
        write_idx = push_write_statement(arena, "10", [str_idx])
        prog_idx = push_program(arena, "test", [write_idx])

        ! Generate MLIR code
        options%compile_mode = .true.
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0 .and. allocated(output)) then
            ! Check for error handling and status checking
            if (index(output, "call @fortran_io_check_status") > 0 .or. &  ! Status checking
                index(output, "cf.cond_br") > 0 .or. &                    ! Conditional branching
                index(output, "call @fortran_io_error_handler") > 0) then  ! Error handling
                print *, "PASS: Error handling generates proper control flow"
                passed = .true.
            else
                print *, "FAIL: Missing error handling and status checking"
             print *, "Expected: Status checking, conditional branching, error handlers"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating MLIR: ", trim(error_msg)
        end if
    end function test_error_handling

    function test_format_parsing() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: write_idx, int_idx, prog_idx

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        error_msg = ""

        ! Test format descriptor parsing: write(10, '(I5)') 42
        arena = create_ast_stack()

        int_idx = push_literal(arena, "42", LITERAL_INTEGER)
        write_idx = push_write_statement(arena, "10", [int_idx], "(I5)")
        prog_idx = push_program(arena, "test", [write_idx])

        ! Generate MLIR code
        options%compile_mode = .true.
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0 .and. allocated(output)) then
            ! Check for format descriptor parsing
            if (index(output, "call @fortran_format_parse") > 0 .and. &  ! Format parsing
                index(output, "call @fortran_format_i5") > 0 .or. &      ! Specific format call
                index(output, "memref.alloca() : memref<5xi8>") > 0) then ! Format buffer
                print *, "PASS: Format descriptors are properly parsed and used"
                passed = .true.
            else
                print *, "FAIL: Missing format descriptor parsing"
              print *, "Expected: Format parsing, specific format calls, format buffers"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating MLIR: ", trim(error_msg)
        end if
    end function test_format_parsing

    function test_io_status_variables() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: write_idx, str_idx, prog_idx

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        error_msg = ""

        ! Test I/O status variable support (iostat, err, end)
        arena = create_ast_stack()

        str_idx = push_literal(arena, "Test", LITERAL_STRING)
        write_idx = push_write_statement(arena, "10", [str_idx])
        prog_idx = push_program(arena, "test", [write_idx])

        ! Generate MLIR code
        options%compile_mode = .true.
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0 .and. allocated(output)) then
            ! Check for I/O status variable support
            if (index(output, "memref.alloca() : memref<1xi32>") > 0 .and. &  ! iostat variable
                index(output, "memref.store") > 0 .and. &                     ! Store status
                index(output, "call @fortran_io_get_status") > 0) then        ! Get status
                print *, "PASS: I/O status variables are properly supported"
                passed = .true.
            else
                print *, "FAIL: Missing I/O status variable support"
                print *, "Expected: iostat allocation, status storage, status retrieval"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating MLIR: ", trim(error_msg)
        end if
    end function test_io_status_variables

end program test_io_proper_implementation
