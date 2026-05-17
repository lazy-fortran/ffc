program test_io_statement_options
    use iso_fortran_env, only: error_unit
    use backend_interface
    use backend_factory
    use ast_core, only: ast_arena_t, create_ast_arena, LITERAL_STRING, LITERAL_INTEGER
    use ast_factory
    implicit none

    logical :: all_tests_passed

    print *, "=== I/O Statement Options Tests ==="
    print *

    all_tests_passed = .true.

    ! Run all tests following TDD RED phase - these should fail initially
    if (.not. test_advance_specifier()) all_tests_passed = .false.
    if (.not. test_rec_specifier()) all_tests_passed = .false.
    if (.not. test_pos_specifier()) all_tests_passed = .false.
    if (.not. test_multiple_io_options()) all_tests_passed = .false.

    print *
    if (all_tests_passed) then
        print *, "All I/O statement options tests passed!"
        stop 0
    else
        print *, "Some I/O statement options tests failed!"
        stop 1
    end if

contains

    ! RED PHASE: Test advance='NO' specifier for non-advancing I/O
    function test_advance_specifier() result(passed)
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

        ! Create AST arena for: write(*,'(A)', advance='NO') "prompt: "
        arena = create_ast_arena()
        
        ! Create string to write
        str_idx = push_literal(arena, "prompt: ", LITERAL_STRING, 1, 1)
        
        ! Create write statement with advance='NO'
        write_idx = push_write_statement_with_advance(arena, "*", [str_idx], "(A)", "NO", 1, 1)
        
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

        ! Check for non-advancing I/O handling
        if (index(output, "fir.call @_FortranAioEnableNonAdvancing") > 0 .or. &
            index(output, "advance") > 0) then
            print *, "PASS: Advance specifier generates proper MLIR"
            passed = .true.
        else
            print *, "FAIL: Missing advance specifier handling"
            print *, "Expected: Non-advancing I/O runtime call"
            print *, "Generated MLIR:"
            print *, trim(output)
        end if
    end function test_advance_specifier

    ! RED PHASE: Test rec= specifier for direct access
    function test_rec_specifier() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg, factory_error
        integer :: read_idx, var_idx, rec_idx, prog_idx

        passed = .false.
        error_msg = ""
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        ! Create AST arena for: read(10, rec=5) data
        arena = create_ast_arena()
        
        ! Create variable to read into
        var_idx = push_identifier(arena, "data", 1, 1)
        
        ! Create record number
        rec_idx = push_literal(arena, "5", LITERAL_INTEGER, 1, 1)
        
        ! Create read statement with rec specifier
        read_idx = push_read_statement_with_rec(arena, "10", [var_idx], rec_idx, 1, 1)
        
        ! Create program
        prog_idx = push_program(arena, "test_prog", [read_idx], 1, 1)

        ! Set backend options  
        options%compile_mode = .false.
        options%debug_info = .false.

        ! Generate MLIR output
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) > 0) then
            print *, "FAIL: Code generation failed: ", trim(error_msg)
            return
        end if

        ! Check for direct access I/O handling
        if (index(output, "fir.call @_FortranAioSetRec") > 0 .or. &
            index(output, "fir.call @_FortranAioBeginUnformattedInput") > 0) then
            print *, "PASS: Rec specifier generates proper direct access I/O"
            passed = .true.
        else
            print *, "FAIL: Missing rec specifier handling"
            print *, "Expected: Direct access I/O with record positioning"
            print *, "Generated MLIR:"
            print *, trim(output)
        end if
    end function test_rec_specifier

    ! RED PHASE: Test pos= specifier for stream access
    function test_pos_specifier() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg, factory_error
        integer :: write_idx, data_idx, pos_idx, prog_idx

        passed = .false.
        error_msg = ""
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        ! Create AST arena for: write(20, pos=100) data
        arena = create_ast_arena()
        
        ! Create data to write
        data_idx = push_identifier(arena, "data", 1, 1)
        
        ! Create position
        pos_idx = push_literal(arena, "100", LITERAL_INTEGER, 1, 1)
        
        ! Create write statement with pos specifier
        write_idx = push_write_statement_with_pos(arena, "20", [data_idx], pos_idx, 1, 1)
        
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

        ! Check for stream positioning
        if (index(output, "fir.call @_FortranAioSetPos") > 0) then
            print *, "PASS: Pos specifier generates proper stream positioning"
            passed = .true.
        else
            print *, "FAIL: Missing pos specifier handling"
            print *, "Expected: Stream positioning call"
            print *, "Generated MLIR:"
            print *, trim(output)
        end if
    end function test_pos_specifier

    ! RED PHASE: Test multiple I/O options together
    function test_multiple_io_options() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg, factory_error
        integer :: write_idx, data_idx, iostat_idx, prog_idx

        passed = .false.
        error_msg = ""
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        ! Create AST arena for: write(*, '(A)', advance='NO', iostat=ios) data
        arena = create_ast_arena()
        
        ! Create data and iostat variable
        data_idx = push_identifier(arena, "data", 1, 1)
        iostat_idx = push_identifier(arena, "ios", 1, 1)
        
        ! Create write statement with multiple options
        write_idx = push_write_statement_with_multiple_options(arena, "*", [data_idx], &
                                                              "(A)", "NO", iostat_idx, 1, 1)
        
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

        ! Check for both advance and iostat handling
        if ((index(output, "advance") > 0 .or. index(output, "@_FortranAioEnableNonAdvancing") > 0) .and. &
            index(output, "fir.store") > 0) then
            print *, "PASS: Multiple I/O options handled correctly"
            passed = .true.
        else
            print *, "FAIL: Missing proper handling of multiple I/O options"
            print *, "Expected: Both advance and iostat handling"
            print *, "Generated MLIR:"
            print *, trim(output)
        end if
    end function test_multiple_io_options

    ! Helper functions that need to be implemented (RED phase placeholders)
    function push_write_statement_with_advance(arena, unit_spec, arg_indices, format_spec, &
                                              advance_spec, line, column) result(write_index)
        type(ast_arena_t), intent(inout) :: arena
        character(len=*), intent(in) :: unit_spec, format_spec, advance_spec
        integer, intent(in) :: arg_indices(:)
        integer, intent(in), optional :: line, column
        integer :: write_index
        
        ! Placeholder - should create write statement with advance specifier
        write_index = 0
    end function push_write_statement_with_advance

    function push_read_statement_with_rec(arena, unit_spec, var_indices, rec_expr, &
                                         line, column) result(read_index)
        type(ast_arena_t), intent(inout) :: arena
        character(len=*), intent(in) :: unit_spec
        integer, intent(in) :: var_indices(:), rec_expr
        integer, intent(in), optional :: line, column
        integer :: read_index
        
        ! Placeholder - should create read statement with rec specifier
        read_index = 0
    end function push_read_statement_with_rec

    function push_write_statement_with_pos(arena, unit_spec, arg_indices, pos_expr, &
                                          line, column) result(write_index)
        type(ast_arena_t), intent(inout) :: arena
        character(len=*), intent(in) :: unit_spec
        integer, intent(in) :: arg_indices(:), pos_expr
        integer, intent(in), optional :: line, column
        integer :: write_index
        
        ! Placeholder - should create write statement with pos specifier
        write_index = 0
    end function push_write_statement_with_pos

    function push_write_statement_with_multiple_options(arena, unit_spec, arg_indices, &
                                                       format_spec, advance_spec, iostat_var, &
                                                       line, column) result(write_index)
        type(ast_arena_t), intent(inout) :: arena
        character(len=*), intent(in) :: unit_spec, format_spec, advance_spec
        integer, intent(in) :: arg_indices(:), iostat_var
        integer, intent(in), optional :: line, column
        integer :: write_index
        
        ! Placeholder - should create write statement with multiple options
        write_index = 0
    end function push_write_statement_with_multiple_options

end program test_io_statement_options