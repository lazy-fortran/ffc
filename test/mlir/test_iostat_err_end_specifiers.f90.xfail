program test_iostat_err_end_specifiers
    use iso_fortran_env, only: error_unit
    use backend_interface
    use backend_factory
    use fortfront, only: ast_arena_t, create_ast_arena, LITERAL_STRING, LITERAL_INTEGER
    use ast_factory
    implicit none

    logical :: all_tests_passed

    print *, "=== iostat/err/end Specifiers Tests ==="
    print *

    all_tests_passed = .true.

    ! Run all tests following TDD RED phase - these should fail initially
    if (.not. test_iostat_specifier()) all_tests_passed = .false.
    if (.not. test_err_specifier()) all_tests_passed = .false.
    if (.not. test_end_specifier()) all_tests_passed = .false.
    if (.not. test_combined_specifiers()) all_tests_passed = .false.

    print *
    if (all_tests_passed) then
        print *, "All iostat/err/end specifiers tests passed!"
        stop 0
    else
        print *, "Some iostat/err/end specifiers tests failed!"
        stop 1
    end if

contains

    ! RED PHASE: Test iostat specifier in I/O statements
    function test_iostat_specifier() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg, factory_error
        integer :: write_idx, str_idx, prog_idx, iostat_idx

        passed = .false.
        error_msg = ""
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        ! Create AST arena for: write(*,*,iostat=stat) "test"
        arena = create_ast_arena()
        
        ! Create string literal and iostat variable
        str_idx = push_literal(arena, "test", LITERAL_STRING, 1, 1)
        iostat_idx = push_identifier(arena, "stat", 1, 1)
        
        ! Create write statement with iostat specifier
        write_idx = push_write_statement_with_iostat(arena, "*", [str_idx], "*", iostat_idx, 1, 1)
        
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

        ! Check for HLFIR-compliant iostat error handling
        ! Should generate error checking code after FIR runtime calls
        if (index(output, "fir.call @_FortranAioEndIoStatement") > 0 .and. &
            index(output, "arith.cmpi sgt") > 0) then
            print *, "PASS: iostat specifier generates HLFIR error handling"
            passed = .true.
        else
            print *, "FAIL: Missing HLFIR iostat error handling"
            print *, "Expected: fir.call @_FortranAioEndIoStatement followed by error check"
            print *, "Generated MLIR:"
            print *, trim(output)
        end if
    end function test_iostat_specifier

    ! RED PHASE: Test err specifier in I/O statements
    function test_err_specifier() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg, factory_error
        integer :: read_idx, var_idx, prog_idx, err_label_idx

        passed = .false.
        error_msg = ""
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        ! Create AST arena for: read(*,*,err=100) var
        arena = create_ast_arena()
        
        ! Create variable and error label
        var_idx = push_identifier(arena, "var", 1, 1)
        err_label_idx = push_literal(arena, "100", LITERAL_INTEGER, 1, 1)
        
        ! Create read statement with err specifier
        read_idx = push_read_statement_with_err(arena, "*", [var_idx], "*", err_label_idx, 1, 1)
        
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

        ! Check for HLFIR-compliant err handling with branch to label
        if (index(output, "fir.call @_FortranAioEndIoStatement") > 0 .and. &
            index(output, "cf.br ^bb100") > 0) then
            print *, "PASS: err specifier generates HLFIR error handling with branch"
            passed = .true.
        else
            print *, "FAIL: Missing HLFIR err specifier handling"
            print *, "Expected: error check with cf.br ^bb100 branch"
            print *, "Generated MLIR:"
            print *, trim(output)
        end if
    end function test_err_specifier

    ! RED PHASE: Test end specifier in I/O statements  
    function test_end_specifier() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg, factory_error
        integer :: read_idx, var_idx, prog_idx, end_label_idx

        passed = .false.
        error_msg = ""
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        ! Create AST arena for: read(*,*,end=200) var
        arena = create_ast_arena()
        
        ! Create variable and end label
        var_idx = push_identifier(arena, "var", 1, 1)
        end_label_idx = push_literal(arena, "200", LITERAL_INTEGER, 1, 1)
        
        ! Create read statement with end specifier
        read_idx = push_read_statement_with_end(arena, "*", [var_idx], "*", end_label_idx, 1, 1)
        
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

        ! Check for HLFIR-compliant end-of-file handling
        if (index(output, "fir.call @_FortranAioEndIoStatement") > 0 .and. &
            index(output, "arith.cmpi eq") > 0 .and. &
            index(output, "cf.br ^bb200") > 0) then
            print *, "PASS: end specifier generates HLFIR EOF handling with branch"
            passed = .true.
        else
            print *, "FAIL: Missing HLFIR end specifier handling"
            print *, "Expected: EOF check with cf.br ^bb200 branch"
            print *, "Generated MLIR:"
            print *, trim(output)
        end if
    end function test_end_specifier

    ! RED PHASE: Test combined iostat, err, and end specifiers
    function test_combined_specifiers() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg, factory_error
        integer :: read_idx, var_idx, prog_idx, iostat_idx, err_idx, end_idx

        passed = .false.
        error_msg = ""
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        ! Create AST arena for: read(*,*,iostat=stat,err=100,end=200) var
        arena = create_ast_arena()
        
        ! Create variable and specifiers
        var_idx = push_identifier(arena, "var", 1, 1)
        iostat_idx = push_identifier(arena, "stat", 1, 1)
        err_idx = push_literal(arena, "100", LITERAL_INTEGER, 1, 1)
        end_idx = push_literal(arena, "200", LITERAL_INTEGER, 1, 1)
        
        ! Create read statement with combined specifiers
        read_idx = push_read_statement_with_all_specifiers(arena, "*", [var_idx], "*", &
                                                          iostat_idx, err_idx, end_idx, 1, 1)
        
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

        ! Check for comprehensive HLFIR error handling
        if (index(output, "fir.call @_FortranAioEndIoStatement") > 0 .and. &
            index(output, "arith.cmpi") > 0 .and. &
            index(output, "cf.br ^bb100") > 0 .and. &
            index(output, "cf.br ^bb200") > 0) then
            print *, "PASS: Combined specifiers generate comprehensive HLFIR handling"
            passed = .true.
        else
            print *, "FAIL: Missing comprehensive HLFIR specifier handling"
            print *, "Expected: iostat assignment, error branch, EOF branch"
            print *, "Generated MLIR:"
            print *, trim(output)
        end if
    end function test_combined_specifiers

    ! Note: Factory functions now implemented in ast_factory module

end program test_iostat_err_end_specifiers