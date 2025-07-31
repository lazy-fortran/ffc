program test_llvm_to_object
    use ast_core
    use ast_factory
    use backend_interface
    use backend_factory
    use temp_utils
    implicit none

    logical :: all_passed = .true.

    print *, "=== Testing LLVM IR to Object Code Generation ==="
    print *, ""

    all_passed = all_passed .and. test_simple_object_generation()
    all_passed = all_passed .and. test_function_object_generation()
    all_passed = all_passed .and. test_optimized_object_generation()

    if (all_passed) then
        print *, ""
        print *, "All LLVM to object code tests passed!"
        stop 0
    else
        print *, ""
        print *, "Some LLVM to object code tests failed!"
        stop 1
    end if

contains

    function test_simple_object_generation() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        type(backend_options_t) :: backend_opts
        character(len=:), allocatable :: mlir_code, llvm_ir
        character(len=256) :: error_msg
        type(temp_dir_manager) :: temp_mgr
        character(len=:), allocatable :: mlir_file, llvm_file, object_file
        integer :: unit, exit_code
        integer :: prog_idx, body_idx

        print *, "Testing simple LLVM IR to object code generation..."

        passed = .false.
        call temp_mgr%create('llvm_to_obj_test')

        ! Create simple AST
        body_idx = push_print_statement(arena, "*", &
                     [push_literal(arena, "'Hello from object file!'", LITERAL_STRING)])
        prog_idx = push_program(arena, "test_obj", [body_idx])

        ! Configure backend for object code generation
        backend_opts%compile_mode = .true.
        backend_opts%generate_llvm = .true.
        backend_opts%generate_executable = .false.
        backend_opts%output_file = temp_mgr%get_file_path('test.o')

        ! Create MLIR backend
        call create_backend("mlir", backend, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, "FAIL: Error creating backend:", trim(error_msg)
            return
        end if

        ! Generate MLIR code
        call backend%generate_code(arena, prog_idx, backend_opts, mlir_code, error_msg)

        if (len_trim(error_msg) > 0) then
            print *, "FAIL: Error generating MLIR:", trim(error_msg)
            return
        end if

        ! Save MLIR to file
        mlir_file = temp_mgr%get_file_path('test.mlir')
        open (newunit=unit, file=mlir_file, action='write')
        write (unit, '(A)') mlir_code
        close (unit)

        ! Lower MLIR to LLVM IR using mlir-translate
        llvm_file = temp_mgr%get_file_path('test.ll')
        call execute_command_line( &
            'mlir-translate --mlir-to-llvmir '//mlir_file//' -o '//llvm_file, &
            exitstat=exit_code)

        if (exit_code /= 0) then
            print *, "FAIL: mlir-translate failed"
            return
        end if

        ! Compile LLVM IR to object code using llc
        object_file = backend_opts%output_file
        call execute_command_line( &
            'llc -filetype=obj '//llvm_file//' -o '//object_file, &
            exitstat=exit_code)

        if (exit_code /= 0) then
            print *, "FAIL: llc failed with exit code:", exit_code
            return
        end if

        ! Check if object file was created
        inquire (file=object_file, exist=passed)

        if (passed) then
            print *, "PASS: Successfully generated object code"
        else
            print *, "FAIL: Object file not created"
        end if
    end function test_simple_object_generation

    function test_function_object_generation() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        type(backend_options_t) :: backend_opts
        character(len=:), allocatable :: mlir_code
        character(len=256) :: error_msg
        type(temp_dir_manager) :: temp_mgr
        character(len=:), allocatable :: mlir_file, llvm_file, object_file
        integer :: unit, exit_code
        integer :: prog_idx, func_idx, body_idx, param_idx

        print *, "Testing function compilation to object code..."

        passed = .false.
        call temp_mgr%create('func_obj_test')

        ! Create AST with function
        param_idx = push_parameter_declaration(arena, "n", "integer", 4)
        body_idx = push_assignment(arena, &
                                   push_identifier(arena, "factorial"), &
                                   push_binary_op(arena, &
                                                  push_identifier(arena, "n"), &
                                            push_literal(arena, "1", LITERAL_INTEGER), &
                                                  "*"))
    func_idx = push_function_def(arena, "factorial", [param_idx], "integer", [body_idx])

        body_idx = push_print_statement(arena, "*", &
                                        [push_call_or_subscript(arena, "factorial", &
                                          [push_literal(arena, "5", LITERAL_INTEGER)])])
        prog_idx = push_program(arena, "test_func_obj", [func_idx, body_idx])

        ! Configure backend
        backend_opts%compile_mode = .true.
        backend_opts%generate_llvm = .true.
        backend_opts%output_file = temp_mgr%get_file_path('func_test.o')

        ! Create MLIR backend
        call create_backend("mlir", backend, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, "FAIL: Error creating backend:", trim(error_msg)
            return
        end if

        ! Generate MLIR code
        call backend%generate_code(arena, prog_idx, backend_opts, mlir_code, error_msg)

        if (len_trim(error_msg) > 0) then
            print *, "FAIL: Error generating MLIR:", trim(error_msg)
            return
        end if

        ! Save and compile to object
        mlir_file = temp_mgr%get_file_path('func_test.mlir')
        open (newunit=unit, file=mlir_file, action='write')
        write (unit, '(A)') mlir_code
        close (unit)

        llvm_file = temp_mgr%get_file_path('func_test.ll')
        call execute_command_line( &
            'mlir-translate --mlir-to-llvmir '//mlir_file//' -o '//llvm_file, &
            exitstat=exit_code)

        if (exit_code /= 0) then
            print *, "FAIL: mlir-translate failed"
            return
        end if

        object_file = backend_opts%output_file
        call execute_command_line( &
            'llc -filetype=obj '//llvm_file//' -o '//object_file, &
            exitstat=exit_code)

        if (exit_code == 0) then
            inquire (file=object_file, exist=passed)
            if (passed) then
                print *, "PASS: Successfully compiled function to object code"
            else
                print *, "FAIL: Object file not created"
            end if
        else
            print *, "FAIL: Function compilation failed"
        end if
    end function test_function_object_generation

    function test_optimized_object_generation() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        type(backend_options_t) :: backend_opts
        character(len=:), allocatable :: mlir_code
        character(len=256) :: error_msg
        type(temp_dir_manager) :: temp_mgr
        character(len=:), allocatable :: mlir_file, llvm_file, object_file
        integer :: unit, exit_code
        integer :: prog_idx, body_idx

        print *, "Testing optimized object code generation..."

        passed = .false.
        call temp_mgr%create('opt_obj_test')

        ! Create AST with simple arithmetic for optimization
        body_idx = push_assignment(arena, &
                                   push_identifier(arena, "result"), &
                                   push_binary_op(arena, &
                                                  push_binary_op(arena, &
                                           push_literal(arena, "10", LITERAL_INTEGER), &
                                           push_literal(arena, "20", LITERAL_INTEGER), &
                                                                 "+"), &
                                           push_literal(arena, "30", LITERAL_INTEGER), &
                                                  "*"))
        prog_idx = push_program(arena, "test_opt", [body_idx])

        ! Configure backend with optimization
        backend_opts%compile_mode = .true.
        backend_opts%generate_llvm = .true.
        backend_opts%optimize = .true.
        backend_opts%output_file = temp_mgr%get_file_path('opt_test.o')

        ! Create MLIR backend
        call create_backend("mlir", backend, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, "FAIL: Error creating backend:", trim(error_msg)
            return
        end if

        ! Generate MLIR code
        call backend%generate_code(arena, prog_idx, backend_opts, mlir_code, error_msg)

        if (len_trim(error_msg) > 0) then
            print *, "FAIL: Error generating MLIR:", trim(error_msg)
            return
        end if

        ! Save and compile with optimization
        mlir_file = temp_mgr%get_file_path('opt_test.mlir')
        open (newunit=unit, file=mlir_file, action='write')
        write (unit, '(A)') mlir_code
        close (unit)

        llvm_file = temp_mgr%get_file_path('opt_test.ll')
        call execute_command_line( &
            'mlir-translate --mlir-to-llvmir '//mlir_file//' -o '//llvm_file, &
            exitstat=exit_code)

        if (exit_code /= 0) then
            print *, "FAIL: mlir-translate failed"
            return
        end if

        object_file = backend_opts%output_file
        call execute_command_line( &
            'llc -O3 -filetype=obj '//llvm_file//' -o '//object_file, &
            exitstat=exit_code)

        if (exit_code == 0) then
            inquire (file=object_file, exist=passed)
            if (passed) then
                print *, "PASS: Successfully generated optimized object code"
            else
                print *, "FAIL: Object file not created"
            end if
        else
            print *, "FAIL: Optimized compilation failed"
        end if
    end function test_optimized_object_generation

end program test_llvm_to_object
