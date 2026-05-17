program test_object_to_executable
    use ast_core
    use ast_factory
    use backend_interface
    use backend_factory
    use temp_utils
    implicit none

    logical :: all_passed = .true.

    print *, "=== Testing Object Code to Executable Linking ==="
    print *, ""

    all_passed = all_passed .and. test_simple_executable()
    all_passed = all_passed .and. test_executable_with_runtime()
    all_passed = all_passed .and. test_executable_execution()

    if (all_passed) then
        print *, ""
        print *, "All object to executable tests passed!"
        stop 0
    else
        print *, ""
        print *, "Some object to executable tests failed!"
        stop 1
    end if

contains

    function test_simple_executable() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        type(backend_options_t) :: backend_opts
        character(len=:), allocatable :: mlir_code
        character(len=256) :: error_msg
        type(temp_dir_manager) :: temp_mgr
        character(len=:), allocatable :: exe_file
        integer :: exit_code
        integer :: prog_idx, body_idx

        print *, "Testing simple executable generation..."

        passed = .false.
        call temp_mgr%create('obj_to_exe_test')

        ! Create simple AST with main function
        body_idx = push_return(arena)
        prog_idx = push_program(arena, "main", [body_idx])

        ! Configure backend for executable generation
        backend_opts%compile_mode = .true.
        backend_opts%generate_llvm = .true.
        backend_opts%generate_executable = .true.
        backend_opts%debug_info = .true.  ! Enable debug output
        backend_opts%output_file = temp_mgr%get_file_path('test_exe')

        ! Create MLIR backend
        call create_backend("mlir", backend, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, "FAIL: Error creating backend:", trim(error_msg)
            return
        end if

        ! Generate code and compile to executable
        ! When compile_mode is true, the backend handles the entire compilation
        call backend%generate_code(arena, prog_idx, backend_opts, mlir_code, error_msg)

        if (len_trim(error_msg) > 0) then
            print *, "FAIL: Error during compilation:", trim(error_msg)
            return
        end if

        ! Check if executable was created
        exe_file = backend_opts%output_file
        inquire (file=exe_file, exist=passed)

        if (passed) then
            print *, "PASS: Successfully created executable"
        else
            print *, "FAIL: Executable not created"
        end if
    end function test_simple_executable

    function test_executable_with_runtime() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        type(backend_options_t) :: backend_opts
        character(len=:), allocatable :: mlir_code
        character(len=256) :: error_msg
        type(temp_dir_manager) :: temp_mgr
        character(len=:), allocatable :: exe_file
        integer :: exit_code
        integer :: prog_idx, body_idx

        print *, "Testing executable with runtime library..."

        passed = .false.
        call temp_mgr%create('exe_runtime_test')

        ! Create AST with print statement (requires runtime)
        body_idx = push_print_statement(arena, "*", &
                               [push_literal(arena, "'Hello, World!'", LITERAL_STRING)])
        prog_idx = push_program(arena, "main", [body_idx])

        ! Configure backend
        backend_opts%compile_mode = .true.
        backend_opts%generate_llvm = .true.
        backend_opts%generate_executable = .true.
        backend_opts%link_runtime = .true.
        backend_opts%output_file = temp_mgr%get_file_path('hello_exe')

        ! Create MLIR backend
        call create_backend("mlir", backend, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, "FAIL: Error creating backend:", trim(error_msg)
            return
        end if

        ! Generate code and compile to executable
        ! When compile_mode is true, the backend handles the entire compilation
        call backend%generate_code(arena, prog_idx, backend_opts, mlir_code, error_msg)

        if (len_trim(error_msg) > 0) then
            print *, "FAIL: Error during compilation:", trim(error_msg)
            return
        end if

        ! Check if executable was created
        exe_file = backend_opts%output_file
        inquire (file=exe_file, exist=passed)

        if (passed) then
            print *, "PASS: Successfully created executable with runtime"
        else
            print *, "FAIL: Executable not created"
        end if
    end function test_executable_with_runtime

    function test_executable_execution() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        type(backend_options_t) :: backend_opts
        character(len=:), allocatable :: mlir_code
        character(len=256) :: error_msg
        type(temp_dir_manager) :: temp_mgr
        character(len=:), allocatable :: exe_file
        integer :: exit_code
        integer :: prog_idx, body_idx

        print *, "Testing executable execution..."

        passed = .false.
        call temp_mgr%create('exe_run_test')

        ! Create AST that writes to stdout
        body_idx = push_stop(arena, push_literal(arena, "0", LITERAL_INTEGER))
        prog_idx = push_program(arena, "main", [body_idx])

        ! Configure backend
        backend_opts%compile_mode = .true.
        backend_opts%generate_llvm = .true.
        backend_opts%generate_executable = .true.
        backend_opts%output_file = temp_mgr%get_file_path('test_run')

        ! Create MLIR backend
        call create_backend("mlir", backend, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, "FAIL: Error creating backend:", trim(error_msg)
            return
        end if

        ! Generate code and compile to executable
        ! When compile_mode is true, the backend handles the entire compilation
        call backend%generate_code(arena, prog_idx, backend_opts, mlir_code, error_msg)

        if (len_trim(error_msg) > 0) then
            print *, "FAIL: Error during compilation:", trim(error_msg)
            return
        end if

        ! Check if executable was created
        exe_file = backend_opts%output_file
        inquire (file=exe_file, exist=passed)

        if (.not. passed) then
            print *, "FAIL: Executable not created"
            return
        end if

        ! Execute the program
        call execute_command_line(exe_file, exitstat=exit_code)

        if (exit_code == 0) then
            print *, "PASS: Executable ran successfully"
            passed = .true.
        else
            print *, "FAIL: Executable failed with exit code:", exit_code
            passed = .false.
        end if
    end function test_executable_execution

end program test_object_to_executable
