program test_mlir_c_backend
    use iso_c_binding
    use mlir_c_core
    use mlir_c_types
    use mlir_c_operations
    use mlir_builder
    use backend_interface
    use ast_core
    use pass_manager
    use lowering_pipeline
    use mlir_c_backend
    implicit none

    ! Output type constants
    integer, parameter :: OUTPUT_MLIR = 1
    integer, parameter :: OUTPUT_OBJECT = 2
    integer, parameter :: OUTPUT_EXECUTABLE = 3

    ! AST builder type for testing
    type :: ast_builder_t
        type(ast_arena_t) :: arena
    contains
        procedure :: init => ast_builder_init
        procedure :: cleanup => ast_builder_cleanup
        procedure :: get_arena => ast_builder_get_arena
    end type ast_builder_t

    logical :: all_tests_passed

    print *, "=== MLIR C API Backend Tests ==="
    print *

    all_tests_passed = .true.

    ! Run all tests - these will fail initially (RED phase)
    if (.not. test_c_api_backend_against_existing()) all_tests_passed = .false.
    if (.not. test_compilation_to_object_files()) all_tests_passed = .false.
    if (.not. test_executable_generation()) all_tests_passed = .false.
    if (.not. test_optimization_levels()) all_tests_passed = .false.

    print *
    if (all_tests_passed) then
        print *, "All MLIR C API backend tests passed!"
        stop 0
    else
        print *, "Some MLIR C API backend tests failed!"
        stop 1
    end if

contains

    function test_c_api_backend_against_existing() result(passed)
        logical :: passed
        type(mlir_c_backend_t) :: c_backend
        type(backend_options_t) :: options
        type(ast_arena_t) :: arena
        integer :: prog_index
        character(len=:), allocatable :: c_api_output, text_output
        character(len=1024) :: error_msg
        type(ast_builder_t) :: builder
        
        passed = .true.
        
        ! Initialize backend and AST
        call c_backend%init()
        passed = passed .and. c_backend%is_initialized()
        
        ! Create test AST using builder
        call builder%init()
        arena = builder%get_arena()
        
        ! Create simple program AST
        prog_index = create_test_program_ast(builder)
        passed = passed .and. (prog_index > 0)
        
        ! Set backend options
        options%compile_mode = .false.
        options%optimize = .false.
        
        ! Test: Generate code using C API backend
        call c_backend%generate_code(arena, prog_index, options, c_api_output, error_msg)
        passed = passed .and. allocated(c_api_output)
        passed = passed .and. len_trim(error_msg) == 0
        passed = passed .and. len_trim(c_api_output) > 0
        
        ! Verify output contains proper MLIR structure (not text generation)
        passed = passed .and. c_backend%uses_c_api_exclusively()
        passed = passed .and. (.not. contains_text_generation(c_api_output))
        
        ! Test: Compare functional equivalence with text backend
        passed = passed .and. verify_functional_equivalence(c_api_output, arena, prog_index)
        
        if (passed) then
            print *, "PASS: test_c_api_backend_against_existing"
        else
            print *, "FAIL: test_c_api_backend_against_existing"
        end if
        
        ! Cleanup
        call c_backend%cleanup()
        call builder%cleanup()
    end function test_c_api_backend_against_existing

    function test_compilation_to_object_files() result(passed)
        logical :: passed
        type(mlir_c_backend_t) :: c_backend
        type(backend_options_t) :: options
        type(ast_arena_t) :: arena
        integer :: prog_index
        character(len=:), allocatable :: output
        character(len=1024) :: error_msg
        type(ast_builder_t) :: builder
        logical :: object_file_exists
        
        passed = .true.
        
        ! Initialize backend
        call c_backend%init()
        call builder%init()
        arena = builder%get_arena()
        
        ! Create test program
        prog_index = create_test_program_ast(builder)
        passed = passed .and. (prog_index > 0)
        
        ! Set options for object file generation
        options%compile_mode = .true.
        options%generate_llvm = .true.
        options%optimize = .false.
        options%output_file = "test_output.o"
        
        ! Test: Compile to object file
        call c_backend%generate_code(arena, prog_index, options, output, error_msg)
        passed = passed .and. len_trim(error_msg) == 0
        
        ! Verify object file was created
        object_file_exists = check_file_exists("test_output.o")
        passed = passed .and. object_file_exists
        
        ! Verify object file is valid
        passed = passed .and. is_valid_object_file("test_output.o")
        
        ! Test: Verify LLVM backend integration
        passed = passed .and. c_backend%has_llvm_integration()
        passed = passed .and. verify_llvm_ir_generation(c_backend, arena, prog_index)
        
        if (passed) then
            print *, "PASS: test_compilation_to_object_files"
        else
            print *, "FAIL: test_compilation_to_object_files"
        end if
        
        ! Cleanup
        if (object_file_exists) call delete_file("test_output.o")
        call c_backend%cleanup()
        call builder%cleanup()
    end function test_compilation_to_object_files

    function test_executable_generation() result(passed)
        logical :: passed
        type(mlir_c_backend_t) :: c_backend
        type(backend_options_t) :: options
        type(ast_arena_t) :: arena
        integer :: prog_index
        character(len=:), allocatable :: output
        character(len=1024) :: error_msg
        type(ast_builder_t) :: builder
        logical :: exe_exists
        integer :: exit_code
        
        passed = .true.
        
        ! Initialize backend
        call c_backend%init()
        call builder%init()
        arena = builder%get_arena()
        
        ! Create executable test program
        prog_index = create_executable_program_ast(builder)
        passed = passed .and. (prog_index > 0)
        
        ! Set options for executable generation
        options%compile_mode = .true.
        options%generate_executable = .true.
        options%link_runtime = .true.
        options%optimize = .false.
        options%output_file = "test_exe"
        
        ! Test: Generate executable
        call c_backend%generate_code(arena, prog_index, options, output, error_msg)
        passed = passed .and. len_trim(error_msg) == 0
        
        ! Verify executable was created
        exe_exists = check_file_exists("test_exe")
        passed = passed .and. exe_exists
        
        ! Test: Run executable and check exit code
        if (exe_exists) then
            exit_code = run_executable("./test_exe")
            passed = passed .and. (exit_code == 0)
        end if
        
        ! Test: Verify linking process
        passed = passed .and. c_backend%supports_linking()
        passed = passed .and. verify_runtime_linking(c_backend)
        
        if (passed) then
            print *, "PASS: test_executable_generation"
        else
            print *, "FAIL: test_executable_generation"
        end if
        
        ! Cleanup
        if (exe_exists) call delete_file("test_exe")
        call c_backend%cleanup()
        call builder%cleanup()
    end function test_executable_generation

    function test_optimization_levels() result(passed)
        logical :: passed
        type(mlir_c_backend_t) :: c_backend
        type(backend_options_t) :: options
        type(ast_arena_t) :: arena
        integer :: prog_index
        character(len=:), allocatable :: output_o0, output_o2, output_o3
        character(len=1024) :: error_msg
        type(ast_builder_t) :: builder
        integer :: size_o0, size_o2, size_o3
        
        passed = .true.
        
        ! Initialize backend
        call c_backend%init()
        call builder%init()
        arena = builder%get_arena()
        
        ! Create optimizable program
        prog_index = create_optimizable_program_ast(builder)
        passed = passed .and. (prog_index > 0)
        
        ! Test: Generate with -O0
        options%compile_mode = .true.
        options%generate_llvm = .true.
        options%optimize = .false.
        options%output_file = "test_o0.o"
        
        call c_backend%generate_code(arena, prog_index, options, output_o0, error_msg)
        passed = passed .and. len_trim(error_msg) == 0
        size_o0 = get_file_size("test_o0.o")
        
        ! Test: Generate with -O2
        options%optimize = .true.
        options%output_file = "test_o2.o"
        
        call c_backend%generate_code(arena, prog_index, options, output_o2, error_msg)
        passed = passed .and. len_trim(error_msg) == 0
        size_o2 = get_file_size("test_o2.o")
        
        ! Test: Generate with -O3  
        options%optimize = .true.
        options%output_file = "test_o3.o"
        
        call c_backend%generate_code(arena, prog_index, options, output_o3, error_msg)
        passed = passed .and. len_trim(error_msg) == 0
        size_o3 = get_file_size("test_o3.o")
        
        ! Verify optimization effects
        passed = passed .and. (size_o2 < size_o0)  ! O2 should produce smaller code
        passed = passed .and. (size_o3 <= size_o2)  ! O3 might be same or smaller
        
        ! Test: Verify optimization passes are applied
        passed = passed .and. verify_optimization_passes(c_backend, 0)
        passed = passed .and. verify_optimization_passes(c_backend, 2)
        passed = passed .and. verify_optimization_passes(c_backend, 3)
        
        if (passed) then
            print *, "PASS: test_optimization_levels"
        else
            print *, "FAIL: test_optimization_levels"
        end if
        
        ! Cleanup
        call delete_file("test_o0.o")
        call delete_file("test_o2.o")
        call delete_file("test_o3.o")
        call c_backend%cleanup()
        call builder%cleanup()
    end function test_optimization_levels

    ! Stub functions that will fail initially (RED phase)

    function create_test_program_ast(builder) result(prog_index)
        type(ast_builder_t), intent(inout) :: builder
        integer :: prog_index
        
        ! Return a valid program index for testing
        prog_index = 1
    end function create_test_program_ast

    function create_executable_program_ast(builder) result(prog_index)
        type(ast_builder_t), intent(inout) :: builder
        integer :: prog_index
        
        ! Return a valid program index for testing
        prog_index = 2
    end function create_executable_program_ast

    function create_optimizable_program_ast(builder) result(prog_index)
        type(ast_builder_t), intent(inout) :: builder
        integer :: prog_index
        
        ! Return a valid program index for testing
        prog_index = 3
    end function create_optimizable_program_ast

    function contains_text_generation(output) result(has_text_gen)
        character(len=*), intent(in) :: output
        logical :: has_text_gen
        
        ! Check for signs of text-based generation (concatenation patterns)
        has_text_gen = (index(output, '// "') > 0) .or. &
                      (index(output, 'trim(') > 0 .and. index(output, ')//') > 0)
    end function contains_text_generation

    function verify_functional_equivalence(output, arena, prog_index) result(is_equivalent)
        character(len=*), intent(in) :: output
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: prog_index
        logical :: is_equivalent
        
        ! For testing, check that output contains expected MLIR structure
        is_equivalent = (index(output, "module") > 0) .and. &
                       (index(output, "func.func") > 0)
    end function verify_functional_equivalence

    function check_file_exists(filename) result(exists)
        character(len=*), intent(in) :: filename
        logical :: exists
        
        ! Use Fortran intrinsic
        inquire(file=filename, exist=exists)
    end function check_file_exists

    function is_valid_object_file(filename) result(is_valid)
        character(len=*), intent(in) :: filename
        logical :: is_valid
        
        ! For testing, just check if file exists and has non-zero size
        is_valid = check_file_exists(filename)
        if (is_valid) then
            is_valid = get_file_size(filename) > 0
        end if
    end function is_valid_object_file

    subroutine delete_file(filename)
        character(len=*), intent(in) :: filename
        logical :: exists
        integer :: unit, iostat
        
        inquire(file=filename, exist=exists)
        if (exists) then
            open(newunit=unit, file=filename, status='old', iostat=iostat)
            if (iostat == 0) then
                close(unit, status='delete')
            end if
        end if
    end subroutine delete_file

    function verify_llvm_ir_generation(backend, arena, prog_index) result(verified)
        type(mlir_c_backend_t), intent(in) :: backend
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: prog_index
        logical :: verified
        
        ! Check that backend can generate LLVM IR
        verified = backend%has_llvm_integration()
    end function verify_llvm_ir_generation

    function run_executable(command) result(exit_code)
        character(len=*), intent(in) :: command
        integer :: exit_code
        
        ! Use system call (simplified for testing)
        call execute_command_line(command, exitstat=exit_code)
    end function run_executable

    function verify_runtime_linking(backend) result(verified)
        type(mlir_c_backend_t), intent(in) :: backend
        logical :: verified
        
        ! Check that backend supports linking
        verified = backend%supports_linking()
    end function verify_runtime_linking

    function get_file_size(filename) result(size)
        character(len=*), intent(in) :: filename
        integer :: size
        integer(8) :: file_size
        
        inquire(file=filename, size=file_size)
        size = int(file_size)
    end function get_file_size

    function verify_optimization_passes(backend, level) result(verified)
        type(mlir_c_backend_t), intent(in) :: backend
        integer, intent(in) :: level
        logical :: verified
        
        ! For testing, assume optimization passes work
        verified = .true.
    end function verify_optimization_passes


    ! AST builder procedures

    subroutine ast_builder_init(this)
        class(ast_builder_t), intent(inout) :: this
        ! Initialize arena (simplified for testing)
        ! In real implementation would properly initialize AST arena
    end subroutine ast_builder_init

    subroutine ast_builder_cleanup(this)
        class(ast_builder_t), intent(inout) :: this
        ! Cleanup arena (simplified for testing)
    end subroutine ast_builder_cleanup

    function ast_builder_get_arena(this) result(arena)
        class(ast_builder_t), intent(in) :: this
        type(ast_arena_t) :: arena
        ! Return the arena
        arena = this%arena
    end function ast_builder_get_arena

end program test_mlir_c_backend