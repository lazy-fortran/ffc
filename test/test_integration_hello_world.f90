module test_integration_hello_world
    use iso_c_binding
    use mlir_c_core
    use mlir_c_types
    use mlir_c_operations
    use mlir_builder
    use hlfir_dialect
    use standard_dialects
    use program_gen
    use mlir_c_backend
    use backend_interface
    use fortfront
    implicit none
    private
    
    public :: test_hello_world_compilation_full
    
contains

    function test_hello_world_compilation_full() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_builder_t) :: builder
        type(mlir_module_t) :: module
        type(mlir_location_t) :: loc
        type(mlir_c_backend_t) :: backend
        type(backend_options_t) :: options
        type(ast_arena_t) :: arena
        integer :: prog_index
        character(len=:), allocatable :: output
        character(len=1024) :: error_msg
        
        passed = .true.
        
        print *, "Testing Hello World compilation with MLIR C API..."
        
        ! Initialize MLIR infrastructure
        context = create_mlir_context()
        builder = create_mlir_builder(context)
        loc = create_unknown_location(context)
        module = create_empty_module(loc)
        
        ! Register dialects
        call register_func_dialect(context)
        call register_hlfir_dialect(context)
        call register_fir_dialect(context)
        
        ! Generate Hello World program using HLFIR
        call generate_hello_world_hlfir(builder, module)
        
        ! Verify module is valid
        passed = passed .and. verify_module(module)
        
        ! Initialize backend
        call backend%init()
        passed = passed .and. backend%is_initialized()
        
        ! Create AST arena (simplified for testing)
        arena = create_test_arena()
        prog_index = 1
        
        ! Test 1: Generate HLFIR output
        options%compile_mode = .false.
        options%emit_hlfir = .true.
        call backend%generate_code(arena, prog_index, options, output, error_msg)
        
        passed = passed .and. (len_trim(error_msg) == 0)
        passed = passed .and. allocated(output)
        passed = passed .and. verify_hlfir_output(output)
        
        if (passed) then
            print *, "  HLFIR generation: PASS"
            print *, "  Output:"
            print *, trim(output)
        else
            print *, "  HLFIR generation: FAIL"
            if (len_trim(error_msg) > 0) print *, "  Error: ", trim(error_msg)
        end if
        
        ! Test 2: Lower to FIR
        options%emit_hlfir = .false.
        options%emit_fir = .true.
        call backend%generate_code(arena, prog_index, options, output, error_msg)
        
        passed = passed .and. (len_trim(error_msg) == 0)
        passed = passed .and. verify_fir_output(output)
        
        if (passed) then
            print *, "  FIR lowering: PASS"
        else
            print *, "  FIR lowering: FAIL"
        end if
        
        ! Test 3: Lower to LLVM
        options%emit_fir = .false.
        options%generate_llvm = .true.
        call backend%generate_code(arena, prog_index, options, output, error_msg)
        
        passed = passed .and. (len_trim(error_msg) == 0)
        passed = passed .and. verify_llvm_output(output)
        
        if (passed) then
            print *, "  LLVM lowering: PASS"
        else
            print *, "  LLVM lowering: FAIL"
        end if
        
        ! Test 4: Compile to object file
        options%compile_mode = .true.
        options%generate_llvm = .true.
        options%output_file = "hello_world_test.o"
        call backend%generate_code(arena, prog_index, options, output, error_msg)
        
        passed = passed .and. (len_trim(error_msg) == 0)
        passed = passed .and. file_exists("hello_world_test.o")
        
        if (passed) then
            print *, "  Object file generation: PASS"
        else
            print *, "  Object file generation: FAIL"
        end if
        
        ! Clean up
        if (file_exists("hello_world_test.o")) then
            call delete_file("hello_world_test.o")
        end if
        
        call backend%cleanup()
        call destroy_mlir_builder(builder)
        call destroy_mlir_context(context)
        
        if (passed) then
            print *, "Hello World compilation test: PASS"
        else
            print *, "Hello World compilation test: FAIL"
        end if
    end function test_hello_world_compilation_full

    subroutine generate_hello_world_hlfir(builder, module)
        type(mlir_builder_t), intent(inout) :: builder
        type(mlir_module_t), intent(inout) :: module
        
        type(mlir_operation_t) :: func_op, entry_block
        type(mlir_operation_t) :: declare_op, assign_op, print_op, return_op
        type(mlir_type_t) :: func_type, char_type, i32_type
        type(mlir_value_t) :: str_const, unit_const
        type(mlir_location_t) :: loc
        
        ! Get location
        loc = builder%get_unknown_location()
        
        ! Create function type: () -> i32
        i32_type = create_i32_type(builder%get_context())
        func_type = create_function_type(builder%get_context(), [], [i32_type])
        
        ! Create main function
        func_op = create_func_op(builder, "main", func_type, loc)
        entry_block = func_op%get_entry_block()
        
        ! Set insertion point to entry block
        call builder%set_insertion_point_to_start(entry_block)
        
        ! Create string constant for "Hello, World!"
        char_type = create_character_type(builder%get_context(), 13)
        str_const = create_string_constant(builder, "Hello, World!", char_type, loc)
        
        ! Create unit number constant (6 for stdout)
        unit_const = create_integer_constant(builder, 6, i32_type, loc)
        
        ! Create print statement using HLFIR
        print_op = create_hlfir_print_op(builder, unit_const, str_const, loc)
        
        ! Create return statement
        return_op = create_return_op(builder, create_integer_constant(builder, 0, i32_type, loc), loc)
        
        ! Add function to module
        call module%add_operation(func_op)
    end subroutine generate_hello_world_hlfir

    ! Helper functions
    
    function create_test_arena() result(arena)
        type(ast_arena_t) :: arena
        ! Create a minimal AST arena for testing
        ! In real implementation, would create proper AST nodes
    end function create_test_arena

    function verify_module(module) result(valid)
        type(mlir_module_t), intent(in) :: module
        logical :: valid
        ! Verify module structure
        valid = module%is_valid()
    end function verify_module

    function verify_hlfir_output(output) result(valid)
        character(len=*), intent(in) :: output
        logical :: valid
        
        ! Check for expected HLFIR constructs
        valid = (index(output, "func.func @main") > 0) .and. &
                (index(output, "hlfir") > 0) .and. &
                (index(output, "return") > 0)
    end function verify_hlfir_output

    function verify_fir_output(output) result(valid)
        character(len=*), intent(in) :: output
        logical :: valid
        
        ! Check for FIR constructs (after HLFIR lowering)
        valid = (index(output, "fir.") > 0) .and. &
                (index(output, "func.func @main") > 0)
    end function verify_fir_output

    function verify_llvm_output(output) result(valid)
        character(len=*), intent(in) :: output
        logical :: valid
        
        ! Check for LLVM IR constructs
        valid = (index(output, "llvm.") > 0) .or. &
                (index(output, "define") > 0)
    end function verify_llvm_output

    function file_exists(filename) result(exists)
        character(len=*), intent(in) :: filename
        logical :: exists
        
        inquire(file=filename, exist=exists)
    end function file_exists

    subroutine delete_file(filename)
        character(len=*), intent(in) :: filename
        integer :: unit, iostat
        
        open(newunit=unit, file=filename, status='old', iostat=iostat)
        if (iostat == 0) then
            close(unit, status='delete')
        end if
    end subroutine delete_file

    ! Stub implementations for missing functions
    
    function create_character_type(context, length) result(char_type)
        type(mlir_context_t), intent(in) :: context
        integer, intent(in) :: length
        type(mlir_type_t) :: char_type
        
        ! Stub: would create !fir.char<1,length>
        char_type%ptr = c_null_ptr
    end function create_character_type

    function create_function_type(context, inputs, outputs) result(func_type)
        type(mlir_context_t), intent(in) :: context
        type(mlir_type_t), intent(in) :: inputs(:), outputs(:)
        type(mlir_type_t) :: func_type
        
        ! Stub: would create function type
        func_type%ptr = c_null_ptr
    end function create_function_type

    function create_func_op(builder, name, func_type, loc) result(func_op)
        type(mlir_builder_t), intent(in) :: builder
        character(len=*), intent(in) :: name
        type(mlir_type_t), intent(in) :: func_type
        type(mlir_location_t), intent(in) :: loc
        type(mlir_operation_t) :: func_op
        
        ! Stub: would create func.func operation
        func_op%ptr = c_null_ptr
    end function create_func_op

    function create_string_constant(builder, str, char_type, loc) result(str_val)
        type(mlir_builder_t), intent(in) :: builder
        character(len=*), intent(in) :: str
        type(mlir_type_t), intent(in) :: char_type
        type(mlir_location_t), intent(in) :: loc
        type(mlir_value_t) :: str_val
        
        ! Stub: would create string constant
        str_val%ptr = c_null_ptr
    end function create_string_constant

    function create_integer_constant(builder, value, int_type, loc) result(int_val)
        type(mlir_builder_t), intent(in) :: builder
        integer, intent(in) :: value
        type(mlir_type_t), intent(in) :: int_type
        type(mlir_location_t), intent(in) :: loc
        type(mlir_value_t) :: int_val
        
        ! Stub: would create integer constant
        int_val%ptr = c_null_ptr
    end function create_integer_constant

    function create_hlfir_print_op(builder, unit, str_val, loc) result(print_op)
        type(mlir_builder_t), intent(in) :: builder
        type(mlir_value_t), intent(in) :: unit, str_val
        type(mlir_location_t), intent(in) :: loc
        type(mlir_operation_t) :: print_op
        
        ! Stub: would create HLFIR print operation
        print_op%ptr = c_null_ptr
    end function create_hlfir_print_op

    function create_return_op(builder, value, loc) result(return_op)
        type(mlir_builder_t), intent(in) :: builder
        type(mlir_value_t), intent(in) :: value
        type(mlir_location_t), intent(in) :: loc
        type(mlir_operation_t) :: return_op
        
        ! Stub: would create return operation
        return_op%ptr = c_null_ptr
    end function create_return_op

end module test_integration_hello_world