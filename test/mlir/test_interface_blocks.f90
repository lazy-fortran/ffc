program test_interface_blocks
    use ast_core, only: ast_arena_t, create_ast_stack
    use ast_factory
    use backend_interface
    use backend_factory
    use temp_utils
    implicit none

    logical :: all_passed = .true.

    print *, "=== Testing Interface Blocks ==="
    print *, ""

    all_passed = all_passed .and. test_simple_interface_block()
    all_passed = all_passed .and. test_generic_interface()
    all_passed = all_passed .and. test_operator_interface()

    if (all_passed) then
        print *, ""
        print *, "All interface block tests passed!"
        stop 0
    else
        print *, ""
        print *, "Some interface block tests failed!"
        stop 1
    end if

contains

    function test_simple_interface_block() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        type(backend_options_t) :: backend_opts
        character(len=:), allocatable :: mlir_code
        character(len=256) :: error_msg
        integer :: prog_idx, interface_idx, func_decl_idx

        print *, "Testing simple interface block..."

        passed = .false.

        ! Initialize arena
        arena = create_ast_stack()

        ! Create AST for:
        ! interface
        !   function compute() result(y)
        !     real :: y
        !   end function
        ! end interface

        ! Create function declaration inside interface (no body for interface)
        func_decl_idx = push_function_def(arena, "compute", &
                                          param_indices=[integer ::], &
                                          return_type="real", &
                                          body_indices=[integer ::])

        ! Create interface block
        interface_idx = push_interface_block(arena, "", [func_decl_idx])

        prog_idx = push_program(arena, "test_interface", [interface_idx])

        ! Configure backend for compile mode
        backend_opts%compile_mode = .true.

        ! Create MLIR backend
        call create_backend("mlir", backend, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, "FAIL: Error creating backend:", trim(error_msg)
            return
        end if

        ! Generate MLIR code
        call backend%generate_code(arena, prog_idx, backend_opts, mlir_code, &
                                   error_msg)

        ! Check if interface block generates proper MLIR function declaration
        if (index(mlir_code, "func.func private @compute") > 0 .and. &
            index(mlir_code, "-> f32") > 0) then
            print *, "PASS: Interface block generates proper MLIR"
            print *, "  Generated function declaration with correct signature"
            passed = .true.
        else
            print *, "FAIL: No interface block support"
            print *, "  Expected: func.func private @compute declaration"
            print *, "  Got: ", trim(mlir_code)
            passed = .false.
        end if
    end function test_simple_interface_block

    function test_generic_interface() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        type(backend_options_t) :: backend_opts
        character(len=:), allocatable :: mlir_code
        character(len=256) :: error_msg
        integer :: prog_idx, interface_idx, func1_idx, func2_idx

        print *, "Testing generic interface..."

        passed = .false.

        ! Initialize arena
        arena = create_ast_stack()

        ! Create AST for:
        ! interface add
        !   function add_int() result(c)
        !     integer :: c
        !   end function
        !   function add_real() result(c)
        !     real :: c
        !   end function
        ! end interface

        ! Create function declarations without parameters for now
        func1_idx = push_function_def(arena, "add_int", &
                                      param_indices=[integer ::], &
                                      return_type="integer", &
                                      body_indices=[integer ::])
        func2_idx = push_function_def(arena, "add_real", &
                                      param_indices=[integer ::], &
                                      return_type="real", &
                                      body_indices=[integer ::])

        interface_idx = push_interface_block(arena, "add", [func1_idx, func2_idx])

        prog_idx = push_program(arena, "test_generic", [interface_idx])

        ! Configure backend for compile mode
        backend_opts%compile_mode = .true.

        ! Create MLIR backend
        call create_backend("mlir", backend, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, "FAIL: Error creating backend:", trim(error_msg)
            return
        end if

        ! Generate MLIR code
        call backend%generate_code(arena, prog_idx, backend_opts, mlir_code, &
                                   error_msg)

        ! Check if generic interface generates multiple function declarations
        if (index(mlir_code, "func.func private @add_int") > 0 .and. &
            index(mlir_code, "func.func private @add_real") > 0) then
            print *, "PASS: Generic interface generates proper MLIR"
            print *, "  Generated multiple function declarations"
            passed = .true.
        else
            print *, "FAIL: No generic interface support"
            print *, "  Expected: Multiple function declarations for overloads"
            print *, "  Got: ", trim(mlir_code)
            passed = .false.
        end if
    end function test_generic_interface

    function test_operator_interface() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        type(backend_options_t) :: backend_opts
        character(len=:), allocatable :: mlir_code
        character(len=256) :: error_msg
        integer :: prog_idx, interface_idx, func_idx

        print *, "Testing operator interface..."

        passed = .false.

        ! Initialize arena
        arena = create_ast_stack()

        ! Create AST for:
        ! interface operator(+)
        !   function vector_add() result(c)
        !     type(vector) :: c
        !   end function
        ! end interface

        ! Create function declaration without parameters for now
        func_idx = push_function_def(arena, "vector_add", &
                                     param_indices=[integer ::], &
                                     return_type="vector", &
                                     body_indices=[integer ::])

        interface_idx = push_interface_block(arena, "operator(+)", [func_idx])

        prog_idx = push_program(arena, "test_operator", [interface_idx])

        ! Configure backend for compile mode
        backend_opts%compile_mode = .true.

        ! Create MLIR backend
        call create_backend("mlir", backend, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, "FAIL: Error creating backend:", trim(error_msg)
            return
        end if

        ! Generate MLIR code
        call backend%generate_code(arena, prog_idx, backend_opts, mlir_code, &
                                   error_msg)

        ! Check if operator interface generates proper MLIR
        if (index(mlir_code, "func.func private @vector_add") > 0) then
            print *, "PASS: Operator interface generates proper MLIR"
            print *, "  Generated operator function declaration"
            passed = .true.
        else
            print *, "FAIL: No operator interface support"
            print *, "  Expected: Function declaration for operator overload"
            print *, "  Got: ", trim(mlir_code)
            passed = .false.
        end if
    end function test_operator_interface

end program test_interface_blocks
