program test_assignment_debug
    use ast_core, only: ast_arena_t, create_ast_stack
    use ast_factory
    use backend_interface
    use backend_factory
    implicit none

    logical :: passed
    type(ast_arena_t) :: arena
    class(backend_t), allocatable :: backend
    type(backend_options_t) :: backend_opts
    character(len=:), allocatable :: mlir_code
    character(len=256) :: error_msg
    integer :: prog_idx, interface_idx, func_idx, assign_idx
    integer :: var1_idx, var2_idx, decl1_idx, decl2_idx

    print *, "=== DEBUG: Assignment Operator Overloading ==="

    passed = .false.
    arena = create_ast_stack()

    ! Create subroutine for assignment operator overload
    func_idx = push_subroutine_def(arena, "vector_assign", &
                                   param_indices=[integer ::], &
                                   body_indices=[integer ::])
    print *, "Created subroutine at index:", func_idx

    ! Create assignment interface
    interface_idx = push_interface_block(arena, "assignment(=)", [func_idx])
    print *, "Created interface at index:", interface_idx

    ! Create variable declarations first
    decl1_idx = push_declaration(arena, "vector", "v1")
    decl2_idx = push_declaration(arena, "vector", "v2")
    print *, "Created declarations at indices:", decl1_idx, decl2_idx

    ! Create variable identifiers
    var1_idx = push_identifier(arena, "v1")
    var2_idx = push_identifier(arena, "v2")
    print *, "Created variables at indices:", var1_idx, var2_idx

    ! Create assignment: v1 = v2
    assign_idx = push_assignment(arena, var1_idx, var2_idx)
    print *, "Created assignment at index:", assign_idx

    prog_idx = push_program(arena, "test_assignment", [interface_idx, decl1_idx, decl2_idx, assign_idx])
    print *, "Created program at index:", prog_idx

    ! Print arena contents
    print *, "Arena size:", arena%size

    ! Configure backend for compile mode
    backend_opts%compile_mode = .true.

    ! Create MLIR backend
    call create_backend("mlir", backend, error_msg)
    if (len_trim(error_msg) > 0) then
        print *, "FAIL: Error creating backend:", trim(error_msg)
        stop 1
    end if

    ! Generate MLIR code
    call backend%generate_code(arena, prog_idx, backend_opts, mlir_code, &
                               error_msg)

    print *, "MLIR code length:", len(mlir_code)
    if (len_trim(mlir_code) > 0) then
        print *, "MLIR code:"
        print *, mlir_code
    end if

    if (len_trim(error_msg) > 0) then
        print *, "Error message:", trim(error_msg)
    end if

    stop 0

end program test_assignment_debug
