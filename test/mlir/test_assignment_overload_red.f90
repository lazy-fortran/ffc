program test_assignment_overload_red
    use ast_core, only: ast_arena_t, create_ast_arena
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
    integer :: var1_idx, var2_idx

    print *, "=== RED phase: Assignment Operator Overloading Test ==="

    passed = .false.
    arena = create_ast_arena()

    ! Create AST for:
    ! interface assignment(=)
    !   subroutine vector_assign(lhs, rhs)
    !     type(vector), intent(out) :: lhs
    !     type(vector), intent(in) :: rhs
    !   end subroutine
    ! end interface
    !
    ! program test
    !   type(vector) :: v1, v2
    !   v1 = v2  ! Should resolve to vector_assign
    ! end program

    ! Create subroutine for assignment operator overload
    func_idx = push_subroutine_def(arena, "vector_assign", &
                                   param_indices=[integer ::], &
                                   body_indices=[integer ::])

    ! Create assignment interface
    interface_idx = push_interface_block(arena, "assignment(=)", [func_idx])

    ! Create variable identifiers
    var1_idx = push_identifier(arena, "v1")
    var2_idx = push_identifier(arena, "v2")

    ! Create assignment: v1 = v2
    assign_idx = push_assignment(arena, var1_idx, var2_idx)

    prog_idx = push_program(arena, "test_assignment", [interface_idx, assign_idx])

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

    ! Check if assignment operator overload resolves to subroutine call
    ! This SHOULD FAIL in RED phase
    if (index(mlir_code, "call @vector_assign") > 0) then
        print *, "PASS: Assignment operator overload works (unexpected in RED phase)"
        passed = .true.
    else
   print *, "FAIL: Assignment operator overload not implemented (expected in RED phase)"
        print *, "  Expected: call @vector_assign for = operator"
        print *, "  Got empty or wrong MLIR output"
        passed = .false.
    end if

    if (.not. passed) then
        print *, "RED PHASE CONFIRMED: Assignment overloading needs implementation"
        stop 0  ! This is expected failure in TDD RED phase
    else
        print *, "RED PHASE UNEXPECTED: Assignment overloading already works"
        stop 1
    end if

end program test_assignment_overload_red
