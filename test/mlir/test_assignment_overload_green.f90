program test_assignment_overload_green
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
    integer :: var1_idx, var2_idx, decl1_idx, decl2_idx

    print *, "=== GREEN phase: Assignment Operator Overloading Test ==="

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

    ! Create variable declarations first
    decl1_idx = push_declaration(arena, "vector", "v1")
    decl2_idx = push_declaration(arena, "vector", "v2")

    ! Create variable identifiers
    var1_idx = push_identifier(arena, "v1")
    var2_idx = push_identifier(arena, "v2")

    ! Create assignment: v1 = v2
    assign_idx = push_assignment(arena, var1_idx, var2_idx)

    prog_idx = push_program(arena, "test_assignment", [interface_idx, decl1_idx, decl2_idx, assign_idx])

    ! Configure backend for MLIR generation (not compile mode)
    backend_opts%compile_mode = .false.

    ! Create MLIR backend
    call create_backend("mlir", backend, error_msg)
    if (len_trim(error_msg) > 0) then
        print *, "FAIL: Error creating backend:", trim(error_msg)
        stop 1
    end if

    ! Generate MLIR code
    call backend%generate_code(arena, prog_idx, backend_opts, mlir_code, &
                               error_msg)

    ! Check for errors first
    if (len_trim(error_msg) > 0) then
        print *, "ERROR in generate_code:", trim(error_msg)
        stop 1
    end if

    ! DEBUG: Show full MLIR output for analysis
    print *, "=== FULL MLIR OUTPUT ==="
    if (allocated(mlir_code)) then
        print *, "MLIR code allocated, length:", len(mlir_code)
        if (len(mlir_code) > 0) then
            print *, mlir_code
        else
            print *, "(MLIR code is empty string)"
        end if
    else
        print *, "(MLIR code not allocated)"
    end if
    print *, "=== END MLIR OUTPUT ==="
    
    ! Check if assignment operator overload resolves to subroutine call
    if (index(mlir_code, "call @vector_assign") > 0) then
        print *, "PASS: Assignment operator overload works!"
        print *, "  = operator resolved to vector_assign subroutine"
        
        ! Check if it uses HLFIR operations
        if (index(mlir_code, "hlfir.assign") > 0) then
            print *, "  Uses HLFIR hlfir.assign: YES"
        else
            print *, "  Uses HLFIR hlfir.assign: NO (using fir.call instead)"
        end if
        
        if (index(mlir_code, "!fir.ref") > 0) then
            print *, "  Uses proper !fir.* types: YES"
        else
            print *, "  Uses proper !fir.* types: NO"
        end if
        
        passed = .true.
    else
        print *, "FAIL: Assignment operator overload still not working"
        print *, "  Expected: call @vector_assign for = operator"
        passed = .false.
    end if

    if (passed) then
        print *, "GREEN PHASE SUCCESS: Assignment overloading implemented"
        stop 0
    else
        print *, "GREEN PHASE FAILED: Assignment overloading needs more work"
        stop 1
    end if

end program test_assignment_overload_green
