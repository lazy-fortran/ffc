program test_operator_overloading
    use ast_core, only: ast_arena_t, create_ast_arena, LITERAL_INTEGER, LITERAL_REAL
    use ast_factory
    use backend_interface
    use backend_factory
    use temp_utils
    implicit none

    logical :: all_passed = .true.

    print *, "=== Testing Operator Overloading (Call Resolution) ==="
    print *, ""

    all_passed = all_passed .and. test_arithmetic_operator_overload()
    all_passed = all_passed .and. test_comparison_operator_overload()
    all_passed = all_passed .and. test_assignment_operator_overload()

    if (all_passed) then
        print *, ""
        print *, "All operator overloading tests passed!"
        stop 0
    else
        print *, ""
        print *, "Some operator overloading tests failed!"
        stop 1
    end if

contains

    function test_arithmetic_operator_overload() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        type(backend_options_t) :: backend_opts
        character(len=:), allocatable :: mlir_code
        character(len=256) :: error_msg
        integer :: prog_idx, interface_idx, func_idx, add_expr_idx
        integer :: var1_idx, var2_idx

        print *, "Testing arithmetic operator overloading (+)..."

        passed = .false.

        ! Initialize arena
        arena = create_ast_arena()

        ! Create AST for:
        ! interface operator(+)
        !   function vector_add(a, b) result(c)
        !     type(vector), intent(in) :: a, b
        !     type(vector) :: c
        !   end function
        ! end interface
        !
        ! program test
        !   type(vector) :: v1, v2, result
        !   result = v1 + v2  ! Should resolve to vector_add
        ! end program

        ! Create function for operator overload
        func_idx = push_function_def(arena, "vector_add", &
                                     param_indices=[integer ::], &
                                     return_type="vector", &
                                     body_indices=[integer ::])

        ! Create operator interface
        interface_idx = push_interface_block(arena, "operator(+)", [func_idx])

        ! Create variable declarations for the operands
        var1_idx = push_identifier(arena, "v1")
        var2_idx = push_identifier(arena, "v2")

        ! Create binary expression: v1 + v2
        add_expr_idx = push_binary_op(arena, var1_idx, var2_idx, "+")

        prog_idx = push_program(arena, "test_operator", [interface_idx, add_expr_idx])

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

        ! Check if operator overload resolves to function call
        if (index(mlir_code, "call @vector_add") > 0) then
            print *, "PASS: Arithmetic operator overload resolves to function call"
            print *, "  + operator resolved to vector_add function"
            passed = .true.
        else
            print *, "FAIL: No arithmetic operator overload resolution"
            print *, "  Expected: call @vector_add for + operator"
            if (len_trim(mlir_code) < 200) then
                print *, "  Got: ", trim(mlir_code)
            else
                print *, "  Got long MLIR output (showing end):"
                print *, "  ", mlir_code(len(mlir_code) - 200:)
            end if
            passed = .false.
        end if
    end function test_arithmetic_operator_overload

    function test_comparison_operator_overload() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        type(backend_options_t) :: backend_opts
        character(len=:), allocatable :: mlir_code
        character(len=256) :: error_msg
        integer :: prog_idx, interface_idx, func_idx, eq_expr_idx
        integer :: var1_idx, var2_idx

        print *, "Testing comparison operator overloading (==)..."

        passed = .false.

        ! Initialize arena
        arena = create_ast_arena()

        ! Create AST for:
        ! interface operator(==)
        !   function vector_equal(a, b) result(c)
        !     type(vector), intent(in) :: a, b
        !     logical :: c
        !   end function
        ! end interface
        !
        ! program test
        !   type(vector) :: v1, v2
        !   logical :: result
        !   result = v1 == v2  ! Should resolve to vector_equal
        ! end program

        ! Create function for comparison operator overload
        func_idx = push_function_def(arena, "vector_equal", &
                                     param_indices=[integer ::], &
                                     return_type="logical", &
                                     body_indices=[integer ::])

        ! Create operator interface
        interface_idx = push_interface_block(arena, "operator(==)", [func_idx])

        ! Create variable identifiers
        var1_idx = push_identifier(arena, "v1")
        var2_idx = push_identifier(arena, "v2")

        ! Create binary expression: v1 == v2
        eq_expr_idx = push_binary_op(arena, var1_idx, var2_idx, "==")

        prog_idx = push_program(arena, "test_comparison", [interface_idx, eq_expr_idx])

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

        ! Check if comparison operator overload resolves to function call
        if (index(mlir_code, "call @vector_equal") > 0) then
            print *, "PASS: Comparison operator overload resolves to function call"
            print *, "  == operator resolved to vector_equal function"
            passed = .true.
        else
            print *, "FAIL: No comparison operator overload resolution"
            print *, "  Expected: call @vector_equal for == operator"
            print *, "  Got: ", trim(mlir_code)
            passed = .false.
        end if
    end function test_comparison_operator_overload

    function test_assignment_operator_overload() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        type(backend_options_t) :: backend_opts
        character(len=:), allocatable :: mlir_code
        character(len=256) :: error_msg
        integer :: prog_idx, interface_idx, func_idx, assign_idx
        integer :: var1_idx, var2_idx, decl1_idx, decl2_idx

        print *, "Testing assignment operator overloading (=)..."

        passed = .false.

        ! Initialize arena
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

        ! Check if assignment operator overload resolves to subroutine call
        if (index(mlir_code, "call @vector_assign") > 0) then
            print *, "PASS: Assignment operator overload resolves to subroutine call"
            print *, "  = operator resolved to vector_assign subroutine"
            passed = .true.
        else
            print *, "FAIL: No assignment operator overload resolution"
            print *, "  Expected: call @vector_assign for = operator"
            print *, "  Got: ", trim(mlir_code)
            passed = .false.
        end if
    end function test_assignment_operator_overload

end program test_operator_overloading
