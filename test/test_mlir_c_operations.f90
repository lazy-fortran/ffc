program test_mlir_c_operations
    use, intrinsic :: iso_c_binding, only: c_int64_t, c_associated
    use mlir_c_core
    use mlir_c_types
    use mlir_c_attributes
    use mlir_c_operations
    use mlir_c_dialects
    implicit none

    integer :: n_tests, n_passed
    n_tests = 0
    n_passed = 0

    call test_operation_state_creation()
    call test_named_attribute()
    call test_value_accessors()

    print "(A)", ""
    print "(A,I0,A,I0,A)", "Results: ", n_passed, "/", n_tests, " tests passed"
    if (n_passed == n_tests) then
        print "(A)", "PASS: test_mlir_c_operations"
    else
        error stop "FAIL: test_mlir_c_operations"
    end if

contains

    subroutine assert(condition, test_name)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: test_name

        n_tests = n_tests + 1
        if (condition) then
            n_passed = n_passed + 1
            print "(A,A)", "  PASS: ", test_name
        else
            print "(A,A)", "  FAIL: ", test_name
        end if
    end subroutine assert

    subroutine test_operation_state_creation()
        type(mlir_context_t) :: ctx
        type(mlir_location_t) :: loc
        type(mlir_operation_state_t) :: state
        character(len=*), parameter :: op_name = "builtin.module"

        print "(A)", "test_operation_state_creation:"
        ctx = mlir_context_create()
        loc = mlir_location_unknown_get(ctx)

        call mlir_operation_state_get(state, op_name, loc)
        call assert(state%n_results == 0, "state has 0 results initially")
        call assert(state%n_operands == 0, "state has 0 operands initially")
        call assert(state%n_regions == 0, "state has 0 regions initially")
        call assert(state%n_attributes == 0, "state has 0 attributes initially")

        call mlir_context_destroy(ctx)
    end subroutine test_operation_state_creation

    subroutine test_named_attribute()
        type(mlir_context_t) :: ctx
        type(mlir_type_t) :: i32_ty
        type(mlir_identifier_t) :: ident
        type(mlir_attribute_t) :: attr
        type(mlir_named_attribute_t) :: named_attr
        character(len=*), parameter :: attr_name = "value"

        print "(A)", "test_named_attribute:"
        ctx = mlir_context_create()

        ident = mlir_identifier_get(ctx, attr_name)
        i32_ty = mlir_integer_type_get(ctx, 32)
        attr = mlir_integer_attr_get(i32_ty, 42_c_int64_t)

        named_attr = mlir_named_attribute_get(ident, attr)
        call assert(c_associated(named_attr%name%ptr), &
            "named attr has identifier")
        call assert(.not. mlir_attribute_is_null(named_attr%attribute), &
            "named attr has attribute")

        call mlir_context_destroy(ctx)
    end subroutine test_named_attribute

    subroutine test_value_accessors()
        type(mlir_value_t) :: val

        print "(A)", "test_value_accessors:"
        call assert(mlir_value_is_null(val), "default value is null")
    end subroutine test_value_accessors

end program test_mlir_c_operations
