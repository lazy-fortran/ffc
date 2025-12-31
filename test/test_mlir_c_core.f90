program test_mlir_c_core
    use, intrinsic :: iso_c_binding, only: c_null_ptr, c_associated
    use mlir_c_core
    implicit none

    integer :: n_tests, n_passed
    n_tests = 0
    n_passed = 0

    call test_context_create_destroy()
    call test_location_unknown()
    call test_location_file_line_col()
    call test_module_create_destroy()
    call test_module_get_body()
    call test_dialect_registry()
    call test_region_create_destroy()
    call test_block_create()

    print "(A)", ""
    print "(A,I0,A,I0,A)", "Results: ", n_passed, "/", n_tests, " tests passed"
    if (n_passed == n_tests) then
        print "(A)", "PASS: test_mlir_c_core"
    else
        error stop "FAIL: test_mlir_c_core"
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

    subroutine test_context_create_destroy()
        type(mlir_context_t) :: ctx

        print "(A)", "test_context_create_destroy:"
        ctx = mlir_context_create()
        call assert(.not. mlir_context_is_null(ctx), "context created is not null")
        call mlir_context_destroy(ctx)
        call assert(mlir_context_is_null(ctx), "context destroyed is null")
    end subroutine test_context_create_destroy

    subroutine test_location_unknown()
        type(mlir_context_t) :: ctx
        type(mlir_location_t) :: loc

        print "(A)", "test_location_unknown:"
        ctx = mlir_context_create()
        loc = mlir_location_unknown_get(ctx)
        call assert(.not. mlir_location_is_null(loc), "unknown location is not null")
        call mlir_context_destroy(ctx)
    end subroutine test_location_unknown

    subroutine test_location_file_line_col()
        type(mlir_context_t) :: ctx
        type(mlir_location_t) :: loc
        character(len=*), parameter :: filename = "test.f90"

        print "(A)", "test_location_file_line_col:"
        ctx = mlir_context_create()
        loc = mlir_location_file_line_col_get(ctx, filename, 10, 5)
        call assert(.not. mlir_location_is_null(loc), "file location is not null")
        call mlir_context_destroy(ctx)
    end subroutine test_location_file_line_col

    subroutine test_module_create_destroy()
        type(mlir_context_t) :: ctx
        type(mlir_location_t) :: loc
        type(mlir_module_t) :: mod

        print "(A)", "test_module_create_destroy:"
        ctx = mlir_context_create()
        loc = mlir_location_unknown_get(ctx)
        mod = mlir_module_create_empty(loc)
        call assert(.not. mlir_module_is_null(mod), "module created is not null")
        call mlir_module_destroy(mod)
        call assert(mlir_module_is_null(mod), "module destroyed is null")
        call mlir_context_destroy(ctx)
    end subroutine test_module_create_destroy

    subroutine test_module_get_body()
        type(mlir_context_t) :: ctx
        type(mlir_location_t) :: loc
        type(mlir_module_t) :: mod
        type(mlir_block_t) :: body
        type(mlir_operation_t) :: module_op

        print "(A)", "test_module_get_body:"
        ctx = mlir_context_create()
        loc = mlir_location_unknown_get(ctx)
        mod = mlir_module_create_empty(loc)

        body = mlir_module_get_body(mod)
        call assert(.not. mlir_block_is_null(body), "module body is not null")

        module_op = mlir_module_get_operation(mod)
        call assert(.not. mlir_operation_is_null(module_op), &
            "module operation is not null")

        call mlir_module_destroy(mod)
        call mlir_context_destroy(ctx)
    end subroutine test_module_get_body

    subroutine test_dialect_registry()
        type(mlir_context_t) :: ctx
        type(mlir_dialect_registry_t) :: registry

        print "(A)", "test_dialect_registry:"
        ctx = mlir_context_create()
        registry = mlir_dialect_registry_create()
        call assert(c_associated(registry%ptr), "registry created is not null")
        call mlir_context_append_dialect_registry(ctx, registry)
        call mlir_dialect_registry_destroy(registry)
        call mlir_context_destroy(ctx)
    end subroutine test_dialect_registry

    subroutine test_region_create_destroy()
        type(mlir_region_t) :: region

        print "(A)", "test_region_create_destroy:"
        region = mlir_region_create()
        call assert(.not. mlir_region_is_null(region), "region created is not null")
        call mlir_region_destroy(region)
        call assert(mlir_region_is_null(region), "region destroyed is null")
    end subroutine test_region_create_destroy

    subroutine test_block_create()
        type(mlir_block_t) :: block

        print "(A)", "test_block_create:"
        block = mlir_block_create()
        call assert(.not. mlir_block_is_null(block), "block created is not null")
    end subroutine test_block_create

end program test_mlir_c_core
