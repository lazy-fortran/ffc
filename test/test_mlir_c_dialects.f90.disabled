program test_mlir_c_dialects
    use, intrinsic :: iso_c_binding, only: c_associated
    use mlir_c_core
    use mlir_c_dialects
    implicit none

    integer :: n_tests, n_passed
    n_tests = 0
    n_passed = 0

    call test_register_all_dialects()
    call test_get_dialect_handles()
    call test_load_specific_dialects()

    print "(A)", ""
    print "(A,I0,A,I0,A)", "Results: ", n_passed, "/", n_tests, " tests passed"
    if (n_passed == n_tests) then
        print "(A)", "PASS: test_mlir_c_dialects"
    else
        error stop "FAIL: test_mlir_c_dialects"
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

    subroutine test_register_all_dialects()
        type(mlir_context_t) :: ctx
        type(mlir_dialect_registry_t) :: registry
        integer :: num_dialects_before, num_dialects_after

        print "(A)", "test_register_all_dialects:"
        ctx = mlir_context_create()
        num_dialects_before = mlir_context_get_num_loaded_dialects(ctx)

        registry = mlir_dialect_registry_create()
        call mlir_register_all_dialects(registry)
        call mlir_context_append_dialect_registry(ctx, registry)
        call mlir_context_load_all_available_dialects(ctx)

        num_dialects_after = mlir_context_get_num_loaded_dialects(ctx)
        call assert(num_dialects_after > num_dialects_before, &
            "more dialects loaded after register_all")

        call mlir_dialect_registry_destroy(registry)
        call mlir_context_destroy(ctx)
    end subroutine test_register_all_dialects

    subroutine test_get_dialect_handles()
        type(mlir_dialect_handle_t) :: func_handle, arith_handle
        type(mlir_dialect_handle_t) :: scf_handle, llvm_handle

        print "(A)", "test_get_dialect_handles:"

        func_handle = mlir_get_dialect_handle_func()
        call assert(c_associated(func_handle%ptr), "func handle is not null")

        arith_handle = mlir_get_dialect_handle_arith()
        call assert(c_associated(arith_handle%ptr), "arith handle is not null")

        scf_handle = mlir_get_dialect_handle_scf()
        call assert(c_associated(scf_handle%ptr), "scf handle is not null")

        llvm_handle = mlir_get_dialect_handle_llvm()
        call assert(c_associated(llvm_handle%ptr), "llvm handle is not null")
    end subroutine test_get_dialect_handles

    subroutine test_load_specific_dialects()
        type(mlir_context_t) :: ctx
        type(mlir_dialect_registry_t) :: registry
        type(mlir_dialect_handle_t) :: func_handle, arith_handle
        type(mlir_dialect_t) :: func_dialect, arith_dialect

        print "(A)", "test_load_specific_dialects:"
        ctx = mlir_context_create()
        registry = mlir_dialect_registry_create()

        func_handle = mlir_get_dialect_handle_func()
        arith_handle = mlir_get_dialect_handle_arith()

        call mlir_dialect_handle_insert_dialect(func_handle, registry)
        call mlir_dialect_handle_insert_dialect(arith_handle, registry)
        call mlir_context_append_dialect_registry(ctx, registry)

        func_dialect = mlir_dialect_handle_load_dialect(func_handle, ctx)
        call assert(.not. mlir_dialect_is_null(func_dialect), &
            "func dialect loaded")

        arith_dialect = mlir_dialect_handle_load_dialect(arith_handle, ctx)
        call assert(.not. mlir_dialect_is_null(arith_dialect), &
            "arith dialect loaded")

        call mlir_dialect_registry_destroy(registry)
        call mlir_context_destroy(ctx)
    end subroutine test_load_specific_dialects

end program test_mlir_c_dialects
