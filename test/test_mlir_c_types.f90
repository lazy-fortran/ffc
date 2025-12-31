program test_mlir_c_types
    use, intrinsic :: iso_c_binding, only: c_int64_t
    use mlir_c_core
    use mlir_c_types
    implicit none

    integer :: n_tests, n_passed
    n_tests = 0
    n_passed = 0

    call test_integer_types()
    call test_index_type()
    call test_float_types()
    call test_none_type()
    call test_complex_type()
    call test_function_type()
    call test_type_equality()

    print "(A)", ""
    print "(A,I0,A,I0,A)", "Results: ", n_passed, "/", n_tests, " tests passed"
    if (n_passed == n_tests) then
        print "(A)", "PASS: test_mlir_c_types"
    else
        error stop "FAIL: test_mlir_c_types"
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

    subroutine test_integer_types()
        type(mlir_context_t) :: ctx
        type(mlir_type_t) :: i32_ty, i64_ty, si32_ty, ui32_ty

        print "(A)", "test_integer_types:"
        ctx = mlir_context_create()

        i32_ty = mlir_integer_type_get(ctx, 32)
        call assert(.not. mlir_type_is_null(i32_ty), "i32 type is not null")
        call assert(mlir_type_is_a_integer(i32_ty), "i32 is an integer type")
        call assert(mlir_integer_type_get_width(i32_ty) == 32, "i32 has width 32")

        i64_ty = mlir_integer_type_get(ctx, 64)
        call assert(mlir_integer_type_get_width(i64_ty) == 64, "i64 has width 64")

        si32_ty = mlir_integer_type_signed_get(ctx, 32)
        call assert(mlir_type_is_a_integer(si32_ty), "si32 is an integer type")

        ui32_ty = mlir_integer_type_unsigned_get(ctx, 32)
        call assert(mlir_type_is_a_integer(ui32_ty), "ui32 is an integer type")

        call mlir_context_destroy(ctx)
    end subroutine test_integer_types

    subroutine test_index_type()
        type(mlir_context_t) :: ctx
        type(mlir_type_t) :: idx_ty

        print "(A)", "test_index_type:"
        ctx = mlir_context_create()

        idx_ty = mlir_index_type_get(ctx)
        call assert(.not. mlir_type_is_null(idx_ty), "index type is not null")
        call assert(mlir_type_is_a_index(idx_ty), "index type is an index")
        call assert(.not. mlir_type_is_a_integer(idx_ty), &
            "index type is not an integer")

        call mlir_context_destroy(ctx)
    end subroutine test_index_type

    subroutine test_float_types()
        type(mlir_context_t) :: ctx
        type(mlir_type_t) :: f16_ty, f32_ty, f64_ty, bf16_ty

        print "(A)", "test_float_types:"
        ctx = mlir_context_create()

        f16_ty = mlir_f16_type_get(ctx)
        call assert(.not. mlir_type_is_null(f16_ty), "f16 type is not null")
        call assert(mlir_type_is_a_float(f16_ty), "f16 is a float type")
        call assert(mlir_float_type_get_width(f16_ty) == 16, "f16 has width 16")

        f32_ty = mlir_f32_type_get(ctx)
        call assert(mlir_type_is_a_float(f32_ty), "f32 is a float type")
        call assert(mlir_float_type_get_width(f32_ty) == 32, "f32 has width 32")

        f64_ty = mlir_f64_type_get(ctx)
        call assert(mlir_type_is_a_float(f64_ty), "f64 is a float type")
        call assert(mlir_float_type_get_width(f64_ty) == 64, "f64 has width 64")

        bf16_ty = mlir_bf16_type_get(ctx)
        call assert(mlir_type_is_a_float(bf16_ty), "bf16 is a float type")

        call mlir_context_destroy(ctx)
    end subroutine test_float_types

    subroutine test_none_type()
        type(mlir_context_t) :: ctx
        type(mlir_type_t) :: none_ty

        print "(A)", "test_none_type:"
        ctx = mlir_context_create()

        none_ty = mlir_none_type_get(ctx)
        call assert(.not. mlir_type_is_null(none_ty), "none type is not null")
        call assert(mlir_type_is_a_none(none_ty), "none type is a none")

        call mlir_context_destroy(ctx)
    end subroutine test_none_type

    subroutine test_complex_type()
        type(mlir_context_t) :: ctx
        type(mlir_type_t) :: f32_ty, complex_ty, elem_ty

        print "(A)", "test_complex_type:"
        ctx = mlir_context_create()

        f32_ty = mlir_f32_type_get(ctx)
        complex_ty = mlir_complex_type_get(f32_ty)
        call assert(.not. mlir_type_is_null(complex_ty), "complex type is not null")
        call assert(mlir_type_is_a_complex(complex_ty), "complex type is a complex")

        elem_ty = mlir_complex_type_get_element_type(complex_ty)
        call assert(mlir_type_equal(elem_ty, f32_ty), "complex element type is f32")

        call mlir_context_destroy(ctx)
    end subroutine test_complex_type

    subroutine test_function_type()
        type(mlir_context_t) :: ctx
        type(mlir_type_t) :: i32_ty, f32_ty, func_ty
        type(mlir_type_t) :: inputs(2), results(1)

        print "(A)", "test_function_type:"
        ctx = mlir_context_create()

        i32_ty = mlir_integer_type_get(ctx, 32)
        f32_ty = mlir_f32_type_get(ctx)

        inputs(1) = i32_ty
        inputs(2) = f32_ty
        results(1) = f32_ty

        func_ty = mlir_function_type_get(ctx, inputs, results)
        call assert(.not. mlir_type_is_null(func_ty), "function type is not null")
        call assert(mlir_type_is_a_function(func_ty), "function type is a function")
        call assert(mlir_function_type_get_num_inputs(func_ty) == 2, &
            "function has 2 inputs")
        call assert(mlir_function_type_get_num_results(func_ty) == 1, &
            "function has 1 result")

        call mlir_context_destroy(ctx)
    end subroutine test_function_type

    subroutine test_type_equality()
        type(mlir_context_t) :: ctx
        type(mlir_type_t) :: i32_a, i32_b, i64_ty

        print "(A)", "test_type_equality:"
        ctx = mlir_context_create()

        i32_a = mlir_integer_type_get(ctx, 32)
        i32_b = mlir_integer_type_get(ctx, 32)
        i64_ty = mlir_integer_type_get(ctx, 64)

        call assert(mlir_type_equal(i32_a, i32_b), "same i32 types are equal")
        call assert(.not. mlir_type_equal(i32_a, i64_ty), &
            "i32 and i64 are not equal")

        call mlir_context_destroy(ctx)
    end subroutine test_type_equality

end program test_mlir_c_types
