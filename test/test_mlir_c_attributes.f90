program test_mlir_c_attributes
    use, intrinsic :: iso_c_binding, only: c_int64_t, c_double
    use, intrinsic :: iso_fortran_env, only: real64
    use mlir_c_core
    use mlir_c_types
    use mlir_c_attributes
    implicit none

    integer :: n_tests, n_passed
    n_tests = 0
    n_passed = 0

    call test_string_attr()
    call test_integer_attr()
    call test_float_attr()
    call test_bool_attr()
    call test_type_attr()
    call test_unit_attr()
    call test_array_attr()
    call test_flat_symbol_ref_attr()
    call test_attribute_equality()

    print "(A)", ""
    print "(A,I0,A,I0,A)", "Results: ", n_passed, "/", n_tests, " tests passed"
    if (n_passed == n_tests) then
        print "(A)", "PASS: test_mlir_c_attributes"
    else
        error stop "FAIL: test_mlir_c_attributes"
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

    subroutine test_string_attr()
        type(mlir_context_t) :: ctx
        type(mlir_attribute_t) :: attr
        character(len=*), parameter :: test_str = "hello"

        print "(A)", "test_string_attr:"
        ctx = mlir_context_create()

        attr = mlir_string_attr_get(ctx, test_str)
        call assert(.not. mlir_attribute_is_null(attr), "string attr is not null")
        call assert(mlir_attribute_is_a_string(attr), "attr is a string attr")

        call mlir_context_destroy(ctx)
    end subroutine test_string_attr

    subroutine test_integer_attr()
        type(mlir_context_t) :: ctx
        type(mlir_type_t) :: i64_ty
        type(mlir_attribute_t) :: attr
        integer(c_int64_t) :: value

        print "(A)", "test_integer_attr:"
        ctx = mlir_context_create()

        i64_ty = mlir_integer_type_get(ctx, 64)
        attr = mlir_integer_attr_get(i64_ty, 42_c_int64_t)
        call assert(.not. mlir_attribute_is_null(attr), "integer attr is not null")
        call assert(mlir_attribute_is_a_integer(attr), "attr is an integer attr")

        value = mlir_integer_attr_get_value_int(attr)
        call assert(value == 42, "integer attr value is 42")

        call mlir_context_destroy(ctx)
    end subroutine test_integer_attr

    subroutine test_float_attr()
        type(mlir_context_t) :: ctx
        type(mlir_type_t) :: f64_ty
        type(mlir_attribute_t) :: attr
        real(c_double) :: value

        print "(A)", "test_float_attr:"
        ctx = mlir_context_create()

        f64_ty = mlir_f64_type_get(ctx)
        attr = mlir_float_attr_double_get(ctx, f64_ty, 3.14_real64)
        call assert(.not. mlir_attribute_is_null(attr), "float attr is not null")
        call assert(mlir_attribute_is_a_float(attr), "attr is a float attr")

        value = mlir_float_attr_get_value_double(attr)
        call assert(abs(value - 3.14_real64) < 1.0e-10_real64, &
            "float attr value is 3.14")

        call mlir_context_destroy(ctx)
    end subroutine test_float_attr

    subroutine test_bool_attr()
        type(mlir_context_t) :: ctx
        type(mlir_attribute_t) :: true_attr, false_attr

        print "(A)", "test_bool_attr:"
        ctx = mlir_context_create()

        true_attr = mlir_bool_attr_get(ctx, .true.)
        call assert(.not. mlir_attribute_is_null(true_attr), &
            "bool true attr is not null")
        call assert(mlir_attribute_is_a_bool(true_attr), "attr is a bool attr")
        call assert(mlir_bool_attr_get_value(true_attr), "bool true attr is true")

        false_attr = mlir_bool_attr_get(ctx, .false.)
        call assert(.not. mlir_bool_attr_get_value(false_attr), &
            "bool false attr is false")

        call mlir_context_destroy(ctx)
    end subroutine test_bool_attr

    subroutine test_type_attr()
        type(mlir_context_t) :: ctx
        type(mlir_type_t) :: i32_ty, extracted_ty
        type(mlir_attribute_t) :: attr

        print "(A)", "test_type_attr:"
        ctx = mlir_context_create()

        i32_ty = mlir_integer_type_get(ctx, 32)
        attr = mlir_type_attr_get(i32_ty)
        call assert(.not. mlir_attribute_is_null(attr), "type attr is not null")
        call assert(mlir_attribute_is_a_type(attr), "attr is a type attr")

        extracted_ty = mlir_type_attr_get_value(attr)
        call assert(mlir_type_equal(extracted_ty, i32_ty), &
            "type attr value is i32")

        call mlir_context_destroy(ctx)
    end subroutine test_type_attr

    subroutine test_unit_attr()
        type(mlir_context_t) :: ctx
        type(mlir_attribute_t) :: attr

        print "(A)", "test_unit_attr:"
        ctx = mlir_context_create()

        attr = mlir_unit_attr_get(ctx)
        call assert(.not. mlir_attribute_is_null(attr), "unit attr is not null")
        call assert(mlir_attribute_is_a_unit(attr), "attr is a unit attr")

        call mlir_context_destroy(ctx)
    end subroutine test_unit_attr

    subroutine test_array_attr()
        type(mlir_context_t) :: ctx
        type(mlir_type_t) :: i32_ty
        type(mlir_attribute_t) :: attr, elem
        type(mlir_attribute_t) :: elements(3)
        integer :: i

        print "(A)", "test_array_attr:"
        ctx = mlir_context_create()

        i32_ty = mlir_integer_type_get(ctx, 32)
        do i = 1, 3
            elements(i) = mlir_integer_attr_get(i32_ty, int(i, c_int64_t))
        end do

        attr = mlir_array_attr_get(ctx, elements)
        call assert(.not. mlir_attribute_is_null(attr), "array attr is not null")
        call assert(mlir_attribute_is_a_array(attr), "attr is an array attr")
        call assert(mlir_array_attr_get_num_elements(attr) == 3, &
            "array has 3 elements")

        elem = mlir_array_attr_get_element(attr, 0)
        call assert(mlir_integer_attr_get_value_int(elem) == 1, &
            "first element is 1")

        call mlir_context_destroy(ctx)
    end subroutine test_array_attr

    subroutine test_flat_symbol_ref_attr()
        type(mlir_context_t) :: ctx
        type(mlir_attribute_t) :: attr
        character(len=*), parameter :: sym_name = "my_symbol"

        print "(A)", "test_flat_symbol_ref_attr:"
        ctx = mlir_context_create()

        attr = mlir_flat_symbol_ref_attr_get(ctx, sym_name)
        call assert(.not. mlir_attribute_is_null(attr), &
            "flat symbol ref attr is not null")
        call assert(mlir_attribute_is_a_flat_symbol_ref(attr), &
            "attr is a flat symbol ref attr")

        call mlir_context_destroy(ctx)
    end subroutine test_flat_symbol_ref_attr

    subroutine test_attribute_equality()
        type(mlir_context_t) :: ctx
        type(mlir_type_t) :: i32_ty
        type(mlir_attribute_t) :: a1, a2, a3

        print "(A)", "test_attribute_equality:"
        ctx = mlir_context_create()

        i32_ty = mlir_integer_type_get(ctx, 32)
        a1 = mlir_integer_attr_get(i32_ty, 42_c_int64_t)
        a2 = mlir_integer_attr_get(i32_ty, 42_c_int64_t)
        a3 = mlir_integer_attr_get(i32_ty, 99_c_int64_t)

        call assert(mlir_attribute_equal(a1, a2), "same value attrs are equal")
        call assert(.not. mlir_attribute_equal(a1, a3), &
            "different value attrs are not equal")

        call mlir_context_destroy(ctx)
    end subroutine test_attribute_equality

end program test_mlir_c_attributes
