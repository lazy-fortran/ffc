module mlir_c_compat
    use iso_c_binding
    use mlir_c_core
    use mlir_c_types
    implicit none
    private

    public :: create_integer_type, create_float_type
    public :: create_reference_type, create_array_type

contains

    function create_integer_type(context, bit_width) result(ty)
        type(mlir_context_t), intent(in) :: context
        integer, intent(in) :: bit_width
        type(mlir_type_t) :: ty
        ty = mlir_integer_type_get(context, bit_width)
    end function create_integer_type

    function create_float_type(context, bit_width) result(ty)
        type(mlir_context_t), intent(in) :: context
        integer, intent(in) :: bit_width
        type(mlir_type_t) :: ty
        select case (bit_width)
        case (16)
            ty = mlir_f16_type_get(context)
        case (32)
            ty = mlir_f32_type_get(context)
        case (64)
            ty = mlir_f64_type_get(context)
        case default
            ty = mlir_f64_type_get(context)
        end select
    end function create_float_type

    function create_reference_type(context, element_type) result(ty)
        type(mlir_context_t), intent(in) :: context
        type(mlir_type_t), intent(in) :: element_type
        type(mlir_type_t) :: ty
        integer(c_int64_t), dimension(0) :: empty_shape
        ty = mlir_memref_type_contiguous_get(element_type, empty_shape)
    end function create_reference_type

    function create_array_type(context, element_type, shape) result(ty)
        type(mlir_context_t), intent(in) :: context
        type(mlir_type_t), intent(in) :: element_type
        integer(c_int64_t), intent(in) :: shape(:)
        type(mlir_type_t) :: ty
        ty = mlir_ranked_tensor_type_get(shape, element_type)
    end function create_array_type

end module mlir_c_compat
