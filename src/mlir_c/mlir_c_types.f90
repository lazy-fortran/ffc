module mlir_c_types
    use, intrinsic :: iso_c_binding, only: c_ptr, c_null_ptr, c_bool, c_int, &
        c_int64_t, c_intptr_t, c_associated, c_loc
    use mlir_c_core, only: mlir_context_t, mlir_type_t, mlir_location_t, &
        mlir_attribute_t
    implicit none
    private

    public :: mlir_integer_type_get
    public :: mlir_integer_type_signed_get
    public :: mlir_integer_type_unsigned_get
    public :: mlir_integer_type_get_width
    public :: mlir_type_is_a_integer

    public :: mlir_index_type_get
    public :: mlir_type_is_a_index

    public :: mlir_f16_type_get
    public :: mlir_f32_type_get
    public :: mlir_f64_type_get
    public :: mlir_bf16_type_get
    public :: mlir_type_is_a_float
    public :: mlir_float_type_get_width

    public :: mlir_none_type_get
    public :: mlir_type_is_a_none

    public :: mlir_complex_type_get
    public :: mlir_type_is_a_complex
    public :: mlir_complex_type_get_element_type

    public :: mlir_function_type_get
    public :: mlir_type_is_a_function
    public :: mlir_function_type_get_num_inputs
    public :: mlir_function_type_get_num_results
    public :: mlir_function_type_get_input
    public :: mlir_function_type_get_result

    public :: mlir_ranked_tensor_type_get
    public :: mlir_type_is_a_ranked_tensor

    public :: mlir_memref_type_contiguous_get
    public :: mlir_type_is_a_memref

    public :: mlir_type_is_null
    public :: mlir_type_equal
    public :: mlir_type_dump

    interface
        function mlirIntegerTypeGet(ctx, bitwidth) &
                bind(C, name="mlirIntegerTypeGet")
            import :: c_ptr, c_int
            type(c_ptr), value :: ctx
            integer(c_int), value :: bitwidth
            type(c_ptr) :: mlirIntegerTypeGet
        end function mlirIntegerTypeGet

        function mlirIntegerTypeSignedGet(ctx, bitwidth) &
                bind(C, name="mlirIntegerTypeSignedGet")
            import :: c_ptr, c_int
            type(c_ptr), value :: ctx
            integer(c_int), value :: bitwidth
            type(c_ptr) :: mlirIntegerTypeSignedGet
        end function mlirIntegerTypeSignedGet

        function mlirIntegerTypeUnsignedGet(ctx, bitwidth) &
                bind(C, name="mlirIntegerTypeUnsignedGet")
            import :: c_ptr, c_int
            type(c_ptr), value :: ctx
            integer(c_int), value :: bitwidth
            type(c_ptr) :: mlirIntegerTypeUnsignedGet
        end function mlirIntegerTypeUnsignedGet

        function mlirIntegerTypeGetWidth(ty) &
                bind(C, name="mlirIntegerTypeGetWidth")
            import :: c_ptr, c_int
            type(c_ptr), value :: ty
            integer(c_int) :: mlirIntegerTypeGetWidth
        end function mlirIntegerTypeGetWidth

        function mlirTypeIsAInteger(ty) bind(C, name="mlirTypeIsAInteger")
            import :: c_ptr, c_bool
            type(c_ptr), value :: ty
            logical(c_bool) :: mlirTypeIsAInteger
        end function mlirTypeIsAInteger

        function mlirIndexTypeGet(ctx) bind(C, name="mlirIndexTypeGet")
            import :: c_ptr
            type(c_ptr), value :: ctx
            type(c_ptr) :: mlirIndexTypeGet
        end function mlirIndexTypeGet

        function mlirTypeIsAIndex(ty) bind(C, name="mlirTypeIsAIndex")
            import :: c_ptr, c_bool
            type(c_ptr), value :: ty
            logical(c_bool) :: mlirTypeIsAIndex
        end function mlirTypeIsAIndex

        function mlirF16TypeGet(ctx) bind(C, name="mlirF16TypeGet")
            import :: c_ptr
            type(c_ptr), value :: ctx
            type(c_ptr) :: mlirF16TypeGet
        end function mlirF16TypeGet

        function mlirF32TypeGet(ctx) bind(C, name="mlirF32TypeGet")
            import :: c_ptr
            type(c_ptr), value :: ctx
            type(c_ptr) :: mlirF32TypeGet
        end function mlirF32TypeGet

        function mlirF64TypeGet(ctx) bind(C, name="mlirF64TypeGet")
            import :: c_ptr
            type(c_ptr), value :: ctx
            type(c_ptr) :: mlirF64TypeGet
        end function mlirF64TypeGet

        function mlirBF16TypeGet(ctx) bind(C, name="mlirBF16TypeGet")
            import :: c_ptr
            type(c_ptr), value :: ctx
            type(c_ptr) :: mlirBF16TypeGet
        end function mlirBF16TypeGet

        function mlirTypeIsAFloat(ty) bind(C, name="mlirTypeIsAFloat")
            import :: c_ptr, c_bool
            type(c_ptr), value :: ty
            logical(c_bool) :: mlirTypeIsAFloat
        end function mlirTypeIsAFloat

        function mlirFloatTypeGetWidth(ty) bind(C, name="mlirFloatTypeGetWidth")
            import :: c_ptr, c_int
            type(c_ptr), value :: ty
            integer(c_int) :: mlirFloatTypeGetWidth
        end function mlirFloatTypeGetWidth

        function mlirNoneTypeGet(ctx) bind(C, name="mlirNoneTypeGet")
            import :: c_ptr
            type(c_ptr), value :: ctx
            type(c_ptr) :: mlirNoneTypeGet
        end function mlirNoneTypeGet

        function mlirTypeIsANone(ty) bind(C, name="mlirTypeIsANone")
            import :: c_ptr, c_bool
            type(c_ptr), value :: ty
            logical(c_bool) :: mlirTypeIsANone
        end function mlirTypeIsANone

        function mlirComplexTypeGet(element_type) &
                bind(C, name="mlirComplexTypeGet")
            import :: c_ptr
            type(c_ptr), value :: element_type
            type(c_ptr) :: mlirComplexTypeGet
        end function mlirComplexTypeGet

        function mlirTypeIsAComplex(ty) bind(C, name="mlirTypeIsAComplex")
            import :: c_ptr, c_bool
            type(c_ptr), value :: ty
            logical(c_bool) :: mlirTypeIsAComplex
        end function mlirTypeIsAComplex

        function mlirComplexTypeGetElementType(ty) &
                bind(C, name="mlirComplexTypeGetElementType")
            import :: c_ptr
            type(c_ptr), value :: ty
            type(c_ptr) :: mlirComplexTypeGetElementType
        end function mlirComplexTypeGetElementType

        function mlirFunctionTypeGet(ctx, num_inputs, inputs, num_results, &
                results) bind(C, name="mlirFunctionTypeGet")
            import :: c_ptr, c_intptr_t
            type(c_ptr), value :: ctx
            integer(c_intptr_t), value :: num_inputs
            type(c_ptr), value :: inputs
            integer(c_intptr_t), value :: num_results
            type(c_ptr), value :: results
            type(c_ptr) :: mlirFunctionTypeGet
        end function mlirFunctionTypeGet

        function mlirTypeIsAFunction(ty) bind(C, name="mlirTypeIsAFunction")
            import :: c_ptr, c_bool
            type(c_ptr), value :: ty
            logical(c_bool) :: mlirTypeIsAFunction
        end function mlirTypeIsAFunction

        function mlirFunctionTypeGetNumInputs(ty) &
                bind(C, name="mlirFunctionTypeGetNumInputs")
            import :: c_ptr, c_intptr_t
            type(c_ptr), value :: ty
            integer(c_intptr_t) :: mlirFunctionTypeGetNumInputs
        end function mlirFunctionTypeGetNumInputs

        function mlirFunctionTypeGetNumResults(ty) &
                bind(C, name="mlirFunctionTypeGetNumResults")
            import :: c_ptr, c_intptr_t
            type(c_ptr), value :: ty
            integer(c_intptr_t) :: mlirFunctionTypeGetNumResults
        end function mlirFunctionTypeGetNumResults

        function mlirFunctionTypeGetInput(ty, pos) &
                bind(C, name="mlirFunctionTypeGetInput")
            import :: c_ptr, c_intptr_t
            type(c_ptr), value :: ty
            integer(c_intptr_t), value :: pos
            type(c_ptr) :: mlirFunctionTypeGetInput
        end function mlirFunctionTypeGetInput

        function mlirFunctionTypeGetResult(ty, pos) &
                bind(C, name="mlirFunctionTypeGetResult")
            import :: c_ptr, c_intptr_t
            type(c_ptr), value :: ty
            integer(c_intptr_t), value :: pos
            type(c_ptr) :: mlirFunctionTypeGetResult
        end function mlirFunctionTypeGetResult

        function mlirRankedTensorTypeGet(rank, shape, element_type, encoding) &
                bind(C, name="mlirRankedTensorTypeGet")
            import :: c_ptr, c_intptr_t
            integer(c_intptr_t), value :: rank
            type(c_ptr), value :: shape
            type(c_ptr), value :: element_type
            type(c_ptr), value :: encoding
            type(c_ptr) :: mlirRankedTensorTypeGet
        end function mlirRankedTensorTypeGet

        function mlirTypeIsARankedTensor(ty) &
                bind(C, name="mlirTypeIsARankedTensor")
            import :: c_ptr, c_bool
            type(c_ptr), value :: ty
            logical(c_bool) :: mlirTypeIsARankedTensor
        end function mlirTypeIsARankedTensor

        function mlirMemRefTypeContiguousGet(element_type, rank, shape, &
                memory_space) bind(C, name="mlirMemRefTypeContiguousGet")
            import :: c_ptr, c_intptr_t
            type(c_ptr), value :: element_type
            integer(c_intptr_t), value :: rank
            type(c_ptr), value :: shape
            type(c_ptr), value :: memory_space
            type(c_ptr) :: mlirMemRefTypeContiguousGet
        end function mlirMemRefTypeContiguousGet

        function mlirTypeIsAMemRef(ty) bind(C, name="mlirTypeIsAMemRef")
            import :: c_ptr, c_bool
            type(c_ptr), value :: ty
            logical(c_bool) :: mlirTypeIsAMemRef
        end function mlirTypeIsAMemRef

        function mlirTypeEqual(t1, t2) bind(C, name="mlirTypeEqual")
            import :: c_ptr, c_bool
            type(c_ptr), value :: t1
            type(c_ptr), value :: t2
            logical(c_bool) :: mlirTypeEqual
        end function mlirTypeEqual

        subroutine mlirTypeDump(ty) bind(C, name="mlirTypeDump")
            import :: c_ptr
            type(c_ptr), value :: ty
        end subroutine mlirTypeDump
    end interface

contains

    function mlir_integer_type_get(ctx, bitwidth) result(ty)
        type(mlir_context_t), intent(in) :: ctx
        integer, intent(in) :: bitwidth
        type(mlir_type_t) :: ty
        ty%ptr = mlirIntegerTypeGet(ctx%ptr, int(bitwidth, c_int))
    end function mlir_integer_type_get

    function mlir_integer_type_signed_get(ctx, bitwidth) result(ty)
        type(mlir_context_t), intent(in) :: ctx
        integer, intent(in) :: bitwidth
        type(mlir_type_t) :: ty
        ty%ptr = mlirIntegerTypeSignedGet(ctx%ptr, int(bitwidth, c_int))
    end function mlir_integer_type_signed_get

    function mlir_integer_type_unsigned_get(ctx, bitwidth) result(ty)
        type(mlir_context_t), intent(in) :: ctx
        integer, intent(in) :: bitwidth
        type(mlir_type_t) :: ty
        ty%ptr = mlirIntegerTypeUnsignedGet(ctx%ptr, int(bitwidth, c_int))
    end function mlir_integer_type_unsigned_get

    function mlir_integer_type_get_width(ty) result(width)
        type(mlir_type_t), intent(in) :: ty
        integer :: width
        width = int(mlirIntegerTypeGetWidth(ty%ptr))
    end function mlir_integer_type_get_width

    function mlir_type_is_a_integer(ty) result(is_int)
        type(mlir_type_t), intent(in) :: ty
        logical :: is_int
        is_int = mlirTypeIsAInteger(ty%ptr)
    end function mlir_type_is_a_integer

    function mlir_index_type_get(ctx) result(ty)
        type(mlir_context_t), intent(in) :: ctx
        type(mlir_type_t) :: ty
        ty%ptr = mlirIndexTypeGet(ctx%ptr)
    end function mlir_index_type_get

    function mlir_type_is_a_index(ty) result(is_idx)
        type(mlir_type_t), intent(in) :: ty
        logical :: is_idx
        is_idx = mlirTypeIsAIndex(ty%ptr)
    end function mlir_type_is_a_index

    function mlir_f16_type_get(ctx) result(ty)
        type(mlir_context_t), intent(in) :: ctx
        type(mlir_type_t) :: ty
        ty%ptr = mlirF16TypeGet(ctx%ptr)
    end function mlir_f16_type_get

    function mlir_f32_type_get(ctx) result(ty)
        type(mlir_context_t), intent(in) :: ctx
        type(mlir_type_t) :: ty
        ty%ptr = mlirF32TypeGet(ctx%ptr)
    end function mlir_f32_type_get

    function mlir_f64_type_get(ctx) result(ty)
        type(mlir_context_t), intent(in) :: ctx
        type(mlir_type_t) :: ty
        ty%ptr = mlirF64TypeGet(ctx%ptr)
    end function mlir_f64_type_get

    function mlir_bf16_type_get(ctx) result(ty)
        type(mlir_context_t), intent(in) :: ctx
        type(mlir_type_t) :: ty
        ty%ptr = mlirBF16TypeGet(ctx%ptr)
    end function mlir_bf16_type_get

    function mlir_type_is_a_float(ty) result(is_flt)
        type(mlir_type_t), intent(in) :: ty
        logical :: is_flt
        is_flt = mlirTypeIsAFloat(ty%ptr)
    end function mlir_type_is_a_float

    function mlir_float_type_get_width(ty) result(width)
        type(mlir_type_t), intent(in) :: ty
        integer :: width
        width = int(mlirFloatTypeGetWidth(ty%ptr))
    end function mlir_float_type_get_width

    function mlir_none_type_get(ctx) result(ty)
        type(mlir_context_t), intent(in) :: ctx
        type(mlir_type_t) :: ty
        ty%ptr = mlirNoneTypeGet(ctx%ptr)
    end function mlir_none_type_get

    function mlir_type_is_a_none(ty) result(is_none)
        type(mlir_type_t), intent(in) :: ty
        logical :: is_none
        is_none = mlirTypeIsANone(ty%ptr)
    end function mlir_type_is_a_none

    function mlir_complex_type_get(element_type) result(ty)
        type(mlir_type_t), intent(in) :: element_type
        type(mlir_type_t) :: ty
        ty%ptr = mlirComplexTypeGet(element_type%ptr)
    end function mlir_complex_type_get

    function mlir_type_is_a_complex(ty) result(is_complex)
        type(mlir_type_t), intent(in) :: ty
        logical :: is_complex
        is_complex = mlirTypeIsAComplex(ty%ptr)
    end function mlir_type_is_a_complex

    function mlir_complex_type_get_element_type(ty) result(elem_ty)
        type(mlir_type_t), intent(in) :: ty
        type(mlir_type_t) :: elem_ty
        elem_ty%ptr = mlirComplexTypeGetElementType(ty%ptr)
    end function mlir_complex_type_get_element_type

    function mlir_function_type_get(ctx, inputs, results) result(ty)
        type(mlir_context_t), intent(in) :: ctx
        type(mlir_type_t), intent(in), target :: inputs(:)
        type(mlir_type_t), intent(in), target :: results(:)
        type(mlir_type_t) :: ty
        integer(c_intptr_t) :: n_inputs, n_results
        type(c_ptr) :: inputs_ptr, results_ptr

        n_inputs = int(size(inputs), c_intptr_t)
        n_results = int(size(results), c_intptr_t)

        if (n_inputs > 0) then
            inputs_ptr = c_loc(inputs(1))
        else
            inputs_ptr = c_null_ptr
        end if

        if (n_results > 0) then
            results_ptr = c_loc(results(1))
        else
            results_ptr = c_null_ptr
        end if

        ty%ptr = mlirFunctionTypeGet(ctx%ptr, n_inputs, inputs_ptr, &
            n_results, results_ptr)
    end function mlir_function_type_get

    function mlir_type_is_a_function(ty) result(is_func)
        type(mlir_type_t), intent(in) :: ty
        logical :: is_func
        is_func = mlirTypeIsAFunction(ty%ptr)
    end function mlir_type_is_a_function

    function mlir_function_type_get_num_inputs(ty) result(num)
        type(mlir_type_t), intent(in) :: ty
        integer :: num
        num = int(mlirFunctionTypeGetNumInputs(ty%ptr))
    end function mlir_function_type_get_num_inputs

    function mlir_function_type_get_num_results(ty) result(num)
        type(mlir_type_t), intent(in) :: ty
        integer :: num
        num = int(mlirFunctionTypeGetNumResults(ty%ptr))
    end function mlir_function_type_get_num_results

    function mlir_function_type_get_input(ty, pos) result(input_ty)
        type(mlir_type_t), intent(in) :: ty
        integer, intent(in) :: pos
        type(mlir_type_t) :: input_ty
        input_ty%ptr = mlirFunctionTypeGetInput(ty%ptr, &
            int(pos, c_intptr_t))
    end function mlir_function_type_get_input

    function mlir_function_type_get_result(ty, pos) result(result_ty)
        type(mlir_type_t), intent(in) :: ty
        integer, intent(in) :: pos
        type(mlir_type_t) :: result_ty
        result_ty%ptr = mlirFunctionTypeGetResult(ty%ptr, &
            int(pos, c_intptr_t))
    end function mlir_function_type_get_result

    function mlir_ranked_tensor_type_get(dims, element_type, encoding) &
            result(ty)
        integer(c_int64_t), intent(in), target :: dims(:)
        type(mlir_type_t), intent(in) :: element_type
        type(mlir_attribute_t), intent(in), optional :: encoding
        type(mlir_type_t) :: ty
        integer(c_intptr_t) :: rank
        type(c_ptr) :: encoding_ptr

        rank = int(size(dims), c_intptr_t)
        if (present(encoding)) then
            encoding_ptr = encoding%ptr
        else
            encoding_ptr = c_null_ptr
        end if

        ty%ptr = mlirRankedTensorTypeGet(rank, c_loc(dims(1)), &
            element_type%ptr, encoding_ptr)
    end function mlir_ranked_tensor_type_get

    function mlir_type_is_a_ranked_tensor(ty) result(is_tensor)
        type(mlir_type_t), intent(in) :: ty
        logical :: is_tensor
        is_tensor = mlirTypeIsARankedTensor(ty%ptr)
    end function mlir_type_is_a_ranked_tensor

    function mlir_memref_type_contiguous_get(element_type, dims, &
            memory_space) result(ty)
        type(mlir_type_t), intent(in) :: element_type
        integer(c_int64_t), intent(in), target :: dims(:)
        type(mlir_attribute_t), intent(in), optional :: memory_space
        type(mlir_type_t) :: ty
        integer(c_intptr_t) :: rank
        type(c_ptr) :: mem_space_ptr

        rank = int(size(dims), c_intptr_t)
        if (present(memory_space)) then
            mem_space_ptr = memory_space%ptr
        else
            mem_space_ptr = c_null_ptr
        end if

        ty%ptr = mlirMemRefTypeContiguousGet(element_type%ptr, rank, &
            c_loc(dims(1)), mem_space_ptr)
    end function mlir_memref_type_contiguous_get

    function mlir_type_is_a_memref(ty) result(is_memref)
        type(mlir_type_t), intent(in) :: ty
        logical :: is_memref
        is_memref = mlirTypeIsAMemRef(ty%ptr)
    end function mlir_type_is_a_memref

    pure function mlir_type_is_null(ty) result(is_null)
        type(mlir_type_t), intent(in) :: ty
        logical :: is_null
        is_null = .not. c_associated(ty%ptr)
    end function mlir_type_is_null

    function mlir_type_equal(t1, t2) result(eq)
        type(mlir_type_t), intent(in) :: t1
        type(mlir_type_t), intent(in) :: t2
        logical :: eq
        eq = mlirTypeEqual(t1%ptr, t2%ptr)
    end function mlir_type_equal

    subroutine mlir_type_dump(ty)
        type(mlir_type_t), intent(in) :: ty
        call mlirTypeDump(ty%ptr)
    end subroutine mlir_type_dump

end module mlir_c_types
