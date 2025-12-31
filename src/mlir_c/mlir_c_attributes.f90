module mlir_c_attributes
    use, intrinsic :: iso_c_binding, only: c_ptr, c_null_ptr, c_bool, c_int, &
        c_int64_t, c_intptr_t, c_size_t, c_double, c_associated, c_loc, &
        c_f_pointer
    use mlir_c_core, only: mlir_context_t, mlir_attribute_t, mlir_type_t
    implicit none
    private

    public :: mlir_attribute_get_null
    public :: mlir_attribute_is_null
    public :: mlir_attribute_equal
    public :: mlir_attribute_dump

    public :: mlir_string_attr_get
    public :: mlir_string_attr_typed_get
    public :: mlir_string_attr_get_value
    public :: mlir_attribute_is_a_string

    public :: mlir_integer_attr_get
    public :: mlir_integer_attr_get_value_int
    public :: mlir_attribute_is_a_integer

    public :: mlir_float_attr_double_get
    public :: mlir_float_attr_get_value_double
    public :: mlir_attribute_is_a_float

    public :: mlir_bool_attr_get
    public :: mlir_bool_attr_get_value
    public :: mlir_attribute_is_a_bool

    public :: mlir_type_attr_get
    public :: mlir_type_attr_get_value
    public :: mlir_attribute_is_a_type

    public :: mlir_unit_attr_get
    public :: mlir_attribute_is_a_unit

    public :: mlir_array_attr_get
    public :: mlir_array_attr_get_num_elements
    public :: mlir_array_attr_get_element
    public :: mlir_attribute_is_a_array

    public :: mlir_flat_symbol_ref_attr_get
    public :: mlir_flat_symbol_ref_attr_get_value
    public :: mlir_attribute_is_a_flat_symbol_ref

    public :: mlir_dense_i64_array_get
    public :: mlir_dense_i64_array_get_element

    interface
        function mlirAttributeGetNull() bind(C, name="mlirAttributeGetNull")
            import :: c_ptr
            type(c_ptr) :: mlirAttributeGetNull
        end function mlirAttributeGetNull

        function mlirAttributeEqual(a1, a2) bind(C, name="mlirAttributeEqual")
            import :: c_ptr, c_bool
            type(c_ptr), value :: a1
            type(c_ptr), value :: a2
            logical(c_bool) :: mlirAttributeEqual
        end function mlirAttributeEqual

        subroutine mlirAttributeDump(attr) bind(C, name="mlirAttributeDump")
            import :: c_ptr
            type(c_ptr), value :: attr
        end subroutine mlirAttributeDump

        function mlirStringAttrGet_c(ctx, str_data, str_len) &
                bind(C, name="mlirStringAttrGet")
            import :: c_ptr, c_size_t
            type(c_ptr), value :: ctx
            type(c_ptr), value :: str_data
            integer(c_size_t), value :: str_len
            type(c_ptr) :: mlirStringAttrGet_c
        end function mlirStringAttrGet_c

        function mlirStringAttrTypedGet_c(ty, str_data, str_len) &
                bind(C, name="mlirStringAttrTypedGet")
            import :: c_ptr, c_size_t
            type(c_ptr), value :: ty
            type(c_ptr), value :: str_data
            integer(c_size_t), value :: str_len
            type(c_ptr) :: mlirStringAttrTypedGet_c
        end function mlirStringAttrTypedGet_c

        subroutine mlirStringAttrGetValue_c(attr, out_data, out_len) &
                bind(C, name="mlirStringAttrGetValue")
            import :: c_ptr, c_size_t
            type(c_ptr), value :: attr
            type(c_ptr) :: out_data
            integer(c_size_t) :: out_len
        end subroutine mlirStringAttrGetValue_c

        function mlirAttributeIsAString(attr) &
                bind(C, name="mlirAttributeIsAString")
            import :: c_ptr, c_bool
            type(c_ptr), value :: attr
            logical(c_bool) :: mlirAttributeIsAString
        end function mlirAttributeIsAString

        function mlirIntegerAttrGet(ty, value) &
                bind(C, name="mlirIntegerAttrGet")
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: ty
            integer(c_int64_t), value :: value
            type(c_ptr) :: mlirIntegerAttrGet
        end function mlirIntegerAttrGet

        function mlirIntegerAttrGetValueInt(attr) &
                bind(C, name="mlirIntegerAttrGetValueInt")
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: attr
            integer(c_int64_t) :: mlirIntegerAttrGetValueInt
        end function mlirIntegerAttrGetValueInt

        function mlirAttributeIsAInteger(attr) &
                bind(C, name="mlirAttributeIsAInteger")
            import :: c_ptr, c_bool
            type(c_ptr), value :: attr
            logical(c_bool) :: mlirAttributeIsAInteger
        end function mlirAttributeIsAInteger

        function mlirFloatAttrDoubleGet(ctx, ty, value) &
                bind(C, name="mlirFloatAttrDoubleGet")
            import :: c_ptr, c_double
            type(c_ptr), value :: ctx
            type(c_ptr), value :: ty
            real(c_double), value :: value
            type(c_ptr) :: mlirFloatAttrDoubleGet
        end function mlirFloatAttrDoubleGet

        function mlirFloatAttrGetValueDouble(attr) &
                bind(C, name="mlirFloatAttrGetValueDouble")
            import :: c_ptr, c_double
            type(c_ptr), value :: attr
            real(c_double) :: mlirFloatAttrGetValueDouble
        end function mlirFloatAttrGetValueDouble

        function mlirAttributeIsAFloat(attr) &
                bind(C, name="mlirAttributeIsAFloat")
            import :: c_ptr, c_bool
            type(c_ptr), value :: attr
            logical(c_bool) :: mlirAttributeIsAFloat
        end function mlirAttributeIsAFloat

        function mlirBoolAttrGet(ctx, value) bind(C, name="mlirBoolAttrGet")
            import :: c_ptr, c_int
            type(c_ptr), value :: ctx
            integer(c_int), value :: value
            type(c_ptr) :: mlirBoolAttrGet
        end function mlirBoolAttrGet

        function mlirBoolAttrGetValue(attr) bind(C, name="mlirBoolAttrGetValue")
            import :: c_ptr, c_bool
            type(c_ptr), value :: attr
            logical(c_bool) :: mlirBoolAttrGetValue
        end function mlirBoolAttrGetValue

        function mlirAttributeIsABool(attr) bind(C, name="mlirAttributeIsABool")
            import :: c_ptr, c_bool
            type(c_ptr), value :: attr
            logical(c_bool) :: mlirAttributeIsABool
        end function mlirAttributeIsABool

        function mlirTypeAttrGet(ty) bind(C, name="mlirTypeAttrGet")
            import :: c_ptr
            type(c_ptr), value :: ty
            type(c_ptr) :: mlirTypeAttrGet
        end function mlirTypeAttrGet

        function mlirTypeAttrGetValue(attr) bind(C, name="mlirTypeAttrGetValue")
            import :: c_ptr
            type(c_ptr), value :: attr
            type(c_ptr) :: mlirTypeAttrGetValue
        end function mlirTypeAttrGetValue

        function mlirAttributeIsAType(attr) bind(C, name="mlirAttributeIsAType")
            import :: c_ptr, c_bool
            type(c_ptr), value :: attr
            logical(c_bool) :: mlirAttributeIsAType
        end function mlirAttributeIsAType

        function mlirUnitAttrGet(ctx) bind(C, name="mlirUnitAttrGet")
            import :: c_ptr
            type(c_ptr), value :: ctx
            type(c_ptr) :: mlirUnitAttrGet
        end function mlirUnitAttrGet

        function mlirAttributeIsAUnit(attr) bind(C, name="mlirAttributeIsAUnit")
            import :: c_ptr, c_bool
            type(c_ptr), value :: attr
            logical(c_bool) :: mlirAttributeIsAUnit
        end function mlirAttributeIsAUnit

        function mlirArrayAttrGet(ctx, num_elements, elements) &
                bind(C, name="mlirArrayAttrGet")
            import :: c_ptr, c_intptr_t
            type(c_ptr), value :: ctx
            integer(c_intptr_t), value :: num_elements
            type(c_ptr), value :: elements
            type(c_ptr) :: mlirArrayAttrGet
        end function mlirArrayAttrGet

        function mlirArrayAttrGetNumElements(attr) &
                bind(C, name="mlirArrayAttrGetNumElements")
            import :: c_ptr, c_intptr_t
            type(c_ptr), value :: attr
            integer(c_intptr_t) :: mlirArrayAttrGetNumElements
        end function mlirArrayAttrGetNumElements

        function mlirArrayAttrGetElement(attr, pos) &
                bind(C, name="mlirArrayAttrGetElement")
            import :: c_ptr, c_intptr_t
            type(c_ptr), value :: attr
            integer(c_intptr_t), value :: pos
            type(c_ptr) :: mlirArrayAttrGetElement
        end function mlirArrayAttrGetElement

        function mlirAttributeIsAArray(attr) &
                bind(C, name="mlirAttributeIsAArray")
            import :: c_ptr, c_bool
            type(c_ptr), value :: attr
            logical(c_bool) :: mlirAttributeIsAArray
        end function mlirAttributeIsAArray

        function mlirFlatSymbolRefAttrGet_c(ctx, symbol_data, symbol_len) &
                bind(C, name="mlirFlatSymbolRefAttrGet")
            import :: c_ptr, c_size_t
            type(c_ptr), value :: ctx
            type(c_ptr), value :: symbol_data
            integer(c_size_t), value :: symbol_len
            type(c_ptr) :: mlirFlatSymbolRefAttrGet_c
        end function mlirFlatSymbolRefAttrGet_c

        subroutine mlirFlatSymbolRefAttrGetValue_c(attr, out_data, out_len) &
                bind(C, name="mlirFlatSymbolRefAttrGetValue")
            import :: c_ptr, c_size_t
            type(c_ptr), value :: attr
            type(c_ptr) :: out_data
            integer(c_size_t) :: out_len
        end subroutine mlirFlatSymbolRefAttrGetValue_c

        function mlirAttributeIsAFlatSymbolRef(attr) &
                bind(C, name="mlirAttributeIsAFlatSymbolRef")
            import :: c_ptr, c_bool
            type(c_ptr), value :: attr
            logical(c_bool) :: mlirAttributeIsAFlatSymbolRef
        end function mlirAttributeIsAFlatSymbolRef

        function mlirDenseI64ArrayGet(ctx, size, values) &
                bind(C, name="mlirDenseI64ArrayGet")
            import :: c_ptr, c_intptr_t
            type(c_ptr), value :: ctx
            integer(c_intptr_t), value :: size
            type(c_ptr), value :: values
            type(c_ptr) :: mlirDenseI64ArrayGet
        end function mlirDenseI64ArrayGet

        function mlirDenseI64ArrayGetElement(attr, pos) &
                bind(C, name="mlirDenseI64ArrayGetElement")
            import :: c_ptr, c_intptr_t, c_int64_t
            type(c_ptr), value :: attr
            integer(c_intptr_t), value :: pos
            integer(c_int64_t) :: mlirDenseI64ArrayGetElement
        end function mlirDenseI64ArrayGetElement
    end interface

contains

    function mlir_attribute_get_null() result(attr)
        type(mlir_attribute_t) :: attr
        attr%ptr = mlirAttributeGetNull()
    end function mlir_attribute_get_null

    pure function mlir_attribute_is_null(attr) result(is_null)
        type(mlir_attribute_t), intent(in) :: attr
        logical :: is_null
        is_null = .not. c_associated(attr%ptr)
    end function mlir_attribute_is_null

    function mlir_attribute_equal(a1, a2) result(eq)
        type(mlir_attribute_t), intent(in) :: a1
        type(mlir_attribute_t), intent(in) :: a2
        logical :: eq
        eq = mlirAttributeEqual(a1%ptr, a2%ptr)
    end function mlir_attribute_equal

    subroutine mlir_attribute_dump(attr)
        type(mlir_attribute_t), intent(in) :: attr
        call mlirAttributeDump(attr%ptr)
    end subroutine mlir_attribute_dump

    function mlir_string_attr_get(ctx, str) result(attr)
        type(mlir_context_t), intent(in) :: ctx
        character(len=*), intent(in), target :: str
        type(mlir_attribute_t) :: attr

        attr%ptr = mlirStringAttrGet_c(ctx%ptr, c_loc(str), &
            int(len(str), c_size_t))
    end function mlir_string_attr_get

    function mlir_string_attr_typed_get(ty, str) result(attr)
        type(mlir_type_t), intent(in) :: ty
        character(len=*), intent(in), target :: str
        type(mlir_attribute_t) :: attr

        attr%ptr = mlirStringAttrTypedGet_c(ty%ptr, c_loc(str), &
            int(len(str), c_size_t))
    end function mlir_string_attr_typed_get

    subroutine mlir_string_attr_get_value(attr, str)
        type(mlir_attribute_t), intent(in) :: attr
        character(len=:), allocatable, intent(out) :: str
        type(c_ptr) :: data_ptr
        integer(c_size_t) :: length
        character(len=1), pointer :: chars(:)

        call mlirStringAttrGetValue_c(attr%ptr, data_ptr, length)
        if (c_associated(data_ptr)) then
            call c_f_pointer(data_ptr, chars, [int(length)])
            allocate(character(len=int(length)) :: str)
            block
                integer :: i
                do i = 1, int(length)
                    str(i:i) = chars(i)
                end do
            end block
        else
            str = ""
        end if
    end subroutine mlir_string_attr_get_value

    function mlir_attribute_is_a_string(attr) result(is_str)
        type(mlir_attribute_t), intent(in) :: attr
        logical :: is_str
        is_str = mlirAttributeIsAString(attr%ptr)
    end function mlir_attribute_is_a_string

    function mlir_integer_attr_get(ty, value) result(attr)
        type(mlir_type_t), intent(in) :: ty
        integer(c_int64_t), intent(in) :: value
        type(mlir_attribute_t) :: attr
        attr%ptr = mlirIntegerAttrGet(ty%ptr, value)
    end function mlir_integer_attr_get

    function mlir_integer_attr_get_value_int(attr) result(value)
        type(mlir_attribute_t), intent(in) :: attr
        integer(c_int64_t) :: value
        value = mlirIntegerAttrGetValueInt(attr%ptr)
    end function mlir_integer_attr_get_value_int

    function mlir_attribute_is_a_integer(attr) result(is_int)
        type(mlir_attribute_t), intent(in) :: attr
        logical :: is_int
        is_int = mlirAttributeIsAInteger(attr%ptr)
    end function mlir_attribute_is_a_integer

    function mlir_float_attr_double_get(ctx, ty, value) result(attr)
        type(mlir_context_t), intent(in) :: ctx
        type(mlir_type_t), intent(in) :: ty
        real(c_double), intent(in) :: value
        type(mlir_attribute_t) :: attr
        attr%ptr = mlirFloatAttrDoubleGet(ctx%ptr, ty%ptr, value)
    end function mlir_float_attr_double_get

    function mlir_float_attr_get_value_double(attr) result(value)
        type(mlir_attribute_t), intent(in) :: attr
        real(c_double) :: value
        value = mlirFloatAttrGetValueDouble(attr%ptr)
    end function mlir_float_attr_get_value_double

    function mlir_attribute_is_a_float(attr) result(is_flt)
        type(mlir_attribute_t), intent(in) :: attr
        logical :: is_flt
        is_flt = mlirAttributeIsAFloat(attr%ptr)
    end function mlir_attribute_is_a_float

    function mlir_bool_attr_get(ctx, value) result(attr)
        type(mlir_context_t), intent(in) :: ctx
        logical, intent(in) :: value
        type(mlir_attribute_t) :: attr
        integer(c_int) :: int_val

        if (value) then
            int_val = 1
        else
            int_val = 0
        end if
        attr%ptr = mlirBoolAttrGet(ctx%ptr, int_val)
    end function mlir_bool_attr_get

    function mlir_bool_attr_get_value(attr) result(value)
        type(mlir_attribute_t), intent(in) :: attr
        logical :: value
        value = mlirBoolAttrGetValue(attr%ptr)
    end function mlir_bool_attr_get_value

    function mlir_attribute_is_a_bool(attr) result(is_bool)
        type(mlir_attribute_t), intent(in) :: attr
        logical :: is_bool
        is_bool = mlirAttributeIsABool(attr%ptr)
    end function mlir_attribute_is_a_bool

    function mlir_type_attr_get(ty) result(attr)
        type(mlir_type_t), intent(in) :: ty
        type(mlir_attribute_t) :: attr
        attr%ptr = mlirTypeAttrGet(ty%ptr)
    end function mlir_type_attr_get

    function mlir_type_attr_get_value(attr) result(ty)
        type(mlir_attribute_t), intent(in) :: attr
        type(mlir_type_t) :: ty
        ty%ptr = mlirTypeAttrGetValue(attr%ptr)
    end function mlir_type_attr_get_value

    function mlir_attribute_is_a_type(attr) result(is_type)
        type(mlir_attribute_t), intent(in) :: attr
        logical :: is_type
        is_type = mlirAttributeIsAType(attr%ptr)
    end function mlir_attribute_is_a_type

    function mlir_unit_attr_get(ctx) result(attr)
        type(mlir_context_t), intent(in) :: ctx
        type(mlir_attribute_t) :: attr
        attr%ptr = mlirUnitAttrGet(ctx%ptr)
    end function mlir_unit_attr_get

    function mlir_attribute_is_a_unit(attr) result(is_unit)
        type(mlir_attribute_t), intent(in) :: attr
        logical :: is_unit
        is_unit = mlirAttributeIsAUnit(attr%ptr)
    end function mlir_attribute_is_a_unit

    function mlir_array_attr_get(ctx, elements) result(attr)
        type(mlir_context_t), intent(in) :: ctx
        type(mlir_attribute_t), intent(in), target :: elements(:)
        type(mlir_attribute_t) :: attr
        integer(c_intptr_t) :: n_elements

        n_elements = int(size(elements), c_intptr_t)
        if (n_elements > 0) then
            attr%ptr = mlirArrayAttrGet(ctx%ptr, n_elements, c_loc(elements(1)))
        else
            attr%ptr = mlirArrayAttrGet(ctx%ptr, 0_c_intptr_t, c_null_ptr)
        end if
    end function mlir_array_attr_get

    function mlir_array_attr_get_num_elements(attr) result(num)
        type(mlir_attribute_t), intent(in) :: attr
        integer :: num
        num = int(mlirArrayAttrGetNumElements(attr%ptr))
    end function mlir_array_attr_get_num_elements

    function mlir_array_attr_get_element(attr, pos) result(elem)
        type(mlir_attribute_t), intent(in) :: attr
        integer, intent(in) :: pos
        type(mlir_attribute_t) :: elem
        elem%ptr = mlirArrayAttrGetElement(attr%ptr, int(pos, c_intptr_t))
    end function mlir_array_attr_get_element

    function mlir_attribute_is_a_array(attr) result(is_array)
        type(mlir_attribute_t), intent(in) :: attr
        logical :: is_array
        is_array = mlirAttributeIsAArray(attr%ptr)
    end function mlir_attribute_is_a_array

    function mlir_flat_symbol_ref_attr_get(ctx, symbol) result(attr)
        type(mlir_context_t), intent(in) :: ctx
        character(len=*), intent(in), target :: symbol
        type(mlir_attribute_t) :: attr

        attr%ptr = mlirFlatSymbolRefAttrGet_c(ctx%ptr, c_loc(symbol), &
            int(len(symbol), c_size_t))
    end function mlir_flat_symbol_ref_attr_get

    subroutine mlir_flat_symbol_ref_attr_get_value(attr, str)
        type(mlir_attribute_t), intent(in) :: attr
        character(len=:), allocatable, intent(out) :: str
        type(c_ptr) :: data_ptr
        integer(c_size_t) :: length
        character(len=1), pointer :: chars(:)

        call mlirFlatSymbolRefAttrGetValue_c(attr%ptr, data_ptr, length)
        if (c_associated(data_ptr)) then
            call c_f_pointer(data_ptr, chars, [int(length)])
            allocate(character(len=int(length)) :: str)
            block
                integer :: i
                do i = 1, int(length)
                    str(i:i) = chars(i)
                end do
            end block
        else
            str = ""
        end if
    end subroutine mlir_flat_symbol_ref_attr_get_value

    function mlir_attribute_is_a_flat_symbol_ref(attr) result(is_sym)
        type(mlir_attribute_t), intent(in) :: attr
        logical :: is_sym
        is_sym = mlirAttributeIsAFlatSymbolRef(attr%ptr)
    end function mlir_attribute_is_a_flat_symbol_ref

    function mlir_dense_i64_array_get(ctx, values) result(attr)
        type(mlir_context_t), intent(in) :: ctx
        integer(c_int64_t), intent(in), target :: values(:)
        type(mlir_attribute_t) :: attr
        integer(c_intptr_t) :: n_values

        n_values = int(size(values), c_intptr_t)
        if (n_values > 0) then
            attr%ptr = mlirDenseI64ArrayGet(ctx%ptr, n_values, c_loc(values(1)))
        else
            attr%ptr = mlirDenseI64ArrayGet(ctx%ptr, 0_c_intptr_t, c_null_ptr)
        end if
    end function mlir_dense_i64_array_get

    function mlir_dense_i64_array_get_element(attr, pos) result(value)
        type(mlir_attribute_t), intent(in) :: attr
        integer, intent(in) :: pos
        integer(c_int64_t) :: value
        value = mlirDenseI64ArrayGetElement(attr%ptr, int(pos, c_intptr_t))
    end function mlir_dense_i64_array_get_element

end module mlir_c_attributes
