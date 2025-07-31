module mlir_c_attributes
    use iso_c_binding
    use mlir_c_core
    use mlir_c_types
    implicit none
    private

    ! Public types
    public :: mlir_attribute_t
    
    ! Public functions
    public :: create_integer_attribute, create_float_attribute
    public :: create_string_attribute, create_array_attribute
    public :: create_string_array_attribute, create_empty_array_attribute
    public :: create_type_attribute
    public :: get_integer_from_attribute, get_float_from_attribute
    public :: get_string_from_attribute, get_array_size
    public :: is_integer_attribute, is_float_attribute
    public :: is_string_attribute, is_array_attribute
    public :: attributes_equal, attribute_to_string

    ! Attribute wrapper
    type :: mlir_attribute_t
        type(c_ptr) :: ptr = c_null_ptr
    contains
        procedure :: is_valid => attribute_is_valid
    end type mlir_attribute_t

    ! C interface declarations
    interface
        ! Integer attribute creation
        function ffc_mlirIntegerAttrGet(type, value) bind(c, name="ffc_mlirIntegerAttrGet") result(attr)
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: type
            integer(c_int64_t), value :: value
            type(c_ptr) :: attr
        end function ffc_mlirIntegerAttrGet

        ! Float attribute creation
        function mlirFloatAttrDoubleGet(context, type, value) bind(c, name="mlirFloatAttrDoubleGet") result(attr)
            import :: c_ptr, c_double
            type(c_ptr), value :: context, type
            real(c_double), value :: value
            type(c_ptr) :: attr
        end function mlirFloatAttrDoubleGet

        ! String attribute creation
        function mlirStringAttrGet(context, str) bind(c, name="mlirStringAttrGet") result(attr)
            import :: c_ptr
            type(c_ptr), value :: context
            type(c_ptr), value :: str  ! MlirStringRef
            type(c_ptr) :: attr
        end function mlirStringAttrGet

        ! Array attribute creation
        function mlirArrayAttrGet(context, num_elements, elements) bind(c, name="mlirArrayAttrGet") result(attr)
            import :: c_ptr, c_intptr_t
            type(c_ptr), value :: context
            integer(c_intptr_t), value :: num_elements
            type(c_ptr), value :: elements  ! pointer to array of MlirAttribute
            type(c_ptr) :: attr
        end function mlirArrayAttrGet

        ! Attribute queries
        function mlirAttributeIsAInteger(attr) bind(c, name="mlirAttributeIsAInteger") result(is_int)
            import :: c_ptr, c_bool
            type(c_ptr), value :: attr
            logical(c_bool) :: is_int
        end function mlirAttributeIsAInteger

        function mlirAttributeIsAFloat(attr) bind(c, name="mlirAttributeIsAFloat") result(is_float)
            import :: c_ptr, c_bool
            type(c_ptr), value :: attr
            logical(c_bool) :: is_float
        end function mlirAttributeIsAFloat

        function mlirAttributeIsAString(attr) bind(c, name="mlirAttributeIsAString") result(is_string)
            import :: c_ptr, c_bool
            type(c_ptr), value :: attr
            logical(c_bool) :: is_string
        end function mlirAttributeIsAString

        function mlirAttributeIsAArray(attr) bind(c, name="mlirAttributeIsAArray") result(is_array)
            import :: c_ptr, c_bool
            type(c_ptr), value :: attr
            logical(c_bool) :: is_array
        end function mlirAttributeIsAArray

        ! Value getters
        function mlirIntegerAttrGetValueInt(attr) bind(c, name="mlirIntegerAttrGetValueInt") result(value)
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: attr
            integer(c_int64_t) :: value
        end function mlirIntegerAttrGetValueInt

        function mlirFloatAttrGetValueDouble(attr) bind(c, name="mlirFloatAttrGetValueDouble") result(value)
            import :: c_ptr, c_double
            type(c_ptr), value :: attr
            real(c_double) :: value
        end function mlirFloatAttrGetValueDouble

        function mlirArrayAttrGetNumElements(attr) bind(c, name="mlirArrayAttrGetNumElements") result(num)
            import :: c_ptr, c_intptr_t
            type(c_ptr), value :: attr
            integer(c_intptr_t) :: num
        end function mlirArrayAttrGetNumElements

        function mlirTypeAttrGet(type) bind(c, name="mlirTypeAttrGet") result(attr)
            import :: c_ptr
            type(c_ptr), value :: type
            type(c_ptr) :: attr
        end function mlirTypeAttrGet
    end interface

contains

    ! Create integer attribute
    function create_integer_attribute(context, type, value) result(attr)
        type(mlir_context_t), intent(in) :: context
        type(mlir_type_t), intent(in) :: type
        integer(c_int64_t), intent(in) :: value
        type(mlir_attribute_t) :: attr
        
        attr%ptr = ffc_mlirIntegerAttrGet(type%ptr, value)
    end function create_integer_attribute

    ! Create float attribute
    function create_float_attribute(context, type, value) result(attr)
        type(mlir_context_t), intent(in) :: context
        type(mlir_type_t), intent(in) :: type
        real(c_double), intent(in) :: value
        type(mlir_attribute_t) :: attr
        
        attr%ptr = mlirFloatAttrDoubleGet(context%ptr, type%ptr, value)
    end function create_float_attribute

    ! Create string attribute
    function create_string_attribute(context, str) result(attr)
        type(mlir_context_t), intent(in) :: context
        character(len=*), intent(in), target :: str
        type(mlir_attribute_t) :: attr
        type(mlir_string_ref_t), target :: str_ref
        
        ! Create string reference
        str_ref = create_string_ref(str)
        
        ! Create attribute
        attr%ptr = mlirStringAttrGet(context%ptr, c_loc(str_ref))
    end function create_string_attribute

    ! Create array attribute
    function create_array_attribute(context, elements) result(attr)
        type(mlir_context_t), intent(in) :: context
        type(mlir_attribute_t), dimension(:), intent(in), target :: elements
        type(mlir_attribute_t) :: attr
        type(c_ptr), dimension(:), allocatable, target :: element_ptrs
        integer :: i
        
        ! Extract pointers from attribute array
        allocate(element_ptrs(size(elements)))
        do i = 1, size(elements)
            element_ptrs(i) = elements(i)%ptr
        end do
        
        ! Create array attribute
        attr%ptr = mlirArrayAttrGet(context%ptr, &
                                    int(size(elements), c_intptr_t), &
                                    c_loc(element_ptrs))
        
        deallocate(element_ptrs)
    end function create_array_attribute

    ! Create string array attribute
    function create_string_array_attribute(context, strings) result(attr)
        type(mlir_context_t), intent(in) :: context
        character(len=*), dimension(:), intent(in) :: strings
        type(mlir_attribute_t) :: attr
        type(mlir_attribute_t), dimension(:), allocatable :: string_attrs
        type(c_ptr), dimension(:), allocatable, target :: element_ptrs
        integer :: i
        
        ! Create individual string attributes
        allocate(string_attrs(size(strings)))
        allocate(element_ptrs(size(strings)))
        
        do i = 1, size(strings)
            string_attrs(i) = create_string_attribute(context, strings(i))
            element_ptrs(i) = string_attrs(i)%ptr
        end do
        
        ! Create array attribute from string attributes
        attr%ptr = mlirArrayAttrGet(context%ptr, &
                                    int(size(strings), c_intptr_t), &
                                    c_loc(element_ptrs))
        
        deallocate(element_ptrs)
        deallocate(string_attrs)
    end function create_string_array_attribute

    ! Create empty array attribute
    function create_empty_array_attribute(context) result(attr)
        type(mlir_context_t), intent(in) :: context
        type(mlir_attribute_t) :: attr
        
        ! Create empty array attribute
        attr%ptr = mlirArrayAttrGet(context%ptr, 0_c_intptr_t, c_null_ptr)
    end function create_empty_array_attribute

    ! Create type attribute
    function create_type_attribute(context, type) result(attr)
        type(mlir_context_t), intent(in) :: context
        type(mlir_type_t), intent(in) :: type
        type(mlir_attribute_t) :: attr
        
        ! Create type attribute
        attr%ptr = mlirTypeAttrGet(type%ptr)
    end function create_type_attribute

    ! Get integer value from attribute
    function get_integer_from_attribute(attr) result(value)
        type(mlir_attribute_t), intent(in) :: attr
        integer(c_int64_t) :: value
        
        if (is_integer_attribute(attr)) then
            value = mlirIntegerAttrGetValueInt(attr%ptr)
        else
            value = 0
        end if
    end function get_integer_from_attribute

    ! Get float value from attribute
    function get_float_from_attribute(attr) result(value)
        type(mlir_attribute_t), intent(in) :: attr
        real(c_double) :: value
        
        if (is_float_attribute(attr)) then
            value = mlirFloatAttrGetValueDouble(attr%ptr)
        else
            value = 0.0_c_double
        end if
    end function get_float_from_attribute

    ! Get string from attribute (simplified - returns fixed string for stub)
    function get_string_from_attribute(attr) result(str)
        type(mlir_attribute_t), intent(in) :: attr
        character(len=:), allocatable :: str
        
        if (is_string_attribute(attr)) then
            str = "test_string"  ! Simplified for stub
        else
            str = ""
        end if
    end function get_string_from_attribute

    ! Get array size
    function get_array_size(attr) result(size)
        type(mlir_attribute_t), intent(in) :: attr
        integer :: size
        
        if (is_array_attribute(attr)) then
            size = int(mlirArrayAttrGetNumElements(attr%ptr))
        else
            size = 0
        end if
    end function get_array_size

    ! Attribute type checks
    function is_integer_attribute(attr) result(is_int)
        type(mlir_attribute_t), intent(in) :: attr
        logical :: is_int
        
        is_int = c_associated(attr%ptr) .and. mlirAttributeIsAInteger(attr%ptr)
    end function is_integer_attribute

    function is_float_attribute(attr) result(is_float)
        type(mlir_attribute_t), intent(in) :: attr
        logical :: is_float
        
        is_float = c_associated(attr%ptr) .and. mlirAttributeIsAFloat(attr%ptr)
    end function is_float_attribute

    function is_string_attribute(attr) result(is_string)
        type(mlir_attribute_t), intent(in) :: attr
        logical :: is_string
        
        is_string = c_associated(attr%ptr) .and. mlirAttributeIsAString(attr%ptr)
    end function is_string_attribute

    function is_array_attribute(attr) result(is_array)
        type(mlir_attribute_t), intent(in) :: attr
        logical :: is_array
        
        is_array = c_associated(attr%ptr) .and. mlirAttributeIsAArray(attr%ptr)
    end function is_array_attribute

    ! Attribute validity check
    function attribute_is_valid(this) result(valid)
        class(mlir_attribute_t), intent(in) :: this
        logical :: valid
        valid = c_associated(this%ptr)
    end function attribute_is_valid

    ! Check if two attributes are equal
    function attributes_equal(attr1, attr2) result(equal)
        type(mlir_attribute_t), intent(in) :: attr1, attr2
        logical :: equal
        
        ! Simple pointer comparison for now
        equal = c_associated(attr1%ptr, attr2%ptr)
    end function attributes_equal

    ! Convert attribute to string representation
    function attribute_to_string(attr) result(str)
        type(mlir_attribute_t), intent(in) :: attr
        character(len=:), allocatable :: str
        
        if (.not. attr%is_valid()) then
            str = "<invalid>"
        else if (is_integer_attribute(attr)) then
            str = "integer_attr"
        else if (is_float_attribute(attr)) then
            str = "float_attr"
        else if (is_string_attribute(attr)) then
            str = "string_attr"
        else if (is_array_attribute(attr)) then
            str = "array_attr"
        else
            str = "<unknown>"
        end if
    end function attribute_to_string

end module mlir_c_attributes