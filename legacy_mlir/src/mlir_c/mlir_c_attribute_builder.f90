module mlir_c_attribute_builder
    use iso_c_binding
    use mlir_c_core
    use mlir_c_types
    use mlir_c_attributes
    implicit none
    private

    ! Public types
    public :: attribute_builder_t
    
    ! Attribute builder with validation
    type :: attribute_builder_t
        private
        type(mlir_context_t) :: context
    contains
        procedure :: init => builder_init
        procedure :: integer_attr => builder_integer_attr
        procedure :: float_attr => builder_float_attr
        procedure :: string_attr => builder_string_attr
        procedure :: array_attr => builder_array_attr
        procedure :: bool_attr => builder_bool_attr
        procedure :: validate_integer_value => builder_validate_integer_value
        procedure :: validate_float_value => builder_validate_float_value
    end type attribute_builder_t

contains

    ! Initialize builder
    subroutine builder_init(this, context)
        class(attribute_builder_t), intent(out) :: this
        type(mlir_context_t), intent(in) :: context
        
        this%context = context
    end subroutine builder_init

    ! Create integer attribute with validation
    function builder_integer_attr(this, width, value, signed) result(attr)
        class(attribute_builder_t), intent(in) :: this
        integer, intent(in) :: width
        integer(c_int64_t), intent(in) :: value
        logical, intent(in), optional :: signed
        type(mlir_attribute_t) :: attr
        type(mlir_type_t) :: int_type
        logical :: is_signed
        
        is_signed = .true.
        if (present(signed)) is_signed = signed
        
        ! Validate width
        if (.not. validate_integer_width(width)) then
            attr%ptr = c_null_ptr
            return
        end if
        
        ! Validate value fits in width
        if (.not. this%validate_integer_value(value, width, is_signed)) then
            attr%ptr = c_null_ptr
            return
        end if
        
        ! Create type and attribute
        int_type = create_integer_type(this%context, width, is_signed)
        attr = create_integer_attribute(this%context, int_type, value)
    end function builder_integer_attr

    ! Create float attribute with validation
    function builder_float_attr(this, width, value) result(attr)
        class(attribute_builder_t), intent(in) :: this
        integer, intent(in) :: width
        real(c_double), intent(in) :: value
        type(mlir_attribute_t) :: attr
        type(mlir_type_t) :: float_type
        
        ! Validate width
        if (.not. validate_float_width(width)) then
            attr%ptr = c_null_ptr
            return
        end if
        
        ! Validate value
        if (.not. this%validate_float_value(value, width)) then
            attr%ptr = c_null_ptr
            return
        end if
        
        ! Create type and attribute
        float_type = create_float_type(this%context, width)
        attr = create_float_attribute(this%context, float_type, value)
    end function builder_float_attr

    ! Create string attribute
    function builder_string_attr(this, str) result(attr)
        class(attribute_builder_t), intent(in) :: this
        character(len=*), intent(in) :: str
        type(mlir_attribute_t) :: attr
        
        ! No special validation for strings
        attr = create_string_attribute(this%context, str)
    end function builder_string_attr

    ! Create array attribute
    function builder_array_attr(this, elements) result(attr)
        class(attribute_builder_t), intent(in) :: this
        type(mlir_attribute_t), dimension(:), intent(in) :: elements
        type(mlir_attribute_t) :: attr
        integer :: i
        
        ! Validate all elements are valid
        do i = 1, size(elements)
            if (.not. elements(i)%is_valid()) then
                attr%ptr = c_null_ptr
                return
            end if
        end do
        
        attr = create_array_attribute(this%context, elements)
    end function builder_array_attr

    ! Create boolean attribute (i1 integer)
    function builder_bool_attr(this, value) result(attr)
        class(attribute_builder_t), intent(in) :: this
        logical, intent(in) :: value
        type(mlir_attribute_t) :: attr
        integer(c_int64_t) :: int_value
        
        int_value = 0
        if (value) int_value = 1
        
        attr = this%integer_attr(1, int_value, signed=.false.)
    end function builder_bool_attr

    ! Validate integer value fits in width
    function builder_validate_integer_value(this, value, width, signed) result(valid)
        class(attribute_builder_t), intent(in) :: this
        integer(c_int64_t), intent(in) :: value
        integer, intent(in) :: width
        logical, intent(in) :: signed
        logical :: valid
        integer(c_int64_t) :: min_val, max_val
        
        valid = .true.
        
        if (signed) then
            ! Signed range: -2^(width-1) to 2^(width-1)-1
            if (width < 64) then
                min_val = -2_c_int64_t**(width-1)
                max_val = 2_c_int64_t**(width-1) - 1
                valid = (value >= min_val) .and. (value <= max_val)
            end if
        else
            ! Unsigned range: 0 to 2^width-1
            if (value < 0) then
                valid = .false.
            else if (width < 64) then
                max_val = 2_c_int64_t**width - 1
                valid = value <= max_val
            end if
        end if
    end function builder_validate_integer_value

    ! Validate float value
    function builder_validate_float_value(this, value, width) result(valid)
        class(attribute_builder_t), intent(in) :: this
        real(c_double), intent(in) :: value
        integer, intent(in) :: width
        logical :: valid
        real(c_double) :: max_val
        
        valid = .true.
        
        ! Check for NaN
        if (value /= value) then
            valid = .false.
            return
        end if
        
        ! Check range for f32
        if (width == 32) then
            max_val = huge(1.0_c_float)
            valid = abs(value) <= max_val
        end if
    end function builder_validate_float_value

end module mlir_c_attribute_builder