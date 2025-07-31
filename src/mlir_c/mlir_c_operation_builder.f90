module mlir_c_operation_builder
    use iso_c_binding
    use mlir_c_core
    use mlir_c_types
    use mlir_c_attributes
    use mlir_c_operations
    implicit none
    private

    ! Public types
    public :: operation_builder_t
    
    ! Public functions
    public :: build_binary_op, build_constant_op, build_unary_op

    ! Fluent operation builder
    type :: operation_builder_t
        private
        type(mlir_context_t) :: context
        type(mlir_location_t) :: loc
        type(mlir_operation_state_t) :: state
        character(len=:), allocatable :: op_name
        logical :: initialized = .false.
    contains
        procedure :: init => builder_init
        procedure :: operand => builder_add_operand
        procedure :: operands => builder_add_operands
        procedure :: result => builder_add_result
        procedure :: results => builder_add_results
        procedure :: attr => builder_add_attribute
        procedure :: location => builder_set_location
        procedure :: build => builder_build
        procedure :: reset => builder_reset
    end type operation_builder_t

contains

    ! Initialize builder
    subroutine builder_init(this, context, op_name)
        class(operation_builder_t), intent(out) :: this
        type(mlir_context_t), intent(in) :: context
        character(len=*), intent(in) :: op_name
        type(mlir_string_ref_t) :: name_ref
        
        this%context = context
        this%op_name = op_name
        this%loc = create_unknown_location(context)
        
        ! Create operation state
        name_ref = create_string_ref(op_name)
        this%state = create_operation_state(name_ref, this%loc)
        this%initialized = .true.
    end subroutine builder_init

    ! Add operand
    subroutine builder_add_operand(this, value)
        class(operation_builder_t), intent(inout) :: this
        type(mlir_value_t), intent(in) :: value
        
        if (this%initialized) then
            call add_operand(this%state, value)
        end if
    end subroutine builder_add_operand

    ! Add multiple operands
    subroutine builder_add_operands(this, values)
        class(operation_builder_t), intent(inout) :: this
        type(mlir_value_t), dimension(:), intent(in) :: values
        
        if (this%initialized) then
            call add_operands(this%state, values)
        end if
    end subroutine builder_add_operands

    ! Add result type
    subroutine builder_add_result(this, result_type)
        class(operation_builder_t), intent(inout) :: this
        type(mlir_type_t), intent(in) :: result_type
        
        if (this%initialized) then
            call add_result(this%state, result_type)
        end if
    end subroutine builder_add_result

    ! Add multiple result types
    subroutine builder_add_results(this, result_types)
        class(operation_builder_t), intent(inout) :: this
        type(mlir_type_t), dimension(:), intent(in) :: result_types
        
        if (this%initialized) then
            call add_results(this%state, result_types)
        end if
    end subroutine builder_add_results

    ! Add attribute
    subroutine builder_add_attribute(this, name, attribute)
        class(operation_builder_t), intent(inout) :: this
        character(len=*), intent(in) :: name
        type(mlir_attribute_t), intent(in) :: attribute
        type(mlir_string_ref_t) :: name_ref
        
        if (this%initialized) then
            name_ref = create_string_ref(name)
            call add_attribute(this%state, name_ref, attribute)
        end if
    end subroutine builder_add_attribute

    ! Set location
    subroutine builder_set_location(this, loc)
        class(operation_builder_t), intent(inout) :: this
        type(mlir_location_t), intent(in) :: loc
        
        this%loc = loc
    end subroutine builder_set_location

    ! Build the operation
    function builder_build(this) result(op)
        class(operation_builder_t), intent(inout) :: this
        type(mlir_operation_t) :: op
        
        if (this%initialized .and. len_trim(this%op_name) > 0) then
            op = create_operation(this%state)
        else
            op%ptr = c_null_ptr
        end if
        
        ! Reset builder after use
        call this%reset()
    end function builder_build

    ! Reset builder state
    subroutine builder_reset(this)
        class(operation_builder_t), intent(inout) :: this
        
        if (this%initialized) then
            call destroy_operation_state(this%state)
            this%initialized = .false.
        end if
        
        if (allocated(this%op_name)) deallocate(this%op_name)
    end subroutine builder_reset

    ! Template for binary operations
    function build_binary_op(context, op_name, lhs, rhs, result_type) result(op)
        type(mlir_context_t), intent(in) :: context
        character(len=*), intent(in) :: op_name
        type(mlir_value_t), intent(in) :: lhs, rhs
        type(mlir_type_t), intent(in) :: result_type
        type(mlir_operation_t) :: op
        type(operation_builder_t) :: builder
        
        call builder%init(context, op_name)
        call builder%operand(lhs)
        call builder%operand(rhs)
        call builder%result(result_type)
        
        op = builder%build()
    end function build_binary_op

    ! Template for constant operations
    function build_constant_op(context, result_type, value_attr) result(op)
        type(mlir_context_t), intent(in) :: context
        type(mlir_type_t), intent(in) :: result_type
        type(mlir_attribute_t), intent(in) :: value_attr
        type(mlir_operation_t) :: op
        type(operation_builder_t) :: builder
        
        call builder%init(context, "arith.constant")
        call builder%result(result_type)
        call builder%attr("value", value_attr)
        
        op = builder%build()
    end function build_constant_op

    ! Template for unary operations
    function build_unary_op(context, op_name, operand, result_type) result(op)
        type(mlir_context_t), intent(in) :: context
        character(len=*), intent(in) :: op_name
        type(mlir_value_t), intent(in) :: operand
        type(mlir_type_t), intent(in) :: result_type
        type(mlir_operation_t) :: op
        type(operation_builder_t) :: builder
        
        call builder%init(context, op_name)
        call builder%operand(operand)
        call builder%result(result_type)
        
        op = builder%build()
    end function build_unary_op

end module mlir_c_operation_builder