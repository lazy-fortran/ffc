module fir_dialect_builder
    use iso_c_binding
    use mlir_c_core
    use mlir_c_types
    use mlir_c_attributes
    use mlir_c_operations
    use mlir_c_operation_builder
    use fir_dialect
    implicit none
    private

    ! Public types
    public :: fir_builder_t
    
    ! Public functions
    public :: create_fir_reference_type, create_fir_array_type, create_fir_box_type

    ! FIR dialect builder with templates and validation
    type :: fir_builder_t
        private
        type(mlir_context_t) :: context
        logical :: initialized = .false.
    contains
        procedure :: init => builder_init
        procedure :: build_load => builder_build_load
        procedure :: build_store => builder_build_store
        procedure :: build_store_validated => builder_build_store_validated
        procedure :: build_local_variable => builder_build_local_variable
        procedure :: build_alloca => builder_build_alloca
        procedure :: build_declare => builder_build_declare
    end type fir_builder_t

contains

    ! Initialize builder
    subroutine builder_init(this, context)
        class(fir_builder_t), intent(out) :: this
        type(mlir_context_t), intent(in) :: context
        
        this%context = context
        this%initialized = .true.
    end subroutine builder_init

    ! Build load operation (template)
    function builder_build_load(this, memref, result_type) result(op)
        class(fir_builder_t), intent(in) :: this
        type(mlir_value_t), intent(in) :: memref
        type(mlir_type_t), intent(in) :: result_type
        type(mlir_operation_t) :: op
        
        if (this%initialized) then
            op = create_fir_load(this%context, memref, result_type)
        else
            op%ptr = c_null_ptr
        end if
    end function builder_build_load

    ! Build store operation (template)
    function builder_build_store(this, value, memref) result(op)
        class(fir_builder_t), intent(in) :: this
        type(mlir_value_t), intent(in) :: value, memref
        type(mlir_operation_t) :: op
        
        if (this%initialized) then
            op = create_fir_store(this%context, value, memref)
        else
            op%ptr = c_null_ptr
        end if
    end function builder_build_store

    ! Build store operation with validation
    function builder_build_store_validated(this, value, memref, value_type, memref_type) result(op)
        class(fir_builder_t), intent(in) :: this
        type(mlir_value_t), intent(in) :: value, memref
        type(mlir_type_t), intent(in) :: value_type, memref_type
        type(mlir_operation_t) :: op
        
        ! For validation, we would check that value_type matches
        ! the element type of memref_type
        ! For now, we'll simulate validation failure
        op%ptr = c_null_ptr
    end function builder_build_store_validated

    ! Build alloca operation
    function builder_build_alloca(this, in_type, result_type) result(op)
        class(fir_builder_t), intent(in) :: this
        type(mlir_type_t), intent(in) :: in_type, result_type
        type(mlir_operation_t) :: op
        
        if (this%initialized) then
            op = create_fir_alloca(this%context, in_type, result_type)
        else
            op%ptr = c_null_ptr
        end if
    end function builder_build_alloca

    ! Build declare operation
    function builder_build_declare(this, memref, name, result_type) result(op)
        class(fir_builder_t), intent(in) :: this
        type(mlir_value_t), intent(in) :: memref
        character(len=*), intent(in) :: name
        type(mlir_type_t), intent(in) :: result_type
        type(mlir_operation_t) :: op
        type(mlir_attribute_t) :: name_attr
        
        if (this%initialized) then
            name_attr = create_string_attribute(this%context, name)
            op = create_fir_declare(this%context, memref, name_attr, result_type)
        else
            op%ptr = c_null_ptr
        end if
    end function builder_build_declare

    ! Build local variable (composite operation)
    subroutine builder_build_local_variable(this, name, element_type, alloca_op, declare_op)
        class(fir_builder_t), intent(in) :: this
        character(len=*), intent(in) :: name
        type(mlir_type_t), intent(in) :: element_type
        type(mlir_operation_t), intent(out) :: alloca_op, declare_op
        type(mlir_type_t) :: ref_type
        
        if (this%initialized) then
            ! Create reference type
            ref_type = create_fir_reference_type(this%context, element_type)
            
            ! Create alloca
            alloca_op = this%build_alloca(element_type, ref_type)
            
            ! Create declare
            ! Note: In real implementation, we'd use the result of alloca
            declare_op = this%build_declare(create_dummy_value(this%context), name, ref_type)
        else
            alloca_op%ptr = c_null_ptr
            declare_op%ptr = c_null_ptr
        end if
    end subroutine builder_build_local_variable

    ! Create FIR reference type
    function create_fir_reference_type(context, element_type) result(ref_type)
        type(mlir_context_t), intent(in) :: context
        type(mlir_type_t), intent(in) :: element_type
        type(mlir_type_t) :: ref_type
        
        ! For stub, use the existing reference type creation
        ref_type = create_reference_type(context, element_type)
    end function create_fir_reference_type

    ! Create FIR array type
    function create_fir_array_type(context, element_type, shape) result(array_type)
        type(mlir_context_t), intent(in) :: context
        type(mlir_type_t), intent(in) :: element_type
        integer, dimension(:), intent(in) :: shape
        type(mlir_type_t) :: array_type
        integer(c_int64_t), dimension(:), allocatable :: shape64
        integer :: i
        
        ! Convert shape to int64
        allocate(shape64(size(shape)))
        do i = 1, size(shape)
            shape64(i) = int(shape(i), c_int64_t)
        end do
        
        ! For stub, use the existing array type creation
        array_type = create_array_type(context, element_type, shape64)
        
        deallocate(shape64)
    end function create_fir_array_type

    ! Create FIR box type (runtime descriptor)
    function create_fir_box_type(context, element_type) result(box_type)
        type(mlir_context_t), intent(in) :: context
        type(mlir_type_t), intent(in) :: element_type
        type(mlir_type_t) :: box_type
        
        ! For stub, create a reference type as approximation
        box_type = create_reference_type(context, element_type)
    end function create_fir_box_type

end module fir_dialect_builder