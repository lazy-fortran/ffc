module fir_dialect
    use iso_c_binding
    use mlir_c_core
    use mlir_c_types
    use mlir_c_attributes
    use mlir_c_operations
    use mlir_c_operation_builder
    implicit none
    private

    ! Public functions
    public :: register_fir_dialect, is_fir_dialect_registered
    public :: create_fir_declare, create_fir_load, create_fir_store
    public :: create_fir_alloca, create_fir_do_loop, create_fir_if

    ! Dialect handle
    type :: fir_dialect_handle_t
        type(c_ptr) :: ptr = c_null_ptr
    end type fir_dialect_handle_t

    ! Module variable to track registration
    logical :: dialect_registered = .false.

    ! C interface declarations
    interface
        ! Dialect registration
        function mlirGetDialectHandle__fir__() bind(c, name="ffc_mlirGetDialectHandle__fir__") result(handle)
            import :: c_ptr
            type(c_ptr) :: handle
        end function mlirGetDialectHandle__fir__

        subroutine mlirDialectHandleRegisterDialect(handle, context) &
            bind(c, name="ffc_mlirDialectHandleRegisterDialect")
            import :: c_ptr
            type(c_ptr), value :: handle
            type(c_ptr), value :: context
        end subroutine mlirDialectHandleRegisterDialect
    end interface

contains

    ! Register FIR dialect with context
    subroutine register_fir_dialect(context)
        type(mlir_context_t), intent(in) :: context
        type(fir_dialect_handle_t) :: handle
        
        handle%ptr = mlirGetDialectHandle__fir__()
        call mlirDialectHandleRegisterDialect(handle%ptr, context%ptr)
        dialect_registered = .true.
    end subroutine register_fir_dialect

    ! Check if FIR dialect is registered
    function is_fir_dialect_registered(context) result(registered)
        type(mlir_context_t), intent(in) :: context
        logical :: registered
        
        ! For now, just return the module variable
        registered = dialect_registered
    end function is_fir_dialect_registered

    ! Create fir.declare operation
    function create_fir_declare(context, memref, name_attr, result_type) result(op)
        type(mlir_context_t), intent(in) :: context
        type(mlir_value_t), intent(in) :: memref
        type(mlir_attribute_t), intent(in) :: name_attr
        type(mlir_type_t), intent(in) :: result_type
        type(mlir_operation_t) :: op
        type(operation_builder_t) :: builder
        
        call builder%init(context, "fir.declare")
        call builder%operand(memref)
        call builder%result(result_type)
        call builder%attr("uniq_name", name_attr)
        
        op = builder%build()
    end function create_fir_declare

    ! Create fir.load operation
    function create_fir_load(context, memref, result_type) result(op)
        type(mlir_context_t), intent(in) :: context
        type(mlir_value_t), intent(in) :: memref
        type(mlir_type_t), intent(in) :: result_type
        type(mlir_operation_t) :: op
        type(operation_builder_t) :: builder
        
        call builder%init(context, "fir.load")
        call builder%operand(memref)
        call builder%result(result_type)
        
        op = builder%build()
    end function create_fir_load

    ! Create fir.store operation
    function create_fir_store(context, value, memref) result(op)
        type(mlir_context_t), intent(in) :: context
        type(mlir_value_t), intent(in) :: value, memref
        type(mlir_operation_t) :: op
        type(operation_builder_t) :: builder
        
        call builder%init(context, "fir.store")
        call builder%operand(value)
        call builder%operand(memref)
        
        op = builder%build()
    end function create_fir_store

    ! Create fir.alloca operation
    function create_fir_alloca(context, in_type, result_type) result(op)
        type(mlir_context_t), intent(in) :: context
        type(mlir_type_t), intent(in) :: in_type, result_type
        type(mlir_operation_t) :: op
        type(operation_builder_t) :: builder
        
        call builder%init(context, "fir.alloca")
        call builder%result(result_type)
        call builder%attr("in_type", create_fir_type_attribute(context, in_type))
        
        op = builder%build()
    end function create_fir_alloca

    ! Create fir.do_loop operation
    function create_fir_do_loop(context, lower, upper, step, body_region) result(op)
        type(mlir_context_t), intent(in) :: context
        type(mlir_value_t), intent(in) :: lower, upper, step
        type(mlir_region_t), intent(in) :: body_region
        type(mlir_operation_t) :: op
        type(operation_builder_t) :: builder
        
        call builder%init(context, "fir.do_loop")
        call builder%operand(lower)
        call builder%operand(upper)
        call builder%operand(step)
        ! Note: Regions are not yet supported in our builder
        ! For now, we'll create without the region
        
        op = builder%build()
    end function create_fir_do_loop

    ! Create fir.if operation
    function create_fir_if(context, condition, then_region, else_region) result(op)
        type(mlir_context_t), intent(in) :: context
        type(mlir_value_t), intent(in) :: condition
        type(mlir_region_t), intent(in) :: then_region, else_region
        type(mlir_operation_t) :: op
        type(operation_builder_t) :: builder
        
        call builder%init(context, "fir.if")
        call builder%operand(condition)
        ! Note: Regions are not yet supported in our builder
        ! For now, we'll create without the regions
        
        op = builder%build()
    end function create_fir_if

    ! Helper to create FIR type attribute (needed for fir.alloca)
    function create_fir_type_attribute(context, type) result(attr)
        type(mlir_context_t), intent(in) :: context
        type(mlir_type_t), intent(in) :: type
        type(mlir_attribute_t) :: attr
        
        ! Use the real type attribute creation from mlir_c_attributes
        attr = create_type_attribute(context, type)
    end function create_fir_type_attribute

end module fir_dialect