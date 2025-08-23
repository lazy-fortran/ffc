module hlfir_dialect
    use iso_c_binding
    use mlir_c_core
    use mlir_c_types
    use mlir_c_attributes
    use mlir_c_operations
    use mlir_c_operation_builder
    implicit none
    private

    ! Public functions
    public :: register_hlfir_dialect, is_hlfir_dialect_registered
    public :: create_hlfir_declare, create_hlfir_designate, create_hlfir_elemental
    public :: create_hlfir_associate, create_hlfir_end_associate
    public :: create_hlfir_expr_type, create_hlfir_var_type
    public :: create_hlfir_fortran_attrs

    ! Dialect handle
    type :: hlfir_dialect_handle_t
        type(c_ptr) :: ptr = c_null_ptr
    end type hlfir_dialect_handle_t

    ! Module variable to track registration
    logical :: dialect_registered = .false.

    ! C interface declarations
    interface
        ! Dialect registration
        function mlirGetDialectHandle__hlfir__() bind(c, name="mlirGetDialectHandle__hlfir__") result(handle)
            import :: c_ptr
            type(c_ptr) :: handle
        end function mlirGetDialectHandle__hlfir__

        subroutine mlirDialectHandleRegisterDialect(handle, context) &
            bind(c, name="mlirDialectHandleRegisterDialect")
            import :: c_ptr
            type(c_ptr), value :: handle
            type(c_ptr), value :: context
        end subroutine mlirDialectHandleRegisterDialect
    end interface

contains

    ! Register HLFIR dialect with context
    subroutine register_hlfir_dialect(context)
        type(mlir_context_t), intent(in) :: context
        type(hlfir_dialect_handle_t) :: handle
        
        ! Reuse FIR's registration function (they're in same stub)
        handle%ptr = mlirGetDialectHandle__hlfir__()
        call mlirDialectHandleRegisterDialect(handle%ptr, context%ptr)
        dialect_registered = .true.
    end subroutine register_hlfir_dialect

    ! Check if HLFIR dialect is registered
    function is_hlfir_dialect_registered(context) result(registered)
        type(mlir_context_t), intent(in) :: context
        logical :: registered
        
        registered = dialect_registered
    end function is_hlfir_dialect_registered

    ! Create hlfir.declare operation
    function create_hlfir_declare(context, memref, name_attr, result_type, fortran_attrs) result(op)
        type(mlir_context_t), intent(in) :: context
        type(mlir_value_t), intent(in) :: memref
        type(mlir_attribute_t), intent(in) :: name_attr, fortran_attrs
        type(mlir_type_t), intent(in) :: result_type
        type(mlir_operation_t) :: op
        type(operation_builder_t) :: builder
        
        call builder%init(context, "hlfir.declare")
        call builder%operand(memref)
        call builder%result(result_type)
        call builder%attr("uniq_name", name_attr)
        call builder%attr("fortran_attrs", fortran_attrs)
        
        op = builder%build()
    end function create_hlfir_declare

    ! Create hlfir.designate operation
    function create_hlfir_designate(context, base, indices, result_type) result(op)
        type(mlir_context_t), intent(in) :: context
        type(mlir_value_t), intent(in) :: base
        type(mlir_value_t), dimension(:), intent(in) :: indices
        type(mlir_type_t), intent(in) :: result_type
        type(mlir_operation_t) :: op
        type(operation_builder_t) :: builder
        integer :: i
        
        call builder%init(context, "hlfir.designate")
        call builder%operand(base)
        
        ! Add indices as operands
        do i = 1, size(indices)
            call builder%operand(indices(i))
        end do
        
        call builder%result(result_type)
        
        op = builder%build()
    end function create_hlfir_designate

    ! Create hlfir.elemental operation
    function create_hlfir_elemental(context, shape, result_type, body_region) result(op)
        type(mlir_context_t), intent(in) :: context
        type(mlir_value_t), dimension(:), intent(in) :: shape
        type(mlir_type_t), intent(in) :: result_type
        type(mlir_region_t), intent(in) :: body_region
        type(mlir_operation_t) :: op
        type(operation_builder_t) :: builder
        integer :: i
        
        call builder%init(context, "hlfir.elemental")
        
        ! Add shape operands
        do i = 1, size(shape)
            call builder%operand(shape(i))
        end do
        
        call builder%result(result_type)
        ! Note: Regions are not yet supported in our builder
        
        op = builder%build()
    end function create_hlfir_elemental

    ! Create hlfir.associate operation
    function create_hlfir_associate(context, expr, result_type, body_region) result(op)
        type(mlir_context_t), intent(in) :: context
        type(mlir_value_t), intent(in) :: expr
        type(mlir_type_t), intent(in) :: result_type
        type(mlir_region_t), intent(in) :: body_region
        type(mlir_operation_t) :: op
        type(operation_builder_t) :: builder
        
        call builder%init(context, "hlfir.associate")
        call builder%operand(expr)
        call builder%result(result_type)
        ! Note: Regions are not yet supported in our builder
        
        op = builder%build()
    end function create_hlfir_associate

    ! Create hlfir.end_associate operation
    function create_hlfir_end_associate(context, var) result(op)
        type(mlir_context_t), intent(in) :: context
        type(mlir_value_t), intent(in) :: var
        type(mlir_operation_t) :: op
        type(operation_builder_t) :: builder
        
        call builder%init(context, "hlfir.end_associate")
        call builder%operand(var)
        
        op = builder%build()
    end function create_hlfir_end_associate

    ! Create HLFIR expression type
    function create_hlfir_expr_type(context, element_type, rank) result(expr_type)
        type(mlir_context_t), intent(in) :: context
        type(mlir_type_t), intent(in) :: element_type
        integer, intent(in) :: rank
        type(mlir_type_t) :: expr_type
        integer(c_int64_t), dimension(:), allocatable :: shape
        integer :: i
        
        ! For stub, create array type for rank > 0, element type for rank = 0
        if (rank > 0) then
            allocate(shape(rank))
            do i = 1, rank
                shape(i) = 10_c_int64_t  ! Default size
            end do
            expr_type = create_array_type(context, element_type, shape)
            deallocate(shape)
        else
            expr_type = element_type
        end if
    end function create_hlfir_expr_type

    ! Create HLFIR variable type
    function create_hlfir_var_type(context, element_type) result(var_type)
        type(mlir_context_t), intent(in) :: context
        type(mlir_type_t), intent(in) :: element_type
        type(mlir_type_t) :: var_type
        
        ! For stub, use reference type
        var_type = create_reference_type(context, element_type)
    end function create_hlfir_var_type

    ! Create HLFIR Fortran attributes
    function create_hlfir_fortran_attrs(context, contiguous, target, optional) result(attrs)
        type(mlir_context_t), intent(in) :: context
        logical, intent(in) :: contiguous, target, optional
        type(mlir_attribute_t) :: attrs
        character(len=64) :: attr_str
        
        ! Create a string representation of attributes
        write(attr_str, '(A,L1,A,L1,A,L1)') &
            'contiguous=', contiguous, &
            ',target=', target, &
            ',optional=', optional
        
        attrs = create_string_attribute(context, trim(attr_str))
    end function create_hlfir_fortran_attrs

end module hlfir_dialect