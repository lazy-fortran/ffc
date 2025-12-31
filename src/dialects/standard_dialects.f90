module standard_dialects
    use iso_c_binding
    use mlir_c_core
    use mlir_c_types
    use mlir_c_attributes
    use mlir_c_operations
    use mlir_c_operation_builder
    implicit none
    private

    ! Public functions for dialect registration
    public :: register_func_dialect, register_arith_dialect, register_scf_dialect
    
    ! Public functions for func dialect
    public :: create_func_func, create_func_return, create_func_call
    public :: create_function_type
    
    ! Public functions for arith dialect
    public :: create_arith_addi, create_arith_muli, create_arith_cmpf
    public :: create_arith_constant
    
    ! Public functions for scf dialect
    public :: create_scf_if, create_scf_for, create_scf_while, create_scf_yield

    ! Module variables to track registration
    logical :: func_registered = .false.
    logical :: arith_registered = .false.
    logical :: scf_registered = .false.

    ! C interface declarations
    interface
        ! Dialect registration
        function mlirGetDialectHandle__func__() bind(c, name="ffc_mlirGetDialectHandle__func__") result(handle)
            import :: c_ptr
            type(c_ptr) :: handle
        end function mlirGetDialectHandle__func__

        function mlirGetDialectHandle__arith__() bind(c, name="ffc_mlirGetDialectHandle__arith__") result(handle)
            import :: c_ptr
            type(c_ptr) :: handle
        end function mlirGetDialectHandle__arith__

        function mlirGetDialectHandle__scf__() bind(c, name="ffc_mlirGetDialectHandle__scf__") result(handle)
            import :: c_ptr
            type(c_ptr) :: handle
        end function mlirGetDialectHandle__scf__

        subroutine mlirDialectHandleRegisterDialect(handle, context) &
            bind(c, name="ffc_mlirDialectHandleRegisterDialect")
            import :: c_ptr
            type(c_ptr), value :: handle
            type(c_ptr), value :: context
        end subroutine mlirDialectHandleRegisterDialect
    end interface

contains

    ! Helper function to create simple operations 
    function create_memory_operation(context, name, operands) result(op)
        type(mlir_context_t), intent(in) :: context
        character(len=*), intent(in) :: name
        type(mlir_value_t), dimension(:), intent(in) :: operands
        type(mlir_operation_t) :: op
        type(operation_builder_t) :: builder
        integer :: i
        
        call builder%init(context, name)
        
        ! Add operands
        do i = 1, size(operands)
            call builder%operand(operands(i))
        end do
        
        op = builder%build()
    end function create_memory_operation

    ! Register func dialect
    subroutine register_func_dialect(context)
        type(mlir_context_t), intent(in) :: context
        type(c_ptr) :: handle
        
        handle = mlirGetDialectHandle__func__()
        call mlirDialectHandleRegisterDialect(handle, context%ptr)
        func_registered = .true.
    end subroutine register_func_dialect

    ! Register arith dialect
    subroutine register_arith_dialect(context)
        type(mlir_context_t), intent(in) :: context
        type(c_ptr) :: handle
        
        handle = mlirGetDialectHandle__arith__()
        call mlirDialectHandleRegisterDialect(handle, context%ptr)
        arith_registered = .true.
    end subroutine register_arith_dialect

    ! Register scf dialect
    subroutine register_scf_dialect(context)
        type(mlir_context_t), intent(in) :: context
        type(c_ptr) :: handle
        
        handle = mlirGetDialectHandle__scf__()
        call mlirDialectHandleRegisterDialect(handle, context%ptr)
        scf_registered = .true.
    end subroutine register_scf_dialect

    ! Create function type
    function create_function_type(context, input_types, result_types) result(func_type)
        type(mlir_context_t), intent(in) :: context
        type(mlir_type_t), dimension(:), intent(in) :: input_types
        type(mlir_type_t), dimension(:), intent(in) :: result_types
        type(mlir_type_t) :: func_type
        type(mlir_type_t) :: void_type

        ! For stub, create a reference type as approximation
        if (size(result_types) > 0) then
            func_type = create_reference_type(context, result_types(1))
        else if (size(input_types) > 0) then
            func_type = create_reference_type(context, input_types(1))
        else
            ! No inputs or results - create a void function type
            void_type = create_void_type(context)
            func_type = create_reference_type(context, void_type)
        end if
    end function create_function_type

    ! Create func.func operation
    function create_func_func(context, name, func_type, body_region) result(op)
        type(mlir_context_t), intent(in) :: context
        character(len=*), intent(in) :: name
        type(mlir_type_t), intent(in) :: func_type
        type(mlir_region_t), intent(in) :: body_region
        type(mlir_operation_t) :: op
        type(operation_builder_t) :: builder
        type(mlir_attribute_t) :: name_attr, type_attr
        
        call builder%init(context, "func.func")
        
        name_attr = create_string_attribute(context, name)
        call builder%attr("sym_name", name_attr)
        
        ! For stub, use string attribute for function type
        type_attr = create_string_attribute(context, "function_type")
        call builder%attr("function_type", type_attr)
        
        ! Note: Regions not yet supported in builder
        
        op = builder%build()
    end function create_func_func

    ! Create func.return operation
    function create_func_return(context, operands) result(op)
        type(mlir_context_t), intent(in) :: context
        type(mlir_value_t), dimension(:), intent(in) :: operands
        type(mlir_operation_t) :: op
        
        op = create_memory_operation(context, "func.return", operands)
    end function create_func_return

    ! Create func.call operation
    function create_func_call(context, callee, operands, result_types) result(op)
        type(mlir_context_t), intent(in) :: context
        character(len=*), intent(in) :: callee
        type(mlir_value_t), dimension(:), intent(in) :: operands
        type(mlir_type_t), dimension(:), intent(in) :: result_types
        type(mlir_operation_t) :: op
        type(operation_builder_t) :: builder
        type(mlir_attribute_t) :: callee_attr
        integer :: i
        
        call builder%init(context, "func.call")
        
        callee_attr = create_string_attribute(context, callee)
        call builder%attr("callee", callee_attr)
        
        ! Add operands
        do i = 1, size(operands)
            call builder%operand(operands(i))
        end do
        
        ! Add result types
        do i = 1, size(result_types)
            call builder%result(result_types(i))
        end do
        
        op = builder%build()
    end function create_func_call

    ! Create arith.addi operation
    function create_arith_addi(context, lhs, rhs, result_type) result(op)
        type(mlir_context_t), intent(in) :: context
        type(mlir_value_t), intent(in) :: lhs, rhs
        type(mlir_type_t), intent(in) :: result_type
        type(mlir_operation_t) :: op
        
        op = build_binary_op(context, "arith.addi", lhs, rhs, result_type)
    end function create_arith_addi

    ! Create arith.muli operation
    function create_arith_muli(context, lhs, rhs, result_type) result(op)
        type(mlir_context_t), intent(in) :: context
        type(mlir_value_t), intent(in) :: lhs, rhs
        type(mlir_type_t), intent(in) :: result_type
        type(mlir_operation_t) :: op
        
        op = build_binary_op(context, "arith.muli", lhs, rhs, result_type)
    end function create_arith_muli

    ! Create arith.cmpf operation
    function create_arith_cmpf(context, predicate, lhs, rhs, result_type) result(op)
        type(mlir_context_t), intent(in) :: context
        character(len=*), intent(in) :: predicate
        type(mlir_value_t), intent(in) :: lhs, rhs
        type(mlir_type_t), intent(in) :: result_type
        type(mlir_operation_t) :: op
        type(operation_builder_t) :: builder
        type(mlir_attribute_t) :: pred_attr
        
        call builder%init(context, "arith.cmpf")
        call builder%operand(lhs)
        call builder%operand(rhs)
        call builder%result(result_type)
        
        pred_attr = create_string_attribute(context, predicate)
        call builder%attr("predicate", pred_attr)
        
        op = builder%build()
    end function create_arith_cmpf

    ! Create arith.constant operation
    function create_arith_constant(context, value, result_type) result(op)
        type(mlir_context_t), intent(in) :: context
        type(mlir_attribute_t), intent(in) :: value
        type(mlir_type_t), intent(in) :: result_type
        type(mlir_operation_t) :: op
        
        op = build_constant_op(context, result_type, value)
    end function create_arith_constant

    ! Create scf.if operation
    function create_scf_if(context, condition, result_types, then_region, else_region) result(op)
        type(mlir_context_t), intent(in) :: context
        type(mlir_value_t), intent(in) :: condition
        type(mlir_type_t), dimension(:), intent(in) :: result_types
        type(mlir_region_t), intent(in) :: then_region, else_region
        type(mlir_operation_t) :: op
        type(operation_builder_t) :: builder
        integer :: i
        
        call builder%init(context, "scf.if")
        call builder%operand(condition)
        
        ! Add result types
        do i = 1, size(result_types)
            call builder%result(result_types(i))
        end do
        
        ! Note: Regions not yet supported in builder
        
        op = builder%build()
    end function create_scf_if

    ! Create scf.for operation
    function create_scf_for(context, lower, upper, step, init_vals, body_region) result(op)
        type(mlir_context_t), intent(in) :: context
        type(mlir_value_t), intent(in) :: lower, upper, step
        type(mlir_value_t), dimension(:), intent(in) :: init_vals
        type(mlir_region_t), intent(in) :: body_region
        type(mlir_operation_t) :: op
        type(operation_builder_t) :: builder
        integer :: i
        
        call builder%init(context, "scf.for")
        call builder%operand(lower)
        call builder%operand(upper)
        call builder%operand(step)
        
        ! Add initial values
        do i = 1, size(init_vals)
            call builder%operand(init_vals(i))
        end do
        
        ! Note: Regions not yet supported in builder
        
        op = builder%build()
    end function create_scf_for

    ! Create scf.while operation
    function create_scf_while(context, init_vals, before_region, after_region) result(op)
        type(mlir_context_t), intent(in) :: context
        type(mlir_value_t), dimension(:), intent(in) :: init_vals
        type(mlir_region_t), intent(in) :: before_region, after_region
        type(mlir_operation_t) :: op
        type(operation_builder_t) :: builder
        integer :: i
        
        call builder%init(context, "scf.while")
        
        ! Add initial values
        do i = 1, size(init_vals)
            call builder%operand(init_vals(i))
        end do
        
        ! Note: Regions not yet supported in builder
        
        op = builder%build()
    end function create_scf_while

    ! Create scf.yield operation
    function create_scf_yield(context, operands) result(op)
        type(mlir_context_t), intent(in) :: context
        type(mlir_value_t), dimension(:), intent(in) :: operands
        type(mlir_operation_t) :: op
        
        op = create_memory_operation(context, "scf.yield", operands)
    end function create_scf_yield

end module standard_dialects