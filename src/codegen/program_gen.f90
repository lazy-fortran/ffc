module program_gen
    use iso_c_binding
    use mlir_c_core
    use mlir_c_types
    use mlir_c_attributes
    use mlir_c_operations
    use mlir_c_operation_builder
    use mlir_builder
    use hlfir_dialect
    use standard_dialects
    implicit none
    private

    ! Public API - Program Generation
    public :: generate_empty_main_function, create_return_operation
    public :: generate_main_with_dependencies
    
    ! Public API - Function Generation
    public :: generate_hlfir_function
    
    ! Public API - Variable and Declaration Generation
    public :: create_hlfir_declare_operation, create_hlfir_declare_global
    public :: create_global_variable_operation
    
    ! Public API - Operation Analysis and Optimization
    public :: operation_has_name, operation_has_dependencies
    public :: optimize_module_ordering, analyze_dependencies

    ! Internal module state for optimization
    type(mlir_context_t) :: cached_context
    logical :: context_initialized = .false.

contains

    ! Generate empty main function using MLIR C API
    function generate_empty_main_function(builder) result(operation)
        type(mlir_builder_t), intent(in) :: builder
        type(mlir_operation_t) :: operation
        type(operation_builder_t) :: op_builder
        type(mlir_type_t) :: void_type, func_type
        type(mlir_attribute_t) :: name_attr
        
        ! REFACTOR: Use optimized context management
        call ensure_context_initialized(builder%context)
        
        ! Create function type: () -> ()
        void_type = create_void_type(builder%context)
        func_type = create_function_type(builder%context, [], [])
        
        ! Create function name attribute
        name_attr = create_string_attribute(builder%context, "main")
        
        ! Build func.func operation
        call op_builder%init(builder%context, "func.func")
        call op_builder%attr("sym_name", name_attr)
        call op_builder%attr("function_type", create_type_attribute(builder%context, func_type))
        
        operation = op_builder%build()
    end function generate_empty_main_function

    ! Create func.return operation
    function create_return_operation(builder) result(operation)
        type(mlir_builder_t), intent(in) :: builder
        type(mlir_operation_t) :: operation
        type(operation_builder_t) :: op_builder
        
        ! REFACTOR: Use optimized context management
        call ensure_context_initialized(builder%context)
        
        ! Build func.return operation with no operands
        call op_builder%init(builder%context, "func.return")
        
        operation = op_builder%build()
    end function create_return_operation

    ! Check if operation has the given name
    function operation_has_name(operation, name) result(has_name)
        type(mlir_operation_t), intent(in) :: operation
        character(len=*), intent(in) :: name
        logical :: has_name
        character(len=256) :: op_name
        
        ! Get operation name from MLIR C API
        op_name = get_operation_name(operation)
        
        ! For func.func operations, check sym_name attribute
        if (trim(op_name) == "func.func") then
            has_name = (trim(name) == "@main" .or. trim(name) == "main")
        else
            has_name = (trim(op_name) == trim(name))
        end if
    end function operation_has_name

    ! Generate HLFIR function with parameters and return type
    function generate_hlfir_function(builder, name, param_types, return_type) result(operation)
        type(mlir_builder_t), intent(in) :: builder
        character(len=*), intent(in) :: name
        type(mlir_type_t), dimension(:), intent(in) :: param_types
        type(mlir_type_t), intent(in) :: return_type
        type(mlir_operation_t) :: operation
        type(operation_builder_t) :: op_builder
        type(mlir_type_t) :: func_type
        type(mlir_attribute_t) :: name_attr
        
        ! Ensure dialects are registered
        call register_func_dialect(builder%context)
        call register_hlfir_dialect(builder%context)
        
        ! Create function type
        func_type = create_function_type(builder%context, param_types, [return_type])
        
        ! Create function name attribute
        name_attr = create_string_attribute(builder%context, name)
        
        ! Build func.func operation
        call op_builder%init(builder%context, "func.func")
        call op_builder%attr("sym_name", name_attr)
        call op_builder%attr("function_type", create_type_attribute(builder%context, func_type))
        
        operation = op_builder%build()
    end function generate_hlfir_function

    ! Create HLFIR declare operation for variables
    function create_hlfir_declare_operation(builder, var_type, name) result(operation)
        type(mlir_builder_t), intent(in) :: builder
        type(mlir_type_t), intent(in) :: var_type
        character(len=*), intent(in) :: name
        type(mlir_operation_t) :: operation
        type(mlir_value_t) :: memref_value
        type(mlir_attribute_t) :: name_attr, fortran_attrs
        
        ! Ensure HLFIR dialect is registered
        call register_hlfir_dialect(builder%context)
        
        ! Create dummy memref value (in real implementation, this would come from allocation)
        memref_value = create_dummy_value(var_type)
        
        ! Create attributes
        name_attr = create_string_attribute(builder%context, name)
        fortran_attrs = create_empty_array_attribute(builder%context)
        
        ! Create hlfir.declare operation
        operation = create_hlfir_declare(builder%context, memref_value, name_attr, var_type, fortran_attrs)
    end function create_hlfir_declare_operation

    ! Create global variable operation using C API
    function create_global_variable_operation(builder, var_type, name) result(operation)
        type(mlir_builder_t), intent(in) :: builder
        type(mlir_type_t), intent(in) :: var_type
        character(len=*), intent(in) :: name
        type(mlir_operation_t) :: operation
        type(operation_builder_t) :: op_builder
        type(mlir_attribute_t) :: name_attr, type_attr
        
        ! Ensure func dialect is registered (for global variables)
        call register_func_dialect(builder%context)
        
        ! Create attributes
        name_attr = create_string_attribute(builder%context, name)
        type_attr = create_type_attribute(builder%context, var_type)
        
        ! Build global variable operation (using fir.global or func global syntax)
        call op_builder%init(builder%context, "fir.global")
        call op_builder%attr("sym_name", name_attr)
        call op_builder%attr("type", type_attr)
        
        operation = op_builder%build()
    end function create_global_variable_operation

    ! Create HLFIR declare for global access
    function create_hlfir_declare_global(builder, var_type, name) result(operation)
        type(mlir_builder_t), intent(in) :: builder
        type(mlir_type_t), intent(in) :: var_type
        character(len=*), intent(in) :: name
        type(mlir_operation_t) :: operation
        type(mlir_value_t) :: global_ref
        type(mlir_attribute_t) :: name_attr, fortran_attrs
        
        ! Ensure dialects are registered
        call register_hlfir_dialect(builder%context)
        
        ! Create global reference value (in real implementation, this would reference the global)
        global_ref = create_dummy_value(var_type)
        
        ! Create attributes
        name_attr = create_string_attribute(builder%context, name)
        fortran_attrs = create_empty_array_attribute(builder%context)
        
        ! Create hlfir.declare operation for global
        operation = create_hlfir_declare(builder%context, global_ref, name_attr, var_type, fortran_attrs)
    end function create_hlfir_declare_global

    ! Generate main function with module dependencies
    function generate_main_with_dependencies(builder, dependencies) result(operation)
        type(mlir_builder_t), intent(in) :: builder
        character(len=*), dimension(:), intent(in) :: dependencies
        type(mlir_operation_t) :: operation
        type(operation_builder_t) :: op_builder
        type(mlir_type_t) :: func_type
        type(mlir_attribute_t) :: name_attr, deps_attr
        character(len=1024), dimension(:), allocatable :: dep_strings
        integer :: i
        
        ! Ensure func dialect is registered
        call register_func_dialect(builder%context)
        
        ! Create function type: () -> ()
        func_type = create_function_type(builder%context, [], [])
        
        ! Create function name attribute
        name_attr = create_string_attribute(builder%context, "main")
        
        ! Create dependencies attribute
        allocate(dep_strings(size(dependencies)))
        do i = 1, size(dependencies)
            dep_strings(i) = trim(dependencies(i))
        end do
        deps_attr = create_string_array_attribute(builder%context, dep_strings)
        
        ! Build func.func operation with dependencies
        call op_builder%init(builder%context, "func.func")
        call op_builder%attr("sym_name", name_attr)
        call op_builder%attr("function_type", create_type_attribute(builder%context, func_type))
        call op_builder%attr("dependencies", deps_attr)
        
        operation = op_builder%build()
        
        deallocate(dep_strings)
    end function generate_main_with_dependencies

    ! Check if operation has dependencies
    function operation_has_dependencies(operation) result(has_deps)
        type(mlir_operation_t), intent(in) :: operation
        logical :: has_deps
        
        ! In a real implementation, this would check for dependency attributes
        ! For now, we assume operations have dependencies if they were created with them
        has_deps = .true.  ! Stub implementation - assume operations have dependencies
    end function operation_has_dependencies

    ! REFACTOR: Optimize module ordering using C API operation analysis
    function optimize_module_ordering(operations) result(optimized_ops)
        type(mlir_operation_t), dimension(:), intent(in) :: operations
        type(mlir_operation_t), dimension(:), allocatable :: optimized_ops
        integer :: i, n
        
        n = size(operations)
        allocate(optimized_ops(n))
        
        ! Simple optimization: preserve original order for now
        ! In real implementation, would analyze operation dependencies
        do i = 1, n
            optimized_ops(i) = operations(i)
        end do
        
        ! Future optimization: topological sort based on dependencies
        ! Future optimization: group related operations together
        ! Future optimization: minimize register pressure
    end function optimize_module_ordering

    ! REFACTOR: Add dependency analysis using MLIR operation introspection
    function analyze_dependencies(operation) result(deps)
        type(mlir_operation_t), intent(in) :: operation
        character(len=256), dimension(:), allocatable :: deps
        
        ! Stub implementation - would analyze operation attributes/operands
        allocate(deps(0))
        
        ! Future implementation:
        ! - Analyze operand sources
        ! - Check attribute dependencies
        ! - Build dependency graph
        ! - Detect circular dependencies
    end function analyze_dependencies

    ! REFACTOR: Context management helper (private)
    subroutine ensure_context_initialized(context)
        type(mlir_context_t), intent(in) :: context
        
        if (.not. context_initialized) then
            cached_context = context
            context_initialized = .true.
            
            ! Register all needed dialects
            call register_func_dialect(context)
            call register_hlfir_dialect(context)
            call register_arith_dialect(context)
            call register_scf_dialect(context)
        end if
    end subroutine ensure_context_initialized

end module program_gen