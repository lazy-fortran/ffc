module test_helpers
    use iso_c_binding
    use mlir_c_core
    use mlir_c_types
    use mlir_c_attributes
    use mlir_c_operations
    implicit none
    private

    public :: create_dummy_operation
    public :: create_operation_with_region

contains

    function create_dummy_operation(context) result(op)
        type(mlir_context_t), intent(in) :: context
        type(mlir_operation_t) :: op
        type(mlir_operation_state_t) :: state
        type(mlir_string_ref_t) :: name_ref
        
        name_ref = create_string_ref("test.dummy")
        state = create_operation_state(name_ref, create_unknown_location(context))
        op = create_operation(state)
    end function create_dummy_operation

    function create_operation_with_region(context, name) result(op)
        type(mlir_context_t), intent(in) :: context
        character(len=*), intent(in) :: name
        type(mlir_operation_t) :: op
        type(mlir_operation_state_t) :: state
        type(mlir_region_t) :: region
        type(mlir_string_ref_t) :: name_ref
        
        name_ref = create_string_ref(name)
        state = create_operation_state(name_ref, create_unknown_location(context))
        region = create_empty_region(context)
        call state%add_region(region)
        op = create_operation(state)
    end function create_operation_with_region

end module test_helpers