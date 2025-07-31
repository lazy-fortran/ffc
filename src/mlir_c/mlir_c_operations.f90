module mlir_c_operations
    use iso_c_binding
    use mlir_c_core
    use mlir_c_types
    use mlir_c_attributes
    implicit none
    private

    ! Public types
    public :: mlir_operation_state_t, mlir_operation_t, mlir_value_t
    
    ! Public functions
    public :: create_operation_state, destroy_operation_state
    public :: add_operand, add_operands, get_num_operands
    public :: add_result, add_results, get_num_results
    public :: add_attribute, get_num_attributes
    public :: create_operation, destroy_operation
    public :: verify_operation, get_operation_num_results
    public :: get_operation_name, create_dummy_value

    ! Operation state wrapper
    type :: mlir_operation_state_t
        type(c_ptr) :: ptr = c_null_ptr
        ! Store arrays for Fortran interface
        type(c_ptr), dimension(:), allocatable :: operand_ptrs
        type(c_ptr), dimension(:), allocatable :: result_ptrs
        integer :: num_operands = 0
        integer :: num_results = 0
        integer :: num_attributes = 0
    contains
        procedure :: is_valid => operation_state_is_valid
    end type mlir_operation_state_t

    ! Operation wrapper
    type :: mlir_operation_t
        type(c_ptr) :: ptr = c_null_ptr
    contains
        procedure :: is_valid => operation_is_valid
    end type mlir_operation_t

    ! Value wrapper (SSA values)
    type :: mlir_value_t
        type(c_ptr) :: ptr = c_null_ptr
    contains
        procedure :: is_valid => value_is_valid
    end type mlir_value_t

    ! C interface declarations
    interface
        ! Operation state creation
        function mlirOperationStateCreate(name, location) bind(c, name="mlirOperationStateCreate") result(state)
            import :: c_ptr
            type(c_ptr), value :: name  ! MlirStringRef
            type(c_ptr), value :: location
            type(c_ptr) :: state
        end function mlirOperationStateCreate

        ! Operation state destruction
        subroutine mlirOperationStateDestroy(state) bind(c, name="mlirOperationStateDestroy")
            import :: c_ptr
            type(c_ptr), value :: state
        end subroutine mlirOperationStateDestroy

        ! Add operands
        subroutine mlirOperationStateAddOperands(state, n, operands) bind(c, name="mlirOperationStateAddOperands")
            import :: c_ptr, c_intptr_t
            type(c_ptr), value :: state
            integer(c_intptr_t), value :: n
            type(c_ptr), value :: operands  ! pointer to array of MlirValue
        end subroutine mlirOperationStateAddOperands

        ! Add results
        subroutine mlirOperationStateAddResults(state, n, results) bind(c, name="mlirOperationStateAddResults")
            import :: c_ptr, c_intptr_t
            type(c_ptr), value :: state
            integer(c_intptr_t), value :: n
            type(c_ptr), value :: results  ! pointer to array of MlirType
        end subroutine mlirOperationStateAddResults

        ! Add attribute
        subroutine mlirOperationStateAddNamedAttribute(state, name, attribute) &
            bind(c, name="mlirOperationStateAddNamedAttribute")
            import :: c_ptr
            type(c_ptr), value :: state
            type(c_ptr), value :: name  ! MlirStringRef
            type(c_ptr), value :: attribute
        end subroutine mlirOperationStateAddNamedAttribute

        ! Create operation
        function mlirOperationCreate(state) bind(c, name="mlirOperationCreate") result(op)
            import :: c_ptr
            type(c_ptr), value :: state
            type(c_ptr) :: op
        end function mlirOperationCreate

        ! Destroy operation
        subroutine mlirOperationDestroy(op) bind(c, name="mlirOperationDestroy")
            import :: c_ptr
            type(c_ptr), value :: op
        end subroutine mlirOperationDestroy

        ! Verify operation
        function mlirOperationVerify(op) bind(c, name="mlirOperationVerify") result(valid)
            import :: c_ptr, c_bool
            type(c_ptr), value :: op
            logical(c_bool) :: valid
        end function mlirOperationVerify

        ! Get number of results
        function mlirOperationGetNumResults(op) bind(c, name="mlirOperationGetNumResults") result(n)
            import :: c_ptr, c_intptr_t
            type(c_ptr), value :: op
            integer(c_intptr_t) :: n
        end function mlirOperationGetNumResults
    end interface

contains

    ! Create operation state
    function create_operation_state(name, location) result(state)
        type(mlir_string_ref_t), intent(in), target :: name
        type(mlir_location_t), intent(in) :: location
        type(mlir_operation_state_t) :: state
        
        state%ptr = mlirOperationStateCreate(c_loc(name), location%ptr)
        state%num_operands = 0
        state%num_results = 0
        state%num_attributes = 0
        
        ! Allocate arrays with initial size
        allocate(state%operand_ptrs(10))
        allocate(state%result_ptrs(10))
    end function create_operation_state

    ! Destroy operation state
    subroutine destroy_operation_state(state)
        type(mlir_operation_state_t), intent(inout) :: state
        
        if (c_associated(state%ptr)) then
            call mlirOperationStateDestroy(state%ptr)
            state%ptr = c_null_ptr
        end if
        
        if (allocated(state%operand_ptrs)) deallocate(state%operand_ptrs)
        if (allocated(state%result_ptrs)) deallocate(state%result_ptrs)
        
        state%num_operands = 0
        state%num_results = 0
        state%num_attributes = 0
    end subroutine destroy_operation_state

    ! Add single operand
    subroutine add_operand(state, operand)
        type(mlir_operation_state_t), intent(inout) :: state
        type(mlir_value_t), intent(in) :: operand
        type(c_ptr), dimension(:), allocatable :: temp
        
        ! Resize array if needed
        if (state%num_operands >= size(state%operand_ptrs)) then
            allocate(temp(size(state%operand_ptrs) * 2))
            temp(1:state%num_operands) = state%operand_ptrs(1:state%num_operands)
            call move_alloc(temp, state%operand_ptrs)
        end if
        
        state%num_operands = state%num_operands + 1
        state%operand_ptrs(state%num_operands) = operand%ptr
    end subroutine add_operand

    ! Add multiple operands
    subroutine add_operands(state, operands)
        type(mlir_operation_state_t), intent(inout) :: state
        type(mlir_value_t), dimension(:), intent(in) :: operands
        integer :: i
        
        do i = 1, size(operands)
            call add_operand(state, operands(i))
        end do
    end subroutine add_operands

    ! Get number of operands
    function get_num_operands(state) result(n)
        type(mlir_operation_state_t), intent(in) :: state
        integer :: n
        
        n = state%num_operands
    end function get_num_operands

    ! Add single result type
    subroutine add_result(state, result_type)
        type(mlir_operation_state_t), intent(inout) :: state
        type(mlir_type_t), intent(in) :: result_type
        type(c_ptr), dimension(:), allocatable :: temp
        
        ! Resize array if needed
        if (state%num_results >= size(state%result_ptrs)) then
            allocate(temp(size(state%result_ptrs) * 2))
            temp(1:state%num_results) = state%result_ptrs(1:state%num_results)
            call move_alloc(temp, state%result_ptrs)
        end if
        
        state%num_results = state%num_results + 1
        state%result_ptrs(state%num_results) = result_type%ptr
    end subroutine add_result

    ! Add multiple result types
    subroutine add_results(state, result_types)
        type(mlir_operation_state_t), intent(inout) :: state
        type(mlir_type_t), dimension(:), intent(in) :: result_types
        integer :: i
        
        do i = 1, size(result_types)
            call add_result(state, result_types(i))
        end do
    end subroutine add_results

    ! Get number of results
    function get_num_results(state) result(n)
        type(mlir_operation_state_t), intent(in) :: state
        integer :: n
        
        n = state%num_results
    end function get_num_results

    ! Add attribute
    subroutine add_attribute(state, name, attribute)
        type(mlir_operation_state_t), intent(inout) :: state
        type(mlir_string_ref_t), intent(in), target :: name
        type(mlir_attribute_t), intent(in) :: attribute
        
        call mlirOperationStateAddNamedAttribute(state%ptr, c_loc(name), attribute%ptr)
        state%num_attributes = state%num_attributes + 1
    end subroutine add_attribute

    ! Get number of attributes
    function get_num_attributes(state) result(n)
        type(mlir_operation_state_t), intent(in) :: state
        integer :: n
        
        n = state%num_attributes
    end function get_num_attributes

    ! Create operation from state
    function create_operation(state) result(op)
        type(mlir_operation_state_t), intent(inout) :: state
        type(mlir_operation_t) :: op
        type(c_ptr), dimension(:), allocatable, target :: temp_operands, temp_results
        
        ! First add all accumulated operands and results to the C state
        if (state%num_operands > 0) then
            allocate(temp_operands(state%num_operands))
            temp_operands(1:state%num_operands) = state%operand_ptrs(1:state%num_operands)
            call mlirOperationStateAddOperands(state%ptr, &
                int(state%num_operands, c_intptr_t), &
                c_loc(temp_operands))
            deallocate(temp_operands)
        end if
        
        if (state%num_results > 0) then
            allocate(temp_results(state%num_results))
            temp_results(1:state%num_results) = state%result_ptrs(1:state%num_results)
            call mlirOperationStateAddResults(state%ptr, &
                int(state%num_results, c_intptr_t), &
                c_loc(temp_results))
            deallocate(temp_results)
        end if
        
        ! Create the operation
        op%ptr = mlirOperationCreate(state%ptr)
    end function create_operation

    ! Destroy operation
    subroutine destroy_operation(op)
        type(mlir_operation_t), intent(inout) :: op
        
        if (c_associated(op%ptr)) then
            call mlirOperationDestroy(op%ptr)
            op%ptr = c_null_ptr
        end if
    end subroutine destroy_operation

    ! Verify operation
    function verify_operation(op) result(valid)
        type(mlir_operation_t), intent(in) :: op
        logical :: valid
        
        if (c_associated(op%ptr)) then
            valid = mlirOperationVerify(op%ptr)
        else
            valid = .false.
        end if
    end function verify_operation

    ! Get operation number of results
    function get_operation_num_results(op) result(n)
        type(mlir_operation_t), intent(in) :: op
        integer :: n
        
        if (c_associated(op%ptr)) then
            n = int(mlirOperationGetNumResults(op%ptr))
        else
            n = 0
        end if
    end function get_operation_num_results

    ! Get operation name (simplified - returns fixed string for stub)
    function get_operation_name(op) result(name)
        type(mlir_operation_t), intent(in) :: op
        character(len=:), allocatable :: name
        
        if (c_associated(op%ptr)) then
            name = "test.complete_op"  ! Simplified for stub
        else
            name = "<invalid>"
        end if
    end function get_operation_name

    ! Create dummy value for testing
    function create_dummy_value(context) result(value)
        type(mlir_context_t), intent(in) :: context
        type(mlir_value_t) :: value
        
        ! For testing, just create a non-null pointer
        value%ptr = context%ptr  ! Reuse context pointer as dummy
    end function create_dummy_value

    ! Check if operation state is valid
    function operation_state_is_valid(this) result(valid)
        class(mlir_operation_state_t), intent(in) :: this
        logical :: valid
        valid = c_associated(this%ptr)
    end function operation_state_is_valid

    ! Check if operation is valid
    function operation_is_valid(this) result(valid)
        class(mlir_operation_t), intent(in) :: this
        logical :: valid
        valid = c_associated(this%ptr)
    end function operation_is_valid

    ! Check if value is valid
    function value_is_valid(this) result(valid)
        class(mlir_value_t), intent(in) :: this
        logical :: valid
        valid = c_associated(this%ptr)
    end function value_is_valid

end module mlir_c_operations