module mlir_c_operations
    use, intrinsic :: iso_c_binding, only: c_ptr, c_null_ptr, c_bool, c_int, &
        c_intptr_t, c_size_t, c_associated, c_loc
    use mlir_c_core, only: mlir_context_t, mlir_location_t, mlir_operation_t, &
        mlir_block_t, mlir_region_t, mlir_value_t, mlir_type_t, &
        mlir_attribute_t, mlir_identifier_t, mlir_string_ref_t
    implicit none
    private

    public :: mlir_named_attribute_t
    public :: mlir_operation_state_t

    public :: mlir_operation_state_get
    public :: mlir_operation_state_add_results
    public :: mlir_operation_state_add_operands
    public :: mlir_operation_state_add_owned_regions
    public :: mlir_operation_state_add_attributes
    public :: mlir_operation_state_enable_result_type_inference

    public :: mlir_operation_create
    public :: mlir_operation_destroy
    public :: mlir_operation_get_num_results
    public :: mlir_operation_get_result
    public :: mlir_operation_get_num_operands
    public :: mlir_operation_get_operand
    public :: mlir_operation_get_num_regions
    public :: mlir_operation_get_region

    public :: mlir_named_attribute_get
    public :: mlir_value_is_null
    public :: mlir_value_get_type

    type :: mlir_named_attribute_t
        type(mlir_identifier_t) :: name
        type(mlir_attribute_t) :: attribute
    end type mlir_named_attribute_t

    type :: mlir_operation_state_t
        type(c_ptr) :: name_data = c_null_ptr
        integer(c_size_t) :: name_length = 0
        type(c_ptr) :: location_ptr = c_null_ptr
        integer(c_intptr_t) :: n_results = 0
        type(c_ptr) :: results = c_null_ptr
        integer(c_intptr_t) :: n_operands = 0
        type(c_ptr) :: operands = c_null_ptr
        integer(c_intptr_t) :: n_regions = 0
        type(c_ptr) :: regions = c_null_ptr
        integer(c_intptr_t) :: n_successors = 0
        type(c_ptr) :: successors = c_null_ptr
        integer(c_intptr_t) :: n_attributes = 0
        type(c_ptr) :: attributes = c_null_ptr
        logical(c_bool) :: enable_result_type_inference = .false.
    end type mlir_operation_state_t

    interface
        subroutine mlirOperationStateGet_c(state, name_data, name_len, loc) &
                bind(C, name="mlirOperationStateGet")
            import :: c_ptr, c_size_t
            type(c_ptr) :: state
            type(c_ptr), value :: name_data
            integer(c_size_t), value :: name_len
            type(c_ptr), value :: loc
        end subroutine mlirOperationStateGet_c

        subroutine mlirOperationStateAddResults(state, n, results) &
                bind(C, name="mlirOperationStateAddResults")
            import :: c_ptr, c_intptr_t
            type(c_ptr), value :: state
            integer(c_intptr_t), value :: n
            type(c_ptr), value :: results
        end subroutine mlirOperationStateAddResults

        subroutine mlirOperationStateAddOperands(state, n, operands) &
                bind(C, name="mlirOperationStateAddOperands")
            import :: c_ptr, c_intptr_t
            type(c_ptr), value :: state
            integer(c_intptr_t), value :: n
            type(c_ptr), value :: operands
        end subroutine mlirOperationStateAddOperands

        subroutine mlirOperationStateAddOwnedRegions(state, n, regions) &
                bind(C, name="mlirOperationStateAddOwnedRegions")
            import :: c_ptr, c_intptr_t
            type(c_ptr), value :: state
            integer(c_intptr_t), value :: n
            type(c_ptr), value :: regions
        end subroutine mlirOperationStateAddOwnedRegions

        subroutine mlirOperationStateAddAttributes(state, n, attrs) &
                bind(C, name="mlirOperationStateAddAttributes")
            import :: c_ptr, c_intptr_t
            type(c_ptr), value :: state
            integer(c_intptr_t), value :: n
            type(c_ptr), value :: attrs
        end subroutine mlirOperationStateAddAttributes

        subroutine mlirOperationStateEnableResultTypeInference(state) &
                bind(C, name="mlirOperationStateEnableResultTypeInference")
            import :: c_ptr
            type(c_ptr), value :: state
        end subroutine mlirOperationStateEnableResultTypeInference

        function mlirOperationCreate(state) bind(C, name="mlirOperationCreate")
            import :: c_ptr
            type(c_ptr), value :: state
            type(c_ptr) :: mlirOperationCreate
        end function mlirOperationCreate

        subroutine mlirOperationDestroy(op) bind(C, name="mlirOperationDestroy")
            import :: c_ptr
            type(c_ptr), value :: op
        end subroutine mlirOperationDestroy

        function mlirOperationGetNumResults(op) &
                bind(C, name="mlirOperationGetNumResults")
            import :: c_ptr, c_intptr_t
            type(c_ptr), value :: op
            integer(c_intptr_t) :: mlirOperationGetNumResults
        end function mlirOperationGetNumResults

        function mlirOperationGetResult(op, pos) &
                bind(C, name="mlirOperationGetResult")
            import :: c_ptr, c_intptr_t
            type(c_ptr), value :: op
            integer(c_intptr_t), value :: pos
            type(c_ptr) :: mlirOperationGetResult
        end function mlirOperationGetResult

        function mlirOperationGetNumOperands(op) &
                bind(C, name="mlirOperationGetNumOperands")
            import :: c_ptr, c_intptr_t
            type(c_ptr), value :: op
            integer(c_intptr_t) :: mlirOperationGetNumOperands
        end function mlirOperationGetNumOperands

        function mlirOperationGetOperand(op, pos) &
                bind(C, name="mlirOperationGetOperand")
            import :: c_ptr, c_intptr_t
            type(c_ptr), value :: op
            integer(c_intptr_t), value :: pos
            type(c_ptr) :: mlirOperationGetOperand
        end function mlirOperationGetOperand

        function mlirOperationGetNumRegions(op) &
                bind(C, name="mlirOperationGetNumRegions")
            import :: c_ptr, c_intptr_t
            type(c_ptr), value :: op
            integer(c_intptr_t) :: mlirOperationGetNumRegions
        end function mlirOperationGetNumRegions

        function mlirOperationGetRegion(op, pos) &
                bind(C, name="mlirOperationGetRegion")
            import :: c_ptr, c_intptr_t
            type(c_ptr), value :: op
            integer(c_intptr_t), value :: pos
            type(c_ptr) :: mlirOperationGetRegion
        end function mlirOperationGetRegion

        function mlirNamedAttributeGet_c(name_ptr, attr_ptr) &
                bind(C, name="mlirNamedAttributeGet")
            import :: c_ptr
            type(c_ptr), value :: name_ptr
            type(c_ptr), value :: attr_ptr
            type(c_ptr) :: mlirNamedAttributeGet_c
        end function mlirNamedAttributeGet_c

        function mlirValueGetType(value) bind(C, name="mlirValueGetType")
            import :: c_ptr
            type(c_ptr), value :: value
            type(c_ptr) :: mlirValueGetType
        end function mlirValueGetType
    end interface

contains

    subroutine mlir_operation_state_get(state, name, loc)
        type(mlir_operation_state_t), intent(out), target :: state
        character(len=*), intent(in), target :: name
        type(mlir_location_t), intent(in) :: loc

        state%name_data = c_loc(name)
        state%name_length = int(len(name), c_size_t)
        state%location_ptr = loc%ptr
        state%n_results = 0
        state%results = c_null_ptr
        state%n_operands = 0
        state%operands = c_null_ptr
        state%n_regions = 0
        state%regions = c_null_ptr
        state%n_successors = 0
        state%successors = c_null_ptr
        state%n_attributes = 0
        state%attributes = c_null_ptr
        state%enable_result_type_inference = .false.
    end subroutine mlir_operation_state_get

    subroutine mlir_operation_state_add_results(state, results)
        type(mlir_operation_state_t), intent(inout), target :: state
        type(mlir_type_t), intent(in), target :: results(:)

        state%n_results = int(size(results), c_intptr_t)
        if (state%n_results > 0) then
            state%results = c_loc(results(1))
        end if
    end subroutine mlir_operation_state_add_results

    subroutine mlir_operation_state_add_operands(state, operands)
        type(mlir_operation_state_t), intent(inout), target :: state
        type(mlir_value_t), intent(in), target :: operands(:)

        state%n_operands = int(size(operands), c_intptr_t)
        if (state%n_operands > 0) then
            state%operands = c_loc(operands(1))
        end if
    end subroutine mlir_operation_state_add_operands

    subroutine mlir_operation_state_add_owned_regions(state, regions)
        type(mlir_operation_state_t), intent(inout), target :: state
        type(mlir_region_t), intent(in), target :: regions(:)

        state%n_regions = int(size(regions), c_intptr_t)
        if (state%n_regions > 0) then
            state%regions = c_loc(regions(1))
        end if
    end subroutine mlir_operation_state_add_owned_regions

    subroutine mlir_operation_state_add_attributes(state, attrs)
        type(mlir_operation_state_t), intent(inout), target :: state
        type(mlir_named_attribute_t), intent(in), target :: attrs(:)

        state%n_attributes = int(size(attrs), c_intptr_t)
        if (state%n_attributes > 0) then
            state%attributes = c_loc(attrs(1))
        end if
    end subroutine mlir_operation_state_add_attributes

    subroutine mlir_operation_state_enable_result_type_inference(state)
        type(mlir_operation_state_t), intent(inout) :: state
        state%enable_result_type_inference = .true.
    end subroutine mlir_operation_state_enable_result_type_inference

    function mlir_operation_create(state) result(op)
        type(mlir_operation_state_t), intent(inout), target :: state
        type(mlir_operation_t) :: op
        op%ptr = mlirOperationCreate(c_loc(state))
    end function mlir_operation_create

    subroutine mlir_operation_destroy(op)
        type(mlir_operation_t), intent(inout) :: op
        if (c_associated(op%ptr)) then
            call mlirOperationDestroy(op%ptr)
            op%ptr = c_null_ptr
        end if
    end subroutine mlir_operation_destroy

    function mlir_operation_get_num_results(op) result(num)
        type(mlir_operation_t), intent(in) :: op
        integer :: num
        num = int(mlirOperationGetNumResults(op%ptr))
    end function mlir_operation_get_num_results

    function mlir_operation_get_result(op, pos) result(val)
        type(mlir_operation_t), intent(in) :: op
        integer, intent(in) :: pos
        type(mlir_value_t) :: val
        val%ptr = mlirOperationGetResult(op%ptr, int(pos, c_intptr_t))
    end function mlir_operation_get_result

    function mlir_operation_get_num_operands(op) result(num)
        type(mlir_operation_t), intent(in) :: op
        integer :: num
        num = int(mlirOperationGetNumOperands(op%ptr))
    end function mlir_operation_get_num_operands

    function mlir_operation_get_operand(op, pos) result(val)
        type(mlir_operation_t), intent(in) :: op
        integer, intent(in) :: pos
        type(mlir_value_t) :: val
        val%ptr = mlirOperationGetOperand(op%ptr, int(pos, c_intptr_t))
    end function mlir_operation_get_operand

    function mlir_operation_get_num_regions(op) result(num)
        type(mlir_operation_t), intent(in) :: op
        integer :: num
        num = int(mlirOperationGetNumRegions(op%ptr))
    end function mlir_operation_get_num_regions

    function mlir_operation_get_region(op, pos) result(region)
        type(mlir_operation_t), intent(in) :: op
        integer, intent(in) :: pos
        type(mlir_region_t) :: region
        region%ptr = mlirOperationGetRegion(op%ptr, int(pos, c_intptr_t))
    end function mlir_operation_get_region

    function mlir_named_attribute_get(name, attr) result(named_attr)
        type(mlir_identifier_t), intent(in) :: name
        type(mlir_attribute_t), intent(in) :: attr
        type(mlir_named_attribute_t) :: named_attr

        named_attr%name = name
        named_attr%attribute = attr
    end function mlir_named_attribute_get

    pure function mlir_value_is_null(val) result(is_null)
        type(mlir_value_t), intent(in) :: val
        logical :: is_null
        is_null = .not. c_associated(val%ptr)
    end function mlir_value_is_null

    function mlir_value_get_type(val) result(ty)
        type(mlir_value_t), intent(in) :: val
        type(mlir_type_t) :: ty
        ty%ptr = mlirValueGetType(val%ptr)
    end function mlir_value_get_type

end module mlir_c_operations
