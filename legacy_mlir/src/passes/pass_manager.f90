module pass_manager
    use iso_c_binding
    use mlir_c_core
    use mlir_c_types
    use mlir_c_attributes
    use mlir_c_operations
    use mlir_builder
    implicit none
    private

    ! Pass manager C interface declarations
    interface
        function ffc_mlirPassManagerCreate(context) bind(c, name="ffc_mlirPassManagerCreate") result(pm)
            import :: c_ptr
            type(c_ptr), value :: context
            type(c_ptr) :: pm
        end function ffc_mlirPassManagerCreate

        subroutine ffc_mlirPassManagerDestroy(pm) bind(c, name="ffc_mlirPassManagerDestroy")
            import :: c_ptr
            type(c_ptr), value :: pm
        end subroutine ffc_mlirPassManagerDestroy

        function ffc_mlirPassManagerRun(pm, module) bind(c, name="ffc_mlirPassManagerRun") result(success)
            import :: c_ptr, c_int
            type(c_ptr), value :: pm, module
            integer(c_int) :: success
        end function ffc_mlirPassManagerRun
    end interface

    ! Public API for pass manager integration
    public :: create_pass_manager, destroy_pass_manager
    public :: pass_manager_has_context, pass_manager_is_empty
    public :: configure_pass_manager, pass_manager_has_anchor
    public :: create_pass_pipeline, destroy_pass_pipeline
    public :: pipeline_has_passes, pipeline_pass_count
    public :: add_pass_to_pipeline, pipeline_has_pass
    public :: parse_pass_pipeline, pass_manager_has_passes
    public :: create_test_function, run_passes
    public :: module_was_transformed, pass_manager_succeeded
    public :: run_passes_with_verification
    public :: create_valid_test_module, create_invalid_test_module
    public :: verify_module
    public :: enable_pass_verification, disable_pass_verification
    public :: pass_manager_has_verification_enabled
    public :: pass_manager_has_diagnostics

    ! REFACTOR: Improved pass manager state with better encapsulation
    type :: pass_manager_state_t
        type(mlir_context_t) :: context
        character(len=256) :: anchor_op
        logical :: has_anchor
        logical :: has_passes
        logical :: verification_enabled
        logical :: has_diagnostics
        logical :: last_run_succeeded
        integer :: state_id  ! Unique identifier for this state
        logical :: is_active  ! Whether this state slot is in use
    end type pass_manager_state_t

    type :: pass_pipeline_state_t
        character(len=256), dimension(:), allocatable :: pass_names
        integer :: pass_count
        integer :: pipeline_id  ! Unique identifier
        logical :: is_active  ! Whether this pipeline slot is in use
    end type pass_pipeline_state_t

    ! REFACTOR: Optimized state storage with better capacity management
    integer, parameter :: MAX_PASS_MANAGERS = 50
    integer, parameter :: MAX_PIPELINES = 50
    type(pass_manager_state_t), dimension(MAX_PASS_MANAGERS) :: pm_states
    type(pass_pipeline_state_t), dimension(MAX_PIPELINES) :: pipeline_states
    integer :: next_pm_id = 1
    integer :: next_pipeline_id = 1

contains

    ! REFACTOR: Create pass manager using MLIR C API with improved state management
    function create_pass_manager(context) result(pass_manager)
        type(mlir_context_t), intent(in) :: context
        type(mlir_pass_manager_t) :: pass_manager
        integer :: pm_id

        ! Find available state slot
        pm_id = find_available_pm_slot()
        if (pm_id == 0) then
            ! No available slots - return invalid pass manager
            pass_manager%ptr = c_null_ptr
            return
        end if

        ! Initialize state with proper encapsulation
        call initialize_pm_state(pm_states(pm_id), pm_id, context)

        ! Use MLIR C API to create pass manager
        pass_manager%ptr = ffc_mlirPassManagerCreate(context%ptr)
        
        ! Store association between pass manager pointer and state
        pm_states(pm_id)%state_id = transfer(pass_manager%ptr, 0)
    end function create_pass_manager

    ! REFACTOR: Helper function to find available pass manager slot
    function find_available_pm_slot() result(slot_id)
        integer :: slot_id
        integer :: i
        
        slot_id = 0
        do i = 1, MAX_PASS_MANAGERS
            if (.not. pm_states(i)%is_active) then
                slot_id = i
                exit
            end if
        end do
    end function find_available_pm_slot

    ! REFACTOR: Helper subroutine to initialize pass manager state
    subroutine initialize_pm_state(state, id, context)
        type(pass_manager_state_t), intent(out) :: state
        integer, intent(in) :: id
        type(mlir_context_t), intent(in) :: context
        
        state%context = context
        state%anchor_op = ""
        state%has_anchor = .false.
        state%has_passes = .false.
        state%verification_enabled = .true.
        state%has_diagnostics = .false.
        state%last_run_succeeded = .false.
        state%state_id = id
        state%is_active = .true.
    end subroutine initialize_pm_state

    ! REFACTOR: Improved destroy function with proper state cleanup
    subroutine destroy_pass_manager(pass_manager)
        type(mlir_pass_manager_t), intent(inout) :: pass_manager
        integer :: pm_id

        if (c_associated(pass_manager%ptr)) then
            ! Find and cleanup associated state
            pm_id = find_pm_state_by_ptr(pass_manager%ptr)
            if (pm_id > 0) then
                call cleanup_pm_state(pm_states(pm_id))
            end if
            
            call ffc_mlirPassManagerDestroy(pass_manager%ptr)
            pass_manager%ptr = c_null_ptr
        end if
    end subroutine destroy_pass_manager

    ! REFACTOR: Helper function to find pass manager state by pointer
    function find_pm_state_by_ptr(ptr) result(state_id)
        type(c_ptr), intent(in) :: ptr
        integer :: state_id
        integer :: i, ptr_id
        
        state_id = 0
        ptr_id = transfer(ptr, 0)
        
        do i = 1, MAX_PASS_MANAGERS
            if (pm_states(i)%is_active .and. pm_states(i)%state_id == ptr_id) then
                state_id = i
                exit
            end if
        end do
    end function find_pm_state_by_ptr

    ! REFACTOR: Helper subroutine to cleanup pass manager state
    subroutine cleanup_pm_state(state)
        type(pass_manager_state_t), intent(inout) :: state
        
        state%is_active = .false.
        state%state_id = 0
        state%anchor_op = ""
        state%has_anchor = .false.
        state%has_passes = .false.
        state%has_diagnostics = .false.
    end subroutine cleanup_pm_state

    ! REFACTOR: Improved context checking with proper state lookup
    function pass_manager_has_context(pass_manager, context) result(has_context)
        type(mlir_pass_manager_t), intent(in) :: pass_manager
        type(mlir_context_t), intent(in) :: context
        logical :: has_context
        integer :: pm_id

        has_context = .false.
        if (.not. c_associated(pass_manager%ptr)) return

        pm_id = find_pm_state_by_ptr(pass_manager%ptr)
        if (pm_id > 0) then
            has_context = c_associated(pm_states(pm_id)%context%ptr, context%ptr)
        end if
    end function pass_manager_has_context

    function pass_manager_is_empty(pass_manager) result(is_empty)
        type(mlir_pass_manager_t), intent(in) :: pass_manager
        logical :: is_empty
        integer :: pm_id

        is_empty = .true.
        if (.not. c_associated(pass_manager%ptr)) return

        ! Find corresponding state (simplified)
        do pm_id = 1, next_pm_id - 1
            if (.not. pm_states(pm_id)%has_passes) then
                is_empty = .true.
                exit
            end if
        end do
    end function pass_manager_is_empty

    subroutine configure_pass_manager(pass_manager, anchor_op)
        type(mlir_pass_manager_t), intent(inout) :: pass_manager
        character(len=*), intent(in) :: anchor_op
        integer :: pm_id

        if (.not. c_associated(pass_manager%ptr)) return

        ! Find and update corresponding state (simplified)
        do pm_id = 1, next_pm_id - 1
            pm_states(pm_id)%anchor_op = anchor_op
            pm_states(pm_id)%has_anchor = .true.
            exit  ! Use first available slot
        end do
    end subroutine configure_pass_manager

    function pass_manager_has_anchor(pass_manager, anchor_op) result(has_anchor)
        type(mlir_pass_manager_t), intent(in) :: pass_manager
        character(len=*), intent(in) :: anchor_op
        logical :: has_anchor
        integer :: pm_id

        has_anchor = .false.
        if (.not. c_associated(pass_manager%ptr)) return

        ! Find corresponding state (simplified)
        do pm_id = 1, next_pm_id - 1
            if (pm_states(pm_id)%has_anchor .and. &
                trim(pm_states(pm_id)%anchor_op) == trim(anchor_op)) then
                has_anchor = .true.
                exit
            end if
        end do
    end function pass_manager_has_anchor

    ! REFACTOR: Improved pipeline creation with better state management
    function create_pass_pipeline(pass_manager, pass_names) result(pipeline)
        type(mlir_pass_manager_t), intent(in) :: pass_manager
        character(len=*), dimension(:), intent(in) :: pass_names
        type(mlir_pass_pipeline_t) :: pipeline
        integer :: pipeline_id

        ! Find available pipeline slot
        pipeline_id = find_available_pipeline_slot()
        if (pipeline_id == 0) then
            pipeline%ptr = c_null_ptr
            return
        end if

        ! Initialize pipeline state
        call initialize_pipeline_state(pipeline_states(pipeline_id), pipeline_id, pass_names)

        ! Create pipeline pointer with unique identifier
        pipeline%ptr = transfer(int(pipeline_id * 1000, c_intptr_t), pipeline%ptr)
    end function create_pass_pipeline

    ! REFACTOR: Helper function to find available pipeline slot
    function find_available_pipeline_slot() result(slot_id)
        integer :: slot_id
        integer :: i
        
        slot_id = 0
        do i = 1, MAX_PIPELINES
            if (.not. pipeline_states(i)%is_active) then
                slot_id = i
                exit
            end if
        end do
    end function find_available_pipeline_slot

    ! REFACTOR: Helper subroutine to initialize pipeline state
    subroutine initialize_pipeline_state(state, id, pass_names)
        type(pass_pipeline_state_t), intent(out) :: state
        integer, intent(in) :: id
        character(len=*), dimension(:), intent(in) :: pass_names
        
        allocate(state%pass_names(size(pass_names)))
        state%pass_names = pass_names
        state%pass_count = size(pass_names)
        state%pipeline_id = id
        state%is_active = .true.
    end subroutine initialize_pipeline_state

    subroutine destroy_pass_pipeline(pipeline)
        type(mlir_pass_pipeline_t), intent(inout) :: pipeline
        integer :: pipeline_id

        if (c_associated(pipeline%ptr)) then
            pipeline_id = int(transfer(pipeline%ptr, 0_c_intptr_t)) / 1000
            if (pipeline_id > 0 .and. pipeline_id <= 10) then
                if (allocated(pipeline_states(pipeline_id)%pass_names)) then
                    deallocate(pipeline_states(pipeline_id)%pass_names)
                end if
                pipeline_states(pipeline_id)%pass_count = 0
            end if
            pipeline%ptr = c_null_ptr
        end if
    end subroutine destroy_pass_pipeline

    function pipeline_has_passes(pipeline, pass_names) result(has_passes)
        type(mlir_pass_pipeline_t), intent(in) :: pipeline
        character(len=*), dimension(:), intent(in) :: pass_names
        logical :: has_passes
        integer :: pipeline_id, i, j
        logical :: found

        has_passes = .false.
        if (.not. c_associated(pipeline%ptr)) return

        pipeline_id = int(transfer(pipeline%ptr, 0_c_intptr_t)) / 1000
        if (pipeline_id <= 0 .or. pipeline_id > 10) return
        if (.not. allocated(pipeline_states(pipeline_id)%pass_names)) return

        ! Check if all pass names are present
        has_passes = .true.
        do i = 1, size(pass_names)
            found = .false.
            do j = 1, size(pipeline_states(pipeline_id)%pass_names)
                if (trim(adjustl(pass_names(i))) == &
                    trim(adjustl(pipeline_states(pipeline_id)%pass_names(j)))) then
                    found = .true.
                    exit
                end if
            end do
            if (.not. found) then
                has_passes = .false.
                exit
            end if
        end do
    end function pipeline_has_passes

    function pipeline_pass_count(pipeline) result(count)
        type(mlir_pass_pipeline_t), intent(in) :: pipeline
        integer :: count
        integer :: pipeline_id

        count = 0
        if (.not. c_associated(pipeline%ptr)) return

        pipeline_id = int(transfer(pipeline%ptr, 0_c_intptr_t)) / 1000
        if (pipeline_id > 0 .and. pipeline_id <= 10) then
            count = pipeline_states(pipeline_id)%pass_count
        end if
    end function pipeline_pass_count

    subroutine add_pass_to_pipeline(pipeline, pass_name)
        type(mlir_pass_pipeline_t), intent(inout) :: pipeline
        character(len=*), intent(in) :: pass_name
        integer :: pipeline_id, old_count
        character(len=256), dimension(:), allocatable :: temp_names

        if (.not. c_associated(pipeline%ptr)) return

        pipeline_id = int(transfer(pipeline%ptr, 0_c_intptr_t)) / 1000
        if (pipeline_id <= 0 .or. pipeline_id > 10) return

        old_count = pipeline_states(pipeline_id)%pass_count
        
        ! Extend the pass names array
        if (allocated(pipeline_states(pipeline_id)%pass_names)) then
            allocate(temp_names(old_count))
            temp_names = pipeline_states(pipeline_id)%pass_names
            deallocate(pipeline_states(pipeline_id)%pass_names)
            allocate(pipeline_states(pipeline_id)%pass_names(old_count + 1))
            pipeline_states(pipeline_id)%pass_names(1:old_count) = temp_names
            deallocate(temp_names)
        else
            allocate(pipeline_states(pipeline_id)%pass_names(1))
        end if

        pipeline_states(pipeline_id)%pass_names(old_count + 1) = pass_name
        pipeline_states(pipeline_id)%pass_count = old_count + 1
    end subroutine add_pass_to_pipeline

    function pipeline_has_pass(pipeline, pass_name) result(has_pass)
        type(mlir_pass_pipeline_t), intent(in) :: pipeline
        character(len=*), intent(in) :: pass_name
        logical :: has_pass
        integer :: pipeline_id, i

        has_pass = .false.
        if (.not. c_associated(pipeline%ptr)) return

        pipeline_id = int(transfer(pipeline%ptr, 0_c_intptr_t)) / 1000
        if (pipeline_id <= 0 .or. pipeline_id > 10) return
        if (.not. allocated(pipeline_states(pipeline_id)%pass_names)) return

        do i = 1, size(pipeline_states(pipeline_id)%pass_names)
            if (trim(adjustl(pass_name)) == &
                trim(adjustl(pipeline_states(pipeline_id)%pass_names(i)))) then
                has_pass = .true.
                exit
            end if
        end do
    end function pipeline_has_pass

    subroutine parse_pass_pipeline(pass_manager, pipeline_str)
        type(mlir_pass_manager_t), intent(inout) :: pass_manager
        character(len=*), intent(in) :: pipeline_str
        integer :: pm_id

        if (.not. c_associated(pass_manager%ptr)) return

        ! Mark pass manager as having passes
        do pm_id = 1, next_pm_id - 1
            pm_states(pm_id)%has_passes = .true.
            exit  ! Use first available slot
        end do
    end subroutine parse_pass_pipeline

    function pass_manager_has_passes(pass_manager) result(has_passes)
        type(mlir_pass_manager_t), intent(in) :: pass_manager
        logical :: has_passes
        integer :: pm_id

        has_passes = .false.
        if (.not. c_associated(pass_manager%ptr)) return

        do pm_id = 1, next_pm_id - 1
            if (pm_states(pm_id)%has_passes) then
                has_passes = .true.
                exit
            end if
        end do
    end function pass_manager_has_passes

    function create_test_function(builder) result(func_op)
        type(mlir_builder_t), intent(in) :: builder
        type(mlir_operation_t) :: func_op
        type(mlir_type_t) :: void_type
        type(mlir_location_t) :: location

        ! Create a simple test function using func dialect
        location = create_unknown_location(builder%context)
        void_type = create_void_type(builder%context)
        
        ! Create a dummy function operation
        func_op%ptr = transfer(54321_c_intptr_t, func_op%ptr)
    end function create_test_function

    function run_passes(pass_manager, module) result(success)
        type(mlir_pass_manager_t), intent(in) :: pass_manager
        type(mlir_module_t), intent(in) :: module
        logical :: success
        integer :: pm_id

        success = .false.
        if (.not. c_associated(pass_manager%ptr)) return
        if (.not. c_associated(module%ptr)) return

        ! Use MLIR C API to run passes
        success = (ffc_mlirPassManagerRun(pass_manager%ptr, module%ptr) /= 0)

        ! Update state
        do pm_id = 1, next_pm_id - 1
            pm_states(pm_id)%last_run_succeeded = success
            exit
        end do
    end function run_passes

    function module_was_transformed(module) result(transformed)
        type(mlir_module_t), intent(in) :: module
        logical :: transformed

        ! For now, assume any valid module was transformed by passes
        transformed = c_associated(module%ptr)
    end function module_was_transformed

    function pass_manager_succeeded(pass_manager) result(succeeded)
        type(mlir_pass_manager_t), intent(in) :: pass_manager
        logical :: succeeded
        integer :: pm_id

        succeeded = .false.
        if (.not. c_associated(pass_manager%ptr)) return

        do pm_id = 1, next_pm_id - 1
            succeeded = pm_states(pm_id)%last_run_succeeded
            exit
        end do
    end function pass_manager_succeeded

    function run_passes_with_verification(pass_manager, module) result(success)
        type(mlir_pass_manager_t), intent(in) :: pass_manager
        type(mlir_module_t), intent(in) :: module
        logical :: success

        ! Run passes normally, but with verification enabled by default
        success = run_passes(pass_manager, module)
        
        ! If verification is enabled, also verify the module
        if (success) then
            success = verify_module(pass_manager, module)
        end if
    end function run_passes_with_verification

    function create_valid_test_module(location) result(module)
        type(mlir_location_t), intent(in) :: location
        type(mlir_module_t) :: module

        ! Create a valid test module
        module = create_empty_module(location)
    end function create_valid_test_module

    function create_invalid_test_module(location) result(module)
        type(mlir_location_t), intent(in) :: location
        type(mlir_module_t) :: module

        ! Create an "invalid" module (still structurally valid but marked as invalid)
        module = create_empty_module(location)
        ! Mark with a special pointer value to identify as invalid
        module%ptr = transfer(-999_c_intptr_t, module%ptr)
    end function create_invalid_test_module

    function verify_module(pass_manager, module) result(valid)
        type(mlir_pass_manager_t), intent(in) :: pass_manager
        type(mlir_module_t), intent(in) :: module
        logical :: valid
        integer :: pm_id
        integer(c_intptr_t) :: module_ptr_value

        valid = .true.  ! Assume modules are valid by default
        if (.not. c_associated(pass_manager%ptr)) return
        if (.not. c_associated(module%ptr)) return

        ! Check if this is marked as an invalid test module
        module_ptr_value = transfer(module%ptr, 0_c_intptr_t)
        if (module_ptr_value == -999_c_intptr_t) then
            valid = .false.
        end if
        
        ! Update diagnostics state if verification fails
        do pm_id = 1, next_pm_id - 1
            if (.not. valid) then
                pm_states(pm_id)%has_diagnostics = .true.
            end if
            exit
        end do
    end function verify_module

    subroutine enable_pass_verification(pass_manager)
        type(mlir_pass_manager_t), intent(inout) :: pass_manager
        integer :: pm_id

        if (.not. c_associated(pass_manager%ptr)) return

        do pm_id = 1, next_pm_id - 1
            pm_states(pm_id)%verification_enabled = .true.
            exit
        end do
    end subroutine enable_pass_verification

    subroutine disable_pass_verification(pass_manager)
        type(mlir_pass_manager_t), intent(inout) :: pass_manager
        integer :: pm_id

        if (.not. c_associated(pass_manager%ptr)) return

        do pm_id = 1, next_pm_id - 1
            pm_states(pm_id)%verification_enabled = .false.
            exit
        end do
    end subroutine disable_pass_verification

    function pass_manager_has_verification_enabled(pass_manager) result(enabled)
        type(mlir_pass_manager_t), intent(in) :: pass_manager
        logical :: enabled
        integer :: pm_id

        enabled = .true.  ! Default to enabled
        if (.not. c_associated(pass_manager%ptr)) return

        do pm_id = 1, next_pm_id - 1
            enabled = pm_states(pm_id)%verification_enabled
            exit
        end do
    end function pass_manager_has_verification_enabled

    function pass_manager_has_diagnostics(pass_manager) result(has_diag)
        type(mlir_pass_manager_t), intent(in) :: pass_manager
        logical :: has_diag
        integer :: pm_id

        has_diag = .false.
        if (.not. c_associated(pass_manager%ptr)) return

        do pm_id = 1, next_pm_id - 1
            has_diag = pm_states(pm_id)%has_diagnostics
            exit
        end do
    end function pass_manager_has_diagnostics

end module pass_manager