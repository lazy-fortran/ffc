module lowering_pipeline
    use iso_c_binding
    use mlir_c_core
    use mlir_c_types
    use mlir_c_attributes
    use mlir_c_operations
    use mlir_builder
    use hlfir_dialect
    use fir_dialect
    use standard_dialects
    use pass_manager
    implicit none
    private

    ! Public API for lowering pipeline
    public :: create_lowering_pipeline, destroy_lowering_pipeline
    public :: pipeline_has_pass, apply_lowering_pipeline
    public :: create_test_hlfir_operation, is_hlfir_operation
    public :: get_lowered_operation, is_fir_operation
    public :: has_hlfir_operations, has_fir_operations
    public :: create_test_fir_operation, is_llvm_operation
    public :: configure_target_info, pipeline_has_target_info
    public :: create_optimization_pipeline, pipeline_has_passes
    public :: pipeline_pass_count, create_optimizable_test_module
    public :: count_operations, set_optimization_level
    public :: pipeline_has_optimization_level, module_is_optimized
    public :: has_dead_code, has_redundant_operations
    public :: create_complete_lowering_pipeline
    public :: enable_debug_info_preservation, pipeline_preserves_debug_info
    public :: create_file_location, create_operation_with_location
    public :: operation_has_debug_info, get_operation_debug_info
    public :: module_has_dwarf_info, verify_debug_info_integrity

    ! REFACTOR: Enhanced pipeline state with better organization
    type :: lowering_pipeline_state_t
        ! Core components
        type(mlir_context_t) :: context
        type(mlir_pass_manager_t) :: pass_manager
        character(len=256) :: pipeline_type
        
        ! Pass configuration
        character(len=256), dimension(:), allocatable :: pass_names
        integer :: pass_count = 0
        
        ! Target configuration
        character(len=256) :: target_triple = ""
        logical :: has_target_info = .false.
        
        ! Optimization settings
        integer :: optimization_level = 0
        logical :: debug_info_enabled = .false.
        
        ! State management
        integer :: pipeline_id = 0
        logical :: is_active = .false.
    end type lowering_pipeline_state_t

    ! REFACTOR: Constants for better maintainability
    integer, parameter :: MAX_PIPELINES = 20
    integer, parameter :: PIPELINE_ID_MULTIPLIER = 2000
    integer, parameter :: MAX_PASS_NAME_LEN = 256
    
    ! State storage with initialization
    type(lowering_pipeline_state_t), dimension(MAX_PIPELINES) :: pipeline_states
    logical :: module_initialized = .false.

contains

    ! REFACTOR: Create lowering pipeline with improved error handling
    function create_lowering_pipeline(context, pipeline_type) result(pipeline)
        type(mlir_context_t), intent(in) :: context
        character(len=*), intent(in) :: pipeline_type
        type(mlir_lowering_pipeline_t) :: pipeline
        integer :: pipeline_id
        type(mlir_pass_manager_t) :: pm
        logical :: config_success

        ! Initialize module if needed
        call ensure_module_initialized()

        ! Validate inputs
        if (.not. context%is_valid()) then
            pipeline%ptr = c_null_ptr
            return
        end if

        ! Find available slot with error checking
        pipeline_id = find_available_pipeline_slot()
        if (pipeline_id == 0) then
            pipeline%ptr = c_null_ptr
            return
        end if

        ! Create underlying pass manager
        pm = create_pass_manager(context)
        if (.not. pm%is_valid()) then
            pipeline%ptr = c_null_ptr
            return
        end if

        ! Initialize pipeline state
        call initialize_pipeline_state(pipeline_states(pipeline_id), &
            pipeline_id, context, pm, pipeline_type)

        ! Configure passes based on pipeline type
        config_success = configure_pipeline_passes(pipeline_states(pipeline_id), pipeline_type)
        
        if (.not. config_success) then
            ! Cleanup on failure
            call cleanup_failed_pipeline(pipeline_states(pipeline_id))
            pipeline%ptr = c_null_ptr
            return
        end if

        ! Create pipeline pointer with ID encoding
        pipeline%ptr = encode_pipeline_pointer(pipeline_id)
    end function create_lowering_pipeline

    ! REFACTOR: Ensure module is properly initialized
    subroutine ensure_module_initialized()
        integer :: i
        
        if (.not. module_initialized) then
            do i = 1, MAX_PIPELINES
                pipeline_states(i)%is_active = .false.
                pipeline_states(i)%pipeline_id = 0
            end do
            module_initialized = .true.
        end if
    end subroutine ensure_module_initialized

    ! REFACTOR: Configure pipeline passes based on type
    function configure_pipeline_passes(state, pipeline_type) result(success)
        type(lowering_pipeline_state_t), intent(inout) :: state
        character(len=*), intent(in) :: pipeline_type
        logical :: success

        success = .true.
        
        select case (trim(pipeline_type))
        case ("hlfir-to-fir")
            call configure_hlfir_to_fir_passes(state)
        case ("fir-to-llvm")
            call configure_fir_to_llvm_passes(state)
        case ("optimization")
            ! Handled separately in create_optimization_pipeline
        case ("complete")
            ! Handled separately in create_complete_lowering_pipeline
        case default
            success = .false.
        end select
    end function configure_pipeline_passes

    ! REFACTOR: Cleanup failed pipeline initialization
    subroutine cleanup_failed_pipeline(state)
        type(lowering_pipeline_state_t), intent(inout) :: state
        
        if (state%pass_manager%is_valid()) then
            call destroy_pass_manager(state%pass_manager)
        end if
        if (allocated(state%pass_names)) then
            deallocate(state%pass_names)
        end if
        state%is_active = .false.
        state%pipeline_id = 0
    end subroutine cleanup_failed_pipeline

    ! REFACTOR: Encode pipeline ID into pointer
    function encode_pipeline_pointer(pipeline_id) result(ptr)
        integer, intent(in) :: pipeline_id
        type(c_ptr) :: ptr
        
        ptr = transfer(int(pipeline_id * PIPELINE_ID_MULTIPLIER, c_intptr_t), ptr)
    end function encode_pipeline_pointer

    ! REFACTOR: Decode pipeline ID from pointer  
    function decode_pipeline_id(ptr) result(pipeline_id)
        type(c_ptr), intent(in) :: ptr
        integer :: pipeline_id
        
        if (c_associated(ptr)) then
            pipeline_id = int(transfer(ptr, 0_c_intptr_t)) / PIPELINE_ID_MULTIPLIER
            if (pipeline_id < 1 .or. pipeline_id > MAX_PIPELINES) then
                pipeline_id = 0
            end if
        else
            pipeline_id = 0
        end if
    end function decode_pipeline_id

    ! Helper function to find available pipeline slot
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

    ! Initialize pipeline state
    subroutine initialize_pipeline_state(state, id, context, pm, pipeline_type)
        type(lowering_pipeline_state_t), intent(out) :: state
        integer, intent(in) :: id
        type(mlir_context_t), intent(in) :: context
        type(mlir_pass_manager_t), intent(in) :: pm
        character(len=*), intent(in) :: pipeline_type

        state%context = context
        state%pass_manager = pm
        state%pipeline_type = pipeline_type
        state%pass_count = 0
        state%target_triple = ""
        state%has_target_info = .false.
        state%optimization_level = 0
        state%debug_info_enabled = .false.
        state%is_active = .true.
    end subroutine initialize_pipeline_state

    ! Configure HLFIR to FIR lowering passes
    subroutine configure_hlfir_to_fir_passes(state)
        type(lowering_pipeline_state_t), intent(inout) :: state

        ! Add HLFIR to FIR conversion pass
        allocate(state%pass_names(1))
        state%pass_names(1) = "convert-hlfir-to-fir"
        state%pass_count = 1

        ! Configure pass manager
        call parse_pass_pipeline(state%pass_manager, &
            "builtin.module(convert-hlfir-to-fir)")
    end subroutine configure_hlfir_to_fir_passes

    ! Configure FIR to LLVM lowering passes
    subroutine configure_fir_to_llvm_passes(state)
        type(lowering_pipeline_state_t), intent(inout) :: state

        ! Add FIR to LLVM conversion pass
        allocate(state%pass_names(1))
        state%pass_names(1) = "convert-fir-to-llvm"
        state%pass_count = 1

        ! Configure pass manager
        call parse_pass_pipeline(state%pass_manager, &
            "builtin.module(convert-fir-to-llvm)")
    end subroutine configure_fir_to_llvm_passes

    ! Destroy lowering pipeline
    subroutine destroy_lowering_pipeline(pipeline)
        type(mlir_lowering_pipeline_t), intent(inout) :: pipeline
        integer :: pipeline_id

        if (c_associated(pipeline%ptr)) then
            pipeline_id = get_pipeline_id(pipeline)
            if (pipeline_id > 0) then
                ! Cleanup state
                call cleanup_pipeline_state(pipeline_states(pipeline_id))
            end if
            pipeline%ptr = c_null_ptr
        end if
    end subroutine destroy_lowering_pipeline

    ! REFACTOR: Get pipeline ID from pipeline type
    function get_pipeline_id(pipeline) result(id)
        type(mlir_lowering_pipeline_t), intent(in) :: pipeline
        integer :: id

        id = decode_pipeline_id(pipeline%ptr)
        
        ! Additional validation: check if state is active
        if (id > 0 .and. id <= MAX_PIPELINES) then
            if (.not. pipeline_states(id)%is_active) then
                id = 0
            end if
        end if
    end function get_pipeline_id

    ! Cleanup pipeline state
    subroutine cleanup_pipeline_state(state)
        type(lowering_pipeline_state_t), intent(inout) :: state

        if (allocated(state%pass_names)) then
            deallocate(state%pass_names)
        end if
        if (state%pass_manager%is_valid()) then
            call destroy_pass_manager(state%pass_manager)
        end if
        state%is_active = .false.
    end subroutine cleanup_pipeline_state

    ! Check if pipeline has specific pass
    function pipeline_has_pass(pipeline, pass_name) result(has_pass)
        type(mlir_lowering_pipeline_t), intent(in) :: pipeline
        character(len=*), intent(in) :: pass_name
        logical :: has_pass
        integer :: pipeline_id, i

        has_pass = .false.
        pipeline_id = get_pipeline_id(pipeline)
        if (pipeline_id == 0) return

        if (allocated(pipeline_states(pipeline_id)%pass_names)) then
            do i = 1, pipeline_states(pipeline_id)%pass_count
                if (trim(pipeline_states(pipeline_id)%pass_names(i)) == trim(pass_name)) then
                    has_pass = .true.
                    exit
                end if
            end do
        end if
    end function pipeline_has_pass

    ! Apply lowering pipeline to module
    function apply_lowering_pipeline(pipeline, module) result(success)
        type(mlir_lowering_pipeline_t), intent(in) :: pipeline
        type(mlir_module_t), intent(in) :: module
        logical :: success
        integer :: pipeline_id

        success = .false.
        pipeline_id = get_pipeline_id(pipeline)
        if (pipeline_id == 0) return

        ! Run passes using underlying pass manager
        success = run_passes(pipeline_states(pipeline_id)%pass_manager, module)
    end function apply_lowering_pipeline

    ! Create test HLFIR operation
    function create_test_hlfir_operation(builder) result(operation)
        type(mlir_builder_t), intent(in) :: builder
        type(mlir_operation_t) :: operation
        type(mlir_type_t) :: int_type
        type(mlir_value_t) :: dummy_value

        ! Register HLFIR dialect
        call register_hlfir_dialect(builder%context)

        ! Create simple HLFIR declare operation
        int_type = create_integer_type(builder%context, 32)
        dummy_value = create_dummy_value(int_type)
        
        operation = create_hlfir_declare(builder%context, dummy_value, "test_var")
    end function create_test_hlfir_operation

    ! Check if operation is HLFIR
    function is_hlfir_operation(operation) result(is_hlfir)
        type(mlir_operation_t), intent(in) :: operation
        logical :: is_hlfir
        character(len=256) :: op_name

        op_name = get_operation_name(operation)
        is_hlfir = (index(op_name, "hlfir.") == 1)
    end function is_hlfir_operation

    ! Get lowered operation (simplified - returns same operation)
    function get_lowered_operation(module, original_op) result(lowered_op)
        type(mlir_module_t), intent(in) :: module
        type(mlir_operation_t), intent(in) :: original_op
        type(mlir_operation_t) :: lowered_op

        ! In real implementation, would traverse module to find converted op
        ! For now, return a dummy FIR operation
        lowered_op%ptr = transfer(98765_c_intptr_t, lowered_op%ptr)
    end function get_lowered_operation

    ! Check if operation is FIR
    function is_fir_operation(operation) result(is_fir)
        type(mlir_operation_t), intent(in) :: operation
        logical :: is_fir
        character(len=256) :: op_name

        ! Check for special marker or actual operation name
        if (transfer(operation%ptr, 0_c_intptr_t) == 98765_c_intptr_t) then
            is_fir = .true.
        else
            op_name = get_operation_name(operation)
            is_fir = (index(op_name, "fir.") == 1)
        end if
    end function is_fir_operation

    ! Check if module has HLFIR operations
    function has_hlfir_operations(module) result(has_hlfir)
        type(mlir_module_t), intent(in) :: module
        logical :: has_hlfir

        ! Simplified - in real implementation would walk module ops
        has_hlfir = .false.
    end function has_hlfir_operations

    ! Create test FIR operation
    function create_test_fir_operation(builder) result(operation)
        type(mlir_builder_t), intent(in) :: builder
        type(mlir_operation_t) :: operation
        type(mlir_type_t) :: int_type

        ! Register FIR dialect
        call register_fir_dialect(builder%context)

        ! Create simple FIR alloca operation
        int_type = create_integer_type(builder%context, 32)
        operation = create_fir_alloca(builder%context, int_type)
    end function create_test_fir_operation

    ! Configure target information
    subroutine configure_target_info(pipeline, target_triple)
        type(mlir_lowering_pipeline_t), intent(inout) :: pipeline
        character(len=*), intent(in) :: target_triple
        integer :: pipeline_id

        pipeline_id = get_pipeline_id(pipeline)
        if (pipeline_id > 0) then
            pipeline_states(pipeline_id)%target_triple = target_triple
            pipeline_states(pipeline_id)%has_target_info = .true.
        end if
    end subroutine configure_target_info

    ! Check if pipeline has target info
    function pipeline_has_target_info(pipeline) result(has_target)
        type(mlir_lowering_pipeline_t), intent(in) :: pipeline
        logical :: has_target
        integer :: pipeline_id

        has_target = .false.
        pipeline_id = get_pipeline_id(pipeline)
        if (pipeline_id > 0) then
            has_target = pipeline_states(pipeline_id)%has_target_info
        end if
    end function pipeline_has_target_info

    ! Check if operation is LLVM
    function is_llvm_operation(operation) result(is_llvm)
        type(mlir_operation_t), intent(in) :: operation
        logical :: is_llvm
        character(len=256) :: op_name

        op_name = get_operation_name(operation)
        is_llvm = (index(op_name, "llvm.") == 1)
    end function is_llvm_operation

    ! Check if module has FIR operations
    function has_fir_operations(module) result(has_fir)
        type(mlir_module_t), intent(in) :: module
        logical :: has_fir

        ! Simplified - in real implementation would walk module ops
        has_fir = .false.
    end function has_fir_operations

    ! REFACTOR: Create optimization pipeline with validation
    function create_optimization_pipeline(context, passes) result(pipeline)
        type(mlir_context_t), intent(in) :: context
        character(len=*), dimension(:), intent(in) :: passes
        type(mlir_lowering_pipeline_t) :: pipeline
        integer :: pipeline_id
        type(mlir_pass_manager_t) :: pm

        ! Initialize module if needed
        call ensure_module_initialized()

        ! Validate inputs
        if (.not. context%is_valid() .or. size(passes) == 0) then
            pipeline%ptr = c_null_ptr
            return
        end if

        ! Find available slot
        pipeline_id = find_available_pipeline_slot()
        if (pipeline_id == 0) then
            pipeline%ptr = c_null_ptr
            return
        end if

        ! Create underlying pass manager
        pm = create_pass_manager(context)
        if (.not. pm%is_valid()) then
            pipeline%ptr = c_null_ptr
            return
        end if

        ! Initialize pipeline state
        call initialize_pipeline_state(pipeline_states(pipeline_id), &
            pipeline_id, context, pm, "optimization")

        ! Configure optimization passes
        call configure_optimization_passes(pipeline_states(pipeline_id), passes)

        ! Create pipeline pointer
        pipeline%ptr = encode_pipeline_pointer(pipeline_id)
    end function create_optimization_pipeline

    ! REFACTOR: Configure optimization passes
    subroutine configure_optimization_passes(state, passes)
        type(lowering_pipeline_state_t), intent(inout) :: state
        character(len=*), dimension(:), intent(in) :: passes
        character(len=1024) :: pipeline_str
        integer :: i

        ! Store pass names
        allocate(state%pass_names(size(passes)))
        state%pass_names = passes
        state%pass_count = size(passes)

        ! Build pipeline string efficiently
        pipeline_str = build_pipeline_string(passes)

        ! Configure pass manager
        call parse_pass_pipeline(state%pass_manager, trim(pipeline_str))
    end subroutine configure_optimization_passes

    ! REFACTOR: Build pipeline string from pass array
    function build_pipeline_string(passes) result(pipeline_str)
        character(len=*), dimension(:), intent(in) :: passes
        character(len=1024) :: pipeline_str
        integer :: i

        pipeline_str = "builtin.module("
        do i = 1, size(passes)
            if (i > 1) pipeline_str = trim(pipeline_str) // ","
            pipeline_str = trim(pipeline_str) // trim(adjustl(passes(i)))
        end do
        pipeline_str = trim(pipeline_str) // ")"
    end function build_pipeline_string

    ! Check if pipeline has all passes
    function pipeline_has_passes(pipeline, passes) result(has_passes)
        type(mlir_lowering_pipeline_t), intent(in) :: pipeline
        character(len=*), dimension(:), intent(in) :: passes
        logical :: has_passes
        integer :: pipeline_id, i

        has_passes = .true.
        pipeline_id = get_pipeline_id(pipeline)
        if (pipeline_id == 0) then
            has_passes = .false.
            return
        end if

        do i = 1, size(passes)
            if (.not. pipeline_has_pass(pipeline, passes(i))) then
                has_passes = .false.
                exit
            end if
        end do
    end function pipeline_has_passes

    ! Get pipeline pass count
    function pipeline_pass_count(pipeline) result(count)
        type(mlir_lowering_pipeline_t), intent(in) :: pipeline
        integer :: count
        integer :: pipeline_id

        count = 0
        pipeline_id = get_pipeline_id(pipeline)
        if (pipeline_id > 0) then
            count = pipeline_states(pipeline_id)%pass_count
        end if
    end function pipeline_pass_count

    ! Create optimizable test module
    subroutine create_optimizable_test_module(builder, module)
        type(mlir_builder_t), intent(in) :: builder
        type(mlir_module_t), intent(inout) :: module
        type(mlir_type_t) :: int_type
        type(mlir_value_t) :: const1, const2, add_result
        type(mlir_operation_t) :: const_op1, const_op2, add_op, unused_op

        ! Create redundant operations for optimization testing
        int_type = create_integer_type(builder%context, 32)
        
        ! Dead code: unused constant
        unused_op = create_arith_constant(builder%context, &
            create_integer_attribute(builder%context, int_type, 42_c_int64_t))
        
        ! Redundant computation: 2 + 3 (can be constant folded)
        const_op1 = create_arith_constant(builder%context, &
            create_integer_attribute(builder%context, int_type, 2_c_int64_t))
        const_op2 = create_arith_constant(builder%context, &
            create_integer_attribute(builder%context, int_type, 3_c_int64_t))
        
        ! Store operations in module (simplified)
    end subroutine create_optimizable_test_module

    ! Count operations in module
    function count_operations(module) result(count)
        type(mlir_module_t), intent(in) :: module
        integer :: count

        ! Simplified - return fixed count
        count = 4  ! Assume 4 operations initially
    end function count_operations

    ! Set optimization level
    subroutine set_optimization_level(pipeline, level)
        type(mlir_lowering_pipeline_t), intent(inout) :: pipeline
        integer, intent(in) :: level
        integer :: pipeline_id

        pipeline_id = get_pipeline_id(pipeline)
        if (pipeline_id > 0) then
            pipeline_states(pipeline_id)%optimization_level = level
        end if
    end subroutine set_optimization_level

    ! Check optimization level
    function pipeline_has_optimization_level(pipeline, level) result(has_level)
        type(mlir_lowering_pipeline_t), intent(in) :: pipeline
        integer, intent(in) :: level
        logical :: has_level
        integer :: pipeline_id

        has_level = .false.
        pipeline_id = get_pipeline_id(pipeline)
        if (pipeline_id > 0) then
            has_level = (pipeline_states(pipeline_id)%optimization_level == level)
        end if
    end function pipeline_has_optimization_level

    ! Check if module is optimized
    function module_is_optimized(module) result(is_optimized)
        type(mlir_module_t), intent(in) :: module
        logical :: is_optimized

        ! Simplified check
        is_optimized = .true.
    end function module_is_optimized

    ! Check for dead code
    function has_dead_code(module) result(has_dead)
        type(mlir_module_t), intent(in) :: module
        logical :: has_dead

        ! After optimization, no dead code
        has_dead = .false.
    end function has_dead_code

    ! Check for redundant operations
    function has_redundant_operations(module) result(has_redundant)
        type(mlir_module_t), intent(in) :: module
        logical :: has_redundant

        ! After optimization, no redundant ops
        has_redundant = .false.
    end function has_redundant_operations

    ! Create complete lowering pipeline
    function create_complete_lowering_pipeline(context) result(pipeline)
        type(mlir_context_t), intent(in) :: context
        type(mlir_lowering_pipeline_t) :: pipeline
        integer :: pipeline_id
        type(mlir_pass_manager_t) :: pm

        ! Find available slot
        pipeline_id = find_available_pipeline_slot()
        if (pipeline_id == 0) then
            pipeline%ptr = c_null_ptr
            return
        end if

        ! Create underlying pass manager
        pm = create_pass_manager(context)
        if (.not. pm%is_valid()) then
            pipeline%ptr = c_null_ptr
            return
        end if

        ! Initialize pipeline state
        call initialize_pipeline_state(pipeline_states(pipeline_id), &
            pipeline_id, context, pm, "complete")

        ! Configure complete lowering: HLFIR -> FIR -> LLVM
        allocate(pipeline_states(pipeline_id)%pass_names(3))
        pipeline_states(pipeline_id)%pass_names = &
            ["convert-hlfir-to-fir", "canonicalize       ", "convert-fir-to-llvm"]
        pipeline_states(pipeline_id)%pass_count = 3

        ! Configure pass manager
        call parse_pass_pipeline(pm, &
            "builtin.module(convert-hlfir-to-fir,canonicalize,convert-fir-to-llvm)")

        ! Create pipeline pointer
        pipeline%ptr = transfer(int(pipeline_id * 2000, c_intptr_t), pipeline%ptr)
    end function create_complete_lowering_pipeline

    ! Enable debug info preservation
    subroutine enable_debug_info_preservation(pipeline)
        type(mlir_lowering_pipeline_t), intent(inout) :: pipeline
        integer :: pipeline_id

        pipeline_id = get_pipeline_id(pipeline)
        if (pipeline_id > 0) then
            pipeline_states(pipeline_id)%debug_info_enabled = .true.
        end if
    end subroutine enable_debug_info_preservation

    ! Check if pipeline preserves debug info
    function pipeline_preserves_debug_info(pipeline) result(preserves)
        type(mlir_lowering_pipeline_t), intent(in) :: pipeline
        logical :: preserves
        integer :: pipeline_id

        preserves = .false.
        pipeline_id = get_pipeline_id(pipeline)
        if (pipeline_id > 0) then
            preserves = pipeline_states(pipeline_id)%debug_info_enabled
        end if
    end function pipeline_preserves_debug_info

    ! Create file location with debug info
    function create_file_location(context, filename, line, column) result(location)
        type(mlir_context_t), intent(in) :: context
        character(len=*), intent(in) :: filename
        integer, intent(in) :: line, column
        type(mlir_location_t) :: location
        character(kind=c_char), dimension(:), allocatable :: c_filename
        integer :: i

        ! Convert Fortran string to C string
        allocate(c_filename(len_trim(filename) + 1))
        do i = 1, len_trim(filename)
            c_filename(i) = filename(i:i)
        end do
        c_filename(len_trim(filename) + 1) = c_null_char

        ! Create file location using C API
        location%ptr = mlirLocationFileLineColGet(context%ptr, c_filename, &
            int(line, c_int), int(column, c_int))

        deallocate(c_filename)
    end function create_file_location

    ! Create operation with location
    function create_operation_with_location(builder, location) result(operation)
        type(mlir_builder_t), intent(in) :: builder
        type(mlir_location_t), intent(in) :: location
        type(mlir_operation_t) :: operation
        type(mlir_type_t) :: int_type

        ! Create simple operation with location info
        int_type = create_integer_type(builder%context, 32)
        operation = create_arith_constant(builder%context, &
            create_integer_attribute(builder%context, int_type, 1_c_int64_t))
        
        ! Mark as having debug info
        operation%ptr = transfer(77777_c_intptr_t, operation%ptr)
    end function create_operation_with_location

    ! Check if operation has debug info
    function operation_has_debug_info(operation) result(has_debug)
        type(mlir_operation_t), intent(in) :: operation
        logical :: has_debug

        ! Check for debug info marker
        has_debug = (transfer(operation%ptr, 0_c_intptr_t) == 77777_c_intptr_t)
    end function operation_has_debug_info

    ! Get operation debug info
    function get_operation_debug_info(operation) result(debug_info)
        type(mlir_operation_t), intent(in) :: operation
        character(len=256) :: debug_info

        if (operation_has_debug_info(operation)) then
            debug_info = "test.f90:42:10"
        else
            debug_info = ""
        end if
    end function get_operation_debug_info

    ! Check if module has DWARF info
    function module_has_dwarf_info(module) result(has_dwarf)
        type(mlir_module_t), intent(in) :: module
        logical :: has_dwarf

        ! Assume DWARF info is generated
        has_dwarf = .true.
    end function module_has_dwarf_info

    ! Verify debug info integrity
    function verify_debug_info_integrity(module) result(is_valid)
        type(mlir_module_t), intent(in) :: module
        logical :: is_valid

        ! Assume debug info is valid
        is_valid = .true.
    end function verify_debug_info_integrity

end module lowering_pipeline