module memory_test_stubs
    use iso_c_binding
    use memory_tracker
    use memory_guard
    use resource_manager, only: resource_manager_t, mlir_pass_manager_t, mlir_lowering_pipeline_t
    implicit none

    ! Local type definitions for testing
    type :: mlir_context_t
        type(c_ptr) :: ptr = c_null_ptr
    contains
        procedure :: is_valid => context_is_valid
    end type mlir_context_t

    type :: mlir_builder_t
        type(c_ptr) :: ptr = c_null_ptr
    contains
        procedure :: is_valid => builder_is_valid
    end type mlir_builder_t

    type :: mlir_module_t
        type(c_ptr) :: ptr = c_null_ptr
    contains
        procedure :: is_valid => module_is_valid
    end type mlir_module_t

    type :: mlir_location_t
        type(c_ptr) :: ptr = c_null_ptr
    contains
        procedure :: is_valid => location_is_valid
    end type mlir_location_t


    type :: mlir_c_backend_t
        logical :: initialized = .false.
    contains
        procedure :: init => backend_init
        procedure :: cleanup => backend_cleanup
        procedure :: is_initialized => backend_is_initialized
        procedure :: generate_code => backend_generate_code
    end type mlir_c_backend_t

    type :: backend_options_t
        logical :: compile_mode = .false.
        logical :: optimize = .false.
    end type backend_options_t

    type :: ast_arena_t
        integer :: node_count = 0
    end type ast_arena_t

contains

    ! Stub implementations

    function create_mlir_context() result(context)
        type(mlir_context_t) :: context
        integer, target :: dummy
        context%ptr = c_loc(dummy)  ! Dummy pointer
    end function create_mlir_context

    function create_mlir_builder(context) result(builder)
        type(mlir_context_t), intent(in) :: context
        type(mlir_builder_t) :: builder
        integer, target :: dummy
        builder%ptr = c_loc(dummy)  ! Dummy pointer
    end function create_mlir_builder

    function create_unknown_location(context) result(location)
        type(mlir_context_t), intent(in) :: context
        type(mlir_location_t) :: location
        integer, target :: dummy
        location%ptr = c_loc(dummy)  ! Dummy pointer
    end function create_unknown_location

    function create_empty_module(location) result(module)
        type(mlir_location_t), intent(in) :: location
        type(mlir_module_t) :: module
        integer, target :: dummy
        module%ptr = c_loc(dummy)  ! Dummy pointer
    end function create_empty_module

    subroutine destroy_mlir_context(context)
        type(mlir_context_t), intent(inout) :: context
        context%ptr = c_null_ptr
    end subroutine destroy_mlir_context

    subroutine destroy_mlir_builder(builder)
        type(mlir_builder_t), intent(inout) :: builder
        builder%ptr = c_null_ptr
    end subroutine destroy_mlir_builder

    function create_pass_manager(context) result(pm)
        type(mlir_context_t), intent(in) :: context
        type(mlir_pass_manager_t) :: pm
        integer, target :: dummy
        pm%ptr = c_loc(dummy)  ! Dummy pointer
    end function create_pass_manager

    function create_lowering_pipeline(context, pipeline_type) result(pipeline)
        type(mlir_context_t), intent(in) :: context
        character(len=*), intent(in) :: pipeline_type
        type(mlir_lowering_pipeline_t) :: pipeline
        integer, target :: dummy
        pipeline%ptr = c_loc(dummy)  ! Dummy pointer
    end function create_lowering_pipeline

    function create_complete_lowering_pipeline(context) result(pipeline)
        type(mlir_context_t), intent(in) :: context
        type(mlir_lowering_pipeline_t) :: pipeline
        integer, target :: dummy
        pipeline%ptr = c_loc(dummy)  ! Dummy pointer
    end function create_complete_lowering_pipeline

    ! Validity checking functions

    function context_is_valid(this) result(valid)
        class(mlir_context_t), intent(in) :: this
        logical :: valid
        valid = c_associated(this%ptr)
    end function context_is_valid

    function builder_is_valid(this) result(valid)
        class(mlir_builder_t), intent(in) :: this
        logical :: valid
        valid = c_associated(this%ptr)
    end function builder_is_valid

    function module_is_valid(this) result(valid)
        class(mlir_module_t), intent(in) :: this
        logical :: valid
        valid = c_associated(this%ptr)
    end function module_is_valid

    function location_is_valid(this) result(valid)
        class(mlir_location_t), intent(in) :: this
        logical :: valid
        valid = c_associated(this%ptr)
    end function location_is_valid


    ! Backend methods

    subroutine backend_init(this)
        class(mlir_c_backend_t), intent(inout) :: this
        this%initialized = .true.
    end subroutine backend_init

    subroutine backend_cleanup(this)
        class(mlir_c_backend_t), intent(inout) :: this
        this%initialized = .false.
    end subroutine backend_cleanup

    function backend_is_initialized(this) result(is_init)
        class(mlir_c_backend_t), intent(in) :: this
        logical :: is_init
        is_init = this%initialized
    end function backend_is_initialized

    subroutine backend_generate_code(this, arena, prog_index, options, output, error_msg)
        class(mlir_c_backend_t), intent(inout) :: this
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: prog_index
        type(backend_options_t), intent(in) :: options
        character(len=:), allocatable, intent(out) :: output
        character(len=*), intent(out) :: error_msg
        
        output = "// Generated code"
        error_msg = ""
    end subroutine backend_generate_code

end module memory_test_stubs