module mlir_c_core
    use iso_c_binding
    implicit none
    private

    ! Public types
    public :: mlir_context_t, mlir_module_t, mlir_location_t, mlir_string_ref_t
    public :: mlir_region_t, mlir_block_t
    public :: mlir_pass_manager_t, mlir_pass_pipeline_t
    public :: mlir_lowering_pipeline_t
    
    ! Public functions
    public :: create_mlir_context, destroy_mlir_context
    public :: create_empty_module, create_unknown_location
    public :: create_string_ref, get_string_from_ref
    public :: create_empty_region, create_mlir_block

    ! Opaque types for MLIR C API objects
    type :: mlir_context_t
        type(c_ptr) :: ptr = c_null_ptr
    contains
        procedure :: is_valid => context_is_valid
    end type mlir_context_t

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

    type :: mlir_string_ref_t
        type(c_ptr) :: data = c_null_ptr
        integer(c_size_t) :: length = 0
    end type mlir_string_ref_t

    type :: mlir_region_t
        type(c_ptr) :: ptr = c_null_ptr
    contains
        procedure :: is_valid => region_is_valid
    end type mlir_region_t

    type :: mlir_block_t
        type(c_ptr) :: ptr = c_null_ptr
    contains
        procedure :: is_valid => block_is_valid
        procedure :: equals => block_equals
    end type mlir_block_t

    type :: mlir_pass_manager_t
        type(c_ptr) :: ptr = c_null_ptr
    contains
        procedure :: is_valid => pass_manager_is_valid
    end type mlir_pass_manager_t

    type :: mlir_pass_pipeline_t
        type(c_ptr) :: ptr = c_null_ptr
    contains
        procedure :: is_valid => pass_pipeline_is_valid
    end type mlir_pass_pipeline_t

    type :: mlir_lowering_pipeline_t
        type(c_ptr) :: ptr = c_null_ptr
    contains
        procedure :: is_valid => lowering_pipeline_is_valid
    end type mlir_lowering_pipeline_t

    ! C interface declarations
    interface
        ! Context management
        function ffc_mlirContextCreate() bind(c, name="ffc_mlirContextCreate") result(ctx)
            import :: c_ptr
            type(c_ptr) :: ctx
        end function ffc_mlirContextCreate

        subroutine ffc_mlirContextDestroy(context) bind(c, name="ffc_mlirContextDestroy")
            import :: c_ptr
            type(c_ptr), value :: context
        end subroutine ffc_mlirContextDestroy

        ! Module operations
        function ffc_mlirModuleCreateEmpty(location) bind(c, name="ffc_mlirModuleCreateEmpty") result(module)
            import :: c_ptr
            type(c_ptr), value :: location
            type(c_ptr) :: module
        end function ffc_mlirModuleCreateEmpty

        ! Location operations
        function ffc_mlirLocationUnknownGet(context) bind(c, name="ffc_mlirLocationUnknownGet") result(location)
            import :: c_ptr
            type(c_ptr), value :: context
            type(c_ptr) :: location
        end function ffc_mlirLocationUnknownGet

        function mlirLocationFileLineColGet(context, filename, line, col) &
            bind(c, name="mlirLocationFileLineColGet") result(location)
            import :: c_ptr, c_char, c_int
            type(c_ptr), value :: context
            character(kind=c_char), intent(in) :: filename(*)
            integer(c_int), value :: line, col
            type(c_ptr) :: location
        end function mlirLocationFileLineColGet

        ! Pass manager operations
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

contains

    ! Create a new MLIR context
    function create_mlir_context() result(context)
        type(mlir_context_t) :: context
        context%ptr = ffc_mlirContextCreate()
    end function create_mlir_context

    ! Destroy an MLIR context
    subroutine destroy_mlir_context(context)
        type(mlir_context_t), intent(inout) :: context
        if (c_associated(context%ptr)) then
            call ffc_mlirContextDestroy(context%ptr)
            context%ptr = c_null_ptr
        end if
    end subroutine destroy_mlir_context

    ! Create an empty MLIR module
    function create_empty_module(location) result(module)
        type(mlir_location_t), intent(in) :: location
        type(mlir_module_t) :: module
        module%ptr = ffc_mlirModuleCreateEmpty(location%ptr)
    end function create_empty_module

    ! Create an unknown location
    function create_unknown_location(context) result(location)
        type(mlir_context_t), intent(in) :: context
        type(mlir_location_t) :: location
        location%ptr = ffc_mlirLocationUnknownGet(context%ptr)
    end function create_unknown_location

    ! Create a string reference
    function create_string_ref(str) result(ref)
        character(len=*), intent(in), target :: str
        type(mlir_string_ref_t) :: ref
        ref%data = c_loc(str)
        ref%length = len(str)
    end function create_string_ref

    ! Get string from reference
    function get_string_from_ref(ref) result(str)
        type(mlir_string_ref_t), intent(in) :: ref
        character(len=:), allocatable :: str
        character(len=ref%length), pointer :: temp
        
        if (c_associated(ref%data) .and. ref%length > 0) then
            call c_f_pointer(ref%data, temp)
            str = temp
        else
            str = ""
        end if
    end function get_string_from_ref

    ! Type-bound procedures for validity checking
    function context_is_valid(this) result(valid)
        class(mlir_context_t), intent(in) :: this
        logical :: valid
        valid = c_associated(this%ptr)
    end function context_is_valid

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

    ! Create empty region
    function create_empty_region(context) result(region)
        type(mlir_context_t), intent(in) :: context
        type(mlir_region_t) :: region
        
        ! For stub, just create a non-null pointer
        region%ptr = context%ptr
    end function create_empty_region

    function region_is_valid(this) result(valid)
        class(mlir_region_t), intent(in) :: this
        logical :: valid
        valid = c_associated(this%ptr)
    end function region_is_valid

    ! Create block
    function create_mlir_block() result(block)
        type(mlir_block_t) :: block
        ! For stub, create a unique pointer
        block%ptr = transfer(12345_c_intptr_t, block%ptr)
    end function create_mlir_block

    function block_is_valid(this) result(valid)
        class(mlir_block_t), intent(in) :: this
        logical :: valid
        valid = c_associated(this%ptr)
    end function block_is_valid

    function block_equals(this, other) result(equal)
        class(mlir_block_t), intent(in) :: this
        type(mlir_block_t), intent(in) :: other
        logical :: equal
        equal = c_associated(this%ptr, other%ptr)
    end function block_equals

    function pass_manager_is_valid(this) result(valid)
        class(mlir_pass_manager_t), intent(in) :: this
        logical :: valid
        valid = c_associated(this%ptr)
    end function pass_manager_is_valid

    function pass_pipeline_is_valid(this) result(valid)
        class(mlir_pass_pipeline_t), intent(in) :: this
        logical :: valid
        valid = c_associated(this%ptr)
    end function pass_pipeline_is_valid

    function lowering_pipeline_is_valid(this) result(valid)
        class(mlir_lowering_pipeline_t), intent(in) :: this
        logical :: valid
        valid = c_associated(this%ptr)
    end function lowering_pipeline_is_valid

end module mlir_c_core