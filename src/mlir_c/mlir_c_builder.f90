module mlir_c_builder
    use mlir_c_core
    use iso_c_binding
    implicit none
    private

    ! Public types
    public :: mlir_builder_t
    
    ! Builder type with RAII-style resource management
    type :: mlir_builder_t
        private
        type(mlir_context_t) :: context
        type(mlir_module_t) :: module
        type(mlir_location_t) :: location
        logical :: owns_context = .false.
    contains
        procedure :: init => builder_init
        procedure :: init_with_context => builder_init_with_context
        procedure :: get_context => builder_get_context
        procedure :: get_module => builder_get_module
        procedure :: get_location => builder_get_location
        procedure :: finalize => builder_finalize
        ! final :: builder_destructor  ! Disabled to avoid double-free issues
    end type mlir_builder_t

    ! Constructor interface
    interface mlir_builder_t
        module procedure create_builder
        module procedure create_builder_with_context
    end interface mlir_builder_t

contains

    ! Constructor: Create builder with new context
    function create_builder() result(builder)
        type(mlir_builder_t) :: builder
        call builder%init()
    end function create_builder

    ! Constructor: Create builder with existing context
    function create_builder_with_context(context) result(builder)
        type(mlir_context_t), intent(in) :: context
        type(mlir_builder_t) :: builder
        call builder%init_with_context(context)
    end function create_builder_with_context

    ! Initialize builder with new context
    subroutine builder_init(this)
        class(mlir_builder_t), intent(out) :: this
        
        ! Create new context
        this%context = create_mlir_context()
        this%owns_context = .true.
        
        ! Create default location
        this%location = create_unknown_location(this%context)
        
        ! Create empty module
        this%module = create_empty_module(this%location)
    end subroutine builder_init

    ! Initialize builder with existing context
    subroutine builder_init_with_context(this, context)
        class(mlir_builder_t), intent(out) :: this
        type(mlir_context_t), intent(in) :: context
        
        ! Use existing context
        this%context = context
        this%owns_context = .false.
        
        ! Create default location
        this%location = create_unknown_location(this%context)
        
        ! Create empty module
        this%module = create_empty_module(this%location)
    end subroutine builder_init_with_context

    ! Get context
    function builder_get_context(this) result(context)
        class(mlir_builder_t), intent(in) :: this
        type(mlir_context_t) :: context
        context = this%context
    end function builder_get_context

    ! Get module
    function builder_get_module(this) result(module)
        class(mlir_builder_t), intent(in) :: this
        type(mlir_module_t) :: module
        module = this%module
    end function builder_get_module

    ! Get location
    function builder_get_location(this) result(location)
        class(mlir_builder_t), intent(in) :: this
        type(mlir_location_t) :: location
        location = this%location
    end function builder_get_location

    ! Explicit finalization
    subroutine builder_finalize(this)
        class(mlir_builder_t), intent(inout) :: this
        
        ! Only destroy context if we own it
        if (this%owns_context .and. this%context%is_valid()) then
            call destroy_mlir_context(this%context)
        end if
        
        ! Clear all references
        this%context%ptr = c_null_ptr
        this%module%ptr = c_null_ptr
        this%location%ptr = c_null_ptr
        this%owns_context = .false.
    end subroutine builder_finalize

    ! ! Automatic destructor for RAII (disabled to avoid double-free issues)
    ! subroutine builder_destructor(this)
    !     type(mlir_builder_t), intent(inout) :: this
    !     ! Only finalize if we own the context and it hasn't been finalized yet
    !     if (this%owns_context .and. c_associated(this%context%ptr)) then
    !         call this%finalize()
    !     end if
    ! end subroutine builder_destructor

end module mlir_c_builder