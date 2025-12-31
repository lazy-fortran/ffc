module mlir_c_smart_ptr
    use mlir_c_core
    use iso_c_binding
    implicit none
    private

    ! Public types
    public :: mlir_context_ptr, mlir_module_ptr, mlir_location_ptr

    ! Smart pointer for MLIR context with reference counting
    type :: mlir_context_ptr
        private
        type(mlir_context_t) :: context
        integer, pointer :: ref_count => null()
    contains
        procedure :: init => context_ptr_init
        procedure :: release => context_ptr_release
        procedure :: get => context_ptr_get
        procedure :: is_valid => context_ptr_is_valid
        procedure :: share => context_ptr_share
        procedure, private :: context_ptr_assign
        generic :: assignment(=) => context_ptr_assign
        final :: context_ptr_destructor
    end type mlir_context_ptr

    ! Smart pointer for MLIR module
    type :: mlir_module_ptr
        private
        type(mlir_module_t) :: module
        type(mlir_context_ptr) :: context_ref  ! Keep context alive
    contains
        procedure :: init => module_ptr_init
        procedure :: get => module_ptr_get
        procedure :: is_valid => module_ptr_is_valid
        final :: module_ptr_destructor
    end type mlir_module_ptr

    ! Smart pointer for MLIR location
    type :: mlir_location_ptr
        private
        type(mlir_location_t) :: location
        type(mlir_context_ptr) :: context_ref  ! Keep context alive
    contains
        procedure :: init => location_ptr_init
        procedure :: get => location_ptr_get
        procedure :: is_valid => location_ptr_is_valid
        final :: location_ptr_destructor
    end type mlir_location_ptr

contains

    ! Context smart pointer methods
    subroutine context_ptr_init(this)
        class(mlir_context_ptr), intent(out) :: this
        
        ! Create new context
        this%context = create_mlir_context()
        
        ! Initialize reference count
        allocate(this%ref_count)
        this%ref_count = 1
    end subroutine context_ptr_init

    subroutine context_ptr_release(this)
        class(mlir_context_ptr), intent(inout) :: this
        
        if (associated(this%ref_count)) then
            this%ref_count = this%ref_count - 1
            
            if (this%ref_count <= 0) then
                ! Last reference - destroy context
                if (this%context%is_valid()) then
                    call destroy_mlir_context(this%context)
                end if
                deallocate(this%ref_count)
            end if
            
            nullify(this%ref_count)
        end if
    end subroutine context_ptr_release

    function context_ptr_get(this) result(context)
        class(mlir_context_ptr), intent(in) :: this
        type(mlir_context_t) :: context
        context = this%context
    end function context_ptr_get

    function context_ptr_is_valid(this) result(valid)
        class(mlir_context_ptr), intent(in) :: this
        logical :: valid
        valid = associated(this%ref_count) .and. this%context%is_valid()
    end function context_ptr_is_valid

    function context_ptr_share(this) result(shared)
        class(mlir_context_ptr), intent(inout) :: this
        type(mlir_context_ptr) :: shared
        
        if (associated(this%ref_count)) then
            this%ref_count = this%ref_count + 1
            shared%context = this%context
            shared%ref_count => this%ref_count
        end if
    end function context_ptr_share

    subroutine context_ptr_assign(lhs, rhs)
        class(mlir_context_ptr), intent(inout) :: lhs
        type(mlir_context_ptr), intent(in) :: rhs
        
        ! Release existing reference
        call lhs%release()
        
        ! Copy and increment reference count
        if (associated(rhs%ref_count)) then
            lhs%context = rhs%context
            lhs%ref_count => rhs%ref_count
            lhs%ref_count = lhs%ref_count + 1
        end if
    end subroutine context_ptr_assign

    subroutine context_ptr_destructor(this)
        type(mlir_context_ptr), intent(inout) :: this
        call this%release()
    end subroutine context_ptr_destructor

    ! Module smart pointer methods
    subroutine module_ptr_init(this, location, context_ref)
        class(mlir_module_ptr), intent(out) :: this
        type(mlir_location_t), intent(in) :: location
        type(mlir_context_ptr), intent(in) :: context_ref
        
        this%module = create_empty_module(location)
        this%context_ref = context_ref  ! Keep context alive
    end subroutine module_ptr_init

    function module_ptr_get(this) result(module)
        class(mlir_module_ptr), intent(in) :: this
        type(mlir_module_t) :: module
        module = this%module
    end function module_ptr_get

    function module_ptr_is_valid(this) result(valid)
        class(mlir_module_ptr), intent(in) :: this
        logical :: valid
        valid = this%module%is_valid() .and. this%context_ref%is_valid()
    end function module_ptr_is_valid

    subroutine module_ptr_destructor(this)
        type(mlir_module_ptr), intent(inout) :: this
        ! Module will be cleaned up when context is destroyed
        ! Just clear the reference
        this%module%ptr = c_null_ptr
    end subroutine module_ptr_destructor

    ! Location smart pointer methods
    subroutine location_ptr_init(this, context, context_ref)
        class(mlir_location_ptr), intent(out) :: this
        type(mlir_context_t), intent(in) :: context
        type(mlir_context_ptr), intent(in) :: context_ref
        
        this%location = create_unknown_location(context)
        this%context_ref = context_ref  ! Keep context alive
    end subroutine location_ptr_init

    function location_ptr_get(this) result(location)
        class(mlir_location_ptr), intent(in) :: this
        type(mlir_location_t) :: location
        location = this%location
    end function location_ptr_get

    function location_ptr_is_valid(this) result(valid)
        class(mlir_location_ptr), intent(in) :: this
        logical :: valid
        valid = this%location%is_valid() .and. this%context_ref%is_valid()
    end function location_ptr_is_valid

    subroutine location_ptr_destructor(this)
        type(mlir_location_ptr), intent(inout) :: this
        ! Location will be cleaned up when context is destroyed
        ! Just clear the reference
        this%location%ptr = c_null_ptr
    end subroutine location_ptr_destructor

end module mlir_c_smart_ptr