module memory_guard
    use iso_c_binding
    implicit none
    private

    public :: memory_guard_t

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

    ! Resource entry for tracking
    type :: resource_entry_t
        type(c_ptr) :: ptr = c_null_ptr
        character(len=64) :: name = ""
        character(len=32) :: type = ""
        logical :: freed = .false.
    end type resource_entry_t

    ! REFACTOR: Memory guard type with enhanced RAII-style cleanup
    type :: memory_guard_t
        private
        logical :: active = .false.
        type(resource_entry_t), allocatable :: resources(:)
        integer :: resource_count = 0
        integer :: max_resources = 100
        ! Statistics
        integer :: total_registered = 0
        integer :: total_freed = 0
        logical :: auto_cleanup = .true.
    contains
        ! Lifecycle
        procedure :: init => guard_init
        procedure :: cleanup => guard_cleanup
        procedure :: is_active => guard_is_active
        ! Resource management
        procedure :: register_resource => guard_register_resource
        procedure :: free_resource_by_name => guard_free_resource_by_name
        procedure :: all_resources_freed => guard_all_resources_freed
        ! Configuration
        procedure :: set_auto_cleanup => guard_set_auto_cleanup
        ! Destructor
        final :: guard_destructor
    end type memory_guard_t

contains

    ! REFACTOR: Initialize guard with error handling
    subroutine guard_init(this)
        class(memory_guard_t), intent(inout) :: this
        integer :: stat
        
        if (this%active) return
        
        allocate(this%resources(this%max_resources), stat=stat)
        if (stat /= 0) then
            this%active = .false.
            return
        end if
        
        this%resource_count = 0
        this%total_registered = 0
        this%total_freed = 0
        this%auto_cleanup = .true.
        this%active = .true.
    end subroutine guard_init

    subroutine guard_cleanup(this)
        class(memory_guard_t), intent(inout) :: this
        integer :: i
        
        if (.not. this%active) return
        
        ! Free all unfreed resources
        do i = 1, this%resource_count
            if (.not. this%resources(i)%freed) then
                call free_resource(this%resources(i))
            end if
        end do
        
        if (allocated(this%resources)) then
            deallocate(this%resources)
        end if
        
        this%resource_count = 0
        this%active = .false.
    end subroutine guard_cleanup

    function guard_is_active(this) result(active)
        class(memory_guard_t), intent(in) :: this
        logical :: active
        
        active = this%active
    end function guard_is_active

    ! REFACTOR: Register resource with validation
    subroutine guard_register_resource(this, resource, name)
        class(memory_guard_t), intent(inout) :: this
        class(*), intent(in) :: resource
        character(len=*), intent(in) :: name
        
        if (.not. this%active) return
        
        if (this%resource_count >= this%max_resources) then
            call expand_resources_array(this)
            if (this%resource_count >= this%max_resources) then
                ! Expansion failed
                return
            end if
        end if
        
        this%resource_count = this%resource_count + 1
        this%total_registered = this%total_registered + 1
        this%resources(this%resource_count)%name = name
        this%resources(this%resource_count)%freed = .false.
        
        ! Store type information based on resource type
        select type (resource)
        type is (mlir_context_t)
            this%resources(this%resource_count)%type = "context"
            this%resources(this%resource_count)%ptr = resource%ptr
        type is (mlir_builder_t)
            this%resources(this%resource_count)%type = "builder"
            this%resources(this%resource_count)%ptr = resource%ptr
        type is (mlir_module_t)
            this%resources(this%resource_count)%type = "module"
            this%resources(this%resource_count)%ptr = resource%ptr
        class default
            this%resources(this%resource_count)%type = "unknown"
        end select
    end subroutine guard_register_resource

    function guard_all_resources_freed(this) result(all_freed)
        class(memory_guard_t), intent(in) :: this
        logical :: all_freed
        integer :: i
        
        all_freed = .true.
        
        if (.not. this%active) return
        
        do i = 1, this%resource_count
            if (.not. this%resources(i)%freed) then
                all_freed = .false.
                exit
            end if
        end do
    end function guard_all_resources_freed

    ! Destructor for automatic cleanup
    subroutine guard_destructor(this)
        type(memory_guard_t), intent(inout) :: this
        
        call this%cleanup()
    end subroutine guard_destructor

    ! Helper subroutines

    subroutine free_resource(resource)
        type(resource_entry_t), intent(inout) :: resource
        
        if (resource%freed) return
        
        ! Mark as freed (actual cleanup would depend on resource type)
        resource%freed = .true.
    end subroutine free_resource

    subroutine expand_resources_array(this)
        class(memory_guard_t), intent(inout) :: this
        type(resource_entry_t), allocatable :: new_resources(:)
        integer :: new_size
        
        new_size = this%max_resources * 2
        allocate(new_resources(new_size))
        
        new_resources(1:this%max_resources) = this%resources
        
        deallocate(this%resources)
        this%resources = new_resources
        this%max_resources = new_size
    end subroutine expand_resources_array

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

    ! REFACTOR: Free resource by name
    subroutine guard_free_resource_by_name(this, name)
        class(memory_guard_t), intent(inout) :: this
        character(len=*), intent(in) :: name
        integer :: i
        
        if (.not. this%active) return
        
        do i = 1, this%resource_count
            if (trim(this%resources(i)%name) == trim(name) .and. &
                .not. this%resources(i)%freed) then
                call free_resource(this%resources(i))
                this%total_freed = this%total_freed + 1
                exit
            end if
        end do
    end subroutine guard_free_resource_by_name

    ! REFACTOR: Set auto cleanup behavior
    subroutine guard_set_auto_cleanup(this, auto_cleanup)
        class(memory_guard_t), intent(inout) :: this
        logical, intent(in) :: auto_cleanup
        
        this%auto_cleanup = auto_cleanup
    end subroutine guard_set_auto_cleanup

end module memory_guard