module resource_manager
    use iso_c_binding
    implicit none
    private

    public :: resource_manager_t
    public :: mlir_pass_manager_t
    public :: mlir_lowering_pipeline_t

    ! Local type definitions for testing
    type :: mlir_pass_manager_t
        type(c_ptr) :: ptr = c_null_ptr
    contains
        procedure :: is_valid => pm_is_valid
    end type mlir_pass_manager_t

    type :: mlir_lowering_pipeline_t
        type(c_ptr) :: ptr = c_null_ptr
    contains
        procedure :: is_valid => pipeline_is_valid
    end type mlir_lowering_pipeline_t

    ! Resource types
    integer, parameter :: RESOURCE_PASS_MANAGER = 1
    integer, parameter :: RESOURCE_PIPELINE = 2
    integer, parameter :: RESOURCE_MODULE = 3
    integer, parameter :: RESOURCE_CONTEXT = 4

    ! Resource tracking entry
    type :: managed_resource_t
        character(len=64) :: name = ""
        integer :: resource_type = 0
        type(c_ptr) :: ptr = c_null_ptr
        logical :: freed = .false.
    end type managed_resource_t

    ! REFACTOR: Resource manager with enhanced tracking capabilities
    type :: resource_manager_t
        private
        logical :: initialized = .false.
        type(managed_resource_t), allocatable :: resources(:)
        integer :: resource_count = 0
        integer :: max_resources = 1000
        integer :: peak_resource_count = 0
        integer :: total_allocated = 0
        integer :: total_freed = 0
        ! Resource type counters
        integer :: pass_manager_count = 0
        integer :: pipeline_count = 0
        integer :: module_count = 0
        integer :: context_count = 0
    contains
        ! Lifecycle
        procedure :: init => manager_init
        procedure :: cleanup => manager_cleanup
        procedure :: is_initialized => manager_is_initialized
        ! Resource registration
        procedure :: register_pass_manager => manager_register_pass_manager
        procedure :: register_pipeline => manager_register_pipeline
        procedure, private :: register_generic_resource
        ! Resource cleanup
        procedure :: cleanup_resource => manager_cleanup_resource
        procedure :: cleanup_all => manager_cleanup_all
        procedure :: cleanup_by_type => manager_cleanup_by_type
        ! Queries
        procedure :: verify_all_freed => manager_verify_all_freed
        procedure :: get_resource_count => manager_get_resource_count
        procedure :: get_peak_resource_count => manager_get_peak_resource_count
        procedure :: get_type_count => manager_get_type_count
        ! Reporting
        procedure :: print_statistics => manager_print_statistics
        procedure :: print_detailed_report => manager_print_detailed_report
    end type resource_manager_t

contains

    subroutine manager_init(this)
        class(resource_manager_t), intent(inout) :: this
        
        if (this%initialized) return
        
        allocate(this%resources(this%max_resources))
        this%resource_count = 0
        this%peak_resource_count = 0
        this%total_allocated = 0
        this%total_freed = 0
        this%initialized = .true.
    end subroutine manager_init

    subroutine manager_cleanup(this)
        class(resource_manager_t), intent(inout) :: this
        
        if (.not. this%initialized) return
        
        call this%cleanup_all()
        
        if (allocated(this%resources)) then
            deallocate(this%resources)
        end if
        
        this%initialized = .false.
    end subroutine manager_cleanup

    function manager_is_initialized(this) result(is_init)
        class(resource_manager_t), intent(in) :: this
        logical :: is_init
        
        is_init = this%initialized
    end function manager_is_initialized

    subroutine manager_register_pass_manager(this, pm, name)
        class(resource_manager_t), intent(inout) :: this
        type(mlir_pass_manager_t), intent(in) :: pm
        character(len=*), intent(in) :: name
        
        call register_generic_resource(this, pm%ptr, name, RESOURCE_PASS_MANAGER)
    end subroutine manager_register_pass_manager

    subroutine manager_register_pipeline(this, pipeline, name)
        class(resource_manager_t), intent(inout) :: this
        type(mlir_lowering_pipeline_t), intent(in) :: pipeline
        character(len=*), intent(in) :: name
        
        call register_generic_resource(this, pipeline%ptr, name, RESOURCE_PIPELINE)
    end subroutine manager_register_pipeline

    ! REFACTOR: Register generic resource with type tracking
    subroutine register_generic_resource(this, ptr, name, resource_type)
        class(resource_manager_t), intent(inout) :: this
        type(c_ptr), intent(in) :: ptr
        character(len=*), intent(in) :: name
        integer, intent(in) :: resource_type
        
        if (.not. this%initialized) return
        
        if (this%resource_count >= this%max_resources) then
            call expand_resources_array(this)
            if (this%resource_count >= this%max_resources) then
                ! Expansion failed
                return
            end if
        end if
        
        this%resource_count = this%resource_count + 1
        this%total_allocated = this%total_allocated + 1
        
        this%resources(this%resource_count)%name = name
        this%resources(this%resource_count)%resource_type = resource_type
        this%resources(this%resource_count)%ptr = ptr
        this%resources(this%resource_count)%freed = .false.
        
        ! Update type counters
        select case (resource_type)
        case (RESOURCE_PASS_MANAGER)
            this%pass_manager_count = this%pass_manager_count + 1
        case (RESOURCE_PIPELINE)
            this%pipeline_count = this%pipeline_count + 1
        case (RESOURCE_MODULE)
            this%module_count = this%module_count + 1
        case (RESOURCE_CONTEXT)
            this%context_count = this%context_count + 1
        end select
        
        if (this%resource_count > this%peak_resource_count) then
            this%peak_resource_count = this%resource_count
        end if
    end subroutine register_generic_resource

    subroutine manager_cleanup_resource(this, name)
        class(resource_manager_t), intent(inout) :: this
        character(len=*), intent(in) :: name
        integer :: i, j
        
        if (.not. this%initialized) return
        
        do i = 1, this%resource_count
            if (trim(this%resources(i)%name) == trim(name) .and. &
                .not. this%resources(i)%freed) then
                
                ! Mark as freed
                this%resources(i)%freed = .true.
                this%total_freed = this%total_freed + 1
                
                ! Compact array by removing freed resource
                do j = i, this%resource_count - 1
                    this%resources(j) = this%resources(j + 1)
                end do
                this%resource_count = this%resource_count - 1
                
                exit
            end if
        end do
    end subroutine manager_cleanup_resource

    subroutine manager_cleanup_all(this)
        class(resource_manager_t), intent(inout) :: this
        integer :: i
        
        if (.not. this%initialized) return
        
        do i = 1, this%resource_count
            if (.not. this%resources(i)%freed) then
                this%resources(i)%freed = .true.
                this%total_freed = this%total_freed + 1
            end if
        end do
        
        this%resource_count = 0
    end subroutine manager_cleanup_all

    function manager_verify_all_freed(this) result(all_freed)
        class(resource_manager_t), intent(in) :: this
        logical :: all_freed
        
        all_freed = (this%resource_count == 0) .and. &
                   (this%total_allocated == this%total_freed)
    end function manager_verify_all_freed

    function manager_get_resource_count(this) result(count)
        class(resource_manager_t), intent(in) :: this
        integer :: count
        
        count = this%resource_count
    end function manager_get_resource_count

    function manager_get_peak_resource_count(this) result(count)
        class(resource_manager_t), intent(in) :: this
        integer :: count
        
        count = this%peak_resource_count
    end function manager_get_peak_resource_count

    subroutine manager_print_statistics(this)
        class(resource_manager_t), intent(in) :: this
        
        print *, "=== Resource Manager Statistics ==="
        print '(A,I0)', "Current resources: ", this%resource_count
        print '(A,I0)', "Peak resources: ", this%peak_resource_count
        print '(A,I0)', "Total allocated: ", this%total_allocated
        print '(A,I0)', "Total freed: ", this%total_freed
        print *, "=================================="
    end subroutine manager_print_statistics

    ! Helper subroutines

    subroutine expand_resources_array(this)
        class(resource_manager_t), intent(inout) :: this
        type(managed_resource_t), allocatable :: new_resources(:)
        integer :: new_size
        
        new_size = this%max_resources * 2
        allocate(new_resources(new_size))
        
        new_resources(1:this%max_resources) = this%resources
        
        deallocate(this%resources)
        this%resources = new_resources
        this%max_resources = new_size
    end subroutine expand_resources_array

    function pm_is_valid(this) result(valid)
        class(mlir_pass_manager_t), intent(in) :: this
        logical :: valid
        valid = c_associated(this%ptr)
    end function pm_is_valid

    function pipeline_is_valid(this) result(valid)
        class(mlir_lowering_pipeline_t), intent(in) :: this
        logical :: valid
        valid = c_associated(this%ptr)
    end function pipeline_is_valid

    ! REFACTOR: Cleanup resources by type
    subroutine manager_cleanup_by_type(this, resource_type)
        class(resource_manager_t), intent(inout) :: this
        integer, intent(in) :: resource_type
        integer :: i, j
        
        if (.not. this%initialized) return
        
        i = 1
        do while (i <= this%resource_count)
            if (this%resources(i)%resource_type == resource_type .and. &
                .not. this%resources(i)%freed) then
                this%resources(i)%freed = .true.
                this%total_freed = this%total_freed + 1
                
                ! Compact array
                do j = i, this%resource_count - 1
                    this%resources(j) = this%resources(j + 1)
                end do
                this%resource_count = this%resource_count - 1
            else
                i = i + 1
            end if
        end do
    end subroutine manager_cleanup_by_type

    ! REFACTOR: Get count of resources by type
    function manager_get_type_count(this, resource_type) result(count)
        class(resource_manager_t), intent(in) :: this
        integer, intent(in) :: resource_type
        integer :: count
        
        select case (resource_type)
        case (RESOURCE_PASS_MANAGER)
            count = this%pass_manager_count
        case (RESOURCE_PIPELINE)
            count = this%pipeline_count
        case (RESOURCE_MODULE)
            count = this%module_count
        case (RESOURCE_CONTEXT)
            count = this%context_count
        case default
            count = 0
        end select
    end function manager_get_type_count

    ! REFACTOR: Print detailed resource report
    subroutine manager_print_detailed_report(this)
        class(resource_manager_t), intent(in) :: this
        integer :: i
        
        print *, "=== Detailed Resource Report ==="
        print '(A,I0)', "Total resources: ", this%resource_count
        
        if (this%resource_count > 0) then
            print *, "Active resources:"
            do i = 1, this%resource_count
                if (.not. this%resources(i)%freed) then
                    print '(A,A,A,I0,A)', "  - ", trim(this%resources(i)%name), &
                           " (type=", this%resources(i)%resource_type, ")"
                end if
            end do
        end if
        
        print *, "Resource types:"
        print '(A,I0)', "  Pass managers: ", this%pass_manager_count
        print '(A,I0)', "  Pipelines: ", this%pipeline_count
        print '(A,I0)', "  Modules: ", this%module_count
        print '(A,I0)', "  Contexts: ", this%context_count
        print *, "==============================="
    end subroutine manager_print_detailed_report

end module resource_manager