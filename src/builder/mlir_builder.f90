module mlir_builder
    use iso_c_binding
    use mlir_c_core
    use mlir_c_types
    use mlir_c_attributes
    use mlir_c_operations
    implicit none
    private

    ! Public types
    public :: mlir_builder_t, builder_scope_t

    ! Public functions
    public :: create_mlir_builder, destroy_mlir_builder

    ! Insertion point type
    type :: insertion_point_t
        type(mlir_block_t) :: block
        type(mlir_operation_t) :: after_op
        logical :: has_op = .false.
    end type insertion_point_t

    ! Builder scope for region management
    type :: builder_scope_t
        type(insertion_point_t), allocatable :: saved_point
        logical :: valid = .false.
    contains
        procedure :: is_valid => scope_is_valid
    end type builder_scope_t

    ! Main builder type
    type :: mlir_builder_t
        type(mlir_context_t), pointer :: context => null()
        type(mlir_module_t), allocatable :: module
        type(insertion_point_t), allocatable :: insertion_point
        type(insertion_point_t), allocatable :: insertion_stack(:)
        integer :: stack_size = 0
    contains
        procedure :: is_valid => builder_is_valid
        procedure :: has_module => builder_has_module
        procedure :: has_insertion_point => builder_has_insertion_point
        procedure :: set_module => builder_set_module
        procedure :: set_insertion_point_to_start => builder_set_insertion_point_to_start
        procedure :: set_insertion_point_after => builder_set_insertion_point_after
        procedure :: get_insertion_block => builder_get_insertion_block
        procedure :: push_insertion_point => builder_push_insertion_point
        procedure :: pop_insertion_point => builder_pop_insertion_point
        procedure :: create_block => builder_create_block
        procedure :: create_block_with_args => builder_create_block_with_args
        procedure :: enter_region => builder_enter_region
        procedure :: exit_scope => builder_exit_scope
        procedure :: with_region => builder_with_region
        procedure :: get_or_create_entry_block => builder_get_or_create_entry_block
    end type mlir_builder_t

contains

    ! Create a new builder
    function create_mlir_builder(context) result(builder)
        type(mlir_context_t), intent(in), target :: context
        type(mlir_builder_t) :: builder
        
        builder%context => context
        allocate(builder%insertion_stack(10))
        builder%stack_size = 0
    end function create_mlir_builder

    ! Destroy builder
    subroutine destroy_mlir_builder(builder)
        type(mlir_builder_t), intent(inout) :: builder
        
        builder%context => null()
        if (allocated(builder%module)) deallocate(builder%module)
        if (allocated(builder%insertion_point)) deallocate(builder%insertion_point)
        if (allocated(builder%insertion_stack)) deallocate(builder%insertion_stack)
        builder%stack_size = 0
    end subroutine destroy_mlir_builder

    ! Check if builder is valid
    function builder_is_valid(this) result(valid)
        class(mlir_builder_t), intent(in) :: this
        logical :: valid
        valid = associated(this%context)
    end function builder_is_valid

    ! Check if builder has module
    function builder_has_module(this) result(has_mod)
        class(mlir_builder_t), intent(in) :: this
        logical :: has_mod
        has_mod = allocated(this%module)
    end function builder_has_module

    ! Check if builder has insertion point
    function builder_has_insertion_point(this) result(has_point)
        class(mlir_builder_t), intent(in) :: this
        logical :: has_point
        has_point = allocated(this%insertion_point)
        if (has_point) then
            has_point = this%insertion_point%block%is_valid()
        end if
    end function builder_has_insertion_point

    ! Set module
    subroutine builder_set_module(this, module)
        class(mlir_builder_t), intent(inout) :: this
        type(mlir_module_t), intent(in) :: module
        
        if (allocated(this%module)) deallocate(this%module)
        allocate(this%module, source=module)
    end subroutine builder_set_module

    ! Set insertion point to start of block
    subroutine builder_set_insertion_point_to_start(this, block)
        class(mlir_builder_t), intent(inout) :: this
        type(mlir_block_t), intent(in) :: block
        
        ! Validation: ensure builder and block are valid
        if (.not. this%is_valid()) return
        if (.not. block%is_valid()) return
        
        if (.not. allocated(this%insertion_point)) then
            allocate(this%insertion_point)
        end if
        
        this%insertion_point%block = block
        this%insertion_point%has_op = .false.
    end subroutine builder_set_insertion_point_to_start

    ! Set insertion point after operation
    subroutine builder_set_insertion_point_after(this, op)
        class(mlir_builder_t), intent(inout) :: this
        type(mlir_operation_t), intent(in) :: op
        
        if (.not. allocated(this%insertion_point)) then
            allocate(this%insertion_point)
        end if
        
        ! For now, just mark that we have an operation
        this%insertion_point%after_op = op
        this%insertion_point%has_op = .true.
        ! In real implementation, would get block from operation
        this%insertion_point%block = create_mlir_block()
    end subroutine builder_set_insertion_point_after

    ! Get current insertion block
    function builder_get_insertion_block(this) result(block)
        class(mlir_builder_t), intent(in) :: this
        type(mlir_block_t) :: block
        
        if (allocated(this%insertion_point)) then
            block = this%insertion_point%block
        else
            block%ptr = c_null_ptr
        end if
    end function builder_get_insertion_block

    ! Push insertion point onto stack
    subroutine builder_push_insertion_point(this)
        class(mlir_builder_t), intent(inout) :: this
        type(insertion_point_t), allocatable :: new_stack(:)
        
        ! Validation: ensure we have something to push
        if (.not. allocated(this%insertion_point)) return
        if (.not. this%is_valid()) return
        
        ! Grow stack if needed (optimize: use reasonable growth factor)
        if (this%stack_size >= size(this%insertion_stack)) then
            ! Grow by 50% or at least 5 slots, whichever is larger
            allocate(new_stack(max(size(this%insertion_stack) + 5, &
                                  int(size(this%insertion_stack) * 1.5))))
            new_stack(1:this%stack_size) = this%insertion_stack(1:this%stack_size)
            call move_alloc(new_stack, this%insertion_stack)
        end if
        
        this%stack_size = this%stack_size + 1
        this%insertion_stack(this%stack_size) = this%insertion_point
    end subroutine builder_push_insertion_point

    ! Pop insertion point from stack
    subroutine builder_pop_insertion_point(this)
        class(mlir_builder_t), intent(inout) :: this
        
        ! Validation: ensure valid state and non-empty stack
        if (.not. this%is_valid()) return
        if (this%stack_size <= 0) return
        
        if (.not. allocated(this%insertion_point)) then
            allocate(this%insertion_point)
        end if
        this%insertion_point = this%insertion_stack(this%stack_size)
        this%stack_size = this%stack_size - 1
        
        ! Optimization: shrink stack if it's become too large
        ! (Keep reasonable working size but prevent excessive memory use)
        if (size(this%insertion_stack) > 50 .and. &
            this%stack_size < size(this%insertion_stack) / 4) then
            block
                type(insertion_point_t), allocatable :: smaller_stack(:)
                integer :: new_size
                new_size = max(10, this%stack_size * 2)
                allocate(smaller_stack(new_size))
                if (this%stack_size > 0) then
                    smaller_stack(1:this%stack_size) = &
                        this%insertion_stack(1:this%stack_size)
                end if
                call move_alloc(smaller_stack, this%insertion_stack)
            end block
        end if
    end subroutine builder_pop_insertion_point

    ! Create block in region
    function builder_create_block(this, region) result(block)
        class(mlir_builder_t), intent(inout) :: this
        type(mlir_region_t), intent(in) :: region
        type(mlir_block_t) :: block
        
        ! Validation: ensure builder and region are valid
        if (.not. this%is_valid() .or. .not. region%is_valid()) then
            block%ptr = c_null_ptr
            return
        end if
        
        block = create_mlir_block()
        ! In real implementation, would add block to region
    end function builder_create_block

    ! Create block with arguments
    function builder_create_block_with_args(this, region, arg_types) result(block)
        class(mlir_builder_t), intent(inout) :: this
        type(mlir_region_t), intent(in) :: region
        type(mlir_type_t), dimension(:), intent(in) :: arg_types
        type(mlir_block_t) :: block
        
        ! Validation: ensure builder and region are valid
        if (.not. this%is_valid() .or. .not. region%is_valid()) then
            block%ptr = c_null_ptr
            return
        end if
        
        ! Validate argument types
        if (size(arg_types) > 0) then
            ! In real implementation, would validate each type
        end if
        
        ! Create block with different pointer to indicate it has arguments
        block%ptr = transfer(54321_c_intptr_t, block%ptr)
        ! In real implementation, would create block with arguments
    end function builder_create_block_with_args

    ! Enter region scope
    function builder_enter_region(this, region) result(scope)
        class(mlir_builder_t), intent(inout) :: this
        type(mlir_region_t), intent(in) :: region
        type(builder_scope_t) :: scope
        
        scope%valid = .true.
        if (allocated(this%insertion_point)) then
            allocate(scope%saved_point, source=this%insertion_point)
        end if
    end function builder_enter_region

    ! Exit scope
    subroutine builder_exit_scope(this, scope)
        class(mlir_builder_t), intent(inout) :: this
        type(builder_scope_t), intent(inout) :: scope
        
        if (scope%valid .and. allocated(scope%saved_point)) then
            if (.not. allocated(this%insertion_point)) then
                allocate(this%insertion_point)
            end if
            this%insertion_point = scope%saved_point
        else
            if (allocated(this%insertion_point)) deallocate(this%insertion_point)
        end if
        scope%valid = .false.
    end subroutine builder_exit_scope

    ! With region helper
    subroutine builder_with_region(this, region)
        class(mlir_builder_t), intent(inout) :: this
        type(mlir_region_t), intent(in) :: region
        
        ! Simple version - just ensure region exists
    end subroutine builder_with_region

    ! Get or create entry block
    function builder_get_or_create_entry_block(this, region) result(block)
        class(mlir_builder_t), intent(inout) :: this
        type(mlir_region_t), intent(in) :: region
        type(mlir_block_t) :: block
        
        ! For stub, always create new block
        block = this%create_block(region)
    end function builder_get_or_create_entry_block

    ! Scope validity check
    function scope_is_valid(this) result(valid)
        class(builder_scope_t), intent(in) :: this
        logical :: valid
        valid = this%valid
    end function scope_is_valid

end module mlir_builder