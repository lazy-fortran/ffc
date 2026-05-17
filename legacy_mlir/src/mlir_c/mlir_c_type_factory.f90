module mlir_c_type_factory
    use iso_c_binding
    use mlir_c_core
    use mlir_c_types
    implicit none
    private

    ! Public types
    public :: type_factory_t
    
    ! Type cache entry
    type :: type_cache_entry_t
        character(len=:), allocatable :: key
        type(mlir_type_t) :: cached_type
    end type type_cache_entry_t

    ! Type factory with caching
    type :: type_factory_t
        private
        type(mlir_context_t) :: context
        type(type_cache_entry_t), allocatable :: cache(:)
        integer :: cache_size = 0
        integer :: cache_capacity = 0
    contains
        procedure :: init => factory_init
        procedure :: get_integer_type => factory_get_integer_type
        procedure :: get_float_type => factory_get_float_type
        procedure :: get_array_type => factory_get_array_type
        procedure :: get_reference_type => factory_get_reference_type
        procedure :: finalize => factory_finalize
        procedure, private :: lookup_cache => factory_lookup_cache
        procedure, private :: add_to_cache => factory_add_to_cache
    end type type_factory_t

    ! Initial cache capacity
    integer, parameter :: INITIAL_CACHE_SIZE = 16

contains

    ! Initialize factory
    subroutine factory_init(this, context)
        class(type_factory_t), intent(out) :: this
        type(mlir_context_t), intent(in) :: context
        
        this%context = context
        this%cache_capacity = INITIAL_CACHE_SIZE
        allocate(this%cache(this%cache_capacity))
        this%cache_size = 0
    end subroutine factory_init

    ! Get integer type with caching
    function factory_get_integer_type(this, width, signed) result(type)
        class(type_factory_t), intent(inout) :: this
        integer, intent(in) :: width
        logical, intent(in), optional :: signed
        type(mlir_type_t) :: type
        character(len=32) :: key
        logical :: is_signed
        
        is_signed = .true.
        if (present(signed)) is_signed = signed
        
        ! Create cache key
        if (is_signed) then
            write(key, '("i", I0, "s")') width
        else
            write(key, '("i", I0, "u")') width
        end if
        
        ! Check cache
        type = this%lookup_cache(trim(key))
        if (type%is_valid()) return
        
        ! Create new type
        type = create_integer_type(this%context, width, is_signed)
        
        ! Add to cache
        if (type%is_valid()) then
            call this%add_to_cache(trim(key), type)
        end if
    end function factory_get_integer_type

    ! Get float type with caching
    function factory_get_float_type(this, width) result(type)
        class(type_factory_t), intent(inout) :: this
        integer, intent(in) :: width
        type(mlir_type_t) :: type
        character(len=32) :: key
        
        ! Validate width
        if (width /= 32 .and. width /= 64) then
            type%ptr = c_null_ptr
            return
        end if
        
        ! Create cache key
        write(key, '("f", I0)') width
        
        ! Check cache
        type = this%lookup_cache(trim(key))
        if (type%is_valid()) return
        
        ! Create new type
        type = create_float_type(this%context, width)
        
        ! Add to cache
        if (type%is_valid()) then
            call this%add_to_cache(trim(key), type)
        end if
    end function factory_get_float_type

    ! Get array type with caching
    function factory_get_array_type(this, element_type, shape) result(type)
        class(type_factory_t), intent(inout) :: this
        type(mlir_type_t), intent(in) :: element_type
        integer(c_int64_t), dimension(:), intent(in) :: shape
        type(mlir_type_t) :: type
        character(len=256) :: key
        character(len=32) :: shape_str
        integer :: i
        
        ! Create cache key - simplified for now
        key = "array_"
        do i = 1, size(shape)
            write(shape_str, '(I0)') shape(i)
            key = trim(key) // "x" // trim(shape_str)
        end do
        
        ! Check cache
        type = this%lookup_cache(trim(key))
        if (type%is_valid()) return
        
        ! Create new type
        type = create_array_type(this%context, element_type, shape)
        
        ! Add to cache
        if (type%is_valid()) then
            call this%add_to_cache(trim(key), type)
        end if
    end function factory_get_array_type

    ! Get reference type with caching
    function factory_get_reference_type(this, element_type) result(type)
        class(type_factory_t), intent(inout) :: this
        type(mlir_type_t), intent(in) :: element_type
        type(mlir_type_t) :: type
        character(len=64) :: key
        
        ! Create cache key - simplified
        key = "ref_type"
        
        ! For now, don't cache reference types as they depend on element type
        type = create_reference_type(this%context, element_type)
    end function factory_get_reference_type

    ! Lookup in cache
    function factory_lookup_cache(this, key) result(type)
        class(type_factory_t), intent(in) :: this
        character(len=*), intent(in) :: key
        type(mlir_type_t) :: type
        integer :: i
        
        type%ptr = c_null_ptr
        
        do i = 1, this%cache_size
            if (allocated(this%cache(i)%key)) then
                if (this%cache(i)%key == key) then
                    type = this%cache(i)%cached_type
                    return
                end if
            end if
        end do
    end function factory_lookup_cache

    ! Add to cache
    subroutine factory_add_to_cache(this, key, type)
        class(type_factory_t), intent(inout) :: this
        character(len=*), intent(in) :: key
        type(mlir_type_t), intent(in) :: type
        type(type_cache_entry_t), allocatable :: new_cache(:)
        
        ! Grow cache if needed
        if (this%cache_size >= this%cache_capacity) then
            allocate(new_cache(this%cache_capacity * 2))
            new_cache(1:this%cache_size) = this%cache(1:this%cache_size)
            call move_alloc(new_cache, this%cache)
            this%cache_capacity = this%cache_capacity * 2
        end if
        
        ! Add entry
        this%cache_size = this%cache_size + 1
        this%cache(this%cache_size)%key = key
        this%cache(this%cache_size)%cached_type = type
    end subroutine factory_add_to_cache

    ! Finalize factory
    subroutine factory_finalize(this)
        class(type_factory_t), intent(inout) :: this
        
        if (allocated(this%cache)) then
            deallocate(this%cache)
        end if
        this%cache_size = 0
        this%cache_capacity = 0
    end subroutine factory_finalize

end module mlir_c_type_factory