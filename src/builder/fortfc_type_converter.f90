module fortfc_type_converter
    use iso_c_binding
    use mlir_c_core
    use mlir_c_types
    implicit none
    private

    ! Public types
    public :: mlir_type_converter_t

    ! Public functions
    public :: create_type_converter, destroy_type_converter

    ! Type cache entry
    type :: type_cache_entry_t
        character(len=:), allocatable :: key
        type(mlir_type_t) :: cached_type
        type(type_cache_entry_t), pointer :: next => null()
    end type type_cache_entry_t

    ! Type converter
    type :: mlir_type_converter_t
        type(mlir_context_t), pointer :: context => null()
        ! Type cache for performance (hash table)
        type(type_cache_entry_t), pointer :: cache(:) => null()
        integer :: cache_size = 31  ! Prime number for good distribution
        integer :: cache_hits = 0
        integer :: cache_misses = 0
    contains
        procedure :: is_valid => converter_is_valid
        procedure :: create_integer_type => converter_create_integer_type
        procedure :: create_float_type => converter_create_float_type
        procedure :: create_character_type => converter_create_character_type
        procedure :: create_complex_type => converter_create_complex_type
        procedure :: create_array_type => converter_create_array_type
        procedure :: get_integer_type_string => converter_get_integer_type_string
        procedure :: get_float_type_string => converter_get_float_type_string
        procedure :: get_logical_type_string => converter_get_logical_type_string
        procedure :: get_character_type_string => converter_get_character_type_string
        procedure :: get_complex_type_string => converter_get_complex_type_string
        procedure :: get_array_type_string => converter_get_array_type_string
        ! Cache management
        procedure :: get_cached_type => converter_get_cached_type
        procedure :: cache_type => converter_cache_type
        procedure :: get_cache_stats => converter_get_cache_stats
        procedure :: clear_cache => converter_clear_cache
    end type mlir_type_converter_t

contains

    ! Create type converter
    function create_type_converter(context) result(converter)
        type(mlir_context_t), intent(in), target :: context
        type(mlir_type_converter_t) :: converter
        integer :: i
        
        converter%context => context
        
        ! Initialize cache
        allocate(converter%cache(converter%cache_size))
        do i = 1, converter%cache_size
            converter%cache(i)%next => null()
        end do
        converter%cache_hits = 0
        converter%cache_misses = 0
    end function create_type_converter

    ! Destroy type converter
    subroutine destroy_type_converter(converter)
        type(mlir_type_converter_t), intent(inout) :: converter
        
        call converter%clear_cache()
        if (associated(converter%cache)) deallocate(converter%cache)
        converter%context => null()
    end subroutine destroy_type_converter

    ! Check if converter is valid
    function converter_is_valid(this) result(valid)
        class(mlir_type_converter_t), intent(in) :: this
        logical :: valid
        valid = associated(this%context)
    end function converter_is_valid

    ! Create integer type (with caching)
    function converter_create_integer_type(this, bit_width) result(mlir_type)
        class(mlir_type_converter_t), intent(inout) :: this
        integer, intent(in) :: bit_width
        type(mlir_type_t) :: mlir_type
        character(len=:), allocatable :: cache_key
        
        if (.not. this%is_valid()) then
            mlir_type%ptr = c_null_ptr
            return
        end if
        
        ! Create cache key
        cache_key = "int_" // trim(adjustl(int_to_string(bit_width)))
        
        ! Try to get from cache first
        mlir_type = this%get_cached_type(cache_key)
        if (mlir_type%is_valid()) return
        
        ! Create new type and cache it
        mlir_type = create_integer_type(this%context, bit_width)
        call this%cache_type(cache_key, mlir_type)
    end function converter_create_integer_type

    ! Create float type
    function converter_create_float_type(this, bit_width) result(mlir_type)
        class(mlir_type_converter_t), intent(in) :: this
        integer, intent(in) :: bit_width
        type(mlir_type_t) :: mlir_type
        
        if (.not. this%is_valid()) then
            mlir_type%ptr = c_null_ptr
            return
        end if
        
        mlir_type = create_float_type(this%context, bit_width)
    end function converter_create_float_type

    ! Get integer type string
    function converter_get_integer_type_string(this, bit_width) result(type_str)
        class(mlir_type_converter_t), intent(in) :: this
        integer, intent(in) :: bit_width
        character(len=:), allocatable :: type_str
        character(len=10) :: width_str
        
        write(width_str, '(I0)') bit_width
        type_str = "i" // trim(width_str)
    end function converter_get_integer_type_string

    ! Get float type string
    function converter_get_float_type_string(this, bit_width) result(type_str)
        class(mlir_type_converter_t), intent(in) :: this
        integer, intent(in) :: bit_width
        character(len=:), allocatable :: type_str
        character(len=10) :: width_str
        
        write(width_str, '(I0)') bit_width
        type_str = "f" // trim(width_str)
    end function converter_get_float_type_string

    ! Get logical type string
    function converter_get_logical_type_string(this) result(type_str)
        class(mlir_type_converter_t), intent(in) :: this
        character(len=:), allocatable :: type_str
        
        type_str = "i1"
    end function converter_get_logical_type_string

    ! Create character type (FIR character type)
    function converter_create_character_type(this, length) result(mlir_type)
        class(mlir_type_converter_t), intent(in) :: this
        integer, intent(in) :: length
        type(mlir_type_t) :: mlir_type
        
        if (.not. this%is_valid()) then
            mlir_type%ptr = c_null_ptr
            return
        end if
        
        ! For now, create as opaque type - in real implementation would use FIR char type
        mlir_type = create_integer_type(this%context, 8)  ! Stub: use i8 for characters
    end function converter_create_character_type

    ! Create complex type (FIR complex type)
    function converter_create_complex_type(this, element_width) result(mlir_type)
        class(mlir_type_converter_t), intent(in) :: this
        integer, intent(in) :: element_width  ! Width of each component (32 or 64)
        type(mlir_type_t) :: mlir_type
        type(mlir_type_t) :: element_type

        interface
            function ffc_mlirComplexTypeGet(elem_type) bind(c, name="ffc_mlirComplexTypeGet") result(ctype)
                import :: c_ptr
                type(c_ptr), value :: elem_type
                type(c_ptr) :: ctype
            end function ffc_mlirComplexTypeGet
        end interface

        if (.not. this%is_valid()) then
            mlir_type%ptr = c_null_ptr
            return
        end if

        ! Create element type (float32 or float64)
        element_type = create_float_type(this%context, element_width)
        if (.not. element_type%is_valid()) then
            mlir_type%ptr = c_null_ptr
            return
        end if

        ! Create complex type from element type
        mlir_type%ptr = ffc_mlirComplexTypeGet(element_type%ptr)
    end function converter_create_complex_type

    ! Create array type (FIR array type)
    function converter_create_array_type(this, element_type, dimensions) result(mlir_type)
        class(mlir_type_converter_t), intent(in) :: this
        type(mlir_type_t), intent(in) :: element_type
        integer, dimension(:), intent(in) :: dimensions
        type(mlir_type_t) :: mlir_type
        
        if (.not. this%is_valid()) then
            mlir_type%ptr = c_null_ptr
            return
        end if
        
        ! For now, just return element type - in real implementation would create array type
        mlir_type = element_type
    end function converter_create_array_type

    ! Get character type string
    function converter_get_character_type_string(this, length) result(type_str)
        class(mlir_type_converter_t), intent(in) :: this
        integer, intent(in) :: length
        character(len=:), allocatable :: type_str
        character(len=10) :: length_str
        
        write(length_str, '(I0)') length
        type_str = "!fir.char<1," // trim(length_str) // ">"
    end function converter_get_character_type_string

    ! Get complex type string
    function converter_get_complex_type_string(this, element_width) result(type_str)
        class(mlir_type_converter_t), intent(in) :: this
        integer, intent(in) :: element_width  ! Width of each component
        character(len=:), allocatable :: type_str
        character(len=10) :: width_str
        
        ! For FIR complex types, the parameter is the size of each component in bytes
        write(width_str, '(I0)') element_width / 8  ! Convert bits to bytes
        type_str = "!fir.complex<" // trim(width_str) // ">"
    end function converter_get_complex_type_string

    ! Get array type string
    function converter_get_array_type_string(this, element_type_str, dimensions) result(type_str)
        class(mlir_type_converter_t), intent(in) :: this
        character(len=*), intent(in) :: element_type_str
        integer, dimension(:), intent(in) :: dimensions
        character(len=:), allocatable :: type_str
        character(len=200) :: dim_str
        integer :: i
        
        ! Build dimension string
        dim_str = ""
        do i = 1, size(dimensions)
            if (dimensions(i) > 0) then
                write(dim_str, '(A, I0)') trim(dim_str), dimensions(i)
            else
                dim_str = trim(dim_str) // "?"
            end if
            if (i < size(dimensions)) then
                dim_str = trim(dim_str) // "x"
            end if
        end do
        
        type_str = "!fir.array<" // trim(dim_str) // "x" // element_type_str // ">"
    end function converter_get_array_type_string

    ! Get cached type
    function converter_get_cached_type(this, key) result(mlir_type)
        class(mlir_type_converter_t), intent(inout) :: this
        character(len=*), intent(in) :: key
        type(mlir_type_t) :: mlir_type
        integer :: hash_index
        type(type_cache_entry_t), pointer :: current
        
        mlir_type%ptr = c_null_ptr  ! Default to invalid
        
        if (.not. associated(this%cache)) return
        
        hash_index = mod(hash_string_simple(key), this%cache_size) + 1
        current => this%cache(hash_index)%next
        
        do while (associated(current))
            if (allocated(current%key) .and. current%key == key) then
                mlir_type = current%cached_type
                this%cache_hits = this%cache_hits + 1
                return
            end if
            current => current%next
        end do
        
        this%cache_misses = this%cache_misses + 1
    end function converter_get_cached_type

    ! Cache type
    subroutine converter_cache_type(this, key, mlir_type)
        class(mlir_type_converter_t), intent(inout) :: this
        character(len=*), intent(in) :: key
        type(mlir_type_t), intent(in) :: mlir_type
        integer :: hash_index
        type(type_cache_entry_t), pointer :: new_entry
        
        if (.not. associated(this%cache)) return
        
        hash_index = mod(hash_string_simple(key), this%cache_size) + 1
        
        ! Create new cache entry
        allocate(new_entry)
        new_entry%key = key
        new_entry%cached_type = mlir_type
        new_entry%next => this%cache(hash_index)%next
        this%cache(hash_index)%next => new_entry
    end subroutine converter_cache_type

    ! Get cache statistics
    subroutine converter_get_cache_stats(this, hits, misses, hit_rate)
        class(mlir_type_converter_t), intent(in) :: this
        integer, intent(out) :: hits, misses
        real, intent(out) :: hit_rate
        
        hits = this%cache_hits
        misses = this%cache_misses
        if (hits + misses > 0) then
            hit_rate = real(hits) / real(hits + misses) * 100.0
        else
            hit_rate = 0.0
        end if
    end subroutine converter_get_cache_stats

    ! Clear cache
    subroutine converter_clear_cache(this)
        class(mlir_type_converter_t), intent(inout) :: this
        integer :: i
        type(type_cache_entry_t), pointer :: current, next_entry
        
        if (.not. associated(this%cache)) return
        
        do i = 1, this%cache_size
            current => this%cache(i)%next
            do while (associated(current))
                next_entry => current%next
                if (allocated(current%key)) deallocate(current%key)
                deallocate(current)
                current => next_entry
            end do
            this%cache(i)%next => null()
        end do
        
        this%cache_hits = 0
        this%cache_misses = 0
    end subroutine converter_clear_cache

    ! Simple hash function for strings
    function hash_string_simple(str) result(hash)
        character(len=*), intent(in) :: str
        integer :: hash
        integer :: i
        
        hash = 0
        do i = 1, len(str)
            hash = mod(hash * 31 + ichar(str(i:i)), huge(hash) / 2)
        end do
        hash = abs(hash)
    end function hash_string_simple

    ! Convert integer to string
    function int_to_string(value) result(str)
        integer, intent(in) :: value
        character(len=:), allocatable :: str
        character(len=20) :: temp
        
        write(temp, '(I0)') value
        str = trim(temp)
    end function int_to_string

end module fortfc_type_converter