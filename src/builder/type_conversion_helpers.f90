module type_conversion_helpers
    use iso_c_binding
    use mlir_c_core
    use mlir_c_types
    use fortfc_type_converter
    implicit none
    private

    ! Public types
    public :: type_helper_t, array_info_t

    ! Public functions
    public :: create_type_helpers, destroy_type_helpers
    public :: assert_valid_type, assert_integer_type, assert_array_type

    ! Array information type (simplified for testing)
    type :: array_info_t
        integer, allocatable :: dimensions(:)
        logical :: is_allocatable = .false.
        logical :: is_pointer = .false.
        integer :: rank = 0
    end type array_info_t

    ! Type helper utilities
    type :: type_helper_t
        type(mlir_type_converter_t), pointer :: converter => null()
    contains
        procedure :: is_valid => helper_is_valid
        procedure :: get_array_descriptor => helper_get_array_descriptor
        procedure :: mangle_derived_type_name => helper_mangle_derived_type_name
        procedure :: extract_array_shape => helper_extract_array_shape
        procedure :: is_assumed_shape => helper_is_assumed_shape
        procedure :: is_allocatable => helper_is_allocatable
        procedure :: is_pointer => helper_is_pointer
        procedure :: wrap_with_reference_type => helper_wrap_with_reference_type
        procedure :: get_reference_type_string => helper_get_reference_type_string
        procedure :: wrap_with_box_type => helper_wrap_with_box_type
        procedure :: wrap_with_heap_type => helper_wrap_with_heap_type
        procedure :: wrap_with_pointer_type => helper_wrap_with_pointer_type
        procedure :: create_allocatable_array_type_string => helper_create_allocatable_array_type_string
        procedure :: types_equivalent => helper_types_equivalent
        procedure :: types_compatible => helper_types_compatible
        procedure :: get_element_type => helper_get_element_type
        procedure :: is_integer_type_string => helper_is_integer_type_string
        procedure :: is_float_type_string => helper_is_float_type_string
        procedure :: is_array_type_string => helper_is_array_type_string
        procedure :: extract_from_wrapper_type => helper_extract_from_wrapper_type
        procedure :: wrap_type_with_pattern => helper_wrap_type_with_pattern
    end type type_helper_t

contains

    ! Create type helpers
    function create_type_helpers(converter) result(helpers)
        type(mlir_type_converter_t), intent(in), target :: converter
        type(type_helper_t) :: helpers
        
        helpers%converter => converter
    end function create_type_helpers

    ! Destroy type helpers
    subroutine destroy_type_helpers(helpers)
        type(type_helper_t), intent(inout) :: helpers
        
        helpers%converter => null()
    end subroutine destroy_type_helpers

    ! Check if helpers are valid
    function helper_is_valid(this) result(valid)
        class(type_helper_t), intent(in) :: this
        logical :: valid
        valid = associated(this%converter)
    end function helper_is_valid

    ! Get array descriptor string
    function helper_get_array_descriptor(this, element_type, dimensions) result(descriptor)
        class(type_helper_t), intent(in) :: this
        character(len=*), intent(in) :: element_type
        integer, dimension(:), intent(in) :: dimensions
        character(len=:), allocatable :: descriptor
        character(len=200) :: dim_str
        integer :: i
        
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
        
        descriptor = trim(dim_str) // "x" // element_type
    end function helper_get_array_descriptor

    ! Mangle derived type name (Fortran name mangling convention)
    function helper_mangle_derived_type_name(this, type_name) result(mangled)
        class(type_helper_t), intent(in) :: this
        character(len=*), intent(in) :: type_name
        character(len=:), allocatable :: mangled
        
        mangled = "_QT" // type_name
    end function helper_mangle_derived_type_name

    ! Extract array shape from array info
    function helper_extract_array_shape(this, array_info) result(shape)
        class(type_helper_t), intent(in) :: this
        type(array_info_t), intent(in) :: array_info
        integer, allocatable :: shape(:)
        
        if (allocated(array_info%dimensions)) then
            allocate(shape(size(array_info%dimensions)))
            shape = array_info%dimensions
        else
            allocate(shape(0))
        end if
    end function helper_extract_array_shape

    ! Check if array is assumed-shape
    function helper_is_assumed_shape(this, array_info) result(is_assumed)
        class(type_helper_t), intent(in) :: this
        type(array_info_t), intent(in) :: array_info
        logical :: is_assumed
        integer :: i
        
        is_assumed = .false.
        if (allocated(array_info%dimensions)) then
            do i = 1, size(array_info%dimensions)
                if (array_info%dimensions(i) == -1) then
                    is_assumed = .true.
                    return
                end if
            end do
        end if
    end function helper_is_assumed_shape

    ! Check if array is allocatable
    function helper_is_allocatable(this, array_info) result(is_alloc)
        class(type_helper_t), intent(in) :: this
        type(array_info_t), intent(in) :: array_info
        logical :: is_alloc
        
        is_alloc = array_info%is_allocatable
    end function helper_is_allocatable

    ! Check if array is pointer
    function helper_is_pointer(this, array_info) result(is_ptr)
        class(type_helper_t), intent(in) :: this
        type(array_info_t), intent(in) :: array_info
        logical :: is_ptr
        
        is_ptr = array_info%is_pointer
    end function helper_is_pointer

    ! Wrap type with reference
    function helper_wrap_with_reference_type(this, base_type) result(wrapped_type)
        class(type_helper_t), intent(in) :: this
        type(mlir_type_t), intent(in) :: base_type
        type(mlir_type_t) :: wrapped_type
        
        ! For stub, just return the base type - in real implementation would create ref type
        wrapped_type = base_type
    end function helper_wrap_with_reference_type

    ! Get reference type string
    function helper_get_reference_type_string(this, base_type_str) result(ref_str)
        class(type_helper_t), intent(in) :: this
        character(len=*), intent(in) :: base_type_str
        character(len=:), allocatable :: ref_str
        
        ref_str = this%wrap_type_with_pattern(base_type_str, "!fir.ref<")
    end function helper_get_reference_type_string

    ! Wrap with box type
    function helper_wrap_with_box_type(this, base_type_str) result(boxed_str)
        class(type_helper_t), intent(in) :: this
        character(len=*), intent(in) :: base_type_str
        character(len=:), allocatable :: boxed_str
        
        boxed_str = this%wrap_type_with_pattern(base_type_str, "!fir.box<")
    end function helper_wrap_with_box_type

    ! Wrap with heap type
    function helper_wrap_with_heap_type(this, base_type_str) result(heap_str)
        class(type_helper_t), intent(in) :: this
        character(len=*), intent(in) :: base_type_str
        character(len=:), allocatable :: heap_str
        
        heap_str = this%wrap_type_with_pattern(base_type_str, "!fir.heap<")
    end function helper_wrap_with_heap_type

    ! Wrap with pointer type
    function helper_wrap_with_pointer_type(this, base_type_str) result(ptr_str)
        class(type_helper_t), intent(in) :: this
        character(len=*), intent(in) :: base_type_str
        character(len=:), allocatable :: ptr_str
        
        ptr_str = this%wrap_type_with_pattern(base_type_str, "!fir.ptr<")
    end function helper_wrap_with_pointer_type

    ! Create allocatable array type string
    function helper_create_allocatable_array_type_string(this, element_type, rank) result(alloc_str)
        class(type_helper_t), intent(in) :: this
        character(len=*), intent(in) :: element_type
        integer, intent(in) :: rank
        character(len=:), allocatable :: alloc_str
        character(len=50) :: array_str
        integer :: i
        
        ! Build array dimension string with ? for assumed-shape
        array_str = ""
        do i = 1, rank
            if (i > 1) array_str = trim(array_str) // "x"
            array_str = trim(array_str) // "?"
        end do
        
        ! Build the complete allocatable type: !fir.ref<!fir.box<!fir.heap<!fir.array<?xT>>>>
        alloc_str = "!fir.ref<!fir.box<!fir.heap<!fir.array<" // trim(array_str) // "x" // element_type // ">>>>"
    end function helper_create_allocatable_array_type_string

    ! Check if types are equivalent
    function helper_types_equivalent(this, type1, type2) result(equivalent)
        class(type_helper_t), intent(in) :: this
        type(mlir_type_t), intent(in) :: type1, type2
        logical :: equivalent
        integer :: kind1, kind2
        
        ! First check if both types are valid
        if (.not. type1%is_valid() .or. .not. type2%is_valid()) then
            equivalent = .false.
            return
        end if
        
        ! Check if they have the same kind
        kind1 = get_type_kind(type1)
        kind2 = get_type_kind(type2)
        
        if (kind1 /= kind2) then
            equivalent = .false.
            return
        end if
        
        ! For integer types, check width
        if (kind1 == TYPE_INTEGER) then
            equivalent = (get_integer_width(type1) == get_integer_width(type2))
        else
            ! For other types, fall back to pointer comparison for now
            equivalent = c_associated(type1%ptr, type2%ptr)
        end if
    end function helper_types_equivalent

    ! Check if types are compatible
    function helper_types_compatible(this, type1, type2) result(compatible)
        class(type_helper_t), intent(in) :: this
        type(mlir_type_t), intent(in) :: type1, type2
        logical :: compatible
        
        ! For stub, same as equivalent
        compatible = this%types_equivalent(type1, type2)
    end function helper_types_compatible

    ! Get element type from complex type string
    recursive function helper_get_element_type(this, type_string) result(element_type)
        class(type_helper_t), intent(in) :: this
        character(len=*), intent(in) :: type_string
        character(len=:), allocatable :: element_type
        integer :: pos1, pos2
        
        
        ! Extract element type from various patterns - check outer types first
        ! Use consolidated pattern for wrapper types
        element_type = this%extract_from_wrapper_type(type_string, ["!fir.ref< ", "!fir.box< ", "!fir.heap<"])
        
        if (element_type /= "unknown") then
            return
        else if (index(type_string, "!fir.array<") > 0) then
            ! Pattern: !fir.array<10xi32> -> i32
            pos1 = index(type_string, "x", .true.) + 1  ! Find last 'x'
            pos2 = len(type_string)
            ! Find the matching closing bracket
            if (type_string(pos2:pos2) == ">") then
                pos2 = pos2 - 1  ! Point to character before '>'
            end if
            if (pos1 > 1 .and. pos2 >= pos1) then
                element_type = type_string(pos1:pos2)
            else
                element_type = "unknown"
            end if
        else
            element_type = "unknown"
        end if
    end function helper_get_element_type

    ! Check if type string represents integer type
    function helper_is_integer_type_string(this, type_string) result(is_int)
        class(type_helper_t), intent(in) :: this
        character(len=*), intent(in) :: type_string
        logical :: is_int
        
        is_int = (index(type_string, "i") == 1)  ! Starts with 'i'
    end function helper_is_integer_type_string

    ! Check if type string represents float type
    function helper_is_float_type_string(this, type_string) result(is_float)
        class(type_helper_t), intent(in) :: this
        character(len=*), intent(in) :: type_string
        logical :: is_float
        
        is_float = (index(type_string, "f") == 1)  ! Starts with 'f'
    end function helper_is_float_type_string

    ! Check if type string represents array type
    function helper_is_array_type_string(this, type_string) result(is_array)
        class(type_helper_t), intent(in) :: this
        character(len=*), intent(in) :: type_string
        logical :: is_array
        
        is_array = (index(type_string, "!fir.array<") > 0)
    end function helper_is_array_type_string

    ! Helper function to extract inner type from wrapped type
    function extract_inner_type(type_string, start_pattern, end_pattern) result(inner)
        character(len=*), intent(in) :: type_string
        character(len=*), intent(in) :: start_pattern, end_pattern
        character(len=:), allocatable :: inner
        integer :: start_pos, end_pos, bracket_count
        integer :: i
        
        start_pos = index(type_string, start_pattern)
        if (start_pos == 0) then
            inner = type_string
            return
        end if
        
        start_pos = start_pos + len(start_pattern)
        
        ! Find matching closing bracket
        bracket_count = 1
        end_pos = start_pos
        do i = start_pos, len(type_string)
            if (type_string(i:i) == "<") then
                bracket_count = bracket_count + 1
            else if (type_string(i:i) == ">") then
                bracket_count = bracket_count - 1
                if (bracket_count == 0) then
                    end_pos = i - 1
                    exit
                end if
            end if
        end do
        
        if (end_pos > start_pos) then
            inner = type_string(start_pos:end_pos)
        else
            inner = type_string
        end if
    end function extract_inner_type

    ! Consolidated pattern: extract from wrapper type if present
    recursive function helper_extract_from_wrapper_type(this, type_string, wrapper_patterns) result(element_type)
        class(type_helper_t), intent(in) :: this
        character(len=*), intent(in) :: type_string
        character(len=*), dimension(:), intent(in) :: wrapper_patterns
        character(len=:), allocatable :: element_type
        integer :: i
        
        ! Check each wrapper pattern in order
        do i = 1, size(wrapper_patterns)
            if (index(type_string, trim(wrapper_patterns(i))) > 0) then
                element_type = this%get_element_type(extract_inner_type(type_string, trim(wrapper_patterns(i)), ">"))
                return
            end if
        end do
        
        ! If no wrapper found, return unknown
        element_type = "unknown"
    end function helper_extract_from_wrapper_type

    ! Consolidated pattern: wrap type string with prefix and suffix
    function helper_wrap_type_with_pattern(this, base_type_str, prefix, suffix) result(wrapped_str)
        class(type_helper_t), intent(in) :: this
        character(len=*), intent(in) :: base_type_str
        character(len=*), intent(in) :: prefix
        character(len=*), intent(in), optional :: suffix
        character(len=:), allocatable :: wrapped_str
        character(len=:), allocatable :: end_part
        
        if (present(suffix)) then
            end_part = suffix
        else
            end_part = ">"
        end if
        
        wrapped_str = prefix // base_type_str // end_part
    end function helper_wrap_type_with_pattern

    ! Type assertion utilities

    ! Assert that a type is valid (non-null pointer)
    subroutine assert_valid_type(mlir_type, context_msg)
        type(mlir_type_t), intent(in) :: mlir_type
        character(len=*), intent(in), optional :: context_msg
        character(len=:), allocatable :: msg
        
        if (.not. mlir_type%is_valid()) then
            if (present(context_msg)) then
                msg = "Type assertion failed in " // context_msg // ": invalid MLIR type (null pointer)"
            else
                msg = "Type assertion failed: invalid MLIR type (null pointer)"
            end if
            error stop msg
        end if
    end subroutine assert_valid_type

    ! Assert that a type is an integer type
    subroutine assert_integer_type(mlir_type, context_msg)
        type(mlir_type_t), intent(in) :: mlir_type
        character(len=*), intent(in), optional :: context_msg
        character(len=:), allocatable :: msg
        
        call assert_valid_type(mlir_type, context_msg)
        
        if (.not. is_integer_type(mlir_type)) then
            if (present(context_msg)) then
                msg = "Type assertion failed in " // context_msg // ": expected integer type"
            else
                msg = "Type assertion failed: expected integer type"
            end if
            error stop msg
        end if
    end subroutine assert_integer_type

    ! Assert that a type is an array type
    subroutine assert_array_type(mlir_type, context_msg)
        type(mlir_type_t), intent(in) :: mlir_type
        character(len=*), intent(in), optional :: context_msg
        character(len=:), allocatable :: msg
        
        call assert_valid_type(mlir_type, context_msg)
        
        if (.not. is_array_type(mlir_type)) then
            if (present(context_msg)) then
                msg = "Type assertion failed in " // context_msg // ": expected array type"
            else
                msg = "Type assertion failed: expected array type"
            end if
            error stop msg
        end if
    end subroutine assert_array_type

end module type_conversion_helpers