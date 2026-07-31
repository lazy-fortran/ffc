module ffc_array_descriptor
    use, intrinsic :: iso_c_binding, only: c_associated, c_int32_t, c_int64_t, &
        c_intptr_t, c_null_ptr, c_ptr, c_size_t
    implicit none
    private

    public :: array_dimension_t
    public :: array_descriptor_t
    public :: set_array_descriptor_null
    public :: set_contiguous_array_descriptor
    public :: set_strided_array_descriptor
    public :: array_descriptor_element_offset
    public :: array_descriptor_element_address
    public :: array_descriptor_size
    public :: array_descriptor_extent
    public :: array_descriptor_lower_bound
    public :: array_descriptor_upper_bound
    public :: array_descriptor_stride_bytes
    public :: array_descriptor_is_allocated
    public :: array_descriptor_is_owned
    public :: array_descriptor_is_contiguous
    public :: release_array_descriptor

    integer, parameter, public :: ARRAY_DESCRIPTOR_MAX_RANK = 7

    integer(c_intptr_t), parameter, public :: &
        ARRAY_DESCRIPTOR_BASE_OFFSET = 0_c_intptr_t
    integer(c_intptr_t), parameter, public :: &
        ARRAY_DESCRIPTOR_ELEMENT_SIZE_OFFSET = 8_c_intptr_t
    integer(c_intptr_t), parameter, public :: &
        ARRAY_DESCRIPTOR_ELEMENT_TYPE_OFFSET = 16_c_intptr_t
    integer(c_intptr_t), parameter, public :: &
        ARRAY_DESCRIPTOR_RANK_OFFSET = 20_c_intptr_t
    integer(c_intptr_t), parameter, public :: &
        ARRAY_DESCRIPTOR_FLAGS_OFFSET = 24_c_intptr_t
    integer(c_intptr_t), parameter, public :: &
        ARRAY_DESCRIPTOR_RESERVED_OFFSET = 28_c_intptr_t
    integer(c_intptr_t), parameter, public :: &
        ARRAY_DESCRIPTOR_DIM_OFFSET = 32_c_intptr_t
    integer(c_size_t), parameter, public :: ARRAY_DIMENSION_BYTES = 24_c_size_t
    integer(c_size_t), parameter, public :: ARRAY_DESCRIPTOR_BYTES = 200_c_size_t

    integer(c_intptr_t), parameter, public :: &
        ARRAY_DIMENSION_LOWER_OFFSET = 0_c_intptr_t
    integer(c_intptr_t), parameter, public :: &
        ARRAY_DIMENSION_EXTENT_OFFSET = 8_c_intptr_t
    integer(c_intptr_t), parameter, public :: &
        ARRAY_DIMENSION_STRIDE_OFFSET = 16_c_intptr_t

    integer(c_int32_t), parameter, public :: ARRAY_FLAG_NONE = 0_c_int32_t
    integer(c_int32_t), parameter, public :: ARRAY_FLAG_ALLOCATED = 1_c_int32_t
    integer(c_int32_t), parameter, public :: ARRAY_FLAG_ASSOCIATED = 2_c_int32_t
    integer(c_int32_t), parameter, public :: ARRAY_FLAG_OWNS_DATA = 4_c_int32_t
    integer(c_int32_t), parameter, public :: ARRAY_FLAG_CONTIGUOUS = 8_c_int32_t

    integer(c_int32_t), parameter, public :: ARRAY_ELEMENT_NONE = 0_c_int32_t
    integer(c_int32_t), parameter, public :: ARRAY_ELEMENT_INTEGER = 1_c_int32_t
    integer(c_int32_t), parameter, public :: ARRAY_ELEMENT_REAL = 2_c_int32_t
    integer(c_int32_t), parameter, public :: ARRAY_ELEMENT_LOGICAL = 3_c_int32_t
    integer(c_int32_t), parameter, public :: ARRAY_ELEMENT_COMPLEX = 4_c_int32_t
    integer(c_int32_t), parameter, public :: ARRAY_ELEMENT_CHARACTER = 5_c_int32_t
    integer(c_int32_t), parameter, public :: ARRAY_ELEMENT_DERIVED = 6_c_int32_t

    integer, parameter, public :: ARRAY_DESCRIPTOR_OK = 0
    integer, parameter, public :: ARRAY_DESCRIPTOR_INVALID_RANK = 1
    integer, parameter, public :: ARRAY_DESCRIPTOR_INVALID_EXTENT = 2
    integer, parameter, public :: ARRAY_DESCRIPTOR_INVALID_ELEMENT_SIZE = 3
    integer, parameter, public :: ARRAY_DESCRIPTOR_INVALID_ELEMENT_TYPE = 4
    integer, parameter, public :: ARRAY_DESCRIPTOR_NULL_DATA = 5
    integer, parameter, public :: ARRAY_DESCRIPTOR_INVALID_OWNERSHIP = 6
    integer, parameter, public :: ARRAY_DESCRIPTOR_INVALID_INDEX = 7

    type, bind(c) :: array_dimension_t
        integer(c_int64_t) :: lower_bound = 1_c_int64_t
        integer(c_int64_t) :: extent = 0_c_int64_t
        integer(c_int64_t) :: stride_bytes = 0_c_int64_t
    end type array_dimension_t

    type, bind(c) :: array_descriptor_t
        type(c_ptr) :: base = c_null_ptr
        integer(c_int64_t) :: element_size = 0_c_int64_t
        integer(c_int32_t) :: element_type = ARRAY_ELEMENT_NONE
        integer(c_int32_t) :: rank = 0_c_int32_t
        integer(c_int32_t) :: flags = ARRAY_FLAG_NONE
        integer(c_int32_t) :: reserved = 0_c_int32_t
        type(array_dimension_t) :: dim(ARRAY_DESCRIPTOR_MAX_RANK)
    end type array_descriptor_t

contains

    subroutine set_array_descriptor_null(descriptor)
        type(array_descriptor_t), intent(out) :: descriptor
        integer :: dim

        descriptor%base = c_null_ptr
        descriptor%element_size = 0_c_int64_t
        descriptor%element_type = ARRAY_ELEMENT_NONE
        descriptor%rank = 0_c_int32_t
        descriptor%flags = ARRAY_FLAG_NONE
        descriptor%reserved = 0_c_int32_t
        do dim = 1, ARRAY_DESCRIPTOR_MAX_RANK
            descriptor%dim(dim)%lower_bound = 1_c_int64_t
            descriptor%dim(dim)%extent = 0_c_int64_t
            descriptor%dim(dim)%stride_bytes = 0_c_int64_t
        end do
    end subroutine set_array_descriptor_null

    subroutine set_contiguous_array_descriptor(descriptor, base, element_type, &
            element_size, rank, lower_bounds, extents, owns_data, status)
        type(array_descriptor_t), intent(out) :: descriptor
        type(c_ptr), intent(in) :: base
        integer(c_int32_t), intent(in) :: element_type
        integer(c_int64_t), intent(in) :: element_size
        integer, intent(in) :: rank
        integer(c_int64_t), intent(in) :: lower_bounds(:)
        integer(c_int64_t), intent(in) :: extents(:)
        logical, intent(in) :: owns_data
        integer, intent(out) :: status

        integer(c_int64_t) :: strides(ARRAY_DESCRIPTOR_MAX_RANK)
        integer(c_int64_t) :: running
        integer :: dim

        call set_array_descriptor_null(descriptor)
        call validate_metadata(element_type, element_size, rank, lower_bounds, &
            extents, status)
        if (status /= ARRAY_DESCRIPTOR_OK) return

        running = element_size
        do dim = 1, rank
            strides(dim) = running
            running = running*extents(dim)
        end do

        call install(descriptor, base, element_type, element_size, rank, &
            lower_bounds, extents, strides(1:rank), owns_data, .true., status)
    end subroutine set_contiguous_array_descriptor

    subroutine set_strided_array_descriptor(descriptor, base, element_type, &
            element_size, rank, lower_bounds, extents, strides, status)
        type(array_descriptor_t), intent(out) :: descriptor
        type(c_ptr), intent(in) :: base
        integer(c_int32_t), intent(in) :: element_type
        integer(c_int64_t), intent(in) :: element_size
        integer, intent(in) :: rank
        integer(c_int64_t), intent(in) :: lower_bounds(:)
        integer(c_int64_t), intent(in) :: extents(:)
        integer(c_int64_t), intent(in) :: strides(:)
        integer, intent(out) :: status

        logical :: contiguous
        integer(c_int64_t) :: running
        integer :: dim

        call set_array_descriptor_null(descriptor)
        call validate_metadata(element_type, element_size, rank, lower_bounds, &
            extents, status)
        if (status /= ARRAY_DESCRIPTOR_OK) return
        if (size(strides) < rank) then
            status = ARRAY_DESCRIPTOR_INVALID_RANK
            return
        end if

        contiguous = .true.
        running = element_size
        do dim = 1, rank
            if (strides(dim) /= running) contiguous = .false.
            running = running*extents(dim)
        end do

        call install(descriptor, base, element_type, element_size, rank, &
            lower_bounds, extents, strides(1:rank), .false., contiguous, status)
    end subroutine set_strided_array_descriptor

    subroutine validate_metadata(element_type, element_size, rank, lower_bounds, &
            extents, status)
        integer(c_int32_t), intent(in) :: element_type
        integer(c_int64_t), intent(in) :: element_size
        integer, intent(in) :: rank
        integer(c_int64_t), intent(in) :: lower_bounds(:)
        integer(c_int64_t), intent(in) :: extents(:)
        integer, intent(out) :: status

        integer :: dim

        status = ARRAY_DESCRIPTOR_OK
        if (rank < 1) then
            status = ARRAY_DESCRIPTOR_INVALID_RANK
            return
        end if
        if (rank > ARRAY_DESCRIPTOR_MAX_RANK) then
            status = ARRAY_DESCRIPTOR_INVALID_RANK
            return
        end if
        if (size(lower_bounds) < rank) then
            status = ARRAY_DESCRIPTOR_INVALID_RANK
            return
        end if
        if (size(extents) < rank) then
            status = ARRAY_DESCRIPTOR_INVALID_RANK
            return
        end if
        if (element_size <= 0_c_int64_t) then
            status = ARRAY_DESCRIPTOR_INVALID_ELEMENT_SIZE
            return
        end if
        if (element_type < ARRAY_ELEMENT_INTEGER) then
            status = ARRAY_DESCRIPTOR_INVALID_ELEMENT_TYPE
            return
        end if
        if (element_type > ARRAY_ELEMENT_DERIVED) then
            status = ARRAY_DESCRIPTOR_INVALID_ELEMENT_TYPE
            return
        end if
        do dim = 1, rank
            if (extents(dim) < 0_c_int64_t) then
                status = ARRAY_DESCRIPTOR_INVALID_EXTENT
                return
            end if
        end do
    end subroutine validate_metadata

    subroutine install(descriptor, base, element_type, element_size, rank, &
            lower_bounds, extents, strides, owns_data, contiguous, status)
        type(array_descriptor_t), intent(inout) :: descriptor
        type(c_ptr), intent(in) :: base
        integer(c_int32_t), intent(in) :: element_type
        integer(c_int64_t), intent(in) :: element_size
        integer, intent(in) :: rank
        integer(c_int64_t), intent(in) :: lower_bounds(:)
        integer(c_int64_t), intent(in) :: extents(:)
        integer(c_int64_t), intent(in) :: strides(:)
        logical, intent(in) :: owns_data
        logical, intent(in) :: contiguous
        integer, intent(out) :: status

        integer(c_int64_t) :: total
        integer :: dim

        total = 1_c_int64_t
        do dim = 1, rank
            total = total*extents(dim)
        end do
        if (total > 0_c_int64_t) then
            if (.not. c_associated(base)) then
                status = ARRAY_DESCRIPTOR_NULL_DATA
                return
            end if
        end if
        if (owns_data) then
            if (.not. c_associated(base)) then
                status = ARRAY_DESCRIPTOR_INVALID_OWNERSHIP
                return
            end if
        end if

        descriptor%base = base
        descriptor%element_size = element_size
        descriptor%element_type = element_type
        descriptor%rank = int(rank, c_int32_t)
        descriptor%flags = ARRAY_FLAG_ALLOCATED + ARRAY_FLAG_ASSOCIATED
        if (owns_data) descriptor%flags = descriptor%flags + ARRAY_FLAG_OWNS_DATA
        if (contiguous) descriptor%flags = descriptor%flags + ARRAY_FLAG_CONTIGUOUS
        do dim = 1, rank
            descriptor%dim(dim)%lower_bound = lower_bounds(dim)
            descriptor%dim(dim)%extent = extents(dim)
            descriptor%dim(dim)%stride_bytes = strides(dim)
        end do
        status = ARRAY_DESCRIPTOR_OK
    end subroutine install

    subroutine array_descriptor_element_offset(descriptor, indices, offset, status)
        type(array_descriptor_t), intent(in) :: descriptor
        integer(c_int64_t), intent(in) :: indices(:)
        integer(c_int64_t), intent(out) :: offset
        integer, intent(out) :: status

        integer(c_int64_t) :: position
        integer :: dim

        offset = 0_c_int64_t
        if (descriptor%rank < 1_c_int32_t) then
            status = ARRAY_DESCRIPTOR_INVALID_RANK
            return
        end if
        if (size(indices) < int(descriptor%rank)) then
            status = ARRAY_DESCRIPTOR_INVALID_RANK
            return
        end if

        do dim = 1, int(descriptor%rank)
            position = indices(dim) - descriptor%dim(dim)%lower_bound
            if (position < 0_c_int64_t) then
                offset = 0_c_int64_t
                status = ARRAY_DESCRIPTOR_INVALID_INDEX
                return
            end if
            if (position >= descriptor%dim(dim)%extent) then
                offset = 0_c_int64_t
                status = ARRAY_DESCRIPTOR_INVALID_INDEX
                return
            end if
            offset = offset + position*descriptor%dim(dim)%stride_bytes
        end do
        status = ARRAY_DESCRIPTOR_OK
    end subroutine array_descriptor_element_offset

    subroutine array_descriptor_element_address(descriptor, indices, address, &
            status)
        type(array_descriptor_t), intent(in) :: descriptor
        integer(c_int64_t), intent(in) :: indices(:)
        integer(c_intptr_t), intent(out) :: address
        integer, intent(out) :: status

        integer(c_intptr_t) :: base_address
        integer(c_int64_t) :: offset

        address = 0_c_intptr_t
        call array_descriptor_element_offset(descriptor, indices, offset, status)
        if (status /= ARRAY_DESCRIPTOR_OK) return
        if (.not. c_associated(descriptor%base)) then
            status = ARRAY_DESCRIPTOR_NULL_DATA
            return
        end if
        base_address = transfer(descriptor%base, base_address)
        address = base_address + int(offset, c_intptr_t)
    end subroutine array_descriptor_element_address

    pure function array_descriptor_size(descriptor) result(total)
        type(array_descriptor_t), intent(in) :: descriptor
        integer(c_int64_t) :: total
        integer :: dim

        total = 0_c_int64_t
        if (descriptor%rank < 1_c_int32_t) return
        total = 1_c_int64_t
        do dim = 1, int(descriptor%rank)
            total = total*descriptor%dim(dim)%extent
        end do
    end function array_descriptor_size

    pure function array_descriptor_extent(descriptor, dim) result(extent)
        type(array_descriptor_t), intent(in) :: descriptor
        integer, intent(in) :: dim
        integer(c_int64_t) :: extent

        extent = 0_c_int64_t
        if (dim < 1) return
        if (dim > int(descriptor%rank)) return
        extent = descriptor%dim(dim)%extent
    end function array_descriptor_extent

    pure function array_descriptor_lower_bound(descriptor, dim) result(lower)
        type(array_descriptor_t), intent(in) :: descriptor
        integer, intent(in) :: dim
        integer(c_int64_t) :: lower

        lower = 1_c_int64_t
        if (dim < 1) return
        if (dim > int(descriptor%rank)) return
        lower = descriptor%dim(dim)%lower_bound
    end function array_descriptor_lower_bound

    pure function array_descriptor_upper_bound(descriptor, dim) result(upper)
        type(array_descriptor_t), intent(in) :: descriptor
        integer, intent(in) :: dim
        integer(c_int64_t) :: upper

        upper = 0_c_int64_t
        if (dim < 1) return
        if (dim > int(descriptor%rank)) return
        upper = descriptor%dim(dim)%lower_bound + descriptor%dim(dim)%extent &
                - 1_c_int64_t
    end function array_descriptor_upper_bound

    pure function array_descriptor_stride_bytes(descriptor, dim) result(stride)
        type(array_descriptor_t), intent(in) :: descriptor
        integer, intent(in) :: dim
        integer(c_int64_t) :: stride

        stride = 0_c_int64_t
        if (dim < 1) return
        if (dim > int(descriptor%rank)) return
        stride = descriptor%dim(dim)%stride_bytes
    end function array_descriptor_stride_bytes

    pure function array_descriptor_is_allocated(descriptor) result(is_allocated)
        type(array_descriptor_t), intent(in) :: descriptor
        logical :: is_allocated

        is_allocated = has_flag(descriptor%flags, ARRAY_FLAG_ALLOCATED)
    end function array_descriptor_is_allocated

    pure function array_descriptor_is_owned(descriptor) result(is_owned)
        type(array_descriptor_t), intent(in) :: descriptor
        logical :: is_owned

        is_owned = has_flag(descriptor%flags, ARRAY_FLAG_OWNS_DATA)
    end function array_descriptor_is_owned

    pure function array_descriptor_is_contiguous(descriptor) result(is_contiguous)
        type(array_descriptor_t), intent(in) :: descriptor
        logical :: is_contiguous

        is_contiguous = has_flag(descriptor%flags, ARRAY_FLAG_CONTIGUOUS)
    end function array_descriptor_is_contiguous

    pure function has_flag(flags, flag) result(present_flag)
        integer(c_int32_t), intent(in) :: flags
        integer(c_int32_t), intent(in) :: flag
        logical :: present_flag

        present_flag = iand(flags, flag) /= 0_c_int32_t
    end function has_flag

    function release_array_descriptor(descriptor) result(base)
        type(array_descriptor_t), intent(inout) :: descriptor
        type(c_ptr) :: base

        base = c_null_ptr
        if (array_descriptor_is_owned(descriptor)) base = descriptor%base
        call set_array_descriptor_null(descriptor)
    end function release_array_descriptor

end module ffc_array_descriptor
