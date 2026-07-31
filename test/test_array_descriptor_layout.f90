program test_array_descriptor_layout
    use ffc_array_descriptor, only: array_descriptor_t, array_dimension_t, &
        ARRAY_DESCRIPTOR_MAX_RANK, ARRAY_DESCRIPTOR_BASE_OFFSET, &
        ARRAY_DESCRIPTOR_ELEMENT_SIZE_OFFSET, &
        ARRAY_DESCRIPTOR_ELEMENT_TYPE_OFFSET, ARRAY_DESCRIPTOR_RANK_OFFSET, &
        ARRAY_DESCRIPTOR_FLAGS_OFFSET, ARRAY_DESCRIPTOR_RESERVED_OFFSET, &
        ARRAY_DESCRIPTOR_DIM_OFFSET, ARRAY_DIMENSION_BYTES, &
        ARRAY_DESCRIPTOR_BYTES, ARRAY_DIMENSION_LOWER_OFFSET, &
        ARRAY_DIMENSION_EXTENT_OFFSET, ARRAY_DIMENSION_STRIDE_OFFSET, &
        ARRAY_FLAG_NONE, ARRAY_FLAG_ALLOCATED, ARRAY_FLAG_ASSOCIATED, &
        ARRAY_FLAG_OWNS_DATA, ARRAY_FLAG_CONTIGUOUS, ARRAY_ELEMENT_NONE, &
        ARRAY_ELEMENT_INTEGER, ARRAY_ELEMENT_DERIVED, ARRAY_DESCRIPTOR_OK, &
        ARRAY_DESCRIPTOR_INVALID_RANK, ARRAY_DESCRIPTOR_INVALID_EXTENT, &
        ARRAY_DESCRIPTOR_INVALID_ELEMENT_SIZE, &
        ARRAY_DESCRIPTOR_INVALID_ELEMENT_TYPE, ARRAY_DESCRIPTOR_NULL_DATA, &
        ARRAY_DESCRIPTOR_INVALID_OWNERSHIP, ARRAY_DESCRIPTOR_INVALID_INDEX, &
        set_array_descriptor_null, set_contiguous_array_descriptor, &
        set_strided_array_descriptor, array_descriptor_element_offset, &
        array_descriptor_element_address, array_descriptor_size, &
        array_descriptor_extent, array_descriptor_lower_bound, &
        array_descriptor_upper_bound, array_descriptor_stride_bytes, &
        array_descriptor_is_allocated, array_descriptor_is_owned, &
        array_descriptor_is_contiguous, release_array_descriptor
    use, intrinsic :: iso_c_binding, only: c_associated, c_int32_t, c_int64_t, &
        c_intptr_t, c_loc, c_null_ptr, c_ptr, c_sizeof
    implicit none

    integer(c_int32_t), parameter :: INT32_ELEMENT_SIZE = 4_c_int32_t

    type(array_descriptor_t), target :: descriptor
    type(array_dimension_t), target :: dimension_probe
    integer(c_int32_t), target :: storage(6)
    type(c_ptr) :: released
    integer(c_intptr_t) :: base_address
    integer(c_intptr_t) :: address
    integer(c_int64_t) :: lower_bounds(2)
    integer(c_int64_t) :: extents(2)
    integer(c_int64_t) :: strides(2)
    integer(c_int64_t) :: offset
    integer(c_int64_t) :: linear(3)
    integer :: status

    storage = 0_c_int32_t

    call set_array_descriptor_null(descriptor)
    call require_descriptor_layout(descriptor)
    call require_dimension_layout(dimension_probe)
    call require_null_state(descriptor, 'initial null')

    lower_bounds = [1_c_int64_t, 1_c_int64_t]
    extents = [2_c_int64_t, 3_c_int64_t]
    call set_contiguous_array_descriptor(descriptor, c_loc(storage), &
        ARRAY_ELEMENT_INTEGER, 4_c_int64_t, 2, lower_bounds, extents, .true., &
        status)
    call require(status == ARRAY_DESCRIPTOR_OK, 'contiguous status')
    call require(descriptor%rank == 2_c_int32_t, 'rank field')
    call require(array_descriptor_size(descriptor) == 6_c_int64_t, 'size')
    call require(array_descriptor_extent(descriptor, 1) == 2_c_int64_t, 'extent 1')
    call require(array_descriptor_extent(descriptor, 2) == 3_c_int64_t, 'extent 2')
    call require(array_descriptor_upper_bound(descriptor, 2) == 3_c_int64_t, &
        'upper bound 2')
    call require(array_descriptor_stride_bytes(descriptor, 1) == 4_c_int64_t, &
        'column-major stride 1')
    call require(array_descriptor_stride_bytes(descriptor, 2) == 8_c_int64_t, &
        'column-major stride 2')
    call require(array_descriptor_is_allocated(descriptor), 'allocated flag')
    call require(array_descriptor_is_owned(descriptor), 'owned flag')
    call require(array_descriptor_is_contiguous(descriptor), 'contiguous flag')

    linear(1) = element_index([1_c_int64_t, 1_c_int64_t])
    linear(2) = element_index([2_c_int64_t, 2_c_int64_t])
    linear(3) = element_index([2_c_int64_t, 3_c_int64_t])
    call require(linear(1) == 1_c_int64_t, 'first element index')
    call require(linear(2) == 4_c_int64_t, 'middle element index')
    call require(linear(3) == 6_c_int64_t, 'last element index')

    base_address = transfer(c_loc(storage), base_address)
    call array_descriptor_element_address(descriptor, &
        [2_c_int64_t, 3_c_int64_t], address, status)
    call require(status == ARRAY_DESCRIPTOR_OK, 'address status')
    call require(address - base_address == 20_c_intptr_t, 'last element address')

    call array_descriptor_element_offset(descriptor, &
        [0_c_int64_t, 1_c_int64_t], offset, status)
    call require(status == ARRAY_DESCRIPTOR_INVALID_INDEX, 'index below lower')
    call array_descriptor_element_offset(descriptor, &
        [3_c_int64_t, 1_c_int64_t], offset, status)
    call require(status == ARRAY_DESCRIPTOR_INVALID_INDEX, 'index above upper')

    released = release_array_descriptor(descriptor)
    call require(c_associated(released, c_loc(storage)), 'owned release')
    call require_null_state(descriptor, 'owned release')

    lower_bounds = [-1_c_int64_t, 2_c_int64_t]
    extents = [2_c_int64_t, 3_c_int64_t]
    call set_contiguous_array_descriptor(descriptor, c_loc(storage), &
        ARRAY_ELEMENT_INTEGER, 4_c_int64_t, 2, lower_bounds, extents, .false., &
        status)
    call require(status == ARRAY_DESCRIPTOR_OK, 'nonunit lower status')
    call require(array_descriptor_lower_bound(descriptor, 1) == -1_c_int64_t, &
        'nonunit lower bound')
    call require(array_descriptor_upper_bound(descriptor, 1) == 0_c_int64_t, &
        'nonunit upper bound')
    call require(element_index([-1_c_int64_t, 2_c_int64_t]) == 1_c_int64_t, &
        'nonunit first element')
    call require(element_index([0_c_int64_t, 4_c_int64_t]) == 6_c_int64_t, &
        'nonunit last element')
    released = release_array_descriptor(descriptor)
    call require(.not. c_associated(released), 'borrowed release')

    lower_bounds = [1_c_int64_t, 1_c_int64_t]
    extents = [2_c_int64_t, 2_c_int64_t]
    strides = [8_c_int64_t, 8_c_int64_t]
    call set_strided_array_descriptor(descriptor, c_loc(storage), &
        ARRAY_ELEMENT_INTEGER, 4_c_int64_t, 2, lower_bounds, extents, strides, &
        status)
    call require(status == ARRAY_DESCRIPTOR_OK, 'strided status')
    call require(.not. array_descriptor_is_contiguous(descriptor), &
        'strided view not contiguous')
    call require(.not. array_descriptor_is_owned(descriptor), 'view never owns')
    call require(element_index([2_c_int64_t, 2_c_int64_t]) == 5_c_int64_t, &
        'strided last element')
    released = release_array_descriptor(descriptor)
    call require(.not. c_associated(released), 'view release keeps base')

    call set_contiguous_array_descriptor(descriptor, c_loc(storage), &
        ARRAY_ELEMENT_INTEGER, 4_c_int64_t, 0, lower_bounds, extents, .false., &
        status)
    call require(status == ARRAY_DESCRIPTOR_INVALID_RANK, 'rank zero rejected')
    call require_null_state(descriptor, 'rank zero')

    call set_contiguous_array_descriptor(descriptor, c_loc(storage), &
        ARRAY_ELEMENT_INTEGER, 4_c_int64_t, ARRAY_DESCRIPTOR_MAX_RANK + 1, &
        lower_bounds, extents, .false., status)
    call require(status == ARRAY_DESCRIPTOR_INVALID_RANK, 'rank eight rejected')
    call require_null_state(descriptor, 'rank eight')

    extents = [2_c_int64_t, -1_c_int64_t]
    call set_contiguous_array_descriptor(descriptor, c_loc(storage), &
        ARRAY_ELEMENT_INTEGER, 4_c_int64_t, 2, lower_bounds, extents, .false., &
        status)
    call require(status == ARRAY_DESCRIPTOR_INVALID_EXTENT, &
        'negative extent rejected')
    call require_null_state(descriptor, 'negative extent')

    extents = [2_c_int64_t, 3_c_int64_t]
    call set_contiguous_array_descriptor(descriptor, c_loc(storage), &
        ARRAY_ELEMENT_INTEGER, 0_c_int64_t, 2, lower_bounds, extents, .false., &
        status)
    call require(status == ARRAY_DESCRIPTOR_INVALID_ELEMENT_SIZE, &
        'zero element size rejected')

    call set_contiguous_array_descriptor(descriptor, c_loc(storage), &
        ARRAY_ELEMENT_DERIVED + 1_c_int32_t, 4_c_int64_t, 2, lower_bounds, &
        extents, .false., status)
    call require(status == ARRAY_DESCRIPTOR_INVALID_ELEMENT_TYPE, &
        'unknown element type rejected')

    call set_contiguous_array_descriptor(descriptor, c_null_ptr, &
        ARRAY_ELEMENT_INTEGER, 4_c_int64_t, 2, lower_bounds, extents, .false., &
        status)
    call require(status == ARRAY_DESCRIPTOR_NULL_DATA, 'null base rejected')

    extents = [0_c_int64_t, 3_c_int64_t]
    call set_contiguous_array_descriptor(descriptor, c_null_ptr, &
        ARRAY_ELEMENT_INTEGER, 4_c_int64_t, 2, lower_bounds, extents, .true., &
        status)
    call require(status == ARRAY_DESCRIPTOR_INVALID_OWNERSHIP, &
        'ownership without storage rejected')

    print *, linear(1), linear(2), linear(3)
    print *, 'PASS: array descriptor layout and addressing'

contains

    function element_index(indices) result(index_value)
        integer(c_int64_t), intent(in) :: indices(:)
        integer(c_int64_t) :: index_value
        integer(c_int64_t) :: byte_offset
        integer :: local_status

        call array_descriptor_element_offset(descriptor, indices, byte_offset, &
            local_status)
        call require(local_status == ARRAY_DESCRIPTOR_OK, 'element offset status')
        index_value = byte_offset/int(INT32_ELEMENT_SIZE, c_int64_t) + 1_c_int64_t
    end function element_index

    subroutine require_descriptor_layout(value)
        type(array_descriptor_t), target, intent(inout) :: value
        integer(c_intptr_t) :: origin
        integer(c_intptr_t) :: field

        origin = transfer(c_loc(value), origin)
        field = transfer(c_loc(value%base), field)
        call require(field - origin == ARRAY_DESCRIPTOR_BASE_OFFSET, 'base offset')
        field = transfer(c_loc(value%element_size), field)
        call require(field - origin == ARRAY_DESCRIPTOR_ELEMENT_SIZE_OFFSET, &
            'element size offset')
        field = transfer(c_loc(value%element_type), field)
        call require(field - origin == ARRAY_DESCRIPTOR_ELEMENT_TYPE_OFFSET, &
            'element type offset')
        field = transfer(c_loc(value%rank), field)
        call require(field - origin == ARRAY_DESCRIPTOR_RANK_OFFSET, 'rank offset')
        field = transfer(c_loc(value%flags), field)
        call require(field - origin == ARRAY_DESCRIPTOR_FLAGS_OFFSET, &
            'flags offset')
        field = transfer(c_loc(value%reserved), field)
        call require(field - origin == ARRAY_DESCRIPTOR_RESERVED_OFFSET, &
            'reserved offset')
        field = transfer(c_loc(value%dim(1)%lower_bound), field)
        call require(field - origin == ARRAY_DESCRIPTOR_DIM_OFFSET, 'dim offset')
        field = transfer(c_loc(value%dim(2)%lower_bound), field)
        call require(field - origin == ARRAY_DESCRIPTOR_DIM_OFFSET &
            + int(ARRAY_DIMENSION_BYTES, c_intptr_t), 'dim stride')
        field = transfer(c_loc(value%dim(ARRAY_DESCRIPTOR_MAX_RANK)%lower_bound), &
            field)
        call require(field - origin == ARRAY_DESCRIPTOR_DIM_OFFSET &
            + 6_c_intptr_t*int(ARRAY_DIMENSION_BYTES, c_intptr_t), 'dim seven')
        call require(c_sizeof(value) == ARRAY_DESCRIPTOR_BYTES, 'total size')
    end subroutine require_descriptor_layout

    subroutine require_dimension_layout(value)
        type(array_dimension_t), target, intent(inout) :: value
        integer(c_intptr_t) :: origin
        integer(c_intptr_t) :: field

        origin = transfer(c_loc(value), origin)
        field = transfer(c_loc(value%lower_bound), field)
        call require(field - origin == ARRAY_DIMENSION_LOWER_OFFSET, &
            'dimension lower offset')
        field = transfer(c_loc(value%extent), field)
        call require(field - origin == ARRAY_DIMENSION_EXTENT_OFFSET, &
            'dimension extent offset')
        field = transfer(c_loc(value%stride_bytes), field)
        call require(field - origin == ARRAY_DIMENSION_STRIDE_OFFSET, &
            'dimension stride offset')
        call require(c_sizeof(value) == ARRAY_DIMENSION_BYTES, 'dimension size')
    end subroutine require_dimension_layout

    subroutine require_null_state(value, label)
        type(array_descriptor_t), intent(in) :: value
        character(len=*), intent(in) :: label
        integer :: dim

        call require(.not. c_associated(value%base), label//' base')
        call require(value%element_size == 0_c_int64_t, label//' element size')
        call require(value%element_type == ARRAY_ELEMENT_NONE, label//' type')
        call require(value%rank == 0_c_int32_t, label//' rank')
        call require(value%flags == ARRAY_FLAG_NONE, label//' flags')
        do dim = 1, ARRAY_DESCRIPTOR_MAX_RANK
            call require(value%dim(dim)%extent == 0_c_int64_t, label//' extent')
            call require(value%dim(dim)%stride_bytes == 0_c_int64_t, &
                label//' stride')
        end do
    end subroutine require_null_state

    subroutine require(condition, message)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: message

        if (.not. condition) then
            print *, 'FAIL: ', message
            stop 1
        end if
    end subroutine require

end program test_array_descriptor_layout
