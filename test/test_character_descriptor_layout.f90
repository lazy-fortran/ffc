program test_character_descriptor_layout
    use ffc_character_descriptor, only: character_descriptor_t, &
        CHARACTER_DESCRIPTOR_DATA_OFFSET, CHARACTER_DESCRIPTOR_LENGTH_OFFSET, &
        CHARACTER_DESCRIPTOR_CAPACITY_OFFSET, &
        CHARACTER_DESCRIPTOR_STORAGE_OFFSET, CHARACTER_DESCRIPTOR_SIZE, &
        CHARACTER_STORAGE_NULL, CHARACTER_STORAGE_STATIC, &
        CHARACTER_STORAGE_STACK, CHARACTER_STORAGE_OWNED, &
        CHARACTER_DESCRIPTOR_OK, CHARACTER_DESCRIPTOR_NEGATIVE_LENGTH, &
        CHARACTER_DESCRIPTOR_INVALID_CAPACITY, &
        CHARACTER_DESCRIPTOR_INVALID_STORAGE, CHARACTER_DESCRIPTOR_NULL_DATA, &
        set_character_descriptor_null, set_borrowed_character_descriptor, &
        set_owned_character_descriptor, character_descriptor_length, &
        character_descriptor_capacity, character_descriptor_is_owned, &
        release_character_descriptor
    use, intrinsic :: iso_c_binding, only: c_associated, c_char, c_int64_t, &
        c_intptr_t, c_loc, c_null_ptr, c_ptr, c_sizeof
    implicit none

    type(character_descriptor_t), target :: descriptor
    character(kind=c_char), target :: borrowed_data(3)
    character(kind=c_char), target :: owned_data(8)
    type(c_ptr) :: released
    integer(c_intptr_t) :: base_address
    integer :: status

    call set_character_descriptor_null(descriptor)
    call require_layout(descriptor)
    call require_null_state(descriptor, 'initial null')

    call set_borrowed_character_descriptor(descriptor, c_loc(borrowed_data), &
        3_c_int64_t, CHARACTER_STORAGE_STATIC, status)
    call require(status == CHARACTER_DESCRIPTOR_OK, 'borrowed status')
    call require(character_descriptor_length(descriptor) == 3_c_int64_t, &
        'borrowed length')
    call require(character_descriptor_capacity(descriptor) == 3_c_int64_t, &
        'borrowed capacity')
    call require(.not. character_descriptor_is_owned(descriptor), &
        'borrowed ownership')
    released = release_character_descriptor(descriptor)
    call require(.not. c_associated(released), 'borrowed release')
    call require_null_state(descriptor, 'borrowed release')

    call set_borrowed_character_descriptor(descriptor, c_loc(borrowed_data), &
        3_c_int64_t, CHARACTER_STORAGE_STACK, status)
    call require(status == CHARACTER_DESCRIPTOR_OK, 'stack status')
    released = release_character_descriptor(descriptor)
    call require(.not. c_associated(released), 'stack release')
    call require_null_state(descriptor, 'stack release')

    call set_owned_character_descriptor(descriptor, c_loc(owned_data), &
        5_c_int64_t, 8_c_int64_t, status)
    call require(status == CHARACTER_DESCRIPTOR_OK, 'owned status')
    call require(character_descriptor_length(descriptor) == 5_c_int64_t, &
        'owned length')
    call require(character_descriptor_capacity(descriptor) == 8_c_int64_t, &
        'owned capacity')
    call require(character_descriptor_is_owned(descriptor), 'owned ownership')
    released = release_character_descriptor(descriptor)
    call require(c_associated(released, c_loc(owned_data)), 'owned release')
    call require_null_state(descriptor, 'owned release')

    call set_borrowed_character_descriptor(descriptor, c_loc(borrowed_data), &
        -1_c_int64_t, CHARACTER_STORAGE_STATIC, status)
    call require(status == CHARACTER_DESCRIPTOR_NEGATIVE_LENGTH, &
        'negative length status')
    call require_null_state(descriptor, 'negative length')

    call set_borrowed_character_descriptor(descriptor, c_loc(borrowed_data), &
        3_c_int64_t, CHARACTER_STORAGE_OWNED, status)
    call require(status == CHARACTER_DESCRIPTOR_INVALID_STORAGE, &
        'invalid borrowed storage status')
    call require_null_state(descriptor, 'invalid borrowed storage')

    call set_borrowed_character_descriptor(descriptor, c_null_ptr, &
        1_c_int64_t, CHARACTER_STORAGE_STATIC, status)
    call require(status == CHARACTER_DESCRIPTOR_NULL_DATA, &
        'borrowed null data status')
    call require_null_state(descriptor, 'borrowed null data')

    call set_owned_character_descriptor(descriptor, c_loc(owned_data), &
        -1_c_int64_t, 0_c_int64_t, status)
    call require(status == CHARACTER_DESCRIPTOR_NEGATIVE_LENGTH, &
        'owned negative length status')
    call require_null_state(descriptor, 'owned negative length')

    call set_owned_character_descriptor(descriptor, c_loc(owned_data), &
        0_c_int64_t, -1_c_int64_t, status)
    call require(status == CHARACTER_DESCRIPTOR_INVALID_CAPACITY, &
        'negative capacity status')
    call require_null_state(descriptor, 'negative capacity')

    call set_owned_character_descriptor(descriptor, c_loc(owned_data), &
        5_c_int64_t, 4_c_int64_t, status)
    call require(status == CHARACTER_DESCRIPTOR_INVALID_CAPACITY, &
        'invalid capacity status')
    call require_null_state(descriptor, 'invalid capacity')

    call set_owned_character_descriptor(descriptor, c_null_ptr, 1_c_int64_t, &
        1_c_int64_t, status)
    call require(status == CHARACTER_DESCRIPTOR_NULL_DATA, 'null data status')
    call require_null_state(descriptor, 'null data')

    print *, 'PASS: character descriptor layout and ownership'

contains

    subroutine require_layout(value)
        type(character_descriptor_t), target, intent(inout) :: value
        integer(c_intptr_t) :: address

        base_address = transfer(c_loc(value), base_address)
        address = transfer(c_loc(value%data), address)
        call require(address - base_address == CHARACTER_DESCRIPTOR_DATA_OFFSET, &
            'data offset')
        address = transfer(c_loc(value%length), address)
        call require(address - base_address == CHARACTER_DESCRIPTOR_LENGTH_OFFSET, &
            'length offset')
        address = transfer(c_loc(value%capacity), address)
        call require(address - base_address == &
            CHARACTER_DESCRIPTOR_CAPACITY_OFFSET, 'capacity offset')
        address = transfer(c_loc(value%storage_class), address)
        call require(address - base_address == CHARACTER_DESCRIPTOR_STORAGE_OFFSET, &
            'storage offset')
        call require(c_sizeof(value) == CHARACTER_DESCRIPTOR_SIZE, 'total size')
    end subroutine require_layout

    subroutine require_null_state(value, label)
        type(character_descriptor_t), intent(in) :: value
        character(len=*), intent(in) :: label

        call require(.not. c_associated(value%data), label//' data')
        call require(value%length == 0_c_int64_t, label//' length')
        call require(value%capacity == 0_c_int64_t, label//' capacity')
        call require(value%storage_class == CHARACTER_STORAGE_NULL, &
            label//' storage class')
    end subroutine require_null_state

    subroutine require(condition, message)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: message

        if (.not. condition) then
            print *, 'FAIL: ', message
            stop 1
        end if
    end subroutine require

end program test_character_descriptor_layout
