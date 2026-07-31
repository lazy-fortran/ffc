program test_polymorphic_descriptor_layout
    use ffc_polymorphic_descriptor, only: polymorphic_descriptor_t, &
        POLYMORPHIC_DESCRIPTOR_DATA_OFFSET, &
        POLYMORPHIC_DESCRIPTOR_DECLARED_TYPE_OFFSET, &
        POLYMORPHIC_DESCRIPTOR_DYNAMIC_TYPE_OFFSET, &
        POLYMORPHIC_DESCRIPTOR_OWNERSHIP_OFFSET, &
        POLYMORPHIC_DESCRIPTOR_SIZE, POLYMORPHIC_TYPE_ID_NONE, &
        POLYMORPHIC_OWNERSHIP_NONE, POLYMORPHIC_OWNERSHIP_BORROWED, &
        POLYMORPHIC_OWNERSHIP_OWNED, POLYMORPHIC_DESCRIPTOR_OK, &
        POLYMORPHIC_DESCRIPTOR_INVALID_DECLARED_TYPE, &
        POLYMORPHIC_DESCRIPTOR_INVALID_DYNAMIC_TYPE, &
        POLYMORPHIC_DESCRIPTOR_NULL_DATA, &
        set_polymorphic_descriptor_null, set_borrowed_polymorphic_descriptor, &
        set_owned_polymorphic_descriptor, polymorphic_descriptor_declared_type, &
        polymorphic_descriptor_dynamic_type, polymorphic_descriptor_is_owned, &
        polymorphic_descriptor_is_extension, release_polymorphic_descriptor
    use, intrinsic :: iso_c_binding, only: c_associated, c_int32_t, c_int64_t, &
        c_intptr_t, c_loc, c_null_ptr, c_ptr, c_sizeof
    implicit none

    integer(c_int64_t), parameter :: BASE_ID = 1_c_int64_t
    integer(c_int64_t), parameter :: EXTENSION_ID = 2_c_int64_t

    type(polymorphic_descriptor_t), target :: descriptor
    integer(c_int32_t), target :: base_storage(4)
    integer(c_int32_t), target :: extension_storage(8)
    type(c_ptr) :: released
    integer(c_intptr_t) :: base_address
    integer :: status

    call set_polymorphic_descriptor_null(descriptor)
    call require_layout(descriptor)
    call require_null_state(descriptor, 'initial null')

    ! A borrowed base value: declared and dynamic identity agree.
    call set_borrowed_polymorphic_descriptor(descriptor, c_loc(base_storage), &
        BASE_ID, BASE_ID, status)
    call require(status == POLYMORPHIC_DESCRIPTOR_OK, 'borrowed base status')
    call require(polymorphic_descriptor_declared_type(descriptor) == BASE_ID, &
        'borrowed base declared type')
    call require(polymorphic_descriptor_dynamic_type(descriptor) == BASE_ID, &
        'borrowed base dynamic type')
    call require(.not. polymorphic_descriptor_is_extension(descriptor), &
        'borrowed base is not an extension')
    call require(.not. polymorphic_descriptor_is_owned(descriptor), &
        'borrowed base ownership')
    released = release_polymorphic_descriptor(descriptor)
    call require(.not. c_associated(released), 'borrowed base release')
    call require_null_state(descriptor, 'borrowed base release')

    ! A borrowed extension value: only the dynamic identity differs.
    call set_borrowed_polymorphic_descriptor(descriptor, &
        c_loc(extension_storage), BASE_ID, EXTENSION_ID, status)
    call require(status == POLYMORPHIC_DESCRIPTOR_OK, 'borrowed extension status')
    call require(polymorphic_descriptor_declared_type(descriptor) == BASE_ID, &
        'borrowed extension declared type')
    call require(polymorphic_descriptor_dynamic_type(descriptor) == &
        EXTENSION_ID, 'borrowed extension dynamic type')
    call require(polymorphic_descriptor_is_extension(descriptor), &
        'borrowed extension is an extension')
    call require(.not. polymorphic_descriptor_is_owned(descriptor), &
        'borrowed extension ownership')
    released = release_polymorphic_descriptor(descriptor)
    call require(.not. c_associated(released), 'borrowed extension release')

    ! An owned extension value: release hands the data address back once.
    call set_owned_polymorphic_descriptor(descriptor, c_loc(extension_storage), &
        BASE_ID, EXTENSION_ID, status)
    call require(status == POLYMORPHIC_DESCRIPTOR_OK, 'owned status')
    call require(polymorphic_descriptor_is_owned(descriptor), 'owned ownership')
    call require(polymorphic_descriptor_is_extension(descriptor), &
        'owned is an extension')
    released = release_polymorphic_descriptor(descriptor)
    call require(c_associated(released, c_loc(extension_storage)), 'owned release')
    call require_null_state(descriptor, 'owned release')
    released = release_polymorphic_descriptor(descriptor)
    call require(.not. c_associated(released), 'released descriptor is inert')

    ! A null data address with an associated dynamic type is not a descriptor.
    call set_borrowed_polymorphic_descriptor(descriptor, c_null_ptr, BASE_ID, &
        BASE_ID, status)
    call require(status == POLYMORPHIC_DESCRIPTOR_NULL_DATA, &
        'borrowed null data status')
    call require_null_state(descriptor, 'borrowed null data')

    call set_owned_polymorphic_descriptor(descriptor, c_null_ptr, BASE_ID, &
        EXTENSION_ID, status)
    call require(status == POLYMORPHIC_DESCRIPTOR_NULL_DATA, &
        'owned null data status')
    call require_null_state(descriptor, 'owned null data')

    call set_borrowed_polymorphic_descriptor(descriptor, c_loc(base_storage), &
        POLYMORPHIC_TYPE_ID_NONE, BASE_ID, status)
    call require(status == POLYMORPHIC_DESCRIPTOR_INVALID_DECLARED_TYPE, &
        'missing declared type status')
    call require_null_state(descriptor, 'missing declared type')

    call set_borrowed_polymorphic_descriptor(descriptor, c_loc(base_storage), &
        BASE_ID, POLYMORPHIC_TYPE_ID_NONE, status)
    call require(status == POLYMORPHIC_DESCRIPTOR_INVALID_DYNAMIC_TYPE, &
        'missing dynamic type status')
    call require_null_state(descriptor, 'missing dynamic type')

    call set_owned_polymorphic_descriptor(descriptor, c_loc(base_storage), &
        -3_c_int64_t, BASE_ID, status)
    call require(status == POLYMORPHIC_DESCRIPTOR_INVALID_DECLARED_TYPE, &
        'negative declared type status')
    call require_null_state(descriptor, 'negative declared type')

    call set_owned_polymorphic_descriptor(descriptor, c_loc(base_storage), &
        BASE_ID, -3_c_int64_t, status)
    call require(status == POLYMORPHIC_DESCRIPTOR_INVALID_DYNAMIC_TYPE, &
        'negative dynamic type status')
    call require_null_state(descriptor, 'negative dynamic type')

    print *, 'PASS: polymorphic descriptor layout and identity'

contains

    subroutine require_layout(value)
        type(polymorphic_descriptor_t), target, intent(inout) :: value
        integer(c_intptr_t) :: address

        base_address = transfer(c_loc(value), base_address)
        address = transfer(c_loc(value%data), address)
        call require(address - base_address == &
            POLYMORPHIC_DESCRIPTOR_DATA_OFFSET, 'data offset')
        address = transfer(c_loc(value%declared_type), address)
        call require(address - base_address == &
            POLYMORPHIC_DESCRIPTOR_DECLARED_TYPE_OFFSET, 'declared type offset')
        address = transfer(c_loc(value%dynamic_type), address)
        call require(address - base_address == &
            POLYMORPHIC_DESCRIPTOR_DYNAMIC_TYPE_OFFSET, 'dynamic type offset')
        address = transfer(c_loc(value%ownership), address)
        call require(address - base_address == &
            POLYMORPHIC_DESCRIPTOR_OWNERSHIP_OFFSET, 'ownership offset')
        call require(c_sizeof(value) == POLYMORPHIC_DESCRIPTOR_SIZE, 'total size')
    end subroutine require_layout

    subroutine require_null_state(value, label)
        type(polymorphic_descriptor_t), intent(in) :: value
        character(len=*), intent(in) :: label

        call require(.not. c_associated(value%data), label//' data')
        call require(value%declared_type == POLYMORPHIC_TYPE_ID_NONE, &
            label//' declared type')
        call require(value%dynamic_type == POLYMORPHIC_TYPE_ID_NONE, &
            label//' dynamic type')
        call require(value%ownership == POLYMORPHIC_OWNERSHIP_NONE, &
            label//' ownership')
        call require(.not. polymorphic_descriptor_is_owned(value), &
            label//' not owned')
    end subroutine require_null_state

    subroutine require(condition, message)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: message

        if (.not. condition) then
            print *, 'FAIL: ', message
            stop 1
        end if
    end subroutine require

end program test_polymorphic_descriptor_layout
