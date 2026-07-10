module ffc_character_descriptor
    use, intrinsic :: iso_c_binding, only: c_associated, c_int32_t, c_int64_t, &
        c_intptr_t, c_null_ptr, c_ptr, c_size_t
    implicit none
    private

    public :: character_descriptor_t
    public :: set_character_descriptor_null
    public :: set_borrowed_character_descriptor
    public :: set_owned_character_descriptor
    public :: character_descriptor_length
    public :: character_descriptor_capacity
    public :: character_descriptor_is_owned
    public :: release_character_descriptor

    integer(c_intptr_t), parameter, public :: &
        CHARACTER_DESCRIPTOR_DATA_OFFSET = 0_c_intptr_t
    integer(c_intptr_t), parameter, public :: &
        CHARACTER_DESCRIPTOR_LENGTH_OFFSET = 8_c_intptr_t
    integer(c_intptr_t), parameter, public :: &
        CHARACTER_DESCRIPTOR_CAPACITY_OFFSET = 16_c_intptr_t
    integer(c_intptr_t), parameter, public :: &
        CHARACTER_DESCRIPTOR_STORAGE_OFFSET = 24_c_intptr_t
    integer(c_size_t), parameter, public :: CHARACTER_DESCRIPTOR_SIZE = 32_c_size_t

    integer(c_int32_t), parameter, public :: CHARACTER_STORAGE_NULL = 0_c_int32_t
    integer(c_int32_t), parameter, public :: CHARACTER_STORAGE_STATIC = 1_c_int32_t
    integer(c_int32_t), parameter, public :: CHARACTER_STORAGE_STACK = 2_c_int32_t
    integer(c_int32_t), parameter, public :: CHARACTER_STORAGE_OWNED = 3_c_int32_t

    integer, parameter, public :: CHARACTER_DESCRIPTOR_OK = 0
    integer, parameter, public :: CHARACTER_DESCRIPTOR_NEGATIVE_LENGTH = 1
    integer, parameter, public :: CHARACTER_DESCRIPTOR_INVALID_CAPACITY = 2
    integer, parameter, public :: CHARACTER_DESCRIPTOR_INVALID_STORAGE = 3
    integer, parameter, public :: CHARACTER_DESCRIPTOR_NULL_DATA = 4

    type, bind(c) :: character_descriptor_t
        type(c_ptr) :: data = c_null_ptr
        integer(c_int64_t) :: length = 0_c_int64_t
        integer(c_int64_t) :: capacity = 0_c_int64_t
        integer(c_int32_t) :: storage_class = CHARACTER_STORAGE_NULL
    end type character_descriptor_t

contains

    subroutine set_character_descriptor_null(descriptor)
        type(character_descriptor_t), intent(out) :: descriptor

        descriptor%data = c_null_ptr
        descriptor%length = 0_c_int64_t
        descriptor%capacity = 0_c_int64_t
        descriptor%storage_class = CHARACTER_STORAGE_NULL
    end subroutine set_character_descriptor_null

    subroutine set_borrowed_character_descriptor(descriptor, data, length, &
            storage_class, status)
        type(character_descriptor_t), intent(out) :: descriptor
        type(c_ptr), intent(in) :: data
        integer(c_int64_t), intent(in) :: length
        integer(c_int32_t), intent(in) :: storage_class
        integer, intent(out) :: status

        call set_character_descriptor_null(descriptor)
        if (length < 0_c_int64_t) then
            status = CHARACTER_DESCRIPTOR_NEGATIVE_LENGTH
            return
        end if
        if (storage_class /= CHARACTER_STORAGE_STATIC .and. &
            storage_class /= CHARACTER_STORAGE_STACK) then
            status = CHARACTER_DESCRIPTOR_INVALID_STORAGE
            return
        end if
        if (length > 0_c_int64_t) then
            if (.not. c_associated(data)) then
                status = CHARACTER_DESCRIPTOR_NULL_DATA
                return
            end if
        end if

        descriptor%data = data
        descriptor%length = length
        descriptor%capacity = length
        descriptor%storage_class = storage_class
        status = CHARACTER_DESCRIPTOR_OK
    end subroutine set_borrowed_character_descriptor

    subroutine set_owned_character_descriptor(descriptor, data, length, capacity, &
            status)
        type(character_descriptor_t), intent(out) :: descriptor
        type(c_ptr), intent(in) :: data
        integer(c_int64_t), intent(in) :: length
        integer(c_int64_t), intent(in) :: capacity
        integer, intent(out) :: status

        call set_character_descriptor_null(descriptor)
        if (length < 0_c_int64_t) then
            status = CHARACTER_DESCRIPTOR_NEGATIVE_LENGTH
            return
        end if
        if (capacity < 0_c_int64_t .or. capacity < length) then
            status = CHARACTER_DESCRIPTOR_INVALID_CAPACITY
            return
        end if
        if (capacity > 0_c_int64_t) then
            if (.not. c_associated(data)) then
                status = CHARACTER_DESCRIPTOR_NULL_DATA
                return
            end if
        end if

        descriptor%data = data
        descriptor%length = length
        descriptor%capacity = capacity
        descriptor%storage_class = CHARACTER_STORAGE_OWNED
        status = CHARACTER_DESCRIPTOR_OK
    end subroutine set_owned_character_descriptor

    pure function character_descriptor_length(descriptor) result(length)
        type(character_descriptor_t), intent(in) :: descriptor
        integer(c_int64_t) :: length

        length = descriptor%length
    end function character_descriptor_length

    pure function character_descriptor_capacity(descriptor) result(capacity)
        type(character_descriptor_t), intent(in) :: descriptor
        integer(c_int64_t) :: capacity

        capacity = descriptor%capacity
    end function character_descriptor_capacity

    pure function character_descriptor_is_owned(descriptor) result(is_owned)
        type(character_descriptor_t), intent(in) :: descriptor
        logical :: is_owned

        is_owned = descriptor%storage_class == CHARACTER_STORAGE_OWNED
    end function character_descriptor_is_owned

    function release_character_descriptor(descriptor) result(data)
        type(character_descriptor_t), intent(inout) :: descriptor
        type(c_ptr) :: data

        data = c_null_ptr
        if (character_descriptor_is_owned(descriptor)) data = descriptor%data
        call set_character_descriptor_null(descriptor)
    end function release_character_descriptor

end module ffc_character_descriptor
