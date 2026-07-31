module ffc_polymorphic_descriptor
    ! Canonical scalar class descriptor (#400). One descriptor carries the data
    ! address, the declared type identity, the dynamic type identity, and the
    ! ownership of the data. Type identities are the dense per-link-unit type
    ! info ids documented in docs/RUNTIME_ABI.md; they are stable within one
    ! linked program and are only ever compared for equality.
    use, intrinsic :: iso_c_binding, only: c_associated, c_int32_t, c_int64_t, &
        c_intptr_t, c_null_ptr, c_ptr, c_size_t
    implicit none
    private

    public :: polymorphic_descriptor_t
    public :: set_polymorphic_descriptor_null
    public :: set_borrowed_polymorphic_descriptor
    public :: set_owned_polymorphic_descriptor
    public :: polymorphic_descriptor_declared_type
    public :: polymorphic_descriptor_dynamic_type
    public :: polymorphic_descriptor_is_owned
    public :: polymorphic_descriptor_is_extension
    public :: release_polymorphic_descriptor

    integer(c_intptr_t), parameter, public :: &
        POLYMORPHIC_DESCRIPTOR_DATA_OFFSET = 0_c_intptr_t
    integer(c_intptr_t), parameter, public :: &
        POLYMORPHIC_DESCRIPTOR_DECLARED_TYPE_OFFSET = 8_c_intptr_t
    integer(c_intptr_t), parameter, public :: &
        POLYMORPHIC_DESCRIPTOR_DYNAMIC_TYPE_OFFSET = 16_c_intptr_t
    integer(c_intptr_t), parameter, public :: &
        POLYMORPHIC_DESCRIPTOR_OWNERSHIP_OFFSET = 24_c_intptr_t
    integer(c_size_t), parameter, public :: &
        POLYMORPHIC_DESCRIPTOR_SIZE = 32_c_size_t

    ! Reserved identity: no type is associated with the descriptor.
    integer(c_int64_t), parameter, public :: POLYMORPHIC_TYPE_ID_NONE = 0_c_int64_t

    integer(c_int32_t), parameter, public :: POLYMORPHIC_OWNERSHIP_NONE = 0_c_int32_t
    integer(c_int32_t), parameter, public :: &
        POLYMORPHIC_OWNERSHIP_BORROWED = 1_c_int32_t
    integer(c_int32_t), parameter, public :: &
        POLYMORPHIC_OWNERSHIP_OWNED = 2_c_int32_t

    integer, parameter, public :: POLYMORPHIC_DESCRIPTOR_OK = 0
    integer, parameter, public :: POLYMORPHIC_DESCRIPTOR_NULL_DATA = 1
    integer, parameter, public :: POLYMORPHIC_DESCRIPTOR_INVALID_DECLARED_TYPE = 2
    integer, parameter, public :: POLYMORPHIC_DESCRIPTOR_INVALID_DYNAMIC_TYPE = 3

    type, bind(c) :: polymorphic_descriptor_t
        type(c_ptr) :: data = c_null_ptr
        integer(c_int64_t) :: declared_type = POLYMORPHIC_TYPE_ID_NONE
        integer(c_int64_t) :: dynamic_type = POLYMORPHIC_TYPE_ID_NONE
        integer(c_int32_t) :: ownership = POLYMORPHIC_OWNERSHIP_NONE
    end type polymorphic_descriptor_t

contains

    subroutine set_polymorphic_descriptor_null(descriptor)
        type(polymorphic_descriptor_t), intent(out) :: descriptor

        descriptor%data = c_null_ptr
        descriptor%declared_type = POLYMORPHIC_TYPE_ID_NONE
        descriptor%dynamic_type = POLYMORPHIC_TYPE_ID_NONE
        descriptor%ownership = POLYMORPHIC_OWNERSHIP_NONE
    end subroutine set_polymorphic_descriptor_null

    subroutine set_borrowed_polymorphic_descriptor(descriptor, data, &
            declared_type, dynamic_type, status)
        ! A class dummy borrows its actual argument's storage: the callee never
        ! frees it and its lifetime is the caller's.
        type(polymorphic_descriptor_t), intent(out) :: descriptor
        type(c_ptr), intent(in) :: data
        integer(c_int64_t), intent(in) :: declared_type
        integer(c_int64_t), intent(in) :: dynamic_type
        integer, intent(out) :: status

        call set_associated_descriptor(descriptor, data, declared_type, &
            dynamic_type, POLYMORPHIC_OWNERSHIP_BORROWED, &
            status)
    end subroutine set_borrowed_polymorphic_descriptor

    subroutine set_owned_polymorphic_descriptor(descriptor, data, &
            declared_type, dynamic_type, status)
        ! An allocatable class value owns its storage: releasing the descriptor
        ! hands the data address back exactly once, to be freed by the owner.
        type(polymorphic_descriptor_t), intent(out) :: descriptor
        type(c_ptr), intent(in) :: data
        integer(c_int64_t), intent(in) :: declared_type
        integer(c_int64_t), intent(in) :: dynamic_type
        integer, intent(out) :: status

        call set_associated_descriptor(descriptor, data, declared_type, &
            dynamic_type, POLYMORPHIC_OWNERSHIP_OWNED, &
            status)
    end subroutine set_owned_polymorphic_descriptor

    subroutine set_associated_descriptor(descriptor, data, declared_type, &
            dynamic_type, ownership, status)
        type(polymorphic_descriptor_t), intent(out) :: descriptor
        type(c_ptr), intent(in) :: data
        integer(c_int64_t), intent(in) :: declared_type
        integer(c_int64_t), intent(in) :: dynamic_type
        integer(c_int32_t), intent(in) :: ownership
        integer, intent(out) :: status

        call set_polymorphic_descriptor_null(descriptor)
        if (declared_type <= POLYMORPHIC_TYPE_ID_NONE) then
            status = POLYMORPHIC_DESCRIPTOR_INVALID_DECLARED_TYPE
            return
        end if
        if (dynamic_type <= POLYMORPHIC_TYPE_ID_NONE) then
            status = POLYMORPHIC_DESCRIPTOR_INVALID_DYNAMIC_TYPE
            return
        end if
        ! An associated dynamic type without storage is not a class value.
        if (.not. c_associated(data)) then
            status = POLYMORPHIC_DESCRIPTOR_NULL_DATA
            return
        end if

        descriptor%data = data
        descriptor%declared_type = declared_type
        descriptor%dynamic_type = dynamic_type
        descriptor%ownership = ownership
        status = POLYMORPHIC_DESCRIPTOR_OK
    end subroutine set_associated_descriptor

    pure function polymorphic_descriptor_declared_type(descriptor) result(type_id)
        type(polymorphic_descriptor_t), intent(in) :: descriptor
        integer(c_int64_t) :: type_id

        type_id = descriptor%declared_type
    end function polymorphic_descriptor_declared_type

    pure function polymorphic_descriptor_dynamic_type(descriptor) result(type_id)
        type(polymorphic_descriptor_t), intent(in) :: descriptor
        integer(c_int64_t) :: type_id

        type_id = descriptor%dynamic_type
    end function polymorphic_descriptor_dynamic_type

    pure function polymorphic_descriptor_is_owned(descriptor) result(is_owned)
        type(polymorphic_descriptor_t), intent(in) :: descriptor
        logical :: is_owned

        is_owned = descriptor%ownership == POLYMORPHIC_OWNERSHIP_OWNED
    end function polymorphic_descriptor_is_owned

    pure function polymorphic_descriptor_is_extension(descriptor) result(is_ext)
        ! True when the value's dynamic type is an extension of its declared
        ! type, i.e. the two recorded identities differ.
        type(polymorphic_descriptor_t), intent(in) :: descriptor
        logical :: is_ext

        is_ext = .false.
        if (descriptor%declared_type == POLYMORPHIC_TYPE_ID_NONE) return
        if (descriptor%dynamic_type == POLYMORPHIC_TYPE_ID_NONE) return
        is_ext = descriptor%declared_type /= descriptor%dynamic_type
    end function polymorphic_descriptor_is_extension

    function release_polymorphic_descriptor(descriptor) result(data)
        ! Return the address the caller must free and reset the descriptor, so
        ! ownership is transferred exactly once and a borrowed or already
        ! released descriptor yields a null address.
        type(polymorphic_descriptor_t), intent(inout) :: descriptor
        type(c_ptr) :: data

        data = c_null_ptr
        if (polymorphic_descriptor_is_owned(descriptor)) data = descriptor%data
        call set_polymorphic_descriptor_null(descriptor)
    end function release_polymorphic_descriptor

end module ffc_polymorphic_descriptor
