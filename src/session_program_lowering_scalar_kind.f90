module session_program_lowering_scalar_kind
    use session_program_lowering_types, only: SCALAR_REAL_NONE, VALUE_F32, VALUE_F64
    implicit none
    private
    public :: real_value_kind_of, wider_real_kind, real_kind_from_kind_number

contains

    integer function real_value_kind_of(kind_code) result(vk)
        !! Narrow an arbitrary VALUE_* code to the real kinds this engine reports.
        integer, intent(in) :: kind_code

        if (kind_code == VALUE_F32 .or. kind_code == VALUE_F64) then
            vk = kind_code
        else
            vk = SCALAR_REAL_NONE
        end if
    end function real_value_kind_of

    integer function wider_real_kind(left_kind, right_kind) result(vk)
        !! Combine two operand kinds using the widest real operand.
        integer, intent(in) :: left_kind, right_kind

        if (left_kind == VALUE_F64 .or. right_kind == VALUE_F64) then
            vk = VALUE_F64
        else if (left_kind == VALUE_F32 .or. right_kind == VALUE_F32) then
            vk = VALUE_F32
        else
            vk = SCALAR_REAL_NONE
        end if
    end function wider_real_kind

    integer function real_kind_from_kind_number(kind_number) result(vk)
        !! Map a resolved numeric KIND selector (4 or 8) onto a value kind.
        integer, intent(in) :: kind_number

        select case (kind_number)
        case (4)
            vk = VALUE_F32
        case (8)
            vk = VALUE_F64
        case default
            vk = SCALAR_REAL_NONE
        end select
    end function real_kind_from_kind_number

end module session_program_lowering_scalar_kind
