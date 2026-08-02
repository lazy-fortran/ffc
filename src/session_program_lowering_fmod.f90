! Compile-safe portion of the FMod lowering support.
!
! Token operations live here as ordinary module procedures. The host-associated
! exporter itself is implemented by the sibling session_program_lowering
! submodule, which retains access to the lowerer's private AST helpers.
module session_program_lowering_fmod
    use session_program_lowering_types, only: VALUE_I32, VALUE_I64, VALUE_I8, &
        VALUE_I16, VALUE_F32, VALUE_F64, VALUE_LOGICAL, VALUE_CHARACTER, &
        VALUE_C4, VALUE_C8
    implicit none
    private
    public :: integer_token, scalar_kind_token, value_kind_of_token

contains

    function integer_token(value) result(token)
        integer, intent(in) :: value
        character(len=:), allocatable :: token
        character(len=32) :: buffer

        write (buffer, '(I0)') value
        token = trim(buffer)
    end function integer_token

    function scalar_kind_token(value_kind) result(token)
        ! The .fmod token for a by-reference scalar dummy kind, or '' when the
        ! kind is not exportable across separate compilation (#284).
        integer, intent(in) :: value_kind
        character(len=:), allocatable :: token

        select case (value_kind)
        case (VALUE_I32)
            token = 'integer'
        case (VALUE_I64)
            token = 'integer8'
        case (VALUE_I8)
            token = 'integer1'
        case (VALUE_I16)
            token = 'integer2'
        case (VALUE_F32)
            token = 'real'
        case (VALUE_F64)
            token = 'real8'
        case (VALUE_LOGICAL)
            token = 'logical'
        case (VALUE_CHARACTER)
            token = 'character'
        case (VALUE_C4)
            token = 'complex'
        case (VALUE_C8)
            token = 'complex8'
        case default
            token = ''
        end select
    end function scalar_kind_token

    integer function value_kind_of_token(token) result(value_kind)
        ! Inverse of scalar_kind_token: the value kind a .fmod scalar token
        ! denotes, or 0 when unrecognised (#284).
        character(len=*), intent(in) :: token

        select case (trim(token))
        case ('integer')
            value_kind = VALUE_I32
        case ('integer8')
            value_kind = VALUE_I64
        case ('integer1')
            value_kind = VALUE_I8
        case ('integer2')
            value_kind = VALUE_I16
        case ('real')
            value_kind = VALUE_F32
        case ('real8')
            value_kind = VALUE_F64
        case ('logical')
            value_kind = VALUE_LOGICAL
        case ('character')
            value_kind = VALUE_CHARACTER
        case ('complex')
            value_kind = VALUE_C4
        case ('complex8')
            value_kind = VALUE_C8
        case default
            value_kind = 0
        end select
    end function value_kind_of_token

end module session_program_lowering_fmod
