module session_lowering_ops
    use, intrinsic :: iso_c_binding, only: c_int, c_int64_t
    use liric_session_bindings, only: LR_OP_ADD, LR_OP_MUL, LR_OP_SDIV, &
                                      LR_OP_SUB
    use liric_session_control_bindings, only: LR_CMP_EQ, LR_CMP_NE, &
                                              LR_CMP_SGE, LR_CMP_SGT, &
                                              LR_CMP_SLE, LR_CMP_SLT
    implicit none
    private

    public :: integer_compare_predicate
    public :: integer_opcode
    public :: parse_i32_literal

contains

    subroutine parse_i32_literal(text, value, error_msg)
        character(len=*), intent(in) :: text
        integer(c_int64_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: io_status

        read (text, *, iostat=io_status) value
        if (io_status == 0) then
            call set_empty(error_msg)
        else
            error_msg = 'invalid integer literal for direct LIRIC lowering: '// &
                        trim(text)
        end if
    end subroutine parse_i32_literal

    subroutine integer_opcode(source_op, opcode, error_msg)
        character(len=*), intent(in) :: source_op
        integer, intent(out) :: opcode
        character(len=:), allocatable, intent(out) :: error_msg

        call set_empty(error_msg)
        select case (trim(source_op))
        case ('+')
            opcode = LR_OP_ADD
        case ('-')
            opcode = LR_OP_SUB
        case ('*')
            opcode = LR_OP_MUL
        case ('/')
            opcode = LR_OP_SDIV
        case default
            error_msg = 'direct LIRIC session MVP does not support operator: '// &
                        trim(source_op)
            opcode = 0
        end select
    end subroutine integer_opcode

    subroutine integer_compare_predicate(source_op, predicate, error_msg)
        character(len=*), intent(in) :: source_op
        integer(c_int), intent(out) :: predicate
        character(len=:), allocatable, intent(out) :: error_msg

        call set_empty(error_msg)
        select case (trim(source_op))
        case ('==', '=')
            predicate = LR_CMP_EQ
        case ('/=', '!=')
            predicate = LR_CMP_NE
        case ('>')
            predicate = LR_CMP_SGT
        case ('>=')
            predicate = LR_CMP_SGE
        case ('<')
            predicate = LR_CMP_SLT
        case ('<=')
            predicate = LR_CMP_SLE
        case default
            error_msg = 'direct LIRIC session MVP does not support comparison: '// &
                        trim(source_op)
            predicate = LR_CMP_EQ
        end select
    end subroutine integer_compare_predicate

    subroutine set_empty(value)
        character(len=:), allocatable, intent(out) :: value

        allocate (character(len=0) :: value)
    end subroutine set_empty

end module session_lowering_ops
