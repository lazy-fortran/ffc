module session_lowering_ops
    use, intrinsic :: iso_c_binding, only: c_int, c_int64_t
    use ffc_strings, only: set_empty
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

        ! BOZ constants (DATA initializers, #2349) carry a base prefix or
        ! postfix that list-directed read cannot parse; decode them by radix.
        call parse_boz_literal(text, value, io_status)
        if (io_status == 0) then
            call set_empty(error_msg)
            return
        end if

        ! A kind-suffixed literal (123_int32, 0_C_INT, 1_4) carries an
        ! underscore-delimited kind parameter that list-directed read cannot
        ! parse; the magnitude is the digit string before the underscore.
        call parse_kind_suffixed_integer(text, value, io_status)
        if (io_status == 0) then
            call set_empty(error_msg)
            return
        end if

        read (text, *, iostat=io_status) value
        if (io_status == 0) then
            call set_empty(error_msg)
        else
            error_msg = 'invalid integer literal for direct LIRIC lowering: '// &
                        trim(text)
        end if
    end subroutine parse_i32_literal

    subroutine parse_kind_suffixed_integer(text, value, status)
        !! Decode digit-string_kind into its integer value. status==0 on
        !! success; nonzero leaves value untouched so the caller falls back.
        character(len=*), intent(in) :: text
        integer(c_int64_t), intent(out) :: value
        integer, intent(out) :: status
        character(len=:), allocatable :: trimmed
        integer :: us

        status = 1
        value = 0_c_int64_t
        trimmed = trim(adjustl(text))
        us = index(trimmed, '_')
        if (us <= 1) return
        read (trimmed(1:us - 1), *, iostat=status) value
    end subroutine parse_kind_suffixed_integer

    subroutine parse_boz_literal(text, value, status)
        !! Decode a BOZ constant into an integer value. Recognises the standard
        !! prefix forms B'..'/O'..'/Z'..', the gfortran-rejected X'..' hex
        !! alias, and the nonstandard postfix forms '..'B/'..'O/'..'Z/'..'X.
        !! status==0 on success; nonzero leaves the value untouched so the
        !! caller falls back to ordinary integer parsing.
        character(len=*), intent(in) :: text
        integer(c_int64_t), intent(out) :: value
        integer, intent(out) :: status
        character(len=:), allocatable :: trimmed, digits
        character(len=1) :: base
        integer :: radix, n, qopen, qclose

        status = 1
        value = 0_c_int64_t
        trimmed = trim(adjustl(text))
        n = len(trimmed)
        if (n < 3) return
        if (trimmed(1:1) /= "'" .and. trimmed(1:1) /= '"') then
            base = to_lower(trimmed(1:1))
        else
            base = to_lower(trimmed(n:n))
        end if
        radix = boz_radix(base)
        if (radix == 0) return
        qopen = index(trimmed, "'")
        if (qopen == 0) qopen = index(trimmed, '"')
        if (qopen == 0) return
        qclose = index(trimmed(qopen + 1:), trimmed(qopen:qopen))
        if (qclose == 0) return
        qclose = qopen + qclose
        if (qclose <= qopen + 1) return
        digits = trimmed(qopen + 1:qclose - 1)
        call decode_radix_digits(digits, radix, value, status)
    end subroutine parse_boz_literal

    integer function boz_radix(base)
        character(len=1), intent(in) :: base
        select case (base)
        case ('b')
            boz_radix = 2
        case ('o')
            boz_radix = 8
        case ('z', 'x')
            boz_radix = 16
        case default
            boz_radix = 0
        end select
    end function boz_radix

    subroutine decode_radix_digits(digits, radix, value, status)
        character(len=*), intent(in) :: digits
        integer, intent(in) :: radix
        integer(c_int64_t), intent(out) :: value
        integer, intent(out) :: status
        integer :: i, d
        character(len=1) :: c

        status = 1
        value = 0_c_int64_t
        if (len_trim(digits) == 0) return
        do i = 1, len(digits)
            c = to_lower(digits(i:i))
            if (c >= '0' .and. c <= '9') then
                d = iachar(c) - iachar('0')
            else if (c >= 'a' .and. c <= 'f') then
                d = iachar(c) - iachar('a') + 10
            else
                return
            end if
            if (d >= radix) return
            value = value*int(radix, c_int64_t) + int(d, c_int64_t)
        end do
        status = 0
    end subroutine decode_radix_digits

    pure function to_lower(c) result(lc)
        character(len=1), intent(in) :: c
        character(len=1) :: lc
        if (c >= 'A' .and. c <= 'Z') then
            lc = achar(iachar(c) + 32)
        else
            lc = c
        end if
    end function to_lower

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
            error_msg = 'ffc direct-session lowering does not support operator: '// &
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
            error_msg = 'ffc direct-session lowering does not support comparison: '// &
                        trim(source_op)
            predicate = LR_CMP_EQ
        end select
    end subroutine integer_compare_predicate

end module session_lowering_ops
