    ! I/O character-specifier type checks. A character-valued I/O specifier
    ! (STATUS=, ACCESS=, ADVANCE=, ...) must be given a CHARACTER value; a
    ! numeric or logical literal is invalid Fortran. Only values whose leading
    ! token is a numeric or logical literal are rejected, so identifiers and
    ! quoted strings (which can never be a numeric literal) are never flagged.
    subroutine check_io_spec_types(spec, context, error_msg)
        character(len=*), intent(in) :: spec
        type(lowering_context_t), intent(in) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: text
        integer :: n, p, arg_start, arg_end

        call set_empty(error_msg)
        text = trim(adjustl(spec))
        n = len(text)
        if (n == 0) return
        ! Position-family statements (REWIND/BACKSPACE/ENDFILE) keep the whole
        ! parenthesised specifier list; strip a matched outer paren pair so the
        ! inner comma-separated specifiers tokenise the same as OPEN/CLOSE.
        if (n >= 2) then
            if (text(1:1) == '(' .and. text(n:n) == ')') then
                text = trim(adjustl(text(2:n - 1)))
                n = len(text)
                if (n == 0) return
            end if
        end if
        p = 1
        do while (p <= n)
            call next_io_spec_arg(text, n, p, arg_start, arg_end)
            if (arg_start == 0) exit
            call check_io_spec_arg(text(arg_start:arg_end), context, error_msg)
            if (len_trim(error_msg) > 0) return
        end do
    end subroutine check_io_spec_types

    ! Advance p to the next comma-separated top-level argument, returning its
    ! [arg_start, arg_end] bounds. Commas inside quotes or (), [] nesting do not
    ! split arguments. arg_start is 0 when no further argument exists.
    subroutine next_io_spec_arg(spec, n, p, arg_start, arg_end)
        character(len=*), intent(in) :: spec
        integer, intent(in) :: n
        integer, intent(inout) :: p
        integer, intent(out) :: arg_start, arg_end
        integer :: depth
        character :: qc

        arg_start = 0
        arg_end = 0
        do while (p <= n)
            if (spec(p:p) == ' ' .or. spec(p:p) == ',') then
                p = p + 1
            else
                exit
            end if
        end do
        if (p > n) return
        arg_start = p
        depth = 0
        do while (p <= n)
            if (spec(p:p) == "'" .or. spec(p:p) == '"') then
                qc = spec(p:p)
                p = p + 1
                do while (p <= n)
                    if (spec(p:p) == qc) exit
                    p = p + 1
                end do
            else if (spec(p:p) == '(' .or. spec(p:p) == '[') then
                depth = depth + 1
            else if (spec(p:p) == ')' .or. spec(p:p) == ']') then
                if (depth > 0) depth = depth - 1
            else if (depth == 0) then
                if (spec(p:p) == ',') exit
            end if
            p = p + 1
        end do
        arg_end = p - 1
        if (p <= n) p = p + 1
    end subroutine next_io_spec_arg

    subroutine check_io_spec_arg(arg, context, error_msg)
        character(len=*), intent(in) :: arg
        type(lowering_context_t), intent(in) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: eqp
        character(len=:), allocatable :: kw, val

        call set_empty(error_msg)
        eqp = index(arg, '=')
        if (eqp <= 1) return
        kw = trim(adjustl(spec_lower(arg(1:eqp - 1))))
        if (.not. is_character_io_spec(kw)) return
        val = arg(eqp + 1:)
        if (noncharacter_io_value(val)) then
            error_msg = io_spec_upper(kw)//' specifier must be of type CHARACTER'
            return
        end if
        if (noncharacter_io_identifier(val, context)) then
            error_msg = io_spec_upper(kw)//' specifier must be of type CHARACTER'
        end if
    end subroutine check_io_spec_arg

    logical function is_character_io_spec(kw) result(is_char)
        character(len=*), intent(in) :: kw

        select case (trim(kw))
        case ('access', 'action', 'advance', 'asynchronous', 'blank', &
              'convert', 'decimal', 'delim', 'dispose', 'encoding', 'form', &
              'pad', 'position', 'round', 'sign', 'status', 'iomsg', &
              'carriagecontrol')
            is_char = .true.
        case default
            is_char = .false.
        end select
    end function is_character_io_spec

    ! True when val is a numeric or logical literal, i.e. a value that can never
    ! be of type CHARACTER. Leading digit, signed number, .digit real, or a
    ! logical literal all qualify; anything else (identifier, quoted string,
    ! null(), array/complex constructor) is left for other checks.
    logical function noncharacter_io_value(val) result(bad)
        character(len=*), intent(in) :: val
        character(len=:), allocatable :: t
        character :: c

        bad = .false.
        t = trim(adjustl(val))
        if (len(t) == 0) return
        if (spec_lower(t) == '.true.' .or. spec_lower(t) == '.false.') then
            bad = .true.
            return
        end if
        c = t(1:1)
        if (c >= '0' .and. c <= '9') then
            bad = .true.
            return
        end if
        if (len(t) < 2) return
        if ((c == '+' .or. c == '-') .and. t(2:2) >= '0' .and. t(2:2) <= '9') &
            bad = .true.
        if ((c == '+' .or. c == '-') .and. t(2:2) == '.') bad = .true.
        if (c == '.' .and. t(2:2) >= '0' .and. t(2:2) <= '9') bad = .true.
    end function noncharacter_io_value

    ! True when val is a bare identifier that resolves to a numeric or logical
    ! symbol, which can never be a CHARACTER I/O specifier value. Non-identifier
    ! text, an unknown name, or a character/derived symbol are all left alone.
    logical function noncharacter_io_identifier(val, context) result(bad)
        character(len=*), intent(in) :: val
        type(lowering_context_t), intent(in) :: context
        character(len=:), allocatable :: t
        integer :: i, sym
        character :: c

        bad = .false.
        t = trim(adjustl(val))
        if (len(t) == 0) return
        c = t(1:1)
        if (.not. ((c >= 'a' .and. c <= 'z') .or. (c >= 'A' .and. c <= 'Z'))) &
            return
        do i = 2, len(t)
            c = t(i:i)
            if (c >= 'a' .and. c <= 'z') cycle
            if (c >= 'A' .and. c <= 'Z') cycle
            if (c >= '0' .and. c <= '9') cycle
            if (c == '_') cycle
            return
        end do
        sym = find_symbol(context, t)
        if (sym <= 0) return
        select case (context%symbols(sym)%value_kind)
        case (VALUE_I32, VALUE_I64, VALUE_I8, VALUE_I16, VALUE_F32, VALUE_F64, &
              VALUE_LOGICAL, VALUE_C4, VALUE_C8)
            bad = .true.
        end select
    end function noncharacter_io_identifier

    function io_spec_upper(s) result(t)
        character(len=*), intent(in) :: s
        character(len=len(s)) :: t
        integer :: i, c

        do i = 1, len(s)
            c = ichar(s(i:i))
            if (c >= ichar('a') .and. c <= ichar('z')) then
                t(i:i) = char(c - 32)
            else
                t(i:i) = s(i:i)
            end if
        end do
    end function io_spec_upper
