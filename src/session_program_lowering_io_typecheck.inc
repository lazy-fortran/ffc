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
        sym = find_symbol_compat(context, t)
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

    ! I/O control-list constraints (#385). Several invalid control lists never
    ! reach the typed AST: the parser drops the statement and the program would
    ! otherwise compile silently. They are validated on the statement text, the
    ! earliest layer that still holds them, with quote-aware tokenising so no
    ! valid form is flagged.
    subroutine check_io_control_source(arena, error_msg)
        type(ast_arena_t), intent(in) :: arena
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: source, stmt
        logical :: found
        integer :: pos, line_no, stmt_line

        call set_empty(error_msg)
        call get_source_text(arena, source, found)
        if (.not. found) return
        pos = 1
        line_no = 0
        do while (pos <= len(source))
            call next_io_source_statement(source, pos, line_no, stmt, stmt_line)
            if (len_trim(stmt) == 0) cycle
            call check_io_statement_text(stmt, stmt_line, error_msg)
            if (len_trim(error_msg) > 0) return
        end do
    end subroutine check_io_control_source

    ! Take the next logical statement: one source line with its comment removed
    ! and any '&' continuations appended.
    subroutine next_io_source_statement(source, pos, line_no, stmt, stmt_line)
        character(len=*), intent(in) :: source
        integer, intent(inout) :: pos, line_no
        character(len=:), allocatable, intent(out) :: stmt
        integer, intent(out) :: stmt_line
        character(len=:), allocatable :: line
        integer :: next_nl, n

        call set_empty(stmt)
        stmt_line = line_no + 1
        do while (pos <= len(source))
            next_nl = index(source(pos:), new_line('a'))
            if (next_nl == 0) then
                line = source(pos:)
                pos = len(source) + 1
            else if (next_nl == 1) then
                line = ''
                pos = pos + 1
            else
                line = source(pos:pos + next_nl - 2)
                pos = pos + next_nl
            end if
            line_no = line_no + 1
            line = strip_data_source_comment(line)
            line = trim(adjustl(line))
            n = len(line)
            if (n > 0) then
                if (line(1:1) == '&') then
                    line = trim(adjustl(line(2:)))
                    n = len(line)
                end if
            end if
            if (n > 0) then
                if (line(n:n) == '&') then
                    stmt = stmt//' '//line(1:n - 1)
                    cycle
                end if
            end if
            stmt = stmt//' '//line
            return
        end do
    end subroutine next_io_source_statement

    subroutine check_io_statement_text(stmt, line_no, error_msg)
        character(len=*), intent(in) :: stmt
        integer, intent(in) :: line_no
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: code, low, kw, list, tail
        character(len=64) :: location
        integer :: open_p, close_p, n

        call set_empty(error_msg)
        code = trim(adjustl(stmt))
        n = len(code)
        if (n == 0) return
        write (location, '(" at line ",I0)') line_no
        if (io_paren_balance(code) /= 0) then
            error_msg = 'unbalanced parentheses in statement'//trim(location)
            return
        end if
        code = trim(adjustl(strip_statement_label(code)))
        n = len(code)
        if (n == 0) return
        low = lowercase_text(code)
        kw = leading_io_keyword(low)
        select case (kw)
        case ('print')
            if (code(n:n) == ',') then
                error_msg = 'output item list required after comma in '// &
                    'PRINT statement'//trim(location)
                return
            end if
            call check_io_output_items(code(len(kw) + 1:), location, error_msg)
        case ('write', 'read', 'open', 'inquire', 'close', 'rewind', &
              'backspace', 'endfile')
            call io_statement_paren_span(code, open_p, close_p)
            if (open_p == 0) return
            if (close_p <= open_p) return
            list = code(open_p + 1:close_p - 1)
            call check_io_control_list(list, kw, location, error_msg)
            if (len_trim(error_msg) > 0) return
            if (kw /= 'write') return
            tail = code(close_p + 1:)
            call check_io_output_items(tail, location, error_msg)
        end select
    end subroutine check_io_statement_text

    ! Apply every control-list constraint to one comma-separated specifier list.
    subroutine check_io_control_list(list, kw, location, error_msg)
        character(len=*), intent(in) :: list, kw, location
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: arg, name, val
        integer :: n, p, arg_start, arg_end, idx
        logical :: has_unit

        call set_empty(error_msg)
        n = len(list)
        p = 1
        idx = 0
        has_unit = .false.
        do while (p <= n)
            call next_io_spec_arg(list, n, p, arg_start, arg_end)
            if (arg_start == 0) exit
            arg = list(arg_start:arg_end)
            call split_io_spec_arg(arg, name, val)
            idx = idx + 1
            if (len(name) == 0) then
                if (is_dec_io_spec(lowercase_text(trim(adjustl(arg))))) then
                    error_msg = io_spec_upper(trim(adjustl(arg)))// &
                        ' specifier is a DEC extension'//trim(location)
                    return
                end if
                if (idx == 1) then
                    has_unit = .true.
                    call check_io_unit_value(val, kw, location, error_msg)
                    if (len_trim(error_msg) > 0) return
                end if
                cycle
            end if
            if (name == 'unit') then
                has_unit = .true.
                call check_io_unit_value(val, kw, location, error_msg)
                if (len_trim(error_msg) > 0) return
                cycle
            end if
            if (is_dec_io_spec(name)) then
                error_msg = io_spec_upper(name)// &
                    ' specifier is a DEC extension'//trim(location)
                return
            end if
            if (name == 'end' .and. kw == 'write') then
                error_msg = 'END= specifier is not allowed in an output '// &
                    'statement'//trim(location)
                return
            end if
            if (.not. is_character_io_spec(name)) cycle
            if (io_value_is_array(val)) then
                error_msg = io_spec_upper(name)//' specifier must be scalar'// &
                    trim(location)
                return
            end if
            if (io_value_nondefault_kind(val)) then
                error_msg = io_spec_upper(name)//' specifier must be a '// &
                    'character string of default kind'//trim(location)
                return
            end if
        end do
        if (kw /= 'write') return
        if (has_unit) return
        error_msg = 'UNIT not specified in WRITE statement'//trim(location)
    end subroutine check_io_control_list

    subroutine check_io_unit_value(val, kw, location, error_msg)
        character(len=*), intent(in) :: val, kw, location
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: t
        integer :: i
        logical :: digits

        call set_empty(error_msg)
        t = trim(adjustl(val))
        if (len(t) == 0) return
        if (kw == 'write') then
            if (t(1:1) == '''' .or. t(1:1) == '"') then
                error_msg = 'Invalid form of WRITE statement: a character '// &
                    'literal cannot be a unit'//trim(location)
                return
            end if
        end if
        if (t(1:1) /= '-') return
        if (len(t) < 2) return
        digits = .true.
        do i = 2, len(t)
            if (t(i:i) < '0' .or. t(i:i) > '9') digits = .false.
        end do
        if (.not. digits) return
        error_msg = 'UNIT number cannot be negative'//trim(location)
    end subroutine check_io_unit_value

    ! A BOZ literal constant has no type and can never be an output item.
    subroutine check_io_output_items(items, location, error_msg)
        character(len=*), intent(in) :: items, location
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: low
        integer :: i, n, close_q
        character :: c, prev, qc
        logical :: is_boz

        call set_empty(error_msg)
        low = lowercase_text(items)
        n = len(low)
        i = 1
        do while (i < n)
            c = low(i:i)
            if (c == '''' .or. c == '"') then
                ! A plain character literal: skip it whole so its content
                ! cannot be mistaken for a radix prefix.
                qc = c
                i = i + 1
                do while (i <= n)
                    if (low(i:i) == qc) exit
                    i = i + 1
                end do
                i = i + 1
                cycle
            end if
            if (c /= 'b' .and. c /= 'o' .and. c /= 'z') then
                i = i + 1
                cycle
            end if
            qc = low(i + 1:i + 1)
            if (qc /= '''' .and. qc /= '"') then
                i = i + 1
                cycle
            end if
            is_boz = .true.
            if (i > 1) then
                prev = low(i - 1:i - 1)
                if (prev == '_') is_boz = .false.
                if (prev >= 'a' .and. prev <= 'z') is_boz = .false.
                if (prev >= '0' .and. prev <= '9') is_boz = .false.
            end if
            close_q = index(low(i + 2:), qc)
            if (close_q == 0) then
                i = i + 1
                cycle
            end if
            close_q = i + 1 + close_q
            if (is_boz) then
                if (.not. is_boz_digits(low(i + 2:close_q - 1), c)) then
                    is_boz = .false.
                end if
            end if
            if (is_boz) then
                error_msg = 'BOZ literal constant cannot appear in an '// &
                    'output IO list'//trim(location)
                return
            end if
            i = close_q + 1
        end do
    end subroutine check_io_output_items

    ! Digits valid for a BOZ radix prefix: binary, octal, or hexadecimal.
    logical function is_boz_digits(text, radix) result(ok)
        character(len=*), intent(in) :: text
        character, intent(in) :: radix
        integer :: i
        character :: c

        ok = .false.
        if (len(text) == 0) return
        do i = 1, len(text)
            c = text(i:i)
            select case (radix)
            case ('b')
                if (c /= '0' .and. c /= '1') return
            case ('o')
                if (c < '0' .or. c > '7') return
            case default
                if (c >= '0' .and. c <= '9') cycle
                if (c < 'a' .or. c > 'f') return
            end select
        end do
        ok = .true.
    end function is_boz_digits

    ! Split "name=value" into a lowercased specifier name and its value text.
    ! A positional item yields an empty name.
    subroutine split_io_spec_arg(arg, name, val)
        character(len=*), intent(in) :: arg
        character(len=:), allocatable, intent(out) :: name, val
        character(len=:), allocatable :: t, head
        integer :: eqp, i
        character :: c

        call set_empty(name)
        t = trim(adjustl(arg))
        val = t
        eqp = io_top_level_equals(t)
        if (eqp <= 1) return
        head = trim(adjustl(t(1:eqp - 1)))
        if (len(head) == 0) return
        c = head(1:1)
        if (.not. ((c >= 'a' .and. c <= 'z') .or. (c >= 'A' .and. c <= 'Z'))) &
            return
        do i = 2, len(head)
            c = head(i:i)
            if (c >= 'a' .and. c <= 'z') cycle
            if (c >= 'A' .and. c <= 'Z') cycle
            if (c >= '0' .and. c <= '9') cycle
            if (c == '_') cycle
            return
        end do
        name = lowercase_text(head)
        val = trim(adjustl(t(eqp + 1:)))
    end subroutine split_io_spec_arg

    ! Position of the first '=' outside quotes and nesting, 0 when absent.
    integer function io_top_level_equals(text) result(eqp)
        character(len=*), intent(in) :: text
        integer :: i, depth
        character :: qc

        eqp = 0
        depth = 0
        i = 1
        do while (i <= len(text))
            if (text(i:i) == '''' .or. text(i:i) == '"') then
                qc = text(i:i)
                i = i + 1
                do while (i <= len(text))
                    if (text(i:i) == qc) exit
                    i = i + 1
                end do
            else if (text(i:i) == '(' .or. text(i:i) == '[') then
                depth = depth + 1
            else if (text(i:i) == ')' .or. text(i:i) == ']') then
                if (depth > 0) depth = depth - 1
            else if (depth == 0 .and. text(i:i) == '=') then
                eqp = i
                return
            end if
            i = i + 1
        end do
    end function io_top_level_equals

    logical function io_value_is_array(val) result(is_array)
        character(len=*), intent(in) :: val
        character(len=:), allocatable :: t

        is_array = .false.
        t = trim(adjustl(val))
        if (len(t) == 0) return
        if (t(1:1) == '[') then
            is_array = .true.
            return
        end if
        if (len(t) < 2) return
        if (t(1:2) == '(/') is_array = .true.
    end function io_value_is_array

    ! True when the specifier value is a character value of a kind other than
    ! default: char(c, k) with k /= 1, or a kind-prefixed literal k_'...'.
    logical function io_value_nondefault_kind(val) result(bad)
        character(len=*), intent(in) :: val
        character(len=:), allocatable :: t, low, inner, arg2, kindtext
        integer :: n, p, arg_start, arg_end, i, upos

        bad = .false.
        t = trim(adjustl(val))
        if (len(t) < 2) return
        low = lowercase_text(t)
        upos = index(low, '_')
        if (upos > 1) then
            if (low(upos + 1:upos + 1) == '''' .or. &
                low(upos + 1:upos + 1) == '"') then
                kindtext = low(1:upos - 1)
                if (io_text_is_integer(kindtext)) then
                    if (kindtext /= '1') bad = .true.
                    return
                end if
            end if
        end if
        if (len(low) < 6) return
        if (low(1:5) /= 'char(') return
        if (low(len(low):len(low)) /= ')') return
        inner = t(6:len(t) - 1)
        n = len(inner)
        p = 1
        call next_io_spec_arg(inner, n, p, arg_start, arg_end)
        if (arg_start == 0) return
        call next_io_spec_arg(inner, n, p, arg_start, arg_end)
        if (arg_start == 0) return
        arg2 = trim(adjustl(inner(arg_start:arg_end)))
        i = io_top_level_equals(arg2)
        if (i > 0) arg2 = trim(adjustl(arg2(i + 1:)))
        if (.not. io_text_is_integer(arg2)) return
        if (arg2 /= '1') bad = .true.
    end function io_value_nondefault_kind

    logical function io_text_is_integer(text) result(is_int)
        character(len=*), intent(in) :: text
        integer :: i

        is_int = .false.
        if (len_trim(text) == 0) return
        do i = 1, len_trim(text)
            if (text(i:i) < '0' .or. text(i:i) > '9') return
        end do
        is_int = .true.
    end function io_text_is_integer

    integer function io_paren_balance(text) result(balance)
        character(len=*), intent(in) :: text
        integer :: i
        character :: qc

        balance = 0
        i = 1
        do while (i <= len(text))
            if (text(i:i) == '''' .or. text(i:i) == '"') then
                qc = text(i:i)
                i = i + 1
                do while (i <= len(text))
                    if (text(i:i) == qc) exit
                    i = i + 1
                end do
            else if (text(i:i) == '(') then
                balance = balance + 1
            else if (text(i:i) == ')') then
                balance = balance - 1
            end if
            i = i + 1
        end do
    end function io_paren_balance

    function strip_statement_label(code) result(rest)
        character(len=*), intent(in) :: code
        character(len=:), allocatable :: rest
        integer :: i

        i = 1
        do while (i <= len(code))
            if (code(i:i) < '0' .or. code(i:i) > '9') exit
            i = i + 1
        end do
        if (i == 1) then
            rest = code
        else
            rest = code(i:)
        end if
    end function strip_statement_label

    function leading_io_keyword(low) result(kw)
        character(len=*), intent(in) :: low
        character(len=:), allocatable :: kw
        integer :: i

        i = 1
        do while (i <= len(low))
            if (low(i:i) < 'a' .or. low(i:i) > 'z') exit
            i = i + 1
        end do
        if (i == 1) then
            kw = ''
        else
            kw = low(1:i - 1)
        end if
    end function leading_io_keyword

    ! Bounds of the control-list parentheses in an I/O statement.
    subroutine io_statement_paren_span(code, open_p, close_p)
        character(len=*), intent(in) :: code
        integer, intent(out) :: open_p, close_p
        integer :: i, depth
        character :: qc

        open_p = 0
        close_p = 0
        depth = 0
        i = 1
        do while (i <= len(code))
            if (code(i:i) == '''' .or. code(i:i) == '"') then
                qc = code(i:i)
                i = i + 1
                do while (i <= len(code))
                    if (code(i:i) == qc) exit
                    i = i + 1
                end do
            else if (code(i:i) == '(') then
                depth = depth + 1
                if (depth == 1) open_p = i
            else if (code(i:i) == ')') then
                depth = depth - 1
                if (depth == 0) then
                    close_p = i
                    return
                end if
            end if
            i = i + 1
        end do
    end subroutine io_statement_paren_span

    ! DEC I/O extension specifiers. Without -fdec they are not Fortran, and the
    ! standard has no equivalent, so they are rejected rather than ignored.
    logical function is_dec_io_spec(name) result(is_dec)
        character(len=*), intent(in) :: name

        select case (trim(name))
        case ('carriagecontrol', 'share', 'shared', 'noshared', 'readonly')
            is_dec = .true.
        case default
            is_dec = .false.
        end select
    end function is_dec_io_spec

    ! NAMELIST transfer-statement constraints (#389). A group object that is
    ! polymorphic, or of a derived type with ALLOCATABLE or POINTER components,
    ! requires a defined input/output procedure; an INTENT(IN) group object
    ! cannot be read. The group list is taken from the source text (see
    ! check_namelist_members); the second positional item of a data transfer
    ! statement is a namelist group name whenever the source declares that
    ! group.
    subroutine check_namelist_transfers(arena, source, error_msg)
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: source
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: group
        integer :: n

        call set_empty(error_msg)
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (nd => arena%entries(n)%node)
            type is (read_statement_node)
                call namelist_transfer_group(nd%namelist_group, nd%format_spec, &
                                             source, nd%line, group)
                if (len_trim(group) == 0) cycle
                call check_namelist_transfer_group(arena, source, group, .true., &
                                                   nd%line, error_msg)
            type is (write_statement_node)
                call namelist_transfer_group(nd%namelist_group, nd%format_spec, &
                                             source, nd%line, group)
                if (len_trim(group) == 0) cycle
                call check_namelist_transfer_group(arena, source, group, .false., &
                                                   nd%line, error_msg)
            end select
            if (len_trim(error_msg) > 0) return
        end do
    end subroutine check_namelist_transfers

    subroutine namelist_transfer_group(nml_spec, format_spec, source, line, group)
        character(len=:), allocatable, intent(in) :: nml_spec
        character(len=:), allocatable, intent(in) :: format_spec
        character(len=*), intent(in) :: source
        integer, intent(in) :: line
        character(len=:), allocatable, intent(out) :: group
        character(len=:), allocatable :: candidate

        group = ''
        if (allocated(nml_spec)) then
            if (len_trim(nml_spec) > 0) then
                group = trim(nml_spec)
                return
            end if
        end if
        candidate = ''
        if (allocated(format_spec)) candidate = trim(format_spec)
        if (len_trim(candidate) == 0) then
            ! A bare positional group name is not always kept on the typed node,
            ! so recover it from the transfer statement itself.
            call namelist_source_control_item(source, line, candidate)
        end if
        if (len_trim(candidate) == 0) return
        if (.not. namelist_source_has_group(source, trim(candidate))) return
        group = trim(candidate)
    end subroutine namelist_transfer_group

    ! Second control-list item of the data transfer statement on `line`, which
    ! is the format or namelist group name in positional form.
    subroutine namelist_source_control_item(source, line, item)
        character(len=*), intent(in) :: source
        integer, intent(in) :: line
        character(len=:), allocatable, intent(out) :: item
        character(len=:), allocatable :: text
        integer :: lp, comma, close_pos, eq_pos

        item = ''
        call namelist_source_line(source, line, text)
        text = trim(adjustl(strip_data_source_comment(text)))
        lp = index(text, '(')
        if (lp == 0) return
        comma = index(text(lp + 1:), ',')
        if (comma == 0) return
        text = text(lp + comma + 1:)
        close_pos = index(text, ')')
        if (close_pos == 0) return
        text = text(1:close_pos - 1)
        comma = index(text, ',')
        if (comma > 0) text = text(1:comma - 1)
        eq_pos = index(text, '=')
        if (eq_pos > 0) text = text(eq_pos + 1:)
        item = trim(adjustl(text))
        if (verify(item, 'abcdefghijklmnopqrstuvwxyz' &
                   //'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_') /= 0) item = ''
    end subroutine namelist_source_control_item

    subroutine namelist_source_line(source, line, text)
        character(len=*), intent(in) :: source
        integer, intent(in) :: line
        character(len=:), allocatable, intent(out) :: text
        integer :: pos, next_nl, current

        text = ''
        if (line < 1) return
        pos = 1
        current = 1
        do while (pos <= len(source))
            next_nl = index(source(pos:), new_line('a'))
            if (next_nl == 0) then
                if (current == line) text = source(pos:)
                return
            end if
            if (current == line) then
                text = source(pos:pos + next_nl - 2)
                return
            end if
            pos = pos + next_nl
            current = current + 1
        end do
    end subroutine namelist_source_line

    logical function namelist_source_has_group(source, name) result(has_group)
        character(len=*), intent(in) :: source
        character(len=*), intent(in) :: name
        character(len=:), allocatable :: stmt, group
        integer :: pos, line_no, stmt_line, n, p
        logical :: found, ok

        has_group = .false.
        pos = 1
        line_no = 1
        do
            call namelist_next_statement(source, pos, line_no, stmt, stmt_line, &
                                         found)
            if (.not. found) exit
            n = len_trim(stmt)
            p = 9
            call namelist_take_group(stmt, n, p, group, ok)
            if (.not. ok) cycle
            if (same_name(group, name)) then
                has_group = .true.
                return
            end if
        end do
    end function namelist_source_has_group

    subroutine check_namelist_transfer_group(arena, source, group, is_read, &
                                             line, error_msg)
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: source
        character(len=*), intent(in) :: group
        logical, intent(in) :: is_read
        integer, intent(in) :: line
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: stmt, this_group, member
        integer :: pos, line_no, stmt_line, n, p
        logical :: found, ok, new_group, in_group

        call set_empty(error_msg)
        pos = 1
        line_no = 1
        do
            call namelist_next_statement(source, pos, line_no, stmt, stmt_line, &
                                         found)
            if (.not. found) exit
            n = len_trim(stmt)
            p = 9
            new_group = .true.
            in_group = .false.
            do
                if (new_group) then
                    call namelist_take_group(stmt, n, p, this_group, ok)
                    if (.not. ok) exit
                    in_group = same_name(this_group, group)
                    new_group = .false.
                end if
                call namelist_take_member(stmt, n, p, member, new_group, ok)
                if (.not. ok) exit
                if (in_group .and. len_trim(member) > 0) then
                    call check_namelist_transfer_member(arena, source, member, &
                        group, is_read, line, error_msg)
                    if (len_trim(error_msg) > 0) return
                end if
                if (p > n) exit
            end do
        end do
    end subroutine check_namelist_transfer_group

    subroutine check_namelist_transfer_member(arena, source, name, group, &
                                              is_read, line, error_msg)
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: source
        character(len=*), intent(in) :: name
        character(len=*), intent(in) :: group
        logical, intent(in) :: is_read
        integer, intent(in) :: line
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: base, low
        character(len=32) :: location
        integer :: decl_index

        call set_empty(error_msg)
        decl_index = namelist_find_declaration(arena, name, huge(1))
        if (decl_index == 0) return
        write (location, '(" at line ",I0)') line
        select type (decl => arena%entries(decl_index)%node)
        type is (declaration_node)
            if (allocated(decl%type_name)) then
                low = trim(lowercase_text(decl%type_name))
                if (index(low, 'class(') == 1) then
                    error_msg = 'NAMELIST object '''//trim(name)// &
                        ''' in namelist '''//trim(group)//''' is polymorphic'// &
                        ' and requires a defined input/output procedure'// &
                        trim(location)
                    return
                end if
                base = namelist_derived_base_name(decl%type_name)
                if (len_trim(base) > 0) then
                    if (namelist_type_needs_dtio(arena, source, base, 0)) then
                        error_msg = 'NAMELIST object '''//trim(name)// &
                            ''' in namelist '''//trim(group)//''' has '// &
                            'ALLOCATABLE or POINTER components and thus '// &
                            'requires a defined input/output procedure'// &
                            trim(location)
                        return
                    end if
                end if
            end if
            if (.not. is_read) return
            if (.not. decl%has_intent) return
            if (.not. allocated(decl%intent)) return
            if (trim(lowercase_text(decl%intent)) == 'in') then
                error_msg = 'NAMELIST object '''//trim(name)//''' in namelist '''// &
                    trim(group)//''' is INTENT(IN) and cannot be read'// &
                    trim(location)
            end if
        end select
    end subroutine check_namelist_transfer_member

    ! A derived type needs a defined input/output procedure for namelist
    ! transfer when it has, directly or through a component, an ALLOCATABLE or
    ! POINTER component. Types defined in another program unit of the same file
    ! reach the arena without their component list, so the type definition is
    ! also read from the source text.
    recursive logical function namelist_type_needs_dtio(arena, source, name, &
            depth) result(needs)
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: source
        character(len=*), intent(in) :: name
        integer, intent(in) :: depth
        character(len=:), allocatable :: base
        integer :: t, i

        needs = .false.
        if (depth > 8) return
        t = namelist_find_derived_type(arena, name)
        if (t == 0) then
            needs = namelist_source_type_needs_dtio(source, name, depth)
            return
        end if
        select type (td => arena%entries(t)%node)
        type is (derived_type_node)
            if (.not. allocated(td%component_indices)) then
                needs = namelist_source_type_needs_dtio(source, name, depth)
                return
            end if
            do i = 1, size(td%component_indices)
                if (.not. node_exists(arena, td%component_indices(i))) cycle
                select type (comp => arena%entries(td%component_indices(i))%node)
                type is (declaration_node)
                    if (comp%is_allocatable .or. comp%is_pointer) then
                        needs = .true.
                        return
                    end if
                    if (.not. allocated(comp%type_name)) cycle
                    base = namelist_derived_base_name(comp%type_name)
                    if (len_trim(base) == 0) cycle
                    needs = namelist_type_needs_dtio(arena, source, base, &
                                                     depth + 1)
                    if (needs) return
                end select
            end do
        end select
    end function namelist_type_needs_dtio

    ! Same question answered from the source text: walk the type definition
    ! block of `name` and report an ALLOCATABLE or POINTER component, recursing
    ! into derived-type components.
    recursive logical function namelist_source_type_needs_dtio(source, name, &
            depth) result(needs)
        character(len=*), intent(in) :: source
        character(len=*), intent(in) :: name
        integer, intent(in) :: depth
        character(len=:), allocatable :: text, low, attrs, base
        integer :: pos, next_nl, sep
        logical :: inside

        needs = .false.
        if (depth > 8) return
        pos = 1
        inside = .false.
        do while (pos <= len(source))
            next_nl = index(source(pos:), new_line('a'))
            if (next_nl == 0) then
                text = source(pos:)
                pos = len(source) + 1
            else
                text = source(pos:pos + next_nl - 2)
                pos = pos + next_nl
            end if
            low = trim(adjustl(lowercase_text( &
                               strip_data_source_comment(text))))
            if (.not. inside) then
                inside = namelist_type_header_names(low, name)
                cycle
            end if
            if (index(low, 'end type') == 1) return
            if (index(low, 'contains') == 1) return
            sep = index(low, '::')
            if (sep == 0) cycle
            attrs = low(1:sep - 1)
            if (index(attrs, 'allocatable') > 0 .or. &
                index(attrs, 'pointer') > 0) then
                needs = .true.
                return
            end if
            base = namelist_derived_base_name(attrs)
            if (len_trim(base) == 0) cycle
            if (same_name(base, name)) cycle
            needs = namelist_source_type_needs_dtio(source, base, depth + 1)
            if (needs) return
        end do
    end function namelist_source_type_needs_dtio

    ! True when the lowercased statement is the type definition header of
    ! `name`, in either `type name`, `type :: name`, or `type, attrs :: name`
    ! form.
    logical function namelist_type_header_names(low, name) result(is_header)
        character(len=*), intent(in) :: low
        character(len=*), intent(in) :: name
        character(len=:), allocatable :: rest
        integer :: sep

        is_header = .false.
        if (len(low) < 5) return
        if (low(1:4) /= 'type') return
        if (low(5:5) /= ' ' .and. low(5:5) /= ',' .and. low(5:5) /= ':') return
        rest = trim(adjustl(low(5:)))
        sep = index(rest, '::')
        if (sep > 0) then
            rest = trim(adjustl(rest(sep + 2:)))
        else if (len(rest) > 0) then
            if (rest(1:1) == ',') return
        end if
        if (index(rest, '(') > 0) rest = rest(1:index(rest, '(') - 1)
        is_header = same_name(trim(rest), name)
    end function namelist_type_header_names
