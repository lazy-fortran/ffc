    logical function is_f64_intrinsic_expression(arena, node, context) &
        result(is_f64)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        integer :: i
        integer :: intrinsic_id

        is_f64 = .false.
        if (.not. allocated(node%name)) return

        intrinsic_id = f64_intrinsic_id(node%name)
        if (intrinsic_id == F64_INTRINSIC_REAL) then
            is_f64 = .true.
            return
        end if
        ! dble() always returns real(8), regardless of argument kind.
        if (same_name(node%name, 'dble')) then
            is_f64 = .true.
            return
        end if
        if (intrinsic_id == F64_INTRINSIC_NONE) return
        if (.not. allocated(node%arg_indices)) return

        do i = 1, size(node%arg_indices)
            if (is_f64_expression(arena, node%arg_indices(i), context)) then
                is_f64 = .true.
                return
            end if
        end do
    end function is_f64_intrinsic_expression

    logical function is_real_array_reduction(arena, node, context, vk) result(ok)
        !! True when node is a sum/product/maxval/minval/norm2 call whose first
        !! argument is a fixed-size array of value kind vk.
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: vk
        character(len=:), allocatable :: arg_name, err
        integer :: sym

        ok = .false.
        if (.not. allocated(node%name)) return
        select case (trim(node%name))
        case ('sum', 'product', 'maxval', 'minval', 'norm2')
        case default
            return
        end select
        if (.not. allocated(node%arg_indices)) return
        if (size(node%arg_indices) /= 1) return
        if (node_exists(arena, node%arg_indices(1))) then
            ! sum(a(lo:hi)) and friends: the section reduces over its base
            ! array's element kind, so gate on that base's value kind.
            select type (arg => arena%entries(node%arg_indices(1))%node)
            type is (array_slice_node)
                call get_identifier_name(arena, arg%array_index, arg_name, err)
                if (len_trim(err) > 0) return
                sym = find_symbol(context, arg_name)
                if (sym <= 0) return
                ok = context%symbols(sym)%is_array .and. &
                     context%symbols(sym)%value_kind == vk
                return
            end select
        end if
        call get_identifier_name(arena, node%arg_indices(1), arg_name, err)
        if (len_trim(err) > 0) return
        sym = find_symbol(context, arg_name)
        if (sym <= 0) return
        ! A 1-D allocatable reduces over its element kind just like a fixed
        ! array, so the result print kind follows the allocatable element kind.
        ok = (context%symbols(sym)%is_array .or. &
              context%symbols(sym)%is_allocatable) .and. &
             context%symbols(sym)%value_kind == vk
    end function is_real_array_reduction

    logical function is_real_dot_product(arena, node, context, vk) result(ok)
        !! True when node is a dot_product call whose two arguments are
        !! rank-1 arrays of value kind vk.
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: vk
        character(len=:), allocatable :: a_name, b_name, err
        integer :: a_sym, b_sym

        ok = .false.
        if (.not. allocated(node%name)) return
        if (trim(node%name) /= 'dot_product') return
        if (.not. allocated(node%arg_indices)) return
        if (size(node%arg_indices) /= 2) return
        call get_identifier_name(arena, node%arg_indices(1), a_name, err)
        if (len_trim(err) > 0) return
        call get_identifier_name(arena, node%arg_indices(2), b_name, err)
        if (len_trim(err) > 0) return
        a_sym = find_symbol(context, a_name)
        b_sym = find_symbol(context, b_name)
        if (a_sym <= 0 .or. b_sym <= 0) return
        ok = context%symbols(a_sym)%is_array .and. &
             context%symbols(a_sym)%array_rank == 1 .and. &
             context%symbols(a_sym)%value_kind == vk .and. &
             context%symbols(b_sym)%is_array .and. &
             context%symbols(b_sym)%array_rank == 1 .and. &
             context%symbols(b_sym)%value_kind == vk
    end function is_real_dot_product

    subroutine real_opcode(source_op, line, column, opcode, error_msg)
        character(len=*), intent(in) :: source_op
        integer, intent(in) :: line
        integer, intent(in) :: column
        integer, intent(out) :: opcode
        character(len=:), allocatable, intent(out) :: error_msg

        call set_empty(error_msg)
        select case (trim(source_op))
        case ('+')
            opcode = 18
        case ('-')
            opcode = 19
        case ('*')
            opcode = 20
        case ('/')
            opcode = 21
        case default
            call unsupported_feature_error('real operator', line, column, &
                                           'direct LIRIC session supports '// &
                                           '+, -, *, and / for real expressions', &
                                           error_msg)
        end select
    end subroutine real_opcode

    logical function is_real_literal(arena, node_index)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        character(len=:), allocatable :: value, literal_type, err

        is_real_literal = .false.
        call get_literal_info(arena, node_index, value, literal_type, err)
        if (len_trim(err) > 0) return
        ! A logical literal (.true./.false.) contains a '.' but is not real.
        if (trim(literal_type) == 'logical') return
        if (trim(value) == '.true.' .or. trim(value) == '.false.') return
        ! A BOZ constant (B'..'/O'..'/Z'..'/X'..' or postfix '..'B/'..'O/'..'Z)
        ! is always integer-valued, even when its hex digits spell a letter
        ! ('e', 'd') that the substring heuristic below would mistake for a
        ! real exponent marker.
        if (is_boz_literal_text(value)) return
        is_real_literal = trim(literal_type) == 'real' .or. &
                          index(value, '.') > 0 .or. index(value, 'e') > 0 .or. &
                          index(value, 'E') > 0
    end function is_real_literal

    logical function is_boz_literal_text(text) result(is_boz)
        !! True when text is a BOZ-literal-constant spelling: a b/o/z/x radix
        !! designator adjacent to a quoted digit string, prefix or postfix.
        character(len=*), intent(in) :: text
        character(len=:), allocatable :: trimmed, lo
        integer :: n

        is_boz = .false.
        trimmed = trim(adjustl(text))
        n = len(trimmed)
        if (n < 3) return
        lo = lowercase_text(trimmed)
        if (trimmed(1:1) == "'" .or. trimmed(1:1) == '"') then
            is_boz = is_boz_designator(lo(n:n))
        else
            is_boz = is_boz_designator(lo(1:1)) .and. &
                     (trimmed(2:2) == "'" .or. trimmed(2:2) == '"')
        end if
    end function is_boz_literal_text

    logical function is_boz_designator(c)
        character(len=1), intent(in) :: c

        is_boz_designator = c == 'b' .or. c == 'o' .or. c == 'z' .or. c == 'x'
    end function is_boz_designator

    logical function is_character_literal(arena, node_index)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        character(len=:), allocatable :: value, literal_type, err

        is_character_literal = .false.
        call get_literal_info(arena, node_index, value, literal_type, err)
        if (len_trim(err) > 0) return
        is_character_literal = trim(literal_type) == 'character' .or. &
                               starts_with_quote(value)
    end function is_character_literal

    logical function is_logical_literal(arena, node_index)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        character(len=:), allocatable :: value, literal_type, err

        is_logical_literal = .false.
        call get_literal_info(arena, node_index, value, literal_type, err)
        if (len_trim(err) > 0) return
        is_logical_literal = trim(value) == '.true.' .or. &
                             trim(value) == '.false.' .or. &
                             trim(literal_type) == 'logical'
    end function is_logical_literal

    logical function starts_with_quote(text)
        character(len=*), intent(in) :: text

        starts_with_quote = len_trim(text) >= 2 .and. &
                            (text(1:1) == '"' .or. text(1:1) == "'")
    end function starts_with_quote

    subroutine strip_literal_quotes(text, value)
        character(len=*), intent(in) :: text
        character(len=:), allocatable, intent(out) :: value
        integer :: text_len

        text_len = len_trim(text)
        if (text_len >= 2 .and. &
            ((text(1:1) == '"' .and. text(text_len:text_len) == '"') .or. &
             (text(1:1) == "'" .and. text(text_len:text_len) == "'"))) then
            value = text(2:text_len - 1)
        else
            value = trim(text)
        end if
    end subroutine strip_literal_quotes

    integer(c_int64_t) function logical_i32_value(text) result(value)
        character(len=*), intent(in) :: text

        if (trim(text) == '.true.') then
            value = 1_c_int64_t
        else
            value = 0_c_int64_t
        end if
    end function logical_i32_value

    logical function literal_is_f64(text)
        ! Returns true when a real literal is explicitly f64 — i.e. its numeric
        ! part carries a 'd'/'D' exponent, or it carries a _8 / _real64 /
        ! _c_double / _c_long_double / _dp / _wp kind suffix (the same names
        ! real_kind_is_f64 accepts for real(kind_spec) declarations).  A bare
        ! literal such as "2.5" or "3.0e0" is default real (f32).  The exponent
        ! check is scoped to the numeric part so a suffix that merely contains
        ! the letter 'd' (e.g. a BOZ digit string reaching this function by
        ! mistake) cannot misfire.
        character(len=*), intent(in) :: text
        character(len=:), allocatable :: lo, numeric_part, suffix
        integer :: uscore

        lo = lowercase_text(trim(text))
        uscore = scan(lo, '_')
        if (uscore > 1 .and. scan(lo(uscore - 1:uscore - 1), '0123456789.de') > 0) &
            then
            numeric_part = lo(1:uscore - 1)
            suffix = lo(uscore + 1:)
        else
            numeric_part = lo
            call set_empty(suffix)
        end if

        if (scan(numeric_part, 'd') > 0) then
            literal_is_f64 = .true.
            return
        end if

        select case (trim(suffix))
        case ('8', 'real64', 'c_double', 'c_long_double', 'dp', 'wp')
            literal_is_f64 = .true.
        case default
            literal_is_f64 = .false.
        end select
    end function literal_is_f64

    subroutine parse_f64_literal(text, value, error_msg)
        character(len=*), intent(in) :: text
        real(c_double), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: clean
        integer :: io_stat, uscore
        integer(c_int64_t) :: int_value
        real(c_float) :: single_value

        ! Strip the kind suffix (_8, _dp, _real64, _c_double, etc.) which Fortran
        ! internal reads do not accept. The suffix begins at the first underscore
        ! that follows a digit, dot, or exponent letter; a kind name such as
        ! c_double contains further underscores that must not be the split point.
        clean = trim(text)
        uscore = scan(clean, '_')
        if (uscore > 1) then
            if (scan(clean(uscore - 1:uscore - 1), '0123456789.dDeE') > 0) then
                clean = clean(1:uscore - 1)
            end if
        end if

        ! A literal without an explicit f64 marker is default real (f32): it
        ! rounds to single precision first, and only that rounded value widens
        ! to f64, matching gfortran (e.g. 1.05 -> 1.0499999523162842d0, not the
        ! double-precision-accurate 1.05d0).
        if (literal_is_f64(text)) then
            read (clean, *, iostat=io_stat) value
        else
            read (clean, *, iostat=io_stat) single_value
            value = real(single_value, c_double)
        end if
        if (io_stat == 0) then
            call set_empty(error_msg)
            return
        end if

        ! If real parsing failed, try parsing as a BOZ literal (e.g. Z'ABC')
        ! and convert the integer to real. This handles real(b'010101') expressions.
        call parse_i32_literal(text, int_value, error_msg)
        if (len_trim(error_msg) == 0) then
            value = real(int_value, c_double)
            return
        end if

        error_msg = 'invalid real literal for direct LIRIC session: '// &
                    trim(text)
    end subroutine parse_f64_literal
