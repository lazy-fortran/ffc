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
        !! True when node is a sum/product/maxval/minval call whose first
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
        case ('sum', 'product', 'maxval', 'minval')
        case default
            return
        end select
        if (.not. allocated(node%arg_indices)) return
        if (size(node%arg_indices) /= 1) return
        call get_identifier_name(arena, node%arg_indices(1), arg_name, err)
        if (len_trim(err) > 0) return
        sym = find_symbol(context, arg_name)
        if (sym <= 0) return
        ok = context%symbols(sym)%is_array .and. &
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
        is_real_literal = trim(literal_type) == 'real' .or. &
                          index(value, '.') > 0 .or. index(value, 'e') > 0 .or. &
                          index(value, 'E') > 0
    end function is_real_literal

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
        ! Returns true when a real literal is explicitly f64 — i.e. it carries a
        ! 'd'/'D' exponent, or a _8 / _real64 / _c_double / _c_long_double kind
        ! suffix.  A bare literal such as "2.5" or "3.0e0" is default real (f32).
        character(len=*), intent(in) :: text
        character(len=:), allocatable :: lo, suffix
        integer :: i, n, dpos

        lo = lowercase_text(trim(text))
        n = len_trim(lo)
        ! d/D exponent marks a double-precision literal.
        dpos = scan(lo, 'd')
        if (dpos > 0) then
            literal_is_f64 = .true.
            return
        end if
        ! Explicit kind suffix: find the first underscore preceded by a digit,
        ! dot, or exponent letter. Everything from that underscore onward is the
        ! kind specifier (e.g. _8, _real64, _c_double, _c_long_double).
        do i = 2, n
            if (lo(i:i) == '_' .and. &
                scan(lo(i - 1:i - 1), '0123456789.') > 0) then
                suffix = lo(i + 1:)
                select case (trim(suffix))
                case ('8', 'real64', 'c_double', 'c_long_double')
                    literal_is_f64 = .true.
                    return
                end select
                exit
            end if
        end do
        literal_is_f64 = .false.
    end function literal_is_f64

    subroutine parse_f64_literal(text, value, error_msg)
        character(len=*), intent(in) :: text
        real(c_double), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: clean
        integer :: io_stat, i, n

        ! Strip trailing kind suffix (_8, _dp, _real64, _c_double, etc.).
        ! Scan forward for the first underscore that follows a digit, dot, or
        ! exponent letter; that marks the start of the kind suffix.
        clean = trim(text)
        n = len_trim(clean)
        do i = 2, n
            if (clean(i:i) == '_' .and. &
                scan(clean(i - 1:i - 1), '0123456789.dDeE') > 0) then
                clean = clean(1:i - 1)
                exit
            end if
        end do

        read (clean, *, iostat=io_stat) value
        if (io_stat == 0) then
            call set_empty(error_msg)
        else
            error_msg = 'invalid real literal for direct LIRIC session: '// &
                        trim(text)
        end if
    end subroutine parse_f64_literal
