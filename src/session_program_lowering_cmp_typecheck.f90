submodule (session_program_lowering_impl) session_program_lowering_cmp_typecheck
    use session_program_lowering_cmp_typecheck_order
contains

    ! Comparison operand type-mismatch check. A relational comparison whose two
    ! operands are of clearly incompatible intrinsic type classes (numeric vs
    ! character, or logical vs anything else) is invalid Fortran and must be
    ! rejected with a diagnostic rather than silently mislowered. Only operands
    ! whose type class is statically obvious (a literal or a declared scalar
    ! identifier) are classified; when either operand's class is unknown the
    ! comparison is left for the normal lowering path, so a valid program is
    ! never rejected here.
    module procedure check_comparison_operand_types
    integer :: left_class, right_class
    character(len=64) :: location

    call set_empty(error_msg)
    if (.not. is_relational_operator(bin_op)) return
    left_class = comparison_operand_class(arena, left_idx, context)
    if (left_class == CMP_CLASS_UNKNOWN) return
    right_class = comparison_operand_class(arena, right_idx, context)
    if (right_class == CMP_CLASS_UNKNOWN) return
    if (left_class == right_class) return
    if (line > 0 .and. col > 0) then
        write (location, '(" at line ",I0,", column ",I0)') line, col
        error_msg = 'operands of comparison operator '// &
            trim(adjustl(bin_op))//trim(location)// &
            ' have mismatched types'
    else
        error_msg = 'operands of comparison operator '// &
            trim(adjustl(bin_op))//' have mismatched types'
    end if
    end procedure check_comparison_operand_types

    module procedure is_relational_operator
    select case (trim(adjustl(lowercase_text(op))))
    case ('==', '/=', '<', '<=', '>', '>=', &
            '.eq.', '.ne.', '.lt.', '.le.', '.gt.', '.ge.')
        is_rel = .true.
    case default
        is_rel = .false.
    end select
    end procedure is_relational_operator

    ! Coarse intrinsic type class of a scalar comparison operand.
    ! CMP_CLASS_NUMERIC = integer/real/complex, CMP_CLASS_CHAR = character,
    ! CMP_CLASS_LOGICAL = logical, CMP_CLASS_UNKNOWN = anything not statically
    ! resolvable (nested expressions, calls, derived types, unknown symbols).
    module procedure comparison_operand_class
    character(len=:), allocatable :: id_name, id_err
    integer :: symbol_index

    cls = CMP_CLASS_UNKNOWN
    if (.not. node_exists(arena, node_index)) return
    if (is_character_operand(arena, node_index, context)) then
        cls = CMP_CLASS_CHAR
        return
    end if
    if (is_literal(arena, node_index)) then
        if (is_hollerith_literal(arena, node_index)) then
            cls = CMP_CLASS_CHAR
            return
        end if
        if (is_logical_literal(arena, node_index)) then
            cls = CMP_CLASS_LOGICAL
        else if (is_character_literal(arena, node_index)) then
            cls = CMP_CLASS_CHAR
        else
            cls = CMP_CLASS_NUMERIC
        end if
        return
    end if
    if (is_identifier(arena, node_index)) then
        call get_identifier_name(arena, node_index, id_name, id_err)
        if (len_trim(id_err) > 0) return
        symbol_index = find_symbol_compat(context, id_name)
        if (symbol_index <= 0) return
        cls = comparison_value_kind_class( &
            context%symbols(symbol_index)%value_kind)
    end if
    end procedure comparison_operand_class

    module procedure comparison_value_kind_class
    select case (value_kind)
    case (VALUE_I32, VALUE_I64, VALUE_I8, VALUE_I16, VALUE_F32, VALUE_F64, &
            VALUE_C4, VALUE_C8)
        cls = CMP_CLASS_NUMERIC
    case (VALUE_CHARACTER)
        cls = CMP_CLASS_CHAR
    case (VALUE_LOGICAL)
        cls = CMP_CLASS_LOGICAL
    case default
        cls = CMP_CLASS_UNKNOWN
    end select
    end procedure comparison_value_kind_class

    module procedure is_hollerith_literal
    character(len=:), allocatable :: value, literal_type, err, text
    integer :: h_pos, n, count, ios

    is_holl = .false.
    call get_literal_info(arena, node_index, value, literal_type, err)
    if (len_trim(err) > 0) return
    text = trim(adjustl(value))
    h_pos = index(lowercase_text(text), 'h')
    if (h_pos <= 1) return
    read (text(1:h_pos - 1), *, iostat=ios) count
    if (ios /= 0 .or. count <= 0) return
    n = len_trim(text)
    is_holl = n - h_pos >= count
    end procedure is_hollerith_literal

end submodule session_program_lowering_cmp_typecheck
