    ! Comparison operand type-mismatch check. A relational comparison whose two
    ! operands are of clearly incompatible intrinsic type classes (numeric vs
    ! character, or logical vs anything else) is invalid Fortran and must be
    ! rejected with a diagnostic rather than silently mislowered. Only operands
    ! whose type class is statically obvious (a literal or a declared scalar
    ! identifier) are classified; when either operand's class is unknown the
    ! comparison is left for the normal lowering path, so a valid program is
    ! never rejected here.
    subroutine check_comparison_operand_types(arena, bin_op, left_idx, &
                                              right_idx, line, col, context, &
                                              error_msg)
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: bin_op
        integer, intent(in) :: left_idx, right_idx, line, col
        type(lowering_context_t), intent(in) :: context
        character(len=:), allocatable, intent(out) :: error_msg
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
    end subroutine check_comparison_operand_types

    logical function is_relational_operator(op) result(is_rel)
        character(len=*), intent(in) :: op

        select case (trim(adjustl(lowercase_text(op))))
        case ('==', '/=', '<', '<=', '>', '>=', &
              '.eq.', '.ne.', '.lt.', '.le.', '.gt.', '.ge.')
            is_rel = .true.
        case default
            is_rel = .false.
        end select
    end function is_relational_operator

    ! Coarse intrinsic type class of a scalar comparison operand.
    ! CMP_CLASS_NUMERIC = integer/real/complex, CMP_CLASS_CHAR = character,
    ! CMP_CLASS_LOGICAL = logical, CMP_CLASS_UNKNOWN = anything not statically
    ! resolvable (nested expressions, calls, derived types, unknown symbols).
    recursive integer function comparison_operand_class(arena, node_index, &
                                                        context) result(cls)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        character(len=:), allocatable :: id_name, id_err
        integer :: symbol_index

        cls = CMP_CLASS_UNKNOWN
        if (.not. node_exists(arena, node_index)) return
        if (is_character_operand(arena, node_index, context)) then
            cls = CMP_CLASS_CHAR
            return
        end if
        if (is_literal(arena, node_index)) then
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
            symbol_index = find_symbol(context, id_name)
            if (symbol_index <= 0) return
            cls = comparison_value_kind_class( &
                  context%symbols(symbol_index)%value_kind)
        end if
    end function comparison_operand_class

    integer function comparison_value_kind_class(value_kind) result(cls)
        integer, intent(in) :: value_kind

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
    end function comparison_value_kind_class
