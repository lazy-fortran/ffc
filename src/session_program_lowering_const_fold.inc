    ! ISO_C_BINDING integer-kind named constants used as compile-time integer
    ! values (e.g. "integer, parameter :: lp = c_bool"), matching the actual
    ! kind numbers gfortran reports at runtime.
    logical function iso_c_binding_kind_value(name, value) result(found)
        character(len=*), intent(in) :: name
        integer(c_int64_t), intent(out) :: value
        character(len=:), allocatable :: lo

        lo = lowercase_text(trim(name))
        found = .true.
        select case (lo)
        case ('c_bool', 'c_signed_char', 'c_int8_t', 'c_char')
            value = 1_c_int64_t
        case ('c_short', 'c_int16_t')
            value = 2_c_int64_t
        case ('c_int', 'c_int32_t', 'c_float')
            value = 4_c_int64_t
        case ('c_long', 'c_long_long', 'c_int64_t', 'c_double', 'c_size_t', &
              'c_intptr_t')
            value = 8_c_int64_t
        case default
            found = .false.
            value = 0_c_int64_t
        end select
    end function iso_c_binding_kind_value

    ! Determine the kind number of an integer literal's optional kind suffix
    ! (12_8, 12_int64, 12_c_long); a bare literal is default kind 4.
    subroutine integer_literal_kind_number(text, k, error_msg)
        character(len=*), intent(in) :: text
        integer(c_int64_t), intent(out) :: k
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: lo, suffix
        integer :: uscore
        integer(c_int64_t) :: iso_value

        lo = lowercase_text(trim(text))
        k = 4_c_int64_t
        call set_empty(error_msg)
        uscore = scan(lo, '_')
        if (uscore <= 0) return
        suffix = lo(uscore + 1:)
        select case (suffix)
        case ('1', 'int8')
            k = 1_c_int64_t
        case ('2', 'int16')
            k = 2_c_int64_t
        case ('4', 'int32')
            k = 4_c_int64_t
        case ('8', 'int64')
            k = 8_c_int64_t
        case default
            if (iso_c_binding_kind_value(suffix, iso_value)) then
                k = iso_value
            else
                error_msg = 'unsupported integer literal kind suffix: '// &
                            trim(suffix)
            end if
        end select
    end subroutine integer_literal_kind_number

    ! Constant-fold min(...)/max(...) over two or more compile-time integer
    ! arguments; real/character overloads carry a different declared value
    ! kind and route through the runtime expression lowerer instead.
    subroutine eval_min_max_constant(arena, node, context, want_max, &
                                     constant_value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        logical, intent(in) :: want_max
        integer(c_int64_t), intent(out) :: constant_value
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int64_t) :: arg_value
        integer :: i

        constant_value = 0_c_int64_t
        if (.not. allocated(node%arg_indices) .or. size(node%arg_indices) < 2) then
            error_msg = trim(node%name)//' requires at least two arguments'
            return
        end if

        call eval_i32_constant(arena, node%arg_indices(1), context, &
                               constant_value, error_msg)
        if (len_trim(error_msg) > 0) return

        do i = 2, size(node%arg_indices)
            call eval_i32_constant(arena, node%arg_indices(i), context, &
                                   arg_value, error_msg)
            if (len_trim(error_msg) > 0) return
            if (want_max) then
                if (arg_value > constant_value) constant_value = arg_value
            else
                if (arg_value < constant_value) constant_value = arg_value
            end if
        end do
        call set_empty(error_msg)
    end subroutine eval_min_max_constant

    ! Constant-fold int(x[, kind]): the compile-time integer folder only ever
    ! reaches this for an already-integer x (a real x carries a different
    ! declared value kind and routes through the runtime expression
    ! lowerer), so folding is a value pass-through.
    subroutine eval_int_constant(arena, node, context, constant_value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        integer(c_int64_t), intent(out) :: constant_value
        character(len=:), allocatable, intent(out) :: error_msg

        constant_value = 0_c_int64_t
        if (.not. allocated(node%arg_indices)) then
            error_msg = 'int requires one or two arguments'
            return
        end if
        if (size(node%arg_indices) < 1 .or. size(node%arg_indices) > 2) then
            error_msg = 'int requires one or two arguments'
            return
        end if
        call eval_i32_constant(arena, node%arg_indices(1), context, &
                               constant_value, error_msg)
    end subroutine eval_int_constant

    ! Determine the Fortran kind number of a compile-time-foldable argument
    ! (a literal or a named entity), for huge()'s kind-dependent bound.
    subroutine constant_arg_kind_number(arena, context, arg_index, k, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: arg_index
        integer(c_int64_t), intent(out) :: k
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: id_name
        integer :: sym

        k = 0_c_int64_t
        if (is_literal(arena, arg_index)) then
            call kind_of_literal(arena, arg_index, k, error_msg)
            return
        end if
        if (is_identifier(arena, arg_index)) then
            call get_identifier_name(arena, arg_index, id_name, error_msg)
            if (len_trim(error_msg) > 0) return
            sym = find_symbol(context, id_name)
            if (sym <= 0) then
                error_msg = 'huge argument was not declared: '//trim(id_name)
                return
            end if
            k = value_kind_number(context%symbols(sym)%value_kind)
            if (k < 0) then
                error_msg = 'huge argument has no compile-time kind: '// &
                            trim(id_name)
            end if
            return
        end if
        error_msg = 'huge argument is not a literal or named entity'
    end subroutine constant_arg_kind_number

    ! Constant-fold huge(x) for an integer x: the largest representable value
    ! of x's kind.
    subroutine eval_huge_constant(arena, node, context, constant_value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        integer(c_int64_t), intent(out) :: constant_value
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int64_t) :: k

        constant_value = 0_c_int64_t
        if (.not. allocated(node%arg_indices) .or. size(node%arg_indices) /= 1) then
            error_msg = 'huge requires one argument'
            return
        end if
        call constant_arg_kind_number(arena, context, node%arg_indices(1), k, &
                                      error_msg)
        if (len_trim(error_msg) > 0) return
        select case (k)
        case (1_c_int64_t)
            constant_value = 127_c_int64_t
        case (2_c_int64_t)
            constant_value = 32767_c_int64_t
        case (4_c_int64_t)
            constant_value = 2147483647_c_int64_t
        case (8_c_int64_t)
            constant_value = 9223372036854775807_c_int64_t
        case default
            error_msg = 'huge argument kind is not a supported integer kind'
            return
        end select
        call set_empty(error_msg)
    end subroutine eval_huge_constant

    ! Constant-fold precision(x) for a real x: decimal precision of x's kind.
    subroutine eval_precision_constant(arena, node, context, constant_value, &
                                       error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        integer(c_int64_t), intent(out) :: constant_value
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int64_t) :: k

        constant_value = 0_c_int64_t
        if (.not. allocated(node%arg_indices) .or. size(node%arg_indices) /= 1) then
            error_msg = 'precision requires one argument'
            return
        end if
        call constant_arg_kind_number(arena, context, node%arg_indices(1), k, &
                                      error_msg)
        if (len_trim(error_msg) > 0) return
        select case (k)
        case (4_c_int64_t)
            constant_value = 6_c_int64_t
        case (8_c_int64_t)
            constant_value = 15_c_int64_t
        case default
            error_msg = 'precision argument kind is not a supported real kind'
            return
        end select
        call set_empty(error_msg)
    end subroutine eval_precision_constant

    ! Constant-fold bit_size(x) for an integer x: 8 times its byte width.
    subroutine eval_bit_size_constant(arena, node, context, constant_value, &
                                      error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        integer(c_int64_t), intent(out) :: constant_value
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int64_t) :: k

        constant_value = 0_c_int64_t
        if (.not. allocated(node%arg_indices) .or. size(node%arg_indices) /= 1) then
            error_msg = 'bit_size requires one argument'
            return
        end if
        call constant_arg_kind_number(arena, context, node%arg_indices(1), k, &
                                      error_msg)
        if (len_trim(error_msg) > 0) return
        select case (k)
        case (1_c_int64_t, 2_c_int64_t, 4_c_int64_t, 8_c_int64_t)
            constant_value = k*8_c_int64_t
        case default
            error_msg = 'bit_size argument kind is not a supported integer kind'
            return
        end select
        call set_empty(error_msg)
    end subroutine eval_bit_size_constant

    ! Constant-fold range(x): decimal exponent range of x's kind, which
    ! differs for integer versus real kinds sharing the same kind number.
    subroutine eval_range_constant(arena, node, context, constant_value, &
                                   error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        integer(c_int64_t), intent(out) :: constant_value
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: arg_index, sym
        integer(c_int64_t) :: k
        logical :: arg_is_real
        character(len=:), allocatable :: id_name

        constant_value = 0_c_int64_t
        if (.not. allocated(node%arg_indices) .or. size(node%arg_indices) /= 1) then
            error_msg = 'range requires one argument'
            return
        end if
        arg_index = node%arg_indices(1)

        if (is_literal(arena, arg_index)) then
            arg_is_real = is_real_literal(arena, arg_index)
            call kind_of_literal(arena, arg_index, k, error_msg)
            if (len_trim(error_msg) > 0) return
        else if (is_identifier(arena, arg_index)) then
            call get_identifier_name(arena, arg_index, id_name, error_msg)
            if (len_trim(error_msg) > 0) return
            sym = find_symbol(context, id_name)
            if (sym <= 0) then
                error_msg = 'range argument was not declared: '//trim(id_name)
                return
            end if
            arg_is_real = context%symbols(sym)%value_kind == VALUE_F32 .or. &
                          context%symbols(sym)%value_kind == VALUE_F64
            k = value_kind_number(context%symbols(sym)%value_kind)
            if (k < 0) then
                error_msg = 'range argument has no compile-time kind: '// &
                            trim(id_name)
                return
            end if
        else
            error_msg = 'range argument is not a literal or named entity'
            return
        end if

        if (arg_is_real) then
            select case (k)
            case (4_c_int64_t)
                constant_value = 37_c_int64_t
            case (8_c_int64_t)
                constant_value = 307_c_int64_t
            case default
                error_msg = 'range argument kind is not a supported real kind'
                return
            end select
        else
            select case (k)
            case (1_c_int64_t)
                constant_value = 2_c_int64_t
            case (2_c_int64_t)
                constant_value = 4_c_int64_t
            case (4_c_int64_t)
                constant_value = 9_c_int64_t
            case (8_c_int64_t)
                constant_value = 18_c_int64_t
            case default
                error_msg = 'range argument kind is not a supported integer kind'
                return
            end select
        end if
        call set_empty(error_msg)
    end subroutine eval_range_constant

    ! Constant-fold selected_char_kind(name), matching the kind numbers
    ! gfortran reports for the character sets it recognizes.
    subroutine eval_selected_char_kind_constant(arena, node, constant_value, &
                                                error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        integer(c_int64_t), intent(out) :: constant_value
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: lit_value, lit_type, stripped, lo

        constant_value = -1_c_int64_t
        if (.not. allocated(node%arg_indices) .or. size(node%arg_indices) /= 1) then
            error_msg = 'selected_char_kind requires one argument'
            return
        end if
        if (.not. is_literal(arena, node%arg_indices(1))) then
            error_msg = 'selected_char_kind argument is not a character literal'
            return
        end if
        if (.not. is_character_literal(arena, node%arg_indices(1))) then
            error_msg = 'selected_char_kind argument is not a character literal'
            return
        end if
        call get_literal_info(arena, node%arg_indices(1), lit_value, lit_type, &
                              error_msg)
        if (len_trim(error_msg) > 0) return
        call strip_literal_quotes(lit_value, stripped)
        lo = lowercase_text(trim(stripped))
        select case (lo)
        case ('default', 'ascii')
            constant_value = 1_c_int64_t
        case ('iso_10646')
            constant_value = 4_c_int64_t
        case default
            constant_value = -1_c_int64_t
        end select
        call set_empty(error_msg)
    end subroutine eval_selected_char_kind_constant
