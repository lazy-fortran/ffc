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
            call kind_of_literal(arena, arg_index, context, k, error_msg)
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
            call kind_of_literal(arena, arg_index, context, k, error_msg)
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

    ! Constant-fold a one-argument integer intrinsic whose Fortran generic
    ! name and semantics match the target intrinsic exactly (abs, not), so
    ! the host Fortran intrinsic performs the fold directly.
    subroutine eval_one_arg_i32_intrinsic(arena, node, context, name, &
                                          constant_value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: name
        integer(c_int64_t), intent(out) :: constant_value
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int64_t) :: a

        constant_value = 0_c_int64_t
        if (.not. allocated(node%arg_indices) .or. size(node%arg_indices) /= 1) then
            error_msg = trim(name)//' requires one argument'
            return
        end if
        call eval_i32_constant(arena, node%arg_indices(1), context, a, error_msg)
        if (len_trim(error_msg) > 0) return
        select case (name)
        case ('abs')
            constant_value = abs(a)
        case ('not')
            constant_value = not(a)
        case default
            error_msg = 'unsupported one-argument compile-time integer call: '// &
                        trim(name)
            return
        end select
        call set_empty(error_msg)
    end subroutine eval_one_arg_i32_intrinsic

    ! Constant-fold a two-argument integer intrinsic whose Fortran generic
    ! name and semantics match the target intrinsic exactly (mod, modulo,
    ! sign, dim, iand, ior, ieor/xor, ishft, ishftc, ibset, ibclr).
    subroutine eval_two_arg_i32_intrinsic(arena, node, context, name, &
                                          constant_value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: name
        integer(c_int64_t), intent(out) :: constant_value
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int64_t) :: a, b

        constant_value = 0_c_int64_t
        if (.not. allocated(node%arg_indices) .or. size(node%arg_indices) /= 2) then
            error_msg = trim(name)//' requires two arguments'
            return
        end if
        call eval_i32_constant(arena, node%arg_indices(1), context, a, error_msg)
        if (len_trim(error_msg) > 0) return
        call eval_i32_constant(arena, node%arg_indices(2), context, b, error_msg)
        if (len_trim(error_msg) > 0) return
        select case (name)
        case ('mod')
            if (b == 0_c_int64_t) then
                error_msg = 'mod by zero is not a compile-time integer constant'
                return
            end if
            constant_value = mod(a, b)
        case ('modulo')
            if (b == 0_c_int64_t) then
                error_msg = 'modulo by zero is not a compile-time integer constant'
                return
            end if
            constant_value = modulo(a, b)
        case ('sign')
            constant_value = sign(a, b)
        case ('dim')
            constant_value = dim(a, b)
        case ('iand')
            constant_value = iand(a, b)
        case ('ior')
            constant_value = ior(a, b)
        case ('ieor', 'xor')
            constant_value = ieor(a, b)
        case ('ishft')
            constant_value = ishft(a, int(b))
        case ('ishftc')
            constant_value = ishftc(a, int(b))
        case ('ibset')
            constant_value = ibset(a, int(b))
        case ('ibclr')
            constant_value = ibclr(a, int(b))
        case default
            error_msg = 'unsupported two-argument compile-time integer call: '// &
                        trim(name)
            return
        end select
        call set_empty(error_msg)
    end subroutine eval_two_arg_i32_intrinsic

    ! Constant-fold ibits(i, pos, len): extract len bits of i starting at
    ! bit position pos, matching the IBITS intrinsic exactly.
    subroutine eval_ibits_constant(arena, node, context, constant_value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        integer(c_int64_t), intent(out) :: constant_value
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int64_t) :: i_val, pos_val, len_val

        constant_value = 0_c_int64_t
        if (.not. allocated(node%arg_indices) .or. size(node%arg_indices) /= 3) then
            error_msg = 'ibits requires three arguments'
            return
        end if
        call eval_i32_constant(arena, node%arg_indices(1), context, i_val, error_msg)
        if (len_trim(error_msg) > 0) return
        call eval_i32_constant(arena, node%arg_indices(2), context, pos_val, error_msg)
        if (len_trim(error_msg) > 0) return
        call eval_i32_constant(arena, node%arg_indices(3), context, len_val, error_msg)
        if (len_trim(error_msg) > 0) return
        constant_value = ibits(i_val, int(pos_val), int(len_val))
        call set_empty(error_msg)
    end subroutine eval_ibits_constant

    ! Constant-fold merge(tsource, fsource, mask) for a scalar compile-time
    ! integer mask (a comparison or a folded logical literal, both 0/1).
    ! mask may arrive positionally or as a keyword argument (mask=..., an
    ! assignment_node); eval_i32_constant already unwraps the keyword form,
    ! so the positional order (tsource, fsource, mask) always applies.
    subroutine eval_merge_constant(arena, node, context, constant_value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        integer(c_int64_t), intent(out) :: constant_value
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int64_t) :: tsource, fsource, mask

        constant_value = 0_c_int64_t
        if (.not. allocated(node%arg_indices) .or. size(node%arg_indices) /= 3) then
            error_msg = 'merge requires three arguments'
            return
        end if
        call eval_i32_constant(arena, node%arg_indices(1), context, tsource, &
                               error_msg)
        if (len_trim(error_msg) > 0) return
        call eval_i32_constant(arena, node%arg_indices(2), context, fsource, &
                               error_msg)
        if (len_trim(error_msg) > 0) return
        call eval_i32_constant(arena, node%arg_indices(3), context, mask, &
                               error_msg)
        if (len_trim(error_msg) > 0) return
        constant_value = merge(tsource, fsource, mask /= 0_c_int64_t)
        call set_empty(error_msg)
    end subroutine eval_merge_constant

    ! Constant-fold digits(x): the number of significant base-radix digits
    ! in x's kind (mantissa bits for real, value bits for integer).
    subroutine eval_digits_constant(arena, node, context, constant_value, error_msg)
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
            error_msg = 'digits requires one argument'
            return
        end if
        arg_index = node%arg_indices(1)
        if (is_literal(arena, arg_index)) then
            arg_is_real = is_real_literal(arena, arg_index)
            call kind_of_literal(arena, arg_index, context, k, error_msg)
            if (len_trim(error_msg) > 0) return
        else if (is_identifier(arena, arg_index)) then
            call get_identifier_name(arena, arg_index, id_name, error_msg)
            if (len_trim(error_msg) > 0) return
            sym = find_symbol(context, id_name)
            if (sym <= 0) then
                error_msg = 'digits argument was not declared: '//trim(id_name)
                return
            end if
            arg_is_real = context%symbols(sym)%value_kind == VALUE_F32 .or. &
                          context%symbols(sym)%value_kind == VALUE_F64
            k = value_kind_number(context%symbols(sym)%value_kind)
            if (k < 0) then
                error_msg = 'digits argument has no compile-time kind: '// &
                            trim(id_name)
                return
            end if
        else
            error_msg = 'digits argument is not a literal or named entity'
            return
        end if

        if (arg_is_real) then
            select case (k)
            case (4_c_int64_t)
                constant_value = 24_c_int64_t
            case (8_c_int64_t)
                constant_value = 53_c_int64_t
            case default
                error_msg = 'digits argument kind is not a supported real kind'
                return
            end select
        else
            select case (k)
            case (1_c_int64_t)
                constant_value = 7_c_int64_t
            case (2_c_int64_t)
                constant_value = 15_c_int64_t
            case (4_c_int64_t)
                constant_value = 31_c_int64_t
            case (8_c_int64_t)
                constant_value = 63_c_int64_t
            case default
                error_msg = 'digits argument kind is not a supported integer kind'
                return
            end select
        end if
        call set_empty(error_msg)
    end subroutine eval_digits_constant

    ! Constant-fold radix(x): 2 for every integer and real kind ffc supports.
    subroutine eval_radix_constant(arena, node, constant_value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        integer(c_int64_t), intent(out) :: constant_value
        character(len=:), allocatable, intent(out) :: error_msg

        constant_value = 2_c_int64_t
        if (.not. allocated(node%arg_indices) .or. size(node%arg_indices) /= 1) then
            error_msg = 'radix requires one argument'
            return
        end if
        if (.not. node_exists(arena, node%arg_indices(1))) then
            error_msg = 'radix argument is not a valid expression'
            return
        end if
        call set_empty(error_msg)
    end subroutine eval_radix_constant

    ! Constant-fold minexponent(x)/maxexponent(x): the minimum/maximum
    ! binary exponent representable in x's real kind.
    subroutine eval_exponent_range_constant(arena, node, context, want_max, &
                                            constant_value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        logical, intent(in) :: want_max
        integer(c_int64_t), intent(out) :: constant_value
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int64_t) :: k

        constant_value = 0_c_int64_t
        if (.not. allocated(node%arg_indices) .or. size(node%arg_indices) /= 1) then
            error_msg = 'minexponent/maxexponent requires one argument'
            return
        end if
        call constant_arg_kind_number(arena, context, node%arg_indices(1), k, &
                                      error_msg)
        if (len_trim(error_msg) > 0) return
        select case (k)
        case (4_c_int64_t)
            constant_value = merge(128_c_int64_t, -125_c_int64_t, want_max)
        case (8_c_int64_t)
            constant_value = merge(1024_c_int64_t, -1021_c_int64_t, want_max)
        case default
            error_msg = 'minexponent/maxexponent argument kind is not a '// &
                        'supported real kind'
            return
        end select
        call set_empty(error_msg)
    end subroutine eval_exponent_range_constant

    ! Constant-fold selected_logical_kind(bits): ffc only ever materialises
    ! default logical (kind 4), matching gfortran's kind for any bit count
    ! it can satisfy.
    subroutine eval_selected_logical_kind_constant(arena, node, context, &
                                                    constant_value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        integer(c_int64_t), intent(out) :: constant_value
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int64_t) :: bits

        constant_value = -1_c_int64_t
        if (.not. allocated(node%arg_indices) .or. size(node%arg_indices) /= 1) then
            error_msg = 'selected_logical_kind requires one argument'
            return
        end if
        call eval_i32_constant(arena, node%arg_indices(1), context, bits, error_msg)
        if (len_trim(error_msg) > 0) return
        if (bits <= 32_c_int64_t) then
            constant_value = 4_c_int64_t
        else
            constant_value = -1_c_int64_t
        end if
        call set_empty(error_msg)
    end subroutine eval_selected_logical_kind_constant

    ! Declaration of name's array PARAMETER, anywhere in the unit, if one
    ! exists with a compile-time array-constructor initializer.
    function find_integer_parameter_array_decl(arena, name) result(decl)
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: name
        type(declaration_node), pointer :: decl
        type(declaration_node), pointer :: candidate
        integer :: n

        decl => null()
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            candidate => declaration_named(arena, n, name)
            if (.not. associated(candidate)) cycle
            if (.not. candidate%is_array) cycle
            if (.not. candidate%is_parameter) cycle
            if (.not. candidate%has_initializer) cycle
            if (candidate%initializer_index <= 0) cycle
            decl => candidate
            return
        end do
    end function find_integer_parameter_array_decl

    recursive subroutine fold_i32_array_literal(arena, context, node_index, &
                                                values, error_msg)
        ! Fold a compile-time integer array: an array constructor of
        ! foldable elements, or an identifier naming an integer PARAMETER
        ! array (recurses into that array's own constructor initializer).
        type(ast_arena_t), intent(in) :: arena
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: node_index
        integer(c_int64_t), allocatable, intent(out) :: values(:)
        character(len=:), allocatable, intent(out) :: error_msg
        integer, allocatable :: flat(:)
        character(len=:), allocatable :: id_name
        integer :: i, n

        if (.not. node_exists(arena, node_index)) then
            error_msg = 'array argument index does not reference an AST node'
            return
        end if

        select type (elem_node => arena%entries(node_index)%node)
        type is (array_literal_node)
            if (.not. allocated(elem_node%element_indices)) then
                error_msg = 'array constructor has no elements'
                return
            end if
            allocate (flat(0))
            call flatten_constructor_elements(arena, elem_node%element_indices, &
                                              flat, error_msg)
            if (len_trim(error_msg) > 0) return
            n = size(flat)
            allocate (values(n))
            do i = 1, n
                call eval_i32_constant(arena, flat(i), context, values(i), error_msg)
                if (len_trim(error_msg) > 0) return
            end do
            call set_empty(error_msg)
            return
        end select

        if (is_identifier(arena, node_index)) then
            call get_identifier_name(arena, node_index, id_name, error_msg)
            if (len_trim(error_msg) > 0) return
            call fold_i32_array_literal_by_name(arena, context, id_name, values, &
                                                error_msg)
            return
        end if

        error_msg = 'array argument is not a compile-time integer array constant'
    end subroutine fold_i32_array_literal

    recursive subroutine fold_i32_array_literal_by_name(arena, context, name, &
                                                         values, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: name
        integer(c_int64_t), allocatable, intent(out) :: values(:)
        character(len=:), allocatable, intent(out) :: error_msg
        type(declaration_node), pointer :: decl

        decl => find_integer_parameter_array_decl(context%arena, name)
        if (.not. associated(decl)) then
            error_msg = 'compile-time integer array parameter was not declared: '// &
                        trim(name)
            return
        end if
        call fold_i32_array_literal(arena, context, decl%initializer_index, &
                                    values, error_msg)
    end subroutine fold_i32_array_literal_by_name

    ! call_or_subscript_node whose name is not a known intrinsic may index
    ! into an integer PARAMETER array (arr(k)); fold the whole array from
    ! its declaration and read one element.
    subroutine fold_param_array_element_from_arena(arena, node, context, &
                                                    constant_value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        integer(c_int64_t), intent(out) :: constant_value
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int64_t), allocatable :: values(:)
        integer(c_int64_t) :: idx

        constant_value = 0_c_int64_t
        if (.not. allocated(node%name)) then
            error_msg = 'compile-time call has no callee name'
            return
        end if
        if (.not. allocated(node%arg_indices) .or. size(node%arg_indices) /= 1) then
            error_msg = 'compile-time integer array parameter was not declared: '// &
                        trim(node%name)
            return
        end if
        call fold_i32_array_literal_by_name(arena, context, node%name, values, &
                                            error_msg)
        if (len_trim(error_msg) > 0) return
        call eval_i32_constant(arena, node%arg_indices(1), context, idx, error_msg)
        if (len_trim(error_msg) > 0) return
        if (idx < 1_c_int64_t .or. idx > int(size(values), c_int64_t)) then
            error_msg = 'parameter array index out of range: '//trim(node%name)
            return
        end if
        constant_value = values(int(idx))
        call set_empty(error_msg)
    end subroutine fold_param_array_element_from_arena

    ! Constant-fold product(array): product of a compile-time integer array
    ! constructor or named integer PARAMETER array.
    subroutine eval_product_constant(arena, node, context, constant_value, &
                                     error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        integer(c_int64_t), intent(out) :: constant_value
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int64_t), allocatable :: values(:)
        integer :: i

        constant_value = 1_c_int64_t
        if (.not. allocated(node%arg_indices) .or. size(node%arg_indices) /= 1) then
            error_msg = 'product requires one array argument'
            return
        end if
        call fold_i32_array_literal(arena, context, node%arg_indices(1), values, &
                                    error_msg)
        if (len_trim(error_msg) > 0) return
        do i = 1, size(values)
            constant_value = constant_value*values(i)
        end do
        call set_empty(error_msg)
    end subroutine eval_product_constant

    ! Constant-fold dot_product(a, b): sum of elementwise products of two
    ! same-length compile-time integer arrays.
    subroutine eval_dot_product_constant(arena, node, context, constant_value, &
                                         error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        integer(c_int64_t), intent(out) :: constant_value
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int64_t), allocatable :: a_values(:), b_values(:)
        integer :: i

        constant_value = 0_c_int64_t
        if (.not. allocated(node%arg_indices) .or. size(node%arg_indices) /= 2) then
            error_msg = 'dot_product requires two array arguments'
            return
        end if
        call fold_i32_array_literal(arena, context, node%arg_indices(1), &
                                    a_values, error_msg)
        if (len_trim(error_msg) > 0) return
        call fold_i32_array_literal(arena, context, node%arg_indices(2), &
                                    b_values, error_msg)
        if (len_trim(error_msg) > 0) return
        if (size(a_values) /= size(b_values)) then
            error_msg = 'dot_product argument arrays have mismatched size'
            return
        end if
        do i = 1, size(a_values)
            constant_value = constant_value + a_values(i)*b_values(i)
        end do
        call set_empty(error_msg)
    end subroutine eval_dot_product_constant
