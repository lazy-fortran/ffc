    subroutine lower_constant_declaration(node, context, value_kind, error_msg)
        type(declaration_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: value_kind
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: name
        integer(c_int64_t) :: constant_value
        integer :: index

        if (node%is_array) then
            call unsupported_feature_error('array parameter declaration', &
                                           node%line, node%column, &
                                           'array parameters are not supported '// &
                                           'by direct LIRIC session', error_msg)
            return
        end if
        if (node%is_multi_declaration) then
            call unsupported_feature_error('parameter declaration', &
                                           node%line, node%column, &
                                           'direct LIRIC session only supports '// &
                                           'single scalar parameters', error_msg)
            return
        end if
        if (.not. allocated(node%var_name)) then
            error_msg = 'parameter declaration did not expose a variable name'
            return
        end if
        if (.not. node%has_initializer .or. node%initializer_index <= 0) then
            error_msg = 'parameter requires a compile-time initializer: '// &
                        trim(node%var_name)
            return
        end if

        ! Real/logical/character named constants bind a typed value into the
        ! symbol; integer scalars stay on the dedicated path because the
        ! derived-type bound pre-pass caches their folded i32 constant.
        if (value_kind /= VALUE_I32) then
            call lower_nonint_constant_declaration(node, context, value_kind, &
                                                   error_msg)
            return
        end if

        name = node%var_name
        index = find_symbol(context, name)
        if (index > 0) then
            ! Already collected by the parameter pre-pass (used for derived-type
            ! array component bounds). Re-declaring the same integer parameter is
            ! a no-op; a genuine clash is a different symbol kind.
            if (context%symbols(index)%is_parameter .and. &
                context%symbols(index)%has_i32_constant) then
                call set_empty(error_msg)
                return
            end if
            error_msg = 'duplicate parameter declaration: '//trim(name)
            return
        end if
     call grow_symbols(context)

        call eval_i32_constant(context%arena, node%initializer_index, context, &
                               constant_value, error_msg)
        if (len_trim(error_msg) > 0) return

        index = context%symbol_count + 1
        context%symbols(index)%name = trim(name)
        context%symbols(index)%value_kind = VALUE_I32
        context%symbols(index)%value = i32_immediate(context%session, constant_value)
        context%symbols(index)%is_parameter = .true.
        context%symbols(index)%has_i32_constant = .true.
        context%symbols(index)%i32_constant = constant_value
        context%symbol_count = index
        call set_empty(error_msg)
    end subroutine lower_constant_declaration

    subroutine lower_nonint_constant_declaration(node, context, value_kind, &
                                                 error_msg)
        ! Bind a scalar REAL/LOGICAL/CHARACTER (or non-default integer) named
        ! constant: define a symbol of the declared kind, fold the literal
        ! initializer into its value operand, and mark it a parameter so later
        ! references read the constant directly.
        type(declaration_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: value_kind
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: name
        integer :: index

        name = node%var_name
        if (find_symbol(context, name) > 0) then
            error_msg = 'duplicate parameter declaration: '//trim(name)
            return
        end if

        if (value_kind == VALUE_CHARACTER) then
            call lower_character_constant_declaration(node, context, name, &
                                                      error_msg)
            return
        end if

        call define_symbol(context, name, value_kind, error_msg)
        if (len_trim(error_msg) > 0) return
        index = find_symbol(context, name)
        if (index <= 0) then
            error_msg = 'parameter declaration did not register a symbol: '// &
                        trim(name)
            return
        end if
        context%symbols(index)%is_parameter = .true.
        call lower_scalar_initializer(context, name, value_kind, &
                                      node%initializer_index, error_msg)
        if (len_trim(error_msg) > 0) return
        ! A non-default-kind integer parameter (integer(int64) etc.) still has a
        ! compile-time integer value; record it so array bounds and kind/size/len
        ! inquiries that reference it fold like a default integer parameter.
        if (value_kind == VALUE_I8 .or. value_kind == VALUE_I16 .or. &
            value_kind == VALUE_I64) then
            block
                integer(c_int64_t) :: int_const
                character(len=:), allocatable :: fold_err
                call eval_i32_constant(context%arena, node%initializer_index, &
                                       context, int_const, fold_err)
                if (len_trim(fold_err) == 0) then
                    context%symbols(index)%has_i32_constant = .true.
                    context%symbols(index)%i32_constant = int_const
                end if
            end block
        end if
    end subroutine lower_nonint_constant_declaration

    subroutine lower_character_constant_declaration(node, context, name, error_msg)
        ! A character named constant with an assumed length
        ! (character(len=*), parameter :: s = "Hello") takes its length from
        ! the folded initializer text. A declared length
        ! (character(len=N), parameter :: s = "Hello") instead pads or
        ! truncates the folded text to N, matching gfortran's fixed-length
        ! assignment. The initializer may reference an earlier character
        ! named constant (z = x // y), not just literals. The value operand
        ! holds a materialized LIRIC string so references and print read it
        ! directly.
        type(declaration_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: name
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: literal_text
        character(len=64) :: string_name
        logical :: fold_ok
        integer :: index, declared_length

        call fold_char_constant_expr(context, node%initializer_index, &
                                     literal_text, fold_ok)
        if (.not. fold_ok) then
            call unsupported_feature_error('character parameter declaration', &
                node%line, node%column, 'only character-literal initializers '// &
                'are supported for named constants', error_msg)
            return
        end if

        call declaration_character_length(context, node, declared_length, &
                                          error_msg)
        if (len_trim(error_msg) > 0) return
        if (declared_length > 0) then
            call normalize_character_literal(literal_text, declared_length)
        end if

        call define_character_symbol(context, name, len(literal_text), error_msg)
        if (len_trim(error_msg) > 0) return
        index = find_symbol(context, name)
        if (index <= 0) then
            error_msg = 'character parameter did not register a symbol: '// &
                        trim(name)
            return
        end if

        context%string_literal_count = context%string_literal_count + 1
        write (string_name, '(A,I0)') '.ffc.char.', context%string_literal_count
        call materialize_liric_string(context%session, trim(string_name), &
                                      literal_text, &
                                      context%symbols(index)%value, error_msg)
        if (len_trim(error_msg) > 0) return
        context%symbols(index)%has_character_value = .true.
        context%symbols(index)%is_parameter = .true.
        context%symbols(index)%character_constant_text = literal_text
        call set_empty(error_msg)
    end subroutine lower_character_constant_declaration

    recursive subroutine fold_char_constant_expr(context, node_index, &
                                                 accumulated, ok)
        ! Fold a character-parameter initializer (a literal, a // chain of
        ! literals and/or earlier character named constants) into its
        ! compile-time text.
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: node_index
        character(len=:), allocatable, intent(out) :: accumulated
        logical, intent(out) :: ok
        character(len=:), allocatable :: left_text, right_text
        character(len=:), allocatable :: bin_op, bin_err
        integer :: bin_left, bin_right, bin_line, bin_col
        character(len=:), allocatable :: id_name, id_err
        integer :: symbol_index

        ok = .false.
        if (.not. node_exists(context%arena, node_index)) return

        if (is_binary_op(context%arena, node_index)) then
            call get_binary_op_info(context%arena, node_index, bin_op, &
                                    bin_left, bin_right, bin_line, bin_col, &
                                    bin_err)
            if (len_trim(bin_err) > 0) return
            if (trim(bin_op) /= '//') return
            call fold_char_constant_expr(context, bin_left, left_text, ok)
            if (.not. ok) return
            call fold_char_constant_expr(context, bin_right, right_text, ok)
            if (.not. ok) return
            accumulated = left_text//right_text
            ok = .true.
            return
        end if

        if (is_identifier(context%arena, node_index)) then
            call get_identifier_name(context%arena, node_index, id_name, &
                                     id_err)
            if (len_trim(id_err) > 0) return
            symbol_index = find_symbol(context, id_name)
            if (symbol_index <= 0) return
            if (.not. &
                allocated(context%symbols(symbol_index)%character_constant_text)) &
                return
            accumulated = context%symbols(symbol_index)%character_constant_text
            ok = .true.
            return
        end if

        call concat_character_literals(context%arena, node_index, &
                                       accumulated, ok)
    end subroutine fold_char_constant_expr

    subroutine type_name_value_kind(type_name, line, column, value_kind, error_msg)
        character(len=*), intent(in) :: type_name
        integer, intent(in) :: line, column
        integer, intent(out) :: value_kind
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: base
        character(len=:), allocatable :: kind_spec
        integer :: paren

        value_kind = VALUE_I32
        call set_empty(error_msg)
        if (is_character_type_name(type_name)) then
            value_kind = VALUE_CHARACTER
            return
        end if

        ! Split "integer(kind)" into the base type and the kind spec, which may
        ! be a numeric literal (4, 8) or an iso_c_binding kind name (c_int, ...).
        paren = index(type_name, '(')
        if (paren > 0) then
            base = lowercase_text(trim(adjustl(type_name(1:paren - 1))))
            kind_spec = lowercase_text(trim(adjustl( &
                type_name(paren + 1:index(type_name, ')', back=.true.) - 1))))
            ! Normalize the explicit "kind=" keyword form so kind-name matching
            ! below sees the bare selector (kind=8 -> 8, real(kind=8) -> 8).
            if (index(kind_spec, 'kind=') == 1) then
                kind_spec = trim(adjustl(kind_spec(6:)))
            end if
        else
            base = lowercase_text(trim(adjustl(type_name)))
            call set_empty(kind_spec)
        end if

        select case (base)
        case ('integer')
            call integer_kind_to_value_kind(kind_spec, line, column, &
                                            value_kind, error_msg)
        case ('real')
            if (real_kind_is_f64(kind_spec)) then
                value_kind = VALUE_F64
            else
                value_kind = VALUE_F32
            end if
        case ('complex')
            if (complex_kind_is_c8(kind_spec)) then
                value_kind = VALUE_C8
            else
                value_kind = VALUE_C4
            end if
        case ('double precision')
            value_kind = VALUE_F64
        case ('logical')
            value_kind = VALUE_LOGICAL
        case ('type')
            ! c_ptr is an opaque 8-byte pointer value; c_loc/c_associated/
            ! c_f_pointer operate on it (#139).
            if (trim(kind_spec) == 'c_ptr') then
                value_kind = VALUE_C_PTR
            else
                call unsupported_feature_error('scalar type', line, column, &
                    'direct LIRIC session only supports integer, real, '// &
                    'logical, character, and c_ptr scalars', error_msg)
            end if
        case ('class')
            ! Unlimited polymorphic class(*); other class(name) forms are not
            ! supported yet (#141).
            if (trim(kind_spec) == '*') then
                value_kind = VALUE_CLASS_STAR
            else
                call unsupported_feature_error('scalar type', line, column, &
                    'direct LIRIC session supports class(*) but not '// &
                    'class('//trim(kind_spec)//')', error_msg)
            end if
        case default
            call unsupported_feature_error('scalar type', line, column, &
                                           'direct LIRIC session only supports '// &
                                           'integer, real, logical, and character '// &
                                           'scalars', error_msg)
        end select
    end subroutine type_name_value_kind

    subroutine integer_kind_to_value_kind(kind_spec, line, column, &
                                          value_kind, error_msg)
        character(len=*), intent(in) :: kind_spec
        integer, intent(in) :: line, column
        integer, intent(out) :: value_kind
        character(len=:), allocatable, intent(out) :: error_msg

        call set_empty(error_msg)
        select case (trim(kind_spec))
        case ('1', 'int8', 'c_int8_t', 'c_signed_char', &
              'c_int_least8_t', 'c_int_fast8_t')
            value_kind = VALUE_I8
        case ('2', 'int16', 'c_int16_t', 'c_short', &
              'c_int_least16_t', 'c_int_fast16_t')
            value_kind = VALUE_I16
        case ('', '4', 'c_int', 'c_int32_t', 'c_bool', 'c_char', 'int32', &
              'c_int_least32_t', 'c_int_fast32_t')
            value_kind = VALUE_I32
        case ('8', 'int64', 'c_int64_t', 'c_long', 'c_long_long', &
              'c_int_least64_t', 'c_int_fast64_t', &
              'c_size_t', 'c_intptr_t', 'c_ptrdiff_t', 'c_intmax_t')
            ! LP64 Linux: size_t/intptr_t/ptrdiff_t/intmax_t are 8 bytes,
            ! matching the liric session's native 64-bit target.
            value_kind = VALUE_I64
        case default
            value_kind = VALUE_I32
            call unsupported_feature_error('integer kind', line, column, &
                'direct LIRIC session supports integer kinds 1 (int8), '// &
                '2 (int16), 4 (default), and 8 (int64); kind '// &
                trim(kind_spec)//' is not yet supported', &
                error_msg)
        end select
    end subroutine integer_kind_to_value_kind

    ! Returns true when the real kind spec selects f64 (kind 8 or higher).
    ! Bare real and real(4)/c_float/real32 are f32.
    logical function real_kind_is_f64(kind_spec)
        character(len=*), intent(in) :: kind_spec

        select case (trim(kind_spec))
        case ('8', 'real64', 'c_double', 'c_long_double', 'dp', 'wp')
            ! dp/wp are the conventional double-precision kind parameters
            ! (dp/wp => real64 or selected_real_kind(15)); sp/default stay f32.
            real_kind_is_f64 = .true.
        case default
            real_kind_is_f64 = .false.
        end select
    end function real_kind_is_f64

    ! Returns true when the complex kind spec selects double precision (kind 8).
    ! Bare complex and complex(4) are c4 (two f32s).
    logical function complex_kind_is_c8(kind_spec)
        character(len=*), intent(in) :: kind_spec

        select case (trim(kind_spec))
        case ('8', 'real64', 'c_double', 'c_double_complex')
            complex_kind_is_c8 = .true.
        case default
            complex_kind_is_c8 = .false.
        end select
    end function complex_kind_is_c8

    subroutine declaration_value_kind(node, value_kind, error_msg)
        type(declaration_node), intent(in) :: node
        integer, intent(out) :: value_kind
        character(len=:), allocatable, intent(out) :: error_msg

        value_kind = VALUE_I32
        call set_empty(error_msg)
        if (.not. allocated(node%type_name)) return
        call type_name_value_kind(node%type_name, node%line, node%column, &
                                  value_kind, error_msg)
    end subroutine declaration_value_kind
