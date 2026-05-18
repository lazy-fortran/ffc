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
        if (value_kind /= VALUE_I32) then
            call unsupported_feature_error('parameter declaration', &
                                           node%line, node%column, &
                                           'direct LIRIC session only supports '// &
                                           'integer scalar parameters', error_msg)
            return
        end if
        if (node%is_multi_declaration) then
            call unsupported_feature_error('parameter declaration', &
                                           node%line, node%column, &
                                           'direct LIRIC session only supports '// &
                                           'single integer parameters', error_msg)
            return
        end if
        if (.not. allocated(node%var_name)) then
            error_msg = 'parameter declaration did not expose a variable name'
            return
        end if
        if (.not. node%has_initializer .or. node%initializer_index <= 0) then
            error_msg = 'integer parameter requires a compile-time initializer: '// &
                        trim(node%var_name)
            return
        end if

        name = node%var_name
        if (find_symbol(context, name) > 0) then
            error_msg = 'duplicate parameter declaration: '//trim(name)
            return
        end if
        if (context%symbol_count >= MAX_SYMBOLS) then
            error_msg = 'too many symbols for ffc direct-session lowering'
            return
        end if

        call eval_i32_constant(context%arena, node%initializer_index, context, &
                               constant_value, error_msg)
        if (len_trim(error_msg) > 0) return

        index = context%symbol_count + 1
        context%symbols(index)%name = trim(name)
        context%symbols(index)%value_kind = VALUE_I32
        context%symbols(index)%value = context%session%i32_immediate(constant_value)
        context%symbols(index)%is_parameter = .true.
        context%symbols(index)%has_i32_constant = .true.
        context%symbols(index)%i32_constant = constant_value
        context%symbol_count = index
        call set_empty(error_msg)
    end subroutine lower_constant_declaration

    subroutine type_name_value_kind(type_name, line, column, value_kind, error_msg)
        character(len=*), intent(in) :: type_name
        integer, intent(in) :: line, column
        integer, intent(out) :: value_kind
        character(len=:), allocatable, intent(out) :: error_msg
        value_kind = VALUE_I32
        call set_empty(error_msg)
        if (is_character_type_name(type_name)) then
            value_kind = VALUE_CHARACTER
            return
        end if
        select case (trim(type_name))
        case ('integer')
            value_kind = VALUE_I32
        case ('real')
            value_kind = VALUE_F64
        case ('logical')
            value_kind = VALUE_LOGICAL
        case default
            call unsupported_feature_error('scalar type', line, column, &
                                           'direct LIRIC session only supports '// &
                                           'integer, real, logical, and character '// &
                                           'scalars', error_msg)
        end select
    end subroutine type_name_value_kind

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
