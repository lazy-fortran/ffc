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
        else
            base = lowercase_text(trim(adjustl(type_name)))
            call set_empty(kind_spec)
        end if

        select case (base)
        case ('integer')
            if (integer_kind_is_i32(kind_spec)) then
                value_kind = VALUE_I32
            else
                call unsupported_feature_error('integer kind', line, column, &
                    'direct LIRIC session only supports default/c_int/'// &
                    'c_int32_t (32-bit) integer kinds', error_msg)
            end if
        case ('real')
            if (real_kind_is_f64(kind_spec)) then
                value_kind = VALUE_F64
            else
                value_kind = VALUE_F32
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

    logical function integer_kind_is_i32(kind_spec)
        character(len=*), intent(in) :: kind_spec

        select case (trim(kind_spec))
        case ('', '4', 'c_int', 'c_int32_t', 'c_bool', 'c_char')
            integer_kind_is_i32 = .true.
        case default
            integer_kind_is_i32 = .false.
        end select
    end function integer_kind_is_i32

    ! Returns true when the real kind spec selects f64 (kind 8 or higher).
    ! Bare real and real(4)/c_float/real32 are f32.
    logical function real_kind_is_f64(kind_spec)
        character(len=*), intent(in) :: kind_spec

        select case (trim(kind_spec))
        case ('8', 'real64', 'c_double', 'c_long_double')
            real_kind_is_f64 = .true.
        case default
            real_kind_is_f64 = .false.
        end select
    end function real_kind_is_f64

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
