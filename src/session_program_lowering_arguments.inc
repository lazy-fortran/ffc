    subroutine store_reference_value(context, symbol_index, value, error_msg)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        type(lr_operand_desc_t), intent(in) :: value
        character(len=:), allocatable, intent(out) :: error_msg

        select case (context%symbols(symbol_index)%value_kind)
        case (VALUE_I32, VALUE_LOGICAL)
            if (.not. emit_i32_store(context%session,  &
                value, context%symbols(symbol_index)%address, error_msg)) return
        case (VALUE_F32)
            if (.not. emit_liric_f32_store( &
                context%session, value, context%symbols(symbol_index)%address, &
                error_msg)) return
        case (VALUE_F64)
            if (.not. emit_liric_f64_store( &
                context%session, value, context%symbols(symbol_index)%address, &
                error_msg)) return
        case default
            error_msg = 'direct LIRIC session only stores numeric reference arguments'
            return
        end select
        call set_empty(error_msg)
    end subroutine store_reference_value

    subroutine prepare_reference_args(arena, arg_indices, context, default_kind, &
                                      callee_name, args, copyback_indices, &
                                      error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: arg_indices(:)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: default_kind
        character(len=*), intent(in) :: callee_name
        type(lr_operand_desc_t), allocatable, intent(out) :: args(:)
        integer, allocatable, intent(out) :: copyback_indices(:)
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i

        allocate (args(size(arg_indices)))
        allocate (copyback_indices(size(arg_indices)))
        copyback_indices = 0

        do i = 1, size(arg_indices)
            ! A character actual passed to a character(len=*) dummy is a single
            ! reference parameter pointing at a {data, length} descriptor. Use
            ! that ABI only when the callee actually declares the dummy as
            ! character(len=*); otherwise fall through to the scalar path (which
            ! rejects an unsupported character actual).
            if (actual_is_character(arena, arg_indices(i), context) .and. &
                callee_dummy_char_writable(arena, callee_name, i)) then
                ! intent(out)/(inout) allocatable character: pass the actual's
                ! own descriptor so the callee's writes land in the caller.
                call char_actual_inout_descriptor(arena, arg_indices(i), &
                                                   context, args(i), error_msg)
                if (len_trim(error_msg) > 0) return
            else if (actual_is_character(arena, arg_indices(i), context) .and. &
                     callee_dummy_is_character(arena, callee_name, i)) then
                call char_actual_descriptor(arena, arg_indices(i), context, &
                                            args(i), error_msg)
                if (len_trim(error_msg) > 0) return
            else if (actual_is_derived(arena, arg_indices(i), context)) then
                ! A derived actual is passed by reference: the base address of
                ! its component storage.
                call derived_actual_address(arena, arg_indices(i), context, &
                                             args(i), error_msg)
                if (len_trim(error_msg) > 0) return
            else if (callee_dummy_is_class_star(arena, callee_name, i)) then
                ! Box the actual into a {data, type_id} descriptor for a
                ! class(*) dummy (#141).
                call box_class_star_actual(arena, arg_indices(i), context, &
                                           args(i), error_msg)
                if (len_trim(error_msg) > 0) return
            else
                call prepare_reference_arg(arena, arg_indices(i), context, &
                                           default_kind, args(i), &
                                           copyback_indices(i), error_msg)
                if (len_trim(error_msg) > 0) return
            end if
        end do

        call set_empty(error_msg)
    end subroutine prepare_reference_args

    logical function actual_is_character(arena, node_index, context)
        ! Whether an actual argument evaluates to a character value.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        character(len=:), allocatable :: id_name, id_err
        integer :: symbol_index

        actual_is_character = .false.
        if (.not. node_exists(arena, node_index)) return
        if (is_character_literal(arena, node_index)) then
            actual_is_character = .true.
            return
        end if
        if (is_char_expr_call(arena, node_index)) then
            actual_is_character = .true.
            return
        end if
        if (is_identifier(arena, node_index)) then
            call get_identifier_name(arena, node_index, id_name, id_err)
            if (len_trim(id_err) > 0) return
            symbol_index = find_symbol(context, id_name)
            if (symbol_index > 0) actual_is_character = &
                context%symbols(symbol_index)%value_kind == VALUE_CHARACTER
        end if
    end function actual_is_character

    subroutine box_class_star_actual(arena, node_index, context, descriptor, &
                                      error_msg)
        ! Box a scalar actual into a stack {void* data; i64 type_id} descriptor
        ! and return its address (the class(*) ABI).
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: descriptor
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: value, data_addr_slot, type_slot, value_addr
        integer :: value_kind
        integer(c_int64_t) :: type_id

        value_kind = expression_value_kind(arena, node_index, context, VALUE_I32)
        select case (value_kind)
        case (VALUE_F64)
            type_id = int(TYPE_ID_REAL, c_int64_t)
        case (VALUE_LOGICAL)
            type_id = int(TYPE_ID_LOGICAL, c_int64_t)
        case (VALUE_I32)
            type_id = int(TYPE_ID_INTEGER, c_int64_t)
        case default
            error_msg = 'class(*) actual must be a scalar integer, real, or '// &
                        'logical value'
            return
        end select

        call lower_expression_by_kind(arena, node_index, context, value_kind, &
                                      value, error_msg)
        if (len_trim(error_msg) > 0) return
        ! Materialise the value in memory so the descriptor can point at it.
        call make_reference_argument(context, value_kind, value, value_addr, &
                                     error_msg)
        if (len_trim(error_msg) > 0) return
        call allocate_deferred_char_descriptor(context, descriptor, &
                                               data_addr_slot, type_slot, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_ptr_store(context%session, value_addr, data_addr_slot, &
                                 error_msg)) return
        if (.not. emit_i64_store(context%session, &
                i64_immediate(context%session, type_id), type_slot, error_msg)) &
            return
        call set_empty(error_msg)
    end subroutine box_class_star_actual

    logical function actual_is_derived(arena, node_index, context)
        ! Whether an actual argument is a derived-type variable.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        character(len=:), allocatable :: id_name, id_err
        integer :: symbol_index

        actual_is_derived = .false.
        if (.not. is_identifier(arena, node_index)) return
        call get_identifier_name(arena, node_index, id_name, id_err)
        if (len_trim(id_err) > 0) return
        symbol_index = find_symbol(context, id_name)
        if (symbol_index > 0) actual_is_derived = &
            context%symbols(symbol_index)%is_derived
    end function actual_is_derived

    subroutine derived_actual_address(arena, node_index, context, address, &
                                      error_msg)
        ! Pass a derived actual by reference: the base address of its component
        ! storage (element_address).
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: address
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: id_name
        integer :: symbol_index

        call get_identifier_name(arena, node_index, id_name, error_msg)
        if (len_trim(error_msg) > 0) return
        symbol_index = find_symbol(context, id_name)
        if (symbol_index <= 0) then
            error_msg = 'derived argument was not declared: '//trim(id_name)
            return
        end if
        address = context%symbols(symbol_index)%element_address
        call set_empty(error_msg)
    end subroutine derived_actual_address

    subroutine char_actual_inout_descriptor(arena, node_index, context, &
                                             descriptor, error_msg)
        ! For an intent(out)/(inout) allocatable character dummy, pass the
        ! address of the actual variable's own {data, length} descriptor so
        ! the callee writes through to the caller's variable.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: descriptor
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: id_name
        integer :: symbol_index

        if (.not. is_identifier(arena, node_index)) then
            error_msg = 'an intent(out) allocatable character argument must '// &
                        'be a deferred-length character variable'
            return
        end if
        call get_identifier_name(arena, node_index, id_name, error_msg)
        if (len_trim(error_msg) > 0) return
        symbol_index = find_symbol(context, id_name)
        if (symbol_index <= 0) then
            error_msg = 'character argument was not declared: '//trim(id_name)
            return
        end if
        if (.not. context%symbols(symbol_index)%is_deferred_character) then
            error_msg = 'an intent(out) allocatable character argument must '// &
                        'be a deferred-length character variable: '//trim(id_name)
            return
        end if
        ! deferred_data is the address of the descriptor's first slot, i.e. the
        ! descriptor base pointer the callee binds against.
        descriptor = context%symbols(symbol_index)%deferred_data
        call set_empty(error_msg)
    end subroutine char_actual_inout_descriptor

    subroutine char_actual_descriptor(arena, node_index, context, descriptor, &
                                       error_msg)
        ! Lower a character actual to a stack {data pointer, i64 length}
        ! descriptor and return its address. This matches the character(len=*)
        ! dummy ABI bound by bind_character_parameter_symbol.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: descriptor
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: data_ptr, length, length_i64
        type(lr_operand_desc_t) :: data_addr, len_addr

        call char_expr_operands(arena, node_index, context, data_ptr, length, &
                                error_msg)
        if (len_trim(error_msg) > 0) return
        call allocate_deferred_char_descriptor(context, descriptor, data_addr, &
                                               len_addr, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_ptr_store(context%session, data_ptr, data_addr, &
                                 error_msg)) return
        if (.not. emit_liric_i32_to_i64(context%session, length, length_i64, &
                                        error_msg)) return
        if (.not. emit_i64_store(context%session, length_i64, len_addr, &
                                 error_msg)) return
        call set_empty(error_msg)
    end subroutine char_actual_descriptor

    subroutine prepare_reference_arg(arena, node_index, context, default_kind, &
                                     arg, copyback_index, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: default_kind
        type(lr_operand_desc_t), intent(out) :: arg
        integer, intent(out) :: copyback_index
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: value
        integer :: value_kind
        logical :: handled

        copyback_index = 0
        if (.not. node_exists(arena, node_index)) then
            error_msg = 'argument index does not reference an AST node'
            return
        end if

        if (is_identifier(arena, node_index)) then
            call prepare_identifier_reference_arg(arena, node_index, context, &
                                                  arg, copyback_index, handled, &
                                                  error_msg)
            if (len_trim(error_msg) > 0 .or. handled) return
        else
            select type (node => arena%entries(node_index)%node)
            type is (call_or_subscript_node)
                if (node%is_array_access) then
                    call lower_i32_array_element_address(arena, node, context, &
                                                         arg, error_msg)
                    return
                end if
            end select
        end if

        value_kind = expression_value_kind(arena, node_index, context, default_kind)
        call lower_expression_by_kind(arena, node_index, context, value_kind, value, &
                                      error_msg)
        if (len_trim(error_msg) > 0) return
        call make_reference_argument(context, value_kind, value, arg, error_msg)
    end subroutine prepare_reference_arg

    subroutine prepare_identifier_reference_arg(arena, node_index, context, arg, &
                                                copyback_index, handled, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: arg
        integer, intent(out) :: copyback_index
        logical, intent(out) :: handled
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: symbol_index
        character(len=:), allocatable :: id_name

        copyback_index = 0
        handled = .false.
        call get_identifier_name(arena, node_index, id_name, error_msg)
        if (len_trim(error_msg) > 0) return
        symbol_index = find_symbol(context, id_name)
        if (symbol_index <= 0) then
            call set_empty(error_msg)
            return
        end if
        handled = .true.
        if (context%symbols(symbol_index)%is_array) then
            call unsupported_feature_error('array argument', &
                                           get_node_line(arena, node_index), &
                                           get_node_column(arena, node_index), &
                                           'whole-array procedure arguments '// &
                                           'are not supported', error_msg)
            return
        end if
        if (context%symbols(symbol_index)%is_reference) then
            arg = context%symbols(symbol_index)%address
            call set_empty(error_msg)
            return
        end if

        call make_reference_argument(context, &
                                     context%symbols(symbol_index)%value_kind, &
                                     context%symbols(symbol_index)%value, arg, &
                                     error_msg)
        if (len_trim(error_msg) > 0) return
        copyback_index = symbol_index
        call set_empty(error_msg)
    end subroutine prepare_identifier_reference_arg

    integer function expression_value_kind(arena, node_index, context, &
                                           default_kind) result(value_kind)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: default_kind
        integer :: symbol_index
        character(len=:), allocatable :: id_name, id_err

        value_kind = default_kind
        if (.not. node_exists(arena, node_index)) return

        if (is_binary_op(arena, node_index)) then
            if (is_f32_expression(arena, node_index, context)) then
                value_kind = VALUE_F32
            else if (is_f64_expression(arena, node_index, context)) then
                value_kind = VALUE_F64
            end if
            return
        end if

        if (is_literal(arena, node_index)) then
            if (is_real_literal(arena, node_index)) then
                ! Real literals take on the caller's expected kind (f32 or f64).
                if (default_kind == VALUE_F32) then
                    value_kind = VALUE_F32
                else
                    value_kind = VALUE_F64
                end if
            else if (is_logical_literal(arena, node_index)) then
                value_kind = VALUE_LOGICAL
            end if
            return
        end if

        if (is_identifier(arena, node_index)) then
            call get_identifier_name(arena, node_index, id_name, id_err)
            symbol_index = find_symbol(context, id_name)
            if (symbol_index > 0) value_kind = context%symbols(symbol_index)%value_kind
            return
        end if

        select type (node => arena%entries(node_index)%node)
        type is (call_or_subscript_node)
            if (allocated(node%name)) then
                if (is_contained_f32_function(context, node%name)) then
                    value_kind = VALUE_F32
                else if (is_contained_f64_function(context, node%name)) then
                    value_kind = VALUE_F64
                else if (is_contained_logical_function(context, node%name)) then
                    value_kind = VALUE_LOGICAL
                end if
            end if
        type is (component_access_node)
            value_kind = VALUE_I32
        end select
    end function expression_value_kind

    subroutine lower_expression_by_kind(arena, node_index, context, value_kind, &
                                        value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: value_kind
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg

        select case (value_kind)
        case (VALUE_F32)
            call lower_f32_expression(arena, node_index, context, value, error_msg)
        case (VALUE_F64)
            call lower_f64_expression(arena, node_index, context, value, error_msg)
        case (VALUE_LOGICAL)
            call lower_logical_expression(arena, node_index, context, value, &
                                          error_msg)
        case default
            call lower_i32_expression(arena, node_index, context, value, error_msg)
        end select
    end subroutine lower_expression_by_kind

    subroutine make_reference_argument(context, value_kind, value, address, &
                                       error_msg)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: value_kind
        type(lr_operand_desc_t), intent(in) :: value
        type(lr_operand_desc_t), intent(out) :: address
        character(len=:), allocatable, intent(out) :: error_msg

        select case (value_kind)
        case (VALUE_F32)
            if (.not. emit_liric_f32_alloca(context%session, address, error_msg)) &
                return
            if (.not. emit_liric_f32_store(context%session, value, address, &
                                           error_msg)) return
        case (VALUE_F64)
            if (.not. emit_liric_f64_alloca(context%session, address, error_msg)) &
                return
            if (.not. emit_liric_f64_store(context%session, value, address, &
                                           error_msg)) return
        case (VALUE_I32, VALUE_LOGICAL)
            if (.not. emit_i32_alloca(context%session, address, error_msg)) return
            if (.not. emit_i32_store(context%session, value, address, error_msg)) &
                return
        case default
            error_msg = 'direct LIRIC session cannot pass this scalar argument'
            return
        end select
        call set_empty(error_msg)
    end subroutine make_reference_argument

    subroutine copy_back_reference_args(context, args, copyback_indices, &
                                        error_msg)
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(in) :: args(:)
        integer, intent(in) :: copyback_indices(:)
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: value
        integer :: i

        do i = 1, size(copyback_indices)
            if (copyback_indices(i) <= 0) cycle
            select case (context%symbols(copyback_indices(i))%value_kind)
            case (VALUE_F32)
                if (.not. emit_liric_f32_load(context%session, args(i), value, &
                                              error_msg)) return
            case (VALUE_F64)
                if (.not. emit_liric_f64_load(context%session, args(i), value, &
                                              error_msg)) return
            case (VALUE_I32, VALUE_LOGICAL)
                if (.not. emit_i32_load(context%session, args(i), value, &
                                                        error_msg)) return
            case default
                error_msg = 'direct LIRIC session cannot copy back this argument'
                return
            end select
            context%symbols(copyback_indices(i))%value = value
        end do
        call set_empty(error_msg)
    end subroutine copy_back_reference_args
