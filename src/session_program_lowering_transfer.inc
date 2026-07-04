    ! TRANSFER(source, mold) for scalar intrinsic types of matching byte size:
    ! integer(4)<->real(4) and integer(8)<->real(8). The result reinterprets
    ! source's bit pattern as mold's type, done here by storing through a
    ! typed stack slot and loading the same address back with the other
    ! type - the same memory-punning trick EQUIVALENCE already relies on.
    !
    ! Scope: two-argument scalar TRANSFER (no SIZE argument, no array mold),
    ! source drawn from an identifier, a declared/allocatable array element,
    ! or a literal. Any other form declines (handled=.false., error_msg
    ! empty) so the caller falls back to its normal unsupported-feature
    ! diagnostic.

    subroutine lower_transfer_intrinsic(arena, node, context, target_kind, &
                                        value, handled, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: target_kind
        type(lr_operand_desc_t), intent(out) :: value
        logical, intent(out) :: handled
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: source_kind
        type(lr_operand_desc_t) :: source_value, address

        handled = .false.
        call set_empty(error_msg)
        if (.not. allocated(node%arg_indices)) return
        if (size(node%arg_indices) /= 2) return

        source_kind = transfer_operand_kind(arena, node%arg_indices(1), context)
        if (source_kind < 0) return
        if (.not. transfer_pair_supported(source_kind, target_kind)) return

        select case (source_kind)
        case (VALUE_I32)
            call lower_i32_expression(arena, node%arg_indices(1), context, &
                                      source_value, error_msg)
        case (VALUE_I64)
            call lower_i64_expression(arena, node%arg_indices(1), context, &
                                      source_value, error_msg)
        case (VALUE_F32)
            call lower_f32_expression(arena, node%arg_indices(1), context, &
                                      source_value, error_msg)
        case default
            call lower_f64_expression(arena, node%arg_indices(1), context, &
                                      source_value, error_msg)
        end select
        if (len_trim(error_msg) > 0) return

        if (source_kind == target_kind) then
            value = source_value
            handled = .true.
            return
        end if

        if (.not. transfer_alloca(context, source_kind, address, error_msg)) &
            return
        if (.not. transfer_store(context, source_kind, source_value, address, &
                                 error_msg)) return
        if (.not. transfer_load(context, target_kind, address, value, &
                                error_msg)) return
        handled = .true.
    end subroutine lower_transfer_intrinsic

    integer function transfer_operand_kind(arena, node_index, context) &
            result(kind)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        integer :: symbol_index
        character(len=:), allocatable :: id_name, id_err
        character(len=:), allocatable :: lv, lt, le

        kind = -1
        if (.not. node_exists(arena, node_index)) return

        if (is_identifier(arena, node_index)) then
            call get_identifier_name(arena, node_index, id_name, id_err)
            if (len_trim(id_err) > 0) return
            symbol_index = find_symbol(context, id_name)
            if (symbol_index > 0) then
                if (.not. context%symbols(symbol_index)%is_array) &
                    kind = context%symbols(symbol_index)%value_kind
            end if
            return
        end if

        if (is_real_literal(arena, node_index)) then
            call get_literal_info(arena, node_index, lv, lt, le)
            if (len_trim(le) > 0) return
            if (literal_is_f64(lv, context)) then
                kind = VALUE_F64
            else
                kind = VALUE_F32
            end if
            return
        end if

        if (is_logical_literal(arena, node_index)) then
            kind = VALUE_LOGICAL
            return
        end if

        if (is_literal(arena, node_index)) then
            kind = VALUE_I32
            return
        end if

        select type (node => arena%entries(node_index)%node)
        type is (call_or_subscript_node)
            if (node%base_expr_index > 0) return
            if (.not. allocated(node%name)) return
            if (node%is_array_access .or. &
                is_declared_array_element_ref(node, context) .or. &
                is_allocatable_element_ref(node, context)) then
                symbol_index = find_symbol(context, node%name)
                if (symbol_index > 0) &
                    kind = context%symbols(symbol_index)%value_kind
            end if
        end select
    end function transfer_operand_kind

    logical function transfer_pair_supported(source_kind, target_kind) &
            result(ok)
        integer, intent(in) :: source_kind, target_kind

        ok = .false.
        if (source_kind == target_kind) then
            select case (source_kind)
            case (VALUE_I32, VALUE_I64, VALUE_F32, VALUE_F64)
                ok = .true.
            end select
            return
        end if
        ok = (source_kind == VALUE_I32 .and. target_kind == VALUE_F32) .or. &
             (source_kind == VALUE_F32 .and. target_kind == VALUE_I32) .or. &
             (source_kind == VALUE_I64 .and. target_kind == VALUE_F64) .or. &
             (source_kind == VALUE_F64 .and. target_kind == VALUE_I64)
    end function transfer_pair_supported

    logical function transfer_alloca(context, kind, address, error_msg) &
            result(ok)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: kind
        type(lr_operand_desc_t), intent(out) :: address
        character(len=:), allocatable, intent(out) :: error_msg

        select case (kind)
        case (VALUE_I32)
            ok = emit_i32_alloca(context%session, address, error_msg)
        case (VALUE_I64)
            ok = emit_i64_alloca(context%session, address, error_msg)
        case (VALUE_F32)
            ok = emit_liric_f32_alloca(context%session, address, error_msg)
        case default
            ok = emit_liric_f64_alloca(context%session, address, error_msg)
        end select
    end function transfer_alloca

    logical function transfer_store(context, kind, value, address, error_msg) &
            result(ok)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: kind
        type(lr_operand_desc_t), intent(in) :: value
        type(lr_operand_desc_t), intent(in) :: address
        character(len=:), allocatable, intent(out) :: error_msg

        select case (kind)
        case (VALUE_I32)
            ok = emit_i32_store(context%session, value, address, error_msg)
        case (VALUE_I64)
            ok = emit_i64_store(context%session, value, address, error_msg)
        case (VALUE_F32)
            ok = emit_liric_f32_store(context%session, value, address, &
                                      error_msg)
        case default
            ok = emit_liric_f64_store(context%session, value, address, &
                                      error_msg)
        end select
    end function transfer_store

    logical function transfer_load(context, kind, address, value, error_msg) &
            result(ok)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: kind
        type(lr_operand_desc_t), intent(in) :: address
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg

        select case (kind)
        case (VALUE_I32)
            ok = emit_i32_load(context%session, address, value, error_msg)
        case (VALUE_I64)
            ok = emit_i64_load(context%session, address, value, error_msg)
        case (VALUE_F32)
            ok = emit_liric_f32_load(context%session, address, value, error_msg)
        case default
            ok = emit_liric_f64_load(context%session, address, value, error_msg)
        end select
    end function transfer_load
