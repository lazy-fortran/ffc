    ! Complex assignment and load helpers.
    ! Storage: re in symbol%address (f32 or f64 alloca), im in element_address.
    ! symbol%value holds the last-loaded re SSA value (used as fallback).

    subroutine lower_c4_assignment(arena, rhs_index, sym_idx, context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: rhs_index, sym_idx
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: re_val, im_val

        call set_empty(error_msg)
        if (.not. node_exists(arena, rhs_index)) then
            error_msg = 'complex(4) assignment: rhs index does not exist'
            return
        end if

        call eval_c4_rhs(arena, rhs_index, context, re_val, im_val, error_msg)
        if (len_trim(error_msg) > 0) return

        if (.not. emit_liric_f32_store(context%session, re_val, &
                context%symbols(sym_idx)%address, error_msg)) return
        if (.not. emit_liric_f32_store(context%session, im_val, &
                context%symbols(sym_idx)%element_address, error_msg)) return
        context%symbols(sym_idx)%value = re_val
        call set_empty(error_msg)
    end subroutine lower_c4_assignment

    subroutine lower_c8_assignment(arena, rhs_index, sym_idx, context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: rhs_index, sym_idx
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: re_val, im_val

        call set_empty(error_msg)
        if (.not. node_exists(arena, rhs_index)) then
            error_msg = 'complex(8) assignment: rhs index does not exist'
            return
        end if

        call eval_c8_rhs(arena, rhs_index, context, re_val, im_val, error_msg)
        if (len_trim(error_msg) > 0) return

        if (.not. emit_liric_f64_store(context%session, re_val, &
                context%symbols(sym_idx)%address, error_msg)) return
        if (.not. emit_liric_f64_store(context%session, im_val, &
                context%symbols(sym_idx)%element_address, error_msg)) return
        context%symbols(sym_idx)%value = re_val
        call set_empty(error_msg)
    end subroutine lower_c8_assignment

    ! True when node_index evaluates to a genuinely complex value of want_kind:
    ! a cmplx() call, a (re,im) literal, an identifier already declared with
    ! that complex kind, or a +/-/*// combination where either side is. Any
    ! other expression (real identifier, real/int literal, real function call,
    ! ** etc.) is real and must be promoted with a zero imaginary part instead
    ! of being decomposed as complex.
    recursive function is_complex_valued(arena, node_index, context, want_kind) &
        result(is_c)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: want_kind
        logical :: is_c
        character(len=:), allocatable :: op, name, err
        integer :: li, ri, ln, cl, sym

        is_c = .false.
        if (complex_call_args(arena, node_index, 'cmplx', li, ri)) then
            is_c = .true.
            return
        end if
        if (is_binary_op(arena, node_index)) then
            call get_binary_op_info(arena, node_index, op, li, ri, ln, cl, err)
            if (len_trim(err) > 0) return
            is_c = is_complex_valued(arena, li, context, want_kind) .or. &
                   is_complex_valued(arena, ri, context, want_kind)
            return
        end if
        select type (node => arena%entries(node_index)%node)
        type is (complex_literal_node)
            is_c = .true.
        class default
            if (.not. is_identifier(arena, node_index)) return
            call get_identifier_name(arena, node_index, name, err)
            if (len_trim(err) > 0) return
            sym = find_symbol(context, name)
            if (sym <= 0) return
            is_c = context%symbols(sym)%value_kind == want_kind
        end select
    end function is_complex_valued

    ! Evaluate an operand of a complex(4) binary combination: route through
    ! eval_c4_rhs when it is genuinely complex, otherwise widen the real value
    ! with a zero imaginary part.
    recursive subroutine eval_c4_operand(arena, node_index, context, re_val, &
                                         im_val, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: re_val, im_val
        character(len=:), allocatable, intent(out) :: error_msg

        call set_empty(error_msg)
        if (is_complex_valued(arena, node_index, context, VALUE_C4)) then
            call eval_c4_rhs(arena, node_index, context, re_val, im_val, error_msg)
            return
        end if
        call lower_f32_expression(arena, node_index, context, re_val, error_msg)
        if (len_trim(error_msg) > 0) return
        im_val = liric_f32_immediate(context%session, 0.0_c_float)
    end subroutine eval_c4_operand

    recursive subroutine eval_c8_operand(arena, node_index, context, re_val, &
                                         im_val, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: re_val, im_val
        character(len=:), allocatable, intent(out) :: error_msg

        call set_empty(error_msg)
        if (is_complex_valued(arena, node_index, context, VALUE_C8)) then
            call eval_c8_rhs(arena, node_index, context, re_val, im_val, error_msg)
            return
        end if
        call lower_f64_expression(arena, node_index, context, re_val, error_msg)
        if (len_trim(error_msg) > 0) return
        im_val = liric_f64_immediate(context%session, 0.0_c_double)
    end subroutine eval_c8_operand

    ! Evaluate a complex(4) rhs into its real/imag SSA components. Handles the
    ! (re, im) literal, a complex(4) identifier load, cmplx(re, im), and binary
    ! +/-/*// where either operand may be real (promoted with im=0) or
    ! complex(4). f32 components come from lower_f32_expression so real scalar
    ! literals widen correctly.
    recursive subroutine eval_c4_rhs(arena, rhs_index, context, re_val, im_val, &
                                     error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: rhs_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: re_val, im_val
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: lre, lim, rre, rim
        character(len=:), allocatable :: op, name
        integer :: li, ri, opc, ln, cl, sym

        call set_empty(error_msg)
        if (complex_call_args(arena, rhs_index, 'cmplx', li, ri)) then
            call lower_f32_expression(arena, li, context, re_val, error_msg)
            if (len_trim(error_msg) > 0) return
            call lower_f32_expression(arena, ri, context, im_val, error_msg)
            return
        end if
        if (is_binary_op(arena, rhs_index)) then
            call complex_binary_op(arena, rhs_index, op, li, ri, opc, ln, cl, &
                                   'complex(4)', error_msg)
            if (len_trim(error_msg) > 0) return
            call eval_c4_operand(arena, li, context, lre, lim, error_msg)
            if (len_trim(error_msg) > 0) return
            call eval_c4_operand(arena, ri, context, rre, rim, error_msg)
            if (len_trim(error_msg) > 0) return
            call complex_combine_c4(context, op, opc, lre, lim, rre, rim, &
                                    re_val, im_val, error_msg)
            return
        end if
        select type (rhs => arena%entries(rhs_index)%node)
        type is (complex_literal_node)
            call lower_f32_expression(arena, rhs%real_index, context, re_val, &
                                      error_msg)
            if (len_trim(error_msg) > 0) return
            call lower_f32_expression(arena, rhs%imag_index, context, im_val, &
                                      error_msg)
        class default
            sym = 0
            if (is_identifier(arena, rhs_index)) then
                call get_identifier_name(arena, rhs_index, name, error_msg)
                if (len_trim(error_msg) > 0) return
                sym = find_symbol(context, name)
            end if
            if (sym > 0 .and. context%symbols(sym)%value_kind == VALUE_C4) then
                call load_complex_identifier(arena, rhs_index, context, VALUE_C4, &
                                             .true., re_val, im_val, error_msg)
                return
            end if
            call lower_f32_expression(arena, rhs_index, context, re_val, error_msg)
            if (len_trim(error_msg) > 0) return
            im_val = liric_f32_immediate(context%session, 0.0_c_float)
        end select
    end subroutine eval_c4_rhs

    recursive subroutine eval_c8_rhs(arena, rhs_index, context, re_val, im_val, &
                                     error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: rhs_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: re_val, im_val
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: lre, lim, rre, rim
        character(len=:), allocatable :: op, name
        integer :: li, ri, opc, ln, cl, sym

        call set_empty(error_msg)
        if (complex_call_args(arena, rhs_index, 'cmplx', li, ri)) then
            call lower_f64_expression(arena, li, context, re_val, error_msg)
            if (len_trim(error_msg) > 0) return
            call lower_f64_expression(arena, ri, context, im_val, error_msg)
            return
        end if
        if (is_binary_op(arena, rhs_index)) then
            call complex_binary_op(arena, rhs_index, op, li, ri, opc, ln, cl, &
                                   'complex(8)', error_msg)
            if (len_trim(error_msg) > 0) return
            call eval_c8_operand(arena, li, context, lre, lim, error_msg)
            if (len_trim(error_msg) > 0) return
            call eval_c8_operand(arena, ri, context, rre, rim, error_msg)
            if (len_trim(error_msg) > 0) return
            call complex_combine_c8(context, op, opc, lre, lim, rre, rim, &
                                    re_val, im_val, error_msg)
            return
        end if
        select type (rhs => arena%entries(rhs_index)%node)
        type is (complex_literal_node)
            call lower_f64_expression(arena, rhs%real_index, context, re_val, &
                                      error_msg)
            if (len_trim(error_msg) > 0) return
            call lower_f64_expression(arena, rhs%imag_index, context, im_val, &
                                      error_msg)
        class default
            sym = 0
            if (is_identifier(arena, rhs_index)) then
                call get_identifier_name(arena, rhs_index, name, error_msg)
                if (len_trim(error_msg) > 0) return
                sym = find_symbol(context, name)
            end if
            if (sym > 0 .and. context%symbols(sym)%value_kind == VALUE_C8) then
                call load_complex_identifier(arena, rhs_index, context, VALUE_C8, &
                                             .false., re_val, im_val, error_msg)
                return
            end if
            call lower_f64_expression(arena, rhs_index, context, re_val, error_msg)
            if (len_trim(error_msg) > 0) return
            im_val = liric_f64_immediate(context%session, 0.0_c_double)
        end select
    end subroutine eval_c8_rhs

    ! Validate a complex binary node as +/-/*// and return the operands plus the
    ! float opcode. +/- combine componentwise; * and / use the full complex
    ! formula (see complex_combine_c4/c8). ** and other operators stay rejected
    ! so unsupported forms remain xfail. kind_tag names the kind in the error.
    subroutine complex_binary_op(arena, node_index, op, left, right, opc, &
                                 line, col, kind_tag, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        character(len=:), allocatable, intent(out) :: op
        integer, intent(out) :: left, right, opc, line, col
        character(len=*), intent(in) :: kind_tag
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: clean

        call get_binary_op_info(arena, node_index, op, left, right, line, col, &
                                error_msg)
        if (len_trim(error_msg) > 0) return
        clean = trim(adjustl(op))
        if (clean /= '+' .and. clean /= '-' .and. clean /= '*' .and. &
            clean /= '/') then
            error_msg = kind_tag//' assignment: unsupported operator '//trim(op)
            return
        end if
        call real_opcode(op, line, col, opc, error_msg)
    end subroutine complex_binary_op

    ! Combine two complex(4) operands by op into result components.
    ! +/- are componentwise; * and / use the Cartesian formulas:
    !   (a+bi)*(c+di) = (ac-bd) + (ad+bc)i
    !   (a+bi)/(c+di) = ((ac+bd) + (bc-ad)i) / (c*c+d*d)
    subroutine complex_combine_c4(context, op, opc, ar, ai, br, bi, &
                                  re_val, im_val, error_msg)
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: op
        integer, intent(in) :: opc
        type(lr_operand_desc_t), intent(in) :: ar, ai, br, bi
        type(lr_operand_desc_t), intent(out) :: re_val, im_val
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: p1, p2, p3, p4, denom, num_re, num_im
        character(len=:), allocatable :: clean

        clean = trim(adjustl(op))
        if (clean == '+' .or. clean == '-') then
            if (.not. emit_liric_f32_binary(context%session, opc, ar, br, &
                                            re_val, error_msg)) return
            if (.not. emit_liric_f32_binary(context%session, opc, ai, bi, &
                                            im_val, error_msg)) return
            return
        end if
        ! products shared by mul and div: ac, bd, ad, bc
        if (.not. emit_liric_f32_binary(context%session, 20, ar, br, p1, &
                                        error_msg)) return
        if (.not. emit_liric_f32_binary(context%session, 20, ai, bi, p2, &
                                        error_msg)) return
        if (.not. emit_liric_f32_binary(context%session, 20, ar, bi, p3, &
                                        error_msg)) return
        if (.not. emit_liric_f32_binary(context%session, 20, ai, br, p4, &
                                        error_msg)) return
        if (clean == '*') then
            if (.not. emit_liric_f32_binary(context%session, 19, p1, p2, &
                                            re_val, error_msg)) return
            if (.not. emit_liric_f32_binary(context%session, 18, p3, p4, &
                                            im_val, error_msg)) return
            return
        end if
        ! division: num_re = ac+bd, num_im = bc-ad, denom = c*c+d*d
        if (.not. emit_liric_f32_binary(context%session, 18, p1, p2, num_re, &
                                        error_msg)) return
        if (.not. emit_liric_f32_binary(context%session, 19, p4, p3, num_im, &
                                        error_msg)) return
        call complex_denom_c4(context, br, bi, denom, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_liric_f32_binary(context%session, 21, num_re, denom, &
                                        re_val, error_msg)) return
        if (.not. emit_liric_f32_binary(context%session, 21, num_im, denom, &
                                        im_val, error_msg)) return
    end subroutine complex_combine_c4

    subroutine complex_denom_c4(context, cr, ci, denom, error_msg)
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(in) :: cr, ci
        type(lr_operand_desc_t), intent(out) :: denom
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: cc, dd

        if (.not. emit_liric_f32_binary(context%session, 20, cr, cr, cc, &
                                        error_msg)) return
        if (.not. emit_liric_f32_binary(context%session, 20, ci, ci, dd, &
                                        error_msg)) return
        if (.not. emit_liric_f32_binary(context%session, 18, cc, dd, denom, &
                                        error_msg)) return
    end subroutine complex_denom_c4

    subroutine complex_combine_c8(context, op, opc, ar, ai, br, bi, &
                                  re_val, im_val, error_msg)
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: op
        integer, intent(in) :: opc
        type(lr_operand_desc_t), intent(in) :: ar, ai, br, bi
        type(lr_operand_desc_t), intent(out) :: re_val, im_val
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: p1, p2, p3, p4, denom, num_re, num_im
        character(len=:), allocatable :: clean

        clean = trim(adjustl(op))
        if (clean == '+' .or. clean == '-') then
            if (.not. emit_liric_f64_binary(context%session, opc, ar, br, &
                                            re_val, error_msg)) return
            if (.not. emit_liric_f64_binary(context%session, opc, ai, bi, &
                                            im_val, error_msg)) return
            return
        end if
        if (.not. emit_liric_f64_binary(context%session, 20, ar, br, p1, &
                                        error_msg)) return
        if (.not. emit_liric_f64_binary(context%session, 20, ai, bi, p2, &
                                        error_msg)) return
        if (.not. emit_liric_f64_binary(context%session, 20, ar, bi, p3, &
                                        error_msg)) return
        if (.not. emit_liric_f64_binary(context%session, 20, ai, br, p4, &
                                        error_msg)) return
        if (clean == '*') then
            if (.not. emit_liric_f64_binary(context%session, 19, p1, p2, &
                                            re_val, error_msg)) return
            if (.not. emit_liric_f64_binary(context%session, 18, p3, p4, &
                                            im_val, error_msg)) return
            return
        end if
        if (.not. emit_liric_f64_binary(context%session, 18, p1, p2, num_re, &
                                        error_msg)) return
        if (.not. emit_liric_f64_binary(context%session, 19, p4, p3, num_im, &
                                        error_msg)) return
        call complex_denom_c8(context, br, bi, denom, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_liric_f64_binary(context%session, 21, num_re, denom, &
                                        re_val, error_msg)) return
        if (.not. emit_liric_f64_binary(context%session, 21, num_im, denom, &
                                        im_val, error_msg)) return
    end subroutine complex_combine_c8

    subroutine complex_denom_c8(context, cr, ci, denom, error_msg)
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(in) :: cr, ci
        type(lr_operand_desc_t), intent(out) :: denom
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: cc, dd

        if (.not. emit_liric_f64_binary(context%session, 20, cr, cr, cc, &
                                        error_msg)) return
        if (.not. emit_liric_f64_binary(context%session, 20, ci, ci, dd, &
                                        error_msg)) return
        if (.not. emit_liric_f64_binary(context%session, 18, cc, dd, denom, &
                                        error_msg)) return
    end subroutine complex_denom_c8

    ! Load the re/im components of a complex identifier rhs of want_kind.
    ! is_f32 selects f32 vs f64 component loads.
    subroutine load_complex_identifier(arena, rhs_index, context, want_kind, &
                                       is_f32, re_val, im_val, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: rhs_index, want_kind
        type(lowering_context_t), intent(inout) :: context
        logical, intent(in) :: is_f32
        type(lr_operand_desc_t), intent(out) :: re_val, im_val
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: src_name, tag
        integer :: src_idx

        call set_empty(error_msg)
        tag = 'complex(4)'
        if (.not. is_f32) tag = 'complex(8)'
        if (.not. is_identifier(arena, rhs_index)) then
            error_msg = tag//' assignment: unsupported rhs form'
            return
        end if
        call get_identifier_name(arena, rhs_index, src_name, error_msg)
        if (len_trim(error_msg) > 0) return
        src_idx = find_symbol(context, src_name)
        if (src_idx <= 0) then
            error_msg = tag//' rhs not declared: '//trim(src_name)
            return
        end if
        if (context%symbols(src_idx)%value_kind /= want_kind) then
            error_msg = tag//' rhs is not '//tag//': '//trim(src_name)
            return
        end if
        if (is_f32) then
            if (.not. emit_liric_f32_load(context%session, &
                    context%symbols(src_idx)%address, re_val, error_msg)) return
            if (.not. emit_liric_f32_load(context%session, &
                    context%symbols(src_idx)%element_address, im_val, error_msg)) &
                return
        else
            if (.not. emit_liric_f64_load(context%session, &
                    context%symbols(src_idx)%address, re_val, error_msg)) return
            if (.not. emit_liric_f64_load(context%session, &
                    context%symbols(src_idx)%element_address, im_val, error_msg)) &
                return
        end if
    end subroutine load_complex_identifier

    ! True when node_index is a two-argument call to the named intrinsic
    ! (cmplx(re, im)); returns the two argument node indices.
    logical function complex_call_args(arena, node_index, name, arg1, arg2) &
        result(is_call)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        character(len=*), intent(in) :: name
        integer, intent(out) :: arg1, arg2

        is_call = .false.
        arg1 = 0
        arg2 = 0
        if (.not. node_exists(arena, node_index)) return
        select type (node => arena%entries(node_index)%node)
        type is (call_or_subscript_node)
            if (lowercase_text(trim(node%name)) /= name) return
            if (.not. allocated(node%arg_indices)) return
            if (size(node%arg_indices) /= 2) return
            arg1 = node%arg_indices(1)
            arg2 = node%arg_indices(2)
            is_call = .true.
        end select
    end function complex_call_args

    ! True when node is real(z)/aimag(z) on a complex symbol of want_kind
    ! (VALUE_C4 or VALUE_C8). Used to route the extraction through the matching
    ! f32/f64 expression path so the result prints in the component's precision.
    logical function is_complex_component_extract(arena, node, context, &
                                                  want_kind) result(matches)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        integer, intent(in) :: want_kind
        type(lowering_context_t), intent(in) :: context
        character(len=:), allocatable :: name, arg_name, err
        integer :: sym

        matches = .false.
        if (.not. allocated(node%name)) return
        if (node%is_array_access) return
        name = lowercase_text(trim(node%name))
        if (name /= 'real' .and. name /= 'aimag') return
        if (.not. allocated(node%arg_indices)) return
        if (size(node%arg_indices) /= 1) return
        if (.not. is_identifier(arena, node%arg_indices(1))) return
        call get_identifier_name(arena, node%arg_indices(1), arg_name, err)
        if (len_trim(err) > 0) return
        sym = find_symbol(context, arg_name)
        if (sym <= 0) return
        matches = context%symbols(sym)%value_kind == want_kind
    end function is_complex_component_extract

    ! Lower real(z)/aimag(z) by loading the matching component slot of the
    ! complex symbol. is_f32 selects f32 vs f64 component loads.
    subroutine lower_complex_component_extract(arena, node, context, &
                                               is_f32, value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        logical, intent(in) :: is_f32
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: arg_name
        type(lr_operand_desc_t) :: slot
        integer :: sym
        logical :: want_imag

        call set_empty(error_msg)
        want_imag = lowercase_text(trim(node%name)) == 'aimag'
        call get_identifier_name(arena, node%arg_indices(1), arg_name, error_msg)
        if (len_trim(error_msg) > 0) return
        sym = find_symbol(context, arg_name)
        if (want_imag) then
            slot = context%symbols(sym)%element_address
        else
            slot = context%symbols(sym)%address
        end if
        if (is_f32) then
            if (.not. emit_liric_f32_load(context%session, slot, value, &
                                          error_msg)) return
        else
            if (.not. emit_liric_f64_load(context%session, slot, value, &
                                          error_msg)) return
        end if
    end subroutine lower_complex_component_extract

    subroutine lower_print_complex4(sym_idx, context, error_msg)
        integer, intent(in) :: sym_idx
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: re_val, im_val

        if (.not. emit_liric_f32_load(context%session, &
                context%symbols(sym_idx)%address, re_val, error_msg)) return
        if (.not. emit_liric_f32_load(context%session, &
                context%symbols(sym_idx)%element_address, im_val, error_msg)) return
        if (.not. emit_complex4_print_call(context%session, re_val, im_val, &
                                           error_msg)) return
        call set_empty(error_msg)
    end subroutine lower_print_complex4

    logical function is_contained_complex_function(context, name) result(is_c)
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: name
        integer :: kind

        kind = contained_function_kind(context, name)
        is_c = kind == VALUE_C4 .or. kind == VALUE_C8
    end function is_contained_complex_function

    subroutine lower_print_complex_result_call(arena, node_index, context, &
                                               error_msg)
        ! print *, cfunc(): emit the sret call into a scratch complex buffer,
        ! then load and print re/im. The callee writes re@+0, im@+4 (c4)/+8 (c8).
        use, intrinsic :: iso_c_binding, only: c_int64_t
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: buf, re_addr, im_addr, re_val, im_val
        type(lr_operand_desc_t), allocatable :: user_args(:), all_args(:)
        integer, allocatable :: copyback_indices(:)
        character(len=:), allocatable :: func_name
        integer :: kind, i
        integer(c_int64_t) :: nbytes, im_offset

        call set_empty(error_msg)
        select type (call_node => arena%entries(node_index)%node)
        type is (call_or_subscript_node)
            func_name = trim(call_node%name)
            kind = contained_function_kind(context, func_name)
            if (kind == VALUE_C8) then
                nbytes = 16_c_int64_t
                im_offset = 8_c_int64_t
            else
                nbytes = 8_c_int64_t
                im_offset = 4_c_int64_t
            end if
            if (.not. emit_alloca_bytes(context%session, &
                    i64_immediate(context%session, nbytes), buf, error_msg)) return
            if (allocated(call_node%arg_indices)) then
                call prepare_reference_args(arena, call_node%arg_indices, context, &
                                            VALUE_I32, func_name, user_args, &
                                            copyback_indices, error_msg)
                if (len_trim(error_msg) > 0) return
            else
                allocate (user_args(0)); allocate (copyback_indices(0))
            end if
            allocate (all_args(size(user_args) + 1))
            all_args(1) = buf
            do i = 1, size(user_args)
                all_args(i + 1) = user_args(i)
            end do
            if (.not. emit_void_call(context%session, func_name, all_args, &
                                     error_msg)) return
            call copy_back_reference_args(context, user_args, copyback_indices, &
                                          error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_ptr_offset(context%session, buf, 0_c_int64_t, &
                                      re_addr, error_msg)) return
            if (.not. emit_ptr_offset(context%session, buf, im_offset, &
                                      im_addr, error_msg)) return
            if (kind == VALUE_C8) then
                if (.not. emit_liric_f64_load(context%session, re_addr, re_val, &
                                              error_msg)) return
                if (.not. emit_liric_f64_load(context%session, im_addr, im_val, &
                                              error_msg)) return
                if (.not. emit_complex8_print_call(context%session, re_val, &
                                                   im_val, error_msg)) return
            else
                if (.not. emit_liric_f32_load(context%session, re_addr, re_val, &
                                              error_msg)) return
                if (.not. emit_liric_f32_load(context%session, im_addr, im_val, &
                                              error_msg)) return
                if (.not. emit_complex4_print_call(context%session, re_val, &
                                                   im_val, error_msg)) return
            end if
        class default
            error_msg = 'expected function call for complex result print'
        end select
    end subroutine lower_print_complex_result_call

    subroutine lower_print_array_result_call(arena, node_index, context, error_msg)
        ! print *, vec_fn(): allocate a scratch array buffer, run the sret call
        ! into it, then print each element in storage order with list-directed
        ! spacing (array function results in a print list).
        use, intrinsic :: iso_c_binding, only: c_int64_t
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: buf, value
        type(lr_operand_desc_t), allocatable :: user_args(:), all_args(:)
        integer, allocatable :: copyback_indices(:)
        character(len=:), allocatable :: func_name
        integer :: element_kind, array_size, sym, i
        integer(c_int64_t) :: k

        call set_empty(error_msg)
        select type (call_node => arena%entries(node_index)%node)
        type is (call_or_subscript_node)
            func_name = trim(call_node%name)
            call contained_array_result_info(arena, context, func_name, &
                                             element_kind, array_size, error_msg)
            if (len_trim(error_msg) > 0) return
            if (array_size <= 0) then
                error_msg = 'array result print requires a positive size'
                return
            end if

            select case (element_kind)
            case (VALUE_F32)
                if (.not. emit_f32_array_alloca(context%session, array_size, &
                                                buf, error_msg)) return
            case (VALUE_F64)
                if (.not. emit_f64_array_alloca(context%session, array_size, &
                                                buf, error_msg)) return
            case default
                if (.not. emit_i32_array_alloca(context%session, array_size, &
                                                buf, error_msg)) return
            end select

            if (allocated(call_node%arg_indices)) then
                call prepare_reference_args(arena, call_node%arg_indices, context, &
                                            VALUE_I32, func_name, user_args, &
                                            copyback_indices, error_msg)
                if (len_trim(error_msg) > 0) return
            else
                allocate (user_args(0)); allocate (copyback_indices(0))
            end if
            allocate (all_args(size(user_args) + 1))
            all_args(1) = buf
            do i = 1, size(user_args)
                all_args(i + 1) = user_args(i)
            end do
            if (.not. emit_void_call(context%session, &
                    call_emit_name(arena, func_name), all_args, error_msg)) return
            call copy_back_reference_args(context, user_args, copyback_indices, &
                                          error_msg)
            if (len_trim(error_msg) > 0) return

            ! Bind a scratch symbol onto the buffer so element loads reuse the
            ! standard typed array-element addressing.
            call grow_symbols(context)
            sym = context%symbol_count + 1
            context%symbols(sym)%name = '__array_result_tmp'
            context%symbols(sym)%value_kind = element_kind
            context%symbols(sym)%is_array = .true.
            context%symbols(sym)%array_rank = 1
            context%symbols(sym)%array_size = array_size
            context%symbols(sym)%array_lower_bound = 1
            context%symbols(sym)%array_dim_sizes(1) = array_size
            context%symbols(sym)%array_dim_lowers(1) = 1
            context%symbols(sym)%element_address = buf
            context%symbol_count = sym

            ! The print-item loop already emitted the leading separator for this
            ! item; supply only the inter-element separators here.
            do k = 0_c_int64_t, int(array_size, c_int64_t) - 1_c_int64_t
                if (k > 0_c_int64_t) then
                    if (.not. emit_liric_print_space(context%session, error_msg)) &
                        return
                end if
                call load_array_linear_element(context, sym, k, value, error_msg)
                if (len_trim(error_msg) > 0) return
                if (element_kind == VALUE_F32) then
                    if (.not. emit_liric_print_f32_value(context%session, value, &
                                                         error_msg)) return
                else if (element_kind == VALUE_F64) then
                    if (.not. emit_liric_print_f64_value(context%session, value, &
                                                         error_msg)) return
                else
                    if (.not. emit_liric_print_i32_value(context%session, &
                            context%i32_print_format_id, value, error_msg)) return
                end if
            end do
        class default
            error_msg = 'expected function call for array result print'
        end select
    end subroutine lower_print_array_result_call

    subroutine lower_print_complex8(sym_idx, context, error_msg)
        integer, intent(in) :: sym_idx
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: re_val, im_val

        if (.not. emit_liric_f64_load(context%session, &
                context%symbols(sym_idx)%address, re_val, error_msg)) return
        if (.not. emit_liric_f64_load(context%session, &
                context%symbols(sym_idx)%element_address, im_val, error_msg)) return
        if (.not. emit_complex8_print_call(context%session, re_val, im_val, &
                                           error_msg)) return
        call set_empty(error_msg)
    end subroutine lower_print_complex8
