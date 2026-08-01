    ! Typed scalar expression kind engine (#447).
    !
    ! One authority answers "what real kind does this scalar expression have?".
    ! Before this engine the answer was computed by two mirror-image boolean
    ! predicates (is_f32_expression / is_f64_expression) plus a third for
    ! intrinsic calls, each re-deriving the mixed-kind rule independently. Every
    ! caller now asks scalar_real_expr_kind once and compares the result.
    !
    ! The engine reports SCALAR_REAL_NONE for an expression that is not a real
    ! scalar (integer, logical, character, complex, derived, unresolved). It
    ! never guesses: an unresolved symbol or an unrecognized construct yields
    ! SCALAR_REAL_NONE rather than a default width.

    integer function real_value_kind_of(kind_code) result(vk)
        !! Narrow an arbitrary VALUE_* code to the real kinds this engine
        !! reports. Complex, integer, logical and derived codes are not real
        !! scalars and map to SCALAR_REAL_NONE.
        integer, intent(in) :: kind_code

        if (kind_code == VALUE_F32 .or. kind_code == VALUE_F64) then
            vk = kind_code
        else
            vk = SCALAR_REAL_NONE
        end if
    end function real_value_kind_of

    integer function wider_real_kind(left_kind, right_kind) result(vk)
        !! Combine two operand kinds under F2018 10.1.5.2.1: mixed-kind
        !! arithmetic takes the widest operand kind. f64 dominates f32, which
        !! dominates "not a real operand", so "2.0*x" with x real(8) is f64.
        integer, intent(in) :: left_kind, right_kind

        if (left_kind == VALUE_F64 .or. right_kind == VALUE_F64) then
            vk = VALUE_F64
        else if (left_kind == VALUE_F32 .or. right_kind == VALUE_F32) then
            vk = VALUE_F32
        else
            vk = SCALAR_REAL_NONE
        end if
    end function wider_real_kind

    integer function real_kind_from_kind_number(kind_number) result(vk)
        !! Map a resolved numeric KIND selector (4 or 8) onto a value kind.
        integer, intent(in) :: kind_number

        select case (kind_number)
        case (4)
            vk = VALUE_F32
        case (8)
            vk = VALUE_F64
        case default
            vk = SCALAR_REAL_NONE
        end select
    end function real_kind_from_kind_number

    recursive integer function scalar_real_expr_kind(arena, node_index, context) &
        result(vk)
        !! The real kind of a scalar expression, or SCALAR_REAL_NONE when the
        !! expression is not a real scalar.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        integer :: symbol_index
        character(len=:), allocatable :: bin_op, bin_err
        integer :: bin_left, bin_right, bin_line, bin_col
        character(len=:), allocatable :: id_name, id_err

        vk = SCALAR_REAL_NONE
        if (.not. node_exists(arena, node_index)) return

        if (is_binary_op(arena, node_index)) then
            call get_binary_op_info(arena, node_index, bin_op, bin_left, &
                                    bin_right, bin_line, bin_col, bin_err)
            vk = wider_real_kind( &
                 scalar_real_expr_kind(arena, bin_left, context), &
                 scalar_real_expr_kind(arena, bin_right, context))
            return
        end if

        if (is_literal(arena, node_index)) then
            ! A bare real literal ("2.5", "3.0e0") is default real (f32); a
            ! literal carrying a d exponent or an _8 / _real64 / named-kind
            ! suffix is f64 and must not be rounded through the f32 path.
            if (is_real_literal(arena, node_index)) then
                block
                    character(len=:), allocatable :: lit_value, lit_type, lit_err
                    call get_literal_info(arena, node_index, lit_value, &
                                          lit_type, lit_err)
                    if (literal_is_f64(lit_value, context, node_index)) then
                        vk = VALUE_F64
                    else
                        vk = VALUE_F32
                    end if
                end block
            end if
            return
        end if

        if (is_identifier(arena, node_index)) then
            call get_identifier_name(arena, node_index, id_name, id_err)
            symbol_index = resolve_symbol_at_node(context, node_index, id_name)
            if (symbol_index > 0) then
                vk = real_value_kind_of(context%symbols(symbol_index)%value_kind)
            end if
            return
        end if

        select type (node => arena%entries(node_index)%node)
        type is (call_or_subscript_node)
            vk = scalar_real_call_kind(arena, node, context)
        end select
    end function scalar_real_expr_kind

    recursive integer function scalar_real_call_kind(arena, node, context) &
        result(vk)
        !! The real kind of a call, subscript, or component reference.
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        integer :: symbol_index
        integer :: call_arg_count
        integer :: call_arg_kinds(MAX_PROC_ARGS)
        integer :: call_arg_ranks(MAX_PROC_ARGS)

        vk = SCALAR_REAL_NONE

        if (node%base_expr_index > 0) then
            ! obj%comp(i): element of an array component.
            vk = real_value_kind_of( &
                 component_element_access_kind(arena, node, context))
            return
        end if

        ! real(z)/aimag(z) yields the component kind of the complex operand.
        if (is_complex_component_extract(arena, node, context, VALUE_C8)) then
            vk = VALUE_F64
            return
        end if
        if (is_complex_component_extract(arena, node, context, VALUE_C4)) then
            vk = VALUE_F32
            return
        end if

        if (.not. allocated(node%name)) return

        if (same_name(node%name, 'transfer') .and. allocated(node%arg_indices)) then
            if (size(node%arg_indices) == 2) then
                vk = real_value_kind_of( &
                     transfer_operand_kind(arena, node%arg_indices(2), context))
            end if
            return
        end if

        if (.not. is_contained_function_reference(node, context) .and. &
            (node%is_array_access .or. &
             is_declared_array_element_ref(node, context))) then
            symbol_index = find_symbol_compat(context, node%name)
            if (symbol_index > 0) then
                vk = real_value_kind_of(context%symbols(symbol_index)%value_kind)
            end if
            return
        end if

        if (is_real_inquiry_intrinsic(node%name)) then
            ! tiny/huge/epsilon return a constant of the argument's kind.
            vk = real_kind_from_kind_number( &
                 inquiry_arg_real_kind(arena, node, context))
            return
        end if

        if (f64_intrinsic_id(node%name) /= F64_INTRINSIC_NONE) then
            vk = scalar_real_intrinsic_kind(arena, node, context)
            return
        end if

        ! dble() always returns real(8), regardless of argument kind.
        if (same_name(node%name, 'dble')) then
            vk = VALUE_F64
            return
        end if

        if (is_contained_f64_function(context, node%name)) then
            vk = VALUE_F64
            return
        end if
        if (is_contained_f32_function(context, node%name)) then
            vk = VALUE_F32
            return
        end if

        call call_argument_kinds(arena, node, context, VALUE_I32, &
                                 call_arg_count, call_arg_kinds)
        call call_argument_ranks(arena, node, context, call_arg_count, &
                                 call_arg_ranks)
        vk = real_value_kind_of(generic_call_return_kind(context, node%name, &
                 call_arg_count, call_arg_kinds, call_arg_ranks))
        if (vk /= SCALAR_REAL_NONE) return

        if (is_real_array_reduction(arena, node, context, VALUE_F64)) then
            vk = VALUE_F64
        else if (is_real_array_reduction(arena, node, context, VALUE_F32)) then
            vk = VALUE_F32
        else if (is_real_dot_product(arena, node, context, VALUE_F64)) then
            vk = VALUE_F64
        else if (is_real_dot_product(arena, node, context, VALUE_F32)) then
            vk = VALUE_F32
        end if
    end function scalar_real_call_kind

    recursive integer function scalar_real_intrinsic_kind(arena, node, context) &
        result(vk)
        !! Result kind of a real-valued elemental or conversion intrinsic.
        !! REAL(A[,KIND]) and AINT/ANINT(A[,KIND]) take their kind from the
        !! selector; every other intrinsic in the table is kind-preserving and
        !! takes the widest kind among its arguments.
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        integer :: i

        vk = SCALAR_REAL_NONE

        if (f64_intrinsic_id(node%name) == F64_INTRINSIC_REAL) then
            if (real_intrinsic_is_f64(arena, node, context)) then
                vk = VALUE_F64
            else
                vk = VALUE_F32
            end if
            return
        end if

        if (real_conversion_intrinsic(node%name) .and. allocated(node%arg_indices)) then
            if (size(node%arg_indices) >= 2) then
                if (real_conversion_kind_is_f64(arena, node%arg_indices(2))) then
                    vk = VALUE_F64
                else
                    vk = VALUE_F32
                end if
                return
            end if
        end if

        if (.not. allocated(node%arg_indices)) return
        if (size(node%arg_indices) < 1) return

        ! abs() of a complex yields a real magnitude of the component kind.
        if (same_name(node%name, 'abs')) then
            if (is_complex_valued(arena, node%arg_indices(1), context, VALUE_C8)) then
                vk = VALUE_F64
                return
            end if
            if (is_complex_valued(arena, node%arg_indices(1), context, VALUE_C4)) then
                vk = VALUE_F32
                return
            end if
        end if

        do i = 1, size(node%arg_indices)
            vk = wider_real_kind(vk, &
                 scalar_real_expr_kind(arena, node%arg_indices(i), context))
        end do
    end function scalar_real_intrinsic_kind
