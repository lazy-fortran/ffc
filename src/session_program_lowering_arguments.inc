    subroutine store_reference_value(context, symbol_index, value, error_msg)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        type(lr_operand_desc_t), intent(in) :: value
        character(len=:), allocatable, intent(out) :: error_msg

        select case (context%symbols(symbol_index)%value_kind)
        case (VALUE_I32, VALUE_LOGICAL)
            if (.not. emit_i32_store(context%session,  &
                value, context%symbols(symbol_index)%address, error_msg)) return
        case (VALUE_I64)
            if (.not. emit_i64_store(context%session, value, &
                context%symbols(symbol_index)%address, error_msg)) return
        case (VALUE_I8)
            if (.not. emit_i8_store(context%session, value, &
                context%symbols(symbol_index)%address, error_msg)) return
        case (VALUE_I16)
            if (.not. emit_i16_store(context%session, value, &
                context%symbols(symbol_index)%address, error_msg)) return
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

    subroutine prepare_reference_args(arena, actual_indices, context, default_kind, &
                                      callee_name, args, copyback_indices, &
                                      error_msg, self_position)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: actual_indices(:)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: default_kind
        character(len=*), intent(in) :: callee_name
        type(lr_operand_desc_t), allocatable, intent(out) :: args(:)
        integer, allocatable, intent(out) :: copyback_indices(:)
        character(len=:), allocatable, intent(out) :: error_msg
        integer, intent(in), optional :: self_position
        integer :: i
        integer :: visible_count
        integer :: hidden_count
        integer :: sp
        integer :: param_count
        integer :: explicit_param_count
        logical :: handled_by_value
        logical :: supplied, is_optional, is_value
        integer :: intent_code, p
        integer :: assumed_rank_actual_symbol
        ! Actual arguments in dummy-argument order: keyword actuals are moved
        ! to the slot their dummy name selects, and a slot no actual reaches
        ! holds 0 (an omitted optional, passed as a null reference).
        integer, allocatable :: arg_indices(:)
        character(len=:), allocatable :: proc_actual

        sp = 0
        if (present(self_position)) sp = self_position
        call resolve_keyword_actuals(arena, context, actual_indices, callee_name, &
                                     sp, arg_indices, error_msg)
        if (len_trim(error_msg) > 0) return
        visible_count = size(arg_indices)
        ! Assumed-shape dummies now travel as one descriptor pointer each
        ! (#334), so no hidden extent arguments are appended any more.
        hidden_count = 0

        ! More actual arguments than the callee (defined or interfaced in this
        ! unit) declares is always invalid, whatever the argument keywords or
        ! optionality. Missing optional dummies must still be materialized as
        ! null reference arguments: same-unit functions have fixed LIRIC
        ! parameter slots even when Fortran omits the actual. Type-bound calls
        ! include the receiver in arg_indices, so the position mapper keeps
        ! the contract check aligned with the implementation's dummies.
        param_count = arena_proc_param_count(arena, callee_name)
        if (param_count < 0) param_count = proc_param_count(context, callee_name)
        explicit_param_count = param_count
        if (sp > 0 .and. explicit_param_count >= 0) then
            explicit_param_count = max(explicit_param_count - 1, 0)
        end if
        if (explicit_param_count >= 0 .and. visible_count > explicit_param_count) then
            error_msg = 'More actual than formal arguments in call to '// &
                trim(callee_name)
            return
        end if
        if (param_count >= 0) then
            if (size(arg_indices) < explicit_param_count) then
                block
                    integer, allocatable :: padded_indices(:)
                    allocate (padded_indices(explicit_param_count))
                    padded_indices = 0
                    if (size(arg_indices) > 0) padded_indices(1:size(arg_indices)) = &
                        arg_indices
                    call move_alloc(padded_indices, arg_indices)
                end block
            end if
            do p = 1, param_count
                ! lower_method_subroutine_call supplies the passed-object
                ! receiver after this routine returns; it is not among the
                ! explicit actuals being checked here.
                if (sp > 0 .and. p == sp) cycle
                supplied = .false.
                do i = 1, size(arg_indices)
                    if (arg_indices(i) <= 0) cycle
                    if (call_arg_param_pos(i, sp) == p) then
                        supplied = .true.
                        exit
                    end if
                end do
                if (supplied) cycle
                call callee_dummy_contracts(arena, context, callee_name, p, &
                                            is_optional, is_value, intent_code)
                if (.not. is_optional) then
                    error_msg = 'Missing required actual argument in call to '// &
                        trim(callee_name)
                    return
                end if
            end do
        end if

        visible_count = size(arg_indices)

        ! Keep array-interface diagnostics (rank and storage extent) useful even
        ! when the interface has no body in this lowering unit.  Scalar calls
        ! can only be lowered when their same-unit body is available.
        ! A call through a procedure dummy or procedure pointer is indirect:
        ! the callee address arrives at run time, so no body is needed here.
        if (.not. call_has_array_dummy(arena, callee_name, arg_indices, sp) &
            .and. .not. is_proc_pointer_call(context, callee_name)) then
            if (registered_procedure_body_missing(arena, context, callee_name)) then
                error_msg = 'procedure body unavailable for call to '//trim(callee_name)
                return
            end if
        end if

        allocate (args(visible_count + hidden_count))
        allocate (copyback_indices(visible_count + hidden_count))
        copyback_indices = 0

        do i = 1, size(arg_indices)
            if (arg_indices(i) <= 0) then
                ! Keyword association skipped this dummy: it is optional and
                ! absent, which the reference ABI encodes as a null pointer.
                args(i) = absent_optional_operand(context)
                cycle
            end if
            ! A procedure passed as an actual argument travels as its callable
            ! address, never through data storage (#467).
            if (procedure_actual_name(arena, arg_indices(i), context, &
                                      proc_actual)) then
                call procedure_actual_operand(arena, context, proc_actual, &
                                              args(i), error_msg)
                if (len_trim(error_msg) > 0) return
                cycle
            end if
            if (actual_is_rank2_allocatable_component(arena, arg_indices(i), &
                                                      context)) then
                error_msg = 'passing a rank-2 allocatable component as an actual '// &
                            'argument is not supported by the direct LIRIC session'
                return
            end if
            call check_scalar_actual_compatibility(arena, arg_indices(i), context, &
                callee_name, call_arg_param_pos(i, sp), error_msg)
            if (len_trim(error_msg) > 0) return
            call check_dummy_contract_actual(arena, arg_indices(i), context, &
                callee_name, call_arg_param_pos(i, sp), handled_by_value, &
                args(i), error_msg)
            if (len_trim(error_msg) > 0) return
            if (handled_by_value) cycle
            ! A scalar actual (literal or scalar variable, not an array element)
            ! passed where the callee declares an array dummy is a rank mismatch
            ! (F2018 15.5.2.4). callee_dummy_is_array only resolves against a
            ! procedure defined or interfaced in this unit, so external calls
            ! without an explicit interface never trip this. An assumed-rank
            ! arr(..) dummy binds to any rank, scalar included, so it is excluded.
            if (callee_dummy_is_array_ctx(arena, context, callee_name, &
                                          call_arg_param_pos(i, sp)) .and. &
                actual_is_definite_scalar(arena, arg_indices(i), context) .and. &
                .not. callee_dummy_is_assumed_rank(arena, callee_name, &
                                                   call_arg_param_pos(i, sp))) then
                error_msg = 'Rank mismatch in argument to '//trim(callee_name)// &
                    ': scalar actual passed to array dummy argument'
                return
            end if
            call check_explicit_shape_actual_size(arena, arg_indices(i), context, &
                                                  callee_name, &
                                                  call_arg_param_pos(i, sp), &
                                                  error_msg)
            if (len_trim(error_msg) > 0) return
            ! A type(t) dummy borrows the actual's slot layout whichever ABI
            ! carries it (bare address, array descriptor), so check declared
            ! type conformance once here rather than per ABI branch.
            call check_derived_actual_type(arena, arg_indices(i), context, &
                                           callee_name, &
                                           call_arg_param_pos(i, sp), error_msg)
            if (len_trim(error_msg) > 0) return
            ! An allocatable array actual has a descriptor ABI even when its
            ! element kind is CHARACTER. Handle this before the scalar
            ! character descriptor path, which otherwise reinterprets the
            ! array descriptor as a {data,length} scalar descriptor.
            if (actual_is_allocatable_array(arena, arg_indices(i), context) &
                    .and. callee_dummy_is_allocatable_array(arena, callee_name, &
                                                            call_arg_param_pos(i, sp))) then
                call allocatable_array_actual_descriptor_address(arena, &
                    arg_indices(i), context, args(i), error_msg)
                if (len_trim(error_msg) > 0) return
            else if (actual_is_character(arena, arg_indices(i), context) .and. &
                     callee_dummy_char_writable(arena, callee_name, &
                                                call_arg_param_pos(i, sp))) then
                ! intent(out)/(inout) allocatable character: pass the actual's
                ! own descriptor so the callee's writes land in the caller.
                call char_actual_inout_descriptor(arena, arg_indices(i), &
                                                   context, args(i), error_msg)
                if (len_trim(error_msg) > 0) return
            else if (actual_is_character(arena, arg_indices(i), context) .and. &
                     (callee_dummy_is_character(arena, callee_name, &
                                                call_arg_param_pos(i, sp)) .or. &
                      external_dummy_value_kind(context, callee_name, &
                          call_arg_param_pos(i, sp)) == VALUE_CHARACTER)) then
                call char_actual_descriptor(arena, arg_indices(i), context, &
                                            args(i), error_msg)
                if (len_trim(error_msg) > 0) return
            else if (actual_is_derived(arena, arg_indices(i), context) .and. &
                     callee_dummy_assumed_shape_dims(arena, callee_name, &
                                        call_arg_param_pos(i, sp)) < 1) then
                ! A derived actual reaching a class(t) dummy is passed as the
                ! scalar class descriptor so the dummy carries the actual's
                ! dynamic type; a type(t) dummy takes the bare base address of
                ! the component storage.
                call class_or_derived_actual(arena, arg_indices(i), context, &
                                             callee_name, &
                                             call_arg_param_pos(i, sp), &
                                             args(i), error_msg)
                if (len_trim(error_msg) > 0) return
            else if (callee_dummy_is_class_star(arena, callee_name, &
                                                call_arg_param_pos(i, sp))) then
                ! Box the actual into a {data, type_id} descriptor for a
                ! class(*) dummy (#141).
                call box_class_star_actual(arena, arg_indices(i), context, &
                                           args(i), error_msg)
                if (len_trim(error_msg) > 0) return
            else if (callee_dummy_is_assumed_rank(arena, callee_name, &
                                        call_arg_param_pos(i, sp)) .and. &
                     (callee_dummy_assumed_rank_value_kind(arena, callee_name, &
                        call_arg_param_pos(i, sp), context) == VALUE_F32 .or. &
                      callee_dummy_assumed_rank_value_kind(arena, callee_name, &
                        call_arg_param_pos(i, sp), context) == VALUE_F64)) then
                ! The narrow assumed-rank slice is a rank-1/rank-2/rank-3 REAL
                ! descriptor boundary. Rank is checked from the actual before
                ! emitting the descriptor; scalar, higher-rank, section, alias,
                ! global, and unsupported-kind forms do not silently fall back
                ! to a base-pointer ABI.
                if (.not. actual_is_whole_array(arena, arg_indices(i), context)) then
                    error_msg = 'unsupported assumed-rank actual in '//trim(callee_name)// &
                        ': only a whole array is supported; sections and aliases are refused'
                    return
                end if
                assumed_rank_actual_symbol = assumed_shape_actual_symbol(arena, &
                    arg_indices(i), context)
                if (assumed_rank_actual_symbol > 0) then
                    if (symbol_has_global_storage( &
                            context%symbols(assumed_rank_actual_symbol))) then
                        error_msg = 'unsupported assumed-rank actual in '//trim(callee_name)// &
                            ': global storage is refused'
                        return
                    end if
                    if (context%symbols(assumed_rank_actual_symbol)%is_pointer .or. &
                        context%symbols(assumed_rank_actual_symbol)%is_reference) then
                        error_msg = 'unsupported assumed-rank actual in '//trim(callee_name)// &
                            ': pointer and alias actuals are refused'
                        return
                    end if
                    if (context%symbols(assumed_rank_actual_symbol)%is_allocatable .or. &
                        context%symbols(assumed_rank_actual_symbol)%is_runtime_array) then
                        error_msg = 'unsupported assumed-rank actual in '//trim(callee_name)// &
                            ': dynamic shapes are not supported'
                        return
                    end if
                    if (context%symbols(assumed_rank_actual_symbol)%value_kind /= VALUE_F32 .and. &
                        context%symbols(assumed_rank_actual_symbol)%value_kind /= VALUE_F64) then
                        error_msg = 'unsupported assumed-rank actual in '//trim(callee_name)// &
                            ': only REAL whole arrays are supported'
                        return
                    end if
                end if
                if (assumed_shape_actual_rank(arena, arg_indices(i), context) /= 1 .and. &
                    assumed_shape_actual_rank(arena, arg_indices(i), context) /= 2 .and. &
                    assumed_shape_actual_rank(arena, arg_indices(i), context) /= 3) then
                    error_msg = 'unsupported assumed-rank actual in '//trim(callee_name)// &
                        ': only a rank-1, rank-2, or rank-3 whole REAL array is supported'
                    return
                end if
                call build_assumed_shape_descriptor(arena, arg_indices(i), &
                    context, assumed_shape_actual_rank(arena, arg_indices(i), context), &
                    callee_name, call_arg_param_pos(i, sp), &
                    args(i), error_msg, assumed_rank=.true.)
                if (len_trim(error_msg) > 0) return
            else if (callee_dummy_assumed_shape_dims(arena, callee_name, &
                                        call_arg_param_pos(i, sp)) >= 1) then
                ! Assumed-shape dummy: pass one canonical array descriptor
                ! (#334, docs/ARRAY_DESCRIPTOR_ABI.md). It carries the actual's
                ! base address, element size and type, rank, extents, and
                ! column-major byte strides, so the callee needs no hidden
                ! extent arguments and no compile-time knowledge of the actual.
                call build_assumed_shape_descriptor(arena, arg_indices(i), &
                    context, callee_dummy_assumed_shape_dims(arena, &
                        callee_name, call_arg_param_pos(i, sp)), callee_name, &
                    call_arg_param_pos(i, sp), &
                    args(i), error_msg)
                if (len_trim(error_msg) > 0) return
            else if (actual_is_whole_array(arena, arg_indices(i), context) .and. &
                     callee_dummy_is_array_ctx(arena, context, callee_name, &
                                               call_arg_param_pos(i, sp))) then
                ! Explicit-shape array actual to an array dummy: pass the
                ! contiguous base address; the callee binds it as its dummy view.
                call whole_array_actual_address(arena, arg_indices(i), context, &
                                                 args(i), error_msg)
                if (len_trim(error_msg) > 0) return
            else if (external_dummy_is_opaque(context, callee_name, &
                                              call_arg_param_pos(i, sp)) .and. &
                     actual_is_allocatable_component(arena, arg_indices(i), &
                                                     context)) then
                ! An opaque imported dummy has no scalar kind or local AST
                ! contract. An allocatable component still has a precise,
                ! reusable descriptor ABI; pass that descriptor rather than
                ! inventing a scalar placeholder type.
                call alloc_array_component_descriptor(arena, arg_indices(i), &
                    context, args(i), error_msg)
                if (len_trim(error_msg) > 0) return
            else if (callee_dummy_is_array_ctx(arena, context, callee_name, &
                                               call_arg_param_pos(i, sp)) .and. &
                     callee_dummy_assumed_shape_dims(arena, callee_name, &
                                         call_arg_param_pos(i, sp)) < 1 .and. &
                     actual_is_array_expression(arena, arg_indices(i), context)) then
                ! An array-valued expression has no storage of its own. Materialise
                ! it into a temporary with the dummy's element kind, then pass the
                ! temporary's contiguous base address just like a named array.
                call array_expression_actual_address(arena, arg_indices(i), &
                    context, callee_name, call_arg_param_pos(i, sp), default_kind, &
                    args(i), error_msg)
                if (len_trim(error_msg) > 0) return
            else if (actual_is_contiguous_section(arena, arg_indices(i), context) &
                     .and. callee_dummy_is_array_ctx(arena, context, &
                            callee_name, call_arg_param_pos(i, sp))) then
                ! Contiguous rank-1 array section a(lo:hi) or whole column
                ! a(:,j): pass the base address of the section's first element,
                ! so the callee's rank-1 dummy views the section's storage in
                ! place. The dummy's extent is resolved from the same section at
                ! compile time (section_actual_decl_extents).
                call section_actual_address(arena, arg_indices(i), context, &
                                            args(i), error_msg)
                if (len_trim(error_msg) > 0) return
            else
                call prepare_reference_arg(arena, arg_indices(i), context, &
                                           arg_default_kind(arena, context, &
                                                callee_name, &
                                                call_arg_param_pos(i, sp), &
                                                default_kind), &
                                           args(i), copyback_indices(i), error_msg)
                if (len_trim(error_msg) > 0) return
            end if
        end do

        call set_empty(error_msg)
    end subroutine prepare_reference_args

    logical function external_dummy_is_opaque(context, callee_name, param_pos)
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: callee_name
        integer, intent(in) :: param_pos
        integer :: idx

        external_dummy_is_opaque = .false.
        if (param_pos < 1) return
        idx = external_procedure_index(context, callee_name)
        if (idx <= 0) return
        if (param_pos > context%external_procedures(idx)%arg_count) return
        external_dummy_is_opaque = &
            context%external_procedures(idx)%arg_is_opaque(param_pos)
    end function external_dummy_is_opaque

    logical function actual_is_allocatable_component(arena, node_index, context)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context

        actual_is_allocatable_component = .false.
        if (.not. node_exists(arena, node_index)) return
        select type (node => arena%entries(node_index)%node)
        type is (component_access_node)
            actual_is_allocatable_component = &
                component_access_is_alloc_array(arena, node, context)
        end select
    end function actual_is_allocatable_component

    logical function actual_is_rank2_allocatable_component(arena, node_index, &
                                                           context)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        integer :: base_type, component_index

        actual_is_rank2_allocatable_component = .false.
        if (.not. node_exists(arena, node_index)) return
        select type (node => arena%entries(node_index)%node)
        type is (component_access_node)
            if (.not. component_access_is_alloc_array(arena, node, context)) return
            base_type = component_base_type_index(arena, node%base_expr_index, &
                                                  context)
            component_index = 0
            if (base_type > 0) component_index = find_derived_component(context, &
                base_type, node%component_name)
            actual_is_rank2_allocatable_component = &
                component_alloc_rank_at(context, base_type, component_index) == 2
        end select
    end function actual_is_rank2_allocatable_component

    recursive logical function actual_is_array_expression(arena, node_index, context) &
            result(is_array)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        integer :: i, left_index, right_index, line, column
        character(len=:), allocatable :: op, error_text

        is_array = .false.
        if (.not. node_exists(arena, node_index)) return
        if (actual_is_whole_array(arena, node_index, context)) then
            is_array = .true.
            return
        end if
        if (is_binary_op(arena, node_index)) then
            call get_binary_op_info(arena, node_index, op, left_index, &
                right_index, line, column, error_text)
            if (len_trim(error_text) > 0) return
            is_array = actual_is_array_expression(arena, left_index, context) .or. &
                       actual_is_array_expression(arena, right_index, context)
            return
        end if
        select type (node => arena%entries(node_index)%node)
        type is (array_literal_node)
            is_array = allocated(node%element_indices)
        type is (array_slice_node)
            is_array = .true.
        type is (call_or_subscript_node)
            if (is_array_result_call(arena, node_index, context)) then
                is_array = .true.
            else if (allocated(node%arg_indices)) then
                ! Elementwise intrinsic expressions (for example abs(a)) carry
                ! their array rank through an array actual argument. Reductions
                ! are deliberately excluded: they return a scalar.
                if (allocated(node%name)) then
                    if (same_name(node%name, 'sum') .or. &
                        same_name(node%name, 'product') .or. &
                        same_name(node%name, 'count') .or. &
                        same_name(node%name, 'any') .or. &
                        same_name(node%name, 'all')) return
                    if (same_name(node%name, 'aint') .or. &
                        same_name(node%name, 'anint')) then
                        do i = 1, size(node%arg_indices)
                            if (actual_is_array_expression(arena, node%arg_indices(i), &
                                                            context)) then
                                is_array = .true.
                                return
                            end if
                        end do
                    end if
                end if
                if (.not. node%is_intrinsic) return
                do i = 1, size(node%arg_indices)
                    if (actual_is_array_expression(arena, node%arg_indices(i), &
                                                    context)) then
                        is_array = .true.
                        return
                    end if
                end do
            end if
        end select
    end function actual_is_array_expression

    subroutine array_expression_actual_address(arena, node_index, context, &
                                               callee_name, param_pos, default_kind, &
                                               address, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: callee_name
        integer, intent(in) :: param_pos
        integer, intent(in) :: default_kind
        type(lr_operand_desc_t), intent(out) :: address
        character(len=:), allocatable, intent(out) :: error_msg
        type(array_expr_plan_t) :: plan
        integer :: extent, target_kind, target_symbol
        logical :: known

        call set_empty(error_msg)
        call array_expression_static_extent(arena, node_index, context, extent, &
                                            known)
        if (.not. known .or. extent <= 0) then
            error_msg = 'array-valued actual has no statically known extent in '// &
                        trim(callee_name)
            return
        end if
        target_kind = callee_dummy_value_kind(arena, callee_name, param_pos, &
                                              context)
        if (target_kind <= 0) target_kind = default_kind
        call create_array_expression_temp(context, target_kind, extent, 0, &
                                          target_symbol, error_msg)
        if (len_trim(error_msg) > 0) return
        call build_array_expression(context, target_symbol, node_index, plan, &
                                    error_msg)
        if (len_trim(error_msg) > 0) return
        call emit_array_expression_assignment(arena, plan, context, error_msg)
        if (len_trim(error_msg) > 0) return
        address = context%symbols(target_symbol)%element_address
    end subroutine array_expression_actual_address

    logical function registered_procedure_body_missing(arena, context, name) &
            result(missing)
        ! A signature declared by an interface inside a contained scope names a
        ! dummy procedure of that scope only when the interfaced name also
        ! appears in that scope's dummy-argument list; such a call is lowerable
        ! only when the body is present in the lowering unit.  An interface for
        ! any other name declares an external procedure the linker resolves
        ! (#582), exactly like a host-level interface block, which a
        ! procedure-only object supplies at link time (#416).  A real
        ! procedure with an empty body is still available; external procedures
        ! are deliberately left unknown so the separate-compilation and bind(C)
        ! paths can handle them.
        type(ast_arena_t), intent(in) :: arena
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: name
        integer :: nested_index

        missing = .false.
        nested_index = visible_nested_procedure_node(context, name)
        if (nested_index > 0) then
            missing = procedure_node_is_interface_signature(arena, nested_index)
            if (missing) then
                if (.not. enclosing_dummy_argument_name(arena, context, name)) &
                    missing = .false.
            end if
            if (missing .and. procedure_body_node_present(arena, context, name)) &
                missing = .false.
            return
        end if
    end function registered_procedure_body_missing

    logical function enclosing_dummy_argument_name(arena, context, name) &
            result(is_dummy)
        ! Whether name is a dummy argument of the procedure currently being
        ! lowered.  An interface block in a procedure's specification part
        ! declares a dummy procedure only for such a name; every other name it
        ! declares is an external procedure resolved at link time (#582).
        type(ast_arena_t), intent(in) :: arena
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: name
        character(len=:), allocatable :: param, name_err
        integer, allocatable :: params(:)
        integer :: i

        is_dummy = .false.
        call procedure_node_param_indices(arena, context%current_proc_node_index, &
                                          params)
        if (.not. allocated(params)) return
        do i = 1, size(params)
            call parameter_name(arena, params(i), param, name_err)
            if (len_trim(name_err) > 0) cycle
            if (same_name(param, name)) then
                is_dummy = .true.
                return
            end if
        end do
    end function enclosing_dummy_argument_name

    subroutine procedure_node_param_indices(arena, node_index, params)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        integer, allocatable, intent(out) :: params(:)
        character(len=:), allocatable :: node_type
        type(function_def_node), pointer :: fn_node
        type(subroutine_def_node), pointer :: sb_node

        if (node_index <= 0) return
        if (.not. node_exists(arena, node_index)) return
        node_type = get_node_type_at(arena, node_index)
        if (node_type == 'function_def_node' .or. node_type == 'function_def') then
            fn_node => get_node_as_function_def(arena, node_index)
            if (.not. associated(fn_node)) return
            if (allocated(fn_node%param_indices)) params = fn_node%param_indices
        else if (node_type == 'subroutine_def_node' .or. &
                 node_type == 'subroutine_def') then
            sb_node => get_node_as_subroutine_def(arena, node_index)
            if (.not. associated(sb_node)) return
            if (allocated(sb_node%param_indices)) params = sb_node%param_indices
        end if
    end subroutine procedure_node_param_indices

    logical function procedure_node_is_interface_signature(arena, node_index) &
            result(is_signature)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        integer :: parent_index
        character(len=:), allocatable :: parent_type

        is_signature = .false.
        if (node_index <= 0 .or. .not. node_exists(arena, node_index)) return
        parent_index = arena%entries(node_index)%parent_index
        do while (parent_index > 0)
            parent_type = get_node_type_at(arena, parent_index)
            if (parent_type == 'interface_block_node' .or. &
                parent_type == 'interface_block') then
                is_signature = .true.
                return
            end if
            parent_index = arena%entries(parent_index)%parent_index
        end do
    end function procedure_node_is_interface_signature

    logical function procedure_body_node_present(arena, context, name) &
            result(present)
        type(ast_arena_t), intent(in) :: arena
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: name
        character(len=:), allocatable :: candidate_name
        integer :: i

        present = .false.
        do i = 1, arena%size
            if (.not. procedure_unit_name(arena, i, candidate_name)) cycle
            if (.not. same_name(candidate_name, name)) cycle
            if (context%current_proc_node_index > 0) then
                if (.not. procedure_scope_is(arena, i, &
                                              context%current_proc_node_index)) cycle
            end if
            if (.not. procedure_node_is_interface_signature(arena, i)) then
                present = .true.
                return
            end if
        end do
    end function procedure_body_node_present

    logical function call_has_array_dummy(arena, callee_name, arg_indices, sp) &
            result(has_array)
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: callee_name
        integer, intent(in) :: arg_indices(:)
        integer, intent(in) :: sp
        integer :: i, param_pos

        has_array = .false.
        do i = 1, size(arg_indices)
            param_pos = call_arg_param_pos(i, sp)
            if (callee_dummy_is_array(arena, callee_name, param_pos)) then
                has_array = .true.
                return
            end if
        end do
    end function call_has_array_dummy

    subroutine check_scalar_actual_compatibility(arena, actual_index, context, &
                                                 callee_name, param_pos, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: actual_index
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: callee_name
        integer, intent(in) :: param_pos
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: dummy_kind, actual_kind
        integer :: dummy_class, actual_class

        call set_empty(error_msg)
        if (callee_dummy_is_array(arena, callee_name, param_pos)) return
        if (callee_dummy_is_class_star(arena, callee_name, param_pos)) return
        if (callee_dummy_is_inferred(arena, callee_name, param_pos)) return
        dummy_kind = callee_dummy_value_kind(arena, callee_name, param_pos, &
                                             context)
        if (dummy_kind == 0) dummy_kind = external_dummy_value_kind(context, &
            callee_name, param_pos)
        if (dummy_kind == 0) return
        actual_kind = actual_argument_value_kind(arena, actual_index, context, &
                                                 dummy_kind)
        if (actual_kind == 0) return
        dummy_class = scalar_value_class(dummy_kind)
        actual_class = scalar_value_class(actual_kind)
        if (dummy_class == 0 .or. actual_class == 0) return
        if (dummy_class /= actual_class) then
            error_msg = 'mismatched scalar type in argument to '//trim(callee_name)
            return
        end if
        if (dummy_kind /= actual_kind .and. is_identifier(arena, actual_index)) then
            error_msg = 'mismatched scalar kind in argument to '//trim(callee_name)
            return
        end if
    end subroutine check_scalar_actual_compatibility

    logical function callee_dummy_is_inferred(arena, callee_name, param_pos) &
            result(is_inferred)
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: callee_name
        integer, intent(in) :: param_pos
        character(len=:), allocatable :: node_type, name, name_err
        type(function_def_node), pointer :: fn_node
        type(subroutine_def_node), pointer :: sb_node
        integer, allocatable :: param_indices(:), body_indices(:)
        integer :: n, i, j

        is_inferred = .true.
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            node_type = get_node_type_at(arena, n)
            if (node_type == 'function_def_node' .or. node_type == 'function_def') then
                fn_node => get_node_as_function_def(arena, n)
                if (.not. associated(fn_node) .or. .not. allocated(fn_node%name)) cycle
                if (.not. same_name(fn_node%name, callee_name)) cycle
                param_indices = fn_node%param_indices
                body_indices = fn_node%body_indices
            else if (node_type == 'subroutine_def_node' .or. &
                     node_type == 'subroutine_def') then
                sb_node => get_node_as_subroutine_def(arena, n)
                if (.not. associated(sb_node) .or. .not. allocated(sb_node%name)) cycle
                if (.not. same_name(sb_node%name, callee_name)) cycle
                param_indices = sb_node%param_indices
                body_indices = sb_node%body_indices
            else
                cycle
            end if
            if (.not. allocated(param_indices)) return
            if (param_pos < 1 .or. param_pos > size(param_indices)) return
            call parameter_name(arena, param_indices(param_pos), name, name_err)
            if (len_trim(name_err) > 0) return
            if (.not. allocated(body_indices)) return
            do i = 1, size(body_indices)
                if (.not. node_exists(arena, body_indices(i))) cycle
                select type (decl => arena%entries(body_indices(i))%node)
                type is (declaration_node)
                    if (allocated(decl%var_name)) then
                        if (same_name(decl%var_name, name)) then
                            is_inferred = decl%is_inferred
                            return
                        end if
                    end if
                    if (decl%is_multi_declaration .and. &
                        allocated(decl%var_names)) then
                        do j = 1, size(decl%var_names)
                            if (same_name(decl%var_names(j), name)) then
                                is_inferred = decl%is_inferred
                                return
                            end if
                        end do
                    end if
                end select
            end do
            return
        end do
    end function callee_dummy_is_inferred

    integer function actual_argument_value_kind(arena, node_index, context, &
                                                dummy_kind) result(value_kind)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: dummy_kind
        character(len=:), allocatable :: name, name_err
        integer :: symbol_index

        value_kind = 0
        if (actual_is_character(arena, node_index, context)) then
            value_kind = VALUE_CHARACTER
            return
        end if
        if (actual_is_derived(arena, node_index, context)) then
            value_kind = VALUE_DERIVED
            return
        end if
        if (is_identifier(arena, node_index)) then
            call get_identifier_name(arena, node_index, name, name_err)
            if (len_trim(name_err) > 0) return
            symbol_index = resolve_symbol_at_node(context, node_index, name)
            if (symbol_index > 0) then
                value_kind = context%symbols(symbol_index)%value_kind
                return
            end if
        end if
        if ((dummy_kind == VALUE_C4 .or. dummy_kind == VALUE_C8) .and. &
            is_complex_valued(arena, node_index, context, dummy_kind)) then
            value_kind = dummy_kind
            return
        end if
        value_kind = expression_value_kind(arena, node_index, context, dummy_kind)
    end function actual_argument_value_kind

    integer function scalar_value_class(value_kind) result(value_class)
        integer, intent(in) :: value_kind

        select case (value_kind)
        case (VALUE_I8, VALUE_I16, VALUE_I32, VALUE_I64)
            value_class = 1
        case (VALUE_F32, VALUE_F64)
            value_class = 2
        case (VALUE_C4, VALUE_C8)
            value_class = 3
        case (VALUE_LOGICAL)
            value_class = 4
        case (VALUE_CHARACTER)
            value_class = 5
        case (VALUE_DERIVED)
            value_class = 6
        case default
            value_class = 0
        end select
    end function scalar_value_class

    subroutine check_explicit_shape_actual_size(arena, actual_index, context, &
                                                callee_name, param_pos, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: actual_index
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: callee_name
        integer, intent(in) :: param_pos
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int64_t) :: dummy_count
        integer(c_int64_t) :: actual_count
        logical :: dummy_known
        logical :: actual_known

        call callee_dummy_explicit_element_count(arena, context, callee_name, &
                                                 param_pos, dummy_count, &
                                                 dummy_known, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. dummy_known) return

        call actual_available_element_count(arena, actual_index, context, &
                                            actual_count, actual_known, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. actual_known) return

        if (actual_count < dummy_count) then
            error_msg = 'Actual argument contains too few elements in call to '// &
                        trim(callee_name)
            return
        end if
        call set_empty(error_msg)
    end subroutine check_explicit_shape_actual_size

    subroutine callee_dummy_explicit_element_count(arena, context, callee_name, &
                                                   param_pos, element_count, known, &
                                                   error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: callee_name
        integer, intent(in) :: param_pos
        integer(c_int64_t), intent(out) :: element_count
        logical, intent(out) :: known
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: n
        character(len=:), allocatable :: name
        character(len=:), allocatable :: name_err
        character(len=:), allocatable :: node_type
        type(function_def_node), pointer :: fn_node
        type(subroutine_def_node), pointer :: sb_node

        element_count = 0_c_int64_t
        known = .false.
        call set_empty(error_msg)
        call registered_dummy_explicit_element_count(context, callee_name, param_pos, &
                                                     element_count, known, &
                                                     error_msg)
        if (len_trim(error_msg) > 0) return
        if (known) return
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            node_type = get_node_type_at(arena, n)
            if (node_type == 'subroutine_def_node' .or. &
                node_type == 'subroutine_def') then
                sb_node => get_node_as_subroutine_def(arena, n)
                if (.not. associated(sb_node)) cycle
                if (.not. allocated(sb_node%name)) cycle
                if (.not. same_name(sb_node%name, callee_name)) cycle
                if (.not. allocated(sb_node%param_indices)) return
                if (param_pos < 1) return
                if (param_pos > size(sb_node%param_indices)) return
                call parameter_name(arena, sb_node%param_indices(param_pos), &
                                    name, name_err)
                if (len_trim(name_err) > 0) return
                call dummy_explicit_element_count(arena, context, &
                                                  sb_node%body_indices, name, &
                                                  element_count, known, error_msg)
                return
            else if (node_type == 'function_def_node' .or. &
                     node_type == 'function_def') then
                fn_node => get_node_as_function_def(arena, n)
                if (.not. associated(fn_node)) cycle
                if (.not. allocated(fn_node%name)) cycle
                if (.not. same_name(fn_node%name, callee_name)) cycle
                if (.not. allocated(fn_node%param_indices)) return
                if (param_pos < 1) return
                if (param_pos > size(fn_node%param_indices)) return
                call parameter_name(arena, fn_node%param_indices(param_pos), &
                                    name, name_err)
                if (len_trim(name_err) > 0) return
                call dummy_explicit_element_count(arena, context, &
                                                  fn_node%body_indices, name, &
                                                  element_count, known, error_msg)
                return
            end if
        end do
    end subroutine callee_dummy_explicit_element_count

    subroutine registered_dummy_explicit_element_count(context, callee_name, &
                                                       param_pos, element_count, &
                                                       known, error_msg)
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: callee_name
        integer, intent(in) :: param_pos
        integer(c_int64_t), intent(out) :: element_count
        logical, intent(out) :: known
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i
        integer :: node_idx
        character(len=:), allocatable :: name
        character(len=:), allocatable :: name_err
        character(len=:), allocatable :: node_type
        type(function_def_node), pointer :: fn_node
        type(subroutine_def_node), pointer :: sb_node

        element_count = 0_c_int64_t
        known = .false.
        call set_empty(error_msg)
        do i = 1, context%function_count
            if (.not. same_name(context%function_names(i), callee_name)) cycle
            node_idx = context%function_node_indices(i)
            if (node_idx <= 0) return
            if (.not. node_exists(context%arena, node_idx)) return
            node_type = get_node_type_at(context%arena, node_idx)
            if (node_type == 'subroutine_def_node' .or. &
                node_type == 'subroutine_def') then
                sb_node => get_node_as_subroutine_def(context%arena, node_idx)
                if (.not. associated(sb_node)) return
                if (.not. allocated(sb_node%param_indices)) return
                if (param_pos < 1) return
                if (param_pos > size(sb_node%param_indices)) return
                call parameter_name(context%arena, sb_node%param_indices(param_pos), &
                                    name, name_err)
                if (len_trim(name_err) > 0) return
                call dummy_explicit_element_count(context%arena, context, &
                                                  sb_node%body_indices, name, &
                                                  element_count, known, error_msg)
                return
            else if (node_type == 'function_def_node' .or. &
                     node_type == 'function_def') then
                fn_node => get_node_as_function_def(context%arena, node_idx)
                if (.not. associated(fn_node)) return
                if (.not. allocated(fn_node%param_indices)) return
                if (param_pos < 1) return
                if (param_pos > size(fn_node%param_indices)) return
                call parameter_name(context%arena, fn_node%param_indices(param_pos), &
                                    name, name_err)
                if (len_trim(name_err) > 0) return
                call dummy_explicit_element_count(context%arena, context, &
                                                  fn_node%body_indices, name, &
                                                  element_count, known, error_msg)
                return
            end if
        end do
    end subroutine registered_dummy_explicit_element_count

    subroutine dummy_explicit_element_count(arena, context, body_indices, &
                                            param_name, element_count, known, &
                                            error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(lowering_context_t), intent(in) :: context
        integer, allocatable, intent(in) :: body_indices(:)
        character(len=*), intent(in) :: param_name
        integer(c_int64_t), intent(out) :: element_count
        logical, intent(out) :: known
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i
        integer :: lower_bound
        integer :: array_size

        element_count = 0_c_int64_t
        known = .false.
        call set_empty(error_msg)
        if (.not. allocated(body_indices)) return
        do i = 1, size(body_indices)
            if (.not. node_exists(arena, body_indices(i))) cycle
            select type (decl => arena%entries(body_indices(i))%node)
            type is (declaration_node)
                if (.not. decl%is_array) cycle
                if (decl%is_allocatable) cycle
                if (.not. declaration_names_var(decl, param_name)) cycle
                if (declaration_is_assumed_shape(decl, context)) return
                if (declaration_is_assumed_rank(decl, context)) return
                if (declaration_is_assumed_size(decl, context)) return
                call get_array_bounds(decl, context, lower_bound, array_size, &
                                      error_msg)
                if (len_trim(error_msg) > 0) then
                    call set_empty(error_msg)
                    return
                end if
                element_count = int(array_size, c_int64_t)
                known = .true.
                return
            end select
        end do
    end subroutine dummy_explicit_element_count

    subroutine actual_available_element_count(arena, node_index, context, &
                                              element_count, known, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        integer(c_int64_t), intent(out) :: element_count
        logical, intent(out) :: known
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: id_name
        integer :: symbol_index
        integer, allocatable :: flat(:)

        element_count = 0_c_int64_t
        known = .false.
        call set_empty(error_msg)
        if (.not. node_exists(arena, node_index)) return

        if (is_identifier(arena, node_index)) then
            call get_identifier_name(arena, node_index, id_name, error_msg)
            if (len_trim(error_msg) > 0) return
            symbol_index = resolve_symbol_at_node(context, node_index, id_name)
            if (symbol_index <= 0) return
            if (.not. context%symbols(symbol_index)%is_array) return
            element_count = int(context%symbols(symbol_index)%array_size, &
                                c_int64_t)
            known = .true.
            return
        end if

        select type (node => arena%entries(node_index)%node)
        type is (array_slice_node)
            call section_available_element_count(arena, node, context, &
                                                 element_count, known, error_msg)
        type is (call_or_subscript_node)
            call call_actual_available_element_count(arena, node, node_index, context, &
                                                     element_count, known, &
                                                     error_msg)
        type is (array_literal_node)
            if (.not. allocated(node%element_indices)) return
            allocate (flat(0))
            call flatten_constructor_elements(arena, node%element_indices, flat, &
                                              error_msg)
            if (len_trim(error_msg) > 0) return
            element_count = int(size(flat), c_int64_t)
            known = .true.
        end select
    end subroutine actual_available_element_count

    subroutine section_available_element_count(arena, node, context, element_count, &
                                               known, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(array_slice_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        integer(c_int64_t), intent(out) :: element_count
        logical, intent(out) :: known
        character(len=:), allocatable, intent(out) :: error_msg
        type(array_section_info_t) :: info
        integer(c_int64_t) :: first_linear

        element_count = 0_c_int64_t
        known = .false.
        call describe_array_section(arena, node, context, info, error_msg, &
                                    allow_derived=.true.)
        if (len_trim(error_msg) > 0) return
        if (info%result_rank == 0) then
            first_linear = array_section_source_linear_index(info, 0_c_int64_t)
            element_count = int(context%symbols(info%source_index)%array_size, &
                                c_int64_t) - first_linear
        else
            element_count = array_section_total_elements(info)
        end if
        if (element_count < 0_c_int64_t) element_count = 0_c_int64_t
        known = .true.
        call set_empty(error_msg)
    end subroutine section_available_element_count

    subroutine call_actual_available_element_count(arena, node, node_index, context, &
                                                   element_count, known, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        integer(c_int64_t), intent(out) :: element_count
        logical, intent(out) :: known
        character(len=:), allocatable, intent(out) :: error_msg
        logical :: is_element_ref

        element_count = 0_c_int64_t
        known = .false.
        call set_empty(error_msg)
        if (call_is_reshape(node)) then
            call reshape_actual_element_count(arena, node, context, element_count, &
                                              known, error_msg)
            return
        end if

        is_element_ref = node%is_array_access
        if (.not. is_element_ref) is_element_ref = &
            is_declared_array_element_ref(node, context)
        if (.not. is_element_ref) return
        call array_element_available_count(arena, node, node_index, context, element_count, &
                                           known, error_msg)
    end subroutine call_actual_available_element_count

    logical function call_is_reshape(node)
        type(call_or_subscript_node), intent(in) :: node

        call_is_reshape = .false.
        if (.not. allocated(node%name)) return
        call_is_reshape = same_name(node%name, 'reshape')
    end function call_is_reshape

    subroutine reshape_actual_element_count(arena, node, context, element_count, &
                                            known, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        integer(c_int64_t), intent(out) :: element_count
        logical, intent(out) :: known
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: source_index
        integer :: shape_index
        integer :: pad_index
        integer :: order_index

        element_count = 0_c_int64_t
        known = .false.
        call resolve_reshape_args(arena, node, source_index, shape_index, &
                                  pad_index, order_index, error_msg)
        if (len_trim(error_msg) > 0) return
        call reshape_shape_element_count(arena, shape_index, context, &
                                         element_count, known, error_msg)
    end subroutine reshape_actual_element_count

    subroutine reshape_shape_element_count(arena, shape_index, context, &
                                           element_count, known, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: shape_index
        type(lowering_context_t), intent(in) :: context
        integer(c_int64_t), intent(out) :: element_count
        logical, intent(out) :: known
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i
        integer :: shape_sym
        integer(c_int64_t) :: shape_value

        element_count = 1_c_int64_t
        known = .false.
        call set_empty(error_msg)
        if (.not. node_exists(arena, shape_index)) return

        select type (shape_node => arena%entries(shape_index)%node)
        type is (array_literal_node)
            if (.not. allocated(shape_node%element_indices)) return
            do i = 1, size(shape_node%element_indices)
                call eval_i32_constant(arena, shape_node%element_indices(i), &
                                       context, shape_value, error_msg)
                if (len_trim(error_msg) > 0) return
                element_count = element_count * shape_value
            end do
            known = .true.
        class default
            call resolve_shape_of_symbol(arena, shape_index, context, shape_sym, &
                                         error_msg)
            if (len_trim(error_msg) > 0) then
                call set_empty(error_msg)
                element_count = 0_c_int64_t
                return
            end if
            if (shape_sym <= 0) then
                element_count = 0_c_int64_t
                return
            end if
            do i = 1, context%symbols(shape_sym)%array_rank
                element_count = element_count * &
                    int(context%symbols(shape_sym)%array_dim_sizes(i), c_int64_t)
            end do
            known = .true.
        end select
    end subroutine reshape_shape_element_count

    subroutine array_element_available_count(arena, node, node_index, context, element_count, &
                                             known, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        integer(c_int64_t), intent(out) :: element_count
        logical, intent(out) :: known
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: symbol_index
        integer :: rank
        integer :: arg_count
        integer :: k
        integer(c_int64_t) :: idx
        integer(c_int64_t) :: lower_k
        integer(c_int64_t) :: extent_k
        integer(c_int64_t) :: stride
        integer(c_int64_t) :: linear_index

        element_count = 0_c_int64_t
        known = .false.
        call set_empty(error_msg)
        if (.not. allocated(node%name)) return
        if (.not. allocated(node%arg_indices)) return
        symbol_index = resolve_symbol_at_node(context, node_index, node%name)
        if (symbol_index <= 0) return
        if (.not. context%symbols(symbol_index)%is_array) return
        rank = context%symbols(symbol_index)%array_rank
        arg_count = size(node%arg_indices)
        if (arg_count /= rank) return

        linear_index = 0_c_int64_t
        stride = 1_c_int64_t
        do k = 1, rank
            call eval_i32_constant(arena, node%arg_indices(k), context, idx, &
                                   error_msg)
            if (len_trim(error_msg) > 0) then
                call set_empty(error_msg)
                return
            end if
            lower_k = int(context%symbols(symbol_index)%array_dim_lowers(k), &
                          c_int64_t)
            extent_k = int(context%symbols(symbol_index)%array_dim_sizes(k), &
                           c_int64_t)
            linear_index = linear_index + (idx - lower_k) * stride
            stride = stride * extent_k
        end do

        element_count = int(context%symbols(symbol_index)%array_size, c_int64_t) - &
                        linear_index
        if (element_count < 0_c_int64_t) element_count = 0_c_int64_t
        known = .true.
        call set_empty(error_msg)
    end subroutine array_element_available_count

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
        symbol_index = resolve_symbol_at_node(context, node_index, id_name)
        if (symbol_index > 0) actual_is_derived = &
            context%symbols(symbol_index)%is_derived
    end function actual_is_derived

    logical function actual_is_whole_array(arena, node_index, context)
        ! Whether an actual argument is a plain whole-array identifier (not an
        ! element access) backed by explicit-shape, non-allocatable storage.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        character(len=:), allocatable :: id_name, id_err
        integer :: symbol_index

        actual_is_whole_array = .false.
        if (.not. is_identifier(arena, node_index)) return
        call get_identifier_name(arena, node_index, id_name, id_err)
        if (len_trim(id_err) > 0) return
        symbol_index = resolve_symbol_at_node(context, node_index, id_name)
        if (symbol_index <= 0) return
        actual_is_whole_array = context%symbols(symbol_index)%is_array .and. &
                                .not. context%symbols(symbol_index)%is_allocatable
    end function actual_is_whole_array

    logical function actual_is_definite_scalar(arena, node_index, context)
        ! Whether an actual argument is statically certain to be scalar (rank 0):
        ! a plain scalar literal, or an identifier bound to a non-array symbol.
        ! Array elements and sections are call_or_subscript nodes, not
        ! identifiers, so sequence-associated scalars (e.g. b(3)) never match.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        character(len=:), allocatable :: id_name, id_err
        integer :: symbol_index

        actual_is_definite_scalar = .false.
        if (.not. node_exists(arena, node_index)) return
        if (get_node_type_at(arena, node_index) == 'literal') then
            actual_is_definite_scalar = .true.
            return
        end if
        if (.not. is_identifier(arena, node_index)) return
        call get_identifier_name(arena, node_index, id_name, id_err)
        if (len_trim(id_err) > 0) return
        symbol_index = resolve_symbol_at_node(context, node_index, id_name)
        if (symbol_index <= 0) return
        if (context%symbols(symbol_index)%is_array) return
        if (context%symbols(symbol_index)%array_rank > 0) return
        actual_is_definite_scalar = .true.
    end function actual_is_definite_scalar

    subroutine whole_array_actual_address(arena, node_index, context, address, &
                                          error_msg)
        ! Pass an explicit-shape array actual by reference: its contiguous base
        ! address (element_address), matching the dummy parameter-pointer view.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: address
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: id_name
        integer :: symbol_index

        call get_identifier_name(arena, node_index, id_name, error_msg)
        if (len_trim(error_msg) > 0) return
        symbol_index = resolve_symbol_at_node(context, node_index, id_name)
        if (symbol_index <= 0) then
            error_msg = 'array argument was not declared: '//trim(id_name)
            return
        end if
        address = context%symbols(symbol_index)%element_address
        call set_empty(error_msg)
    end subroutine whole_array_actual_address

    logical function actual_is_contiguous_section(arena, node_index, context)
        ! Whether an actual argument is a rank-1 array section whose elements are
        ! adjacent in storage: a stride-1 subscript triplet on a rank-1 array
        ! (a(lo:hi)) or a whole leading dimension of a rank-2 array (a(:,j)).
        ! Such a section can bind a rank-1 dummy through its base pointer.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        type(array_section_info_t) :: info
        character(len=:), allocatable :: err
        integer :: keep, dd
        integer(c_int64_t) :: step

        actual_is_contiguous_section = .false.
        if (.not. node_exists(arena, node_index)) return
        select type (slice => arena%entries(node_index)%node)
        type is (array_slice_node)
            call describe_array_section(arena, slice, context, info, err, &
                                        allow_derived=.true.)
            if (len_trim(err) > 0) return
            if (info%result_rank /= 1) return
            keep = info%kept_dims(1)
            if (info%section_strides(keep) /= 1_c_int64_t) return
            step = 1_c_int64_t
            do dd = 1, keep - 1
                step = step*info%source_sizes(dd)
            end do
            if (step /= 1_c_int64_t) return
            actual_is_contiguous_section = .true.
        end select
    end function actual_is_contiguous_section

    subroutine section_actual_address(arena, node_index, context, address, &
                                      error_msg)
        ! Base address of a contiguous rank-1 section's first element, computed
        ! from the source array's storage and the section's compile-time start.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: address
        character(len=:), allocatable, intent(out) :: error_msg
        type(array_section_info_t) :: info
        type(lr_operand_desc_t) :: index_op
        integer(c_int64_t) :: first_linear
        integer :: vk, arr_size, src

        select type (slice => arena%entries(node_index)%node)
        type is (array_slice_node)
            call describe_array_section(arena, slice, context, info, error_msg, &
                                        allow_derived=.true.)
            if (len_trim(error_msg) > 0) return
            src = info%source_index
            first_linear = array_section_source_linear_index(info, 0_c_int64_t)
            vk = context%symbols(src)%value_kind
            arr_size = context%symbols(src)%array_size
            if (vk == VALUE_DERIVED) then
                ! Derived storage is a flat i32 slot array: one element spans
                ! slots(type) slots, so the linear element index scales.
                first_linear = first_linear* &
                    int(total_component_slots(context, &
                        context%symbols(src)%derived_type_index), c_int64_t)
            end if
            index_op = i32_immediate(context%session, first_linear)
            select case (vk)
            case (VALUE_F32)
                if (.not. emit_f32_array_element_addr(context%session, arr_size, &
                        context%symbols(src)%element_address, index_op, address, &
                        error_msg)) return
            case (VALUE_F64)
                if (.not. emit_f64_array_element_addr(context%session, arr_size, &
                        context%symbols(src)%element_address, index_op, address, &
                        error_msg)) return
            case default
                if (.not. emit_i32_array_element_addr(context%session, arr_size, &
                        context%symbols(src)%element_address, index_op, address, &
                        error_msg)) return
            end select
            call set_empty(error_msg)
        class default
            error_msg = 'array section actual expected an array slice node'
        end select
    end subroutine section_actual_address

    logical function actual_is_allocatable_array(arena, node_index, context)
        ! Whether an actual argument is a plain allocatable-array identifier.
        ! Allocatable arrays never set is_array (only array_rank), unlike
        ! fixed-size arrays, so array_rank > 0 is the array/scalar signal.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        character(len=:), allocatable :: id_name, id_err
        integer :: symbol_index

        actual_is_allocatable_array = .false.
        if (.not. is_identifier(arena, node_index)) return
        call get_identifier_name(arena, node_index, id_name, id_err)
        if (len_trim(id_err) > 0) return
        symbol_index = resolve_symbol_at_node(context, node_index, id_name)
        if (symbol_index <= 0) return
        actual_is_allocatable_array = &
            context%symbols(symbol_index)%is_allocatable .and. &
            context%symbols(symbol_index)%array_rank > 0
    end function actual_is_allocatable_array

    subroutine allocatable_array_actual_descriptor_address(arena, node_index, &
                                                            context, address, &
                                                            error_msg)
        ! Pass an allocatable array actual by reference to its own 40-byte
        ! descriptor (data pointer + bounds); the callee aliases the same
        ! descriptor so allocate/deallocate inside the callee are visible to
        ! the caller (W11).
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: address
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: id_name
        integer :: symbol_index

        call get_identifier_name(arena, node_index, id_name, error_msg)
        if (len_trim(error_msg) > 0) return
        symbol_index = resolve_symbol_at_node(context, node_index, id_name)
        if (symbol_index <= 0) then
            error_msg = 'allocatable array argument was not declared: '// &
                        trim(id_name)
            return
        end if
        address = context%symbols(symbol_index)%allocatable_descriptor_address
        call set_empty(error_msg)
    end subroutine allocatable_array_actual_descriptor_address

    subroutine class_or_derived_actual(arena, node_index, context, callee_name, &
                                       param_pos, address, error_msg)
        ! Route a derived actual by the dummy's spelling. class(t) takes the
        ! canonical scalar class descriptor (#400) so declared and dynamic type
        ! identity both reach the callee; type(t) keeps the existing bare
        ! by-reference ABI unchanged.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: callee_name
        integer, intent(in) :: param_pos
        type(lr_operand_desc_t), intent(out) :: address
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: id_name
        integer :: actual_index, declared_index, symbol_index

        call class_dummy_declared_type_index(context, arena, callee_name, &
                                             param_pos, declared_index)
        if (declared_index <= 0) then
            call derived_actual_address(arena, node_index, context, address, &
                                        error_msg)
            return
        end if

        call get_identifier_name(arena, node_index, id_name, error_msg)
        if (len_trim(error_msg) > 0) return
        symbol_index = resolve_symbol_at_node(context, node_index, id_name)
        if (symbol_index <= 0) then
            error_msg = 'derived argument was not declared: '//trim(id_name)
            return
        end if
        ! Fortran 2018 7.3.2.3: the actual must be type compatible with the
        ! class(t) dummy, i.e. its declared type is t or an extension of t.
        actual_index = context%symbols(symbol_index)%derived_type_index
        if (actual_index <= 0) then
            error_msg = 'Type mismatch in argument to '//trim(callee_name)// &
                ': actual '//trim(id_name)//' has no resolvable declared type'// &
                ' for class('//trim(context%derived_types(declared_index)%name)// &
                ') dummy'
            return
        end if
        if (.not. derived_type_extends(context, actual_index, declared_index)) then
            error_msg = 'Type mismatch in argument to '//trim(callee_name)// &
                ': actual of type '// &
                trim(context%derived_types(actual_index)%name)// &
                ' is not type compatible with class('// &
                trim(context%derived_types(declared_index)%name)//') dummy'
            return
        end if
        call build_class_scalar_descriptor(context, symbol_index, &
                                           declared_index, address, error_msg)
    end subroutine class_or_derived_actual

    logical function name_is_generic_specific(context, name) result(is_specific)
        ! True when name is listed as a specific procedure of some registered
        ! generic interface.
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: name
        integer :: g, s

        is_specific = .false.
        do g = 1, context%generic_count
            do s = 1, context%generics(g)%specific_count
                if (same_name(context%generics(g)%specific_names(s), name)) then
                    is_specific = .true.
                    return
                end if
            end do
        end do
    end function name_is_generic_specific

    subroutine check_derived_actual_type(arena, node_index, context, &
                                         callee_name, param_pos, error_msg)
        ! Fortran 2018 15.5.2.4: an actual associated with a type(t) dummy must
        ! have declared type t. The dummy borrows the actual's storage layout
        ! (including any element component descriptors), so a mismatched type
        ! would silently reinterpret those slots.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: callee_name
        integer, intent(in) :: param_pos
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: id_name, dummy_type_name, name_err
        integer :: symbol_index, actual_index

        call set_empty(error_msg)
        ! Generic resolution currently discriminates on argument kind and rank
        ! only, so a call to a generic whose specifics differ only in their
        ! derived type can reach this point already bound to a sibling
        ! specific. Leave those calls to the generic path rather than
        ! reporting a mismatch the source does not contain.
        if (name_is_generic_specific(context, callee_name)) return
        call callee_dummy_declared_derived_type(arena, callee_name, param_pos, &
                                                .false., dummy_type_name)
        if (len_trim(dummy_type_name) == 0) return
        if (.not. is_identifier(arena, node_index)) return
        call get_identifier_name(arena, node_index, id_name, name_err)
        if (len_trim(name_err) > 0) return
        symbol_index = resolve_symbol_at_node(context, node_index, id_name)
        if (symbol_index <= 0) return
        actual_index = context%symbols(symbol_index)%derived_type_index
        if (actual_index <= 0) return
        if (same_name(context%derived_types(actual_index)%name, &
                      dummy_type_name)) return
        error_msg = 'Type mismatch in argument to '//trim(callee_name)// &
            ': actual of type '// &
            trim(context%derived_types(actual_index)%name)// &
            ' cannot be associated with type('//trim(dummy_type_name)// &
            ') dummy'
    end subroutine check_derived_actual_type

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
        symbol_index = resolve_symbol_at_node(context, node_index, id_name)
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
        symbol_index = resolve_symbol_at_node(context, node_index, id_name)
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

    integer function arg_default_kind(arena, context, callee_name, param_pos, &
                                      fallback_kind) result(kind)
        ! The kind a scalar actual should adopt at param_pos: the callee dummy's
        ! declared kind when it is a contained procedure whose dummy is known, or
        ! a separately compiled module procedure whose dummy kinds came from its
        ! .fmod, otherwise the caller's fallback. This lets a real literal actual
        ! match a real(4)/real(8) dummy independent of the call-site default kind,
        ! which matters when a single call mixes integer and real dummies (e.g. an
        ! array function result called as f(n, step)).
        type(ast_arena_t), intent(in) :: arena
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: callee_name
        integer, intent(in) :: param_pos
        integer, intent(in) :: fallback_kind
        integer :: dummy_kind

        kind = fallback_kind
        dummy_kind = callee_dummy_value_kind(arena, callee_name, param_pos, &
                                             context)
        if (dummy_kind == 0) then
            dummy_kind = external_dummy_value_kind(context, callee_name, &
                                                   param_pos)
        end if
        if (dummy_kind == VALUE_I32 .or. dummy_kind == VALUE_F32 .or. &
            dummy_kind == VALUE_F64 .or. dummy_kind == VALUE_LOGICAL .or. &
            dummy_kind == VALUE_I64 .or. dummy_kind == VALUE_I8 .or. &
            dummy_kind == VALUE_I16 .or. dummy_kind == VALUE_C4 .or. &
            dummy_kind == VALUE_C8 .or. dummy_kind == VALUE_CHARACTER) then
            kind = dummy_kind
        end if
    end function arg_default_kind

    integer function external_dummy_value_kind(context, callee_name, param_pos) &
            result(value_kind)
        ! The registered scalar kind of the param_pos-th dummy of a separately
        ! compiled by-reference module procedure, or 0 when the name is not such
        ! a procedure. Lets a call site materialise each actual with the kind the
        ! .fmod recorded (#284).
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: callee_name
        integer, intent(in) :: param_pos
        integer :: idx

        value_kind = 0
        idx = external_procedure_index(context, callee_name)
        if (idx <= 0) return
        if (.not. context%external_procedures(idx)%by_reference) return
        if (param_pos < 1) return
        if (param_pos > context%external_procedures(idx)%arg_count) return
        value_kind = context%external_procedures(idx)%arg_value_kinds(param_pos)
    end function external_dummy_value_kind

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
        if (value_kind /= VALUE_C4 .and. value_kind /= VALUE_C8) then
            if (is_complex_valued(arena, node_index, context, VALUE_C8)) then
                value_kind = VALUE_C8
            else if (is_complex_valued(arena, node_index, context, VALUE_C4)) then
                value_kind = VALUE_C4
            end if
        end if
        if (value_kind == VALUE_C4 .or. value_kind == VALUE_C8) then
            call make_complex_reference_argument(arena, node_index, context, &
                value_kind, arg, error_msg)
            return
        end if
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
        symbol_index = resolve_symbol_at_node(context, node_index, id_name)
        if (symbol_index <= 0) then
            call set_empty(error_msg)
            return
        end if
        handled = .true.
        if (context%symbols(symbol_index)%is_array) then
            ! Array-to-array passing is handled earlier in prepare_reference_args
            ! after checking the dummy's rank. Reaching here means the dummy is
            ! scalar: a rank mismatch this scalar path cannot honor.
            call unsupported_feature_error('array argument', &
                                           get_node_line(arena, node_index), &
                                           get_node_column(arena, node_index), &
                                           'whole-array procedure arguments '// &
                                           'are not supported', error_msg)
            return
        end if
        if (context%symbols(symbol_index)%value_kind == VALUE_C4 .or. &
            context%symbols(symbol_index)%value_kind == VALUE_C8) then
            if (.not. context%symbols(symbol_index)%has_address) then
                error_msg = 'complex scalar argument has no storage: '//trim(id_name)
                return
            end if
            arg = context%symbols(symbol_index)%address
            call set_empty(error_msg)
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

    subroutine make_complex_reference_argument(arena, node_index, context, &
                                               value_kind, address, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: value_kind
        type(lr_operand_desc_t), intent(out) :: address
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: re_value, im_value, im_address
        integer(c_int64_t) :: bytes, im_offset

        if (value_kind == VALUE_C8) then
            bytes = 16_c_int64_t
            im_offset = 8_c_int64_t
            call eval_c8_rhs(arena, node_index, context, re_value, im_value, &
                             error_msg)
        else
            bytes = 8_c_int64_t
            im_offset = 4_c_int64_t
            call eval_c4_rhs(arena, node_index, context, re_value, im_value, &
                             error_msg)
        end if
        if (len_trim(error_msg) > 0) return
        if (.not. emit_alloca_bytes(context%session, i64_immediate( &
                context%session, bytes), address, error_msg)) return
        if (.not. emit_ptr_offset(context%session, address, im_offset, &
                                  im_address, error_msg)) return
        if (value_kind == VALUE_C8) then
            if (.not. emit_liric_f64_store(context%session, re_value, address, &
                                           error_msg)) return
            if (.not. emit_liric_f64_store(context%session, im_value, im_address, &
                                           error_msg)) return
        else
            if (.not. emit_liric_f32_store(context%session, re_value, address, &
                                           error_msg)) return
            if (.not. emit_liric_f32_store(context%session, im_value, im_address, &
                                           error_msg)) return
        end if
        call set_empty(error_msg)
    end subroutine make_complex_reference_argument

recursive integer function expression_value_kind(arena, node_index, context, &
                                            default_kind) result(value_kind)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: default_kind
        integer :: symbol_index
        integer :: call_arg_count
        integer :: call_arg_kinds(MAX_PROC_ARGS)
        integer :: call_arg_ranks(MAX_PROC_ARGS)
        integer :: external_index
        character(len=:), allocatable :: id_name, id_err, callee_name

        value_kind = default_kind
        if (.not. node_exists(arena, node_index)) return

        if (is_binary_op(arena, node_index)) then
            if (scalar_real_expr_kind(arena, node_index, context) == VALUE_F32) then
                value_kind = VALUE_F32
            else if (scalar_real_expr_kind(arena, node_index, context) &
                     == VALUE_F64) then
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
            else
                ! An integer (or BOZ) literal is i32 unless the dummy is a wider
                ! integer kind: widening the literal to an integer(8/1/2) dummy is
                ! type-safe. A non-integer hint (e.g. a type-bound call's
                ! passed-object offset) still keeps the literal at i32 so it is
                ! never rerouted through a logical/real path.
                if (default_kind == VALUE_I64 .or. default_kind == VALUE_I8 .or. &
                    default_kind == VALUE_I16) then
                    value_kind = default_kind
                else
                    value_kind = VALUE_I32
                end if
            end if
            return
        end if

        if (is_identifier(arena, node_index)) then
            call get_identifier_name(arena, node_index, id_name, id_err)
            symbol_index = resolve_symbol_at_node(context, node_index, id_name)
            if (symbol_index > 0) value_kind = context%symbols(symbol_index)%value_kind
            return
        end if

        select type (node => arena%entries(node_index)%node)
        type is (call_or_subscript_node)
            if (allocated(node%name)) then
                ! Generic interface: resolve full call signature and look up the
                ! matching specific return kind (#249 B7c).
                if (find_generic(context, node%name) > 0) then
                    call call_argument_kinds(arena, node, context, default_kind, &
                        call_arg_count, call_arg_kinds)
                    call call_argument_ranks(arena, node, context, call_arg_count, &
                        call_arg_ranks)
                    value_kind = generic_call_return_kind(context, node%name, &
                        call_arg_count, call_arg_kinds, call_arg_ranks)
                else if (is_contained_f32_function(context, node%name)) then
                    value_kind = VALUE_F32
                else if (is_contained_f64_function(context, node%name)) then
                    value_kind = VALUE_F64
                else if (is_contained_logical_function(context, node%name)) then
                    value_kind = VALUE_LOGICAL
                else if (is_proc_pointer_call(context, node%name)) then
                    call resolve_proc_pointer_callee_name(arena, node%name, &
                                                         callee_name)
                    if (is_contained_f32_function(context, callee_name)) then
                        value_kind = VALUE_F32
                    else if (is_contained_f64_function(context, callee_name)) then
                        value_kind = VALUE_F64
                    end if
                else
                    external_index = external_procedure_index(context, node%name)
                    if (external_index > 0) then
                        value_kind = context%external_procedures( &
                            external_index)%return_value_kind
                    end if
                end if
            end if
        type is (component_access_node)
            value_kind = derived_component_access_kind(arena, node, context)
        end select
    end function expression_value_kind

 recursive subroutine call_argument_kinds(arena, node, context, default_kind, &
                                   arg_count, arg_kinds)
        ! Expand the actual argument kinds at a call site for generic signature
        ! matching.
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: default_kind
        integer, intent(out) :: arg_count
        integer, intent(out) :: arg_kinds(MAX_PROC_ARGS)
        integer :: i

        arg_count = 0
        arg_kinds = default_kind
        if (.not. allocated(node%arg_indices)) return
        arg_count = size(node%arg_indices)
        if (arg_count > MAX_PROC_ARGS) arg_count = MAX_PROC_ARGS
        do i = 1, arg_count
            ! Generic matching uses a real literal's source kind before the
            ! selected procedure applies any contextual conversion.
            if (is_real_literal(arena, node%arg_indices(i))) then
                if (scalar_real_expr_kind(arena, node%arg_indices(i), context) &
                    == VALUE_F32) then
                    arg_kinds(i) = VALUE_F32
                else
                    arg_kinds(i) = VALUE_F64
                end if
            else
                arg_kinds(i) = expression_value_kind(arena, &
                    node%arg_indices(i), context, default_kind)
            end if
        end do
    end subroutine call_argument_kinds

    subroutine call_argument_ranks(arena, node, context, arg_count, arg_ranks)
        ! Expand the actual argument ranks at a call site for generic signature
        ! matching. 0 = scalar, >0 = array rank (refs #303).
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        integer, intent(out) :: arg_count
        integer, intent(out) :: arg_ranks(MAX_PROC_ARGS)
        integer :: i

        arg_count = 0
        arg_ranks = 0
        if (.not. allocated(node%arg_indices)) return
        arg_count = size(node%arg_indices)
        if (arg_count > MAX_PROC_ARGS) arg_count = MAX_PROC_ARGS
        do i = 1, arg_count
            arg_ranks(i) = expression_value_rank(arena, node%arg_indices(i), &
                                                 context)
        end do
    end subroutine call_argument_ranks

    integer function expression_value_rank(arena, node_index, context) result(rank)
        ! Array rank of an expression used as a call argument. 0 for scalar,
        ! >0 for array. Used by generic dispatch to distinguish same-kind
        ! specifics that differ in rank (refs #303).
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        integer :: symbol_index
        character(len=:), allocatable :: id_name, id_err

        rank = 0
        if (.not. node_exists(arena, node_index)) return

        if (is_literal(arena, node_index)) return
        if (is_binary_op(arena, node_index)) return

        if (is_identifier(arena, node_index)) then
            call get_identifier_name(arena, node_index, id_name, id_err)
            if (len_trim(id_err) > 0) return
            symbol_index = resolve_symbol_at_node(context, node_index, id_name)
            if (symbol_index > 0) then
                if (context%symbols(symbol_index)%is_array) then
                    rank = context%symbols(symbol_index)%array_rank
                else if (context%symbols(symbol_index)%array_rank > 0) then
                    rank = context%symbols(symbol_index)%array_rank
                end if
            end if
            return
        end if

        select type (node => arena%entries(node_index)%node)
        type is (call_or_subscript_node)
            if (node%is_array_access) then
                rank = expression_array_access_rank(arena, node, node_index, context)
                return
            end if
        end select
    end function expression_value_rank

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
        case (VALUE_I64)
            call lower_i64_expression(arena, node_index, context, value, error_msg)
        case (VALUE_I8)
            call lower_i8_expression(arena, node_index, context, value, error_msg)
        case (VALUE_I16)
            call lower_i16_expression(arena, node_index, context, value, error_msg)
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
        case (VALUE_I64)
            if (.not. emit_i64_alloca(context%session, address, error_msg)) return
            if (.not. emit_i64_store(context%session, value, address, error_msg)) &
                return
        case (VALUE_I8)
            if (.not. emit_i8_alloca(context%session, address, error_msg)) return
            if (.not. emit_i8_store(context%session, value, address, error_msg)) &
                return
        case (VALUE_I16)
            if (.not. emit_i16_alloca(context%session, address, error_msg)) return
            if (.not. emit_i16_store(context%session, value, address, error_msg)) &
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
            case (VALUE_I64)
                if (.not. emit_i64_load(context%session, args(i), value, &
                                                        error_msg)) return
            case (VALUE_I8)
                if (.not. emit_i8_load(context%session, args(i), value, &
                                                        error_msg)) return
            case (VALUE_I16)
                if (.not. emit_i16_load(context%session, args(i), value, &
                                                        error_msg)) return
            case default
                error_msg = 'direct LIRIC session cannot copy back this argument'
                return
            end select
            context%symbols(copyback_indices(i))%value = value
        end do
        call set_empty(error_msg)
    end subroutine copy_back_reference_args

    integer function expression_array_access_rank(arena, node, node_index, context) result(rank)
        ! Rank of an array access expression: 0 for element access (all indices
        ! are scalar), or the remaining rank for a section (e.g. a(:) keeps rank
        ! 1 of a rank-1 array).
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        integer :: symbol_index, base_rank

        rank = 0
        if (.not. allocated(node%name)) return
        if (len_trim(node%name) == 0) return
        symbol_index = resolve_symbol_at_node(context, node_index, node%name)
        if (symbol_index <= 0) return
        base_rank = 0
        if (context%symbols(symbol_index)%is_array) then
            base_rank = context%symbols(symbol_index)%array_rank
        else if (context%symbols(symbol_index)%array_rank > 0) then
            base_rank = context%symbols(symbol_index)%array_rank
        end if
        if (base_rank <= 0) return
        rank = base_rank - count_scalar_indices(arena, node)
        if (rank < 0) rank = 0
    end function expression_array_access_rank

    integer function count_scalar_indices(arena, node) result(count)
        ! Number of scalar indices in an array access. If all indices are scalar,
        ! the result is a scalar element (rank 0). If some are sections (:), the
        ! result retains the un-indexed dimensions.
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        integer :: i

        count = 0
        if (.not. allocated(node%arg_indices)) return
        do i = 1, size(node%arg_indices)
            if (.not. is_section_index(arena, node%arg_indices(i))) count = count + 1
        end do
    end function count_scalar_indices

    logical function is_section_index(arena, node_index)
        ! Whether a subscript is a full-section selector (:), which preserves
        ! the dimension in the result rank.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index

        is_section_index = .false.
        if (.not. node_exists(arena, node_index)) return
        if (is_literal(arena, node_index)) return
        if (is_identifier(arena, node_index)) return
        if (is_binary_op(arena, node_index)) return
        is_section_index = .true.
    end function is_section_index

    ! A dummy argument that is INTENT(IN) can never be redefined. Lazy Fortran
    ! makes INTENT(IN) the default for a dummy declared without an explicit
    ! INTENT (#438), so in lazy mode an intent-less dummy is protected too,
    ! while a standard Fortran source keeps the language rule that only an
    ! explicit INTENT(IN) protects. The check runs before any code is emitted
    ! so an offending unit is diagnosed instead of silently lowered.
    subroutine check_dummy_intent_definitions(arena, context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(lowering_context_t), intent(in) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: n

        call set_empty(error_msg)
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (nd => arena%entries(n)%node)
            type is (function_def_node)
                if (.not. allocated(nd%body_indices)) cycle
                if (.not. allocated(nd%param_indices)) cycle
                call check_scope_intent_writes(arena, context, &
                                               nd%param_indices, &
                                               nd%body_indices, error_msg)
            type is (subroutine_def_node)
                if (.not. allocated(nd%body_indices)) cycle
                if (.not. allocated(nd%param_indices)) cycle
                call check_scope_intent_writes(arena, context, &
                                               nd%param_indices, &
                                               nd%body_indices, error_msg)
            end select
            if (len_trim(error_msg) > 0) return
        end do
    end subroutine check_dummy_intent_definitions

    subroutine check_scope_intent_writes(arena, context, param_indices, &
                                         body_indices, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: param_indices(:)
        integer, intent(in) :: body_indices(:)
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i

        call set_empty(error_msg)
        do i = 1, size(body_indices)
            if (.not. node_exists(arena, body_indices(i))) cycle
            select type (stmt => arena%entries(body_indices(i))%node)
            type is (declaration_node)
                cycle
            class default
                call check_statement_intent_writes(arena, context, &
                                                   param_indices, &
                                                   body_indices, &
                                                   body_indices(i), error_msg)
            end select
            if (len_trim(error_msg) > 0) return
        end do
    end subroutine check_scope_intent_writes

    recursive subroutine check_statement_intent_writes(arena, context, &
                                                       param_indices, &
                                                       body_indices, &
                                                       stmt_index, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: param_indices(:)
        integer, intent(in) :: body_indices(:)
        integer, intent(in) :: stmt_index
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i

        call set_empty(error_msg)
        if (.not. node_exists(arena, stmt_index)) return
        select type (stmt => arena%entries(stmt_index)%node)
        type is (assignment_node)
            call check_intent_write_target(arena, context, param_indices, &
                                           body_indices, stmt%target_index, &
                                           error_msg)
        type is (if_node)
            if (allocated(stmt%then_body_indices)) then
                do i = 1, size(stmt%then_body_indices)
                    call check_statement_intent_writes(arena, context, &
                        param_indices, body_indices, &
                        stmt%then_body_indices(i), error_msg)
                    if (len_trim(error_msg) > 0) return
                end do
            end if
            if (allocated(stmt%else_body_indices)) then
                do i = 1, size(stmt%else_body_indices)
                    call check_statement_intent_writes(arena, context, &
                        param_indices, body_indices, &
                        stmt%else_body_indices(i), error_msg)
                    if (len_trim(error_msg) > 0) return
                end do
            end if
        type is (do_loop_node)
            if (.not. allocated(stmt%body_indices)) return
            do i = 1, size(stmt%body_indices)
                call check_statement_intent_writes(arena, context, &
                    param_indices, body_indices, stmt%body_indices(i), &
                    error_msg)
                if (len_trim(error_msg) > 0) return
            end do
        type is (do_while_node)
            if (.not. allocated(stmt%body_indices)) return
            do i = 1, size(stmt%body_indices)
                call check_statement_intent_writes(arena, context, &
                    param_indices, body_indices, stmt%body_indices(i), &
                    error_msg)
                if (len_trim(error_msg) > 0) return
            end do
        end select
    end subroutine check_statement_intent_writes

    subroutine check_intent_write_target(arena, context, param_indices, &
                                         body_indices, target_index, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: param_indices(:)
        integer, intent(in) :: body_indices(:)
        integer, intent(in) :: target_index
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: name
        character(len=64) :: location
        integer :: decl_index

        call set_empty(error_msg)
        call target_base_name(arena, target_index, name)
        if (len_trim(name) == 0) return
        if (.not. name_is_dummy(arena, param_indices, name)) return
        call scope_decl_for_name(arena, body_indices, name, decl_index)
        if (decl_index <= 0) return
        if (.not. declaration_is_intent_in(arena, decl_index, context)) return
        write (location, '(" at line ",I0,", column ",I0)') &
            get_node_line(arena, target_index), &
            get_node_column(arena, target_index)
        error_msg = 'INTENT(IN) dummy argument '''//trim(name)// &
            ''' cannot appear in a variable definition context '// &
            '(assignment target)'//trim(location)
    end subroutine check_intent_write_target

    logical function name_is_dummy(arena, param_indices, name) result(is_dummy)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: param_indices(:)
        character(len=*), intent(in) :: name
        character(len=:), allocatable :: param, name_error
        integer :: i

        is_dummy = .false.
        do i = 1, size(param_indices)
            call parameter_name(arena, param_indices(i), param, name_error)
            if (len_trim(name_error) > 0) cycle
            if (.not. allocated(param)) cycle
            if (same_name(param, name)) then
                is_dummy = .true.
                return
            end if
        end do
    end function name_is_dummy

    ! Effective INTENT(IN): declared so, or - under the Lazy Fortran default
    ! intent policy - declared with no INTENT at all (#438).
    logical function declaration_is_intent_in(arena, decl_index, context) &
            result(is_in)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: decl_index
        type(lowering_context_t), intent(in) :: context

        is_in = .false.
        select type (decl => arena%entries(decl_index)%node)
        type is (declaration_node)
            if (decl%is_pointer) return
            if (decl%has_intent) then
                if (.not. allocated(decl%intent)) return
                is_in = trim(lowercase_text(decl%intent)) == 'in'
            else
                is_in = context%lazy_mode
            end if
        end select
    end function declaration_is_intent_in

    logical function is_keyword_actual(arena, idx, dummy_name, value_index)
        ! An actual written as dummy=expr parses as an assignment node whose
        ! target names the dummy and whose value is the expression actually
        ! passed (F2018 15.5.1). A CALL statement's argument list also flags
        ! the node is_keyword_argument, but a function reference's does not,
        ! so the shape of the node is what identifies it: an assignment can
        ! never appear as an actual argument in any other role.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: idx
        character(len=:), allocatable, intent(out) :: dummy_name
        integer, intent(out) :: value_index

        is_keyword_actual = .false.
        dummy_name = ''
        value_index = 0
        if (idx <= 0) return
        if (.not. node_exists(arena, idx)) return
        select type (nd => arena%entries(idx)%node)
        type is (assignment_node)
            if (.not. node_exists(arena, nd%target_index)) return
            select type (tgt => arena%entries(nd%target_index)%node)
            type is (identifier_node)
                if (.not. allocated(tgt%name)) return
                dummy_name = trim(tgt%name)
                value_index = nd%value_index
                is_keyword_actual = .true.
            end select
        end select
    end function is_keyword_actual

    integer function dummy_name_position(dummy_names, count, dummy_name) &
            result(pos)
        ! Slot of dummy_name in the callee's dummy list, or 0 when it names no
        ! dummy of that procedure.
        character(len=*), intent(in) :: dummy_names(:)
        integer, intent(in) :: count
        character(len=*), intent(in) :: dummy_name
        integer :: i

        pos = 0
        do i = 1, min(count, size(dummy_names))
            if (same_name(dummy_names(i), dummy_name)) then
                pos = i
                return
            end if
        end do
    end function dummy_name_position

    subroutine resolve_keyword_actuals(arena, context, actual_indices, &
                                       callee_name, self_position, mapped, &
                                       error_msg)
        ! Reorder the call's actual arguments into the callee's dummy order
        ! (#408). Positional actuals fill the leading slots; each keyword actual
        ! goes to the slot its dummy name selects. Slots no actual reaches stay
        ! 0, which the caller passes as an absent optional. A call without
        ! keyword actuals keeps its written order untouched.
        type(ast_arena_t), intent(in) :: arena
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: actual_indices(:)
        character(len=*), intent(in) :: callee_name
        integer, intent(in) :: self_position
        integer, allocatable, intent(out) :: mapped(:)
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=64), allocatable :: dummy_names(:)
        integer, allocatable :: slots(:)
        character(len=:), allocatable :: kw_name
        integer :: i, pos, name_count, value_index, last
        logical :: seen_keyword

        call set_empty(error_msg)
        mapped = actual_indices
        seen_keyword = .false.
        do i = 1, size(actual_indices)
            if (is_keyword_actual(arena, actual_indices(i), kw_name, &
                                  value_index)) seen_keyword = .true.
        end do
        if (.not. seen_keyword) return
        ! A type-bound call carries an implicit passed-object receiver, whose
        ! slot the keyword mapping below does not model.
        if (self_position > 0) return

        call callee_dummy_names(arena, context, callee_name, dummy_names, &
                                name_count)
        if (name_count < 0) then
            error_msg = 'keyword argument in call to '//trim(callee_name)// &
                ' requires a visible explicit interface'
            return
        end if
        allocate (slots(name_count))
        slots = 0
        seen_keyword = .false.
        do i = 1, size(actual_indices)
            if (is_keyword_actual(arena, actual_indices(i), kw_name, &
                                  value_index)) then
                seen_keyword = .true.
                pos = dummy_name_position(dummy_names, name_count, kw_name)
                if (pos <= 0) then
                    error_msg = 'Keyword actual '//kw_name//' in call to '// &
                        trim(callee_name)//' names no dummy argument'
                    return
                end if
                if (slots(pos) /= 0) then
                    error_msg = 'Dummy argument '//kw_name//' of '// &
                        trim(callee_name)//' is supplied more than once'
                    return
                end if
                slots(pos) = value_index
            else
                if (seen_keyword) then
                    error_msg = 'Positional actual in call to '// &
                        trim(callee_name)//' appears after a keyword argument'
                    return
                end if
                if (i > name_count) then
                    error_msg = 'More actual than formal arguments in call to '// &
                        trim(callee_name)
                    return
                end if
                slots(i) = actual_indices(i)
            end if
        end do
        last = 0
        do i = 1, name_count
            if (slots(i) /= 0) last = i
        end do
        deallocate (mapped)
        allocate (mapped(last))
        do i = 1, last
            mapped(i) = slots(i)
        end do
    end subroutine resolve_keyword_actuals

    type(lr_operand_desc_t) function absent_optional_operand(context) &
            result(operand)
        ! The reference-ABI encoding of an absent optional dummy: a null
        ! pointer, matching the trailing padding in
        ! emit_call_with_optional_padding.
        type(lowering_context_t), intent(in) :: context

        operand%kind = LR_OP_KIND_IMM_I64
        operand%payload = 0_c_int64_t
        operand%typ = lr_type_ptr_s(context%session%handle)
        operand%global_offset = 0_c_int64_t
    end function absent_optional_operand

    subroutine check_dummy_contract_actual(arena, node_index, context, &
                                           callee_name, param_pos, &
                                           handled_by_value, arg, error_msg)
        ! Apply the dummy's declared contract to one actual argument. A VALUE
        ! dummy receives a copy, so the callee's assignments to it never reach
        ! the caller's actual; handled_by_value reports that arg was filled
        ! here. An INTENT(OUT) or INTENT(INOUT) dummy requires a definable
        ! actual, so a literal is diagnosed instead of lowered. Both contracts
        ! read the same way for a procedure defined in this unit and for one
        ! imported from a .fmod artefact (#397).
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: callee_name
        integer, intent(in) :: param_pos
        logical, intent(out) :: handled_by_value
        type(lr_operand_desc_t), intent(out) :: arg
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: value
        logical :: is_optional, is_value
        integer :: intent_code, value_kind

        call set_empty(error_msg)
        handled_by_value = .false.
        if (param_pos < 1) return
        call callee_dummy_contracts(arena, context, callee_name, param_pos, &
                                    is_optional, is_value, intent_code)
        if (intent_code == ARG_INTENT_OUT .or. intent_code == ARG_INTENT_INOUT) then
            if (is_literal(arena, node_index)) then
                error_msg = 'Actual argument to '//intent_text(intent_code)// &
                    ' dummy argument of '//trim(callee_name)// &
                    ' must be a variable'
                return
            end if
        end if
        if (.not. is_value) return
        value_kind = arg_default_kind(arena, context, callee_name, param_pos, &
                                      VALUE_I32)
        call lower_expression_by_kind(arena, node_index, context, value_kind, &
                                      value, error_msg)
        if (len_trim(error_msg) > 0) return
        call make_reference_argument(context, value_kind, value, arg, error_msg)
        if (len_trim(error_msg) > 0) return
        handled_by_value = .true.
    end subroutine check_dummy_contract_actual

    function intent_text(intent_code) result(text)
        ! The INTENT spelling used in a diagnostic (#397).
        integer, intent(in) :: intent_code
        character(len=:), allocatable :: text

        if (intent_code == ARG_INTENT_OUT) then
            text = 'INTENT(OUT)'
        else
            text = 'INTENT(INOUT)'
        end if
    end function intent_text

    logical function callee_dummy_is_array_ctx(arena, context, callee_name, &
                                               param_pos) result(is_array)
        ! Whether the callee's param_pos-th dummy is an array. A procedure
        ! defined or interfaced in this unit answers from its declaration; a
        ! separately compiled module procedure answers from the rank the .fmod
        ! recorded, so an array actual takes the array-argument ABI instead of
        ! being rejected as an unsupported scalar (#415).
        type(ast_arena_t), intent(in) :: arena
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: callee_name
        integer, intent(in) :: param_pos
        integer :: idx

        is_array = callee_dummy_is_array(arena, callee_name, param_pos)
        if (is_array) return
        if (param_pos < 1) return
        idx = external_procedure_index(context, callee_name)
        if (idx <= 0) return
        if (.not. context%external_procedures(idx)%by_reference) return
        if (param_pos > context%external_procedures(idx)%arg_count) return
        is_array = context%external_procedures(idx)%arg_ranks(param_pos) > 0
    end function callee_dummy_is_array_ctx
