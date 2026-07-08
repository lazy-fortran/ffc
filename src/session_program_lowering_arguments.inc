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

    subroutine prepare_reference_args(arena, arg_indices, context, default_kind, &
                                      callee_name, args, copyback_indices, &
                                      error_msg, self_position)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: arg_indices(:)
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

        sp = 0
        if (present(self_position)) sp = self_position
        visible_count = size(arg_indices)
        hidden_count = callee_hidden_extent_count(context, callee_name)

        ! More actual arguments than the callee (defined or interfaced in this
        ! unit) declares is always invalid, whatever the argument keywords or
        ! optionality. A missing trailing dummy is left to the optional-padding
        ! path, which cannot tell optional from required, so only the excess is
        ! diagnosed here. Type-bound calls (sp > 0) carry a hidden receiver and
        ! are skipped to keep the count comparison exact.
        if (sp <= 0) then
            param_count = arena_proc_param_count(arena, callee_name)
            if (param_count >= 0 .and. visible_count > param_count) then
                error_msg = 'More actual than formal arguments in call to '// &
                    trim(callee_name)
                return
            end if
        end if

        allocate (args(visible_count + hidden_count))
        allocate (copyback_indices(visible_count + hidden_count))
        copyback_indices = 0

        do i = 1, size(arg_indices)
            ! A scalar actual (literal or scalar variable, not an array element)
            ! passed where the callee declares an array dummy is a rank mismatch
            ! (F2018 15.5.2.4). callee_dummy_is_array only resolves against a
            ! procedure defined or interfaced in this unit, so external calls
            ! without an explicit interface never trip this. An assumed-rank
            ! arr(..) dummy binds to any rank, scalar included, so it is excluded.
            if (callee_dummy_is_array(arena, callee_name, &
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
            else if (actual_is_whole_array(arena, arg_indices(i), context) .and. &
                     callee_dummy_is_array(arena, callee_name, i)) then
                ! Explicit-shape array actual to an array dummy: pass the
                ! contiguous base address; the callee binds it as its dummy view.
                call whole_array_actual_address(arena, arg_indices(i), context, &
                                                 args(i), error_msg)
                if (len_trim(error_msg) > 0) return
            else if (actual_is_allocatable_array(arena, arg_indices(i), context) &
                     .and. callee_dummy_is_allocatable_array(arena, callee_name, &
                                                             i)) then
                ! Allocatable array actual to an allocatable array dummy: pass
                ! the caller's own descriptor address so the callee aliases it
                ! (allocate/deallocate inside the callee are visible to the
                ! caller too, W11).
                call allocatable_array_actual_descriptor_address(arena, &
                    arg_indices(i), context, args(i), error_msg)
                if (len_trim(error_msg) > 0) return
            else if (actual_is_allocatable_array(arena, arg_indices(i), context) &
                     .and. callee_dummy_assumed_shape_runtime(arena, callee_name, &
                                            call_arg_param_pos(i, sp))) then
                ! Allocatable array actual to a rank-1/rank-2 assumed-shape dummy:
                ! pass the descriptor's runtime data pointer (the W2 runtime-extent
                ! ABI); its per-dimension extents travel separately as hidden i64
                ! arguments appended below.
                call allocatable_array_actual_data_address(arena, arg_indices(i), &
                    context, args(i), error_msg)
                if (len_trim(error_msg) > 0) return
            else if (actual_is_contiguous_section(arena, arg_indices(i), context) &
                     .and. callee_dummy_is_array(arena, callee_name, &
                                                 call_arg_param_pos(i, sp))) then
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
                                                callee_name, i, default_kind), &
                                           args(i), copyback_indices(i), error_msg)
                if (len_trim(error_msg) > 0) return
            end if
        end do

        if (hidden_count > 0) then
            call append_hidden_extent_args(arena, arg_indices, context, &
                                           callee_name, visible_count, args, &
                                           error_msg, self_position=sp)
            if (len_trim(error_msg) > 0) return
        end if

        call set_empty(error_msg)
    end subroutine prepare_reference_args

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
            symbol_index = find_symbol(context, id_name)
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
            call call_actual_available_element_count(arena, node, context, &
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
        call describe_array_section(arena, node, context, info, error_msg)
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

    subroutine call_actual_available_element_count(arena, node, context, &
                                                   element_count, known, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
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
        call array_element_available_count(arena, node, context, element_count, &
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

        element_count = 0_c_int64_t
        known = .false.
        call resolve_reshape_args(arena, node, source_index, shape_index, error_msg)
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

    subroutine array_element_available_count(arena, node, context, element_count, &
                                             known, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
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
        symbol_index = find_symbol(context, node%name)
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
        select type (n => arena%entries(node_index)%node)
        type is (component_access_node)
            actual_is_character = derived_component_access_kind(arena, n, &
                context) == VALUE_CHARACTER
            return
        end select
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
        symbol_index = find_symbol(context, id_name)
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
        symbol_index = find_symbol(context, id_name)
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
        symbol_index = find_symbol(context, id_name)
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
            call describe_array_section(arena, slice, context, info, err)
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
            call describe_array_section(arena, slice, context, info, error_msg)
            if (len_trim(error_msg) > 0) return
            src = info%source_index
            first_linear = array_section_source_linear_index(info, 0_c_int64_t)
            vk = context%symbols(src)%value_kind
            arr_size = context%symbols(src)%array_size
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
        symbol_index = find_symbol(context, id_name)
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
        symbol_index = find_symbol(context, id_name)
        if (symbol_index <= 0) then
            error_msg = 'allocatable array argument was not declared: '// &
                        trim(id_name)
            return
        end if
        address = context%symbols(symbol_index)%allocatable_descriptor_address
        call set_empty(error_msg)
    end subroutine allocatable_array_actual_descriptor_address

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
        dummy_kind = callee_dummy_value_kind(arena, callee_name, param_pos)
        if (dummy_kind == 0) then
            dummy_kind = external_dummy_value_kind(context, callee_name, &
                                                   param_pos)
        end if
        if (dummy_kind == VALUE_I32 .or. dummy_kind == VALUE_F32 .or. &
            dummy_kind == VALUE_F64 .or. dummy_kind == VALUE_LOGICAL .or. &
            dummy_kind == VALUE_I64 .or. dummy_kind == VALUE_I8 .or. &
            dummy_kind == VALUE_I16) then
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
        integer :: call_arg_count
        integer :: call_arg_kinds(MAX_PROC_ARGS)
        integer :: call_arg_ranks(MAX_PROC_ARGS)
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
            symbol_index = find_symbol(context, id_name)
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
                end if
            end if
        type is (component_access_node)
            value_kind = derived_component_access_kind(arena, node, context)
        end select
    end function expression_value_kind

 subroutine call_argument_kinds(arena, node, context, default_kind, &
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
            arg_kinds(i) = expression_value_kind(arena, node%arg_indices(i), &
                                                 context, default_kind)
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
            symbol_index = find_symbol(context, id_name)
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
                rank = expression_array_access_rank(arena, node, context)
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

    integer function expression_array_access_rank(arena, node, context) result(rank)
        ! Rank of an array access expression: 0 for element access (all indices
        ! are scalar), or the remaining rank for a section (e.g. a(:) keeps rank
        ! 1 of a rank-1 array).
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        integer :: symbol_index, base_rank

        rank = 0
        if (.not. allocated(node%name)) return
        if (len_trim(node%name) == 0) return
        symbol_index = find_symbol(context, node%name)
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
