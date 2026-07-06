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
                    value_kind = generic_call_return_kind(context, node%name, &
                        call_arg_count, call_arg_kinds)
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
            case default
                error_msg = 'direct LIRIC session cannot copy back this argument'
                return
            end select
            context%symbols(copyback_indices(i))%value = value
        end do
        call set_empty(error_msg)
    end subroutine copy_back_reference_args
