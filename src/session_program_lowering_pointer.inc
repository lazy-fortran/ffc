    subroutine lower_scalar_pointer_target(node, context, value_kind, error_msg)
        ! Scalar integer/real/logical POINTER/TARGET (#245 slice B3a). A target
        ! is given a stable in-memory slot so a pointer can address it; a pointer
        ! starts unassociated.
        type(declaration_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: value_kind
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i, character_length

        character_length = 0
        if (value_kind == VALUE_CHARACTER) then
            if (.not. allocated(node%type_name)) then
                error_msg = 'character pointer/target declaration has no type'
                return
            end if
            call parse_character_length(node%type_name, character_length, error_msg)
            if (len_trim(error_msg) > 0) return
            if (character_length < 0 .and. node%is_pointer) then
                ! character(len=:), pointer: a deferred-length character view.
                ! It carries the same {data,len} descriptor as a deferred-length
                ! allocatable, but the storage is borrowed, so it is never
                ! freed through this symbol (#361).
                call define_deferred_character_pointer(node, context, error_msg)
                return
            end if
            if (character_length <= 0) then
                call unsupported_feature_error('character pointer/target declaration', &
                    node%line, node%column, &
                    'direct LIRIC session supports fixed-length character '// &
                    'pointer/target declarations only (#452)', error_msg)
                return
            end if
        end if

        if (node%is_multi_declaration .and. allocated(node%var_names)) then
            do i = 1, size(node%var_names)
                call define_scalar_pointer_target_symbol(context, node, &
                    node%var_names(i), value_kind, character_length, error_msg)
                if (len_trim(error_msg) > 0) return
            end do
        else if (allocated(node%var_name)) then
            call define_scalar_pointer_target_symbol(context, node, node%var_name, &
                value_kind, character_length, error_msg)
        else
            error_msg = 'pointer/target declaration did not expose a variable name'
        end if
    end subroutine lower_scalar_pointer_target

    subroutine define_deferred_character_pointer(node, context, error_msg)
        ! Define every name in a character(len=:), pointer declaration as a
        ! deferred-length character symbol. Association is done later by
        ! c_f_strpointer (#361).
        type(declaration_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i

        if (node%is_multi_declaration .and. allocated(node%var_names)) then
            do i = 1, size(node%var_names)
                call define_deferred_character_symbol(context, node%var_names(i), &
                                                      error_msg)
                if (len_trim(error_msg) > 0) return
            end do
        else if (allocated(node%var_name)) then
            call define_deferred_character_symbol(context, node%var_name, error_msg)
        else
            error_msg = 'pointer declaration did not expose a variable name'
        end if
    end subroutine define_deferred_character_pointer

    subroutine scalar_character_cstring_symbol(arena, node_index, context, &
                                               symbol_index)
        ! Index of the symbol named by a C_F_STRPOINTER CSTRING actual when that
        ! actual is a scalar fixed-length character variable whose bytes are
        ! already materialised; zero otherwise (#361).
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        integer, intent(out) :: symbol_index
        character(len=:), allocatable :: name, name_error
        integer :: index

        symbol_index = 0
        if (.not. is_identifier(arena, node_index)) return
        call get_identifier_name(arena, node_index, name, name_error)
        if (len_trim(name_error) > 0) return
        index = find_symbol_compat(context, name)
        if (index <= 0) return
        if (context%symbols(index)%value_kind /= VALUE_CHARACTER) return
        if (context%symbols(index)%is_array) return
        if (.not. context%symbols(index)%has_character_value) return
        if (context%symbols(index)%character_length <= 0) return
        symbol_index = index
    end subroutine scalar_character_cstring_symbol

    subroutine lower_c_f_strpointer(arena, arg_indices, context, error_msg)
        ! c_f_strpointer(cstring, fstrptr [, nchars]): associate a
        ! deferred-length character pointer with borrowed C character storage.
        ! The Fortran length is the number of characters before the first C
        ! null byte, capped at nchars. The C target is never freed (#361).
        type(ast_arena_t), intent(in) :: arena
        integer, allocatable, intent(in) :: arg_indices(:)
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: data_ptr, max_chars, length
        character(len=:), allocatable :: name
        integer :: symbol_index, cstring_index

        call set_empty(error_msg)
        if (.not. allocated(arg_indices)) then
            error_msg = 'c_f_strpointer requires at least two arguments'
            return
        end if
        if (size(arg_indices) < 2 .or. size(arg_indices) > 3) then
            error_msg = 'c_f_strpointer requires two or three arguments'
            return
        end if

        ! A scalar character cstring carries its own declared length, so NCHARS
        ! is optional there; a c_ptr cstring has no length metadata at all.
        call scalar_character_cstring_symbol(arena, arg_indices(1), context, &
                                             cstring_index)
        if (cstring_index <= 0 .and. size(arg_indices) < 3) then
            error_msg = 'c_f_strpointer with a c_ptr cstring requires the '// &
                        'nchars argument'
            return
        end if

        if (cstring_index > 0) then
            data_ptr = context%symbols(cstring_index)%value
        else
            call lower_c_ptr_expression(arena, arg_indices(1), context, data_ptr, &
                                        error_msg)
            if (len_trim(error_msg) > 0) then
                error_msg = 'c_f_strpointer cstring must be a c_ptr expression '// &
                            'or a scalar character variable; character array '// &
                            'cstrings are not yet supported'
                return
            end if
        end if

        if (.not. is_identifier(arena, arg_indices(2))) then
            error_msg = 'c_f_strpointer fstrptr must be a character pointer'
            return
        end if
        call get_identifier_name(arena, arg_indices(2), name, error_msg)
        if (len_trim(error_msg) > 0) return
        symbol_index = find_symbol_compat(context, name)
        if (symbol_index <= 0) then
            error_msg = 'c_f_strpointer fstrptr was not declared: '//trim(name)
            return
        end if
        if (.not. context%symbols(symbol_index)%is_deferred_character) then
            error_msg = 'c_f_strpointer fstrptr must be a deferred-length '// &
                        'character pointer: '//trim(name)
            return
        end if

        if (size(arg_indices) >= 3) then
            call lower_i64_expression(arena, arg_indices(3), context, max_chars, &
                                      error_msg)
            if (len_trim(error_msg) > 0) return
        else
            max_chars = i64_immediate(context%session, &
                int(context%symbols(cstring_index)%character_length, c_int64_t))
        end if

        if (.not. emit_strnlen(context%session, data_ptr, max_chars, length, &
                               error_msg)) return
        if (.not. emit_ptr_store(context%session, data_ptr, &
                context%symbols(symbol_index)%deferred_data, error_msg)) return
        if (.not. emit_i64_store(context%session, length, &
                context%symbols(symbol_index)%deferred_length, error_msg)) return
        ! Uses of the view read the borrowed data pointer directly, exactly as
        ! a deferred-length assignment binds its data operand.
        context%symbols(symbol_index)%value = data_ptr
        context%symbols(symbol_index)%has_character_value = .true.
        call set_empty(error_msg)
    end subroutine lower_c_f_strpointer

    subroutine lower_pointer_target_array(node, context, value_kind, error_msg)
        ! Rank-1/rank-2 fixed-size integer/real/logical/complex POINTER/TARGET
        ! arrays (#245 B3a). A target array gets ordinary explicit-shape
        ! storage; a deferred-shape pointer array (p(:) or p(:,:)) starts
        ! unassociated and adopts a target array's base, shape, and element
        ! kind at `p => t`. Anything outside this subset (rank>2, allocatable,
        ! derived, character, bounds-remapped pointers) keeps the prior
        ! unsupported diagnostic.
        type(declaration_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: value_kind
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: dim_count, array_lower_bound, array_size, i

        if (node%is_allocatable) then
            call unsupported_pointer_target_array(node, error_msg)
            return
        end if
        dim_count = 0
        if (allocated(node%dimension_indices)) &
            dim_count = size(node%dimension_indices)
        if (dim_count < 1 .or. dim_count > 2) then
            call unsupported_pointer_target_array(node, error_msg)
            return
        end if
        if (value_kind /= VALUE_I32 .and. value_kind /= VALUE_F32 .and. &
            value_kind /= VALUE_F64 .and. value_kind /= VALUE_LOGICAL .and. &
            value_kind /= VALUE_C4 .and. value_kind /= VALUE_C8) then
            call unsupported_pointer_target_array(node, error_msg)
            return
        end if

        ! A pointer array is deferred-shape p(:) [,(:)]: no compile-time
        ! storage, just a stub symbol that `p => t` fills in. A target array
        ! is explicit-shape and is laid out exactly like an ordinary
        ! fixed-size array.
        if (node%is_pointer .and. declaration_is_assumed_shape(node, context)) then
            if (node%is_multi_declaration .and. allocated(node%var_names)) then
                do i = 1, size(node%var_names)
                    call define_pointer_array_stub(context, node%var_names(i), &
                                                   value_kind, dim_count, error_msg)
                    if (len_trim(error_msg) > 0) return
                end do
            else if (allocated(node%var_name)) then
                call define_pointer_array_stub(context, node%var_name, &
                                               value_kind, dim_count, error_msg)
            else
                error_msg = 'pointer array declaration did not expose a name'
            end if
            return
        end if

        ! Explicit-shape target array (or an explicit-shape pointer, which we
        ! treat like a target with storage). A deferred-shape target is invalid
        ! Fortran, so the only remaining shape here is explicit rank-1.
        if (declaration_is_assumed_shape(node, context)) then
            call unsupported_pointer_target_array(node, error_msg)
            return
        end if
        call get_array_bounds(node, context, array_lower_bound, array_size, &
                              error_msg)
        if (len_trim(error_msg) > 0) return
        if (node%is_multi_declaration .and. allocated(node%var_names)) then
            do i = 1, size(node%var_names)
                call define_target_array_symbol(context, node, node%var_names(i), &
                                                array_lower_bound, array_size, &
                                                value_kind, error_msg)
                if (len_trim(error_msg) > 0) return
            end do
        else if (allocated(node%var_name)) then
            call define_target_array_symbol(context, node, node%var_name, &
                                            array_lower_bound, array_size, &
                                            value_kind, error_msg)
        else
            error_msg = 'target array declaration did not expose a name'
        end if
    end subroutine lower_pointer_target_array

    subroutine unsupported_pointer_target_array(node, error_msg)
        type(declaration_node), intent(in) :: node
        character(len=:), allocatable, intent(out) :: error_msg

        call unsupported_feature_error('pointer/target array declaration', &
            node%line, node%column, &
            'direct LIRIC session supports rank-1 and rank-2 fixed-size '// &
            'integer, real, logical, and complex pointer/target arrays '// &
            'only (#245 slice B3a)', error_msg)
    end subroutine unsupported_pointer_target_array

    subroutine define_target_array_symbol(context, node, name, array_lower_bound, &
                                          array_size, value_kind, error_msg)
        ! Lay out a fixed-size target array exactly like an ordinary array, then
        ! mark it as an addressable target so a pointer can alias its storage.
        type(lowering_context_t), intent(inout) :: context
        type(declaration_node), intent(in) :: node
        character(len=*), intent(in) :: name
        integer, intent(in) :: array_lower_bound
        integer, intent(in) :: array_size
        integer, intent(in) :: value_kind
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: index

        call define_declared_array_symbol(context, node, name, array_lower_bound, &
                                          array_size, value_kind, error_msg)
        if (len_trim(error_msg) > 0) return
        index = find_symbol_compat(context, name)
        if (index <= 0) then
            error_msg = 'target array declaration lost its symbol: '//trim(name)
            return
        end if
        if (node%is_target) context%symbols(index)%is_target = .true.
        if (node%is_pointer) then
            context%symbols(index)%is_pointer = .true.
            context%symbols(index)%is_associated = .true.
        end if
        ! element_address is the array base; expose it as the alias address so
        ! the shared `p => t` path can adopt it. A complex array already keeps
        ! its imaginary-part array base in `address` (define_declared_array_
        ! symbol), so leave that field alone there instead of overwriting it.
        if (value_kind /= VALUE_C4 .and. value_kind /= VALUE_C8) then
            context%symbols(index)%address = context%symbols(index)%element_address
        end if
        context%symbols(index)%has_address = .true.
    end subroutine define_target_array_symbol

    subroutine define_pointer_array_stub(context, name, value_kind, array_rank, &
                                         error_msg)
        ! A deferred-shape pointer array p(:) [,(:)]: no storage and no shape
        ! until a `p => t` association supplies the target's base and extent.
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: name
        integer, intent(in) :: value_kind
        integer, intent(in) :: array_rank
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: index

        index = find_symbol_compat(context, name)
        if (index > 0) then
            ! Re-declaration (e.g. dummy refresh): just record the attributes.
            context%symbols(index)%value_kind = value_kind
            context%symbols(index)%is_pointer = .true.
            context%symbols(index)%is_array = .true.
            context%symbols(index)%array_rank = array_rank
            context%symbols(index)%array_size = 0
            context%symbols(index)%is_associated = .false.
            context%symbols(index)%array_lower_bound = 1
            context%symbols(index)%array_dim_lowers = 1
            ! A pointer dummy is passed as the associated target's base address
            ! through the canonical descriptor ABI. Preserve the bound base and
            ! descriptor-derived stride instead of leaving element_address unset.
            if (context%symbols(index)%is_parameter) then
                context%symbols(index)%element_address = &
                    context%symbols(index)%address
                context%symbols(index)%has_address = .true.
                context%symbols(index)%is_reference = .true.
                context%symbols(index)%is_associated = .true.
            end if
            call set_empty(error_msg)
            return
        end if
        call grow_symbols(context)
        index = context%symbol_count + 1
        context%symbols(index)%name = trim(name)
        context%symbols(index)%value_kind = value_kind
        context%symbols(index)%is_array = .true.
        context%symbols(index)%array_rank = array_rank
        context%symbols(index)%array_size = 0
        context%symbols(index)%is_pointer = .true.
        context%symbols(index)%is_associated = .false.
        context%symbols(index)%has_address = .false.
        context%symbols(index)%is_reference = .false.
        context%symbols(index)%array_lower_bound = 1
        context%symbols(index)%array_dim_lowers = 1
        context%symbol_count = index
        call set_empty(error_msg)
    end subroutine define_pointer_array_stub

    subroutine define_scalar_pointer_target_symbol(context, node, name, value_kind, &
                                                   character_length, error_msg)
        type(lowering_context_t), intent(inout) :: context
        type(declaration_node), intent(in) :: node
        character(len=*), intent(in) :: name
        integer, intent(in) :: value_kind
        integer, intent(in) :: character_length
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: symbol_index

        symbol_index = find_symbol_compat(context, name)
        if (symbol_index > 0) then
            ! A dummy argument already has its own pass-by-reference storage, so a
            ! `target` attribute on it needs no new slot; just record the flag.
            ! A genuine local pointer/target re-declaration is benign here too.
            if (node%is_target) context%symbols(symbol_index)%is_target = .true.
            if (node%is_pointer) then
                context%symbols(symbol_index)%is_pointer = .true.
                context%symbols(symbol_index)%is_associated = .false.
            end if
            call set_empty(error_msg)
            return
        end if
        if (value_kind == VALUE_CHARACTER) then
            call define_character_symbol(context, name, character_length, error_msg)
        else
            call define_symbol(context, name, value_kind, error_msg)
        end if
        if (len_trim(error_msg) > 0) return
        symbol_index = find_symbol_compat(context, name)
        if (symbol_index <= 0) then
            error_msg = 'pointer/target declaration lost its symbol: '//trim(name)
            return
        end if
        if (node%is_target) then
            ! Materialise the target in memory so a pointer can hold its address.
            context%symbols(symbol_index)%is_target = .true.
            call emit_pointer_target_slot(context, symbol_index, value_kind, error_msg)
            if (len_trim(error_msg) > 0) return
            context%symbols(symbol_index)%has_address = .true.
            context%symbols(symbol_index)%is_reference = .true.
            if (node%has_initializer .and. node%initializer_index > 0) then
                if (value_kind == VALUE_CHARACTER) then
                    call lower_character_initializer(context, name, &
                        node%initializer_index, error_msg)
                else
                    call lower_scalar_initializer(context, name, value_kind, &
                        node%initializer_index, error_msg)
                end if
                if (len_trim(error_msg) > 0) return
            end if
        else
            ! A pointer starts unassociated; `p => t` fills in the slot address.
            context%symbols(symbol_index)%is_pointer = .true.
            context%symbols(symbol_index)%is_associated = .false.
            context%symbols(symbol_index)%has_address = .false.
            context%symbols(symbol_index)%is_reference = .false.
        end if
        call set_empty(error_msg)
    end subroutine define_scalar_pointer_target_symbol

    subroutine emit_pointer_target_slot(context, symbol_index, value_kind, error_msg)
        ! Allocate a typed slot for a target scalar and store its initial value,
        ! so a pointer can later adopt the slot address (deref via the typed
        ! is_reference load/store paths).
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        integer, intent(in) :: value_kind
        character(len=:), allocatable, intent(out) :: error_msg

        select case (value_kind)
        case (VALUE_I32, VALUE_LOGICAL)
            if (.not. emit_i32_alloca(context%session, &
                context%symbols(symbol_index)%address, error_msg)) return
            if (.not. emit_i32_store(context%session, &
                context%symbols(symbol_index)%value, &
                context%symbols(symbol_index)%address, error_msg)) return
        case (VALUE_F32)
            if (.not. emit_liric_f32_alloca(context%session, &
                context%symbols(symbol_index)%address, error_msg)) return
            if (.not. emit_liric_f32_store(context%session, &
                context%symbols(symbol_index)%value, &
                context%symbols(symbol_index)%address, error_msg)) return
        case (VALUE_F64)
            if (.not. emit_liric_f64_alloca(context%session, &
                context%symbols(symbol_index)%address, error_msg)) return
            if (.not. emit_liric_f64_store(context%session, &
                context%symbols(symbol_index)%value, &
                context%symbols(symbol_index)%address, error_msg)) return
        case (VALUE_CHARACTER)
            if (context%symbols(symbol_index)%character_length <= 0) then
                error_msg = 'character target has no fixed storage length'
                return
            end if
            if (.not. emit_alloca_bytes(context%session, &
                i64_immediate(context%session, int( &
                    context%symbols(symbol_index)%character_length + 1, &
                    c_int64_t)), context%symbols(symbol_index)%address, &
                error_msg)) return
            call fill_spaces(context, context%symbols(symbol_index)%address, &
                i32_immediate(context%session, int( &
                    context%symbols(symbol_index)%character_length, c_int64_t)), &
                error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_liric_store_char_byte(context%session, &
                context%symbols(symbol_index)%address, &
                i32_immediate(context%session, int( &
                    context%symbols(symbol_index)%character_length, c_int64_t)), &
                i32_immediate(context%session, 0_c_int64_t), error_msg)) return
            context%symbols(symbol_index)%value = &
                context%symbols(symbol_index)%address
            context%symbols(symbol_index)%has_character_value = .true.
        case default
            error_msg = 'unsupported target scalar kind for direct LIRIC session'
            return
        end select
        call set_empty(error_msg)
    end subroutine emit_pointer_target_slot

    subroutine lower_pointer_assignment(arena, node, context, error_msg)
        ! p => t : the pointer adopts the target's storage address.
        ! Dispatches to lower_proc_pointer_assignment when p is a procedure pointer.
        type(ast_arena_t), intent(in) :: arena
        type(pointer_assignment_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: ptr_name, target_name
        integer :: ptr_index, target_index
        logical :: declared_data_pointer

        ! obj%p => proc binds a procedure-pointer component (#372).
        if (is_proc_pointer_component_access(arena, node%pointer_index, context)) then
            call lower_proc_component_assignment(arena, node, context, error_msg)
            return
        end if
        ! obj%p => t binds a scalar data-pointer component (#401).
        if (is_data_pointer_component_access(arena, node%pointer_index, context)) &
        then
            call lower_data_component_assignment(arena, node, context, error_msg)
            return
        end if
        call get_identifier_name(arena, node%pointer_index, ptr_name, error_msg)
        if (len_trim(error_msg) > 0) return
        ptr_index = find_symbol_compat(context, ptr_name)
        if (ptr_index > 0 .and. context%symbols(ptr_index)%is_proc_pointer) then
            call lower_proc_pointer_assignment(arena, node, context, error_msg)
            return
        end if
        if (ptr_index <= 0) then
            error_msg = 'pointer assignment target was not declared: '//trim(ptr_name)
            return
        end if
        ! An entity FortFront resolved to a declaration; a lazy-Fortran name
        ! that no declaration introduced carries no binding.
        declared_data_pointer = context%symbols(ptr_index)%has_binding
        if (.not. context%symbols(ptr_index)%is_pointer) then
            if (promote_inferred_data_pointer(context, ptr_index)) then
                ! FortFront currently keeps a bare POINTER statement out of the
                ! public AST. Its inferred symbol is binding-less and has no
                ! storage; promote that compatibility symbol at its first =>
                ! association instead of treating it as an ordinary scalar.
            else
                error_msg = 'left of => is not a pointer: '//trim(ptr_name)
                return
            end if
        end if
        ! p => null(): the pointer becomes disassociated and drops its storage.
        ! Only a declared pointer qualifies: an inferred lazy-Fortran name that
        ! no declaration introduced keeps its undeclared-pointer diagnostic.
        if (pointer_target_is_null(arena, node%target_index) .and. &
            declared_data_pointer) then
            context%symbols(ptr_index)%is_associated = .false.
            context%symbols(ptr_index)%has_runtime_association = .false.
            context%symbols(ptr_index)%has_address = .false.
            context%symbols(ptr_index)%is_reference = .false.
            call set_empty(error_msg)
            return
        end if
        if (context%symbols(ptr_index)%is_class_pointer) then
            error_msg = 'scalar class pointer reassociation is not supported'
            return
        end if
        ! p => f(...): a data-pointer function result hands back the target
        ! address itself, so the pointer adopts the returned address (#407).
        if (data_pointer_result_call(arena, node%target_index, context)) then
            call bind_data_pointer_result(arena, node%target_index, context, &
                                          ptr_index, error_msg)
            return
        end if
        if (component_access_is_scalar_derived_pointer(arena, &
                node%target_index, context)) then
            call alias_pointer_to_derived_component(arena, node%target_index, &
                context, ptr_index, error_msg)
            return
        end if
        if (component_access_is_derived_array(arena, node%target_index, context)) then
            call alias_pointer_to_derived_array_component(arena, node%target_index, &
                context, ptr_index, error_msg)
            return
        end if
        ! p => a(i): a scalar pointer may bind to an element of an intrinsic
        ! TARGET array.  FortFront represents the element as a call-shaped
        ! node, distinct from the array-section node handled below.
        if (is_declared_scalar_array_element_target(arena, node%target_index, &
                                                    context)) then
            call alias_pointer_to_scalar_array_element(arena, node%target_index, &
                                                       context, ptr_index, &
                                                       error_msg)
            return
        end if
        ! p => a(lo:hi:s): a strided array-section view bound through the
        ! pointer's descriptor byte-stride path (#338).
        if (.not. is_identifier(arena, node%target_index)) then
            select type (target_entry => arena%entries(node%target_index)%node)
            type is (array_slice_node)
                call alias_pointer_array_section(arena, node%target_index, context, &
                                                 ptr_index, error_msg)
                return
            class default
                error_msg = 'pointer assignment target AST node is not an '// &
                            'array slice node (#338)'
                return
            end select
        end if
        call get_identifier_name(arena, node%target_index, target_name, error_msg)
        if (len_trim(error_msg) > 0) return
        target_index = find_symbol_compat(context, target_name)
        if (target_index <= 0) then
            error_msg = 'pointer assignment source was not declared: '// &
                        trim(target_name)
            return
        end if
        if (.not. (context%symbols(target_index)%is_target .or. &
                   context%symbols(target_index)%is_pointer)) then
            error_msg = 'right of => is not a target or pointer: '// &
                        trim(target_name)
            return
        end if
        if (.not. context%symbols(target_index)%has_address) then
            error_msg = 'pointer assignment source has no storage: '// &
                        trim(target_name)
            return
        end if
        ! A data-pointer result must not carry back the address of storage that
        ! dies with the invocation: only a dummy's storage or static storage
        ! survives the return (#407).
        if (context%in_internal_function .and. &
            ptr_index == context%current_function_result_index) then
            if (.not. target_outlives_procedure(context%symbols(target_index))) then
                error_msg = 'pointer result target does not outlive the '// &
                            'function: '//trim(target_name)
                return
            end if
        end if
        ! Whole-array p => t: the pointer adopts the target array's base,
        ! shape, and element kind. The bounds-remapped form p(lo:hi) => t is
        ! out of scope and carries pointer bounds on the LHS, which we reject.
        if (context%symbols(ptr_index)%is_array .or. &
            context%symbols(target_index)%is_array) then
            if (.not. is_identifier(arena, node%pointer_index)) then
                error_msg = 'direct LIRIC session pointer assignment does not '// &
                            'support pointer bounds remapping'
                return
            end if
            call alias_pointer_array(context, ptr_index, target_index, error_msg)
            return
        end if
        if (context%symbols(ptr_index)%value_kind /= &
            context%symbols(target_index)%value_kind) then
            error_msg = 'pointer and target scalar kinds do not match'
            return
        end if
        if (context%symbols(ptr_index)%is_derived .neqv. &
            context%symbols(target_index)%is_derived) then
            error_msg = 'pointer and target scalar kinds do not match'
            return
        end if
        if (context%symbols(ptr_index)%is_derived .and. &
            context%symbols(ptr_index)%derived_type_index /= &
            context%symbols(target_index)%derived_type_index) then
            ! A class(parent) pointer may associate with a target declared as
            ! an extension, while a type(parent) pointer keeps the same
            ! compatibility relation for this storage-only ABI. Reject only
            ! unrelated derived types; exact identity remains the common path.
            if (.not. derived_type_extends(context, &
                    context%symbols(target_index)%derived_type_index, &
                    context%symbols(ptr_index)%derived_type_index)) then
                error_msg = 'pointer and target derived types do not match'
                return
            end if
        end if
        if (context%symbols(ptr_index)%value_kind == VALUE_CHARACTER .and. &
            context%symbols(ptr_index)%character_length /= &
            context%symbols(target_index)%character_length) then
            error_msg = 'pointer and target character lengths do not match'
            return
        end if
        if (context%symbols(ptr_index)%is_parameter .and. &
            context%symbols(ptr_index)%is_pointer) then
            if (.not. emit_ptr_store(context%session, &
                    context%symbols(target_index)%element_address, &
                    context%symbols(ptr_index)%address, error_msg)) return
        end if
        context%symbols(ptr_index)%address = context%symbols(target_index)%address
        if (context%symbols(ptr_index)%is_derived) then
            context%symbols(ptr_index)%element_address = &
                context%symbols(target_index)%element_address
            context%symbols(ptr_index)%array_size = &
                context%symbols(target_index)%array_size
        else if (context%symbols(ptr_index)%value_kind == VALUE_CHARACTER) then
            context%symbols(ptr_index)%value = &
                context%symbols(target_index)%value
        end if
        context%symbols(ptr_index)%has_address = .true.
        context%symbols(ptr_index)%is_reference = .true.
        context%symbols(ptr_index)%is_associated = .true.
        context%symbols(ptr_index)%has_runtime_association = .false.
        call set_empty(error_msg)
    end subroutine lower_pointer_assignment

    logical function is_declared_scalar_array_element_target(arena, target_index, &
                                                              context) result(is_target)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: target_index
        type(lowering_context_t), intent(in) :: context

        is_target = .false.
        if (.not. node_exists(arena, target_index)) return
        select type (target_node => arena%entries(target_index)%node)
        type is (call_or_subscript_node)
            if (.not. is_declared_array_element_ref(target_node, context)) return
            is_target = .true.
        end select
    end function is_declared_scalar_array_element_target

    subroutine alias_pointer_to_scalar_array_element(arena, target_index, context, &
                                                      ptr_index, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: target_index, ptr_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: element_address
        integer :: target_symbol, value_kind

        select type (target_node => arena%entries(target_index)%node)
        type is (call_or_subscript_node)
            target_symbol = find_symbol_compat(context, target_node%name)
            if (target_symbol <= 0) then
                error_msg = 'pointer assignment source was not declared: '// &
                            trim(target_node%name)
                return
            end if
            if (.not. (context%symbols(target_symbol)%is_target .or. &
                       context%symbols(target_symbol)%is_pointer)) then
                error_msg = 'right of => is not a target or pointer: '// &
                            trim(target_node%name)
                return
            end if
            if (.not. context%symbols(target_symbol)%has_address) then
                error_msg = 'pointer assignment source has no storage: '// &
                            trim(target_node%name)
                return
            end if
            if (context%symbols(ptr_index)%is_array) then
                error_msg = 'left of => is an array pointer but the target is '// &
                            'a scalar array element'
                return
            end if
            value_kind = context%symbols(target_symbol)%value_kind
            if (context%symbols(ptr_index)%value_kind /= value_kind) then
                error_msg = 'pointer and target scalar kinds do not match'
                return
            end if
            select case (value_kind)
            case (VALUE_I32)
                call lower_i32_array_element_address(arena, target_node, context, &
                                                     element_address, error_msg, &
                                                     'pointer assignment target')
            case (VALUE_F32, VALUE_F64)
                call lower_array_element_address_for_kind(arena, target_node, &
                    context, value_kind, element_address, error_msg, &
                    'pointer assignment target')
            case default
                error_msg = 'pointer assignment target array element kind is not '// &
                            'supported'
            end select
            if (len_trim(error_msg) > 0) return
            context%symbols(ptr_index)%address = element_address
            context%symbols(ptr_index)%has_address = .true.
            context%symbols(ptr_index)%is_reference = .true.
            context%symbols(ptr_index)%is_associated = .true.
            context%symbols(ptr_index)%has_runtime_association = .false.
            call set_empty(error_msg)
        class default
            error_msg = 'pointer assignment target AST node is not a scalar '// &
                        'array element'
        end select
    end subroutine alias_pointer_to_scalar_array_element

    logical function component_access_is_scalar_derived_pointer(arena, access_index, &
                                                                context) result(is_ptr)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: access_index
        type(lowering_context_t), intent(in) :: context
        integer :: symbol_index, type_index, component_index
        character(len=:), allocatable :: name, error_msg

        is_ptr = .false.
        call resolve_component_ref(arena, access_index, context, symbol_index, &
                                   type_index, component_index, name, error_msg)
        if (len_trim(error_msg) > 0 .or. symbol_index <= 0) return
        is_ptr = component_is_pointer(context, type_index, component_index) .and. &
                 component_kind(context, type_index, component_index) == VALUE_DERIVED
    end function component_access_is_scalar_derived_pointer

    subroutine alias_pointer_to_derived_component(arena, access_index, context, &
                                                   ptr_index, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: access_index, ptr_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: slot_address, target_address
        character(len=:), allocatable :: component_name
        integer :: symbol_index, type_index, component_index, target_type

        call resolve_component_ref(arena, access_index, context, symbol_index, &
                                   type_index, component_index, component_name, &
                                   error_msg)
        if (len_trim(error_msg) > 0) return
        target_type = context%derived_types(type_index)%component_type_index( &
            component_index)
        if (.not. context%symbols(ptr_index)%is_derived .or. &
            (.not. (target_type == context%symbols(ptr_index)%derived_type_index .or. &
                    derived_type_extends(context, target_type, &
                                         context%symbols(ptr_index)%derived_type_index)))) then
            error_msg = 'pointer and derived component target types do not match'
            return
        end if
        call component_element_address_at(context, symbol_index, type_index, &
                                           component_index, 0, slot_address, &
                                           error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_ptr_load(context%session, slot_address, target_address, &
                                error_msg)) return
        context%symbols(ptr_index)%address = target_address
        context%symbols(ptr_index)%element_address = target_address
        context%symbols(ptr_index)%has_address = .true.
        context%symbols(ptr_index)%is_reference = .true.
        context%symbols(ptr_index)%is_associated = .true.
        context%symbols(ptr_index)%has_runtime_association = .true.
        call set_empty(error_msg)
    end subroutine alias_pointer_to_derived_component

    subroutine alias_pointer_to_derived_array_component(arena, target_index, &
                                                        context, ptr_index, &
                                                        error_msg)
        ! Bind p => arr%component. The pointer is a rank-1 view whose base is
        ! the first component slot and whose stride is one complete arr element.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: target_index, ptr_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: target_symbol, type_index, component_index, extent, rank, i
        integer :: value_kind
        type(lr_operand_desc_t) :: first_address

        call resolve_derived_array_component(arena, target_index, context, &
                                             target_symbol, type_index, &
                                             component_index, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. context%symbols(target_symbol)%is_target .and. &
            .not. context%symbols(target_symbol)%is_pointer) then
            error_msg = 'pointer assignment source is not a target or pointer'
            return
        end if
        if (.not. context%symbols(target_symbol)%has_address) then
            error_msg = 'pointer assignment source has no storage'
            return
        end if
        if (.not. context%symbols(ptr_index)%is_array) then
            error_msg = 'left of => is a scalar pointer but the target is an array'
            return
        end if
        rank = context%symbols(target_symbol)%array_rank
        if (rank /= 1) then
            error_msg = 'derived array component pointer association supports rank 1 only'
            return
        end if
        if (context%symbols(ptr_index)%array_rank /= rank) then
            error_msg = 'pointer and target array ranks do not match'
            return
        end if
        value_kind = component_kind(context, type_index, component_index)
        if (context%symbols(ptr_index)%value_kind /= value_kind) then
            error_msg = 'pointer and target array element kinds do not match'
            return
        end if
        extent = derived_array_component_extent(context, target_symbol)
        if (extent <= 0) then
            error_msg = 'pointer assignment source has no positive extent'
            return
        end if
        call derived_array_component_address_at(context, target_symbol, type_index, &
            component_index, i32_immediate(context%session, 0_c_int64_t), &
            first_address, error_msg)
        if (len_trim(error_msg) > 0) return

        context%symbols(ptr_index)%is_array = .true.
        context%symbols(ptr_index)%array_size = extent
        context%symbols(ptr_index)%array_lower_bound = &
            context%symbols(target_symbol)%array_lower_bound
        context%symbols(ptr_index)%array_dim_sizes = 0
        context%symbols(ptr_index)%array_dim_lowers = 0
        do i = 1, rank
            context%symbols(ptr_index)%array_dim_sizes(i) = &
                context%symbols(target_symbol)%array_dim_sizes(i)
            context%symbols(ptr_index)%array_dim_lowers(i) = &
                context%symbols(target_symbol)%array_dim_lowers(i)
        end do
        context%symbols(ptr_index)%element_address = first_address
        context%symbols(ptr_index)%address = first_address
        context%symbols(ptr_index)%has_address = .true.
        context%symbols(ptr_index)%is_reference = .true.
        context%symbols(ptr_index)%is_associated = .true.
        context%symbols(ptr_index)%has_runtime_association = .false.
        context%symbols(ptr_index)%array_stride_bytes = &
            total_component_slots(context, type_index) * 4
        context%symbols(ptr_index)%has_runtime_array_stride = .false.
        call set_empty(error_msg)
    end subroutine alias_pointer_to_derived_array_component

    logical function target_outlives_procedure(symbol) result(outlives)
        ! A target whose storage survives the current invocation: a dummy
        ! argument (the caller owns the storage) or an entity in static
        ! storage (module variable, COMMON block, saved local).
        type(symbol_t), intent(in) :: symbol

        outlives = symbol%is_dummy_argument .or. symbol_has_global_storage(symbol)
    end function target_outlives_procedure

    logical function data_pointer_result_call(arena, node_index, context) &
        result(is_call)
        ! True when node_index is a call to a function whose result is a pointer
        ! (data or procedure pointer).
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context

        is_call = .false.
        if (.not. node_exists(arena, node_index)) return
        select type (call_node => arena%entries(node_index)%node)
        type is (call_or_subscript_node)
            if (.not. allocated(call_node%name)) return
            is_call = contained_function_kind(context, call_node%name) == &
                      VALUE_DATA_PTR_RESULT .or. &
                      contained_function_kind(context, call_node%name) == &
                      VALUE_PROC_PTR
        end select
    end function data_pointer_result_call

    subroutine bind_data_pointer_result(arena, node_index, context, ptr_index, &
                                        error_msg)
        ! p => f(...): call f, which returns the target address, and bind that
        ! address into p. No pointee storage is copied, so a write through p
        ! reaches the same target the callee returned. The association state is
        ! only known at run time, so ASSOCIATED(p) compares against null.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: ptr_index
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t), allocatable :: args(:)
        integer, allocatable :: copyback_indices(:)
        type(lr_operand_desc_t) :: result_address
        integer :: pointee_kind

        call set_empty(error_msg)
        select type (call_node => arena%entries(node_index)%node)
        type is (call_or_subscript_node)
            if (contained_function_kind(context, call_node%name) == VALUE_PROC_PTR) then
                if (allocated(call_node%arg_indices)) then
                    call prepare_reference_args(arena, call_node%arg_indices, &
                        context, VALUE_F32, call_node%name, args, &
                        copyback_indices, error_msg)
                else
                    call prepare_reference_args(arena, NO_ACTUAL_ARGS, context, &
                        VALUE_F32, call_node%name, args, copyback_indices, error_msg)
                end if
                if (len_trim(error_msg) > 0) return
                if (.not. emit_ptr_call(context%session, &
                        call_emit_name(arena, call_node%name, context), args, &
                        result_address, error_msg)) return
                call copy_back_reference_args(context, args, copyback_indices, &
                    error_msg)
                if (len_trim(error_msg) > 0) return
                if (.not. emit_ptr_store(context%session, result_address, &
                        context%symbols(ptr_index)%address, error_msg)) return
                context%symbols(ptr_index)%is_associated = .true.
                call set_empty(error_msg)
                return
            end if
            pointee_kind = data_pointer_result_kind(context, call_node%name)
            if (pointee_kind /= context%symbols(ptr_index)%value_kind) then
                error_msg = 'pointer and function result scalar kinds do not '// &
                            'match: '//trim(call_node%name)
                return
            end if
            if (allocated(call_node%arg_indices)) then
                call prepare_reference_args(arena, call_node%arg_indices, &
                                            context, pointee_kind, &
                                            call_node%name, args, &
                                            copyback_indices, error_msg)
                if (len_trim(error_msg) > 0) return
            else
                call prepare_reference_args(arena, NO_ACTUAL_ARGS, context, &
                                            pointee_kind, call_node%name, &
                                            args, copyback_indices, error_msg)
                if (len_trim(error_msg) > 0) return
            end if
            if (.not. emit_ptr_call(context%session, &
                    call_emit_name(arena, call_node%name, context), args, &
                    result_address, error_msg)) return
            call copy_back_reference_args(context, args, copyback_indices, &
                                          error_msg)
            if (len_trim(error_msg) > 0) return
        class default
            error_msg = 'pointer assignment source is not a function call'
            return
        end select
        context%symbols(ptr_index)%address = result_address
        context%symbols(ptr_index)%has_address = .true.
        context%symbols(ptr_index)%is_reference = .true.
        context%symbols(ptr_index)%is_associated = .true.
        context%symbols(ptr_index)%has_runtime_association = .true.
    end subroutine bind_data_pointer_result

    integer function data_pointer_result_kind(context, name) result(pointee_kind)
        ! Pointee scalar kind of a data-pointer function result, read back from
        ! the callee's own result declaration.
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: name
        type(function_def_node), pointer :: fn_node
        integer :: table_index, node_index

        pointee_kind = VALUE_I32
        table_index = find_function_index(context, name)
        if (table_index <= 0) return
        node_index = context%function_node_indices(table_index)
        if (node_index <= 0) return
        if (.not. node_exists(context%arena, node_index)) return
        fn_node => get_node_as_function_def(context%arena, node_index)
        if (.not. associated(fn_node)) return
        if (.not. result_var_is_data_pointer(fn_node, context, pointee_kind)) &
            pointee_kind = VALUE_I32
    end function data_pointer_result_kind

    logical function promote_inferred_data_pointer(context, symbol_index)
        ! Recognize a data pointer declared only by a bare POINTER statement
        ! that the current FortFront AST does not retain. Explicit declarations
        ! already carry a binding or the POINTER flag and never use this path.
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        integer :: value_kind

        promote_inferred_data_pointer = .false.
        if (symbol_index <= 0 .or. symbol_index > context%symbol_count) return
        if (context%symbols(symbol_index)%has_binding) return
        if (context%symbols(symbol_index)%is_target .or. &
            context%symbols(symbol_index)%is_proc_pointer) return
        if (context%symbols(symbol_index)%has_address) return
        value_kind = context%symbols(symbol_index)%value_kind
        if (value_kind /= VALUE_I32 .and. value_kind /= VALUE_F32 .and. &
            value_kind /= VALUE_F64 .and. value_kind /= VALUE_LOGICAL) return
        context%symbols(symbol_index)%is_pointer = .true.
        context%symbols(symbol_index)%is_associated = .false.
        context%symbols(symbol_index)%is_reference = .false.
        promote_inferred_data_pointer = .true.
    end function promote_inferred_data_pointer

    subroutine alias_pointer_array(context, ptr_index, target_index, error_msg)
        ! Whole-array p => t: the pointer array adopts the target array's base,
        ! extent, element kind, and bounds, so element access through the pointer
        ! GEPs the same storage. Both sides must be rank-1 fixed-size arrays of
        ! the same element kind (the supported subset of #245 B3a).
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: ptr_index, target_index
        character(len=:), allocatable :: error_msg
        integer :: i

        if (.not. context%symbols(ptr_index)%is_array) then
            error_msg = 'left of => is a scalar pointer but the target is an array'
            return
        end if
        if (.not. context%symbols(target_index)%is_array) then
            error_msg = 'left of => is an array pointer but the target is a scalar'
            return
        end if
        if (context%symbols(ptr_index)%array_rank /= &
            context%symbols(target_index)%array_rank) then
            error_msg = 'pointer and target array ranks do not match'
            return
        end if
        if (context%symbols(ptr_index)%value_kind /= &
            context%symbols(target_index)%value_kind) then
            error_msg = 'pointer and target array element kinds do not match'
            return
        end if
        ! Adopt the target's element storage and shape. element_address is the
        ! array base used by the GEP element-addressing path; array_size and the
        ! per-dimension shape drive size(), bounds, and whole-array print.
        context%symbols(ptr_index)%element_address = &
            context%symbols(target_index)%element_address
        context%symbols(ptr_index)%array_size = &
            context%symbols(target_index)%array_size
        context%symbols(ptr_index)%array_lower_bound = &
            context%symbols(target_index)%array_lower_bound
        do i = 1, 2
            context%symbols(ptr_index)%array_dim_sizes(i) = &
                context%symbols(target_index)%array_dim_sizes(i)
            context%symbols(ptr_index)%array_dim_lowers(i) = &
                context%symbols(target_index)%array_dim_lowers(i)
        end do
        context%symbols(ptr_index)%address = &
            context%symbols(target_index)%address
        context%symbols(ptr_index)%has_address = .true.
        context%symbols(ptr_index)%is_reference = .true.
        context%symbols(ptr_index)%is_associated = .true.
        context%symbols(ptr_index)%array_stride_bytes = &
            context%symbols(target_index)%array_stride_bytes
        if (context%symbols(ptr_index)%array_stride_bytes == 0) then
            context%symbols(ptr_index)%array_stride_bytes = &
                equivalence_kind_size(context%symbols(ptr_index)%value_kind)
        end if
        context%symbols(ptr_index)%has_runtime_array_stride = &
            context%symbols(target_index)%has_runtime_array_stride
        context%symbols(ptr_index)%runtime_array_stride_bytes = &
            context%symbols(target_index)%runtime_array_stride_bytes
        call set_empty(error_msg)
    end subroutine alias_pointer_array

    subroutine alias_pointer_array_section(arena, target_index, context, &
                                            ptr_index, error_msg)
        ! p => a(lo:hi:s): bind p to a strided section view of the target
        ! array.  Element addressing through p uses the descriptor byte-stride
        ! path, so p(i) steps by the section's stride across the target's
        ! storage (#338).
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: target_index
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: ptr_index
        character(len=:), allocatable, intent(out) :: error_msg
        type(array_section_info_t) :: info
        type(lr_operand_desc_t) :: index_op, base_addr
        integer(c_int64_t) :: first_linear, stride_bytes
        integer :: src, vk, arr_size

        select type (slice => arena%entries(target_index)%node)
        type is (array_slice_node)
            call describe_array_section(arena, slice, context, info, error_msg, &
                                        allow_derived=.true.)
            if (len_trim(error_msg) > 0) return
        class default
            error_msg = 'pointer assignment section expected an array slice node'
            return
        end select
        if (info%result_rank /= 1) then
            error_msg = 'pointer array sections support rank-1 views only'
            return
        end if
        if (info%has_runtime_bounds) then
            error_msg = 'pointer array sections require compile-time bounds'
            return
        end if
        src = info%source_index
        if (src <= 0 .or. src > context%symbol_count) then
            error_msg = 'pointer array section source was not declared'
            return
        end if
        if (.not. context%symbols(ptr_index)%is_array) then
            error_msg = 'left of => is a scalar pointer but the target is an array'
            return
        end if
        if (.not. context%symbols(src)%is_array) then
            error_msg = 'pointer array section source is not an array'
            return
        end if
        if (.not. (context%symbols(src)%is_target .or. &
                   context%symbols(src)%is_pointer)) then
            error_msg = 'right of => is not a target or pointer'
            return
        end if
        if (context%symbols(src)%is_pointer) then
            ! A section of a pointer may already carry a non-unit or negative
            ! descriptor stride. Compose that stride before accepting the
            ! association, or reject it rather than treating the source as
            ! contiguous (#338).
            error_msg = 'pointer array section source must be a contiguous target'
            return
        end if
        if (.not. context%symbols(src)%has_address) then
            error_msg = 'pointer array section source has no storage'
            return
        end if
        if (context%symbols(ptr_index)%array_rank /= info%result_rank) then
            error_msg = 'pointer and section ranks do not match'
            return
        end if
        if (context%symbols(ptr_index)%value_kind /= &
            context%symbols(src)%value_kind) then
            error_msg = 'pointer and target array element kinds do not match'
            return
        end if
        if (context%symbols(ptr_index)%is_derived .neqv. &
            context%symbols(src)%is_derived) then
            error_msg = 'pointer and target array types do not match'
            return
        end if
        if (context%symbols(ptr_index)%is_derived .and. &
            context%symbols(ptr_index)%derived_type_index /= &
            context%symbols(src)%derived_type_index) then
            if (.not. derived_type_extends(context, &
                    context%symbols(src)%derived_type_index, &
                    context%symbols(ptr_index)%derived_type_index)) then
                error_msg = 'pointer and target array derived types do not match'
                return
            end if
        end if
        if (context%symbols(src)%is_allocatable) then
            error_msg = 'pointer array section of an allocatable target is '// &
                        'not supported'
            return
        end if
        vk = context%symbols(src)%value_kind
        if (vk /= VALUE_I32 .and. vk /= VALUE_LOGICAL .and. &
            vk /= VALUE_F32 .and. vk /= VALUE_F64) then
            error_msg = 'pointer array sections do not support this element kind'
            return
        end if
        arr_size = context%symbols(src)%array_size
        first_linear = array_section_source_linear_index(info, 0_c_int64_t)
        index_op = i32_immediate(context%session, first_linear)
        select case (vk)
        case (VALUE_F32)
            if (.not. emit_f32_array_element_addr(context%session, arr_size, &
                    context%symbols(src)%element_address, index_op, base_addr, &
                    error_msg)) return
        case (VALUE_F64)
            if (.not. emit_f64_array_element_addr(context%session, arr_size, &
                    context%symbols(src)%element_address, index_op, base_addr, &
                    error_msg)) return
        case (VALUE_I32, VALUE_LOGICAL)
            if (.not. emit_i32_array_element_addr(context%session, arr_size, &
                    context%symbols(src)%element_address, index_op, base_addr, &
                    error_msg)) return
        end select

        stride_bytes = info%section_strides(info%kept_dims(1)) * &
                       allocatable_elem_size(vk)
        context%symbols(ptr_index)%element_address = base_addr
        context%symbols(ptr_index)%array_size = int(info%section_extents( &
            info%kept_dims(1)))
        context%symbols(ptr_index)%array_lower_bound = 1
        context%symbols(ptr_index)%array_rank = 1
        context%symbols(ptr_index)%has_runtime_array_stride = .true.
        context%symbols(ptr_index)%runtime_array_stride_bytes = &
            i64_immediate(context%session, stride_bytes)
        context%symbols(ptr_index)%has_address = .true.
        context%symbols(ptr_index)%is_reference = .true.
        context%symbols(ptr_index)%is_associated = .true.
        context%symbols(ptr_index)%has_runtime_association = .false.
        call set_empty(error_msg)
    end subroutine alias_pointer_array_section

    subroutine lower_nullify(arena, node, context, error_msg)
        ! nullify(p [, q ...]) : disassociate each pointer.
        type(ast_arena_t), intent(in) :: arena
        type(nullify_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: name
        integer :: i, symbol_index

        call set_empty(error_msg)
        if (.not. allocated(node%pointer_indices)) then
            error_msg = 'nullify has no pointer arguments'
            return
        end if
        do i = 1, size(node%pointer_indices)
            if (is_data_pointer_component_access(arena, node%pointer_indices(i), &
                                                 context)) then
                call nullify_data_pointer_component(arena, node%pointer_indices(i), &
                                                    context, error_msg)
                if (len_trim(error_msg) > 0) return
                cycle
            end if
            call get_identifier_name(arena, node%pointer_indices(i), name, error_msg)
            if (len_trim(error_msg) > 0) return
            symbol_index = find_symbol_compat(context, name)
            if (symbol_index <= 0) then
                error_msg = 'nullify argument was not declared: '//trim(name)
                return
            end if
            if (.not. context%symbols(symbol_index)%is_pointer) then
                error_msg = 'nullify argument is not a pointer: '//trim(name)
                return
            end if
            if (context%symbols(symbol_index)%is_proc_pointer) then
                if (.not. emit_ptr_store(context%session, &
                        null_ptr_operand(context), &
                        context%symbols(symbol_index)%address, error_msg)) return
            end if
            context%symbols(symbol_index)%is_associated = .false.
            context%symbols(symbol_index)%has_runtime_association = .false.
            context%symbols(symbol_index)%has_address = .false.
            context%symbols(symbol_index)%is_reference = .false.
        end do
    end subroutine lower_nullify

    subroutine lower_associated(arena, arg_indices, context, value, error_msg)
        ! associated(p) / associated(p, t): compile-time association check
        ! (#245 B3a / B3c). One-argument form folds to the `is_associated` flag.
        ! Two-argument form additionally checks that p and t point to the same
        ! target (identical address operand in compile-time straight-line code).
        type(ast_arena_t), intent(in) :: arena
        integer, allocatable, intent(in) :: arg_indices(:)
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: ptr_name, target_name
        type(lr_operand_desc_t) :: field_addr, ptr_value
        integer :: ptr_idx, tgt_idx
        logical :: same_target

        call set_empty(error_msg)
        if (.not. allocated(arg_indices)) then
            error_msg = 'associated requires at least one argument'
            return
        end if
        if (size(arg_indices) < 1 .or. size(arg_indices) > 2) then
            error_msg = 'associated requires one or two arguments'
            return
        end if
        ! A procedure-pointer component operand has no compile-time association
        ! flag; compare its stored callee address at run time instead (#372).
        if (is_data_pointer_component_access(arena, arg_indices(1), context)) then
            call lower_data_pointer_component_associated(arena, arg_indices, &
                                                         context, value, error_msg)
            return
        end if
        if (associated_uses_proc_component(arena, arg_indices, context)) then
            call lower_proc_pointer_associated(arena, arg_indices, context, &
                                               value, error_msg)
            return
        end if
        if (.not. is_identifier(arena, arg_indices(1))) then
            error_msg = 'associated first argument must be a pointer variable'
            return
        end if
        call get_identifier_name(arena, arg_indices(1), ptr_name, error_msg)
        if (len_trim(error_msg) > 0) return
        ptr_idx = find_symbol_compat(context, ptr_name)
        if (ptr_idx <= 0) then
            error_msg = 'associated argument was not declared: '//trim(ptr_name)
            return
        end if
        if (.not. context%symbols(ptr_idx)%is_pointer .and. &
            .not. context%symbols(ptr_idx)%is_proc_pointer) then
            error_msg = 'associated argument is not a pointer: '//trim(ptr_name)
            return
        end if
        ! Procedure-pointer association is a run-time property of the slot.
        ! Folding the lowering-time flag is wrong when an assignment occurs
        ! inside a branch (the slot remains null when that branch is not taken).
        if (context%symbols(ptr_idx)%is_proc_pointer) then
            call lower_proc_pointer_associated(arena, arg_indices, context, &
                                               value, error_msg)
            return
        end if
        if (size(arg_indices) == 1) then
            ! A scalar class pointer has a runtime descriptor even before its
            ! first ALLOCATE. Read its data field rather than folding the
            ! lowering-time association flag, so descriptor nulling is part of
            ! the same runtime contract as dynamic dispatch.
            if (context%symbols(ptr_idx)%is_class_pointer) then
                if (.not. emit_ptr_offset(context%session, &
                        context%symbols(ptr_idx)%class_descriptor_address, &
                        int(POLYMORPHIC_DESCRIPTOR_DATA_OFFSET, c_int64_t), &
                        field_addr, error_msg)) return
                if (.not. emit_ptr_load(context%session, field_addr, ptr_value, &
                        error_msg)) return
                if (.not. emit_liric_i32_icmp(context%session, LR_CMP_NE, &
                        ptr_value, null_ptr_operand(context), value, error_msg)) &
                    return
                return
            end if
            ! A pointer bound to a data-pointer function result has no
            ! compile-time association flag: test its address against null.
            if (context%symbols(ptr_idx)%has_runtime_association) then
                if (.not. emit_liric_i32_icmp(context%session, LR_CMP_NE, &
                        context%symbols(ptr_idx)%address, &
                        null_ptr_operand(context), value, error_msg)) return
                return
            end if
            ! One-argument: fold to compile-time association flag.
            if (context%symbols(ptr_idx)%is_associated) then
                value = i32_immediate(context%session, 1_c_int64_t)
            else
                value = i32_immediate(context%session, 0_c_int64_t)
            end if
            return
        end if
        ! Two-argument form: also require same target (B3c).
        if (.not. is_identifier(arena, arg_indices(2))) then
            error_msg = 'associated second argument must be a target variable'
            return
        end if
        call get_identifier_name(arena, arg_indices(2), target_name, error_msg)
        if (len_trim(error_msg) > 0) return
        tgt_idx = find_symbol_compat(context, target_name)
        if (tgt_idx <= 0) then
            error_msg = 'associated second argument was not declared: '// &
                        trim(target_name)
            return
        end if
        ! For compile-time straight-line code compare address vreg payloads.
        same_target = .false.
        if (context%symbols(ptr_idx)%is_associated .and. &
            context%symbols(tgt_idx)%has_address) then
            same_target = &
                context%symbols(ptr_idx)%address%payload == &
                context%symbols(tgt_idx)%address%payload
        end if
        if (same_target) then
            value = i32_immediate(context%session, 1_c_int64_t)
        else
            value = i32_immediate(context%session, 0_c_int64_t)
        end if
    end subroutine lower_associated

    subroutine lower_proc_pointer_declaration(node, context, error_msg)
        ! procedure(iface), pointer :: fp => null() (#245 slice B3d).
        ! A procedure pointer is stored as a ptr alloca slot; the callee address
        ! is loaded from that slot at each indirect call site.
        type(declaration_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i

        if (node%is_multi_declaration .and. allocated(node%var_names)) then
            do i = 1, size(node%var_names)
                call define_proc_pointer_symbol(context, node%var_names(i), &
                                                error_msg)
                if (len_trim(error_msg) > 0) return
            end do
        else if (allocated(node%var_name)) then
            call define_proc_pointer_symbol(context, node%var_name, error_msg)
        else
            error_msg = 'procedure pointer declaration has no variable name'
        end if
    end subroutine lower_proc_pointer_declaration

    subroutine lower_c_f_procpointer(arena, arg_indices, context, error_msg)
        ! c_f_procpointer(cfunptr, fptr): copy a C function address into a
        ! Fortran procedure-pointer slot. C_FUNPTR uses the same pointer-sized
        ! operand representation as C_PTR in the direct session backend.
        type(ast_arena_t), intent(in) :: arena
        integer, allocatable, intent(in) :: arg_indices(:)
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: ptr_value
        character(len=:), allocatable :: name
        integer :: symbol_index

        call set_empty(error_msg)
        if (.not. allocated(arg_indices)) then
            error_msg = 'c_f_procpointer requires exactly two arguments'
            return
        end if
        if (size(arg_indices) /= 2) then
            error_msg = 'c_f_procpointer requires exactly two arguments'
            return
        end if
        call lower_c_ptr_expression(arena, arg_indices(1), context, ptr_value, &
                                    error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. is_identifier(arena, arg_indices(2))) then
            error_msg = 'c_f_procpointer target must be a procedure pointer'
            return
        end if
        call get_identifier_name(arena, arg_indices(2), name, error_msg)
        if (len_trim(error_msg) > 0) return
        symbol_index = find_symbol_compat(context, name)
        if (symbol_index <= 0) then
            error_msg = 'c_f_procpointer target was not declared as a procedure pointer: '// &
                trim(name)
            return
        end if
        if (.not. context%symbols(symbol_index)%is_proc_pointer) then
            error_msg = 'c_f_procpointer target was not declared as a procedure pointer: '// &
                trim(name)
            return
        end if
        if (.not. emit_ptr_store(context%session, ptr_value, &
                                 context%symbols(symbol_index)%address, error_msg)) return
        context%symbols(symbol_index)%is_associated = .true.
    end subroutine lower_c_f_procpointer

    subroutine define_proc_pointer_symbol(context, name, error_msg)
        ! Allocate a ptr slot on the stack; the slot holds the function address.
        ! Initially null (zeroed by alloca semantics at declaration).
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: name
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: idx

        idx = find_symbol_compat(context, name)
        if (idx > 0) then
            ! Already declared (e.g. a dummy re-declaration); benign.
            call set_empty(error_msg)
            return
        end if
        call grow_symbols(context)
        idx = context%symbol_count + 1
        context%symbols(idx)%name = trim(name)
        context%symbols(idx)%value_kind = VALUE_PROC_PTR
        context%symbols(idx)%is_pointer = .true.
        context%symbols(idx)%is_proc_pointer = .true.
        context%symbols(idx)%is_associated = .false.
        ! Allocate a ptr slot on the stack; address stores the slot address.
        if (.not. emit_ptr_alloca(context%session, &
                                  context%symbols(idx)%address, error_msg)) return
        ! A fresh procedure pointer is disassociated: null the slot so a runtime
        ! association test reads false before any => binding (#372).
        if (.not. emit_ptr_store(context%session, null_ptr_operand(context), &
                                 context%symbols(idx)%address, error_msg)) return
        context%symbols(idx)%has_address = .true.
        context%symbol_count = idx
        call set_empty(error_msg)
    end subroutine define_proc_pointer_symbol

    subroutine lower_proc_pointer_assignment(arena, node, context, error_msg)
        ! fp => my_func : store the function address into fp's ptr slot.
        ! Also handles fp => null() which clears the association.
        type(ast_arena_t), intent(in) :: arena
        type(pointer_assignment_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: ptr_name, target_name
        integer :: ptr_idx, target_idx
        type(lr_operand_desc_t) :: callee_global
        integer(c_int32_t) :: symbol_id
        character(kind=c_char), allocatable :: c_name(:)

        call get_identifier_name(arena, node%pointer_index, ptr_name, error_msg)
        if (len_trim(error_msg) > 0) return
        ptr_idx = find_symbol_compat(context, ptr_name)
        if (ptr_idx <= 0) then
            error_msg = 'procedure pointer was not declared: '//trim(ptr_name)
            return
        end if
        if (.not. context%symbols(ptr_idx)%is_proc_pointer) then
            error_msg = 'left of => is not a procedure pointer: '//trim(ptr_name)
            return
        end if
        ! null() clears the association and nulls the slot.
        if (pointer_target_is_null(arena, node%target_index)) then
            context%symbols(ptr_idx)%is_associated = .false.
            if (.not. emit_ptr_store(context%session, null_ptr_operand(context), &
                                     context%symbols(ptr_idx)%address, &
                                     error_msg)) return
            call set_empty(error_msg)
            return
        end if
        if (data_pointer_result_call(arena, node%target_index, context)) then
            call bind_data_pointer_result(arena, node%target_index, context, &
                                          ptr_idx, error_msg)
            return
        end if
        if (.not. is_identifier(arena, node%target_index)) then
            error_msg = 'procedure pointer assignment requires a procedure name'
            return
        end if
        call get_identifier_name(arena, node%target_index, target_name, error_msg)
        if (len_trim(error_msg) > 0) return
        ! Check whether target is another procedure pointer being copied.
        target_idx = find_symbol_compat(context, target_name)
        if (target_idx > 0 .and. context%symbols(target_idx)%is_proc_pointer) then
            ! Copy the callee ptr from the source slot into the destination slot.
            if (.not. emit_ptr_load(context%session, &
                                    context%symbols(target_idx)%address, &
                                    callee_global, error_msg)) return
            if (.not. emit_ptr_store(context%session, callee_global, &
                                     context%symbols(ptr_idx)%address, &
                                     error_msg)) return
            context%symbols(ptr_idx)%is_associated = &
                context%symbols(target_idx)%is_associated
            call set_empty(error_msg)
            return
        end if
        ! Target is a named procedure: intern its symbol and store as ptr.
        call to_c_chars(trim(call_emit_name(context%arena, target_name, context)), c_name)
        symbol_id = lr_session_intern(context%session%handle, c_name)
        if (symbol_id < 0_c_int32_t) then
            error_msg = 'could not intern procedure symbol: '//trim(target_name)
            return
        end if
        callee_global%kind = LR_OP_KIND_GLOBAL
        callee_global%payload = int(symbol_id, c_int64_t)
        callee_global%typ = lr_type_ptr_s(context%session%handle)
        callee_global%global_offset = 0_c_int64_t
        if (.not. emit_ptr_store(context%session, callee_global, &
                                 context%symbols(ptr_idx)%address, error_msg)) return
        context%symbols(ptr_idx)%is_associated = .true.
        call set_empty(error_msg)
    end subroutine lower_proc_pointer_assignment

    logical function is_proc_pointer_call(context, name) result(ok)
        ! Return .true. when name is a declared procedure pointer symbol.
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: name
        integer :: idx

        idx = find_symbol_compat(context, name)
        ok = .false.
        if (idx <= 0) return
        ok = context%symbols(idx)%is_proc_pointer
    end function is_proc_pointer_call

    subroutine resolve_proc_pointer_callee_name(arena, ptr_name, callee_name)
        ! Name to validate an indirect call against (#362). A procedure pointer
        ! associated with a procedure defined or interfaced in this unit keeps
        ! that procedure's interface: the call must use the same descriptor
        ! passing and rank checks as a direct call, so argument lowering is
        ! driven by the target's name. Without a resolvable target the pointer's
        ! own name is kept, which carries no dummy information.
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: ptr_name
        character(len=:), allocatable, intent(out) :: callee_name
        integer :: n
        character(len=:), allocatable :: lhs_name, rhs_name, err

        callee_name = trim(ptr_name)
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (pnode => arena%entries(n)%node)
            type is (pointer_assignment_node)
                if (.not. is_identifier(arena, pnode%pointer_index)) cycle
                if (.not. is_identifier(arena, pnode%target_index)) cycle
                call get_identifier_name(arena, pnode%pointer_index, lhs_name, err)
                if (len_trim(err) > 0) cycle
                if (.not. same_name(lhs_name, ptr_name)) cycle
                call get_identifier_name(arena, pnode%target_index, rhs_name, err)
                if (len_trim(err) > 0) cycle
                if (arena_proc_param_count(arena, rhs_name) < 0) cycle
                callee_name = trim(rhs_name)
                return
            end select
        end do
    end subroutine resolve_proc_pointer_callee_name

    subroutine lower_i32_proc_ptr_call(arena, node, context, value, error_msg)
        ! Indirect integer function call through a procedure pointer.
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: callee_ptr
        type(lr_operand_desc_t), allocatable :: args(:)
        integer, allocatable :: copyback_indices(:)
        integer :: idx
        character(len=:), allocatable :: callee_name

        idx = find_symbol_compat(context, node%name)
        call resolve_proc_pointer_callee_name(arena, node%name, callee_name)
        if (.not. emit_ptr_load(context%session, context%symbols(idx)%address, &
                                callee_ptr, error_msg)) return
        if (allocated(node%arg_indices)) then
            call prepare_reference_args(arena, node%arg_indices, context, &
                                        VALUE_I32, callee_name, args, &
                                        copyback_indices, error_msg)
            if (len_trim(error_msg) > 0) return
        else
            allocate (args(0))
            allocate (copyback_indices(0))
        end if
        if (.not. emit_i32_indirect_call(context%session, callee_ptr, args, &
                                         value, error_msg)) return
        call copy_back_reference_args(context, args, copyback_indices, error_msg)
    end subroutine lower_i32_proc_ptr_call

    subroutine lower_void_proc_ptr_call(arena, node_index, context, error_msg)
        ! Indirect void (subroutine) call through a procedure pointer.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: name
        integer, allocatable :: arg_indices(:)
        type(lr_operand_desc_t) :: callee_ptr
        type(lr_operand_desc_t), allocatable :: args(:)
        integer, allocatable :: copyback_indices(:)
        integer :: idx
        character(len=:), allocatable :: callee_name

        call get_subroutine_call_name(arena, node_index, name, error_msg)
        if (len_trim(error_msg) > 0) return
        idx = find_symbol_compat(context, name)
        call resolve_proc_pointer_callee_name(arena, name, callee_name)
        if (.not. emit_ptr_load(context%session, context%symbols(idx)%address, &
                                callee_ptr, error_msg)) return
        call get_subroutine_call_arg_indices(arena, node_index, arg_indices, &
                                             error_msg)
        if (len_trim(error_msg) > 0) return
        call prepare_reference_args(arena, arg_indices, context, VALUE_I32, &
                                    callee_name, args, copyback_indices, &
                                    error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_void_indirect_call(context%session, callee_ptr, args, &
                                          error_msg)) return
        call copy_back_reference_args(context, args, copyback_indices, error_msg)
    end subroutine lower_void_proc_ptr_call

    logical function is_proc_pointer_component_access(arena, node_index, context) &
        result(is_proc)
        ! True when node_index is obj%p naming a procedure-pointer component
        ! (#372). Unresolvable component references answer .false. so the
        ! caller's other paths keep their own diagnostics.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        integer :: sym, type_index, comp_index
        character(len=:), allocatable :: comp_name, local_error

        is_proc = .false.
        if (node_index <= 0) return
        if (.not. node_exists(arena, node_index)) return
        select type (acc => arena%entries(node_index)%node)
        type is (component_access_node)
        class default
            return
        end select
        call resolve_component_ref(arena, node_index, context, sym, type_index, &
                                   comp_index, comp_name, local_error)
        if (len_trim(local_error) > 0) return
        is_proc = component_kind(context, type_index, comp_index) == VALUE_PROC_PTR
    end function is_proc_pointer_component_access

    logical function associated_uses_proc_component(arena, arg_indices, context) &
        result(uses)
        ! True when either ASSOCIATED argument is a procedure-pointer component.
        type(ast_arena_t), intent(in) :: arena
        integer, allocatable, intent(in) :: arg_indices(:)
        type(lowering_context_t), intent(in) :: context

        uses = .false.
        if (.not. allocated(arg_indices)) return
        if (size(arg_indices) < 1 .or. size(arg_indices) > 2) return
        uses = is_proc_pointer_component_access(arena, arg_indices(1), context)
        if (uses) return
        if (size(arg_indices) < 2) return
        uses = is_proc_pointer_component_access(arena, arg_indices(2), context)
    end function associated_uses_proc_component

    logical function pointer_target_is_null(arena, node_index) result(is_null)
        ! True for the null() intrinsic reference on the right of =>, whether it
        ! reaches us as a bare identifier or as an argumentless call node.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        character(len=:), allocatable :: name, local_error

        is_null = .false.
        if (node_index <= 0) return
        if (.not. node_exists(arena, node_index)) return
        if (is_identifier(arena, node_index)) then
            call get_identifier_name(arena, node_index, name, local_error)
            if (len_trim(local_error) > 0) return
            is_null = same_name(name, 'null')
            return
        end if
        select type (target_node => arena%entries(node_index)%node)
        type is (call_or_subscript_node)
            if (.not. allocated(target_node%name)) return
            is_null = same_name(target_node%name, 'null')
        end select
    end function pointer_target_is_null

    subroutine proc_component_slot_address(arena, access_index, context, addr, &
                                           error_msg)
        ! Address of the inline pointer slot holding obj%p's callee address.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: access_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: addr
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: comp_name
        integer :: sym, type_index, comp_index

        call resolve_component_ref(arena, access_index, context, sym, type_index, &
                                   comp_index, comp_name, error_msg)
        if (len_trim(error_msg) > 0) return
        if (component_kind(context, type_index, comp_index) /= VALUE_PROC_PTR) then
            error_msg = 'component is not a procedure pointer: '//trim(comp_name)
            return
        end if
        call component_element_address_at(context, sym, type_index, comp_index, 0, &
                                          addr, error_msg)
    end subroutine proc_component_slot_address

    subroutine load_proc_pointer_operand(arena, node_index, context, ptr_value, &
                                         error_msg)
        ! Callee address currently held by a procedure-pointer entity: either a
        ! procedure-pointer component obj%p or a scalar procedure pointer.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: ptr_value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: addr
        character(len=:), allocatable :: name
        integer :: idx

        if (is_proc_pointer_component_access(arena, node_index, context)) then
            call proc_component_slot_address(arena, node_index, context, addr, &
                                             error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_ptr_load(context%session, addr, ptr_value, error_msg)) &
                return
            call set_empty(error_msg)
            return
        end if
        if (.not. is_identifier(arena, node_index)) then
            error_msg = 'expected a procedure pointer variable or component'
            return
        end if
        call get_identifier_name(arena, node_index, name, error_msg)
        if (len_trim(error_msg) > 0) return
        idx = find_symbol_compat(context, name)
        if (idx <= 0) then
            error_msg = 'procedure pointer was not declared: '//trim(name)
            return
        end if
        if (.not. context%symbols(idx)%is_proc_pointer) then
            error_msg = 'not a procedure pointer: '//trim(name)
            return
        end if
        if (.not. emit_ptr_load(context%session, context%symbols(idx)%address, &
                                ptr_value, error_msg)) return
        call set_empty(error_msg)
    end subroutine load_proc_pointer_operand

    subroutine proc_target_address(arena, node_index, context, ptr_value, error_msg)
        ! Callee address of a => target or of ASSOCIATED's second argument: a
        ! procedure-pointer entity (loaded) or a named procedure (its symbol).
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: ptr_value
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: name
        character(kind=c_char), allocatable :: c_name(:)
        integer(c_int32_t) :: symbol_id
        integer :: idx

        if (is_proc_pointer_component_access(arena, node_index, context)) then
            call load_proc_pointer_operand(arena, node_index, context, ptr_value, &
                                           error_msg)
            return
        end if
        if (.not. is_identifier(arena, node_index)) then
            error_msg = 'procedure pointer target must be a procedure name or '// &
                        'a procedure pointer'
            return
        end if
        call get_identifier_name(arena, node_index, name, error_msg)
        if (len_trim(error_msg) > 0) return
        idx = find_symbol_compat(context, name)
        if (idx > 0) then
            if (context%symbols(idx)%is_proc_pointer) then
                call load_proc_pointer_operand(arena, node_index, context, &
                                               ptr_value, error_msg)
                return
            end if
            ! A declared data object never has a compatible procedure interface.
            error_msg = 'procedure pointer target is not a procedure: '//trim(name)
            return
        end if
        call to_c_chars(trim(call_emit_name(context%arena, name, context)), c_name)
        symbol_id = lr_session_intern(context%session%handle, c_name)
        if (symbol_id < 0_c_int32_t) then
            error_msg = 'could not intern procedure symbol: '//trim(name)
            return
        end if
        ptr_value%kind = LR_OP_KIND_GLOBAL
        ptr_value%payload = int(symbol_id, c_int64_t)
        ptr_value%typ = lr_type_ptr_s(context%session%handle)
        ptr_value%global_offset = 0_c_int64_t
        call set_empty(error_msg)
    end subroutine proc_target_address

    subroutine lower_proc_component_assignment(arena, node, context, error_msg)
        ! obj%p => proc / obj%p => other%q / obj%p => null(): store the callee
        ! address into the component's inline pointer slot (#372).
        type(ast_arena_t), intent(in) :: arena
        type(pointer_assignment_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: slot_addr, callee

        call proc_component_slot_address(arena, node%pointer_index, context, &
                                         slot_addr, error_msg)
        if (len_trim(error_msg) > 0) return
        if (pointer_target_is_null(arena, node%target_index)) then
            callee = null_ptr_operand(context)
        else
            call proc_target_address(arena, node%target_index, context, callee, &
                                     error_msg)
            if (len_trim(error_msg) > 0) return
        end if
        if (.not. emit_ptr_store(context%session, callee, slot_addr, error_msg)) &
            return
        call set_empty(error_msg)
    end subroutine lower_proc_component_assignment

    logical function is_data_pointer_component_access(arena, node_index, context) &
        result(is_ptr)
        ! True when node_index is obj%p naming a scalar data-pointer component
        ! (#401). Unresolvable component references answer .false. so the
        ! caller's other paths keep their own diagnostics.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        integer :: sym, type_index, comp_index
        character(len=:), allocatable :: comp_name, local_error

        is_ptr = .false.
        if (node_index <= 0) return
        if (.not. node_exists(arena, node_index)) return
        select type (acc => arena%entries(node_index)%node)
        type is (component_access_node)
        class default
            return
        end select
        call resolve_component_ref(arena, node_index, context, sym, type_index, &
                                   comp_index, comp_name, local_error)
        if (len_trim(local_error) > 0) return
        is_ptr = component_is_pointer(context, type_index, comp_index)
    end function is_data_pointer_component_access

    subroutine data_component_slot_address(arena, access_index, context, addr, &
                                           error_msg)
        ! Address of the inline slot holding obj%p's target address.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: access_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: addr
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: comp_name
        integer :: sym, type_index, comp_index

        call resolve_component_ref(arena, access_index, context, sym, type_index, &
                                   comp_index, comp_name, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. component_is_pointer(context, type_index, comp_index)) then
            error_msg = 'component is not a data pointer: '//trim(comp_name)
            return
        end if
        call component_element_address_at(context, sym, type_index, comp_index, 0, &
                                          addr, error_msg)
    end subroutine data_component_slot_address

    subroutine lower_data_component_assignment(arena, node, context, error_msg)
        ! obj%p => t / obj%p => null(): store the target's address into the
        ! component's inline pointer slot (#401). Target validation matches the
        ! whole-variable pointer path: the right side must be a TARGET or a
        ! pointer with storage, of the component's scalar kind.
        type(ast_arena_t), intent(in) :: arena
        type(pointer_assignment_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: slot_addr, target_addr

        call data_component_slot_address(arena, node%pointer_index, context, &
                                         slot_addr, error_msg)
        if (len_trim(error_msg) > 0) return
        if (pointer_target_is_null(arena, node%target_index)) then
            target_addr = null_ptr_operand(context)
        else
            call data_pointer_target_address(arena, node%pointer_index, &
                                             node%target_index, context, &
                                             target_addr, error_msg)
            if (len_trim(error_msg) > 0) return
        end if
        if (.not. emit_ptr_store(context%session, target_addr, slot_addr, &
                                 error_msg)) return
        call set_empty(error_msg)
    end subroutine lower_data_component_assignment

    subroutine data_pointer_target_address(arena, access_index, target_index, &
                                           context, target_addr, error_msg)
        ! Storage address a data-pointer component may adopt: a scalar TARGET
        ! or pointer variable of the component's kind.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: access_index
        integer, intent(in) :: target_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: target_addr
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: target_name, comp_name
        integer :: sym, type_index, comp_index, tgt

        call resolve_component_ref(arena, access_index, context, sym, type_index, &
                                   comp_index, comp_name, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. is_identifier(arena, target_index)) then
            error_msg = 'direct LIRIC session pointer component assignment '// &
                        'requires a scalar target variable'
            return
        end if
        call get_identifier_name(arena, target_index, target_name, error_msg)
        if (len_trim(error_msg) > 0) return
        tgt = find_symbol_compat(context, target_name)
        if (tgt <= 0) then
            error_msg = 'pointer assignment source was not declared: '// &
                        trim(target_name)
            return
        end if
        if (.not. (context%symbols(tgt)%is_target .or. &
                   context%symbols(tgt)%is_pointer)) then
            error_msg = 'right of => is not a target or pointer: '// &
                        trim(target_name)
            return
        end if
        if (component_kind(context, type_index, comp_index) == VALUE_DERIVED) then
            if (.not. context%symbols(tgt)%is_derived .or. &
                context%symbols(tgt)%derived_type_index /= &
                    context%derived_types(type_index)%component_type_index(comp_index)) then
                error_msg = 'pointer component target derived types do not match'
                return
            end if
            if (.not. context%symbols(tgt)%has_address) then
                if (.not. context%symbols(tgt)%is_pointer) then
                    error_msg = 'pointer assignment source has no storage: '// &
                                trim(target_name)
                    return
                end if
                target_addr = null_ptr_operand(context)
            else
                target_addr = context%symbols(tgt)%element_address
            end if
        else
            if (.not. context%symbols(tgt)%has_address) then
                error_msg = 'pointer assignment source has no storage: '// &
                            trim(target_name)
                return
            end if
            if (context%symbols(tgt)%is_array .or. context%symbols(tgt)%is_derived) then
                error_msg = 'pointer component target must be a scalar of '// &
                            'intrinsic type: '//trim(target_name)
                return
            end if
            if (context%symbols(tgt)%value_kind /= &
                component_kind(context, type_index, comp_index)) then
                error_msg = 'pointer and target scalar kinds do not match'
                return
            end if
            target_addr = context%symbols(tgt)%address
        end if
        call set_empty(error_msg)
    end subroutine data_pointer_target_address

    subroutine nullify_data_pointer_component(arena, access_index, context, &
                                              error_msg)
        ! nullify(obj%p): clear the component's inline pointer slot (#401).
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: access_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: slot_addr

        call data_component_slot_address(arena, access_index, context, slot_addr, &
                                         error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_ptr_store(context%session, null_ptr_operand(context), &
                                 slot_addr, error_msg)) return
        call set_empty(error_msg)
    end subroutine nullify_data_pointer_component

    subroutine lower_data_pointer_component_associated(arena, arg_indices, &
                                                       context, value, error_msg)
        ! associated(obj%p) / associated(obj%p, t): run-time comparison of the
        ! stored target address (#401). F2018 16.9.20: a disassociated pointer
        ! is never associated with a target, so the identity test is ANDed with
        ! the non-null test.
        type(ast_arena_t), intent(in) :: arena
        integer, allocatable, intent(in) :: arg_indices(:)
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: slot_addr, ptr_value, target_addr, nonnull, same

        call data_component_slot_address(arena, arg_indices(1), context, &
                                         slot_addr, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_ptr_load(context%session, slot_addr, ptr_value, error_msg)) &
            return
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_NE, ptr_value, &
                null_ptr_operand(context), nonnull, error_msg)) return
        if (size(arg_indices) == 1) then
            value = nonnull
            call set_empty(error_msg)
            return
        end if
        call data_pointer_target_address(arena, arg_indices(1), arg_indices(2), &
                                         context, target_addr, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_EQ, ptr_value, &
                target_addr, same, error_msg)) return
        if (.not. emit_i32_binary(context%session, LR_OP_AND, nonnull, same, &
                                  value, error_msg)) return
        call set_empty(error_msg)
    end subroutine lower_data_pointer_component_associated

    subroutine lower_proc_pointer_associated(arena, arg_indices, context, value, &
                                             error_msg)
        ! associated(obj%p) / associated(obj%p, target) with a run-time callee
        ! address comparison (#372). F2018 16.9.20: a disassociated pointer is
        ! never associated with a target, so the identity test is ANDed with the
        ! non-null test.
        type(ast_arena_t), intent(in) :: arena
        integer, allocatable, intent(in) :: arg_indices(:)
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: ptr_value, target_value, nonnull, same

        call load_proc_pointer_operand(arena, arg_indices(1), context, ptr_value, &
                                       error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_NE, ptr_value, &
                null_ptr_operand(context), nonnull, error_msg)) return
        if (size(arg_indices) == 1) then
            value = nonnull
            call set_empty(error_msg)
            return
        end if
        call proc_target_address(arena, arg_indices(2), context, target_value, &
                                 error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_EQ, ptr_value, &
                target_value, same, error_msg)) return
        if (.not. emit_i32_binary(context%session, LR_OP_AND, nonnull, same, &
                                  value, error_msg)) return
        call set_empty(error_msg)
    end subroutine lower_proc_pointer_associated
