    subroutine validate_allocate_status_specifiers(arena, node, context, error_msg)
        ! #382: an ALLOCATE stat= specifier must name a scalar INTEGER
        ! variable and errmsg= must name a scalar default CHARACTER variable.
        ! Only resolved plain identifiers carry enough information here; other
        ! forms (array elements, components) stay with the existing handling.
        type(ast_arena_t), intent(in) :: arena
        type(allocate_statement_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: sym
        logical :: is_scalar

        call set_empty(error_msg)

        call resolve_status_specifier_symbol(arena, node%stat_var_index, &
                                             context, sym, is_scalar)
        if (sym > 0) then
            if (.not. is_scalar) then
                call allocate_status_error(node, 'stat', &
                    'a scalar INTEGER variable', error_msg)
                return
            end if
            if (context%symbols(sym)%value_kind /= VALUE_I32 .and. &
                context%symbols(sym)%value_kind /= VALUE_I64) then
                call allocate_status_error(node, 'stat', &
                    'a scalar INTEGER variable', error_msg)
                return
            end if
        end if

        call resolve_status_specifier_symbol(arena, node%errmsg_var_index, &
                                             context, sym, is_scalar)
        if (sym > 0) then
            if (.not. is_scalar) then
                call allocate_status_error(node, 'errmsg', &
                    'a scalar default CHARACTER variable', error_msg)
                return
            end if
            if (context%symbols(sym)%value_kind /= VALUE_CHARACTER) then
                call allocate_status_error(node, 'errmsg', &
                    'a scalar default CHARACTER variable', error_msg)
                return
            end if
        end if
    end subroutine validate_allocate_status_specifiers

    subroutine resolve_status_specifier_symbol(arena, node_index, context, &
                                               symbol_index, is_scalar)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        integer, intent(out) :: symbol_index
        logical, intent(out) :: is_scalar
        character(len=:), allocatable :: name, local_error

        symbol_index = 0
        is_scalar = .true.
        if (node_index <= 0) return
        if (.not. is_identifier(arena, node_index)) return
        call get_identifier_name(arena, node_index, name, local_error)
        if (len_trim(local_error) > 0) return
        symbol_index = find_symbol_compat(context, name)
        if (symbol_index <= 0) return
        if (context%symbols(symbol_index)%is_array) is_scalar = .false.
        if (context%symbols(symbol_index)%array_rank > 0) is_scalar = .false.
    end subroutine resolve_status_specifier_symbol

    subroutine allocate_status_error(node, specifier, requirement, error_msg)
        type(allocate_statement_node), intent(in) :: node
        character(len=*), intent(in) :: specifier
        character(len=*), intent(in) :: requirement
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=64) :: location

        write (location, '(" at line ",I0,", column ",I0)') node%line, node%column
        error_msg = 'ALLOCATE '//specifier//'= must be '//requirement// &
                    trim(location)
    end subroutine allocate_status_error

    subroutine lower_allocatable_declaration(node, context, error_msg, &
                                             value_kind_override)
        ! integer/real/logical/character, allocatable :: a(:) or a(:,:)
        type(declaration_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer, intent(in), optional :: value_kind_override
        integer :: i, rank, value_kind, character_length, symbol_index

        if (present(value_kind_override)) then
            value_kind = value_kind_override
            call set_empty(error_msg)
        else
            call declaration_value_kind(node, value_kind, error_msg, context)
            if (len_trim(error_msg) > 0) return
        end if
        if (value_kind /= VALUE_I32 .and. value_kind /= VALUE_F32 .and. &
            value_kind /= VALUE_F64 .and. value_kind /= VALUE_LOGICAL .and. &
            value_kind /= VALUE_CHARACTER .and. value_kind /= VALUE_C4 .and. &
            value_kind /= VALUE_C8) then
            call unsupported_feature_error('allocatable declaration', &
                node%line, node%column, &
                'direct LIRIC session supports integer, real, logical, complex, '// &
                'and fixed-length character allocatable arrays only', error_msg)
            return
        end if

        character_length = 0
        if (value_kind == VALUE_CHARACTER) then
            call declaration_character_length(context, node, character_length, &
                                              error_msg)
            if (len_trim(error_msg) > 0) return
            if (character_length <= 0) then
                call unsupported_feature_error('allocatable declaration', &
                    node%line, node%column, &
                    'direct LIRIC session supports a fixed-length '// &
                    'character(len=N), allocatable array; a deferred-length '// &
                    'character(len=:) array element type is not supported', &
                    error_msg)
                return
            end if
        end if

        rank = 1
        if (allocated(node%dimension_indices)) then
            rank = size(node%dimension_indices)
            if (rank < 1 .or. rank > 2) then
                call unsupported_feature_error('allocatable declaration', &
                    node%line, node%column, &
                    'direct LIRIC session supports rank-1 and rank-2 '// &
                    'allocatables', error_msg)
                return
            end if
            if (rank == 2 .and. value_kind == VALUE_CHARACTER) then
                call unsupported_feature_error('allocatable declaration', &
                    node%line, node%column, &
                    'direct LIRIC session supports rank-1 character '// &
                    'allocatable arrays only', error_msg)
                return
            end if
        end if

        if (node%is_multi_declaration .and. allocated(node%var_names)) then
            do i = 1, size(node%var_names)
                call declare_allocatable(context, node%var_names(i), rank, &
                                         value_kind, error_msg)
                if (len_trim(error_msg) > 0) return
                if (value_kind == VALUE_CHARACTER) then
                    symbol_index = find_symbol_compat(context, node%var_names(i))
                    if (symbol_index > 0) &
                        context%symbols(symbol_index)%character_length = &
                            character_length
                end if
            end do
        else if (allocated(node%var_name)) then
            call declare_allocatable(context, node%var_name, rank, value_kind, &
                                     error_msg)
            if (len_trim(error_msg) == 0 .and. value_kind == VALUE_CHARACTER) then
                symbol_index = find_symbol_compat(context, node%var_name)
                if (symbol_index > 0) &
                    context%symbols(symbol_index)%character_length = &
                        character_length
            end if
        else
            error_msg = 'allocatable declaration did not expose a variable name'
        end if
    end subroutine lower_allocatable_declaration

    subroutine lower_allocate_statement(arena, node, context, error_msg)
        ! allocate(a(N)) or allocate(a(M,N)) for one integer allocatable.
        ! N and M may be runtime integer expressions. Multi-variable and
        ! lower:upper forms are rejected.
        type(ast_arena_t), intent(in) :: arena
        type(allocate_statement_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: name
        integer :: size_index1, size_index2, symbol_index
        logical :: handled, no_shape

        if (.not. allocated(node%var_indices) .or. size(node%var_indices) < 1) then
            call unsupported_feature_error('allocate statement', node%line, &
                node%column, 'allocate statement has no target variable', &
                error_msg)
            return
        end if
        if (size(node%var_indices) > 1) then
            call lower_multi_var_allocate(arena, node, context, error_msg)
            return
        end if
        no_shape = .not. allocated(node%shape_indices)
        if (allocated(node%shape_indices)) no_shape = size(node%shape_indices) == 0

        if (allocated(node%type_spec)) then
            call lower_allocate_deferred_char_array(arena, node, context, &
                                                    handled, error_msg)
            if (len_trim(error_msg) > 0) return
            if (handled) then
                call set_alloc_stat_zero(arena, node, context, error_msg)
                return
            end if
        end if

        if (node%mold_expr_index <= 0 .and. node%source_expr_index <= 0) then
            call lower_allocate_array_component(arena, node%var_indices(1), &
                                                context, handled, error_msg)
            if (len_trim(error_msg) > 0) return
            if (handled) return
            call lower_allocate_scalar_component(arena, node%var_indices(1), &
                                                 context, handled, error_msg)
            if (len_trim(error_msg) > 0) return
            if (handled) return
        end if

        if (node%mold_expr_index > 0 .or. node%source_expr_index > 0) then
            call lower_allocate_mold_source(arena, node, context, error_msg)
            return
        end if

        ! allocate(<type-spec> :: x): no shape, the element type/length lives in
        ! node%type_spec as a string. Route by the type-spec text.
        if (allocated(node%type_spec)) then
            if (len_trim(node%type_spec) > 0 .and. &
                is_identifier(arena, node%var_indices(1)) .and. &
                no_shape) then
                call lower_allocate_type_spec(arena, node, context, error_msg)
                return
            end if
        end if

        ! allocate(x) for a bare scalar allocatable: no shape at all.
        if (is_identifier(arena, node%var_indices(1)) .and. &
            no_shape) then
            call get_identifier_name(arena, node%var_indices(1), name, error_msg)
            if (len_trim(error_msg) > 0) return
            symbol_index = find_symbol_compat(context, name)
            if (symbol_index > 0) then
                if (context%symbols(symbol_index)%is_allocatable .and. &
                    context%symbols(symbol_index)%array_rank == 0) then
                    call lower_allocate_scalar(symbol_index, context, error_msg)
                    return
                end if
                ! allocate(p) for a scalar POINTER: give the pointer fresh
                ! heap storage and mark it associated (F2018 9.7.1.1).
                if (context%symbols(symbol_index)%is_pointer .and. &
                    .not. context%symbols(symbol_index)%is_array) then
                    if (context%symbols(symbol_index)%is_class_pointer) then
                        if (context%symbols(symbol_index)%derived_type_index > 0) then
                            if (len_trim(context%derived_types(context%symbols( &
                                symbol_index)%derived_type_index)%final_proc_name) > 0) then
                                error_msg = 'class pointer allocation of a finalizable '// &
                                    'type has unsupported ownership/finalization'
                                return
                            end if
                        end if
                    end if
                    call lower_allocate_scalar(symbol_index, context, error_msg)
                    if (len_trim(error_msg) == 0) &
                        context%symbols(symbol_index)%is_associated = .true.
                    return
                end if
            end if
        end if

        call allocate_target(arena, node, name, size_index1, size_index2, error_msg)
        if (len_trim(error_msg) > 0) then
            call unsupported_feature_error('allocate statement', node%line, &
                node%column, trim(error_msg), error_msg)
            return
        end if

        symbol_index = find_symbol_compat(context, name)
        if (symbol_index <= 0) then
            call unsupported_feature_error('allocate statement', node%line, &
                node%column, 'allocate target is not declared: '//trim(name), &
                error_msg)
            return
        end if
        if (.not. context%symbols(symbol_index)%is_allocatable) then
            call unsupported_feature_error('allocate statement', node%line, &
                node%column, 'allocate target is not an allocatable array: '// &
                trim(name), error_msg)
            return
        end if

        if (context%symbols(symbol_index)%is_derived) then
            call lower_allocate_derived_array(arena, size_index1, size_index2, &
                                               symbol_index, context, error_msg)
        else if (context%symbols(symbol_index)%array_rank == 2) then
            call lower_allocate_i32_2d(arena, size_index1, size_index2, &
                                       symbol_index, context, error_msg)
            if (len_trim(error_msg) == 0) &
                call record_allocatable_static_size_2d(arena, size_index1, &
                                                    size_index2, symbol_index, &
                                                    context)
        else
            call lower_allocate_i32_1d(arena, size_index1, symbol_index, &
                                       context, error_msg)
            if (len_trim(error_msg) == 0) &
                call record_allocatable_static_size(arena, size_index1, &
                                                    symbol_index, context)
            if (len_trim(error_msg) == 0 .and. &
                context%symbols(symbol_index)%value_kind == VALUE_CHARACTER) &
                call init_allocatable_character_blanks(context, symbol_index, &
                                                       error_msg)
        end if
        if (len_trim(error_msg) == 0) &
            call set_alloc_stat_zero(arena, node, context, error_msg)
    end subroutine lower_allocate_statement

    subroutine lower_multi_var_allocate(arena, node, context, error_msg)
        ! allocate(a(N), b(N), c(M,K), ...): allocate each target independently.
        ! Each target carries its own shape in its call_or_subscript subscripts.
        ! The shared type-spec / mold= / source= forms are single-target only and
        ! stay unsupported here (they previously errored for any multi-target
        ! allocate, so declining them regresses nothing).
        type(ast_arena_t), intent(in) :: arena
        type(allocate_statement_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: k
        logical :: handled

        call set_empty(error_msg)
        if (node%mold_expr_index > 0 .or. node%source_expr_index > 0) then
            call unsupported_feature_error('allocate statement', node%line, &
                node%column, 'multi-variable allocate with mold= or source= is '// &
                'not supported', error_msg)
            return
        end if

        do k = 1, size(node%var_indices)
            call lower_one_allocate_var(arena, node%var_indices(k), context, &
                                        handled, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. handled) then
                call unsupported_feature_error('allocate statement', node%line, &
                    node%column, 'unsupported target in multi-variable allocate', &
                    error_msg)
                return
            end if
        end do
        call set_alloc_stat_zero(arena, node, context, error_msg)
    end subroutine lower_multi_var_allocate

    subroutine lower_one_allocate_var(arena, vi, context, handled, error_msg)
        ! Allocate a single target within a multi-variable allocate. Handles an
        ! allocatable array/scalar component, a bare scalar allocatable, and the
        ! rank-1/rank-2 integer/real/logical/character array a(N)[,(M)] form.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: vi
        type(lowering_context_t), intent(inout) :: context
        logical, intent(out) :: handled
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: name
        integer :: symbol_index, size_index1, size_index2

        handled = .false.
        call set_empty(error_msg)

        call lower_allocate_array_component(arena, vi, context, handled, error_msg)
        if (len_trim(error_msg) > 0) return
        if (handled) return
        call lower_allocate_scalar_component(arena, vi, context, handled, error_msg)
        if (len_trim(error_msg) > 0) return
        if (handled) return

        if (is_identifier(arena, vi)) then
            call get_identifier_name(arena, vi, name, error_msg)
            if (len_trim(error_msg) > 0) return
            symbol_index = find_symbol_compat(context, name)
            if (symbol_index <= 0) return
            if (context%symbols(symbol_index)%is_allocatable .and. &
                context%symbols(symbol_index)%array_rank == 0) then
                call lower_allocate_scalar(symbol_index, context, error_msg)
                handled = len_trim(error_msg) == 0
            end if
            return
        end if

        call allocate_var_target(arena, vi, name, size_index1, size_index2, &
                                 error_msg)
        if (len_trim(error_msg) > 0) then
            call set_empty(error_msg)
            return
        end if
        symbol_index = find_symbol_compat(context, name)
        if (symbol_index <= 0) return
        if (.not. context%symbols(symbol_index)%is_allocatable) return

        if (context%symbols(symbol_index)%array_rank == 2) then
            call lower_allocate_i32_2d(arena, size_index1, size_index2, &
                                       symbol_index, context, error_msg)
            if (len_trim(error_msg) == 0) &
                call record_allocatable_static_size_2d(arena, size_index1, &
                                                    size_index2, symbol_index, &
                                                    context)
        else
            call lower_allocate_i32_1d(arena, size_index1, symbol_index, &
                                       context, error_msg)
            if (len_trim(error_msg) == 0) &
                call record_allocatable_static_size(arena, size_index1, &
                                                    symbol_index, context)
            if (len_trim(error_msg) == 0 .and. &
                context%symbols(symbol_index)%value_kind == VALUE_CHARACTER) &
                call init_allocatable_character_blanks(context, symbol_index, &
                                                       error_msg)
        end if
        handled = len_trim(error_msg) == 0
    end subroutine lower_one_allocate_var

    subroutine allocate_var_target(arena, vi, name, size_index1, size_index2, &
                                   error_msg)
        ! Extract the array name and rank-1/rank-2 shape expressions from a single
        ! allocate target node a(N) or a(M,K) (a call_or_subscript). size_index2
        ! is 0 for rank-1.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: vi
        character(len=:), allocatable, intent(out) :: name
        integer, intent(out) :: size_index1, size_index2
        character(len=:), allocatable, intent(out) :: error_msg

        call set_empty(error_msg)
        size_index1 = 0
        size_index2 = 0
        if (.not. node_exists(arena, vi)) then
            error_msg = 'allocate target is not an AST node'
            return
        end if
        select type (v => arena%entries(vi)%node)
        type is (call_or_subscript_node)
            if (.not. allocated(v%name) .or. .not. allocated(v%arg_indices)) then
                error_msg = 'allocate target must name an array with a size'
                return
            end if
            if (size(v%arg_indices) < 1 .or. size(v%arg_indices) > 2) then
                error_msg = 'only rank-1 and rank-2 allocate is supported'
                return
            end if
            name = v%name
            size_index1 = v%arg_indices(1)
            if (size(v%arg_indices) == 2) size_index2 = v%arg_indices(2)
        class default
            error_msg = 'allocate target must be a scalar array name'
        end select
    end subroutine allocate_var_target

    subroutine set_alloc_stat_zero(arena, node, context, error_msg)
        ! A successful allocate sets an optional stat= integer variable to 0.
        ! Best-effort: an unresolved or non-integer stat target is left untouched
        ! rather than raising an error.
        use, intrinsic :: iso_c_binding, only: c_int64_t
        type(ast_arena_t), intent(in) :: arena
        type(allocate_statement_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: name
        integer :: sym

        call set_empty(error_msg)
        if (node%stat_var_index <= 0) return
        if (.not. is_identifier(arena, node%stat_var_index)) return
        call get_identifier_name(arena, node%stat_var_index, name, error_msg)
        if (len_trim(error_msg) > 0) then
            call set_empty(error_msg)
            return
        end if
        sym = find_symbol_compat(context, name)
        if (sym <= 0) return
        if (context%symbols(sym)%value_kind /= VALUE_I32) return
        call assign_i32_to_symbol(context, sym, &
            i32_immediate(context%session, 0_c_int64_t), error_msg)
    end subroutine set_alloc_stat_zero

    subroutine lower_allocate_type_spec(arena, node, context, error_msg)
        ! allocate(<type-spec> :: x): the element type/length comes from the
        ! type-spec STRING, not a shape expression. A character type-spec sizes
        ! a deferred-length character scalar; a derived type-spec allocates one
        ! instance of a scalar allocatable derived variable.
        type(ast_arena_t), intent(in) :: arena
        type(allocate_statement_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: name, spec_lc
        integer :: symbol_index, spec_type_index

        call set_empty(error_msg)
        spec_type_index = 0
        if (.not. node_exists(arena, node%var_indices(1)) .or. &
            .not. is_identifier(arena, node%var_indices(1))) then
            call unsupported_feature_error('allocate statement', node%line, &
                node%column, 'type-spec allocate target must be a scalar name', &
                error_msg)
            return
        end if
        call get_identifier_name(arena, node%var_indices(1), name, error_msg)
        if (len_trim(error_msg) > 0) return

        symbol_index = find_symbol_compat(context, name)
        if (symbol_index <= 0) then
            call unsupported_feature_error('allocate statement', node%line, &
                node%column, 'allocate target is not declared: '//trim(name), &
                error_msg)
            return
        end if

        spec_lc = lowercase_text(trim(adjustl(node%type_spec)))
        if (len(spec_lc) >= 9) then
            if (spec_lc(1:9) == 'character') then
                call lower_allocate_char_type_spec(context, node, symbol_index, &
                                                   name, error_msg)
                return
            end if
        end if

        ! Otherwise treat the type-spec as a derived type name: allocate(t :: x)
        ! mallocs one instance and default-inits it, identical to the bare
        ! allocate(x) form for a scalar allocatable derived variable.
        if (context%symbols(symbol_index)%is_derived .and. &
            (context%symbols(symbol_index)%is_allocatable .or. &
             context%symbols(symbol_index)%is_class_pointer) .and. &
            context%symbols(symbol_index)%array_rank == 0) then
            ! allocate(t :: x) names the dynamic type explicitly. For a class
            ! allocatable it may be an extension of the declared type, in which
            ! case that extension's layout is allocated and recorded (#421).
            spec_type_index = find_derived_type(context, trim(spec_lc))
            if (spec_type_index > 0) then
                if (.not. derived_type_extends(context, spec_type_index, &
                        context%symbols(symbol_index)%derived_type_index)) then
                    error_msg = 'allocate type-spec "'//trim(spec_lc)// &
                        '" is not type compatible with "'//trim(name)// &
                        '" of declared type "'// &
                        trim(context%derived_types(context%symbols( &
                            symbol_index)%derived_type_index)%name)//'"'
                    return
                end if
            end if
            if (context%symbols(symbol_index)%is_class_pointer) then
                if (spec_type_index <= 0) then
                    error_msg = 'class pointer ALLOCATE requires a known derived '// &
                        'type-spec'
                    return
                end if
                if (len_trim(context%derived_types(spec_type_index)%final_proc_name) > 0) then
                    error_msg = 'class pointer allocation of a finalizable type has '// &
                        'unsupported ownership/finalization'
                    return
                end if
                call lower_allocate_scalar_derived(symbol_index, context, error_msg, &
                                                    spec_type_index)
                if (len_trim(error_msg) == 0) &
                    context%symbols(symbol_index)%is_associated = .true.
                return
            end if
            call lower_allocate_scalar_derived(symbol_index, context, error_msg, &
                                               spec_type_index)
            return
        end if
        if (context%symbols(symbol_index)%is_derived) then
            call set_empty(error_msg)
            return
        end if
        call unsupported_feature_error('allocate statement', node%line, &
            node%column, 'unsupported type-spec allocate for: '//trim(name), &
            error_msg)
    end subroutine lower_allocate_type_spec

    subroutine lower_allocate_char_type_spec(context, node, symbol_index, name, &
                                             error_msg)
        ! allocate(character(len=<expr>) :: str): malloc len+1 bytes (room for a
        ! trailing NUL so %s prints stay bounded), store the data pointer and the
        ! length into str's deferred {data,len} descriptor. Mirrors the malloc /
        ! store path in lower_deferred_concat_assignment.
        use, intrinsic :: iso_c_binding, only: c_int64_t
        type(lowering_context_t), intent(inout) :: context
        type(allocate_statement_node), intent(in) :: node
        integer, intent(in) :: symbol_index
        character(len=*), intent(in) :: name
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: len_i32, len_i64, bytes_i64, data_ptr
        character(len=:), allocatable :: len_expr

        if (.not. context%symbols(symbol_index)%is_deferred_character) then
            call unsupported_feature_error('allocate statement', node%line, &
                node%column, 'character type-spec allocate requires a '// &
                'character(len=:), allocatable target: '//trim(name), error_msg)
            return
        end if

        call extract_type_spec_len(node%type_spec, len_expr, error_msg)
        if (len_trim(error_msg) > 0) then
            call unsupported_feature_error('allocate statement', node%line, &
                node%column, trim(error_msg), error_msg)
            return
        end if

        call lower_type_spec_len_value(context, len_expr, len_i32, error_msg)
        if (len_trim(error_msg) > 0) then
            call unsupported_feature_error('allocate statement', node%line, &
                node%column, trim(error_msg), error_msg)
            return
        end if

        ! bytes = len + 1 (trailing NUL), computed in i64.
        if (.not. emit_liric_i32_to_i64(context%session, len_i32, len_i64, &
                error_msg)) return
        if (.not. emit_i64_binary(context%session, LR_OP_ADD, len_i64, &
                i64_immediate(context%session, 1_c_int64_t), bytes_i64, &
                error_msg)) return
        ! An explicit allocate over an already-allocated deferred character is
        ! an error in standard Fortran; releasing first keeps the descriptor
        ! from silently leaking its former owned block if it happens anyway.
        call release_owned_character_storage(context, symbol_index, error_msg)
        if (len_trim(error_msg) > 0) return

        if (.not. emit_malloc(context%session, bytes_i64, data_ptr, error_msg)) &
            return

        if (.not. emit_ptr_store(context%session, data_ptr, &
                context%symbols(symbol_index)%deferred_data, error_msg)) return
        if (.not. emit_i64_store(context%session, len_i64, &
                context%symbols(symbol_index)%deferred_length, error_msg)) return
        ! capacity is the requested length: the extra byte is the trailing NUL
        ! the printers rely on, not usable character storage.
        call set_character_storage(context, symbol_index, len_i64, &
                                   LOWERING_CHARACTER_STORAGE_OWNED, error_msg)
        if (len_trim(error_msg) > 0) return
        context%symbols(symbol_index)%value = data_ptr
        context%symbols(symbol_index)%has_character_value = .true.
        call set_empty(error_msg)
    end subroutine lower_allocate_char_type_spec

    subroutine extract_type_spec_len(type_spec, len_expr, error_msg)
        ! Pull the length expression out of a "character(len=<expr>)" or
        ! "character(<expr>)" type-spec string, balancing parentheses so a nested
        ! call like len(input) survives intact.
        character(len=*), intent(in) :: type_spec
        character(len=:), allocatable, intent(out) :: len_expr
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: spec_lc
        integer :: open_paren, start, depth, i, content_end

        call set_empty(error_msg)
        len_expr = ''
        spec_lc = lowercase_text(trim(adjustl(type_spec)))
        open_paren = index(spec_lc, '(')
        if (open_paren <= 0) then
            error_msg = 'character type-spec without a length'
            return
        end if

        ! Find the inner content up to the matching close paren.
        start = open_paren + 1
        depth = 1
        content_end = 0
        do i = start, len(spec_lc)
            if (spec_lc(i:i) == '(') then
                depth = depth + 1
            else if (spec_lc(i:i) == ')') then
                depth = depth - 1
                if (depth == 0) then
                    content_end = i - 1
                    exit
                end if
            end if
        end do
        if (content_end < start) then
            error_msg = 'malformed character type-spec length'
            return
        end if

        ! Drop a leading "len=" if present; keep the original (cased) text so a
        ! variable name resolves with its real spelling.
        block
            character(len=:), allocatable :: inner, inner_lc
            integer :: eq_pos
            inner = trim(adjustl(type_spec(open_paren + 1:open_paren + 1 + &
                                            (content_end - start))))
            inner_lc = lowercase_text(inner)
            if (len(inner_lc) >= 4) then
                if (inner_lc(1:4) == 'len=') then
                    inner = trim(adjustl(inner(5:)))
                end if
            end if
            len_expr = inner
        end block
        if (len_trim(len_expr) == 0) error_msg = 'empty character type-spec length'
    end subroutine extract_type_spec_len

    subroutine lower_type_spec_len_value(context, len_expr, value, error_msg)
        ! Lower a type-spec length expression string to an i32 operand. Supports
        ! an integer literal, a bare integer variable, and len(<name>) of a
        ! character symbol. More complex expressions stay unsupported here.
        use, intrinsic :: iso_c_binding, only: c_int64_t
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: len_expr
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: expr, expr_lc, inner_name
        type(lr_operand_desc_t) :: data_ptr
        integer(c_int64_t) :: literal
        integer :: symbol_index, ios, open_paren, close_paren

        call set_empty(error_msg)
        expr = trim(adjustl(len_expr))
        expr_lc = lowercase_text(expr)

        ! Integer literal.
        if (verify(trim(expr), '0123456789') == 0 .and. len_trim(expr) > 0) then
            read (expr, *, iostat=ios) literal
            if (ios /= 0) then
                error_msg = 'invalid integer length in character type-spec'
                return
            end if
            value = i32_immediate(context%session, literal)
            return
        end if

        ! len(<name>) of a character symbol.
        if (len(expr_lc) > 4) then
            if (expr_lc(1:4) == 'len(') then
                open_paren = index(expr, '(')
                close_paren = index(expr, ')', back=.true.)
                if (close_paren <= open_paren + 1) then
                    error_msg = 'malformed len() in character type-spec'
                    return
                end if
                inner_name = trim(adjustl(expr(open_paren + 1:close_paren - 1)))
                symbol_index = find_symbol_compat(context, inner_name)
                if (symbol_index <= 0 .or. &
                    context%symbols(max(symbol_index, 1))%value_kind /= &
                    VALUE_CHARACTER) then
                    error_msg = 'len() argument is not a character variable: '// &
                                trim(inner_name)
                    return
                end if
                call char_length_operands(context, symbol_index, data_ptr, &
                                          value, error_msg)
                return
            end if
        end if

        ! Bare integer variable.
        if (verify(trim(expr), &
            'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_') &
            == 0) then
            symbol_index = find_symbol_compat(context, trim(expr))
            if (symbol_index <= 0) then
                error_msg = 'length variable is not declared: '//trim(expr)
                return
            end if
            if (context%symbols(symbol_index)%value_kind /= VALUE_I32) then
                error_msg = 'length variable is not integer: '//trim(expr)
                return
            end if
            if (context%symbols(symbol_index)%has_address .and. &
                context%symbols(symbol_index)%is_reference) then
                if (.not. emit_i32_load(context%session, &
                    context%symbols(symbol_index)%address, value, error_msg)) &
                    return
            else
                value = context%symbols(symbol_index)%value
            end if
            return
        end if

        error_msg = 'unsupported character type-spec length expression: '// &
                    trim(expr)
    end subroutine lower_type_spec_len_value

    subroutine lower_allocate_mold_source(arena, node, context, error_msg)
        ! allocate(b, mold=a) / allocate(c, source=a): give the target the same
        ! rank-1 extent as the mold/source allocatable. SOURCE also copies the
        ! element bytes; MOLD leaves contents undefined. The source extent must
        ! be a compile-time constant (allocatable_static_size), which covers the
        ! straight-line allocate(a(N)); a=... case in the corpus.
        type(ast_arena_t), intent(in) :: arena
        type(allocate_statement_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: target_name, source_name
        integer :: target_sym, source_sym, source_expr, n
        logical :: copy_values
        type(lr_operand_desc_t) :: n_i32, dst_desc, src_desc, dst_data, src_data

        copy_values = node%source_expr_index > 0
        if (copy_values) then
            source_expr = node%source_expr_index
        else
            source_expr = node%mold_expr_index
        end if

        if (.not. is_identifier(arena, node%var_indices(1))) then
            error_msg = 'mold/source allocate target must be a scalar array name'
            return
        end if
        call get_identifier_name(arena, node%var_indices(1), target_name, error_msg)
        if (len_trim(error_msg) > 0) return
        target_sym = find_symbol_compat(context, target_name)
        if (target_sym > 0) then
            if (context%symbols(target_sym)%is_allocatable .and. &
                context%symbols(target_sym)%array_rank == 0) then
                call lower_scalar_allocate_mold_source(arena, node, target_sym, &
                    context, error_msg)
                return
            end if
        end if
        if (.not. is_identifier(arena, source_expr)) then
            if (target_sym <= 0 .or. .not. context%symbols(target_sym)%is_allocatable) &
                    then
                error_msg = 'mold/source allocate references an undeclared variable'
                return
            end if
            call lower_allocate_array_expression_source(arena, source_expr, &
                target_sym, context, copy_values, error_msg)
            return
        end if
        if (.not. is_identifier(arena, source_expr)) then
            error_msg = 'mold/source must name an allocatable array'
            return
        end if
        call get_identifier_name(arena, source_expr, source_name, error_msg)
        if (len_trim(error_msg) > 0) return

        target_sym = find_symbol_compat(context, target_name)
        source_sym = find_symbol_compat(context, source_name)
        if (target_sym <= 0 .or. source_sym <= 0) then
            error_msg = 'mold/source allocate references an undeclared variable'
            return
        end if
        if (.not. context%symbols(target_sym)%is_allocatable .or. &
            .not. context%symbols(source_sym)%is_allocatable) then
            error_msg = 'mold/source allocate requires allocatable operands'
            return
        end if
        if (context%symbols(source_sym)%array_rank == 2) then
            error_msg = 'mold/source allocate supports only rank-1 arrays'
            return
        end if
        n = context%symbols(source_sym)%allocatable_static_size
        if (n <= 0) then
            error_msg = 'mold/source allocate requires a constant source extent'
            return
        end if

        n_i32 = i32_immediate(context%session, int(n, c_int64_t))
        call lower_allocate_i32_1d_operand(n_i32, target_sym, context, error_msg)
        if (len_trim(error_msg) > 0) return
        context%symbols(target_sym)%allocatable_static_size = n

        if (.not. copy_values) then
            call set_empty(error_msg)
            return
        end if

        dst_desc = context%symbols(target_sym)%allocatable_descriptor_address
        src_desc = context%symbols(source_sym)%allocatable_descriptor_address
        if (.not. emit_ptr_load(context%session, dst_desc, dst_data, error_msg)) &
            return
        if (.not. emit_ptr_load(context%session, src_desc, src_data, error_msg)) &
            return
        if (.not. emit_memcpy(context%session, dst_data, src_data, &
                i64_immediate(context%session, int(n, c_int64_t) * &
                    allocatable_elem_size(context%symbols(source_sym)%value_kind)), &
                error_msg)) return
        call set_empty(error_msg)
    end subroutine lower_allocate_mold_source

    subroutine lower_allocate_array_expression_source(arena, source_expr, &
                                                       target_sym, context, &
                                                       copy_values, error_msg)
        !! SOURCE= may be an array-valued expression, not only an allocatable
        !! array name.  Infer its fixed extent, allocate the target, and reuse
        !! the ordinary whole-array evaluator to materialise the values.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: source_expr, target_sym
        type(lowering_context_t), intent(inout) :: context
        logical, intent(in) :: copy_values
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: extent
        logical :: known
        type(array_expr_plan_t) :: plan
        type(lr_operand_desc_t) :: n_i32

        call set_empty(error_msg)
        if (context%symbols(target_sym)%array_rank == 2) then
            error_msg = 'mold/source allocate supports only rank-1 arrays'
            return
        end if
        call array_expression_static_extent(arena, source_expr, context, extent, &
                                            known)
        if (.not. known .or. extent <= 0) then
            error_msg = 'mold/source allocate requires a constant source extent'
            return
        end if
        n_i32 = i32_immediate(context%session, int(extent, c_int64_t))
        call lower_allocate_i32_1d_operand(n_i32, target_sym, context, error_msg)
        if (len_trim(error_msg) > 0) return
        context%symbols(target_sym)%allocatable_static_size = extent
        if (.not. copy_values) return

        call build_array_expression(context, target_sym, source_expr, plan, &
                                    error_msg)
        if (len_trim(error_msg) > 0) return
        call emit_array_expression_assignment(arena, plan, context, error_msg)
    end subroutine lower_allocate_array_expression_source

    recursive subroutine array_expression_static_extent(arena, expr_index, &
                                                         context, extent, known)
        !! Recover the element count of the small fixed-shape expressions that
        !! SOURCE= can evaluate here: constructors, identifiers, elemental
        !! calls, and conforming binary expressions.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: expr_index
        type(lowering_context_t), intent(in) :: context
        integer, intent(out) :: extent
        logical, intent(out) :: known
        integer :: i, left_index, right_index, line, column, sym
        integer :: left_extent, right_extent
        integer, allocatable :: flat(:)
        character(len=:), allocatable :: op, name, err

        extent = 0
        known = .false.
        if (.not. node_exists(arena, expr_index)) return

        select type (node => arena%entries(expr_index)%node)
        type is (array_literal_node)
            if (.not. allocated(node%element_indices)) return
            allocate (flat(0))
            call flatten_constructor_elements(arena, node%element_indices, flat, err)
            if (len_trim(err) > 0) return
            extent = size(flat)
            known = .true.
            return
        type is (call_or_subscript_node)
            if (.not. allocated(node%arg_indices)) return
            do i = 1, size(node%arg_indices)
                call array_expression_static_extent(arena, node%arg_indices(i), &
                    context, extent, known)
                if (known) return
            end do
            return
        end select

        if (is_identifier(arena, expr_index)) then
            call get_identifier_name(arena, expr_index, name, err)
            if (len_trim(err) > 0) return
            sym = find_symbol_compat(context, name)
            if (sym <= 0) return
            if (context%symbols(sym)%is_allocatable) then
                extent = context%symbols(sym)%allocatable_static_size
            else if (context%symbols(sym)%array_rank == 1) then
                extent = context%symbols(sym)%array_dim_sizes(1)
                if (extent <= 0) extent = context%symbols(sym)%array_size
            end if
            known = extent > 0
            return
        end if

        if (.not. is_binary_op(arena, expr_index)) return
        call get_binary_op_info(arena, expr_index, op, left_index, right_index, &
                                line, column, err)
        if (len_trim(err) > 0) return
        call array_expression_static_extent(arena, left_index, context, &
                                            left_extent, known)
        if (.not. known) then
            call array_expression_static_extent(arena, right_index, context, &
                                                right_extent, known)
            if (known) extent = right_extent
            return
        end if
        call array_expression_static_extent(arena, right_index, context, &
                                            right_extent, known)
        if (known .and. left_extent /= right_extent) then
            extent = 0
            known = .false.
            return
        end if
        extent = left_extent
    end subroutine array_expression_static_extent

    subroutine record_allocatable_static_size(arena, size_index, symbol_index, &
                                              context)
        ! Remember the rank-1 allocation extent when it is a compile-time
        ! constant; this lets whole-array print unroll without a runtime loop.
        ! A non-constant size leaves the size unknown (0), keeping print xfail.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: size_index
        integer, intent(in) :: symbol_index
        type(lowering_context_t), intent(inout) :: context
        integer(c_int64_t) :: n
        character(len=:), allocatable :: eval_err

        context%symbols(symbol_index)%allocatable_static_size = 0
        call eval_i32_constant(arena, size_index, context, n, eval_err)
        if (len_trim(eval_err) == 0 .and. n >= 0_c_int64_t) &
            context%symbols(symbol_index)%allocatable_static_size = int(n)
    end subroutine record_allocatable_static_size

    subroutine record_allocatable_static_size_2d(arena, size_index1, size_index2, &
                                                 symbol_index, context)
        ! Remember the total rank-2 element count (M*N) when both extents are
        ! compile-time constants, so a whole-array scalar broadcast can unroll
        ! over the flat storage. A non-constant extent leaves the size 0.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: size_index1, size_index2
        integer, intent(in) :: symbol_index
        type(lowering_context_t), intent(inout) :: context
        integer(c_int64_t) :: m, n
        character(len=:), allocatable :: err1, err2

        context%symbols(symbol_index)%allocatable_static_size = 0
        call eval_i32_constant(arena, size_index1, context, m, err1)
        call eval_i32_constant(arena, size_index2, context, n, err2)
        if (len_trim(err1) == 0 .and. len_trim(err2) == 0 .and. &
            m >= 0_c_int64_t .and. n >= 0_c_int64_t) &
            context%symbols(symbol_index)%allocatable_static_size = int(m * n)
    end subroutine record_allocatable_static_size_2d

    subroutine allocate_target(arena, node, name, size_index1, size_index2, &
                               error_msg)
        ! Extract the variable name and size expression(s). allocate parses
        ! a(N) or a(M,N) as a call_or_subscript in var_indices (or as an
        ! identifier in var_indices with shape in shape_indices for rank-1).
        ! size_index2 is 0 for rank-1 allocations.
        type(ast_arena_t), intent(in) :: arena
        type(allocate_statement_node), intent(in) :: node
        character(len=:), allocatable, intent(out) :: name
        integer, intent(out) :: size_index1, size_index2
        character(len=:), allocatable, intent(out) :: error_msg

        call set_empty(error_msg)
        size_index1 = 0
        size_index2 = 0
        if (.not. node_exists(arena, node%var_indices(1))) then
            error_msg = 'allocate target is not an AST node'
            return
        end if
        if (is_identifier(arena, node%var_indices(1))) then
            call get_identifier_name(arena, node%var_indices(1), name, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. allocated(node%shape_indices) .or. &
                size(node%shape_indices) /= 1) then
                error_msg = 'only one-dimensional allocate is supported '// &
                            'for bare-identifier form'
                return
            end if
            size_index1 = node%shape_indices(1)
            return
        end if
        select type (v => arena%entries(node%var_indices(1))%node)
        type is (call_or_subscript_node)
            if (.not. allocated(v%name) .or. .not. allocated(v%arg_indices)) then
                error_msg = 'allocate target must name an array with a size'
                return
            end if
            if (size(v%arg_indices) < 1 .or. size(v%arg_indices) > 2) then
                error_msg = 'only rank-1 and rank-2 allocate is supported'
                return
            end if
            name = v%name
            size_index1 = v%arg_indices(1)
            if (size(v%arg_indices) == 2) size_index2 = v%arg_indices(2)
        type is (array_slice_node)
            if (v%array_index <= 0 .or. v%num_dimensions < 1 .or. &
                v%num_dimensions > 2) then
                error_msg = 'allocate target must name an array with a size'
                return
            end if
            call get_identifier_name(arena, v%array_index, name, error_msg)
            if (len_trim(error_msg) > 0) return
            size_index1 = v%bounds_indices(1)
            if (v%num_dimensions == 2) size_index2 = v%bounds_indices(2)
        class default
            error_msg = 'allocate target must be a scalar array name ('// &
                        trim(arena%entries(node%var_indices(1))%node_type)//')'
        end select
    end subroutine allocate_target

    subroutine lower_allocate_i32_1d(arena, size_index, symbol_index, context, &
                                     error_msg)
        ! data = malloc(N*4); descriptor.data = data; .lower = 1; .upper = N
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: size_index
        integer, intent(in) :: symbol_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: n_i32

        call lower_i32_expression(arena, size_index, context, n_i32, error_msg)
        if (len_trim(error_msg) > 0) return
        call lower_allocate_i32_1d_operand(n_i32, symbol_index, context, error_msg)
    end subroutine lower_allocate_i32_1d

    subroutine lower_allocate_i32_2d(arena, size_index1, size_index2, &
                                     symbol_index, context, error_msg)
        ! data = malloc(M*N*4); descriptor: ptr, lower1=1, upper1=M,
        !                                       lower2=1, upper2=N.
        ! Convert M and N to i64 before arithmetic so the i32 values are
        ! not consumed by the multiply before they are stored in the descriptor.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: size_index1, size_index2
        integer, intent(in) :: symbol_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: m_i32, n_i32, m_i64, n_i64
        type(lr_operand_desc_t) :: total_i64, data_ptr, descriptor
        type(lr_operand_desc_t) :: extents_i64(2)

        call lower_i32_expression(arena, size_index1, context, m_i32, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_liric_i32_to_i64(context%session, m_i32, m_i64, error_msg)) &
            return
        call lower_i32_expression(arena, size_index2, context, n_i32, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_liric_i32_to_i64(context%session, n_i32, n_i64, error_msg)) &
            return

        ! count = M * N; the element size goes to the runtime separately so
        ! the byte product is formed and range-checked there (#428).
        if (.not. emit_i64_binary(context%session, LR_OP_MUL, m_i64, n_i64, &
                total_i64, error_msg)) return
        if (.not. emit_malloc(context%session, total_i64, data_ptr, error_msg, &
                elem_size=i64_immediate(context%session, &
                    allocatable_elem_size( &
                        context%symbols(symbol_index)%value_kind)))) return

        descriptor = context%symbols(symbol_index)%allocatable_descriptor_address
        if (.not. emit_ptr_store(context%session, data_ptr, descriptor, &
                error_msg)) return
        ! Unit lower bounds with extents M and N, plus column-major strides.
        extents_i64(1) = m_i64
        extents_i64(2) = n_i64
        call emit_alloc_desc_allocate_shape(context, descriptor, &
            context%symbols(symbol_index)%value_kind, 2, extents_i64, error_msg)
    end subroutine lower_allocate_i32_2d

    subroutine lower_allocate_i32_1d_operand(n_i32, symbol_index, context, error_msg)
        ! data = malloc(N*4); descriptor.data = data; .lower = 1; .upper = N
        ! n_i32 is the already-lowered size operand (i32).
        type(lr_operand_desc_t), intent(in) :: n_i32
        integer, intent(in) :: symbol_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: n_i64, data_ptr
        type(lr_operand_desc_t) :: descriptor
        type(lr_operand_desc_t) :: extents_i64(2)

        descriptor = context%symbols(symbol_index)%allocatable_descriptor_address
        if (.not. emit_liric_i32_to_i64(context%session, n_i32, n_i64, error_msg)) &
            return
        ! The element count and the element size go to the runtime separately,
        ! so the product is formed and range-checked there (#428). The former
        ! i32 multiply here silently wrapped for a large extent.
        if (.not. emit_malloc(context%session, n_i64, data_ptr, error_msg, &
                elem_size=i64_immediate(context%session, &
                    int(allocatable_elem_size( &
                        context%symbols(symbol_index)%value_kind), &
                        c_int64_t)))) return

        if (.not. emit_ptr_store(context%session, data_ptr, descriptor, &
                error_msg)) return
        extents_i64(1) = n_i64
        call emit_alloc_desc_allocate_shape(context, descriptor, &
            context%symbols(symbol_index)%value_kind, 1, extents_i64, error_msg)
    end subroutine lower_allocate_i32_1d_operand

    subroutine lower_allocate_i32_2d_operands(m_i32, n_i32, symbol_index, &
                                              context, error_msg)
        ! Runtime-shape counterpart of lower_allocate_i32_2d: m_i32 and n_i32
        ! have already been read from an array descriptor, so no AST bound is
        ! available to lower here.
        type(lr_operand_desc_t), intent(in) :: m_i32, n_i32
        integer, intent(in) :: symbol_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: m_i64, n_i64, total_i64
        type(lr_operand_desc_t) :: data_ptr, descriptor
        type(lr_operand_desc_t) :: extents_i64(2)

        if (.not. emit_liric_i32_to_i64(context%session, m_i32, m_i64, &
                error_msg)) return
        if (.not. emit_liric_i32_to_i64(context%session, n_i32, n_i64, &
                error_msg)) return
        if (.not. emit_i64_binary(context%session, LR_OP_MUL, m_i64, n_i64, &
                total_i64, error_msg)) return
        if (.not. emit_malloc(context%session, total_i64, data_ptr, error_msg, &
                elem_size=i64_immediate(context%session, &
                    allocatable_elem_size( &
                        context%symbols(symbol_index)%value_kind)))) return

        descriptor = context%symbols(symbol_index)%allocatable_descriptor_address
        if (.not. emit_ptr_store(context%session, data_ptr, descriptor, &
                error_msg)) return
        extents_i64(1) = m_i64
        extents_i64(2) = n_i64
        call emit_alloc_desc_allocate_shape(context, descriptor, &
            context%symbols(symbol_index)%value_kind, 2, extents_i64, error_msg)
    end subroutine lower_allocate_i32_2d_operands

    subroutine lower_allocatable_constructor_assignment(arena, ctor, symbol_index, &
                                                       context, error_msg)
        ! a = [e1, e2, ...] where a is a 1-D allocatable of integer, real, or
        ! logical elements (B2c). Deallocates old storage, allocates for N
        ! elements, then stores each at the kind-specific stride.
        type(ast_arena_t), intent(in) :: arena
        type(array_literal_node), intent(in) :: ctor
        integer, intent(in) :: symbol_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: n_i32, elem_addr, value
        type(lr_operand_desc_t) :: descriptor, data_ptr
        integer :: i, n, vk
        integer(c_int64_t) :: stride
        integer, allocatable :: flat(:), source_symbols(:), source_offsets(:)
        type(lr_operand_desc_t), allocatable :: values(:), re_values(:), im_values(:)

        if (context%symbols(symbol_index)%value_kind == VALUE_CHARACTER) then
            error_msg = 'array constructor assignment to a character '// &
                        'allocatable is not supported'
            return
        end if
        if (.not. allocated(ctor%element_indices)) then
            error_msg = 'array constructor has no elements'
            return
        end if
        call expand_allocatable_constructor_elements(arena, ctor%element_indices, &
            context, flat, source_symbols, source_offsets, error_msg)
        if (len_trim(error_msg) > 0) return
        n = size(flat)
        if (n <= 0) then
            error_msg = 'array constructor has no elements'
            return
        end if
        vk = context%symbols(symbol_index)%value_kind

        ! Materialize every RHS value before freeing the old allocation. Lazy
        ! array constructors permit a whole array operand, e.g. [x, 3], and
        ! Fortran evaluates that operand from the pre-assignment value.
        allocate(values(n))
        if (vk == VALUE_C4 .or. vk == VALUE_C8) then
            allocate(re_values(n), im_values(n))
        end if
        do i = 1, n
            if (source_symbols(i) > 0) then
                if (vk == VALUE_C4 .or. vk == VALUE_C8) then
                    call load_complex_array_linear_element(context, &
                        source_symbols(i), source_offsets(i), vk, re_values(i), &
                        im_values(i), error_msg)
                else
                    call load_array_linear_element(context, source_symbols(i), &
                        int(source_offsets(i), c_int64_t), values(i), error_msg)
                end if
            else if (vk == VALUE_C4) then
                call eval_c4_rhs(arena, flat(i), context, re_values(i), &
                                 im_values(i), error_msg)
            else if (vk == VALUE_C8) then
                call eval_c8_rhs(arena, flat(i), context, re_values(i), &
                                 im_values(i), error_msg)
            else
                call lower_allocatable_element_value(arena, flat(i), vk, context, &
                                                     values(i), error_msg)
            end if
            if (len_trim(error_msg) > 0) return
        end do

        ! Free old allocation if any.
        descriptor = context%symbols(symbol_index)%allocatable_descriptor_address
        if (.not. emit_ptr_load(context%session, descriptor, data_ptr, error_msg)) &
            return
        if (.not. emit_free(context%session, data_ptr, error_msg)) return

        ! Allocate N elements.
        n_i32 = i32_immediate(context%session, int(n, c_int64_t))
        call lower_allocate_i32_1d_operand(n_i32, symbol_index, context, error_msg)
        if (len_trim(error_msg) > 0) return
        context%symbols(symbol_index)%allocatable_static_size = n

        ! Store each element. After lower_allocate_i32_1d_operand the descriptor
        ! data pointer points to the fresh allocation with lower = 1.
        vk = context%symbols(symbol_index)%value_kind
        stride = allocatable_elem_size(vk)
        if (.not. emit_ptr_load(context%session, descriptor, data_ptr, error_msg)) &
            return
        do i = 1, n
            if (vk == VALUE_C4) then
                call store_allocatable_complex_element(context, symbol_index, &
                    i - 1, vk, re_values(i), im_values(i), error_msg)
                if (len_trim(error_msg) > 0) return
                cycle
            else if (vk == VALUE_C8) then
                call store_allocatable_complex_element(context, symbol_index, &
                    i - 1, vk, re_values(i), im_values(i), error_msg)
                if (len_trim(error_msg) > 0) return
                cycle
            end if
            value = values(i)
            if (.not. emit_ptr_offset(context%session, data_ptr, &
                    int(i - 1, c_int64_t) * stride, elem_addr, error_msg)) return
            if (.not. emit_allocatable_element_store(context, vk, value, &
                                                     elem_addr, error_msg)) return
        end do
        call set_empty(error_msg)
    end subroutine lower_allocatable_constructor_assignment

    subroutine expand_allocatable_constructor_elements(arena, indices, context, &
                                                        elements, source_symbols, &
                                                        source_offsets, error_msg)
        ! Flatten nested constructors and expand a whole rank-one array operand
        ! into element references. The latter is Lazy Fortran's concatenation
        ! form, used by the corpus cases `[x, 3]` and `[x, 5.0]`.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: indices(:)
        type(lowering_context_t), intent(in) :: context
        integer, allocatable, intent(out) :: elements(:), source_symbols(:), &
            source_offsets(:)
        character(len=:), allocatable, intent(out) :: error_msg
        integer, allocatable :: flat(:)
        integer :: i, j, sym, extent
        character(len=:), allocatable :: name, local_error

        allocate(flat(0), elements(0), source_symbols(0), source_offsets(0))
        call flatten_constructor_elements(arena, indices, flat, error_msg)
        if (len_trim(error_msg) > 0) return
        do i = 1, size(flat)
            sym = 0
            if (is_identifier(arena, flat(i))) then
                call get_identifier_name(arena, flat(i), name, local_error)
                if (len_trim(local_error) > 0) then
                    error_msg = local_error
                    return
                end if
                sym = find_symbol_compat(context, name)
                if (sym > 0) then
                    if (.not. is_elementwise_array_operand(context, sym) .or. &
                        context%symbols(sym)%array_rank > 1) sym = 0
                end if
            end if
            if (sym <= 0) then
                elements = [elements, flat(i)]
                source_symbols = [source_symbols, 0]
                source_offsets = [source_offsets, 0]
                cycle
            end if
            extent = effective_array_extent(context, sym)
            if (extent <= 0) then
                error_msg = 'whole-array constructor operand requires a positive extent'
                return
            end if
            do j = 0, extent - 1
                elements = [elements, flat(i)]
                source_symbols = [source_symbols, sym]
                source_offsets = [source_offsets, j]
            end do
        end do
        call set_empty(error_msg)
    end subroutine expand_allocatable_constructor_elements

    subroutine store_allocatable_complex_element(context, symbol_index, index, &
                                                 vk, re_value, im_value, &
                                                 error_msg)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index, index, vk
        type(lr_operand_desc_t), intent(in) :: re_value, im_value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: descriptor, data_ptr, re_addr, im_addr
        integer(c_int64_t) :: component_stride

        descriptor = context%symbols(symbol_index)%allocatable_descriptor_address
        if (.not. emit_ptr_load(context%session, descriptor, data_ptr, error_msg)) &
            return
        if (vk == VALUE_C4) then
            component_stride = 8_c_int64_t
            if (.not. emit_ptr_offset(context%session, data_ptr, &
                    int(index, c_int64_t) * component_stride, re_addr, &
                    error_msg)) return
            if (.not. emit_ptr_offset(context%session, re_addr, 4_c_int64_t, &
                    im_addr, error_msg)) return
            if (.not. emit_liric_f32_store(context%session, re_value, re_addr, &
                    error_msg)) return
            if (.not. emit_liric_f32_store(context%session, im_value, im_addr, &
                    error_msg)) return
        else
            component_stride = 16_c_int64_t
            if (.not. emit_ptr_offset(context%session, data_ptr, &
                    int(index, c_int64_t) * component_stride, re_addr, &
                    error_msg)) return
            if (.not. emit_ptr_offset(context%session, re_addr, 8_c_int64_t, &
                    im_addr, error_msg)) return
            if (.not. emit_liric_f64_store(context%session, re_value, re_addr, &
                    error_msg)) return
            if (.not. emit_liric_f64_store(context%session, im_value, im_addr, &
                    error_msg)) return
        end if
        call set_empty(error_msg)
    end subroutine store_allocatable_complex_element

    subroutine lower_allocatable_character_array_assignment(arena, node, &
                                                            symbol_index, context, &
                                                            handled, error_msg)
        ! Copy a conforming rank-1 character allocatable from another character
        ! allocatable. Character allocatable arrays use one immutable string
        ! pointer per element, so copying the pointer slots is the element-value
        ! copy; later element assignment replaces a slot and does not mutate the
        ! shared string data. Keep this path deliberately narrow: a known source
        ! extent and equal known element lengths avoid silently walking an
        ! unknown or nonconforming descriptor.
        use, intrinsic :: iso_fortran_env, only: int64
        type(ast_arena_t), intent(in) :: arena
        type(assignment_node), intent(in) :: node
        integer, intent(in) :: symbol_index
        type(lowering_context_t), intent(inout) :: context
        logical, intent(out) :: handled
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: source_name
        integer :: source_index, source_extent, target_extent
        type(lr_operand_desc_t) :: target_data, source_data
        integer(int64) :: byte_count
        logical :: empty_constructor

        handled = .false.
        call set_empty(error_msg)
        if (context%symbols(symbol_index)%array_rank /= 1) return
        if (context%symbols(symbol_index)%value_kind /= VALUE_CHARACTER) return
        if (.not. node_exists(arena, node%value_index)) return

        empty_constructor = .false.
        select type (rhs => arena%entries(node%value_index)%node)
        type is (array_literal_node)
            if (.not. allocated(rhs%element_indices)) then
                empty_constructor = .true.
            else if (size(rhs%element_indices) == 0) then
                empty_constructor = .true.
            end if
        class default
        end select
        if (empty_constructor) then
            if (.not. emit_ptr_load(context%session, &
                    context%symbols(symbol_index)%allocatable_descriptor_address, &
                    target_data, error_msg)) return
            if (.not. emit_free(context%session, target_data, error_msg)) return
            call lower_allocate_i32_1d_operand( &
                i32_immediate(context%session, 0_c_int64_t), symbol_index, &
                context, error_msg)
            if (len_trim(error_msg) > 0) return
            context%symbols(symbol_index)%allocatable_static_size = 0
            handled = .true.
            return
        end if
        if (.not. is_identifier(arena, node%value_index)) return

        call get_identifier_name(arena, node%value_index, source_name, error_msg)
        if (len_trim(error_msg) > 0) return
        source_index = find_symbol_compat(context, source_name)
        if (source_index <= 0) return
        if (.not. context%symbols(source_index)%is_allocatable) return
        if (context%symbols(source_index)%array_rank /= 1) return
        if (context%symbols(source_index)%value_kind /= VALUE_CHARACTER) return

        source_extent = context%symbols(source_index)%allocatable_static_size
        if (source_extent <= 0) return
        target_extent = context%symbols(symbol_index)%allocatable_static_size
        if (target_extent > 0 .and. target_extent /= source_extent) then
            error_msg = 'whole-array assignment requires conforming shapes: '// &
                trim(context%symbols(symbol_index)%name)//' and '// &
                trim(context%symbols(source_index)%name)//' differ in extent'
            return
        end if
        if (context%symbols(symbol_index)%character_length > 0 .and. &
            context%symbols(source_index)%character_length > 0 .and. &
            context%symbols(symbol_index)%character_length /= &
                context%symbols(source_index)%character_length) then
            error_msg = 'whole-array assignment requires equal character lengths'
            return
        end if

        if (.not. emit_ptr_load(context%session, &
                context%symbols(symbol_index)%allocatable_descriptor_address, &
                target_data, error_msg)) return
        if (.not. emit_ptr_load(context%session, &
                context%symbols(source_index)%allocatable_descriptor_address, &
                source_data, error_msg)) return
        byte_count = int(source_extent, int64) * &
            int(allocatable_elem_size(VALUE_CHARACTER), int64)
        if (.not. emit_memcpy(context%session, target_data, source_data, &
                i64_immediate(context%session, byte_count), error_msg)) return

        context%symbols(symbol_index)%allocatable_static_size = source_extent
        handled = .true.
    end subroutine lower_allocatable_character_array_assignment

    subroutine lower_allocatable_runtime_whole_copy(arena, node, target_sym, &
                                                    context, handled, error_msg)
        ! a = b where b is a separately allocated rank-1 intrinsic array whose
        ! extent is present only in its descriptor. Reallocate a from b's
        ! runtime extent, then copy b through a counted loop. Keep this path
        ! separate from the compile-time expression machinery: it is one
        ! concrete whole-copy form, and unsupported RHS patterns still reach
        ! their existing diagnostics.
        type(ast_arena_t), intent(in) :: arena
        type(assignment_node), intent(in) :: node
        integer, intent(in) :: target_sym
        type(lowering_context_t), intent(inout) :: context
        logical, intent(out) :: handled
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: source_name
        integer :: source_sym, vk
        type(lr_operand_desc_t) :: source_extent, descriptor, data_ptr
        type(lr_operand_desc_t) :: entry_idx, header_idx, backedge_idx
        type(lr_operand_desc_t) :: condition, element, next_idx
        type(lr_operand_desc_t) :: idx64, byte_offset, element_addr
        integer(c_int32_t) :: entry_block, header_block, body_block, &
            latch_block, exit_block, idx_vreg

        handled = .false.
        call set_empty(error_msg)
        if (.not. node_exists(arena, node%value_index)) return
        if (context%symbols(target_sym)%array_rank == 2) then
            call lower_allocatable_runtime_rank2_copy(arena, node, target_sym, &
                context, handled, error_msg)
            return
        end if
        if (.not. is_identifier(arena, node%value_index)) return
        if (context%symbols(target_sym)%array_rank /= 1) return
        vk = context%symbols(target_sym)%value_kind
        if (vk /= VALUE_I32 .and. vk /= VALUE_F32 .and. vk /= VALUE_F64 .and. &
            vk /= VALUE_LOGICAL) return

        call get_identifier_name(arena, node%value_index, source_name, error_msg)
        if (len_trim(error_msg) > 0) return
        source_sym = find_symbol_compat(context, source_name)
        if (source_sym <= 0) return
        if (.not. context%symbols(source_sym)%is_allocatable) return
        if (context%symbols(source_sym)%array_rank /= 1) return
        if (context%symbols(source_sym)%value_kind /= vk) return
        if (context%symbols(source_sym)%allocatable_static_size > 0) return
        if (source_sym == target_sym) return

        call allocatable_descriptor_extent_i32(context, source_sym, 1, &
                                               source_extent, error_msg)
        if (len_trim(error_msg) > 0) return
        descriptor = context%symbols(target_sym)%allocatable_descriptor_address
        if (.not. emit_ptr_load(context%session, descriptor, data_ptr, error_msg)) &
            return
        if (.not. emit_free(context%session, data_ptr, error_msg)) return
        call lower_allocate_i32_1d_operand(source_extent, target_sym, context, &
                                            error_msg)
        if (len_trim(error_msg) > 0) return
        context%symbols(target_sym)%allocatable_static_size = 0

        if (.not. emit_ptr_load(context%session, descriptor, data_ptr, error_msg)) &
            return
        entry_idx = i32_immediate(context%session, 0_c_int64_t)
        entry_block = context%current_block_id
        header_block = create_liric_block(context%session)
        body_block = create_liric_block(context%session)
        latch_block = create_liric_block(context%session)
        exit_block = create_liric_block(context%session)
        if (.not. emit_liric_br(context%session, header_block, error_msg)) return

        if (.not. set_liric_block(context%session, header_block, error_msg)) return
        context%current_block_id = header_block
        context%current_block_terminated = .false.
        idx_vreg = reserve_i32_vreg(context%session)
        if (idx_vreg == 0_c_int32_t) idx_vreg = reserve_i32_vreg(context%session)
        if (idx_vreg <= 0_c_int32_t) then
            error_msg = 'LIRIC could not reserve a positive backedge vreg'
            return
        end if
        backedge_idx = i32_vreg(context%session, idx_vreg)
        if (.not. emit_liric_i32_phi(context%session, entry_idx, entry_block, &
                backedge_idx, latch_block, header_idx, error_msg)) return
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_SLT, header_idx, &
                source_extent, condition, error_msg)) return
        if (.not. emit_liric_condbr(context%session, condition, body_block, &
                exit_block, error_msg)) return

        if (.not. set_liric_block(context%session, body_block, error_msg)) return
        context%current_block_id = body_block
        context%current_block_terminated = .false.
        call load_alloc_element_runtime(context, source_sym, vk, header_idx, &
                                        element, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_liric_i32_to_i64(context%session, header_idx, idx64, &
                error_msg)) return
        if (.not. emit_i64_binary(context%session, LR_OP_MUL, idx64, &
                i64_immediate(context%session, allocatable_elem_size(vk)), &
                byte_offset, error_msg)) return
        if (.not. emit_ptr_offset_dyn(context%session, data_ptr, byte_offset, &
                element_addr, error_msg)) return
        if (.not. emit_allocatable_element_store(context, vk, element, &
                                                 element_addr, error_msg)) return
        if (.not. emit_liric_br(context%session, latch_block, error_msg)) return

        if (.not. set_liric_block(context%session, latch_block, error_msg)) return
        context%current_block_id = latch_block
        context%current_block_terminated = .false.
        if (.not. emit_i32_binary_into(context%session, LR_OP_ADD, header_idx, &
                i32_immediate(context%session, 1_c_int64_t), idx_vreg, next_idx, &
                error_msg)) return
        if (.not. emit_liric_br(context%session, header_block, error_msg)) return

        if (.not. set_liric_block(context%session, exit_block, error_msg)) return
        context%current_block_id = exit_block
        context%current_block_terminated = .false.
        handled = .true.
        call set_empty(error_msg)
    end subroutine lower_allocatable_runtime_whole_copy

    subroutine lower_allocatable_runtime_rank2_copy(arena, node, target_sym, &
                                                    context, handled, error_msg)
        ! Copy a separate rank-2 intrinsic allocatable whose two extents are
        ! available only in the RHS descriptor. The target is resized before a
        ! bounded column-major loop copies the source. This intentionally does
        ! not cover global descriptors, components, aliases, or unsupported
        ! kinds; those continue through the existing diagnostic paths.
        type(ast_arena_t), intent(in) :: arena
        type(assignment_node), intent(in) :: node
        integer, intent(in) :: target_sym
        type(lowering_context_t), intent(inout) :: context
        logical, intent(out) :: handled
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: source_name
        integer :: source_sym, vk
        type(lr_operand_desc_t) :: extent1, extent2, descriptor, data_ptr
        type(lr_operand_desc_t) :: source_data, target_data, idx64, byte_offset
        type(lr_operand_desc_t) :: source_addr, target_addr
        type(lr_operand_desc_t) :: entry_j, header_j, backedge_j, next_j
        type(lr_operand_desc_t) :: entry_i, header_i, backedge_i, next_i
        type(lr_operand_desc_t) :: column_offset, linear_index, value, condition
        type(lr_operand_desc_t) :: one
        integer(c_int32_t) :: outer_entry, outer_header, outer_body, outer_latch
        integer(c_int32_t) :: outer_exit, inner_header, inner_body
        integer(c_int32_t) :: i_vreg, j_vreg

        handled = .false.
        call set_empty(error_msg)
        if (.not. node_exists(arena, node%value_index)) return
        if (is_identifier(arena, node%value_index)) then
            call get_identifier_name(arena, node%value_index, source_name, &
                                     error_msg)
        else
            select type (rhs => arena%entries(node%value_index)%node)
            type is (call_or_subscript_node)
                if (.not. allocated(rhs%name)) then
                    return
                end if
                if (allocated(rhs%arg_indices)) then
                    if (size(rhs%arg_indices) /= 0) then
                        return
                    end if
                end if
                source_name = rhs%name
            class default
                return
            end select
        end if
        if (len_trim(error_msg) > 0) return

        vk = context%symbols(target_sym)%value_kind
        if (vk /= VALUE_I32 .and. vk /= VALUE_F32 .and. &
            vk /= VALUE_F64 .and. vk /= VALUE_LOGICAL) return
        if (context%symbols(target_sym)%allocatable_descriptor_address%kind == &
            LR_OP_KIND_GLOBAL) return

        source_sym = find_symbol_compat(context, source_name)
        if (source_sym <= 0) return
        if (.not. context%symbols(source_sym)%is_allocatable) return
        if (context%symbols(source_sym)%array_rank /= 2) return
        if (context%symbols(source_sym)%value_kind /= vk) return
        if (context%symbols(source_sym)%allocatable_static_size > 0) return
        if (source_sym == target_sym) return
        if (context%symbols(source_sym)%allocatable_descriptor_address%kind == &
            LR_OP_KIND_GLOBAL) return

        call allocatable_descriptor_extent_i32(context, source_sym, 1, extent1, &
                                               error_msg)
        if (len_trim(error_msg) > 0) return
        call allocatable_descriptor_extent_i32(context, source_sym, 2, extent2, &
                                               error_msg)
        if (len_trim(error_msg) > 0) return

        descriptor = context%symbols(target_sym)%allocatable_descriptor_address
        if (.not. emit_ptr_load(context%session, descriptor, data_ptr, error_msg)) &
            return
        if (.not. emit_free(context%session, data_ptr, error_msg)) return
        call lower_allocate_i32_2d_operands(extent1, extent2, target_sym, &
                                             context, error_msg)
        if (len_trim(error_msg) > 0) return
        context%symbols(target_sym)%allocatable_static_size = 0

        one = i32_immediate(context%session, 1_c_int64_t)
        entry_j = i32_immediate(context%session, 0_c_int64_t)
        outer_entry = context%current_block_id
        outer_header = create_liric_block(context%session)
        outer_body = create_liric_block(context%session)
        outer_latch = create_liric_block(context%session)
        outer_exit = create_liric_block(context%session)
        inner_header = create_liric_block(context%session)
        inner_body = create_liric_block(context%session)
        if (.not. emit_liric_br(context%session, outer_header, error_msg)) return

        if (.not. enter_liric_block(context, outer_header, error_msg)) return
        j_vreg = reserve_i32_vreg(context%session)
        if (j_vreg <= 0_c_int32_t) then
            error_msg = 'LIRIC could not reserve the rank-2 column loop vreg'
            return
        end if
        backedge_j = i32_vreg(context%session, j_vreg)
        if (.not. emit_liric_i32_phi(context%session, entry_j, outer_entry, &
                backedge_j, outer_latch, header_j, error_msg)) return
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_SLT, header_j, &
                extent2, condition, error_msg)) return
        if (.not. emit_liric_condbr(context%session, condition, outer_body, &
                outer_exit, error_msg)) return

        if (.not. enter_liric_block(context, outer_body, error_msg)) return
        entry_i = i32_immediate(context%session, 0_c_int64_t)
        if (.not. emit_liric_br(context%session, inner_header, error_msg)) return

        if (.not. enter_liric_block(context, inner_header, error_msg)) return
        i_vreg = reserve_i32_vreg(context%session)
        if (i_vreg <= 0_c_int32_t) then
            error_msg = 'LIRIC could not reserve the rank-2 row loop vreg'
            return
        end if
        backedge_i = i32_vreg(context%session, i_vreg)
        if (.not. emit_liric_i32_phi(context%session, entry_i, outer_body, &
                backedge_i, inner_body, header_i, error_msg)) return
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_SLT, header_i, &
                extent1, condition, error_msg)) return
        if (.not. emit_liric_condbr(context%session, condition, inner_body, &
                outer_latch, error_msg)) return

        if (.not. enter_liric_block(context, inner_body, error_msg)) return
        if (.not. emit_i32_binary(context%session, LR_OP_MUL, header_j, extent1, &
                column_offset, error_msg)) return
        if (.not. emit_i32_binary(context%session, LR_OP_ADD, header_i, &
                column_offset, linear_index, error_msg)) return
        if (.not. emit_ptr_load(context%session, &
                context%symbols(source_sym)%allocatable_descriptor_address, &
                source_data, error_msg)) return
        if (.not. emit_ptr_load(context%session, &
                context%symbols(target_sym)%allocatable_descriptor_address, &
                target_data, error_msg)) return
        if (.not. emit_liric_i32_to_i64(context%session, linear_index, idx64, &
                error_msg)) return
        if (.not. emit_i64_binary(context%session, LR_OP_MUL, idx64, &
                i64_immediate(context%session, allocatable_elem_size(vk)), &
                byte_offset, error_msg)) return
        if (.not. emit_ptr_offset_dyn(context%session, source_data, byte_offset, &
                source_addr, error_msg)) return
        if (.not. emit_allocatable_element_load(context, vk, source_addr, value, &
                error_msg)) return
        if (len_trim(error_msg) > 0) return
        if (.not. emit_ptr_offset_dyn(context%session, target_data, byte_offset, &
                target_addr, error_msg)) return
        if (.not. emit_allocatable_element_store(context, vk, value, target_addr, &
                error_msg)) return
        if (len_trim(error_msg) > 0) return
        if (.not. emit_i32_binary_into(context%session, LR_OP_ADD, header_i, one, &
                i_vreg, next_i, error_msg)) return
        if (.not. emit_liric_br(context%session, inner_header, error_msg)) return

        if (.not. enter_liric_block(context, outer_latch, error_msg)) return
        if (.not. emit_i32_binary_into(context%session, LR_OP_ADD, header_j, one, &
                j_vreg, next_j, error_msg)) return
        if (.not. emit_liric_br(context%session, outer_header, error_msg)) return

        if (.not. enter_liric_block(context, outer_exit, error_msg)) return
        handled = .true.
        call set_empty(error_msg)
    end subroutine lower_allocatable_runtime_rank2_copy

    subroutine lower_allocatable_scalar_broadcast(arena, value_index, &
                                                  symbol_index, context, error_msg)
        ! a = scalar where a is a rank-1 allocatable: store the scalar into every
        ! element. Requires the current extent to be a compile-time constant
        ! (allocatable_static_size), set by the preceding allocate.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: value_index
        integer, intent(in) :: symbol_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: value, descriptor, data_ptr, addr
        integer :: vk, n, i
        integer(c_int64_t) :: stride

        if (context%symbols(symbol_index)%array_rank > 2) then
            error_msg = 'allocatable scalar broadcast supports only rank-1 and '// &
                        'rank-2 arrays'
            return
        end if
        if (context%symbols(symbol_index)%value_kind == VALUE_CHARACTER) then
            error_msg = 'scalar broadcast assignment to a character '// &
                        'allocatable is not supported'
            return
        end if
        n = context%symbols(symbol_index)%allocatable_static_size
        if (n <= 0) then
            error_msg = 'allocatable scalar broadcast requires a constant extent'
            return
        end if

        vk = context%symbols(symbol_index)%value_kind
        call lower_allocatable_element_value(arena, value_index, vk, context, &
                                             value, error_msg)
        if (len_trim(error_msg) > 0) return

        stride = allocatable_elem_size(vk)
        descriptor = context%symbols(symbol_index)%allocatable_descriptor_address
        if (.not. emit_ptr_load(context%session, descriptor, data_ptr, error_msg)) &
            return
        do i = 0, n - 1
            if (.not. emit_ptr_offset(context%session, data_ptr, &
                    int(i, c_int64_t) * stride, addr, error_msg)) return
            if (.not. emit_allocatable_element_store(context, vk, value, addr, &
                                                     error_msg)) return
        end do
        call set_empty(error_msg)
    end subroutine lower_allocatable_scalar_broadcast

    subroutine lower_allocatable_elementwise_assignment(arena, node, &
                                                         symbol_index, context, &
                                                         error_msg)
        ! a = expr where a is a rank-1 allocatable and expr is a general
        ! elementwise expression (arithmetic over other arrays and/or
        ! scalars). Requires the current extent to be a compile-time
        ! constant (allocatable_static_size), set by a preceding allocate.
        type(ast_arena_t), intent(in) :: arena
        type(assignment_node), intent(in) :: node
        integer, intent(in) :: symbol_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(array_expr_plan_t) :: plan
        integer :: array_size
        integer :: rhs_extent

        if (context%symbols(symbol_index)%array_rank /= 1) then
            error_msg = 'allocatable whole-array assignment from an expression '// &
                        'supports only rank-1 arrays'
            return
        end if
        if (context%symbols(symbol_index)%value_kind == VALUE_CHARACTER) then
            error_msg = 'whole-array assignment from an expression to a '// &
                        'character allocatable is not supported'
            return
        end if
        array_size = context%symbols(symbol_index)%allocatable_static_size
        rhs_extent = elementwise_expression_extent(arena, node%value_index, context)
        if (rhs_extent > 0 .and. rhs_extent /= array_size) then
            ! F2018 10.2.1.3: an unallocated or nonconforming allocatable
            ! destination is (re)allocated to the shape of the right-hand side.
            call reallocate_allocatable_extent(symbol_index, rhs_extent, context, &
                                               error_msg)
            if (len_trim(error_msg) > 0) return
            array_size = rhs_extent
        end if
        if (array_size <= 0) then
            error_msg = 'allocatable whole-array assignment from an expression '// &
                        'requires a previously allocated constant extent'
            return
        end if
        call build_array_expression(context, symbol_index, node%value_index, &
                                    plan, error_msg)
        if (len_trim(error_msg) > 0) return
        call emit_array_expression_assignment(arena, plan, context, error_msg)
    end subroutine lower_allocatable_elementwise_assignment

    logical function allocatable_assignment_rhs_is_matmul(arena, node) result(is_matmul)
        type(ast_arena_t), intent(in) :: arena
        type(assignment_node), intent(in) :: node

        is_matmul = .false.
        if (.not. node_exists(arena, node%value_index)) return
        select type (rhs => arena%entries(node%value_index)%node)
        type is (call_or_subscript_node)
            if (allocated(rhs%name)) is_matmul = same_name(rhs%name, 'matmul')
        class default
            return
        end select
    end function allocatable_assignment_rhs_is_matmul

    subroutine lower_allocatable_matmul_assignment(arena, node, target_sym, &
                                                    context, error_msg)
        !! Evaluate MATMUL into a previously allocated rank-2 target with three
        !! runtime LIRIC loops. Allocatable extents are descriptor values, so a
        !! 96x96 benchmark does not become nearly a million compile-time IR
        !! operations or require a large temporary matrix in the compiler.
        type(ast_arena_t), intent(in) :: arena
        type(assignment_node), intent(in) :: node
        integer, intent(in) :: target_sym
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: a_sym, b_sym, value_kind
        type(lr_operand_desc_t) :: a_rows, a_cols, b_rows, b_cols
        type(lr_operand_desc_t) :: entry_j, header_j, backedge_j, next_j
        type(lr_operand_desc_t) :: entry_i, header_i, backedge_i, next_i
        type(lr_operand_desc_t) :: entry_k, header_k, backedge_k, next_k
        type(lr_operand_desc_t) :: sum_entry, sum_header, sum_backedge
        type(lr_operand_desc_t) :: a_col_offset, a_linear, b_col_offset, b_linear
        type(lr_operand_desc_t) :: a_value, b_value, product, next_sum, output_index
        type(lr_operand_desc_t) :: target_desc, old_data, target_data
        type(lr_operand_desc_t) :: a_rows64, b_cols64, total64, extents64(2)
        type(lr_operand_desc_t) :: condition
        type(lr_operand_desc_t) :: one
        integer(c_int32_t) :: outer_entry, outer_header, outer_body, outer_latch, &
                               outer_exit, inner_header, inner_body, k_header, &
                               k_body, k_latch, k_exit
        integer(c_int32_t) :: j_vreg, i_vreg, k_vreg, sum_vreg

        call set_empty(error_msg)
        if (.not. node_exists(arena, node%value_index)) then
            error_msg = 'allocatable matmul assignment has no right-hand side'
            return
        end if
        select type (rhs => arena%entries(node%value_index)%node)
        type is (call_or_subscript_node)
            if (.not. allocated(rhs%arg_indices) .or. size(rhs%arg_indices) /= 2) then
                error_msg = 'matmul requires exactly two array arguments'
                return
            end if
            call resolve_i32_array_argument(arena, rhs, 1, context, 'matmul', &
                                            'left operand', a_sym, error_msg)
            if (len_trim(error_msg) > 0) return
            call resolve_i32_array_argument(arena, rhs, 2, context, 'matmul', &
                                            'right operand', b_sym, error_msg)
            if (len_trim(error_msg) > 0) return
        class default
            error_msg = 'matmul requires an intrinsic call'
            return
        end select

        if (context%symbols(a_sym)%array_rank /= 2 .or. &
            context%symbols(b_sym)%array_rank /= 2 .or. &
            context%symbols(target_sym)%array_rank /= 2) then
            error_msg = 'allocatable matmul requires rank-2 arrays'
            return
        end if
        value_kind = context%symbols(target_sym)%value_kind
        if (context%symbols(a_sym)%value_kind /= value_kind .or. &
            context%symbols(b_sym)%value_kind /= value_kind) then
            error_msg = 'matmul operands and result must have the same kind'
            return
        end if
        if (value_kind /= VALUE_F32 .and. value_kind /= VALUE_F64 .and. &
            value_kind /= VALUE_I32) then
            error_msg = 'allocatable matmul supports real and default integer arrays'
            return
        end if

        call allocatable_descriptor_extent_i32(context, a_sym, 1, a_rows, error_msg)
        if (len_trim(error_msg) > 0) return
        call allocatable_descriptor_extent_i32(context, a_sym, 2, a_cols, error_msg)
        if (len_trim(error_msg) > 0) return
        call allocatable_descriptor_extent_i32(context, b_sym, 1, b_rows, error_msg)
        if (len_trim(error_msg) > 0) return
        call allocatable_descriptor_extent_i32(context, b_sym, 2, b_cols, error_msg)
        if (len_trim(error_msg) > 0) return

        ! Assignment to an unallocated allocatable (and to a shape-changing
        ! allocatable) owns the result shape.  Materialize the target storage
        ! from the operand extents before the runtime loops first dereference
        ! its descriptor; otherwise the null data pointer reaches the element
        ! address helper and the generated program crashes.
        if (.not. emit_liric_i32_to_i64(context%session, a_rows, a_rows64, &
                                       error_msg)) return
        if (.not. emit_liric_i32_to_i64(context%session, b_cols, b_cols64, &
                                       error_msg)) return
        if (.not. emit_i64_binary(context%session, LR_OP_MUL, a_rows64, b_cols64, &
                                  total64, error_msg)) return
        target_desc = context%symbols(target_sym)%allocatable_descriptor_address
        if (.not. emit_ptr_load(context%session, target_desc, old_data, error_msg)) &
            return
        if (.not. emit_free(context%session, old_data, error_msg)) return
        if (.not. emit_malloc(context%session, total64, target_data, error_msg, &
                elem_size=i64_immediate(context%session, int( &
                    allocatable_elem_size(value_kind), c_int64_t)))) return
        if (.not. emit_ptr_store(context%session, target_data, target_desc, &
                                error_msg)) return
        extents64(1) = a_rows64
        extents64(2) = b_cols64
        call emit_alloc_desc_allocate_shape(context, target_desc, value_kind, 2, &
                                            extents64, error_msg)
        if (len_trim(error_msg) > 0) return

        one = i32_immediate(context%session, 1_c_int64_t)
        entry_j = i32_immediate(context%session, 0_c_int64_t)
        outer_entry = context%current_block_id
        outer_header = create_liric_block(context%session)
        outer_body = create_liric_block(context%session)
        outer_latch = create_liric_block(context%session)
        outer_exit = create_liric_block(context%session)
        inner_header = create_liric_block(context%session)
        inner_body = create_liric_block(context%session)
        k_header = create_liric_block(context%session)
        k_body = create_liric_block(context%session)
        k_latch = create_liric_block(context%session)
        k_exit = create_liric_block(context%session)
        if (.not. emit_liric_br(context%session, outer_header, error_msg)) return

        if (.not. enter_liric_block(context, outer_header, error_msg)) return
        j_vreg = reserve_i32_vreg(context%session)
        if (j_vreg <= 0_c_int32_t) then
            error_msg = 'LIRIC could not reserve the matmul column loop vreg'
            return
        end if
        backedge_j = i32_vreg(context%session, j_vreg)
        if (.not. emit_liric_i32_phi(context%session, entry_j, outer_entry, &
                backedge_j, outer_latch, header_j, error_msg)) return
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_SLT, header_j, &
                b_cols, condition, error_msg)) return
        if (.not. emit_liric_condbr(context%session, condition, outer_body, &
                outer_exit, error_msg)) return

        if (.not. enter_liric_block(context, outer_body, error_msg)) return
        entry_i = i32_immediate(context%session, 0_c_int64_t)
        if (.not. emit_liric_br(context%session, inner_header, error_msg)) return

        if (.not. enter_liric_block(context, inner_header, error_msg)) return
        i_vreg = reserve_i32_vreg(context%session)
        if (i_vreg <= 0_c_int32_t) then
            error_msg = 'LIRIC could not reserve the matmul row loop vreg'
            return
        end if
        backedge_i = i32_vreg(context%session, i_vreg)
        if (.not. emit_liric_i32_phi(context%session, entry_i, outer_body, &
                backedge_i, k_exit, header_i, error_msg)) return
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_SLT, header_i, &
                a_rows, condition, error_msg)) return
        if (.not. emit_liric_condbr(context%session, condition, inner_body, &
                outer_latch, error_msg)) return

        if (.not. enter_liric_block(context, inner_body, error_msg)) return
        entry_k = i32_immediate(context%session, 0_c_int64_t)
        if (value_kind == VALUE_F32) then
            sum_entry = liric_f32_immediate(context%session, 0.0_c_float)
        else if (value_kind == VALUE_F64) then
            sum_entry = liric_f64_immediate(context%session, 0.0_c_double)
        else
            sum_entry = i32_immediate(context%session, 0_c_int64_t)
        end if
        if (.not. emit_liric_br(context%session, k_header, error_msg)) return

        if (.not. enter_liric_block(context, k_header, error_msg)) return
        k_vreg = reserve_i32_vreg(context%session)
        sum_vreg = reserve_i32_vreg(context%session)
        if (k_vreg <= 0_c_int32_t .or. sum_vreg <= 0_c_int32_t) then
            error_msg = 'LIRIC could not reserve the matmul reduction vregs'
            return
        end if
        backedge_k = i32_vreg(context%session, k_vreg)
        if (value_kind == VALUE_F32) then
            sum_backedge = f32_vreg(context%session, sum_vreg)
        else if (value_kind == VALUE_F64) then
            sum_backedge = f64_vreg(context%session, sum_vreg)
        else
            sum_backedge = i32_vreg(context%session, sum_vreg)
        end if
        if (.not. emit_liric_i32_phi(context%session, entry_k, inner_body, &
                backedge_k, k_latch, header_k, error_msg)) return
        if (.not. emit_liric_phi(context%session, sum_entry, inner_body, &
                sum_backedge, k_latch, sum_header, error_msg)) return
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_SLT, header_k, &
                a_cols, condition, error_msg)) return
        if (.not. emit_liric_condbr(context%session, condition, k_body, &
                k_exit, error_msg)) return

        if (.not. enter_liric_block(context, k_body, error_msg)) return
        if (.not. emit_i32_binary(context%session, LR_OP_MUL, header_k, a_rows, &
                a_col_offset, error_msg)) return
        if (.not. emit_i32_binary(context%session, LR_OP_ADD, header_i, &
                a_col_offset, a_linear, error_msg)) return
        if (.not. emit_i32_binary(context%session, LR_OP_MUL, header_j, b_rows, &
                b_col_offset, error_msg)) return
        if (.not. emit_i32_binary(context%session, LR_OP_ADD, header_k, &
                b_col_offset, b_linear, error_msg)) return
        call load_array_element_at_operand(context, a_sym, a_linear, a_value, &
                                           error_msg)
        if (len_trim(error_msg) > 0) return
        call load_array_element_at_operand(context, b_sym, b_linear, b_value, &
                                           error_msg)
        if (len_trim(error_msg) > 0) return
        if (value_kind == VALUE_F32) then
            if (.not. emit_liric_f32_binary(context%session, LR_OP_FMUL, a_value, &
                    b_value, product, error_msg)) return
            if (.not. emit_liric_f32_binary(context%session, LR_OP_FADD, sum_header, &
                    product, next_sum, error_msg)) return
        else if (value_kind == VALUE_F64) then
            if (.not. emit_liric_f64_binary(context%session, LR_OP_FMUL, a_value, &
                    b_value, product, error_msg)) return
            if (.not. emit_liric_f64_binary(context%session, LR_OP_FADD, sum_header, &
                    product, next_sum, error_msg)) return
        else
            if (.not. emit_i32_binary(context%session, LR_OP_MUL, a_value, b_value, &
                    product, error_msg)) return
            if (.not. emit_i32_binary(context%session, LR_OP_ADD, sum_header, &
                    product, next_sum, error_msg)) return
        end if
        if (.not. emit_liric_br(context%session, k_latch, error_msg)) return

        if (.not. enter_liric_block(context, k_latch, error_msg)) return
        if (.not. emit_i32_binary_into(context%session, LR_OP_ADD, header_k, one, &
                k_vreg, next_k, error_msg)) return
        if (value_kind == VALUE_F32 .or. value_kind == VALUE_F64) then
            if (.not. emit_real_copy_to(context%session, next_sum, sum_vreg, &
                    sum_backedge, error_msg)) return
        else
            if (.not. emit_i32_copy_to(context%session, next_sum, sum_vreg, &
                    sum_backedge, error_msg)) return
        end if
        if (.not. emit_liric_br(context%session, k_header, error_msg)) return

        if (.not. enter_liric_block(context, k_exit, error_msg)) return
        if (.not. emit_i32_binary(context%session, LR_OP_MUL, header_j, a_rows, &
                b_col_offset, error_msg)) return
        if (.not. emit_i32_binary(context%session, LR_OP_ADD, header_i, &
                b_col_offset, output_index, error_msg)) return
        call store_array_element_at_operand(context, target_sym, output_index, &
                                            sum_header, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_i32_binary_into(context%session, LR_OP_ADD, header_i, one, &
                i_vreg, next_i, error_msg)) return
        if (.not. emit_liric_br(context%session, inner_header, error_msg)) return

        if (.not. enter_liric_block(context, outer_latch, error_msg)) return
        if (.not. emit_i32_binary_into(context%session, LR_OP_ADD, header_j, one, &
                j_vreg, next_j, error_msg)) return
        if (.not. emit_liric_br(context%session, outer_header, error_msg)) return

        if (.not. enter_liric_block(context, outer_exit, error_msg)) return
        call set_empty(error_msg)
    end subroutine lower_allocatable_matmul_assignment

    subroutine lower_allocatable_matvec_assignment(a_sym, b_sym, target_sym, &
                                                   context, error_msg)
        !! Evaluate a rank-2 allocatable matrix times a rank-1 vector into a
        !! fixed-size rank-1 target.  The matrix extents come from its
        !! descriptor, so the generated loop remains bounded at runtime.
        integer, intent(in) :: a_sym, b_sym, target_sym
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: value_kind
        type(lr_operand_desc_t) :: a_rows, a_cols
        type(lr_operand_desc_t) :: entry_i, header_i, backedge_i, next_i
        type(lr_operand_desc_t) :: entry_k, header_k, backedge_k, next_k
        type(lr_operand_desc_t) :: sum_entry, sum_header, sum_backedge
        type(lr_operand_desc_t) :: a_col_offset, a_linear
        type(lr_operand_desc_t) :: a_value, b_value, product, next_sum
        type(lr_operand_desc_t) :: condition, one
        integer(c_int32_t) :: outer_entry, outer_header, outer_body, outer_latch, &
                               outer_exit, inner_header, inner_body, inner_latch, &
                               inner_exit
        integer(c_int32_t) :: i_vreg, k_vreg, sum_vreg

        call set_empty(error_msg)
        if (context%symbols(a_sym)%array_rank /= 2 .or. &
            context%symbols(b_sym)%array_rank /= 1 .or. &
            context%symbols(target_sym)%array_rank /= 1) then
            error_msg = 'allocatable matvec requires matrix and vector arrays'
            return
        end if
        value_kind = context%symbols(target_sym)%value_kind
        if (context%symbols(a_sym)%value_kind /= value_kind .or. &
            context%symbols(b_sym)%value_kind /= value_kind) then
            error_msg = 'matvec operands and result must have the same kind'
            return
        end if
        if (value_kind /= VALUE_F32 .and. value_kind /= VALUE_F64 .and. &
            value_kind /= VALUE_I32) then
            error_msg = 'allocatable matvec supports real and default integer arrays'
            return
        end if

        call allocatable_descriptor_extent_i32(context, a_sym, 1, a_rows, error_msg)
        if (len_trim(error_msg) > 0) return
        call allocatable_descriptor_extent_i32(context, a_sym, 2, a_cols, error_msg)
        if (len_trim(error_msg) > 0) return

        one = i32_immediate(context%session, 1_c_int64_t)
        entry_i = i32_immediate(context%session, 0_c_int64_t)
        outer_entry = context%current_block_id
        outer_header = create_liric_block(context%session)
        outer_body = create_liric_block(context%session)
        outer_latch = create_liric_block(context%session)
        outer_exit = create_liric_block(context%session)
        inner_header = create_liric_block(context%session)
        inner_body = create_liric_block(context%session)
        inner_latch = create_liric_block(context%session)
        inner_exit = create_liric_block(context%session)
        if (.not. emit_liric_br(context%session, outer_header, error_msg)) return

        if (.not. enter_liric_block(context, outer_header, error_msg)) return
        i_vreg = reserve_i32_vreg(context%session)
        if (i_vreg <= 0_c_int32_t) then
            error_msg = 'LIRIC could not reserve the matvec row loop vreg'
            return
        end if
        backedge_i = i32_vreg(context%session, i_vreg)
        if (.not. emit_liric_i32_phi(context%session, entry_i, outer_entry, &
                backedge_i, outer_latch, header_i, error_msg)) return
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_SLT, header_i, &
                a_rows, condition, error_msg)) return
        if (.not. emit_liric_condbr(context%session, condition, outer_body, &
                outer_exit, error_msg)) return

        if (.not. enter_liric_block(context, outer_body, error_msg)) return
        entry_k = i32_immediate(context%session, 0_c_int64_t)
        if (.not. emit_liric_br(context%session, inner_header, error_msg)) return

        if (.not. enter_liric_block(context, inner_header, error_msg)) return
        k_vreg = reserve_i32_vreg(context%session)
        sum_vreg = reserve_i32_vreg(context%session)
        if (k_vreg <= 0_c_int32_t .or. sum_vreg <= 0_c_int32_t) then
            error_msg = 'LIRIC could not reserve the matvec reduction vregs'
            return
        end if
        backedge_k = i32_vreg(context%session, k_vreg)
        if (value_kind == VALUE_F32) then
            sum_entry = liric_f32_immediate(context%session, 0.0_c_float)
            sum_backedge = f32_vreg(context%session, sum_vreg)
        else if (value_kind == VALUE_F64) then
            sum_entry = liric_f64_immediate(context%session, 0.0_c_double)
            sum_backedge = f64_vreg(context%session, sum_vreg)
        else
            sum_entry = i32_immediate(context%session, 0_c_int64_t)
            sum_backedge = i32_vreg(context%session, sum_vreg)
        end if
        if (.not. emit_liric_i32_phi(context%session, entry_k, outer_body, &
                backedge_k, inner_latch, header_k, error_msg)) return
        if (.not. emit_liric_phi(context%session, sum_entry, outer_body, &
                sum_backedge, inner_latch, sum_header, error_msg)) return
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_SLT, header_k, &
                a_cols, condition, error_msg)) return
        if (.not. emit_liric_condbr(context%session, condition, inner_body, &
                inner_exit, error_msg)) return

        if (.not. enter_liric_block(context, inner_body, error_msg)) return
        if (.not. emit_i32_binary(context%session, LR_OP_MUL, header_k, a_rows, &
                a_col_offset, error_msg)) return
        if (.not. emit_i32_binary(context%session, LR_OP_ADD, header_i, &
                a_col_offset, a_linear, error_msg)) return
        call load_array_element_at_operand(context, a_sym, a_linear, a_value, &
                                           error_msg)
        if (len_trim(error_msg) > 0) return
        call load_array_element_at_operand(context, b_sym, header_k, b_value, &
                                           error_msg)
        if (len_trim(error_msg) > 0) return
        if (value_kind == VALUE_F32) then
            if (.not. emit_liric_f32_binary(context%session, LR_OP_FMUL, a_value, &
                    b_value, product, error_msg)) return
            if (.not. emit_liric_f32_binary(context%session, LR_OP_FADD, sum_header, &
                    product, next_sum, error_msg)) return
        else if (value_kind == VALUE_F64) then
            if (.not. emit_liric_f64_binary(context%session, LR_OP_FMUL, a_value, &
                    b_value, product, error_msg)) return
            if (.not. emit_liric_f64_binary(context%session, LR_OP_FADD, sum_header, &
                    product, next_sum, error_msg)) return
        else
            if (.not. emit_i32_binary(context%session, LR_OP_MUL, a_value, b_value, &
                    product, error_msg)) return
            if (.not. emit_i32_binary(context%session, LR_OP_ADD, sum_header, &
                    product, next_sum, error_msg)) return
        end if
        if (.not. emit_liric_br(context%session, inner_latch, error_msg)) return

        if (.not. enter_liric_block(context, inner_latch, error_msg)) return
        if (.not. emit_i32_binary_into(context%session, LR_OP_ADD, header_k, one, &
                k_vreg, next_k, error_msg)) return
        if (value_kind == VALUE_F32 .or. value_kind == VALUE_F64) then
            if (.not. emit_real_copy_to(context%session, next_sum, sum_vreg, &
                    sum_backedge, error_msg)) return
        else
            if (.not. emit_i32_copy_to(context%session, next_sum, sum_vreg, &
                    sum_backedge, error_msg)) return
        end if
        if (.not. emit_liric_br(context%session, inner_header, error_msg)) return

        if (.not. enter_liric_block(context, inner_exit, error_msg)) return
        call store_array_element_at_operand(context, target_sym, header_i, &
                                            sum_header, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_liric_br(context%session, outer_latch, error_msg)) return

        if (.not. enter_liric_block(context, outer_latch, error_msg)) return
        if (.not. emit_i32_binary_into(context%session, LR_OP_ADD, header_i, one, &
                i_vreg, next_i, error_msg)) return
        if (.not. emit_liric_br(context%session, outer_header, error_msg)) return

        if (.not. enter_liric_block(context, outer_exit, error_msg)) return
        call set_empty(error_msg)
    end subroutine lower_allocatable_matvec_assignment

    logical function enter_liric_block(context, block_id, error_msg) result(ok)
        type(lowering_context_t), intent(inout) :: context
        integer(c_int32_t), intent(in) :: block_id
        character(len=:), allocatable, intent(out) :: error_msg

        ok = set_liric_block(context%session, block_id, error_msg)
        if (.not. ok) return
        context%current_block_id = block_id
        context%current_block_terminated = .false.
        call set_empty(error_msg)
    end function enter_liric_block

    subroutine allocatable_descriptor_extent_i32(context, symbol_index, dim, &
                                                 extent_i32, error_msg)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index, dim
        type(lr_operand_desc_t), intent(out) :: extent_i32
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: extent_i64

        call emit_alloc_desc_load_extent(context, &
            context%symbols(symbol_index)%allocatable_descriptor_address, dim, &
            extent_i64, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_liric_i64_to_i32(context%session, extent_i64, extent_i32, &
                error_msg)) return
        call set_empty(error_msg)
    end subroutine allocatable_descriptor_extent_i32

    subroutine reallocate_allocatable_extent(symbol_index, n, context, error_msg)
        ! Free the current allocation (a null data pointer marks the array
        ! unallocated, so freeing is unconditional) and allocate n elements.
        integer, intent(in) :: symbol_index
        integer, intent(in) :: n
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: descriptor, data_ptr, n_i32

        descriptor = context%symbols(symbol_index)%allocatable_descriptor_address
        if (.not. emit_ptr_load(context%session, descriptor, data_ptr, error_msg)) &
            return
        if (.not. emit_free(context%session, data_ptr, error_msg)) return
        n_i32 = i32_immediate(context%session, int(n, c_int64_t))
        call lower_allocate_i32_1d_operand(n_i32, symbol_index, context, error_msg)
        if (len_trim(error_msg) > 0) return
        context%symbols(symbol_index)%allocatable_static_size = n
        call set_empty(error_msg)
    end subroutine reallocate_allocatable_extent

    recursive integer function elementwise_expression_extent(arena, node_index, &
                                                             context) result(n)
        ! Compile-time element count of a rank-1 elementwise right-hand side:
        ! the extent of its first array operand. Zero when no operand carries a
        ! known extent, which leaves the destination's current extent in force.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        character(len=:), allocatable :: id_name, err, bin_op
        integer :: sym, bin_left, bin_right, bin_line, bin_col

        n = 0
        if (.not. node_exists(arena, node_index)) return
        if (is_identifier(arena, node_index)) then
            call get_identifier_name(arena, node_index, id_name, err)
            if (len_trim(err) > 0) return
            sym = find_symbol_compat(context, id_name)
            if (sym <= 0) return
            if (.not. is_elementwise_array_operand(context, sym)) return
            if (context%symbols(sym)%array_rank > 1) return
            n = effective_array_extent(context, sym)
            return
        end if
        if (.not. is_binary_op(arena, node_index)) return
        call get_binary_op_info(arena, node_index, bin_op, bin_left, bin_right, &
                                bin_line, bin_col, err)
        if (len_trim(err) > 0) return
        n = elementwise_expression_extent(arena, bin_left, context)
        if (n > 0) return
        n = elementwise_expression_extent(arena, bin_right, context)
    end function elementwise_expression_extent

    logical function is_scalar_broadcast_to_allocatable(arena, node_index, &
                                                        context) result(scalar)
        ! True when the RHS broadcasts into an allocatable: a literal, or an
        ! identifier naming a genuine scalar. An allocatable, array, or pointer
        ! identifier is not a scalar, so a = b (both allocatable) is rejected.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        character(len=:), allocatable :: name, err
        integer :: sym

        scalar = .false.
        if (.not. node_exists(arena, node_index)) return
        select type (n => arena%entries(node_index)%node)
        type is (literal_node)
            scalar = .true.
            return
        end select
        if (.not. is_identifier(arena, node_index)) return
        call get_identifier_name(arena, node_index, name, err)
        if (len_trim(err) > 0) return
        sym = find_symbol_compat(context, name)
        if (sym <= 0) return
        scalar = .not. context%symbols(sym)%is_array .and. &
                 .not. context%symbols(sym)%is_allocatable .and. &
                 .not. context%symbols(sym)%is_pointer
    end function is_scalar_broadcast_to_allocatable

    subroutine lower_allocatable_element_value(arena, expr_index, vk, context, &
                                               value, error_msg)
        ! Lower one element-typed value for an allocatable, by element kind.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: expr_index
        integer, intent(in) :: vk
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg

        select case (vk)
        case (VALUE_F32)
            call lower_f32_expression(arena, expr_index, context, value, error_msg)
        case (VALUE_F64)
            call lower_f64_expression(arena, expr_index, context, value, error_msg)
        case (VALUE_LOGICAL)
            call lower_logical_expression(arena, expr_index, context, value, &
                                          error_msg)
        case default
            call lower_i32_expression(arena, expr_index, context, value, error_msg)
        end select
    end subroutine lower_allocatable_element_value

    logical function emit_allocatable_element_store(context, vk, value, addr, &
                                                    error_msg) result(ok)
        ! Typed store of one allocatable element by element kind.
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: vk
        type(lr_operand_desc_t), intent(in) :: value, addr
        character(len=:), allocatable, intent(out) :: error_msg
        select case (vk)
        case (VALUE_F32)
            ok = emit_liric_f32_store(context%session, value, addr, error_msg)
        case (VALUE_F64)
            ok = emit_liric_f64_store(context%session, value, addr, error_msg)
        case default
            ok = emit_i32_store(context%session, value, addr, error_msg)
        end select
    end function emit_allocatable_element_store

    logical function emit_allocatable_element_load(context, vk, addr, value, &
                                                   error_msg) result(ok)
        ! Typed load of one allocatable element by element kind.
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: vk
        type(lr_operand_desc_t), intent(in) :: addr
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        select case (vk)
        case (VALUE_F32)
            ok = emit_liric_f32_load(context%session, addr, value, error_msg)
        case (VALUE_F64)
            ok = emit_liric_f64_load(context%session, addr, value, error_msg)
        case default
            ok = emit_i32_load(context%session, addr, value, error_msg)
        end select
    end function emit_allocatable_element_load

    subroutine lower_deallocate_statement(arena, node, context, error_msg)
        ! deallocate(a) or deallocate(a, b, c): free each target independently.
        type(ast_arena_t), intent(in) :: arena
        type(deallocate_statement_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg

        if (.not. allocated(node%var_indices) .or. size(node%var_indices) < 1) then
            call unsupported_feature_error('deallocate statement', node%line, &
                node%column, 'deallocate statement has no target variable', &
                error_msg)
            return
        end if
        if (size(node%var_indices) > 1) then
            block
                integer :: k
                do k = 1, size(node%var_indices)
                    call lower_one_deallocate_var(arena, node, &
                        node%var_indices(k), context, error_msg)
                    if (len_trim(error_msg) > 0) return
                end do
            end block
            return
        end if
        call lower_one_deallocate_var(arena, node, node%var_indices(1), context, &
                                      error_msg)
    end subroutine lower_deallocate_statement

    subroutine lower_one_deallocate_var(arena, node, vi, context, error_msg)
        ! Deallocate a single target (shared by single- and multi-variable
        ! deallocate). Handles an allocatable scalar component, a deferred-length
        ! character allocatable, a scalar allocatable, and a rank-1/rank-2 array.
        type(ast_arena_t), intent(in) :: arena
        type(deallocate_statement_node), intent(in) :: node
        integer, intent(in) :: vi
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: name
        integer :: symbol_index
        logical :: handled

        if (.not. node_exists(arena, vi)) then
            error_msg = 'deallocate target is not an AST node'
            return
        end if
        call lower_deallocate_scalar_component(arena, vi, &
                                               context, handled, error_msg)
        if (len_trim(error_msg) > 0) return
        if (handled) return
        if (is_identifier(arena, vi)) then
            call get_identifier_name(arena, vi, name, error_msg)
            if (len_trim(error_msg) > 0) return
        else
            call unsupported_feature_error('deallocate statement', node%line, &
                node%column, 'deallocate target must be a scalar array name', &
                error_msg)
            return
        end if

        symbol_index = find_symbol_compat(context, name)
        if (symbol_index <= 0) then
            call unsupported_feature_error('deallocate statement', node%line, &
                node%column, 'deallocate target is not declared: '//trim(name), &
                error_msg)
            return
        end if
        if (context%symbols(symbol_index)%is_class_pointer) then
            call unsupported_feature_error('deallocate statement', node%line, &
                node%column, 'scalar class pointer ownership and finalization are '// &
                'not supported', error_msg)
            return
        end if
        if (context%symbols(symbol_index)%is_deferred_character) then
            call lower_deallocate_deferred_char(symbol_index, context, error_msg)
            return
        end if
        if (.not. context%symbols(symbol_index)%is_allocatable) then
            call unsupported_feature_error('deallocate statement', node%line, &
                node%column, 'deallocate target is not an allocatable array: '// &
                trim(name), error_msg)
            return
        end if
        if (context%symbols(symbol_index)%array_rank == 0) then
            call lower_deallocate_scalar(symbol_index, context, error_msg)
            return
        end if

        call lower_deallocate_i32(symbol_index, context, error_msg)
    end subroutine lower_one_deallocate_var

    subroutine lower_deallocate_deferred_char(symbol_index, context, error_msg)
        ! deallocate(str) for a character(len=:), allocatable scalar. Release
        ! the storage first: the data slot may hold a static string pointer
        ! (after str = literal) or a stack buffer, which the storage class
        ! distinguishes from heap storage this descriptor owns, so only owned
        ! storage is returned and it is returned exactly once. Then reset the
        ! descriptor to the ABI null state, which is what later reads see.
        use, intrinsic :: iso_c_binding, only: c_int64_t
        integer, intent(in) :: symbol_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg

        call release_owned_character_storage(context, symbol_index, error_msg)
        if (len_trim(error_msg) > 0) return
        if (context%symbols(symbol_index)%has_character_ownership) then
            if (.not. emit_i64_store(context%session, &
                    i64_immediate(context%session, 0_c_int64_t), &
                    context%symbols(symbol_index)%deferred_capacity, &
                    error_msg)) return
        end if
        if (.not. emit_i64_store(context%session, &
                i64_immediate(context%session, 0_c_int64_t), &
                context%symbols(symbol_index)%deferred_data, error_msg)) return
        if (.not. emit_i64_store(context%session, &
                i64_immediate(context%session, 0_c_int64_t), &
                context%symbols(symbol_index)%deferred_length, error_msg)) return
        context%symbols(symbol_index)%has_character_value = .false.
        call set_empty(error_msg)
    end subroutine lower_deallocate_deferred_char

    subroutine lower_deallocate_i32(symbol_index, context, error_msg)
        ! free(descriptor.data); zero descriptor. Works for rank-1 and rank-2.
        ! free(NULL) is a no-op, so an unallocated variable is handled safely.
        integer, intent(in) :: symbol_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: descriptor, data_ptr, zero

        descriptor = context%symbols(symbol_index)%allocatable_descriptor_address
        if (.not. emit_ptr_load(context%session, descriptor, data_ptr, &
                error_msg)) return
        if (.not. emit_free(context%session, data_ptr, error_msg)) return

        call emit_alloc_desc_clear(context, descriptor, error_msg)
    end subroutine lower_deallocate_i32

    subroutine lower_move_alloc(arena, arg_indices, context, error_msg)
        ! move_alloc(from, to): copy the canonical descriptor from -> to, then
        ! return from to the unallocated state. The whole record is copied, so
        ! this works for every rank the descriptor supports.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: arg_indices(:)
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: from_name, to_name
        integer :: from_sym, to_sym, k
        type(lr_operand_desc_t) :: from_desc, to_desc, val, zero

        if (size(arg_indices) /= 2) then
            error_msg = 'move_alloc requires exactly two arguments'
            return
        end if
        if (.not. is_identifier(arena, arg_indices(1)) .or. &
            .not. is_identifier(arena, arg_indices(2))) then
            error_msg = 'move_alloc arguments must be allocatable array names'
            return
        end if
        call get_identifier_name(arena, arg_indices(1), from_name, error_msg)
        if (len_trim(error_msg) > 0) return
        call get_identifier_name(arena, arg_indices(2), to_name, error_msg)
        if (len_trim(error_msg) > 0) return
        from_sym = find_symbol_compat(context, from_name)
        to_sym   = find_symbol_compat(context, to_name)
        if (from_sym <= 0 .or. .not. context%symbols(from_sym)%is_allocatable) then
            error_msg = 'move_alloc: from argument is not an allocatable: '// &
                        trim(from_name)
            return
        end if
        if (to_sym <= 0 .or. .not. context%symbols(to_sym)%is_allocatable) then
            error_msg = 'move_alloc: to argument is not an allocatable: '// &
                        trim(to_name)
            return
        end if

        ! Scalar allocatable derived: the association state is a compile-time
        ! folded heap pointer (element_address + has_address), not a runtime
        ! descriptor. move_alloc hands the instance over and unassociates from.
        if (context%symbols(from_sym)%is_derived .and. &
            context%symbols(from_sym)%array_rank == 0 .and. &
            context%symbols(to_sym)%is_derived .and. &
            context%symbols(to_sym)%array_rank == 0) then
            context%symbols(to_sym)%element_address = &
                context%symbols(from_sym)%element_address
            context%symbols(to_sym)%array_size = &
                context%symbols(from_sym)%array_size
            context%symbols(to_sym)%has_address = &
                context%symbols(from_sym)%has_address
            context%symbols(from_sym)%has_address = .false.
            call set_empty(error_msg)
            return
        end if

        from_desc = context%symbols(from_sym)%allocatable_descriptor_address
        to_desc   = context%symbols(to_sym)%allocatable_descriptor_address
        zero = i64_immediate(context%session, 0_c_int64_t)

        ! Hand the whole descriptor over: base, element metadata, flags, and
        ! every dimension's lower bound, extent, and stride. Copying only a
        ! fixed prefix would silently drop the shape.
        do k = 0, ALLOC_DESCRIPTOR_BYTES/8 - 1
            if (.not. emit_i64_load_at(context%session, from_desc, &
                    int(k*8, c_int64_t), val, error_msg)) return
            if (.not. emit_i64_store_at(context%session, val, to_desc, &
                    int(k*8, c_int64_t), error_msg)) return
        end do

        ! The source becomes unallocated. Ownership moved with the base
        ! pointer, so clearing must not free anything.
        call emit_alloc_desc_clear(context, from_desc, error_msg)
    end subroutine lower_move_alloc

    logical function is_allocatable_element_ref(node, context) result(is_ref)
        ! a(i) or a(i,j) where a is an allocatable, even when FortFront did not
        ! flag the call_or_subscript node as an array access. The subscript count
        ! must match the allocatable rank.
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        integer :: symbol_index, rank, nargs

        is_ref = .false.
        if (.not. allocated(node%name)) return
        if (.not. allocated(node%arg_indices)) return
        if (is_contained_function_reference(node, context)) return
        nargs = size(node%arg_indices)
        if (nargs < 1 .or. nargs > 2) return
        symbol_index = find_symbol_compat(context, node%name)
        if (symbol_index <= 0) return
        if (.not. context%symbols(symbol_index)%is_allocatable) return
        rank = context%symbols(symbol_index)%array_rank
        if (rank == 0) rank = 1   ! pre-B2e rank-1 default
        is_ref = (nargs == rank)
    end function is_allocatable_element_ref

    logical function is_declared_array_element_ref(node, context) result(is_ref)
        ! a(i) or a(i,j) where a is a declared (non-allocatable) array symbol,
        ! even when FortFront did not flag the node as an array access. This gap
        ! shows up for array dummy arguments inside module procedures, where the
        ! usage node keeps is_array_access = .false. The subscript count must
        ! match the array rank and every subscript must be scalar (a range
        ! subscript is an array section, handled elsewhere).
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        integer :: symbol_index, rank, nargs, i

        is_ref = .false.
        if (.not. allocated(node%name)) return
        if (.not. allocated(node%arg_indices)) return
        nargs = size(node%arg_indices)
        if (nargs < 1 .or. nargs > 2) return
        symbol_index = find_symbol_compat(context, node%name)
        if (symbol_index <= 0) return
        if (is_contained_function_reference(node, context)) then
            ! An ASSOCIATE name can be represented by the same call-shaped AST
            ! node as a contained function reference. A declared array symbol
            ! still wins: Fortran cannot declare a function and an array with
            ! the same local name.
            if (.not. context%symbols(symbol_index)%is_array) return
        end if
        if (.not. context%symbols(symbol_index)%is_array) return
        if (context%symbols(symbol_index)%is_allocatable) return
        do i = 1, nargs
            if (.not. node_exists(context%arena, node%arg_indices(i))) return
            select type (arg => context%arena%entries(node%arg_indices(i))%node)
            type is (range_expression_node)
                return
            end select
        end do
        rank = context%symbols(symbol_index)%array_rank
        if (rank == 0) rank = 1
        is_ref = (nargs == rank)
    end function is_declared_array_element_ref

    subroutine lower_allocatable_element_address(arena, node, symbol_index, &
                                                 context, element_address, error_msg)
        ! Runtime element address for a rank-1 or rank-2 integer allocatable.
        ! Descriptor layout (all fields are i64 at the given byte offsets):
        !   offset  0: data pointer
        !   offset  8: lower1
        !   offset 16: upper1
        !   offset 24: lower2  (rank-2 only)
        !   offset 32: upper2  (rank-2 only)
        ! rank-1: linear = i - lower1
        ! rank-2: linear = (i - lower1) + (upper1 - lower1 + 1) * (j - lower2)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        integer, intent(in) :: symbol_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: element_address
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: descriptor, data_ptr
        type(lr_operand_desc_t) :: lower1, upper1, lower2
        type(lr_operand_desc_t) :: idx1_i32, idx1_i64, offset1_i64
        type(lr_operand_desc_t) :: idx2_i32, idx2_i64, offset2_i64
        type(lr_operand_desc_t) :: extent1, linear_i64, byte_offset, tmp
        integer :: rank, arg_count

        rank = context%symbols(symbol_index)%array_rank
        if (rank == 0) rank = 1   ! pre-B2e rank-1 default
        arg_count = 0
        if (allocated(node%arg_indices)) arg_count = size(node%arg_indices)
        if (arg_count /= rank) then
            call unsupported_feature_error('allocatable element access', node%line, &
                node%column, 'subscript count does not match allocatable rank', &
                error_msg)
            return
        end if

        call validate_i32_array_subscript(arena, node%arg_indices(1), context, &
                                          error_msg)
        if (len_trim(error_msg) > 0) return
        call lower_i32_expression(arena, node%arg_indices(1), context, idx1_i32, &
                                  error_msg)
        if (len_trim(error_msg) > 0) return

        descriptor = context%symbols(symbol_index)%allocatable_descriptor_address
        if (.not. emit_ptr_load(context%session, descriptor, data_ptr, error_msg)) &
            return
        call emit_alloc_desc_load_lower(context, descriptor, 1, lower1, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_liric_i32_to_i64(context%session, idx1_i32, idx1_i64, &
                error_msg)) return
        if (.not. emit_i64_binary(context%session, LR_OP_SUB, idx1_i64, lower1, &
                offset1_i64, error_msg)) return

        if (rank == 1) then
            linear_i64 = offset1_i64
        else
            ! rank == 2: linear = offset1 + extent1 * (j - lower2)
            ! The descriptor stores the extent directly, so the old
            ! upper - lower + 1 reconstruction is gone.
            call emit_alloc_desc_load_extent(context, descriptor, 1, extent1, &
                                             error_msg)
            if (len_trim(error_msg) > 0) return
            call emit_alloc_desc_load_lower(context, descriptor, 2, lower2, &
                                            error_msg)
            if (len_trim(error_msg) > 0) return
            call validate_i32_array_subscript(arena, node%arg_indices(2), context, &
                                              error_msg)
            if (len_trim(error_msg) > 0) return
            call lower_i32_expression(arena, node%arg_indices(2), context, &
                                      idx2_i32, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_liric_i32_to_i64(context%session, idx2_i32, idx2_i64, &
                    error_msg)) return
            if (.not. emit_i64_binary(context%session, LR_OP_SUB, idx2_i64, lower2, &
                    offset2_i64, error_msg)) return
            ! linear = offset1 + extent1 * offset2
            ! Use tmp to avoid aliasing (linear_i64 as both rhs and result is UB).
            if (.not. emit_i64_binary(context%session, LR_OP_MUL, extent1, &
                    offset2_i64, tmp, error_msg)) return
            if (.not. emit_i64_binary(context%session, LR_OP_ADD, offset1_i64, &
                    tmp, linear_i64, error_msg)) return
        end if

        if (.not. emit_i64_binary(context%session, LR_OP_MUL, linear_i64, &
                i64_immediate(context%session, &
                    allocatable_elem_size(context%symbols(symbol_index)%value_kind)), &
                byte_offset, error_msg)) return
        if (.not. emit_ptr_offset_dyn(context%session, data_ptr, byte_offset, &
                element_address, error_msg)) return
        call set_empty(error_msg)
    end subroutine lower_allocatable_element_address

    subroutine init_allocatable_character_blanks(context, symbol_index, error_msg)
        ! Blank-fill every slot of a freshly allocated 1-D fixed-length
        ! character allocatable array, mirroring init_character_array_blanks
        ! for the fixed-size sibling. Requires a compile-time constant extent
        ! (set by record_allocatable_static_size); a runtime-only extent
        ! leaves slots as raw malloc bytes (null pointers are not guaranteed).
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=64) :: string_name
        character(len=:), allocatable :: blanks
        type(lr_operand_desc_t) :: blank_ptr, descriptor, data_ptr, slot_addr
        integer :: n, character_length, i

        call set_empty(error_msg)
        n = context%symbols(symbol_index)%allocatable_static_size
        if (n <= 0) return
        character_length = context%symbols(symbol_index)%character_length

        allocate (character(len=character_length) :: blanks)
        blanks = repeat(' ', character_length)
        context%string_literal_count = context%string_literal_count + 1
        string_name = ffc_unit_global_name( &
            context, 'char.', context%string_literal_count)
        call materialize_liric_string(context%session, trim(string_name), blanks, &
                                      blank_ptr, error_msg)
        if (len_trim(error_msg) > 0) return

        descriptor = context%symbols(symbol_index)%allocatable_descriptor_address
        if (.not. emit_ptr_load(context%session, descriptor, data_ptr, error_msg)) &
            return
        do i = 0, n - 1
            if (.not. emit_ptr_offset(context%session, data_ptr, &
                    int(i, c_int64_t) * allocatable_elem_size(VALUE_CHARACTER), &
                    slot_addr, error_msg)) return
            if (.not. emit_ptr_store(context%session, blank_ptr, slot_addr, &
                                     error_msg)) return
        end do
        call set_empty(error_msg)
    end subroutine init_allocatable_character_blanks

    subroutine lower_allocatable_character_element_assignment(arena, node, &
                                                              target, context, &
                                                              error_msg)
        ! a(i) = "text" for a 1-D fixed-length character allocatable array:
        ! materialize the normalized literal and store its pointer in the
        ! runtime-addressed slot. Mirrors
        ! lower_character_array_element_assignment for the fixed-size sibling.
        type(ast_arena_t), intent(in) :: arena
        type(assignment_node), intent(in) :: node
        type(call_or_subscript_node), intent(in) :: target
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: symbol_index
        type(lr_operand_desc_t) :: slot_addr, string_ptr
        character(len=:), allocatable :: literal_text
        character(len=64) :: string_name
        logical :: fold_ok

        if (.not. node_exists(arena, node%value_index)) then
            error_msg = 'character allocatable assignment value does not exist'
            return
        end if
        call concat_character_literals(arena, node%value_index, literal_text, &
                                       fold_ok)
        if (.not. fold_ok) then
            call unsupported_feature_error('character allocatable assignment', &
                node%line, node%column, 'direct LIRIC session supports only '// &
                'character-literal assignment into a character allocatable '// &
                'element', error_msg)
            return
        end if

        if (.not. allocated(target%name)) then
            error_msg = 'character allocatable assignment target has no name'
            return
        end if
        symbol_index = find_symbol_compat(context, target%name)
        if (symbol_index <= 0) then
            error_msg = 'character allocatable array was not declared: '// &
                        trim(target%name)
            return
        end if
        call lower_allocatable_element_address(arena, target, symbol_index, &
                                               context, slot_addr, error_msg)
        if (len_trim(error_msg) > 0) return

        call normalize_character_literal(literal_text, &
            context%symbols(symbol_index)%character_length)
        context%string_literal_count = context%string_literal_count + 1
        string_name = ffc_unit_global_name( &
            context, 'char.', context%string_literal_count)
        call materialize_liric_string(context%session, trim(string_name), &
                                      literal_text, string_ptr, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_ptr_store(context%session, string_ptr, slot_addr, &
                                 error_msg)) then
            error_msg = 'char_alloc_assign_store: '//error_msg
            return
        end if
        call set_empty(error_msg)
    end subroutine lower_allocatable_character_element_assignment

    subroutine lower_allocatable_character_element_print(arena, node, context, &
                                                          use_value, error_msg)
        ! print *, a(i) for a 1-D fixed-length character allocatable array:
        ! load the slot pointer and reuse the scalar string print path.
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        logical, intent(in) :: use_value
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: symbol_index
        type(lr_operand_desc_t) :: slot_addr, string_ptr

        if (.not. allocated(node%name)) then
            error_msg = 'character allocatable print target has no name'
            return
        end if
        symbol_index = find_symbol_compat(context, node%name)
        if (symbol_index <= 0) then
            error_msg = 'character allocatable array was not declared: '// &
                        trim(node%name)
            return
        end if
        call lower_allocatable_element_address(arena, node, symbol_index, &
                                               context, slot_addr, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_ptr_load(context%session, slot_addr, string_ptr, &
                                error_msg)) then
            error_msg = 'char_alloc_print_load: '//error_msg
            return
        end if
        if (use_value) then
            if (.not. emit_liric_print_string_operand_value(context%session, &
                    context%str_print_format_id, string_ptr, error_msg)) return
        else
            if (.not. emit_liric_print_string_operand(context%session, &
                    context%str_print_format_id, string_ptr, error_msg)) return
        end if
        call set_empty(error_msg)
    end subroutine lower_allocatable_character_element_print

    subroutine emit_allocatable_character_print_items(context, sym, handled, &
                                                       error_msg)
        ! Whole-array print of a rank-1 fixed-length character allocatable
        ! whose extent is known at compile time: one leading list-directed
        ! space, then every element with no inter-element separator (matching
        ! gfortran's character list output).
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: sym
        logical, intent(out) :: handled
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: descriptor, data_ptr, slot_addr, string_ptr
        integer(c_int64_t) :: stride, k, n

        handled = .false.
        call set_empty(error_msg)
        if (context%symbols(sym)%array_rank == 2) return
        n = int(context%symbols(sym)%allocatable_static_size, c_int64_t)
        if (n <= 0_c_int64_t) return

        handled = .true.
        stride = allocatable_elem_size(VALUE_CHARACTER)
        descriptor = context%symbols(sym)%allocatable_descriptor_address
        if (.not. emit_ptr_load(context%session, descriptor, data_ptr, error_msg)) &
            return
        if (.not. emit_liric_print_space(context%session, error_msg)) return
        do k = 0_c_int64_t, n - 1_c_int64_t
            if (.not. emit_ptr_offset(context%session, data_ptr, k * stride, &
                                      slot_addr, error_msg)) return
            if (.not. emit_ptr_load(context%session, slot_addr, string_ptr, &
                                    error_msg)) return
            if (.not. emit_liric_print_string_operand_value(context%session, &
                    context%str_print_format_id, string_ptr, error_msg)) return
        end do
        call set_empty(error_msg)
    end subroutine emit_allocatable_character_print_items

    pure integer(c_int64_t) function allocatable_elem_size(value_kind) result(sz)
        ! Byte stride of one allocatable element. i32/f32/logical occupy 4 bytes;
        ! i64/f64/a character slot pointer occupy 8. Logical uses an i32 slot,
        ! matching the array path.
        integer, intent(in) :: value_kind
        select case (value_kind)
        case (VALUE_C8)
            sz = 16_c_int64_t
        case (VALUE_C4)
            sz = 8_c_int64_t
        case (VALUE_F64, VALUE_I64, VALUE_CHARACTER)
            sz = 8_c_int64_t
        case default
            sz = 4_c_int64_t
        end select
    end function allocatable_elem_size

    subroutine declare_allocatable(context, name, rank, value_kind, error_msg, &
                                   element_bytes)
        ! Stack-allocate a 40-byte descriptor and zero it. Layout:
        !   offset  0: data pointer (i64)
        !   offset  8: lower1 (i64)
        !   offset 16: upper1 (i64)
        !   offset 24: lower2 (i64, rank-2; 0 for rank-1)
        !   offset 32: upper2 (i64, rank-2; 0 for rank-1)
        ! data == 0 marks the array unallocated. The element kind lives on the
        ! symbol's value_kind; the descriptor itself is kind-agnostic.
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: name
        integer, intent(in) :: rank
        integer, intent(in) :: value_kind
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int64_t), intent(in), optional :: element_bytes
        type(lr_operand_desc_t) :: descriptor, zero
        integer :: index, k

        index = find_symbol_compat(context, name)
        if (index > 0) then
            ! The allocatable array function result was pre-registered onto the
            ! caller's descriptor (param 0). Its body declaration re-states the
            ! type; adopt it rather than flagging a duplicate.
            if (index == context%current_function_result_index) then
                if (context%symbols(index)%is_allocatable .and. &
                    context%symbols(index)%is_array) then
                    context%symbols(index)%value_kind = value_kind
                    context%symbols(index)%is_array = rank > 0
                    context%symbols(index)%array_rank = rank
                    call set_empty(error_msg)
                    return
                end if
            end if
            ! An allocatable dummy argument: the parameter symbol was already
            ! registered generically at call-entry binding. The incoming
            ! parameter pointer IS the caller's own descriptor address
            ! (passed by allocatable_array_actual_descriptor_address), so
            ! adopt it directly instead of allocating fresh storage -- writes
            ! to the descriptor (allocate/deallocate) then land in the
            ! caller's frame too.
            if (context%symbols(index)%is_parameter .and. &
                .not. context%symbols(index)%is_array .and. &
                .not. context%symbols(index)%is_allocatable) then
                context%symbols(index)%value_kind = value_kind
                context%symbols(index)%is_allocatable = .true.
                context%symbols(index)%is_array = rank > 0
                context%symbols(index)%array_rank = rank
                context%symbols(index)%allocatable_descriptor_address = &
                    context%symbols(index)%address
                call set_empty(error_msg)
                return
            end if
            error_msg = 'duplicate allocatable declaration: '//trim(name)
            return
        end if
        call grow_symbols(context)

        if (.not. emit_alloca_bytes(context%session, &
                i64_immediate(context%session, &
                    int(ALLOC_DESCRIPTOR_BYTES, c_int64_t)), descriptor, &
                error_msg)) return
        ! Zero the whole descriptor before installing the header: an
        ! unallocated allocatable must read as null base, no flags, and zero
        ! extents, never as whatever the stack slot happened to hold.
        zero = i64_immediate(context%session, 0_c_int64_t)
        do k = 0, ALLOC_DESCRIPTOR_BYTES/8 - 1
            if (.not. emit_i64_store_at(context%session, zero, descriptor, &
                    int(k*8, c_int64_t), error_msg)) return
        end do
        if (rank >= 1) then
            call emit_alloc_desc_header(context, descriptor, value_kind, rank, &
                                        error_msg, element_bytes)
            if (len_trim(error_msg) > 0) return
        end if

        index = context%symbol_count + 1
        context%symbols(index)%name = trim(name)
        context%symbols(index)%value_kind = value_kind
        context%symbols(index)%is_allocatable = .true.
        context%symbols(index)%is_array = rank > 0
        context%symbols(index)%array_rank = rank
        if (rank >= 1) then
            context%symbols(index)%array_lower_bound = 1
            context%symbols(index)%array_dim_lowers(1) = 1
        end if
        context%symbols(index)%allocatable_descriptor_address = descriptor
        context%symbol_count = index
        call set_empty(error_msg)
    end subroutine declare_allocatable

    subroutine lower_allocated(arena, arg_indices, context, value, error_msg)
        ! allocated(a): true when a's data pointer is non-null. Works for a
        ! rank-1/2 allocatable array (data ptr in allocatable_descriptor_address
        ! offset 0) and a deferred-length character allocatable (data ptr in
        ! deferred_data).
        type(ast_arena_t), intent(in) :: arena
        integer, allocatable, intent(in) :: arg_indices(:)
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: name
        type(lr_operand_desc_t) :: data_ptr
        integer :: symbol_index
        logical :: handled
        integer :: actual_index

        if (.not. allocated(arg_indices)) then
            error_msg = 'allocated requires exactly one argument'
            return
        end if
        if (size(arg_indices) /= 1) then
            error_msg = 'allocated requires exactly one argument'
            return
        end if
        call unwrap_allocated_keyword_argument(arena, arg_indices(1), actual_index, &
                                               error_msg)
        if (len_trim(error_msg) > 0) return
        call lower_allocated_scalar_component(arena, actual_index, context, &
                                              value, handled, error_msg)
        if (len_trim(error_msg) > 0) return
        if (handled) return
        if (.not. is_identifier(arena, actual_index)) then
            error_msg = 'allocated argument must be an allocatable variable'
            return
        end if
        call identifier_name(arena, actual_index, name, error_msg)
        if (len_trim(error_msg) > 0) return

        symbol_index = find_symbol_compat(context, name)
        if (symbol_index <= 0) then
            error_msg = 'allocated argument was not declared: '//trim(name)
            return
        end if
        if (context%symbols(symbol_index)%is_deferred_character) then
            if (.not. emit_ptr_load(context%session, &
                    context%symbols(symbol_index)%deferred_data, data_ptr, &
                    error_msg)) return
        else if (context%symbols(symbol_index)%is_allocatable .and. &
                 context%symbols(symbol_index)%array_rank == 0) then
            ! A scalar allocatable folds its straight-line association state,
            ! mirroring associated()'s compile-time limitation.
            if (context%symbols(symbol_index)%has_address) then
                value = i32_immediate(context%session, 1_c_int64_t)
            else
                value = i32_immediate(context%session, 0_c_int64_t)
            end if
            call set_empty(error_msg)
            return
        else if (context%symbols(symbol_index)%is_allocatable) then
            if (.not. emit_ptr_load(context%session, &
                    context%symbols(symbol_index)%allocatable_descriptor_address, &
                    data_ptr, error_msg)) return
        else
            error_msg = 'allocated argument is not an allocatable variable: '// &
                        trim(name)
            return
        end if
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_NE, data_ptr, &
                null_ptr_operand(context), value, error_msg)) return
        call set_empty(error_msg)
    end subroutine lower_allocated

    subroutine unwrap_allocated_keyword_argument(arena, arg_index, actual_index, &
                                                  error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: arg_index
        integer, intent(out) :: actual_index
        character(len=:), allocatable, intent(out) :: error_msg

        actual_index = arg_index
        call set_empty(error_msg)
        if (.not. node_exists(arena, arg_index)) return
        select type (arg => arena%entries(arg_index)%node)
        type is (assignment_node)
            ! Function-reference actuals use this assignment shape without
            ! setting is_keyword_argument; an assignment cannot be a normal
            ! expression in an intrinsic actual-argument slot.
            actual_index = arg%value_index
        end select
    end subroutine unwrap_allocated_keyword_argument

    subroutine lower_allocated_scalar_component(arena, arg_index, context, value, &
                                                handled, error_msg)
        ! allocated(x%c) for a scalar allocatable component: true when its inline
        ! data pointer is non-null. Leaves handled .false. for any other argument.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: arg_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        logical, intent(out) :: handled
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: slot_address, data_ptr

        handled = .false.
        call set_empty(error_msg)
        if (.not. node_exists(arena, arg_index)) return
        select type (arg => arena%entries(arg_index)%node)
        type is (component_access_node)
            if (.not. component_access_is_allocatable(arena, arg, context) .and. &
                .not. component_access_is_alloc_array(arena, arg, context)) return
            call lower_derived_component_address(arena, arg, context, &
                slot_address, error_msg, 'allocated component argument')
            if (len_trim(error_msg) > 0) return
            if (.not. emit_ptr_load(context%session, slot_address, data_ptr, &
                                    error_msg)) return
            if (.not. emit_liric_i32_icmp(context%session, LR_CMP_NE, data_ptr, &
                    null_ptr_operand(context), value, error_msg)) return
            handled = .true.
        end select
    end subroutine lower_allocated_scalar_component

    subroutine lower_deallocate_scalar_component(arena, target_index, context, &
                                                 handled, error_msg)
        ! deallocate(x%c) for a scalar allocatable component: free the current
        ! data slot and null the inline descriptor. Leaves handled .false. for
        ! any other target.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: target_index
        type(lowering_context_t), intent(inout) :: context
        logical, intent(out) :: handled
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: slot_address, data_ptr

        handled = .false.
        call set_empty(error_msg)
        if (.not. node_exists(arena, target_index)) return
        select type (target => arena%entries(target_index)%node)
        type is (component_access_node)
            block
                logical :: is_alloc_array
                is_alloc_array = component_access_is_alloc_array(arena, target, &
                                                                context)
                if (.not. component_access_is_allocatable(arena, target, context) &
                    .and. .not. is_alloc_array) return
                call lower_derived_component_address(arena, target, context, &
                    slot_address, error_msg, 'deallocate component target')
                if (len_trim(error_msg) > 0) return
                if (.not. emit_ptr_load(context%session, slot_address, data_ptr, &
                                        error_msg)) return
                if (.not. emit_free(context%session, data_ptr, error_msg)) return
                if (.not. emit_ptr_store(context%session, null_ptr_operand(context), &
                                         slot_address, error_msg)) return
                if (is_alloc_array .or. &
                    component_access_is_alloc_char(arena, target, context)) then
                    ! Both an array descriptor and a deferred-length character
                    ! descriptor carry an i64 extent/length at +8; zero it so
                    ! the component reads as empty and unallocated (#402).
                    if (.not. emit_i64_store_at(context%session, &
                            i64_immediate(context%session, 0_c_int64_t), &
                            slot_address, 8_c_int64_t, error_msg)) return
                end if
                handled = .true.
            end block
        end select
    end subroutine lower_deallocate_scalar_component
