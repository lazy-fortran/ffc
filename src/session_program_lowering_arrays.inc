    logical function dim_is_assumed_shape(arena, dim_index) result(is_assumed)
        ! a(:) is parsed as a range_expression with no start/end/stride, or as
        ! an array_bounds_node flagged is_assumed_shape. Either marks one
        ! assumed-shape dimension whose extent comes from the caller's actual.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: dim_index

        is_assumed = .false.
        if (.not. node_exists(arena, dim_index)) return
        select type (dim_node => arena%entries(dim_index)%node)
        type is (range_expression_node)
            is_assumed = dim_node%start_index <= 0 .and. &
                         dim_node%end_index <= 0 .and. dim_node%stride_index <= 0
        type is (array_bounds_node)
            is_assumed = dim_node%is_assumed_shape
        end select
    end function dim_is_assumed_shape

    logical function declaration_is_assumed_shape(node, context) result(is_assumed)
        ! A declaration is assumed-shape when every dimension is a colon a(:).
        type(declaration_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        integer :: i

        is_assumed = .false.
        if (.not. allocated(node%dimension_indices)) return
        if (size(node%dimension_indices) < 1) return
        do i = 1, size(node%dimension_indices)
            if (.not. dim_is_assumed_shape(context%arena, &
                                           node%dimension_indices(i))) return
        end do
        is_assumed = .true.
    end function declaration_is_assumed_shape

    logical function declaration_is_runtime_rank1(node, context) result(is_runtime)
        ! A rank-1 explicit-shape array whose single extent is a runtime integer
        ! expression (a dummy-argument value such as dimension(n)) rather than a
        ! compile-time constant. Allocatable and assumed-shape declarations are
        ! excluded; they have their own lowering paths.
        type(declaration_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        integer :: dim_index
        integer(c_int64_t) :: folded
        character(len=:), allocatable :: fold_error

        is_runtime = .false.
        if (.not. node%is_array) return
        if (node%is_allocatable) return
        if (.not. allocated(node%dimension_indices)) return
        if (size(node%dimension_indices) /= 1) return
        if (declaration_is_assumed_shape(node, context)) return
        dim_index = node%dimension_indices(1)
        if (.not. node_exists(context%arena, dim_index)) return
        select type (dim_node => context%arena%entries(dim_index)%node)
        type is (range_expression_node)
            call eval_i32_constant(context%arena, dim_node%end_index, context, &
                                   folded, fold_error)
        class default
            call eval_i32_constant(context%arena, dim_index, context, folded, &
                                   fold_error)
        end select
        is_runtime = len_trim(fold_error) > 0
    end function declaration_is_runtime_rank1

    logical function declaration_rebinds_runtime_array_result(node, context) &
            result(is_rebind)
        ! True when this body array declaration names the pre-bound array function
        ! result symbol and has a runtime (non-foldable) rank-1 extent. Such a
        ! declaration rebinds the sret buffer view rather than allocating storage,
        ! so its bounds must not be folded at compile time.
        type(declaration_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        integer :: index, k

        is_rebind = .false.
        if (.not. declaration_is_runtime_rank1(node, context)) return
        if (allocated(node%var_name)) then
            index = find_symbol(context, node%var_name)
            if (index > 0) then
                if (context%symbols(index)%is_array_result) is_rebind = .true.
            end if
            if (is_rebind) return
        end if
        if (node%is_multi_declaration .and. allocated(node%var_names)) then
            do k = 1, size(node%var_names)
                index = find_symbol(context, node%var_names(k))
                if (index > 0) then
                    if (context%symbols(index)%is_array_result) is_rebind = .true.
                end if
                if (is_rebind) return
            end do
        end if
    end function declaration_rebinds_runtime_array_result

    logical function declaration_is_assumed_rank(node, context) result(is_rank)
        ! An assumed-rank dummy arr(..) parses to a single array_bounds_node
        ! flagged is_assumed_rank. Its runtime rank comes from the caller's
        ! actual, so it bypasses compile-time bound folding (#273).
        type(declaration_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        integer :: dim_index

        is_rank = .false.
        if (.not. allocated(node%dimension_indices)) return
        if (size(node%dimension_indices) /= 1) return
        dim_index = node%dimension_indices(1)
        if (.not. node_exists(context%arena, dim_index)) return
        select type (dim_node => context%arena%entries(dim_index)%node)
        type is (array_bounds_node)
            is_rank = dim_node%is_assumed_rank
        end select
    end function declaration_is_assumed_rank

    subroutine lower_assumed_rank_declaration(node, context, value_kind, &
                                              error_msg)
        ! Bind an assumed-rank dummy arr(..) to its parameter base and to the
        ! rank of the caller's actual. ffc resolves that rank statically from
        ! the single call site (matching the assumed-shape extent recovery), so
        ! a later select rank dispatches to the matching arm at compile time.
        type(declaration_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: value_kind
        character(len=:), allocatable, intent(out) :: error_msg

        call set_empty(error_msg)
        if (node%is_multi_declaration .and. allocated(node%var_names)) then
            call unsupported_feature_error('assumed-rank array', node%line, &
                node%column, 'one assumed-rank dummy per declaration', error_msg)
            return
        end if
        if (.not. allocated(node%var_name)) then
            error_msg = 'assumed-rank declaration did not expose a variable name'
            return
        end if
        call bind_one_assumed_rank(node, context, node%var_name, value_kind, &
                                   error_msg)
    end subroutine lower_assumed_rank_declaration

    subroutine bind_one_assumed_rank(node, context, name, value_kind, error_msg)
        type(declaration_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: name
        integer, intent(in) :: value_kind
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: index, actual_rank
        logical :: ok
        type(lr_operand_desc_t) :: param_base

        if (.not. allocated(context%current_proc_name)) then
            call unsupported_feature_error('assumed-rank array', node%line, &
                node%column, 'assumed-rank dummies are supported only inside a '// &
                'contained procedure', error_msg)
            return
        end if
        index = find_symbol(context, name)
        ! Guard the symbol-table access on its own line: .or. does not
        ! short-circuit, so index <= 0 must gate the array read separately.
        if (index <= 0) then
            call unsupported_feature_error('assumed-rank array', node%line, &
                node%column, 'assumed-rank declaration must name a dummy '// &
                'argument', error_msg)
            return
        end if
        if (.not. context%symbols(index)%is_parameter) then
            call unsupported_feature_error('assumed-rank array', node%line, &
                node%column, 'assumed-rank declaration must name a dummy '// &
                'argument', error_msg)
            return
        end if
        param_base = context%symbols(index)%address
        call resolve_assumed_rank(context, context%current_proc_name, name, &
                                  actual_rank, ok)
        if (.not. ok) then
            call unsupported_feature_error('assumed-rank array', node%line, &
                node%column, 'assumed-rank dummy rank must come from a '// &
                'whole-array or scalar actual', error_msg)
            return
        end if
        context%symbols(index)%value_kind = value_kind
        context%symbols(index)%is_array = actual_rank > 0
        context%symbols(index)%array_rank = actual_rank
        context%symbols(index)%element_address = param_base
        call set_empty(error_msg)
    end subroutine bind_one_assumed_rank

    subroutine resolve_assumed_rank(context, proc_name, param_name, rank, ok)
        ! Recover an assumed-rank dummy's runtime rank from the caller's actual:
        ! find the call to proc_name and the rank of the actual at the dummy's
        ! position. A whole-array actual contributes its declared rank; a scalar
        ! actual contributes rank 0.
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: proc_name
        character(len=*), intent(in) :: param_name
        integer, intent(out) :: rank
        logical, intent(out) :: ok
        integer :: param_pos, n
        character(len=:), allocatable :: actual_name
        type(declaration_node), pointer :: decl

        ok = .false.
        rank = 0
        param_pos = proc_param_position(context%arena, proc_name, param_name)
        if (param_pos <= 0) return
        call find_call_actual_identifier(context%arena, proc_name, param_pos, &
                                         actual_name)
        if (.not. allocated(actual_name)) return
        if (len_trim(actual_name) == 0) return
        do n = 1, context%arena%size
            if (.not. node_exists(context%arena, n)) cycle
            decl => declaration_named(context%arena, n, actual_name)
            if (.not. associated(decl)) cycle
            if (decl%is_array .and. allocated(decl%dimension_indices)) then
                rank = size(decl%dimension_indices)
            else
                rank = 0
            end if
            ok = .true.
            return
        end do
    end subroutine resolve_assumed_rank

    subroutine lower_assumed_shape_declaration(node, context, value_kind, &
                                               error_msg)
        ! Bind an assumed-shape dummy a(:) (or a(:,:)) to its parameter base and
        ! to the caller-derived extent, then reuse the fixed-array machinery for
        ! element access, size(), and whole-array operations.
        type(declaration_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: value_kind
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: dim_count, i

        dim_count = 0
        if (allocated(node%dimension_indices)) &
            dim_count = size(node%dimension_indices)
        if (dim_count < 1 .or. dim_count > 2) then
            call unsupported_feature_error('assumed-shape array', node%line, &
                node%column, 'direct LIRIC session supports rank-1 and rank-2 '// &
                'assumed-shape dummies', error_msg)
            return
        end if

        if (node%is_multi_declaration .and. allocated(node%var_names)) then
            do i = 1, size(node%var_names)
                call bind_one_assumed_shape(node, context, node%var_names(i), &
                                            dim_count, value_kind, error_msg)
                if (len_trim(error_msg) > 0) return
            end do
        else if (allocated(node%var_name)) then
            call bind_one_assumed_shape(node, context, node%var_name, dim_count, &
                                        value_kind, error_msg)
        else
            error_msg = 'assumed-shape declaration did not expose a variable name'
        end if
    end subroutine lower_assumed_shape_declaration

    subroutine bind_one_assumed_shape(node, context, name, dim_count, &
                                      value_kind, error_msg)
        type(declaration_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: name
        integer, intent(in) :: dim_count
        integer, intent(in) :: value_kind
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: extents(2), total, i
        logical :: ok

        if (.not. allocated(context%current_proc_name)) then
            call unsupported_feature_error('assumed-shape array', node%line, &
                node%column, 'assumed-shape dummies are supported only inside '// &
                'a contained procedure', error_msg)
            return
        end if
        extents = 0
        call resolve_assumed_shape_extents(context, context%current_proc_name, &
                                           name, dim_count, extents, ok)
        if (.not. ok) then
            call unsupported_feature_error('assumed-shape array', node%line, &
                node%column, 'assumed-shape dummy extent must come from a '// &
                'whole-array actual of compile-time size', error_msg)
            return
        end if
        total = 1
        do i = 1, dim_count
            total = total * extents(i)
        end do
        call define_declared_array_symbol(context, node, name, 1, total, &
                                          value_kind, error_msg, &
                                          assumed_extents=extents(1:dim_count))
    end subroutine bind_one_assumed_shape

    subroutine resolve_assumed_shape_extents(context, proc_name, param_name, &
                                             dim_count, extents, ok)
        ! Recover an assumed-shape dummy's per-dimension extents from the
        ! caller's actual. Find a call to proc_name whose actual at the dummy's
        ! position is a whole-array identifier, then read that array's declared
        ! compile-time shape. ok stays false when no constant extent is found.
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: proc_name
        character(len=*), intent(in) :: param_name
        integer, intent(in) :: dim_count
        integer, intent(out) :: extents(:)
        logical, intent(out) :: ok
        integer :: param_pos
        character(len=:), allocatable :: actual_name

        ok = .false.
        extents = 0
        param_pos = proc_param_position(context%arena, proc_name, param_name)
        if (param_pos <= 0) return
        call find_call_actual_identifier(context%arena, proc_name, param_pos, &
                                         actual_name)
        if (.not. allocated(actual_name)) return
        if (len_trim(actual_name) == 0) return
        call array_decl_extents(context, actual_name, dim_count, extents, ok)
    end subroutine resolve_assumed_shape_extents

    integer function proc_param_position(arena, proc_name, param_name) &
            result(pos)
        ! 1-based position of param_name among proc_name's dummy arguments.
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: proc_name
        character(len=*), intent(in) :: param_name
        integer :: n, i
        character(len=:), allocatable :: node_type, name, name_err
        type(function_def_node), pointer :: fn_node
        type(subroutine_def_node), pointer :: sb_node
        integer, allocatable :: params(:)

        pos = 0
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            node_type = get_node_type_at(arena, n)
            if (node_type == 'subroutine_def_node' .or. &
                node_type == 'subroutine_def') then
                sb_node => get_node_as_subroutine_def(arena, n)
                if (.not. associated(sb_node)) cycle
                if (.not. allocated(sb_node%name)) cycle
                if (.not. same_name(sb_node%name, proc_name)) cycle
                if (.not. allocated(sb_node%param_indices)) return
                params = sb_node%param_indices
            else if (node_type == 'function_def_node' .or. &
                     node_type == 'function_def') then
                fn_node => get_node_as_function_def(arena, n)
                if (.not. associated(fn_node)) cycle
                if (.not. allocated(fn_node%name)) cycle
                if (.not. same_name(fn_node%name, proc_name)) cycle
                if (.not. allocated(fn_node%param_indices)) return
                params = fn_node%param_indices
            else
                cycle
            end if
            do i = 1, size(params)
                call parameter_name(arena, params(i), name, name_err)
                if (len_trim(name_err) > 0) cycle
                if (trim(name) == trim(param_name)) then
                    pos = i
                    return
                end if
            end do
            return
        end do
    end function proc_param_position

    subroutine find_call_actual_identifier(arena, proc_name, param_pos, &
                                           actual_name)
        ! Scan the arena for a call to proc_name and return the identifier name
        ! of its param_pos-th actual argument, if that actual is a bare name.
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: proc_name
        integer, intent(in) :: param_pos
        character(len=:), allocatable, intent(out) :: actual_name
        integer :: n
        character(len=:), allocatable :: call_name, name_err
        integer, allocatable :: arg_indices(:)

        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            if (.not. call_targets_proc(arena, n, proc_name, call_name, &
                                        arg_indices)) cycle
            if (.not. allocated(arg_indices)) cycle
            if (param_pos < 1 .or. param_pos > size(arg_indices)) cycle
            if (.not. is_identifier(arena, arg_indices(param_pos))) cycle
            call get_identifier_name(arena, arg_indices(param_pos), actual_name, &
                                     name_err)
            if (len_trim(name_err) > 0) cycle
            return
        end do
    end subroutine find_call_actual_identifier

    logical function call_targets_proc(arena, node_index, proc_name, call_name, &
                                       arg_indices) result(matches)
        ! True when node_index is a subroutine-call statement or a
        ! call_or_subscript expression naming proc_name; returns its arguments.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        character(len=*), intent(in) :: proc_name
        character(len=:), allocatable, intent(out) :: call_name
        integer, allocatable, intent(out) :: arg_indices(:)
        character(len=:), allocatable :: err

        matches = .false.
        if (is_subroutine_call_statement(arena, node_index)) then
            call get_subroutine_call_name(arena, node_index, call_name, err)
            if (len_trim(err) > 0) return
            if (.not. same_name(call_name, proc_name)) return
            call get_subroutine_call_arg_indices(arena, node_index, arg_indices, &
                                                 err)
            if (len_trim(err) > 0) return
            matches = .true.
            return
        end if
        select type (cnode => arena%entries(node_index)%node)
        type is (call_or_subscript_node)
            if (.not. allocated(cnode%name)) return
            if (.not. same_name(cnode%name, proc_name)) return
            if (allocated(cnode%arg_indices)) arg_indices = cnode%arg_indices
            matches = .true.
        end select
    end function call_targets_proc

    subroutine array_decl_extents(context, var_name, dim_count, extents, ok)
        ! Find var_name's array declaration anywhere in the arena and compute its
        ! per-dimension compile-time extents. ok stays false on any non-constant
        ! bound or a rank mismatch with the assumed-shape dummy.
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: var_name
        integer, intent(in) :: dim_count
        integer, intent(out) :: extents(:)
        logical, intent(out) :: ok
        integer :: n, lower_bound, total
        character(len=:), allocatable :: err
        type(declaration_node), pointer :: decl

        ok = .false.
        extents = 0
        do n = 1, context%arena%size
            if (.not. node_exists(context%arena, n)) cycle
            decl => declaration_named(context%arena, n, var_name)
            if (.not. associated(decl)) cycle
            if (.not. decl%is_array) cycle
            if (decl%is_allocatable) cycle
            if (.not. allocated(decl%dimension_indices)) cycle
            if (size(decl%dimension_indices) /= dim_count) cycle
            ! Skip the assumed-shape dummy's own colon declaration: its extents
            ! are exactly what we are resolving. Only a caller-side whole-array
            ! declaration with foldable bounds contributes a compile-time extent.
            if (declaration_is_assumed_shape(decl, context)) cycle
            call array_decl_dim_extents(context, decl, dim_count, extents, &
                                        lower_bound, total, err)
            if (len_trim(err) > 0) cycle
            ok = .true.
            return
        end do
    end subroutine array_decl_extents

    function declaration_named(arena, node_index, var_name) result(decl)
        ! Pointer to the declaration node at node_index when it declares
        ! var_name (single or multi-declaration); null otherwise.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        character(len=*), intent(in) :: var_name
        type(declaration_node), pointer :: decl
        integer :: k

        decl => null()
        select type (node => arena%entries(node_index)%node)
        type is (declaration_node)
            if (allocated(node%var_name)) then
                if (trim(node%var_name) == trim(var_name)) then
                    decl => node
                    return
                end if
            end if
            if (node%is_multi_declaration .and. allocated(node%var_names)) then
                do k = 1, size(node%var_names)
                    if (trim(node%var_names(k)) == trim(var_name)) then
                        decl => node
                        return
                    end if
                end do
            end if
        end select
    end function declaration_named

    subroutine array_decl_dim_extents(context, decl, dim_count, extents, &
                                      lower_bound, total, error_msg)
        ! Per-dimension extents of an explicit-shape array declaration, folding
        ! literal and named-constant bounds via eval_i32_constant.
        type(lowering_context_t), intent(in) :: context
        type(declaration_node), intent(in) :: decl
        integer, intent(in) :: dim_count
        integer, intent(out) :: extents(:)
        integer, intent(out) :: lower_bound
        integer, intent(out) :: total
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i, dim_index
        integer(c_int64_t) :: lower_value, upper_value, extent

        lower_bound = 1
        total = 1
        call set_empty(error_msg)
        do i = 1, dim_count
            dim_index = decl%dimension_indices(i)
            if (.not. node_exists(context%arena, dim_index)) then
                error_msg = 'actual array dimension index is invalid'
                return
            end if
            select type (dim_node => context%arena%entries(dim_index)%node)
            type is (range_expression_node)
                call eval_i32_constant(context%arena, dim_node%start_index, &
                                       context, lower_value, error_msg)
                if (len_trim(error_msg) > 0) return
                call eval_i32_constant(context%arena, dim_node%end_index, &
                                       context, upper_value, error_msg)
                if (len_trim(error_msg) > 0) return
                extent = upper_value - lower_value + 1_c_int64_t
            class default
                call eval_i32_constant(context%arena, dim_index, context, &
                                       upper_value, error_msg)
                if (len_trim(error_msg) > 0) return
                lower_value = 1_c_int64_t
                extent = upper_value
            end select
            if (extent <= 0_c_int64_t) then
                error_msg = 'actual array has a non-positive extent'
                return
            end if
            if (i == 1) lower_bound = int(lower_value)
            extents(i) = int(extent)
            total = total * int(extent)
        end do
    end subroutine array_decl_dim_extents

    subroutine get_array_bounds(node, context, array_lower_bound, array_size, &
                                error_msg)
        type(declaration_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        integer, intent(out) :: array_lower_bound
        integer, intent(out) :: array_size
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: dim_count
        integer :: dim_index
        integer(c_int64_t) :: extent
        integer(c_int64_t) :: lower_value
        integer(c_int64_t) :: upper_value
        integer :: i

        array_lower_bound = 1
        array_size = 0
        call set_empty(error_msg)
        if (.not. allocated(node%dimension_indices)) return

        dim_count = size(node%dimension_indices)
        if (dim_count < 1 .or. dim_count > ARRAY_MAX_RANK) then
            call unsupported_feature_error('array declaration', node%line, &
                                           node%column, &
                                           'direct LIRIC session supports '// &
                                           'at most rank-7 arrays', error_msg)
            return
        end if

        do i = 1, dim_count
            dim_index = node%dimension_indices(i)
            if (.not. node_exists(context%arena, dim_index)) then
                error_msg = 'array dimension index does not reference an AST node'
                return
            end if

            select type (dim_node => context%arena%entries(dim_index)%node)
            type is (range_expression_node)
                if (dim_node%stride_index > 0) then
                    call unsupported_feature_error('array declaration', node%line, &
                                                   node%column, &
                                                   'array declaration bounds do not '// &
                                                   'support strides', error_msg)
                    return
                end if
                if (dim_node%start_index <= 0 .or. dim_node%end_index <= 0) then
                    call unsupported_feature_error('array declaration', node%line, &
                                                   node%column, &
                                                   'array declaration requires '// &
                                                   'explicit lower and upper bounds', &
                                                   error_msg)
                    return
                end if
                call eval_i32_constant(context%arena, dim_node%start_index, context, &
                                       lower_value, error_msg)
                if (len_trim(error_msg) > 0) return
                call eval_i32_constant(context%arena, dim_node%end_index, context, &
                                       upper_value, error_msg)
                if (len_trim(error_msg) > 0) return
                extent = upper_value - lower_value + 1_c_int64_t
            class default
                call eval_i32_constant(context%arena, dim_index, context, &
                                       upper_value, error_msg)
                if (len_trim(error_msg) > 0) return
                lower_value = 1_c_int64_t
                extent = upper_value
            end select

            if (extent <= 0_c_int64_t) then
                call unsupported_feature_error('array declaration', node%line, &
                                               node%column, &
                                               'array bounds produce a '// &
                                               'non-positive extent', error_msg)
                return
            end if

            if (i == 1) then
                array_lower_bound = int(lower_value)
                array_size = int(extent)
            else
                array_size = array_size * int(extent)
            end if
        end do
        call set_empty(error_msg)
    end subroutine get_array_bounds

    recursive subroutine eval_i32_constant(arena, node_index, context, &
                                           constant_value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        integer(c_int64_t), intent(out) :: constant_value
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int64_t) :: lhs
        integer(c_int64_t) :: rhs
        integer :: symbol_index
        character(len=:), allocatable :: bin_op
        integer :: bin_left, bin_right, bin_line, bin_col
        character(len=:), allocatable :: lit_value, lit_type
        character(len=:), allocatable :: id_name

        constant_value = 0_c_int64_t
        if (.not. node_exists(arena, node_index)) then
            error_msg = 'constant expression index does not reference an AST node'
            return
        end if

        if (is_binary_op(arena, node_index)) then
            call get_binary_op_info(arena, node_index, bin_op, bin_left, &
                                    bin_right, bin_line, bin_col, error_msg)
            if (len_trim(error_msg) > 0) return
            call eval_i32_constant(arena, bin_left, context, lhs, error_msg)
            if (len_trim(error_msg) > 0) return
            call eval_i32_constant(arena, bin_right, context, rhs, error_msg)
            if (len_trim(error_msg) > 0) return
            select case (trim(bin_op))
            case ('+')
                constant_value = lhs + rhs
            case ('-')
                constant_value = lhs - rhs
            case ('*')
                constant_value = lhs*rhs
            case ('/')
                if (rhs == 0_c_int64_t) then
                    error_msg = 'integer constant division by zero'
                    return
                end if
                constant_value = lhs/rhs
            case default
                error_msg = 'unsupported integer constant operator: '// &
                            trim(bin_op)
                return
            end select
            call set_empty(error_msg)
            return
        end if

        if (is_literal(arena, node_index)) then
            call get_literal_info(arena, node_index, lit_value, lit_type, &
                                  error_msg)
            if (len_trim(error_msg) > 0) return
            call parse_i32_literal(lit_value, constant_value, error_msg)
            return
        end if

        if (is_identifier(arena, node_index)) then
            call get_identifier_name(arena, node_index, id_name, error_msg)
            if (len_trim(error_msg) > 0) return
            symbol_index = find_symbol(context, id_name)
            if (symbol_index > 0) then
                if (context%symbols(symbol_index)%has_i32_constant) then
                    constant_value = context%symbols(symbol_index)%i32_constant
                    call set_empty(error_msg)
                    return
                end if
            else
                if (iso_c_binding_kind_value(id_name, constant_value)) then
                    call set_empty(error_msg)
                    return
                end if
            end if
            ! The identifier is not a compile-time constant in the current scope.
            ! When resolving an assumed-shape dummy's extent from a caller's actual
            ! declared with a named-constant bound (dimension(n)), that constant
            ! lives in the caller's scope, not the callee's. Fold it from its
            ! PARAMETER declaration anywhere in the unit.
            call fold_named_constant_from_arena(arena, context, id_name, &
                                                constant_value, error_msg)
            return
        end if

        select type (node => arena%entries(node_index)%node)
        type is (call_or_subscript_node)
            call eval_kind_inquiry_constant(arena, node, context, &
                                            constant_value, error_msg)
        class default
            error_msg = 'array bound is not a compile-time integer expression'
        end select
    end subroutine eval_i32_constant

    recursive subroutine fold_named_constant_from_arena(arena, context, name, &
                                                        constant_value, error_msg)
        ! Fold an integer named constant by finding its PARAMETER declaration
        ! anywhere in the unit and evaluating its initializer. Used when the
        ! constant is not in the current lowering scope, e.g. a caller-scope
        ! parameter naming an assumed-shape actual's declared extent.
        type(ast_arena_t), intent(in) :: arena
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: name
        integer(c_int64_t), intent(out) :: constant_value
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: n

        constant_value = 0_c_int64_t
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (node => arena%entries(n)%node)
            type is (declaration_node)
                if (.not. node%is_parameter) cycle
                if (.not. node%has_initializer) cycle
                if (node%initializer_index <= 0) cycle
                if (.not. allocated(node%var_name)) cycle
                if (trim(node%var_name) /= trim(name)) cycle
                call eval_i32_constant(arena, node%initializer_index, context, &
                                       constant_value, error_msg)
                return
            end select
        end do
        error_msg = 'compile-time integer parameter was not declared: '//trim(name)
    end subroutine fold_named_constant_from_arena

    ! Constant-fold selected_int_kind / selected_real_kind so they may appear
    ! in a compile-time integer parameter initializer (kind selection).
    subroutine eval_kind_inquiry_constant(arena, node, context, constant_value, &
                                          error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        integer(c_int64_t), intent(out) :: constant_value
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int64_t) :: p, r

        constant_value = 0_c_int64_t
        if (.not. allocated(node%name)) then
            error_msg = 'compile-time call has no callee name'
            return
        end if
        if (same_name(node%name, 'selected_int_kind')) then
            if (.not. allocated(node%arg_indices) .or. &
                size(node%arg_indices) /= 1) then
                error_msg = 'selected_int_kind requires one argument'
                return
            end if
            call eval_i32_constant(arena, node%arg_indices(1), context, r, error_msg)
            if (len_trim(error_msg) > 0) return
            constant_value = selected_int_kind_value(r)
            call set_empty(error_msg)
        else if (same_name(node%name, 'selected_real_kind')) then
            if (.not. allocated(node%arg_indices) .or. &
                size(node%arg_indices) < 1 .or. size(node%arg_indices) > 2) then
                error_msg = 'selected_real_kind requires one or two arguments'
                return
            end if
            p = 0_c_int64_t
            r = 0_c_int64_t
            call eval_i32_constant(arena, node%arg_indices(1), context, p, error_msg)
            if (len_trim(error_msg) > 0) return
            if (size(node%arg_indices) >= 2) then
                call eval_i32_constant(arena, node%arg_indices(2), context, r, &
                                       error_msg)
                if (len_trim(error_msg) > 0) return
            end if
            constant_value = selected_real_kind_value(p, r)
            call set_empty(error_msg)
        else if (same_name(node%name, 'kind')) then
            call eval_kind_of_arg(arena, node, context, constant_value, error_msg)
        else if (same_name(node%name, 'size')) then
            call eval_size_constant(arena, node, context, constant_value, error_msg)
        else if (same_name(node%name, 'len')) then
            call eval_len_constant(arena, node, context, constant_value, error_msg)
        else if (same_name(node%name, 'min')) then
            call eval_min_max_constant(arena, node, context, .false., &
                                       constant_value, error_msg)
        else if (same_name(node%name, 'max')) then
            call eval_min_max_constant(arena, node, context, .true., &
                                       constant_value, error_msg)
        else if (same_name(node%name, 'int')) then
            call eval_int_constant(arena, node, context, constant_value, error_msg)
        else if (same_name(node%name, 'huge')) then
            call eval_huge_constant(arena, node, context, constant_value, error_msg)
        else if (same_name(node%name, 'selected_char_kind')) then
            call eval_selected_char_kind_constant(arena, node, constant_value, &
                                                  error_msg)
        else if (same_name(node%name, 'precision')) then
            call eval_precision_constant(arena, node, context, constant_value, &
                                         error_msg)
        else if (same_name(node%name, 'bit_size')) then
            call eval_bit_size_constant(arena, node, context, constant_value, &
                                        error_msg)
        else if (same_name(node%name, 'range')) then
            call eval_range_constant(arena, node, context, constant_value, &
                                     error_msg)
        else
            error_msg = 'compile-time integer expression uses unsupported call: '// &
                        trim(node%name)
        end if
    end subroutine eval_kind_inquiry_constant

    ! Map an internal value kind to the Fortran KIND number gfortran reports.
    pure integer(c_int64_t) function value_kind_number(value_kind) result(k)
        integer, intent(in) :: value_kind

        select case (value_kind)
        case (VALUE_I8)
            k = 1_c_int64_t
        case (VALUE_I16)
            k = 2_c_int64_t
        case (VALUE_I32, VALUE_F32, VALUE_LOGICAL, VALUE_C4)
            k = 4_c_int64_t
        case (VALUE_I64, VALUE_F64, VALUE_C8)
            k = 8_c_int64_t
        case (VALUE_CHARACTER)
            k = 1_c_int64_t
        case default
            k = -1_c_int64_t
        end select
    end function value_kind_number

    ! Fold kind(x): the kind of a literal argument follows its type and any
    ! kind suffix; the kind of a named entity follows its declared value kind.
    subroutine eval_kind_of_arg(arena, node, context, constant_value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        integer(c_int64_t), intent(out) :: constant_value
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: arg_index, sym
        integer(c_int64_t) :: k
        character(len=:), allocatable :: id_name

        constant_value = 0_c_int64_t
        if (.not. allocated(node%arg_indices) .or. size(node%arg_indices) /= 1) then
            error_msg = 'kind requires one argument'
            return
        end if
        arg_index = node%arg_indices(1)
        if (.not. node_exists(arena, arg_index)) then
            error_msg = 'kind argument does not reference an AST node'
            return
        end if

        if (is_literal(arena, arg_index)) then
            call kind_of_literal(arena, arg_index, k, error_msg)
            if (len_trim(error_msg) > 0) return
            constant_value = k
            call set_empty(error_msg)
            return
        end if

        if (is_identifier(arena, arg_index)) then
            call get_identifier_name(arena, arg_index, id_name, error_msg)
            if (len_trim(error_msg) > 0) return
            sym = find_symbol(context, id_name)
            if (sym <= 0) then
                error_msg = 'kind argument was not declared: '//trim(id_name)
                return
            end if
            k = value_kind_number(context%symbols(sym)%value_kind)
            if (k < 0) then
                error_msg = 'kind argument has no compile-time kind: '// &
                            trim(id_name)
                return
            end if
            constant_value = k
            call set_empty(error_msg)
            return
        end if

        error_msg = 'kind argument is not a literal or named entity'
    end subroutine eval_kind_of_arg

    ! Determine the Fortran kind of a literal node from its type and suffix.
    subroutine kind_of_literal(arena, node_index, k, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        integer(c_int64_t), intent(out) :: k
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: lit_value, lit_type

        k = 0_c_int64_t
        call set_empty(error_msg)
        if (is_character_literal(arena, node_index)) then
            k = 1_c_int64_t
        else if (is_logical_literal(arena, node_index)) then
            k = 4_c_int64_t
        else if (is_real_literal(arena, node_index)) then
            call get_literal_info(arena, node_index, lit_value, lit_type, error_msg)
            if (len_trim(error_msg) > 0) return
            if (literal_is_f64(lit_value)) then
                k = 8_c_int64_t
            else
                k = 4_c_int64_t
            end if
        else
            call get_literal_info(arena, node_index, lit_value, lit_type, error_msg)
            if (len_trim(error_msg) > 0) return
            call integer_literal_kind_number(lit_value, k, error_msg)
        end if
    end subroutine kind_of_literal

    ! Fold size(arr[, dim]) for a fixed-size array symbol.
    subroutine eval_size_constant(arena, node, context, constant_value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        integer(c_int64_t), intent(out) :: constant_value
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: sym, nargs
        integer(c_int64_t) :: dim
        character(len=:), allocatable :: id_name

        constant_value = 0_c_int64_t
        if (.not. allocated(node%arg_indices)) then
            error_msg = 'size requires an array argument'
            return
        end if
        nargs = size(node%arg_indices)
        if (nargs < 1 .or. nargs > 2) then
            error_msg = 'size requires one or two arguments'
            return
        end if
        if (.not. is_identifier(arena, node%arg_indices(1))) then
            error_msg = 'size argument is not a named array'
            return
        end if
        call get_identifier_name(arena, node%arg_indices(1), id_name, error_msg)
        if (len_trim(error_msg) > 0) return
        sym = find_symbol(context, id_name)
        if (sym <= 0) then
            error_msg = 'size argument was not declared: '//trim(id_name)
            return
        end if
        if (.not. context%symbols(sym)%is_array) then
            error_msg = 'size argument is not an array: '//trim(id_name)
            return
        end if
        if (context%symbols(sym)%array_size <= 0) then
            error_msg = 'size argument is not a fixed-size array: '//trim(id_name)
            return
        end if

        if (nargs == 1) then
            constant_value = int(context%symbols(sym)%array_size, c_int64_t)
            call set_empty(error_msg)
            return
        end if

        call eval_i32_constant(arena, node%arg_indices(2), context, dim, error_msg)
        if (len_trim(error_msg) > 0) return
        if (dim < 1 .or. dim > int(context%symbols(sym)%array_rank, c_int64_t)) then
            error_msg = 'size dim out of range for: '//trim(id_name)
            return
        end if
        constant_value = int(context%symbols(sym)%array_dim_sizes(int(dim)), &
                             c_int64_t)
        call set_empty(error_msg)
    end subroutine eval_size_constant

    ! Fold len(s) for a fixed-length character entity or character literal.
    subroutine eval_len_constant(arena, node, context, constant_value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        integer(c_int64_t), intent(out) :: constant_value
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: sym
        character(len=:), allocatable :: id_name, lit_value, lit_type
        character(len=:), allocatable :: stripped

        constant_value = 0_c_int64_t
        if (.not. allocated(node%arg_indices) .or. size(node%arg_indices) /= 1) then
            error_msg = 'len requires one argument'
            return
        end if

        if (is_literal(arena, node%arg_indices(1))) then
            if (.not. is_character_literal(arena, node%arg_indices(1))) then
                error_msg = 'len argument is not a character entity'
                return
            end if
            call get_literal_info(arena, node%arg_indices(1), lit_value, lit_type, &
                                  error_msg)
            if (len_trim(error_msg) > 0) return
            call strip_literal_quotes(lit_value, stripped)
            constant_value = int(len(stripped), c_int64_t)
            call set_empty(error_msg)
            return
        end if

        if (.not. is_identifier(arena, node%arg_indices(1))) then
            error_msg = 'len argument is not a named character entity'
            return
        end if
        call get_identifier_name(arena, node%arg_indices(1), id_name, error_msg)
        if (len_trim(error_msg) > 0) return
        sym = find_symbol(context, id_name)
        if (sym <= 0) then
            error_msg = 'len argument was not declared: '//trim(id_name)
            return
        end if
        if (context%symbols(sym)%value_kind /= VALUE_CHARACTER) then
            error_msg = 'len argument is not a character entity: '//trim(id_name)
            return
        end if
        if (context%symbols(sym)%character_length <= 0) then
            error_msg = 'len argument is not a fixed-length character: '// &
                        trim(id_name)
            return
        end if
        constant_value = int(context%symbols(sym)%character_length, c_int64_t)
        call set_empty(error_msg)
    end subroutine eval_len_constant

    subroutine define_declared_array_symbol(context, node, name, array_lower_bound, &
                                            array_size, value_kind, error_msg, &
                                            assumed_extents)
        type(lowering_context_t), intent(inout) :: context
        type(declaration_node), intent(in) :: node
        character(len=*), intent(in) :: name
        integer, intent(in) :: array_lower_bound
        integer, intent(in) :: array_size
        integer, intent(in) :: value_kind
        character(len=:), allocatable, intent(out) :: error_msg
        ! For an assumed-shape dummy a(:), the caller-derived extent per
        ! dimension (lower bound 1). Present and positive only for assumed-shape.
        integer, intent(in), optional :: assumed_extents(:)
        integer :: index
        integer :: dim_count
        integer :: i
        integer :: dim_index
        integer(c_int64_t) :: lower_value
        integer(c_int64_t) :: upper_value
        integer(c_int64_t) :: extent
        logical :: is_dummy
        logical :: is_assumed
        type(lr_operand_desc_t) :: param_base

        is_assumed = present(assumed_extents)

        ! An array body declaration whose name matches a dummy argument binds
        ! the explicit-shape array view onto the parameter pointer instead of
        ! allocating fresh storage; the caller passes the actual's base address.
        index = find_symbol(context, name)
        is_dummy = .false.
        if (index > 0) then
            if (context%symbols(index)%is_parameter .and. &
                .not. context%symbols(index)%is_array) then
                is_dummy = .true.
                param_base = context%symbols(index)%address
            else if (context%symbols(index)%is_array_result) then
                ! The result array was pre-bound to the sret buffer (param 0);
                ! this body declaration fills in its shape and element kind and
                ! keeps the buffer as element storage (array function results).
                is_dummy = .true.
                param_base = context%symbols(index)%element_address
            else
                error_msg = 'duplicate array declaration: '//trim(name)
                return
            end if
        end if

        if ((value_kind == VALUE_C4 .or. value_kind == VALUE_C8) .and. is_dummy) &
                then
            call unsupported_feature_error('complex array declaration', &
                node%line, node%column, &
                'direct LIRIC session supports complex arrays as '// &
                'fixed-size local declarations only, not dummy arguments '// &
                'or function results', error_msg)
            return
        end if

        dim_count = 0
        if (allocated(node%dimension_indices)) then
            dim_count = size(node%dimension_indices)
        end if

        if ((value_kind == VALUE_C4 .or. value_kind == VALUE_C8) .and. &
                (dim_count < 1 .or. dim_count > 2)) then
            call unsupported_feature_error('complex array declaration', &
                node%line, node%column, &
                'direct LIRIC session supports rank-1 and rank-2 complex '// &
                'arrays only', error_msg)
            return
        end if

        if (.not. is_dummy) then
            call grow_symbols(context)
            index = context%symbol_count + 1
        end if
        context%symbols(index)%name = trim(name)
        context%symbols(index)%value_kind = value_kind
        context%symbols(index)%is_array = .true.
        context%symbols(index)%array_rank = dim_count
        context%symbols(index)%array_size = array_size
        context%symbols(index)%array_lower_bound = array_lower_bound

        if (is_assumed) then
            ! Assumed-shape dummy a(:): the (:) dimensions carry no compile-time
            ! bound, so take the extents derived from the caller's actual instead
            ! of folding the colon dimension nodes. Lower bound is 1 (the dummy's
            ! local indexing), matching gfortran for an assumed-shape view.
            do i = 1, dim_count
                context%symbols(index)%array_dim_lowers(i) = 1
                context%symbols(index)%array_dim_sizes(i) = assumed_extents(i)
            end do
            context%symbols(index)%element_address = param_base
            call set_empty(error_msg)
            return
        end if

        ! An array function result whose extent is a dummy-argument value (e.g.
        ! `real, dimension(n) :: seq`) has no compile-time size. The result is
        ! bound to the caller's sret buffer (param 0), so element addressing GEPs
        ! the runtime subscript with the element stride and the array-type extent
        ! is irrelevant. Skip folding and keep array_size as the sentinel 0; the
        ! caller-side destination supplies the real storage and the do-loop bound
        ! n drives the element count at run time.
        if (context%symbols(index)%is_array_result .and. &
            declaration_is_runtime_rank1(node, context)) then
            context%symbols(index)%array_lower_bound = 1
            context%symbols(index)%array_dim_lowers(1) = 1
            context%symbols(index)%array_dim_sizes(1) = 0
            context%symbols(index)%array_size = 0
            context%symbols(index)%element_address = param_base
            call set_empty(error_msg)
            return
        end if

        if (dim_count >= 1 .and. allocated(node%dimension_indices)) then
            do i = 1, dim_count
                dim_index = node%dimension_indices(i)
                if (.not. node_exists(context%arena, dim_index)) cycle
                select type (dim_node => context%arena%entries(dim_index)%node)
                type is (range_expression_node)
                    call eval_i32_constant(context%arena, dim_node%start_index, context, &
                                           lower_value, error_msg)
                    if (len_trim(error_msg) > 0) return
                    call eval_i32_constant(context%arena, dim_node%end_index, context, &
                                           upper_value, error_msg)
                    if (len_trim(error_msg) > 0) return
                    context%symbols(index)%array_dim_lowers(i) = int(lower_value)
                    extent = upper_value - lower_value + 1_c_int64_t
                    context%symbols(index)%array_dim_sizes(i) = int(extent)
                class default
                    context%symbols(index)%array_dim_lowers(i) = 1
                    call eval_i32_constant(context%arena, dim_index, context, &
                                           upper_value, error_msg)
                    if (len_trim(error_msg) > 0) return
                    context%symbols(index)%array_dim_sizes(i) = int(upper_value)
                end select
            end do
        end if

        if (is_dummy) then
            ! Element addressing GEPs the [N x T] type onto this base, so the
            ! dummy reads and writes the caller's contiguous storage in place.
            context%symbols(index)%element_address = param_base
            call set_empty(error_msg)
            return
        end if

        select case (value_kind)
        case (VALUE_F32)
            if (.not. emit_f32_array_alloca(context%session, array_size, &
                    context%symbols(index)%element_address, error_msg)) then
                error_msg = 'decl_f32_array['//trim(name)//']: '//error_msg
                return
            end if
        case (VALUE_F64)
            if (.not. emit_f64_array_alloca(context%session, array_size, &
                    context%symbols(index)%element_address, error_msg)) then
                error_msg = 'decl_f64_array['//trim(name)//']: '//error_msg
                return
            end if
        case (VALUE_I64)
            if (.not. emit_i64_array_alloca(context%session, array_size, &
                    context%symbols(index)%element_address, error_msg)) then
                error_msg = 'decl_i64_array['//trim(name)//']: '//error_msg
                return
            end if
        case (VALUE_I8)
            if (.not. emit_i8_array_alloca(context%session, array_size, &
                    context%symbols(index)%element_address, error_msg)) then
                error_msg = 'decl_i8_array['//trim(name)//']: '//error_msg
                return
            end if
        case (VALUE_I16)
            if (.not. emit_i16_array_alloca(context%session, array_size, &
                    context%symbols(index)%element_address, error_msg)) then
                error_msg = 'decl_i16_array['//trim(name)//']: '//error_msg
                return
            end if
        case (VALUE_C4)
            ! Complex arrays store re/im components in two parallel f32 arrays:
            ! element_address is the re array base, address the im array base
            ! (mirrors the scalar complex(4) re/im slot convention).
            if (.not. emit_f32_array_alloca(context%session, array_size, &
                    context%symbols(index)%element_address, error_msg)) then
                error_msg = 'decl_c4_array_re['//trim(name)//']: '//error_msg
                return
            end if
            if (.not. emit_f32_array_alloca(context%session, array_size, &
                    context%symbols(index)%address, error_msg)) then
                error_msg = 'decl_c4_array_im['//trim(name)//']: '//error_msg
                return
            end if
        case (VALUE_C8)
            if (.not. emit_f64_array_alloca(context%session, array_size, &
                    context%symbols(index)%element_address, error_msg)) then
                error_msg = 'decl_c8_array_re['//trim(name)//']: '//error_msg
                return
            end if
            if (.not. emit_f64_array_alloca(context%session, array_size, &
                    context%symbols(index)%address, error_msg)) then
                error_msg = 'decl_c8_array_im['//trim(name)//']: '//error_msg
                return
            end if
        case default
            if (.not. emit_i32_array_alloca(context%session, array_size, &
                    context%symbols(index)%element_address, error_msg)) then
                error_msg = 'decl_array['//trim(name)//']: '//error_msg
                return
            end if
        end select

        context%symbol_count = index

        if (node%has_initializer .and. node%initializer_index > 0) then
            if (node_exists(context%arena, node%initializer_index)) then
                select type (init => context%arena%entries( &
                             node%initializer_index)%node)
                type is (array_literal_node)
                    call lower_array_constructor_assignment(context%arena, init, &
                                                            index, context, error_msg)
                    if (len_trim(error_msg) > 0) return
                class default
                    call broadcast_scalar_initializer(context, index, &
                                                      node%initializer_index, &
                                                      array_size, error_msg)
                    if (len_trim(error_msg) > 0) return
                end select
            end if
        end if

        call set_empty(error_msg)
    end subroutine define_declared_array_symbol

    subroutine broadcast_scalar_initializer(context, symbol_index, value_index, &
                                            array_size, error_msg)
        !! Scalar inline initializer (integer :: a(n) = expr): lower the scalar
        !! once and store it into every element of linear storage.
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        integer, intent(in) :: value_index
        integer, intent(in) :: array_size
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: value
        integer :: i

        select case (context%symbols(symbol_index)%value_kind)
        case (VALUE_F32)
            call lower_f32_expression(context%arena, value_index, context, &
                                      value, error_msg)
        case (VALUE_F64)
            call lower_f64_expression(context%arena, value_index, context, &
                                      value, error_msg)
        case (VALUE_LOGICAL)
            call lower_logical_expression(context%arena, value_index, context, &
                                          value, error_msg)
        case (VALUE_I64)
            call lower_i64_expression(context%arena, value_index, context, &
                                      value, error_msg)
        case (VALUE_I8)
            call lower_i8_expression(context%arena, value_index, context, &
                                     value, error_msg)
        case (VALUE_I16)
            call lower_i16_expression(context%arena, value_index, context, &
                                      value, error_msg)
        case (VALUE_C4, VALUE_C8)
            error_msg = 'complex array scalar initializer is not supported'
            return
        case default
            call lower_i32_expression(context%arena, value_index, context, &
                                      value, error_msg)
        end select
        if (len_trim(error_msg) > 0) return
        do i = 1, array_size
            call store_array_linear_element(context, symbol_index, &
                                            int(i - 1, c_int64_t), value, error_msg)
            if (len_trim(error_msg) > 0) return
        end do
        call set_empty(error_msg)
    end subroutine broadcast_scalar_initializer

    logical function is_implied_do_constructor(arena, ctor) result(is_implied)
        !! True when a constructor is an implied-do: tagged syntax_style or a
        !! lone do_loop_node element (fortfront omits the tag in some paths).
        type(ast_arena_t), intent(in) :: arena
        type(array_literal_node), intent(in) :: ctor

        is_implied = .false.
        if (allocated(ctor%syntax_style)) then
            if (ctor%syntax_style == 'implied_do') then
                is_implied = .true.
                return
            end if
        end if
        if (.not. allocated(ctor%element_indices)) return
        if (size(ctor%element_indices) /= 1) return
        if (.not. node_exists(arena, ctor%element_indices(1))) return
        select type (n => arena%entries(ctor%element_indices(1))%node)
        type is (do_loop_node)
            is_implied = .true.
        class default
        end select
    end function is_implied_do_constructor

    subroutine lower_array_constructor_assignment(arena, ctor, symbol_index, &
                                                  context, error_msg)
        ! arr = [a, b, c] for any fixed-size array: lower each element and
        ! store it at the matching slot in linear storage order.
        type(ast_arena_t), intent(in) :: arena
        type(array_literal_node), intent(in) :: ctor
        integer, intent(in) :: symbol_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: value
        integer :: i
        integer :: n
        integer :: vk
        integer, allocatable :: flat(:)

        ! An implied-do constructor wraps a single do_loop_node. fortfront does
        ! not always tag syntax_style, so detect either marker structurally.
        if (is_implied_do_constructor(arena, ctor)) then
            call expand_implied_do_constructor(arena, ctor, symbol_index, &
                                               context, error_msg)
            return
        end if

        if (.not. allocated(ctor%element_indices)) then
            error_msg = 'array constructor has no elements'
            return
        end if
        ! Flatten nested array literals ([[1,2],[3,4]] -> [1,2,3,4]) so a
        ! rank-2 array constructed from row literals stores in linear order.
        allocate (flat(0))
        call flatten_constructor_elements(arena, ctor%element_indices, flat, &
                                          error_msg)
        if (len_trim(error_msg) > 0) return
        n = size(flat)
        if (n /= context%symbols(symbol_index)%array_size) then
            error_msg = 'array constructor element count does not match the '// &
                'declared size of '//trim(context%symbols(symbol_index)%name)
            return
        end if
        vk = context%symbols(symbol_index)%value_kind
        do i = 1, n
            if (vk == VALUE_F32) then
                call lower_f32_expression(arena, flat(i), context, value, error_msg)
            else if (vk == VALUE_F64) then
                call lower_f64_expression(arena, flat(i), context, value, error_msg)
            else if (vk == VALUE_LOGICAL) then
                call lower_logical_expression(arena, flat(i), context, value, &
                                              error_msg)
            else if (vk == VALUE_I64) then
                call lower_i64_expression(arena, flat(i), context, value, error_msg)
            else if (vk == VALUE_I8) then
                call lower_i8_expression(arena, flat(i), context, value, error_msg)
            else if (vk == VALUE_I16) then
                call lower_i16_expression(arena, flat(i), context, value, error_msg)
            else if (vk == VALUE_C4 .or. vk == VALUE_C8) then
                error_msg = 'complex array constructor initializer is not '// &
                            'supported'
                return
            else
                call lower_constructor_i32_element(arena, flat(i), context, &
                                                   value, error_msg)
            end if
            if (len_trim(error_msg) > 0) return
            call store_array_linear_element(context, symbol_index, &
                                            int(i - 1, c_int64_t), value, error_msg)
            if (len_trim(error_msg) > 0) return
        end do
        call set_empty(error_msg)
    end subroutine lower_array_constructor_assignment

    subroutine lower_constructor_i32_element(arena, node_index, context, value, &
                                             error_msg)
        !! Lower one element of an integer array constructor. A typed constructor
        !! [integer :: 1.5, ...] may carry real literals; coerce them by Fortran
        !! int() truncation toward zero so they store as integers.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: lit_value, lit_type
        real(c_double) :: rval

        if (is_literal(arena, node_index)) then
            call get_literal_info(arena, node_index, lit_value, lit_type, error_msg)
            if (len_trim(error_msg) > 0) return
            if (index(lit_value, '.') > 0 .or. index(lit_value, 'e') > 0 .or. &
                index(lit_value, 'E') > 0 .or. index(lit_value, 'd') > 0 .or. &
                index(lit_value, 'D') > 0) then
                call parse_f64_literal(lit_value, rval, error_msg)
                if (len_trim(error_msg) > 0) return
                value = i32_immediate(context%session, int(rval, c_int64_t))
                call set_empty(error_msg)
                return
            end if
        end if
        call lower_i32_expression(arena, node_index, context, value, error_msg)
    end subroutine lower_constructor_i32_element

    recursive subroutine flatten_constructor_elements(arena, indices, flat, error_msg)
        !! Collect the leaf element indices of a (possibly nested) array
        !! constructor in order. A nested array literal contributes its own
        !! flattened elements; an implied-do nested inside is not supported.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: indices(:)
        integer, allocatable, intent(inout) :: flat(:)
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i

        call set_empty(error_msg)
        do i = 1, size(indices)
            if (.not. node_exists(arena, indices(i))) then
                flat = [flat, indices(i)]
                cycle
            end if
            select type (child => arena%entries(indices(i))%node)
            type is (array_literal_node)
                if (allocated(child%syntax_style)) then
                    if (child%syntax_style == 'implied_do') then
                        error_msg = 'nested implied-do array constructors are '// &
                                    'not supported'
                        return
                    end if
                end if
                if (allocated(child%element_indices)) then
                    call flatten_constructor_elements(arena, child%element_indices, &
                                                      flat, error_msg)
                    if (len_trim(error_msg) > 0) return
                end if
            class default
                flat = [flat, indices(i)]
            end select
        end do
    end subroutine flatten_constructor_elements

    subroutine expand_implied_do_constructor(arena, ctor, symbol_index, context, &
                                             error_msg)
        !! Expand a compile-time implied-do array constructor, arr = [(expr, i =
        !! lo, hi)], by folding the body for each i and storing the result. A
        !! nested implied-do body, e.g. [((expr, j=1,m), i=1,n)], recurses with a
        !! shared running slot, so the inner loop fills consecutive elements.
        type(ast_arena_t), intent(in) :: arena
        type(array_literal_node), intent(in) :: ctor
        integer, intent(in) :: symbol_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: slot

        slot = 0
        call expand_implied_do_into_slot(arena, ctor, symbol_index, slot, &
                                         context, error_msg)
        if (len_trim(error_msg) > 0) return
        if (slot /= context%symbols(symbol_index)%array_size) then
            error_msg = 'implied-do constructor element count does not match '// &
                        'the array size'
            return
        end if
        call set_empty(error_msg)
    end subroutine expand_implied_do_constructor

    recursive subroutine expand_implied_do_into_slot(arena, ctor, symbol_index, &
                                                     slot, context, error_msg)
        !! Fold one implied-do constructor into the array starting at slot,
        !! advancing slot past the elements it produces. The loop variable is
        !! bound as a compile-time constant per iteration so eval_i32_constant
        !! can fold the body. A body element that is itself an implied-do
        !! constructor recurses; any other body element is one scalar element.
        type(ast_arena_t), intent(in) :: arena
        type(array_literal_node), intent(in) :: ctor
        integer, intent(in) :: symbol_index
        integer, intent(inout) :: slot
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int64_t) :: lo, hi, step, ival
        integer :: vsym, b
        logical :: created_temp
        type(symbol_t) :: saved_sym

        if (.not. allocated(ctor%element_indices)) then
            error_msg = 'implied-do constructor has no loop'
            return
        end if
        if (size(ctor%element_indices) /= 1) then
            error_msg = 'implied-do constructor is missing its loop node'
            return
        end if

        select type (dnode => arena%entries(ctor%element_indices(1))%node)
        type is (do_loop_node)
            if (.not. allocated(dnode%body_indices)) then
                error_msg = 'implied-do constructor has no body'
                return
            end if
            call eval_i32_constant(arena, dnode%start_expr_index, context, lo, &
                                   error_msg)
            if (len_trim(error_msg) > 0) return
            call eval_i32_constant(arena, dnode%end_expr_index, context, hi, &
                                   error_msg)
            if (len_trim(error_msg) > 0) return
            step = 1_c_int64_t
            if (dnode%step_expr_index > 0) then
                call eval_i32_constant(arena, dnode%step_expr_index, context, step, &
                                       error_msg)
                if (len_trim(error_msg) > 0) return
            end if
            if (step == 0_c_int64_t) then
                error_msg = 'implied-do constructor step is zero'
                return
            end if
            ! The implied-do index is locally scoped to its constructor and
            ! needs no explicit declaration. When it is not an existing symbol,
            ! bind a temporary i32 slot for the loop and drop it afterwards.
            created_temp = .false.
            vsym = find_symbol(context, dnode%var_name)
            if (vsym <= 0) then
                call grow_symbols(context)
                vsym = context%symbol_count + 1
                context%symbols(vsym)%name = trim(dnode%var_name)
                context%symbol_count = vsym
                created_temp = .true.
            end if
            saved_sym = context%symbols(vsym)
            context%symbols(vsym)%has_i32_constant = .true.
            context%symbols(vsym)%value_kind = VALUE_I32
            ival = lo
            do while ((step > 0 .and. ival <= hi) .or. (step < 0 .and. ival >= hi))
                ! Bind the loop var as an i32 immediate both for compile-time
                ! folding (eval_i32_constant) and for emit-time lowering of a
                ! real body (lower_iX/fX read %value when has_address is off).
                context%symbols(vsym)%i32_constant = ival
                context%symbols(vsym)%value = i32_immediate(context%session, ival)
                context%symbols(vsym)%has_address = .false.
                context%symbols(vsym)%is_reference = .false.
                do b = 1, size(dnode%body_indices)
                    call expand_implied_do_body_element(arena, &
                        dnode%body_indices(b), symbol_index, slot, context, &
                        error_msg)
                    if (len_trim(error_msg) > 0) then
                        if (created_temp) then
                            context%symbol_count = context%symbol_count - 1
                        else
                            context%symbols(vsym) = saved_sym
                        end if
                        return
                    end if
                end do
                ival = ival + step
            end do
            if (created_temp) then
                context%symbol_count = context%symbol_count - 1
            else
                context%symbols(vsym) = saved_sym
            end if
        class default
            error_msg = 'implied-do constructor is missing its loop node'
            return
        end select
        call set_empty(error_msg)
    end subroutine expand_implied_do_into_slot

    recursive subroutine expand_implied_do_body_element(arena, body_index, &
                                                        symbol_index, slot, &
                                                        context, error_msg)
        !! Store one implied-do body item at the running slot. A nested implied-do
        !! constructor recurses; any other item is a single scalar element.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: body_index
        integer, intent(in) :: symbol_index
        integer, intent(inout) :: slot
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg

        if (node_exists(arena, body_index)) then
            select type (inner => arena%entries(body_index)%node)
            type is (array_literal_node)
                if (is_implied_do_constructor(arena, inner)) then
                    call expand_implied_do_into_slot(arena, inner, symbol_index, &
                                                     slot, context, error_msg)
                    return
                end if
            type is (do_loop_node)
                call expand_implied_do_inner_loop(arena, body_index, &
                                                  symbol_index, slot, context, &
                                                  error_msg)
                return
            end select
        end if

        if (slot >= context%symbols(symbol_index)%array_size) then
            error_msg = 'implied-do constructor produces more elements than the '// &
                        'array holds'
            return
        end if
        call lower_implied_do_element(arena, body_index, symbol_index, &
                                      int(slot, c_int64_t), context, error_msg)
        if (len_trim(error_msg) > 0) return
        slot = slot + 1
    end subroutine expand_implied_do_body_element

    recursive subroutine expand_implied_do_inner_loop(arena, loop_index, &
                                                      symbol_index, slot, context, &
                                                      error_msg)
        !! A nested implied-do whose constructor wrapper is absent appears as a
        !! bare do_loop_node body element. Fold it into the running slot by
        !! reusing the same loop expansion path through a synthetic wrapper.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: loop_index
        integer, intent(in) :: symbol_index
        integer, intent(inout) :: slot
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(array_literal_node) :: wrapper

        wrapper%element_indices = [loop_index]
        call expand_implied_do_into_slot(arena, wrapper, symbol_index, slot, &
                                         context, error_msg)
    end subroutine expand_implied_do_inner_loop

    subroutine lower_implied_do_element(arena, body_index, symbol_index, slot, &
                                        context, error_msg)
        !! Lower one implied-do body expression for the current loop-var binding
        !! and store it at linear slot. Integer arrays fold to an immediate;
        !! real arrays emit a real expression so bodies like real(i)/2.0 work.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: body_index
        integer, intent(in) :: symbol_index
        integer(c_int64_t), intent(in) :: slot
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: value
        integer(c_int64_t) :: cval

        select case (context%symbols(symbol_index)%value_kind)
        case (VALUE_F32)
            call lower_f32_expression(arena, body_index, context, value, error_msg)
        case (VALUE_F64)
            call lower_f64_expression(arena, body_index, context, value, error_msg)
        case (VALUE_LOGICAL)
            call lower_logical_expression(arena, body_index, context, value, &
                                          error_msg)
        case (VALUE_I64, VALUE_I8, VALUE_I16)
            error_msg = 'implied-do array constructor does not support '// &
                        'non-default-kind integer elements'
            return
        case default
            call eval_i32_constant(arena, body_index, context, cval, error_msg)
            if (len_trim(error_msg) > 0) return
            value = i32_immediate(context%session, cval)
        end select
        if (len_trim(error_msg) > 0) return
        call store_array_linear_element(context, symbol_index, slot, value, &
                                        error_msg)
    end subroutine lower_implied_do_element

    logical function is_whole_array_intrinsic_name(name) result(is_intrinsic)
        ! True for a whole-array-shaped RHS intrinsic name handled by
        ! lower_array_whole_assignment's per-intrinsic dispatch below.
        character(len=*), intent(in) :: name

        is_intrinsic = same_name(name, 'shape') .or. &
                       same_name(name, 'matmul') .or. &
                       same_name(name, 'transpose') .or. &
                       same_name(name, 'reshape') .or. &
                       same_name(name, 'cshift') .or. &
                       same_name(name, 'eoshift') .or. &
                       same_name(name, 'merge') .or. &
                       same_name(name, 'spread') .or. &
                       same_name(name, 'pack') .or. &
                       same_name(name, 'unpack')
    end function is_whole_array_intrinsic_name

    subroutine lower_array_whole_assignment(arena, node, symbol_index, context, &
                                            error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(assignment_node), intent(in) :: node
        integer, intent(in) :: symbol_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: value, wide_value, narrow_value
        integer :: i
        integer :: array_size

        array_size = context%symbols(symbol_index)%array_size
        if (array_size <= 0) then
            error_msg = 'whole-array assignment requires a positive array size'
            return
        end if

        if (node_exists(arena, node%value_index)) then
            select type (rhs => arena%entries(node%value_index)%node)
            type is (array_literal_node)
                ! a = [...] / a = [(expr, i=lo,hi)]: reuse the constructor
                ! lowering so implied-do, typed, and nested forms share one path.
                call lower_array_constructor_assignment(arena, rhs, symbol_index, &
                                                        context, error_msg)
                return
            type is (call_or_subscript_node)
                if (allocated(rhs%name)) then
                    if (is_whole_array_intrinsic_name(rhs%name)) then
                        if (context%symbols(symbol_index)%value_kind == &
                                VALUE_I64 .or. &
                            context%symbols(symbol_index)%value_kind == &
                                VALUE_I8 .or. &
                            context%symbols(symbol_index)%value_kind == &
                                VALUE_I16) then
                            call unsupported_feature_error( &
                                'whole-array intrinsic assignment', &
                                node%line, node%column, &
                                'direct LIRIC session supports '// &
                                trim(rhs%name)//' for default-kind integer, '// &
                                'real, and logical arrays only', error_msg)
                            return
                        end if
                    end if
                end if
                if (allocated(rhs%name)) then
                    if (same_name(rhs%name, 'shape')) then
                        call lower_shape_intrinsic_assignment(arena, node, &
                                                              symbol_index, context, &
                                                              error_msg)
                        return
                    end if
                    if (same_name(rhs%name, 'matmul')) then
                        call lower_matmul_assignment(arena, node, symbol_index, &
                                                     context, error_msg)
                        return
                    end if
                    if (same_name(rhs%name, 'transpose')) then
                        call lower_transpose_assignment(arena, node, symbol_index, &
                                                        context, error_msg)
                        return
                    end if
                    if (same_name(rhs%name, 'reshape')) then
                        call lower_reshape_assignment(arena, node, symbol_index, &
                                                      context, error_msg)
                        return
                    end if
                    if (same_name(rhs%name, 'cshift')) then
                        call lower_shift_assignment(arena, node, symbol_index, &
                                                    context, .true., error_msg)
                        return
                    end if
                    if (same_name(rhs%name, 'eoshift')) then
                        call lower_shift_assignment(arena, node, symbol_index, &
                                                    context, .false., error_msg)
                        return
                    end if
                    if (same_name(rhs%name, 'merge')) then
                        call lower_merge_assignment(arena, node, symbol_index, &
                                                    context, error_msg)
                        return
                    end if
                    if (same_name(rhs%name, 'spread')) then
                        call lower_spread_assignment(arena, node, symbol_index, &
                                                     context, error_msg)
                        return
                    end if
                    if (same_name(rhs%name, 'pack')) then
                        call lower_pack_assignment(arena, node, symbol_index, &
                                                   context, error_msg)
                        return
                    end if
                    if (same_name(rhs%name, 'unpack')) then
                        call lower_unpack_assignment(arena, node, symbol_index, &
                                                     context, error_msg)
                        return
                    end if
                end if
            end select
        end if

        ! Scalar broadcast: a = expr where expr is a scalar. Lower the scalar
        ! once and store it into every element.
        if (is_scalar_broadcast_rhs(arena, node%value_index, context)) then
            select case (context%symbols(symbol_index)%value_kind)
            case (VALUE_F32)
                call lower_f32_expression(arena, node%value_index, context, &
                                          value, error_msg)
            case (VALUE_F64)
                call lower_f64_expression(arena, node%value_index, context, &
                                          value, error_msg)
            case (VALUE_LOGICAL)
                call lower_logical_expression(arena, node%value_index, context, &
                                              value, error_msg)
            case (VALUE_I64)
                call lower_i64_expression(arena, node%value_index, context, &
                                          value, error_msg)
            case (VALUE_I8)
                call lower_i8_expression(arena, node%value_index, context, &
                                         value, error_msg)
            case (VALUE_I16)
                call lower_i16_expression(arena, node%value_index, context, &
                                          value, error_msg)
            case default
                if (is_f64_expression(arena, node%value_index, context)) then
                    ! Integer array with a real scalar rhs: assignment
                    ! converts each element by truncation toward zero.
                    call lower_f64_expression(arena, node%value_index, &
                                              context, wide_value, error_msg)
                    if (len_trim(error_msg) > 0) return
                    if (.not. emit_liric_f64_to_i32(context%session, &
                        wide_value, value, error_msg)) return
                else if (is_f32_expression(arena, node%value_index, context)) then
                    call lower_f32_expression(arena, node%value_index, &
                                              context, narrow_value, error_msg)
                    if (len_trim(error_msg) > 0) return
                    if (.not. emit_liric_f32_to_f64(context%session, &
                        narrow_value, wide_value, error_msg)) return
                    if (.not. emit_liric_f64_to_i32(context%session, &
                        wide_value, value, error_msg)) return
                else
                    call lower_i32_expression(arena, node%value_index, context, &
                                              value, error_msg)
                end if
            end select
            if (len_trim(error_msg) > 0) return
            do i = 1, array_size
                call store_array_linear_element(context, symbol_index, &
                                                int(i - 1, c_int64_t), value, &
                                                error_msg)
                if (len_trim(error_msg) > 0) return
            end do
            call set_empty(error_msg)
            return
        end if

        do i = 1, array_size
            call lower_array_elementwise_value(arena, node%value_index, &
                                               symbol_index, i - 1, context, &
                                               value, error_msg)
            if (len_trim(error_msg) > 0) return
            call store_array_linear_element(context, symbol_index, &
                                            int(i - 1, c_int64_t), value, error_msg)
            if (len_trim(error_msg) > 0) return
        end do
        call set_empty(error_msg)
    end subroutine lower_array_whole_assignment

    recursive logical function is_scalar_broadcast_rhs(arena, node_index, &
                                                        context) result(scalar)
        !! True when the RHS of a whole-array assignment is a scalar
        !! expression that should broadcast: a literal, an identifier naming
        !! a non-array scalar, an array element access, or a call/binary-op
        !! tree built purely from those (e.g. real(i), a + 1).
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        character(len=:), allocatable :: name, err, bin_op
        integer :: sym, i, bin_left, bin_right, bin_line, bin_col

        scalar = .false.
        if (.not. node_exists(arena, node_index)) return
        select type (n => arena%entries(node_index)%node)
        type is (literal_node)
            scalar = .true.
            return
        type is (call_or_subscript_node)
            if (n%is_array_access) then
                scalar = .true.
                return
            end if
            if (n%base_expr_index > 0) return
            if (allocated(n%name)) then
                if (is_contained_array_function(context, n%name)) return
            end if
            if (.not. allocated(n%arg_indices)) then
                scalar = .true.
                return
            end if
            do i = 1, size(n%arg_indices)
                if (.not. is_scalar_broadcast_rhs(arena, n%arg_indices(i), &
                                                  context)) return
            end do
            scalar = .true.
            return
        end select
        if (is_binary_op(arena, node_index)) then
            call get_binary_op_info(arena, node_index, bin_op, bin_left, &
                                    bin_right, bin_line, bin_col, err)
            if (len_trim(err) > 0) return
            scalar = is_scalar_broadcast_rhs(arena, bin_left, context) .and. &
                     is_scalar_broadcast_rhs(arena, bin_right, context)
            return
        end if
        if (is_identifier(arena, node_index)) then
            call get_identifier_name(arena, node_index, name, err)
            if (len_trim(err) > 0) return
            sym = find_symbol(context, name)
            if (sym > 0) scalar = .not. is_elementwise_array_operand(context, sym)
        end if
    end function is_scalar_broadcast_rhs

    subroutine lower_shape_intrinsic_assignment(arena, node, symbol_index, &
                                                context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(assignment_node), intent(in) :: node
        integer, intent(in) :: symbol_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: addr
        type(lr_operand_desc_t) :: value
        character(len=:), allocatable :: source_name
        integer :: source_index
        integer :: source_rank
        integer :: i

        select type (rhs => arena%entries(node%value_index)%node)
        type is (call_or_subscript_node)
            if (.not. allocated(rhs%arg_indices) .or. size(rhs%arg_indices) /= 1) then
                error_msg = 'shape requires exactly one array argument'
                return
            end if
            call get_identifier_name(arena, rhs%arg_indices(1), source_name, &
                                     error_msg)
            if (len_trim(error_msg) > 0) return
            source_index = find_symbol(context, source_name)
            if (source_index <= 0) then
                error_msg = 'shape argument was not declared: '//trim(source_name)
                return
            end if
            if (.not. context%symbols(source_index)%is_array) then
                error_msg = 'shape requires an array argument: '//trim(source_name)
                return
            end if
        class default
            error_msg = 'shape requires an array argument'
            return
        end select

        source_rank = context%symbols(source_index)%array_rank
        if (context%symbols(symbol_index)%array_rank /= 1) then
            error_msg = 'shape result must be assigned to a rank-1 integer array'
            return
        end if
        if (context%symbols(symbol_index)%array_size /= source_rank) then
            error_msg = 'shape result length does not match the target array size'
            return
        end if

        do i = 1, source_rank
            if (.not. emit_i32_array_element_addr(context%session, &
                    context%symbols(symbol_index)%array_size, &
                    context%symbols(symbol_index)%element_address, &
                    i32_immediate(context%session, int(i - 1, c_int64_t)), addr, &
                    error_msg)) return
            value = i32_immediate(context%session, &
                                  int(context%symbols(source_index)%array_dim_sizes(i), &
                                      c_int64_t))
            if (.not. emit_i32_store(context%session, value, addr, error_msg)) &
                return
        end do
        call set_empty(error_msg)
    end subroutine lower_shape_intrinsic_assignment

    subroutine lower_shape_intrinsic_element(arena, node, target_symbol_index, &
                                             linear_index, context, value, &
                                             error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        integer, intent(in) :: target_symbol_index
        integer, intent(in) :: linear_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: source_name
        integer :: source_index
        integer :: source_rank

        if (.not. allocated(node%arg_indices) .or. size(node%arg_indices) /= 1) then
            error_msg = 'shape requires exactly one array argument'
            return
        end if
        call get_identifier_name(arena, node%arg_indices(1), source_name, &
                                 error_msg)
        if (len_trim(error_msg) > 0) return
        source_index = find_symbol(context, source_name)
        if (source_index <= 0) then
            error_msg = 'shape argument was not declared: '//trim(source_name)
            return
        end if
        if (.not. context%symbols(source_index)%is_array) then
            error_msg = 'shape requires an array argument: '//trim(source_name)
            return
        end if

        source_rank = context%symbols(source_index)%array_rank
        if (context%symbols(target_symbol_index)%array_rank /= 1) then
            error_msg = 'shape result must be assigned to a rank-1 integer array'
            return
        end if
        if (context%symbols(target_symbol_index)%array_size /= source_rank) then
            error_msg = 'shape result length does not match the target array size'
            return
        end if
        if (linear_index < 0 .or. linear_index >= source_rank) then
            error_msg = 'shape result index is out of range'
            return
        end if

        value = i32_immediate(context%session, &
                              int(context%symbols(source_index)%array_dim_sizes( &
                                  linear_index + 1), c_int64_t))
        call set_empty(error_msg)
    end subroutine lower_shape_intrinsic_element

    subroutine lower_array_size_intrinsic(arena, node, context, value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: array_name
        integer :: source_index
        integer :: nargs
        integer(c_int64_t) :: dim_val

        nargs = 0
        if (allocated(node%arg_indices)) nargs = size(node%arg_indices)
        if (nargs < 1 .or. nargs > 2) then
            error_msg = 'size requires one or two arguments'
            return
        end if
        call get_identifier_name(arena, node%arg_indices(1), array_name, error_msg)
        if (len_trim(error_msg) > 0) return
        source_index = find_symbol(context, array_name)
        if (source_index <= 0) then
            error_msg = 'size argument was not declared: '//trim(array_name)
            return
        end if
        if (context%symbols(source_index)%is_allocatable) then
            if (nargs /= 1) then
                error_msg = 'size with a dim over an allocatable is not supported'
                return
            end if
            if (context%symbols(source_index)%allocatable_static_size <= 0) then
                error_msg = 'size requires a constant allocatable extent: '// &
                            trim(array_name)
                return
            end if
            value = i32_immediate(context%session, &
                int(context%symbols(source_index)%allocatable_static_size, &
                    c_int64_t))
            call set_empty(error_msg)
            return
        end if
        if (.not. context%symbols(source_index)%is_array) then
            error_msg = 'size requires an array argument: '//trim(array_name)
            return
        end if
        if (nargs == 1) then
            value = i32_immediate(context%session, &
                                  int(context%symbols(source_index)%array_size, &
                                      c_int64_t))
            call set_empty(error_msg)
            return
        end if
        call eval_i32_constant(arena, node%arg_indices(2), context, dim_val, &
                               error_msg)
        if (len_trim(error_msg) > 0) return
        if (dim_val < 1 .or. &
            dim_val > int(context%symbols(source_index)%array_rank, c_int64_t)) then
            error_msg = 'size dim out of range for: '//trim(array_name)
            return
        end if
        value = i32_immediate(context%session, &
                    int(context%symbols(source_index)%array_dim_sizes(int(dim_val)), &
                        c_int64_t))
        call set_empty(error_msg)
    end subroutine lower_array_size_intrinsic

    subroutine lower_array_reduction_intrinsic(arena, node, context, value, &
                                               error_msg, reduction_name)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=*), intent(in) :: reduction_name
        character(len=:), allocatable :: array_name
        type(lr_operand_desc_t) :: candidate
        type(lr_operand_desc_t) :: next_value
        type(lr_operand_desc_t) :: condition
        type(lr_operand_desc_t) :: selected
        integer :: source_index
        integer :: array_size
        integer :: i
        integer(c_int) :: predicate

        if (.not. allocated(node%arg_indices) .or. size(node%arg_indices) /= 1) then
            error_msg = trim(reduction_name)//' requires exactly one array argument'
            return
        end if

        ! A section argument (sum(a(lo:hi))) reduces over the section extent
        ! rather than the whole declared array.
        if (node_exists(arena, node%arg_indices(1))) then
            select type (arg => arena%entries(node%arg_indices(1))%node)
            type is (array_slice_node)
                call lower_section_reduction(arena, arg, context, value, &
                                             reduction_name, error_msg)
                return
            end select
        end if

        call get_identifier_name(arena, node%arg_indices(1), array_name, error_msg)
        if (len_trim(error_msg) > 0) return
        source_index = find_symbol(context, array_name)
        if (source_index <= 0) then
            error_msg = trim(reduction_name)//' argument was not declared: '// &
                        trim(array_name)
            return
        end if
        ! A 1-D allocatable reduces over its compile-time extent recorded by the
        ! preceding allocate(a(N)); the element loop loads from the descriptor.
        if (context%symbols(source_index)%is_allocatable) then
            if (context%symbols(source_index)%array_rank == 2) then
                error_msg = trim(reduction_name)// &
                            ' over a rank-2 allocatable is not supported'
                return
            end if
            array_size = context%symbols(source_index)%allocatable_static_size
            if (array_size <= 0) then
                error_msg = trim(reduction_name)// &
                            ' requires a constant allocatable extent: '// &
                            trim(array_name)
                return
            end if
        else
            if (.not. context%symbols(source_index)%is_array) then
                error_msg = trim(reduction_name)// &
                            ' requires an array argument: '//trim(array_name)
                return
            end if
            array_size = context%symbols(source_index)%array_size
            if (array_size <= 0) then
                error_msg = trim(reduction_name)//' requires a non-empty array'
                return
            end if
        end if

        select case (context%symbols(source_index)%value_kind)
        case (VALUE_F32)
            select case (trim(reduction_name))
            case ('sum')
                value = liric_f32_immediate(context%session, 0.0_c_float)
                do i = 0, array_size - 1
                    call load_array_linear_element(context, source_index, &
                                                   int(i, c_int64_t), candidate, &
                                                   error_msg)
                    if (len_trim(error_msg) > 0) return
                    if (.not. emit_liric_f32_binary(context%session, LR_OP_FADD, &
                                                    value, candidate, next_value, &
                                                    error_msg)) return
                    value = next_value
                end do
            case ('product')
                value = liric_f32_immediate(context%session, 1.0_c_float)
                do i = 0, array_size - 1
                    call load_array_linear_element(context, source_index, &
                                                   int(i, c_int64_t), candidate, &
                                                   error_msg)
                    if (len_trim(error_msg) > 0) return
                    if (.not. emit_liric_f32_binary(context%session, LR_OP_FMUL, &
                                                    value, candidate, next_value, &
                                                    error_msg)) return
                    value = next_value
                end do
            case ('norm2')
                ! L2 norm: sqrt of the sum of squares of the elements.
                value = liric_f32_immediate(context%session, 0.0_c_float)
                do i = 0, array_size - 1
                    call load_array_linear_element(context, source_index, &
                                                   int(i, c_int64_t), candidate, &
                                                   error_msg)
                    if (len_trim(error_msg) > 0) return
                    if (.not. emit_liric_f32_binary(context%session, LR_OP_FMUL, &
                                                    candidate, candidate, &
                                                    next_value, error_msg)) return
                    if (.not. emit_liric_f32_binary(context%session, LR_OP_FADD, &
                                                    value, next_value, candidate, &
                                                    error_msg)) return
                    value = candidate
                end do
                if (.not. emit_liric_f32_call(context%session, 'sqrtf', &
                                              [value], next_value, error_msg)) &
                    return
                value = next_value
            case ('maxval')
                call load_array_linear_element(context, source_index, 0_c_int64_t, &
                                               value, error_msg)
                if (len_trim(error_msg) > 0) return
                do i = 1, array_size - 1
                    call load_array_linear_element(context, source_index, &
                                                   int(i, c_int64_t), candidate, &
                                                   error_msg)
                    if (len_trim(error_msg) > 0) return
                    if (.not. emit_liric_f32_fcmp(context%session, LR_FCMP_OGT, &
                                                   candidate, value, condition, &
                                                   error_msg)) return
                    call select_value(context, condition, candidate, value, selected, &
                                      error_msg)
                    if (len_trim(error_msg) > 0) return
                    value = selected
                end do
            case ('minval')
                call load_array_linear_element(context, source_index, 0_c_int64_t, &
                                               value, error_msg)
                if (len_trim(error_msg) > 0) return
                do i = 1, array_size - 1
                    call load_array_linear_element(context, source_index, &
                                                   int(i, c_int64_t), candidate, &
                                                   error_msg)
                    if (len_trim(error_msg) > 0) return
                    if (.not. emit_liric_f32_fcmp(context%session, LR_FCMP_OLT, &
                                                   candidate, value, condition, &
                                                   error_msg)) return
                    call select_value(context, condition, candidate, value, selected, &
                                      error_msg)
                    if (len_trim(error_msg) > 0) return
                    value = selected
                end do
            case default
                error_msg = 'unsupported array reduction intrinsic: '// &
                            trim(reduction_name)
                return
            end select
        case (VALUE_F64)
            select case (trim(reduction_name))
            case ('sum')
                value = liric_f64_immediate(context%session, 0.0_c_double)
                do i = 0, array_size - 1
                    call load_array_linear_element(context, source_index, &
                                                   int(i, c_int64_t), candidate, &
                                                   error_msg)
                    if (len_trim(error_msg) > 0) return
                    if (.not. emit_liric_f64_binary(context%session, LR_OP_FADD, &
                                                    value, candidate, next_value, &
                                                    error_msg)) return
                    value = next_value
                end do
            case ('product')
                value = liric_f64_immediate(context%session, 1.0_c_double)
                do i = 0, array_size - 1
                    call load_array_linear_element(context, source_index, &
                                                   int(i, c_int64_t), candidate, &
                                                   error_msg)
                    if (len_trim(error_msg) > 0) return
                    if (.not. emit_liric_f64_binary(context%session, LR_OP_FMUL, &
                                                    value, candidate, next_value, &
                                                    error_msg)) return
                    value = next_value
                end do
            case ('norm2')
                ! L2 norm: sqrt of the sum of squares of the elements.
                value = liric_f64_immediate(context%session, 0.0_c_double)
                do i = 0, array_size - 1
                    call load_array_linear_element(context, source_index, &
                                                   int(i, c_int64_t), candidate, &
                                                   error_msg)
                    if (len_trim(error_msg) > 0) return
                    if (.not. emit_liric_f64_binary(context%session, LR_OP_FMUL, &
                                                    candidate, candidate, &
                                                    next_value, error_msg)) return
                    if (.not. emit_liric_f64_binary(context%session, LR_OP_FADD, &
                                                    value, next_value, candidate, &
                                                    error_msg)) return
                    value = candidate
                end do
                if (.not. emit_liric_f64_call(context%session, 'sqrt', &
                                              [value], next_value, error_msg)) &
                    return
                value = next_value
            case ('maxval')
                call load_array_linear_element(context, source_index, 0_c_int64_t, &
                                               value, error_msg)
                if (len_trim(error_msg) > 0) return
                do i = 1, array_size - 1
                    call load_array_linear_element(context, source_index, &
                                                   int(i, c_int64_t), candidate, &
                                                   error_msg)
                    if (len_trim(error_msg) > 0) return
                    if (.not. emit_liric_f64_fcmp(context%session, LR_FCMP_OGT, &
                                                   candidate, value, condition, &
                                                   error_msg)) return
                    call select_value(context, condition, candidate, value, selected, &
                                      error_msg)
                    if (len_trim(error_msg) > 0) return
                    value = selected
                end do
            case ('minval')
                call load_array_linear_element(context, source_index, 0_c_int64_t, &
                                               value, error_msg)
                if (len_trim(error_msg) > 0) return
                do i = 1, array_size - 1
                    call load_array_linear_element(context, source_index, &
                                                   int(i, c_int64_t), candidate, &
                                                   error_msg)
                    if (len_trim(error_msg) > 0) return
                    if (.not. emit_liric_f64_fcmp(context%session, LR_FCMP_OLT, &
                                                   candidate, value, condition, &
                                                   error_msg)) return
                    call select_value(context, condition, candidate, value, selected, &
                                      error_msg)
                    if (len_trim(error_msg) > 0) return
                    value = selected
                end do
            case default
                error_msg = 'unsupported array reduction intrinsic: '// &
                            trim(reduction_name)
                return
            end select
        case (VALUE_I64, VALUE_I8, VALUE_I16)
            error_msg = trim(reduction_name)//' over a non-default-kind '// &
                        'integer array is not supported'
            return
        case default
            select case (trim(reduction_name))
            case ('sum')
                value = i32_immediate(context%session, 0_c_int64_t)
                do i = 0, array_size - 1
                    call load_array_linear_element(context, source_index, &
                                                   int(i, c_int64_t), candidate, &
                                                   error_msg)
                    if (len_trim(error_msg) > 0) return
                    if (.not. emit_i32_binary(context%session, LR_OP_ADD, value, &
                                              candidate, next_value, error_msg)) return
                    value = next_value
                end do
            case ('product')
                value = i32_immediate(context%session, 1_c_int64_t)
                do i = 0, array_size - 1
                    call load_array_linear_element(context, source_index, &
                                                   int(i, c_int64_t), candidate, &
                                                   error_msg)
                    if (len_trim(error_msg) > 0) return
                    if (.not. emit_i32_binary(context%session, LR_OP_MUL, value, &
                                              candidate, next_value, error_msg)) return
                    value = next_value
                end do
            case ('maxval')
                predicate = LR_CMP_SGT
                call load_array_linear_element(context, source_index, 0_c_int64_t, &
                                               value, error_msg)
                if (len_trim(error_msg) > 0) return
                do i = 1, array_size - 1
                    call load_array_linear_element(context, source_index, &
                                                   int(i, c_int64_t), candidate, &
                                                   error_msg)
                    if (len_trim(error_msg) > 0) return
                    if (.not. emit_liric_i32_icmp(context%session, predicate, &
                                                   candidate, value, condition, &
                                                   error_msg)) return
                    call select_value(context, condition, candidate, value, selected, &
                                      error_msg)
                    if (len_trim(error_msg) > 0) return
                    value = selected
                end do
            case ('minval')
                predicate = LR_CMP_SLT
                call load_array_linear_element(context, source_index, 0_c_int64_t, &
                                               value, error_msg)
                if (len_trim(error_msg) > 0) return
                do i = 1, array_size - 1
                    call load_array_linear_element(context, source_index, &
                                                   int(i, c_int64_t), candidate, &
                                                   error_msg)
                    if (len_trim(error_msg) > 0) return
                    if (.not. emit_liric_i32_icmp(context%session, predicate, &
                                                   candidate, value, condition, &
                                                   error_msg)) return
                    call select_value(context, condition, candidate, value, selected, &
                                      error_msg)
                    if (len_trim(error_msg) > 0) return
                    value = selected
                end do
            case default
                error_msg = 'unsupported array reduction intrinsic: '// &
                            trim(reduction_name)
                return
            end select
        end select

        call set_empty(error_msg)
    end subroutine lower_array_reduction_intrinsic

    subroutine lower_section_reduction(arena, slice_node, context, value, &
                                       reduction_name, error_msg)
        !! Reduce sum/product/maxval/minval over an array section a(lo:hi:st).
        !! Walks the section in storage order via the section element loader and
        !! folds with the accumulator step matching the section's element kind.
        type(ast_arena_t), intent(in) :: arena
        type(array_slice_node), intent(in) :: slice_node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=*), intent(in) :: reduction_name
        character(len=:), allocatable, intent(out) :: error_msg
        type(array_section_info_t) :: info
        type(lr_operand_desc_t) :: candidate
        integer(c_int64_t) :: total, k
        integer :: vk

        call describe_array_section(arena, slice_node, context, info, error_msg)
        if (len_trim(error_msg) > 0) return
        total = array_section_total_elements(info)
        if (total <= 0_c_int64_t) then
            error_msg = trim(reduction_name)//' requires a non-empty section'
            return
        end if
        vk = context%symbols(info%source_index)%value_kind

        call lower_array_section_element_from_info(info, 0_c_int64_t, context, &
                                                   value, error_msg)
        if (len_trim(error_msg) > 0) return
        select case (trim(reduction_name))
        case ('sum', 'product', 'maxval', 'minval')
        case default
            error_msg = 'unsupported array reduction intrinsic: '// &
                        trim(reduction_name)
            return
        end select
        do k = 1_c_int64_t, total - 1_c_int64_t
            call lower_array_section_element_from_info(info, k, context, &
                                                       candidate, error_msg)
            if (len_trim(error_msg) > 0) return
            call fold_reduction_step(context, vk, reduction_name, value, &
                                     candidate, error_msg)
            if (len_trim(error_msg) > 0) return
        end do
        call set_empty(error_msg)
    end subroutine lower_section_reduction

    subroutine fold_reduction_step(context, vk, reduction_name, acc, candidate, &
                                   error_msg)
        !! One accumulation step for a section reduction: acc := acc op candidate
        !! for the element kind vk. maxval/minval pick the candidate via a typed
        !! comparison and select.
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: vk
        character(len=*), intent(in) :: reduction_name
        type(lr_operand_desc_t), intent(inout) :: acc
        type(lr_operand_desc_t), intent(in) :: candidate
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: next_value, condition, selected
        integer(c_int) :: predicate

        call set_empty(error_msg)
        if (vk == VALUE_I64 .or. vk == VALUE_I8 .or. vk == VALUE_I16) then
            error_msg = trim(reduction_name)//' over a non-default-kind '// &
                        'integer array section is not supported'
            return
        end if
        select case (trim(reduction_name))
        case ('sum')
            if (vk == VALUE_F32) then
                if (.not. emit_liric_f32_binary(context%session, LR_OP_FADD, acc, &
                                                candidate, next_value, error_msg)) return
            else if (vk == VALUE_F64) then
                if (.not. emit_liric_f64_binary(context%session, LR_OP_FADD, acc, &
                                                candidate, next_value, error_msg)) return
            else
                if (.not. emit_i32_binary(context%session, LR_OP_ADD, acc, &
                                          candidate, next_value, error_msg)) return
            end if
            acc = next_value
        case ('product')
            if (vk == VALUE_F32) then
                if (.not. emit_liric_f32_binary(context%session, LR_OP_FMUL, acc, &
                                                candidate, next_value, error_msg)) return
            else if (vk == VALUE_F64) then
                if (.not. emit_liric_f64_binary(context%session, LR_OP_FMUL, acc, &
                                                candidate, next_value, error_msg)) return
            else
                if (.not. emit_i32_binary(context%session, LR_OP_MUL, acc, &
                                          candidate, next_value, error_msg)) return
            end if
            acc = next_value
        case ('maxval', 'minval')
            if (vk == VALUE_F32) then
                if (trim(reduction_name) == 'maxval') then
                    if (.not. emit_liric_f32_fcmp(context%session, LR_FCMP_OGT, &
                            candidate, acc, condition, error_msg)) return
                else
                    if (.not. emit_liric_f32_fcmp(context%session, LR_FCMP_OLT, &
                            candidate, acc, condition, error_msg)) return
                end if
            else if (vk == VALUE_F64) then
                if (trim(reduction_name) == 'maxval') then
                    if (.not. emit_liric_f64_fcmp(context%session, LR_FCMP_OGT, &
                            candidate, acc, condition, error_msg)) return
                else
                    if (.not. emit_liric_f64_fcmp(context%session, LR_FCMP_OLT, &
                            candidate, acc, condition, error_msg)) return
                end if
            else
                if (trim(reduction_name) == 'maxval') then
                    predicate = LR_CMP_SGT
                else
                    predicate = LR_CMP_SLT
                end if
                if (.not. emit_liric_i32_icmp(context%session, predicate, &
                        candidate, acc, condition, error_msg)) return
            end if
            call select_value(context, condition, candidate, acc, selected, error_msg)
            if (len_trim(error_msg) > 0) return
            acc = selected
        end select
    end subroutine fold_reduction_step

    subroutine resolve_i32_array_argument(arena, node, arg_pos, context, &
                                          intrinsic_name, arg_role, symbol_index, &
                                          error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        integer, intent(in) :: arg_pos
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: intrinsic_name
        character(len=*), intent(in) :: arg_role
        integer, intent(out) :: symbol_index
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: arg_name

        symbol_index = 0
        if (.not. allocated(node%arg_indices) .or. size(node%arg_indices) < arg_pos) then
            error_msg = trim(intrinsic_name)//' requires '//trim(arg_role)//' arguments'
            return
        end if
        if (.not. is_identifier(arena, node%arg_indices(arg_pos))) then
            error_msg = trim(intrinsic_name)//' requires identifier '// &
                        trim(arg_role)//' arguments'
            return
        end if

        call get_identifier_name(arena, node%arg_indices(arg_pos), arg_name, error_msg)
        if (len_trim(error_msg) > 0) return

        symbol_index = find_symbol(context, arg_name)
        if (symbol_index <= 0) then
            error_msg = trim(intrinsic_name)//' '//trim(arg_role)// &
                        ' was not declared: '//trim(arg_name)
            return
        end if
        if (.not. context%symbols(symbol_index)%is_array) then
            error_msg = trim(intrinsic_name)//' requires array '//trim(arg_role)// &
                        ': '//trim(arg_name)
            return
        end if
        select case (context%symbols(symbol_index)%value_kind)
        case (VALUE_I32, VALUE_F32, VALUE_F64)
            ! integer, real(4), and real(8) arrays are supported
        case default
            error_msg = trim(intrinsic_name)//' only supports integer and real arrays'
            return
        end select

        call set_empty(error_msg)
    end subroutine resolve_i32_array_argument

    subroutine lower_matmul_assignment(arena, node, symbol_index, context, &
                                       error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(assignment_node), intent(in) :: node
        integer, intent(in) :: symbol_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: a_value
        type(lr_operand_desc_t) :: b_value
        type(lr_operand_desc_t) :: product
        type(lr_operand_desc_t) :: sum
        type(lr_operand_desc_t) :: next_sum
        integer :: a_index
        integer :: b_index
        integer :: i
        integer :: j
        integer :: k
        integer(c_int64_t) :: a_rows
        integer(c_int64_t) :: a_cols
        integer(c_int64_t) :: b_rows
        integer(c_int64_t) :: b_cols
        integer(c_int64_t) :: target_rows
        integer(c_int64_t) :: target_cols
        integer(c_int64_t) :: a_row
        integer(c_int64_t) :: a_col
        integer(c_int64_t) :: b_row
        integer(c_int64_t) :: b_col
        integer(c_int64_t) :: linear_index
        integer(c_int64_t) :: a_row_base
        integer(c_int64_t) :: a_col_base
        integer(c_int64_t) :: b_row_base
        integer(c_int64_t) :: b_col_base

        select type (rhs => arena%entries(node%value_index)%node)
        type is (call_or_subscript_node)
            if (.not. allocated(rhs%name) .or. .not. same_name(rhs%name, 'matmul')) then
                error_msg = 'matmul requires a matmul intrinsic'
                return
            end if
            if (.not. allocated(rhs%arg_indices) .or. size(rhs%arg_indices) /= 2) then
                error_msg = 'matmul requires exactly two array arguments'
                return
            end if
            call resolve_i32_array_argument(arena, rhs, 1, context, 'matmul', &
                                            'left operand', a_index, error_msg)
            if (len_trim(error_msg) > 0) return
            call resolve_i32_array_argument(arena, rhs, 2, context, 'matmul', &
                                            'right operand', b_index, error_msg)
            if (len_trim(error_msg) > 0) return

            if (context%symbols(a_index)%array_rank /= 2) then
                error_msg = 'matmul requires rank-2 left operand arrays'
                return
            end if
            if (context%symbols(b_index)%array_rank /= 2) then
                error_msg = 'matmul requires rank-2 right operand arrays'
                return
            end if
            if (context%symbols(symbol_index)%array_rank /= 2) then
                error_msg = 'matmul result must be assigned to a rank-2 array'
                return
            end if

            a_rows = int(context%symbols(a_index)%array_dim_sizes(1), c_int64_t)
            a_cols = int(context%symbols(a_index)%array_dim_sizes(2), c_int64_t)
            b_rows = int(context%symbols(b_index)%array_dim_sizes(1), c_int64_t)
            b_cols = int(context%symbols(b_index)%array_dim_sizes(2), c_int64_t)
            target_rows = int(context%symbols(symbol_index)%array_dim_sizes(1), &
                              c_int64_t)
            target_cols = int(context%symbols(symbol_index)%array_dim_sizes(2), &
                              c_int64_t)
            if (a_cols /= b_rows) then
                error_msg = 'matmul requires conforming matrix operands'
                return
            end if
            if (target_rows /= a_rows .or. target_cols /= b_cols) then
                error_msg = 'matmul result shape does not match the target array'
                return
            end if

            a_row_base = int(context%symbols(a_index)%array_dim_lowers(1), c_int64_t)
            a_col_base = int(context%symbols(a_index)%array_dim_lowers(2), c_int64_t)
            b_row_base = int(context%symbols(b_index)%array_dim_lowers(1), c_int64_t)
            b_col_base = int(context%symbols(b_index)%array_dim_lowers(2), c_int64_t)

            associate (vk => context%symbols(symbol_index)%value_kind)
            do j = 0, int(target_cols) - 1
                do i = 0, int(target_rows) - 1
                    if (vk == VALUE_F32) then
                        sum = liric_f32_immediate(context%session, 0.0_c_float)
                    else if (vk == VALUE_F64) then
                        sum = liric_f64_immediate(context%session, 0.0_c_double)
                    else
                        sum = i32_immediate(context%session, 0_c_int64_t)
                    end if
                    do k = 0, int(a_cols) - 1
                        a_row = a_row_base + int(i, c_int64_t)
                        a_col = a_col_base + int(k, c_int64_t)
                        b_row = b_row_base + int(k, c_int64_t)
                        b_col = b_col_base + int(j, c_int64_t)
                        call load_i32_array_element(context, a_index, a_row, &
                                                    a_value, error_msg, a_col)
                        if (len_trim(error_msg) > 0) return
                        call load_i32_array_element(context, b_index, b_row, &
                                                    b_value, error_msg, b_col)
                        if (len_trim(error_msg) > 0) return
                        if (vk == VALUE_F32) then
                            if (.not. emit_liric_f32_binary(context%session, LR_OP_FMUL, &
                                                            a_value, b_value, product, &
                                                            error_msg)) return
                            if (.not. emit_liric_f32_binary(context%session, LR_OP_FADD, &
                                                            sum, product, next_sum, &
                                                            error_msg)) return
                        else if (vk == VALUE_F64) then
                            if (.not. emit_liric_f64_binary(context%session, LR_OP_FMUL, &
                                                            a_value, b_value, product, &
                                                            error_msg)) return
                            if (.not. emit_liric_f64_binary(context%session, LR_OP_FADD, &
                                                            sum, product, next_sum, &
                                                            error_msg)) return
                        else
                            if (.not. emit_i32_binary(context%session, LR_OP_MUL, &
                                                      a_value, b_value, product, &
                                                      error_msg)) return
                            if (.not. emit_i32_binary(context%session, LR_OP_ADD, sum, &
                                                      product, next_sum, error_msg)) return
                        end if
                        sum = next_sum
                    end do
                    linear_index = int(i, c_int64_t) + target_rows * int(j, c_int64_t)
                    call store_array_linear_element(context, symbol_index, &
                                                    linear_index, sum, error_msg)
                    if (len_trim(error_msg) > 0) return
                end do
            end do
            end associate
            call set_empty(error_msg)
        class default
            error_msg = 'matmul requires an intrinsic call'
            return
        end select
    end subroutine lower_matmul_assignment

    subroutine lower_transpose_assignment(arena, node, symbol_index, context, &
                                          error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(assignment_node), intent(in) :: node
        integer, intent(in) :: symbol_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: value
        integer :: source_index
        integer :: i
        integer :: j
        integer(c_int64_t) :: source_rows
        integer(c_int64_t) :: source_cols
        integer(c_int64_t) :: target_rows
        integer(c_int64_t) :: target_cols
        integer(c_int64_t) :: source_row_base
        integer(c_int64_t) :: source_col_base
        integer(c_int64_t) :: source_row
        integer(c_int64_t) :: source_col
        integer(c_int64_t) :: linear_index

        select type (rhs => arena%entries(node%value_index)%node)
        type is (call_or_subscript_node)
            if (.not. allocated(rhs%name) .or. .not. same_name(rhs%name, 'transpose')) then
                error_msg = 'transpose requires a transpose intrinsic'
                return
            end if
            if (.not. allocated(rhs%arg_indices) .or. size(rhs%arg_indices) /= 1) then
                error_msg = 'transpose requires exactly one array argument'
                return
            end if
            call resolve_i32_array_argument(arena, rhs, 1, context, 'transpose', &
                                            'source array', source_index, error_msg)
            if (len_trim(error_msg) > 0) return
            if (context%symbols(source_index)%array_rank /= 2) then
                error_msg = 'transpose requires a rank-2 source array'
                return
            end if
            if (context%symbols(symbol_index)%array_rank /= 2) then
                error_msg = 'transpose result must be assigned to a rank-2 array'
                return
            end if

            source_rows = int(context%symbols(source_index)%array_dim_sizes(1), &
                              c_int64_t)
            source_cols = int(context%symbols(source_index)%array_dim_sizes(2), &
                              c_int64_t)
            target_rows = int(context%symbols(symbol_index)%array_dim_sizes(1), &
                              c_int64_t)
            target_cols = int(context%symbols(symbol_index)%array_dim_sizes(2), &
                              c_int64_t)
            if (target_rows /= source_cols .or. target_cols /= source_rows) then
                error_msg = 'transpose result shape does not match the target array'
                return
            end if

            source_row_base = int(context%symbols(source_index)%array_dim_lowers(1), &
                                   c_int64_t)
            source_col_base = int(context%symbols(source_index)%array_dim_lowers(2), &
                                   c_int64_t)

            do j = 0, int(target_cols) - 1
                do i = 0, int(target_rows) - 1
                    source_row = source_row_base + int(j, c_int64_t)
                    source_col = source_col_base + int(i, c_int64_t)
                    call load_i32_array_element(context, source_index, source_row, &
                                                value, error_msg, source_col)
                    if (len_trim(error_msg) > 0) return
                    linear_index = int(i, c_int64_t) + target_rows * int(j, c_int64_t)
                    call store_array_linear_element(context, symbol_index, &
                                                    linear_index, value, error_msg)
                    if (len_trim(error_msg) > 0) return
                end do
            end do
            call set_empty(error_msg)
        class default
            error_msg = 'transpose requires an intrinsic call'
            return
        end select
    end subroutine lower_transpose_assignment

    ! cshift(a, s) / eoshift(a, s) for a rank-1 array. circular selects the
    ! circular variant (cshift); otherwise eoshift fills shifted-out positions
    ! with the type zero. The shift count s must be a compile-time integer; only
    ! default eoshift boundary (zero / .false.) is supported.
    subroutine lower_shift_assignment(arena, node, symbol_index, context, &
                                      circular, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(assignment_node), intent(in) :: node
        integer, intent(in) :: symbol_index
        type(lowering_context_t), intent(inout) :: context
        logical, intent(in) :: circular
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: value
        integer :: source_index
        integer :: i, n
        integer(c_int64_t) :: shift
        integer(c_int64_t) :: src
        character(len=:), allocatable :: intrinsic_name

        if (circular) then
            intrinsic_name = 'cshift'
        else
            intrinsic_name = 'eoshift'
        end if
        select type (rhs => arena%entries(node%value_index)%node)
        type is (call_or_subscript_node)
            if (.not. allocated(rhs%arg_indices) .or. size(rhs%arg_indices) < 2) then
                error_msg = trim(intrinsic_name)//' requires an array and a shift'
                return
            end if
            if (size(rhs%arg_indices) > 2) then
                error_msg = trim(intrinsic_name)// &
                            ' supports only positional array and shift arguments'
                return
            end if
            call resolve_i32_array_argument(arena, rhs, 1, context, &
                                            intrinsic_name, 'source array', &
                                            source_index, error_msg)
            if (len_trim(error_msg) > 0) return
            if (context%symbols(source_index)%array_rank /= 1) then
                error_msg = trim(intrinsic_name)//' supports only rank-1 arrays'
                return
            end if
            if (context%symbols(symbol_index)%array_rank /= 1) then
                error_msg = trim(intrinsic_name)// &
                            ' result must be assigned to a rank-1 array'
                return
            end if
            n = context%symbols(source_index)%array_size
            if (context%symbols(symbol_index)%array_size /= n) then
                error_msg = trim(intrinsic_name)// &
                            ' result shape does not match the target array'
                return
            end if
            call eval_i32_constant(arena, rhs%arg_indices(2), context, shift, &
                                   error_msg)
            if (len_trim(error_msg) > 0) return
            do i = 0, n - 1
                src = int(i, c_int64_t) + shift
                if (circular) then
                    src = modulo(src, int(n, c_int64_t))
                    call load_array_linear_element(context, source_index, src, &
                                                   value, error_msg)
                    if (len_trim(error_msg) > 0) return
                else if (src >= 0_c_int64_t .and. src < int(n, c_int64_t)) then
                    call load_array_linear_element(context, source_index, src, &
                                                   value, error_msg)
                    if (len_trim(error_msg) > 0) return
                else
                    call zero_operand(context, &
                                      context%symbols(symbol_index)%value_kind, &
                                      value)
                end if
                call store_array_linear_element(context, symbol_index, &
                                                int(i, c_int64_t), value, error_msg)
                if (len_trim(error_msg) > 0) return
            end do
            call set_empty(error_msg)
        class default
            error_msg = trim(intrinsic_name)//' requires an intrinsic call'
            return
        end select
    end subroutine lower_shift_assignment

    ! merge(tsource, fsource, mask) for rank-1 arrays: per element pick from
    ! tsource where mask is .true., else from fsource. All three arrays must
    ! share the result shape; mask is a logical array stored as i32 (nonzero
    ! means .true.).
    subroutine lower_merge_assignment(arena, node, symbol_index, context, &
                                      error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(assignment_node), intent(in) :: node
        integer, intent(in) :: symbol_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: t_val, f_val, mask_val, cond, selected
        type(lr_operand_desc_t) :: zero
        integer :: t_index, f_index, m_index
        integer :: i, n

        select type (rhs => arena%entries(node%value_index)%node)
        type is (call_or_subscript_node)
            if (.not. allocated(rhs%arg_indices) .or. size(rhs%arg_indices) /= 3) then
                error_msg = 'merge requires tsource, fsource, and mask arrays'
                return
            end if
            call resolve_i32_array_argument(arena, rhs, 1, context, 'merge', &
                                            'tsource', t_index, error_msg)
            if (len_trim(error_msg) > 0) return
            call resolve_i32_array_argument(arena, rhs, 2, context, 'merge', &
                                            'fsource', f_index, error_msg)
            if (len_trim(error_msg) > 0) return
            call resolve_logical_mask_argument(arena, rhs, 3, context, 'merge', &
                                               m_index, error_msg)
            if (len_trim(error_msg) > 0) return
            if (context%symbols(symbol_index)%array_rank /= 1 .or. &
                context%symbols(t_index)%array_rank /= 1 .or. &
                context%symbols(f_index)%array_rank /= 1 .or. &
                context%symbols(m_index)%array_rank /= 1) then
                error_msg = 'merge supports only rank-1 arrays'
                return
            end if
            n = context%symbols(symbol_index)%array_size
            if (context%symbols(t_index)%array_size /= n .or. &
                context%symbols(f_index)%array_size /= n .or. &
                context%symbols(m_index)%array_size /= n) then
                error_msg = 'merge operands must share the result shape'
                return
            end if
            zero = i32_immediate(context%session, 0_c_int64_t)
            do i = 0, n - 1
                call load_array_linear_element(context, m_index, &
                                               int(i, c_int64_t), mask_val, &
                                               error_msg)
                if (len_trim(error_msg) > 0) return
                if (.not. emit_liric_i32_icmp(context%session, LR_CMP_NE, &
                        mask_val, zero, cond, error_msg)) return
                call load_array_linear_element(context, t_index, &
                                               int(i, c_int64_t), t_val, error_msg)
                if (len_trim(error_msg) > 0) return
                call load_array_linear_element(context, f_index, &
                                               int(i, c_int64_t), f_val, error_msg)
                if (len_trim(error_msg) > 0) return
                call select_value(context, cond, t_val, f_val, selected, error_msg)
                if (len_trim(error_msg) > 0) return
                call store_array_linear_element(context, symbol_index, &
                                                int(i, c_int64_t), selected, &
                                                error_msg)
                if (len_trim(error_msg) > 0) return
            end do
            call set_empty(error_msg)
        class default
            error_msg = 'merge requires an intrinsic call'
            return
        end select
    end subroutine lower_merge_assignment

    ! spread(source, dim, ncopies) for a rank-1 source replicated into a rank-2
    ! target. dim and ncopies must be compile-time integers. With dim=2 the
    ! result is (m, ncopies) and result(i,j) = source(i); with dim=1 it is
    ! (ncopies, m) and result(i,j) = source(j).
    subroutine lower_spread_assignment(arena, node, symbol_index, context, &
                                       error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(assignment_node), intent(in) :: node
        integer, intent(in) :: symbol_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: value
        integer :: source_index
        integer :: i, j, m
        integer(c_int64_t) :: dim_val, ncopies
        integer(c_int64_t) :: target_rows, target_cols
        integer(c_int64_t) :: src_index, linear_index

        select type (rhs => arena%entries(node%value_index)%node)
        type is (call_or_subscript_node)
            if (.not. allocated(rhs%arg_indices) .or. size(rhs%arg_indices) /= 3) then
                error_msg = 'spread requires source, dim, and ncopies arguments'
                return
            end if
            call resolve_i32_array_argument(arena, rhs, 1, context, 'spread', &
                                            'source array', source_index, error_msg)
            if (len_trim(error_msg) > 0) return
            if (context%symbols(source_index)%array_rank /= 1) then
                error_msg = 'spread supports only a rank-1 source array'
                return
            end if
            if (context%symbols(symbol_index)%array_rank /= 2) then
                error_msg = 'spread result must be assigned to a rank-2 array'
                return
            end if
            call eval_i32_constant(arena, rhs%arg_indices(2), context, dim_val, &
                                   error_msg)
            if (len_trim(error_msg) > 0) return
            call eval_i32_constant(arena, rhs%arg_indices(3), context, ncopies, &
                                   error_msg)
            if (len_trim(error_msg) > 0) return
            if (dim_val /= 1_c_int64_t .and. dim_val /= 2_c_int64_t) then
                error_msg = 'spread dim must be 1 or 2 for a rank-1 source'
                return
            end if
            m = context%symbols(source_index)%array_size
            target_rows = int(context%symbols(symbol_index)%array_dim_sizes(1), &
                              c_int64_t)
            target_cols = int(context%symbols(symbol_index)%array_dim_sizes(2), &
                              c_int64_t)
            if (dim_val == 2_c_int64_t) then
                if (target_rows /= int(m, c_int64_t) .or. target_cols /= ncopies) then
                    error_msg = 'spread result shape does not match the target array'
                    return
                end if
            else
                if (target_rows /= ncopies .or. target_cols /= int(m, c_int64_t)) then
                    error_msg = 'spread result shape does not match the target array'
                    return
                end if
            end if
            do j = 0, int(target_cols) - 1
                do i = 0, int(target_rows) - 1
                    if (dim_val == 2_c_int64_t) then
                        src_index = int(i, c_int64_t)
                    else
                        src_index = int(j, c_int64_t)
                    end if
                    call load_array_linear_element(context, source_index, &
                                                   src_index, value, error_msg)
                    if (len_trim(error_msg) > 0) return
                    linear_index = int(i, c_int64_t) + target_rows * int(j, c_int64_t)
                    call store_array_linear_element(context, symbol_index, &
                                                    linear_index, value, error_msg)
                    if (len_trim(error_msg) > 0) return
                end do
            end do
            call set_empty(error_msg)
        class default
            error_msg = 'spread requires an intrinsic call'
            return
        end select
    end subroutine lower_spread_assignment

    ! Number of elements a pack/unpack mask iterates over: the logical array's
    ! size, or the compared array's size for a comparison mask. Zero when the
    ! mask argument is neither.
    integer function mask_extent(arena, mask_index, context) result(extent)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: mask_index
        type(lowering_context_t), intent(in) :: context
        integer :: sym, scalar_index
        logical :: array_on_left
        character(len=:), allocatable :: source_op, name, err, cmp_err

        extent = 0
        call resolve_comparison_mask(arena, mask_index, context, sym, &
                                     scalar_index, array_on_left, source_op, &
                                     cmp_err)
        if (len_trim(cmp_err) == 0 .and. sym > 0) then
            extent = context%symbols(sym)%array_size
            return
        end if
        if (is_identifier(arena, mask_index)) then
            call get_identifier_name(arena, mask_index, name, err)
            if (len_trim(err) > 0) return
            sym = find_symbol(context, name)
            if (sym <= 0) return
            if (context%symbols(sym)%is_array) extent = &
                context%symbols(sym)%array_size
        end if
    end function mask_extent

    ! The i-th element of a pack/unpack mask as a 0/1 i32 operand. Handles a
    ! logical array variable and an elementwise comparison expression.
    subroutine mask_element_value(arena, mask_index, context, linear_index, &
                                  value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: mask_index
        type(lowering_context_t), intent(inout) :: context
        integer(c_int64_t), intent(in) :: linear_index
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: sym, scalar_index
        logical :: array_on_left
        character(len=:), allocatable :: source_op, name

        call resolve_comparison_mask(arena, mask_index, context, sym, &
                                     scalar_index, array_on_left, source_op, &
                                     error_msg)
        if (len_trim(error_msg) > 0) return
        if (sym > 0) then
            call lower_comparison_mask_element(context, sym, scalar_index, &
                array_on_left, source_op, arena, linear_index, value, error_msg)
            return
        end if
        call get_identifier_name(arena, mask_index, name, error_msg)
        if (len_trim(error_msg) > 0) return
        sym = find_symbol(context, name)
        if (sym <= 0) then
            error_msg = 'pack/unpack mask was not declared: '//trim(name)
            return
        end if
        call load_array_linear_element(context, sym, linear_index, value, &
                                       error_msg)
    end subroutine mask_element_value

    ! pack(array, mask): store the source elements whose mask is .true. into the
    ! fixed-size target in order, tracking the destination position in a cursor.
    subroutine lower_pack_assignment(arena, node, target_index, context, &
                                     error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(assignment_node), intent(in) :: node
        integer, intent(in) :: target_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: source_index, extent, i
        type(lr_operand_desc_t) :: cursor_addr, mask_val, cond
        type(lr_operand_desc_t) :: elem, cursor, next_cursor
        integer(c_int32_t) :: store_block, skip_block

        call resolve_pack_arguments(arena, node%value_index, context, &
                                    source_index, extent, error_msg)
        if (len_trim(error_msg) > 0) return

        if (.not. emit_i32_alloca(context%session, cursor_addr, error_msg)) return
        if (.not. emit_i32_store(context%session, &
                i32_immediate(context%session, 0_c_int64_t), cursor_addr, &
                error_msg)) return

        do i = 0, extent - 1
            call mask_element_value(arena, pack_mask_index(arena, node%value_index), &
                                    context, int(i, c_int64_t), mask_val, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_liric_i32_icmp(context%session, LR_CMP_NE, mask_val, &
                    i32_immediate(context%session, 0_c_int64_t), cond, &
                    error_msg)) return
            store_block = create_liric_block(context%session)
            skip_block = create_liric_block(context%session)
            if (.not. emit_liric_condbr(context%session, cond, store_block, &
                                        skip_block, error_msg)) return

            ! store block: target[cursor] = source[i]; cursor = cursor + 1
            if (.not. set_liric_block(context%session, store_block, error_msg)) &
                return
            context%current_block_id = store_block
            call load_array_linear_element(context, source_index, &
                                           int(i, c_int64_t), elem, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_i32_load(context%session, cursor_addr, cursor, &
                                    error_msg)) return
            call store_array_element_at_operand(context, target_index, cursor, &
                                                elem, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_i32_binary(context%session, LR_OP_ADD, cursor, &
                    i32_immediate(context%session, 1_c_int64_t), next_cursor, &
                    error_msg)) return
            if (.not. emit_i32_store(context%session, next_cursor, cursor_addr, &
                                     error_msg)) return
            if (.not. emit_liric_br(context%session, skip_block, error_msg)) return

            ! skip block continues the loop.
            if (.not. set_liric_block(context%session, skip_block, error_msg)) &
                return
            context%current_block_id = skip_block
            context%current_block_terminated = .false.
        end do
        call set_empty(error_msg)
    end subroutine lower_pack_assignment

    ! Resolve pack(array, mask): the source array symbol and the mask extent
    ! (which must match the source size). The mask shares the source's extent.
    subroutine resolve_pack_arguments(arena, call_index, context, source_index, &
                                      extent, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: call_index
        type(lowering_context_t), intent(in) :: context
        integer, intent(out) :: source_index
        integer, intent(out) :: extent
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: name

        source_index = 0
        extent = 0
        select type (rhs => arena%entries(call_index)%node)
        type is (call_or_subscript_node)
            if (.not. allocated(rhs%arg_indices)) then
                error_msg = 'pack requires an array and a mask argument'
                return
            end if
            if (size(rhs%arg_indices) < 2) then
                error_msg = 'pack requires an array and a mask argument'
                return
            end if
            call get_identifier_name(arena, rhs%arg_indices(1), name, error_msg)
            if (len_trim(error_msg) > 0) return
            source_index = find_symbol(context, name)
            if (source_index <= 0) then
                error_msg = 'pack array was not declared: '//trim(name)
                return
            end if
            if (.not. context%symbols(source_index)%is_array) then
                error_msg = 'pack requires an array argument: '//trim(name)
                return
            end if
            extent = mask_extent(arena, rhs%arg_indices(2), context)
            if (extent <= 0) then
                error_msg = 'pack mask must be a logical array or comparison'
                return
            end if
        class default
            error_msg = 'pack requires an intrinsic call'
        end select
    end subroutine resolve_pack_arguments

    ! AST index of the mask argument of a pack/unpack call.
    integer function pack_mask_index(arena, call_index) result(mask_index)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: call_index

        mask_index = 0
        select type (rhs => arena%entries(call_index)%node)
        type is (call_or_subscript_node)
            if (allocated(rhs%arg_indices)) then
                if (size(rhs%arg_indices) >= 2) mask_index = rhs%arg_indices(2)
            end if
        end select
    end function pack_mask_index

    ! unpack(vector, mask, field): where mask is .true. take the next vector
    ! element in order, else take field. field may be an array or a scalar.
    subroutine lower_unpack_assignment(arena, node, target_index, context, &
                                       error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(assignment_node), intent(in) :: node
        integer, intent(in) :: target_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: vector_index, mask_idx, extent, i
        integer :: field_sym, field_scalar_index
        logical :: field_is_scalar
        type(lr_operand_desc_t) :: cursor_addr, mask_val, cond
        type(lr_operand_desc_t) :: vec_elem, field_val, cursor, next_cursor
        integer(c_int32_t) :: take_block, else_block, merge_block

        call resolve_unpack_arguments(arena, node%value_index, context, &
                                      vector_index, mask_idx, field_sym, &
                                      field_is_scalar, field_scalar_index, &
                                      extent, error_msg)
        if (len_trim(error_msg) > 0) return

        if (.not. emit_i32_alloca(context%session, cursor_addr, error_msg)) return
        if (.not. emit_i32_store(context%session, &
                i32_immediate(context%session, 0_c_int64_t), cursor_addr, &
                error_msg)) return

        do i = 0, extent - 1
            call mask_element_value(arena, mask_idx, context, int(i, c_int64_t), &
                                    mask_val, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_liric_i32_icmp(context%session, LR_CMP_NE, mask_val, &
                    i32_immediate(context%session, 0_c_int64_t), cond, &
                    error_msg)) return
            take_block = create_liric_block(context%session)
            else_block = create_liric_block(context%session)
            merge_block = create_liric_block(context%session)
            if (.not. emit_liric_condbr(context%session, cond, take_block, &
                                        else_block, error_msg)) return

            ! take block: target[i] = vector[cursor]; cursor = cursor + 1
            if (.not. set_liric_block(context%session, take_block, error_msg)) &
                return
            context%current_block_id = take_block
            if (.not. emit_i32_load(context%session, cursor_addr, cursor, &
                                    error_msg)) return
            call load_array_element_at_operand(context, vector_index, cursor, &
                                               vec_elem, error_msg)
            if (len_trim(error_msg) > 0) return
            call store_array_linear_element(context, target_index, &
                                            int(i, c_int64_t), vec_elem, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_i32_binary(context%session, LR_OP_ADD, cursor, &
                    i32_immediate(context%session, 1_c_int64_t), next_cursor, &
                    error_msg)) return
            if (.not. emit_i32_store(context%session, next_cursor, cursor_addr, &
                                     error_msg)) return
            if (.not. emit_liric_br(context%session, merge_block, error_msg)) return

            ! else block: target[i] = field[i] (or scalar field)
            if (.not. set_liric_block(context%session, else_block, error_msg)) &
                return
            context%current_block_id = else_block
            if (field_is_scalar) then
                call lower_unpack_scalar_field(arena, field_scalar_index, &
                    field_sym, target_index, context, field_val, error_msg)
                if (len_trim(error_msg) > 0) return
            else
                call load_array_linear_element(context, field_sym, &
                                               int(i, c_int64_t), field_val, &
                                               error_msg)
                if (len_trim(error_msg) > 0) return
            end if
            call store_array_linear_element(context, target_index, &
                                            int(i, c_int64_t), field_val, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_liric_br(context%session, merge_block, error_msg)) return

            if (.not. set_liric_block(context%session, merge_block, error_msg)) &
                return
            context%current_block_id = merge_block
            context%current_block_terminated = .false.
        end do
        call set_empty(error_msg)
    end subroutine lower_unpack_assignment

    ! Lower the scalar field value of unpack, formatted to the target element
    ! kind. A literal field has no symbol; lower its expression directly.
    subroutine lower_unpack_scalar_field(arena, field_scalar_index, field_sym, &
                                         target_index, context, value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: field_scalar_index, field_sym, target_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg

        call set_empty(error_msg)
        if (field_sym > 0) then
            value = context%symbols(field_sym)%value
            return
        end if
        select case (context%symbols(target_index)%value_kind)
        case (VALUE_F32)
            call lower_f32_expression(arena, field_scalar_index, context, value, &
                                      error_msg)
        case (VALUE_F64)
            call lower_f64_expression(arena, field_scalar_index, context, value, &
                                      error_msg)
        case default
            call lower_i32_expression(arena, field_scalar_index, context, value, &
                                      error_msg)
        end select
    end subroutine lower_unpack_scalar_field

    ! Resolve unpack(vector, mask, field): the vector symbol, the mask AST
    ! index, the field symbol (array or scalar; 0 for a literal scalar field
    ! whose AST index is returned in field_scalar_index), and the result extent
    ! (the mask extent, which equals the result and field array shape).
    subroutine resolve_unpack_arguments(arena, call_index, context, &
                                        vector_index, mask_idx, field_sym, &
                                        field_is_scalar, field_scalar_index, &
                                        extent, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: call_index
        type(lowering_context_t), intent(in) :: context
        integer, intent(out) :: vector_index, mask_idx, field_sym, extent
        integer, intent(out) :: field_scalar_index
        logical, intent(out) :: field_is_scalar
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: name, name_err

        vector_index = 0
        mask_idx = 0
        field_sym = 0
        field_scalar_index = 0
        field_is_scalar = .false.
        extent = 0
        select type (rhs => arena%entries(call_index)%node)
        type is (call_or_subscript_node)
            if (.not. allocated(rhs%arg_indices)) then
                error_msg = 'unpack requires vector, mask, and field arguments'
                return
            end if
            if (size(rhs%arg_indices) < 3) then
                error_msg = 'unpack requires vector, mask, and field arguments'
                return
            end if
            call get_identifier_name(arena, rhs%arg_indices(1), name, error_msg)
            if (len_trim(error_msg) > 0) return
            vector_index = find_symbol(context, name)
            if (vector_index <= 0 .or. .not. &
                context%symbols(max(vector_index, 1))%is_array) then
                error_msg = 'unpack vector must be a declared array: '//trim(name)
                return
            end if
            mask_idx = rhs%arg_indices(2)
            extent = mask_extent(arena, mask_idx, context)
            if (extent <= 0) then
                error_msg = 'unpack mask must be a logical array or comparison'
                return
            end if
            ! The field is an array variable, a scalar variable, or a scalar
            ! literal/expression. A non-identifier field is a literal scalar.
            if (is_identifier(arena, rhs%arg_indices(3))) then
                call get_identifier_name(arena, rhs%arg_indices(3), name, &
                                         name_err)
                if (len_trim(name_err) == 0) field_sym = find_symbol(context, name)
            end if
            if (field_sym > 0) then
                field_is_scalar = .not. context%symbols(field_sym)%is_array
            else
                field_is_scalar = .true.
                field_scalar_index = rhs%arg_indices(3)
            end if
        class default
            error_msg = 'unpack requires an intrinsic call'
        end select
    end subroutine resolve_unpack_arguments

    ! Resolve a logical mask array argument (merge): an identifier naming a
    ! declared logical array. Stored as i32, so callers compare against zero.
    subroutine resolve_logical_mask_argument(arena, node, arg_pos, context, &
                                             intrinsic_name, symbol_index, &
                                             error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        integer, intent(in) :: arg_pos
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: intrinsic_name
        integer, intent(out) :: symbol_index
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: arg_name

        symbol_index = 0
        if (.not. is_identifier(arena, node%arg_indices(arg_pos))) then
            error_msg = trim(intrinsic_name)//' requires an identifier mask argument'
            return
        end if
        call get_identifier_name(arena, node%arg_indices(arg_pos), arg_name, &
                                 error_msg)
        if (len_trim(error_msg) > 0) return
        symbol_index = find_symbol(context, arg_name)
        if (symbol_index <= 0) then
            error_msg = trim(intrinsic_name)//' mask was not declared: '// &
                        trim(arg_name)
            return
        end if
        if (.not. context%symbols(symbol_index)%is_array .or. &
            context%symbols(symbol_index)%value_kind /= VALUE_LOGICAL) then
            error_msg = trim(intrinsic_name)//' requires a logical array mask: '// &
                        trim(arg_name)
            return
        end if
        call set_empty(error_msg)
    end subroutine resolve_logical_mask_argument

    ! Type-appropriate zero immediate for shift fill values.
    subroutine zero_operand(context, value_kind, value)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: value_kind
        type(lr_operand_desc_t), intent(out) :: value

        select case (value_kind)
        case (VALUE_F32)
            value = liric_f32_immediate(context%session, 0.0_c_float)
        case (VALUE_F64)
            value = liric_f64_immediate(context%session, 0.0_c_double)
        case default
            value = i32_immediate(context%session, 0_c_int64_t)
        end select
    end subroutine zero_operand

    subroutine lower_reshape_assignment(arena, node, symbol_index, context, &
                                        error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(assignment_node), intent(in) :: node
        integer, intent(in) :: symbol_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg

        select type (rhs => arena%entries(node%value_index)%node)
        type is (call_or_subscript_node)
            if (.not. allocated(rhs%name) .or. .not. same_name(rhs%name, 'reshape')) then
                error_msg = 'reshape requires a reshape intrinsic'
                return
            end if
            if (.not. allocated(rhs%arg_indices) .or. size(rhs%arg_indices) /= 2) then
                error_msg = 'reshape requires a source array and a shape array'
                return
            end if
            if (context%symbols(symbol_index)%array_rank < 1 .or. &
                context%symbols(symbol_index)%array_rank > 2) then
                error_msg = 'reshape result must be assigned to a rank-1 or rank-2 array'
                return
            end if
            call check_reshape_shape(arena, rhs%arg_indices(2), symbol_index, &
                                     context, error_msg)
            if (len_trim(error_msg) > 0) return
            call lower_reshape_source(arena, rhs%arg_indices(1), symbol_index, &
                                      context, error_msg)
        class default
            error_msg = 'reshape requires an intrinsic call'
            return
        end select
    end subroutine lower_reshape_assignment

    subroutine check_reshape_shape(arena, shape_index, symbol_index, context, &
                                   error_msg)
        !! The shape argument must be a literal whose rank and extents match the
        !! target array. reshape produces column-major order, so the linear copy
        !! below is correct only when the target's compile-time shape is known.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: shape_index
        integer, intent(in) :: symbol_index
        type(lowering_context_t), intent(in) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i
        integer :: shape_rank
        integer(c_int64_t) :: shape_value

        select type (shape_node => arena%entries(shape_index)%node)
        type is (array_literal_node)
            if (.not. allocated(shape_node%element_indices)) then
                error_msg = 'reshape requires a non-empty shape literal'
                return
            end if
            shape_rank = size(shape_node%element_indices)
            if (shape_rank /= context%symbols(symbol_index)%array_rank) then
                error_msg = 'reshape shape rank does not match the target array'
                return
            end if
            do i = 1, shape_rank
                call parse_i32_constant(arena, shape_node%element_indices(i), &
                                        shape_value, 'reshape shape', error_msg)
                if (len_trim(error_msg) > 0) return
                if (shape_value /= &
                    int(context%symbols(symbol_index)%array_dim_sizes(i), c_int64_t)) then
                    error_msg = 'reshape shape does not match the target array'
                    return
                end if
            end do
        class default
            error_msg = 'reshape requires a literal integer shape argument'
            return
        end select
        call set_empty(error_msg)
    end subroutine check_reshape_shape

    subroutine lower_reshape_source(arena, source_index_in, symbol_index, &
                                    context, error_msg)
        !! Copy the source elements into the target's linear storage. The source
        !! is either a declared rank-1 array or an inline array literal; both are
        !! already laid out in column-major order, so a flat copy suffices.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: source_index_in
        integer, intent(in) :: symbol_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: value
        integer :: source_sym
        integer(c_int64_t) :: target_size
        integer(c_int64_t) :: linear_index
        logical :: source_is_literal

        target_size = int(context%symbols(symbol_index)%array_size, c_int64_t)

        source_is_literal = .false.
        if (node_exists(arena, source_index_in)) then
            select type (s => arena%entries(source_index_in)%node)
            type is (array_literal_node)
                source_is_literal = .true.
            end select
        end if
        if (source_is_literal) then
            call lower_reshape_literal_source(arena, source_index_in, &
                                              symbol_index, context, error_msg)
            return
        end if

        call resolve_reshape_identifier_source(arena, source_index_in, context, &
                                               source_sym, error_msg)
        if (len_trim(error_msg) > 0) return
        if (int(context%symbols(source_sym)%array_size, c_int64_t) /= target_size) then
            error_msg = 'reshape requires source and target arrays of equal size'
            return
        end if
        do linear_index = 0_c_int64_t, target_size - 1_c_int64_t
            call load_array_linear_element(context, source_sym, linear_index, &
                                           value, error_msg)
            if (len_trim(error_msg) > 0) return
            call store_array_linear_element(context, symbol_index, linear_index, &
                                            value, error_msg)
            if (len_trim(error_msg) > 0) return
        end do
        call set_empty(error_msg)
    end subroutine lower_reshape_source

    subroutine lower_reshape_literal_source(arena, source_index, symbol_index, &
                                            context, error_msg)
        !! reshape([...], shape): the literal supplies the column-major elements
        !! directly. Lower each in the target's element kind and store linearly.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: source_index
        integer, intent(in) :: symbol_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: value
        integer, allocatable :: flat(:)
        integer :: i
        integer :: vk

        select type (lit => arena%entries(source_index)%node)
        type is (array_literal_node)
            allocate (flat(0))
            call flatten_constructor_elements(arena, lit%element_indices, flat, &
                                              error_msg)
            if (len_trim(error_msg) > 0) return
        class default
            error_msg = 'reshape source literal is not an array constructor'
            return
        end select

        if (size(flat) /= context%symbols(symbol_index)%array_size) then
            error_msg = 'reshape requires source and target arrays of equal size'
            return
        end if

        vk = context%symbols(symbol_index)%value_kind
        do i = 1, size(flat)
            if (vk == VALUE_F32) then
                call lower_f32_expression(arena, flat(i), context, value, error_msg)
            else if (vk == VALUE_F64) then
                call lower_f64_expression(arena, flat(i), context, value, error_msg)
            else if (vk == VALUE_LOGICAL) then
                call lower_logical_expression(arena, flat(i), context, value, error_msg)
            else
                call lower_constructor_i32_element(arena, flat(i), context, value, &
                                                   error_msg)
            end if
            if (len_trim(error_msg) > 0) return
            call store_array_linear_element(context, symbol_index, &
                                            int(i - 1, c_int64_t), value, error_msg)
            if (len_trim(error_msg) > 0) return
        end do
        call set_empty(error_msg)
    end subroutine lower_reshape_literal_source

    subroutine resolve_reshape_identifier_source(arena, source_index, context, &
                                                 source_sym, error_msg)
        !! A declared array source must be a named array of a supported element
        !! kind; its column-major storage is copied linearly into the target.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: source_index
        type(lowering_context_t), intent(in) :: context
        integer, intent(out) :: source_sym
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: arg_name

        source_sym = 0
        if (.not. is_identifier(arena, source_index)) then
            error_msg = 'reshape requires identifier or literal source array arguments'
            return
        end if
        call get_identifier_name(arena, source_index, arg_name, error_msg)
        if (len_trim(error_msg) > 0) return
        source_sym = find_symbol(context, arg_name)
        if (source_sym <= 0) then
            error_msg = 'reshape source array was not declared: '//trim(arg_name)
            return
        end if
        if (.not. context%symbols(source_sym)%is_array) then
            error_msg = 'reshape requires array source array: '//trim(arg_name)
            return
        end if
        select case (context%symbols(source_sym)%value_kind)
        case (VALUE_I32, VALUE_F32, VALUE_F64)
        case default
            error_msg = 'reshape only supports integer and real arrays'
            return
        end select
        call set_empty(error_msg)
    end subroutine resolve_reshape_identifier_source
