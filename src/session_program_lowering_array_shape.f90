submodule (session_program_lowering_impl) session_program_lowering_array_shape
    implicit none
contains
    module function dim_is_assumed_shape(arena, dim_index) result(is_assumed)
        ! a(:) is parsed as a range_expression with no start/end/stride, or as
        ! an array_bounds_node flagged is_assumed_shape. Either marks one
        ! assumed-shape dimension whose extent comes from the caller's actual.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: dim_index
        logical :: is_assumed

        is_assumed = .false.
        ! FortFront's public declaration contract uses a zero dimension index
        ! for a deferred-shape colon when there is no AST bounds node to point
        ! at.  A non-allocatable declaration with that rank metadata is an
        ! assumed-shape dummy; declaration_is_assumed_shape keeps allocatable
        ! deferred-shape arrays on their separate lowering path.
        if (dim_index == 0) then
            is_assumed = .true.
            return
        end if
        if (.not. node_exists(arena, dim_index)) return
        select type (dim_node => arena%entries(dim_index)%node)
        type is (range_expression_node)
            ! An assumed-shape dummy may specify an explicit lower bound,
            ! e.g. a(0:).  The omitted upper bound, rather than an omitted
            ! lower bound, identifies the assumed-shape dimension.
            is_assumed = dim_node%end_index <= 0 .and. &
                         dim_node%stride_index <= 0
        type is (array_bounds_node)
            is_assumed = dim_node%is_assumed_shape
        end select
    end function dim_is_assumed_shape

    module function declaration_is_assumed_shape(node, context) result(is_assumed)
        ! A declaration is assumed-shape when every dimension is a colon a(:).
        type(declaration_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        logical :: is_assumed
        integer :: i

        is_assumed = .false.
        ! An allocatable colon is a deferred-shape specification, not an
        ! assumed-shape dummy. Its extent is supplied by ALLOCATE later.
        if (node%is_allocatable) return
        if (.not. allocated(node%dimension_indices)) return
        if (size(node%dimension_indices) < 1) return
        do i = 1, size(node%dimension_indices)
            if (.not. dim_is_assumed_shape(context%arena, &
                                           node%dimension_indices(i))) return
        end do
        is_assumed = .true.
    end function declaration_is_assumed_shape

    module function declaration_is_runtime_rank1(node, context) result(is_runtime)
        ! A rank-1 through rank-4 explicit-shape array at least one of whose extents
        ! is a runtime integer expression (a dummy-argument value such as
        ! dimension(n)) rather than a compile-time constant. Allocatable and
        ! assumed-shape declarations are excluded; they have their own lowering
        ! paths.
        type(declaration_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        logical :: is_runtime
        integer :: dim_index, d, rank
        integer(c_int64_t) :: folded
        character(len=:), allocatable :: fold_error

        is_runtime = .false.
        if (.not. node%is_array) return
        if (node%is_allocatable) return
        if (.not. allocated(node%dimension_indices)) return
        rank = size(node%dimension_indices)
        if (rank < 1 .or. rank > 4) return
        if (declaration_is_assumed_shape(node, context)) return
        ! Assumed-size dummies have an intentionally unknown trailing extent.
        ! They use the dedicated base-address path below; treating the bare
        ! asterisk as a runtime integer bound sends external procedures through
        ! the adjustable-array classifier instead (#584).
        if (declaration_is_assumed_size(node, context)) return
        do d = 1, rank
            dim_index = node%dimension_indices(d)
            if (.not. node_exists(context%arena, dim_index)) return
            if (bound_expr_references_variable(context%arena, dim_index, context)) then
                is_runtime = .true.
                cycle
            end if
            select type (dim_node => context%arena%entries(dim_index)%node)
            type is (range_expression_node)
                call eval_i32_constant(context%arena, dim_node%end_index, &
                                       context, folded, fold_error)
            class default
                call eval_i32_constant(context%arena, dim_index, context, &
                                       folded, fold_error)
            end select
            if (len_trim(fold_error) > 0) is_runtime = .true.
        end do
    end function declaration_is_runtime_rank1

    recursive module function bound_expr_references_variable(arena, idx, context) &
            result(has_var)
        ! True when the bound expression names a runtime variable (a non-parameter
        ! symbol in scope). A bound built only from literals/parameters is a
        ! constant expression that must fold at compile time (and be rejected if
        ! invalid, e.g. a(0/0)); a bare function call is not a runtime-variable
        ! extent either. Only genuine variable-driven extents are runtime arrays.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: idx
        type(lowering_context_t), intent(in) :: context
        logical :: has_var
        integer(c_int64_t) :: folded
        character(len=:), allocatable :: fold_error

        ! Default to runtime (safe: never decline a valid runtime array). Only a
        ! provably pure-constant bound - a literal, or arithmetic over literals -
        ! is non-runtime, so it folds at compile time and its errors (a(0/0)) are
        ! reported. An identifier, an intrinsic call (a(count(mask))), or any
        ! other form stays runtime.
        has_var = .true.
        if (.not. node_exists(arena, idx)) return
        select type (n => arena%entries(idx)%node)
        type is (literal_node)
            has_var = .false.
        type is (identifier_node)
            has_var = bound_identifier_references_variable(arena, idx, context)
        type is (binary_op_node)
            has_var = bound_expr_references_variable(arena, n%left_index, context)
            if (.not. has_var) then
                has_var = bound_expr_references_variable(arena, n%right_index, &
                                                         context)
            end if
        type is (range_expression_node)
            has_var = bound_expr_references_variable(arena, n%end_index, context)
        type is (call_or_subscript_node)
            if (.not. allocated(n%name)) return
            if (same_name(n%name, 'size')) then
                call eval_i32_constant(arena, idx, context, folded, fold_error)
                has_var = len_trim(fold_error) > 0
            else if (same_name(n%name, 'huge')) then
                if (.not. allocated(n%arg_indices) .or. &
                    size(n%arg_indices) /= 1) return
                has_var = bound_expr_references_variable(arena, &
                    n%arg_indices(1), context)
            end if
        end select
    end function bound_expr_references_variable

    module function bound_identifier_references_variable(arena, idx, context) &
        result(has_var)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: idx
        type(lowering_context_t), intent(in) :: context
        logical :: has_var
        type(declaration_binding_t) :: binding
        character(len=:), allocatable :: id_name
        character(len=:), allocatable :: error_msg
        integer(c_int64_t) :: kind_value
        integer :: symbol_index

        has_var = .true.
        call resolve_identifier_binding(arena, idx, binding, error_msg)
        if (len_trim(error_msg) > 0) return
        if (binding%found) then
            if (binding%binding_kind == BINDING_NAMED_CONSTANT) has_var = .false.
            return
        end if
        call get_identifier_name(arena, idx, id_name, error_msg)
        if (len_trim(error_msg) > 0) return
        symbol_index = find_symbol_compat(context, id_name)
        if (symbol_index > 0) then
            if (context%symbols(symbol_index)%has_i32_constant) has_var = .false.
            return
        end if
        if (iso_c_binding_kind_value(id_name, kind_value)) has_var = .false.
        if (iso_fortran_env_kind_value(id_name, kind_value)) has_var = .false.
    end function bound_identifier_references_variable

    module function declaration_bound_is_variable_driven(node, context) &
            result(is_var)
        type(declaration_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        logical :: is_var

        integer :: d

        is_var = .false.
        if (.not. allocated(node%dimension_indices)) return
        if (size(node%dimension_indices) < 1 .or. &
            size(node%dimension_indices) > 4) return
        do d = 1, size(node%dimension_indices)
            if (bound_expr_references_variable(context%arena, &
                    node%dimension_indices(d), context)) is_var = .true.
        end do
    end function declaration_bound_is_variable_driven

    module function declaration_is_runtime_local_array(node, context, value_kind) &
            result(is_local)
        ! A rank-1 through rank-4 explicit-shape local automatic array whose extent is a runtime
        ! expression and whose name is not already bound (not a dummy argument,
        ! function result, or COMMON member). Such a symbol owns dynamic storage
        ! allocated at its declaration; adjustable-array dummies keep their
        ! parameter-base binding and the existing compile-time fold path.
        type(declaration_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: value_kind
        logical :: is_local
        integer :: k

        is_local = .false.
        if (.not. runtime_local_array_kind_ok(value_kind)) return
        if (.not. declaration_is_runtime_rank1(node, context)) return
        if (.not. declaration_bound_is_variable_driven(node, context)) return
        if (node%is_multi_declaration .and. allocated(node%var_names)) then
            do k = 1, size(node%var_names)
                if (find_symbol_compat(context, node%var_names(k)) > 0) return
            end do
            is_local = .true.
        else if (allocated(node%var_name)) then
            is_local = find_symbol_compat(context, node%var_name) <= 0
        end if
    end function declaration_is_runtime_local_array

    module function declaration_rebinds_runtime_array_result(node, context) &
            result(is_rebind)
        ! True when this body array declaration names the pre-bound array function
        ! result symbol and has a runtime (non-foldable) rank-1 extent. Such a
        ! declaration rebinds the sret buffer view rather than allocating storage,
        ! so its bounds must not be folded at compile time.
        type(declaration_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        logical :: is_rebind
        integer :: index, k

        is_rebind = .false.
        if (.not. declaration_is_runtime_rank1(node, context)) return
        if (allocated(node%var_name)) then
            index = find_symbol_compat(context, node%var_name)
            if (index > 0) then
                if (context%symbols(index)%is_array_result) is_rebind = .true.
            end if
            if (is_rebind) return
        end if
        if (node%is_multi_declaration .and. allocated(node%var_names)) then
            do k = 1, size(node%var_names)
                index = find_symbol_compat(context, node%var_names(k))
                if (index > 0) then
                    if (context%symbols(index)%is_array_result) is_rebind = .true.
                end if
                if (is_rebind) return
            end do
        end if
    end function declaration_rebinds_runtime_array_result

    module function declaration_is_assumed_rank(node, context) result(is_rank)
        ! An assumed-rank dummy arr(..) parses to a single array_bounds_node
        ! flagged is_assumed_rank. Its runtime rank comes from the caller's
        ! actual, so it bypasses compile-time bound folding (#273).
        type(declaration_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        logical :: is_rank
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

    module function dim_is_assumed_size(arena, dim_index) result(is_assumed)
        ! A bare asterisk dimension a(*) parses to an array_bounds_node
        ! flagged is_assumed_size.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: dim_index
        logical :: is_assumed

        is_assumed = .false.
        if (.not. node_exists(arena, dim_index)) return
        select type (dim_node => arena%entries(dim_index)%node)
        type is (array_bounds_node)
            is_assumed = dim_node%is_assumed_size
        end select
    end function dim_is_assumed_size

    module function declaration_is_assumed_size(node, context) result(is_assumed)
        ! An assumed-size dummy a(*) or a(n1, ..., *): the last dimension is
        ! a bare asterisk and every leading dimension is an explicit extent
        ! (never a colon or another asterisk).
        type(declaration_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        logical :: is_assumed
        integer :: dim_count, i

        is_assumed = .false.
        if (.not. allocated(node%dimension_indices)) return
        dim_count = size(node%dimension_indices)
        if (dim_count < 1) return
        if (.not. dim_is_assumed_size(context%arena, &
                                      node%dimension_indices(dim_count))) return
        do i = 1, dim_count - 1
            if (dim_is_assumed_size(context%arena, node%dimension_indices(i))) &
                return
            if (dim_is_assumed_shape(context%arena, node%dimension_indices(i))) &
                return
        end do
        is_assumed = .true.
    end function declaration_is_assumed_size

end submodule session_program_lowering_array_shape
