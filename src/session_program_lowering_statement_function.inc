    ! Statement functions (f(x) = x*x + 1): collection and call-site inlining.
    !
    ! FortFront converts a spec-section assignment whose target is a scalar call
    ! into a statement_function_node carrying the name, dummy names, and the
    ! arena index of the defining expression. ffc records these at setup time and
    ! expands each call by binding the actual-argument operands to the dummy
    ! names and lowering the stored body expression in place (scalar only).

    subroutine collect_statement_functions(arena, root_index, context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: root_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: prog_name, ff_error
        integer, allocatable :: body_indices(:)
        integer :: i

        context%statement_function_count = 0
        if (.not. allocated(context%statement_functions)) &
            allocate(context%statement_functions(4))
        call set_empty(error_msg)

        call get_program_body_info(arena, root_index, prog_name, body_indices, &
                                   ff_error)
        if (len_trim(ff_error) > 0) return
        if (.not. allocated(body_indices)) return
        do i = 1, size(body_indices)
            call record_statement_function(arena, body_indices(i), context)
        end do
    end subroutine collect_statement_functions

    ! Record a statement function from either FortFront's dedicated node (the
    ! implicit-program path converts to it) or a spec-section assignment whose
    ! target is an undeclared scalar call f(x,...) with all-identifier arguments
    ! (the explicit-program path leaves this form unconverted).
    subroutine record_statement_function(arena, node_index, context)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        integer :: next, j

        if (.not. node_exists(arena, node_index)) return
        select type (node => arena%entries(node_index)%node)
        type is (statement_function_node)
            if (.not. allocated(node%name)) return
            if (.not. allocated(node%arg_names)) return
            if (size(node%arg_names) > MAX_STMT_FN_ARGS) return
            call grow_statement_functions(context)
            next = context%statement_function_count + 1
            context%statement_functions(next)%name = trim(node%name)
            context%statement_functions(next)%arg_count = size(node%arg_names)
            do j = 1, size(node%arg_names)
                context%statement_functions(next)%arg_names(j) = &
                    trim(node%arg_names(j))
            end do
            context%statement_functions(next)%body_expr_index = &
                node%body_expr_index
            context%statement_function_count = next
        type is (assignment_node)
            call record_assignment_statement_function(arena, node, context)
        end select
    end subroutine record_statement_function

    subroutine record_assignment_statement_function(arena, node, context)
        type(ast_arena_t), intent(in) :: arena
        type(assignment_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        integer :: next, n, j
        character(len=64) :: dummy_names(MAX_STMT_FN_ARGS)

        if (node%value_index <= 0) return
        if (.not. node_exists(arena, node%target_index)) return
        select type (target => arena%entries(node%target_index)%node)
        type is (call_or_subscript_node)
            if (target%base_expr_index /= 0) return
            if (.not. allocated(target%name)) return
            if (.not. allocated(target%arg_indices)) return
            n = size(target%arg_indices)
            if (n <= 0 .or. n > MAX_STMT_FN_ARGS) return
            ! A name already declared (variable or array) is an element
            ! assignment, not a statement-function definition.
            if (is_declared_in_arena(arena, target%name)) return
            if (.not. gather_dummy_names(arena, target%arg_indices, &
                                         dummy_names, n)) return
            call grow_statement_functions(context)
            next = context%statement_function_count + 1
            context%statement_functions(next)%name = trim(target%name)
            context%statement_functions(next)%arg_count = n
            do j = 1, n
                context%statement_functions(next)%arg_names(j) = dummy_names(j)
            end do
            context%statement_functions(next)%body_expr_index = node%value_index
            context%statement_function_count = next
        end select
    end subroutine record_assignment_statement_function

    ! True when name is declared by any declaration node reachable in the arena.
    ! Used to reject array/scalar element assignments from statement-function
    ! detection in the explicit-program path.
    logical function is_declared_in_arena(arena, name) result(declared)
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: name
        integer :: i

        declared = .false.
        do i = 1, arena%size
            if (.not. node_exists(arena, i)) cycle
            select type (node => arena%entries(i)%node)
            type is (declaration_node)
                if (allocated(node%var_name)) then
                    if (same_name(trim(node%var_name), trim(name))) then
                        declared = .true.
                        return
                    end if
                end if
                if (allocated(node%var_names)) then
                    if (any_var_name_matches(node%var_names, name)) then
                        declared = .true.
                        return
                    end if
                end if
            end select
        end do
    end function is_declared_in_arena

    logical function any_var_name_matches(var_names, name) result(matched)
        character(len=*), intent(in) :: var_names(:)
        character(len=*), intent(in) :: name
        integer :: i

        matched = .false.
        do i = 1, size(var_names)
            if (same_name(trim(var_names(i)), trim(name))) then
                matched = .true.
                return
            end if
        end do
    end function any_var_name_matches

    logical function gather_dummy_names(arena, arg_indices, names, n) result(ok)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: arg_indices(:)
        character(len=64), intent(out) :: names(:)
        integer, intent(in) :: n
        integer :: k
        character(len=:), allocatable :: id_name, id_err

        ok = .false.
        do k = 1, n
            if (.not. is_identifier(arena, arg_indices(k))) return
            call get_identifier_name(arena, arg_indices(k), id_name, id_err)
            if (len_trim(id_err) > 0) return
            names(k) = trim(id_name)
        end do
        ok = .true.
    end function gather_dummy_names

    subroutine grow_statement_functions(context)
        type(lowering_context_t), intent(inout) :: context
        type(statement_function_t), allocatable :: tmp(:)
        integer :: old_size

        if (.not. allocated(context%statement_functions)) then
            allocate(context%statement_functions(4))
            return
        end if
        if (context%statement_function_count < &
            size(context%statement_functions)) return
        old_size = size(context%statement_functions)
        call move_alloc(context%statement_functions, tmp)
        allocate(context%statement_functions(2 * old_size))
        context%statement_functions(1:old_size) = tmp
    end subroutine grow_statement_functions

    ! True when an assignment's target names a recorded statement function (the
    ! unconverted explicit-program definition form). Such an assignment defines
    ! the function and emits no code of its own.
    logical function is_statement_function_definition(arena, node, context) &
            result(is_def)
        type(ast_arena_t), intent(in) :: arena
        type(assignment_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context

        is_def = .false.
        if (context%statement_function_count == 0) return
        if (.not. node_exists(arena, node%target_index)) return
        select type (target => arena%entries(node%target_index)%node)
        type is (call_or_subscript_node)
            if (target%base_expr_index /= 0) return
            if (.not. allocated(target%name)) return
            if (.not. allocated(target%arg_indices)) return
            is_def = find_statement_function(context, target%name) > 0
        end select
    end function is_statement_function_definition

    integer function find_statement_function(context, name) result(index)
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: name
        integer :: i

        index = 0
        do i = 1, context%statement_function_count
            if (same_name(trim(context%statement_functions(i)%name), &
                          trim(name))) then
                index = i
                return
            end if
        end do
    end function find_statement_function

    ! True when node_index is a call to a recorded statement function.
    logical function is_statement_function_call(arena, node_index, context)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context

        is_statement_function_call = .false.
        if (context%statement_function_count == 0) return
        if (.not. node_exists(arena, node_index)) return
        select type (node => arena%entries(node_index)%node)
        type is (call_or_subscript_node)
            if (node%base_expr_index /= 0) return
            if (.not. allocated(node%name)) return
            is_statement_function_call = &
                find_statement_function(context, node%name) > 0
        end select
    end function is_statement_function_call

    ! Inline a statement-function call as a scalar expression of value_kind.
    ! Binds each actual-argument operand to its dummy name, lowers the stored
    ! body expression, then restores the shadowed symbols.
    recursive subroutine lower_statement_function_call(arena, node_index, &
            value_kind, context, value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        integer, intent(in) :: value_kind
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: sf, body_index, n, k
        type(symbol_t), allocatable :: saved(:)
        integer, allocatable :: saved_slots(:)
        logical, allocatable :: had_symbol(:)

        call set_empty(error_msg)
        select type (node => arena%entries(node_index)%node)
        type is (call_or_subscript_node)
            sf = find_statement_function(context, node%name)
            if (sf <= 0) then
                error_msg = 'statement function not found: '//trim(node%name)
                return
            end if
            if (.not. allocated(node%arg_indices)) then
                error_msg = 'statement function requires arguments: '// &
                            trim(node%name)
                return
            end if
            n = context%statement_functions(sf)%arg_count
            if (size(node%arg_indices) /= n) then
                error_msg = 'statement function argument count mismatch: '// &
                            trim(node%name)
                return
            end if
            body_index = context%statement_functions(sf)%body_expr_index
            call bind_statement_function_args(arena, node%arg_indices, sf, &
                context, saved, saved_slots, had_symbol, error_msg)
            if (len_trim(error_msg) > 0) return
            call lower_scalar_expression_kind(arena, body_index, value_kind, &
                                              context, value, error_msg)
            call restore_statement_function_args(context, sf, saved, &
                                                 saved_slots, had_symbol)
        class default
            error_msg = 'statement function call expected'
        end select
    end subroutine lower_statement_function_call

    ! Lower each actual to an operand whose kind matches the dummy's declared
    ! kind, then overwrite the dummy's symbol with that value. Existing symbols
    ! for the dummy names are preserved for restoration.
    recursive subroutine bind_statement_function_args(arena, arg_indices, sf, &
            context, saved, saved_slots, had_symbol, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: arg_indices(:)
        integer, intent(in) :: sf
        type(lowering_context_t), intent(inout) :: context
        type(symbol_t), allocatable, intent(out) :: saved(:)
        integer, allocatable, intent(out) :: saved_slots(:)
        logical, allocatable, intent(out) :: had_symbol(:)
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: k, n, slot
        integer :: dummy_kinds(MAX_STMT_FN_ARGS)
        character(len=:), allocatable :: dummy_name
        type(lr_operand_desc_t) :: actuals(MAX_STMT_FN_ARGS)

        call set_empty(error_msg)
        n = context%statement_functions(sf)%arg_count
        allocate(saved(n), saved_slots(n), had_symbol(n))
        ! Lower every actual in the caller's scope first: an actual may name a
        ! dummy (f(a, a+1)) whose binding must not shadow the caller's variable.
        do k = 1, n
            dummy_name = trim(context%statement_functions(sf)%arg_names(k))
            slot = find_symbol(context, dummy_name)
            dummy_kinds(k) = VALUE_I32
            if (slot > 0) dummy_kinds(k) = context%symbols(slot)%value_kind
            call lower_scalar_expression_kind(arena, arg_indices(k), &
                dummy_kinds(k), context, actuals(k), error_msg)
            if (len_trim(error_msg) > 0) return
        end do
        do k = 1, n
            dummy_name = trim(context%statement_functions(sf)%arg_names(k))
            slot = find_symbol(context, dummy_name)
            if (slot > 0) then
                had_symbol(k) = .true.
                saved_slots(k) = slot
                saved(k) = context%symbols(slot)
            else
                had_symbol(k) = .false.
                call grow_symbols(context)
                context%symbol_count = context%symbol_count + 1
                slot = context%symbol_count
                saved_slots(k) = slot
            end if
            call set_bound_arg_symbol(context%symbols(slot), dummy_name, &
                                      dummy_kinds(k), actuals(k))
        end do
    end subroutine bind_statement_function_args

    subroutine set_bound_arg_symbol(sym, name, value_kind, value)
        type(symbol_t), intent(inout) :: sym
        character(len=*), intent(in) :: name
        integer, intent(in) :: value_kind
        type(lr_operand_desc_t), intent(in) :: value

        sym = symbol_t()
        sym%name = name
        sym%value_kind = value_kind
        sym%value = value
        ! A by-value binding: the identifier resolves to the operand directly,
        ! never to storage, so reference/address paths stay disabled.
        sym%is_parameter = .true.
        sym%has_address = .false.
        sym%is_reference = .false.
    end subroutine set_bound_arg_symbol

    subroutine restore_statement_function_args(context, sf, saved, saved_slots, &
                                               had_symbol)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: sf
        type(symbol_t), intent(in) :: saved(:)
        integer, intent(in) :: saved_slots(:)
        logical, intent(in) :: had_symbol(:)
        integer :: k, n

        n = context%statement_functions(sf)%arg_count
        do k = n, 1, -1
            if (had_symbol(k)) then
                context%symbols(saved_slots(k)) = saved(k)
            else
                context%symbols(saved_slots(k)) = symbol_t()
                context%symbol_count = context%symbol_count - 1
            end if
        end do
    end subroutine restore_statement_function_args

    ! Lower a scalar expression to an operand of the requested value kind. Used
    ! by statement-function inlining for both actuals and the body.
    recursive subroutine lower_scalar_expression_kind(arena, node_index, &
            value_kind, context, value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        integer, intent(in) :: value_kind
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg

        select case (value_kind)
        case (VALUE_F32)
            call lower_f32_expression(arena, node_index, context, value, error_msg)
        case (VALUE_F64)
            call lower_f64_expression(arena, node_index, context, value, error_msg)
        case (VALUE_I64)
            call lower_i64_expression(arena, node_index, context, value, error_msg)
        case (VALUE_LOGICAL)
            call lower_logical_expression(arena, node_index, context, value, &
                                          error_msg)
        case default
            call lower_i32_expression(arena, node_index, context, value, error_msg)
        end select
    end subroutine lower_scalar_expression_kind
