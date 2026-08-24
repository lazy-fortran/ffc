    subroutine lower_interface_block(arena, node, context, error_msg)
        ! Register bind(c) external integer functions declared in an interface
        ! block (#192). Generic named interfaces (interface foo ... module
        ! procedure foo_int, foo_real) are registered in the generic table so
        ! calls can be resolved to the matching specific (#249 B7c).
        ! Operator/assignment interfaces record their specifics in the operator
        ! table so a matching binary op or assignment dispatches to them.
        type(ast_arena_t), intent(in) :: arena
        type(interface_block_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i
        character(len=:), allocatable :: lowered_kind

        call set_empty(error_msg)
        ! An abstract interface declares a signature template for procedure
        ! pointers and dummy procedures; it binds no symbol and emits no code.
        ! Dispatch through such a pointer is resolved at the use site, so the
        ! declaration alone needs no registration here.
        if (node%is_abstract) return
        if (allocated(node%kind)) then
            lowered_kind = lowercase_text(trim(node%kind))
            if (lowered_kind == 'operator' .or. &
                lowered_kind == 'assignment') then
                call register_operator_interface(arena, node, context, &
                    lowered_kind == 'assignment', error_msg)
                return
            end if
            if (lowered_kind /= 'interface' .and. &
                lowered_kind /= 'generic') then
                call unsupported_feature_error('interface block', node%line, &
                    node%column, 'only plain bind(c) and generic interfaces '// &
                    'are supported', error_msg)
                return
            end if
        end if
        if (.not. allocated(node%procedure_indices)) return

        ! A generic interface has a name and module procedure lists inside.
        if (allocated(node%name) .and. len_trim(node%name) > 0) then
            call register_generic_interface(arena, node, context, error_msg)
            return
        end if

        do i = 1, size(node%procedure_indices)
            call register_interface_procedure(arena, node%procedure_indices(i), &
                                              node, context, error_msg)
            if (len_trim(error_msg) > 0) return
        end do
    end subroutine lower_interface_block

    subroutine register_generic_interface(arena, node, context, error_msg)
        ! Collect the specific procedure names listed in a named generic
        ! interface block and record them in context%generics so call sites
        ! can resolve the generic name to the right specific (#249 B7c). A
        ! specific may be named either by a `module procedure P` statement or,
        ! in the submodule idiom, by a module-procedure interface body
        ! (`module subroutine P(...)`); both forms register P as a specific.
        type(ast_arena_t), intent(in) :: arena
        type(interface_block_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i, j, next
        type(module_procedure_node), pointer :: mp_node
        character(len=:), allocatable :: node_type, body_name

        call set_empty(error_msg)
        next = find_generic(context, node%name)
        if (next == 0) then
            call grow_generics(context)
            next = context%generic_count + 1
            context%generic_count = next
        end if
        context%generics(next)%generic_name = trim(node%name)
        context%generics(next)%specific_count = 0

        do i = 1, size(node%procedure_indices)
            if (.not. node_exists(arena, node%procedure_indices(i))) cycle
            node_type = get_node_type_at(arena, node%procedure_indices(i))
            if (node_type == 'module_procedure_node' .or. &
                node_type == 'module_procedure') then
                select type (n => arena%entries(node%procedure_indices(i))%node)
                type is (module_procedure_node)
                    mp_node => n
                class default
                    cycle
                end select
                if (.not. allocated(mp_node%procedure_names)) cycle
                do j = 1, size(mp_node%procedure_names)
                    call add_generic_specific(context, next, &
                        trim(mp_node%procedure_names(j)%s), error_msg)
                    if (len_trim(error_msg) > 0) return
                end do
            else if (node_type == 'function_def_node' .or. &
                     node_type == 'function_def' .or. &
                     node_type == 'subroutine_def_node' .or. &
                     node_type == 'subroutine_def') then
                call interface_body_procedure_name(arena, &
                    node%procedure_indices(i), body_name)
                if (len_trim(body_name) == 0) cycle
                call add_generic_specific(context, next, body_name, error_msg)
                if (len_trim(error_msg) > 0) return
            end if
        end do
        call check_generic_ambiguity(arena, context, next, error_msg)
        if (len_trim(error_msg) > 0) return
    end subroutine register_generic_interface

    subroutine add_generic_specific(context, generic_idx, specific_name, error_msg)
        ! Append one specific procedure name to a generic interface entry and
        ! record its return kind, per-argument kinds, and per-argument ranks
        ! from the function table. Ranks enable dispatch to distinguish same-kind
        ! specifics that differ in rank (refs #303).
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: generic_idx
        character(len=*), intent(in) :: specific_name
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: arg_count, j
        integer :: arg_kinds(MAX_PROC_ARGS)
        integer :: arg_ranks(MAX_PROC_ARGS)
        integer :: fn_index

        call set_empty(error_msg)
        if (context%generics(generic_idx)%specific_count >= &
            MAX_GENERIC_SPECIFICS) then
            error_msg = 'generic interface '// &
                trim(context%generics(generic_idx)%generic_name)// &
                ' has too many specific procedures'
            return
        end if
        context%generics(generic_idx)%specific_count = &
            context%generics(generic_idx)%specific_count + 1
        associate (sc => context%generics(generic_idx)%specific_count)
            arg_count = 0
            arg_kinds = VALUE_I32
            arg_ranks = 0
            context%generics(generic_idx)%specific_names(sc) = trim(specific_name)
            call query_specific_kinds(context, trim(specific_name), &
                context%generics(generic_idx)%return_kinds(sc), &
                arg_count, arg_kinds)
            context%generics(generic_idx)%specific_arg_counts(sc) = arg_count
            context%generics(generic_idx)%specific_arg_kinds(:, sc) = arg_kinds
            fn_index = find_function_index(context, trim(specific_name))
            if (fn_index > 0) then
                do j = 1, arg_count
                    arg_ranks(j) = specific_arg_rank(context, fn_index, j)
                end do
            end if
            context%generics(generic_idx)%specific_arg_ranks(:, sc) = arg_ranks
        end associate
    end subroutine add_generic_specific

    subroutine interface_body_procedure_name(arena, node_index, name)
        ! The declared name of a module-procedure interface body (a
        ! function_def/subroutine_def stored inside a generic interface block).
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        character(len=:), allocatable, intent(out) :: name
        character(len=:), allocatable :: node_type
        type(function_def_node), pointer :: fn_node
        type(subroutine_def_node), pointer :: sb_node

        name = ''
        if (.not. node_exists(arena, node_index)) return
        node_type = get_node_type_at(arena, node_index)
        if (node_type == 'function_def_node' .or. node_type == 'function_def') then
            fn_node => get_node_as_function_def(arena, node_index)
            if (.not. associated(fn_node)) return
            if (allocated(fn_node%name)) name = trim(fn_node%name)
        else if (node_type == 'subroutine_def_node' .or. &
                 node_type == 'subroutine_def') then
            sb_node => get_node_as_subroutine_def(arena, node_index)
            if (.not. associated(sb_node)) return
            if (allocated(sb_node%name)) name = trim(sb_node%name)
        end if
    end subroutine interface_body_procedure_name

    subroutine register_operator_interface(arena, node, context, is_assignment, &
                                           error_msg)
        ! Record an interface operator(op)/assignment(=) block in the operator
        ! table. The token in node%operator (e.g. "+", ".dot.", "=") keys the
        ! overload; the listed module procedures become the specifics. Operand
        ! and return kinds are scanned from each specific so distinct overloads
        ! of one token dispatch by operand type.
        type(ast_arena_t), intent(in) :: arena
        type(interface_block_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        logical, intent(in) :: is_assignment
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i, j, next, fn_index
        type(module_procedure_node), pointer :: mp_node
        character(len=:), allocatable :: node_type, op_token

        call set_empty(error_msg)
        if (.not. allocated(node%procedure_indices)) return
        op_token = ''
        if (allocated(node%operator)) op_token = lowercase_text(trim(node%operator))
        if (is_assignment) op_token = '='
        if (len_trim(op_token) == 0) return

        next = find_operator_slot(context, op_token, is_assignment)
        do i = 1, size(node%procedure_indices)
            if (.not. node_exists(arena, node%procedure_indices(i))) cycle
            node_type = get_node_type_at(arena, node%procedure_indices(i))
            if (node_type /= 'module_procedure_node' .and. &
                node_type /= 'module_procedure') cycle
            select type (n => arena%entries(node%procedure_indices(i))%node)
            type is (module_procedure_node)
                mp_node => n
            class default
                cycle
            end select
            if (.not. allocated(mp_node%procedure_names)) cycle
            do j = 1, size(mp_node%procedure_names)
                if (context%operators(next)%specific_count >= &
                    MAX_GENERIC_SPECIFICS) then
                    error_msg = 'operator '//trim(op_token)// &
                                ' has too many overloads'
                    return
                end if
                associate (op => context%operators(next))
                    op%specific_count = op%specific_count + 1
                    op%specific_names(op%specific_count) = &
                        trim(mp_node%procedure_names(j)%s)
                    fn_index = find_function_index(context, &
                        trim(mp_node%procedure_names(j)%s))
                    if (fn_index > 0) then
                        op%return_kinds(op%specific_count) = &
                            context%function_value_kinds(fn_index)
                        op%first_arg_kinds(op%specific_count) = &
                            specific_arg_kind(context, fn_index, 1)
                        op%second_arg_kinds(op%specific_count) = &
                            specific_arg_kind(context, fn_index, 2)
                    end if
                end associate
            end do
        end do
    end subroutine register_operator_interface

    integer function find_operator_slot(context, op_token, is_assignment) &
        result(idx)
        ! Index of the operator-table entry for op_token, creating one if absent
        ! so specifics from interfaces in different scopes accumulate.
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: op_token
        logical, intent(in) :: is_assignment
        integer :: i

        do i = 1, context%operator_count
            if (context%operators(i)%is_assignment .eqv. is_assignment) then
                if (same_name(context%operators(i)%operator_name, op_token)) then
                    idx = i
                    return
                end if
            end if
        end do
        call grow_operators(context)
        idx = context%operator_count + 1
        context%operators(idx)%operator_name = trim(op_token)
        context%operators(idx)%is_assignment = is_assignment
        context%operators(idx)%specific_count = 0
        context%operator_count = idx
    end function find_operator_slot

    subroutine grow_operators(context)
        type(lowering_context_t), intent(inout) :: context
        type(operator_interface_t), allocatable :: old(:)

        if (.not. allocated(context%operators)) then
            allocate (context%operators(4))
            return
        end if
        if (context%operator_count < size(context%operators)) return
        call move_alloc(context%operators, old)
        allocate (context%operators(2 * size(old)))
        context%operators(1:size(old)) = old
    end subroutine grow_operators

    integer function find_function_index(context, name) result(idx)
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: name
        integer :: i

        idx = 0
        do i = 1, context%function_count
            if (same_name(context%function_names(i), name)) then
                idx = i
                return
            end if
        end do
    end function find_function_index

    integer function specific_arg_kind(context, fn_index, arg_pos) result(kind)
        ! Value kind of the arg_pos-th dummy of a registered specific. Derived
        ! dummies report VALUE_DERIVED so overload dispatch can tell a derived
        ! operand from an intrinsic one.
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: fn_index, arg_pos
        integer :: node_idx, i
        character(len=:), allocatable :: param_name, err, node_type
        type(function_def_node), pointer :: fn_node
        type(subroutine_def_node), pointer :: sb_node
        integer, allocatable :: body_indices(:), param_indices(:)

        kind = VALUE_I32
        node_idx = context%function_node_indices(fn_index)
        if (node_idx <= 0) return
        if (.not. node_exists(context%arena, node_idx)) return
        node_type = get_node_type_at(context%arena, node_idx)
        if (node_type == 'function_def_node' .or. node_type == 'function_def') then
            fn_node => get_node_as_function_def(context%arena, node_idx)
            if (.not. associated(fn_node)) return
            if (.not. allocated(fn_node%param_indices)) return
            param_indices = fn_node%param_indices
            body_indices = fn_node%body_indices
        else if (node_type == 'subroutine_def_node' .or. &
                 node_type == 'subroutine_def') then
            sb_node => get_node_as_subroutine_def(context%arena, node_idx)
            if (.not. associated(sb_node)) return
            if (.not. allocated(sb_node%param_indices)) return
            param_indices = sb_node%param_indices
            body_indices = sb_node%body_indices
        else
            return
        end if
        if (arg_pos > size(param_indices)) return
        call parameter_name(context%arena, param_indices(arg_pos), param_name, err)
        if (len_trim(err) > 0) return
        if (.not. allocated(body_indices)) return
        do i = 1, size(body_indices)
            if (.not. node_exists(context%arena, body_indices(i))) cycle
            select type (decl => context%arena%entries(body_indices(i))%node)
            type is (declaration_node)
                if (.not. allocated(decl%type_name)) cycle
                if (.not. declaration_names_var(decl, param_name)) cycle
                if (declaration_derived_type_index(context, decl) > 0) then
                    kind = VALUE_DERIVED
                    return
                end if
                call type_name_value_kind(decl%type_name, decl%line, &
                                          decl%column, kind, err, context)
                if (len_trim(err) > 0) kind = VALUE_DERIVED
                return
            end select
        end do
    end function specific_arg_kind

    integer function specific_arg_rank(context, fn_index, arg_pos) result(rank)
        ! Array rank of the arg_pos-th dummy of a registered specific. Returns 0
        ! for a scalar dummy and the rank (1, 2, ...) for an array dummy. Used by
        ! generic dispatch to distinguish same-kind specifics that differ in rank
        ! (refs #303).
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: fn_index, arg_pos
        integer :: node_idx, i
        character(len=:), allocatable :: param_name, err, node_type
        type(function_def_node), pointer :: fn_node
        type(subroutine_def_node), pointer :: sb_node
        integer, allocatable :: body_indices(:), param_indices(:)

        rank = 0
        node_idx = context%function_node_indices(fn_index)
        if (node_idx <= 0) return
        if (.not. node_exists(context%arena, node_idx)) return
        node_type = get_node_type_at(context%arena, node_idx)
        if (node_type == 'function_def_node' .or. node_type == 'function_def') then
            fn_node => get_node_as_function_def(context%arena, node_idx)
            if (.not. associated(fn_node)) return
            if (.not. allocated(fn_node%param_indices)) return
            param_indices = fn_node%param_indices
            body_indices = fn_node%body_indices
        else if (node_type == 'subroutine_def_node' .or. &
                 node_type == 'subroutine_def') then
            sb_node => get_node_as_subroutine_def(context%arena, node_idx)
            if (.not. associated(sb_node)) return
            if (.not. allocated(sb_node%param_indices)) return
            param_indices = sb_node%param_indices
            body_indices = sb_node%body_indices
        else
            return
        end if
        if (arg_pos > size(param_indices)) return
        call parameter_name(context%arena, param_indices(arg_pos), param_name, err)
        if (len_trim(err) > 0) return
        if (.not. allocated(body_indices)) return
        do i = 1, size(body_indices)
            if (.not. node_exists(context%arena, body_indices(i))) cycle
            select type (decl => context%arena%entries(body_indices(i))%node)
            type is (declaration_node)
                if (.not. declaration_names_var(decl, param_name)) cycle
                if (decl%is_array .and. allocated(decl%dimension_indices)) then
                    rank = size(decl%dimension_indices)
                end if
                return
            end select
        end do
    end function specific_arg_rank

    integer function find_operator(context, op_token, left_kind, right_kind, &
                                   is_assignment) result(specific_slot)
        ! Find the operator-overload specific (encoded as op_index*256+spec_index)
        ! matching op_token and the operand kinds, or 0 when none applies. Exact
        ! operand-kind matches always win. A sole-specific fallback applies only
        ! to named (.op.) operators and assignment, never to the arithmetic
        ! symbols +,-,*,/: those keep their builtin meaning unless the operand
        ! kinds match a declared overload, so normal integer/real arithmetic in
        ! a unit that also overloads a symbol is not hijacked.
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: op_token
        integer, intent(in) :: left_kind, right_kind
        logical, intent(in) :: is_assignment
        integer :: i, j
        character(len=:), allocatable :: token
        integer :: fallback
        logical :: allow_fallback

        specific_slot = 0
        fallback = 0
        token = lowercase_text(trim(op_token))
        allow_fallback = is_assignment .or. is_named_operator_token(token)
        do i = 1, context%operator_count
            if (context%operators(i)%is_assignment .neqv. is_assignment) cycle
            if (.not. same_name(context%operators(i)%operator_name, token)) cycle
            do j = 1, context%operators(i)%specific_count
                if (context%operators(i)%first_arg_kinds(j) == left_kind .and. &
                    context%operators(i)%second_arg_kinds(j) == right_kind) then
                    specific_slot = i * 256 + j
                    return
                end if
                if (fallback == 0) fallback = i * 256 + j
            end do
        end do
        if (allow_fallback) specific_slot = fallback
    end function find_operator

    integer function find_unary_operator(context, op_token, operand_kind) &
                                           result(specific_slot)
        ! Defined unary operators are represented by a binary_op_node whose
        ! left index is zero. Match the one-argument specific against the
        ! actual operand kind; the unused second-kind slot is not meaningful
        ! for a unary procedure.
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: op_token
        integer, intent(in) :: operand_kind
        integer :: i, j, fn_index, fallback
        character(len=:), allocatable :: token

        specific_slot = 0
        fallback = 0
        token = lowercase_text(trim(op_token))
        do i = 1, context%operator_count
            if (context%operators(i)%is_assignment) cycle
            if (.not. same_name(context%operators(i)%operator_name, token)) cycle
            do j = 1, context%operators(i)%specific_count
                fn_index = find_function_index(context, &
                    trim(context%operators(i)%specific_names(j)))
                if (fn_index <= 0) cycle
                if (context%function_param_counts(fn_index) /= 1) cycle
                if (context%operators(i)%first_arg_kinds(j) == operand_kind) then
                    specific_slot = i * 256 + j
                    return
                end if
                if (fallback == 0) fallback = i * 256 + j
            end do
        end do
        if (is_named_operator_token(token)) specific_slot = fallback
    end function find_unary_operator

    logical function is_named_operator_token(token)
        ! A named operator like .dot. lexes as .word. and cannot collide with a
        ! builtin arithmetic symbol, so a sole overload may dispatch even when
        ! operand-kind inference is incomplete.
        character(len=*), intent(in) :: token

        is_named_operator_token = .false.
        if (len_trim(token) < 3) return
        is_named_operator_token = token(1:1) == '.' .and. &
            token(len_trim(token):len_trim(token)) == '.'
    end function is_named_operator_token

    function operator_specific_name(context, slot) result(name)
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: slot
        character(len=:), allocatable :: name
        integer :: op_i, sp_j

        op_i = slot / 256
        sp_j = mod(slot, 256)
        name = trim(context%operators(op_i)%specific_names(sp_j))
    end function operator_specific_name

    integer function operator_return_kind(context, slot) result(kind)
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: slot
        integer :: op_i, sp_j

        op_i = slot / 256
        sp_j = mod(slot, 256)
        kind = context%operators(op_i)%return_kinds(sp_j)
    end function operator_return_kind

    subroutine query_specific_kinds(context, specific_name, &
                                    return_kind, arg_count, arg_kinds)
        ! Look up the return kind and per-argument kinds of a specific
        ! procedure from the function table. Falls back to VALUE_I32 when
        ! the specific is not yet registered (registration order does not
        ! guarantee functions come before the interface block).
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: specific_name
        integer, intent(out) :: return_kind, arg_count
        integer, intent(out) :: arg_kinds(MAX_PROC_ARGS)
        integer :: i, j

        return_kind = VALUE_I32
        arg_count = 0
        arg_kinds = VALUE_I32
        do i = 1, context%function_count
            if (same_name(context%function_names(i), specific_name)) then
                return_kind = context%function_value_kinds(i)
                arg_count = context%function_param_counts(i)
                if (arg_count > MAX_PROC_ARGS) arg_count = MAX_PROC_ARGS
                do j = 1, arg_count
                    arg_kinds(j) = specific_arg_kind(context, i, j)
                end do
                return
            end if
        end do
    end subroutine query_specific_kinds

    subroutine grow_generics(context)
        type(lowering_context_t), intent(inout) :: context
        type(generic_interface_t), allocatable :: old(:)

        if (.not. allocated(context%generics)) then
            allocate (context%generics(4))
            return
        end if
        if (context%generic_count < size(context%generics)) return
        call move_alloc(context%generics, old)
        allocate (context%generics(2 * size(old)))
        context%generics(1:size(old)) = old
    end subroutine grow_generics

    integer function find_generic(context, name) result(idx)
        ! Return the index of a registered generic interface, or 0 if not found.
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: name
        integer :: i

        idx = 0
        do i = 1, context%generic_count
            if (same_name(context%generics(i)%generic_name, name)) then
                idx = i
                return
            end if
        end do
    end function find_generic

  logical function generic_signature_matches(stored_count, stored_kinds, &
                                             stored_ranks, call_count, &
                                             call_kinds, call_ranks) result(ok)
        integer, intent(in) :: stored_count
        integer, intent(in) :: stored_kinds(MAX_PROC_ARGS)
        integer, intent(in) :: stored_ranks(MAX_PROC_ARGS)
        integer, intent(in) :: call_count
        integer, intent(in) :: call_kinds(MAX_PROC_ARGS)
        integer, intent(in) :: call_ranks(MAX_PROC_ARGS)
        integer :: i

        ok = .false.
        if (stored_count /= call_count) return
        if (stored_count == 0) then
            ok = .true.
            return
        end if
        do i = 1, call_count
            if (call_kinds(i) /= stored_kinds(i)) return
            if (call_ranks(i) /= stored_ranks(i)) return
        end do
        ok = .true.
    end function generic_signature_matches

  function resolve_generic(context, generic_idx, call_arg_count, call_arg_kinds, &
                            call_arg_ranks) &
        result(specific_name)
        ! Choose the specific procedure that matches the full argument signature
        ! (kind + rank). Returns the first specific whose signature equals the
        ! actual, or the first specific when no exact match exists.
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: generic_idx, call_arg_count
        integer, intent(in) :: call_arg_kinds(MAX_PROC_ARGS)
        integer, intent(in) :: call_arg_ranks(MAX_PROC_ARGS)
        character(len=:), allocatable :: specific_name
        integer :: i
        type(generic_interface_t) :: g

        g = context%generics(generic_idx)
        if (g%specific_count == 0) then
            specific_name = ''
            return
        end if
        ! Exact match first.
        do i = 1, g%specific_count
            if (generic_signature_matches(g%specific_arg_counts(i), &
                g%specific_arg_kinds(:, i), g%specific_arg_ranks(:, i), &
                call_arg_count, call_arg_kinds, call_arg_ranks)) then
                specific_name = trim(g%specific_names(i))
                return
            end if
        end do
        ! Fallback: first specific.
        specific_name = trim(g%specific_names(1))
    end function resolve_generic

    integer function generic_return_kind(context, generic_idx, &
                                          call_arg_count, call_arg_kinds, &
                                          call_arg_ranks) result(ret_kind)
        ! Return kind of the specific selected for the full argument signature
        ! (kind + rank).
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: generic_idx, call_arg_count
        integer, intent(in) :: call_arg_kinds(MAX_PROC_ARGS)
        integer, intent(in) :: call_arg_ranks(MAX_PROC_ARGS)
        integer :: i
        type(generic_interface_t) :: g

        g = context%generics(generic_idx)
        ret_kind = VALUE_I32
        if (g%specific_count == 0) return
        do i = 1, g%specific_count
            if (generic_signature_matches(g%specific_arg_counts(i), &
                g%specific_arg_kinds(:, i), g%specific_arg_ranks(:, i), &
                call_arg_count, call_arg_kinds, call_arg_ranks)) then
                ret_kind = g%return_kinds(i)
                return
            end if
        end do
        ret_kind = g%return_kinds(1)
    end function generic_return_kind

    subroutine register_interface_procedure(arena, proc_index, block, context, &
                                            error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: proc_index
        type(interface_block_node), intent(in) :: block
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: ret_kind, n
        character(len=:), allocatable :: c_name
        integer, allocatable :: arg_kinds(:), arg_intents(:), arg_ranks(:), &
            arg_extents(:)
        character(len=64), allocatable :: arg_names(:)
        logical, allocatable :: arg_optionals(:), arg_values(:)

        if (.not. node_exists(arena, proc_index)) then
            error_msg = 'interface procedure index does not reference a node'
            return
        end if
        select type (proc => arena%entries(proc_index)%node)
        type is (function_def_node)
            if (.not. has_bind_c(proc%bind_c_clause)) then
                ! A plain explicit interface gives an external or dummy
                ! procedure its signature so call sites type-check. The
                ! interface node carries the parameter declarations and return
                ! type, so registering it lets the ordinary contained-call path
                ! lower arguments by reference; the symbol resolves at link.
                call register_internal_function_by_index(arena, proc_index, &
                                                         context, error_msg)
                return
            end if
            ret_kind = VALUE_I32
            if (allocated(proc%return_type)) then
                call type_name_value_kind(proc%return_type, proc%line, &
                                          proc%column, ret_kind, error_msg, &
                                          context)
                if (len_trim(error_msg) > 0) return
            end if
            call bind_c_name(proc%bind_c_clause, proc%name, c_name)
            n = 0
            if (allocated(proc%param_indices)) n = size(proc%param_indices)
            if (n > MAX_PROC_ARGS) then
                error_msg = 'bind(c) interface has too many arguments'
                return
            end if
            call interface_argument_metadata(arena, context, proc%param_indices, &
                                              proc%body_indices, arg_kinds, &
                                              arg_names, arg_optionals, arg_values, &
                                              arg_intents, arg_ranks, arg_extents, &
                                              error_msg)
            if (len_trim(error_msg) > 0) return
            call add_external_procedure(context, proc%name, c_name, ret_kind, n, &
                                        .false., arg_kinds, arg_names, &
                                        arg_optionals, arg_values, arg_intents, &
                                        arg_ranks, arg_extents, is_bind_c=.true.)
        type is (subroutine_def_node)
            if (.not. has_bind_c(proc%bind_c_clause)) then
                ! Plain explicit interface for an external/dummy subroutine:
                ! record the signature so the CALL type-checks and lowers by
                ! reference through the ordinary path; the symbol links later.
                call register_internal_subroutine_by_index(arena, proc_index, &
                                                          context, error_msg)
                return
            end if
            call bind_c_name(proc%bind_c_clause, proc%name, c_name)
            n = 0
            if (allocated(proc%param_indices)) n = size(proc%param_indices)
            if (n > MAX_PROC_ARGS) then
                error_msg = 'bind(c) interface has too many arguments'
                return
            end if
            call interface_argument_metadata(arena, context, proc%param_indices, &
                                              proc%body_indices, arg_kinds, &
                                              arg_names, arg_optionals, arg_values, &
                                              arg_intents, arg_ranks, arg_extents, &
                                              error_msg)
            if (len_trim(error_msg) > 0) return
            call add_external_procedure(context, proc%name, c_name, &
                                        VALUE_SUBROUTINE, n, .false., arg_kinds, &
                                        arg_names, arg_optionals, arg_values, &
                                        arg_intents, arg_ranks, arg_extents, &
                                        is_bind_c=.true.)
        class default
            call unsupported_feature_error('interface declaration', block%line, &
                block%column, 'only bind(c) interface functions are supported', &
                error_msg)
        end select
        call set_empty(error_msg)
    end subroutine register_interface_procedure

    subroutine interface_argument_metadata(arena, context, param_indices, &
                                           body_indices, arg_kinds, arg_names, &
                                           arg_optionals, arg_values, arg_intents, &
                                           arg_ranks, arg_extents, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(lowering_context_t), intent(in) :: context
        integer, allocatable, intent(in) :: param_indices(:), body_indices(:)
        integer, allocatable, intent(out) :: arg_kinds(:), arg_intents(:), &
            arg_ranks(:), arg_extents(:)
        character(len=64), allocatable, intent(out) :: arg_names(:)
        logical, allocatable, intent(out) :: arg_optionals(:), arg_values(:)
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: name, name_err, intent_token
        integer :: i, value_kind, rank, extent
        logical :: is_optional, is_value

        call set_empty(error_msg)
        if (.not. allocated(param_indices)) then
            allocate (arg_kinds(0), arg_names(0), arg_optionals(0), &
                      arg_values(0), arg_intents(0), arg_ranks(0), arg_extents(0))
            return
        end if
        allocate (arg_kinds(size(param_indices)), arg_names(size(param_indices)), &
                  arg_optionals(size(param_indices)), arg_values(size(param_indices)), &
                  arg_intents(size(param_indices)), arg_ranks(size(param_indices)), &
                  arg_extents(size(param_indices)))
        do i = 1, size(param_indices)
            value_kind = param_at_value_kind(arena, param_indices, body_indices, i, &
                                             context)
            rank = 0
            extent = 0
            if (value_kind == 0) then
                call param_at_array_shape(arena, context, param_indices, &
                                          body_indices, i, value_kind, rank, extent)
                if (value_kind == 0) then
                    call bind_c_interface_array_shape(arena, context, param_indices, &
                                                      body_indices, i, value_kind, &
                                                      rank, extent)
                end if
            end if
            if (value_kind == 0 .and. param_at_is_character(arena, param_indices, &
                                                             body_indices, i)) then
                value_kind = VALUE_CHARACTER
            end if
            if (value_kind == 0) then
                error_msg = 'unsupported BIND(C) interface argument'
                return
            end if
            call parameter_name(arena, param_indices(i), name, name_err)
            if (len_trim(name_err) > 0) then
                error_msg = name_err
                return
            end if
            arg_kinds(i) = value_kind
            arg_names(i) = trim(name)
            call param_at_attributes(arena, param_indices, body_indices, i, &
                                     intent_token, is_optional, is_value)
            arg_optionals(i) = is_optional
            arg_values(i) = is_value
            arg_intents(i) = arg_intent_code(intent_token)
            arg_ranks(i) = rank
            arg_extents(i) = extent
        end do
    end subroutine interface_argument_metadata

    subroutine bind_c_interface_array_shape(arena, context, param_indices, &
                                            body_indices, pos, value_kind, rank, &
                                            extent)
        ! BIND(C) explicit-shape and assumed-size arrays need only their element
        ! kind and rank at the call site. Their extent may be runtime-sized (b(n))
        ! or intentionally omitted (b(*)); unlike a module fmod signature, the C
        ! ABI does not require a compile-time element count.
        type(ast_arena_t), intent(in) :: arena
        type(lowering_context_t), intent(in) :: context
        integer, allocatable, intent(in) :: param_indices(:), body_indices(:)
        integer, intent(in) :: pos
        integer, intent(out) :: value_kind, rank, extent
        character(len=:), allocatable :: name, name_err, kind_err
        integer :: i

        value_kind = 0
        rank = 0
        extent = 0
        if (.not. allocated(param_indices)) return
        if (pos < 1 .or. pos > size(param_indices)) return
        call parameter_name(arena, param_indices(pos), name, name_err)
        if (len_trim(name_err) > 0) return
        if (.not. allocated(body_indices)) return
        do i = 1, size(body_indices)
            if (.not. node_exists(arena, body_indices(i))) cycle
            select type (decl => arena%entries(body_indices(i))%node)
            type is (declaration_node)
                if (.not. decl%is_array) cycle
                if (decl%is_allocatable .or. decl%is_pointer) return
                if (.not. declaration_declares_name(decl, &
                    trim(lowercase_text(name)))) cycle
                if (.not. allocated(decl%dimension_indices)) return
                call declaration_value_kind(decl, value_kind, kind_err, context)
                if (len_trim(kind_err) > 0) then
                    value_kind = 0
                    return
                end if
                rank = size(decl%dimension_indices)
                return
            end select
        end do
    end subroutine bind_c_interface_array_shape

    subroutine add_external_procedure(context, fortran_name, c_name, &
                                      return_kind, arg_count, by_reference, &
                                      arg_kinds, arg_names, arg_optionals, &
                                      arg_values, arg_intents, arg_ranks, &
                                      arg_extents, is_bind_c, arg_opaque, &
                                      arg_class, arg_class_types, &
                                      arg_class_type_identities, &
                                      require_class_type_identities)
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: fortran_name, c_name
        integer, intent(in) :: return_kind, arg_count
        logical, intent(in), optional :: by_reference
        integer, intent(in), optional :: arg_kinds(:)
        character(len=*), intent(in), optional :: arg_names(:)
        logical, intent(in), optional :: arg_optionals(:)
        logical, intent(in), optional :: arg_values(:)
        integer, intent(in), optional :: arg_intents(:)
        integer, intent(in), optional :: arg_ranks(:)
        integer, intent(in), optional :: arg_extents(:)
        logical, intent(in), optional :: is_bind_c
        logical, intent(in), optional :: arg_opaque(:)
        logical, intent(in), optional :: arg_class(:)
        character(len=*), intent(in), optional :: arg_class_types(:)
        character(len=*), intent(in), optional :: arg_class_type_identities(:)
        logical, intent(in), optional :: require_class_type_identities
        integer :: j, identity_length

        if (external_procedure_index(context, trim(fortran_name)) > 0) return
        call grow_external_procedures(context)
        context%external_procedure_count = context%external_procedure_count + 1
        associate (ep => context%external_procedures( &
                   context%external_procedure_count))
            ep%fortran_name = trim(fortran_name)
            ep%c_name = trim(c_name)
            ep%return_value_kind = return_kind
            ep%arg_count = arg_count
            ep%by_reference = .false.
            if (present(by_reference)) ep%by_reference = by_reference
            ep%is_bind_c = .false.
            if (present(is_bind_c)) ep%is_bind_c = is_bind_c
            ep%require_class_type_identities = .false.
            if (present(require_class_type_identities)) ep% &
                require_class_type_identities = require_class_type_identities
            do j = 1, arg_count
                ep%arg_value_kinds(j) = VALUE_I32
            end do
            if (present(arg_kinds)) then
                do j = 1, min(arg_count, size(arg_kinds))
                    ep%arg_value_kinds(j) = arg_kinds(j)
                end do
            end if
            ep%arg_names = ''
            if (present(arg_names)) then
                do j = 1, min(arg_count, size(arg_names))
                    ep%arg_names(j) = trim(arg_names(j))
                end do
            end if
            ep%arg_is_optional = .false.
            ep%arg_is_value = .false.
            ep%arg_intents = ARG_INTENT_NONE
            if (present(arg_optionals)) then
                do j = 1, min(arg_count, size(arg_optionals))
                    ep%arg_is_optional(j) = arg_optionals(j)
                end do
            end if
            if (present(arg_values)) then
                do j = 1, min(arg_count, size(arg_values))
                    ep%arg_is_value(j) = arg_values(j)
                end do
            end if
            if (present(arg_intents)) then
                do j = 1, min(arg_count, size(arg_intents))
                    ep%arg_intents(j) = arg_intents(j)
                end do
            end if
            ep%arg_ranks = 0
            ep%arg_extents = 0
            if (present(arg_ranks)) then
                do j = 1, min(arg_count, size(arg_ranks))
                    ep%arg_ranks(j) = arg_ranks(j)
                end do
            end if
            if (present(arg_extents)) then
                do j = 1, min(arg_count, size(arg_extents))
                    ep%arg_extents(j) = arg_extents(j)
                end do
            end if
            ep%arg_is_opaque = .false.
            if (present(arg_opaque)) then
                do j = 1, min(arg_count, size(arg_opaque))
                    ep%arg_is_opaque(j) = arg_opaque(j)
                end do
            end if
            ep%arg_is_class = .false.
            ep%arg_class_types = ''
            if (allocated(ep%arg_class_type_identities)) &
                deallocate(ep%arg_class_type_identities)
            identity_length = 1
            if (present(arg_class_type_identities)) then
                if (size(arg_class_type_identities) > 0) identity_length = max( &
                    identity_length, maxval(len_trim(arg_class_type_identities)))
            end if
            allocate(character(len=identity_length) :: &
                     ep%arg_class_type_identities(MAX_PROC_ARGS))
            ep%arg_class_type_identities = ''
            if (present(arg_class)) then
                do j = 1, min(arg_count, size(arg_class))
                    ep%arg_is_class(j) = arg_class(j)
                end do
            end if
            if (present(arg_class_types)) then
                do j = 1, min(arg_count, size(arg_class_types))
                    ep%arg_class_types(j) = trim(arg_class_types(j))
                end do
            end if
            if (present(arg_class_type_identities)) then
                do j = 1, min(arg_count, size(arg_class_type_identities))
                    ep%arg_class_type_identities(j) = trim( &
                        arg_class_type_identities(j))
                end do
            end if
        end associate
    end subroutine add_external_procedure

    subroutine lower_module_proc_i32_call(arena, node, ext_idx, context, value, &
                                          error_msg, callee_name, argument_kind)
        ! Lower a call to a separately compiled integer module function. The
        ! arguments pass by reference (Fortran ABI) and the call targets the
        ! module procedure's mangled symbol resolved from the .fmod (#284).
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        integer, intent(in) :: ext_idx
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        ! The name the call resolved to, which differs from the written name
        ! when a use-associated generic selected a specific (#415).
        character(len=*), intent(in), optional :: callee_name
        integer, intent(in), optional :: argument_kind
        type(lr_operand_desc_t), allocatable :: args(:)
        integer, allocatable :: copyback_indices(:)
        character(len=:), allocatable :: resolved_name
        integer :: empty_args(0)
        integer :: actual_kind

        resolved_name = trim(node%name)
        if (present(callee_name)) resolved_name = trim(callee_name)
        actual_kind = VALUE_I32
        if (present(argument_kind)) actual_kind = argument_kind
        if (allocated(node%arg_indices)) then
            call check_module_call_arity(arena, context, ext_idx, &
                                         resolved_name, node%arg_indices, &
                                         error_msg)
            if (len_trim(error_msg) > 0) return
            call prepare_reference_args(arena, node%arg_indices, context, &
                                        actual_kind, resolved_name, args, &
                                        copyback_indices, error_msg)
            if (len_trim(error_msg) > 0) return
        else
            call check_module_call_arity(arena, context, ext_idx, &
                                         resolved_name, empty_args, error_msg)
            if (len_trim(error_msg) > 0) return
            allocate (args(0))
            allocate (copyback_indices(0))
        end if
        call pad_absent_optional_args(context, ext_idx, args, copyback_indices)
        if (.not. emit_i32_call(context%session, &
                trim(context%external_procedures(ext_idx)%c_name), args, value, &
                error_msg)) return
        call copy_back_reference_args(context, args, copyback_indices, error_msg)
    end subroutine lower_module_proc_i32_call

    subroutine lower_module_proc_f32_call(arena, node, ext_idx, context, value, &
                                          error_msg, callee_name)
        ! Lower a separately compiled real(4) module function.  Its arguments
        ! use the ordinary Fortran by-reference ABI, while the result remains
        ! an f32 value.
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        integer, intent(in) :: ext_idx
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=*), intent(in), optional :: callee_name
        type(lr_operand_desc_t), allocatable :: args(:)
        integer, allocatable :: copyback_indices(:)
        character(len=:), allocatable :: resolved_name
        integer :: empty_args(0)

        resolved_name = trim(node%name)
        if (present(callee_name)) resolved_name = trim(callee_name)
        if (allocated(node%arg_indices)) then
            call check_module_call_arity(arena, context, ext_idx, &
                                         resolved_name, node%arg_indices, &
                                         error_msg)
            if (len_trim(error_msg) > 0) return
            call prepare_reference_args(arena, node%arg_indices, context, &
                                        VALUE_F32, resolved_name, args, &
                                        copyback_indices, error_msg)
            if (len_trim(error_msg) > 0) return
        else
            call check_module_call_arity(arena, context, ext_idx, &
                                         resolved_name, empty_args, error_msg)
            if (len_trim(error_msg) > 0) return
            allocate (args(0))
            allocate (copyback_indices(0))
        end if
        call pad_absent_optional_args(context, ext_idx, args, copyback_indices)
        if (.not. emit_liric_f32_call(context%session, &
                trim(context%external_procedures(ext_idx)%c_name), args, value, &
                error_msg)) return
        call copy_back_reference_args(context, args, copyback_indices, error_msg)
    end subroutine lower_module_proc_f32_call

    subroutine lower_module_proc_f64_call(arena, node, ext_idx, context, value, &
                                          error_msg, callee_name)
        ! Lower a separately compiled real(8) function. Arguments use the
        ! ordinary Fortran by-reference ABI, while the result is returned as
        ! an f64 value.
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        integer, intent(in) :: ext_idx
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=*), intent(in), optional :: callee_name
        type(lr_operand_desc_t), allocatable :: args(:)
        integer, allocatable :: copyback_indices(:)
        character(len=:), allocatable :: resolved_name
        integer :: empty_args(0)

        resolved_name = trim(node%name)
        if (present(callee_name)) resolved_name = trim(callee_name)
        if (allocated(node%arg_indices)) then
            call check_module_call_arity(arena, context, ext_idx, &
                                         resolved_name, node%arg_indices, &
                                         error_msg)
            if (len_trim(error_msg) > 0) return
            call prepare_reference_args(arena, node%arg_indices, context, &
                                        VALUE_F64, resolved_name, args, &
                                        copyback_indices, error_msg)
            if (len_trim(error_msg) > 0) return
        else
            call check_module_call_arity(arena, context, ext_idx, &
                                         resolved_name, empty_args, error_msg)
            if (len_trim(error_msg) > 0) return
            allocate (args(0))
            allocate (copyback_indices(0))
        end if
        call pad_absent_optional_args(context, ext_idx, args, copyback_indices)
        if (.not. emit_liric_f64_call(context%session, &
                trim(context%external_procedures(ext_idx)%c_name), args, value, &
                error_msg)) return
        call copy_back_reference_args(context, args, copyback_indices, error_msg)
    end subroutine lower_module_proc_f64_call

    subroutine lower_module_proc_void_call(arena, node_index, ext_idx, context, &
                                           error_msg)
        ! Lower a CALL to a separately compiled module subroutine (#284).
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        integer, intent(in) :: ext_idx
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer, allocatable :: arg_indices(:)
        type(lr_operand_desc_t), allocatable :: args(:)
        integer, allocatable :: copyback_indices(:)
        character(len=:), allocatable :: call_name

        call get_subroutine_call_arg_indices(arena, node_index, arg_indices, &
                                             error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. allocated(arg_indices)) allocate (arg_indices(0))
        call_name = trim(context%external_procedures(ext_idx)%fortran_name)
        call check_module_call_arity(arena, context, ext_idx, call_name, &
                                     arg_indices, error_msg)
        if (len_trim(error_msg) > 0) return
        call prepare_reference_args(arena, arg_indices, context, VALUE_I32, &
                                    call_name, args, copyback_indices, error_msg)
        if (len_trim(error_msg) > 0) return
        call pad_absent_optional_args(context, ext_idx, args, copyback_indices)
        if (.not. emit_void_call(context%session, &
                trim(context%external_procedures(ext_idx)%c_name), args, &
                error_msg)) return
        call copy_back_reference_args(context, args, copyback_indices, error_msg)
    end subroutine lower_module_proc_void_call

    subroutine check_module_call_arity(arena, context, ext_idx, callee_name, &
                                       actual_indices, error_msg)
        ! A call to a separately compiled module procedure is checked against
        ! the dummy list the .fmod carries: no call may supply more actuals than
        ! there are dummies, and every dummy the call leaves unassociated must
        ! be OPTIONAL. Keyword actuals are placed first, so `f(b=1)` is judged
        ! on the dummy it names rather than on its written position (#397).
        type(ast_arena_t), intent(in) :: arena
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: ext_idx
        character(len=*), intent(in) :: callee_name
        integer, intent(in) :: actual_indices(:)
        character(len=:), allocatable, intent(out) :: error_msg
        integer, allocatable :: mapped(:)
        integer :: i, arg_count
        logical :: supplied

        call resolve_keyword_actuals(arena, context, actual_indices, &
                                     callee_name, 0, mapped, error_msg)
        if (len_trim(error_msg) > 0) return
        arg_count = context%external_procedures(ext_idx)%arg_count
        if (size(mapped) > arg_count) then
            error_msg = 'More actual than formal arguments in call to '// &
                trim(callee_name)
            return
        end if
        do i = 1, arg_count
            supplied = i <= size(mapped)
            if (supplied) supplied = mapped(i) /= 0
            if (supplied) cycle
            if (context%external_procedures(ext_idx)%arg_is_optional(i)) cycle
            error_msg = 'Missing actual argument for dummy argument '// &
                trim(context%external_procedures(ext_idx)%arg_names(i))// &
                ' in call to '//trim(callee_name)
            return
        end do
    end subroutine check_module_call_arity

    subroutine pad_absent_optional_args(context, ext_idx, args, copyback_indices)
        ! Pad a call to a separately compiled module procedure up to its
        ! declared dummy count: each dummy no actual reached is OPTIONAL (the
        ! arity check already established that) and passes as a null reference,
        ! the same absent-optional encoding a same-unit call uses (#397).
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: ext_idx
        type(lr_operand_desc_t), allocatable, intent(inout) :: args(:)
        integer, allocatable, intent(inout) :: copyback_indices(:)
        type(lr_operand_desc_t), allocatable :: padded(:)
        integer, allocatable :: padded_copyback(:)
        integer :: arg_count, j

        arg_count = context%external_procedures(ext_idx)%arg_count
        if (size(args) >= arg_count) return
        allocate (padded(arg_count))
        allocate (padded_copyback(arg_count))
        padded_copyback = 0
        do j = 1, size(args)
            padded(j) = args(j)
            padded_copyback(j) = copyback_indices(j)
        end do
        do j = size(args) + 1, arg_count
            padded(j) = absent_optional_operand(context)
        end do
        call move_alloc(padded, args)
        call move_alloc(padded_copyback, copyback_indices)
    end subroutine pad_absent_optional_args

    logical function has_bind_c(clause)
        character(len=:), allocatable, intent(in) :: clause

        has_bind_c = .false.
        if (.not. allocated(clause)) return
        has_bind_c = index(lowercase_text(clause), 'bind') > 0
    end function has_bind_c

    subroutine bind_c_name(clause, fortran_name, c_name)
        ! Extract the C name from a bind(c, name="...") clause; default to the
        ! lowercased Fortran name when no name= is present.
        character(len=*), intent(in) :: clause
        character(len=*), intent(in) :: fortran_name
        character(len=:), allocatable, intent(out) :: c_name
        integer :: name_pos, first_quote, second_quote

        name_pos = index(lowercase_text(clause), 'name')
        if (name_pos > 0) then
            first_quote = index(clause(name_pos:), '"')
            if (first_quote > 0) then
                first_quote = name_pos + first_quote - 1
                second_quote = index(clause(first_quote + 1:), '"')
                if (second_quote > 0) then
                    c_name = clause(first_quote + 1:first_quote + second_quote - 1)
                    return
                end if
            end if
        end if
        c_name = trim(lowercase_text(fortran_name))
    end subroutine bind_c_name

    subroutine prepare_external_args(arena, arg_indices, context, args, error_msg)
        ! Lower bind(c) call arguments by value. A c_null_ptr actual becomes a
        ! null pointer operand (8 bytes); other actuals lower as i32 values.
        type(ast_arena_t), intent(in) :: arena
        integer, allocatable, intent(in) :: arg_indices(:)
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), allocatable, intent(out) :: args(:)
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: n, i
        type(lr_operand_desc_t) :: char_data, char_length

        n = 0
        if (allocated(arg_indices)) n = size(arg_indices)
        allocate (args(n))
        do i = 1, n
            if (is_c_null_ptr_arg(arena, arg_indices(i))) then
                args(i)%kind = LR_OP_KIND_IMM_I64
                args(i)%payload = 0_c_int64_t
                args(i)%typ = lr_type_ptr_s(context%session%handle)
                args(i)%global_offset = 0_c_int64_t
            else if (is_character_operand(arena, arg_indices(i), context)) then
                ! A scalar CHARACTER(len=1) dummy in a BIND(C) interface is
                ! passed as the address of its character storage, not as the
                ! internal character descriptor.  This also handles a
                ! temporary expression such as `s // c_null_char` (#584).
                call char_expr_operands(arena, arg_indices(i), context, &
                                        char_data, char_length, error_msg)
                if (len_trim(error_msg) > 0) return
                args(i) = char_data
            else
                call lower_i32_expression(arena, arg_indices(i), context, &
                                          args(i), error_msg)
                if (len_trim(error_msg) > 0) return
            end if
        end do
        call set_empty(error_msg)
    end subroutine prepare_external_args

    logical function is_c_null_ptr_arg(arena, node_index)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        character(len=:), allocatable :: id_name, id_err

        is_c_null_ptr_arg = .false.
        if (.not. is_identifier(arena, node_index)) return
        call get_identifier_name(arena, node_index, id_name, id_err)
        if (len_trim(id_err) > 0) return
        is_c_null_ptr_arg = same_name(id_name, 'c_null_ptr')
    end function is_c_null_ptr_arg

    subroutine prepare_bind_c_args(arena, actual_indices, ext_idx, context, args, &
                                   copyback_indices, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: actual_indices(:)
        integer, intent(in) :: ext_idx
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), allocatable, intent(out) :: args(:)
        integer, allocatable, intent(out) :: copyback_indices(:)
        character(len=:), allocatable, intent(out) :: error_msg
        integer, allocatable :: mapped(:)
        integer :: p, actual, arg_count, value_kind, actual_kind
        type(lr_operand_desc_t) :: value, complex_value, char_length
        character(len=:), allocatable :: callee_name

        call set_empty(error_msg)
        callee_name = trim(context%external_procedures(ext_idx)%fortran_name)
        call check_module_call_arity(arena, context, ext_idx, callee_name, &
                                     actual_indices, error_msg)
        if (len_trim(error_msg) > 0) return
        call resolve_keyword_actuals(arena, context, actual_indices, callee_name, &
                                     0, mapped, error_msg)
        if (len_trim(error_msg) > 0) return
        arg_count = context%external_procedures(ext_idx)%arg_count
        allocate (args(arg_count), copyback_indices(arg_count))
        copyback_indices = 0
        do p = 1, arg_count
            actual = 0
            if (p <= size(mapped)) actual = mapped(p)
            if (actual <= 0) then
                args(p) = absent_optional_operand(context)
                cycle
            end if
            value_kind = context%external_procedures(ext_idx)%arg_value_kinds(p)
            if (context%external_procedures(ext_idx)%arg_is_value(p)) then
                if (value_kind == VALUE_CHARACTER) then
                    error_msg = 'BIND(C), VALUE character arguments are not lowered'
                    return
                else if (value_kind == VALUE_C4 .or. value_kind == VALUE_C8) then
                    call prepare_reference_arg(arena, actual, context, value_kind, &
                                               args(p), copyback_indices(p), &
                                               error_msg)
                    if (len_trim(error_msg) > 0) return
                    if (.not. emit_complex_value_load(context%session, args(p), &
                            value_kind == VALUE_C8, complex_value, error_msg)) return
                    args(p) = complex_value
                    copyback_indices(p) = 0
                else
                    ! The direct-session representation uses i64 for default
                    ! integers, while a BIND(C), VALUE c_int dummy is i32.
                    ! Convert at this ABI boundary instead of weakening the
                    ! scalar expression evaluators globally.
                    actual_kind = expression_value_kind(arena, actual, context, &
                                                        value_kind)
                    if (value_kind == VALUE_I32 .and. &
                        actual_kind == VALUE_I64) then
                        call lower_i64_expression(arena, actual, context, value, &
                                                  error_msg)
                        if (len_trim(error_msg) > 0) return
                        if (.not. emit_liric_i64_to_i32(context%session, value, &
                                                       args(p), error_msg)) return
                    else
                        call lower_expression_by_kind(arena, actual, context, &
                                                      value_kind, value, error_msg)
                        if (len_trim(error_msg) > 0) return
                        args(p) = value
                    end if
                end if
            else if (value_kind == VALUE_CHARACTER) then
                call char_expr_operands(arena, actual, context, args(p), &
                                        char_length, error_msg)
                if (len_trim(error_msg) > 0) return
            else if (context%external_procedures(ext_idx)%arg_ranks(p) > 0) then
                if (actual_is_whole_array(arena, actual, context)) then
                    call whole_array_actual_address(arena, actual, context, args(p), &
                                                    error_msg)
                else if (actual_is_array_expression(arena, actual, context)) then
                    call array_expression_actual_address(arena, actual, context, &
                        callee_name, p, value_kind, args(p), error_msg)
                else if (actual_is_contiguous_section(arena, actual, context)) then
                    call section_actual_address(arena, actual, context, args(p), &
                                                error_msg)
                else
                    error_msg = 'BIND(C) array actual is not contiguous'
                end if
                if (len_trim(error_msg) > 0) return
            else
                call prepare_reference_arg(arena, actual, context, value_kind, &
                                           args(p), copyback_indices(p), error_msg)
                if (len_trim(error_msg) > 0) return
            end if
        end do
    end subroutine prepare_bind_c_args

    subroutine lower_external_i32_call(arena, node, ext_idx, context, value, &
                                       error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        integer, intent(in) :: ext_idx
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t), allocatable :: args(:)
        integer, allocatable :: copyback_indices(:)
        integer :: empty_args(0)

        if (allocated(node%arg_indices)) then
            call prepare_bind_c_args(arena, node%arg_indices, ext_idx, context, &
                                     args, copyback_indices, error_msg)
        else
            call prepare_bind_c_args(arena, empty_args, ext_idx, context, args, &
                                     copyback_indices, error_msg)
        end if
        if (len_trim(error_msg) > 0) return
        if (.not. emit_c_i32_call(context%session, &
                trim(context%external_procedures(ext_idx)%c_name), args, value, &
                error_msg)) return
        call copy_back_reference_args(context, args, copyback_indices, error_msg)
    end subroutine lower_external_i32_call

    subroutine lower_external_void_call(arena, node_index, ext_idx, context, &
                                        error_msg)
        ! Lower a CALL to a registered bind(c) subroutine.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        integer, intent(in) :: ext_idx
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer, allocatable :: arg_indices(:)
        type(lr_operand_desc_t), allocatable :: args(:)
        type(lr_operand_desc_t) :: value
        integer, allocatable :: copyback_indices(:)
        integer :: n, empty_args(0)

        call get_subroutine_call_arg_indices(arena, node_index, arg_indices, &
                                             error_msg)
        if (len_trim(error_msg) > 0) return
        n = 0
        if (allocated(arg_indices)) n = size(arg_indices)
        if (n > 0) then
            call prepare_bind_c_args(arena, arg_indices, ext_idx, context, args, &
                                     copyback_indices, error_msg)
        else
            call prepare_bind_c_args(arena, empty_args, ext_idx, context, args, &
                                     copyback_indices, error_msg)
        end if
        if (len_trim(error_msg) > 0) return
        if (.not. emit_c_void_call(context%session, &
                trim(context%external_procedures(ext_idx)%c_name), args, &
                error_msg)) return
        call copy_back_reference_args(context, args, copyback_indices, error_msg)
    end subroutine lower_external_void_call

    subroutine lower_external_complex_call(arena, node, ext_idx, context, &
                                           is_double, re_value, im_value, &
                                           error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        integer, intent(in) :: ext_idx
        type(lowering_context_t), intent(inout) :: context
        logical, intent(in) :: is_double
        type(lr_operand_desc_t), intent(out) :: re_value, im_value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t), allocatable :: args(:)
        integer, allocatable :: copyback_indices(:)
        integer :: empty_args(0)

        if (allocated(node%arg_indices)) then
            call prepare_bind_c_args(arena, node%arg_indices, ext_idx, context, &
                                     args, copyback_indices, error_msg)
        else
            call prepare_bind_c_args(arena, empty_args, ext_idx, context, args, &
                                     copyback_indices, error_msg)
        end if
        if (len_trim(error_msg) > 0) return
        if (.not. emit_c_complex_call(context%session, &
                trim(context%external_procedures(ext_idx)%c_name), args, &
                is_double, re_value, im_value, error_msg)) return
        call copy_back_reference_args(context, args, copyback_indices, error_msg)
    end subroutine lower_external_complex_call

    integer function external_procedure_index(context, name) result(idx)
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: name
        integer :: i

        idx = 0
        do i = 1, context%external_procedure_count
            if (same_name(context%external_procedures(i)%fortran_name, name)) then
                idx = i
                return
            end if
        end do
    end function external_procedure_index

    subroutine grow_external_procedures(context)
        type(lowering_context_t), intent(inout) :: context
        type(external_procedure_t), allocatable :: old(:)

        if (.not. allocated(context%external_procedures)) then
            allocate (context%external_procedures(8))
            return
        end if
        if (context%external_procedure_count < size(context%external_procedures)) &
            return
        call move_alloc(context%external_procedures, old)
        allocate (context%external_procedures(2 * size(old)))
        context%external_procedures(1:size(old)) = old
    end subroutine grow_external_procedures

    subroutine copy_external_procedures(source, destination)
        ! Procedure bodies lower in a child context, but USE-associated .fmod
        ! calls remain part of the translation-unit contract. Preserve those
        ! callable identities and ABI records across that context boundary
        ! (#584).
        type(lowering_context_t), intent(in) :: source
        type(lowering_context_t), intent(inout) :: destination
        integer :: capacity

        destination%external_procedure_count = 0
        if (source%external_procedure_count <= 0) return
        capacity = max(8, source%external_procedure_count)
        allocate (destination%external_procedures(capacity))
        destination%external_procedure_count = source%external_procedure_count
        destination%external_procedures(1:source%external_procedure_count) = &
            source%external_procedures(1:source%external_procedure_count)
    end subroutine copy_external_procedures

 function degeneric_call_name(context, name, call_arg_count, call_arg_kinds, &
                                call_arg_ranks) &
        result(resolved)
        ! If name is a USE-rename alias, resolve to the real procedure first
        ! (#274); if it is a registered generic, return the matching specific
        ! name; otherwise return name unchanged (#249 B7c).
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: name
        integer, intent(in) :: call_arg_count
        integer, intent(in) :: call_arg_kinds(MAX_PROC_ARGS)
        integer, intent(in) :: call_arg_ranks(MAX_PROC_ARGS)
        character(len=:), allocatable :: resolved, dealiased
        integer :: g

        if (visible_nested_procedure_node(context, name) > 0) then
            resolved = trim(name)
            return
        end if
        dealiased = resolve_proc_alias(context, name)
        g = find_generic(context, dealiased)
        if (g > 0) then
            resolved = resolve_generic(context, g, call_arg_count, &
                                       call_arg_kinds, call_arg_ranks)
        else
            resolved = dealiased
        end if
    end function degeneric_call_name

    function resolve_proc_alias(context, name) result(target)
        ! Map a USE-rename procedure alias to its real procedure name (#274);
        ! returns name unchanged when it is not an alias.
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: name
        character(len=:), allocatable :: target
        integer :: i

        target = trim(name)
        if (.not. allocated(context%proc_alias_locals)) return
        do i = 1, context%proc_alias_count
            if (same_name(context%proc_alias_locals(i), name)) then
                target = trim(context%proc_alias_targets(i))
                return
            end if
        end do
    end function resolve_proc_alias

 integer function generic_call_return_kind(context, name, call_arg_count, &
                                             call_arg_kinds, call_arg_ranks) &
        result(ret_kind)
        ! Return kind of the specific selected for a generic call, or 0 when
        ! name is not a generic (#249 B7c).
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: name
        integer, intent(in) :: call_arg_count
        integer, intent(in) :: call_arg_kinds(MAX_PROC_ARGS)
        integer, intent(in) :: call_arg_ranks(MAX_PROC_ARGS)
        integer :: g

        g = find_generic(context, name)
        if (g > 0) then
            ret_kind = generic_return_kind(context, g, call_arg_count, &
                                           call_arg_kinds, call_arg_ranks)
        else
            ret_kind = 0
        end if
    end function generic_call_return_kind
