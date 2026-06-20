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
        ! can resolve the generic name to the right specific (#249 B7c).
        type(ast_arena_t), intent(in) :: arena
        type(interface_block_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i, j, next
        type(module_procedure_node), pointer :: mp_node
        character(len=:), allocatable :: node_type

        call set_empty(error_msg)
        call grow_generics(context)
        next = context%generic_count + 1
        context%generics(next)%generic_name = trim(node%name)
        context%generics(next)%specific_count = 0

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
                if (context%generics(next)%specific_count >= &
                    MAX_GENERIC_SPECIFICS) then
                    error_msg = 'generic interface '//trim(node%name)// &
                                ' has too many specific procedures'
                    return
                end if
                context%generics(next)%specific_count = &
                    context%generics(next)%specific_count + 1
                associate (sc => context%generics(next)%specific_count)
                    context%generics(next)%specific_names(sc) = &
                        trim(mp_node%procedure_names(j)%s)
                    ! Look up the specific's return kind and first-arg kind
                    ! from the function table (populated during
                    ! collect_internal_function_names / register_module_procedures).
                    call query_specific_kinds(context, &
                        trim(mp_node%procedure_names(j)%s), &
                        context%generics(next)%return_kinds(sc), &
                        context%generics(next)%first_arg_kinds(sc))
                end associate
            end do
        end do
        context%generic_count = next
    end subroutine register_generic_interface

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
                if (.not. allocated(decl%var_name)) cycle
                if (.not. same_name(decl%var_name, param_name)) cycle
                if (declaration_derived_type_index(context, decl) > 0) then
                    kind = VALUE_DERIVED
                    return
                end if
                call type_name_value_kind(decl%type_name, decl%line, &
                                          decl%column, kind, err)
                if (len_trim(err) > 0) kind = VALUE_DERIVED
                return
            end select
        end do
    end function specific_arg_kind

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
                                    return_kind, first_arg_kind)
        ! Look up the return kind and first-argument kind of a specific
        ! procedure from the function table. Falls back to VALUE_I32 when
        ! the specific is not yet registered (registration order does not
        ! guarantee functions come before the interface block).
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: specific_name
        integer, intent(out) :: return_kind, first_arg_kind
        integer :: i

        return_kind = VALUE_I32
        first_arg_kind = VALUE_I32
        do i = 1, context%function_count
            if (same_name(context%function_names(i), specific_name)) then
                return_kind = context%function_value_kinds(i)
                ! First-arg kind: scan the specific's node for its first
                ! parameter declaration.
                first_arg_kind = specific_first_arg_kind(context, i)
                return
            end if
        end do
    end subroutine query_specific_kinds

    integer function specific_first_arg_kind(context, fn_index) result(kind)
        ! Determine the value kind of the first dummy argument of a registered
        ! specific by looking at its body declarations.
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: fn_index
        integer :: node_idx, i
        character(len=:), allocatable :: param_name, err
        character(len=:), allocatable :: node_type
        type(function_def_node), pointer :: fn_node
        type(subroutine_def_node), pointer :: sb_node
        integer, allocatable :: body_indices(:), param_indices(:)

        kind = VALUE_I32
        node_idx = context%function_node_indices(fn_index)
        if (node_idx <= 0) return
        if (.not. node_exists(context%arena, node_idx)) return

        node_type = get_node_type_at(context%arena, node_idx)
        if (node_type == 'function_def_node' .or. &
            node_type == 'function_def') then
            fn_node => get_node_as_function_def(context%arena, node_idx)
            if (.not. associated(fn_node)) return
            if (.not. allocated(fn_node%param_indices)) return
            if (size(fn_node%param_indices) == 0) return
            call parameter_name(context%arena, fn_node%param_indices(1), &
                                param_name, err)
            if (len_trim(err) > 0) return
            body_indices = fn_node%body_indices
        else if (node_type == 'subroutine_def_node' .or. &
                 node_type == 'subroutine_def') then
            sb_node => get_node_as_subroutine_def(context%arena, node_idx)
            if (.not. associated(sb_node)) return
            if (.not. allocated(sb_node%param_indices)) return
            if (size(sb_node%param_indices) == 0) return
            call parameter_name(context%arena, sb_node%param_indices(1), &
                                param_name, err)
            if (len_trim(err) > 0) return
            body_indices = sb_node%body_indices
        else
            return
        end if

        ! Scan body declarations for the first parameter's type.
        if (.not. allocated(body_indices)) return
        do i = 1, size(body_indices)
            if (.not. node_exists(context%arena, body_indices(i))) cycle
            select type (decl => context%arena%entries(body_indices(i))%node)
            type is (declaration_node)
                if (.not. allocated(decl%type_name)) cycle
                if (.not. allocated(decl%var_name)) cycle
                if (.not. same_name(decl%var_name, param_name)) cycle
                call type_name_value_kind(decl%type_name, decl%line, &
                                          decl%column, kind, err)
                return
            end select
        end do
    end function specific_first_arg_kind

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

    function resolve_generic(context, generic_idx, first_arg_kind) &
        result(specific_name)
        ! Choose the specific procedure that matches first_arg_kind.
        ! Returns the first specific whose first_arg_kind equals the actual,
        ! or the first specific when no exact match exists.
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: generic_idx, first_arg_kind
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
            if (g%first_arg_kinds(i) == first_arg_kind) then
                specific_name = trim(g%specific_names(i))
                return
            end if
        end do
        ! Fallback: first specific.
        specific_name = trim(g%specific_names(1))
    end function resolve_generic

    integer function generic_return_kind(context, generic_idx, &
                                         first_arg_kind) result(ret_kind)
        ! Return kind of the specific selected for first_arg_kind.
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: generic_idx, first_arg_kind
        integer :: i
        type(generic_interface_t) :: g

        g = context%generics(generic_idx)
        ret_kind = VALUE_I32
        if (g%specific_count == 0) return
        do i = 1, g%specific_count
            if (g%first_arg_kinds(i) == first_arg_kind) then
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
                                          proc%column, ret_kind, error_msg)
                if (len_trim(error_msg) > 0) return
            end if
            if (ret_kind /= VALUE_I32) then
                call unsupported_feature_error('interface declaration', &
                    proc%line, proc%column, 'only integer bind(c) interface '// &
                    'functions are supported', error_msg)
                return
            end if
            call bind_c_name(proc%bind_c_clause, proc%name, c_name)
            n = 0
            if (allocated(proc%param_indices)) n = size(proc%param_indices)
            if (n > MAX_PROC_ARGS) then
                error_msg = 'bind(c) interface has too many arguments'
                return
            end if
            call add_external_procedure(context, proc%name, c_name, VALUE_I32, n)
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
            call add_external_procedure(context, proc%name, c_name, &
                                        VALUE_SUBROUTINE, n)
        class default
            call unsupported_feature_error('interface declaration', block%line, &
                block%column, 'only bind(c) interface functions are supported', &
                error_msg)
        end select
        call set_empty(error_msg)
    end subroutine register_interface_procedure

    subroutine add_external_procedure(context, fortran_name, c_name, &
                                      return_kind, arg_count)
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: fortran_name, c_name
        integer, intent(in) :: return_kind, arg_count
        integer :: j

        call grow_external_procedures(context)
        context%external_procedure_count = context%external_procedure_count + 1
        associate (ep => context%external_procedures( &
                   context%external_procedure_count))
            ep%fortran_name = trim(fortran_name)
            ep%c_name = trim(c_name)
            ep%return_value_kind = return_kind
            ep%arg_count = arg_count
            do j = 1, arg_count
                ep%arg_value_kinds(j) = VALUE_I32
            end do
        end associate
    end subroutine add_external_procedure

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

        n = 0
        if (allocated(arg_indices)) n = size(arg_indices)
        allocate (args(n))
        do i = 1, n
            if (is_c_null_ptr_arg(arena, arg_indices(i))) then
                args(i)%kind = LR_OP_KIND_IMM_I64
                args(i)%payload = 0_c_int64_t
                args(i)%typ = lr_type_ptr_s(context%session%handle)
                args(i)%global_offset = 0_c_int64_t
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

    subroutine lower_external_i32_call(arena, node, ext_idx, context, value, &
                                       error_msg)
        ! Lower a call to a registered bind(c) integer function: each argument
        ! is passed by value (bind(c) value semantics) and the call targets the
        ! C symbol name.
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        integer, intent(in) :: ext_idx
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t), allocatable :: args(:)
        integer :: n

        n = 0
        if (allocated(node%arg_indices)) n = size(node%arg_indices)
        if (n /= context%external_procedures(ext_idx)%arg_count) then
            call unsupported_feature_error('external function call', node%line, &
                node%column, 'argument count does not match the bind(c) '// &
                'interface for '//trim(node%name), error_msg)
            return
        end if

        call prepare_external_args(arena, node%arg_indices, context, args, &
                                   error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_i32_call(context%session, &
                trim(context%external_procedures(ext_idx)%c_name), args, value, &
                error_msg)) return
        call set_empty(error_msg)
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
        integer :: n

        call get_subroutine_call_arg_indices(arena, node_index, arg_indices, &
                                             error_msg)
        if (len_trim(error_msg) > 0) return
        n = 0
        if (allocated(arg_indices)) n = size(arg_indices)
        if (n /= context%external_procedures(ext_idx)%arg_count) then
            error_msg = 'argument count does not match the bind(c) interface'
            return
        end if
        call prepare_external_args(arena, arg_indices, context, args, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_void_call(context%session, &
                trim(context%external_procedures(ext_idx)%c_name), args, &
                error_msg)) return
        call set_empty(error_msg)
    end subroutine lower_external_void_call

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

    function degeneric_call_name(context, name, first_arg_kind) result(resolved)
        ! If name is a USE-rename alias, resolve to the real procedure first
        ! (#274); if it is a registered generic, return the matching specific
        ! name; otherwise return name unchanged (#249 B7c).
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: name
        integer, intent(in) :: first_arg_kind
        character(len=:), allocatable :: resolved, dealiased
        integer :: g

        dealiased = resolve_proc_alias(context, name)
        g = find_generic(context, dealiased)
        if (g > 0) then
            resolved = resolve_generic(context, g, first_arg_kind)
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

    integer function generic_call_return_kind(context, name, first_arg_kind) &
        result(ret_kind)
        ! Return kind of the specific selected for a generic call, or 0 when
        ! name is not a generic (#249 B7c).
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: name
        integer, intent(in) :: first_arg_kind
        integer :: g

        g = find_generic(context, name)
        if (g > 0) then
            ret_kind = generic_return_kind(context, g, first_arg_kind)
        else
            ret_kind = 0
        end if
    end function generic_call_return_kind
