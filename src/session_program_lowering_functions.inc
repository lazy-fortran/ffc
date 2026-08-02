    ! Return the unit body for the body-walking passes. A program root yields its
    ! body_indices. A mixed_construct_container root (lazy source mixing a
    ! top-level procedure with bare main-program statements) yields its explicit
    ! program units followed by the implicit main statements, so the existing
    ! walk lowers the procedures as contained functions and the statements as the
    ! main body (#266, #275).
    subroutine get_unit_body_indices(arena, root_index, body_indices, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: root_index
        integer, allocatable, intent(out) :: body_indices(:)
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: prog_name, ff_error
        integer :: n_expl, n_impl

        call set_empty(error_msg)
        select type (root => arena%entries(root_index)%node)
        type is (mixed_construct_container_node)
            n_expl = 0
            n_impl = 0
            if (allocated(root%explicit_program_indices)) &
                n_expl = size(root%explicit_program_indices)
            if (allocated(root%implicit_declaration_indices)) &
                n_impl = size(root%implicit_declaration_indices)
            allocate(body_indices(n_expl + n_impl))
            if (n_expl > 0) &
                body_indices(1:n_expl) = root%explicit_program_indices
            if (n_impl > 0) &
                body_indices(n_expl + 1:n_expl + n_impl) = &
                    root%implicit_declaration_indices
        type is (multi_unit_container_node)
            ! Several top-level units from one source (e.g. a module followed by
            ! its program). FortFront aggregates them under this container
            ! (#2830, formerly the __MULTI_UNIT__ sentinel program); its
            ! body_indices are the contained units, walked like a program body.
            if (allocated(root%body_indices)) then
                allocate(body_indices(size(root%body_indices)))
                body_indices = root%body_indices
            else
                allocate(body_indices(0))
            end if
        type is (submodule_node)
            ! A submodule compiled as its own unit contributes no main-body
            ! statement: its procedures are emitted separately against the
            ! parent module's mangling (#297).
            allocate(body_indices(0))
        class default
            call get_program_body_info(arena, root_index, prog_name, &
                                       body_indices, ff_error)
            if (len_trim(ff_error) > 0) error_msg = ff_error
        end select
        if (len_trim(error_msg) == 0) &
            call drop_nested_procedure_units(arena, body_indices)
    end subroutine get_unit_body_indices

    subroutine drop_nested_procedure_units(arena, body_indices)
        ! FortFront hoists the last CONTAINS procedure of a subroutine or
        ! function out to the enclosing multi_unit_container's top-level list
        ! while leaving a bare `contains` marker in the original body. When that
        ! hoisted procedure shares a name with a genuine top-level procedure (a
        ! contained helper and an external procedure with the same name), the
        ! flat function table sees two same-named top-level units and both are
        ! registered and emitted. Two same-named procedures at file scope are not
        ! valid Fortran, so any repeat is a hoisting artefact: keep the first
        ! occurrence of each procedure name and drop later duplicates.
        type(ast_arena_t), intent(in) :: arena
        integer, allocatable, intent(inout) :: body_indices(:)
        integer, allocatable :: kept(:)
        character(len=:), allocatable :: name
        integer :: i, keep_count

        if (.not. allocated(body_indices)) return
        if (size(body_indices) <= 1) return
        allocate(kept(size(body_indices)))
        keep_count = 0
        do i = 1, size(body_indices)
            if (procedure_unit_name(arena, body_indices(i), name)) then
                if (name_precedes_in_list(arena, body_indices, i, name)) cycle
            end if
            keep_count = keep_count + 1
            kept(keep_count) = body_indices(i)
        end do
        if (keep_count == size(body_indices)) return
        deallocate(body_indices)
        allocate(body_indices(keep_count))
        body_indices = kept(1:keep_count)
    end subroutine drop_nested_procedure_units

    logical function name_precedes_in_list(arena, body_indices, pos, name) &
                                           result(precedes)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: body_indices(:)
        integer, intent(in) :: pos
        character(len=*), intent(in) :: name
        character(len=:), allocatable :: earlier
        integer :: j

        precedes = .false.
        do j = 1, pos - 1
            if (.not. procedure_unit_name(arena, body_indices(j), earlier)) cycle
            if (same_name(earlier, name)) then
                precedes = .true.
                return
            end if
        end do
    end function name_precedes_in_list

    logical function procedure_unit_name(arena, node_index, name) result(is_proc)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        character(len=:), allocatable, intent(out) :: name
        character(len=:), allocatable :: node_type
        type(function_def_node), pointer :: fn_node
        type(subroutine_def_node), pointer :: sb_node

        is_proc = .false.
        name = ''
        if (.not. node_exists(arena, node_index)) return
        node_type = get_node_type_at(arena, node_index)
        if (node_type == 'function_def_node' .or. node_type == 'function_def') then
            fn_node => get_node_as_function_def(arena, node_index)
            if (.not. associated(fn_node)) return
            if (allocated(fn_node%name)) name = fn_node%name
        else if (node_type == 'subroutine_def_node' .or. &
                 node_type == 'subroutine_def') then
            sb_node => get_node_as_subroutine_def(arena, node_index)
            if (.not. associated(sb_node)) return
            if (allocated(sb_node%name)) name = sb_node%name
        else
            return
        end if
        is_proc = .true.
    end function procedure_unit_name

    subroutine collect_internal_function_names(arena, root_index, context, &
                                               error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: root_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: ff_error, node_type
        character(len=:), allocatable :: fn_name, fn_result, sb_name
        integer, allocatable :: body_indices(:), fn_params(:), fn_body(:), &
                               sb_params(:), sb_body(:)
        integer :: i

        context%function_count = 0
        if (allocated(context%function_names)) deallocate(context%function_names)
        if (allocated(context%function_value_kinds)) deallocate(context%function_value_kinds)
        if (allocated(context%function_param_counts)) &
            deallocate(context%function_param_counts)
        if (allocated(context%function_node_indices)) &
            deallocate(context%function_node_indices)
        allocate(context%function_names(8))
        allocate(context%function_value_kinds(8))
        allocate(context%function_param_counts(8))
        allocate(context%function_node_indices(8))
        context%function_value_kinds = VALUE_I32
        context%function_param_counts = 0
        context%function_node_indices = 0
        call set_empty(error_msg)

        node_type = get_node_type_at(arena, root_index)
        if (node_type == 'module_node' .or. node_type == 'module') then
            ! A bare module root: register its procedures so a self- or
            ! sibling-call inside a module procedure resolves (e.g. a recursive
            ! module function referencing itself).
            call register_module_procedures(arena, root_index, context, error_msg)
            return
        end if
        if (is_submodule_unit(node_type)) then
            ! A submodule compiled as its own unit: its procedures register
            ! against the parent module named in its header (#297).
            call register_one_submodule_procedures(arena, root_index, context, &
                                                   error_msg)
            return
        end if
        if (node_type == 'block_data_node' .or. node_type == 'block_data') return
        call get_unit_body_indices(arena, root_index, body_indices, ff_error)
        if (len_trim(ff_error) > 0) then
            error_msg = ff_error
            return
        end if
        if (.not. allocated(body_indices)) return
        do i = 1, size(body_indices)
            node_type = get_node_type_at(arena, body_indices(i))
            if (node_type == 'function_def_node' .or. &
                node_type == 'function_def') then
                call get_function_body_info(arena, body_indices(i), &
                                            fn_name, fn_params, fn_body, &
                                            fn_result, ff_error)
                if (len_trim(ff_error) > 0) then
                    error_msg = ff_error
                    return
                end if
                ! Create a synthetic function_def_node for register_internal_function_name
                ! by passing the node_index directly - the register function accesses
                ! node fields that are available via the query results
                call register_internal_function_by_index(arena, body_indices(i), &
                                                         context, error_msg)
                if (len_trim(error_msg) > 0) return
            else if (node_type == 'subroutine_def_node' .or. &
                     node_type == 'subroutine_def') then
                call get_subroutine_body_info(arena, body_indices(i), &
                                              sb_name, sb_params, sb_body, &
                                              ff_error)
                if (len_trim(ff_error) > 0) then
                    error_msg = ff_error
                    return
                end if
                call register_internal_subroutine_by_index(arena, body_indices(i), &
                                                           context, error_msg)
                if (len_trim(error_msg) > 0) return
            else if (node_type == 'module_node' .or. node_type == 'module') then
                call register_module_procedures(arena, body_indices(i), context, &
                                                error_msg)
                if (len_trim(error_msg) > 0) return
            else if (node_type == 'program_node' .or. &
                     node_type == 'program') then
                call register_program_contained_procedures(arena, &
                    body_indices(i), context, error_msg)
                if (len_trim(error_msg) > 0) return
            else if (is_submodule_unit(node_type)) then
                call register_one_submodule_procedures(arena, body_indices(i), &
                                                       context, error_msg)
                if (len_trim(error_msg) > 0) return
            end if
        end do
    end subroutine collect_internal_function_names

    subroutine register_program_contained_procedures(arena, program_index, &
                                                     context, error_msg)
        ! A multi_unit_container holds a program alongside its modules; the
        ! container walk sees only the program_node, so its own CONTAINS
        ! procedures must be registered here for call dispatch, exactly as a
        ! bare-program root's contained defs already are.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: program_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: prog_name, ff_error, node_type
        integer, allocatable :: body_indices(:)
        integer :: i

        call set_empty(error_msg)
        call get_program_body_info(arena, program_index, prog_name, &
                                   body_indices, ff_error)
        if (len_trim(ff_error) > 0) then
            error_msg = ff_error
            return
        end if
        if (.not. allocated(body_indices)) return
        do i = 1, size(body_indices)
            node_type = get_node_type_at(arena, body_indices(i))
            if (node_type == 'function_def_node' .or. &
                node_type == 'function_def') then
                call register_internal_function_by_index(arena, body_indices(i), &
                                                         context, error_msg)
                if (len_trim(error_msg) > 0) return
            else if (node_type == 'subroutine_def_node' .or. &
                     node_type == 'subroutine_def') then
                call register_internal_subroutine_by_index(arena, &
                    body_indices(i), context, error_msg)
                if (len_trim(error_msg) > 0) return
            end if
        end do
    end subroutine register_program_contained_procedures

    subroutine lower_program_contained_functions(arena, program_index, context, &
                                                 error_msg)
        ! Lower the CONTAINS procedures of a program nested in a
        ! multi_unit_container. The container body-walk reaches only the
        ! program_node, so its contained function and subroutine bodies are
        ! emitted here alongside the sibling modules' procedures.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: program_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: prog_name, ff_error, node_type
        integer, allocatable :: body_indices(:)
        integer :: i

        call set_empty(error_msg)
        call get_program_body_info(arena, program_index, prog_name, &
                                   body_indices, ff_error)
        if (len_trim(ff_error) > 0) then
            error_msg = ff_error
            return
        end if
        if (.not. allocated(body_indices)) return
        do i = 1, size(body_indices)
            node_type = get_node_type_at(arena, body_indices(i))
            if (node_type == 'function_def_node' .or. &
                node_type == 'function_def' .or. &
                node_type == 'subroutine_def_node' .or. &
                node_type == 'subroutine_def') then
                call lower_one_internal_function(arena, body_indices(i), context, &
                                                 error_msg)
                if (len_trim(error_msg) > 0) return
            end if
        end do
    end subroutine lower_program_contained_functions

    logical function proc_node_has_executable_body(arena, node_index) &
                                                   result(has_body)
        ! True when a function/subroutine node carries an executable body (any
        ! body statement that is not a plain declaration). An explicit-interface
        ! signature declares only its dummies, so it reports no executable body
        ! and never displaces a real definition in the procedure table.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        integer, allocatable :: body_indices(:)
        character(len=:), allocatable :: node_type
        type(function_def_node), pointer :: fn_node
        type(subroutine_def_node), pointer :: sb_node
        integer :: i

        has_body = .false.
        if (node_index <= 0) return
        if (.not. node_exists(arena, node_index)) return
        node_type = get_node_type_at(arena, node_index)
        if (node_type == 'function_def_node' .or. node_type == 'function_def') then
            fn_node => get_node_as_function_def(arena, node_index)
            if (.not. associated(fn_node)) return
            if (.not. allocated(fn_node%body_indices)) return
            body_indices = fn_node%body_indices
        else if (node_type == 'subroutine_def_node' .or. &
                 node_type == 'subroutine_def') then
            sb_node => get_node_as_subroutine_def(arena, node_index)
            if (.not. associated(sb_node)) return
            if (.not. allocated(sb_node%body_indices)) return
            body_indices = sb_node%body_indices
        else
            return
        end if
        do i = 1, size(body_indices)
            if (.not. node_exists(arena, body_indices(i))) cycle
            select type (n => arena%entries(body_indices(i))%node)
            type is (declaration_node)
                cycle
            class default
                has_body = .true.
                return
            end select
        end do
    end function proc_node_has_executable_body

    subroutine reconcile_duplicate_procedure(context, existing, node_index, &
                                             param_count, error_msg)
        ! A procedure name registered twice: an explicit-interface signature and
        ! the real definition. Point the table entry at whichever node carries an
        ! executable body so call-site queries and lowering resolve the real
        ! definition. The declared kind is identical in both, so it is left as is.
        ! Reconcile only when the interface and the definition agree on the
        ! function result's rank, base type, and POINTER attribute; a genuine
        ! signature mismatch (F2018 12.4.2.1) is invalid Fortran and is rejected
        ! rather than silently merged.
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: existing
        integer, intent(in) :: node_index
        integer, intent(in) :: param_count
        character(len=:), allocatable, intent(out) :: error_msg

        call set_empty(error_msg)
        if (node_index <= 0) return
        call reject_result_signature_mismatch(context%arena, node_index, &
            context%function_node_indices(existing), error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. proc_node_has_executable_body(context%arena, node_index)) return
        if (proc_node_has_executable_body(context%arena, &
                                          context%function_node_indices(existing))) &
            return
        context%function_node_indices(existing) = node_index
        context%function_param_counts(existing) = param_count
    end subroutine reconcile_duplicate_procedure

    subroutine reject_result_signature_mismatch(arena, index_a, index_b, &
                                                error_msg)
        ! Reject when two same-named function nodes (an interface body and the
        ! real definition) declare incompatible result signatures. Only fires
        ! when BOTH result declarations are determinable; an implicit or missing
        ! result declaration leaves the pair reconcilable, preserving the
        ! compatible interface<->definition case.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: index_a, index_b
        character(len=:), allocatable, intent(out) :: error_msg
        logical :: found_a, found_b, ptr_a, ptr_b
        integer :: rank_a, rank_b
        character(len=:), allocatable :: type_a, type_b

        call set_empty(error_msg)
        call function_result_signature(arena, index_a, found_a, rank_a, &
                                       type_a, ptr_a)
        call function_result_signature(arena, index_b, found_b, rank_b, &
                                       type_b, ptr_b)
        if (.not. found_a .or. .not. found_b) return
        if (rank_a /= rank_b) then
            error_msg = 'rank mismatch in function result between interface '// &
                'and definition'
            return
        end if
        if (len_trim(type_a) > 0 .and. len_trim(type_b) > 0) then
            if (type_a /= type_b) then
                error_msg = 'type mismatch in function result between '// &
                    'interface and definition'
                return
            end if
        end if
        if (ptr_a .neqv. ptr_b) then
            error_msg = 'POINTER attribute mismatch in function result '// &
                'between interface and definition'
            return
        end if
    end subroutine reject_result_signature_mismatch

    subroutine function_result_signature(arena, node_index, found, rank, &
                                         type_name, is_pointer)
        ! Result declaration rank, lowercased base type name, and POINTER
        ! attribute of a function_def_node, read from the body declaration that
        ! names the result variable (result(x) name, else the function name).
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        logical, intent(out) :: found
        integer, intent(out) :: rank
        character(len=:), allocatable, intent(out) :: type_name
        logical, intent(out) :: is_pointer
        type(function_def_node), pointer :: fn
        character(len=:), allocatable :: node_type, result_name
        integer :: i

        found = .false.
        rank = 0
        is_pointer = .false.
        call set_empty(type_name)
        if (node_index <= 0) return
        if (.not. node_exists(arena, node_index)) return
        node_type = get_node_type_at(arena, node_index)
        if (node_type /= 'function_def_node' .and. node_type /= 'function_def') &
            return
        fn => get_node_as_function_def(arena, node_index)
        if (.not. associated(fn)) return
        if (.not. allocated(fn%name)) return
        result_name = trim(fn%name)
        if (allocated(fn%result_variable)) then
            if (len_trim(fn%result_variable) > 0) &
                result_name = trim(fn%result_variable)
        end if
        if (.not. allocated(fn%body_indices)) return
        do i = 1, size(fn%body_indices)
            if (.not. node_exists(arena, fn%body_indices(i))) cycle
            select type (decl => arena%entries(fn%body_indices(i))%node)
            type is (declaration_node)
                if (.not. decl_names_result(decl, result_name)) cycle
                found = .true.
                if (decl%is_array) then
                    if (allocated(decl%dimension_indices)) then
                        rank = size(decl%dimension_indices)
                    else
                        rank = 1
                    end if
                end if
                if (allocated(decl%type_name)) &
                    type_name = trim(lowercase_text(decl%type_name))
                is_pointer = decl%is_pointer
                return
            end select
        end do
    end subroutine function_result_signature

    logical function decl_names_result(decl, result_name) result(names)
        type(declaration_node), intent(in) :: decl
        character(len=*), intent(in) :: result_name
        integer :: k

        names = .false.
        if (allocated(decl%var_name)) then
            if (same_name(decl%var_name, result_name)) then
                names = .true.
                return
            end if
        end if
        if (decl%is_multi_declaration .and. allocated(decl%var_names)) then
            do k = 1, size(decl%var_names)
                if (same_name(decl%var_names(k), result_name)) then
                    names = .true.
                    return
                end if
            end do
        end if
    end function decl_names_result

    subroutine register_internal_subroutine_name(node, context, error_msg, &
                                                 node_index)
        type(subroutine_def_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer, intent(in), optional :: node_index
        integer :: next_index, existing, new_param_count

        call set_empty(error_msg)
        if (.not. allocated(node%name)) return
        call grow_function_names(context)
        existing = find_function_index(context, node%name)
        if (existing /= 0) then
            if (present(node_index)) then
                new_param_count = 0
                if (allocated(node%param_indices)) &
                    new_param_count = size(node%param_indices)
                call reconcile_duplicate_procedure(context, existing, &
                    node_index, new_param_count, error_msg)
            end if
            return
        end if
        next_index = context%function_count + 1
        context%function_names(next_index) = trim(node%name)
        context%function_value_kinds(next_index) = VALUE_SUBROUTINE
        if (allocated(node%param_indices)) then
            context%function_param_counts(next_index) = size(node%param_indices)
        else
            context%function_param_counts(next_index) = 0
        end if
        if (present(node_index)) then
            context%function_node_indices(next_index) = node_index
        else
            context%function_node_indices(next_index) = 0
        end if
        context%function_count = next_index
    end subroutine register_internal_subroutine_name

    subroutine register_internal_function_by_index(arena, node_index, context, &
                                                   error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(function_def_node), pointer :: node

        call set_empty(error_msg)
        node => get_node_as_function_def(arena, node_index)
        if (.not. associated(node)) then
            error_msg = 'internal function index does not reference an AST node'
            return
        end if
        call register_internal_function_name(node, context, error_msg, node_index)
    end subroutine register_internal_function_by_index

    subroutine register_internal_subroutine_by_index(arena, node_index, context, &
                                                      error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(subroutine_def_node), pointer :: node

        call set_empty(error_msg)
        node => get_node_as_subroutine_def(arena, node_index)
        if (.not. associated(node)) then
            error_msg = 'internal subroutine index does not reference an AST node'
            return
        end if
        call register_internal_subroutine_name(node, context, error_msg, node_index)
    end subroutine register_internal_subroutine_by_index

    subroutine register_procedure_alias(context, source_name, alias_name, &
                                         error_msg)
        ! Add a function-table entry under alias_name that shares the node
        ! index, value kind, and parameter count of an already-registered
        ! procedure source_name. Used for USE rename of a module procedure
        ! (#274).
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: source_name
        character(len=*), intent(in) :: alias_name
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i, source_index, next_index

        call set_empty(error_msg)
        source_index = 0
        do i = 1, context%function_count
            if (same_name(context%function_names(i), source_name)) then
                source_index = i
                exit
            end if
        end do
        if (source_index == 0) then
            error_msg = 'use rename source procedure not registered: '// &
                        trim(source_name)
            return
        end if
        call grow_function_names(context)
        next_index = context%function_count + 1
        context%function_names(next_index) = trim(alias_name)
        context%function_value_kinds(next_index) = &
            context%function_value_kinds(source_index)
        context%function_param_counts(next_index) = &
            context%function_param_counts(source_index)
        context%function_node_indices(next_index) = &
            context%function_node_indices(source_index)
        context%function_count = next_index
        ! Record the alias so a call to alias_name mangles to the real
        ! procedure symbol at call_emit_name time.
        call record_proc_alias(context, alias_name, source_name)
    end subroutine register_procedure_alias

    subroutine record_proc_alias(context, local_name, target_name)
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: local_name
        character(len=*), intent(in) :: target_name
        character(len=64), allocatable :: tmp_locals(:), tmp_targets(:)
        integer :: cap, idx

        if (.not. allocated(context%proc_alias_locals)) then
            allocate (context%proc_alias_locals(8))
            allocate (context%proc_alias_targets(8))
            context%proc_alias_count = 0
        end if
        cap = size(context%proc_alias_locals)
        if (context%proc_alias_count >= cap) then
            call move_alloc(context%proc_alias_locals, tmp_locals)
            call move_alloc(context%proc_alias_targets, tmp_targets)
            allocate (context%proc_alias_locals(2 * cap))
            allocate (context%proc_alias_targets(2 * cap))
            context%proc_alias_locals(1:cap) = tmp_locals
            context%proc_alias_targets(1:cap) = tmp_targets
        end if
        idx = context%proc_alias_count + 1
        context%proc_alias_locals(idx) = trim(local_name)
        context%proc_alias_targets(idx) = trim(target_name)
        context%proc_alias_count = idx
    end subroutine record_proc_alias

    integer function proc_param_count(context, name) result(count)
        ! Declared parameter count of a contained procedure, or -1 if unknown.
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: name
        integer :: i, nested_node_index
        character(len=:), allocatable :: node_type
        type(function_def_node), pointer :: fn_node
        type(subroutine_def_node), pointer :: sb_node

        count = -1
        nested_node_index = visible_nested_procedure_node(context, name)
        if (nested_node_index > 0) then
            node_type = get_node_type_at(context%arena, nested_node_index)
            if (node_type == 'function_def_node' .or. &
                node_type == 'function_def') then
                fn_node => get_node_as_function_def(context%arena, nested_node_index)
                if (associated(fn_node)) then
                    if (allocated(fn_node%param_indices)) then
                        count = size(fn_node%param_indices)
                    else
                        count = 0
                    end if
                end if
            else if (node_type == 'subroutine_def_node' .or. &
                     node_type == 'subroutine_def') then
                sb_node => get_node_as_subroutine_def(context%arena, nested_node_index)
                if (associated(sb_node)) then
                    if (allocated(sb_node%param_indices)) then
                        count = size(sb_node%param_indices)
                    else
                        count = 0
                    end if
                end if
            end if
            return
        end if
        do i = 1, context%function_count
            if (same_name(context%function_names(i), name)) then
                count = context%function_param_counts(i)
                return
            end if
        end do
    end function proc_param_count

    subroutine register_internal_function_name(node, context, error_msg, &
                                               node_index)
        type(function_def_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer, intent(in), optional :: node_index
        integer :: next_index
        integer :: value_kind
        integer :: derived_type_index
        integer :: existing, new_param_count

        if (.not. allocated(node%name)) then
            error_msg = 'direct LIRIC session function requires a name'
            return
        end if
        call grow_function_names(context)
        ! A name seen twice is an explicit-interface signature paired with the
        ! real external/module definition (or a FortFront hoisting artefact),
        ! not two rival definitions. Reconcile so the entry keeps the
        ! body-bearing definition rather than hard-erroring.
        existing = find_function_index(context, node%name)
        if (existing /= 0) then
            if (present(node_index)) then
                new_param_count = 0
                if (allocated(node%param_indices)) &
                    new_param_count = size(node%param_indices)
                call reconcile_duplicate_procedure(context, existing, &
                    node_index, new_param_count, error_msg)
                return
            end if
            call set_empty(error_msg)
            return
        end if
        call function_value_kind(node, context, value_kind, derived_type_index, &
                                 error_msg, node_index)
        if (len_trim(error_msg) > 0) return

        next_index = context%function_count + 1
        context%function_names(next_index) = trim(node%name)
        context%function_value_kinds(next_index) = value_kind
        if (allocated(node%param_indices)) then
            context%function_param_counts(next_index) = size(node%param_indices)
        else
            context%function_param_counts(next_index) = 0
        end if
        if (present(node_index)) then
            context%function_node_indices(next_index) = node_index
        else
            context%function_node_indices(next_index) = 0
        end if
        context%function_count = next_index
        call set_empty(error_msg)
    end subroutine register_internal_function_name

  subroutine function_value_kind(node, context, value_kind, derived_type_index, &
                                   error_msg, reference_index)
        type(function_def_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        integer, intent(out) :: value_kind
        integer, intent(out) :: derived_type_index
        integer, intent(in), optional :: reference_index
        integer :: char_len, paren, rparen
        integer :: pointee_kind
        character(len=:), allocatable :: derived_name, base_type, kind_spec
        character(len=:), allocatable, intent(out) :: error_msg

        value_kind = VALUE_I32
        derived_type_index = 0
        call set_empty(error_msg)

        ! A rank-1 fixed-size array result returns via the sret ABI regardless of
        ! whether return_type names the element type or is empty (result(out)
        ! with the array shape declared in the body). Detect it from the result
        ! variable's body declaration before the scalar classification.
        if (result_var_is_fixed_array(node, context)) then
            value_kind = VALUE_ARRAY_RESULT
            return
        end if

        ! An allocatable rank-1/2 array result returns via the allocatable
        ! descriptor sret ABI; the callee allocates the data at run time.
        if (result_var_is_alloc_array(node, context)) then
            value_kind = VALUE_ALLOC_ARRAY_RESULT
            return
        end if

        ! A scalar data POINTER result returns the target address itself (#407).
        if (result_var_is_data_pointer(node, context, pointee_kind)) then
            value_kind = VALUE_DATA_PTR_RESULT
            return
        end if

        if (.not. allocated(node%return_type)) return

        ! When the result type is declared in the body (result(r) with
        ! `<type> :: r`) rather than on the function statement, return_type is
        ! empty. Resolve the intrinsic scalar kind from that body declaration.
        if (len_trim(node%return_type) == 0) then
            call result_var_scalar_kind(node, context, value_kind, &
                                        derived_type_index, error_msg)
            if (value_kind == VALUE_CHARACTER) &
                value_kind = VALUE_DEFERRED_CHARACTER_RESULT
            return
        end if

        ! Split "real(kind)" into base and kind_spec; double precision has none.
        paren = index(node%return_type, '(')
        if (paren > 0) then
            base_type = lowercase_text(trim(adjustl(node%return_type(1:paren - 1))))
            rparen = index(node%return_type, ')', back=.true.)
            if (rparen > paren) then
                kind_spec = lowercase_text(trim(adjustl( &
                    node%return_type(paren + 1:rparen - 1))))
            else
                call set_empty(kind_spec)
            end if
        else
            base_type = lowercase_text(trim(adjustl(node%return_type)))
            call set_empty(kind_spec)
        end if

        select case (base_type)
        case ('integer')
            if (len_trim(kind_spec) == 0) then
                ! FortFront drops an explicit integer(8) header kind to a bare
                ! "integer"; recover it from the result variable's body
                ! declaration before defaulting to i32.
                call result_var_scalar_kind(node, context, value_kind, &
                                             derived_type_index, error_msg)
                if (len_trim(error_msg) > 0) return
                if (value_kind /= VALUE_I64) then
                    ! Body declaration didn't resolve to i64; check if any
                    ! dummy argument is i64 — a strong indicator the result
                    ! is also i64 when the header kind was dropped.
                    if (.not. function_has_i64_parameter(node, context)) then
                        value_kind = VALUE_I32
                    else
                        value_kind = VALUE_I64
                    end if
                end if
            else
                call integer_kind_to_value_kind(kind_spec, node%line, &
                                                node%column, value_kind, error_msg, &
                                                context, reference_index)
                if (len_trim(error_msg) > 0) return
                if (value_kind /= VALUE_I64) value_kind = VALUE_I32
            end if
        case ('real')
            ! Classify by kind: bare real and real(4) are f32; real(8) is f64.
            call check_float_kind_supported('real', kind_spec, node%line, &
                                            node%column, error_msg, context, &
                                            reference_index)
            if (len_trim(error_msg) > 0) return
            if (real_kind_is_f64(kind_spec, context, reference_index)) then
                value_kind = VALUE_F64
            else if (len_trim(kind_spec) == 0 .and. &
                     result_var_body_decl_is_f64(node, context)) then
                ! FortFront sometimes renders an explicit real(8) function return
                ! as a kindless "real" on the function statement (the kind is
                ! dropped during inference). When the result variable carries an
                ! explicit real(8)/real(real64)/double precision body declaration,
                ! trust that over the kindless statement type.
                value_kind = VALUE_F64
            else if (len_trim(kind_spec) == 0 .and. &
                     lazy_defaults_active(context)) then
                ! Lazy Fortran default real is 8 bytes (#438): a kind-less
                ! `real` result matches its kind-less body declaration, which
                ! records kind 8 too (#571).
                value_kind = VALUE_F64
            else
                value_kind = VALUE_F32
            end if
        case ('double precision')
            value_kind = VALUE_F64
        case ('complex')
            ! Bare complex and complex(4) are two f32s; complex(8) is two f64s.
            call check_float_kind_supported('complex', kind_spec, node%line, &
                                            node%column, error_msg, context, &
                                            reference_index)
            if (len_trim(error_msg) > 0) return
            if (complex_kind_is_c8(kind_spec, context, reference_index)) then
                value_kind = VALUE_C8
            else
                value_kind = VALUE_C4
            end if
        case ('logical')
            value_kind = VALUE_LOGICAL
        case default
            if (is_character_type_name(node%return_type)) then
                call parse_character_length(node%return_type, &
                                            char_len, error_msg)
                if (len_trim(error_msg) > 0) return
                ! Both fixed- and deferred-length character results use the
                ! descriptor return ABI. The body result declaration records a
                ! fixed width when one is present, so assignments still pad or
                ! truncate correctly.
                value_kind = VALUE_DEFERRED_CHARACTER_RESULT
                return
            end if
            call extracted_derived_type_name(node%return_type, derived_name)
            if (len_trim(derived_name) > 0) &
                derived_type_index = find_derived_type(context, derived_name)
            ! The result type is often declared in the body (result(p) with
            ! type(t) :: p), not on the function statement.
            if (derived_type_index == 0) &
                derived_type_index = result_var_derived_type(node, context)
            if (derived_type_index > 0) then
                value_kind = VALUE_DERIVED
            else
                call unsupported_feature_error('function result type', &
                                                node%line, node%column, &
                                                'direct LIRIC session supports '// &
                                                'integer, real, logical, complex, '// &
                                                'deferred-length character, and '// &
                                                'derived-type contained functions', &
                                                error_msg)
            end if
        end select
    end subroutine function_value_kind

    logical function result_var_body_decl_is_f64(node, context) result(is_f64)
        ! True when the function result variable has an explicit scalar
        ! real(8)/real(real64)/double precision declaration in the body. Used to
        ! recover the f64 kind when FortFront drops it from the function
        ! statement's return type, leaving a kindless "real".
        type(function_def_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        character(len=:), allocatable :: result_name, decl_err
        integer :: i, k, value_kind, derived_index
        logical :: is_array

        is_f64 = .false.
        result_name = trim(node%name)
        if (allocated(node%result_variable)) then
            if (len_trim(node%result_variable) > 0) &
                result_name = trim(node%result_variable)
        end if
        if (.not. allocated(node%body_indices)) return
        do i = 1, size(node%body_indices)
            if (.not. node_exists(context%arena, node%body_indices(i))) cycle
            select type (decl => context%arena%entries(node%body_indices(i))%node)
            type is (declaration_node)
                if (decl%is_array) cycle
                if (allocated(decl%var_name)) then
                        if (trim(decl%var_name) == result_name) then
                            call resolve_result_decl_kind(decl, context, value_kind, &
                                derived_index, is_array, decl_err, &
                                node%body_indices(i))
                        if (len_trim(decl_err) == 0 .and. .not. is_array .and. &
                            value_kind == VALUE_F64) is_f64 = .true.
                        if (is_f64) return
                    end if
                end if
                if (decl%is_multi_declaration .and. allocated(decl%var_names)) then
                    do k = 1, size(decl%var_names)
                        if (trim(decl%var_names(k)) == result_name) then
                            call resolve_result_decl_kind(decl, context, &
                                value_kind, derived_index, is_array, decl_err, &
                                node%body_indices(i))
                            if (len_trim(decl_err) == 0 .and. .not. is_array .and. &
                                value_kind == VALUE_F64) is_f64 = .true.
                            if (is_f64) return
                        end if
                    end do
                end if
            end select
        end do
    end function result_var_body_decl_is_f64

    logical function function_has_i64_parameter(node, context) result(has_i64)
        ! True when the function body contains an integer(8)/i64 declaration
        ! or an integer literal with kind '8'.
        ! Used as a fallback to recover i64 return kind when FortFront drops
        ! the kind from the function header (e.g. "integer(8) function f" ->
        ! return_type "integer" with no kind spec). Body declarations carry
        ! the full type information even when the header is truncated. When
        ! declarations also lose the kind, i64 literals in the body serve as
        ! a reliable indicator.
        type(function_def_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        integer :: i
        character(len=:), allocatable :: decl_err
        integer :: decl_kind

        has_i64 = .false.
        if (.not. allocated(node%body_indices)) return
        do i = 1, size(node%body_indices)
            if (.not. node_exists(context%arena, node%body_indices(i))) cycle
            select type (decl => context%arena%entries(node%body_indices(i))%node)
            type is (declaration_node)
                if (decl%is_array) cycle
                call declaration_value_kind(decl, decl_kind, decl_err, context, &
                                            node%body_indices(i))
                if (len_trim(decl_err) == 0 .and. decl_kind == VALUE_I64) then
                    has_i64 = .true.
                    return
                end if
            type is (assignment_node)
                if (body_node_has_i64_literal(context%arena, decl%value_index)) then
                    has_i64 = .true.
                    return
                end if
            end select
        end do
    end function function_has_i64_parameter

    recursive logical function body_node_has_i64_literal(arena, node_index) result(has_lit)
        ! Recursively check if an AST subtree contains an integer literal
        ! with kind '8' (i64 indicator when kind specs are dropped).
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        integer :: idx
        character(len=:), allocatable :: lit_val, lit_type, err
        integer(c_int64_t) :: literal_kind

        has_lit = .false.
        if (.not. node_exists(arena, node_index)) return

        if (is_literal(arena, node_index)) then
            call get_literal_info(arena, node_index, lit_val, lit_type, err)
            if (len_trim(err) == 0) then
                if (trim(lowercase_text(trim(lit_type))) == 'integer') then
                    call integer_literal_kind_number(lit_val, literal_kind, err)
                    has_lit = (len_trim(err) == 0 .and. literal_kind == 8_c_int64_t)
                else
                    has_lit = .false.
                end if
            end if
            return
        end if

        select type (n => arena%entries(node_index)%node)
        type is (binary_op_node)
            has_lit = body_node_has_i64_literal(arena, n%left_index) .or. &
                      body_node_has_i64_literal(arena, n%right_index)
        type is (call_or_subscript_node)
            if (allocated(n%arg_indices)) then
                do idx = 1, size(n%arg_indices)
                    if (body_node_has_i64_literal(arena, n%arg_indices(idx))) then
                        has_lit = .true.
                        return
                    end if
                end do
            end if
        end select
    end function body_node_has_i64_literal

    subroutine result_var_scalar_kind(node, context, value_kind, &
                                      derived_type_index, error_msg)
        ! Resolve the result value kind from the body declaration of the result
        ! variable, used when return_type is empty (result(r) with `<type> :: r`
        ! declared in the body). Handles intrinsic scalar kinds (integer, real,
        ! logical, complex) and derived types. Array-valued results are out of
        ! scope and keep the unsupported diagnostic.
        type(function_def_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        integer, intent(out) :: value_kind
        integer, intent(out) :: derived_type_index
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: result_name
        integer :: i, k
        logical :: is_array

        value_kind = VALUE_I32
        derived_type_index = 0
        call set_empty(error_msg)
        result_name = trim(node%name)
        if (allocated(node%result_variable)) then
            if (len_trim(node%result_variable) > 0) &
                result_name = trim(node%result_variable)
        end if

        if (allocated(node%body_indices)) then
            do i = 1, size(node%body_indices)
                if (.not. node_exists(context%arena, node%body_indices(i))) cycle
                select type (decl => context%arena%entries(node%body_indices(i))%node)
                type is (declaration_node)
                    if (allocated(decl%var_name)) then
                        if (trim(decl%var_name) == result_name) then
                            call resolve_result_decl_kind(decl, context, &
                                value_kind, derived_type_index, is_array, error_msg, &
                                node%body_indices(i))
                            if (.not. is_array) then
                                call remap_scalar_char_result_kind(context, decl, &
                                                                   value_kind, &
                                                                   node%body_indices(i))
                                return
                            end if
                        end if
                    end if
                    if (decl%is_multi_declaration .and. allocated(decl%var_names)) then
                        do k = 1, size(decl%var_names)
                            if (trim(decl%var_names(k)) == result_name) then
                                call resolve_result_decl_kind(decl, context, &
                                    value_kind, derived_type_index, is_array, &
                                    error_msg, node%body_indices(i))
                                if (.not. is_array) then
                                    call remap_scalar_char_result_kind(context, &
                                                                       decl, &
                                                                       value_kind, &
                                                                       node%body_indices(i))
                                    return
                                end if
                            end if
                        end do
                    end if
                end select
            end do
        end if

        ! No explicit result declaration and no result type in the header:
        ! FortFront leaves return_type empty for an implicitly-typed result (and
        ! for some explicitly-typed headers it drops the type too). Recover the
        ! kind from the Fortran default letter rule: i-n integer, else real
        ! (#249 cluster C). Array results are excluded above. An in-scope IMPLICIT
        ! statement that overrides the default is not visible here (FortFront
        ! consumes it), so a result whose implicit kind differs from the default
        ! keeps the later output-mismatch diagnosis rather than a wrong guess.
        call implicit_result_kind(node, result_name, value_kind, error_msg)
    end subroutine result_var_scalar_kind

    subroutine implicit_result_kind(node, result_name, value_kind, error_msg)
        type(function_def_node), intent(in) :: node
        character(len=*), intent(in) :: result_name
        integer, intent(out) :: value_kind
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=1) :: first

        value_kind = VALUE_I32
        call set_empty(error_msg)
        if (len_trim(result_name) == 0) then
            call unsupported_result_type(node, error_msg)
            return
        end if
        first = lowercase_text(result_name(1:1))
        if (first >= 'i' .and. first <= 'n') then
            value_kind = VALUE_I32
        else
            value_kind = VALUE_F32
        end if
    end subroutine implicit_result_kind

    subroutine unsupported_result_type(node, error_msg)
        type(function_def_node), intent(in) :: node
        character(len=:), allocatable, intent(out) :: error_msg

        call unsupported_feature_error('function result type', &
                                        node%line, node%column, &
                                        'direct LIRIC session supports '// &
                                        'integer, real, logical, complex, '// &
                                        'deferred-length character, and '// &
                                        'derived-type contained functions', &
                                        error_msg)
    end subroutine unsupported_result_type

    subroutine remap_scalar_char_result_kind(context, decl, value_kind, &
                                             reference_index)
        ! A scalar character result whose length is deferred (character(len=:))
        ! or a runtime expression (character(len=len(arg))) returns through the
        ! deferred descriptor ABI, matching the contained-function path. A fixed
        ! compile-time length keeps VALUE_CHARACTER: the descriptor return does
        ! not pad short assignments, so that case stays a clean rejection rather
        ! than emitting an unpadded (wrong) result.
        type(lowering_context_t), intent(in) :: context
        type(declaration_node), intent(in) :: decl
        integer, intent(inout) :: value_kind
        integer, intent(in), optional :: reference_index
        integer :: char_len
        character(len=:), allocatable :: len_err

        if (value_kind /= VALUE_CHARACTER) return
        if (present(reference_index)) then
            call declaration_character_length(context, decl, char_len, len_err, &
                                               reference_index)
        else
            call declaration_character_length(context, decl, char_len, len_err)
        end if
        if (len_trim(len_err) > 0) then
            value_kind = VALUE_DEFERRED_CHARACTER_RESULT
        else if (char_len <= 0) then
            value_kind = VALUE_DEFERRED_CHARACTER_RESULT
        end if
    end subroutine remap_scalar_char_result_kind

    subroutine resolve_result_decl_kind(decl, context, value_kind, &
                                        derived_type_index, is_array, error_msg, &
                                        reference_index)
        type(declaration_node), intent(in) :: decl
        type(lowering_context_t), intent(in) :: context
        integer, intent(out) :: value_kind
        integer, intent(out) :: derived_type_index
        logical, intent(out) :: is_array
        character(len=:), allocatable, intent(out) :: error_msg
        integer, intent(in), optional :: reference_index

        value_kind = VALUE_I32
        derived_type_index = 0
        is_array = decl%is_array
        call set_empty(error_msg)
        if (is_array) return

        derived_type_index = declaration_derived_type_index(context, decl)
        if (derived_type_index > 0) then
            value_kind = VALUE_DERIVED
            return
        end if
        call declaration_value_kind(decl, value_kind, error_msg, context, &
                                    reference_index)
    end subroutine resolve_result_decl_kind

    logical function result_var_is_fixed_array(node, context) result(is_fixed)
        ! True when the function result variable is declared as a rank-1
        ! fixed-size (compile-time extent) array in the body. Assumed-shape and
        ! allocatable results stay out of scope and keep their later diagnostic.
        type(function_def_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        character(len=:), allocatable :: result_name
        integer :: i, k

        is_fixed = .false.
        result_name = trim(node%name)
        if (allocated(node%result_variable)) then
            if (len_trim(node%result_variable) > 0) &
                result_name = trim(node%result_variable)
        end if
        if (.not. allocated(node%body_indices)) return
        do i = 1, size(node%body_indices)
            if (.not. node_exists(context%arena, node%body_indices(i))) cycle
            select type (decl => context%arena%entries(node%body_indices(i))%node)
            type is (declaration_node)
                if (allocated(decl%var_name)) then
                    if (trim(decl%var_name) == result_name) then
                        if (declaration_is_fixed_rank1(decl)) is_fixed = .true.
                        if (is_fixed) return
                    end if
                end if
                if (decl%is_multi_declaration .and. allocated(decl%var_names)) then
                    do k = 1, size(decl%var_names)
                        if (trim(decl%var_names(k)) == result_name) then
                            if (declaration_is_fixed_rank1(decl)) is_fixed = .true.
                            if (is_fixed) return
                        end if
                    end do
                end if
            end select
        end do
    end function result_var_is_fixed_array

    logical function result_var_is_data_pointer(node, context, pointee_kind) &
        result(is_data_pointer)
        ! True when the function result variable is declared as a scalar data
        ! POINTER in the body (#407). pointee_kind reports the pointee's scalar
        ! value kind. Pointer array and derived results stay out of scope and
        ! keep their existing diagnostics.
        type(function_def_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        integer, intent(out) :: pointee_kind
        character(len=:), allocatable :: result_name, decl_error
        integer :: i, k, derived_index
        logical :: is_array

        is_data_pointer = .false.
        pointee_kind = VALUE_I32
        result_name = trim(node%name)
        if (allocated(node%result_variable)) then
            if (len_trim(node%result_variable) > 0) &
                result_name = trim(node%result_variable)
        end if
        if (.not. allocated(node%body_indices)) return
        do i = 1, size(node%body_indices)
            if (.not. node_exists(context%arena, node%body_indices(i))) cycle
            select type (decl => context%arena%entries(node%body_indices(i))%node)
            type is (declaration_node)
                if (.not. decl%is_pointer) cycle
                if (decl%is_array) cycle
                is_data_pointer = .false.
                if (allocated(decl%var_name)) then
                    if (trim(decl%var_name) == result_name) is_data_pointer = .true.
                end if
                if (decl%is_multi_declaration .and. allocated(decl%var_names)) then
                    do k = 1, size(decl%var_names)
                        if (trim(decl%var_names(k)) == result_name) &
                            is_data_pointer = .true.
                    end do
                end if
                if (.not. is_data_pointer) cycle
                call resolve_result_decl_kind(decl, context, pointee_kind, &
                    derived_index, is_array, decl_error, node%body_indices(i))
                if (len_trim(decl_error) > 0 .or. is_array .or. &
                    derived_index > 0) then
                    is_data_pointer = .false.
                    pointee_kind = VALUE_I32
                    cycle
                end if
                return
            end select
        end do
    end function result_var_is_data_pointer

    logical function declaration_is_fixed_rank1(decl) result(is_fixed)
        ! A rank-1 explicit-shape array declaration with neither allocatable nor
        ! assumed-shape (no colon) dimensions.
        type(declaration_node), intent(in) :: decl

        is_fixed = .false.
        if (.not. decl%is_array) return
        if (decl%is_allocatable) return
        if (.not. allocated(decl%dimension_indices)) return
        if (size(decl%dimension_indices) /= 1) return
        is_fixed = .true.
    end function declaration_is_fixed_rank1

    logical function result_var_is_alloc_array(node, context) result(is_alloc)
        ! True when the function result variable is declared as an allocatable
        ! rank-1 or rank-2 array of an intrinsic element kind in the body.
        type(function_def_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        integer :: elem_kind, rank

        call alloc_array_result_info(node, context, is_alloc, elem_kind, rank)
    end function result_var_is_alloc_array

    subroutine alloc_array_result_info(node, context, is_alloc, elem_kind, rank)
        ! Element scalar kind and rank of an allocatable array function result,
        ! read from the result variable's body declaration. is_alloc is false for
        ! non-allocatable, derived-element, or higher-rank results (out of scope).
        type(function_def_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        logical, intent(out) :: is_alloc
        integer, intent(out) :: elem_kind
        integer, intent(out) :: rank
        character(len=:), allocatable :: result_name, decl_err
        integer :: i, k

        is_alloc = .false.
        elem_kind = VALUE_I32
        rank = 0
        result_name = trim(node%name)
        if (allocated(node%result_variable)) then
            if (len_trim(node%result_variable) > 0) &
                result_name = trim(node%result_variable)
        end if
        if (.not. allocated(node%body_indices)) return
        do i = 1, size(node%body_indices)
            if (.not. node_exists(context%arena, node%body_indices(i))) cycle
            select type (decl => context%arena%entries(node%body_indices(i))%node)
            type is (declaration_node)
                if (allocated(decl%var_name)) then
                        if (trim(decl%var_name) == result_name) then
                            call classify_alloc_array_decl(decl, context, is_alloc, &
                                                       elem_kind, rank, decl_err, &
                                                       node%body_indices(i))
                        if (is_alloc) return
                    end if
                end if
                if (decl%is_multi_declaration .and. allocated(decl%var_names)) then
                    do k = 1, size(decl%var_names)
                        if (trim(decl%var_names(k)) == result_name) then
                            call classify_alloc_array_decl(decl, context, is_alloc, &
                                                           elem_kind, rank, decl_err, &
                                                           node%body_indices(i))
                            if (is_alloc) return
                        end if
                    end do
                end if
            end select
        end do
    end subroutine alloc_array_result_info

    subroutine classify_alloc_array_decl(decl, context, is_alloc, elem_kind, rank, &
                                         decl_err, reference_index)
        ! An allocatable rank-1/2 declaration of an intrinsic element kind.
        type(declaration_node), intent(in) :: decl
        type(lowering_context_t), intent(in) :: context
        logical, intent(out) :: is_alloc
        integer, intent(out) :: elem_kind
        integer, intent(out) :: rank
        character(len=:), allocatable, intent(out) :: decl_err
        integer, intent(in), optional :: reference_index

        is_alloc = .false.
        elem_kind = VALUE_I32
        rank = 0
        call set_empty(decl_err)
        if (.not. decl%is_array) return
        if (.not. decl%is_allocatable) return
        if (declaration_derived_type_index(context, decl) > 0) return
        if (allocated(decl%dimension_indices)) then
            rank = size(decl%dimension_indices)
        else
            rank = 1
        end if
        if (rank < 1 .or. rank > 2) then
            rank = 0
            return
        end if
        call declaration_value_kind(decl, elem_kind, decl_err, context, &
                                    reference_index)
        if (len_trim(decl_err) > 0) then
            rank = 0
            return
        end if
        select case (elem_kind)
        case (VALUE_I32, VALUE_I64, VALUE_F32, VALUE_F64, VALUE_LOGICAL)
            is_alloc = .true.
        case default
            rank = 0
        end select
    end subroutine classify_alloc_array_decl

    integer function result_var_derived_type(node, context) result(type_index)
        ! Find the derived type of the function result variable from its body
        ! declaration (result(p) with type(t) :: p inside the body).
        type(function_def_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        character(len=:), allocatable :: result_name
        integer :: i, k

        type_index = 0
        result_name = trim(node%name)
        if (allocated(node%result_variable)) then
            if (len_trim(node%result_variable) > 0) &
                result_name = trim(node%result_variable)
        end if
        if (.not. allocated(node%body_indices)) return
        do i = 1, size(node%body_indices)
            if (.not. node_exists(context%arena, node%body_indices(i))) cycle
            select type (decl => context%arena%entries(node%body_indices(i))%node)
            type is (declaration_node)
                if (allocated(decl%var_name)) then
                    if (trim(decl%var_name) == result_name) then
                        type_index = declaration_derived_type_index(context, decl)
                        if (type_index > 0) return
                    end if
                end if
                if (decl%is_multi_declaration .and. allocated(decl%var_names)) then
                    do k = 1, size(decl%var_names)
                        if (trim(decl%var_names(k)) == result_name) then
                            type_index = declaration_derived_type_index(context, &
                                                                        decl)
                            if (type_index > 0) return
                        end if
                    end do
                end if
            end select
        end do
    end function result_var_derived_type

    subroutine copy_internal_function_names(source, destination)
        type(lowering_context_t), intent(in) :: source
        type(lowering_context_t), intent(inout) :: destination
        integer :: i

        if (.not. allocated(destination%function_names)) then
            allocate(destination%function_names(8))
        end if
        if (.not. allocated(destination%function_value_kinds)) then
            allocate(destination%function_value_kinds(8))
        end if
        if (.not. allocated(destination%function_param_counts)) then
            allocate(destination%function_param_counts(8))
            destination%function_param_counts = 0
        end if
        if (.not. allocated(destination%function_node_indices)) then
            allocate(destination%function_node_indices(8))
            destination%function_node_indices = 0
        end if
        do i = 1, source%function_count
            call ensure_function_names_capacity(destination, i)
            destination%function_names(i) = source%function_names(i)
            destination%function_value_kinds(i) = source%function_value_kinds(i)
            destination%function_param_counts(i) = source%function_param_counts(i)
            destination%function_node_indices(i) = source%function_node_indices(i)
        end do
        destination%function_count = source%function_count
    end subroutine copy_internal_function_names

    subroutine copy_derived_type_table(source, destination)
        type(lowering_context_t), intent(in) :: source
        type(lowering_context_t), intent(inout) :: destination
        integer :: i

        do i = 1, source%derived_type_count
            call ensure_derived_types_capacity(destination, i)
            destination%derived_types(i) = source%derived_types(i)
        end do
        destination%derived_type_count = source%derived_type_count
    end subroutine copy_derived_type_table

    subroutine copy_module_exports(source, destination)
        ! Carry the module export table into a procedure's fresh context so its
        ! body can host-associate scalar module variables (#263).
        type(lowering_context_t), intent(in) :: source
        type(lowering_context_t), intent(inout) :: destination
        integer :: i

        do i = 1, source%module_export_count
            call grow_module_exports(destination)
            destination%module_exports(i) = source%module_exports(i)
            destination%module_export_count = i
        end do
    end subroutine copy_module_exports

    subroutine copy_declaration_registry(source, destination)
        type(lowering_context_t), intent(in) :: source
        type(lowering_context_t), intent(inout) :: destination

        if (allocated(destination%declaration_records)) then
            deallocate(destination%declaration_records)
        end if
        if (allocated(source%declaration_records)) then
            allocate(destination%declaration_records( &
                size(source%declaration_records)))
            destination%declaration_records = source%declaration_records
        end if
        destination%declaration_record_count = source%declaration_record_count
        destination%declaration_collection_complete = &
            source%declaration_collection_complete
    end subroutine copy_declaration_registry

    subroutine copy_host_global_symbols(source, destination)
        type(lowering_context_t), intent(in) :: source
        type(lowering_context_t), intent(inout) :: destination
        integer :: i
        type(declaration_binding_t) :: binding

        do i = 1, source%symbol_count
            if (.not. symbol_has_global_storage(source%symbols(i))) cycle
            if (source%symbols(i)%has_binding) then
                binding%found = .true.
                binding%name = trim(source%symbols(i)%name)
                binding%declaration_node_index = &
                    source%symbols(i)%binding_declaration_index
                binding%declaration_entity_index = &
                    source%symbols(i)%binding_entity_index
                binding%scope_node_index = source%symbols(i)%binding_scope_index
                if (find_symbol_for_binding(destination, binding) > 0) cycle
            else if (find_symbol_compat(destination, trim(source%symbols(i)%name)) > 0) then
                cycle
            end if
            call grow_symbols(destination)
            destination%symbol_count = destination%symbol_count + 1
            destination%symbols(destination%symbol_count) = source%symbols(i)
            if (source%symbols(i)%has_binding) then
                call destination%binding_table%insert_binding( &
                    source%symbols(i)%binding_declaration_index, &
                    source%symbols(i)%binding_entity_index, &
                    source%symbols(i)%binding_scope_index, &
                    destination%symbol_count)
            end if
        end do
    end subroutine copy_host_global_symbols

    subroutine collect_host_associated_symbols(arena, context, error_msg)
        ! Host variables are referenced by flat LIRIC procedures, so their
        ! storage must exist before either the host body or the contained body
        ! is emitted. The FortFront binding triple is the only identity used;
        ! text names are retained solely for diagnostics (#330).
        type(ast_arena_t), intent(in) :: arena
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(declaration_binding_t) :: binding
        type(mono_type_t), allocatable :: inferred
        character(len=:), allocatable :: name, resolve_error, kind_error
        logical :: found
        integer :: i, value_kind, result_derived_index, procedure_index
        integer :: declaration_index

        call set_empty(error_msg)
        do i = 1, arena%size
            if (.not. node_exists(arena, i)) cycle
            if (.not. is_identifier(arena, i)) cycle
            procedure_index = containing_procedure_node(arena, i)
            if (procedure_index <= 0) cycle
            if (.not. procedure_is_internal(arena, procedure_index)) cycle
            call get_identifier_name(arena, i, name, resolve_error)
            if (len_trim(resolve_error) > 0) cycle
            call resolve_name_at_node(arena, i, name, binding, resolve_error)
            if (len_trim(resolve_error) > 0 .or. .not. binding%found) cycle
            if (binding%association /= ASSOCIATION_HOST) cycle
            ! An entity declared by the internal procedure itself is local, not
            ! host-associated, however the reference is labelled. Nested
            ! ASSOCIATE constructs can report a local dummy as host-associated;
            ! giving such a name host storage would shadow the genuine host
            ! entity of the same spelling and reject its declaration (#584).
            if (binding%scope_node_index == procedure_index) cycle
            if (binding_scope_is_module(arena, binding%scope_node_index)) cycle
            if (binding%binding_kind /= BINDING_DECLARATION .and. &
                binding%binding_kind /= BINDING_FUNCTION_RESULT) cycle
            declaration_index = find_declaration_record(context, binding)
            if (declaration_index > 0 .and. &
                    context%declaration_records(declaration_index)%is_array) then
                call register_host_array_global_symbol(context, binding, name, &
                    context%declaration_records(declaration_index), error_msg)
                if (len_trim(error_msg) > 0) return
                cycle
            end if
            if (binding%declaration_node_index > 0 .and. &
                    node_exists(arena, binding%declaration_node_index)) then
                select type (decl => &
                        arena%entries(binding%declaration_node_index)%node)
                type is (declaration_node)
                    if (.not. decl%is_array .and. decl%is_allocatable .and. &
                            allocated(decl%type_name) .and. &
                            is_class_derived_type_spec(decl%type_name)) then
                        call register_host_class_allocatable_global(context, &
                            binding, name, decl, error_msg)
                        if (len_trim(error_msg) > 0) return
                        cycle
                    end if
                class default
                    continue
                end select
            end if
            if (find_symbol_for_binding(context, binding) > 0) cycle
            value_kind = 0
            call set_empty(kind_error)
            if (binding%binding_kind == BINDING_FUNCTION_RESULT) then
                if (binding%scope_node_index > 0 .and. &
                    node_exists(arena, binding%scope_node_index)) then
                    select type (fn => &
                            arena%entries(binding%scope_node_index)%node)
                    type is (function_def_node)
                        call function_value_kind(fn, context, value_kind, &
                            result_derived_index, kind_error, &
                            binding%scope_node_index)
                    class default
                        value_kind = 0
                    end select
                end if
            else
                call get_type_for_node(arena, i, inferred, found)
                if (found .and. allocated(inferred)) &
                    value_kind = inferred_type_to_value_kind(inferred, context)
            end if
            if (value_kind <= 0 .and. binding%binding_kind == &
                    BINDING_DECLARATION .and. binding%declaration_node_index > 0) then
                if (node_exists(arena, binding%declaration_node_index)) then
                    select type (decl => &
                            arena%entries(binding%declaration_node_index)%node)
                    type is (declaration_node)
                        call declaration_value_kind(decl, value_kind, kind_error, &
                            context, binding%declaration_node_index)
                        if (len_trim(kind_error) > 0) value_kind = 0
                    class default
                        value_kind = 0
                    end select
                end if
            end if
            if (value_kind <= 0) cycle
            select case (value_kind)
            case (VALUE_I32, VALUE_I64, VALUE_F32, VALUE_F64, VALUE_LOGICAL, &
                  VALUE_CHARACTER, VALUE_DEFERRED_CHARACTER_RESULT)
                if (value_kind == VALUE_DEFERRED_CHARACTER_RESULT) &
                    value_kind = VALUE_CHARACTER
                call register_host_global_symbol(context, binding, name, &
                                                  value_kind, error_msg)
                if (len_trim(error_msg) > 0) return
            case default
                ! The current direct-session storage ABI has no scalar global
                ! representation for this host entity. Leave it unresolved so
                ! the existing typed diagnostic remains authoritative.
            end select
        end do
        ! Treat pre-registered host storage as an enclosing scope. A declaration
        ! with the same spelling but a different FortFront identity must append a
        ! local symbol instead of colliding with this pre-pass storage.
        context%block_scope_floor = context%symbol_count
    end subroutine collect_host_associated_symbols

    subroutine register_host_global_symbol(context, binding, name, value_kind, &
                                           error_msg)
        use, intrinsic :: iso_c_binding, only: c_bool, c_char, c_double, c_float, &
            c_int8_t, c_int32_t, c_int64_t, c_size_t, c_loc, c_ptr
        type(lowering_context_t), intent(inout) :: context
        type(declaration_binding_t), intent(in) :: binding
        character(len=*), intent(in) :: name
        integer, intent(in) :: value_kind
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: global_name
        character(kind=c_char), allocatable :: c_name(:)
        integer(c_int32_t), target :: init_i32
        integer(c_int8_t), target :: init_i8(32)
        integer(c_int64_t), target :: init_i64
        real(c_float), target :: init_f32
        real(c_double), target :: init_f64
        type(c_ptr) :: init_ptr, global_type
        integer(c_size_t) :: init_size
        integer(c_int32_t) :: global_id
        logical(c_bool) :: c_false_local
        integer :: index

        call set_empty(error_msg)
        global_name = host_global_name(binding, name)
        call to_c_chars(global_name, c_name)
        init_i32 = 0_c_int32_t
        init_i8 = 0_c_int8_t
        init_i64 = 0_c_int64_t
        init_f32 = 0.0_c_float
        init_f64 = 0.0_c_double
        c_false_local = .false.
        select case (value_kind)
        case (VALUE_I32, VALUE_LOGICAL)
            global_type = lr_type_i32_s(context%session%handle)
            init_ptr = c_loc(init_i32)
            init_size = 4_c_size_t
        case (VALUE_I64)
            global_type = lr_type_i64_s(context%session%handle)
            init_ptr = c_loc(init_i64)
            init_size = 8_c_size_t
        case (VALUE_CHARACTER)
            ! A host-associated character variable or result carries the
            ! canonical 32-byte character descriptor, so its ownership half
            ! travels with it (docs/CHARACTER_DESCRIPTOR_ABI.md).
            global_type = lr_type_array_s(context%session%handle, &
                lr_type_i8_s(context%session%handle), 32_c_int64_t)
            init_ptr = c_loc(init_i8)
            init_size = 32_c_size_t
        case (VALUE_F32)
            global_type = lr_type_f32_s(context%session%handle)
            init_ptr = c_loc(init_f32)
            init_size = 4_c_size_t
        case (VALUE_F64)
            global_type = lr_type_f64_s(context%session%handle)
            init_ptr = c_loc(init_f64)
            init_size = 8_c_size_t
        case default
            error_msg = 'unsupported host-associated scalar: '//trim(name)
            return
        end select
        global_id = lr_session_global(context%session%handle, c_name, global_type, &
            c_false_local, init_ptr, init_size)
        if (global_id < 0_c_int32_t) then
            error_msg = 'LIRIC could not create host-associated global: '// &
                trim(global_name)
            return
        end if
        global_id = lr_session_intern(context%session%handle, c_name)
        if (global_id < 0_c_int32_t) then
            error_msg = 'LIRIC could not intern host-associated global: '// &
                trim(global_name)
            return
        end if

        call grow_symbols(context)
        index = context%symbol_count + 1
        context%symbols(index) = symbol_t()
        context%symbols(index)%name = trim(name)
        context%symbols(index)%value_kind = value_kind
        context%symbols(index)%has_address = .true.
        context%symbols(index)%is_reference = .true.
        context%symbols(index)%address%kind = LR_OP_KIND_GLOBAL
        context%symbols(index)%address%payload = int(global_id, c_int64_t)
        context%symbols(index)%address%typ = lr_type_ptr_s(context%session%handle)
        context%symbols(index)%address%global_offset = 0_c_int64_t
        select case (value_kind)
        case (VALUE_I32, VALUE_LOGICAL)
            context%symbols(index)%value = i32_immediate(context%session, 0_c_int64_t)
        case (VALUE_I64)
            context%symbols(index)%value = i64_immediate(context%session, 0_c_int64_t)
        case (VALUE_F32)
            context%symbols(index)%value = liric_f32_immediate(context%session, &
                0.0_c_float)
        case (VALUE_F64)
            context%symbols(index)%value = liric_f64_immediate(context%session, &
                0.0_c_double)
        case (VALUE_CHARACTER)
            context%symbols(index)%is_deferred_character = .true.
            context%symbols(index)%has_character_value = .false.
        end select
        context%symbols(index)%has_binding = .true.
        context%symbols(index)%binding_declaration_index = &
            binding%declaration_node_index
        context%symbols(index)%binding_entity_index = &
            binding%declaration_entity_index
        context%symbols(index)%binding_scope_index = binding%scope_node_index
        context%symbol_count = index
        call context%binding_table%insert_binding( &
            binding%declaration_node_index, binding%declaration_entity_index, &
            binding%scope_node_index, index)
    end subroutine register_host_global_symbol

    subroutine register_host_class_allocatable_global(context, binding, name, &
                                                       decl, error_msg)
        use, intrinsic :: iso_c_binding, only: c_bool, c_char, c_int32_t, &
            c_int64_t, c_loc, c_ptr, c_size_t
        ! A host-associated CLASS(t), ALLOCATABLE scalar owns the same 32-byte
        ! descriptor as a local class allocatable. Materialize that descriptor
        ! as a global before contained procedures are emitted; the host body's
        ! ALLOCATE then publishes its data pointer and dynamic type into it.
        type(lowering_context_t), intent(inout) :: context
        type(declaration_binding_t), intent(in) :: binding
        character(len=*), intent(in) :: name
        type(declaration_node), intent(in) :: decl
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: global_name
        character(kind=c_char), allocatable :: c_name(:)
        integer(c_int64_t), target :: init_words(POLYMORPHIC_DESCRIPTOR_SIZE / 8)
        type(c_ptr) :: global_type
        type(lr_operand_desc_t) :: descriptor
        integer(c_int32_t) :: global_id
        integer :: declared_index, index
        logical(c_bool) :: c_false_local

        call set_empty(error_msg)
        if (find_symbol_for_binding(context, binding) > 0) return
        declared_index = declaration_derived_type_index(context, decl)
        if (declared_index <= 0) return

        init_words = 0_c_int64_t
        init_words(2) = int(declared_index, c_int64_t)
        global_name = host_global_name(binding, name)
        call to_c_chars(global_name, c_name)
        global_type = lr_type_array_s(context%session%handle, &
            lr_type_i8_s(context%session%handle), &
            int(POLYMORPHIC_DESCRIPTOR_SIZE, c_int64_t))
        c_false_local = .false.
        global_id = lr_session_global(context%session%handle, c_name, global_type, &
            c_false_local, c_loc(init_words), &
            int(POLYMORPHIC_DESCRIPTOR_SIZE, c_size_t))
        if (global_id < 0_c_int32_t) then
            error_msg = 'LIRIC could not create host-associated class descriptor: '// &
                trim(name)
            return
        end if
        global_id = lr_session_intern(context%session%handle, c_name)
        if (global_id < 0_c_int32_t) then
            error_msg = 'LIRIC could not intern host-associated class descriptor: '// &
                trim(name)
            return
        end if

        descriptor = lr_operand_desc_t()
        descriptor%kind = LR_OP_KIND_GLOBAL
        descriptor%payload = int(global_id, c_int64_t)
        descriptor%typ = lr_type_ptr_s(context%session%handle)
        descriptor%global_offset = 0_c_int64_t

        call grow_symbols(context)
        index = context%symbol_count + 1
        context%symbols(index) = symbol_t()
        context%symbols(index)%name = trim(name)
        context%symbols(index)%value_kind = VALUE_DERIVED
        context%symbols(index)%is_derived = .true.
        context%symbols(index)%derived_type_index = declared_index
        context%symbols(index)%is_allocatable = .true.
        context%symbols(index)%is_polymorphic = .true.
        context%symbols(index)%class_descriptor_address = descriptor
        context%symbols(index)%dynamic_type_address = descriptor
        context%symbols(index)%dynamic_type_address%global_offset = &
            int(POLYMORPHIC_DESCRIPTOR_DYNAMIC_TYPE_OFFSET, c_int64_t)
        context%symbols(index)%has_dynamic_type_address = .true.
        context%symbol_count = index
        call attach_symbol_binding(context, index, binding)
    end subroutine register_host_class_allocatable_global

    subroutine register_host_array_global_symbol(context, binding, name, record, &
                                                 error_msg)
        ! A contained procedure can use a fixed-size host array before the host
        ! procedure's body is lowered. Give that binding one static global, just
        ! as module arrays receive, so the later host declaration reuses the
        ! identity instead of allocating a second array.
        use, intrinsic :: iso_c_binding, only: c_bool, c_char, c_double, c_float, &
            c_int32_t, c_int64_t, c_loc, c_ptr, c_size_t
        type(lowering_context_t), intent(inout) :: context
        type(declaration_binding_t), intent(in) :: binding
        character(len=*), intent(in) :: name
        type(declaration_record_t), intent(in) :: record
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: global_name
        character(kind=c_char), allocatable :: c_name(:)
        integer(c_int32_t), target, allocatable :: array_i32(:)
        real(c_float), target, allocatable :: array_f32(:)
        real(c_double), target, allocatable :: array_f64(:)
        type(c_ptr) :: elem_type, global_type, init_ptr
        integer(c_size_t) :: init_size
        integer(c_int32_t) :: global_id
        integer :: array_lower, array_size, index
        logical(c_bool) :: c_false_local

        call set_empty(error_msg)
        if (find_symbol_for_binding(context, binding) > 0) return
        if (record%rank /= 1) return
        if (.not. node_exists(context%arena, binding%declaration_node_index)) return
        if (.not. host_array_scope_is_program(context%arena, &
                                              binding%scope_node_index)) return
        select type (decl => &
                context%arena%entries(binding%declaration_node_index)%node)
        type is (declaration_node)
            if (decl%is_allocatable .or. decl%is_pointer .or. decl%is_target) return
            call get_array_bounds(decl, context, array_lower, array_size, error_msg)
        class default
            return
        end select
        if (len_trim(error_msg) > 0) return
        if (array_size <= 0) return

        c_false_local = .false.
        select case (record%value_kind)
        case (VALUE_I32, VALUE_LOGICAL)
            allocate (array_i32(array_size))
            array_i32 = 0_c_int32_t
            elem_type = lr_type_i32_s(context%session%handle)
            init_ptr = c_loc(array_i32)
            init_size = int(array_size, c_size_t) * 4_c_size_t
        case (VALUE_F32)
            allocate (array_f32(array_size))
            array_f32 = 0.0_c_float
            elem_type = lr_type_f32_s(context%session%handle)
            init_ptr = c_loc(array_f32)
            init_size = int(array_size, c_size_t) * 4_c_size_t
        case (VALUE_F64)
            allocate (array_f64(array_size))
            array_f64 = 0.0_c_double
            elem_type = lr_type_f64_s(context%session%handle)
            init_ptr = c_loc(array_f64)
            init_size = int(array_size, c_size_t) * 8_c_size_t
        case default
            return
        end select
        global_name = host_global_name(binding, name)
        call to_c_chars(global_name, c_name)
        global_type = lr_type_array_s(context%session%handle, elem_type, &
                                      int(array_size, c_int64_t))
        global_id = lr_session_global(context%session%handle, c_name, global_type, &
                                      c_false_local, init_ptr, init_size)
        if (global_id < 0_c_int32_t) then
            error_msg = 'LIRIC could not create host-associated array: '//trim(name)
            return
        end if
        global_id = lr_session_intern(context%session%handle, c_name)
        if (global_id < 0_c_int32_t) then
            error_msg = 'LIRIC could not intern host-associated array: '//trim(name)
            return
        end if

        call grow_symbols(context)
        index = context%symbol_count + 1
        context%symbols(index) = symbol_t()
        context%symbols(index)%name = trim(name)
        context%symbols(index)%value_kind = record%value_kind
        context%symbols(index)%has_address = .true.
        context%symbols(index)%is_reference = .true.
        context%symbols(index)%is_array = .true.
        context%symbols(index)%array_rank = 1
        context%symbols(index)%array_size = array_size
        context%symbols(index)%array_lower_bound = array_lower
        context%symbols(index)%array_dim_sizes(1) = array_size
        context%symbols(index)%array_dim_lowers(1) = array_lower
        context%symbols(index)%address%kind = LR_OP_KIND_GLOBAL
        context%symbols(index)%address%payload = int(global_id, c_int64_t)
        context%symbols(index)%address%typ = lr_type_ptr_s(context%session%handle)
        context%symbols(index)%address%global_offset = 0_c_int64_t
        context%symbols(index)%element_address = context%symbols(index)%address
        context%symbol_count = index
        call attach_symbol_binding(context, index, binding)
        call set_empty(error_msg)
    end subroutine register_host_array_global_symbol

    logical function host_array_scope_is_program(arena, scope_node_index) &
        result(is_program_scope)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: scope_node_index

        is_program_scope = .false.
        if (.not. node_exists(arena, scope_node_index)) return
        select type (scope => arena%entries(scope_node_index)%node)
        type is (program_node)
            is_program_scope = .true.
        end select
    end function host_array_scope_is_program

    subroutine materialize_host_character_globals(context, error_msg)
        ! Host-associated character results and variables use a mutable global
        ! {data, length} descriptor. Its field addresses are function-local LIRIC
        ! operands, so materialise them only after a function has begun.
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i

        call set_empty(error_msg)
        do i = 1, context%symbol_count
            if (context%symbols(i)%value_kind /= VALUE_CHARACTER) cycle
            if (.not. context%symbols(i)%is_deferred_character) cycle
            if (context%symbols(i)%address%kind /= LR_OP_KIND_GLOBAL) cycle
            if (context%symbols(i)%deferred_data%kind /= LR_OP_KIND_IMM_I64) cycle
            if (.not. emit_ptr_offset(context%session, &
                context%symbols(i)%address, 0_c_int64_t, &
                context%symbols(i)%deferred_data, error_msg)) return
            if (.not. emit_ptr_offset(context%session, &
                context%symbols(i)%address, 8_c_int64_t, &
                context%symbols(i)%deferred_length, error_msg)) return
            if (.not. emit_ptr_offset(context%session, &
                context%symbols(i)%address, 16_c_int64_t, &
                context%symbols(i)%deferred_capacity, error_msg)) return
            if (.not. emit_ptr_offset(context%session, &
                context%symbols(i)%address, 24_c_int64_t, &
                context%symbols(i)%deferred_storage, error_msg)) return
            context%symbols(i)%has_character_ownership = .true.
        end do
        call set_empty(error_msg)
    end subroutine materialize_host_character_globals

    subroutine emit_host_character_result_return(context, symbol_index, error_msg)
        ! Copy a host-result global descriptor into the caller's hidden sret
        ! descriptor before returning from the flattened contained function.
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: result_data_addr, result_length_addr
        type(lr_operand_desc_t) :: result_capacity_addr, result_storage_addr
        type(lr_operand_desc_t) :: data_value, length_value
        type(lr_operand_desc_t) :: capacity_value, storage_value

        call set_empty(error_msg)
        if (.not. emit_ptr_offset(context%session, ptr_param(context%session, 0), &
            0_c_int64_t, result_data_addr, error_msg)) return
        if (.not. emit_ptr_offset(context%session, ptr_param(context%session, 0), &
            8_c_int64_t, result_length_addr, error_msg)) return
        if (.not. emit_ptr_offset(context%session, ptr_param(context%session, 0), &
            16_c_int64_t, result_capacity_addr, error_msg)) return
        if (.not. emit_ptr_offset(context%session, ptr_param(context%session, 0), &
            24_c_int64_t, result_storage_addr, error_msg)) return
        if (.not. emit_ptr_load(context%session, &
            context%symbols(symbol_index)%deferred_data, data_value, error_msg)) return
        if (.not. emit_i64_load(context%session, &
            context%symbols(symbol_index)%deferred_length, length_value, error_msg)) return
        if (.not. emit_i64_load(context%session, &
            context%symbols(symbol_index)%deferred_capacity, capacity_value, &
            error_msg)) return
        if (.not. emit_i64_load(context%session, &
            context%symbols(symbol_index)%deferred_storage, storage_value, &
            error_msg)) return
        if (.not. emit_ptr_store(context%session, data_value, result_data_addr, &
            error_msg)) return
        if (.not. emit_i64_store(context%session, length_value, result_length_addr, &
            error_msg)) return
        if (.not. emit_i64_store(context%session, capacity_value, &
            result_capacity_addr, error_msg)) return
        ! Returning the descriptor transfers ownership to the caller: the class
        ! travels with the data, and the callee's own descriptor drops back to
        ! the null class so the block is never released twice. The result
        ! global is reused by the next call, which reinitialises it anyway.
        if (.not. emit_i64_store(context%session, storage_value, &
            result_storage_addr, error_msg)) return
        if (.not. emit_i64_store(context%session, &
            i64_immediate(context%session, 0_c_int64_t), &
            context%symbols(symbol_index)%deferred_storage, error_msg)) return
        if (.not. emit_ret_void(context%session, error_msg)) return
        call set_empty(error_msg)
    end subroutine emit_host_character_result_return

    subroutine load_function_result_value(context, symbol_index, value, error_msg)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg

        call set_empty(error_msg)
        if (.not. context%symbols(symbol_index)%has_address) then
            value = context%symbols(symbol_index)%value
            return
        end if
        select case (context%symbols(symbol_index)%value_kind)
        case (VALUE_I64)
            if (.not. emit_i64_load(context%session, &
                context%symbols(symbol_index)%address, value, error_msg)) return
        case (VALUE_F32)
            if (.not. emit_liric_f32_load(context%session, &
                context%symbols(symbol_index)%address, value, error_msg)) return
        case (VALUE_F64)
            if (.not. emit_liric_f64_load(context%session, &
                context%symbols(symbol_index)%address, value, error_msg)) return
        case default
            if (.not. emit_i32_load(context%session, &
                context%symbols(symbol_index)%address, value, error_msg)) return
        end select
        call set_empty(error_msg)
    end subroutine load_function_result_value

    subroutine resolve_function_result_binding(arena, procedure_node_index, name, &
                                               binding)
        ! The result name can be represented by the body declaration rather than
        ! the synthetic function-result binding (for example, an explicit
        ! `character(4) :: f`). Find the identity FortFront assigns to an actual
        ! reference in this procedure before falling back to the synthetic one.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: procedure_node_index
        character(len=*), intent(in) :: name
        type(declaration_binding_t), intent(out) :: binding
        type(declaration_binding_t) :: candidate
        character(len=:), allocatable :: identifier_name, resolve_error
        integer :: i, parent

        binding%found = .false.
        binding%declaration_node_index = 0
        binding%declaration_entity_index = 0
        binding%scope_node_index = 0
        binding%binding_kind = 0
        binding%association = 0
        do i = 1, arena%size
            if (.not. node_exists(arena, i)) cycle
            if (.not. is_identifier(arena, i)) cycle
            call get_identifier_name(arena, i, identifier_name, resolve_error)
            if (len_trim(resolve_error) > 0 .or. &
                .not. same_name(identifier_name, name)) cycle
            parent = i
            do while (parent > 0 .and. parent /= procedure_node_index)
                if (.not. node_exists(arena, parent)) exit
                parent = arena%entries(parent)%parent_index
            end do
            if (parent /= procedure_node_index) cycle
            call resolve_name_at_node(arena, i, name, candidate, resolve_error)
            if (len_trim(resolve_error) > 0 .or. .not. candidate%found) cycle
            if (candidate%scope_node_index /= procedure_node_index) cycle
            if (candidate%binding_kind /= BINDING_DECLARATION .and. &
                candidate%binding_kind /= BINDING_FUNCTION_RESULT) cycle
            binding = candidate
            return
        end do

        ! No reference was available, but the synthetic function result still
        ! has a stable FortFront identity.
        binding%found = .true.
        binding%declaration_node_index = procedure_node_index
        binding%declaration_entity_index = 1
        binding%scope_node_index = procedure_node_index
        binding%binding_kind = BINDING_FUNCTION_RESULT
    end subroutine resolve_function_result_binding

    function host_global_name(binding, name) result(global_name)
        type(declaration_binding_t), intent(in) :: binding
        character(len=*), intent(in) :: name
        character(len=:), allocatable :: global_name
        character(len=32) :: declaration_text, entity_text, scope_text

        write (declaration_text, '(I0)') binding%declaration_node_index
        write (entity_text, '(I0)') binding%declaration_entity_index
        write (scope_text, '(I0)') binding%scope_node_index
        global_name = '.ffc.host.'//trim(declaration_text)//'.'// &
            trim(entity_text)//'.'//trim(scope_text)//'.'//trim(name)
    end function host_global_name

    logical function symbol_has_global_storage(symbol)
        type(symbol_t), intent(in) :: symbol

        symbol_has_global_storage = &
            symbol%address%kind == LR_OP_KIND_GLOBAL .or. &
            symbol%element_address%kind == LR_OP_KIND_GLOBAL .or. &
            symbol%allocatable_descriptor_address%kind == LR_OP_KIND_GLOBAL .or. &
            symbol%class_descriptor_address%kind == LR_OP_KIND_GLOBAL
    end function symbol_has_global_storage

    logical function is_contained_i32_function(context, name)
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: name

        is_contained_i32_function = &
            contained_function_kind(context, name) == VALUE_I32
    end function is_contained_i32_function

    logical function is_contained_f32_function(context, name)
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: name

        is_contained_f32_function = &
            contained_function_kind(context, name) == VALUE_F32
    end function is_contained_f32_function

    logical function is_contained_f64_function(context, name)
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: name

        is_contained_f64_function = &
            contained_function_kind(context, name) == VALUE_F64
    end function is_contained_f64_function

    logical function is_contained_logical_function(context, name)
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: name

        is_contained_logical_function = &
            contained_function_kind(context, name) == VALUE_LOGICAL
    end function is_contained_logical_function

    logical function is_contained_derived_function(context, name)
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: name

        is_contained_derived_function = &
            contained_function_kind(context, name) == VALUE_DERIVED
    end function is_contained_derived_function

    logical function is_contained_array_function(context, name)
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: name

        is_contained_array_function = &
            contained_function_kind(context, name) == VALUE_ARRAY_RESULT
    end function is_contained_array_function

    ! Signature facts a contained procedure must satisfy before an elemental
    ! call over array actuals may be expanded element-by-element (#405):
    ! how many dummies it takes, whether its prefix says ELEMENTAL, and
    ! whether every dummy is declared INTENT(IN) as F2018 15.8.1 requires of
    ! an elemental function. found stays .false. when the procedure is not a
    ! contained function visible in the arena.
    subroutine contained_elemental_signature(arena, name, found, param_count, &
                                             is_elemental, all_intent_in, &
                                             bad_dummy)
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: name
        logical, intent(out) :: found
        integer, intent(out) :: param_count
        logical, intent(out) :: is_elemental
        logical, intent(out) :: all_intent_in
        character(len=:), allocatable, intent(out) :: bad_dummy
        character(len=:), allocatable :: dummy_name, err
        integer :: i, p

        found = .false.
        param_count = 0
        is_elemental = .false.
        all_intent_in = .true.
        bad_dummy = ''
        do i = 1, arena%size
            if (.not. allocated(arena%entries(i)%node)) cycle
            select type (fn => arena%entries(i)%node)
            type is (function_def_node)
                if (.not. allocated(fn%name)) cycle
                if (.not. same_name(fn%name, name)) cycle
                found = .true.
                if (allocated(fn%param_indices)) param_count = size(fn%param_indices)
                if (allocated(fn%prefix_keywords)) then
                    do p = 1, size(fn%prefix_keywords)
                        if (same_name(trim(fn%prefix_keywords(p)), 'elemental')) &
                            is_elemental = .true.
                    end do
                end if
                if (.not. allocated(fn%param_indices)) return
                if (.not. allocated(fn%body_indices)) return
                do p = 1, size(fn%param_indices)
                    call parameter_name(arena, fn%param_indices(p), dummy_name, err)
                    if (len_trim(err) > 0) cycle
                    if (dummy_is_intent_in(arena, fn%body_indices, dummy_name)) cycle
                    all_intent_in = .false.
                    if (len_trim(bad_dummy) == 0) bad_dummy = trim(dummy_name)
                end do
                return
            end select
        end do
    end subroutine contained_elemental_signature

    ! True when dummy_name is declared INTENT(IN) among the body declarations.
    logical function dummy_is_intent_in(arena, body_indices, dummy_name) &
        result(is_in)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: body_indices(:)
        character(len=*), intent(in) :: dummy_name
        integer :: i, k
        logical :: matches

        is_in = .false.
        do i = 1, size(body_indices)
            if (.not. node_exists(arena, body_indices(i))) cycle
            select type (decl => arena%entries(body_indices(i))%node)
            type is (declaration_node)
                matches = .false.
                if (allocated(decl%var_name)) then
                    if (same_name(trim(decl%var_name), dummy_name)) matches = .true.
                end if
                if (decl%is_multi_declaration .and. allocated(decl%var_names)) then
                    do k = 1, size(decl%var_names)
                        if (same_name(trim(decl%var_names(k)), dummy_name)) &
                            matches = .true.
                    end do
                end if
                if (.not. matches) cycle
                if (.not. decl%has_intent) return
                if (.not. allocated(decl%intent)) return
                is_in = trim(lowercase_text(decl%intent)) == 'in'
                return
            end select
        end do
    end function dummy_is_intent_in

    subroutine contained_array_result_info(arena, context, name, element_kind, &
                                           array_size, error_msg, character_length)
        ! Element scalar kind and total element count of a rank-1 array function
        ! result, read from the result variable's body declaration. Used by the
        ! print path that materialises the result into a scratch buffer.
        type(ast_arena_t), intent(in) :: arena
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: name
        integer, intent(out) :: element_kind
        integer, intent(out) :: array_size
        character(len=:), allocatable, intent(out) :: error_msg
        integer, intent(out), optional :: character_length
        type(function_def_node), pointer :: fn_node
        character(len=:), allocatable :: result_name
        integer :: i, k, node_index, lower_bound

        element_kind = VALUE_I32
        array_size = 0
        if (present(character_length)) character_length = 0
        call set_empty(error_msg)

        node_index = 0
        do i = 1, context%function_count
            if (same_name(context%function_names(i), name)) then
                node_index = context%function_node_indices(i)
                exit
            end if
        end do
        if (node_index <= 0 .or. .not. node_exists(arena, node_index)) then
            error_msg = 'array result function node not found: '//trim(name)
            return
        end if
        fn_node => get_node_as_function_def(arena, node_index)
        if (.not. associated(fn_node)) then
            error_msg = 'array result function node not found: '//trim(name)
            return
        end if

        result_name = trim(fn_node%name)
        if (allocated(fn_node%result_variable)) then
            if (len_trim(fn_node%result_variable) > 0) &
                result_name = trim(fn_node%result_variable)
        end if
        if (.not. allocated(fn_node%body_indices)) then
            error_msg = 'array result declaration not found: '//trim(name)
            return
        end if
        do i = 1, size(fn_node%body_indices)
            if (.not. node_exists(arena, fn_node%body_indices(i))) cycle
            select type (decl => arena%entries(fn_node%body_indices(i))%node)
            type is (declaration_node)
                if (.not. declaration_is_fixed_rank1(decl)) cycle
                if (allocated(decl%var_name)) then
                    if (trim(decl%var_name) == result_name) then
                        call declaration_value_kind(decl, element_kind, &
                                                    error_msg, context, &
                                                    fn_node%body_indices(i))
                        if (len_trim(error_msg) > 0) return
                        if (present(character_length) .and. &
                            element_kind == VALUE_CHARACTER) then
                            call declaration_character_length(context, decl, &
                                                             character_length, &
                                                             error_msg)
                            if (len_trim(error_msg) > 0) return
                        end if
                        call get_array_bounds(decl, context, lower_bound, &
                                              array_size, error_msg)
                        return
                    end if
                end if
                if (decl%is_multi_declaration .and. allocated(decl%var_names)) then
                    do k = 1, size(decl%var_names)
                        if (trim(decl%var_names(k)) == result_name) then
                            call declaration_value_kind(decl, element_kind, &
                                                        error_msg, context, &
                                                        fn_node%body_indices(i))
                            if (len_trim(error_msg) > 0) return
                            if (present(character_length) .and. &
                                element_kind == VALUE_CHARACTER) then
                                call declaration_character_length(context, decl, &
                                                                 character_length, &
                                                                 error_msg)
                                if (len_trim(error_msg) > 0) return
                            end if
                            call get_array_bounds(decl, context, lower_bound, &
                                                  array_size, error_msg)
                            return
                        end if
                    end do
                end if
            end select
        end do
        error_msg = 'array result declaration not found: '//trim(name)
    end subroutine contained_array_result_info

    integer function contained_function_kind(context, name) result(value_kind)
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: name
        integer :: i, nested_node_index

        value_kind = 0
        nested_node_index = visible_nested_procedure_node(context, name)
        if (nested_node_index > 0) then
            value_kind = procedure_value_kind(context, nested_node_index)
            return
        end if
        do i = 1, context%function_count
            if (same_name(context%function_names(i), name)) then
                value_kind = context%function_value_kinds(i)
                return
            end if
        end do
   end function contained_function_kind

    logical function is_contained_function_reference(node, context) result(is_ref)
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context

        is_ref = .false.
        if (node%base_expr_index > 0) return
        if (.not. allocated(node%name)) return
        is_ref = contained_function_kind(context, node%name) /= 0
    end function is_contained_function_reference

    logical function is_contained_deferred_char_function(context, name)
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: name

        is_contained_deferred_char_function = &
            contained_function_kind(context, name) == &
            VALUE_DEFERRED_CHARACTER_RESULT
    end function is_contained_deferred_char_function

    subroutine lower_internal_functions(arena, root_index, context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: root_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: ff_error, node_type
        integer, allocatable :: body_indices(:)
        integer :: i

        call set_empty(error_msg)
        node_type = get_node_type_at(arena, root_index)
        if (node_type == 'module_node' .or. node_type == 'module') return
        if (is_submodule_unit(node_type)) then
            call lower_one_submodule_procedures(arena, root_index, context, &
                                                error_msg)
            return
        end if
        if (node_type == 'block_data_node' .or. node_type == 'block_data') return
        call get_unit_body_indices(arena, root_index, body_indices, ff_error)
        if (len_trim(ff_error) > 0) then
            error_msg = ff_error
            return
        end if
        if (size(body_indices) == 0) return
        do i = 1, size(body_indices)
            node_type = get_node_type_at(arena, body_indices(i))
            if (node_type == 'function_def_node' .or. &
                node_type == 'function_def' .or. &
                node_type == 'subroutine_def_node' .or. &
                node_type == 'subroutine_def') then
                call lower_one_internal_function(arena, body_indices(i), context, &
                                                 error_msg)
                if (len_trim(error_msg) > 0) return
            else if (node_type == 'program_node' .or. &
                     node_type == 'program') then
                call lower_program_contained_functions(arena, body_indices(i), &
                                                       context, error_msg)
                if (len_trim(error_msg) > 0) return
            end if
        end do
    end subroutine lower_internal_functions

    subroutine lower_one_internal_function(arena, node_index, context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: node_type
        type(function_def_node), pointer :: fn_node
        type(subroutine_def_node), pointer :: sb_node

        call set_empty(error_msg)
        if (.not. node_exists(arena, node_index)) then
            error_msg = 'internal procedure index does not reference an AST node'
            return
        end if
        node_type = get_node_type_at(arena, node_index)
        if (node_type == 'function_def_node' .or. node_type == 'function_def') then
            fn_node => get_node_as_function_def(arena, node_index)
            if (.not. associated(fn_node)) then
                error_msg = 'internal procedure index does not reference a function node'
                return
            end if
            call lower_nested_internal_procedures(arena, fn_node%body_indices, &
                                                  context, error_msg)
            if (len_trim(error_msg) > 0) return
            call lower_scalar_function(arena, fn_node, context, error_msg, &
                                       procedure_node_index=node_index)
        else if (node_type == 'subroutine_def_node' .or. &
                 node_type == 'subroutine_def') then
            sb_node => get_node_as_subroutine_def(arena, node_index)
            if (.not. associated(sb_node)) then
                error_msg = 'internal procedure index does not reference a subroutine node'
                return
            end if
            call lower_nested_internal_procedures(arena, sb_node%body_indices, &
                                                  context, error_msg)
            if (len_trim(error_msg) > 0) return
            call lower_void_subroutine(arena, sb_node, context, error_msg, &
                                       procedure_node_index=node_index)
        end if
    end subroutine lower_one_internal_function

    subroutine lower_module_procedures(arena, root_index, context, error_msg)
        ! Emit module-defined procedures under their mangled symbol
        ! __<module>_MOD_<name> (#119, #263). Two unit shapes are handled: a
        ! program with embedded module(s), and a bare module root (module-only
        ! source compiled to an object or a no-main executable). A procedure with
        ! an unsupported signature leaves error_msg set so the unit stays xfail.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: root_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: ff_error, node_type
        integer, allocatable :: body_indices(:)
        integer :: i

        call set_empty(error_msg)
        node_type = get_node_type_at(arena, root_index)
        if (node_type == 'module_node' .or. node_type == 'module') then
            call lower_one_module_unit_procedures(arena, root_index, context, &
                                                  error_msg)
            return
        end if
        if (node_type == 'block_data_node' .or. node_type == 'block_data') return
        call get_unit_body_indices(arena, root_index, body_indices, ff_error)
        if (len_trim(ff_error) > 0) then
            error_msg = ff_error
            return
        end if
        if (size(body_indices) == 0) return
        do i = 1, size(body_indices)
            node_type = get_node_type_at(arena, body_indices(i))
            if (is_submodule_unit(node_type)) then
                call lower_one_submodule_procedures(arena, body_indices(i), &
                                                    context, error_msg)
                if (len_trim(error_msg) > 0) return
                cycle
            end if
            if (node_type /= 'module_node' .and. node_type /= 'module') cycle
            call lower_one_module_unit_procedures(arena, body_indices(i), &
                                                  context, error_msg)
            if (len_trim(error_msg) > 0) return
        end do
    end subroutine lower_module_procedures

    subroutine lower_one_module_unit_procedures(arena, module_index, context, &
                                                error_msg)
        ! Lower every procedure of one module node under its mangled name.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: module_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: mod_name, ff_error
        integer, allocatable :: decl_indices(:), proc_indices(:)
        integer :: p

        call set_empty(error_msg)
        call get_module_body_info(arena, module_index, mod_name, &
                                  decl_indices, proc_indices, ff_error)
        if (len_trim(ff_error) > 0) then
            error_msg = ff_error
            return
        end if
        do p = 1, size(proc_indices)
            call lower_one_module_procedure(arena, proc_indices(p), &
                                            trim(mod_name), context, error_msg)
            if (len_trim(error_msg) > 0) return
        end do
    end subroutine lower_one_module_unit_procedures

    subroutine lower_one_module_procedure(arena, node_index, module_name, &
                                          context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        character(len=*), intent(in) :: module_name
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: node_type
        type(function_def_node), pointer :: fn_node
        type(subroutine_def_node), pointer :: sb_node

        call set_empty(error_msg)
        if (.not. node_exists(arena, node_index)) return
        node_type = get_node_type_at(arena, node_index)
        if (node_type == 'function_def_node' .or. node_type == 'function_def') then
            fn_node => get_node_as_function_def(arena, node_index)
            if (.not. associated(fn_node)) return
            call lower_nested_internal_procedures(arena, fn_node%body_indices, &
                                                  context, error_msg)
            if (len_trim(error_msg) > 0) return
            call lower_scalar_function(arena, fn_node, context, error_msg, &
                module_procedure_mangled(module_name, fn_node%name), &
                procedure_node_index=node_index)
            if (len_trim(error_msg) > 0) return
        else if (node_type == 'subroutine_def_node' .or. &
                 node_type == 'subroutine_def') then
            sb_node => get_node_as_subroutine_def(arena, node_index)
            if (.not. associated(sb_node)) return
            call lower_nested_internal_procedures(arena, sb_node%body_indices, &
                                                  context, error_msg)
            if (len_trim(error_msg) > 0) return
            call lower_void_subroutine(arena, sb_node, context, error_msg, &
                module_procedure_mangled(module_name, sb_node%name), &
                procedure_node_index=node_index)
            if (len_trim(error_msg) > 0) return
        end if
    end subroutine lower_one_module_procedure

    subroutine lower_nested_internal_procedures(arena, body_indices, context, &
                                                error_msg)
        ! Lower the internal procedures contained in a module procedure as flat
        ! functions under their plain Fortran names (#249 B7f). They are emitted
        ! after the containing procedure; calls inside the containing procedure
        ! resolve to these symbols via call_emit_name. Host association of the
        ! containing procedure's local variables is not provided: an internal
        ! procedure that references a host variable is unsupported.
        type(ast_arena_t), intent(in) :: arena
        integer, allocatable, intent(in) :: body_indices(:)
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: node_type
        integer :: i

        call set_empty(error_msg)
        if (.not. allocated(body_indices)) return
        do i = 1, size(body_indices)
            if (.not. node_exists(arena, body_indices(i))) cycle
            node_type = get_node_type_at(arena, body_indices(i))
            if (node_type == 'function_def_node' .or. &
                node_type == 'function_def' .or. &
                node_type == 'subroutine_def_node' .or. &
                node_type == 'subroutine_def') then
                call lower_one_internal_function(arena, body_indices(i), &
                                                 context, error_msg)
                if (len_trim(error_msg) > 0) return
            end if
        end do
    end subroutine lower_nested_internal_procedures

    logical function procedure_has_nested_contains(arena, body_indices)
        ! Whether a procedure body holds an internal (nested) procedure, which
        ! ffc does not lower; such a procedure is skipped rather than failing.
        type(ast_arena_t), intent(in) :: arena
        integer, allocatable, intent(in) :: body_indices(:)
        integer :: i
        character(len=:), allocatable :: node_type

        procedure_has_nested_contains = .false.
        if (.not. allocated(body_indices)) return
        do i = 1, size(body_indices)
            if (.not. node_exists(arena, body_indices(i))) cycle
            node_type = get_node_type_at(arena, body_indices(i))
            if (node_type == 'contains_node' .or. node_type == 'contains' .or. &
                node_type == 'function_def_node' .or. &
                node_type == 'function_def' .or. &
                node_type == 'subroutine_def_node' .or. &
                node_type == 'subroutine_def') then
                procedure_has_nested_contains = .true.
                return
            end if
        end do
    end function procedure_has_nested_contains

    subroutine register_module_procedures(arena, module_index, context, error_msg)
        ! Register module procedures by Fortran name for call dispatch; skip
        ! duplicate names across modules (mangling keeps them distinct) and
        ! procedures with nested internals (not lowered).
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: module_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: mod_name, ff_error
        integer, allocatable :: declaration_indices(:), procedure_indices(:)

        call set_empty(error_msg)
        call get_module_body_info(arena, module_index, mod_name, &
                                  declaration_indices, procedure_indices, &
                                  ff_error)
        if (len_trim(ff_error) > 0) then
            error_msg = ff_error
            return
        end if
        call register_procedures_from_indices(arena, declaration_indices, &
                                              procedure_indices, context, error_msg)
    end subroutine register_module_procedures

    subroutine register_procedures_from_indices(arena, declaration_indices, &
                                                procedure_indices, context, &
                                                error_msg)
        ! Register a module or submodule unit's procedures by Fortran name and
        ! its named/operator generic interfaces. Shared by module and submodule
        ! registration; mangling is resolved later at the call site.
        type(ast_arena_t), intent(in) :: arena
        integer, allocatable, intent(in) :: declaration_indices(:)
        integer, allocatable, intent(in) :: procedure_indices(:)
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: node_type
        integer :: p
        type(function_def_node), pointer :: fn_node
        type(subroutine_def_node), pointer :: sb_node

        call set_empty(error_msg)
        if (.not. allocated(procedure_indices)) return
        do p = 1, size(procedure_indices)
            if (.not. node_exists(arena, procedure_indices(p))) cycle
            node_type = get_node_type_at(arena, procedure_indices(p))
            if (node_type == 'function_def_node' .or. node_type == 'function_def') then
                fn_node => get_node_as_function_def(arena, procedure_indices(p))
                if (.not. associated(fn_node)) cycle
                if (.not. allocated(fn_node%name)) cycle
                if (contained_function_kind(context, fn_node%name) /= 0) cycle
                call register_internal_function_name(fn_node, context, error_msg, &
                                                     procedure_indices(p))
                if (len_trim(error_msg) > 0) return
                call register_nested_internal_names(arena, fn_node%body_indices, &
                                                    context, error_msg)
                if (len_trim(error_msg) > 0) return
            else if (node_type == 'subroutine_def_node' .or. &
                     node_type == 'subroutine_def') then
                sb_node => get_node_as_subroutine_def(arena, procedure_indices(p))
                if (.not. associated(sb_node)) cycle
                if (.not. allocated(sb_node%name)) cycle
                if (contained_function_kind(context, sb_node%name) /= 0) cycle
                call register_internal_subroutine_name(sb_node, context, error_msg, &
                                                       procedure_indices(p))
                if (len_trim(error_msg) > 0) return
                call register_nested_internal_names(arena, sb_node%body_indices, &
                                                    context, error_msg)
                if (len_trim(error_msg) > 0) return
            end if
        end do
        ! Register named generic interfaces from the module declaration section
        ! (#249 B7c). This must run after the procedure loop so the specific
        ! procedures are already in the function table when query_specific_kinds
        ! is called.
        if (.not. allocated(declaration_indices)) return
        do p = 1, size(declaration_indices)
            if (.not. node_exists(arena, declaration_indices(p))) cycle
            node_type = get_node_type_at(arena, declaration_indices(p))
            if (node_type /= 'interface_block_node' .and. &
                node_type /= 'interface_block') cycle
            select type (iblock => arena%entries(declaration_indices(p))%node)
            type is (interface_block_node)
                if (allocated(iblock%kind)) then
                    block
                        character(len=:), allocatable :: ik
                        ik = lowercase_text(trim(iblock%kind))
                        if (ik == 'operator' .or. ik == 'assignment') then
                            call register_operator_interface(arena, iblock, &
                                context, ik == 'assignment', error_msg)
                            if (len_trim(error_msg) > 0) return
                            cycle
                        end if
                    end block
                end if
                if (.not. allocated(iblock%name)) cycle
                if (len_trim(iblock%name) == 0) cycle
                call register_generic_interface(arena, iblock, context, error_msg)
                if (len_trim(error_msg) > 0) return
            end select
        end do
    end subroutine register_procedures_from_indices

    subroutine register_nested_internal_names(arena, body_indices, context, &
                                              error_msg)
        ! Register, by Fortran name, the internal procedures contained in a
        ! module procedure (#249 B7f) so a call inside the containing procedure
        ! resolves to the right value kind and emit name.
        type(ast_arena_t), intent(in) :: arena
        integer, allocatable, intent(in) :: body_indices(:)
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: node_type
        integer :: i
        type(function_def_node), pointer :: fn_node
        type(subroutine_def_node), pointer :: sb_node

        call set_empty(error_msg)
        if (.not. allocated(body_indices)) return
        do i = 1, size(body_indices)
            if (.not. node_exists(arena, body_indices(i))) cycle
            node_type = get_node_type_at(arena, body_indices(i))
            if (node_type == 'function_def_node' .or. &
                node_type == 'function_def') then
                fn_node => get_node_as_function_def(arena, body_indices(i))
                if (.not. associated(fn_node)) cycle
                if (.not. allocated(fn_node%name)) cycle
                if (contained_function_kind(context, fn_node%name) /= 0) cycle
                call register_internal_function_name(fn_node, context, error_msg, &
                                                     body_indices(i))
                if (len_trim(error_msg) > 0) return
            else if (node_type == 'subroutine_def_node' .or. &
                     node_type == 'subroutine_def') then
                sb_node => get_node_as_subroutine_def(arena, body_indices(i))
                if (.not. associated(sb_node)) cycle
                if (.not. allocated(sb_node%name)) cycle
                if (contained_function_kind(context, sb_node%name) /= 0) cycle
                call register_internal_subroutine_name(sb_node, context, &
                                                       error_msg, body_indices(i))
                if (len_trim(error_msg) > 0) return
            end if
        end do
    end subroutine register_nested_internal_names

    logical function is_internal_procedure_entry(arena, node_index)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        character(len=:), allocatable :: node_type

        is_internal_procedure_entry = .false.
        if (.not. node_exists(arena, node_index)) return
        node_type = get_node_type_at(arena, node_index)
        if (node_type == 'function_def_node' .or. node_type == 'function_def' .or. &
            node_type == 'subroutine_def_node' .or. node_type == 'subroutine_def' .or. &
            node_type == 'contains_node' .or. node_type == 'contains') then
            is_internal_procedure_entry = .true.
        end if
    end function is_internal_procedure_entry

    subroutine lower_scalar_function(arena, node, parent_context, error_msg, &
                                     emit_name_override, procedure_node_index)
        type(ast_arena_t), intent(in) :: arena
        type(function_def_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: parent_context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=*), intent(in), optional :: emit_name_override
        integer, intent(in), optional :: procedure_node_index
        type(lowering_context_t) :: context
        type(lr_operand_desc_t) :: value
        integer :: param_count
        integer :: result_index
        integer :: value_kind
        integer :: result_derived_index
        integer :: alloc_elem_kind
        integer :: alloc_rank
        integer :: function_index
        integer :: function_node_index
        integer :: result_binding_index
        logical :: terminated
        logical :: result_is_alloc_array
        logical :: result_is_data_pointer
        integer :: pointee_kind
        logical :: host_character_result
        type(declaration_binding_t) :: result_binding
        character(len=:), allocatable :: result_name
        character(len=:), allocatable :: emit_name

        if (.not. allocated(node%name)) then
            error_msg = 'direct LIRIC session function requires a name'
            return
        end if
        function_node_index = 0
        if (present(procedure_node_index)) then
            function_node_index = procedure_node_index
        else
            function_index = find_function_index(parent_context, node%name)
            if (function_index > 0) &
                function_node_index = parent_context%function_node_indices(function_index)
        end if
        if (function_node_index > 0) then
            call function_value_kind(node, parent_context, value_kind, &
                                     result_derived_index, error_msg, &
                                     function_node_index)
        else
            call function_value_kind(node, parent_context, value_kind, &
                                     result_derived_index, error_msg)
        end if
        if (len_trim(error_msg) > 0) return
        result_name = trim(node%name)
        if (allocated(node%result_variable)) then
            if (len_trim(node%result_variable) > 0) then
                result_name = trim(node%result_variable)
            end if
        end if

        context%session = parent_context%session
        context%i32_print_format_id = parent_context%i32_print_format_id
        context%str_print_format_id = parent_context%str_print_format_id
        context%string_literal_count = parent_context%string_literal_count
        if (allocated(parent_context%unit_symbol_prefix)) &
            context%unit_symbol_prefix = parent_context%unit_symbol_prefix
        context%arena = arena
        context%root_index = parent_context%root_index
        ! The dialect is a property of the translation unit, not of one
        ! procedure: a contained procedure lowers under the same Lazy
        ! Fortran defaults as its host (#571).
        context%lazy_mode = parent_context%lazy_mode
        allocate(context%symbols(16))
        allocate(context%derived_types(8))
        allocate(context%module_exports(4))
        allocate(context%function_names(8))
        allocate(context%function_value_kinds(8))
        context%function_value_kinds = VALUE_I32
        call copy_internal_function_names(parent_context, context)
        call copy_derived_type_table(parent_context, context)
        call copy_module_exports(parent_context, context)
        call copy_declaration_registry(parent_context, context)
        call copy_host_global_symbols(parent_context, context)
        result_binding_index = 0
        if (function_node_index > 0) then
            call resolve_function_result_binding(arena, function_node_index, &
                result_name, result_binding)
            result_binding_index = find_symbol_for_binding(context, &
                result_binding)
        end if
        context%block_scope_floor = context%symbol_count
        context%in_internal_function = .true.
        context%current_proc_name = trim(node%name)
        context%current_proc_node_index = function_node_index

        param_count = 0
        if (allocated(node%param_indices)) param_count = size(node%param_indices)
        if (present(emit_name_override)) then
            emit_name = trim(emit_name_override)
        else
            emit_name = nested_procedure_emit_name(arena, function_node_index, &
                node%name, node%bind_c_clause)
        end if
        if (value_kind == VALUE_F32) then
            if (.not. begin_liric_f32_function(context%session, emit_name, &
                                               param_count, error_msg)) return
        else if (value_kind == VALUE_F64) then
            if (.not. begin_liric_f64_function(context%session, emit_name, &
                                               param_count, error_msg)) return
        else if (value_kind == VALUE_DEFERRED_CHARACTER_RESULT .or. &
                 value_kind == VALUE_DERIVED .or. &
                 value_kind == VALUE_ARRAY_RESULT .or. &
                 value_kind == VALUE_ALLOC_ARRAY_RESULT .or. &
                 value_kind == VALUE_C4 .or. value_kind == VALUE_C8) then
            ! Aggregate / complex / array results return via a leading sret
            ! pointer: the caller passes the result buffer address as param 0.
            if (.not. begin_void_subroutine(context%session, emit_name, &
                                            param_count + 1, error_msg)) return
        else if (value_kind == VALUE_DATA_PTR_RESULT) then
            ! A scalar data pointer result returns the target address itself,
            ! so the LIRIC function returns ptr and copies no pointee storage.
            if (.not. begin_ptr_function(context%session, emit_name, &
                                         param_count, error_msg)) return
        else if (value_kind == VALUE_I64) then
            if (.not. begin_i64_function(context%session, emit_name, param_count, &
                                                          error_msg)) return
        else
            if (.not. begin_i32_function(context%session, emit_name, param_count, &
                                                          error_msg)) return
        end if

        call materialize_host_character_globals(context, error_msg)
        if (len_trim(error_msg) > 0) return

        host_character_result = .false.
        if (result_binding_index > 0) then
            context%current_function_result_index = result_binding_index
            host_character_result = value_kind == VALUE_DEFERRED_CHARACTER_RESULT .and. &
                context%symbols(result_binding_index)%value_kind == VALUE_CHARACTER .and. &
                context%symbols(result_binding_index)%is_deferred_character
        else if (value_kind == VALUE_DEFERRED_CHARACTER_RESULT) then
            call grow_symbols(context)
            context%current_function_result_index = context%symbol_count + 1
            context%symbols(context%current_function_result_index)%name = &
                trim(result_name)
            context%symbols(context%current_function_result_index)%value_kind = &
                VALUE_CHARACTER
            context%symbols(context%current_function_result_index)%&
                is_deferred_character = .true.
            context%symbols(context%current_function_result_index)%&
                has_character_value = .false.
            if (.not. emit_ptr_offset(context%session, &
                ptr_param(context%session, 0), 0_c_int64_t, &
                context%symbols(context%current_function_result_index)%deferred_data, &
                error_msg)) return
            if (.not. emit_ptr_offset(context%session, &
                ptr_param(context%session, 0), 8_c_int64_t, &
                context%symbols(context%current_function_result_index)%deferred_length, &
                error_msg)) return
            ! The caller's return descriptor is the canonical 32-byte record,
            ! so the result symbol also owns its capacity and storage_class
            ! slots. Every assignment to the result then records which storage
            ! class the returned data belongs to, which is what lets the caller
            ! release an owned result and leave a borrowed one alone.
            associate (rsym => &
                context%symbols(context%current_function_result_index))
                if (.not. emit_ptr_offset(context%session, &
                    ptr_param(context%session, 0), 16_c_int64_t, &
                    rsym%deferred_capacity, error_msg)) return
                if (.not. emit_ptr_offset(context%session, &
                    ptr_param(context%session, 0), 24_c_int64_t, &
                    rsym%deferred_storage, error_msg)) return
                rsym%has_character_ownership = .true.
            end associate
            context%symbol_count = context%current_function_result_index
        else if (value_kind == VALUE_DERIVED) then
            call grow_symbols(context)
            context%current_function_result_index = context%symbol_count + 1
            associate (rsym => context%symbols(context%current_function_result_index))
                rsym%name = trim(result_name)
                rsym%value_kind = VALUE_DERIVED
                rsym%is_derived = .true.
                rsym%derived_type_index = result_derived_index
                rsym%array_size = total_component_slots(context, &
                                                        result_derived_index)
                rsym%element_address = ptr_param(context%session, 0)
            end associate
            context%symbol_count = context%current_function_result_index
        else if (value_kind == VALUE_C4 .or. value_kind == VALUE_C8) then
            ! Result re/im live in the sret buffer: re at +0, im at +4 (c4) or
            ! +8 (c8). Complex assignment stores through these address slots.
            call grow_symbols(context)
            context%current_function_result_index = context%symbol_count + 1
            associate (rsym => context%symbols(context%current_function_result_index))
                rsym%name = trim(result_name)
                rsym%value_kind = value_kind
                rsym%has_address = .true.
                if (.not. emit_ptr_offset(context%session, &
                    ptr_param(context%session, 0), 0_c_int64_t, &
                    rsym%address, error_msg)) return
                if (value_kind == VALUE_C4) then
                    if (.not. emit_ptr_offset(context%session, &
                        ptr_param(context%session, 0), 4_c_int64_t, &
                        rsym%element_address, error_msg)) return
                else
                    if (.not. emit_ptr_offset(context%session, &
                        ptr_param(context%session, 0), 8_c_int64_t, &
                        rsym%element_address, error_msg)) return
                end if
            end associate
            context%symbol_count = context%current_function_result_index
        else if (value_kind == VALUE_ARRAY_RESULT) then
            ! Pre-register the result array bound to the sret buffer (param 0).
            ! The body array declaration rebinds its shape and element kind onto
            ! this symbol's element_address, so whole-array assignment writes the
            ! result straight into the caller's destination.
            call grow_symbols(context)
            context%current_function_result_index = context%symbol_count + 1
            associate (rsym => context%symbols(context%current_function_result_index))
                rsym%name = trim(result_name)
                rsym%value_kind = VALUE_I32
                rsym%is_array = .true.
                rsym%is_array_result = .true.
                rsym%element_address = ptr_param(context%session, 0)
            end associate
            context%symbol_count = context%current_function_result_index
        else if (value_kind == VALUE_ALLOC_ARRAY_RESULT) then
            ! Bind the result onto the caller's descriptor (param 0). The body's
            ! allocate/assignment machinery writes through allocatable_descriptor_
            ! address, so the callee allocates straight into the caller's slot.
            call alloc_array_result_info(node, context, result_is_alloc_array, &
                                         alloc_elem_kind, alloc_rank)
            call grow_symbols(context)
            context%current_function_result_index = context%symbol_count + 1
            associate (rsym => context%symbols(context%current_function_result_index))
                rsym%name = trim(result_name)
                rsym%value_kind = alloc_elem_kind
                rsym%is_array = .true.
                rsym%is_allocatable = .true.
                rsym%array_rank = alloc_rank
                rsym%allocatable_descriptor_address = ptr_param(context%session, 0)
            end associate
            context%symbol_count = context%current_function_result_index
        else if (value_kind == VALUE_DATA_PTR_RESULT) then
            ! The result is a data pointer: it starts disassociated and adopts a
            ! target address at the first `r => t` in the body.
            result_is_data_pointer = result_var_is_data_pointer(node, context, &
                                                                pointee_kind)
            call grow_symbols(context)
            context%current_function_result_index = context%symbol_count + 1
            associate (rsym => context%symbols(context%current_function_result_index))
                rsym%name = trim(result_name)
                rsym%value_kind = pointee_kind
                rsym%is_pointer = .true.
                rsym%is_associated = .false.
                rsym%has_address = .false.
                rsym%is_reference = .false.
            end associate
            context%symbol_count = context%current_function_result_index
        else
            call define_symbol(context, result_name, value_kind, error_msg)
            if (len_trim(error_msg) > 0) return
            context%current_function_result_index = find_symbol_compat(context, result_name)
        end if
        ! The result binding is synthetic in FortFront (its identity is the
        ! function scope), so publish it before any specification or executable
        ! expression can reference the result name (#457).
        if (result_binding%found .and. context%current_function_result_index > 0) then
            call attach_symbol_binding(context, context%current_function_result_index, &
                                       result_binding)
        end if
        if (value_kind == VALUE_DEFERRED_CHARACTER_RESULT .or. &
                value_kind == VALUE_DERIVED .or. &
                value_kind == VALUE_ARRAY_RESULT .or. &
                value_kind == VALUE_ALLOC_ARRAY_RESULT .or. &
                value_kind == VALUE_C4 .or. value_kind == VALUE_C8) then
            call define_reference_parameters(arena, node%param_indices, context, &
                                              error_msg, body_indices = &
                                              node%body_indices, param_offset=1, &
                                              return_value_kind=value_kind)
        else
            call define_reference_parameters(arena, node%param_indices, context, &
                                              error_msg, body_indices = &
                                              node%body_indices, &
                                              return_value_kind=value_kind)
        end if
        if (len_trim(error_msg) > 0) return

        ! Rebind every assumed-shape dummy from its descriptor parameter (#334).
        call bind_assumed_shape_descriptor_params(arena, node%param_indices, &
                                                  context, trim(node%name), &
                                                  error_msg)
        if (len_trim(error_msg) > 0) return

        ! Host-associate scalar module variables as addressed globals (#263).
        call bind_module_variables_in_context(context, error_msg)
        if (len_trim(error_msg) > 0) return
        ! Host-associate module-scope named constants and enumerators (#1826).
        call bind_module_constants_in_context(context, error_msg)
        if (len_trim(error_msg) > 0) return

        if (allocated(node%body_indices)) then
            call lower_statement_list(arena, node%body_indices, context, value, &
                                      terminated, error_msg)
            if (len_trim(error_msg) > 0) return
        end if

        result_index = context%current_function_result_index
        if (result_index <= 0) then
            error_msg = 'function result symbol was not declared: '// &
                        trim(result_name)
            return
        end if
        if (.not. context%current_block_terminated) then
            if (value_kind == VALUE_DEFERRED_CHARACTER_RESULT .or. &
                value_kind == VALUE_DERIVED .or. &
                value_kind == VALUE_ARRAY_RESULT .or. &
                value_kind == VALUE_ALLOC_ARRAY_RESULT .or. &
                value_kind == VALUE_C4 .or. value_kind == VALUE_C8) then
                if (host_character_result) then
                    call emit_host_character_result_return(context, result_index, &
                        error_msg)
                    if (len_trim(error_msg) > 0) return
                else if (.not. emit_ret_void(context%session, error_msg)) then
                    return
                end if
            else if (value_kind == VALUE_DATA_PTR_RESULT) then
                ! Return the adopted target address, or a null pointer when the
                ! result was never associated.
                if (context%symbols(result_index)%has_address .and. &
                    context%symbols(result_index)%is_associated) then
                    value = context%symbols(result_index)%address
                else
                    value = null_ptr_operand(context)
                end if
                if (.not. emit_ret_ptr_operand(context%session, value, &
                                               error_msg)) return
            else if (value_kind == VALUE_I64) then
                call load_function_result_value(context, result_index, value, &
                    error_msg)
                if (len_trim(error_msg) > 0) return
                if (.not. emit_ret_i64_operand(context%session, value, error_msg)) return
            else
                call load_function_result_value(context, result_index, value, &
                    error_msg)
                if (len_trim(error_msg) > 0) return
                if (.not. emit_ret_i32_operand(context%session, value, error_msg)) return
            end if
        end if
        if (.not. finish_function(context%session, error_msg)) return

        parent_context%session = context%session
        parent_context%string_literal_count = context%string_literal_count
        call set_empty(error_msg)
    end subroutine lower_scalar_function

    include 'session_program_lowering_functions_tail.inc'
