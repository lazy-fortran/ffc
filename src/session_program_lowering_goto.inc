    ! GOTO / labeled-statement lowering (#270).
    !
    ! Fortran statement labels (the `100` in `100 continue`) plus GOTO form an
    ! arbitrary control-flow graph. The value-tracking lowerer normally keeps a
    ! scalar's current SSA value in the symbol table, which only works inside a
    ! structured construct that can place phi nodes at its merge. Across GOTO
    ! edges that breaks down, so a body with labels promotes every scalar
    ! integer/logical local to a stack slot: reads load and writes store (the
    ! existing is_reference path), and the backend's mem2reg rebuilds SSA. Each
    ! labeled statement owns a LIRIC block; a GOTO is a plain branch to it.

    logical function body_has_statement_labels(arena, body_indices)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: body_indices(:)
        integer :: i

        body_has_statement_labels = .false.
        do i = 1, size(body_indices)
            if (len_trim(get_node_stmt_label(arena, body_indices(i))) > 0) then
                body_has_statement_labels = .true.
                return
            end if
        end do
    end function body_has_statement_labels

    logical function is_executable_body_node(arena, node_index)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        character(len=:), allocatable :: node_type

        is_executable_body_node = .false.
        if (.not. node_exists(arena, node_index)) return
        node_type = get_node_type_at(arena, node_index)
        select case (trim(node_type))
        case ('declaration', 'declaration_node', &
              'parameter_declaration', 'parameter_declaration_node', &
              'implicit_statement', 'implicit_statement_node', &
              'comment', 'comment_node', &
              'use_statement', 'use_statement_node', &
              'interface_block', 'interface_block_node', &
              'derived_type', 'derived_type_node', &
              'module', 'module_node')
            is_executable_body_node = .false.
        case default
            is_executable_body_node = .true.
        end select
    end function is_executable_body_node

    subroutine lower_labeled_program_body(arena, body_indices, &
                                          has_nested_program, context, value, &
                                          error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: body_indices(:)
        logical, intent(in) :: has_nested_program
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i
        integer :: first_exec

        value = i32_immediate(context%session, 0_c_int64_t)
        call set_empty(error_msg)

        ! Reject control-flow forms the simple branch lowering cannot honour
        ! before we promote anything or emit blocks.
        do i = 1, size(body_indices)
            if (goto_is_computed(arena, body_indices(i))) then
                call unsupported_feature_error('computed goto', &
                    get_node_line(arena, body_indices(i)), &
                    get_node_column(arena, body_indices(i)), &
                    'direct LIRIC session lowers only simple goto to a label', &
                    error_msg)
                return
            end if
        end do

        ! Lower leading declarations with the normal path; labels never attach
        ! to them. Find where the executable region begins.
        first_exec = size(body_indices) + 1
        do i = 1, size(body_indices)
            if (is_internal_procedure_entry(arena, body_indices(i))) cycle
            if (is_executable_body_node(arena, body_indices(i))) then
                first_exec = i
                exit
            end if
            if (.not. lower_declaration_region_node(arena, body_indices(i), &
                                                    has_nested_program, context, &
                                                    value, error_msg)) return
        end do

        call promote_scalars_to_memory(context, error_msg)
        if (len_trim(error_msg) > 0) return

        call build_label_table(arena, body_indices(first_exec:), context, &
                               error_msg)
        if (len_trim(error_msg) > 0) return

        call lower_labeled_executables(arena, body_indices(first_exec:), &
                                       context, value, error_msg)

        context%in_labeled_body = .false.
        context%label_count = 0
    end subroutine lower_labeled_program_body

    logical function lower_declaration_region_node(arena, node_index, &
                                                   has_nested_program, context, &
                                                   value, error_msg) result(ok)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        logical, intent(in) :: has_nested_program
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(inout) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: node_type

        ok = .false.
        node_type = get_node_type_at(arena, node_index)
        if (node_type == 'module_node' .or. node_type == 'module') then
            if (.not. has_nested_program) then
                call unsupported_feature_error('module definition', &
                    get_node_line(arena, node_index), &
                    get_node_column(arena, node_index), &
                    'direct LIRIC session only supports modules as USE '// &
                    'targets, not as top-level program units', error_msg)
                return
            end if
            ok = .true.
            return
        end if
        call lower_statement(arena, node_index, context, value, error_msg)
        if (len_trim(error_msg) > 0) return
        ok = .true.
    end function lower_declaration_region_node

    ! Move scalar integer/logical locals into stack slots so their values
    ! survive GOTO edges. Reuses the is_reference load/store path.
    subroutine promote_scalars_to_memory(context, error_msg)
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: address
        integer :: i

        call set_empty(error_msg)
        do i = 1, context%symbol_count
            if (context%symbols(i)%is_parameter) cycle
            if (context%symbols(i)%is_array) cycle
            if (context%symbols(i)%is_allocatable) cycle
            if (context%symbols(i)%is_derived) cycle
            if (context%symbols(i)%has_address) cycle
            select case (context%symbols(i)%value_kind)
            case (VALUE_I32, VALUE_LOGICAL)
                if (.not. emit_i32_alloca(context%session, address, &
                                          error_msg)) return
                if (.not. emit_i32_store(context%session, &
                                         context%symbols(i)%value, address, &
                                         error_msg)) return
                context%symbols(i)%address = address
                context%symbols(i)%has_address = .true.
                context%symbols(i)%is_reference = .true.
            end select
        end do
    end subroutine promote_scalars_to_memory

    subroutine build_label_table(arena, exec_indices, context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: exec_indices(:)
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: label
        integer :: i
        integer :: n

        call set_empty(error_msg)
        n = 0
        do i = 1, size(exec_indices)
            if (len_trim(get_node_stmt_label(arena, exec_indices(i))) > 0) n = n + 1
        end do
        allocate (context%label_names(max(n, 1)))
        allocate (context%label_blocks(max(n, 1)))
        context%label_count = 0
        do i = 1, size(exec_indices)
            label = get_node_stmt_label(arena, exec_indices(i))
            if (len_trim(label) == 0) cycle
            if (len_trim(label) > len(context%label_names)) then
                error_msg = 'statement label too long: '//trim(label)
                return
            end if
            context%label_count = context%label_count + 1
            context%label_names(context%label_count) = trim(label)
            context%label_blocks(context%label_count) = &
                create_liric_block(context%session)
        end do
        context%in_labeled_body = .true.
    end subroutine build_label_table

    integer(c_int32_t) function find_label_block(context, label, found)
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: label
        logical, intent(out) :: found
        integer :: i

        found = .false.
        find_label_block = 0_c_int32_t
        do i = 1, context%label_count
            if (trim(context%label_names(i)) == trim(label)) then
                find_label_block = context%label_blocks(i)
                found = .true.
                return
            end if
        end do
    end function find_label_block

    subroutine lower_labeled_executables(arena, exec_indices, context, value, &
                                         error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: exec_indices(:)
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: label
        integer(c_int32_t) :: label_block
        logical :: found
        integer :: i

        value = i32_immediate(context%session, 0_c_int64_t)
        call set_empty(error_msg)
        do i = 1, size(exec_indices)
            if (is_internal_procedure_entry(arena, exec_indices(i))) cycle
            label = get_node_stmt_label(arena, exec_indices(i))
            if (len_trim(label) > 0) then
                label_block = find_label_block(context, label, found)
                if (.not. found) then
                    error_msg = 'undefined statement label: '//trim(label)
                    return
                end if
                ! Fall through into the labeled block unless the previous
                ! statement already transferred control.
                if (.not. context%current_block_terminated) then
                    if (.not. emit_liric_br(context%session, label_block, &
                                            error_msg)) return
                end if
                if (.not. set_liric_block(context%session, label_block, &
                                          error_msg)) return
                context%current_block_id = label_block
                context%current_block_terminated = .false.
            end if
            ! A statement after an unterminated GOTO/return is dead code with no
            ! label of its own; skip it so we do not emit into a finished block.
            if (context%current_block_terminated) cycle
            call lower_statement(arena, exec_indices(i), context, value, &
                                 error_msg)
            if (len_trim(error_msg) > 0) return
        end do
    end subroutine lower_labeled_executables

    subroutine lower_goto(arena, node_index, context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: label
        integer(c_int32_t) :: target_block
        logical :: found

        if (.not. context%in_labeled_body) then
            call unsupported_feature_error('goto statement', &
                get_node_line(arena, node_index), &
                get_node_column(arena, node_index), &
                'goto is only lowered inside a labeled program body', error_msg)
            return
        end if
        if (goto_is_computed(arena, node_index)) then
            call unsupported_feature_error('computed goto', &
                get_node_line(arena, node_index), &
                get_node_column(arena, node_index), &
                'direct LIRIC session lowers only simple goto to a label', &
                error_msg)
            return
        end if
        label = get_goto_label(arena, node_index)
        if (len_trim(label) == 0) then
            error_msg = 'goto statement is missing a target label'
            return
        end if
        target_block = find_label_block(context, label, found)
        if (.not. found) then
            error_msg = 'goto targets undefined statement label: '//trim(label)
            return
        end if
        if (.not. emit_liric_br(context%session, target_block, error_msg)) return
        context%current_block_terminated = .true.
    end subroutine lower_goto
