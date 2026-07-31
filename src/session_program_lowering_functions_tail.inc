    subroutine lower_void_subroutine(arena, node, parent_context, error_msg, &
                                      emit_name_override, procedure_node_index)
        type(ast_arena_t), intent(in) :: arena
        type(subroutine_def_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: parent_context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=*), intent(in), optional :: emit_name_override
        integer, intent(in), optional :: procedure_node_index
        type(lowering_context_t) :: context
        type(lr_operand_desc_t) :: value
        integer :: param_count
        integer :: hidden_extent_count
        integer :: table_function_index
        logical :: terminated
        character(len=:), allocatable :: emit_name

        if (.not. allocated(node%name)) then
            error_msg = 'direct LIRIC session subroutine requires a name'
            return
        end if

        context%session = parent_context%session
        context%i32_print_format_id = parent_context%i32_print_format_id
        context%str_print_format_id = parent_context%str_print_format_id
        context%string_literal_count = parent_context%string_literal_count
        if (allocated(parent_context%unit_symbol_prefix)) &
            context%unit_symbol_prefix = parent_context%unit_symbol_prefix
        context%arena = arena
        context%root_index = parent_context%root_index
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
        if (present(procedure_node_index)) then
            context%current_proc_node_index = procedure_node_index
        else
            table_function_index = find_function_index(parent_context, node%name)
            if (table_function_index > 0) then
                context%current_proc_node_index = &
                    parent_context%function_node_indices(table_function_index)
            end if
        end if
        call copy_host_global_symbols(parent_context, context)
        context%block_scope_floor = context%symbol_count
        context%in_internal_function = .true.
        context%in_internal_subroutine = .true.
        context%current_proc_name = trim(node%name)

        param_count = 0
        if (allocated(node%param_indices)) param_count = size(node%param_indices)
        ! A rank-1 assumed-shape dummy whose extent does not fold at compile
        ! time (an allocatable actual) carries one hidden i64 extent argument
        ! appended after the visible parameters (#W2 runtime-extent ABI).
        hidden_extent_count = callee_hidden_extent_count(context, trim(node%name))
        if (present(emit_name_override)) then
            emit_name = trim(emit_name_override)
        else
            emit_name = nested_procedure_emit_name(arena, &
                context%current_proc_node_index, node%name, node%bind_c_clause)
        end if
        if (.not. begin_void_subroutine(context%session, emit_name, &
                param_count + hidden_extent_count, error_msg)) return

        call materialize_host_character_globals(context, error_msg)
        if (len_trim(error_msg) > 0) return

        call define_reference_parameters(arena, node%param_indices, context, &
                                         error_msg, node%body_indices)
        if (len_trim(error_msg) > 0) return

        if (hidden_extent_count > 0) then
            call bind_hidden_extent_params(arena, node%param_indices, context, &
                                           trim(node%name), param_count, error_msg)
            if (len_trim(error_msg) > 0) return
        end if

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

        if (.not. context%current_block_terminated) then
            if (.not. emit_ret_void(context%session, error_msg)) return
        end if
        if (.not. finish_function(context%session, error_msg)) return

        parent_context%session = context%session
        parent_context%string_literal_count = context%string_literal_count
        call set_empty(error_msg)
    end subroutine lower_void_subroutine

    logical function dummy_is_character(arena, body_indices, param_name)
        ! A dummy is an assumed-length character when a body declaration names
        ! it with type 'character'. ffc supports only character(len=*) dummies.
        type(ast_arena_t), intent(in) :: arena
        integer, allocatable, intent(in) :: body_indices(:)
        character(len=*), intent(in) :: param_name
        integer :: i, k

        dummy_is_character = .false.
        if (.not. allocated(body_indices)) return
        do i = 1, size(body_indices)
            if (.not. node_exists(arena, body_indices(i))) cycle
            select type (decl => arena%entries(body_indices(i))%node)
            type is (declaration_node)
                if (.not. allocated(decl%type_name)) cycle
                if (.not. is_character_type_name(decl%type_name)) cycle
                if (allocated(decl%var_name)) then
                    if (trim(decl%var_name) == trim(param_name)) then
                        dummy_is_character = .true.
                        return
                    end if
                end if
                if (decl%is_multi_declaration .and. allocated(decl%var_names)) then
                    do k = 1, size(decl%var_names)
                        if (trim(decl%var_names(k)) == trim(param_name)) then
                            dummy_is_character = .true.
                            return
                        end if
                    end do
                end if
            end select
        end do
    end function dummy_is_character

    logical function dummy_char_is_writable(arena, body_indices, param_name)
        ! Whether a character dummy named param_name is allocatable with
        ! intent(out)/intent(inout): the caller passes the actual's own
        ! descriptor so callee writes propagate back (#171).
        type(ast_arena_t), intent(in) :: arena
        integer, allocatable, intent(in) :: body_indices(:)
        character(len=*), intent(in) :: param_name
        integer :: i, k
        logical :: matches

        dummy_char_is_writable = .false.
        if (.not. allocated(body_indices)) return
        do i = 1, size(body_indices)
            if (.not. node_exists(arena, body_indices(i))) cycle
            select type (decl => arena%entries(body_indices(i))%node)
            type is (declaration_node)
                if (.not. allocated(decl%type_name)) cycle
                if (.not. is_character_type_name(decl%type_name)) cycle
                if (.not. decl%is_allocatable) cycle
                if (.not. decl%has_intent) cycle
                if (.not. allocated(decl%intent)) cycle
                if (trim(decl%intent) /= 'out' .and. &
                    trim(decl%intent) /= 'inout') cycle
                matches = .false.
                if (allocated(decl%var_name)) then
                    if (trim(decl%var_name) == trim(param_name)) matches = .true.
                end if
                if (decl%is_multi_declaration .and. allocated(decl%var_names)) then
                    do k = 1, size(decl%var_names)
                        if (trim(decl%var_names(k)) == trim(param_name)) &
                            matches = .true.
                    end do
                end if
                if (matches) then
                    dummy_char_is_writable = .true.
                    return
                end if
            end select
        end do
    end function dummy_char_is_writable

    logical function callee_dummy_char_writable(arena, callee_name, param_pos)
        ! Whether the callee's param_pos-th dummy is an allocatable
        ! intent(out)/(inout) character.
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: callee_name
        integer, intent(in) :: param_pos
        integer :: n
        character(len=:), allocatable :: name, name_err
        character(len=:), allocatable :: node_type
        type(function_def_node), pointer :: fn_node
        type(subroutine_def_node), pointer :: sb_node

        callee_dummy_char_writable = .false.
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            node_type = get_node_type_at(arena, n)
            if (node_type == 'subroutine_def_node' .or. &
                node_type == 'subroutine_def') then
                sb_node => get_node_as_subroutine_def(arena, n)
                if (.not. associated(sb_node)) cycle
                if (allocated(sb_node%name)) then
                    if (same_name(sb_node%name, callee_name)) then
                        if (.not. allocated(sb_node%param_indices)) return
                        if (param_pos < 1 .or. &
                            param_pos > size(sb_node%param_indices)) return
                        call parameter_name(arena, sb_node%param_indices(param_pos), &
                                            name, name_err)
                        if (len_trim(name_err) > 0) return
                        callee_dummy_char_writable = dummy_char_is_writable(arena, &
                            sb_node%body_indices, name)
                        return
                    end if
                end if
            else if (node_type == 'function_def_node' .or. &
                     node_type == 'function_def') then
                fn_node => get_node_as_function_def(arena, n)
                if (.not. associated(fn_node)) cycle
                if (allocated(fn_node%name)) then
                    if (same_name(fn_node%name, callee_name)) then
                        if (.not. allocated(fn_node%param_indices)) return
                        if (param_pos < 1 .or. &
                            param_pos > size(fn_node%param_indices)) return
                        call parameter_name(arena, fn_node%param_indices(param_pos), &
                                            name, name_err)
                        if (len_trim(name_err) > 0) return
                        callee_dummy_char_writable = dummy_char_is_writable(arena, &
                            fn_node%body_indices, name)
                        return
                    end if
                end if
            end if
        end do
    end function callee_dummy_char_writable

    logical function dummy_is_class_star(arena, body_indices, param_name)
        ! A dummy is class(*) (unlimited polymorphic) when a body declaration
        ! names it with the type class(*). A class(t) dummy names a concrete
        ! derived type and is passed by reference (static dispatch), not boxed,
        ! so it must not match here.
        type(ast_arena_t), intent(in) :: arena
        integer, allocatable, intent(in) :: body_indices(:)
        character(len=*), intent(in) :: param_name
        integer :: i, k
        logical :: matches

        dummy_is_class_star = .false.
        if (.not. allocated(body_indices)) return
        do i = 1, size(body_indices)
            if (.not. node_exists(arena, body_indices(i))) cycle
            select type (decl => arena%entries(body_indices(i))%node)
            type is (declaration_node)
                if (.not. allocated(decl%type_name)) cycle
                if (index(lowercase_text(decl%type_name), 'class') /= 1) cycle
                ! class(t) names a concrete type and is passed by reference, not
                ! boxed; only class(*) is the unlimited-polymorphic box.
                if (is_derived_type_spec(decl%type_name)) cycle
                matches = .false.
                if (allocated(decl%var_name)) then
                    if (trim(decl%var_name) == trim(param_name)) matches = .true.
                end if
                if (decl%is_multi_declaration .and. allocated(decl%var_names)) then
                    do k = 1, size(decl%var_names)
                        if (trim(decl%var_names(k)) == trim(param_name)) &
                            matches = .true.
                    end do
                end if
                if (matches) then
                    dummy_is_class_star = .true.
                    return
                end if
            end select
        end do
    end function dummy_is_class_star

    logical function callee_dummy_is_class_star(arena, callee_name, param_pos)
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: callee_name
        integer, intent(in) :: param_pos
        integer :: n
        character(len=:), allocatable :: name, name_err
        character(len=:), allocatable :: node_type
        type(function_def_node), pointer :: fn_node
        type(subroutine_def_node), pointer :: sb_node

        callee_dummy_is_class_star = .false.
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            node_type = get_node_type_at(arena, n)
            if (node_type == 'subroutine_def_node' .or. &
                node_type == 'subroutine_def') then
                sb_node => get_node_as_subroutine_def(arena, n)
                if (.not. associated(sb_node)) cycle
                if (allocated(sb_node%name)) then
                    if (same_name(sb_node%name, callee_name)) then
                        if (.not. allocated(sb_node%param_indices)) return
                        if (param_pos < 1 .or. &
                            param_pos > size(sb_node%param_indices)) return
                        call parameter_name(arena, sb_node%param_indices(param_pos), &
                                            name, name_err)
                        if (len_trim(name_err) > 0) return
                        callee_dummy_is_class_star = dummy_is_class_star(arena, &
                            sb_node%body_indices, name)
                        return
                    end if
                end if
            else if (node_type == 'function_def_node' .or. &
                     node_type == 'function_def') then
                fn_node => get_node_as_function_def(arena, n)
                if (.not. associated(fn_node)) cycle
                if (allocated(fn_node%name)) then
                    if (same_name(fn_node%name, callee_name)) then
                        if (.not. allocated(fn_node%param_indices)) return
                        if (param_pos < 1 .or. &
                            param_pos > size(fn_node%param_indices)) return
                        call parameter_name(arena, fn_node%param_indices(param_pos), &
                                            name, name_err)
                        if (len_trim(name_err) > 0) return
                        callee_dummy_is_class_star = dummy_is_class_star(arena, &
                            fn_node%body_indices, name)
                        return
                    end if
                end if
            end if
        end do
    end function callee_dummy_is_class_star

    logical function dummy_is_array(arena, body_indices, param_name)
        ! A dummy is an array when a body declaration names it with an array
        ! spec (is_array). Mirrors dummy_is_character for the array-argument ABI.
        type(ast_arena_t), intent(in) :: arena
        integer, allocatable, intent(in) :: body_indices(:)
        character(len=*), intent(in) :: param_name
        integer :: i, k

        dummy_is_array = .false.
        if (.not. allocated(body_indices)) return
        do i = 1, size(body_indices)
            if (.not. node_exists(arena, body_indices(i))) cycle
            select type (decl => arena%entries(body_indices(i))%node)
            type is (declaration_node)
                if (.not. decl%is_array) cycle
                if (allocated(decl%var_name)) then
                    if (trim(decl%var_name) == trim(param_name)) then
                        dummy_is_array = .true.
                        return
                    end if
                end if
                if (decl%is_multi_declaration .and. allocated(decl%var_names)) then
                    do k = 1, size(decl%var_names)
                        if (trim(decl%var_names(k)) == trim(param_name)) then
                            dummy_is_array = .true.
                            return
                        end if
                    end do
                end if
            end select
        end do
    end function dummy_is_array

    logical function dummy_is_assumed_rank(arena, body_indices, param_name)
        ! Whether the body declaration of param_name is an assumed-rank dummy
        ! arr(..). Such a dummy binds to an actual of any rank, including a
        ! scalar (rank 0), so it must be excluded from the scalar-to-array
        ! rank-mismatch rejection.
        type(ast_arena_t), intent(in) :: arena
        integer, allocatable, intent(in) :: body_indices(:)
        character(len=*), intent(in) :: param_name
        integer :: i, dim_index
        logical :: named

        dummy_is_assumed_rank = .false.
        if (.not. allocated(body_indices)) return
        do i = 1, size(body_indices)
            if (.not. node_exists(arena, body_indices(i))) cycle
            select type (decl => arena%entries(body_indices(i))%node)
            type is (declaration_node)
                if (.not. decl%is_array) cycle
                named = .false.
                if (allocated(decl%var_name)) then
                    if (trim(decl%var_name) == trim(param_name)) named = .true.
                end if
                if (.not. named) cycle
                if (.not. allocated(decl%dimension_indices)) cycle
                if (size(decl%dimension_indices) /= 1) cycle
                dim_index = decl%dimension_indices(1)
                if (.not. node_exists(arena, dim_index)) cycle
                select type (dim_node => arena%entries(dim_index)%node)
                type is (array_bounds_node)
                    dummy_is_assumed_rank = dim_node%is_assumed_rank
                end select
                return
            end select
        end do
    end function dummy_is_assumed_rank

    logical function dummy_is_allocatable_array(arena, body_indices, param_name)
        ! A dummy is an allocatable array when a body declaration names it
        ! with both an array spec and the allocatable attribute. Selects the
        ! descriptor-aliasing ABI (W11 duplicate-declaration fix).
        type(ast_arena_t), intent(in) :: arena
        integer, allocatable, intent(in) :: body_indices(:)
        character(len=*), intent(in) :: param_name
        integer :: i, k

        dummy_is_allocatable_array = .false.
        if (.not. allocated(body_indices)) return
        do i = 1, size(body_indices)
            if (.not. node_exists(arena, body_indices(i))) cycle
            select type (decl => arena%entries(body_indices(i))%node)
            type is (declaration_node)
                if (.not. decl%is_array .or. .not. decl%is_allocatable) cycle
                if (allocated(decl%var_name)) then
                    if (trim(decl%var_name) == trim(param_name)) then
                        dummy_is_allocatable_array = .true.
                        return
                    end if
                end if
                if (decl%is_multi_declaration .and. allocated(decl%var_names)) then
                    do k = 1, size(decl%var_names)
                        if (trim(decl%var_names(k)) == trim(param_name)) then
                            dummy_is_allocatable_array = .true.
                            return
                        end if
                    end do
                end if
            end select
        end do
    end function dummy_is_allocatable_array

    logical function callee_dummy_is_array(arena, callee_name, param_pos)
        ! Whether the contained procedure named callee_name declares its
        ! param_pos-th (1-based) dummy as an array. Selects the array-argument
        ! base-pointer ABI on the caller side.
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: callee_name
        integer, intent(in) :: param_pos
        integer :: n
        character(len=:), allocatable :: name, name_err
        character(len=:), allocatable :: node_type
        type(function_def_node), pointer :: fn_node
        type(subroutine_def_node), pointer :: sb_node

        callee_dummy_is_array = .false.
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            node_type = get_node_type_at(arena, n)
            if (node_type == 'subroutine_def_node' .or. &
                node_type == 'subroutine_def') then
                sb_node => get_node_as_subroutine_def(arena, n)
                if (.not. associated(sb_node)) cycle
                if (.not. allocated(sb_node%name)) cycle
                if (.not. same_name(sb_node%name, callee_name)) cycle
                if (.not. allocated(sb_node%param_indices)) return
                if (param_pos < 1 .or. param_pos > size(sb_node%param_indices)) return
                call parameter_name(arena, sb_node%param_indices(param_pos), &
                                    name, name_err)
                if (len_trim(name_err) > 0) return
                callee_dummy_is_array = dummy_is_array(arena, &
                    sb_node%body_indices, name)
                return
            else if (node_type == 'function_def_node' .or. &
                     node_type == 'function_def') then
                fn_node => get_node_as_function_def(arena, n)
                if (.not. associated(fn_node)) cycle
                if (.not. allocated(fn_node%name)) cycle
                if (.not. same_name(fn_node%name, callee_name)) cycle
                if (.not. allocated(fn_node%param_indices)) return
                if (param_pos < 1 .or. param_pos > size(fn_node%param_indices)) return
                call parameter_name(arena, fn_node%param_indices(param_pos), &
                                    name, name_err)
                if (len_trim(name_err) > 0) return
                callee_dummy_is_array = dummy_is_array(arena, &
                    fn_node%body_indices, name)
                return
            end if
        end do
    end function callee_dummy_is_array

    logical function callee_dummy_is_assumed_rank(arena, callee_name, param_pos)
        ! Whether callee_name's param_pos-th dummy is an assumed-rank arr(..)
        ! dummy, which accepts an actual of any rank (including scalar) and so
        ! must not trip the scalar-to-array rank-mismatch check.
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: callee_name
        integer, intent(in) :: param_pos
        integer :: n
        character(len=:), allocatable :: name, name_err
        character(len=:), allocatable :: node_type
        type(function_def_node), pointer :: fn_node
        type(subroutine_def_node), pointer :: sb_node

        callee_dummy_is_assumed_rank = .false.
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            node_type = get_node_type_at(arena, n)
            if (node_type == 'subroutine_def_node' .or. &
                node_type == 'subroutine_def') then
                sb_node => get_node_as_subroutine_def(arena, n)
                if (.not. associated(sb_node)) cycle
                if (.not. allocated(sb_node%name)) cycle
                if (.not. same_name(sb_node%name, callee_name)) cycle
                if (.not. allocated(sb_node%param_indices)) return
                if (param_pos < 1 .or. param_pos > size(sb_node%param_indices)) return
                call parameter_name(arena, sb_node%param_indices(param_pos), &
                                    name, name_err)
                if (len_trim(name_err) > 0) return
                callee_dummy_is_assumed_rank = dummy_is_assumed_rank(arena, &
                    sb_node%body_indices, name)
                return
            else if (node_type == 'function_def_node' .or. &
                     node_type == 'function_def') then
                fn_node => get_node_as_function_def(arena, n)
                if (.not. associated(fn_node)) cycle
                if (.not. allocated(fn_node%name)) cycle
                if (.not. same_name(fn_node%name, callee_name)) cycle
                if (.not. allocated(fn_node%param_indices)) return
                if (param_pos < 1 .or. param_pos > size(fn_node%param_indices)) return
                call parameter_name(arena, fn_node%param_indices(param_pos), &
                                    name, name_err)
                if (len_trim(name_err) > 0) return
                callee_dummy_is_assumed_rank = dummy_is_assumed_rank(arena, &
                    fn_node%body_indices, name)
                return
            end if
        end do
    end function callee_dummy_is_assumed_rank

    logical function callee_dummy_is_allocatable_array(arena, callee_name, &
                                                        param_pos)
        ! Whether the contained procedure named callee_name declares its
        ! param_pos-th (1-based) dummy as an allocatable array. Selects the
        ! descriptor-aliasing ABI on the caller side (W11).
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: callee_name
        integer, intent(in) :: param_pos
        integer :: n
        character(len=:), allocatable :: name, name_err
        character(len=:), allocatable :: node_type
        type(function_def_node), pointer :: fn_node
        type(subroutine_def_node), pointer :: sb_node

        callee_dummy_is_allocatable_array = .false.
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            node_type = get_node_type_at(arena, n)
            if (node_type == 'subroutine_def_node' .or. &
                node_type == 'subroutine_def') then
                sb_node => get_node_as_subroutine_def(arena, n)
                if (.not. associated(sb_node)) cycle
                if (.not. allocated(sb_node%name)) cycle
                if (.not. same_name(sb_node%name, callee_name)) cycle
                if (.not. allocated(sb_node%param_indices)) return
                if (param_pos < 1 .or. param_pos > size(sb_node%param_indices)) return
                call parameter_name(arena, sb_node%param_indices(param_pos), &
                                    name, name_err)
                if (len_trim(name_err) > 0) return
                callee_dummy_is_allocatable_array = dummy_is_allocatable_array( &
                    arena, sb_node%body_indices, name)
                return
            else if (node_type == 'function_def_node' .or. &
                     node_type == 'function_def') then
                fn_node => get_node_as_function_def(arena, n)
                if (.not. associated(fn_node)) cycle
                if (.not. allocated(fn_node%name)) cycle
                if (.not. same_name(fn_node%name, callee_name)) cycle
                if (.not. allocated(fn_node%param_indices)) return
                if (param_pos < 1 .or. param_pos > size(fn_node%param_indices)) return
                call parameter_name(arena, fn_node%param_indices(param_pos), &
                                    name, name_err)
                if (len_trim(name_err) > 0) return
                callee_dummy_is_allocatable_array = dummy_is_allocatable_array( &
                    arena, fn_node%body_indices, name)
                return
            end if
        end do
    end function callee_dummy_is_allocatable_array

    logical function callee_dummy_is_character(arena, callee_name, param_pos)
        ! Whether the contained procedure named callee_name declares its
        ! param_pos-th (1-based) dummy as character(len=*). Used by the caller
        ! to select the character descriptor ABI.
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: callee_name
        integer, intent(in) :: param_pos
        integer :: n
        character(len=:), allocatable :: node_type
        type(function_def_node), pointer :: fn_node
        type(subroutine_def_node), pointer :: sb_node

        callee_dummy_is_character = .false.
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            node_type = get_node_type_at(arena, n)
            if (node_type == 'subroutine_def_node' .or. &
                node_type == 'subroutine_def') then
                sb_node => get_node_as_subroutine_def(arena, n)
                if (.not. associated(sb_node)) cycle
                if (allocated(sb_node%name)) then
                    if (same_name(sb_node%name, callee_name)) then
                        callee_dummy_is_character = param_at_is_character(arena, &
                            sb_node%param_indices, sb_node%body_indices, param_pos)
                        return
                    end if
                end if
            else if (node_type == 'function_def_node' .or. &
                     node_type == 'function_def') then
                fn_node => get_node_as_function_def(arena, n)
                if (.not. associated(fn_node)) cycle
                if (allocated(fn_node%name)) then
                    if (same_name(fn_node%name, callee_name)) then
                        callee_dummy_is_character = param_at_is_character(arena, &
                            fn_node%param_indices, fn_node%body_indices, param_pos)
                        return
                    end if
                end if
            end if
        end do
    end function callee_dummy_is_character

    logical function param_at_is_character(arena, param_indices, body_indices, &
                                           param_pos) result(is_char)
        type(ast_arena_t), intent(in) :: arena
        integer, allocatable, intent(in) :: param_indices(:)
        integer, allocatable, intent(in) :: body_indices(:)
        integer, intent(in) :: param_pos
        character(len=:), allocatable :: name, name_err

        is_char = .false.
        if (.not. allocated(param_indices)) return
        if (param_pos < 1 .or. param_pos > size(param_indices)) return
        call parameter_name(arena, param_indices(param_pos), name, name_err)
        if (len_trim(name_err) > 0) return
        is_char = dummy_is_character(arena, body_indices, name)
    end function param_at_is_character

    integer function callee_dummy_value_kind(arena, callee_name, param_pos) &
            result(value_kind)
        ! The scalar value kind of the param_pos-th (1-based) dummy of the
        ! contained procedure callee_name, read from its body declaration.
        ! Returns 0 when the procedure or its dummy declaration is not found, so
        ! the caller falls back to its own default. When the body declaration
        ! lacks a kind spec but the function return type is i64, infer i64 for
        ! the dummy as well (FortFront may drop the kind from body declarations).
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: callee_name
        integer, intent(in) :: param_pos
        integer :: n
        character(len=:), allocatable :: node_type
        type(function_def_node), pointer :: fn_node
        type(subroutine_def_node), pointer :: sb_node

        value_kind = 0
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            node_type = get_node_type_at(arena, n)
            if (node_type == 'subroutine_def_node' .or. &
                node_type == 'subroutine_def') then
                sb_node => get_node_as_subroutine_def(arena, n)
                if (.not. associated(sb_node)) cycle
                if (.not. allocated(sb_node%name)) cycle
                if (.not. same_name(sb_node%name, callee_name)) cycle
                value_kind = param_at_value_kind(arena, sb_node%param_indices, &
                    sb_node%body_indices, param_pos)
                return
            else if (node_type == 'function_def_node' .or. &
                     node_type == 'function_def') then
                fn_node => get_node_as_function_def(arena, n)
                if (.not. associated(fn_node)) cycle
                if (.not. allocated(fn_node%name)) cycle
                if (.not. same_name(fn_node%name, callee_name)) cycle
                value_kind = param_at_value_kind(arena, fn_node%param_indices, &
                    fn_node%body_indices, param_pos)
                if (value_kind == 0 .and. fn_return_type_is_i64(fn_node)) then
                    value_kind = VALUE_I64
                end if
                return
            end if
        end do
    end function callee_dummy_value_kind

    logical function fn_return_type_is_i64(node) result(is_i64)
        ! True when the function return type string indicates integer(8)/i64.
        ! Used as a fallback for parameter kind inference when the body
        ! declaration lacks a kind spec (FortFront drops the kind).
        type(function_def_node), intent(in) :: node
        character(len=:), allocatable :: base, kind_s
        integer :: lp, rp

        is_i64 = .false.
        if (.not. allocated(node%return_type)) return
        if (len_trim(node%return_type) == 0) return
        lp = index(node%return_type, '(')
        if (lp > 0) then
            base = lowercase_text(trim(adjustl(node%return_type(1:lp - 1))))
            rp = index(node%return_type, ')', back=.true.)
            if (rp > lp) then
                kind_s = lowercase_text(trim(adjustl(node%return_type(lp + 1:rp - 1))))
            else
                return
            end if
        else
            base = lowercase_text(trim(adjustl(node%return_type)))
            return
        end if
        if (base == 'integer') then
            is_i64 = (kind_s == '8' .or. kind_s == 'int64' .or. &
                      kind_s == 'int64_t' .or. kind_s == 'c_int64')
        end if
    end function fn_return_type_is_i64

    integer function param_at_value_kind(arena, param_indices, body_indices, &
                                         param_pos) result(value_kind)
        type(ast_arena_t), intent(in) :: arena
        integer, allocatable, intent(in) :: param_indices(:)
        integer, allocatable, intent(in) :: body_indices(:)
        integer, intent(in) :: param_pos
        character(len=:), allocatable :: name, name_err, kind_err
        integer :: i, k

        value_kind = 0
        if (.not. allocated(param_indices)) return
        if (param_pos < 1 .or. param_pos > size(param_indices)) return
        call parameter_name(arena, param_indices(param_pos), name, name_err)
        if (len_trim(name_err) > 0) return
        if (.not. allocated(body_indices)) return
        do i = 1, size(body_indices)
            if (.not. node_exists(arena, body_indices(i))) cycle
            select type (decl => arena%entries(body_indices(i))%node)
            type is (declaration_node)
                if (decl%is_array) cycle
                if (allocated(decl%var_name)) then
                    if (trim(decl%var_name) == trim(name)) then
                        call declaration_value_kind(decl, value_kind, kind_err)
                        if (len_trim(kind_err) > 0) value_kind = 0
                        return
                    end if
                end if
                if (decl%is_multi_declaration .and. allocated(decl%var_names)) then
                    do k = 1, size(decl%var_names)
                        if (trim(decl%var_names(k)) == trim(name)) then
                            call declaration_value_kind(decl, value_kind, kind_err)
                            if (len_trim(kind_err) > 0) value_kind = 0
                            return
                        end if
                    end do
                end if
            end select
        end do
    end function param_at_value_kind

    function procedure_emit_name(fortran_name, bind_c_clause) result(emit_name)
        ! The LIRIC symbol a contained procedure is emitted under: its
        ! bind(c, name="...") C name when present, otherwise its Fortran name.
        character(len=*), intent(in) :: fortran_name
        character(len=:), allocatable, intent(in) :: bind_c_clause
        character(len=:), allocatable :: emit_name

        if (has_bind_c(bind_c_clause)) then
            call bind_c_name(bind_c_clause, fortran_name, emit_name)
        else
            emit_name = lowercase_text(trim(fortran_name))
        end if
    end function procedure_emit_name

    function nested_procedure_emit_name(arena, procedure_node_index, &
                                        fortran_name, bind_c_clause) result(emit_name)
        ! LIRIC procedures share one flat namespace. Give a contained
        ! procedure an owner-qualified name so same-named procedures in sibling
        ! hosts remain distinct (#330).
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: procedure_node_index
        character(len=*), intent(in) :: fortran_name
        character(len=:), allocatable, intent(in) :: bind_c_clause
        character(len=:), allocatable :: emit_name, base_name
        character(len=32) :: host_text, procedure_text
        integer :: host_node_index

        host_node_index = enclosing_procedure_node(arena, procedure_node_index)
        if (host_node_index <= 0) then
            emit_name = procedure_emit_name(fortran_name, bind_c_clause)
            return
        end if

        write (host_text, '(I0)') host_node_index
        write (procedure_text, '(I0)') procedure_node_index
        base_name = procedure_emit_name(fortran_name, bind_c_clause)
        emit_name = '.ffc.nested.'//trim(host_text)//'.'//trim(procedure_text)// &
                    '.'//trim(base_name)
    end function nested_procedure_emit_name

    function procedure_module(arena, fortran_name) result(module_name)
        ! The module a procedure belongs to for mangling, or empty if not
        ! module-defined. A submodule's module procedure is owned by the
        ! ancestor module named in its parent reference (#292), so the call site
        ! mangles to __<parent>_MOD_<name> just like the interface it implements.
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: fortran_name
        character(len=:), allocatable :: module_name
        character(len=:), allocatable :: node_type, mod_name, parent_module, &
                                         ff_error
        integer, allocatable :: declaration_indices(:), procedure_indices(:)
        integer :: n, p

        call set_empty(module_name)
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            node_type = get_node_type_at(arena, n)
            if (node_type == 'module_node' .or. node_type == 'module') then
                call get_module_body_info(arena, n, mod_name, declaration_indices, &
                                          procedure_indices, ff_error)
            else if (node_type == 'submodule_node' .or. &
                     node_type == 'submodule') then
                call get_submodule_body_info(arena, n, mod_name, parent_module, &
                                             declaration_indices, &
                                             procedure_indices, ff_error)
            else
                cycle
            end if
            if (len_trim(ff_error) > 0) cycle
            if (size(procedure_indices) == 0) cycle
            do p = 1, size(procedure_indices)
                if (procedure_node_named(arena, procedure_indices(p), fortran_name)) then
                    if (node_type == 'submodule_node' .or. &
                        node_type == 'submodule') then
                        module_name = trim(parent_module)
                    else
                        module_name = trim(mod_name)
                    end if
                    return
                end if
            end do
        end do
    end function procedure_module

    logical function procedure_node_named(arena, node_index, fortran_name)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        character(len=*), intent(in) :: fortran_name
        character(len=:), allocatable :: node_type
        type(function_def_node), pointer :: fn_node
        type(subroutine_def_node), pointer :: sb_node

        procedure_node_named = .false.
        if (.not. node_exists(arena, node_index)) return
        node_type = get_node_type_at(arena, node_index)
        if (node_type == 'function_def_node' .or. node_type == 'function_def') then
            fn_node => get_node_as_function_def(arena, node_index)
            if (associated(fn_node) .and. allocated(fn_node%name)) then
                procedure_node_named = same_name(fn_node%name, fortran_name)
            end if
        else if (node_type == 'subroutine_def_node' .or. &
                 node_type == 'subroutine_def') then
            sb_node => get_node_as_subroutine_def(arena, node_index)
            if (associated(sb_node) .and. allocated(sb_node%name)) then
                procedure_node_named = same_name(sb_node%name, fortran_name)
            end if
        end if
    end function procedure_node_named

    function module_procedure_mangled(module_name, proc_name) result(mangled)
        ! gfortran convention: __<module>_MOD_<procedure>.
        character(len=*), intent(in) :: module_name, proc_name
        character(len=:), allocatable :: mangled

        mangled = '__'//trim(lowercase_text(module_name))//'_MOD_'// &
                  trim(lowercase_text(proc_name))
    end function module_procedure_mangled

    function call_emit_name(arena, fortran_name, current_context) result(emit_name)
        ! Resolve the LIRIC symbol to call: a module-defined procedure mangles
        ! to __<module>_MOD_<name>; a bind(c) procedure uses its C name; else
        ! the Fortran name.
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: fortran_name
        type(lowering_context_t), intent(in), optional :: current_context
        character(len=:), allocatable :: emit_name, module_name
        character(len=:), allocatable :: node_type
        integer :: nested_node_index
        integer :: n
        type(function_def_node), pointer :: fn_node
        type(subroutine_def_node), pointer :: sb_node

        if (present(current_context)) then
            nested_node_index = visible_nested_procedure_node(current_context, &
                                                               fortran_name)
            if (nested_node_index > 0) then
                emit_name = trim(fortran_name)
                if (node_exists(arena, nested_node_index)) then
                    node_type = get_node_type_at(arena, nested_node_index)
                    if (node_type == 'subroutine_def_node' .or. &
                        node_type == 'subroutine_def') then
                        sb_node => get_node_as_subroutine_def(arena, nested_node_index)
                        if (associated(sb_node)) emit_name = &
                            nested_procedure_emit_name(arena, nested_node_index, &
                                fortran_name, sb_node%bind_c_clause)
                    else if (node_type == 'function_def_node' .or. &
                             node_type == 'function_def') then
                        fn_node => get_node_as_function_def(arena, nested_node_index)
                        if (associated(fn_node)) emit_name = &
                            nested_procedure_emit_name(arena, nested_node_index, &
                                fortran_name, fn_node%bind_c_clause)
                    end if
                end if
                return
            end if
        end if
        module_name = procedure_module(arena, fortran_name)
        if (len_trim(module_name) > 0) then
            emit_name = module_procedure_mangled(module_name, fortran_name)
            return
        end if
        emit_name = trim(fortran_name)
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            node_type = get_node_type_at(arena, n)
            if (node_type == 'subroutine_def_node' .or. &
                node_type == 'subroutine_def') then
                sb_node => get_node_as_subroutine_def(arena, n)
                if (associated(sb_node) .and. allocated(sb_node%name)) then
                    if (same_name(sb_node%name, fortran_name)) then
                        emit_name = procedure_emit_name(fortran_name, &
                                                        sb_node%bind_c_clause)
                        return
                    end if
                end if
            else if (node_type == 'function_def_node' .or. &
                     node_type == 'function_def') then
                fn_node => get_node_as_function_def(arena, n)
                if (associated(fn_node) .and. allocated(fn_node%name)) then
                    if (same_name(fn_node%name, fortran_name)) then
                        emit_name = procedure_emit_name(fortran_name, &
                                                        fn_node%bind_c_clause)
                        return
                    end if
                end if
            end if
        end do
    end function call_emit_name

    integer function visible_nested_procedure_node(context, name) result(index)
        ! Return the procedure declared directly in the current procedure's
        ! scope. The AST binding, rather than a global text-name table, decides
        ! whether it shadows a USE-associated procedure (#330).
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: name
        character(len=:), allocatable :: candidate_name
        integer :: i

        index = 0
        if (context%current_proc_node_index <= 0) return
        do i = 1, context%arena%size
            if (.not. procedure_unit_name(context%arena, i, candidate_name)) cycle
            if (.not. same_name(candidate_name, name)) cycle
            if (.not. procedure_scope_is(context%arena, i, &
                                         context%current_proc_node_index)) cycle
            index = i
            return
        end do
    end function visible_nested_procedure_node

    logical function procedure_scope_is(arena, procedure_node_index, scope_index) &
            result(is_nested)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: procedure_node_index
        integer, intent(in) :: scope_index
        character(len=:), allocatable :: node_type
        integer :: parent

        is_nested = .false.
        if (.not. node_exists(arena, procedure_node_index)) return
        parent = arena%entries(procedure_node_index)%parent_index
        do while (parent > 0)
            if (.not. node_exists(arena, parent)) return
            node_type = get_node_type_at(arena, parent)
            if (node_type == 'function_def_node' .or. node_type == 'function_def' .or. &
                node_type == 'subroutine_def_node' .or. &
                node_type == 'subroutine_def' .or. node_type == 'program_node' .or. &
                node_type == 'program' .or. node_type == 'module_node' .or. &
                node_type == 'module') then
                is_nested = parent == scope_index
                return
            end if
            parent = arena%entries(parent)%parent_index
        end do
    end function procedure_scope_is

    integer function enclosing_procedure_node(arena, procedure_node_index) &
            result(scope_index)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: procedure_node_index
        character(len=:), allocatable :: node_type
        integer :: parent

        scope_index = 0
        if (.not. node_exists(arena, procedure_node_index)) return
        parent = arena%entries(procedure_node_index)%parent_index
        do while (parent > 0)
            if (.not. node_exists(arena, parent)) return
            node_type = get_node_type_at(arena, parent)
            if (node_type == 'function_def_node' .or. node_type == 'function_def' .or. &
                node_type == 'subroutine_def_node' .or. &
                node_type == 'subroutine_def') then
                scope_index = parent
                return
            else if (node_type == 'program_node' .or. node_type == 'program' .or. &
                     node_type == 'module_node' .or. node_type == 'module') then
                return
            end if
            parent = arena%entries(parent)%parent_index
        end do
    end function enclosing_procedure_node

    integer function containing_procedure_node(arena, node_index) result(scope_index)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        character(len=:), allocatable :: node_type
        integer :: parent

        scope_index = 0
        if (.not. node_exists(arena, node_index)) return
        parent = node_index
        do while (parent > 0)
            if (.not. node_exists(arena, parent)) return
            node_type = get_node_type_at(arena, parent)
            if (node_type == 'function_def_node' .or. node_type == 'function_def' .or. &
                node_type == 'subroutine_def_node' .or. &
                node_type == 'subroutine_def') then
                scope_index = parent
                return
            else if (node_type == 'program_node' .or. node_type == 'program' .or. &
                     node_type == 'module_node' .or. node_type == 'module') then
                return
            end if
            parent = arena%entries(parent)%parent_index
        end do
    end function containing_procedure_node

    logical function procedure_is_internal(arena, procedure_node_index) result(is_internal)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: procedure_node_index
        character(len=:), allocatable :: node_type
        integer :: parent

        is_internal = .false.
        if (.not. node_exists(arena, procedure_node_index)) return
        parent = arena%entries(procedure_node_index)%parent_index
        do while (parent > 0)
            if (.not. node_exists(arena, parent)) return
            node_type = get_node_type_at(arena, parent)
            if (node_type == 'function_def_node' .or. node_type == 'function_def' .or. &
                node_type == 'subroutine_def_node' .or. &
                node_type == 'subroutine_def' .or. node_type == 'program_node' .or. &
                node_type == 'program') then
                is_internal = .true.
                return
            else if (node_type == 'module_node' .or. node_type == 'module') then
                return
            end if
            parent = arena%entries(parent)%parent_index
        end do
    end function procedure_is_internal

    logical function binding_scope_is_module(arena, scope_index) result(is_module)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: scope_index
        character(len=:), allocatable :: node_type
        integer :: parent

        is_module = .false.
        if (.not. node_exists(arena, scope_index)) return
        parent = scope_index
        do while (parent > 0)
            if (.not. node_exists(arena, parent)) return
            node_type = get_node_type_at(arena, parent)
            if (node_type == 'module_node' .or. node_type == 'module') then
                is_module = .true.
                return
            else if (node_type == 'function_def_node' .or. &
                     node_type == 'function_def' .or. &
                     node_type == 'subroutine_def_node' .or. &
                     node_type == 'subroutine_def' .or. node_type == 'program_node' .or. &
                     node_type == 'program') then
                return
            end if
            parent = arena%entries(parent)%parent_index
        end do
    end function binding_scope_is_module

    integer function procedure_value_kind(context, node_index) result(value_kind)
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: node_index
        character(len=:), allocatable :: node_type, error_msg
        type(function_def_node), pointer :: fn_node
        type(subroutine_def_node), pointer :: sb_node
        integer :: derived_type_index

        value_kind = 0
        if (.not. node_exists(context%arena, node_index)) return
        node_type = get_node_type_at(context%arena, node_index)
        if (node_type == 'function_def_node' .or. node_type == 'function_def') then
            fn_node => get_node_as_function_def(context%arena, node_index)
            if (.not. associated(fn_node)) return
            call function_value_kind(fn_node, context, value_kind, &
                                     derived_type_index, error_msg, node_index)
            if (len_trim(error_msg) > 0) value_kind = 0
        else if (node_type == 'subroutine_def_node' .or. &
                 node_type == 'subroutine_def') then
            sb_node => get_node_as_subroutine_def(context%arena, node_index)
            if (associated(sb_node)) value_kind = VALUE_SUBROUTINE
        end if
    end function procedure_value_kind

    subroutine define_reference_parameters(arena, param_indices, context, &
                                            error_msg, body_indices, &
                                            param_offset, return_value_kind)
        type(ast_arena_t), intent(in) :: arena
        integer, allocatable, intent(in) :: param_indices(:)
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer, allocatable, intent(in), optional :: body_indices(:)
        integer, intent(in), optional :: param_offset
        integer, intent(in), optional :: return_value_kind
        character(len=:), allocatable :: name
        integer :: offset
        integer :: i
        integer :: value_kind
        integer :: rvk

        call set_empty(error_msg)
        offset = 0
        if (present(param_offset)) offset = param_offset
        rvk = 0
        if (present(return_value_kind)) rvk = return_value_kind
        if (.not. allocated(param_indices)) return
        do i = 1, size(param_indices)
            call parameter_name(arena, param_indices(i), name, error_msg)
            if (len_trim(error_msg) > 0) return
            value_kind = VALUE_I32
            if (present(body_indices)) then
                value_kind = param_at_value_kind(arena, param_indices, &
                    body_indices, i)
                if (value_kind == 0) then
                    ! Body declaration lacks kind spec; fall back to the
                    ! function's return type when it indicates i64.
                    if (rvk == VALUE_I64) then
                        value_kind = VALUE_I64
                    else
                        value_kind = VALUE_I32
                    end if
                end if
            end if
            call define_parameter_symbol(context, name, i - 1 + offset, &
                                         value_kind, error_msg, &
                                         parameter_node_index=param_indices(i))
            if (len_trim(error_msg) > 0) return
        end do
    end subroutine define_reference_parameters

    subroutine parameter_name(arena, node_index, name, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        character(len=:), allocatable, intent(out) :: name
        character(len=:), allocatable, intent(out) :: error_msg

        if (.not. node_exists(arena, node_index)) then
            error_msg = 'parameter index does not reference an AST node'
            call set_empty(name)
            return
        end if

        if (is_identifier(arena, node_index)) then
            call get_identifier_name(arena, node_index, name, error_msg)
            return
        end if

        select type (node => arena%entries(node_index)%node)
        type is (parameter_declaration_node)
            name = node%name
            call set_empty(error_msg)
        class default
            error_msg = 'direct LIRIC session function parameter needs a name'
            call set_empty(name)
        end select
    end subroutine parameter_name

    subroutine define_parameter_symbol(context, name, param_index, value_kind, &
                                      error_msg, parameter_node_index)
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: name
        integer, intent(in) :: param_index
        integer, intent(in) :: value_kind
        character(len=:), allocatable, intent(out) :: error_msg
        integer, intent(in), optional :: parameter_node_index
        integer :: index
        type(declaration_binding_t) :: binding
        character(len=:), allocatable :: binding_error
        logical :: has_binding

        has_binding = .false.
        if (present(parameter_node_index)) then
            call resolve_name_at_node(context%arena, parameter_node_index, name, &
                                      binding, binding_error)
            if (len_trim(binding_error) == 0 .and. binding%found) &
                has_binding = .true.
        end if
        if (has_binding) then
            index = find_symbol_for_binding(context, binding)
        else
            index = find_symbol_compat(context, name)
        end if
        if (index > 0 .and. .not. has_binding) then
            error_msg = 'duplicate function parameter: '//trim(name)
            return
        end if
        if (index <= 0) then
            call grow_symbols(context)
            index = context%symbol_count + 1
            context%symbol_count = index
        end if
        context%symbols(index)%name = trim(name)
        context%symbols(index)%value_kind = value_kind
        context%symbols(index)%address = ptr_param(context%session, param_index)
        context%symbols(index)%has_address = .true.
        context%symbols(index)%is_reference = .true.
        context%symbols(index)%value = i32_immediate(context%session, 0_c_int64_t)
        context%symbols(index)%is_parameter = .true.
        if (has_binding) then
            context%symbols(index)%has_binding = .true.
            context%symbols(index)%binding_declaration_index = &
                binding%declaration_node_index
            context%symbols(index)%binding_entity_index = &
                binding%declaration_entity_index
            context%symbols(index)%binding_scope_index = binding%scope_node_index
            call context%binding_table%insert_binding( &
                binding%declaration_node_index, binding%declaration_entity_index, &
                binding%scope_node_index, index)
        end if
        call set_empty(error_msg)
    end subroutine define_parameter_symbol

    subroutine ensure_function_names_capacity(context, required_index)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: required_index
        character(len=64), allocatable :: tmp_names(:)
        integer, allocatable :: tmp_kinds(:)
        integer, allocatable :: tmp_counts(:)
        integer, allocatable :: tmp_indices(:)
        integer :: old_size, new_size

        if (required_index <= size(context%function_names)) return
        old_size = size(context%function_names)
        new_size = max(2 * old_size, required_index)
        call move_alloc(context%function_names, tmp_names)
        call move_alloc(context%function_value_kinds, tmp_kinds)
        call move_alloc(context%function_param_counts, tmp_counts)
        allocate(context%function_names(new_size))
        allocate(context%function_value_kinds(new_size))
        allocate(context%function_param_counts(new_size))
        context%function_value_kinds = VALUE_I32
        context%function_param_counts = 0
        context%function_names(1:old_size) = tmp_names
        context%function_value_kinds(1:old_size) = tmp_kinds
        context%function_param_counts(1:old_size) = tmp_counts
        if (allocated(context%function_node_indices)) then
            call move_alloc(context%function_node_indices, tmp_indices)
            allocate(context%function_node_indices(new_size))
            context%function_node_indices = 0
            context%function_node_indices(1:size(tmp_indices)) = tmp_indices
        else
            allocate(context%function_node_indices(new_size))
            context%function_node_indices = 0
        end if
    end subroutine ensure_function_names_capacity

    subroutine ensure_derived_types_capacity(context, required_index)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: required_index

        if (required_index <= size(context%derived_types)) return
        call grow_derived_types(context)
        if (required_index <= size(context%derived_types)) return
        allocate(context%derived_types(required_index), &
                 source=context%derived_types)
    end subroutine ensure_derived_types_capacity

    subroutine ensure_all_context_arrays_allocated(context, min_symbols, &
                                                    min_derived_types, &
                                                    min_module_exports, &
                                                    min_functions)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: min_symbols
        integer, intent(in) :: min_derived_types
        integer, intent(in) :: min_module_exports
        integer, intent(in) :: min_functions

        if (.not. allocated(context%symbols)) then
            allocate(context%symbols(max(min_symbols, 16)))
        end if
        if (.not. allocated(context%derived_types)) then
            allocate(context%derived_types(max(min_derived_types, 8)))
        end if
        if (.not. allocated(context%module_exports)) then
            allocate(context%module_exports(max(min_module_exports, 4)))
        end if
        if (.not. allocated(context%function_names)) then
            allocate(context%function_names(max(min_functions, 8)))
            allocate(context%function_value_kinds(max(min_functions, 8)))
            allocate(context%function_param_counts(max(min_functions, 8)))
            context%function_value_kinds = VALUE_I32
            context%function_param_counts = 0
        end if
    end subroutine ensure_all_context_arrays_allocated
