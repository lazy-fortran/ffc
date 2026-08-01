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
        integer :: visible_param_count
        integer :: altret_count
        integer :: hidden_altret_count
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
        ! Assumed-shape dummies are passed as one descriptor pointer each in
        ! their own visible parameter slot (#334), so no hidden extent
        ! parameters are appended to the signature.
        hidden_extent_count = 0
        ! Each `*` dummy is a positional alternate-return slot, not a passed
        ! argument: the slots collapse into one hidden trailing selector
        ! parameter (#353).
        altret_count = alt_return_slot_count(arena, node%param_indices)
        visible_param_count = param_count - altret_count
        hidden_altret_count = 0
        if (altret_count > 0) hidden_altret_count = 1
        context%altret_slot_count = altret_count
        if (altret_count > 0) then
            context%altret_param_index = visible_param_count + hidden_extent_count
        end if
        if (present(emit_name_override)) then
            emit_name = trim(emit_name_override)
        else
            emit_name = nested_procedure_emit_name(arena, &
                context%current_proc_node_index, node%name, node%bind_c_clause)
        end if
        if (.not. begin_void_subroutine(context%session, emit_name, &
                visible_param_count + hidden_extent_count + hidden_altret_count, &
                error_msg)) return

        call materialize_host_character_globals(context, error_msg)
        if (len_trim(error_msg) > 0) return

        call define_reference_parameters(arena, node%param_indices, context, &
                                         error_msg, node%body_indices)
        if (len_trim(error_msg) > 0) return

        ! Rebind every assumed-shape dummy from its descriptor parameter.
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

        if (.not. context%current_block_terminated) then
            ! The procedure ends execution here, so its owned local derived
            ! values reach the end of their lifetime (#403).
            call emit_scope_exit_finalizers(context, error_msg)
            if (len_trim(error_msg) > 0) return
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

    subroutine dummy_declared_derived_type(arena, body_indices, param_name, &
                                           want_class, type_name)
        ! Declared type name of a derived dummy, empty when the dummy is not
        ! declared with the requested spelling. want_class selects class(t),
        ! the polymorphic-derived spelling the class descriptor ABI covers;
        ! otherwise type(t), the monomorphic by-reference spelling.
        type(ast_arena_t), intent(in) :: arena
        integer, allocatable, intent(in) :: body_indices(:)
        character(len=*), intent(in) :: param_name
        logical, intent(in) :: want_class
        character(len=:), allocatable, intent(out) :: type_name
        integer :: i, k
        logical :: matches

        type_name = ''
        if (.not. allocated(body_indices)) return
        do i = 1, size(body_indices)
            if (.not. node_exists(arena, body_indices(i))) cycle
            select type (decl => arena%entries(body_indices(i))%node)
            type is (declaration_node)
                if (.not. allocated(decl%type_name)) cycle
                if (want_class) then
                    if (.not. is_class_derived_type_spec(decl%type_name)) cycle
                else
                    if (is_class_derived_type_spec(decl%type_name)) cycle
                    if (.not. is_derived_type_spec(decl%type_name)) cycle
                end if
                ! An allocatable or pointer class dummy aliases the caller's
                ! own allocatable/pointer descriptor; that ABI is unchanged
                ! here and gains its dynamic identity with #421.
                if (decl%is_allocatable) cycle
                if (decl%is_pointer) cycle
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
                    call extracted_derived_type_name(decl%type_name, type_name)
                    return
                end if
            end select
        end do
    end subroutine dummy_declared_derived_type

    subroutine callee_dummy_declared_derived_type(arena, callee_name, &
                                                  param_pos, want_class, &
                                                  type_name)
        ! Declared type name of the callee's param_pos-th dummy when that dummy
        ! carries the requested derived spelling (class(t) or type(t)); empty
        ! otherwise.
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: callee_name
        integer, intent(in) :: param_pos
        logical, intent(in) :: want_class
        character(len=:), allocatable, intent(out) :: type_name
        integer :: n
        character(len=:), allocatable :: name, name_err
        character(len=:), allocatable :: node_type
        type(function_def_node), pointer :: fn_node
        type(subroutine_def_node), pointer :: sb_node

        type_name = ''
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
                        call dummy_declared_derived_type(arena, &
                            sb_node%body_indices, name, want_class, type_name)
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
                        call dummy_declared_derived_type(arena, &
                            fn_node%body_indices, name, want_class, type_name)
                        return
                    end if
                end if
            end if
        end do
    end subroutine callee_dummy_declared_derived_type

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

    integer function callee_dummy_value_kind(arena, callee_name, param_pos, &
                                             context) result(value_kind)
        ! The scalar value kind of the param_pos-th (1-based) dummy of the
        ! contained procedure callee_name, read from its body declaration.
        ! Returns 0 when the procedure or its dummy declaration is not found, so
        ! the caller falls back to its own default. When the body declaration
        ! lacks a kind spec but the function return type is i64, infer i64 for
        ! the dummy as well (FortFront may drop the kind from body declarations).
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: callee_name
        integer, intent(in) :: param_pos
        type(lowering_context_t), intent(in), optional :: context
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
                    sb_node%body_indices, param_pos, context)
                return
            else if (node_type == 'function_def_node' .or. &
                     node_type == 'function_def') then
                fn_node => get_node_as_function_def(arena, n)
                if (.not. associated(fn_node)) cycle
                if (.not. allocated(fn_node%name)) cycle
                if (.not. same_name(fn_node%name, callee_name)) cycle
                value_kind = param_at_value_kind(arena, fn_node%param_indices, &
                    fn_node%body_indices, param_pos, context)
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
                                         param_pos, context) result(value_kind)
        type(ast_arena_t), intent(in) :: arena
        integer, allocatable, intent(in) :: param_indices(:)
        integer, allocatable, intent(in) :: body_indices(:)
        integer, intent(in) :: param_pos
        ! The dialect decides what a kind-less `real` dummy means, so the
        ! caller passes its context whenever one is in scope (#571).
        type(lowering_context_t), intent(in), optional :: context
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
                ! An array declaration names a shape, not a scalar kind: the
                ! caller reads the rank separately, and reporting a scalar kind
                ! here would hide it. Say "no scalar kind" and stop looking, so
                ! the parameter-entry fallback below cannot answer for it.
                if (decl%is_array) then
                    if (declaration_names_entity(decl, name)) return
                    cycle
                end if
                if (allocated(decl%var_name)) then
                    if (trim(decl%var_name) == trim(name)) then
                        call declaration_value_kind(decl, value_kind, kind_err, &
                                                    context)
                        if (len_trim(kind_err) > 0) value_kind = 0
                        return
                    end if
                end if
                if (decl%is_multi_declaration .and. allocated(decl%var_names)) then
                    do k = 1, size(decl%var_names)
                        if (trim(decl%var_names(k)) == trim(name)) then
                            call declaration_value_kind(decl, value_kind, kind_err, &
                                                    context)
                            if (len_trim(kind_err) > 0) value_kind = 0
                            return
                        end if
                    end do
                end if
            end select
        end do
        ! No specification-part declaration names this dummy. A Lazy unit's
        ! monomorphized specialization carries the dummy's resolved type on
        ! the parameter-list entry itself and adds no declaration node, so
        ! read the kind from there rather than defaulting to i32 (#651).
        value_kind = parameter_entry_value_kind(arena, param_indices(param_pos), &
                                                context)
    end function param_at_value_kind

    logical function declaration_names_entity(decl, name) result(names_it)
        ! Whether this declaration declares the named entity, single or multi.
        type(declaration_node), intent(in) :: decl
        character(len=*), intent(in) :: name
        integer :: k

        names_it = .false.
        if (allocated(decl%var_name)) then
            if (trim(decl%var_name) == trim(name)) then
                names_it = .true.
                return
            end if
        end if
        if (.not. decl%is_multi_declaration) return
        if (.not. allocated(decl%var_names)) return
        do k = 1, size(decl%var_names)
            if (trim(decl%var_names(k)) == trim(name)) then
                names_it = .true.
                return
            end if
        end do
    end function declaration_names_entity

    integer function parameter_entry_value_kind(arena, param_index, context) &
        result(value_kind)
        ! The value kind recorded on a parameter-list entry itself. The dialect
        ! decides what a kind-less `real` dummy means, so a Lazy unit's
        ! kind-less real dummy is 8 bytes like every other kind-less real
        ! there (#438, #651).
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: param_index
        type(lowering_context_t), intent(in), optional :: context
        integer :: kind_value
        character(len=:), allocatable :: type_text

        value_kind = 0
        if (.not. node_exists(arena, param_index)) return
        select type (entry => arena%entries(param_index)%node)
        type is (parameter_declaration_node)
            if (entry%is_array) return
            if (entry%resolved_rank > 0) return
            if (allocated(entry%dimension_indices)) then
                if (size(entry%dimension_indices) > 0) return
            end if
            if (entry%resolved_type_kind <= 0) return
            kind_value = entry%resolved_kind_value
            if (present(context)) &
                call apply_lazy_real_kind(context, entry%resolved_type_kind, &
                                          entry%has_kind, kind_value)
            type_text = ''
            if (allocated(entry%type_name)) type_text = trim(entry%type_name)
            value_kind = declaration_metadata_value_kind( &
                entry%resolved_type_kind, kind_value, type_text)
        end select
    end function parameter_entry_value_kind

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
            ! An interface block in a procedure's specification part declares
            ! an external procedure unless the name is also a dummy argument of
            ! that procedure; the external one keeps its plain link name rather
            ! than the nested-procedure mangling (#582).
            if (nested_node_index > 0) then
                if (procedure_node_is_interface_signature(arena, &
                                                          nested_node_index)) then
                    if (.not. enclosing_dummy_argument_name(arena, &
                            current_context, fortran_name)) nested_node_index = 0
                end if
            end if
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
        integer :: slot
        integer :: value_kind
        integer :: rvk

        call set_empty(error_msg)
        offset = 0
        if (present(param_offset)) offset = param_offset
        rvk = 0
        if (present(return_value_kind)) rvk = return_value_kind
        if (.not. allocated(param_indices)) return
        slot = 0
        do i = 1, size(param_indices)
            ! An alternate-return slot is positional, carries no name and no
            ! passed argument; it never occupies a parameter position (#353).
            if (is_alternate_return_dummy(arena, param_indices(i))) cycle
            slot = slot + 1
            call parameter_name(arena, param_indices(i), name, error_msg)
            if (len_trim(error_msg) > 0) return
            ! A dummy whose signature comes from an interface body in this
            ! scope is a procedure dummy: it carries a callable address, not
            ! data storage (#467, #606).
            if (present(body_indices)) then
                if (dummy_is_procedure(arena, body_indices, name)) then
                    call define_procedure_dummy_symbol(context, name, &
                                                       slot - 1 + offset, &
                                                       error_msg)
                    if (len_trim(error_msg) > 0) return
                    cycle
                end if
            end if
            value_kind = VALUE_I32
            if (present(body_indices)) then
                value_kind = param_at_value_kind(arena, param_indices, &
                    body_indices, i, context)
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
            call define_parameter_symbol(context, name, slot - 1 + offset, &
                                         value_kind, error_msg, &
                                         parameter_node_index=param_indices(i))
            if (len_trim(error_msg) > 0) return
        end do
    end subroutine define_reference_parameters

    subroutine split_alt_return_args(arena, arg_indices, alt_labels)
        ! Remove `*label` alternate-return specifiers from an actual argument
        ! list and report their statement labels in call order (#353).
        type(ast_arena_t), intent(in) :: arena
        integer, allocatable, intent(inout) :: arg_indices(:)
        integer, allocatable, intent(out) :: alt_labels(:)
        integer, allocatable :: plain(:)
        integer, allocatable :: labels(:)
        integer :: i, plain_count, label_count, label

        allocate(alt_labels(0))
        if (.not. allocated(arg_indices)) return
        if (size(arg_indices) == 0) return
        allocate(plain(size(arg_indices)))
        allocate(labels(size(arg_indices)))
        plain_count = 0
        label_count = 0
        do i = 1, size(arg_indices)
            label = get_alternate_return_label(arena, arg_indices(i))
            if (label > 0) then
                label_count = label_count + 1
                labels(label_count) = label
            else
                plain_count = plain_count + 1
                plain(plain_count) = arg_indices(i)
            end if
        end do
        if (label_count == 0) return
        deallocate(alt_labels)
        allocate(alt_labels(label_count))
        alt_labels = labels(1:label_count)
        deallocate(arg_indices)
        allocate(arg_indices(plain_count))
        arg_indices = plain(1:plain_count)
    end subroutine split_alt_return_args

    subroutine emit_alt_return_call(arena, node_index, emit_name, args, &
                                    alt_labels, copyback_indices, context, &
                                    error_msg)
        ! Call a subroutine with alternate-return specifiers: pass a zeroed
        ! hidden selector slot as the trailing argument, then branch to the
        ! n-th caller label when the callee stored n (#353). Selector 0 (a
        ! plain RETURN) falls through to the statement after the CALL.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        character(len=*), intent(in) :: emit_name
        type(lr_operand_desc_t), intent(in) :: args(:)
        integer, intent(in) :: alt_labels(:)
        integer, intent(in) :: copyback_indices(:)
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t), allocatable :: call_args(:)
        type(lr_operand_desc_t) :: slot, selector, ordinal, condition
        integer(c_int32_t) :: target_block, next_block
        character(len=16) :: label_text
        logical :: found
        integer :: i

        call set_empty(error_msg)
        if (.not. context%in_labeled_body) then
            call unsupported_feature_error('alternate return call', &
                get_node_line(arena, node_index), &
                get_node_column(arena, node_index), &
                'a call with alternate-return arguments is only lowered '// &
                'inside a body carrying statement labels', error_msg)
            return
        end if
        if (.not. emit_i32_alloca(context%session, slot, error_msg)) return
        if (.not. emit_i32_store(context%session, &
                i32_immediate(context%session, 0_c_int64_t), slot, &
                error_msg)) return
        allocate(call_args(size(args) + 1))
        if (size(args) > 0) call_args(1:size(args)) = args
        call_args(size(args) + 1) = slot
        if (.not. emit_void_call(context%session, emit_name, call_args, &
                                 error_msg)) return
        call copy_back_reference_args(context, args, copyback_indices, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_i32_load(context%session, slot, selector, error_msg)) &
            return
        do i = 1, size(alt_labels)
            write (label_text, '(i0)') alt_labels(i)
            target_block = find_label_block(context, trim(label_text), found)
            if (.not. found) then
                error_msg = 'alternate return targets undefined statement '// &
                            'label: '//trim(label_text)
                return
            end if
            ordinal = i32_immediate(context%session, int(i, c_int64_t))
            if (.not. emit_liric_i32_icmp(context%session, LR_CMP_EQ, selector, &
                                          ordinal, condition, error_msg)) return
            next_block = create_liric_block(context%session)
            if (.not. emit_liric_condbr(context%session, condition, &
                                        target_block, next_block, error_msg)) &
                return
            if (.not. set_liric_block(context%session, next_block, error_msg)) &
                return
            context%current_block_id = next_block
        end do
    end subroutine emit_alt_return_call

    subroutine lower_alternate_return(arena, node_index, context, error_msg)
        ! `return n` inside a subroutine with alternate-return dummies stores
        ! the selector n into the hidden trailing selector parameter; the
        ! caller branches on it (#353). A plain RETURN leaves the slot at the
        ! zero the caller wrote, which means "continue after the CALL".
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        logical :: has_selector
        integer :: selector_index
        integer(c_int64_t) :: literal
        character(len=:), allocatable :: parse_error
        type(lr_operand_desc_t) :: selector, slot

        call set_empty(error_msg)
        call get_return_selector(arena, node_index, has_selector, selector_index)
        if (.not. has_selector) return
        if (selector_index <= 0) then
            error_msg = 'alternate return is missing its selector expression'
            return
        end if
        if (context%altret_slot_count <= 0 .or. context%altret_param_index < 0) then
            error_msg = 'alternate return used in a subroutine that declares '// &
                        'no alternate-return dummy'
            return
        end if
        call parse_i32_constant(arena, selector_index, literal, 'selector', &
                                parse_error)
        if (len_trim(parse_error) == 0) then
            if (literal < 1_c_int64_t .or. &
                literal > int(context%altret_slot_count, c_int64_t)) then
                error_msg = 'alternate return selector is outside the '// &
                            'declared alternate-return slots'
                return
            end if
        end if
        call lower_i32_expression(arena, selector_index, context, selector, &
                                  error_msg)
        if (len_trim(error_msg) > 0) return
        slot = ptr_param(context%session, context%altret_param_index)
        if (.not. emit_i32_store(context%session, selector, slot, error_msg)) &
            return
    end subroutine lower_alternate_return

    integer function alt_return_slot_count(arena, param_indices) result(count)
        ! Number of `*` alternate-return dummies in a dummy argument list.
        type(ast_arena_t), intent(in) :: arena
        integer, allocatable, intent(in) :: param_indices(:)
        integer :: i

        count = 0
        if (.not. allocated(param_indices)) return
        do i = 1, size(param_indices)
            if (is_alternate_return_dummy(arena, param_indices(i))) &
                count = count + 1
        end do
    end function alt_return_slot_count

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
        context%symbols(index)%is_dummy_argument = .true.
        if (value_kind == VALUE_C4) then
            if (.not. emit_ptr_offset(context%session, &
                    context%symbols(index)%address, 4_c_int64_t, &
                    context%symbols(index)%element_address, error_msg)) return
            context%symbols(index)%value = &
                liric_f32_immediate(context%session, 0.0_c_float)
        else if (value_kind == VALUE_C8) then
            if (.not. emit_ptr_offset(context%session, &
                    context%symbols(index)%address, 8_c_int64_t, &
                    context%symbols(index)%element_address, error_msg)) return
            context%symbols(index)%value = &
                liric_f64_immediate(context%session, 0.0_c_double)
        else
            context%symbols(index)%value = i32_immediate(context%session, &
                0_c_int64_t)
        end if
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

    subroutine collect_param_names(arena, param_indices, names, count)
        ! Dummy-argument names behind a procedure definition's parameter index
        ! list. count is -1 when any parameter name cannot be recovered, so the
        ! caller treats the signature as unusable for keyword association.
        type(ast_arena_t), intent(in) :: arena
        integer, allocatable, intent(in) :: param_indices(:)
        character(len=64), allocatable, intent(inout) :: names(:)
        integer, intent(out) :: count
        character(len=:), allocatable :: name, name_err
        integer :: i

        count = 0
        if (allocated(names)) deallocate (names)
        if (.not. allocated(param_indices)) then
            allocate (names(0))
            return
        end if
        allocate (names(size(param_indices)))
        names = ''
        do i = 1, size(param_indices)
            call parameter_name(arena, param_indices(i), name, name_err)
            if (len_trim(name_err) > 0) then
                count = -1
                return
            end if
            names(i) = name
        end do
        count = size(param_indices)
    end subroutine collect_param_names

    subroutine callee_dummy_names(arena, context, callee_name, names, count)
        ! Public dummy-argument names of callee_name, taken from its definition
        ! or interface in this unit, else from a separately compiled signature
        ! imported from a .fmod. count is -1 when no signature is visible.
        type(ast_arena_t), intent(in) :: arena
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: callee_name
        character(len=64), allocatable, intent(out) :: names(:)
        integer, intent(out) :: count
        integer :: n
        character(len=:), allocatable :: node_type
        type(function_def_node), pointer :: fn_node
        type(subroutine_def_node), pointer :: sb_node

        count = -1
        allocate (names(0))
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            node_type = get_node_type_at(arena, n)
            if (node_type == 'subroutine_def_node' .or. &
                node_type == 'subroutine_def') then
                sb_node => get_node_as_subroutine_def(arena, n)
                if (.not. associated(sb_node)) cycle
                if (.not. allocated(sb_node%name)) cycle
                if (.not. same_name(sb_node%name, callee_name)) cycle
                call collect_param_names(arena, sb_node%param_indices, names, &
                                         count)
                return
            else if (node_type == 'function_def_node' .or. &
                     node_type == 'function_def') then
                fn_node => get_node_as_function_def(arena, n)
                if (.not. associated(fn_node)) cycle
                if (.not. allocated(fn_node%name)) cycle
                if (.not. same_name(fn_node%name, callee_name)) cycle
                call collect_param_names(arena, fn_node%param_indices, names, &
                                         count)
                return
            end if
        end do
        call external_dummy_names(context, callee_name, names, count)
    end subroutine callee_dummy_names

    subroutine external_dummy_names(context, callee_name, names, count)
        ! Dummy-argument names of a separately compiled procedure whose
        ! signature arrived through a .fmod. count stays -1 when the artefact
        ! carried no names (an older .fmod) or the procedure is unknown.
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: callee_name
        character(len=64), allocatable, intent(inout) :: names(:)
        integer, intent(inout) :: count
        integer :: idx, j

        idx = external_procedure_index(context, callee_name)
        if (idx <= 0) return
        associate (ep => context%external_procedures(idx))
            if (ep%arg_count > MAX_PROC_ARGS) return
            do j = 1, ep%arg_count
                if (len_trim(ep%arg_names(j)) == 0) return
            end do
            if (allocated(names)) deallocate (names)
            allocate (names(ep%arg_count))
            do j = 1, ep%arg_count
                names(j) = ep%arg_names(j)
            end do
            count = ep%arg_count
        end associate
    end subroutine external_dummy_names

    subroutine callee_dummy_contracts(arena, context, callee_name, param_pos, &
                                      is_optional, is_value, intent_code)
        ! The OPTIONAL, VALUE, and INTENT contracts of callee_name's param_pos-th
        ! dummy. They come from the procedure's definition in this unit when it
        ! has one, and otherwise from the .fmod artefact of a separately compiled
        ! module procedure, which carries them precisely so a caller never has to
        ! guess or rescan the defining source (#397).
        type(ast_arena_t), intent(in) :: arena
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: callee_name
        integer, intent(in) :: param_pos
        logical, intent(out) :: is_optional
        logical, intent(out) :: is_value
        integer, intent(out) :: intent_code
        character(len=:), allocatable :: node_type, intent_token
        type(function_def_node), pointer :: fn_node
        type(subroutine_def_node), pointer :: sb_node
        integer :: n, idx

        is_optional = .false.
        is_value = .false.
        intent_code = ARG_INTENT_NONE
        if (param_pos < 1) return
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            node_type = get_node_type_at(arena, n)
            if (node_type == 'subroutine_def_node' .or. &
                node_type == 'subroutine_def') then
                sb_node => get_node_as_subroutine_def(arena, n)
                if (.not. associated(sb_node)) cycle
                if (.not. allocated(sb_node%name)) cycle
                if (.not. same_name(sb_node%name, callee_name)) cycle
                call param_at_attributes(arena, sb_node%param_indices, &
                    sb_node%body_indices, param_pos, intent_token, is_optional, &
                    is_value)
                intent_code = arg_intent_code(intent_token)
                return
            else if (node_type == 'function_def_node' .or. &
                     node_type == 'function_def') then
                fn_node => get_node_as_function_def(arena, n)
                if (.not. associated(fn_node)) cycle
                if (.not. allocated(fn_node%name)) cycle
                if (.not. same_name(fn_node%name, callee_name)) cycle
                call param_at_attributes(arena, fn_node%param_indices, &
                    fn_node%body_indices, param_pos, intent_token, is_optional, &
                    is_value)
                intent_code = arg_intent_code(intent_token)
                return
            end if
        end do
        idx = external_procedure_index(context, callee_name)
        if (idx <= 0) return
        if (.not. context%external_procedures(idx)%by_reference) return
        if (param_pos > context%external_procedures(idx)%arg_count) return
        is_optional = context%external_procedures(idx)%arg_is_optional(param_pos)
        is_value = context%external_procedures(idx)%arg_is_value(param_pos)
        intent_code = context%external_procedures(idx)%arg_intents(param_pos)
    end subroutine callee_dummy_contracts

    integer function arg_intent_code(token) result(code)
        ! The intent code an INTENT token denotes (#397).
        character(len=*), intent(in) :: token

        select case (trim(token))
        case ('in')
            code = ARG_INTENT_IN
        case ('out')
            code = ARG_INTENT_OUT
        case ('inout')
            code = ARG_INTENT_INOUT
        case default
            code = ARG_INTENT_NONE
        end select
    end function arg_intent_code
