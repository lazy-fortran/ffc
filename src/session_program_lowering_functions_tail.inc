    subroutine lower_void_subroutine(arena, node, parent_context, error_msg, &
                                      emit_name_override)
        type(ast_arena_t), intent(in) :: arena
        type(subroutine_def_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: parent_context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=*), intent(in), optional :: emit_name_override
        type(lowering_context_t) :: context
        type(lr_operand_desc_t) :: value
        integer :: param_count
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
        context%arena = arena
        allocate(context%symbols(16))
        allocate(context%derived_types(8))
        allocate(context%module_exports(4))
        allocate(context%function_names(8))
        allocate(context%function_value_kinds(8))
        context%function_value_kinds = VALUE_I32
        call copy_internal_function_names(parent_context, context)
        call copy_derived_type_table(parent_context, context)
        call copy_module_exports(parent_context, context)
        context%in_internal_function = .true.
        context%in_internal_subroutine = .true.
        context%current_proc_name = trim(node%name)

        param_count = 0
        if (allocated(node%param_indices)) param_count = size(node%param_indices)
        if (present(emit_name_override)) then
            emit_name = trim(emit_name_override)
        else
            emit_name = procedure_emit_name(node%name, node%bind_c_clause)
        end if
        if (.not. begin_void_subroutine(context%session, emit_name, param_count, &
                                                        error_msg)) return

        call define_reference_parameters(arena, node%param_indices, context, &
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
        ! the caller falls back to its own default. Lets a real literal actual
        ! adopt the dummy's kind (e.g. 2.0 to a real(4) step) regardless of the
        ! caller-side default kind.
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
                return
            end if
        end do
    end function callee_dummy_value_kind

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
            emit_name = trim(fortran_name)
        end if
    end function procedure_emit_name

    function procedure_module(arena, fortran_name) result(module_name)
        ! The module that defines a procedure, or empty if not module-defined.
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: fortran_name
        character(len=:), allocatable :: module_name
        character(len=:), allocatable :: node_type, mod_name, ff_error
        integer, allocatable :: declaration_indices(:), procedure_indices(:)
        integer :: n, p
        type(function_def_node), pointer :: fn_node
        type(subroutine_def_node), pointer :: sb_node

        call set_empty(module_name)
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            node_type = get_node_type_at(arena, n)
            if (node_type /= 'module_node' .and. node_type /= 'module') cycle
            call get_module_body_info(arena, n, mod_name, declaration_indices, &
                                      procedure_indices, ff_error)
            if (len_trim(ff_error) > 0) cycle
            if (size(procedure_indices) == 0) cycle
            do p = 1, size(procedure_indices)
                if (procedure_node_named(arena, procedure_indices(p), fortran_name)) then
                    module_name = trim(mod_name)
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

    function call_emit_name(arena, fortran_name) result(emit_name)
        ! Resolve the LIRIC symbol to call: a module-defined procedure mangles
        ! to __<module>_MOD_<name>; a bind(c) procedure uses its C name; else
        ! the Fortran name.
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: fortran_name
        character(len=:), allocatable :: emit_name, module_name
        character(len=:), allocatable :: node_type
        integer :: n
        type(function_def_node), pointer :: fn_node
        type(subroutine_def_node), pointer :: sb_node

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

    subroutine define_reference_parameters(arena, param_indices, context, &
                                           error_msg, param_offset)
        type(ast_arena_t), intent(in) :: arena
        integer, allocatable, intent(in) :: param_indices(:)
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer, intent(in), optional :: param_offset
        character(len=:), allocatable :: name
        integer :: offset
        integer :: i

        call set_empty(error_msg)
        offset = 0
        if (present(param_offset)) offset = param_offset
        if (.not. allocated(param_indices)) return
        do i = 1, size(param_indices)
            call parameter_name(arena, param_indices(i), name, error_msg)
            if (len_trim(error_msg) > 0) return
            call define_parameter_symbol(context, name, i - 1 + offset, &
                                         error_msg)
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

    subroutine define_parameter_symbol(context, name, param_index, error_msg)
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: name
        integer, intent(in) :: param_index
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: index

        if (find_symbol(context, name) > 0) then
            error_msg = 'duplicate function parameter: '//trim(name)
            return
        end if
       call grow_symbols(context)

        index = context%symbol_count + 1
        context%symbols(index)%name = trim(name)
        context%symbols(index)%value_kind = VALUE_I32
        context%symbols(index)%address = ptr_param(context%session, param_index)
        context%symbols(index)%has_address = .true.
        context%symbols(index)%is_reference = .true.
        context%symbols(index)%value = i32_immediate(context%session, 0_c_int64_t)
        context%symbols(index)%is_parameter = .true.
        context%symbol_count = index
        call set_empty(error_msg)
    end subroutine define_parameter_symbol

    subroutine ensure_function_names_capacity(context, required_index)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: required_index

        if (required_index <= size(context%function_names)) return
        call grow_function_names(context)
        if (required_index <= size(context%function_names)) return
        allocate(context%function_names(required_index), &
                 source=context%function_names)
        allocate(context%function_value_kinds(required_index), &
                 source=context%function_value_kinds)
        allocate(context%function_param_counts(required_index), &
                 source=context%function_param_counts)
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
