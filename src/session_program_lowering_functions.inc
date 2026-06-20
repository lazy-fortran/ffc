    subroutine collect_internal_function_names(arena, root_index, context, &
                                               error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: root_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: prog_name, ff_error, node_type
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
        call get_program_body_info(arena, root_index, &
                                   prog_name, body_indices, ff_error)
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
            end if
        end do
    end subroutine collect_internal_function_names

    subroutine register_internal_subroutine_name(node, context, error_msg, &
                                                 node_index)
        type(subroutine_def_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer, intent(in), optional :: node_index
        integer :: next_index

        call set_empty(error_msg)
        if (.not. allocated(node%name)) return
        call grow_function_names(context)
        if (contained_function_kind(context, node%name) /= 0) return
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

    integer function proc_param_count(context, name) result(count)
        ! Declared parameter count of a contained procedure, or -1 if unknown.
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: name
        integer :: i

        count = -1
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

        if (.not. allocated(node%name)) then
            error_msg = 'direct LIRIC session function requires a name'
            return
        end if
       call grow_function_names(context)
         if (contained_function_kind(context, node%name) /= 0) then
            error_msg = 'duplicate contained function: '//trim(node%name)
            return
        end if
        call function_value_kind(node, context, value_kind, derived_type_index, &
                                 error_msg)
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
                                   error_msg)
        type(function_def_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        integer, intent(out) :: value_kind
        integer, intent(out) :: derived_type_index
        integer :: char_len, paren, rparen
        character(len=:), allocatable :: derived_name, base_type, kind_spec
        character(len=:), allocatable, intent(out) :: error_msg

        value_kind = VALUE_I32
        derived_type_index = 0
        call set_empty(error_msg)
        if (.not. allocated(node%return_type)) return

        ! When the result type is declared in the body (result(r) with
        ! `<type> :: r`) rather than on the function statement, return_type is
        ! empty. Resolve the intrinsic scalar kind from that body declaration.
        if (len_trim(node%return_type) == 0) then
            call result_var_scalar_kind(node, context, value_kind, &
                                        derived_type_index, error_msg)
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
            value_kind = VALUE_I32
        case ('real')
            ! Classify by kind: bare real and real(4) are f32; real(8) is f64.
            if (real_kind_is_f64(kind_spec)) then
                value_kind = VALUE_F64
            else
                value_kind = VALUE_F32
            end if
        case ('double precision')
            value_kind = VALUE_F64
        case ('complex')
            ! Bare complex and complex(4) are two f32s; complex(8) is two f64s.
            if (complex_kind_is_c8(kind_spec)) then
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
                if (char_len <= 0) then
                    value_kind = VALUE_DEFERRED_CHARACTER_RESULT
                else
                    call unsupported_feature_error('function result type', &
                                                    node%line, node%column, &
                                                    'direct LIRIC session '// &
                                                    'supports deferred-length '// &
                                                    'character, integer, real, '// &
                                                    'and logical contained '// &
                                                    'functions', error_msg)
                end if
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
                                value_kind, derived_type_index, is_array, error_msg)
                            if (.not. is_array) return
                        end if
                    end if
                    if (decl%is_multi_declaration .and. allocated(decl%var_names)) then
                        do k = 1, size(decl%var_names)
                            if (trim(decl%var_names(k)) == result_name) then
                                call resolve_result_decl_kind(decl, context, &
                                    value_kind, derived_type_index, is_array, &
                                    error_msg)
                                if (.not. is_array) return
                            end if
                        end do
                    end if
                end select
            end do
        end if

        ! Either no matching declaration or an array-valued result.
        call unsupported_feature_error('function result type', &
                                        node%line, node%column, &
                                        'direct LIRIC session supports '// &
                                        'integer, real, logical, complex, '// &
                                        'deferred-length character, and '// &
                                        'derived-type contained functions', &
                                        error_msg)
    end subroutine result_var_scalar_kind

    subroutine resolve_result_decl_kind(decl, context, value_kind, &
                                        derived_type_index, is_array, error_msg)
        type(declaration_node), intent(in) :: decl
        type(lowering_context_t), intent(in) :: context
        integer, intent(out) :: value_kind
        integer, intent(out) :: derived_type_index
        logical, intent(out) :: is_array
        character(len=:), allocatable, intent(out) :: error_msg

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
        call declaration_value_kind(decl, value_kind, error_msg)
    end subroutine resolve_result_decl_kind

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

    integer function contained_function_kind(context, name) result(value_kind)
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: name
        integer :: i

        value_kind = 0
        do i = 1, context%function_count
            if (same_name(context%function_names(i), name)) then
                value_kind = context%function_value_kinds(i)
                return
            end if
        end do
   end function contained_function_kind

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
        character(len=:), allocatable :: prog_name, ff_error, node_type
        integer, allocatable :: body_indices(:)
        integer :: i

        call set_empty(error_msg)
        node_type = get_node_type_at(arena, root_index)
        if (node_type == 'module_node' .or. node_type == 'module') return
        call get_program_body_info(arena, root_index, prog_name, body_indices, &
                                   ff_error)
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
            call lower_scalar_function(arena, fn_node, context, error_msg)
        else if (node_type == 'subroutine_def_node' .or. &
                 node_type == 'subroutine_def') then
            sb_node => get_node_as_subroutine_def(arena, node_index)
            if (.not. associated(sb_node)) then
                error_msg = 'internal procedure index does not reference a subroutine node'
                return
            end if
            call lower_void_subroutine(arena, sb_node, context, error_msg)
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
        character(len=:), allocatable :: prog_name, ff_error, node_type
        integer, allocatable :: body_indices(:)
        integer :: i

        call set_empty(error_msg)
        node_type = get_node_type_at(arena, root_index)
        if (node_type == 'module_node' .or. node_type == 'module') then
            call lower_one_module_unit_procedures(arena, root_index, context, &
                                                  error_msg)
            return
        end if
        call get_program_body_info(arena, root_index, prog_name, body_indices, &
                                   ff_error)
        if (len_trim(ff_error) > 0) then
            error_msg = ff_error
            return
        end if
        if (size(body_indices) == 0) return
        do i = 1, size(body_indices)
            node_type = get_node_type_at(arena, body_indices(i))
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
            call lower_scalar_function(arena, fn_node, context, error_msg, &
                module_procedure_mangled(module_name, fn_node%name))
            if (len_trim(error_msg) > 0) return
            call lower_nested_internal_procedures(arena, fn_node%body_indices, &
                                                  context, error_msg)
        else if (node_type == 'subroutine_def_node' .or. &
                 node_type == 'subroutine_def') then
            sb_node => get_node_as_subroutine_def(arena, node_index)
            if (.not. associated(sb_node)) return
            call lower_void_subroutine(arena, sb_node, context, error_msg, &
                module_procedure_mangled(module_name, sb_node%name))
            if (len_trim(error_msg) > 0) return
            call lower_nested_internal_procedures(arena, sb_node%body_indices, &
                                                  context, error_msg)
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
        character(len=:), allocatable :: mod_name, ff_error, node_type
        integer, allocatable :: declaration_indices(:), procedure_indices(:)
        integer :: p
        type(function_def_node), pointer :: fn_node
        type(subroutine_def_node), pointer :: sb_node

        call set_empty(error_msg)
        call get_module_body_info(arena, module_index, mod_name, &
                                  declaration_indices, procedure_indices, &
                                  ff_error)
        if (len_trim(ff_error) > 0) then
            error_msg = ff_error
            return
        end if
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
    end subroutine register_module_procedures

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
                                     emit_name_override)
        type(ast_arena_t), intent(in) :: arena
        type(function_def_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: parent_context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=*), intent(in), optional :: emit_name_override
        type(lowering_context_t) :: context
        type(lr_operand_desc_t) :: value
        integer :: param_count
        integer :: result_index
        integer :: value_kind
        integer :: result_derived_index
        logical :: terminated
        character(len=:), allocatable :: result_name
        character(len=:), allocatable :: emit_name

        if (.not. allocated(node%name)) then
            error_msg = 'direct LIRIC session function requires a name'
            return
        end if
        call function_value_kind(node, parent_context, value_kind, &
                                 result_derived_index, error_msg)
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
        context%current_proc_name = trim(node%name)

        param_count = 0
        if (allocated(node%param_indices)) param_count = size(node%param_indices)
        if (present(emit_name_override)) then
            emit_name = trim(emit_name_override)
        else
            emit_name = procedure_emit_name(node%name, node%bind_c_clause)
        end if
        if (value_kind == VALUE_F32) then
            if (.not. begin_liric_f32_function(context%session, emit_name, &
                                               param_count, error_msg)) return
        else if (value_kind == VALUE_F64) then
            if (.not. begin_liric_f64_function(context%session, emit_name, &
                                               param_count, error_msg)) return
        else if (value_kind == VALUE_DEFERRED_CHARACTER_RESULT .or. &
                 value_kind == VALUE_DERIVED .or. &
                 value_kind == VALUE_C4 .or. value_kind == VALUE_C8) then
            ! Aggregate / complex results return via a leading sret pointer:
            ! the caller passes the result buffer address as param 0.
            if (.not. begin_void_subroutine(context%session, emit_name, &
                                            param_count + 1, error_msg)) return
        else
            if (.not. begin_i32_function(context%session, emit_name, param_count, &
                                                          error_msg)) return
        end if

        if (value_kind == VALUE_DEFERRED_CHARACTER_RESULT) then
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
        else
            call define_symbol(context, result_name, value_kind, error_msg)
            if (len_trim(error_msg) > 0) return
            context%current_function_result_index = find_symbol(context, &
                                                                result_name)
        end if
        if (value_kind == VALUE_DEFERRED_CHARACTER_RESULT .or. &
            value_kind == VALUE_DERIVED .or. &
            value_kind == VALUE_C4 .or. value_kind == VALUE_C8) then
            call define_reference_parameters(arena, node%param_indices, context, &
                                             error_msg, 1)
        else
            call define_reference_parameters(arena, node%param_indices, context, &
                                             error_msg)
        end if
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

        result_index = find_symbol(context, result_name)
        if (result_index <= 0) then
            error_msg = 'function result symbol was not declared: '// &
                        trim(result_name)
            return
        end if
        if (.not. context%current_block_terminated) then
            if (value_kind == VALUE_DEFERRED_CHARACTER_RESULT .or. &
                value_kind == VALUE_DERIVED .or. &
                value_kind == VALUE_C4 .or. value_kind == VALUE_C8) then
                if (.not. emit_ret_void(context%session, error_msg)) return
            else
                value = context%symbols(result_index)%value
                if (.not. emit_ret_i32_operand(context%session, value, error_msg)) return
            end if
        end if
        if (.not. finish_function(context%session, error_msg)) return

        parent_context%session = context%session
        parent_context%string_literal_count = context%string_literal_count
        call set_empty(error_msg)
    end subroutine lower_scalar_function

    include 'session_program_lowering_functions_tail.inc'
