    subroutine component_base_name(arena, node_index, name, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        character(len=:), allocatable, intent(out) :: name
        character(len=:), allocatable, intent(out) :: error_msg

        if (.not. node_exists(arena, node_index)) then
            error_msg = 'component base index does not reference an AST node'
            call set_empty(name)
            return
        end if

        if (is_identifier(arena, node_index)) then
            call get_identifier_name(arena, node_index, name, error_msg)
            return
        end if
        error_msg = 'nested component access is not supported'
        call set_empty(name)
    end subroutine component_base_name

    subroutine lower_derived_whole_assignment_diagnostic(arena, node, context, &
                                                         symbol_index, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(assignment_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: symbol_index
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: type_index

        type_index = context%symbols(symbol_index)%derived_type_index
        if (is_derived_constructor_call(arena, node%value_index, context, &
                                        type_index)) then
            call unsupported_feature_error('derived type constructor', &
                                           node%line, node%column, &
                                           'direct LIRIC session does not '// &
                                           'support derived type constructors', &
                                           error_msg)
            return
        end if
        call unsupported_feature_error('derived type assignment', &
                                       node%line, node%column, &
                                       'direct LIRIC session only supports '// &
                                       'component assignment', error_msg)
    end subroutine lower_derived_whole_assignment_diagnostic

    logical function is_derived_constructor_call(arena, node_index, context, &
                                                 type_index)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: type_index

        is_derived_constructor_call = .false.
        if (.not. node_exists(arena, node_index)) return
        select type (node => arena%entries(node_index)%node)
        type is (call_or_subscript_node)
            if (allocated(node%name)) then
                is_derived_constructor_call = &
                    same_name(node%name, context%derived_types(type_index)%name)
            end if
        end select
    end function is_derived_constructor_call

    integer function declaration_derived_type_index(context, node) result(type_index)
        type(lowering_context_t), intent(in) :: context
        type(declaration_node), intent(in) :: node
        character(len=:), allocatable :: type_name

        type_index = 0
        if (.not. allocated(node%type_name)) return
        call extracted_derived_type_name(node%type_name, type_name)
        if (len_trim(type_name) == 0) return
        type_index = find_derived_type(context, type_name)
    end function declaration_derived_type_index

    subroutine extracted_derived_type_name(type_spec, type_name)
        character(len=*), intent(in) :: type_spec
        character(len=:), allocatable, intent(out) :: type_name
        character(len=:), allocatable :: lowered
        integer :: close_pos
        integer :: open_pos

        lowered = trim(lowercase_text(type_spec))
        ! type(t) and class(t) both name derived type t. class(*) is unlimited
        ! polymorphic and stays non-derived (it routes to VALUE_CLASS_STAR). For
        ! a non-polymorphic declared-type object class(t) lowers exactly like
        ! type(t): the passed-object dummy is taken by reference (static
        ! dispatch), so both spellings share the derived-type path here.
        if ((len_trim(lowered) >= 5 .and. lowered(1:5) == 'type(') .or. &
            (len_trim(lowered) >= 6 .and. lowered(1:6) == 'class(')) then
            open_pos = index(lowered, '(')
            close_pos = index(lowered, ')', back=.true.)
            if (close_pos > open_pos + 1) then
                type_name = lowered(open_pos + 1:close_pos - 1)
            else
                type_name = ''
            end if
            if (type_name == '*') type_name = ''
            return
        end if

        if (lowered /= 'integer' .and. lowered /= 'real' .and. &
            lowered /= 'logical' .and. .not. is_character_type_name(lowered)) then
            type_name = lowered
        else
            type_name = ''
        end if
    end subroutine extracted_derived_type_name

    logical function is_derived_type_spec(type_spec)
        character(len=*), intent(in) :: type_spec
        character(len=:), allocatable :: type_name

        call extracted_derived_type_name(type_spec, type_name)
        is_derived_type_spec = len_trim(type_name) > 0
    end function is_derived_type_spec

    integer function find_derived_type(context, name) result(type_index)
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: name
        integer :: i

        type_index = 0
        do i = 1, context%derived_type_count
            if (same_name(context%derived_types(i)%name, name)) then
                type_index = i
                return
            end if
        end do
    end function find_derived_type

    integer function find_derived_component(context, type_index, name) &
        result(component_index)
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: type_index
        character(len=*), intent(in) :: name
        integer :: i

        component_index = 0
        if (type_index <= 0 .or. type_index > context%derived_type_count) return
        do i = 1, context%derived_types(type_index)%component_count
            if (same_name(context%derived_types(type_index)%component_names(i), &
                          name)) then
                component_index = i
                return
            end if
        end do
    end function find_derived_component

    subroutine import_module_derived_types(arena, use_node, context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(use_statement_node), intent(in) :: use_node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: export_idx
        integer :: i
        integer :: type_idx
        integer :: type_idx_in_context
        character(len=:), allocatable :: module_lowered
        character(len=:), allocatable :: type_name

        call set_empty(error_msg)

        if (.not. allocated(use_node%module_name)) return

        module_lowered = lowercase_text(use_node%module_name)
        export_idx = find_module_export_index(context, module_lowered)
        if (export_idx <= 0) then
            ! A module defined in this unit but exporting nothing collectable
            ! (empty or procedure-only) is a no-op. Otherwise resolve through a
            ! .fmod on the include path (#189); intrinsic modules are a no-op.
            if (module_defined_in_unit(arena, module_lowered)) return
            call import_module_from_fmod(use_node, context, error_msg)
            return
        end if

        do i = 1, context%module_exports(export_idx)%derived_type_count
            type_idx = context%module_exports(export_idx)% &
                derived_type_indices(i)
            if (.not. node_exists(arena, type_idx)) then
                error_msg = 'invalid derived type arena index in module '// &
                    trim(context%module_exports(export_idx)%module_name)
                return
            end if
            if (.not. is_derived_type_node(arena, type_idx)) then
                cycle
            end if
            call get_derived_type_name(arena, type_idx, type_name, error_msg)
            if (len_trim(error_msg) > 0) return
            if (len_trim(type_name) == 0) cycle
            if (.not. use_only_wants(use_node, trim(type_name))) cycle
            type_idx_in_context = find_derived_type(context, type_name)
            if (type_idx_in_context > 0) then
                cycle
            end if
            call add_derived_type_from_arena(arena, type_idx, context, &
                error_msg)
            if (len_trim(error_msg) > 0) return
        end do

        call import_module_constants(arena, export_idx, use_node, context, &
                                     error_msg)
        if (len_trim(error_msg) > 0) return
        call import_module_variables(arena, export_idx, use_node, context, &
                                     error_msg)
        if (len_trim(error_msg) > 0) return
        call validate_use_only_names(arena, export_idx, use_node, context, &
                                     error_msg)
        if (len_trim(error_msg) > 0) return
        call import_use_renames(arena, export_idx, use_node, context, error_msg)
    end subroutine import_module_derived_types

    subroutine import_module_from_fmod(use_node, context, error_msg)
        ! Resolve `use <module>` against a .fmod on the include path. Intrinsic
        ! modules are a no-op; an unresolved non-intrinsic module is an error.
        type(use_statement_node), intent(in) :: use_node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(module_info_t) :: info
        character(len=:), allocatable :: name, path
        integer :: i

        call set_empty(error_msg)
        if (.not. allocated(use_node%module_name)) return
        name = trim(use_node%module_name)
        if (is_intrinsic_module(name)) return

        path = find_fmod_on_path(context, name)
        if (len_trim(path) == 0) then
            call unsupported_feature_error('use statement', use_node%line, &
                use_node%column, 'module "'//name//'" was not compiled in '// &
                'this unit and no '//name//'.fmod was found on the include '// &
                'path', error_msg)
            return
        end if

        call read_fmod(path, info, error_msg)
        if (len_trim(error_msg) > 0) return

        if (allocated(info%parameters)) then
            do i = 1, size(info%parameters)
                call import_fmod_parameter(use_node, info%parameters(i), &
                                           context, error_msg)
                if (len_trim(error_msg) > 0) return
            end do
        end if
        if (allocated(info%variables)) then
            do i = 1, size(info%variables)
                call import_fmod_variable(use_node, info%variables(i), &
                                          context, error_msg)
                if (len_trim(error_msg) > 0) return
            end do
        end if
    end subroutine import_module_from_fmod

    logical function module_defined_in_unit(arena, module_lowered)
        ! Whether a module of this name is defined anywhere in this arena.
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: module_lowered
        integer :: n

        module_defined_in_unit = .false.
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (node => arena%entries(n)%node)
            type is (module_node)
                if (allocated(node%name)) then
                    if (lowercase_text(node%name) == module_lowered) then
                        module_defined_in_unit = .true.
                        return
                    end if
                end if
            end select
        end do
    end function module_defined_in_unit

    logical function is_intrinsic_module(name)
        character(len=*), intent(in) :: name
        character(len=:), allocatable :: lowered

        lowered = lowercase_text(name)
        is_intrinsic_module = lowered == 'iso_c_binding' .or. &
                              lowered == 'iso_fortran_env' .or. &
                              lowered == 'ieee_arithmetic' .or. &
                              lowered == 'ieee_exceptions' .or. &
                              lowered == 'ieee_features' .or. &
                              lowered == 'omp_lib'
    end function is_intrinsic_module

    function find_fmod_on_path(context, name) result(path)
        ! First match of <dir>/<name>.fmod across the include paths and the
        ! current directory wins.
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: name
        character(len=:), allocatable :: path
        character(len=:), allocatable :: candidate
        logical :: exists
        integer :: i

        path = ''
        do i = 1, context%include_path_count
            candidate = trim(context%include_paths(i))//'/'//trim(name)//'.fmod'
            inquire (file=candidate, exist=exists)
            if (exists) then
                path = candidate
                return
            end if
        end do
        candidate = trim(name)//'.fmod'
        inquire (file=candidate, exist=exists)
        if (exists) path = candidate
    end function find_fmod_on_path

    subroutine import_fmod_parameter(use_node, param, context, error_msg)
        type(use_statement_node), intent(in) :: use_node
        type(fmod_parameter_t), intent(in) :: param
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int64_t) :: constant_value
        integer :: index, io_stat

        call set_empty(error_msg)
        if (.not. allocated(param%name)) return
        if (len_trim(param%name) == 0) return
        if (.not. use_only_wants(use_node, trim(param%name))) return
        ! Only integer named constants are importable today.
        if (.not. allocated(param%kind)) return
        if (trim(param%kind) /= 'integer') return
        read (param%value, *, iostat=io_stat) constant_value
        if (io_stat /= 0) then
            error_msg = 'malformed integer value in .fmod for parameter '// &
                        trim(param%name)
            return
        end if
        if (find_symbol(context, trim(param%name)) > 0) return
        call grow_symbols(context)
        index = context%symbol_count + 1
        context%symbols(index)%name = trim(param%name)
        context%symbols(index)%value_kind = VALUE_I32
        context%symbols(index)%value = i32_immediate(context%session, &
                                                      constant_value)
        context%symbols(index)%is_parameter = .true.
        context%symbols(index)%has_i32_constant = .true.
        context%symbols(index)%i32_constant = constant_value
        context%symbol_count = index
    end subroutine import_fmod_parameter

    subroutine import_fmod_variable(use_node, var, context, error_msg)
        ! Add a module variable from a .fmod record to the symbol table (#249 B7a).
        ! The global `__<mod>_MOD_<name>` must already exist in the session
        ! (emitted when the defining module was compiled); we resolve it with
        ! lr_session_intern and set up an addressed symbol so that load/store
        ! work in the using unit.
        use, intrinsic :: iso_c_binding, only: c_char, c_int32_t, c_int64_t
        type(use_statement_node), intent(in) :: use_node
        type(fmod_variable_t), intent(in) :: var
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: mangled
        character(kind=c_char), allocatable :: c_name(:)
        integer(c_int32_t) :: global_id
        integer :: sym_idx

        call set_empty(error_msg)
        if (.not. allocated(var%name)) return
        if (len_trim(var%name) == 0) return
        if (.not. use_only_wants(use_node, trim(var%name))) return
        if (trim(var%kind) /= 'integer') return
        if (find_symbol(context, trim(var%name)) > 0) return
        mangled = module_procedure_mangled( &
            trim(use_node%module_name), trim(var%name))
        call to_c_chars(mangled, c_name)
        global_id = lr_session_intern(context%session%handle, c_name)
        if (global_id < 0_c_int32_t) then
            error_msg = 'module variable global not found: '//trim(mangled)
            return
        end if
        call grow_symbols(context)
        sym_idx = context%symbol_count + 1
        context%symbols(sym_idx)%name = trim(var%name)
        context%symbols(sym_idx)%value_kind = VALUE_I32
        context%symbols(sym_idx)%has_address = .true.
        context%symbols(sym_idx)%is_reference = .true.
        context%symbols(sym_idx)%address%kind = LR_OP_KIND_GLOBAL
        context%symbols(sym_idx)%address%payload = int(global_id, c_int64_t)
        context%symbols(sym_idx)%address%typ = lr_type_ptr_s(context%session%handle)
        context%symbols(sym_idx)%address%global_offset = 0_c_int64_t
        context%symbol_count = sym_idx
    end subroutine import_fmod_variable

    subroutine import_use_renames(arena, export_idx, use_node, context, error_msg)
        ! use m, only: local => remote binds module symbol `remote` under the
        ! local name. rename_list is a flat [local, remote, ...] array.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: export_idx
        type(use_statement_node), intent(in) :: use_node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i
        integer :: j
        integer :: param_idx
        integer :: value_kind
        logical :: found
        integer(c_int64_t) :: constant_value
        integer :: init_index
        integer :: line
        integer :: column
        integer :: symbol_idx
        character(len=:), allocatable :: decl_name
        character(len=:), allocatable :: decl_type_name

        call set_empty(error_msg)
        if (.not. allocated(use_node%rename_list)) return

        do i = 1, size(use_node%rename_list) - 1, 2
            found = .false.
            do j = 1, context%module_exports(export_idx)%parameter_count
                param_idx = context%module_exports(export_idx)%parameter_indices(j)
                if (.not. node_exists(arena, param_idx)) cycle
                if (.not. is_declaration_node(arena, param_idx)) cycle
                call get_declaration_var_name(arena, param_idx, decl_name, &
                                               error_msg)
                if (len_trim(error_msg) > 0) return
                if (.not. same_name(decl_name, use_node%rename_list(i + 1)%s)) &
                    cycle
                call get_declaration_type_name(arena, param_idx, decl_type_name, &
                                               error_msg)
                if (len_trim(error_msg) > 0) return
                line = get_node_line(arena, param_idx)
                column = get_node_column(arena, param_idx)
                call type_name_value_kind(decl_type_name, line, column, &
                                          value_kind, error_msg)
                if (len_trim(error_msg) > 0) return
                if (value_kind /= VALUE_I32) then
                    call unsupported_feature_error('renamed parameter', line, &
                                                   column, 'direct LIRIC '// &
                                                   'session only supports '// &
                                                   'integer scalar parameters', &
                                                   error_msg)
                    return
                end if
                if (.not. get_declaration_has_initializer(arena, param_idx)) then
                    error_msg = 'renamed parameter has no initializer: '// &
                                trim(decl_name)
                    return
                end if
                init_index = get_declaration_initializer_index(arena, param_idx)
                if (init_index <= 0) then
                    error_msg = 'renamed parameter has no initializer: '// &
                                trim(decl_name)
                    return
                end if
                call eval_i32_constant(context%arena, init_index, context, &
                                       constant_value, error_msg)
                if (len_trim(error_msg) > 0) return
                if (find_symbol(context, trim(use_node%rename_list(i)%s)) > 0) then
                    error_msg = 'duplicate use rename local name: '// &
                                trim(use_node%rename_list(i)%s)
                    return
                end if
                call grow_symbols(context)
                symbol_idx = context%symbol_count + 1
                context%symbols(symbol_idx)%name = trim(use_node%rename_list(i)%s)
                context%symbols(symbol_idx)%value_kind = VALUE_I32
                context%symbols(symbol_idx)%value = i32_immediate(context%session, &
                                                                  constant_value)
                context%symbols(symbol_idx)%is_parameter = .true.
                context%symbols(symbol_idx)%has_i32_constant = .true.
                context%symbols(symbol_idx)%i32_constant = constant_value
                context%symbol_count = symbol_idx
                found = .true.
                if (found) exit
            end do
            if (.not. found) then
                error_msg = 'use rename target is not an exported parameter of '// &
                    trim(use_node%module_name)//': '// &
                    trim(use_node%rename_list(i + 1)%s)
                return
            end if
        end do
    end subroutine import_use_renames

    subroutine import_renamed_parameter(decl, local_name, value_kind, context, &
                                        error_msg)
        type(declaration_node), intent(in) :: decl
        character(len=*), intent(in) :: local_name
        integer, intent(in) :: value_kind
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int64_t) :: constant_value
        integer :: index

        if (value_kind /= VALUE_I32) then
            call unsupported_feature_error('renamed parameter', decl%line, &
                decl%column, 'direct LIRIC session only supports integer '// &
                'scalar parameters', error_msg)
            return
        end if
        if (find_symbol(context, local_name) > 0) then
            error_msg = 'duplicate use rename local name: '//trim(local_name)
            return
        end if
        if (.not. decl%has_initializer .or. decl%initializer_index <= 0) then
            error_msg = 'renamed parameter has no initializer: '//trim(decl%var_name)
            return
        end if
        call eval_i32_constant(context%arena, decl%initializer_index, context, &
                               constant_value, error_msg)
        if (len_trim(error_msg) > 0) return
        call grow_symbols(context)
        index = context%symbol_count + 1
        context%symbols(index)%name = trim(local_name)
        context%symbols(index)%value_kind = VALUE_I32
        context%symbols(index)%value = i32_immediate(context%session, &
                                                     constant_value)
        context%symbols(index)%is_parameter = .true.
        context%symbols(index)%has_i32_constant = .true.
        context%symbols(index)%i32_constant = constant_value
        context%symbol_count = index
        call set_empty(error_msg)
    end subroutine import_renamed_parameter

    logical function use_only_wants(use_node, name)
        ! Whether a plain USE imports an exported name. A USE without only:
        ! imports everything; a USE with only: imports just the listed names.
        type(use_statement_node), intent(in) :: use_node
        character(len=*), intent(in) :: name
        integer :: i

        use_only_wants = .true.
        if (.not. use_node%has_only) return
        use_only_wants = .false.
        if (.not. allocated(use_node%only_list)) return
        do i = 1, size(use_node%only_list)
            if (same_name(use_node%only_list(i)%s, name)) then
                use_only_wants = .true.
                return
            end if
        end do
    end function use_only_wants

    subroutine validate_use_only_names(arena, export_idx, use_node, context, &
                                       error_msg)
        ! Every name in only: must be a symbol the module actually exports.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: export_idx
        type(use_statement_node), intent(in) :: use_node
        type(lowering_context_t), intent(in) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i

        call set_empty(error_msg)
        if (.not. use_node%has_only) return
        if (.not. allocated(use_node%only_list)) return
        do i = 1, size(use_node%only_list)
            if (.not. module_exports_name(arena, export_idx, context, &
                                          use_node%only_list(i)%s)) then
                error_msg = 'use only: name is not exported by module '// &
                    trim(use_node%module_name)//': '// &
                    trim(use_node%only_list(i)%s)
                return
            end if
        end do
    end subroutine validate_use_only_names

    logical function module_exports_name(arena, export_idx, context, name)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: export_idx
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: name
        integer :: i
        character(len=:), allocatable :: derived_name, decl_name
        character(len=:), allocatable :: error_msg

        module_exports_name = .false.
        do i = 1, context%module_exports(export_idx)%derived_type_count
            if (.not. is_derived_type_node(arena, &
                    context%module_exports(export_idx)%derived_type_indices(i))) cycle
            call get_derived_type_name(arena, &
                                       context%module_exports(export_idx)% &
                                           derived_type_indices(i), &
                                       derived_name, error_msg)
            if (len_trim(error_msg) > 0) cycle
            if (same_name(derived_name, name)) then
                module_exports_name = .true.
                return
            end if
        end do
        do i = 1, context%module_exports(export_idx)%parameter_count
            if (.not. is_declaration_node(arena, &
                    context%module_exports(export_idx)%parameter_indices(i))) cycle
            call get_declaration_var_name(arena, &
                                          context%module_exports(export_idx)% &
                                              parameter_indices(i), &
                                          decl_name, error_msg)
            if (len_trim(error_msg) > 0) cycle
            if (same_name(decl_name, name)) then
                module_exports_name = .true.
                return
            end if
        end do
        do i = 1, context%module_exports(export_idx)%variable_count
            if (.not. is_declaration_node(arena, &
                    context%module_exports(export_idx)%variable_indices(i))) cycle
            call get_declaration_var_name(arena, &
                                          context%module_exports(export_idx)% &
                                              variable_indices(i), &
                                          decl_name, error_msg)
            if (len_trim(error_msg) > 0) cycle
            if (same_name(decl_name, name)) then
                module_exports_name = .true.
                return
            end if
        end do
        ! Module procedures are not recorded in module_exports_t (only data
        ! symbols are), so resolve a procedure name against the module body
        ! itself (#263). A USE-only of a contained function or subroutine is a
        ! valid export.
        if (module_exports_procedure(arena, &
                context%module_exports(export_idx)%module_name, name)) then
            module_exports_name = .true.
            return
        end if
    end function module_exports_name

    logical function module_exports_procedure(arena, module_name, name)
        ! Whether the named module contains a public function or subroutine
        ! called `name`. Used to validate USE-only of a module procedure (#263).
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: module_name
        character(len=*), intent(in) :: name
        character(len=:), allocatable :: module_lowered
        integer :: n

        module_exports_procedure = .false.
        module_lowered = lowercase_text(module_name)
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (node => arena%entries(n)%node)
            type is (module_node)
                if (.not. allocated(node%name)) cycle
                if (lowercase_text(node%name) /= module_lowered) cycle
                if (.not. allocated(node%procedure_indices)) return
                if (module_procedure_named(arena, node, name)) &
                    module_exports_procedure = .true.
                return
            end select
        end do
    end function module_exports_procedure

    logical function module_procedure_named(arena, mod_node, name)
        ! Whether `mod_node` holds a non-private function/subroutine `name`.
        type(ast_arena_t), intent(in) :: arena
        type(module_node), intent(in) :: mod_node
        character(len=*), intent(in) :: name
        character(len=:), allocatable :: proc_name
        integer :: p, idx

        module_procedure_named = .false.
        do p = 1, size(mod_node%procedure_indices)
            idx = mod_node%procedure_indices(p)
            if (.not. node_exists(arena, idx)) cycle
            proc_name = ''
            select type (proc => arena%entries(idx)%node)
            type is (function_def_node)
                if (allocated(proc%name)) proc_name = proc%name
            type is (subroutine_def_node)
                if (allocated(proc%name)) proc_name = proc%name
            end select
            if (len_trim(proc_name) == 0) cycle
            if (.not. same_name(proc_name, name)) cycle
            if (module_symbol_is_private(arena, mod_node, proc_name)) cycle
            module_procedure_named = .true.
            return
        end do
    end function module_procedure_named

    subroutine import_module_constants(arena, export_idx, use_node, context, &
                                       error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: export_idx
        type(use_statement_node), intent(in) :: use_node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i
        integer :: param_idx
        integer :: value_kind
        integer(c_int64_t) :: constant_value
        integer :: init_index
        integer :: line
        integer :: column
        character(len=:), allocatable :: decl_name
        character(len=:), allocatable :: decl_type_name

        call set_empty(error_msg)
        do i = 1, context%module_exports(export_idx)%parameter_count
            param_idx = context%module_exports(export_idx)%parameter_indices(i)
            if (.not. node_exists(arena, param_idx)) cycle
            if (.not. is_declaration_node(arena, param_idx)) cycle
            call get_declaration_var_name(arena, param_idx, decl_name, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. use_only_wants(use_node, trim(decl_name))) cycle
            if (find_symbol(context, decl_name) > 0) cycle
            call get_declaration_type_name(arena, param_idx, decl_type_name, &
                                           error_msg)
            if (len_trim(error_msg) > 0) return
            line = get_node_line(arena, param_idx)
            column = get_node_column(arena, param_idx)
            call type_name_value_kind(decl_type_name, line, column, value_kind, &
                                      error_msg)
            if (len_trim(error_msg) > 0) return
            if (value_kind /= VALUE_I32) then
                call unsupported_feature_error('parameter declaration', line, &
                                               column, 'direct LIRIC session '// &
                                               'only supports integer scalar '// &
                                               'parameters', error_msg)
                return
            end if
            if (.not. get_declaration_has_initializer(arena, param_idx)) then
                error_msg = 'integer parameter requires a compile-time '// &
                            'initializer: '//trim(decl_name)
                return
            end if
            init_index = get_declaration_initializer_index(arena, param_idx)
            if (init_index <= 0) then
                error_msg = 'integer parameter requires a compile-time '// &
                            'initializer: '//trim(decl_name)
                return
            end if
            call eval_i32_constant(context%arena, init_index, context, &
                                   constant_value, error_msg)
            if (len_trim(error_msg) > 0) return
            call grow_symbols(context)
            param_idx = context%symbol_count + 1
            context%symbols(param_idx)%name = trim(decl_name)
            context%symbols(param_idx)%value_kind = VALUE_I32
            context%symbols(param_idx)%value = i32_immediate(context%session, &
                                                             constant_value)
            context%symbols(param_idx)%is_parameter = .true.
            context%symbols(param_idx)%has_i32_constant = .true.
            context%symbols(param_idx)%i32_constant = constant_value
            context%symbol_count = param_idx
        end do
    end subroutine import_module_constants

    subroutine import_module_variables(arena, export_idx, use_node, context, &
                                       error_msg)
        ! Import module-level variables from an in-unit module into the calling
        ! program's symbol table (#263). Each supported scalar variable binds to
        ! the shared global as an addressed reference, honouring use, only:.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: export_idx
        type(use_statement_node), intent(in) :: use_node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i, var_idx, value_kind, status
        character(len=:), allocatable :: var_name, name_err

        call set_empty(error_msg)
        do i = 1, context%module_exports(export_idx)%variable_count
            var_idx = context%module_exports(export_idx)%variable_indices(i)
            if (.not. node_exists(arena, var_idx)) cycle
            call get_declaration_var_name(arena, var_idx, var_name, name_err)
            if (len_trim(name_err) > 0) cycle
            if (.not. use_only_wants(use_node, trim(var_name))) cycle
            call module_variable_kind(arena, var_idx, value_kind, status, &
                                      error_msg)
            ! An unsupported module variable already failed in
            ! lower_module_variables; the using unit just skips it here.
            if (status /= MODVAR_OK) then
                call set_empty(error_msg)
                cycle
            end if
            call bind_one_module_variable(arena, var_idx, value_kind, &
                trim(context%module_exports(export_idx)%module_name), context, &
                error_msg)
            if (len_trim(error_msg) > 0) return
        end do
    end subroutine import_module_variables

    subroutine add_derived_type_from_arena(arena, type_index, context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: type_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: type_index_new

        call set_empty(error_msg)

        if (.not. node_exists(arena, type_index)) then
            error_msg = 'derived type index does not reference an AST node'
            return
        end if

      call grow_derived_types(context)

        type_index_new = context%derived_type_count + 1

        ! Use the existing lower_derived_type_definition pattern but with arena index
        call import_derived_type_from_arena(arena, type_index, context, &
            type_index_new, error_msg)
        if (len_trim(error_msg) > 0) return

        context%derived_type_count = type_index_new
    end subroutine add_derived_type_from_arena

subroutine import_derived_type_from_arena(arena, type_index, context, &
        type_index_new, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: type_index
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: type_index_new
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: type_name

        call set_empty(error_msg)

        if (.not. node_exists(arena, type_index)) then
            error_msg = 'derived type index does not reference an AST node'
            return
        end if

        context%derived_types(type_index_new)%component_count = 0

        call get_derived_type_name(arena, type_index, type_name, error_msg)
        if (len_trim(error_msg) > 0) return
        context%derived_types(type_index_new)%name = trim(type_name)

        call collect_derived_components_via_arena(arena, type_index, context, &
            type_index_new, error_msg)
    end subroutine import_derived_type_from_arena

    subroutine collect_derived_components_via_arena(arena, type_index, &
        context, type_index_new, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: type_index
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: type_index_new
        character(len=:), allocatable, intent(out) :: error_msg

        call set_empty(error_msg)

        if (.not. node_exists(arena, type_index)) then
            error_msg = 'derived type index does not reference an AST node'
            return
        end if

        ! The key insight: we need to access derived_type_node%component_indices
        ! which is an allocatable integer array. When we do select type on
        ! arena%entries(type_index)%node (which is class(ast_node), allocatable),
        ! the allocatable arrays in the derived type should be accessible.
        !
        ! The issue we've been seeing is that component_indices appears unallocated.
        ! This might be because the arena stores nodes differently than expected.
        !
        ! Let's try using context%arena instead of arena to see if that helps.
        call collect_derived_components_from_context_arena(arena, type_index, context, &
            type_index_new, error_msg)
    end subroutine collect_derived_components_via_arena

    subroutine collect_derived_components_from_context_arena(arena, type_index, &
        context, type_index_new, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: type_index
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: type_index_new
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i
        integer :: parent_line
        integer :: parent_column
        integer, allocatable :: component_indices(:)

        call set_empty(error_msg)

        if (.not. node_exists(arena, type_index)) then
            error_msg = 'derived type index does not reference an AST node'
            return
        end if

        if (.not. is_derived_type_node(arena, type_index)) then
            error_msg = 'node at type_index is not a derived_type_node'
            return
        end if
        call get_derived_type_components(arena, type_index, component_indices)
        parent_line = get_node_line(arena, type_index)
        parent_column = get_node_column(arena, type_index)
        if (allocated(component_indices)) then
            do i = 1, size(component_indices)
                call collect_component_declaration(component_indices(i), &
                                                   parent_line, parent_column, &
                                                   context, type_index_new, &
                                                   error_msg)
                if (len_trim(error_msg) > 0) return
            end do
        end if
    end subroutine collect_derived_components_from_context_arena

    subroutine lower_program_body(arena, program, context, value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(program_node), intent(in) :: program
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i

        call set_empty(error_msg)
        value = i32_immediate(context%session, 0_c_int64_t)
        context%current_block_terminated = .false.

        if (.not. allocated(program%body_indices)) return
        do i = 1, size(program%body_indices)
            if (is_internal_procedure_entry(arena, program%body_indices(i))) cycle
            call lower_statement(arena, program%body_indices(i), context, &
                                 value, error_msg)
            if (len_trim(error_msg) > 0) return
            if (context%current_block_terminated) return
        end do
    end subroutine lower_program_body

    subroutine find_module_in_arena(arena, module_lowered, module_index)
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: module_lowered
        integer, intent(out) :: module_index
        integer :: i

        module_index = 0
        if (len_trim(module_lowered) == 0) return

        do i = 1, arena%size
            if (.not. allocated(arena%entries(i)%node)) cycle
            select type (node => arena%entries(i)%node)
            type is (module_node)
                if (.not. allocated(node%name)) cycle
                if (lowercase_text(node%name) == module_lowered) then
                    module_index = i
                    return
                end if
            end select
        end do
    end subroutine find_module_in_arena
