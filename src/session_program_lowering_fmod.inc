    subroutine emit_module_fmod_artifacts(arena, context, output_path, error_msg)
        ! For each module compiled in this unit, write a sibling
        ! <dirname(output)>/<modulename>.fmod describing its exports.
        type(ast_arena_t), intent(in) :: arena
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: output_path
        character(len=:), allocatable, intent(out) :: error_msg
        type(module_info_t) :: info
        character(len=:), allocatable :: dir, path
        integer :: m

        call set_empty(error_msg)
        dir = path_dirname(output_path)
        do m = 1, context%module_export_count
            call build_module_info(arena, context%module_exports(m), info, &
                                    error_msg)
            if (len_trim(error_msg) > 0) return
            path = dir//trim(context%module_exports(m)%module_name)//'.fmod'
            call write_fmod(path, info, error_msg)
            if (len_trim(error_msg) > 0) return
        end do
    end subroutine emit_module_fmod_artifacts

    function path_dirname(path) result(dir)
        character(len=*), intent(in) :: path
        character(len=:), allocatable :: dir
        integer :: slash

        slash = index(path, '/', back=.true.)
        if (slash > 0) then
            dir = path(1:slash)
        else
            dir = ''
        end if
    end function path_dirname

    subroutine build_module_info(arena, export, info, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(module_exports_t), intent(in) :: export
        type(module_info_t), intent(out) :: info
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i

        call set_empty(error_msg)
        info%name = trim(export%module_name)
        allocate (info%parameters(export%parameter_count))
        do i = 1, export%parameter_count
            call build_fmod_parameter(arena, export%parameter_indices(i), &
                                      info%parameters(i), error_msg)
            if (len_trim(error_msg) > 0) return
        end do
        allocate (info%derived_types(export%derived_type_count))
        do i = 1, export%derived_type_count
            call build_fmod_derived_type(arena, export%derived_type_indices(i), &
                                         info%derived_types(i), error_msg)
            if (len_trim(error_msg) > 0) return
        end do
        ! Scalar module variables round-trip so a separately compiled program
        ! can bind the shared global on USE (#274).
        allocate (info%variables(export%variable_count))
        do i = 1, export%variable_count
            call build_fmod_variable(arena, export%variable_indices(i), &
                                     info%variables(i), error_msg)
            if (len_trim(error_msg) > 0) return
        end do
        ! Public module procedures with integer-scalar signatures round-trip so
        ! a separately compiled program can call them by reference and link
        ! against the module object (#284).
        call build_fmod_procedures(arena, trim(export%module_name), &
                                   info%procedures, error_msg)
        if (len_trim(error_msg) > 0) return
        ! Named generic interfaces whose specifics are all exportable procedures
        ! round-trip so a use-associated generic call in a separately compiled
        ! program resolves to the matching specific and links.
        call build_fmod_generics(arena, trim(export%module_name), &
                                 info%procedures, info%generics, error_msg)
    end subroutine build_module_info

    subroutine build_fmod_generics(arena, module_name, procs, generics, error_msg)
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: module_name
        type(fmod_procedure_t), allocatable, intent(in) :: procs(:)
        type(fmod_generic_t), allocatable, intent(out) :: generics(:)
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: module_index, d, count
        integer, allocatable :: declaration_indices(:), procedure_indices(:)
        character(len=:), allocatable :: mod_name, ff_error, specifics

        call set_empty(error_msg)
        allocate (generics(0))
        call find_module_in_arena(arena, lowercase_text(module_name), module_index)
        if (module_index == 0) return
        call get_module_body_info(arena, module_index, mod_name, &
                                  declaration_indices, procedure_indices, ff_error)
        if (len_trim(ff_error) > 0) return
        if (.not. allocated(declaration_indices)) return
        count = 0
        do d = 1, size(declaration_indices)
            call fmod_generic_specifics(arena, declaration_indices(d), procs, &
                                        mod_name, specifics)
            if (len_trim(specifics) == 0) cycle
            count = count + 1
            call grow_fmod_generics(generics, count)
            generics(count)%name = generic_block_name(arena, declaration_indices(d))
            generics(count)%specifics = specifics
        end do
    end subroutine build_fmod_generics

    function generic_block_name(arena, node_index) result(name)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        character(len=:), allocatable :: name

        name = ''
        if (.not. node_exists(arena, node_index)) return
        select type (node => arena%entries(node_index)%node)
        type is (interface_block_node)
            if (allocated(node%name)) name = trim(node%name)
        end select
    end function generic_block_name

    subroutine fmod_generic_specifics(arena, node_index, procs, module_name, &
                                      specifics)
        ! Space-joined specific names of a named generic interface, but only when
        ! every specific is an exportable procedure recorded in procs. An empty
        ! result means the node is not an exportable generic (unnamed, operator,
        ! private, or with a non-scalar specific), so it is skipped.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(fmod_procedure_t), allocatable, intent(in) :: procs(:)
        character(len=*), intent(in) :: module_name
        character(len=:), allocatable, intent(out) :: specifics
        type(module_procedure_node), pointer :: mp_node
        character(len=:), allocatable :: node_type, body_name
        integer :: i, j

        specifics = ''
        if (.not. node_exists(arena, node_index)) return
        select type (node => arena%entries(node_index)%node)
        type is (interface_block_node)
            if (node%is_abstract) return
            if (allocated(node%kind)) then
                node_type = lowercase_text(trim(node%kind))
                if (node_type == 'operator' .or. node_type == 'assignment') return
            end if
            if (.not. allocated(node%name)) return
            if (len_trim(node%name) == 0) return
            if (.not. allocated(node%procedure_indices)) return
            do i = 1, size(node%procedure_indices)
                if (.not. node_exists(arena, node%procedure_indices(i))) return
                node_type = get_node_type_at(arena, node%procedure_indices(i))
                if (node_type == 'module_procedure_node' .or. &
                    node_type == 'module_procedure') then
                    select type (n => arena%entries(node%procedure_indices(i))%node)
                    type is (module_procedure_node)
                        mp_node => n
                    class default
                        specifics = ''
                        return
                    end select
                    if (.not. allocated(mp_node%procedure_names)) return
                    do j = 1, size(mp_node%procedure_names)
                        call append_generic_specific(procs, &
                            trim(mp_node%procedure_names(j)%s), specifics)
                        if (len_trim(specifics) == 0) return
                    end do
                else if (node_type == 'function_def_node' .or. &
                         node_type == 'function_def' .or. &
                         node_type == 'subroutine_def_node' .or. &
                         node_type == 'subroutine_def') then
                    call interface_body_procedure_name(arena, &
                        node%procedure_indices(i), body_name)
                    if (len_trim(body_name) == 0) then
                        specifics = ''
                        return
                    end if
                    call append_generic_specific(procs, body_name, specifics)
                    if (len_trim(specifics) == 0) return
                else
                    specifics = ''
                    return
                end if
            end do
        end select
    end subroutine fmod_generic_specifics

    subroutine append_generic_specific(procs, name, specifics)
        ! Append name to the space-joined specifics list only when it names an
        ! exportable procedure in procs; otherwise clear specifics to signal the
        ! generic cannot be exported.
        type(fmod_procedure_t), allocatable, intent(in) :: procs(:)
        character(len=*), intent(in) :: name
        character(len=:), allocatable, intent(inout) :: specifics
        integer :: p
        logical :: found

        found = .false.
        if (allocated(procs)) then
            do p = 1, size(procs)
                if (.not. allocated(procs(p)%name)) cycle
                if (same_name(procs(p)%name, name)) then
                    found = .true.
                    exit
                end if
            end do
        end if
        if (.not. found) then
            specifics = ''
            return
        end if
        if (len_trim(specifics) == 0) then
            specifics = trim(name)
        else
            specifics = specifics//' '//trim(name)
        end if
    end subroutine append_generic_specific

    subroutine grow_fmod_generics(arr, n)
        type(fmod_generic_t), allocatable, intent(inout) :: arr(:)
        integer, intent(in) :: n
        type(fmod_generic_t), allocatable :: tmp(:)

        if (n <= size(arr)) return
        allocate (tmp(n))
        tmp(1:size(arr)) = arr
        call move_alloc(tmp, arr)
    end subroutine grow_fmod_generics

    subroutine build_fmod_procedures(arena, module_name, procs, error_msg)
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: module_name
        type(fmod_procedure_t), allocatable, intent(out) :: procs(:)
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: module_index, p, count
        integer, allocatable :: declaration_indices(:), procedure_indices(:)
        character(len=:), allocatable :: mod_name, ff_error, kind_text, arg_tokens
        integer :: nargs
        type(module_node), pointer :: mod_node

        call set_empty(error_msg)
        allocate (procs(0))
        call find_module_in_arena(arena, lowercase_text(module_name), &
                                  module_index)
        if (module_index == 0) return
        mod_node => get_module_node_ptr(arena, module_index)
        if (.not. associated(mod_node)) return
        call get_module_body_info(arena, module_index, mod_name, &
                                  declaration_indices, procedure_indices, &
                                  ff_error)
        if (len_trim(ff_error) > 0) return
        count = 0
        do p = 1, size(procedure_indices)
            call fmod_procedure_signature(arena, procedure_indices(p), mod_node, &
                                          kind_text, nargs, arg_tokens)
            if (len_trim(kind_text) == 0) cycle
            count = count + 1
            call grow_fmod_procs(procs, count)
            procs(count)%name = procedure_fortran_name(arena, &
                                                        procedure_indices(p))
            procs(count)%kind = kind_text
            procs(count)%nargs = nargs
            procs(count)%arg_kinds = arg_tokens
        end do
    end subroutine build_fmod_procedures

    subroutine grow_fmod_procs(arr, n)
        type(fmod_procedure_t), allocatable, intent(inout) :: arr(:)
        integer, intent(in) :: n
        type(fmod_procedure_t), allocatable :: tmp(:)

        if (n <= size(arr)) return
        allocate (tmp(n))
        tmp(1:size(arr)) = arr
        call move_alloc(tmp, arr)
    end subroutine grow_fmod_procs

    function get_module_node_ptr(arena, module_index) result(mod_node)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: module_index
        type(module_node), pointer :: mod_node

        mod_node => null()
        if (.not. node_exists(arena, module_index)) return
        select type (node => arena%entries(module_index)%node)
        type is (module_node)
            mod_node => node
        end select
    end function get_module_node_ptr

    function procedure_fortran_name(arena, node_index) result(name)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        character(len=:), allocatable :: name
        type(function_def_node), pointer :: fn_node
        type(subroutine_def_node), pointer :: sb_node

        name = ''
        fn_node => get_node_as_function_def(arena, node_index)
        if (associated(fn_node)) then
            if (allocated(fn_node%name)) name = trim(fn_node%name)
            return
        end if
        sb_node => get_node_as_subroutine_def(arena, node_index)
        if (associated(sb_node)) then
            if (allocated(sb_node%name)) name = trim(sb_node%name)
        end if
    end function procedure_fortran_name

    subroutine fmod_procedure_signature(arena, node_index, mod_node, kind_text, &
                                        nargs, arg_tokens)
        ! Classify a module procedure for .fmod export. kind_text is 'integer'
        ! for an integer function, 'subroutine' for a subroutine, and empty for
        ! anything not exportable (private, non-integer result, unsupported or
        ! non-scalar argument, or a nested internal procedure). nargs is the
        ! argument count and arg_tokens the space-joined per-argument scalar-kind
        ! tokens for an exportable procedure (#284).
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(module_node), intent(in) :: mod_node
        character(len=:), allocatable, intent(out) :: kind_text
        integer, intent(out) :: nargs
        character(len=:), allocatable, intent(out) :: arg_tokens
        type(function_def_node), pointer :: fn_node
        type(subroutine_def_node), pointer :: sb_node

        kind_text = ''
        nargs = 0
        arg_tokens = ''
        if (.not. node_exists(arena, node_index)) return
        fn_node => get_node_as_function_def(arena, node_index)
        if (associated(fn_node)) then
            if (.not. allocated(fn_node%name)) return
            if (module_symbol_is_private(arena, mod_node, fn_node%name)) return
            if (procedure_has_nested_contains(arena, fn_node%body_indices)) return
            if (.not. return_type_is_integer(fn_node%return_type)) return
            if (.not. params_all_supported_scalar(arena, fn_node%param_indices, &
                                    fn_node%body_indices, nargs, arg_tokens)) return
            kind_text = 'integer'
            return
        end if
        sb_node => get_node_as_subroutine_def(arena, node_index)
        if (associated(sb_node)) then
            if (.not. allocated(sb_node%name)) return
            if (module_symbol_is_private(arena, mod_node, sb_node%name)) return
            if (procedure_has_nested_contains(arena, sb_node%body_indices)) return
            if (.not. params_all_supported_scalar(arena, sb_node%param_indices, &
                                    sb_node%body_indices, nargs, arg_tokens)) return
            kind_text = 'subroutine'
        end if
    end subroutine fmod_procedure_signature

    logical function return_type_is_integer(return_type)
        character(len=:), allocatable, intent(in) :: return_type
        character(len=:), allocatable :: lowered

        return_type_is_integer = .false.
        if (.not. allocated(return_type)) return
        lowered = lowercase_text(trim(return_type))
        return_type_is_integer = lowered == 'integer'
    end function return_type_is_integer

    logical function params_all_supported_scalar(arena, param_indices, &
                                    body_indices, nargs, arg_tokens) result(ok)
        ! Whether every dummy of a module procedure is a scalar of a kind ffc can
        ! pass by reference across separate compilation (integer, real, logical
        ! of the supported kinds). arg_tokens receives the space-joined per-dummy
        ! kind tokens. Any array, character, derived, or unsupported-kind dummy
        ! disqualifies the procedure so a using unit never miscompiles a call
        ! (#284).
        type(ast_arena_t), intent(in) :: arena
        integer, allocatable, intent(in) :: param_indices(:)
        integer, allocatable, intent(in) :: body_indices(:)
        integer, intent(out) :: nargs
        character(len=:), allocatable, intent(out) :: arg_tokens
        integer :: i, value_kind
        character(len=:), allocatable :: token

        ok = .false.
        nargs = 0
        arg_tokens = ''
        if (.not. allocated(param_indices)) then
            ok = .true.
            return
        end if
        nargs = size(param_indices)
        do i = 1, nargs
            value_kind = param_at_value_kind(arena, param_indices, &
                                             body_indices, i)
            token = scalar_kind_token(value_kind)
            if (len_trim(token) == 0) return
            if (i == 1) then
                arg_tokens = token
            else
                arg_tokens = arg_tokens//' '//token
            end if
        end do
        ok = .true.
    end function params_all_supported_scalar

    function scalar_kind_token(value_kind) result(token)
        ! The .fmod token for a by-reference scalar dummy kind, or '' when the
        ! kind is not exportable across separate compilation (#284).
        integer, intent(in) :: value_kind
        character(len=:), allocatable :: token

        select case (value_kind)
        case (VALUE_I32)
            token = 'integer'
        case (VALUE_I64)
            token = 'integer8'
        case (VALUE_I8)
            token = 'integer1'
        case (VALUE_I16)
            token = 'integer2'
        case (VALUE_F32)
            token = 'real'
        case (VALUE_F64)
            token = 'real8'
        case (VALUE_LOGICAL)
            token = 'logical'
        case default
            token = ''
        end select
    end function scalar_kind_token

    integer function value_kind_of_token(token) result(value_kind)
        ! Inverse of scalar_kind_token: the value kind a .fmod scalar token
        ! denotes, or 0 when unrecognised (#284).
        character(len=*), intent(in) :: token

        select case (trim(token))
        case ('integer')
            value_kind = VALUE_I32
        case ('integer8')
            value_kind = VALUE_I64
        case ('integer1')
            value_kind = VALUE_I8
        case ('integer2')
            value_kind = VALUE_I16
        case ('real')
            value_kind = VALUE_F32
        case ('real8')
            value_kind = VALUE_F64
        case ('logical')
            value_kind = VALUE_LOGICAL
        case default
            value_kind = 0
        end select
    end function value_kind_of_token

    subroutine build_fmod_variable(arena, node_index, var, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(fmod_variable_t), intent(out) :: var
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: name, type_name
        integer :: cur
        character(len=:), allocatable :: node_type
        character(len=:), allocatable :: module_name
        type(module_node), pointer :: node

        call set_empty(error_msg)
        var%name = ''
        var%kind = ''
        var%c_name = ''
        if (.not. is_declaration_node(arena, node_index)) then
            error_msg = 'module variable export is not a declaration'
            return
        end if
        module_name = ''
        cur = get_parent(arena, node_index)
        do while (cur > 0)
            node_type = get_node_type_at(arena, cur)
            if (node_type == 'module_node') then
                select type (node => arena%entries(cur)%node)
                type is (module_node)
                    if (allocated(node%name)) module_name = trim(node%name)
                end select
                exit
            end if
            cur = get_parent(arena, cur)
        end do
        call get_declaration_var_name(arena, node_index, name, error_msg)
        if (len_trim(error_msg) > 0) return
        var%name = trim(name)
        call get_declaration_type_name(arena, node_index, type_name, error_msg)
        if (len_trim(error_msg) > 0) return
        var%kind = fmod_variable_kind_token(type_name)
        if (len_trim(module_name) > 0) then
            var%c_name = module_procedure_mangled(trim(module_name), trim(name))
        end if
    end subroutine build_fmod_variable

    function fmod_variable_kind_token(type_name) result(token)
        ! The precise .fmod kind token for a scalar module variable so a using
        ! unit imports it with the correct load/store kind (integer vs real,
        ! single vs double). A character or derived type keeps the broad token;
        ! its value is not imported through the scalar path (#284).
        character(len=*), intent(in) :: type_name
        character(len=:), allocatable :: token, kind_err
        integer :: value_kind

        call type_name_value_kind(type_name, 0, 0, value_kind, kind_err)
        if (len_trim(kind_err) == 0) then
            token = scalar_kind_token(value_kind)
            if (len_trim(token) > 0) return
        end if
        token = fmod_kind_string(type_name)
    end function fmod_variable_kind_token

    subroutine build_fmod_parameter(arena, node_index, param, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(fmod_parameter_t), intent(out) :: param
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: lit_value, lit_type, lit_err
        character(len=:), allocatable :: name, type_name
        integer :: init_index

        call set_empty(error_msg)
        param%name = ''
        param%kind = ''
        param%value = ''
        if (.not. is_declaration_node(arena, node_index)) then
            error_msg = 'module parameter export is not a declaration'
            return
        end if
        call get_declaration_var_name(arena, node_index, name, error_msg)
        if (len_trim(error_msg) > 0) return
        param%name = trim(name)
        call get_declaration_type_name(arena, node_index, type_name, error_msg)
        if (len_trim(error_msg) > 0) return
        param%kind = fmod_kind_string(type_name)
        if (get_declaration_has_initializer(arena, node_index)) then
            init_index = get_declaration_initializer_index(arena, node_index)
            if (init_index > 0 .and. is_literal(arena, init_index)) then
                call get_literal_info(arena, init_index, lit_value, lit_type, &
                                      lit_err)
                if (len_trim(lit_err) == 0) param%value = trim(lit_value)
            end if
        end if
    end subroutine build_fmod_parameter

    subroutine build_fmod_derived_type(arena, node_index, dtype, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(fmod_derived_type_t), intent(out) :: dtype
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: n, k
        character(len=:), allocatable :: type_name, comp_name, comp_type
        integer, allocatable :: component_indices(:)

        call set_empty(error_msg)
        dtype%name = ''
        if (.not. is_derived_type_node(arena, node_index)) then
            error_msg = 'module derived-type export is not a derived type'
            return
        end if
        call get_derived_type_name(arena, node_index, type_name, error_msg)
        if (len_trim(error_msg) > 0) return
        dtype%name = trim(type_name)
        call get_derived_type_components(arena, node_index, component_indices)
        n = size(component_indices)
        allocate (dtype%components(n))
        do k = 1, n
            dtype%components(k)%name = ''
            dtype%components(k)%kind = ''
            if (.not. is_declaration_node(arena, component_indices(k))) cycle
            call get_declaration_var_name(arena, component_indices(k), &
                                          comp_name, error_msg)
            if (len_trim(error_msg) > 0) return
            dtype%components(k)%name = trim(comp_name)
            call get_declaration_type_name(arena, component_indices(k), &
                                           comp_type, error_msg)
            if (len_trim(error_msg) > 0) return
            dtype%components(k)%kind = fmod_kind_string(comp_type)
        end do
    end subroutine build_fmod_derived_type

    function fmod_kind_string(type_name) result(kind_text)
        ! Normalise a declaration type name to the .fmod kind token.
        character(len=*), intent(in) :: type_name
        character(len=:), allocatable :: kind_text
        character(len=:), allocatable :: lowered

        lowered = lowercase_text(type_name)
        if (index(lowered, 'character') == 1) then
            kind_text = 'character'
        else if (index(lowered, 'real') == 1 .or. &
                 index(lowered, 'double precision') == 1) then
            kind_text = 'real'
        else if (index(lowered, 'logical') == 1) then
            kind_text = 'logical'
        else if (index(lowered, 'integer') == 1) then
            kind_text = 'integer'
        else if (index(lowered, 'type(') == 1) then
            kind_text = trim(type_name)
        else
            kind_text = trim(type_name)
        end if
    end function fmod_kind_string
