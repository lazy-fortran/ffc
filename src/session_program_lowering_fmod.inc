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
    end subroutine build_module_info

    subroutine build_fmod_variable(arena, node_index, var, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(fmod_variable_t), intent(out) :: var
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: name, type_name

        call set_empty(error_msg)
        var%name = ''
        var%kind = ''
        if (.not. is_declaration_node(arena, node_index)) then
            error_msg = 'module variable export is not a declaration'
            return
        end if
        call get_declaration_var_name(arena, node_index, name, error_msg)
        if (len_trim(error_msg) > 0) return
        var%name = trim(name)
        call get_declaration_type_name(arena, node_index, type_name, error_msg)
        if (len_trim(error_msg) > 0) return
        var%kind = fmod_kind_string(type_name)
    end subroutine build_fmod_variable

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
