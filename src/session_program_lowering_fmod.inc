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
    end subroutine build_module_info

    subroutine build_fmod_parameter(arena, node_index, param, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(fmod_parameter_t), intent(out) :: param
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: lit_value, lit_type, lit_err

        call set_empty(error_msg)
        param%name = ''
        param%kind = ''
        param%value = ''
        select type (decl => arena%entries(node_index)%node)
        type is (declaration_node)
            if (allocated(decl%var_name)) param%name = trim(decl%var_name)
            if (allocated(decl%type_name)) param%kind = fmod_kind_string(decl%type_name)
            if (decl%has_initializer .and. decl%initializer_index > 0) then
                if (is_literal(arena, decl%initializer_index)) then
                    call get_literal_info(arena, decl%initializer_index, &
                                          lit_value, lit_type, lit_err)
                    if (len_trim(lit_err) == 0) param%value = trim(lit_value)
                end if
            end if
        class default
            error_msg = 'module parameter export is not a declaration'
        end select
    end subroutine build_fmod_parameter

    subroutine build_fmod_derived_type(arena, node_index, dtype, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(fmod_derived_type_t), intent(out) :: dtype
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: n, k

        call set_empty(error_msg)
        dtype%name = ''
        select type (node => arena%entries(node_index)%node)
        type is (derived_type_node)
            if (allocated(node%name)) dtype%name = trim(node%name)
            n = 0
            if (allocated(node%component_indices)) n = size(node%component_indices)
            allocate (dtype%components(n))
            do k = 1, n
                dtype%components(k)%name = ''
                dtype%components(k)%kind = ''
                select type (comp => arena%entries(node%component_indices(k))%node)
                type is (declaration_node)
                    if (allocated(comp%var_name)) &
                        dtype%components(k)%name = trim(comp%var_name)
                    if (allocated(comp%type_name)) &
                        dtype%components(k)%kind = fmod_kind_string(comp%type_name)
                end select
            end do
        class default
            error_msg = 'module derived-type export is not a derived type'
        end select
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
