submodule(session_program_lowering_impl) session_program_lowering_fmod_exporter
    use ast_arena_source_text, only: get_source_line
contains
    module subroutine emit_module_fmod_artifacts(arena, context, output_path, error_msg)
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
            call build_module_info(arena, context, context%module_exports(m), &
                                   info, error_msg)
            if (len_trim(error_msg) > 0) return
            path = dir//trim(context%module_exports(m)%module_name)//'.fmod'
            call write_fmod(path, info, error_msg)
            if (len_trim(error_msg) > 0) return
        end do
    end subroutine emit_module_fmod_artifacts

    module function path_dirname(path) result(dir)
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

    module subroutine build_module_info(arena, context, export, info, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(lowering_context_t), intent(in) :: context
        type(module_exports_t), intent(in) :: export
        type(module_info_t), intent(out) :: info
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i, j, module_index, derived_count, imported_count, use_count
        integer :: type_index
        integer, allocatable :: derived_indices(:)
        integer, allocatable :: imported_type_indices(:)
        character(len=:), allocatable :: exported_name
        logical :: reexported

        call set_empty(error_msg)
        info%name = trim(export%module_name)
        allocate (info%parameters(export%parameter_count))
        do i = 1, export%parameter_count
            call build_fmod_parameter(arena, export%parameter_indices(i), &
                                      info%parameters(i), error_msg)
            if (len_trim(error_msg) > 0) return
        end do
        ! A public derived type may contain a private helper type.  The helper
        ! is not a USE export, but its layout is needed to reconstruct the
        ! public type in a separately compiled user (e.g. a private table
        ! component in a public tokenizer).  Include all derived declarations
        ! belonging to this module in the artefact; USE visibility is still
        ! enforced by the source-unit path.
        module_index = 0
        do i = 1, arena%size
            if (.not. node_exists(arena, i)) cycle
            if (arena%entries(i)%parent_index /= 0) cycle
            select type (node => arena%entries(i)%node)
            type is (module_node)
                if (allocated(node%name) .and. same_name(node%name, &
                        export%module_name)) module_index = i
            end select
        end do
        use_count = 0
        if (module_index > 0) then
            select type (module => arena%entries(module_index)%node)
            type is (module_node)
                if (allocated(module%declaration_indices)) then
                    do i = 1, size(module%declaration_indices)
                        if (.not. node_exists(arena, module%declaration_indices(i))) cycle
                        select type (use_node => arena%entries( &
                                module%declaration_indices(i))%node)
                        type is (use_statement_node)
                            if (allocated(use_node%module_name)) use_count = &
                                use_count + 1
                        end select
                    end do
                end if
            end select
        end if
        allocate (info%uses(use_count))
        if (use_count > 0) then
            j = 0
            select type (module => arena%entries(module_index)%node)
            type is (module_node)
                do i = 1, size(module%declaration_indices)
                    if (.not. node_exists(arena, module%declaration_indices(i))) cycle
                    select type (use_node => arena%entries( &
                            module%declaration_indices(i))%node)
                    type is (use_statement_node)
                        if (.not. allocated(use_node%module_name)) cycle
                        j = j + 1
                        info%uses(j)%name = trim(use_node%module_name)
                    end select
                end do
            end select
        end if
        ! The module export collector has already selected the public derived
        ! declarations and retained their arena indices.  Do not reconstruct
        ! that set from parent links: derived declarations in a module can be
        ! represented through a different arena nesting path, which otherwise
        ! silently drops them from the .fmod artefact (#540).
        derived_count = export%derived_type_count
        allocate (derived_indices(derived_count))
        if (derived_count > 0) then
            derived_indices = export%derived_type_indices(1:derived_count)
        end if
        ! A public module also re-exports a public derived type admitted by a
        ! USE statement. The lowering context already reconstructed that type
        ! from the dependency's .fmod; preserve it in this module's artefact
        ! so a further separately compiled user sees the same public name
        ! (#447).
        imported_count = 0
        do i = 1, context%derived_type_count
            if (.not. context%derived_types(i)%is_imported) cycle
            if (module_reexports_type(arena, module_index, &
                                      context%derived_types(i)%name, &
                                      exported_name)) then
                imported_count = imported_count + 1
            end if
        end do
        allocate (imported_type_indices(imported_count))
        j = 0
        do i = 1, context%derived_type_count
            if (.not. context%derived_types(i)%is_imported) cycle
            if (.not. module_reexports_type(arena, module_index, &
                                            context%derived_types(i)%name, &
                                            exported_name)) cycle
            j = j + 1
            imported_type_indices(j) = i
        end do
        allocate (info%derived_types(derived_count + imported_count))
        do i = 1, derived_count
            call build_fmod_derived_type(arena, context, &
                                         derived_indices(i), &
                                         info%derived_types(i), error_msg)
            if (len_trim(error_msg) > 0) return
            info%derived_types(i)%canonical_identity = trim(export%module_name)// &
                '::'//trim(info%derived_types(i)%canonical_name)
        end do
        deallocate (derived_indices)
        do i = 1, imported_count
            type_index = imported_type_indices(i)
            call build_fmod_derived_type_from_context(context, type_index, &
                                                      info%derived_types( &
                                                      derived_count + i), &
                                                      error_msg)
            if (len_trim(error_msg) > 0) return
            reexported = module_reexports_type(arena, module_index, &
                                               context%derived_types(type_index)%name, &
                                               exported_name)
            if (reexported .and. len_trim(exported_name) > 0) then
                info%derived_types(derived_count + i)%name = trim(exported_name)
            end if
        end do
        deallocate (imported_type_indices)
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
        call build_fmod_procedures(arena, context, trim(export%module_name), &
                                   info%procedures, error_msg)
        if (len_trim(error_msg) > 0) return
        ! Named generic interfaces whose specifics are all exportable procedures
        ! round-trip so a use-associated generic call in a separately compiled
        ! program resolves to the matching specific and links.
        call build_fmod_generics(arena, trim(export%module_name), &
                                 info%procedures, info%generics, error_msg)
    end subroutine build_module_info

    module function module_reexports_type(arena, module_index, type_name, &
                                          local_name) &
            result(reexports)
        logical :: reexports
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: module_index
        character(len=*), intent(in) :: type_name
        character(len=:), allocatable, optional, intent(out) :: local_name
        character(len=:), allocatable :: candidate
        logical :: imported
        integer :: i, j

        reexports = .false.
        if (present(local_name)) local_name = ''
        if (module_index <= 0) return
        select type (module => arena%entries(module_index)%node)
        type is (module_node)
            if (.not. allocated(module%declaration_indices)) return
            do i = 1, size(module%declaration_indices)
                if (.not. node_exists(arena, module%declaration_indices(i))) cycle
                select type (use_node => arena%entries( &
                        module%declaration_indices(i))%node)
                type is (use_statement_node)
                    if (.not. allocated(use_node%module_name)) cycle
                    call resolve_export_import_name(use_node, trim(type_name), &
                                                  candidate, imported)
                    if (imported) then
                        reexports = .true.
                        if (present(local_name)) local_name = trim(candidate)
                        return
                    end if
                    if (allocated(use_node%rename_list)) then
                        do j = 1, size(use_node%rename_list) - 1, 2
                            if (.not. allocated(use_node%rename_list(j)%s)) cycle
                            if (.not. same_name(use_node%rename_list(j)%s, &
                                                type_name)) cycle
                            reexports = .true.
                            if (present(local_name)) local_name = &
                                trim(use_node%rename_list(j)%s)
                            return
                        end do
                    end if
                end select
            end do
        end select
    end function module_reexports_type

    subroutine resolve_export_import_name(use_node, remote_name, local_name, &
                                          imported)
        ! The exporter is a separate submodule object. Keep this tiny
        ! read-only USE-name resolver local so a fresh build does not create a
        ! link-time dependency on the parent implementation's internal helper.
        type(use_statement_node), intent(in) :: use_node
        character(len=*), intent(in) :: remote_name
        character(len=:), allocatable, intent(out) :: local_name
        logical, intent(out) :: imported
        integer :: i

        local_name = trim(remote_name)
        imported = .true.
        if (allocated(use_node%rename_list)) then
            do i = 1, size(use_node%rename_list) - 1, 2
                if (.not. allocated(use_node%rename_list(i)%s)) cycle
                if (.not. allocated(use_node%rename_list(i + 1)%s)) cycle
                if (same_name(use_node%rename_list(i + 1)%s, remote_name)) then
                    local_name = trim(use_node%rename_list(i)%s)
                    return
                end if
            end do
        end if
        if (.not. use_node%has_only) return
        imported = .false.
        if (.not. allocated(use_node%only_list)) return
        do i = 1, size(use_node%only_list)
            if (.not. allocated(use_node%only_list(i)%s)) cycle
            if (same_name(use_node%only_list(i)%s, remote_name)) then
                imported = .true.
                return
            end if
        end do
    end subroutine resolve_export_import_name

    module subroutine build_fmod_generics(arena, module_name, procs, generics, error_msg)
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

    module function generic_block_name(arena, node_index) result(name)
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

    module subroutine fmod_generic_specifics(arena, node_index, procs, module_name, &
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

    module subroutine append_generic_specific(procs, name, specifics)
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
                    if (procs(p)%callable) then
                        found = .true.
                        exit
                    end if
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

    module subroutine grow_fmod_generics(arr, n)
        type(fmod_generic_t), allocatable, intent(inout) :: arr(:)
        integer, intent(in) :: n
        type(fmod_generic_t), allocatable :: tmp(:)

        if (n <= size(arr)) return
        allocate (tmp(n))
        tmp(1:size(arr)) = arr
        call move_alloc(tmp, arr)
    end subroutine grow_fmod_generics

    module subroutine build_fmod_procedures(arena, context, module_name, procs, &
                                     error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: module_name
        type(fmod_procedure_t), allocatable, intent(out) :: procs(:)
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: module_index, p, count
        integer, allocatable :: declaration_indices(:), procedure_indices(:)
        character(len=:), allocatable :: mod_name, ff_error, kind_text, arg_tokens
        character(len=:), allocatable :: rank_tokens, extent_tokens
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
            call record_fmod_procedure(arena, context, procedure_indices(p), &
                                       mod_node, .false., procs, count)
        end do
        ! A module procedure whose interface the module declares and whose body
        ! a submodule supplies is exported too: its symbol is mangled to this
        ! module, so a separately compiled caller resolves and links it exactly
        ! like a contained module procedure, and a separately compiled submodule
        ! reads the interface it has to implement from here (#297).
        do p = 1, size(declaration_indices)
            call record_fmod_interface_procedures(arena, context, &
                                                  declaration_indices(p), &
                                                  mod_node, procs, count)
        end do
    end subroutine build_fmod_procedures

    module subroutine record_fmod_interface_procedures(arena, context, node_index, &
                                                mod_node, procs, count)
        ! Record deferred module-procedure bodies and public plain-interface
        ! procedures. The latter bind to their unmangled external symbol.
        type(ast_arena_t), intent(in) :: arena
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: node_index
        type(module_node), intent(in) :: mod_node
        type(fmod_procedure_t), allocatable, intent(inout) :: procs(:)
        integer, intent(inout) :: count
        integer :: i
        character(len=:), allocatable :: node_type

        if (.not. node_exists(arena, node_index)) return
        select type (block => arena%entries(node_index)%node)
        type is (interface_block_node)
            if (block%is_abstract) return
            if (.not. allocated(block%procedure_indices)) return
            do i = 1, size(block%procedure_indices)
                if (procedure_is_deferred_module_body(arena, &
                        block%procedure_indices(i))) then
                    call record_fmod_procedure(arena, context, &
                                               block%procedure_indices(i), &
                                               mod_node, .true., procs, count)
                    cycle
                end if
                node_type = get_node_type_at(arena, block%procedure_indices(i))
                if (node_type == 'function_def_node' .or. &
                    node_type == 'function_def' .or. &
                    node_type == 'subroutine_def_node' .or. &
                    node_type == 'subroutine_def') then
                    call record_fmod_procedure(arena, context, &
                                               block%procedure_indices(i), &
                                               mod_node, .false., procs, count, &
                                               .true.)
                end if
            end do
        end select
    end subroutine record_fmod_interface_procedures

    module function procedure_is_deferred_module_body(arena, node_index) &
            result(is_module_body)
        logical :: is_module_body
        ! Whether an interface body is written `module function` /
        ! `module subroutine`, which makes it a module procedure of the
        ! enclosing module whose body a submodule supplies (F2018 15.6.2.5).
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(function_def_node), pointer :: fn_node
        type(subroutine_def_node), pointer :: sb_node

        is_module_body = .false.
        if (.not. node_exists(arena, node_index)) return
        fn_node => get_node_as_function_def(arena, node_index)
        if (associated(fn_node)) then
            is_module_body = prefix_has_module(fn_node%prefix_keywords)
            return
        end if
        sb_node => get_node_as_subroutine_def(arena, node_index)
        if (associated(sb_node)) &
            is_module_body = prefix_has_module(sb_node%prefix_keywords)
    end function procedure_is_deferred_module_body

    module function prefix_has_module(prefix_keywords) result(has_module)
        logical :: has_module
        character(len=16), allocatable, intent(in) :: prefix_keywords(:)
        integer :: i

        has_module = .false.
        if (.not. allocated(prefix_keywords)) return
        do i = 1, size(prefix_keywords)
            if (trim(lowercase_text(prefix_keywords(i))) == 'module') then
                has_module = .true.
                return
            end if
        end do
    end function prefix_has_module

    module subroutine record_fmod_procedure(arena, context, node_index, mod_node, &
                                     deferred_body, procs, count, &
                                     external_binding)
        ! Append one exportable module procedure's signature record. The same
        ! record describes a procedure whose body this module contains and one
        ! whose body a submodule supplies; deferred_body only says which, since
        ! both carry the same mangled symbol and the same call contract (#297).
        type(ast_arena_t), intent(in) :: arena
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: node_index
        type(module_node), intent(in) :: mod_node
        logical, intent(in) :: deferred_body
        type(fmod_procedure_t), allocatable, intent(inout) :: procs(:)
        integer, intent(inout) :: count
        logical, intent(in), optional :: external_binding
        character(len=:), allocatable :: kind_text, arg_tokens
        character(len=:), allocatable :: rank_tokens, extent_tokens
        character(len=:), allocatable :: proc_name
        character(len=:), allocatable :: arg_names
        character(len=:), allocatable :: token
        integer, allocatable :: param_indices(:)
        integer :: nargs, i, value_kind
        logical :: is_external, opaque_subroutine, is_bind_c_procedure
        type(subroutine_def_node), pointer :: sb_node

        is_external = .false.
        if (present(external_binding)) is_external = external_binding
        is_bind_c_procedure = procedure_has_bind_c(arena, node_index)

        call fmod_procedure_signature(arena, context, node_index, mod_node, &
                                      kind_text, nargs, arg_tokens, &
                                      rank_tokens, extent_tokens, &
                                      allow_runtime_array=is_external)
        if (len_trim(kind_text) == 0) then
            ! A public subroutine can still be useful as interface metadata
            ! when one or more dummies are outside the lowering ABI. Keep a
            ! scalar placeholder for each dummy so a using unit can validate
            ! its call shape; the actual Fortran dummy contract remains in the
            ! separate attribute fields below (#584).
            sb_node => get_node_as_subroutine_def(arena, node_index)
            opaque_subroutine = associated(sb_node)
            if (opaque_subroutine) then
                if (.not. allocated(sb_node%name)) then
                    opaque_subroutine = .false.
                else if (module_symbol_is_private(arena, mod_node, sb_node%name)) then
                    opaque_subroutine = .false.
                else if (allocated(sb_node%param_indices)) then
                    param_indices = sb_node%param_indices
                end if
            end if
            if (opaque_subroutine) then
                count = count + 1
                call grow_fmod_procs(procs, count)
                procs(count)%name = trim(sb_node%name)
                procs(count)%external_name = ''
                if (is_external .or. is_bind_c_procedure) procs(count)%external_name = &
                    fmod_procedure_external_name(arena, node_index)
                procs(count)%kind = 'subroutine'
                procs(count)%nargs = 0
                procs(count)%arg_kinds = ''
                arg_names = fmod_procedure_arg_names(arena, node_index)
                procs(count)%arg_names = arg_names
                procs(count)%arg_intents = ''
                procs(count)%arg_optionals = ''
                procs(count)%arg_values = ''
                procs(count)%result_name = ''
                procs(count)%result_kind = ''
                procs(count)%arg_ranks = ''
                procs(count)%arg_extents = ''
                procs(count)%arg_classes = ''
                procs(count)%arg_class_types = ''
                procs(count)%arg_class_type_identities = ''
                procs(count)%opaque = .true.
                procs(count)%callable = .true.
                procs(count)%external_binding = is_external .or. is_bind_c_procedure
                procs(count)%deferred_body = deferred_body
                if (allocated(param_indices)) then
                    nargs = size(param_indices)
                    call fmod_procedure_name_count(arg_names, nargs)
                    procs(count)%nargs = nargs
                    do i = 1, nargs
                        if (i > 1) then
                            procs(count)%arg_kinds = &
                                procs(count)%arg_kinds//' '
                            procs(count)%arg_ranks = &
                                procs(count)%arg_ranks//' '
                            procs(count)%arg_extents = &
                                procs(count)%arg_extents//' '
                        end if
                        ! Keep the per-dummy ABI precise even when another
                        ! dummy makes the whole procedure opaque. A supported
                        ! scalar or character dummy remains callable through
                        ! its normal path; only the unsupported dummy needs
                        ! the opaque placeholder.
                        value_kind = param_at_value_kind(arena, param_indices, &
                            sb_node%body_indices, i, context)
                        token = scalar_kind_token(value_kind)
                        if (len_trim(token) > 0) then
                            procs(count)%arg_kinds = &
                                procs(count)%arg_kinds//token
                        else if (param_at_is_character(arena, param_indices, &
                                                       sb_node%body_indices, i)) then
                            procs(count)%arg_kinds = &
                                procs(count)%arg_kinds//'character'
                        else
                            procs(count)%arg_kinds = &
                                procs(count)%arg_kinds//'opaque'
                        end if
                        procs(count)%arg_ranks = procs(count)%arg_ranks//'0'
                        procs(count)%arg_extents = procs(count)%arg_extents//'0'
                    end do
                end if
                call fmod_procedure_dummy_attributes(arena, node_index, &
                                                     procs(count)%arg_intents, &
                                                     procs(count)%arg_optionals, &
                                                     procs(count)%arg_values)
                call fmod_procedure_arg_class_info(arena, node_index, context, &
                                                   procs(count)%arg_classes, &
                                                   procs(count)%arg_class_types, &
                                                   procs(count)%arg_class_type_identities)
                call pad_fmod_dummy_attributes(procs(count)%nargs, &
                                               procs(count)%arg_intents, &
                                               procs(count)%arg_optionals, &
                                               procs(count)%arg_values)
                return
            end if
            ! The procedure is still a public module export even when its call
            ! ABI is not supported by the direct session backend. Preserve its
            ! name in the .fmod so USE ONLY validates against the real module
            ! interface; the reader will not register a callable external for
            ! this record (#584).
            proc_name = procedure_fortran_name(arena, node_index)
            if (len_trim(proc_name) == 0) return
            if (module_symbol_is_private(arena, mod_node, proc_name)) return
            count = count + 1
            call grow_fmod_procs(procs, count)
            procs(count)%name = proc_name
            procs(count)%external_name = ''
            if (is_external .or. is_bind_c_procedure) procs(count)%external_name = &
                fmod_procedure_external_name(arena, node_index)
            procs(count)%kind = 'unsupported'
            procs(count)%arg_kinds = ''
            procs(count)%arg_names = ''
            procs(count)%arg_intents = ''
            procs(count)%arg_optionals = ''
            procs(count)%arg_values = ''
            procs(count)%result_name = ''
            procs(count)%result_kind = ''
            procs(count)%arg_ranks = ''
            procs(count)%arg_extents = ''
            procs(count)%arg_classes = ''
            procs(count)%arg_class_types = ''
            procs(count)%arg_class_type_identities = ''
            procs(count)%callable = .false.
            procs(count)%external_binding = is_external .or. is_bind_c_procedure
            procs(count)%deferred_body = deferred_body
            procs(count)%nargs = 0
            return
        end if
        count = count + 1
        call grow_fmod_procs(procs, count)
        procs(count)%name = procedure_fortran_name(arena, node_index)
        procs(count)%external_name = ''
        if (is_external .or. is_bind_c_procedure) procs(count)%external_name = &
            fmod_procedure_external_name(arena, node_index)
        procs(count)%kind = kind_text
        procs(count)%nargs = nargs
        procs(count)%arg_kinds = arg_tokens
        procs(count)%arg_ranks = rank_tokens
        procs(count)%arg_extents = extent_tokens
        call fmod_procedure_arg_class_info(arena, node_index, context, &
                                           procs(count)%arg_classes, &
                                           procs(count)%arg_class_types, &
                                           procs(count)%arg_class_type_identities)
        procs(count)%callable = .true.
        procs(count)%external_binding = is_external .or. is_bind_c_procedure
        procs(count)%arg_names = fmod_procedure_arg_names(arena, node_index)
        procs(count)%deferred_body = deferred_body
        call fmod_procedure_dummy_attributes(arena, node_index, &
                                             procs(count)%arg_intents, &
                                             procs(count)%arg_optionals, &
                                             procs(count)%arg_values)
        call fmod_procedure_result(arena, node_index, kind_text, &
                                   procs(count)%result_name, &
                                   procs(count)%result_kind)
    end subroutine record_fmod_procedure

    logical function procedure_has_bind_c(arena, node_index) result(has_bind)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(function_def_node), pointer :: fn_node
        type(subroutine_def_node), pointer :: sb_node

        has_bind = .false.
        fn_node => get_node_as_function_def(arena, node_index)
        if (associated(fn_node)) then
            if (allocated(fn_node%bind_c_clause)) has_bind = &
                has_bind_c(fn_node%bind_c_clause)
            return
        end if
        sb_node => get_node_as_subroutine_def(arena, node_index)
        if (associated(sb_node)) then
            if (allocated(sb_node%bind_c_clause)) has_bind = &
                has_bind_c(sb_node%bind_c_clause)
        end if
    end function procedure_has_bind_c

    module subroutine fmod_procedure_result(arena, node_index, kind_text, result_name, &
                                     result_kind)
        ! The result-variable name and scalar kind token of an exported module
        ! function, so a using unit reconstructs the same result contract the
        ! defining unit compiled. Both empty for a subroutine (#397).
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        character(len=*), intent(in) :: kind_text
        character(len=:), allocatable, intent(out) :: result_name
        character(len=:), allocatable, intent(out) :: result_kind
        type(function_def_node), pointer :: fn_node

        result_name = ''
        result_kind = ''
        if (trim(kind_text) == 'subroutine') return
        fn_node => get_node_as_function_def(arena, node_index)
        if (.not. associated(fn_node)) return
        if (allocated(fn_node%result_variable)) then
            result_name = trim(fn_node%result_variable)
        else if (allocated(fn_node%name)) then
            result_name = trim(fn_node%name)
        end if
        result_kind = trim(kind_text)
    end subroutine fmod_procedure_result

    module subroutine fmod_procedure_dummy_attributes(arena, node_index, intents, &
                                               optionals, values)
        ! The per-dummy INTENT, OPTIONAL, and VALUE contracts of an exported
        ! module procedure, space-joined one token per dummy. A separately
        ! compiled caller reads them back to omit absent optionals, to pass a
        ! VALUE dummy a copy, and to reject a non-definable actual for an
        ! INTENT(OUT) dummy, none of which it can infer without the source
        ! (#397).
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        character(len=:), allocatable, intent(out) :: intents
        character(len=:), allocatable, intent(out) :: optionals
        character(len=:), allocatable, intent(out) :: values
        type(function_def_node), pointer :: fn_node
        type(subroutine_def_node), pointer :: sb_node
        integer, allocatable :: param_indices(:), body_indices(:)
        character(len=:), allocatable :: intent_token
        logical :: is_optional, is_value
        integer :: i

        intents = ''
        optionals = ''
        values = ''
        fn_node => get_node_as_function_def(arena, node_index)
        if (associated(fn_node)) then
            if (allocated(fn_node%param_indices)) &
                param_indices = fn_node%param_indices
            if (allocated(fn_node%body_indices)) &
                body_indices = fn_node%body_indices
        else
            sb_node => get_node_as_subroutine_def(arena, node_index)
            if (.not. associated(sb_node)) return
            if (allocated(sb_node%param_indices)) &
                param_indices = sb_node%param_indices
            if (allocated(sb_node%body_indices)) &
                body_indices = sb_node%body_indices
        end if
        if (.not. allocated(param_indices)) return
        if (.not. allocated(body_indices)) allocate (body_indices(0))
        do i = 1, size(param_indices)
            call param_at_attributes(arena, param_indices, body_indices, i, &
                                     intent_token, is_optional, is_value)
            if (i > 1) then
                intents = intents//' '
                optionals = optionals//' '
                values = values//' '
            end if
            intents = intents//intent_token
            optionals = optionals//flag_token(is_optional)
            values = values//flag_token(is_value)
        end do
    end subroutine fmod_procedure_dummy_attributes

    subroutine fmod_procedure_arg_class_info(arena, node_index, context, classes, &
                                                    class_types, class_identities)
        ! Preserve the distinction between type(t) and class(t) dummies in
        ! the separate-compilation contract. In particular, an imported
        ! type-bound target cannot recover this from its caller's AST.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        character(len=:), allocatable, intent(out) :: classes
        character(len=:), allocatable, intent(out) :: class_types
        character(len=:), allocatable, intent(out) :: class_identities
        type(function_def_node), pointer :: fn_node
        type(subroutine_def_node), pointer :: sb_node
        integer, allocatable :: param_indices(:), body_indices(:)
        character(len=:), allocatable :: param_name, name_error, type_name
        logical :: is_class
        integer :: i, j, open_pos, close_pos, type_index

        classes = ''
        class_types = ''
        class_identities = ''
        fn_node => get_node_as_function_def(arena, node_index)
        if (associated(fn_node)) then
            if (allocated(fn_node%param_indices)) param_indices = &
                fn_node%param_indices
            if (allocated(fn_node%body_indices)) body_indices = &
                fn_node%body_indices
        else
            sb_node => get_node_as_subroutine_def(arena, node_index)
            if (.not. associated(sb_node)) return
            if (allocated(sb_node%param_indices)) param_indices = &
                sb_node%param_indices
            if (allocated(sb_node%body_indices)) body_indices = &
                sb_node%body_indices
        end if
        if (.not. allocated(param_indices)) return
        if (.not. allocated(body_indices)) allocate (body_indices(0))
        do i = 1, size(param_indices)
            is_class = .false.
            type_name = '-'
            call parameter_name(arena, param_indices(i), param_name, name_error)
            if (len_trim(name_error) == 0) then
                do j = 1, size(body_indices)
                    if (.not. node_exists(arena, body_indices(j))) cycle
                    select type (decl => arena%entries(body_indices(j))%node)
                    type is (declaration_node)
                        if (.not. declaration_declares_name(decl, &
                            trim(lowercase_text(param_name)))) cycle
                        if (.not. allocated(decl%type_name)) exit
                        type_name = trim(lowercase_text(decl%type_name))
                        is_class = .false.
                        if (len_trim(type_name) >= 6) then
                            is_class = type_name(1:6) == 'class('
                        end if
                        if (is_class) then
                            open_pos = index(type_name, '(')
                            close_pos = index(type_name, ')', back=.true.)
                            if (close_pos > open_pos + 1) then
                                if (trim(type_name(open_pos + 1:close_pos - 1)) /= '*') then
                                    type_name = trim(type_name(open_pos + 1:close_pos - 1))
                                else
                                    is_class = .false.
                                    type_name = '-'
                                end if
                            else
                                is_class = .false.
                                type_name = '-'
                            end if
                        else
                            type_name = '-'
                        end if
                        exit
                    end select
                end do
            end if
            if (i > 1) then
                classes = classes//' '
                class_types = class_types//' '
                class_identities = class_identities//' '
            end if
            classes = classes//flag_token(is_class)
            class_types = class_types//type_name
            if (is_class) then
                type_index = find_derived_type(context, type_name)
                if (type_index > 0) then
                    class_identities = class_identities//trim( &
                        context%derived_types(type_index)%canonical_identity)
                else
                    class_identities = class_identities//'-'
                end if
            else
                class_identities = class_identities//'-'
            end if
        end do
    end subroutine fmod_procedure_arg_class_info

    module function flag_token(flag) result(token)
        logical, intent(in) :: flag
        character(len=:), allocatable :: token

        if (flag) then
            token = '1'
        else
            token = '0'
        end if
    end function flag_token

    module subroutine param_at_attributes(arena, param_indices, body_indices, pos, &
                                   intent_token, is_optional, is_value)
        ! The declared INTENT ('in', 'out', 'inout', or 'none'), OPTIONAL, and
        ! VALUE attributes of the pos-th dummy, taken from its declaration in
        ! the procedure body (#397).
        type(ast_arena_t), intent(in) :: arena
        integer, allocatable, intent(in) :: param_indices(:)
        integer, allocatable, intent(in) :: body_indices(:)
        integer, intent(in) :: pos
        character(len=:), allocatable, intent(out) :: intent_token
        logical, intent(out) :: is_optional
        logical, intent(out) :: is_value
        character(len=:), allocatable :: name, name_err
        integer :: i

        intent_token = 'none'
        is_optional = .false.
        is_value = .false.
        if (.not. allocated(param_indices)) return
        if (pos < 1 .or. pos > size(param_indices)) return
        call parameter_name(arena, param_indices(pos), name, name_err)
        if (len_trim(name_err) > 0) return
        if (.not. allocated(body_indices)) return
        do i = 1, size(body_indices)
            if (.not. node_exists(arena, body_indices(i))) cycle
            select type (decl => arena%entries(body_indices(i))%node)
            type is (declaration_node)
                if (.not. declaration_declares_name(decl, &
                    trim(lowercase_text(name)))) cycle
                is_optional = decl%is_optional
                is_value = decl%is_value
                if (decl%has_intent) then
                    if (allocated(decl%intent)) then
                        intent_token = lowercase_text(trim(decl%intent))
                        if (len_trim(intent_token) == 0) intent_token = 'none'
                    end if
                end if
                return
            end select
        end do
    end subroutine param_at_attributes


    module function fmod_procedure_arg_names(arena, node_index) result(tokens)
        ! Space-joined dummy-argument names of an exported module procedure, so
        ! a separately compiled caller can associate keyword actuals with it
        ! (#408). Empty when a name cannot be recovered.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        character(len=:), allocatable :: tokens
        character(len=64), allocatable :: names(:)
        type(function_def_node), pointer :: fn_node
        type(subroutine_def_node), pointer :: sb_node
        integer :: count, i

        tokens = ''
        count = -1
        fn_node => get_node_as_function_def(arena, node_index)
        if (associated(fn_node)) then
            call collect_param_names(arena, fn_node%param_indices, names, count)
        else
            sb_node => get_node_as_subroutine_def(arena, node_index)
            if (.not. associated(sb_node)) return
            call collect_param_names(arena, sb_node%param_indices, names, count)
        end if
        if (count < 0) return
        do i = 1, count
            if (i > 1) tokens = tokens//' '
            tokens = tokens//trim(names(i))
        end do
        call prefer_source_procedure_arg_names(arena, node_index, tokens)
    end function fmod_procedure_arg_names

    module subroutine fmod_procedure_name_count(arg_names, count)
        character(len=*), intent(in) :: arg_names
        integer, intent(inout) :: count
        integer :: n, i

        n = 0
        do i = 1, len_trim(arg_names)
            if (i == 1 .or. arg_names(i - 1:i - 1) == ' ') then
                if (arg_names(i:i) /= ' ') n = n + 1
            end if
        end do
        if (n > count) count = n
    end subroutine fmod_procedure_name_count

    module subroutine prefer_source_procedure_arg_names(arena, node_index, tokens)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        character(len=:), allocatable, intent(inout) :: tokens
        character(len=:), allocatable :: line, source_names
        logical :: found
        integer :: line_number

        line_number = get_node_line(arena, node_index)
        call get_source_line(arena, line_number, line, found)
        if (.not. found) return
        source_names = procedure_header_names(line)
        if (len_trim(source_names) == 0) return
        if (fmod_name_count(source_names) >= fmod_name_count(tokens)) tokens = &
            source_names
    end subroutine prefer_source_procedure_arg_names

    module function fmod_name_count(tokens) result(count)
        character(len=*), intent(in) :: tokens
        integer :: count, i

        count = 0
        do i = 1, len_trim(tokens)
            if (i == 1 .or. tokens(i - 1:i - 1) == ' ') then
                if (tokens(i:i) /= ' ') count = count + 1
            end if
        end do
    end function fmod_name_count

    module function procedure_header_names(line) result(names)
        character(len=*), intent(in) :: line
        character(len=:), allocatable :: names
        character(len=:), allocatable :: lowered, inside, token
        integer :: open_pos, close_pos, start_pos, comma_pos

        names = ''
        lowered = lowercase_text(line)
        open_pos = index(lowered, '(')
        if (open_pos <= 0) return
        close_pos = index(lowered(open_pos + 1:), ')')
        if (close_pos <= 0) return
        close_pos = open_pos + close_pos
        if (close_pos <= open_pos + 1) return
        inside = line(open_pos + 1:close_pos - 1)
        start_pos = 1
        do while (start_pos <= len_trim(inside))
            comma_pos = index(inside(start_pos:), ',')
            if (comma_pos == 0) then
                token = adjustl(inside(start_pos:))
                start_pos = len_trim(inside) + 1
            else
                token = adjustl(inside(start_pos:start_pos + comma_pos - 2))
                start_pos = start_pos + comma_pos
            end if
            if (len_trim(token) == 0) cycle
            if (lowercase_text(trim(token)) == 'ffc_error') token = 'error'
            if (len_trim(names) > 0) names = names//' '
            names = names//trim(token)
        end do
    end function procedure_header_names

    module subroutine pad_fmod_dummy_attributes(nargs, intents, optionals, values)
        integer, intent(in) :: nargs
        character(len=:), allocatable, intent(inout) :: intents, optionals, values

        do while (fmod_name_count(intents) < nargs)
            if (len_trim(intents) > 0) intents = intents//' '
            intents = intents//'none'
        end do
        do while (fmod_name_count(optionals) < nargs)
            if (len_trim(optionals) > 0) optionals = optionals//' '
            optionals = optionals//'0'
        end do
        do while (fmod_name_count(values) < nargs)
            if (len_trim(values) > 0) values = values//' '
            values = values//'0'
        end do
    end subroutine pad_fmod_dummy_attributes

    module subroutine grow_fmod_procs(arr, n)
        type(fmod_procedure_t), allocatable, intent(inout) :: arr(:)
        integer, intent(in) :: n
        type(fmod_procedure_t), allocatable :: tmp(:)

        if (n <= size(arr)) return
        allocate (tmp(n))
        tmp(1:size(arr)) = arr
        call move_alloc(tmp, arr)
    end subroutine grow_fmod_procs

    module function get_module_node_ptr(arena, module_index) result(mod_node)
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

    module function procedure_fortran_name(arena, node_index) result(name)
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

    module function fmod_procedure_external_name(arena, node_index) result(name)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        character(len=:), allocatable :: name
        character(len=:), allocatable :: fortran_name
        type(function_def_node), pointer :: fn_node
        type(subroutine_def_node), pointer :: sb_node

        fortran_name = procedure_fortran_name(arena, node_index)
        name = fortran_name
        fn_node => get_node_as_function_def(arena, node_index)
        if (associated(fn_node)) then
            if (allocated(fn_node%bind_c_clause)) then
                call bind_c_name(fn_node%bind_c_clause, fortran_name, name)
            end if
            return
        end if
        sb_node => get_node_as_subroutine_def(arena, node_index)
        if (associated(sb_node)) then
            if (allocated(sb_node%bind_c_clause)) then
                call bind_c_name(sb_node%bind_c_clause, fortran_name, name)
            end if
        end if
    end function fmod_procedure_external_name

    module subroutine fmod_procedure_signature(arena, context, node_index, mod_node, &
                                        kind_text, nargs, arg_tokens, &
                                        rank_tokens, extent_tokens, &
                                        allow_runtime_array)
        ! Classify a module procedure for .fmod export. kind_text is a scalar
        ! kind token for a function, 'subroutine' for a subroutine, and empty for
        ! anything not exportable (private, unsupported or
        ! non-scalar argument, or a nested internal procedure). nargs is the
        ! argument count and arg_tokens the space-joined per-argument scalar-kind
        ! tokens for an exportable procedure (#284).
        type(ast_arena_t), intent(in) :: arena
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: node_index
        ! Absent when the caller is checking a procedure against an imported
        ! interface rather than exporting this module's own: there is then no
        ! module node to read accessibility from (#297).
        type(module_node), intent(in), optional :: mod_node
        character(len=:), allocatable, intent(out) :: kind_text
        integer, intent(out) :: nargs
        character(len=:), allocatable, intent(out) :: arg_tokens
        character(len=:), allocatable, intent(out) :: rank_tokens
        character(len=:), allocatable, intent(out) :: extent_tokens
        logical, intent(in), optional :: allow_runtime_array
        type(function_def_node), pointer :: fn_node
        type(subroutine_def_node), pointer :: sb_node
        integer :: value_kind
        logical :: runtime_arrays_ok

        kind_text = ''
        nargs = 0
        arg_tokens = ''
        rank_tokens = ''
        extent_tokens = ''
        runtime_arrays_ok = .false.
        if (present(allow_runtime_array)) runtime_arrays_ok = allow_runtime_array
        if (.not. node_exists(arena, node_index)) return
        fn_node => get_node_as_function_def(arena, node_index)
        if (associated(fn_node)) then
            if (.not. allocated(fn_node%name)) return
            if (present(mod_node)) then
                if (module_symbol_is_private(arena, mod_node, fn_node%name)) return
            end if
            if (procedure_has_nested_contains(arena, fn_node%body_indices)) return
            value_kind = fmod_function_result_value_kind(arena, fn_node)
            if (value_kind == 0) return
            if (.not. params_all_supported(arena, context, &
                                    fn_node%param_indices, fn_node%body_indices, &
                                    nargs, arg_tokens, rank_tokens, &
                                    extent_tokens, &
                                    allow_runtime_array=runtime_arrays_ok)) return
            kind_text = scalar_kind_token(value_kind)
            return
        end if
        sb_node => get_node_as_subroutine_def(arena, node_index)
        if (associated(sb_node)) then
            if (.not. allocated(sb_node%name)) return
            if (present(mod_node)) then
                if (module_symbol_is_private(arena, mod_node, sb_node%name)) return
            end if
            if (procedure_has_nested_contains(arena, sb_node%body_indices)) return
            if (.not. params_all_supported(arena, context, &
                                    sb_node%param_indices, sb_node%body_indices, &
                                    nargs, arg_tokens, rank_tokens, &
                                    extent_tokens, &
                                    allow_runtime_array=runtime_arrays_ok)) return
            kind_text = 'subroutine'
        end if
    end subroutine fmod_procedure_signature

    module function fmod_function_result_value_kind(arena, fn_node) &
            result(value_kind)
        integer :: value_kind
        ! Scalar result kind of an exported function. Interface bodies may
        ! state it in the header or in a result declaration (#297).
        type(ast_arena_t), intent(in) :: arena
        type(function_def_node), intent(in) :: fn_node
        character(len=:), allocatable :: result_name, kind_err
        integer :: i

        value_kind = 0
        if (allocated(fn_node%return_type)) then
            if (len_trim(fn_node%return_type) > 0) then
                call type_name_value_kind(fn_node%return_type, 0, 0, &
                                          value_kind, kind_err)
                if (len_trim(kind_err) == 0) return
                value_kind = 0
            end if
        end if
        result_name = ''
        if (allocated(fn_node%result_variable)) result_name = &
            trim(fn_node%result_variable)
        if (len_trim(result_name) == 0 .and. allocated(fn_node%name)) &
            result_name = trim(fn_node%name)
        if (len_trim(result_name) == 0) return
        if (.not. allocated(fn_node%body_indices)) return
        do i = 1, size(fn_node%body_indices)
            if (.not. node_exists(arena, fn_node%body_indices(i))) cycle
            select type (decl => arena%entries(fn_node%body_indices(i))%node)
            type is (declaration_node)
                if (decl%is_array) cycle
                if (.not. declaration_declares_name(decl, &
                    trim(lowercase_text(result_name)))) cycle
                call declaration_value_kind(decl, value_kind, kind_err)
                if (len_trim(kind_err) > 0) return
                return
            end select
        end do
    end function fmod_function_result_value_kind

    module function params_all_supported(arena, context, param_indices, &
                                          body_indices, nargs, arg_tokens, &
                                          rank_tokens, extent_tokens, &
                                          allow_runtime_array) result(ok)
        logical :: ok
        ! Whether every dummy of a module procedure is one this ffc can pass
        ! across separate compilation: a scalar of a supported kind, or an
        ! explicit-shape array of such a scalar, which passes as the base
        ! address of its contiguous storage. Character dummies use the
        ! canonical {data, length} descriptor ABI. arg_tokens receives the
        ! element-kind tokens, rank_tokens the per-dummy rank (0 for a scalar),
        ! and extent_tokens an array dummy's total element count. An assumed-
        ! shape, assumed-rank, allocatable, or derived dummy still disqualifies
        ! the procedure, so a using unit never miscompiles a call (#284, #415).
        type(ast_arena_t), intent(in) :: arena
        type(lowering_context_t), intent(in) :: context
        integer, allocatable, intent(in) :: param_indices(:)
        integer, allocatable, intent(in) :: body_indices(:)
        integer, intent(out) :: nargs
        character(len=:), allocatable, intent(out) :: arg_tokens
        character(len=:), allocatable, intent(out) :: rank_tokens
        character(len=:), allocatable, intent(out) :: extent_tokens
        logical, intent(in), optional :: allow_runtime_array
        integer :: i, value_kind, rank, extent
        character(len=:), allocatable :: token
        logical :: runtime_arrays_ok

        ok = .false.
        nargs = 0
        arg_tokens = ''
        rank_tokens = ''
        extent_tokens = ''
        runtime_arrays_ok = .false.
        if (present(allow_runtime_array)) runtime_arrays_ok = allow_runtime_array
        if (.not. allocated(param_indices)) then
            ok = .true.
            return
        end if
        nargs = size(param_indices)
        do i = 1, nargs
            rank = 0
            extent = 0
            value_kind = param_at_value_kind(arena, param_indices, &
                                             body_indices, i)
            if (value_kind == 0) then
                call param_at_array_shape(arena, context, param_indices, &
                                          body_indices, i, value_kind, rank, &
                                          extent, &
                                          allow_runtime_extent=runtime_arrays_ok)
                if (rank <= 0) return
            end if
            token = scalar_kind_token(value_kind)
            if (len_trim(token) == 0) return
            if (i > 1) then
                arg_tokens = arg_tokens//' '
                rank_tokens = rank_tokens//' '
                extent_tokens = extent_tokens//' '
            end if
            arg_tokens = arg_tokens//token
            rank_tokens = rank_tokens//integer_token(rank)
            extent_tokens = extent_tokens//integer_token(extent)
        end do
        ok = .true.
    end function params_all_supported

    module subroutine param_at_array_shape(arena, context, param_indices, &
                                    body_indices, pos, value_kind, rank, extent, &
                                    allow_runtime_extent)
        ! The element kind, rank, and total element count of an explicit-shape
        ! array dummy. rank stays 0 when the dummy is not an array this ffc can
        ! pass by base address (assumed shape, assumed rank, assumed size, or
        ! allocatable), which keeps such a procedure out of the artefact (#415).
        type(ast_arena_t), intent(in) :: arena
        type(lowering_context_t), intent(in) :: context
        integer, allocatable, intent(in) :: param_indices(:)
        integer, allocatable, intent(in) :: body_indices(:)
        integer, intent(in) :: pos
        integer, intent(out) :: value_kind
        integer, intent(out) :: rank
        integer, intent(out) :: extent
        logical, intent(in), optional :: allow_runtime_extent
        character(len=:), allocatable :: name, name_err, kind_err
        integer(c_int64_t) :: element_count
        logical :: known
        integer :: i
        logical :: runtime_extent_ok

        value_kind = 0
        rank = 0
        extent = 0
        runtime_extent_ok = .false.
        if (present(allow_runtime_extent)) runtime_extent_ok = allow_runtime_extent
        if (.not. allocated(param_indices)) return
        if (pos < 1 .or. pos > size(param_indices)) return
        call parameter_name(arena, param_indices(pos), name, name_err)
        if (len_trim(name_err) > 0) return
        if (.not. allocated(body_indices)) return
        do i = 1, size(body_indices)
            if (.not. node_exists(arena, body_indices(i))) cycle
            select type (decl => arena%entries(body_indices(i))%node)
            type is (declaration_node)
                if (.not. decl%is_array) cycle
                if (decl%is_allocatable .or. decl%is_pointer) return
                if (.not. declaration_declares_name(decl, &
                    trim(lowercase_text(name)))) cycle
                if (.not. allocated(decl%dimension_indices)) return
                call declaration_value_kind(decl, value_kind, kind_err)
                if (len_trim(kind_err) > 0) then
                    value_kind = 0
                    return
                end if
                call dummy_explicit_element_count(arena, context, body_indices, &
                                                  name, element_count, known, &
                                                  kind_err)
                if (len_trim(kind_err) > 0) return
                if (.not. known) then
                    if (runtime_extent_ok .and. &
                        .not. declaration_is_assumed_shape(decl, context) .and. &
                        .not. declaration_is_assumed_rank(decl, context)) then
                        rank = size(decl%dimension_indices)
                    end if
                    return
                end if
                if (element_count <= 0) return
                rank = size(decl%dimension_indices)
                extent = int(element_count)
                return
            end select
        end do
    end subroutine param_at_array_shape

    module subroutine build_fmod_variable(arena, node_index, var, error_msg)
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

    module function fmod_variable_kind_token(type_name) result(token)
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

    module subroutine build_fmod_parameter(arena, node_index, param, error_msg)
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

    module function fmod_kind_string(type_name) result(kind_text)
        ! Normalise a declaration type name to the .fmod kind token.
        ! This remains here because lowercase_text is private to the parent
        ! lowering module; moving it would require exporting that unrelated
        ! parser utility or introducing a second string-normalisation API.
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

    module subroutine build_fmod_derived_type(arena, context, node_index, dtype, &
                                       error_msg)
        ! Serialise the layout the lowering context already computed for this
        ! type, not a second description re-derived from the AST. The context
        ! record is the canonical derived layout every same-unit access uses, so
        ! a using unit that reconstructs it addresses components exactly as the
        ! defining unit did (#414).
        type(ast_arena_t), intent(in) :: arena
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: node_index
        type(fmod_derived_type_t), intent(out) :: dtype
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: type_name
        integer :: type_index

        call set_empty(error_msg)
        dtype%name = ''
        dtype%canonical_name = ''
        dtype%canonical_identity = ''
        dtype%parent_name = ''
        dtype%parent_identity = ''
        if (.not. is_derived_type_node(arena, node_index)) then
            error_msg = 'module derived-type export is not a derived type'
            return
        end if
        call get_derived_type_name(arena, node_index, type_name, error_msg)
        if (len_trim(error_msg) > 0) return
        dtype%name = trim(type_name)
        type_index = find_derived_type(context, trim(type_name))
        if (type_index <= 0) then
            ! The type never reached the lowering context, so this unit has no
            ! layout for it and must not publish a guess.
            allocate (dtype%components(0))
            return
        end if
        call build_fmod_derived_type_from_context(context, type_index, dtype, &
                                                  error_msg)
    end subroutine build_fmod_derived_type

    module subroutine build_fmod_derived_type_from_context(context, type_index, &
                                                           dtype, error_msg)
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: type_index
        type(fmod_derived_type_t), intent(out) :: dtype
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: k, nested, b, parent_index
        integer(c_int64_t) :: offset, slot_product, max_integer

        call set_empty(error_msg)
        dtype%name = ''
        dtype%canonical_name = ''
        dtype%canonical_identity = ''
        dtype%parent_name = ''
        dtype%parent_identity = ''
        if (type_index <= 0 .or. type_index > context%derived_type_count) then
            error_msg = 'invalid derived type context index for .fmod export'
            return
        end if
        dtype%name = trim(context%derived_types(type_index)%name)
        dtype%canonical_name = trim(context%derived_types(type_index)% &
                                    canonical_name)
        if (len_trim(dtype%canonical_name) == 0) &
            dtype%canonical_name = trim(dtype%name)
        dtype%canonical_identity = trim(context%derived_types(type_index)% &
                                        canonical_identity)
        if (len_trim(dtype%canonical_identity) == 0) &
            dtype%canonical_identity = trim(dtype%canonical_name)
        dtype%parent_name = trim(context%derived_types(type_index)%parent_name)
        parent_index = find_derived_type(context, trim(dtype%parent_name))
        if (parent_index > 0) dtype%parent_identity = trim( &
            context%derived_types(parent_index)%canonical_identity)
        allocate (dtype%components( &
            context%derived_types(type_index)%component_count))
        allocate (dtype%bindings( &
            context%derived_types(type_index)%binding_count))
        offset = 0_c_int64_t
        max_integer = int(huge(0), c_int64_t)
        do k = 1, context%derived_types(type_index)%component_count
            associate (comp => dtype%components(k))
                comp%name = trim(context%derived_types(type_index)% &
                                 component_names(k))
                comp%kind = fmod_component_kind_token(context, type_index, k)
                comp%type_name = ''
                comp%type_identity = ''
                nested = context%derived_types(type_index)% &
                         component_type_index(k)
                if (nested > 0) comp%type_name = &
                    trim(context%derived_types(nested)%name)
                if (nested > 0) comp%type_identity = trim( &
                    context%derived_types(nested)%canonical_identity)
                comp%slot_width = component_slot_width(context, type_index, k)
                comp%elem_count = context%derived_types(type_index)% &
                                  component_array_size(k)
                slot_product = int(comp%elem_count, c_int64_t) * &
                                int(comp%slot_width, c_int64_t)
                if (slot_product < 1_c_int64_t .or. &
                    slot_product > max_integer) then
                    error_msg = 'derived type component slot count exceeds '// &
                        'the .fmod integer range'
                    return
                end if
                if (offset > max_integer - slot_product) then
                    error_msg = 'derived type component slot offset exceeds '// &
                        'the .fmod integer range'
                    return
                end if
                comp%slot_count = int(slot_product)
                comp%slot_offset = int(offset)
                comp%char_length = context%derived_types(type_index)% &
                                   component_char_length(k)
                comp%dim1 = context%derived_types(type_index)%component_dim1(k)
                comp%alloc_rank = context%derived_types(type_index)% &
                                  component_alloc_rank(k)
                comp%is_allocatable = context%derived_types(type_index)% &
                                      component_is_allocatable(k)
                comp%is_pointer = context%derived_types(type_index)% &
                                  component_is_pointer(k)
                comp%is_alloc_array = context%derived_types(type_index)% &
                                      component_is_alloc_array(k)
                offset = offset + slot_product
            end associate
        end do
        do b = 1, context%derived_types(type_index)%binding_count
            dtype%bindings(b)%method_name = trim(context%derived_types(type_index)% &
                binding_method_names(b))
            dtype%bindings(b)%target_name = trim(context%derived_types(type_index)% &
                binding_target_names(b))
            dtype%bindings(b)%specific_names = trim(context%derived_types(type_index)% &
                binding_specific_names(b))
            dtype%bindings(b)%pass_name = trim(context%derived_types(type_index)% &
                binding_pass_names(b))
            dtype%bindings(b)%pass_arg = context%derived_types(type_index)% &
                binding_pass_args(b)
        end do
    end subroutine build_fmod_derived_type_from_context

    module function fmod_component_kind_token(context, type_index, comp_index) &
            result(token)
        ! The canonical .fmod token for a component's value kind. A nested
        ! derived component reports 'derived'; its type travels in type_name.
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: type_index
        integer, intent(in) :: comp_index
        character(len=:), allocatable :: token
        integer :: value_kind

        value_kind = context%derived_types(type_index)% &
                     component_value_kind(comp_index)
        select case (value_kind)
        case (VALUE_DERIVED)
            token = 'derived'
        case (VALUE_CHARACTER)
            token = 'character'
        case (VALUE_C_PTR)
            token = 'c_ptr'
        case default
            token = scalar_kind_token(value_kind)
            if (len_trim(token) == 0) token = 'unsupported'
        end select
    end function fmod_component_kind_token

    module function fmod_component_value_kind(token) result(value_kind)
        integer :: value_kind
        ! Inverse of fmod_component_kind_token; 0 when the token names no kind
        ! this ffc can lay out.
        character(len=*), intent(in) :: token

        select case (trim(token))
        case ('derived')
            value_kind = VALUE_DERIVED
        case ('character')
            value_kind = VALUE_CHARACTER
        case ('c_ptr')
            value_kind = VALUE_C_PTR
        case default
            value_kind = value_kind_of_token(token)
        end select
    end function fmod_component_value_kind
end submodule session_program_lowering_fmod_exporter
