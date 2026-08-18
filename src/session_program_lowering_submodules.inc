    logical function is_submodule_unit(node_type)
        ! Whether a stored node-type tag names a submodule unit.
        character(len=*), intent(in) :: node_type

        is_submodule_unit = node_type == 'submodule_node' .or. &
                            node_type == 'submodule'
    end function is_submodule_unit

    function submodule_parent_module(parent_identifier) result(parent_module)
        ! The ancestor module a submodule extends. FortFront stores the parent
        ! chain as 'mod' or 'mod:sub:...'; the module that owns the interface
        ! declarations (and the gfortran mangling prefix) is the first element.
        character(len=*), intent(in) :: parent_identifier
        character(len=:), allocatable :: parent_module
        integer :: colon

        colon = index(parent_identifier, ':')
        if (colon > 0) then
            parent_module = trim(parent_identifier(1:colon - 1))
        else
            parent_module = trim(parent_identifier)
        end if
    end function submodule_parent_module

    subroutine get_submodule_body_info(arena, node_index, name, parent_module, &
                                       declaration_indices, procedure_indices, &
                                       error_msg)
        ! Submodule counterpart of get_module_body_info: expose the submodule's
        ! name, its ancestor module, and its declaration/procedure arena indices.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        character(len=:), allocatable, intent(out) :: name, parent_module
        integer, allocatable, intent(out) :: declaration_indices(:)
        integer, allocatable, intent(out) :: procedure_indices(:)
        character(len=:), allocatable, intent(out) :: error_msg

        call set_empty(name)
        call set_empty(parent_module)
        if (allocated(declaration_indices)) deallocate(declaration_indices)
        if (allocated(procedure_indices)) deallocate(procedure_indices)
        allocate(declaration_indices(0))
        allocate(procedure_indices(0))
        if (.not. node_exists(arena, node_index)) then
            error_msg = 'submodule body index does not reference an AST node'
            return
        end if
        select type (node => arena%entries(node_index)%node)
        type is (submodule_node)
            if (allocated(node%name)) name = node%name
            if (allocated(node%parent_identifier)) then
                parent_module = submodule_parent_module(node%parent_identifier)
            end if
            if (allocated(node%declaration_indices)) then
                declaration_indices = node%declaration_indices
            end if
            if (allocated(node%procedure_indices)) then
                procedure_indices = node%procedure_indices
            end if
            call set_empty(error_msg)
        class default
            error_msg = 'AST node is not a submodule; expected submodule_node'
        end select
    end subroutine get_submodule_body_info

    logical function module_node_defined(arena, name)
        ! Whether a module_node named 'name' exists in the compilation set, so a
        ! submodule's parent reference can be validated before its procedures are
        ! registered against that module.
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: name
        integer :: n
        character(len=:), allocatable :: node_type, mod_name, ff_error
        integer, allocatable :: decl_indices(:), proc_indices(:)

        module_node_defined = .false.
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            node_type = get_node_type_at(arena, n)
            if (node_type /= 'module_node' .and. node_type /= 'module') cycle
            call get_module_body_info(arena, n, mod_name, decl_indices, &
                                      proc_indices, ff_error)
            if (len_trim(ff_error) > 0) cycle
            if (same_name(mod_name, name)) then
                module_node_defined = .true.
                return
            end if
        end do
    end function module_node_defined

    subroutine register_one_submodule_procedures(arena, submodule_index, context, &
                                                 error_msg)
        ! Register a submodule's contained procedures by Fortran name so a call
        ! from the program (or a sibling procedure) resolves to the right value
        ! kind. A separate `module procedure P` body inherits P's kind and
        ! signature from the parent module's interface. Mangling to the parent
        ! module happens at the call site.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: submodule_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: name, parent_module, ff_error
        integer, allocatable :: decl_indices(:), proc_indices(:)
        integer :: p

        call get_submodule_body_info(arena, submodule_index, name, parent_module, &
                                     decl_indices, proc_indices, ff_error)
        if (len_trim(ff_error) > 0) then
            error_msg = ff_error
            return
        end if
        do p = 1, size(proc_indices)
            call register_one_submodule_procedure(arena, proc_indices(p), &
                                                  parent_module, context, error_msg)
            if (len_trim(error_msg) > 0) return
        end do
        call register_submodule_generic_interfaces(arena, decl_indices, context, &
                                                    error_msg)
        if (len_trim(error_msg) > 0) return
        call refresh_parent_module_generics(arena, parent_module, context, &
                                            error_msg)
    end subroutine register_one_submodule_procedures

    subroutine refresh_parent_module_generics(arena, parent_module, context, &
                                              error_msg)
        ! A parent module's named generic can list a module procedure whose
        ! body is supplied later by a submodule. Refresh those specific
        ! signatures after the submodule procedure table is populated so the
        ! generic return kind and argument count are not left at their
        ! interface-only defaults.
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: parent_module
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: node_type, module_name, ff_error
        integer, allocatable :: declaration_indices(:), procedure_indices(:)
        integer, allocatable :: no_procedures(:)
        integer :: n

        call set_empty(error_msg)
        allocate(no_procedures(0))
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            node_type = get_node_type_at(arena, n)
            if (node_type /= 'module_node' .and. node_type /= 'module') cycle
            call get_module_body_info(arena, n, module_name, declaration_indices, &
                                      procedure_indices, ff_error)
            if (len_trim(ff_error) > 0) cycle
            if (.not. same_name(module_name, parent_module)) cycle
            call register_procedures_from_indices(arena, declaration_indices, &
                                                  no_procedures, context, error_msg)
            return
        end do
    end subroutine refresh_parent_module_generics

    subroutine register_one_submodule_procedure(arena, proc_index, parent_module, &
                                                context, error_msg)
        ! Register a single submodule procedure. A separate `module procedure`
        ! form is registered with the parent interface's kind and signature; any
        ! other form (a restated module procedure or a submodule-local helper)
        ! registers directly from its own node.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: proc_index
        character(len=*), intent(in) :: parent_module
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: node_type
        logical :: found, is_function
        type(function_def_node) :: merged_fn
        type(subroutine_def_node) :: merged_sb
        integer, allocatable :: body_source(:)
        type(function_def_node), pointer :: fn_node
        type(subroutine_def_node), pointer :: sb_node

        call set_empty(error_msg)
        if (.not. node_exists(arena, proc_index)) return
        call build_separate_module_procedure(arena, proc_index, parent_module, &
                                             found, is_function, merged_fn, &
                                             merged_sb, body_source, error_msg)
        if (len_trim(error_msg) > 0) return
        if (found) then
            if (is_function) then
                if (contained_function_kind(context, merged_fn%name) /= 0) return
                call register_internal_function_name(merged_fn, context, &
                                                     error_msg, proc_index)
            else
                if (contained_function_kind(context, merged_sb%name) /= 0) return
                call register_internal_subroutine_name(merged_sb, context, &
                                                       error_msg, proc_index)
            end if
            if (len_trim(error_msg) > 0) return
            call register_nested_internal_names(arena, body_source, context, &
                                                error_msg)
            return
        end if
        node_type = get_node_type_at(arena, proc_index)
        if (node_type == 'function_def_node' .or. node_type == 'function_def') then
            fn_node => get_node_as_function_def(arena, proc_index)
            if (.not. associated(fn_node)) return
            if (.not. allocated(fn_node%name)) return
            if (contained_function_kind(context, fn_node%name) /= 0) return
            call register_internal_function_name(fn_node, context, error_msg, &
                                                 proc_index)
            if (len_trim(error_msg) > 0) return
            call register_nested_internal_names(arena, fn_node%body_indices, &
                                                context, error_msg)
        else if (node_type == 'subroutine_def_node' .or. &
                 node_type == 'subroutine_def') then
            sb_node => get_node_as_subroutine_def(arena, proc_index)
            if (.not. associated(sb_node)) return
            if (.not. allocated(sb_node%name)) return
            if (contained_function_kind(context, sb_node%name) /= 0) return
            call register_internal_subroutine_name(sb_node, context, error_msg, &
                                                   proc_index)
            if (len_trim(error_msg) > 0) return
            call register_nested_internal_names(arena, sb_node%body_indices, &
                                                context, error_msg)
        end if
    end subroutine register_one_submodule_procedure

    subroutine register_submodule_generic_interfaces(arena, decl_indices, context, &
                                                     error_msg)
        ! Register named/operator generic interfaces declared in a submodule's
        ! specification part, mirroring the module path in
        ! register_procedures_from_indices.
        type(ast_arena_t), intent(in) :: arena
        integer, allocatable, intent(in) :: decl_indices(:)
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer, allocatable :: no_procs(:)

        allocate(no_procs(0))
        call register_procedures_from_indices(arena, decl_indices, no_procs, &
                                              context, error_msg)
    end subroutine register_submodule_generic_interfaces

    subroutine lower_one_submodule_procedures(arena, submodule_index, context, &
                                              error_msg)
        ! Emit a submodule's module procedures under their ancestor module's
        ! mangled name (__<parent>_MOD_<name>), matching the interface bodies the
        ! parent module declared. A parent module absent from the compilation set
        ! is a hard error: single-file submodules only (#292).
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: submodule_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: name, parent_module, ff_error
        integer, allocatable :: decl_indices(:), proc_indices(:)
        integer :: p

        call get_submodule_body_info(arena, submodule_index, name, parent_module, &
                                     decl_indices, proc_indices, ff_error)
        if (len_trim(ff_error) > 0) then
            error_msg = ff_error
            return
        end if
        call bind_submodule_parent(arena, submodule_index, parent_module, &
                                   proc_indices, context, error_msg)
        if (len_trim(error_msg) > 0) return
        do p = 1, size(proc_indices)
            call lower_one_submodule_procedure(arena, proc_indices(p), &
                                               parent_module, context, error_msg)
            if (len_trim(error_msg) > 0) return
        end do
    end subroutine lower_one_submodule_procedures

    subroutine bind_submodule_parent(arena, submodule_index, parent_module, &
                                     proc_indices, context, error_msg)
        ! Bind a submodule to the parent module it extends. The parent is either
        ! compiled in this unit or described by its .fmod artefact on the include
        ! path; a submodule compiled on its own reads its parent's interfaces
        ! from that artefact and is checked against them, so an implementation
        ! that does not match the interface it claims is diagnosed here rather
        ! than emitted under a symbol callers would then miscall (#297).
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: submodule_index
        character(len=*), intent(in) :: parent_module
        integer, allocatable, intent(in) :: proc_indices(:)
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(module_info_t) :: parent_info
        character(len=:), allocatable :: path
        integer :: p

        call set_empty(error_msg)
        if (module_node_defined(arena, parent_module)) return
        path = find_fmod_on_path(context, trim(parent_module))
        if (len_trim(path) == 0) then
            call unsupported_feature_error('submodule', &
                get_node_line(arena, submodule_index), &
                get_node_column(arena, submodule_index), &
                'submodule parent module not found: '//trim(parent_module)// &
                ' is not compiled in this unit and no '//trim(parent_module)// &
                '.fmod was found on the include path', error_msg)
            return
        end if
        call read_fmod(path, parent_info, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. allocated(proc_indices)) return
        do p = 1, size(proc_indices)
            call check_submodule_procedure_against_parent(arena, &
                proc_indices(p), parent_module, parent_info, context, error_msg)
            if (len_trim(error_msg) > 0) return
        end do
    end subroutine bind_submodule_parent

    subroutine check_submodule_procedure_against_parent(arena, proc_index, &
                                                        parent_module, &
                                                        parent_info, context, &
                                                        error_msg)
        ! Check one implemented module procedure against the interface the
        ! parent artefact declares for it: the parent must declare it, and the
        ! implementation must restate the same dummy count and dummy kinds.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: proc_index
        character(len=*), intent(in) :: parent_module
        type(module_info_t), intent(in) :: parent_info
        type(lowering_context_t), intent(in) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: proc_name, kind_text, arg_tokens
        character(len=:), allocatable :: rank_tokens, extent_tokens
        integer :: idx, nargs

        call set_empty(error_msg)
        proc_name = procedure_fortran_name(arena, proc_index)
        if (len_trim(proc_name) == 0) return
        idx = fmod_procedure_index(parent_info, proc_name)
        if (idx <= 0) then
            call unsupported_feature_error('submodule procedure', &
                get_node_line(arena, proc_index), &
                get_node_column(arena, proc_index), &
                'module procedure '//trim(proc_name)//' is not declared by '// &
                'module '//trim(parent_module), error_msg)
            return
        end if
        if (.not. submodule_body_restates_signature(arena, proc_index)) then
            ! A `module procedure P` body inherits its dummy declarations from
            ! the parent's interface, which this unit does not have as source.
            call unsupported_feature_error('submodule procedure', &
                get_node_line(arena, proc_index), &
                get_node_column(arena, proc_index), &
                'module procedure '//trim(proc_name)//' compiled apart from '// &
                'module '//trim(parent_module)//' must restate its interface', &
                error_msg)
            return
        end if
        call fmod_procedure_signature(arena, context, proc_index, &
                                      kind_text=kind_text, nargs=nargs, &
                                      arg_tokens=arg_tokens, &
                                      rank_tokens=rank_tokens, &
                                      extent_tokens=extent_tokens)
        if (len_trim(kind_text) == 0) return
        if (nargs /= parent_info%procedures(idx)%nargs) then
            call submodule_signature_mismatch(arena, proc_index, proc_name, &
                parent_module, 'dummy argument count', error_msg)
            return
        end if
        if (trim(arg_tokens) /= trim(field_text(parent_info%procedures(idx)% &
                                                arg_kinds))) then
            call submodule_signature_mismatch(arena, proc_index, proc_name, &
                parent_module, 'dummy argument kinds', error_msg)
            return
        end if
        if (trim(kind_text) /= trim(field_text(parent_info%procedures(idx)%kind))) &
            call submodule_signature_mismatch(arena, proc_index, proc_name, &
                parent_module, 'result kind', error_msg)
    end subroutine check_submodule_procedure_against_parent

    subroutine submodule_signature_mismatch(arena, proc_index, proc_name, &
                                            parent_module, what, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: proc_index
        character(len=*), intent(in) :: proc_name
        character(len=*), intent(in) :: parent_module
        character(len=*), intent(in) :: what
        character(len=:), allocatable, intent(out) :: error_msg

        call unsupported_feature_error('submodule procedure', &
            get_node_line(arena, proc_index), &
            get_node_column(arena, proc_index), &
            'module procedure '//trim(proc_name)//' does not match the '// &
            what//' its interface in module '//trim(parent_module)//' declares', &
            error_msg)
    end subroutine submodule_signature_mismatch

    logical function submodule_body_restates_signature(arena, proc_index) &
            result(restates)
        ! Whether a submodule procedure restates its dummy list, as
        ! `module function f(x) result(y)` does and `module procedure f` does
        ! not.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: proc_index
        type(function_def_node), pointer :: fn_node
        type(subroutine_def_node), pointer :: sb_node

        restates = .false.
        fn_node => get_node_as_function_def(arena, proc_index)
        if (associated(fn_node)) then
            restates = allocated(fn_node%param_indices)
            if (restates) restates = size(fn_node%param_indices) > 0
            return
        end if
        sb_node => get_node_as_subroutine_def(arena, proc_index)
        if (.not. associated(sb_node)) return
        restates = allocated(sb_node%param_indices)
        if (restates) restates = size(sb_node%param_indices) > 0
    end function submodule_body_restates_signature

    integer function fmod_procedure_index(info, name) result(idx)
        ! Index of the module procedure the artefact records under name, or 0.
        type(module_info_t), intent(in) :: info
        character(len=*), intent(in) :: name
        integer :: i

        idx = 0
        if (.not. allocated(info%procedures)) return
        do i = 1, size(info%procedures)
            if (.not. allocated(info%procedures(i)%name)) cycle
            if (same_name(info%procedures(i)%name, name)) then
                idx = i
                return
            end if
        end do
    end function fmod_procedure_index

    function field_text(text) result(out)
        character(len=:), allocatable, intent(in) :: text
        character(len=:), allocatable :: out

        if (allocated(text)) then
            out = text
        else
            out = ''
        end if
    end function field_text

    subroutine lower_one_submodule_procedure(arena, proc_index, parent_module, &
                                             context, error_msg)
        ! Lower one submodule procedure under the parent module's mangling. A
        ! separate `module procedure` body is spliced onto its parent interface
        ! signature; any other form lowers directly from its own node.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: proc_index
        character(len=*), intent(in) :: parent_module
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        logical :: found, is_function
        type(function_def_node) :: merged_fn
        type(subroutine_def_node) :: merged_sb
        integer, allocatable :: body_source(:)

        call set_empty(error_msg)
        if (.not. node_exists(arena, proc_index)) return
        call build_separate_module_procedure(arena, proc_index, parent_module, &
                                             found, is_function, merged_fn, &
                                             merged_sb, body_source, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. found) then
            call lower_one_module_procedure(arena, proc_index, trim(parent_module), &
                                            context, error_msg)
            return
        end if
        if (is_function) then
            call lower_scalar_function(arena, merged_fn, context, error_msg, &
                module_procedure_mangled(parent_module, merged_fn%name))
        else
            call lower_void_subroutine(arena, merged_sb, context, error_msg, &
                module_procedure_mangled(parent_module, merged_sb%name))
        end if
        if (len_trim(error_msg) > 0) return
        call lower_nested_internal_procedures(arena, body_source, context, &
                                              error_msg)
    end subroutine lower_one_submodule_procedure

    subroutine build_separate_module_procedure(arena, proc_index, parent_module, &
                                               found, is_function, merged_fn, &
                                               merged_sb, body_source, error_msg)
        ! Detect a separate `module procedure P` body (a subroutine_def_node with
        ! no restated parameters) and splice the parent module's interface
        ! signature for P onto it: dummy declarations and result come from the
        ! interface, executable statements from the submodule body. found=.false.
        ! for any other procedure form.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: proc_index
        character(len=*), intent(in) :: parent_module
        logical, intent(out) :: found
        logical, intent(out) :: is_function
        type(function_def_node), intent(out) :: merged_fn
        type(subroutine_def_node), intent(out) :: merged_sb
        integer, allocatable, intent(out) :: body_source(:)
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: node_type
        type(subroutine_def_node), pointer :: body_node
        type(function_def_node), pointer :: iface_fn
        type(subroutine_def_node), pointer :: iface_sb
        integer :: iface_index

        found = .false.
        is_function = .false.
        allocate(body_source(0))
        call set_empty(error_msg)
        node_type = get_node_type_at(arena, proc_index)
        if (node_type /= 'subroutine_def_node' .and. &
            node_type /= 'subroutine_def') return
        body_node => get_node_as_subroutine_def(arena, proc_index)
        if (.not. associated(body_node)) return
        if (.not. allocated(body_node%name)) return
        if (allocated(body_node%param_indices)) then
            if (size(body_node%param_indices) > 0) return
        end if
        iface_index = find_module_interface_procedure(arena, parent_module, &
                                                      body_node%name)
        if (iface_index <= 0) return
        if (allocated(body_node%body_indices)) body_source = body_node%body_indices
        node_type = get_node_type_at(arena, iface_index)
        if (node_type == 'function_def_node' .or. node_type == 'function_def') then
            iface_fn => get_node_as_function_def(arena, iface_index)
            if (.not. associated(iface_fn)) return
            merged_fn = iface_fn
            call concat_indices(iface_fn%body_indices, body_source, &
                                merged_fn%body_indices)
            found = .true.
            is_function = .true.
        else if (node_type == 'subroutine_def_node' .or. &
                 node_type == 'subroutine_def') then
            iface_sb => get_node_as_subroutine_def(arena, iface_index)
            if (.not. associated(iface_sb)) return
            merged_sb = iface_sb
            call concat_indices(iface_sb%body_indices, body_source, &
                                merged_sb%body_indices)
            found = .true.
            is_function = .false.
        end if
    end subroutine build_separate_module_procedure

    subroutine concat_indices(head, tail, joined)
        ! Concatenate two optionally-unallocated arena index lists.
        integer, allocatable, intent(in) :: head(:)
        integer, allocatable, intent(in) :: tail(:)
        integer, allocatable, intent(out) :: joined(:)
        integer :: head_size, tail_size

        head_size = 0
        tail_size = 0
        if (allocated(head)) head_size = size(head)
        if (allocated(tail)) tail_size = size(tail)
        allocate (joined(head_size + tail_size))
        if (head_size > 0) joined(1:head_size) = head
        if (tail_size > 0) joined(head_size + 1:head_size + tail_size) = tail
    end subroutine concat_indices

    integer function find_module_interface_procedure(arena, parent_module, name) &
            result(iface_index)
        ! The arena index of the interface body (module function/subroutine) that
        ! declares module procedure `name` in module `parent_module`, or 0.
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: parent_module
        character(len=*), intent(in) :: name
        character(len=:), allocatable :: mod_lowered, mod_name, ff_error
        integer, allocatable :: decl_indices(:), proc_indices(:)
        integer :: n, d, p

        iface_index = 0
        mod_lowered = lowercase_text(parent_module)
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (node => arena%entries(n)%node)
            type is (module_node)
                if (.not. allocated(node%name)) cycle
                if (lowercase_text(node%name) /= mod_lowered) cycle
                call get_module_body_info(arena, n, mod_name, decl_indices, &
                                          proc_indices, ff_error)
                if (len_trim(ff_error) > 0) return
                if (.not. allocated(decl_indices)) return
                do d = 1, size(decl_indices)
                    if (.not. node_exists(arena, decl_indices(d))) cycle
                    select type (iblock => arena%entries(decl_indices(d))%node)
                    type is (interface_block_node)
                        if (iblock%is_abstract) cycle
                        if (.not. allocated(iblock%procedure_indices)) cycle
                        do p = 1, size(iblock%procedure_indices)
                            if (procedure_node_named(arena, &
                                    iblock%procedure_indices(p), name)) then
                                iface_index = iblock%procedure_indices(p)
                                return
                            end if
                        end do
                    end select
                end do
                return
            end select
        end do
    end function find_module_interface_procedure
