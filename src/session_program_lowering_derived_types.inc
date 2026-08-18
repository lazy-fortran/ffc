    subroutine collect_program_parameters(arena, root_index, context, error_msg)
        ! Pre-collect program-scope integer scalar parameters so derived-type
        ! array component bounds (integer :: a(n)) can resolve them. Parameters
        ! that are not compile-time integer scalars are skipped here and handled
        ! (or rejected) when the body declaration is lowered.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: root_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        call set_empty(error_msg)
        select type (program => arena%entries(root_index)%node)
        type is (program_node)
            if (.not. allocated(program%body_indices)) return
            call collect_parameter_declarations(arena, program%body_indices, &
                                                context, error_msg)
        type is (module_node)
            if (.not. allocated(program%declaration_indices)) return
            call collect_parameter_declarations(arena, &
                program%declaration_indices, context, error_msg)
        end select
        call collect_old_style_parameters(arena, context, error_msg)
    end subroutine collect_program_parameters

    subroutine collect_parameter_declarations(arena, indices, context, error_msg)
        ! Bind scalar integer parameters before derived-type collection. Module
        ! kind parameters are needed while laying out components in the same
        ! module; treating them as a later export leaves an incomplete .fmod.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: indices(:)
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i, value_kind
        character(len=:), allocatable :: kind_err, decl_err

        call set_empty(error_msg)
        do i = 1, size(indices)
            if (.not. node_exists(arena, indices(i))) cycle
            select type (node => arena%entries(indices(i))%node)
            type is (declaration_node)
                if (.not. node%is_parameter) cycle
                if (node%is_array .or. node%is_multi_declaration) cycle
                if (.not. allocated(node%var_name)) cycle
                call declaration_value_kind(node, value_kind, kind_err, context)
                if (len_trim(kind_err) > 0) cycle
                if (value_kind /= VALUE_I32) cycle
                if (.not. node%has_initializer) cycle
                ! Best effort: the declaration pass reports unsupported forms.
                call lower_constant_declaration(node, context, VALUE_I32, &
                                                decl_err)
                if (len_trim(decl_err) == 0) then
                    call register_declaration_bindings(context, node, indices(i))
                end if
            end select
        end do
    end subroutine collect_parameter_declarations

    subroutine collect_old_style_parameters(arena, context, error_msg)
        ! An old-style PARAMETER statement whose name has no type declaration
        ! (parameter (ialen = 42)) reaches ffc as a free-standing declaration
        ! carrying only the name and the initializer: FortFront leaves it out
        ! of the enclosing scope body because implicit typing supplies its
        ! type. Bind it here so later specification expressions (array bounds)
        ! and references fold against the constant. Names that already carry a
        ! type declaration are merged by FortFront into the body declaration
        ! and stay on the ordinary path.
        type(ast_arena_t), intent(in) :: arena
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i, value_kind

        call set_empty(error_msg)
        do i = 1, arena%size
            if (.not. node_exists(arena, i)) cycle
            if (arena%entries(i)%parent_index /= 0) cycle
            select type (node => arena%entries(i)%node)
            type is (declaration_node)
                if (.not. declaration_is_old_style_parameter(node)) cycle
                value_kind = implicit_parameter_value_kind(node%var_name)
                call lower_constant_declaration(node, context, value_kind, &
                                                error_msg)
                if (len_trim(error_msg) > 0) return
                call register_declaration_bindings(context, node, i)
            end select
        end do
    end subroutine collect_old_style_parameters

    logical function declaration_is_old_style_parameter(node) result(is_old)
        ! True for an untyped PARAMETER-statement entity: a named constant
        ! with an initializer and no declared type.
        type(declaration_node), intent(in) :: node

        is_old = .false.
        if (.not. node%is_parameter) return
        if (node%is_array .or. node%is_multi_declaration) return
        if (.not. node%has_initializer) return
        if (node%initializer_index <= 0) return
        if (.not. allocated(node%var_name)) return
        if (len_trim(node%var_name) == 0) return
        if (allocated(node%type_name)) then
            if (len_trim(node%type_name) > 0) return
        end if
        is_old = .true.
    end function declaration_is_old_style_parameter

    integer function implicit_parameter_value_kind(name) result(value_kind)
        ! Default implicit typing: names starting with I-N are integer, all
        ! others are default real. IMPLICIT statements are out of scope here.
        character(len=*), intent(in) :: name
        character(len=1) :: first

        first = name(1:1)
        if (first >= 'A' .and. first <= 'Z') first = achar(iachar(first) + 32)
        if (first >= 'i' .and. first <= 'n') then
            value_kind = VALUE_I32
        else
            value_kind = VALUE_F32
        end if
    end function implicit_parameter_value_kind

    subroutine collect_derived_type_definitions(arena, root_index, context, &
                                                error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: root_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i

        do i = 1, size(context%derived_types)
            context%derived_types(i)%name = ''
            context%derived_types(i)%layout_registered = .false.
            context%derived_types(i)%component_count = 0
            context%derived_types(i)%duplicate_binding_name = ''
            if (allocated(context%derived_types(i)%component_names)) then
                deallocate(context%derived_types(i)%component_names)
            end if
            if (allocated(context%derived_types(i)%component_value_kind)) then
                deallocate(context%derived_types(i)%component_value_kind)
            end if
            if (allocated(context%derived_types(i)%component_type_index)) then
                deallocate(context%derived_types(i)%component_type_index)
            end if
        end do
        context%derived_type_count = 0
        context%derived_type_alias_count = 0
        if (allocated(context%derived_type_alias_names)) &
            deallocate(context%derived_type_alias_names)
        if (allocated(context%derived_type_alias_indices)) &
            deallocate(context%derived_type_alias_indices)
        call set_empty(error_msg)

        select type (program => arena%entries(root_index)%node)
        type is (program_node)
            if (.not. allocated(program%body_indices)) return
            call reserve_derived_type_names(arena, program%body_indices, context)
            do i = 1, size(program%body_indices)
                select type (node => arena%entries(program%body_indices(i))%node)
                type is (derived_type_node)
                    call lower_derived_type_definition(node, context, error_msg)
                    if (len_trim(error_msg) > 0) return
                    call emit_type_info_constant(context, &
                                                 context%derived_type_count, &
                                                 error_msg)
                    if (len_trim(error_msg) > 0) return
                type is (module_node)
                    call collect_module_derived_types(arena, node, context, &
                                                      error_msg)
                    if (len_trim(error_msg) > 0) return
                end select
            end do
        type is (multi_unit_container_node)
            ! Multi-unit source (#2830 container): register derived types from
            ! every contained module and any top-level type, as for a program.
            if (.not. allocated(program%body_indices)) return
            call reserve_derived_type_names(arena, program%body_indices, context)
            do i = 1, size(program%body_indices)
                select type (node => arena%entries(program%body_indices(i))%node)
                type is (derived_type_node)
                    call lower_derived_type_definition(node, context, error_msg)
                    if (len_trim(error_msg) > 0) return
                    call emit_type_info_constant(context, &
                                                 context%derived_type_count, &
                                                 error_msg)
                    if (len_trim(error_msg) > 0) return
                type is (module_node)
                    call collect_module_derived_types(arena, node, context, &
                                                      error_msg)
                    if (len_trim(error_msg) > 0) return
                end select
            end do
        type is (module_node)
            ! Bare module compilation unit: register its types so module
            ! procedures can name them as dummy/local/component types.
            call collect_module_derived_types(arena, program, context, error_msg)
            if (len_trim(error_msg) > 0) return
        end select
    end subroutine collect_derived_type_definitions

    subroutine reserve_derived_type_names(arena, indices, context)
        ! Pointer components may name a derived type defined later in the same
        ! scoping unit. Reserve every local type name before collecting any
        ! layout so those forward references resolve to a stable type index.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: indices(:)
        type(lowering_context_t), intent(inout) :: context
        integer :: i, type_index

        do i = 1, size(indices)
            if (.not. node_exists(arena, indices(i))) cycle
            select type (node => arena%entries(indices(i))%node)
            type is (derived_type_node)
                if (.not. allocated(node%name)) cycle
                if (find_derived_type(context, trim(node%name)) > 0) cycle
                call grow_derived_types(context)
                type_index = context%derived_type_count + 1
                context%derived_types(type_index)%name = trim(node%name)
                context%derived_types(type_index)%layout_registered = .false.
                context%derived_types(type_index)%component_count = 0
                context%derived_type_count = type_index
            end select
        end do
    end subroutine reserve_derived_type_names

    subroutine collect_module_derived_types(arena, mod_node, context, error_msg)
        ! Register module-defined derived types into the derived-type table so a
        ! module procedure body can resolve type(t) dummies, locals, and
        ! components, just as program-scope types resolve. Mirrors the per-type
        ! work of collect_derived_type_definitions (definition + type_info).
        type(ast_arena_t), intent(in) :: arena
        type(module_node), intent(in) :: mod_node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: j

        call set_empty(error_msg)
        if (.not. allocated(mod_node%declaration_indices)) return
        ! Imported derived types must be registered before this module's layouts:
        ! a component may name a type from a sibling module's .fmod.
        do j = 1, size(mod_node%declaration_indices)
            if (.not. node_exists(arena, mod_node%declaration_indices(j))) cycle
            select type (use_node => &
                arena%entries(mod_node%declaration_indices(j))%node)
            type is (use_statement_node)
                if (.not. module_use_is_resolvable(arena, context, use_node)) cycle
                call import_module_derived_types(arena, use_node, context, error_msg)
                if (len_trim(error_msg) > 0) return
            end select
        end do
        do j = 1, size(mod_node%declaration_indices)
            if (.not. node_exists(arena, mod_node%declaration_indices(j))) cycle
            select type (decl => &
                arena%entries(mod_node%declaration_indices(j))%node)
            type is (derived_type_node)
                if (.not. allocated(decl%name)) cycle
                if (find_derived_type(context, trim(decl%name)) > 0) cycle
                ! A type ffc cannot lower (e.g. allocatable/pointer derived
                ! components) is skipped, not fatal: a module that only defines
                ! it stays compilable, and a procedure that actually uses it
                ! fails later at its own declaration with a clean diagnostic.
                call lower_derived_type_definition(decl, context, error_msg)
                if (len_trim(error_msg) > 0) then
                    call set_empty(error_msg)
                    cycle
                end if
                call emit_type_info_constant(context, &
                                             context%derived_type_count, &
                                             error_msg)
                if (len_trim(error_msg) > 0) return
            end select
        end do
    end subroutine collect_module_derived_types

    subroutine emit_type_info_constant(context, type_index, error_msg)
        ! Emit the per-derived-type compile-time type_info_t constant (#140).
        ! The id is the dense per-unit type index; size is the storage in bytes
        ! (each component occupies a 4-byte i32 slot in the MVP layout).
        use, intrinsic :: iso_c_binding, only: c_int64_t
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: type_index
        character(len=:), allocatable, intent(out) :: error_msg

        if (type_index <= 0) then
            call set_empty(error_msg)
            return
        end if
        call create_type_info_global(context%session, &
            '__ffc_type_info_'//trim(context%derived_types(type_index)%name), &
            int(type_index, c_int64_t), &
            int(total_component_slots(context, type_index) * 4, c_int64_t), &
            error_msg)
    end subroutine emit_type_info_constant

    subroutine collect_module_exports(arena, root_index, context, &
                                      error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: root_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i
        integer :: j
        integer :: export_idx
        integer :: type_idx
        integer :: comp_idx

        do i = 1, size(context%module_exports)
            context%module_exports(i)%module_name = ''
            if (allocated(context%module_exports(i)%derived_type_indices)) then
                deallocate(context%module_exports(i)%derived_type_indices)
            end if
            context%module_exports(i)%derived_type_count = 0
            if (allocated(context%module_exports(i)%parameter_indices)) then
                deallocate(context%module_exports(i)%parameter_indices)
            end if
            context%module_exports(i)%parameter_count = 0
            if (allocated(context%module_exports(i)%variable_indices)) then
                deallocate(context%module_exports(i)%variable_indices)
            end if
            context%module_exports(i)%variable_count = 0
            if (allocated(context%module_exports(i)%enum_indices)) then
                deallocate(context%module_exports(i)%enum_indices)
            end if
            context%module_exports(i)%enum_count = 0
        end do
        context%module_export_count = 0
        call set_empty(error_msg)

        select type (program => arena%entries(root_index)%node)
        type is (program_node)
            if (.not. allocated(program%body_indices)) return
            do i = 1, size(program%body_indices)
                select type (mod_node => arena%entries(program%body_indices(i))%node)
                type is (module_node)
                    call collect_one_module_exports(arena, mod_node, context)
                end select
            end do
        type is (multi_unit_container_node)
            ! Several units from one source (a module plus its program). Register
            ! the exports of every contained module so the program's USE-renamed
            ! references resolve (#2830 container; was the __MULTI_UNIT__ program).
            if (.not. allocated(program%body_indices)) return
            do i = 1, size(program%body_indices)
                select type (mod_node => arena%entries(program%body_indices(i))%node)
                type is (module_node)
                    call collect_one_module_exports(arena, mod_node, context)
                end select
            end do
        type is (module_node)
            ! A module-only compilation unit (ffc -c <module>.f90).
            call collect_one_module_exports(arena, program, context)
        end select
    end subroutine collect_module_exports

    subroutine collect_one_module_exports(arena, mod_node, context)
        type(ast_arena_t), intent(in) :: arena
        type(module_node), intent(in) :: mod_node
        type(lowering_context_t), intent(inout) :: context
        integer :: j
        integer :: export_idx
        character(len=:), allocatable :: decl_name, name_err

        if (.not. allocated(mod_node%name)) return
        call grow_module_exports(context)

        export_idx = context%module_export_count + 1
        context%module_exports(export_idx)%module_name = trim(mod_node%name)
        context%module_exports(export_idx)%derived_type_count = 0
        allocate(context%module_exports(export_idx)%derived_type_indices(8))
        context%module_exports(export_idx)%parameter_count = 0
        allocate(context%module_exports(export_idx)%parameter_indices(8))
        context%module_exports(export_idx)%variable_count = 0
        allocate(context%module_exports(export_idx)%variable_indices(8))
        context%module_exports(export_idx)%enum_count = 0
        allocate(context%module_exports(export_idx)%enum_indices(8))

        if (.not. allocated(mod_node%declaration_indices)) then
            context%module_export_count = export_idx
            return
        end if
        do j = 1, size(mod_node%declaration_indices)
            select type (decl => &
                arena%entries(mod_node%declaration_indices(j))%node)
            type is (enum_node)
                ! Record module ENUM definitions so module procedures can
                ! host-associate the enumerators (#1826).
                call grow_module_export_enum_indices(context, export_idx)
                context%module_exports(export_idx)%enum_indices( &
                    context%module_exports(export_idx)%enum_count + 1) = &
                    mod_node%declaration_indices(j)
                context%module_exports(export_idx)%enum_count = &
                    context%module_exports(export_idx)%enum_count + 1
            type is (derived_type_node)
                if (.not. allocated(decl%name)) cycle
                ! A private symbol is not exported through use (#149).
                if (module_symbol_is_private(arena, mod_node, trim(decl%name))) cycle
                call grow_module_export_derived_indices(context, export_idx)
                context%module_exports(export_idx)%derived_type_indices( &
                    context%module_exports(export_idx)%derived_type_count + 1) = &
                    mod_node%declaration_indices(j)
                context%module_exports(export_idx)%derived_type_count = &
                    context%module_exports(export_idx)%derived_type_count + 1
            type is (declaration_node)
                call get_declaration_var_name(arena, mod_node%declaration_indices(j), &
                                              decl_name, name_err)
                if (len_trim(name_err) > 0 .or. len_trim(decl_name) == 0) cycle
                ! A PROCEDURE declaration (procedure(iface) :: p) exports a
                ! procedure name, not module variable storage (#364).
                if (declaration_names_procedure(decl)) then
                    if (.not. decl%is_pointer) cycle
                end if
                ! A scalar derived-type PARAMETER exports like a derived module
                ! variable: it gets a flat slot global (with its constructor
                ! folded into the static bytes) and binds as a derived reference
                ! in using units, which the integer-only constant path cannot do.
                if (decl%is_parameter .and. .not. decl%is_array .and. &
                    declaration_derived_type_index(context, decl) > 0) then
                    if (module_symbol_is_private(arena, mod_node, &
                            trim(decl_name))) cycle
                    call grow_module_export_variable_indices(context, export_idx)
                    context%module_exports(export_idx)%variable_indices( &
                        context%module_exports(export_idx)%variable_count + 1) = &
                        mod_node%declaration_indices(j)
                    context%module_exports(export_idx)%variable_count = &
                        context%module_exports(export_idx)%variable_count + 1
                else if (decl%is_parameter) then
                    if (module_symbol_is_private(arena, mod_node, trim(decl_name))) &
                        cycle
                    call grow_module_export_parameter_indices(context, export_idx)
                    context%module_exports(export_idx)%parameter_indices( &
                        context%module_exports(export_idx)%parameter_count + 1) = &
                        mod_node%declaration_indices(j)
                    context%module_exports(export_idx)%parameter_count = &
                        context%module_exports(export_idx)%parameter_count + 1
                else
                    ! Non-parameter scalar or array: export as module variable
                    ! (#249 B7a). A module array is exported only when ffc can
                    ! store it as a static global; an out-of-scope array form
                    ! (allocatable/pointer, rank > 1, runtime extent) is skipped
                    ! here so a unit that merely declares it still compiles,
                    ! exactly as before fixed-size module arrays were supported.
                    if (module_symbol_is_private(arena, mod_node, trim(decl_name))) &
                        cycle
                    if (module_declaration_is_allocatable(arena, &
                            mod_node%declaration_indices(j))) then
                        block
                            integer :: ma_kind, ma_rank
                            character(len=:), allocatable :: ma_err
                            call module_allocatable_array_info(arena, &
                                mod_node%declaration_indices(j), ma_kind, &
                                ma_rank, context, ma_err)
                            if (len_trim(ma_err) > 0) cycle
                        end block
                    else if (decl%is_array) then
                        block
                            integer :: mv_kind, mv_size, mv_lower
                            character(len=:), allocatable :: mv_err
                            call module_array_variable_info(arena, &
                                mod_node%declaration_indices(j), context, &
                                mv_kind, mv_size, mv_lower, mv_err)
                            if (len_trim(mv_err) > 0) cycle
                            call probe_module_array_initializer(arena, &
                                mod_node%declaration_indices(j), mv_kind, &
                                mv_size, context, mv_err)
                            if (len_trim(mv_err) > 0) cycle
                        end block
                    end if
                    call grow_module_export_variable_indices(context, export_idx)
                    context%module_exports(export_idx)%variable_indices( &
                        context%module_exports(export_idx)%variable_count + 1) = &
                        mod_node%declaration_indices(j)
                    context%module_exports(export_idx)%variable_count = &
                        context%module_exports(export_idx)%variable_count + 1
                end if
            end select
        end do

        context%module_export_count = export_idx
    end subroutine collect_one_module_exports

    logical function module_symbol_is_private(arena, mod_node, name)
        ! Effective accessibility of a module symbol (#149): a name listed in
        ! `private :: ...` is private; one listed in `public :: ...` is public;
        ! otherwise the module default applies (public unless a bare `private`
        ! statement flipped it). The attribute form (`integer, private ::`) is
        ! also honoured when FortFront records it on the declaration.
        type(ast_arena_t), intent(in) :: arena
        type(module_node), intent(in) :: mod_node
        character(len=*), intent(in) :: name
        logical :: default_private
        integer :: j, k

        module_symbol_is_private = .false.
        default_private = .false.
        ! A module with only contained procedures has no declaration part, so no
        ! visibility statement can restrict a symbol: it stays public.
        if (.not. allocated(mod_node%declaration_indices)) return
        ! Pass 1: explicit per-name lists win regardless of position.
        do j = 1, size(mod_node%declaration_indices)
            if (.not. node_exists(arena, mod_node%declaration_indices(j))) cycle
            select type (vis => arena%entries(mod_node%declaration_indices(j))%node)
            type is (visibility_statement_node)
                if (vis%has_list .and. allocated(vis%names)) then
                    do k = 1, size(vis%names)
                        if (allocated(vis%names(k)%s)) then
                            if (trim(lowercase_text(vis%names(k)%s)) == &
                                trim(lowercase_text(name))) then
                                module_symbol_is_private = vis%is_private
                                return
                            end if
                        end if
                    end do
                else if (vis%is_private) then
                    default_private = .true.
                else
                    default_private = .false.
                end if
            type is (declaration_node)
                ! Attribute form: integer, private, parameter :: name.
                if (decl_names_match(vis, name)) then
                    if (allocated(vis%accessibility)) then
                        if (trim(lowercase_text(vis%accessibility)) == 'private') then
                            module_symbol_is_private = .true.
                            return
                        else if (trim(lowercase_text(vis%accessibility)) == 'public') then
                            module_symbol_is_private = .false.
                            return
                        end if
                    end if
                end if
            end select
        end do
        module_symbol_is_private = default_private
    end function module_symbol_is_private

    logical function decl_names_match(decl, name)
        type(declaration_node), intent(in) :: decl
        character(len=*), intent(in) :: name
        integer :: k

        decl_names_match = .false.
        if (allocated(decl%var_name)) then
            if (trim(decl%var_name) == trim(name)) decl_names_match = .true.
        end if
        if (decl%is_multi_declaration .and. allocated(decl%var_names)) then
            do k = 1, size(decl%var_names)
                if (trim(decl%var_names(k)) == trim(name)) decl_names_match = .true.
            end do
        end if
    end function decl_names_match

    integer function find_module_export_index(context, module_name) &
        result(idx)
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: module_name
        integer :: i

        idx = 0
        do i = 1, context%module_export_count
            if (same_name(context%module_exports(i)%module_name, module_name)) then
                idx = i
                return
            end if
        end do
    end function find_module_export_index

    logical function symbol_is_scope_finalizable(context, symbol_index) &
            result(is_finalizable)
        ! An owned local scalar derived variable of a type that declares a
        ! scalar FINAL procedure (#403). Dummies borrow their storage and
        ! globals outlive the scope, so neither is finalized here.
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: symbol_index
        integer :: type_index

        is_finalizable = .false.
        if (.not. context%symbols(symbol_index)%is_derived) return
        if (context%symbols(symbol_index)%is_dummy_argument) return
        ! A pointer names storage it does not own.
        if (context%symbols(symbol_index)%is_pointer) return
        if (context%symbols(symbol_index)%is_array) then
            ! An allocatable derived array owns storage only while its
            ! descriptor data pointer is non-null.  Whether it is currently
            ! allocated is checked at runtime when the finalizer runs, so the
            ! scope-exit path emits the call unconditionally and guards on the
            ! descriptor (#643 / #403).
            if (.not. context%symbols(symbol_index)%is_allocatable) return
        end if
        ! An allocatable owns storage only while it is allocated; a moved-from
        ! or deallocated value has already lost its address.
        if (context%symbols(symbol_index)%is_allocatable .and. &
            context%symbols(symbol_index)%array_rank == 0) then
            if (.not. context%symbols(symbol_index)%has_address) return
        end if
        if (symbol_has_global_storage(context%symbols(symbol_index))) return
        type_index = context%symbols(symbol_index)%derived_type_index
        if (type_index <= 0) return
        if (type_index > context%derived_type_count) return
        if (len_trim(context%derived_types(type_index)%final_proc_name) == 0) return
        is_finalizable = .true.
    end function symbol_is_scope_finalizable

    subroutine emit_scalar_finalizer_call(context, symbol_index, error_msg)
        ! Call the declared scalar FINAL procedure with the value's storage
        ! address, exactly once at the point the value dies (#403).
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: args(1)
        character(len=:), allocatable :: final_name
        integer :: type_index

        call set_empty(error_msg)
        type_index = context%symbols(symbol_index)%derived_type_index
        if (type_index <= 0) return
        if (type_index > context%derived_type_count) return
        final_name = trim(context%derived_types(type_index)%final_proc_name)
        if (len_trim(final_name) == 0) return
        args(1) = context%symbols(symbol_index)%element_address
        if (.not. emit_call_with_optional_padding(context, &
                call_emit_name(context%arena, final_name, context), args, &
                error_msg)) return
        call set_empty(error_msg)
    end subroutine emit_scalar_finalizer_call

    subroutine emit_array_finalizer_call(context, symbol_index, error_msg)
        ! Finalize every element of an allocated rank-1/rank-2 derived
        ! allocatable array by calling the type's scalar FINAL procedure once
        ! on each element's storage address, before the block is released.
        ! An unallocated descriptor (null data pointer) is a no-op, matching
        ! `deallocate` on an unallocated entity (#643 / #403).
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: descriptor, data_ptr
        type(lr_operand_desc_t) :: extent1, extent2, count64, element_bytes
        type(lr_operand_desc_t) :: index64, byte_offset, element_addr
        type(lr_operand_desc_t) :: entry_index, header_index, next_index
        type(lr_operand_desc_t) :: condition, extent32
        type(lr_operand_desc_t) :: args(1)
        character(len=:), allocatable :: final_name
        integer(c_int32_t) :: entry_block, header_block, body_block
        integer(c_int32_t) :: latch_block, exit_block, index_vreg
        integer :: type_index, rank, slots

        call set_empty(error_msg)
        if (context%symbols(symbol_index)%is_polymorphic) then
            ! The descriptor records a dynamic finalizer and element layout,
            ! while this slice is statically specialized to the declared type.
            ! Refuse the case until runtime finalizer/layout dispatch is
            ! available instead of calling the wrong procedure or stepping by
            ! the wrong element width (#643 / #403).
            error_msg = 'polymorphic allocatable array finalization is not '// &
                        'supported'
            return
        end if
        type_index = context%symbols(symbol_index)%derived_type_index
        if (type_index <= 0) return
        if (type_index > context%derived_type_count) return
        final_name = trim(context%derived_types(type_index)%final_proc_name)
        if (len_trim(final_name) == 0) return
        rank = context%symbols(symbol_index)%array_rank
        if (rank < 1 .or. rank > 2) then
            error_msg = 'allocatable derived array finalization supports rank 1 '// &
                        'and rank 2 only'
            return
        end if

        descriptor = context%symbols(symbol_index)%allocatable_descriptor_address
        if (.not. emit_ptr_load(context%session, descriptor, data_ptr, &
                                error_msg)) return
        call emit_alloc_desc_load_extent(context, descriptor, 1, extent1, error_msg)
        if (len_trim(error_msg) > 0) return
        if (rank == 2) then
            call emit_alloc_desc_load_extent(context, descriptor, 2, extent2, &
                                             error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_i64_binary(context%session, LR_OP_MUL, extent1, &
                    extent2, count64, error_msg)) return
        else
            count64 = extent1
        end if
        slots = total_component_slots(context, type_index)
        if (slots <= 0) return
        element_bytes = i64_immediate(context%session, &
            int(slots, c_int64_t) * 4_c_int64_t)
        if (.not. emit_liric_i64_to_i32(context%session, count64, extent32, &
                                        error_msg)) return

        entry_block = context%current_block_id
        header_block = create_liric_block(context%session)
        body_block = create_liric_block(context%session)
        latch_block = create_liric_block(context%session)
        exit_block = create_liric_block(context%session)
        if (.not. emit_liric_br(context%session, header_block, error_msg)) return

        if (.not. set_liric_block(context%session, header_block, error_msg)) return
        context%current_block_id = header_block
        context%current_block_terminated = .false.

        entry_index = i32_immediate(context%session, 0_c_int64_t)
        index_vreg = reserve_i32_vreg(context%session)
        if (index_vreg <= 0_c_int32_t) then
            error_msg = 'LIRIC could not reserve a derived array finalizer index'
            return
        end if
        index64 = i32_vreg(context%session, index_vreg)
        if (.not. emit_liric_i32_phi(context%session, entry_index, entry_block, &
                index64, latch_block, header_index, error_msg)) return
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_SLT, header_index, &
                extent32, condition, error_msg)) return
        if (.not. emit_liric_condbr(context%session, condition, body_block, &
                                    exit_block, error_msg)) return

        if (.not. set_liric_block(context%session, body_block, error_msg)) return
        context%current_block_id = body_block
        context%current_block_terminated = .false.
        if (.not. emit_liric_i32_to_i64(context%session, header_index, index64, &
                                        error_msg)) return
        if (.not. emit_i64_binary(context%session, LR_OP_MUL, index64, &
                element_bytes, byte_offset, error_msg)) return
        if (.not. emit_ptr_offset_dyn(context%session, data_ptr, byte_offset, &
                                      element_addr, error_msg)) return
        args(1) = element_addr
        if (.not. emit_call_with_optional_padding(context, &
                call_emit_name(context%arena, final_name, context), args, &
                error_msg)) return
        if (.not. emit_liric_br(context%session, latch_block, error_msg)) return

        if (.not. set_liric_block(context%session, latch_block, error_msg)) return
        context%current_block_id = latch_block
        context%current_block_terminated = .false.
        if (.not. emit_i32_binary_into(context%session, LR_OP_ADD, header_index, &
                i32_immediate(context%session, 1_c_int64_t), index_vreg, &
                next_index, error_msg)) return
        if (.not. emit_liric_br(context%session, header_block, error_msg)) return

        if (.not. set_liric_block(context%session, exit_block, error_msg)) return
        context%current_block_id = exit_block
        context%current_block_terminated = .false.
        call set_empty(error_msg)
    end subroutine emit_array_finalizer_call

    subroutine emit_scope_exit_finalizers(context, error_msg)
        ! End of a procedure's execution finalizes each owned local derived
        ! value once, in reverse declaration order (#403).
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i

        call set_empty(error_msg)
        do i = context%symbol_count, 1, -1
            if (.not. symbol_is_scope_finalizable(context, i)) cycle
            if (context%symbols(i)%is_array) then
                call emit_array_finalizer_call(context, i, error_msg)
            else
                call emit_scalar_finalizer_call(context, i, error_msg)
            end if
            if (len_trim(error_msg) > 0) return
        end do
    end subroutine emit_scope_exit_finalizers

    subroutine lower_derived_type_definition(node, context, error_msg)
        ! Register a derived type definition. A parameterized type (#411)
        ! carries no layout of its own: it registers as a template, and each
        ! declaration that supplies constant actual parameters instantiates a
        ! concrete layout under a mangled instance name.
        type(derived_type_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg

        call set_empty(error_msg)
        if (.not. allocated(node%name)) then
            error_msg = 'derived type definition did not expose a name'
            return
        end if
        if (derived_type_is_parameterized(node)) then
            call record_pdt_template(context, trim(node%name))
            return
        end if
        ! A type extending a PDT inherits its formals, so it has no layout of
        ! its own either: it is a template whose instances are selected by the
        ! inherited actuals (#621).
        if (allocated(node%extends_parent)) then
            if (find_pdt_template(context, trim(node%extends_parent)) > 0) then
                call record_pdt_template(context, trim(node%name))
                return
            end if
        end if
        if (allocated(node%param_indices)) then
            call unsupported_feature_error('derived type parameters', &
                                           node%line, node%column, &
                                           'direct LIRIC session only supports '// &
                                           'non-parameterized derived types', &
                                           error_msg)
            return
        end if
        call register_derived_type_layout(node, context, trim(node%name), &
                                          error_msg)
    end subroutine lower_derived_type_definition

    recursive subroutine register_derived_type_layout(node, context, layout_name, &
                                                      error_msg, &
                                                      parent_layout_name)
        ! Record the concrete component layout of a derived type definition
        ! under layout_name. For a plain type that is the type name; for a PDT
        ! instance it is the mangled name carrying the actual parameters.
        type(derived_type_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: layout_name
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=*), intent(in), optional :: parent_layout_name
        character(len=:), allocatable :: parent_name
        integer :: type_index

        type_index = find_derived_type(context, layout_name)
        if (type_index > 0) then
            if (context%derived_types(type_index)%layout_registered) then
                call set_empty(error_msg)
                return
            end if
        else
            call grow_derived_types(context)
            type_index = context%derived_type_count + 1
            context%derived_type_count = type_index
        end if
        ! A module may reject a type after partially collecting its layout. The
        ! module collector intentionally keeps going for unrelated declarations;
        ! clear the reusable slot first so the failed type cannot contaminate
        ! the next type's layout or its exported .fmod record.
        context%derived_types(type_index)%component_count = 0
        context%derived_types(type_index)%binding_count = 0
        context%derived_types(type_index)%parent_name = ''
        context%derived_types(type_index)%final_proc_name = ''
        context%derived_types(type_index)%duplicate_binding_name = ''
        context%derived_types(type_index)%is_imported = .false.
        if (allocated(context%derived_types(type_index)%component_names)) &
            deallocate(context%derived_types(type_index)%component_names)
        if (allocated(context%derived_types(type_index)%component_has_default)) &
            deallocate(context%derived_types(type_index)%component_has_default)
        if (allocated(context%derived_types(type_index)%component_default_value)) &
            deallocate(context%derived_types(type_index)%component_default_value)
        if (allocated(context%derived_types(type_index)%component_array_size)) &
            deallocate(context%derived_types(type_index)%component_array_size)
        if (allocated(context%derived_types(type_index)%component_value_kind)) &
            deallocate(context%derived_types(type_index)%component_value_kind)
        if (allocated(context%derived_types(type_index)%component_type_index)) &
            deallocate(context%derived_types(type_index)%component_type_index)
        if (allocated(context%derived_types(type_index)%component_is_allocatable)) &
            deallocate(context%derived_types(type_index)%component_is_allocatable)
        if (allocated(context%derived_types(type_index)%component_is_pointer)) &
            deallocate(context%derived_types(type_index)%component_is_pointer)
        if (allocated(context%derived_types(type_index)%component_is_alloc_array)) &
            deallocate(context%derived_types(type_index)%component_is_alloc_array)
        if (allocated(context%derived_types(type_index)%component_alloc_rank)) &
            deallocate(context%derived_types(type_index)%component_alloc_rank)
        if (allocated(context%derived_types(type_index)%component_char_length)) &
            deallocate(context%derived_types(type_index)%component_char_length)
        if (allocated(context%derived_types(type_index)%component_dim1)) &
            deallocate(context%derived_types(type_index)%component_dim1)
        if (allocated(context%derived_types(type_index)%binding_method_names)) &
            deallocate(context%derived_types(type_index)%binding_method_names)
        if (allocated(context%derived_types(type_index)%binding_target_names)) &
            deallocate(context%derived_types(type_index)%binding_target_names)
        if (allocated(context%derived_types(type_index)%binding_specific_names)) &
            deallocate(context%derived_types(type_index)%binding_specific_names)
        if (allocated(context%derived_types(type_index)%binding_pass_names)) &
            deallocate(context%derived_types(type_index)%binding_pass_names)
        if (allocated(context%derived_types(type_index)%binding_pass_args)) &
            deallocate(context%derived_types(type_index)%binding_pass_args)
        context%derived_types(type_index)%name = trim(layout_name)
        context%derived_types(type_index)%layout_registered = .false.
        context%derived_types(type_index)%binding_count = 0
        if (allocated(node%extends_parent)) then
            parent_name = trim(node%extends_parent)
            if (present(parent_layout_name)) parent_name = trim(parent_layout_name)
            context%derived_types(type_index)%parent_name = parent_name
            call inherit_parent_members(node, context, type_index, parent_name, &
                                        error_msg)
            if (len_trim(error_msg) > 0) return
        end if
        if (allocated(node%component_indices)) then
            call collect_derived_components(node, context, type_index, error_msg)
            if (len_trim(error_msg) > 0) return
        end if
        ! An empty or behavioral-only type (no data components, e.g. a type that
        ! carries only a `contains` type-bound-procedure block, or a bare
        ! `type :: t; end type`) registers with a single hidden i32 placeholder
        ! slot so it has a nonzero footprint and reuses the fully-supported
        ! single-integer layout. This lets class(t)/type(t) declarations,
        ! type-bound dispatch, and allocation resolve against the type instead
        ! of failing because the type never registered.
        if (context%derived_types(type_index)%component_count <= 0) then
            call add_derived_component(context, type_index, '__ffc_placeholder', &
                                       node%line, node%column, .false., &
                                       0_c_int64_t, 1, VALUE_I32, error_msg)
            if (len_trim(error_msg) > 0) return
        end if
        call collect_type_bindings(node, context, type_index, error_msg)
        if (len_trim(error_msg) > 0) return

        context%derived_type_count = type_index
        context%derived_types(type_index)%layout_registered = .true.
        call set_empty(error_msg)
    end subroutine register_derived_type_layout

    subroutine collect_type_bindings(node, context, type_index, error_msg)
        ! Record type-bound procedure bindings (procedure :: m => fn) as
        ! compile-time aliases. When inherited bindings are already present
        ! (from inherit_parent_bindings), child declarations either override
        ! an inherited name or append a new one (#248 B6b).
        type(derived_type_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: type_index
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i, b, inherited, new_total, slot
        character(len=64) :: method_name_buf, target_name_buf, pass_name_buf
        character(len=1024) :: specific_names_buf
        logical :: overrides, pass_arg_buf, generic_alias
        character(len=64), allocatable :: old_methods(:), old_targets(:), old_passes(:)
        character(len=1024), allocatable :: old_specifics(:)
        logical, allocatable :: old_pass_args(:)
        integer :: g

        call set_empty(error_msg)
        if (.not. allocated(node%binding_indices)) return

        inherited = context%derived_types(type_index)%binding_count
        if (inherited == 0) then
            ! No inherited bindings: allocate fresh arrays sized to own bindings.
            allocate(context%derived_types(type_index)% &
                binding_method_names(size(node%binding_indices)))
            allocate(context%derived_types(type_index)% &
                binding_target_names(size(node%binding_indices)))
            allocate(context%derived_types(type_index)% &
                binding_specific_names(size(node%binding_indices)))
            allocate(context%derived_types(type_index)% &
                binding_pass_names(size(node%binding_indices)))
            allocate(context%derived_types(type_index)% &
                binding_pass_args(size(node%binding_indices)))
            context%derived_types(type_index)%binding_pass_names = ''
            context%derived_types(type_index)%binding_specific_names = ''
            context%derived_types(type_index)%binding_pass_args = .false.
        else
            ! Inherited bindings exist: grow arrays to hold inherited + new own.
            new_total = inherited + size(node%binding_indices)
            old_methods = context%derived_types(type_index)%binding_method_names
            old_targets = context%derived_types(type_index)%binding_target_names
            old_specifics = context%derived_types(type_index)%binding_specific_names
            old_passes  = context%derived_types(type_index)%binding_pass_names
            old_pass_args = context%derived_types(type_index)%binding_pass_args
            deallocate(context%derived_types(type_index)%binding_method_names)
            deallocate(context%derived_types(type_index)%binding_target_names)
            deallocate(context%derived_types(type_index)%binding_specific_names)
            deallocate(context%derived_types(type_index)%binding_pass_names)
            deallocate(context%derived_types(type_index)%binding_pass_args)
            allocate(context%derived_types(type_index)%binding_method_names(new_total))
            allocate(context%derived_types(type_index)%binding_target_names(new_total))
            allocate(context%derived_types(type_index)%binding_specific_names(new_total))
            allocate(context%derived_types(type_index)%binding_pass_names(new_total))
            allocate(context%derived_types(type_index)%binding_pass_args(new_total))
            context%derived_types(type_index)%binding_method_names = ''
            context%derived_types(type_index)%binding_target_names = ''
            context%derived_types(type_index)%binding_specific_names = ''
            context%derived_types(type_index)%binding_pass_names = ''
            context%derived_types(type_index)%binding_pass_args = .false.
            context%derived_types(type_index)%binding_method_names(1:inherited) = &
                old_methods(1:inherited)
            context%derived_types(type_index)%binding_target_names(1:inherited) = &
                old_targets(1:inherited)
            context%derived_types(type_index)%binding_specific_names(1:inherited) = &
                old_specifics(1:inherited)
            context%derived_types(type_index)%binding_pass_names(1:inherited) = &
                old_passes(1:inherited)
            context%derived_types(type_index)%binding_pass_args(1:inherited) = &
                old_pass_args(1:inherited)
        end if

        do i = 1, size(node%binding_indices)
            if (.not. node_exists(context%arena, node%binding_indices(i))) cycle
            select type (binding => &
                context%arena%entries(node%binding_indices(i))%node)
            type is (type_binding_node)
                generic_alias = .false.
                ! A deferred binding on an abstract type has no implementation
                ! to emit, but its name still occupies an inherited vtable
                ! slot.  Keep that slot so an extending type can override it;
                ! the vtable emitter already leaves an unresolved target null.
                if (binding%is_generic) then
                    if (.not. allocated(binding%generic_list) .or. &
                        size(binding%generic_list) == 0) cycle
                    generic_alias = .true.
                end if
                ! FINAL names a destructor, not a callable method: record it
                ! so an owned scalar value of this type is finalized once at
                ! the end of its lifetime (#403). A second FINAL binding is
                ! ambiguous for the scalar case ffc models, so it is rejected
                ! rather than resolved by picking one.
                if (binding%is_final) then
                    if (.not. allocated(binding%binding_name)) cycle
                    if (len_trim(context%derived_types(type_index)% &
                                 final_proc_name) > 0) then
                        call unsupported_feature_error('final procedure', &
                            node%line, node%column, &
                            'direct LIRIC session supports one scalar final '// &
                            'procedure per type; "'//trim(node%name)// &
                            '" declares several', error_msg)
                        return
                    end if
                    context%derived_types(type_index)%final_proc_name = &
                        trim(binding%binding_name)
                    cycle
                end if
                if (.not. allocated(binding%binding_name)) cycle
                method_name_buf = trim(binding%binding_name)
                if (binding%is_deferred) then
                    ! An abstract deferred binding occupies a vtable slot but
                    ! has no linkable implementation in this compilation
                    ! unit. Keep the slot for extensions and leave its target
                    ! null so an abstract module can still be linked.
                    target_name_buf = ''
                else if (generic_alias) then
                    target_name_buf = trim(binding%generic_list(1)%s)
                else if (allocated(binding%implementation) .and. &
                         len_trim(binding%implementation) > 0) then
                    target_name_buf = trim(binding%implementation)
                else
                    target_name_buf = trim(binding%binding_name)
                end if
                specific_names_buf = ''
                if (generic_alias) then
                    do g = 1, size(binding%generic_list)
                        if (len_trim(binding%generic_list(g)%s) == 0) cycle
                        if (len_trim(specific_names_buf) > 0) then
                            specific_names_buf = trim(specific_names_buf)//' '// &
                                trim(binding%generic_list(g)%s)
                        else
                            specific_names_buf = trim(binding%generic_list(g)%s)
                        end if
                    end do
                else
                    specific_names_buf = trim(target_name_buf)
                end if
                pass_name_buf = ''
                if (allocated(binding%pass_name)) then
                    if (len_trim(binding%pass_name) > 0) &
                        pass_name_buf = trim(binding%pass_name)
                end if
                pass_arg_buf = binding%pass_arg
                ! Override inherited binding with same method name, or append.
                overrides = .false.
                do b = 1, inherited
                    if (same_name(context%derived_types(type_index)% &
                              binding_method_names(b), method_name_buf)) then
                        context%derived_types(type_index)%binding_target_names(b) = &
                            target_name_buf
                        context%derived_types(type_index)%binding_specific_names(b) = &
                            specific_names_buf
                        context%derived_types(type_index)%binding_pass_names(b) = &
                            pass_name_buf
                        context%derived_types(type_index)%binding_pass_args(b) = &
                            pass_arg_buf
                        overrides = .true.
                        exit
                    end if
                end do
                if (.not. overrides) then
                    ! A name declared twice in the same type has no single
                    ! vtable slot occupant, so dispatch through it would be
                    ! ambiguous: reject it instead of silently keeping one
                    ! (#420).
                    do b = inherited + 1, &
                        context%derived_types(type_index)%binding_count
                        if (trim(context%derived_types(type_index)% &
                                 binding_method_names(b)) /= &
                            trim(method_name_buf)) cycle
                        context%derived_types(type_index)% &
                            duplicate_binding_name = method_name_buf
                        exit
                    end do
                    slot = context%derived_types(type_index)%binding_count + 1
                    context%derived_types(type_index)%binding_method_names(slot) = &
                        method_name_buf
                    context%derived_types(type_index)%binding_target_names(slot) = &
                        target_name_buf
                    context%derived_types(type_index)%binding_specific_names(slot) = &
                        specific_names_buf
                    context%derived_types(type_index)%binding_pass_names(slot) = &
                        pass_name_buf
                    context%derived_types(type_index)%binding_pass_args(slot) = &
                        pass_arg_buf
                    context%derived_types(type_index)%binding_count = slot
                end if
            end select
        end do
    end subroutine collect_type_bindings

    subroutine collect_derived_components(node, context, type_index, error_msg)
        type(derived_type_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: type_index
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i

        ! Own components append after any inherited (parent-first) components,
        ! so the child layout is parent components then own (#248 B6a).
        do i = 1, size(node%component_indices)
            call collect_component_declaration( &
                node%component_indices(i), node%line, node%column, context, &
                type_index, error_msg)
            if (len_trim(error_msg) > 0) return
        end do
        call set_empty(error_msg)
    end subroutine collect_derived_components

    subroutine inherit_parent_members(node, context, type_index, parent_name, &
                                      error_msg)
        ! Copy the parent type's components (parent-first layout) and its
        ! static type-bound bindings into the extending child (#248 B6a). The
        ! parent must already be collected; types are collected in source order
        ! and Fortran requires the parent to be defined before it is extended.
        ! parent_name is the parent's layout name: the plain type name, or the
        ! mangled instance name when the parent is a PDT instance (#621).
        type(derived_type_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: type_index
        character(len=*), intent(in) :: parent_name
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: parent_index, c

        call set_empty(error_msg)
        parent_index = find_derived_type(context, parent_name)
        if (parent_index <= 0) then
            call unsupported_feature_error('derived type inheritance', &
                node%line, node%column, 'parent type "'// &
                trim(parent_name)//'" must be defined before the type '// &
                'that extends it', error_msg)
            return
        end if

        associate (parent => context%derived_types(parent_index))
            do c = 1, parent%component_count
                call add_derived_component(context, type_index, &
                    parent%component_names(c), node%line, node%column, &
                    parent%component_has_default(c), &
                    parent%component_default_value(c), &
                    parent%component_array_size(c), &
                    parent%component_value_kind(c), error_msg, &
                    parent%component_type_index(c), &
                    is_allocatable=parent%component_is_allocatable(c), &
                    comp_char_length=parent%component_char_length(c), &
                    is_alloc_array=parent%component_is_alloc_array(c), &
                    alloc_rank=parent%component_alloc_rank(c), &
                    comp_dim1=parent%component_dim1(c), &
                    is_pointer=parent%component_is_pointer(c))
                if (len_trim(error_msg) > 0) return
            end do
        end associate
        call inherit_parent_bindings(context, type_index, parent_index)
    end subroutine inherit_parent_members

    subroutine inherit_parent_bindings(context, type_index, parent_index)
        ! Copy the parent's static type-bound bindings into the child so a
        ! parent method is callable on a child instance (static dispatch;
        ! overriding and dynamic dispatch are later #248 slices).
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: type_index
        integer, intent(in) :: parent_index
        integer :: pc, slot

        pc = context%derived_types(parent_index)%binding_count
        if (pc <= 0) return
        if (.not. allocated(context%derived_types(parent_index)% &
                            binding_method_names)) return
        allocate(context%derived_types(type_index)%binding_method_names(pc))
        allocate(context%derived_types(type_index)%binding_target_names(pc))
        allocate(context%derived_types(type_index)%binding_specific_names(pc))
        allocate(context%derived_types(type_index)%binding_pass_names(pc))
        allocate(context%derived_types(type_index)%binding_pass_args(pc))
        do slot = 1, pc
            context%derived_types(type_index)%binding_method_names(slot) = &
                context%derived_types(parent_index)%binding_method_names(slot)
            context%derived_types(type_index)%binding_target_names(slot) = &
                context%derived_types(parent_index)%binding_target_names(slot)
            context%derived_types(type_index)%binding_specific_names(slot) = &
                context%derived_types(parent_index)%binding_specific_names(slot)
            context%derived_types(type_index)%binding_pass_names(slot) = &
                context%derived_types(parent_index)%binding_pass_names(slot)
            context%derived_types(type_index)%binding_pass_args(slot) = &
                context%derived_types(parent_index)%binding_pass_args(slot)
        end do
        context%derived_types(type_index)%binding_count = pc
    end subroutine inherit_parent_bindings

    subroutine fold_component_default(context, init_index, value_kind, &
                                      default_value, error_msg)
        ! Fold a scalar component's default initializer to the raw bits that
        ! occupy its slots: an integer value, a logical (0/1), the f32 bit
        ! pattern of a default-real literal, the f64 bit pattern of an
        ! explicitly-typed real(real64) literal, or a null c_ptr. A character
        ! default cannot be summarised in one integer, so default_value carries
        ! the initializer's AST index instead; the text is refolded into the
        ! component's byte slots by character_default_words().
        use, intrinsic :: iso_c_binding, only: c_int32_t, c_int64_t, c_float, &
                                               c_double
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: init_index
        integer, intent(in) :: value_kind
        integer(c_int64_t), intent(out) :: default_value
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: lit_value, lit_type, lit_err
        character(len=:), allocatable :: text, id_name
        real(c_float) :: rval
        real(c_double) :: dval
        logical :: folded
        integer :: io_stat

        default_value = 0_c_int64_t
        call set_empty(error_msg)
        select case (value_kind)
        case (VALUE_I32)
            call parse_i32_constant(context%arena, init_index, default_value, &
                                    'component default', error_msg)
        case (VALUE_LOGICAL)
            if (.not. is_literal(context%arena, init_index)) then
                error_msg = 'non-literal logical component default'
                return
            end if
            call get_literal_info(context%arena, init_index, lit_value, &
                                  lit_type, lit_err)
            if (len_trim(lit_err) > 0) then
                error_msg = lit_err
                return
            end if
            default_value = int(logical_literal_to_i32(lit_value), c_int64_t)
        case (VALUE_F32)
            if (.not. is_literal(context%arena, init_index)) then
                error_msg = 'non-literal real component default'
                return
            end if
            call get_literal_info(context%arena, init_index, lit_value, &
                                  lit_type, lit_err)
            if (len_trim(lit_err) > 0) then
                error_msg = lit_err
                return
            end if
            read (lit_value, *, iostat=io_stat) rval
            if (io_stat /= 0) then
                error_msg = 'malformed real component default: '//lit_value
                return
            end if
            default_value = int(transfer(rval, 0_c_int32_t), c_int64_t)
        case (VALUE_F64)
            if (.not. is_literal(context%arena, init_index)) then
                error_msg = 'non-literal real component default'
                return
            end if
            call get_literal_info(context%arena, init_index, lit_value, &
                                  lit_type, lit_err)
            if (len_trim(lit_err) > 0) then
                error_msg = lit_err
                return
            end if
            call parse_f64_literal(lit_value, context, dval, lit_err, init_index)
            if (len_trim(lit_err) > 0) then
                error_msg = lit_err
                return
            end if
            default_value = transfer(dval, 0_c_int64_t)
        case (VALUE_C_PTR)
            if (.not. is_identifier(context%arena, init_index)) then
                error_msg = 'only c_null_ptr is a supported c_ptr default'
                return
            end if
            call get_identifier_name(context%arena, init_index, id_name, lit_err)
            if (len_trim(lit_err) > 0) then
                error_msg = lit_err
                return
            end if
            if (lowercase_text(trim(id_name)) /= 'c_null_ptr') then
                error_msg = 'only c_null_ptr is a supported c_ptr default'
                return
            end if
            default_value = 0_c_int64_t
        case (VALUE_CHARACTER)
            call concat_character_literals(context%arena, init_index, text, folded)
            if (.not. folded) then
                error_msg = 'non-literal character component default'
                return
            end if
            default_value = int(init_index, c_int64_t)
        case default
            error_msg = 'unsupported component default kind'
        end select
    end subroutine fold_component_default

    subroutine character_default_words(context, type_index, comp_index, words, &
                                       error_msg)
        ! Pack a character component's folded default into the i32 words its
        ! inline byte storage occupies: blank-padded to the declared length,
        ! truncated on overflow, then NUL-terminated in the reserved spare byte
        ! so the stored text prints and concatenates like any other character.
        use, intrinsic :: iso_c_binding, only: c_int32_t, c_int64_t
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: type_index
        integer, intent(in) :: comp_index
        integer(c_int64_t), allocatable, intent(out) :: words(:)
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: text, padded
        integer(c_int32_t) :: word_bits
        character(len=4) :: chunk
        integer :: dest_len, slot_count, init_index, k, copy_len
        logical :: folded

        call set_empty(error_msg)
        dest_len = context%derived_types(type_index)%component_char_length(comp_index)
        slot_count = context%derived_types(type_index)%component_array_size(comp_index)
        init_index = int(context%derived_types(type_index)% &
                         component_default_value(comp_index))
        call concat_character_literals(context%arena, init_index, text, folded)
        if (.not. folded) then
            error_msg = 'non-literal character component default'
            allocate (words(0))
            return
        end if
        allocate (character(len=slot_count*4) :: padded)
        padded(:) = ''
        copy_len = min(len(text), dest_len)
        if (copy_len > 0) padded(1:copy_len) = text(1:copy_len)
        padded(dest_len + 1:) = char(0)
        allocate (words(slot_count))
        do k = 1, slot_count
            chunk = padded(4*(k - 1) + 1:4*k)
            word_bits = transfer(chunk, word_bits)
            words(k) = int(word_bits, c_int64_t)
        end do
    end subroutine character_default_words

subroutine collect_component_declaration(component_index, parent_line, &
                                              parent_column, context, &
                                              type_index, error_msg)
        integer, intent(in) :: component_index
        integer, intent(in) :: parent_line
        integer, intent(in) :: parent_column
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: type_index
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i
        logical :: has_default
        integer(c_int64_t) :: default_value
        integer :: array_size
        integer :: value_kind
        integer :: nested_type_index
        logical :: comp_allocatable
        logical :: comp_alloc_array
        logical :: fixed_char_array
        logical :: comp_pointer
        integer :: char_length
        if (.not. node_exists(context%arena, component_index)) then
            error_msg = 'derived type component index does not reference an AST node'
            return
        end if

        select type (component => context%arena%entries(component_index)%node)
        type is (declaration_node)
            call validate_component_declaration(component, parent_line, &
                                                parent_column, context, &
                                                error_msg)
            if (len_trim(error_msg) > 0) return
            nested_type_index = declaration_derived_type_index(context, component)
            if (nested_type_index > 0) then
                if (component%is_allocatable .and. component%is_array .and. &
                    .not. component%is_pointer .and. &
                    component_alloc_array_rank_one(component)) then
                    call collect_alloc_derived_array_component(component, context, &
                        type_index, nested_type_index, error_msg)
                    return
                end if
                call collect_nested_derived_component(component, context, &
                    type_index, nested_type_index, error_msg)
                return
            end if
            if (is_proc_pointer_component(component)) then
                call collect_proc_pointer_component(component, context, &
                                                    type_index, error_msg)
                return
            end if
            call declaration_value_kind(component, value_kind, error_msg, context)
            if (len_trim(error_msg) > 0) return
            ! A scalar allocatable component of intrinsic numeric/logical type
            ! or character type stores an inline data pointer
            ! (descriptor-in-struct). Array, pointer, and c_ptr allocatables
            ! stay unsupported.
            ! A scalar data-pointer component of intrinsic numeric/logical
            ! type stores an inline 8-byte target address; it starts null and
            ! only `=>` or nullify changes it.
            comp_pointer = component%is_pointer .and. &
                           .not. component%is_array .and. &
                           .not. component%is_allocatable
            if (comp_pointer) then
                select case (value_kind)
                case (VALUE_I32, VALUE_F32, VALUE_F64, VALUE_LOGICAL)
                case default
                    comp_pointer = .false.
                end select
            end if
            comp_allocatable = component_is_scalar_allocatable(component)
            if (comp_allocatable) then
                select case (value_kind)
                case (VALUE_I32, VALUE_F32, VALUE_F64, VALUE_LOGICAL, &
                      VALUE_CHARACTER)
                case default
                    comp_allocatable = .false.
                end select
            end if
            ! A rank-1/rank-2/rank-3/rank-4 allocatable array component of intrinsic type
            ! stores an inline descriptor (data pointer + one i64 extent per
            ! dimension). Character elements use the same descriptor layout for
            ! the older rank-1/rank-2 path; rank three and rank four deliberately exclude
            ! character until its element descriptor semantics are complete.
            ! Higher ranks, pointer, and c_ptr element types stay unsupported.
            comp_alloc_array = component%is_allocatable .and. &
                               component%is_array .and. &
                               .not. component%is_pointer .and. &
                               component_alloc_array_rank_bounded(component)
            if (comp_alloc_array) then
                select case (value_kind)
                case (VALUE_I32, VALUE_F32, VALUE_F64, VALUE_LOGICAL, &
                      VALUE_CHARACTER)
                case default
                    comp_alloc_array = .false.
                end select
            end if
            if (component%is_allocatable .and. component%is_array .and. &
                .not. comp_alloc_array) then
                call unsupported_feature_error('allocatable array component', &
                    component%line, component%column, &
                    'only integer, real, and logical allocatable array '// &
                    'components of rank 1 through rank 4 are supported', error_msg)
                return
            end if
            if (component%is_allocatable .and. component%is_array .and. &
                component_alloc_array_rank(component) >= 3 .and. &
                component%is_target) then
                call unsupported_feature_error( &
                    'rank-3/rank-4 allocatable array component', component%line, &
                    component%column, &
                    'rank-3 and rank-4 allocatable TARGET components are not supported', &
                    error_msg)
                return
            end if
            ! A component default initialiser must be a compile-time integer.
            has_default = .false.
            default_value = 0_c_int64_t
            array_size = 1
            char_length = 0
            fixed_char_array = .false.
            if (comp_alloc_array .and. value_kind == VALUE_CHARACTER) then
                array_size = 1
                call declaration_character_length(context, component, &
                                                  char_length, error_msg)
                if (len_trim(error_msg) > 0) return
            else if (comp_alloc_array) then
                array_size = 1
            else if (component%is_array) then
                call component_array_count(component, context, array_size, &
                                           error_msg)
                if (len_trim(error_msg) > 0) return
                if (value_kind == VALUE_CHARACTER) then
                    call declaration_character_length(context, component, &
                                                      char_length, error_msg)
                    if (len_trim(error_msg) > 0) return
                    fixed_char_array = .true.
                end if
            else if (value_kind == VALUE_CHARACTER .and. .not. comp_allocatable) then
                ! Reuse array_size as the component's slot count: a character
                ! value occupies ceil((len+1)/4) byte-storage slots at width 1,
                ! the extra byte holding the load-time NUL terminator.
                call declaration_character_length(context, component, &
                                                  char_length, error_msg)
                if (len_trim(error_msg) > 0) return
                array_size = character_component_slots(char_length)
            end if
            if (component%has_initializer .and. component%initializer_index > 0) then
                if (component%is_array) then
                    call unsupported_feature_error( &
                        'derived type array component initializer', &
                        component%line, component%column, 'array components '// &
                        'cannot carry a default initialiser', error_msg)
                    return
                end if
                call fold_component_default(context, component%initializer_index, &
                                            value_kind, default_value, error_msg)
                if (len_trim(error_msg) > 0) then
                    call unsupported_feature_error( &
                        'derived type component initializer', component%line, &
                        component%column, 'only a compile-time integer, real, '// &
                        'logical, character, or c_null_ptr default is '// &
                        'supported', error_msg)
                    return
                end if
                has_default = .true.
            end if
            if (component%is_multi_declaration .and. &
                allocated(component%var_names)) then
                do i = 1, size(component%var_names)
                    call add_derived_component(context, type_index, &
                                               component%var_names(i), &
                                               component%line, component%column, &
                                               has_default, default_value, &
                                               array_size, value_kind, error_msg, &
                                               is_allocatable=comp_allocatable, &
                                               comp_char_length=char_length, &
                                               comp_dim1=merge(array_size, 0, &
                                                               fixed_char_array), &
                                               is_alloc_array=comp_alloc_array, &
                                               alloc_rank=merge( &
                                                   component_alloc_array_rank(component), &
                                                   0, comp_alloc_array), &
                                               is_pointer=comp_pointer)
                    if (len_trim(error_msg) > 0) return
                end do
            else if (allocated(component%var_name)) then
                call add_derived_component(context, type_index, component%var_name, &
                                           component%line, component%column, &
                                           has_default, default_value, array_size, &
                                           value_kind, error_msg, &
                                           is_allocatable=comp_allocatable, &
                                           comp_char_length=char_length, &
                                           comp_dim1=merge(array_size, 0, &
                                                           fixed_char_array), &
                                           is_alloc_array=comp_alloc_array, &
                                           alloc_rank=merge( &
                                               component_alloc_array_rank(component), &
                                               0, comp_alloc_array), &
                                           is_pointer=comp_pointer)
            else
                error_msg = 'derived type component declaration did not expose a name'
                return
            end if
        class default
            call unsupported_feature_error('derived type component', &
                                           parent_line, parent_column, &
                                           'direct LIRIC session only supports '// &
                                           'integer component declarations', &
                                           error_msg)
            return
        end select
        call set_empty(error_msg)
    end subroutine collect_component_declaration

    subroutine collect_nested_derived_component(component, context, type_index, &
                                                nested_type_index, error_msg)
        ! Record one or more scalar nested derived components. A nested component
        ! occupies the inner type's full slot count inline; component_array_size
        ! carries that count and component_value_kind is VALUE_DERIVED so the
        ! slot model reserves the right span. value access goes through the leaf
        ! scalar component, never the nested aggregate directly.
        type(declaration_node), intent(in) :: component
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: type_index
        integer, intent(in) :: nested_type_index
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i, inner_slots, element_count, span

        ! A scalar derived POINTER component is an opaque two-slot address,
        ! not an inline nested instance. Keep the pointee type for compatibility
        ! checks, but reserve only the pointer slot in the containing type.
        if (component%is_pointer) then
            if (component%is_array .or. component%is_allocatable) then
                call unsupported_feature_error('nested derived type component', &
                    component%line, component%column, 'only scalar derived '// &
                    'pointer components are supported', error_msg)
                return
            end if
            if (component%is_multi_declaration .and. allocated(component%var_names)) then
                do i = 1, size(component%var_names)
                    call add_derived_component(context, type_index, &
                        component%var_names(i), component%line, component%column, &
                        .false., 0_c_int64_t, 1, VALUE_DERIVED, error_msg, &
                        nested_type_index, is_pointer=.true.)
                    if (len_trim(error_msg) > 0) return
                end do
            else if (allocated(component%var_name)) then
                call add_derived_component(context, type_index, component%var_name, &
                    component%line, component%column, .false., 0_c_int64_t, 1, &
                    VALUE_DERIVED, error_msg, nested_type_index, is_pointer=.true.)
            else
                error_msg = 'derived type component declaration did not expose a name'
            end if
            return
        end if

        if (component%is_allocatable .and. .not. component%is_array) then
            if (component%is_multi_declaration .and. allocated(component%var_names)) then
                do i = 1, size(component%var_names)
                    call add_derived_component(context, type_index, &
                        component%var_names(i), component%line, component%column, &
                        .false., 0_c_int64_t, 1, VALUE_DERIVED, error_msg, &
                        nested_type_index, is_allocatable=.true.)
                    if (len_trim(error_msg) > 0) return
                end do
            else if (allocated(component%var_name)) then
                call add_derived_component(context, type_index, component%var_name, &
                    component%line, component%column, .false., 0_c_int64_t, 1, &
                    VALUE_DERIVED, error_msg, nested_type_index, &
                    is_allocatable=.true.)
            else
                error_msg = 'derived type component declaration did not expose a name'
            end if
            return
        end if

        if (component%has_initializer) then
            ! A bare inner-type default constructor (`type(inner) :: c =
            ! inner()`) equals plain default initialisation, which nested
            ! default propagation already materialises, so accept and ignore it.
            ! Any argument-bearing or non-constructor initialiser stays
            ! unsupported rather than silently dropping its value.
            if (.not. nested_default_constructor_initializer(component, context, &
                                                             nested_type_index)) then
                call unsupported_feature_error('nested derived type component', &
                    component%line, component%column, 'derived components can '// &
                    'only carry a bare default-constructor initialiser', error_msg)
                return
            end if
        end if
        inner_slots = total_component_slots(context, nested_type_index)
        element_count = 1
        if (component%is_array) then
            call component_array_count(component, context, element_count, error_msg)
            if (len_trim(error_msg) > 0) return
        end if
        span = element_count * inner_slots
        if (component%is_multi_declaration .and. allocated(component%var_names)) then
            do i = 1, size(component%var_names)
                call add_derived_component(context, type_index, &
                    component%var_names(i), component%line, component%column, &
                    .false., 0_c_int64_t, span, VALUE_DERIVED, error_msg, &
                    nested_type_index)
                if (len_trim(error_msg) > 0) return
            end do
        else if (allocated(component%var_name)) then
            call add_derived_component(context, type_index, component%var_name, &
                component%line, component%column, .false., 0_c_int64_t, &
                span, VALUE_DERIVED, error_msg, nested_type_index)
            if (len_trim(error_msg) > 0) return
            call record_component_leading_extent(component, context, type_index, &
                                                 error_msg)
        else
            error_msg = 'derived type component declaration did not expose a name'
            return
        end if
    end subroutine collect_nested_derived_component

    subroutine record_component_leading_extent(component, context, type_index, &
                                               error_msg)
        ! Store the leading extent of a rank-2 fixed-size derived array
        ! component so comp(i,j) can linearise column-major.
        type(declaration_node), intent(in) :: component
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: type_index
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int64_t) :: extent, lower_value
        integer :: comp_index

        call set_empty(error_msg)
        if (.not. component%is_array) return
        if (.not. allocated(component%dimension_indices)) return
        if (size(component%dimension_indices) /= 2) return
        if (.not. node_exists(context%arena, component%dimension_indices(1))) return
        select type (dim_node => &
                     context%arena%entries(component%dimension_indices(1))%node)
        type is (range_expression_node)
            call eval_i32_constant(context%arena, dim_node%start_index, context, &
                                   lower_value, error_msg)
            if (len_trim(error_msg) > 0) return
            call eval_i32_constant(context%arena, dim_node%end_index, context, &
                                   extent, error_msg)
            if (len_trim(error_msg) > 0) return
            extent = extent - lower_value + 1_c_int64_t
        class default
            call eval_i32_constant(context%arena, component%dimension_indices(1), &
                                   context, extent, error_msg)
            if (len_trim(error_msg) > 0) return
        end select
        if (extent < 0_c_int64_t) extent = 0_c_int64_t
        comp_index = context%derived_types(type_index)%component_count
        context%derived_types(type_index)%component_dim1(comp_index) = int(extent)
    end subroutine record_component_leading_extent

    subroutine collect_alloc_derived_array_component(component, context, &
                                                     type_index, elem_type_index, &
                                                     error_msg)
        ! Record a rank-1 allocatable array of a derived element type
        ! (type(inner), allocatable :: c(:)). The component stores an inline
        ! 16-byte descriptor (four i32 slots: data pointer at byte 0, i64 element
        ! extent at byte 8), exactly like an intrinsic allocatable array
        ! component, so is_alloc_array carries the descriptor layout. The element
        ! type index rides in component_type_index; addressing loads the data
        ! pointer and steps in by whole inner instances. A component initialiser
        ! is not valid for an allocatable array, so none is accepted here.
        type(declaration_node), intent(in) :: component
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: type_index
        integer, intent(in) :: elem_type_index
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i

        if (component%is_multi_declaration .and. allocated(component%var_names)) then
            do i = 1, size(component%var_names)
                call add_derived_component(context, type_index, &
                    component%var_names(i), component%line, component%column, &
                    .false., 0_c_int64_t, 1, VALUE_DERIVED, error_msg, &
                    elem_type_index, is_alloc_array=.true., alloc_rank=1)
                if (len_trim(error_msg) > 0) return
            end do
        else if (allocated(component%var_name)) then
            call add_derived_component(context, type_index, component%var_name, &
                component%line, component%column, .false., 0_c_int64_t, 1, &
                    VALUE_DERIVED, error_msg, elem_type_index, &
                    is_alloc_array=.true., alloc_rank=1)
        else
            error_msg = 'derived type component declaration did not expose a name'
            return
        end if
    end subroutine collect_alloc_derived_array_component

    logical function nested_default_constructor_initializer(component, context, &
                                                             nested_type_index)
        ! True when a nested derived component's initialiser is a bare
        ! zero-argument constructor of its own type (`inner()`), which is
        ! equivalent to default initialisation.
        type(declaration_node), intent(in) :: component
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: nested_type_index
        integer :: init_index

        nested_default_constructor_initializer = .false.
        init_index = component%initializer_index
        if (.not. node_exists(context%arena, init_index)) return
        if (.not. is_derived_constructor_call(context%arena, init_index, context, &
                                              nested_type_index)) return
        select type (node => context%arena%entries(init_index)%node)
        type is (call_or_subscript_node)
            if (.not. allocated(node%arg_indices)) then
                nested_default_constructor_initializer = .true.
            else if (size(node%arg_indices) == 0) then
                nested_default_constructor_initializer = .true.
            end if
        end select
    end function nested_default_constructor_initializer

    subroutine validate_component_declaration(component, parent_line, &
                                              parent_column, context, &
                                              error_msg)
        type(declaration_node), intent(in) :: component
        integer, intent(in) :: parent_line
        integer, intent(in) :: parent_column
        type(lowering_context_t), intent(in) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: nested_index
        integer :: comp_kind

        if (.not. allocated(component%type_name)) then
            call set_empty(error_msg)
            return
        end if
        ! A procedure-pointer component holds a callee address in an inline
        ! 8-byte slot pair (#372); arrays of them are out of scope.
        if (is_proc_pointer_component(component)) then
            if (component%is_array) then
                call unsupported_feature_error('derived type array component', &
                    component%line, component%column, 'arrays of procedure '// &
                    'pointer components are not supported', error_msg)
                return
            end if
            call set_empty(error_msg)
            return
        end if

        nested_index = declaration_derived_type_index(context, component)
        if (nested_index > 0) then
            ! A scalar or fixed-size-array nested derived component lives inline in
            ! the parent's flat slot layout (an array component spans
            ! element_count * inner slots). A rank-1 allocatable array of a
            ! derived element type instead stores an inline 16-byte descriptor and
            ! reaches its heap elements through a data pointer. Pointer, scalar
            ! allocatable, and higher-rank allocatable derived components stay
            ! unsupported.
            if (component%is_pointer) then
                if (component%is_array .or. component%is_allocatable) then
                    call unsupported_feature_error('nested derived type component', &
                        component%line, component%column, 'only scalar derived '// &
                        'pointer components are supported', error_msg)
                else
                    call set_empty(error_msg)
                end if
                return
            end if
            if (component%is_allocatable) then
                if (component%is_array .and. &
                    .not. component_alloc_array_rank_one(component)) then
                    call unsupported_feature_error('nested derived type component', &
                        component%line, component%column, 'only a rank-1 '// &
                        'allocatable array of a derived type is supported', &
                        error_msg)
                    return
                end if
            end if
            call set_empty(error_msg)
            return
        end if
        if (component%is_allocatable .and. component%is_array) then
            if (.not. component_alloc_array_rank_bounded(component)) then
                call unsupported_feature_error('derived type component', &
                    component%line, component%column, &
                    'only rank-1 through rank-4 intrinsic allocatable components '// &
                    'are supported', error_msg)
                return
            end if
        end if
        ! Accept scalar integer (incl. interop kinds and c_ptr), real (f32/f64),
        ! logical, and fixed-length character components. Numeric/logical
        ! components occupy whole 4-byte slots (f64 spans two, the rest one); a
        ! character(len=N) component occupies ceil(N/4) byte-storage slots.
        ! Arrays of these and nested components are handled elsewhere.
        call type_name_value_kind(component%type_name, component%line, &
                                  component%column, comp_kind, error_msg, &
                                  context)
        if (len_trim(error_msg) > 0) return
        if (component%is_allocatable .and. component%is_array) then
            select case (comp_kind)
            case (VALUE_I32, VALUE_F32, VALUE_F64, VALUE_LOGICAL)
            case default
                call unsupported_feature_error('allocatable array component', &
                    component%line, component%column, &
                    'only integer, real, and logical allocatable array '// &
                    'components of rank 1 through rank 4 are supported', error_msg)
                return
            end select
        end if
        select case (comp_kind)
        case (VALUE_I32, VALUE_I64, VALUE_C_PTR)
            ! Scalar and fixed-size array components are both supported.
        case (VALUE_F64, VALUE_F32, VALUE_LOGICAL)
            ! Scalar and fixed-size array components are both supported; an
            ! f64 array element spans two i32 slots (handled by slot width).
        case (VALUE_CHARACTER)
            if (component%is_array) then
                if (component%is_pointer) then
                    call unsupported_feature_error( &
                        'derived type array component', component%line, &
                        component%column, 'character array components are not '// &
                        'supported for pointer components', error_msg)
                    return
                end if
                if (component%is_allocatable) then
                    if (.not. component_alloc_array_rank_one(component)) then
                        call unsupported_feature_error( &
                            'derived type array component', component%line, &
                            component%column, 'only rank-1 allocatable character '// &
                            'arrays are supported', error_msg)
                        return
                    end if
                    ! A fixed-length character allocatable array has a four-slot
                    ! data/extent descriptor. Element access remains rejected
                    ! until the derived-component character descriptor path is
                    ! complete.
                end if
                call validate_character_component_length(component, context, &
                                                         error_msg)
                return
            end if
            ! A scalar allocatable character component stores an inline data
            ! pointer instead of inline bytes, so a deferred length is fine
            ! at layout time (#360).
            if (component_is_scalar_allocatable(component)) then
                call set_empty(error_msg)
                return
            end if
            ! A fixed positive length is required; deferred/runtime-length
            ! character components have no static slot footprint.
            call validate_character_component_length(component, context, error_msg)
            if (len_trim(error_msg) > 0) return
        case default
            call unsupported_feature_error('derived type component', &
                                           component%line, component%column, &
                                           'direct LIRIC session supports scalar '// &
                                           'integer, real, logical, and c_ptr '// &
                                           'components', error_msg)
            return
        end select
        call set_empty(error_msg)
    end subroutine validate_component_declaration

    ! A character(len=N) component needs a fixed positive length so its
    ! byte-storage slot footprint (ceil(N/4)) is known at layout time.
    subroutine validate_character_component_length(component, context, error_msg)
        type(declaration_node), intent(in) :: component
        type(lowering_context_t), intent(in) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: character_length

        call declaration_character_length(context, component, character_length, &
                                          error_msg)
        if (len_trim(error_msg) > 0) then
            call unsupported_feature_error('derived type character component', &
                component%line, component%column, 'character components need a '// &
                'fixed positive length', error_msg)
            return
        end if
        if (character_length <= 0) then
            call unsupported_feature_error('derived type character component', &
                component%line, component%column, 'deferred and assumed-length '// &
                'character components are not supported', error_msg)
            return
        end if
    end subroutine validate_character_component_length

    ! i32 slots a character(len=N) value occupies: N bytes plus one spare byte
    ! for the load-time NUL terminator (#265 read/write), one slot per 4 bytes
    ! rounded up. len<=0 yields zero slots (rejected before this point).
    integer function character_component_slots(character_length)
        integer, intent(in) :: character_length

        if (character_length <= 0) then
            character_component_slots = 0
        else
            character_component_slots = (character_length + 4) / 4
        end if
    end function character_component_slots

    logical function is_proc_pointer_component(component) result(is_proc)
        ! True for `procedure(iface), pointer :: p` inside a derived type: the
        ! component stores a callee address rather than a data value (#372).
        type(declaration_node), intent(in) :: component
        integer :: n

        ! FortFront reports the component type as "procedure()" or
        ! "procedure(iface)"; the POINTER attribute is not retained on the
        ! component node, and a component of procedure type is always a
        ! procedure pointer.
        is_proc = .false.
        if (.not. allocated(component%type_name)) return
        n = min(9, len_trim(component%type_name))
        if (n <= 0) return
        is_proc = lowercase_text(trim(adjustl(component%type_name(1:n)))) == &
                  'procedure'
    end function is_proc_pointer_component

    subroutine collect_proc_pointer_component(component, context, type_index, &
                                              error_msg)
        ! Record one or more procedure-pointer components. Each occupies an
        ! inline pointer slot pair (see component_slot_width) that a fresh
        ! instance nulls, so associated() reads false before any => binding.
        ! FortFront keeps the component's attribute keywords (nopass, pointer)
        ! in var_names, so only the entity names are taken from that list.
        type(declaration_node), intent(in) :: component
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: type_index
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i, added

        added = 0
        call set_empty(error_msg)
        if (allocated(component%var_names)) then
            do i = 1, size(component%var_names)
                if (is_proc_component_attribute(component%var_names(i))) cycle
                call add_derived_component(context, type_index, &
                                           component%var_names(i), &
                                           component%line, component%column, &
                                           .false., 0_c_int64_t, 1, &
                                           VALUE_PROC_PTR, error_msg)
                if (len_trim(error_msg) > 0) return
                added = added + 1
            end do
        end if
        if (added > 0) return
        if (.not. allocated(component%var_name)) then
            error_msg = 'derived type component declaration did not expose a name'
            return
        end if
        call add_derived_component(context, type_index, component%var_name, &
                                   component%line, component%column, &
                                   .false., 0_c_int64_t, 1, VALUE_PROC_PTR, &
                                   error_msg)
    end subroutine collect_proc_pointer_component

    logical function is_proc_component_attribute(text) result(is_attribute)
        ! True for an attribute keyword FortFront leaves in a procedure
        ! component's entity list rather than an entity name.
        character(len=*), intent(in) :: text
        character(len=:), allocatable :: word

        word = lowercase_text(trim(adjustl(text)))
        is_attribute = word == 'nopass' .or. word == 'pointer' .or. &
                       word == 'pass' .or. word == 'private' .or. &
                       word == 'public'
    end function is_proc_component_attribute
    logical function component_is_scalar_allocatable(component) result(is_scalar)
        ! True for a scalar (non-array, non-pointer) allocatable component,
        ! which is laid out as an inline data pointer.
        type(declaration_node), intent(in) :: component

        is_scalar = component%is_allocatable .and. .not. component%is_array &
                    .and. .not. component%is_pointer
    end function component_is_scalar_allocatable

    logical function component_alloc_array_rank_one(component) result(is_rank_one)
        ! True when an allocatable array component declares exactly one dimension.
        type(declaration_node), intent(in) :: component

        is_rank_one = .false.
        if (.not. allocated(component%dimension_indices)) return
        is_rank_one = size(component%dimension_indices) == 1
    end function component_alloc_array_rank_one

    integer function component_alloc_array_rank(component) result(rank)
        ! Declared rank of an allocatable array component. Zero means that the
        ! declaration is not an allocatable array or uses an unsupported rank.
        type(declaration_node), intent(in) :: component

        rank = 0
        if (.not. component%is_allocatable .or. .not. component%is_array) return
        if (.not. allocated(component%dimension_indices)) return
        if (size(component%dimension_indices) < 1 .or. &
            size(component%dimension_indices) > 4) return
        rank = size(component%dimension_indices)
    end function component_alloc_array_rank

    logical function component_alloc_array_rank_bounded(component) result(is_bounded)
        type(declaration_node), intent(in) :: component

        is_bounded = component_alloc_array_rank(component) >= 1 .and. &
                     component_alloc_array_rank(component) <= 4
    end function component_alloc_array_rank_bounded

    subroutine component_array_count(component, context, array_size, error_msg)
        ! Compile-time element count of an array component (literal or
        ! parameter bound). Only a default lower bound of 1 is assumed.
        type(declaration_node), intent(in) :: component
        type(lowering_context_t), intent(in) :: context
        integer, intent(out) :: array_size
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: array_lower_bound

        call get_array_bounds(component, context, array_lower_bound, array_size, &
                              error_msg)
        if (len_trim(error_msg) > 0) return
        if (array_size <= 0) then
            call unsupported_feature_error('derived type array component', &
                component%line, component%column, 'array components need a '// &
                'positive compile-time size', error_msg)
            return
        end if
        call set_empty(error_msg)
    end subroutine component_array_count

    subroutine add_derived_component(context, type_index, name, line, column, &
                                     has_default, default_value, array_size, &
                                     value_kind, error_msg, comp_type_index, &
                                     is_allocatable, comp_char_length, &
                                     is_alloc_array, alloc_rank, comp_dim1, &
                                     is_pointer)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: type_index
        character(len=*), intent(in) :: name
        integer, intent(in) :: line, column
        logical, intent(in) :: has_default
        integer(c_int64_t), intent(in) :: default_value
        integer, intent(in) :: array_size
        integer, intent(in) :: value_kind
        character(len=:), allocatable, intent(out) :: error_msg
        ! Derived type index of a nested derived component; 0 (or absent) for
        ! scalar and intrinsic-array components.
        integer, intent(in), optional :: comp_type_index
        ! True for a scalar allocatable component (descriptor-in-struct pointer).
        logical, intent(in), optional :: is_allocatable
        ! True for a rank-1 allocatable array component (inline descriptor).
        logical, intent(in), optional :: is_alloc_array
        integer, intent(in), optional :: alloc_rank
        ! True for a scalar data-pointer component (inline target address).
        logical, intent(in), optional :: is_pointer
        ! Declared length of a VALUE_CHARACTER component; 0 (or absent) for
        ! every other kind.
        integer, intent(in), optional :: comp_char_length
        ! Positive first-dimension extent for a fixed-size array component;
        ! also marks fixed-length character arrays, whose component_array_size
        ! stores element count rather than scalar-character slot count.
        integer, intent(in), optional :: comp_dim1
        integer :: component_index

        if (find_derived_component(context, type_index, name) > 0) then
            error_msg = 'duplicate derived type component: '//trim(name)
            return
        end if

        call grow_derived_type_components(context, type_index)

        component_index = context%derived_types(type_index)%component_count + 1
        context%derived_types(type_index)%component_names(component_index) = &
            trim(name)
        context%derived_types(type_index)%component_has_default(component_index) = &
            has_default
        context%derived_types(type_index)%component_default_value(component_index) = &
            default_value
        context%derived_types(type_index)%component_array_size(component_index) = &
            array_size
        context%derived_types(type_index)%component_value_kind(component_index) = &
            value_kind
        context%derived_types(type_index)%component_type_index(component_index) = 0
        if (present(comp_type_index)) &
            context%derived_types(type_index)%component_type_index(component_index) = &
                comp_type_index
        context%derived_types(type_index)%component_is_allocatable(component_index) = &
            .false.
        if (present(is_allocatable)) &
            context%derived_types(type_index)%component_is_allocatable( &
                component_index) = is_allocatable
        context%derived_types(type_index)%component_is_alloc_array(component_index) = &
            .false.
        if (present(is_alloc_array)) &
            context%derived_types(type_index)%component_is_alloc_array( &
                component_index) = is_alloc_array
        context%derived_types(type_index)%component_alloc_rank(component_index) = 0
        if (present(alloc_rank)) &
            context%derived_types(type_index)%component_alloc_rank(component_index) = &
                alloc_rank
        context%derived_types(type_index)%component_is_pointer(component_index) = &
            .false.
        if (present(is_pointer)) &
            context%derived_types(type_index)%component_is_pointer( &
                component_index) = is_pointer
        context%derived_types(type_index)%component_char_length(component_index) = 0
        if (present(comp_char_length)) &
            context%derived_types(type_index)%component_char_length(component_index) = &
                comp_char_length
        context%derived_types(type_index)%component_dim1(component_index) = 0
        if (present(comp_dim1)) &
            context%derived_types(type_index)%component_dim1(component_index) = &
                comp_dim1
        context%derived_types(type_index)%component_count = component_index
        call set_empty(error_msg)
    end subroutine add_derived_component

    subroutine grow_derived_type_components(context, type_index)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: type_index
        character(len=64), allocatable :: old_names(:)
        integer :: new_size, old_size

        old_size = context%derived_types(type_index)%component_count
        if (old_size > 0 .and. allocated(context%derived_types(type_index)%component_names)) then
            if (old_size < size(context%derived_types(type_index)%component_names)) return
        else
            if (allocated(context%derived_types(type_index)%component_names)) &
                deallocate(context%derived_types(type_index)%component_names)
            if (allocated(context%derived_types(type_index)%component_has_default)) &
                deallocate(context%derived_types(type_index)%component_has_default)
            if (allocated(context%derived_types(type_index)%component_default_value)) &
                deallocate(context%derived_types(type_index)%component_default_value)
            if (allocated(context%derived_types(type_index)%component_array_size)) &
                deallocate(context%derived_types(type_index)%component_array_size)
            if (allocated(context%derived_types(type_index)%component_value_kind)) &
                deallocate(context%derived_types(type_index)%component_value_kind)
            if (allocated(context%derived_types(type_index)%component_type_index)) &
                deallocate(context%derived_types(type_index)%component_type_index)
            if (allocated(context%derived_types(type_index)%component_is_allocatable)) &
                deallocate(context%derived_types(type_index)%component_is_allocatable)
            if (allocated(context%derived_types(type_index)%component_is_alloc_array)) &
                deallocate(context%derived_types(type_index)%component_is_alloc_array)
            if (allocated(context%derived_types(type_index)%component_alloc_rank)) &
                deallocate(context%derived_types(type_index)%component_alloc_rank)
            if (allocated(context%derived_types(type_index)%component_is_pointer)) &
                deallocate(context%derived_types(type_index)%component_is_pointer)
            if (allocated(context%derived_types(type_index)%component_char_length)) &
                deallocate(context%derived_types(type_index)%component_char_length)
            if (allocated(context%derived_types(type_index)%component_dim1)) &
                deallocate(context%derived_types(type_index)%component_dim1)
            allocate(context%derived_types(type_index)%component_names(8))
            allocate(context%derived_types(type_index)%component_has_default(8))
            allocate(context%derived_types(type_index)%component_default_value(8))
            allocate(context%derived_types(type_index)%component_array_size(8))
            allocate(context%derived_types(type_index)%component_value_kind(8))
            allocate(context%derived_types(type_index)%component_type_index(8))
            allocate(context%derived_types(type_index)%component_is_allocatable(8))
            allocate(context%derived_types(type_index)%component_is_alloc_array(8))
            allocate(context%derived_types(type_index)%component_alloc_rank(8))
            allocate(context%derived_types(type_index)%component_is_pointer(8))
            allocate(context%derived_types(type_index)%component_char_length(8))
            allocate(context%derived_types(type_index)%component_dim1(8))
            context%derived_types(type_index)%component_has_default = .false.
            context%derived_types(type_index)%component_default_value = 0_c_int64_t
            context%derived_types(type_index)%component_array_size = 1
            context%derived_types(type_index)%component_value_kind = VALUE_I32
            context%derived_types(type_index)%component_type_index = 0
            context%derived_types(type_index)%component_is_allocatable = .false.
            context%derived_types(type_index)%component_is_alloc_array = .false.
            context%derived_types(type_index)%component_alloc_rank = 0
            context%derived_types(type_index)%component_is_pointer = .false.
            context%derived_types(type_index)%component_char_length = 0
            context%derived_types(type_index)%component_dim1 = 0
            context%derived_types(type_index)%component_count = 0
            return
        end if

        old_names = context%derived_types(type_index)%component_names
        new_size = 2 * size(context%derived_types(type_index)%component_names)
        deallocate(context%derived_types(type_index)%component_names)
        allocate(context%derived_types(type_index)%component_names(new_size))
        context%derived_types(type_index)%component_names(1:old_size) = old_names
        deallocate(old_names)
        block
            logical, allocatable :: old_hd(:)
            integer(c_int64_t), allocatable :: old_dv(:)
            integer, allocatable :: old_as(:)
            integer, allocatable :: old_vk(:)
            integer, allocatable :: old_ti(:)
            logical, allocatable :: old_ia(:)
            logical, allocatable :: old_iaa(:)
            integer, allocatable :: old_ar(:)
            logical, allocatable :: old_ip(:)
            integer, allocatable :: old_cl(:)
            integer, allocatable :: old_d1(:)
            associate (dt => context%derived_types(type_index))
                old_hd = dt%component_has_default
                old_dv = dt%component_default_value
                old_as = dt%component_array_size
                old_vk = dt%component_value_kind
                old_ti = dt%component_type_index
                old_ia = dt%component_is_allocatable
                old_iaa = dt%component_is_alloc_array
                old_ar = dt%component_alloc_rank
                old_ip = dt%component_is_pointer
                old_cl = dt%component_char_length
                old_d1 = dt%component_dim1
                deallocate(dt%component_has_default)
                deallocate(dt%component_default_value)
                deallocate(dt%component_array_size)
                deallocate(dt%component_value_kind)
                deallocate(dt%component_type_index)
                deallocate(dt%component_is_allocatable)
                deallocate(dt%component_is_alloc_array)
                deallocate(dt%component_alloc_rank)
                deallocate(dt%component_is_pointer)
                deallocate(dt%component_char_length)
                deallocate(dt%component_dim1)
                allocate(dt%component_has_default(new_size))
                allocate(dt%component_default_value(new_size))
                allocate(dt%component_array_size(new_size))
                allocate(dt%component_value_kind(new_size))
                allocate(dt%component_type_index(new_size))
                allocate(dt%component_is_allocatable(new_size))
                allocate(dt%component_is_alloc_array(new_size))
                allocate(dt%component_alloc_rank(new_size))
                allocate(dt%component_is_pointer(new_size))
                allocate(dt%component_char_length(new_size))
                allocate(dt%component_dim1(new_size))
                dt%component_has_default = .false.
                dt%component_default_value = 0_c_int64_t
                dt%component_array_size = 1
                dt%component_value_kind = VALUE_I32
                dt%component_type_index = 0
                dt%component_is_allocatable = .false.
                dt%component_is_alloc_array = .false.
                dt%component_alloc_rank = 0
                dt%component_is_pointer = .false.
                dt%component_char_length = 0
                dt%component_dim1 = 0
                dt%component_has_default(1:old_size) = old_hd
                dt%component_default_value(1:old_size) = old_dv
                dt%component_array_size(1:old_size) = old_as
                dt%component_value_kind(1:old_size) = old_vk
                dt%component_type_index(1:old_size) = old_ti
                dt%component_is_allocatable(1:old_size) = old_ia
                dt%component_is_alloc_array(1:old_size) = old_iaa
                dt%component_alloc_rank(1:old_size) = old_ar
                dt%component_is_pointer(1:old_size) = old_ip
                dt%component_char_length(1:old_size) = old_cl
                dt%component_dim1(1:old_size) = old_d1
            end associate
        end block
    end subroutine grow_derived_type_components

    subroutine grow_module_export_derived_indices(context, export_idx)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: export_idx
        integer, allocatable :: old_indices(:)
        integer :: new_size, old_size

        old_size = context%module_exports(export_idx)%derived_type_count
        if (old_size > 0 .and. allocated(context%module_exports(export_idx)%derived_type_indices)) then
            if (old_size < size(context%module_exports(export_idx)%derived_type_indices)) return
        else
            return
        end if

        old_indices = context%module_exports(export_idx)%derived_type_indices
        new_size = 2 * size(context%module_exports(export_idx)%derived_type_indices)
        deallocate(context%module_exports(export_idx)%derived_type_indices)
        allocate(context%module_exports(export_idx)%derived_type_indices(new_size))
        context%module_exports(export_idx)%derived_type_indices(1:old_size) = old_indices
        deallocate(old_indices)
    end subroutine grow_module_export_derived_indices

    subroutine grow_module_export_parameter_indices(context, export_idx)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: export_idx
        integer, allocatable :: old_indices(:)
        integer :: new_size, old_size

        old_size = context%module_exports(export_idx)%parameter_count
        if (old_size > 0 .and. &
            allocated(context%module_exports(export_idx)%parameter_indices)) then
            if (old_size < &
                size(context%module_exports(export_idx)%parameter_indices)) return
        else
            return
        end if

        old_indices = context%module_exports(export_idx)%parameter_indices
        new_size = 2 * size(context%module_exports(export_idx)%parameter_indices)
        deallocate(context%module_exports(export_idx)%parameter_indices)
        allocate(context%module_exports(export_idx)%parameter_indices(new_size))
        context%module_exports(export_idx)%parameter_indices(1:old_size) = old_indices
        deallocate(old_indices)
    end subroutine grow_module_export_parameter_indices

    subroutine grow_module_export_variable_indices(context, export_idx)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: export_idx
        integer, allocatable :: old_indices(:)
        integer :: new_size, old_size

        old_size = context%module_exports(export_idx)%variable_count
        if (old_size > 0 .and. &
            allocated(context%module_exports(export_idx)%variable_indices)) then
            if (old_size < &
                size(context%module_exports(export_idx)%variable_indices)) return
        else
            return
        end if

        old_indices = context%module_exports(export_idx)%variable_indices
        new_size = 2 * size(context%module_exports(export_idx)%variable_indices)
        deallocate(context%module_exports(export_idx)%variable_indices)
        allocate(context%module_exports(export_idx)%variable_indices(new_size))
        context%module_exports(export_idx)%variable_indices(1:old_size) = old_indices
        deallocate(old_indices)
    end subroutine grow_module_export_variable_indices

    subroutine grow_module_export_enum_indices(context, export_idx)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: export_idx
        integer, allocatable :: old_indices(:)
        integer :: new_size, old_size

        old_size = context%module_exports(export_idx)%enum_count
        if (old_size > 0 .and. &
            allocated(context%module_exports(export_idx)%enum_indices)) then
            if (old_size < &
                size(context%module_exports(export_idx)%enum_indices)) return
        else
            return
        end if

        old_indices = context%module_exports(export_idx)%enum_indices
        new_size = 2 * size(context%module_exports(export_idx)%enum_indices)
        deallocate(context%module_exports(export_idx)%enum_indices)
        allocate(context%module_exports(export_idx)%enum_indices(new_size))
        context%module_exports(export_idx)%enum_indices(1:old_size) = old_indices
        deallocate(old_indices)
    end subroutine grow_module_export_enum_indices

    ! ---------------------------------------------------------------------
    ! Derived-type definition constraints checked on the source form (#390).
    !
    ! FortFront folds a contained procedure of a module and the component list
    ! of a module-scope type away before the arena is handed to ffc, so the
    ! typed nodes no longer carry the attributes these rules need. The source
    ! form is therefore the earliest - and here the only - layer with enough
    ! information:
    !
    !   * a component-less BIND(C) derived type is accepted: gfortran only
    !     warns that it may be inaccessible by the C companion processor
    !     (lazy-fortran/ffc#581);
    !   * an INTENT(OUT) dummy argument of a type with default initialization
    !     may not be an assumed-size array, because the default initializer
    !     would have to be applied to an unknown number of elements
    !     (gfortran: "is ASSUMED SIZE and so cannot have a default
    !     initializer");
    !   * a SELECT TYPE guard is spelled TYPE IS; TYPEIS is not a keyword and
    !     is parsed as a malformed type definition
    !     (gfortran: "Mangled derived type definition").
    ! ---------------------------------------------------------------------
    subroutine check_derived_type_definition_forms(arena, error_msg)
        type(ast_arena_t), intent(in) :: arena
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: source, init_types
        logical :: found

        call set_empty(error_msg)
        call get_source_text(arena, source, found)
        if (.not. found) return
        call scan_derived_type_bodies(source, init_types, error_msg)
        if (len_trim(error_msg) > 0) return
        call check_assumed_size_dt_dummies(source, init_types, error_msg)
    end subroutine check_derived_type_definition_forms

    subroutine scan_derived_type_bodies(source, init_types, error_msg)
        !! Walk every derived-type definition in the source. Reports the
        !! mangled TYPEIS guard, and returns the
        !! '|'-delimited set of type names that have a default-initialized
        !! component.
        character(len=*), intent(in) :: source
        character(len=:), allocatable, intent(out) :: init_types
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: line, code, packed, name, type_name
        logical :: in_body, is_header
        integer :: pos, line_no
        character(len=64) :: location

        call set_empty(error_msg)
        init_types = '|'
        in_body = .false.
        type_name = ''
        pos = 1
        line_no = 0
        do while (pos <= len(source))
            call next_source_line(source, pos, line)
            line_no = line_no + 1
            code = statement_code_text(line)
            if (len_trim(code) == 0) cycle
            packed = remove_blanks(code)
            if (in_body) then
                if (len(packed) >= 7) then
                    if (packed(1:7) == 'endtype') then
                        in_body = .false.
                        cycle
                    end if
                end if
                if (is_type_body_keyword(packed)) cycle
                if (has_default_initializer(code)) then
                    if (index(init_types, '|'//trim(type_name)//'|') == 0) then
                        init_types = init_types//trim(type_name)//'|'
                    end if
                end if
                cycle
            end if
            if (first_word_of(code) == 'typeis') then
                write (location, '(" at line ",I0)') line_no
                error_msg = 'mangled derived type definition'//trim(location)
                return
            end if
            call derived_type_header_info(code, name, is_header)
            if (.not. is_header) cycle
            in_body = .true.
            type_name = name
        end do
    end subroutine scan_derived_type_bodies

    subroutine check_assumed_size_dt_dummies(source, init_types, error_msg)
        character(len=*), intent(in) :: source
        character(len=*), intent(in) :: init_types
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: line, code, packed, type_name, dummy
        integer :: pos, line_no, close_paren
        character(len=64) :: location

        call set_empty(error_msg)
        pos = 1
        line_no = 0
        do while (pos <= len(source))
            call next_source_line(source, pos, line)
            line_no = line_no + 1
            code = statement_code_text(line)
            if (len_trim(code) == 0) cycle
            packed = remove_blanks(code)
            if (len(packed) < 6) cycle
            if (packed(1:5) /= 'type(') cycle
            if (index(packed, 'intent(out)') == 0) cycle
            if (index(packed, '(*)') == 0) cycle
            close_paren = index(packed, ')')
            if (close_paren <= 6) cycle
            type_name = packed(6:close_paren - 1)
            if (index(init_types, '|'//trim(type_name)//'|') == 0) cycle
            dummy = declared_entity_name(packed)
            write (location, '(" at line ",I0)') line_no
            error_msg = 'INTENT(OUT) dummy argument '''//trim(dummy)// &
                ''' is ASSUMED SIZE and so cannot have a default initializer'// &
                trim(location)
            return
        end do
    end subroutine check_assumed_size_dt_dummies

    subroutine derived_type_header_info(code, name, is_header)
        !! Decide whether a lowered statement opens a derived-type definition,
        !! and if so return its name.
        character(len=*), intent(in) :: code
        character(len=:), allocatable, intent(out) :: name
        logical, intent(out) :: is_header
        character(len=:), allocatable :: rest
        integer :: dc

        call set_empty(name)
        is_header = .false.
        if (len(code) < 6) return
        if (code(1:4) /= 'type') return
        if (code(5:5) /= ' ' .and. code(5:5) /= ',' .and. code(5:5) /= ':') return
        rest = adjustl(code(5:))
        if (len_trim(rest) == 0) return
        dc = index(rest, '::')
        if (dc > 0) then
            name = first_word_of(adjustl(rest(dc + 2:)))
        else
            if (rest(1:1) == ',') return
            name = first_word_of(rest)
        end if
        if (len_trim(name) == 0) return
        if (trim(name) == 'is') return
        is_header = .true.
    end subroutine derived_type_header_info

    logical function is_type_body_keyword(packed) result(is_keyword)
        character(len=*), intent(in) :: packed

        select case (trim(packed))
        case ('private', 'public', 'sequence', 'contains')
            is_keyword = .true.
        case default
            is_keyword = .false.
        end select
    end function is_type_body_keyword

    logical function has_default_initializer(code) result(has_init)
        !! True when a component declaration carries '= value' (but not the
        !! pointer initializer '=>' nor a relational operator).
        character(len=*), intent(in) :: code
        integer :: i

        has_init = .false.
        do i = 1, len(code)
            if (code(i:i) /= '=') cycle
            if (i > 1) then
                if (scan(code(i - 1:i - 1), '=/<>') > 0) cycle
            end if
            if (i < len(code)) then
                if (scan(code(i + 1:i + 1), '=>') > 0) cycle
            end if
            has_init = .true.
            return
        end do
    end function has_default_initializer

    function declared_entity_name(packed) result(name)
        !! First entity name declared after '::' in a packed declaration.
        character(len=*), intent(in) :: packed
        character(len=:), allocatable :: name, rest
        integer :: dc, stop_at

        name = ''
        dc = index(packed, '::')
        if (dc == 0) return
        rest = packed(dc + 2:)
        if (len_trim(rest) == 0) return
        stop_at = scan(rest, ',(')
        if (stop_at == 0) then
            name = trim(rest)
        else
            name = trim(rest(:stop_at - 1))
        end if
    end function declared_entity_name

    subroutine next_source_line(source, pos, line)
        character(len=*), intent(in) :: source
        integer, intent(inout) :: pos
        character(len=:), allocatable, intent(out) :: line
        integer :: next_nl

        next_nl = index(source(pos:), new_line('a'))
        if (next_nl == 0) then
            line = source(pos:)
            pos = len(source) + 1
        else
            line = source(pos:pos + next_nl - 2)
            pos = pos + next_nl
        end if
    end subroutine next_source_line

    function statement_code_text(line) result(code)
        !! Lowered, comment-stripped, left-justified statement text.
        character(len=*), intent(in) :: line
        character(len=:), allocatable :: code
        integer :: comment_pos

        comment_pos = index(line, '!')
        if (comment_pos == 1) then
            code = ''
            return
        else if (comment_pos > 1) then
            code = lowercase_text(line(:comment_pos - 1))
        else
            code = lowercase_text(line)
        end if
        code = trim(adjustl(code))
    end function statement_code_text

    function remove_blanks(text) result(packed)
        character(len=*), intent(in) :: text
        character(len=:), allocatable :: packed
        integer :: i

        packed = ''
        do i = 1, len(text)
            if (text(i:i) == ' ' .or. text(i:i) == char(9)) cycle
            packed = packed//text(i:i)
        end do
    end function remove_blanks

    function first_word_of(text) result(word)
        character(len=*), intent(in) :: text
        character(len=:), allocatable :: word
        character(len=:), allocatable :: trimmed
        integer :: stop_at

        trimmed = adjustl(text)
        stop_at = scan(trimmed, ' '//char(9)//',(:')
        if (stop_at == 0) then
            word = trim(trimmed)
        else
            word = trim(trimmed(:stop_at - 1))
        end if
    end function first_word_of
