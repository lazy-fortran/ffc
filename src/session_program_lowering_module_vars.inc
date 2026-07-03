    ! Module-level variable storage (#263). A scalar module variable becomes a
    ! mutable LIRIC global named __<module>_MOD_<var> (gfortran convention),
    ! carrying its declared initializer. Module procedures bind the same global
    ! as an addressed reference symbol; a using program imports it via
    ! import_module_variables. Supported kinds are the scalar kinds whose
    ! addressed load/store paths are complete: integer(4), real(4), real(8),
    ! and logical. Arrays, multi-declarations, and other kinds stay xfail with
    ! a clean diagnostic.

    subroutine module_variable_kind(arena, var_idx, value_kind, status, context, &
                                    error_msg)
        ! Classify a module-level declaration. status is MODVAR_OK for a scalar
        ! variable we can lower (value_kind set), otherwise MODVAR_UNSUPPORTED
        ! with a diagnostic so the unit stays xfail rather than miscompiling.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: var_idx
        integer, intent(out) :: value_kind
        integer, intent(out) :: status
        type(lowering_context_t), intent(in) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: type_name, var_name, kind_err
        integer :: line, column

        value_kind = VALUE_I32
        status = MODVAR_UNSUPPORTED
        call set_empty(error_msg)
        line = get_node_line(arena, var_idx)
        column = get_node_column(arena, var_idx)
        if (.not. is_declaration_node(arena, var_idx)) then
            error_msg = 'module export is not a declaration'
            return
        end if
        if (module_declaration_is_array(arena, var_idx) .or. &
            module_declaration_is_multi(arena, var_idx)) then
            call unsupported_feature_error('module variable', line, column, &
                'direct LIRIC session stores only scalar single-name module '// &
                'variables', error_msg)
            return
        end if
        ! A scalar derived-type module variable stores as a flat slot global,
        ! provided its type is a registered non-parameterised derived type and
        ! it carries no runtime initializer (component defaults fold statically).
        if (module_var_derived_type_index(arena, var_idx, context) > 0) then
            if (module_declaration_has_initializer(arena, var_idx)) then
                call unsupported_feature_error('module variable', line, column, &
                    'derived module variables with an explicit initializer are '// &
                    'not supported', error_msg)
                return
            end if
            value_kind = VALUE_DERIVED
            status = MODVAR_OK
            return
        end if
        ! PROTECTED restricts write access from using units; the frontend
        ! enforces that at semantic analysis. ffc emits ordinary storage so the
        ! defining module's own procedures (the only legitimate writers) work
        ! (#274).
        call get_declaration_type_name(arena, var_idx, type_name, kind_err)
        if (len_trim(kind_err) > 0) then
            error_msg = kind_err
            return
        end if
        call type_name_value_kind(type_name, line, column, value_kind, kind_err)
        if (len_trim(kind_err) > 0) then
            error_msg = kind_err
            return
        end if
        select case (value_kind)
        case (VALUE_I32, VALUE_F32, VALUE_F64, VALUE_LOGICAL)
            status = MODVAR_OK
        case default
            call get_declaration_var_name(arena, var_idx, var_name, kind_err)
            if (len_trim(kind_err) > 0) var_name = '<unknown>'
            call unsupported_feature_error('module variable', line, column, &
                'direct LIRIC session stores only integer(4), real, and '// &
                'logical scalar module variables; '//trim(var_name)// &
                ' has an unsupported kind', error_msg)
        end select
    end subroutine module_variable_kind

    logical function module_declaration_is_array(arena, node_index)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        module_declaration_is_array = .false.
        if (.not. node_exists(arena, node_index)) return
        select type (decl => arena%entries(node_index)%node)
        type is (declaration_node)
            module_declaration_is_array = decl%is_array
        end select
    end function module_declaration_is_array

    logical function module_declaration_is_multi(arena, node_index)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        module_declaration_is_multi = .false.
        if (.not. node_exists(arena, node_index)) return
        select type (decl => arena%entries(node_index)%node)
        type is (declaration_node)
            module_declaration_is_multi = decl%is_multi_declaration
        end select
    end function module_declaration_is_multi

    logical function module_declaration_has_initializer(arena, node_index)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        module_declaration_has_initializer = .false.
        if (.not. node_exists(arena, node_index)) return
        select type (decl => arena%entries(node_index)%node)
        type is (declaration_node)
            module_declaration_has_initializer = decl%has_initializer
        end select
    end function module_declaration_has_initializer

    integer function module_var_derived_type_index(arena, node_index, context)
        ! Registered derived-type index for a module-variable declaration, or 0
        ! when the declaration does not name a known derived type.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        module_var_derived_type_index = 0
        if (.not. node_exists(arena, node_index)) return
        select type (decl => arena%entries(node_index)%node)
        type is (declaration_node)
            module_var_derived_type_index = &
                declaration_derived_type_index(context, decl)
        end select
    end function module_var_derived_type_index

    subroutine lower_module_variables(arena, root_index, context, error_msg)
        ! Emit a global with its declared initializer for each scalar module
        ! variable. An unsupported module variable leaves error_msg set so the
        ! unit stays xfail (lower_module_unit then no longer re-diagnoses it).
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: root_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: m, i, var_idx, value_kind, status

        call set_empty(error_msg)
        do m = 1, context%module_export_count
            do i = 1, context%module_exports(m)%variable_count
                var_idx = context%module_exports(m)%variable_indices(i)
                if (.not. node_exists(arena, var_idx)) cycle
                call module_variable_kind(arena, var_idx, value_kind, status, &
                                          context, error_msg)
                if (status /= MODVAR_OK) return
                call emit_module_variable_global(arena, var_idx, value_kind, &
                    trim(context%module_exports(m)%module_name), context, error_msg)
                if (len_trim(error_msg) > 0) return
            end do
        end do
    end subroutine lower_module_variables

    subroutine emit_module_variable_global(arena, var_idx, value_kind, &
                                           module_name, context, error_msg)
        use, intrinsic :: iso_c_binding, only: c_bool, c_char, c_int32_t, &
                                               c_loc, c_size_t, c_ptr, c_double, &
                                               c_float, c_int32_t
        use liric_session_bindings, only: lr_type_i32_s, lr_type_f32_s, &
                                          lr_type_f64_s
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: var_idx
        integer, intent(in) :: value_kind
        character(len=*), intent(in) :: module_name
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: var_name, mangled
        character(kind=c_char), allocatable :: c_name(:)
        integer(c_int32_t), target :: init_i32
        real(c_float), target :: init_f32
        real(c_double), target :: init_f64
        type(c_ptr) :: gtype, init_ptr
        integer(c_size_t) :: init_size
        integer(c_int32_t) :: global_id
        logical(c_bool) :: c_false_local

        c_false_local = .false.
        if (value_kind == VALUE_DERIVED) then
            call emit_derived_module_variable_global(arena, var_idx, module_name, &
                                                     context, error_msg)
            return
        end if
        call get_declaration_var_name(arena, var_idx, var_name, error_msg)
        if (len_trim(error_msg) > 0) return
        mangled = module_procedure_mangled(module_name, trim(var_name))
        call to_c_chars(mangled, c_name)
        call fold_module_variable_initializer(arena, var_idx, value_kind, &
            init_i32, init_f32, init_f64, error_msg)
        if (len_trim(error_msg) > 0) return
        select case (value_kind)
        case (VALUE_I32, VALUE_LOGICAL)
            gtype = lr_type_i32_s(context%session%handle)
            init_ptr = c_loc(init_i32)
            init_size = 4_c_size_t
        case (VALUE_F32)
            gtype = lr_type_f32_s(context%session%handle)
            init_ptr = c_loc(init_f32)
            init_size = 4_c_size_t
        case (VALUE_F64)
            gtype = lr_type_f64_s(context%session%handle)
            init_ptr = c_loc(init_f64)
            init_size = 8_c_size_t
        case default
            error_msg = 'unsupported module variable kind for global emission'
            return
        end select
        global_id = lr_session_global(context%session%handle, c_name, gtype, &
                                      c_false_local, init_ptr, init_size)
        if (global_id < 0_c_int32_t) then
            error_msg = 'LIRIC could not create module variable global: '// &
                        trim(mangled)
            return
        end if
        call set_empty(error_msg)
    end subroutine emit_module_variable_global

    subroutine emit_derived_module_variable_global(arena, var_idx, module_name, &
                                                   context, error_msg)
        ! Emit a scalar derived-type module variable as a flat [slots x i32]
        ! global whose static bytes carry the type's compile-time component
        ! defaults (zero where a component has none). Component access in using
        ! units GEPs into this global, exactly as for a local derived instance.
        use, intrinsic :: iso_c_binding, only: c_bool, c_char, c_int32_t, &
                                               c_int64_t, c_loc, c_size_t, c_ptr
        use liric_session_bindings, only: lr_type_i32_s, lr_type_array_s
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: var_idx
        character(len=*), intent(in) :: module_name
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: var_name, mangled
        character(kind=c_char), allocatable :: c_name(:)
        integer(c_int32_t), allocatable, target :: init_slots(:)
        type(c_ptr) :: gtype
        integer(c_int32_t) :: global_id
        integer :: di, slots, c, start_slot
        logical(c_bool) :: c_false_local

        c_false_local = .false.
        di = module_var_derived_type_index(arena, var_idx, context)
        if (di <= 0) then
            error_msg = 'derived module variable lost its type index'
            return
        end if
        call get_declaration_var_name(arena, var_idx, var_name, error_msg)
        if (len_trim(error_msg) > 0) return
        slots = total_component_slots(context, di)
        if (slots <= 0) then
            error_msg = 'derived module variable has no storage: '//trim(var_name)
            return
        end if
        allocate (init_slots(slots))
        init_slots = 0_c_int32_t
        if (allocated(context%derived_types(di)%component_has_default)) then
            do c = 1, context%derived_types(di)%component_count
                if (.not. context%derived_types(di)%component_has_default(c)) cycle
                start_slot = component_start_slot(context, di, c)
                init_slots(start_slot + 1) = int( &
                    context%derived_types(di)%component_default_value(c), c_int32_t)
            end do
        end if
        mangled = module_procedure_mangled(module_name, trim(var_name))
        call to_c_chars(mangled, c_name)
        gtype = lr_type_array_s(context%session%handle, &
                                lr_type_i32_s(context%session%handle), &
                                int(slots, c_int64_t))
        global_id = lr_session_global(context%session%handle, c_name, gtype, &
                                      c_false_local, c_loc(init_slots), &
                                      int(slots * 4, c_size_t))
        if (global_id < 0_c_int32_t) then
            error_msg = 'LIRIC could not create derived module variable global: '// &
                        trim(mangled)
            return
        end if
        call set_empty(error_msg)
    end subroutine emit_derived_module_variable_global

    subroutine fold_module_variable_initializer(arena, var_idx, value_kind, &
                                                init_i32, init_f32, init_f64, &
                                                error_msg)
        ! Fold a declared scalar initializer literal into raw initial bytes.
        ! An absent initializer is zero. Only literal initializers are folded;
        ! a non-literal initializer keeps the unit xfail.
        use, intrinsic :: iso_c_binding, only: c_int32_t, c_int64_t, c_float, &
                                               c_double
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: var_idx
        integer, intent(in) :: value_kind
        integer(c_int32_t), intent(out) :: init_i32
        real(c_float), intent(out) :: init_f32
        real(c_double), intent(out) :: init_f64
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: lit_value, lit_type, lit_err
        integer(c_int64_t) :: iconst
        integer :: init_index, io_stat

        init_i32 = 0_c_int32_t
        init_f32 = 0.0_c_float
        init_f64 = 0.0_c_double
        call set_empty(error_msg)
        if (.not. get_declaration_has_initializer(arena, var_idx)) return
        init_index = get_declaration_initializer_index(arena, var_idx)
        if (init_index <= 0) return
        if (.not. is_literal(arena, init_index)) then
            call unsupported_feature_error('module variable initializer', &
                get_node_line(arena, var_idx), get_node_column(arena, var_idx), &
                'only literal initializers are stored for module variables', &
                error_msg)
            return
        end if
        call get_literal_info(arena, init_index, lit_value, lit_type, lit_err)
        if (len_trim(lit_err) > 0) then
            error_msg = lit_err
            return
        end if
        select case (value_kind)
        case (VALUE_I32)
            call parse_i32_literal(lit_value, iconst, error_msg)
            if (len_trim(error_msg) > 0) return
            init_i32 = int(iconst, c_int32_t)
        case (VALUE_LOGICAL)
            init_i32 = logical_literal_to_i32(lit_value)
        case (VALUE_F32)
            read (lit_value, *, iostat=io_stat) init_f32
            if (io_stat /= 0) error_msg = 'malformed real initializer: '//lit_value
        case (VALUE_F64)
            read (lit_value, *, iostat=io_stat) init_f64
            if (io_stat /= 0) error_msg = 'malformed real initializer: '//lit_value
        end select
    end subroutine fold_module_variable_initializer

    integer(c_int32_t) function logical_literal_to_i32(lit_value) result(v)
        use, intrinsic :: iso_c_binding, only: c_int32_t
        character(len=*), intent(in) :: lit_value
        character(len=:), allocatable :: lowered
        lowered = lowercase_text(trim(adjustl(lit_value)))
        if (index(lowered, 't') > 0) then
            v = 1_c_int32_t
        else
            v = 0_c_int32_t
        end if
    end function logical_literal_to_i32

    subroutine bind_module_variables_in_context(context, error_msg)
        ! Bind every module variable as an addressed reference symbol in a fresh
        ! procedure context so its body can read and write the shared global.
        ! Called after the module's globals have been emitted (#263).
        use, intrinsic :: iso_c_binding, only: c_char, c_int32_t, c_int64_t
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(ast_arena_t) :: arena
        integer :: m, i, var_idx, value_kind, status

        call set_empty(error_msg)
        arena = context%arena
        do m = 1, context%module_export_count
            do i = 1, context%module_exports(m)%variable_count
                var_idx = context%module_exports(m)%variable_indices(i)
                if (.not. node_exists(arena, var_idx)) cycle
                call module_variable_kind(arena, var_idx, value_kind, status, &
                                          context, error_msg)
                if (status /= MODVAR_OK) then
                    ! A module that also declares an unsupported variable already
                    ! failed in lower_module_variables; do not re-diagnose here.
                    call set_empty(error_msg)
                    cycle
                end if
                call bind_one_module_variable(arena, var_idx, value_kind, &
                    trim(context%module_exports(m)%module_name), context, error_msg)
                if (len_trim(error_msg) > 0) return
            end do
        end do
    end subroutine bind_module_variables_in_context

    subroutine bind_one_module_variable(arena, var_idx, value_kind, module_name, &
                                        context, error_msg)
        use, intrinsic :: iso_c_binding, only: c_char, c_int32_t, c_int64_t
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: var_idx
        integer, intent(in) :: value_kind
        character(len=*), intent(in) :: module_name
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: var_name, mangled
        character(kind=c_char), allocatable :: c_name(:)
        integer(c_int32_t) :: global_id
        integer :: sym_idx

        call get_declaration_var_name(arena, var_idx, var_name, error_msg)
        if (len_trim(error_msg) > 0) return
        if (find_symbol(context, var_name) > 0) return
        mangled = module_procedure_mangled(module_name, trim(var_name))
        call to_c_chars(mangled, c_name)
        global_id = lr_session_intern(context%session%handle, c_name)
        if (global_id < 0_c_int32_t) then
            error_msg = 'module variable global not found: '//trim(mangled)
            return
        end if
        if (value_kind == VALUE_DERIVED) then
            call bind_one_derived_module_variable(arena, var_idx, trim(var_name), &
                                                  global_id, context, error_msg)
            return
        end if
        call grow_symbols(context)
        sym_idx = context%symbol_count + 1
        context%symbols(sym_idx)%name = trim(var_name)
        context%symbols(sym_idx)%value_kind = value_kind
        context%symbols(sym_idx)%has_address = .true.
        context%symbols(sym_idx)%is_reference = .true.
        context%symbols(sym_idx)%address%kind = LR_OP_KIND_GLOBAL
        context%symbols(sym_idx)%address%payload = int(global_id, c_int64_t)
        context%symbols(sym_idx)%address%typ = lr_type_ptr_s(context%session%handle)
        context%symbols(sym_idx)%address%global_offset = 0_c_int64_t
        context%symbol_count = sym_idx
        call set_empty(error_msg)
    end subroutine bind_one_module_variable

    subroutine bind_one_derived_module_variable(arena, var_idx, sym_name, &
                                                global_id, context, error_msg)
        ! Bind a derived-type module variable as a derived reference symbol whose
        ! flat slot storage is the interned global. Component access GEPs from
        ! element_address, so it points at the global's base pointer. sym_name is
        ! the local binding name (a USE rename may differ from the module name).
        use, intrinsic :: iso_c_binding, only: c_int32_t, c_int64_t
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: var_idx
        character(len=*), intent(in) :: sym_name
        integer(c_int32_t), intent(in) :: global_id
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: di, sym_idx

        di = module_var_derived_type_index(arena, var_idx, context)
        if (di <= 0) then
            error_msg = 'derived module variable lost its type index: '// &
                        trim(sym_name)
            return
        end if
        call grow_symbols(context)
        sym_idx = context%symbol_count + 1
        context%symbols(sym_idx)%name = trim(sym_name)
        context%symbols(sym_idx)%value_kind = VALUE_DERIVED
        context%symbols(sym_idx)%is_derived = .true.
        context%symbols(sym_idx)%derived_type_index = di
        context%symbols(sym_idx)%array_size = total_component_slots(context, di)
        context%symbols(sym_idx)%element_address%kind = LR_OP_KIND_GLOBAL
        context%symbols(sym_idx)%element_address%payload = int(global_id, c_int64_t)
        context%symbols(sym_idx)%element_address%typ = &
            lr_type_ptr_s(context%session%handle)
        context%symbols(sym_idx)%element_address%global_offset = 0_c_int64_t
        context%symbol_count = sym_idx
        call set_empty(error_msg)
    end subroutine bind_one_derived_module_variable

    subroutine bind_module_constants_in_context(context, error_msg)
        ! Host-associate module-scope named constants into a fresh procedure
        ! context so a contained or module procedure body can reference them
        ! (#1826). Scalar integer/real/logical/character parameters and enum
        ! enumerators are folded into value-based symbols, exactly as the main
        ! program scope binds them. Array parameters and non-literal
        ! initializers stay out of scope: the constant is simply left unbound,
        ! so a body reference surfaces the usual clean "not declared" diagnostic.
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(ast_arena_t) :: arena
        integer :: m, i, param_idx, value_kind
        logical :: enum_handled
        character(len=:), allocatable :: kind_err, decl_err

        call set_empty(error_msg)
        arena = context%arena
        do m = 1, context%module_export_count
            do i = 1, context%module_exports(m)%parameter_count
                param_idx = context%module_exports(m)%parameter_indices(i)
                if (.not. node_exists(arena, param_idx)) cycle
                select type (node => arena%entries(param_idx)%node)
                type is (declaration_node)
                    if (find_symbol(context, node%var_name) > 0) cycle
                    if (node%is_array .or. node%is_multi_declaration) cycle
                    call declaration_value_kind(node, value_kind, kind_err)
                    if (len_trim(kind_err) > 0) cycle
                    ! Out-of-scope kinds leave the symbol unbound; do not fail.
                    call lower_constant_declaration(node, context, value_kind, &
                                                    decl_err)
                end select
            end do
            do i = 1, context%module_exports(m)%enum_count
                param_idx = context%module_exports(m)%enum_indices(i)
                if (.not. node_exists(arena, param_idx)) cycle
                call lower_enum_block(arena, param_idx, context, decl_err, &
                                      enum_handled)
                if (len_trim(decl_err) > 0) then
                    error_msg = decl_err
                    return
                end if
            end do
        end do
    end subroutine bind_module_constants_in_context
