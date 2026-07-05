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
        if (module_declaration_is_array(arena, var_idx)) then
            call module_array_variable_kind(arena, var_idx, value_kind, status, &
                                            context, error_msg)
            return
        end if
        ! A scalar derived-type module variable stores as a flat slot global.
        ! Component defaults fold statically; an explicit structure-constructor
        ! initialiser folds too when its provided arguments are compile-time
        ! integer constants for integer components (checked at emit time).
        if (module_var_derived_type_index(arena, var_idx, context) > 0) then
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
        call type_name_value_kind(type_name, line, column, value_kind, &
                                  kind_err, context)
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

    subroutine module_array_variable_kind(arena, var_idx, value_kind, status, &
                                           context, error_msg)
        ! Classify a module-level ARRAY declaration. A fixed-size rank-1 array of
        ! integer(4), real, or logical whose extent folds to a compile-time
        ! constant is MODVAR_OK (value_kind is the element kind). Allocatable,
        ! pointer, character, derived, multidimensional, or runtime-sized arrays
        ! stay MODVAR_UNSUPPORTED with a clean diagnostic, so a case that never
        ! compiled keeps its xfail rather than miscompiling.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: var_idx
        integer, intent(out) :: value_kind
        integer, intent(out) :: status
        type(lowering_context_t), intent(in) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: array_size, array_lower, line, column

        value_kind = VALUE_I32
        status = MODVAR_UNSUPPORTED
        call set_empty(error_msg)
        line = get_node_line(arena, var_idx)
        column = get_node_column(arena, var_idx)
        call module_array_variable_info(arena, var_idx, context, value_kind, &
                                        array_size, array_lower, error_msg)
        if (len_trim(error_msg) > 0) then
            call unsupported_feature_error('module variable', line, column, &
                trim(error_msg), error_msg)
            return
        end if
        status = MODVAR_OK
    end subroutine module_array_variable_kind

    subroutine module_array_variable_info(arena, var_idx, context, value_kind, &
                                          array_size, array_lower, error_msg)
        ! Resolve the element kind, total extent, and lower bound of a supported
        ! fixed-size module array. error_msg names the reason a form is out of
        ! scope; callers turn that into an xfail diagnostic.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: var_idx
        type(lowering_context_t), intent(in) :: context
        integer, intent(out) :: value_kind
        integer, intent(out) :: array_size
        integer, intent(out) :: array_lower
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: type_name, kind_err
        integer :: line, column

        value_kind = VALUE_I32
        array_size = 0
        array_lower = 1
        call set_empty(error_msg)
        line = get_node_line(arena, var_idx)
        column = get_node_column(arena, var_idx)
        if (.not. node_exists(arena, var_idx)) then
            error_msg = 'module array declaration is missing'
            return
        end if
        select type (decl => arena%entries(var_idx)%node)
        type is (declaration_node)
            if (decl%is_allocatable .or. decl%is_pointer) then
                error_msg = 'allocatable and pointer module arrays are not '// &
                            'stored as static globals'
                return
            end if
            if (.not. allocated(decl%dimension_indices)) then
                error_msg = 'module array has no static dimension'
                return
            end if
            if (size(decl%dimension_indices) /= 1) then
                error_msg = 'only rank-1 module arrays are stored as globals'
                return
            end if
            call get_array_bounds(decl, context, array_lower, array_size, &
                                  error_msg)
            if (len_trim(error_msg) > 0) then
                error_msg = 'module array extent must be a compile-time constant'
                return
            end if
            if (array_size <= 0) then
                error_msg = 'module array has a non-positive extent'
                return
            end if
        class default
            error_msg = 'module export is not a declaration'
            return
        end select
        call get_declaration_type_name(arena, var_idx, type_name, kind_err)
        if (len_trim(kind_err) > 0) then
            error_msg = kind_err
            return
        end if
        call type_name_value_kind(type_name, line, column, value_kind, &
                                  kind_err, context)
        if (len_trim(kind_err) > 0) then
            error_msg = kind_err
            return
        end if
        select case (value_kind)
        case (VALUE_I32, VALUE_F32, VALUE_F64, VALUE_LOGICAL)
        case default
            error_msg = 'direct LIRIC session stores only integer(4), real, '// &
                        'and logical module arrays'
        end select
    end subroutine module_array_variable_info

    subroutine probe_module_array_initializer(arena, var_idx, value_kind, &
                                              array_size, context, error_msg)
        ! A supported-shape module array whose initializer is not a foldable
        ! compile-time constant constructor is still out of scope: fold into
        ! throwaway buffers so the collector can decline it up front, keeping a
        ! unit that only declares such an array compilable.
        use, intrinsic :: iso_c_binding, only: c_int32_t, c_float, c_double
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: var_idx, value_kind, array_size
        type(lowering_context_t), intent(in) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int32_t), allocatable :: probe_i32(:)
        real(c_float), allocatable :: probe_f32(:)
        real(c_double), allocatable :: probe_f64(:)

        call set_empty(error_msg)
        if (array_size <= 0) return
        allocate (probe_i32(array_size), probe_f32(array_size), &
                  probe_f64(array_size))
        probe_i32 = 0_c_int32_t
        probe_f32 = 0.0_c_float
        probe_f64 = 0.0_c_double
        call fold_module_array_initializer(arena, var_idx, value_kind, &
                                           array_size, context, probe_i32, &
                                           probe_f32, probe_f64, error_msg)
    end subroutine probe_module_array_initializer

    logical function module_declaration_is_multi(arena, node_index)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        module_declaration_is_multi = .false.
        if (.not. node_exists(arena, node_index)) return
        select type (decl => arena%entries(node_index)%node)
        type is (declaration_node)
            module_declaration_is_multi = decl%is_multi_declaration
            if (module_declaration_is_multi .and. allocated(decl%var_names)) then
                module_declaration_is_multi = size(decl%var_names) > 1
            end if
        end select
    end function module_declaration_is_multi

    subroutine module_declaration_var_names(arena, node_index, names)
        ! Names declared by a module-variable declaration: the single var_name,
        ! or every entry of a multi-name declaration (integer :: i, j).
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        character(len=64), allocatable, intent(out) :: names(:)
        integer :: k

        if (node_exists(arena, node_index)) then
            select type (decl => arena%entries(node_index)%node)
            type is (declaration_node)
                if (decl%is_multi_declaration .and. allocated(decl%var_names)) then
                    allocate(names(size(decl%var_names)))
                    do k = 1, size(decl%var_names)
                        names(k) = trim(decl%var_names(k))
                    end do
                else if (allocated(decl%var_name)) then
                    allocate(names(1))
                    names(1) = trim(decl%var_name)
                end if
            end select
        end if
        if (.not. allocated(names)) allocate(names(0))
    end subroutine module_declaration_var_names

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
                block
                    character(len=64), allocatable :: names(:)
                    integer :: nm
                    call module_declaration_var_names(arena, var_idx, names)
                    do nm = 1, size(names)
                        call emit_module_variable_global(arena, var_idx, &
                            value_kind, trim(names(nm)), &
                            trim(context%module_exports(m)%module_name), &
                            context, error_msg)
                        if (len_trim(error_msg) > 0) return
                    end do
                end block
            end do
        end do
    end subroutine lower_module_variables

    subroutine emit_module_variable_global(arena, var_idx, value_kind, &
                                           var_name, module_name, context, &
                                           error_msg)
        use, intrinsic :: iso_c_binding, only: c_bool, c_char, c_int32_t, &
                                               c_loc, c_size_t, c_ptr, c_double, &
                                               c_float, c_int32_t
        use liric_session_bindings, only: lr_type_i32_s, lr_type_f32_s, &
                                          lr_type_f64_s
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: var_idx
        integer, intent(in) :: value_kind
        character(len=*), intent(in) :: var_name
        character(len=*), intent(in) :: module_name
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: mangled
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
        if (module_declaration_is_array(arena, var_idx)) then
            call emit_module_array_variable_global(arena, var_idx, value_kind, &
                var_name, module_name, context, error_msg)
            return
        end if
        call set_empty(error_msg)
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

    subroutine emit_module_array_variable_global(arena, var_idx, value_kind, &
                                                 var_name, module_name, context, &
                                                 error_msg)
        ! Emit a fixed-size module array as one [extent x elem] LIRIC global,
        ! zero-filled unless a compile-time array-constructor initialiser folds
        ! into its static bytes. The buffer is local, so lr_session_global (which
        ! copies immediately) runs before it goes out of scope.
        use, intrinsic :: iso_c_binding, only: c_bool, c_char, c_int32_t, &
                                               c_int64_t, c_loc, c_size_t, &
                                               c_ptr, c_double, c_float
        use liric_session_bindings, only: lr_type_i32_s, lr_type_f32_s, &
                                          lr_type_f64_s, lr_type_array_s
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: var_idx
        integer, intent(in) :: value_kind
        character(len=*), intent(in) :: var_name
        character(len=*), intent(in) :: module_name
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: mangled
        character(kind=c_char), allocatable :: c_name(:)
        integer(c_int32_t), target, allocatable :: array_i32(:)
        real(c_float), target, allocatable :: array_f32(:)
        real(c_double), target, allocatable :: array_f64(:)
        type(c_ptr) :: elem_type, gtype, init_ptr
        integer(c_size_t) :: init_size
        integer(c_int32_t) :: global_id
        integer :: value_kind_local, array_size, array_lower
        logical(c_bool) :: c_false_local

        call set_empty(error_msg)
        c_false_local = .false.
        call module_array_variable_info(arena, var_idx, context, value_kind_local, &
                                        array_size, array_lower, error_msg)
        if (len_trim(error_msg) > 0) return
        select case (value_kind)
        case (VALUE_I32, VALUE_LOGICAL)
            allocate (array_i32(array_size))
            array_i32 = 0_c_int32_t
            call fold_module_array_initializer(arena, var_idx, value_kind, &
                array_size, context, array_i32, array_f32, array_f64, error_msg)
            if (len_trim(error_msg) > 0) return
            elem_type = lr_type_i32_s(context%session%handle)
            init_ptr = c_loc(array_i32)
            init_size = int(array_size, c_size_t) * 4_c_size_t
        case (VALUE_F32)
            allocate (array_f32(array_size))
            array_f32 = 0.0_c_float
            call fold_module_array_initializer(arena, var_idx, value_kind, &
                array_size, context, array_i32, array_f32, array_f64, error_msg)
            if (len_trim(error_msg) > 0) return
            elem_type = lr_type_f32_s(context%session%handle)
            init_ptr = c_loc(array_f32)
            init_size = int(array_size, c_size_t) * 4_c_size_t
        case (VALUE_F64)
            allocate (array_f64(array_size))
            array_f64 = 0.0_c_double
            call fold_module_array_initializer(arena, var_idx, value_kind, &
                array_size, context, array_i32, array_f32, array_f64, error_msg)
            if (len_trim(error_msg) > 0) return
            elem_type = lr_type_f64_s(context%session%handle)
            init_ptr = c_loc(array_f64)
            init_size = int(array_size, c_size_t) * 8_c_size_t
        case default
            error_msg = 'unsupported module array kind for global emission'
            return
        end select
        mangled = module_procedure_mangled(module_name, trim(var_name))
        call to_c_chars(mangled, c_name)
        gtype = lr_type_array_s(context%session%handle, elem_type, &
                                int(array_size, c_int64_t))
        global_id = lr_session_global(context%session%handle, c_name, gtype, &
                                      c_false_local, init_ptr, init_size)
        if (global_id < 0_c_int32_t) then
            error_msg = 'LIRIC could not create module array global: '// &
                        trim(mangled)
            return
        end if
        call set_empty(error_msg)
    end subroutine emit_module_array_variable_global

    subroutine fold_module_array_initializer(arena, var_idx, value_kind, &
                                             array_size, context, array_i32, &
                                             array_f32, array_f64, error_msg)
        ! Fold a compile-time array-constructor initialiser into the static
        ! element buffer of the matching kind. An absent initialiser leaves the
        ! zero fill. A non-constructor or non-constant initialiser sets error_msg
        ! so the unit stays xfail rather than storing wrong bytes.
        use, intrinsic :: iso_c_binding, only: c_int32_t, c_int64_t, c_float, &
                                               c_double
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: var_idx
        integer, intent(in) :: value_kind
        integer, intent(in) :: array_size
        type(lowering_context_t), intent(in) :: context
        integer(c_int32_t), intent(inout), allocatable :: array_i32(:)
        real(c_float), intent(inout), allocatable :: array_f32(:)
        real(c_double), intent(inout), allocatable :: array_f64(:)
        character(len=:), allocatable, intent(out) :: error_msg
        integer, allocatable :: flat(:)
        integer :: init_index, i, io_stat, line, column
        integer(c_int64_t) :: iconst
        character(len=:), allocatable :: lit_value, lit_type, lit_err

        call set_empty(error_msg)
        if (.not. get_declaration_has_initializer(arena, var_idx)) return
        init_index = get_declaration_initializer_index(arena, var_idx)
        if (init_index <= 0) return
        line = get_node_line(arena, var_idx)
        column = get_node_column(arena, var_idx)
        allocate (flat(0))
        call module_array_constructor_elements(arena, init_index, flat, error_msg)
        if (len_trim(error_msg) > 0) then
            call unsupported_feature_error('module array initializer', line, &
                column, trim(error_msg), error_msg)
            return
        end if
        if (size(flat) /= array_size) then
            call unsupported_feature_error('module array initializer', line, &
                column, 'array constructor length does not match declared '// &
                'extent', error_msg)
            return
        end if
        do i = 1, array_size
            select case (value_kind)
            case (VALUE_I32)
                call eval_i32_constant(arena, flat(i), context, iconst, error_msg)
            case (VALUE_LOGICAL)
                call get_literal_info(arena, flat(i), lit_value, lit_type, lit_err)
                if (len_trim(lit_err) > 0) error_msg = lit_err
            case (VALUE_F32, VALUE_F64)
                call get_literal_info(arena, flat(i), lit_value, lit_type, lit_err)
                if (len_trim(lit_err) > 0) error_msg = lit_err
            end select
            if (len_trim(error_msg) > 0) then
                call unsupported_feature_error('module array initializer', line, &
                    column, 'only compile-time constant array constructors '// &
                    'initialise module arrays', error_msg)
                return
            end if
            select case (value_kind)
            case (VALUE_I32)
                array_i32(i) = int(iconst, c_int32_t)
            case (VALUE_LOGICAL)
                array_i32(i) = logical_literal_to_i32(lit_value)
            case (VALUE_F32)
                read (lit_value, *, iostat=io_stat) array_f32(i)
                if (io_stat /= 0) error_msg = 'malformed real initializer: '// &
                                              lit_value
            case (VALUE_F64)
                read (lit_value, *, iostat=io_stat) array_f64(i)
                if (io_stat /= 0) error_msg = 'malformed real initializer: '// &
                                              lit_value
            end select
            if (len_trim(error_msg) > 0) return
        end do
    end subroutine fold_module_array_initializer

    subroutine module_array_constructor_elements(arena, init_index, flat, error_msg)
        ! Flatten a plain array-constructor initialiser into ordered leaf element
        ! indices. A non-constructor initialiser (e.g. a scalar broadcast or a
        ! reshape call) is out of scope and reported so the unit stays xfail.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: init_index
        integer, allocatable, intent(inout) :: flat(:)
        character(len=:), allocatable, intent(out) :: error_msg

        call set_empty(error_msg)
        if (.not. node_exists(arena, init_index)) then
            error_msg = 'array constructor initializer is missing'
            return
        end if
        select type (node => arena%entries(init_index)%node)
        type is (array_literal_node)
            if (.not. allocated(node%element_indices)) then
                error_msg = 'empty array constructor'
                return
            end if
            call flatten_constructor_elements(arena, node%element_indices, flat, &
                                              error_msg)
        class default
            error_msg = 'module array initializer must be a plain array '// &
                        'constructor'
        end select
    end subroutine module_array_constructor_elements

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
        integer :: di, slots
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
        call fill_type_default_slots(context, di, 0, init_slots)
        call fold_module_derived_constructor_slots(arena, var_idx, di, context, &
                                                   init_slots, error_msg)
        if (len_trim(error_msg) > 0) return
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

    subroutine fold_module_derived_constructor_slots(arena, var_idx, di, context, &
                                                     init_slots, error_msg)
        ! Fold an explicit structure-constructor initialiser (t(a1, a2, ...))
        ! on a derived module variable into the static slot bytes. Provided
        ! arguments must be compile-time integer constants targeting integer
        ! components; components omitted from the constructor keep the default
        ! already staged in init_slots. Any other form (real/logical/nested
        ! argument, overloaded constructor, non-constant value) raises a clean
        ! unsupported diagnostic instead of mis-initialising.
        use, intrinsic :: iso_c_binding, only: c_int32_t, c_int64_t
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: var_idx
        integer, intent(in) :: di
        type(lowering_context_t), intent(in) :: context
        integer(c_int32_t), intent(inout) :: init_slots(:)
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: init_index, n_comp, c, start_slot, line, column
        integer, allocatable :: comp_value_index(:)
        integer(c_int64_t) :: cval

        call set_empty(error_msg)
        init_index = get_declaration_initializer_index(arena, var_idx)
        if (init_index <= 0) return
        line = get_node_line(arena, var_idx)
        column = get_node_column(arena, var_idx)
        if (.not. is_derived_constructor_call(arena, init_index, context, di) &
            .or. constructor_uses_user_routine(context, &
                context%derived_types(di)%name)) then
            call unsupported_feature_error('module variable', line, column, &
                'a derived module variable initialiser must be a plain '// &
                'structure constructor', error_msg)
            return
        end if
        call map_constructor_args(arena, init_index, context, di, comp_value_index)
        if (.not. allocated(comp_value_index)) then
            call unsupported_feature_error('module variable', line, column, &
                'derived module variable constructor arguments do not map to '// &
                'components', error_msg)
            return
        end if
        n_comp = context%derived_types(di)%component_count
        do c = 1, n_comp
            if (comp_value_index(c) <= 0) cycle
            ! Only compile-time integer component values fold into the static
            ! slots; a non-integer explicit argument (character, real, nested)
            ! leaves that component at its staged default, matching the
            ! defaults-only behaviour of a parameter used purely for its
            ! compile-time inquiries (e.g. x%c%len).
            if (.not. constructor_component_storeable(context, di, c)) cycle
            if (component_kind(context, di, c) /= VALUE_I32) cycle
            call eval_i32_constant(arena, comp_value_index(c), context, cval, &
                                   error_msg)
            if (len_trim(error_msg) > 0) return
            start_slot = component_start_slot(context, di, c)
            init_slots(start_slot + 1) = int(cval, c_int32_t)
        end do
    end subroutine fold_module_derived_constructor_slots

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
                block
                    character(len=64), allocatable :: names(:)
                    integer :: nm
                    call module_declaration_var_names(arena, var_idx, names)
                    do nm = 1, size(names)
                        call bind_one_module_variable(arena, var_idx, &
                            value_kind, trim(names(nm)), &
                            trim(context%module_exports(m)%module_name), &
                            context, error_msg)
                        if (len_trim(error_msg) > 0) return
                    end do
                end block
            end do
        end do
    end subroutine bind_module_variables_in_context

    subroutine bind_one_module_variable(arena, var_idx, value_kind, var_name, &
                                        module_name, context, error_msg)
        use, intrinsic :: iso_c_binding, only: c_char, c_int32_t, c_int64_t
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: var_idx
        integer, intent(in) :: value_kind
        character(len=*), intent(in) :: var_name
        character(len=*), intent(in) :: module_name
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: mangled
        character(kind=c_char), allocatable :: c_name(:)
        integer(c_int32_t) :: global_id
        integer :: sym_idx

        call set_empty(error_msg)
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
        if (module_declaration_is_array(arena, var_idx)) then
            call bind_one_module_array_variable(arena, var_idx, value_kind, &
                trim(var_name), global_id, context, error_msg)
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

    subroutine bind_one_module_array_variable(arena, var_idx, value_kind, &
                                              sym_name, global_id, context, &
                                              error_msg)
        ! Bind a fixed-size module array as an addressed reference whose storage
        ! is the interned [extent x elem] global. element_address mirrors address
        ! so the shared element-GEP path (session_program_lowering_array_elements)
        ! reads and writes the global, and array_size/array_dim_* carry the shape
        ! that size and bounds inquiries need. sym_name is the local binding name
        ! (a USE rename may differ from the module name).
        use, intrinsic :: iso_c_binding, only: c_int32_t, c_int64_t
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: var_idx
        integer, intent(in) :: value_kind
        character(len=*), intent(in) :: sym_name
        integer(c_int32_t), intent(in) :: global_id
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: sym_idx, value_kind_local, array_size, array_lower

        call module_array_variable_info(arena, var_idx, context, value_kind_local, &
                                        array_size, array_lower, error_msg)
        if (len_trim(error_msg) > 0) return
        call grow_symbols(context)
        sym_idx = context%symbol_count + 1
        context%symbols(sym_idx)%name = trim(sym_name)
        context%symbols(sym_idx)%value_kind = value_kind
        context%symbols(sym_idx)%has_address = .true.
        context%symbols(sym_idx)%is_reference = .true.
        context%symbols(sym_idx)%is_array = .true.
        context%symbols(sym_idx)%array_rank = 1
        context%symbols(sym_idx)%array_size = array_size
        context%symbols(sym_idx)%array_dim_sizes(1) = array_size
        context%symbols(sym_idx)%array_dim_lowers(1) = array_lower
        context%symbols(sym_idx)%address%kind = LR_OP_KIND_GLOBAL
        context%symbols(sym_idx)%address%payload = int(global_id, c_int64_t)
        context%symbols(sym_idx)%address%typ = lr_type_ptr_s(context%session%handle)
        context%symbols(sym_idx)%address%global_offset = 0_c_int64_t
        context%symbols(sym_idx)%element_address = context%symbols(sym_idx)%address
        context%symbol_count = sym_idx
        call set_empty(error_msg)
    end subroutine bind_one_module_array_variable

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
                    call declaration_value_kind(node, value_kind, kind_err, &
                                                context)
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
