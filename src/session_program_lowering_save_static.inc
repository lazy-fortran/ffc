    ! Static storage for saved locals beyond the scalar case (#466). A saved
    ! array is one [extent x elem] LIRIC global, exactly like a fixed-size
    ! module array, so every element keeps its value across calls. A DATA
    ! initializer for a saved local folds into that global's static bytes and
    ! its runtime stores are suppressed, so the value is applied once before
    ! the first executable use instead of on every call.

    subroutine emit_saved_array_globals(arena, node, node_index, context, &
                                        error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(declaration_node), intent(in) :: node
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i

        call set_empty(error_msg)
        if (node%is_multi_declaration .and. allocated(node%var_names)) then
            do i = 1, size(node%var_names)
                call emit_one_saved_array_global(arena, node, node_index, &
                    trim(node%var_names(i)), context, error_msg)
                if (len_trim(error_msg) > 0) return
            end do
        else if (allocated(node%var_name)) then
            call emit_one_saved_array_global(arena, node, node_index, &
                trim(node%var_name), context, error_msg)
        else
            error_msg = 'saved array declaration did not expose a variable name'
        end if
    end subroutine emit_saved_array_globals

    subroutine emit_one_saved_array_global(arena, node, node_index, name, &
                                           context, error_msg)
        ! Create the [extent x elem] global backing one saved array name. The
        ! static bytes start zeroed, then take a constant array-constructor
        ! initializer or a DATA value list, whichever the declaration carries.
        use, intrinsic :: iso_c_binding, only: c_bool, c_char, c_int32_t, &
                                               c_int64_t, c_loc, c_size_t, &
                                               c_ptr, c_double, c_float
        use liric_session_bindings, only: lr_type_i32_s, lr_type_f32_s, &
                                          lr_type_f64_s, lr_type_array_s
        type(ast_arena_t), intent(in) :: arena
        type(declaration_node), intent(in) :: node
        integer, intent(in) :: node_index
        character(len=*), intent(in) :: name
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: gname, info_err
        character(kind=c_char), allocatable :: c_name(:)
        integer(c_int32_t), target, allocatable :: array_i32(:)
        real(c_float), target, allocatable :: array_f32(:)
        real(c_double), target, allocatable :: array_f64(:)
        type(c_ptr) :: elem_type, gtype, init_ptr
        integer(c_size_t) :: init_size
        integer(c_int32_t) :: global_id
        integer :: value_kind, array_size, array_lower
        logical(c_bool) :: c_false_local

        call set_empty(error_msg)
        c_false_local = .false.
        call saved_array_info(arena, node, node_index, context, value_kind, &
                              array_size, array_lower, error_msg)
        if (len_trim(error_msg) > 0) return
        allocate (array_i32(max(array_size, 1)))
        allocate (array_f32(max(array_size, 1)))
        allocate (array_f64(max(array_size, 1)))
        array_i32 = 0_c_int32_t
        array_f32 = 0.0_c_float
        array_f64 = 0.0_c_double
        if (node%has_initializer .and. node%initializer_index > 0) then
            call fold_module_array_initializer(arena, node_index, value_kind, &
                array_size, context, array_i32, array_f32, array_f64, info_err)
            if (len_trim(info_err) > 0) then
                error_msg = info_err
                return
            end if
        else
            call fold_saved_data_values(arena, node_index, name, value_kind, &
                array_size, context, array_i32, array_f32, array_f64, error_msg)
            if (len_trim(error_msg) > 0) return
        end if
        select case (value_kind)
        case (VALUE_I32, VALUE_LOGICAL)
            elem_type = lr_type_i32_s(context%session%handle)
            init_ptr = c_loc(array_i32)
            init_size = int(array_size, c_size_t)*4_c_size_t
        case (VALUE_F32)
            elem_type = lr_type_f32_s(context%session%handle)
            init_ptr = c_loc(array_f32)
            init_size = int(array_size, c_size_t)*4_c_size_t
        case (VALUE_F64)
            elem_type = lr_type_f64_s(context%session%handle)
            init_ptr = c_loc(array_f64)
            init_size = int(array_size, c_size_t)*8_c_size_t
        case default
            call unsupported_feature_error('saved array declaration', &
                node%line, node%column, 'direct LIRIC session supports the '// &
                'save attribute on integer, real, and logical arrays only', &
                error_msg)
            return
        end select
        gname = saved_local_global_name(node_index, name)
        call to_c_chars(gname, c_name)
        gtype = lr_type_array_s(context%session%handle, elem_type, &
                                int(array_size, c_int64_t))
        global_id = lr_session_global(context%session%handle, c_name, gtype, &
                                      c_false_local, init_ptr, init_size)
        if (global_id < 0_c_int32_t) then
            error_msg = 'LIRIC could not create saved array global: '//gname
            return
        end if
        call set_empty(error_msg)
    end subroutine emit_one_saved_array_global

    subroutine saved_array_info(arena, node, node_index, context, value_kind, &
                                array_size, array_lower, error_msg)
        ! Shape and element kind of a supported saved array. Any out-of-scope
        ! form keeps the unit xfail with a saved-array diagnostic.
        type(ast_arena_t), intent(in) :: arena
        type(declaration_node), intent(in) :: node
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        integer, intent(out) :: value_kind
        integer, intent(out) :: array_size
        integer, intent(out) :: array_lower
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: info_err

        call module_array_variable_info(arena, node_index, context, value_kind, &
                                        array_size, array_lower, info_err)
        if (len_trim(info_err) > 0) then
            call unsupported_feature_error('saved array declaration', &
                node%line, node%column, trim(info_err), error_msg)
            return
        end if
        if (array_size <= 0) then
            call unsupported_feature_error('saved array declaration', &
                node%line, node%column, 'saved array extent must be a '// &
                'positive compile-time constant', error_msg)
            return
        end if
        call set_empty(error_msg)
    end subroutine saved_array_info

    subroutine lower_saved_array_declaration(arena, node, node_index, context, &
                                             error_msg)
        ! Body-side binding: the backing global already exists from the pre-pass,
        ! so bind each name onto it as an addressed array reference. Re-entering
        ! the same declaration rebinds the same global rather than allocating a
        ! second slot.
        type(ast_arena_t), intent(in) :: arena
        type(declaration_node), intent(in) :: node
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i

        call set_empty(error_msg)
        if (node%is_multi_declaration .and. allocated(node%var_names)) then
            do i = 1, size(node%var_names)
                call bind_saved_array(arena, node, node_index, &
                    trim(node%var_names(i)), context, error_msg)
                if (len_trim(error_msg) > 0) return
            end do
        else if (allocated(node%var_name)) then
            call bind_saved_array(arena, node, node_index, trim(node%var_name), &
                                  context, error_msg)
        else
            error_msg = 'saved array declaration did not expose a variable name'
        end if
    end subroutine lower_saved_array_declaration

    subroutine bind_saved_array(arena, node, node_index, name, context, error_msg)
        use, intrinsic :: iso_c_binding, only: c_char, c_int32_t, c_int64_t
        type(ast_arena_t), intent(in) :: arena
        type(declaration_node), intent(in) :: node
        integer, intent(in) :: node_index
        character(len=*), intent(in) :: name
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: gname
        character(kind=c_char), allocatable :: c_name(:)
        integer(c_int32_t) :: global_id
        integer :: sym_idx, existing_idx, value_kind, array_size, array_lower

        call saved_array_info(arena, node, node_index, context, value_kind, &
                              array_size, array_lower, error_msg)
        if (len_trim(error_msg) > 0) return
        gname = saved_local_global_name(node_index, name)
        call to_c_chars(gname, c_name)
        global_id = lr_session_intern(context%session%handle, c_name)
        if (global_id < 0_c_int32_t) then
            error_msg = 'saved array global not found: '//gname
            return
        end if
        existing_idx = find_symbol_compat(context, name)
        if (existing_idx > 0) then
            sym_idx = existing_idx
        else
            call grow_symbols(context)
            sym_idx = context%symbol_count + 1
            context%symbol_count = sym_idx
        end if
        context%symbols(sym_idx) = symbol_t()
        context%symbols(sym_idx)%name = name
        context%symbols(sym_idx)%value_kind = value_kind
        context%symbols(sym_idx)%has_address = .true.
        context%symbols(sym_idx)%is_reference = .true.
        context%symbols(sym_idx)%is_array = .true.
        context%symbols(sym_idx)%array_rank = 1
        context%symbols(sym_idx)%array_size = array_size
        context%symbols(sym_idx)%array_lower_bound = array_lower
        context%symbols(sym_idx)%array_dim_sizes(1) = array_size
        context%symbols(sym_idx)%array_dim_lowers(1) = array_lower
        context%symbols(sym_idx)%address%kind = LR_OP_KIND_GLOBAL
        context%symbols(sym_idx)%address%payload = int(global_id, c_int64_t)
        context%symbols(sym_idx)%address%typ = lr_type_ptr_s(context%session%handle)
        context%symbols(sym_idx)%address%global_offset = 0_c_int64_t
        context%symbols(sym_idx)%element_address = context%symbols(sym_idx)%address
        context%symbols(sym_idx)%is_static_data_initialized = &
            saved_local_has_data_values(arena, node_index, name, context)
        call set_empty(error_msg)
    end subroutine bind_saved_array

    ! ---- DATA initializers for saved locals -------------------------------

    logical function saved_local_has_data_values(arena, node_index, name, context)
        ! Whether a DATA statement in the declaration's own scope names this
        ! saved local. Such a value is folded into the static global, so the
        ! body walk must not re-apply it on every call.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        character(len=*), intent(in) :: name
        type(lowering_context_t), intent(in) :: context
        integer, allocatable :: value_indices(:)
        character(len=:), allocatable :: err

        allocate (value_indices(0))
        call saved_local_data_value_indices(arena, node_index, name, context, &
                                            value_indices, err)
        saved_local_has_data_values = size(value_indices) > 0
    end function saved_local_has_data_values

    subroutine saved_local_data_value_indices(arena, node_index, name, context, &
                                              value_indices, error_msg)
        ! Collect the DATA value expressions that initialise `name`, searching
        ! the DATA statements of the scope that owns the declaration. Only flat
        ! object lists of plain identifiers are matched; anything else leaves
        ! the list empty so the ordinary runtime DATA path stays in charge.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        character(len=*), intent(in) :: name
        type(lowering_context_t), intent(in) :: context
        integer, allocatable, intent(out) :: value_indices(:)
        character(len=:), allocatable, intent(out) :: error_msg
        integer, allocatable :: body(:)
        integer :: i

        call set_empty(error_msg)
        allocate (value_indices(0))
        call saved_scope_body(arena, node_index, body)
        if (size(body) == 0) return
        do i = 1, size(body)
            if (.not. is_data_statement_at(arena, body(i))) cycle
            select type (dnode => arena%entries(body(i))%node)
            type is (data_statement_node)
                call data_values_for_name(arena, dnode, body, name, context, &
                                          value_indices, error_msg)
                if (len_trim(error_msg) > 0) return
                if (size(value_indices) > 0) return
            end select
        end do
    end subroutine saved_local_data_value_indices

    subroutine saved_scope_body(arena, node_index, body)
        ! Statement list of the program unit that owns a declaration node.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        integer, allocatable, intent(out) :: body(:)
        integer :: parent_index

        allocate (body(0))
        if (node_index <= 0) return
        if (.not. node_exists(arena, node_index)) return
        parent_index = arena%entries(node_index)%parent_index
        if (parent_index <= 0) return
        if (.not. node_exists(arena, parent_index)) return
        select type (parent => arena%entries(parent_index)%node)
        type is (subroutine_def_node)
            if (allocated(parent%body_indices)) body = parent%body_indices
        type is (function_def_node)
            if (allocated(parent%body_indices)) body = parent%body_indices
        type is (program_node)
            if (allocated(parent%body_indices)) body = parent%body_indices
        end select
    end subroutine saved_scope_body

    subroutine data_values_for_name(arena, dnode, body, name, context, &
                                    value_indices, error_msg)
        ! Walk one DATA statement's flat object list, consuming as many values
        ! as each object needs, and return the slice belonging to `name`.
        type(ast_arena_t), intent(in) :: arena
        type(data_statement_node), intent(in) :: dnode
        integer, intent(in) :: body(:)
        character(len=*), intent(in) :: name
        type(lowering_context_t), intent(in) :: context
        integer, allocatable, intent(inout) :: value_indices(:)
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: obj_name, name_err
        integer :: oi, vpos, count, k

        call set_empty(error_msg)
        if (.not. allocated(dnode%object_indices)) return
        if (.not. allocated(dnode%value_indices)) return
        vpos = 0
        do oi = 1, size(dnode%object_indices)
            call identifier_name(arena, dnode%object_indices(oi), obj_name, &
                                 name_err)
            if (len_trim(name_err) > 0) return
            call scope_object_element_count(arena, body, obj_name, context, count)
            if (count <= 0) return
            if (vpos + count > size(dnode%value_indices)) return
            if (same_name(obj_name, name)) then
                deallocate (value_indices)
                allocate (value_indices(count))
                do k = 1, count
                    value_indices(k) = dnode%value_indices(vpos + k)
                end do
                return
            end if
            vpos = vpos + count
        end do
    end subroutine data_values_for_name

    subroutine scope_object_element_count(arena, body, name, context, count)
        ! Element count a DATA object consumes: its declared extent for an
        ! array, one for a scalar, zero when the scope does not declare it.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: body(:)
        character(len=*), intent(in) :: name
        type(lowering_context_t), intent(in) :: context
        integer, intent(out) :: count
        integer :: i, j

        count = 0
        do i = 1, size(body)
            if (.not. node_exists(arena, body(i))) cycle
            select type (decl => arena%entries(body(i))%node)
            type is (declaration_node)
                if (decl%is_multi_declaration .and. allocated(decl%var_names)) then
                    do j = 1, size(decl%var_names)
                        if (.not. same_name(decl%var_names(j), name)) cycle
                        call declared_element_count(decl, context, count)
                        return
                    end do
                else if (allocated(decl%var_name)) then
                    if (same_name(decl%var_name, name)) then
                        call declared_element_count(decl, context, count)
                        return
                    end if
                end if
            end select
        end do
    end subroutine scope_object_element_count

    subroutine declared_element_count(decl, context, count)
        type(declaration_node), intent(in) :: decl
        type(lowering_context_t), intent(in) :: context
        integer, intent(out) :: count
        integer :: lower, extent
        character(len=:), allocatable :: bounds_err

        count = 1
        if (.not. decl%is_array) return
        count = 0
        if (.not. allocated(decl%dimension_indices)) return
        if (size(decl%dimension_indices) /= 1) return
        call get_array_bounds(decl, context, lower, extent, bounds_err)
        if (len_trim(bounds_err) > 0) return
        if (extent <= 0) return
        count = extent
    end subroutine declared_element_count

    subroutine fold_saved_data_values(arena, node_index, name, value_kind, &
                                      array_size, context, array_i32, array_f32, &
                                      array_f64, error_msg)
        ! Fold the DATA values naming this saved local into its static element
        ! buffer. A missing or partial value list leaves the zero fill.
        use, intrinsic :: iso_c_binding, only: c_int32_t, c_int64_t, c_float, &
                                               c_double
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        character(len=*), intent(in) :: name
        integer, intent(in) :: value_kind
        integer, intent(in) :: array_size
        type(lowering_context_t), intent(in) :: context
        integer(c_int32_t), intent(inout), allocatable :: array_i32(:)
        real(c_float), intent(inout), allocatable :: array_f32(:)
        real(c_double), intent(inout), allocatable :: array_f64(:)
        character(len=:), allocatable, intent(out) :: error_msg
        integer, allocatable :: value_indices(:)
        integer :: i

        call set_empty(error_msg)
        allocate (value_indices(0))
        call saved_local_data_value_indices(arena, node_index, name, context, &
                                            value_indices, error_msg)
        if (len_trim(error_msg) > 0) return
        if (size(value_indices) == 0) return
        if (size(value_indices) /= array_size) then
            error_msg = 'saved DATA value count does not match '//trim(name)
            return
        end if
        do i = 1, array_size
            call fold_saved_data_element(arena, value_indices(i), value_kind, &
                array_i32(i), array_f32(i), array_f64(i), error_msg)
            if (len_trim(error_msg) > 0) return
        end do
    end subroutine fold_saved_data_values

    subroutine fold_saved_data_element(arena, vidx, value_kind, out_i32, out_f32, &
                                       out_f64, error_msg)
        ! One DATA literal folded into the static bytes of a saved local.
        use, intrinsic :: iso_c_binding, only: c_int32_t, c_int64_t, c_float, &
                                               c_double
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: vidx
        integer, intent(in) :: value_kind
        integer(c_int32_t), intent(inout) :: out_i32
        real(c_float), intent(inout) :: out_f32
        real(c_double), intent(inout) :: out_f64
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: lit_value, lit_type, lit_err
        integer(c_int64_t) :: iconst
        integer :: io_stat

        call set_empty(error_msg)
        if (.not. is_literal(arena, vidx)) then
            error_msg = 'only literal DATA values initialise a saved local'
            return
        end if
        call get_literal_info(arena, vidx, lit_value, lit_type, lit_err)
        if (len_trim(lit_err) > 0) then
            error_msg = lit_err
            return
        end if
        select case (value_kind)
        case (VALUE_I32)
            call parse_i32_literal(lit_value, iconst, error_msg)
            if (len_trim(error_msg) > 0) return
            out_i32 = int(iconst, c_int32_t)
        case (VALUE_LOGICAL)
            out_i32 = logical_literal_to_i32(lit_value)
        case (VALUE_F32)
            read (lit_value, *, iostat=io_stat) out_f32
            if (io_stat /= 0) error_msg = 'malformed real DATA value: '//lit_value
        case (VALUE_F64)
            read (lit_value, *, iostat=io_stat) out_f64
            if (io_stat /= 0) error_msg = 'malformed real DATA value: '//lit_value
        case default
            error_msg = 'unsupported kind for a saved DATA initializer'
        end select
    end subroutine fold_saved_data_element

    subroutine fold_saved_scalar_data(arena, node_index, name, context, &
                                      value_kind, init_i32, init_f32, init_f64, &
                                      error_msg)
        ! A saved scalar without a declaration initializer may still take a DATA
        ! value; fold it into the static bytes of its backing global so it is
        ! applied once instead of on every call.
        use, intrinsic :: iso_c_binding, only: c_int32_t, c_float, c_double
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        character(len=*), intent(in) :: name
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: value_kind
        integer(c_int32_t), intent(inout) :: init_i32
        real(c_float), intent(inout) :: init_f32
        real(c_double), intent(inout) :: init_f64
        character(len=:), allocatable, intent(out) :: error_msg
        integer, allocatable :: value_indices(:)

        call set_empty(error_msg)
        allocate (value_indices(0))
        call saved_local_data_value_indices(arena, node_index, name, context, &
                                            value_indices, error_msg)
        if (len_trim(error_msg) > 0) return
        if (size(value_indices) /= 1) return
        call fold_saved_data_element(arena, value_indices(1), value_kind, &
                                     init_i32, init_f32, init_f64, error_msg)
    end subroutine fold_saved_scalar_data
