    ! SAVE attribute for scalar locals (#1541). A saved local keeps its value
    ! across procedure calls, so it is backed by a LIRIC global whose literal
    ! initializer is applied once at load. The local symbol becomes an addressed
    ! reference into that global, so ordinary reads and writes go through the
    ! persistent storage.
    !
    ! The backing globals are emitted in a pre-pass before any function body, the
    ! same ordering module variables rely on: a global created while a function
    ! is being compiled gets its address patched directly into the code instead
    ! of a symbolic relocation, which bypasses the data-section initializer. The
    ! global name encodes the declaration's source line and column so the
    ! pre-pass and the in-body binding agree on the same symbol. The arena node
    ! index of the declaration keys the name, so two distinct `count` saved
    ! locals get distinct static storage.

    subroutine emit_saved_local_globals(arena, context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: n

        call set_empty(error_msg)
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            if (.not. is_declaration_node(arena, n)) cycle
            select type (decl => arena%entries(n)%node)
            type is (declaration_node)
                if (.not. decl%is_save) cycle
                if (decl%is_parameter .or. decl%is_array .or. &
                    decl%is_pointer .or. decl%is_target) cycle
                call emit_saved_decl_globals(decl, n, context, error_msg)
                if (len_trim(error_msg) > 0) return
            end select
        end do
    end subroutine emit_saved_local_globals

    subroutine emit_saved_decl_globals(node, node_index, context, error_msg)
        type(declaration_node), intent(in) :: node
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: value_kind, i

        call saved_scalar_value_kind(node, value_kind, error_msg)
        if (len_trim(error_msg) > 0) return
        if (node%is_multi_declaration .and. allocated(node%var_names)) then
            do i = 1, size(node%var_names)
                ! A multi-name save shares at most one initializer expression;
                ! apply it to the first name only, matching declaration order.
                call emit_one_saved_global(node, node_index, context, &
                    trim(node%var_names(i)), value_kind, i == 1, error_msg)
                if (len_trim(error_msg) > 0) return
            end do
        else if (allocated(node%var_name)) then
            call emit_one_saved_global(node, node_index, context, &
                trim(node%var_name), value_kind, .true., error_msg)
        else
            error_msg = 'saved declaration did not expose a variable name'
        end if
    end subroutine emit_saved_decl_globals

    subroutine saved_scalar_value_kind(node, value_kind, error_msg)
        type(declaration_node), intent(in) :: node
        integer, intent(out) :: value_kind
        character(len=:), allocatable, intent(out) :: error_msg

        call declaration_value_kind(node, value_kind, error_msg)
        if (len_trim(error_msg) > 0) return
        select case (value_kind)
        case (VALUE_I32, VALUE_F32, VALUE_F64, VALUE_LOGICAL)
            call set_empty(error_msg)
        case default
            call unsupported_feature_error('saved local declaration', node%line, &
                node%column, 'direct LIRIC session supports the save attribute '// &
                'on scalar integer, real, and logical locals only', error_msg)
        end select
    end subroutine saved_scalar_value_kind

    subroutine emit_one_saved_global(node, node_index, context, name, value_kind, &
                                     take_initializer, error_msg)
        use, intrinsic :: iso_c_binding, only: c_bool, c_char, c_int32_t, &
                                               c_loc, c_size_t, c_ptr, &
                                               c_double, c_float
        use liric_session_bindings, only: lr_type_i32_s, lr_type_f32_s, &
                                          lr_type_f64_s
        type(declaration_node), intent(in) :: node
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: name
        integer, intent(in) :: value_kind
        logical, intent(in) :: take_initializer
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: gname
        character(kind=c_char), allocatable :: c_name(:)
        integer(c_int32_t), target :: init_i32
        real(c_float), target :: init_f32
        real(c_double), target :: init_f64
        type(c_ptr) :: gtype, init_ptr
        integer(c_size_t) :: init_size
        integer(c_int32_t) :: global_id
        logical(c_bool) :: c_false_local

        c_false_local = .false.
        call fold_saved_initializer(node, context, value_kind, take_initializer, &
                                    init_i32, init_f32, init_f64, error_msg)
        if (len_trim(error_msg) > 0) return
        gname = saved_local_global_name(node_index, name)
        call to_c_chars(gname, c_name)
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
        end select
        global_id = lr_session_global(context%session%handle, c_name, gtype, &
                                      c_false_local, init_ptr, init_size)
        if (global_id < 0_c_int32_t) then
            error_msg = 'LIRIC could not create saved local global: '//gname
            return
        end if
        call set_empty(error_msg)
    end subroutine emit_one_saved_global

    subroutine lower_saved_scalar_declaration(node, node_index, context, &
                                              value_kind, error_msg)
        ! Body-side binding: the backing globals already exist from the pre-pass,
        ! so bind each name as an addressed reference into its global.
        type(declaration_node), intent(in) :: node
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: value_kind
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i

        select case (value_kind)
        case (VALUE_I32, VALUE_F32, VALUE_F64, VALUE_LOGICAL)
        case default
            call unsupported_feature_error('saved local declaration', node%line, &
                node%column, 'direct LIRIC session supports the save attribute '// &
                'on scalar integer, real, and logical locals only', error_msg)
            return
        end select

        if (node%is_multi_declaration .and. allocated(node%var_names)) then
            do i = 1, size(node%var_names)
                call bind_saved_scalar(node_index, context, &
                    trim(node%var_names(i)), value_kind, error_msg)
                if (len_trim(error_msg) > 0) return
            end do
        else if (allocated(node%var_name)) then
            call bind_saved_scalar(node_index, context, trim(node%var_name), &
                                   value_kind, error_msg)
        else
            error_msg = 'saved declaration did not expose a variable name'
        end if
    end subroutine lower_saved_scalar_declaration

    subroutine bind_saved_scalar(node_index, context, name, value_kind, error_msg)
        use, intrinsic :: iso_c_binding, only: c_char, c_int32_t, c_int64_t
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: name
        integer, intent(in) :: value_kind
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: gname
        character(kind=c_char), allocatable :: c_name(:)
        integer(c_int32_t) :: global_id
        integer :: sym_idx

        if (find_symbol(context, name) > 0) then
            error_msg = 'duplicate declaration of saved local: '//name
            return
        end if
        gname = saved_local_global_name(node_index, name)
        call to_c_chars(gname, c_name)
        global_id = lr_session_intern(context%session%handle, c_name)
        if (global_id < 0_c_int32_t) then
            error_msg = 'saved local global not found: '//gname
            return
        end if
        call grow_symbols(context)
        sym_idx = context%symbol_count + 1
        context%symbols(sym_idx)%name = name
        context%symbols(sym_idx)%value_kind = value_kind
        context%symbols(sym_idx)%has_address = .true.
        context%symbols(sym_idx)%is_reference = .true.
        context%symbols(sym_idx)%address%kind = LR_OP_KIND_GLOBAL
        context%symbols(sym_idx)%address%payload = int(global_id, c_int64_t)
        context%symbols(sym_idx)%address%typ = lr_type_ptr_s(context%session%handle)
        context%symbols(sym_idx)%address%global_offset = 0_c_int64_t
        context%symbol_count = sym_idx
        call set_empty(error_msg)
    end subroutine bind_saved_scalar

    function saved_local_global_name(node_index, name) result(gname)
        ! The declaration's arena index is unique, so distinct saved locals get
        ! distinct static storage even when they share a variable name.
        integer, intent(in) :: node_index
        character(len=*), intent(in) :: name
        character(len=:), allocatable :: gname
        character(len=32) :: idx_text

        write (idx_text, '(I0)') node_index
        gname = '.ffc.save.'//trim(idx_text)//'.'//name
    end function saved_local_global_name

    subroutine fold_saved_initializer(node, context, value_kind, take_initializer, &
                                      init_i32, init_f32, init_f64, error_msg)
        ! Fold a saved scalar's literal initializer into raw initial bytes. An
        ! absent initializer (plain `save`) starts at zero. Only literal
        ! initializers fold; anything else keeps the unit xfail.
        use, intrinsic :: iso_c_binding, only: c_int32_t, c_int64_t, c_float, &
                                               c_double
        type(declaration_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: value_kind
        logical, intent(in) :: take_initializer
        integer(c_int32_t), intent(out) :: init_i32
        real(c_float), intent(out) :: init_f32
        real(c_double), intent(out) :: init_f64
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: lit_value, lit_type, lit_err
        integer(c_int64_t) :: iconst
        integer :: io_stat

        init_i32 = 0_c_int32_t
        init_f32 = 0.0_c_float
        init_f64 = 0.0_c_double
        call set_empty(error_msg)
        if (.not. take_initializer) return
        if (.not. node%has_initializer .or. node%initializer_index <= 0) return
        if (.not. is_literal(context%arena, node%initializer_index)) then
            call unsupported_feature_error('saved local initializer', node%line, &
                node%column, 'only literal initializers are stored for saved '// &
                'locals', error_msg)
            return
        end if
        call get_literal_info(context%arena, node%initializer_index, lit_value, &
                              lit_type, lit_err)
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
    end subroutine fold_saved_initializer
