submodule (session_program_lowering_impl) session_program_lowering_c_ptr
    implicit none
contains
    module procedure null_ptr_operand
    ! A null c_ptr value: an 8-byte pointer-typed zero.
    op%kind = LR_OP_KIND_IMM_I64
    op%payload = 0_c_int64_t
    op%typ = lr_type_ptr_s(context%session%handle)
    op%global_offset = 0_c_int64_t
    end procedure null_ptr_operand

    module procedure define_c_ptr_symbol
    integer :: index

    if (find_symbol_compat(context, name) > 0) then
        error_msg = 'duplicate c_ptr declaration: '//trim(name)
        return
    end if
    call grow_symbols(context)
    index = context%symbol_count + 1
    context%symbols(index)%name = trim(name)
    context%symbols(index)%value_kind = VALUE_C_PTR
    context%symbols(index)%value = null_ptr_operand(context)
    context%symbol_count = index
    call set_empty(error_msg)
    end procedure define_c_ptr_symbol

    module procedure call_or_subscript_arg_indices
    allocate (arg_indices(0))
    if (.not. node_exists(arena, node_index)) then
        error_msg = 'call index does not reference an AST node'
        return
    end if
    select type (node => arena%entries(node_index)%node)
        type is (call_or_subscript_node)
        if (allocated(node%arg_indices)) then
            if (size(node%arg_indices) > 0) arg_indices = node%arg_indices
        end if
        call set_empty(error_msg)
    class default
        error_msg = 'expected a call expression'
    end select
    end procedure call_or_subscript_arg_indices

    module procedure is_named_call
    ! A scalar call_or_subscript to the intrinsic named `name`.
    is_named_call = .false.
    if (.not. node_exists(arena, node_index)) return
    select type (node => arena%entries(node_index)%node)
        type is (call_or_subscript_node)
        if (.not. node%is_array_access .and. allocated(node%name)) &
            is_named_call = same_name(node%name, name)
    end select
    end procedure is_named_call

    module procedure lower_c_ptr_expression
    ! Lower an expression that yields a c_ptr: c_null_ptr, a c_ptr variable,
    ! or c_loc(x).
    character(len=:), allocatable :: name
    integer :: symbol_index

    if (.not. node_exists(arena, node_index)) then
        error_msg = 'c_ptr expression does not reference an AST node'
        return
    end if
    if (is_named_call(arena, node_index, 'c_loc')) then
        block
            integer, allocatable :: c_loc_args(:)
            call call_or_subscript_arg_indices(arena, node_index, &
                c_loc_args, error_msg)
            if (len_trim(error_msg) > 0) return
            call lower_c_loc(arena, c_loc_args, context, value, error_msg)
        end block
        return
    end if
    if (is_identifier(arena, node_index)) then
        call get_identifier_name(arena, node_index, name, error_msg)
        if (len_trim(error_msg) > 0) return
        if (same_name(name, 'c_null_ptr')) then
            value = null_ptr_operand(context)
            call set_empty(error_msg)
            return
        end if
        symbol_index = find_symbol_compat(context, name)
        if (symbol_index <= 0) then
            error_msg = 'c_ptr identifier was not declared: '//trim(name)
            return
        end if
        if (context%symbols(symbol_index)%value_kind /= VALUE_C_PTR) then
            error_msg = 'expected a c_ptr value: '//trim(name)
            return
        end if
        if (context%symbols(symbol_index)%has_address .and. &
            context%symbols(symbol_index)%is_reference) then
            if (.not. emit_ptr_load(context%session, &
                context%symbols(symbol_index)%address, value, error_msg)) &
                return
        else
            value = context%symbols(symbol_index)%value
            call set_empty(error_msg)
        end if
        return
    end if
    error_msg = 'direct LIRIC session c_ptr expression supports c_null_ptr, '// &
        'a c_ptr variable, or c_loc(x)'
    end procedure lower_c_ptr_expression

    module procedure lower_c_loc
    ! c_loc(x): the address of x. A scalar local is materialised in memory
    ! (alloca + store) to obtain a stable address.
    character(len=:), allocatable :: name
    integer :: symbol_index

    call set_empty(error_msg)
    if (.not. allocated(arg_indices)) then
        error_msg = 'c_loc requires exactly one argument'
        return
    end if
    if (size(arg_indices) /= 1) then
        error_msg = 'c_loc requires exactly one argument'
        return
    end if
    if (.not. is_identifier(arena, arg_indices(1))) then
        error_msg = 'c_loc argument must be a scalar variable'
        return
    end if
    call get_identifier_name(arena, arg_indices(1), name, error_msg)
    if (len_trim(error_msg) > 0) return
    symbol_index = find_symbol_compat(context, name)
    if (symbol_index <= 0) then
        error_msg = 'c_loc argument was not declared: '//trim(name)
        return
    end if
    if (context%symbols(symbol_index)%has_address) then
        value = context%symbols(symbol_index)%address
        call set_empty(error_msg)
        return
    end if
    call make_reference_argument(context, &
        context%symbols(symbol_index)%value_kind, &
        context%symbols(symbol_index)%value, value, &
        error_msg)
    end procedure lower_c_loc

    module procedure c_f_pointer_shape_extents
    ! Read the constant extents of a C_F_POINTER SHAPE actual. Only a
    ! constant rank-1 array constructor is supported: the descriptor extents
    ! must be known when the pointer adopts the C address.
    integer(c_int64_t) :: extent
    logical :: ok
    integer :: i, n

    allocate (extents(0))
    if (.not. node_exists(arena, shape_index)) then
        error_msg = 'C_F_POINTER SHAPE does not reference an AST node'
        return
    end if
    select type (node => arena%entries(shape_index)%node)
        type is (array_literal_node)
        if (.not. allocated(node%element_indices)) then
            error_msg = 'C_F_POINTER SHAPE must supply at least one extent'
            return
        end if
        n = size(node%element_indices)
        if (n < 1) then
            error_msg = 'C_F_POINTER SHAPE must supply at least one extent'
            return
        end if
        deallocate (extents)
        allocate (extents(n))
        do i = 1, n
            call try_const_int64(arena, node%element_indices(i), extent, ok)
            if (.not. ok) then
                error_msg = 'C_F_POINTER SHAPE extent must be a constant '// &
                    'integer expression'
                return
            end if
            if (extent < 0_c_int64_t) then
                error_msg = 'C_F_POINTER SHAPE extent must not be negative'
                return
            end if
            extents(i) = int(extent)
        end do
        call set_empty(error_msg)
    class default
        error_msg = 'C_F_POINTER SHAPE must be a constant array constructor'
    end select
    end procedure c_f_pointer_shape_extents

    module procedure associate_c_f_pointer_array
    ! c_f_pointer(cptr, fptr, shape): the array pointer adopts the C address
    ! as its element base and the validated SHAPE extents as its descriptor
    ! shape. No pointee storage is copied.
    integer, allocatable :: extents(:)
    integer :: i, total

    call c_f_pointer_shape_extents(arena, shape_index, extents, error_msg)
    if (len_trim(error_msg) > 0) return
    if (.not. context%symbols(symbol_index)%is_array) then
        error_msg = 'C_F_POINTER SHAPE was supplied for a scalar FPTR'
        return
    end if
    if (size(extents) /= context%symbols(symbol_index)%array_rank) then
        error_msg = 'C_F_POINTER SHAPE size does not match the rank of FPTR'
        return
    end if
    if (size(extents) > ARRAY_MAX_RANK) then
        error_msg = 'C_F_POINTER SHAPE rank is not supported'
        return
    end if
    total = 1
    do i = 1, size(extents)
        total = total*extents(i)
    end do
    context%symbols(symbol_index)%element_address = ptr_value
    context%symbols(symbol_index)%address = ptr_value
    context%symbols(symbol_index)%array_size = total
    context%symbols(symbol_index)%array_lower_bound = 1
    context%symbols(symbol_index)%array_dim_sizes = 0
    context%symbols(symbol_index)%array_dim_lowers = 0
    do i = 1, size(extents)
        context%symbols(symbol_index)%array_dim_sizes(i) = extents(i)
        context%symbols(symbol_index)%array_dim_lowers(i) = 1
    end do
    context%symbols(symbol_index)%has_address = .true.
    context%symbols(symbol_index)%is_reference = .true.
    context%symbols(symbol_index)%is_pointer = .true.
    context%symbols(symbol_index)%is_associated = .true.
    call set_empty(error_msg)
    end procedure associate_c_f_pointer_array

    module procedure lower_c_associated
    ! c_associated(p): 1 when p is non-null, 0 otherwise (an i32 boolean).
    ! c_associated(p, q): 1 when p is non-null and refers to the same C
    ! address as q (F2018 18.2.3.2).
    type(lr_operand_desc_t) :: ptr_value, other_value, same_value, nonnull_value

    call set_empty(error_msg)
    if (.not. allocated(arg_indices)) then
        error_msg = 'c_associated requires one or two arguments'
        return
    end if
    if (size(arg_indices) < 1 .or. size(arg_indices) > 2) then
        error_msg = 'c_associated requires one or two arguments'
        return
    end if
    call lower_c_ptr_expression(arena, arg_indices(1), context, ptr_value, &
        error_msg)
    if (len_trim(error_msg) > 0) return
    if (.not. emit_liric_i32_icmp(context%session, LR_CMP_NE, ptr_value, &
        null_ptr_operand(context), value, error_msg)) return
    if (size(arg_indices) == 2) then
        call lower_c_ptr_expression(arena, arg_indices(2), context, &
            other_value, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_EQ, ptr_value, &
            other_value, same_value, error_msg)) return
        nonnull_value = value
        if (.not. emit_i32_binary(context%session, LR_OP_AND, nonnull_value, &
            same_value, value, error_msg)) return
    end if
    call set_empty(error_msg)
    end procedure lower_c_associated

    module procedure lower_c_f_pointer
    ! c_f_pointer(cptr, fptr): the MVP stores cptr into fptr's c_ptr slot.
    type(lr_operand_desc_t) :: ptr_value
    character(len=:), allocatable :: name
    integer :: symbol_index

    if (.not. allocated(arg_indices)) then
        error_msg = 'c_f_pointer requires two arguments'
        return
    end if
    if (size(arg_indices) < 2) then
        error_msg = 'c_f_pointer requires at least two arguments'
        return
    end if
    if (size(arg_indices) > 3) then
        error_msg = 'c_f_pointer accepts at most three arguments'
        return
    end if
    call lower_c_ptr_expression(arena, arg_indices(1), context, ptr_value, &
        error_msg)
    if (len_trim(error_msg) > 0) return
    if (.not. is_identifier(arena, arg_indices(2))) then
        error_msg = 'c_f_pointer target must be a scalar pointer variable'
        return
    end if
    call get_identifier_name(arena, arg_indices(2), name, error_msg)
    if (len_trim(error_msg) > 0) return
    symbol_index = find_symbol_compat(context, name)
    if (symbol_index <= 0) then
        error_msg = 'c_f_pointer target was not declared: '//trim(name)
        return
    end if
    ! F2018 18.2.3.3: SHAPE is optional only when FPTR is scalar. An array
    ! FPTR without SHAPE has no extents to take, so reject it here.
    if (context%symbols(symbol_index)%is_array .and. size(arg_indices) < 3) then
        error_msg = 'Expected SHAPE argument to C_F_POINTER with array FPTR'
        return
    end if
    if (size(arg_indices) == 3) then
        call associate_c_f_pointer_array(arena, arg_indices(3), context, &
            symbol_index, ptr_value, error_msg)
        return
    end if
    ! Bind the Fortran pointer to the C address: the pointer adopts cp's
    ! address as its storage, so a later read of p dereferences that address
    ! (emit_*_load) and yields the value the C pointer refers to.
    context%symbols(symbol_index)%value = ptr_value
    context%symbols(symbol_index)%address = ptr_value
    context%symbols(symbol_index)%has_address = .true.
    context%symbols(symbol_index)%is_reference = .true.
    context%symbols(symbol_index)%is_pointer = .true.
    context%symbols(symbol_index)%is_associated = .true.
    call set_empty(error_msg)
    end procedure lower_c_f_pointer

end submodule session_program_lowering_c_ptr
