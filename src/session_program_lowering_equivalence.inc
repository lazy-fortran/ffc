    ! EQUIVALENCE storage overlay (#280, issue_1745). FortFront leaves an
    ! EQUIVALENCE statement as a comment_node ("equivalence (i, r)"), the same as
    ! COMMON. The members of one EQUIVALENCE group name the same storage, so a
    ! write through one member is observable, bit-for-bit, through another.
    !
    ! Model: emit one zero-initialised global per group, sized to hold every
    ! member, and bind every member as an addressed reference into it. Each
    ! member keeps its own value_kind, so a load/store uses that member's type at
    ! the shared address - reading an integer back as a real reinterprets the
    ! bits exactly as gfortran does.
    !
    ! An array-element designator ("a(3)", "a(np)") associates that element with
    ! the group's association point, so the member's own storage starts that many
    ! bytes earlier in the group global (#370). The subscript is a constant
    ! expression: a named constant resolves through its FortFront binding before
    ! the byte offset is computed, and a nonconstant subscript is diagnosed.
    !
    ! Scope: integer(4)/integer(8)/real(4)/real(8)/logical members of a single
    ! (a, b, ...) group, scalars or 1D arrays. Character members and substring
    ! designators leave the symbols plain locals and surface the usual
    ! diagnostics rather than miscompiling.

    logical function is_equivalence_text(text)
        ! An EQUIVALENCE statement begins with the keyword after blanks.
        character(len=*), intent(in) :: text
        character(len=:), allocatable :: low
        is_equivalence_text = .false.
        low = lowercase_text(adjustl(text))
        if (len(low) >= 11) is_equivalence_text = low(1:11) == 'equivalence'
    end function is_equivalence_text

    subroutine emit_equivalence_globals(arena, context, error_msg)
        ! Pre-pass: emit one zero-initialised global per EQUIVALENCE group so the
        ! shared storage exists in the data section before any body is lowered.
        use, intrinsic :: iso_c_binding, only: c_bool, c_char, c_int32_t, &
                                               c_int64_t, c_loc, c_size_t, c_ptr
        use liric_session_bindings, only: lr_type_i8_s, lr_type_array_s
        type(ast_arena_t), intent(in) :: arena
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(equiv_member_t) :: members(EQUIV_MAX_MEMBERS)
        integer :: member_count, group_index, n, byte_size
        character(len=:), allocatable :: gname
        character(kind=c_char), allocatable :: c_name(:)
        character(kind=c_char, len=1), allocatable, target :: zero_bytes(:)
        type(c_ptr) :: gtype
        integer(c_int32_t) :: global_id
        logical(c_bool) :: c_false_local

        call set_empty(error_msg)
        c_false_local = .false.
        group_index = 0
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (node => arena%entries(n)%node)
            type is (comment_node)
                if (.not. is_equivalence_text(node%text)) cycle
                group_index = group_index + 1
                call collect_equivalence_members(arena, context, n, node%text, &
                                                 members, member_count, &
                                                 byte_size, error_msg)
                if (len_trim(error_msg) > 0) return
                if (member_count == 0) cycle
                allocate (zero_bytes(byte_size))
                zero_bytes = char(0)
                gname = equivalence_global_name(group_index)
                call to_c_chars(gname, c_name)
                gtype = lr_type_array_s(context%session%handle, &
                                        lr_type_i8_s(context%session%handle), &
                                        int(byte_size, c_int64_t))
                global_id = lr_session_global(context%session%handle, c_name, &
                                              gtype, c_false_local, &
                                              c_loc(zero_bytes), &
                                              int(byte_size, c_size_t))
                deallocate (zero_bytes)
                if (global_id < 0_c_int32_t) then
                    error_msg = 'LIRIC could not create EQUIVALENCE global: '// &
                                gname
                    return
                end if
            end select
        end do
    end subroutine emit_equivalence_globals

    subroutine bind_equivalence_symbols(arena, context, error_msg)
        ! Body-side binding: rebind every member of each EQUIVALENCE group as an
        ! addressed reference into the group global. Re-derives the group order so
        ! the global name matches the pre-pass; binding is idempotent.
        type(ast_arena_t), intent(in) :: arena
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(equiv_member_t) :: members(EQUIV_MAX_MEMBERS)
        integer :: member_count, group_index, n, i, sym, byte_size

        call set_empty(error_msg)
        group_index = 0
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (node => arena%entries(n)%node)
            type is (comment_node)
                if (.not. is_equivalence_text(node%text)) cycle
                group_index = group_index + 1
                call collect_equivalence_members(arena, context, n, node%text, &
                                                 members, member_count, &
                                                 byte_size, error_msg)
                if (len_trim(error_msg) > 0) return
                do i = 1, member_count
                    sym = find_symbol_compat(context, trim(members(i)%var_name))
                    if (sym <= 0) cycle
                    call rebind_equivalence_symbol(group_index, members(i), sym, &
                                                   context, error_msg)
                    if (len_trim(error_msg) > 0) return
                end do
            end select
        end do
    end subroutine bind_equivalence_symbols

    subroutine collect_equivalence_members(arena, context, node_index, text, &
                                           members, member_count, byte_size, &
                                           error_msg)
        ! Parse "equivalence (a, b(np), ...)" into members with their kinds,
        ! shapes and designated-element offsets, then lay the group out.
        type(ast_arena_t), intent(in) :: arena
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: node_index
        character(len=*), intent(in) :: text
        type(equiv_member_t), intent(out) :: members(:)
        integer, intent(out) :: member_count
        integer, intent(out) :: byte_size
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: group
        character(len=:), allocatable :: items(:)
        integer :: i

        call set_empty(error_msg)
        member_count = 0
        byte_size = 0
        call parse_equivalence_group(text, group, error_msg)
        if (len_trim(error_msg) > 0) return
        call split_equivalence_designators(group, items, member_count)
        if (member_count < 2) then
            error_msg = 'EQUIVALENCE group needs at least two members: '//text
            return
        end if
        if (member_count > size(members)) then
            error_msg = 'too many EQUIVALENCE members for direct lowering'
            return
        end if
        do i = 1, member_count
            call parse_equivalence_member(arena, context, node_index, &
                                          trim(items(i)), members(i), error_msg)
            if (len_trim(error_msg) > 0) return
        end do
        call layout_equivalence_group(members, member_count, byte_size)
    end subroutine collect_equivalence_members

    subroutine parse_equivalence_member(arena, context, node_index, item, &
                                        member, error_msg)
        ! One designator: a bare name, or "name(subscript)" with a constant
        ! subscript. The declaration supplies the kind and shape.
        type(ast_arena_t), intent(in) :: arena
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: node_index
        character(len=*), intent(in) :: item
        type(equiv_member_t), intent(out) :: member
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: base_name, sub_text
        integer :: open_paren, close_paren, subscript

        call set_empty(error_msg)
        open_paren = index(item, '(')
        close_paren = index(item, ')', back=.true.)
        if (open_paren == 0) then
            base_name = trim(item)
            sub_text = ''
        else
            if (close_paren <= open_paren) then
                error_msg = 'malformed EQUIVALENCE designator: '//trim(item)
                return
            end if
            base_name = trim(adjustl(item(1:open_paren - 1)))
            sub_text = trim(adjustl(item(open_paren + 1:close_paren - 1)))
        end if
        member%var_name = base_name
        call program_declaration_kind_full(arena, base_name, member%value_kind, &
                                           member%is_array, member%array_size, &
                                           error_msg)
        if (len_trim(error_msg) > 0) return
        member%byte_offset = 0
        if (len(sub_text) == 0) return
        if (.not. member%is_array) then
            error_msg = 'EQUIVALENCE subscript on a non-array member: '//trim(item)
            return
        end if
        call fold_equivalence_subscript(arena, context, node_index, sub_text, &
                                        subscript, error_msg)
        if (len_trim(error_msg) > 0) return
        if (subscript < 1 .or. subscript > member%array_size) then
            error_msg = 'EQUIVALENCE subscript is out of bounds: '//trim(item)
            return
        end if
        member%byte_offset = (subscript - 1)* &
                             equivalence_kind_size(member%value_kind)
    end subroutine parse_equivalence_member

    subroutine fold_equivalence_subscript(arena, context, node_index, sub_text, &
                                          subscript, error_msg)
        ! An EQUIVALENCE subscript is a constant expression (F2018 8.10.3): an
        ! integer literal, or a named constant that resolves through the
        ! EQUIVALENCE statement's own binding scope before the byte offset is
        ! computed. Anything else is not constant and is diagnosed.
        use, intrinsic :: iso_c_binding, only: c_int64_t
        type(ast_arena_t), intent(in) :: arena
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: node_index
        character(len=*), intent(in) :: sub_text
        integer, intent(out) :: subscript
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int64_t) :: value
        integer :: ios
        logical :: found

        call set_empty(error_msg)
        subscript = 0
        read (sub_text, *, iostat=ios) subscript
        if (ios == 0) return
        call fold_named_constant_at_node(context, node_index, sub_text, value, &
                                         found, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. found) then
            call fold_equivalence_parameter_decl(arena, context, sub_text, &
                                                 value, found)
        end if
        if (.not. found) then
            error_msg = 'EQUIVALENCE subscript is not a constant expression: '// &
                        trim(sub_text)
            return
        end if
        subscript = int(value)
    end subroutine fold_equivalence_subscript

    subroutine fold_equivalence_parameter_decl(arena, context, name, value, found)
        ! Fallback for a named constant FortFront cannot bind at the comment
        ! node that carries the EQUIVALENCE statement: fold the initialiser of
        ! the PARAMETER declaration that spells this name.
        use, intrinsic :: iso_c_binding, only: c_int64_t
        type(ast_arena_t), intent(in) :: arena
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: name
        integer(c_int64_t), intent(out) :: value
        logical, intent(out) :: found
        character(len=:), allocatable :: fold_error
        integer :: n

        value = 0_c_int64_t
        found = .false.
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (decl => arena%entries(n)%node)
            type is (declaration_node)
                if (.not. decl%is_parameter) cycle
                if (.not. decl%has_initializer) cycle
                if (decl%initializer_index <= 0) cycle
                if (.not. declaration_names_var(decl, name)) cycle
                call eval_i32_constant(arena, decl%initializer_index, context, &
                                       value, fold_error)
                if (len_trim(fold_error) > 0) return
                found = .true.
                return
            end select
        end do
    end subroutine fold_equivalence_parameter_decl

    subroutine split_equivalence_designators(text, items, count)
        ! Split the group list on commas that are not inside a subscript.
        character(len=*), intent(in) :: text
        character(len=:), allocatable, intent(out) :: items(:)
        integer, intent(out) :: count
        integer :: i, start, depth, max_len, n
        character(len=:), allocatable :: piece

        count = 0
        n = count_commas(text) + 1
        max_len = max(len(text), 1)
        allocate (character(len=max_len) :: items(n))
        items = ''
        start = 1
        depth = 0
        do i = 1, len(text) + 1
            if (i > len(text)) then
                piece = trim(adjustl(text(start:)))
            else if (text(i:i) == '(') then
                depth = depth + 1
                cycle
            else if (text(i:i) == ')') then
                depth = depth - 1
                cycle
            else if (text(i:i) == ',' .and. depth == 0) then
                if (i > start) then
                    piece = trim(adjustl(text(start:i - 1)))
                else
                    piece = ''
                end if
                start = i + 1
            else
                cycle
            end if
            if (len(piece) > 0) then
                count = count + 1
                items(count) = piece
            end if
        end do
    end subroutine split_equivalence_designators

    subroutine parse_equivalence_group(text, group, error_msg)
        ! "equivalence (a, b)" -> group="a, b". Only a single parenthesised group
        ! is supported.
        character(len=*), intent(in) :: text
        character(len=:), allocatable, intent(out) :: group
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: rest
        integer :: open_paren, close_paren

        call set_empty(error_msg)
        rest = adjustl(text)
        rest = adjustl(rest(12:))   ! drop the leading "equivalence"
        open_paren = index(rest, '(')
        close_paren = index(rest, ')', back=.true.)
        if (open_paren == 0 .or. close_paren <= open_paren) then
            error_msg = 'malformed EQUIVALENCE statement: '//text
            return
        end if
        group = rest(open_paren + 1:close_paren - 1)
    end subroutine parse_equivalence_group

    subroutine layout_equivalence_group(members, member_count, byte_size)
        ! Every member's designated element sits on the group's association
        ! point, so a member whose designator names element k starts that many
        ! bytes before it. The group spans from the earliest start to the last
        ! byte any member occupies.
        type(equiv_member_t), intent(inout) :: members(:)
        integer, intent(in) :: member_count
        integer, intent(out) :: byte_size
        integer :: i, anchor, span

        anchor = 0
        do i = 1, member_count
            if (members(i)%byte_offset > anchor) anchor = members(i)%byte_offset
        end do
        byte_size = 0
        do i = 1, member_count
            members(i)%start_offset = anchor - members(i)%byte_offset
            span = members(i)%start_offset + equivalence_member_size(members(i))
            if (span > byte_size) byte_size = span
        end do
        if (byte_size <= 0) byte_size = 4
    end subroutine layout_equivalence_group

    integer function equivalence_member_size(member) result(sz)
        ! Storage the member itself occupies, in bytes.
        type(equiv_member_t), intent(in) :: member

        sz = equivalence_kind_size(member%value_kind)
        if (member%is_array) sz = sz*max(member%array_size, 1)
    end function equivalence_member_size

    integer function equivalence_kind_size(value_kind) result(sz)
        integer, intent(in) :: value_kind
        select case (value_kind)
        case (VALUE_F64, VALUE_I64)
            sz = 8
        case default
            sz = 4
        end select
    end function equivalence_kind_size

    subroutine rebind_equivalence_symbol(group_index, member, sym, context, &
                                         error_msg)
        use, intrinsic :: iso_c_binding, only: c_char, c_int32_t, c_int64_t
        integer, intent(in) :: group_index
        type(equiv_member_t), intent(in) :: member
        integer, intent(in) :: sym
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: gname
        character(kind=c_char), allocatable :: c_name(:)
        integer(c_int32_t) :: global_id

        gname = equivalence_global_name(group_index)
        call to_c_chars(gname, c_name)
        global_id = lr_session_intern(context%session%handle, c_name)
        if (global_id < 0_c_int32_t) then
            error_msg = 'EQUIVALENCE global not found: '//gname
            return
        end if
        context%symbols(sym)%value_kind = member%value_kind
        context%symbols(sym)%has_address = .true.
        context%symbols(sym)%is_reference = .true.
        context%symbols(sym)%address%kind = LR_OP_KIND_GLOBAL
        context%symbols(sym)%address%payload = int(global_id, c_int64_t)
        context%symbols(sym)%address%typ = lr_type_ptr_s(context%session%handle)
        context%symbols(sym)%address%global_offset = &
            int(member%start_offset, c_int64_t)
        if (member%is_array) then
            context%symbols(sym)%is_array = .true.
            context%symbols(sym)%array_rank = 1
            context%symbols(sym)%array_size = member%array_size
            context%symbols(sym)%array_dim_sizes(1) = member%array_size
            context%symbols(sym)%array_dim_lowers(1) = 1
            ! Element GEP addresses off element_address, the same field a local
            ! array's alloca fills (session_program_lowering_array_elements.inc).
            context%symbols(sym)%element_address = context%symbols(sym)%address
        end if
        call set_empty(error_msg)
    end subroutine rebind_equivalence_symbol

    function equivalence_global_name(group_index) result(gname)
        integer, intent(in) :: group_index
        character(len=:), allocatable :: gname
        character(len=32) :: idx_text
        write (idx_text, '(I0)') group_index
        gname = '.ffc.equiv.'//trim(idx_text)
    end function equivalence_global_name
