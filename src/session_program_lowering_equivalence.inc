    ! EQUIVALENCE storage overlay (#280, issue_1745). FortFront leaves an
    ! EQUIVALENCE statement as a comment_node ("equivalence (i, r)"), the same as
    ! COMMON. The members of one EQUIVALENCE group name the same storage, so a
    ! write through one member is observable, bit-for-bit, through another.
    !
    ! Model: emit one zero-initialised global per group, sized to the largest
    ! member, and bind every member as an addressed reference into it. Each
    ! member keeps its own value_kind, so a load/store uses that member's type at
    ! the shared address - reading an integer back as a real reinterprets the
    ! bits exactly as gfortran does.
    !
    ! Scope: scalar integer(4)/integer(8)/real(4)/real(8)/logical members of a
    ! single (a, b, ...) group. Arrays, character, and offset associations
    ! (substrings, array elements) leave the symbols plain locals and surface the
    ! usual diagnostics rather than miscompiling.

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
        character(len=:), allocatable :: names(:)
        integer :: kinds(EQUIV_MAX_MEMBERS)
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
                call collect_equivalence_members(arena, node%text, names, kinds, &
                                                  member_count, error_msg)
                if (len_trim(error_msg) > 0) return
                if (member_count == 0) cycle
                byte_size = equivalence_group_size(kinds, member_count)
                allocate(zero_bytes(byte_size))
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
                deallocate(zero_bytes)
                if (global_id < 0_c_int32_t) then
                    error_msg = 'LIRIC could not create EQUIVALENCE global: '//gname
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
        character(len=:), allocatable :: names(:)
        integer :: kinds(EQUIV_MAX_MEMBERS)
        integer :: member_count, group_index, n, i, sym

        call set_empty(error_msg)
        group_index = 0
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (node => arena%entries(n)%node)
            type is (comment_node)
                if (.not. is_equivalence_text(node%text)) cycle
                group_index = group_index + 1
                call collect_equivalence_members(arena, node%text, names, kinds, &
                                                  member_count, error_msg)
                if (len_trim(error_msg) > 0) return
                do i = 1, member_count
                    sym = find_symbol(context, trim(names(i)))
                    if (sym <= 0) cycle
                    call rebind_equivalence_symbol(group_index, kinds(i), sym, &
                                                   context, error_msg)
                    if (len_trim(error_msg) > 0) return
                end do
            end select
        end do
    end subroutine bind_equivalence_symbols

    subroutine collect_equivalence_members(arena, text, names, kinds, &
                                           member_count, error_msg)
        ! Parse "equivalence (a, b, ...)" into member names and their kinds,
        ! taken from each member's scalar program declaration.
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: text
        character(len=:), allocatable, intent(out) :: names(:)
        integer, intent(out) :: kinds(:)
        integer, intent(out) :: member_count
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: group
        integer :: i

        call set_empty(error_msg)
        member_count = 0
        call parse_equivalence_group(text, group, error_msg)
        if (len_trim(error_msg) > 0) return
        call split_csv(group, names, member_count)
        if (member_count < 2) then
            error_msg = 'EQUIVALENCE group needs at least two members: '//text
            return
        end if
        if (member_count > size(kinds)) then
            error_msg = 'too many EQUIVALENCE members for direct lowering'
            return
        end if
        do i = 1, member_count
            call program_declaration_kind(arena, trim(names(i)), kinds(i), &
                                          error_msg)
            if (len_trim(error_msg) > 0) return
        end do
    end subroutine collect_equivalence_members

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

    integer function equivalence_group_size(kinds, member_count) result(byte_size)
        ! The group occupies storage as large as its largest member.
        integer, intent(in) :: kinds(:)
        integer, intent(in) :: member_count
        integer :: i, sz
        byte_size = 0
        do i = 1, member_count
            sz = equivalence_kind_size(kinds(i))
            if (sz > byte_size) byte_size = sz
        end do
        if (byte_size <= 0) byte_size = 4
    end function equivalence_group_size

    integer function equivalence_kind_size(value_kind) result(sz)
        integer, intent(in) :: value_kind
        select case (value_kind)
        case (VALUE_F64, VALUE_I64)
            sz = 8
        case default
            sz = 4
        end select
    end function equivalence_kind_size

    subroutine rebind_equivalence_symbol(group_index, value_kind, sym, context, &
                                         error_msg)
        use, intrinsic :: iso_c_binding, only: c_char, c_int32_t, c_int64_t
        integer, intent(in) :: group_index, value_kind, sym
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
        context%symbols(sym)%value_kind = value_kind
        context%symbols(sym)%has_address = .true.
        context%symbols(sym)%is_reference = .true.
        context%symbols(sym)%address%kind = LR_OP_KIND_GLOBAL
        context%symbols(sym)%address%payload = int(global_id, c_int64_t)
        context%symbols(sym)%address%typ = lr_type_ptr_s(context%session%handle)
        context%symbols(sym)%address%global_offset = 0_c_int64_t
        call set_empty(error_msg)
    end subroutine rebind_equivalence_symbol

    function equivalence_global_name(group_index) result(gname)
        integer, intent(in) :: group_index
        character(len=:), allocatable :: gname
        character(len=32) :: idx_text
        write (idx_text, '(I0)') group_index
        gname = '.ffc.equiv.'//trim(idx_text)
    end function equivalence_global_name
