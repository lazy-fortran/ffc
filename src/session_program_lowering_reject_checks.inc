    ! Semantic rejection checks for invalid programs that lower cleanly but are
    ! not conforming Fortran. Each check sets error_msg to a non-empty diagnostic
    ! so the driver exits nonzero. Checks fire only on statically certain
    ! violations, never on a form that could be valid in some context.

    ! Two specific procedures in the same generic interface are ambiguous when
    ! their dummy argument lists are not distinguishable by type, kind, or rank
    ! (F2018 C1514). This check flags only the provably indistinguishable case:
    ! equal argument count with every position a known-kind scalar of identical
    ! kind. Any argument whose type/kind/rank cannot be resolved statically, or
    ! any array argument (where rank could distinguish), leaves the pair unflagged
    ! so a valid generic is never rejected.
    subroutine check_generic_ambiguity(arena, context, generic_idx, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: generic_idx
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: sc, i, j
        character(len=:), allocatable :: name_i, name_j

        call set_empty(error_msg)
        sc = context%generics(generic_idx)%specific_count
        do i = 1, sc - 1
            name_i = trim(context%generics(generic_idx)%specific_names(i))
            do j = i + 1, sc
                name_j = trim(context%generics(generic_idx)%specific_names(j))
                if (specifics_indistinguishable(arena, name_i, name_j)) then
                    error_msg = 'ambiguous interfaces '//name_i//' and '// &
                        name_j//' in generic interface '// &
                        trim(context%generics(generic_idx)%generic_name)
                    return
                end if
            end do
        end do
    end subroutine check_generic_ambiguity

    logical function specifics_indistinguishable(arena, name_a, name_b) &
            result(same)
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: name_a, name_b
        integer :: count_a, count_b, pos, vk_a, vk_b
        logical :: char_a, char_b

        same = .false.
        count_a = arena_proc_param_count(arena, name_a)
        count_b = arena_proc_param_count(arena, name_b)
        if (count_a < 0 .or. count_b < 0) return
        if (count_a /= count_b) return
        do pos = 1, count_a
            if (callee_dummy_is_array(arena, name_a, pos)) return
            if (callee_dummy_is_array(arena, name_b, pos)) return
            char_a = callee_dummy_is_character(arena, name_a, pos)
            char_b = callee_dummy_is_character(arena, name_b, pos)
            if (char_a .neqv. char_b) return
            if (.not. char_a) then
                vk_a = callee_dummy_value_kind(arena, name_a, pos)
                vk_b = callee_dummy_value_kind(arena, name_b, pos)
                if (vk_a == 0 .or. vk_b == 0) return
                if (vk_a /= vk_b) return
            end if
        end do
        same = .true.
    end function specifics_indistinguishable

    integer function arena_proc_param_count(arena, proc_name) result(count)
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: proc_name
        integer :: n
        character(len=:), allocatable :: node_type
        type(function_def_node), pointer :: fn_node
        type(subroutine_def_node), pointer :: sb_node

        count = -1
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            node_type = get_node_type_at(arena, n)
            if (node_type == 'subroutine_def_node' .or. &
                node_type == 'subroutine_def') then
                sb_node => get_node_as_subroutine_def(arena, n)
                if (.not. associated(sb_node)) cycle
                if (.not. allocated(sb_node%name)) cycle
                if (.not. same_name(sb_node%name, proc_name)) cycle
                if (allocated(sb_node%param_indices)) then
                    count = size(sb_node%param_indices)
                else
                    count = 0
                end if
                return
            else if (node_type == 'function_def_node' .or. &
                     node_type == 'function_def') then
                fn_node => get_node_as_function_def(arena, n)
                if (.not. associated(fn_node)) cycle
                if (.not. allocated(fn_node%name)) cycle
                if (.not. same_name(fn_node%name, proc_name)) cycle
                if (allocated(fn_node%param_indices)) then
                    count = size(fn_node%param_indices)
                else
                    count = 0
                end if
                return
            end if
        end do
    end function arena_proc_param_count
