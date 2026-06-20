    ! ENUM / ENUMERATOR lowering (issue #1826).
    !
    ! FortFront parses `enum ... end enum` into an enum_node carrying the
    ! enumerator names and their resolved integer values (the standard rule -
    ! explicit initializer, else previous + 1, starting at 0 - is applied during
    ! parsing). This pass binds each enumerator as an integer named constant
    ! (default kind c_int) so later expressions, print, and case use resolve it
    ! to a folded i32 constant.
    subroutine lower_enum_block(arena, node_index, context, error_msg, handled)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        logical, intent(out) :: handled
        integer :: i
        integer(c_int64_t) :: value

        handled = .false.
        call set_empty(error_msg)
        if (.not. node_exists(arena, node_index)) return
        select type (node => arena%entries(node_index)%node)
        type is (enum_node)
            handled = .true.
            if (.not. allocated(node%enumerator_names)) return
            do i = 1, size(node%enumerator_names)
                value = enumerator_value(node, i)
                call bind_enum_constant(context, trim(node%enumerator_names(i)%s), &
                                        value, error_msg)
                if (len_trim(error_msg) > 0) return
            end do
        end select
    end subroutine lower_enum_block

    function enumerator_value(node, idx) result(value)
        ! Resolved value of enumerator idx; falls back to the implicit
        ! previous + 1 rule when the parser left the value array short.
        type(enum_node), intent(in) :: node
        integer, intent(in) :: idx
        integer(c_int64_t) :: value

        if (allocated(node%enumerator_values)) then
            if (idx <= size(node%enumerator_values)) then
                value = int(node%enumerator_values(idx), c_int64_t)
                return
            end if
        end if
        value = int(idx - 1, c_int64_t)
    end function enumerator_value

    ! Register one enumerator as an integer named constant, matching the
    ! integer-parameter symbol shape so later expressions, print, and case use
    ! resolve it to a folded i32 constant.
    subroutine bind_enum_constant(context, name, value, error_msg)
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: name
        integer(c_int64_t), intent(in) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: index

        call set_empty(error_msg)
        if (find_symbol(context, name) > 0) then
            error_msg = 'duplicate enumerator declaration: '//trim(name)
            return
        end if
        call grow_symbols(context)
        index = context%symbol_count + 1
        context%symbols(index)%name = trim(name)
        context%symbols(index)%value_kind = VALUE_I32
        context%symbols(index)%value = i32_immediate(context%session, value)
        context%symbols(index)%is_parameter = .true.
        context%symbols(index)%has_i32_constant = .true.
        context%symbols(index)%i32_constant = value
        context%symbol_count = index
    end subroutine bind_enum_constant
