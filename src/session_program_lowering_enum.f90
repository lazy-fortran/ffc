submodule (session_program_lowering_impl) session_program_lowering_enum
    use session_program_lowering_enum_order
    implicit none
contains
    ! ENUM / ENUMERATOR lowering (issue #1826).
    !
    ! FortFront parses `enum ... end enum` into an enum_node carrying the
    ! enumerator names and their resolved integer values (the standard rule -
    ! explicit initializer, else previous + 1, starting at 0 - is applied
    ! during parsing). This pass binds each enumerator as an integer named
    ! constant (default kind c_int) so later expressions, print, and case use
    ! resolve it to a folded i32 constant.
    module procedure lower_enum_block
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
                call bind_enum_constant(context, &
                                        trim(node%enumerator_names(i)%s), &
                                        value, error_msg)
                if (len_trim(error_msg) > 0) return
            end do
        end select
    end procedure lower_enum_block

    module procedure enumerator_value
        ! Resolved value of enumerator idx; falls back to the implicit
        ! previous + 1 rule when the parser left the value array short.
        if (allocated(node%enumerator_values)) then
            if (idx <= size(node%enumerator_values)) then
                value = int(node%enumerator_values(idx), c_int64_t)
                return
            end if
        end if
        value = int(idx - 1, c_int64_t)
    end procedure enumerator_value

    module procedure bind_enum_constant
        ! Register one enumerator as an integer named constant, matching the
        ! integer-parameter symbol shape so later expressions, print, and case
        ! use resolve it to a folded i32 constant.
        integer :: index

        call set_empty(error_msg)
        if (find_symbol_compat(context, name) > 0) then
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
    end procedure bind_enum_constant
end submodule session_program_lowering_enum
