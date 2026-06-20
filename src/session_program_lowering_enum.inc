    ! ENUM / ENUMERATOR lowering (issue #1826).
    !
    ! FortFront does not build a structured node for `enum ... end enum`: it
    ! consumes only the `enum` keyword into an error_node and discards the
    ! enumerator declarations. We therefore recover the enumerators from the
    ! source text attached to the arena, starting at the error node's line, and
    ! bind each as an integer named constant (default kind c_int). Values follow
    ! the standard rule: explicit initializer, else previous + 1, starting at 0.
    subroutine lower_enum_block(arena, node_index, context, error_msg, handled)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        logical, intent(out) :: handled
        character(len=:), allocatable :: line_text, head
        logical :: found
        integer :: line_no
        integer(c_int64_t) :: next_value

        handled = .false.
        call set_empty(error_msg)
        line_no = get_node_line(arena, node_index)
        if (line_no <= 0) return
        call get_source_line(arena, line_no, line_text, found)
        if (.not. found) return
        head = enum_first_word(line_text)
        ! Only treat this error_node as an enum opener; other error_nodes keep
        ! their existing unsupported-feature diagnostic.
        if (head /= 'enum') return

        handled = .true.
        next_value = 0_c_int64_t
        line_no = line_no + 1
        do
            call get_source_line(arena, line_no, line_text, found)
            if (.not. found) then
                error_msg = 'enum block is missing end enum'
                return
            end if
            head = enum_first_word(line_text)
            if (head == 'end') return
            if (head == 'enumerator') then
                call bind_enumerator_line(line_text, context, next_value, error_msg)
                if (len_trim(error_msg) > 0) return
            end if
            line_no = line_no + 1
        end do
    end subroutine lower_enum_block

    ! Parse one `enumerator :: NAME [= value] [, NAME [= value]]...` line and
    ! bind each name as an integer parameter. next_value carries the implicit
    ! counter across names and across statements.
    subroutine bind_enumerator_line(line_text, context, next_value, error_msg)
        character(len=*), intent(in) :: line_text
        type(lowering_context_t), intent(inout) :: context
        integer(c_int64_t), intent(inout) :: next_value
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: rest, item, name, value_text
        integer :: eq, comma
        integer(c_int64_t) :: this_value
        logical :: ok

        call set_empty(error_msg)
        rest = enum_after_double_colon(line_text)
        if (len_trim(rest) == 0) then
            error_msg = 'enumerator statement has no names'
            return
        end if
        do
            comma = index(rest, ',')
            if (comma > 0) then
                item = rest(1:comma - 1)
                rest = rest(comma + 1:)
            else
                item = rest
                rest = ''
            end if
            eq = index(item, '=')
            if (eq > 0) then
                name = trim(adjustl(item(1:eq - 1)))
                value_text = trim(adjustl(item(eq + 1:)))
                call parse_enum_value(value_text, this_value, ok)
                if (.not. ok) then
                    error_msg = 'unsupported enumerator initializer: '//value_text
                    return
                end if
            else
                name = trim(adjustl(item))
                this_value = next_value
            end if
            if (len_trim(name) > 0) then
                call bind_enum_constant(context, name, this_value, error_msg)
                if (len_trim(error_msg) > 0) return
            end if
            next_value = this_value + 1_c_int64_t
            if (len_trim(rest) == 0) exit
        end do
    end subroutine bind_enumerator_line

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

    ! Lowercased first whitespace-delimited word of a source line.
    function enum_first_word(line_text) result(word)
        character(len=*), intent(in) :: line_text
        character(len=:), allocatable :: word
        character(len=:), allocatable :: trimmed
        integer :: sp

        trimmed = adjustl(line_text)
        sp = scan(trimmed, ' ,(')
        if (sp > 0) then
            word = lowercase_text(trimmed(1:sp - 1))
        else
            word = lowercase_text(trim(trimmed))
        end if
    end function enum_first_word

    ! Text after `::` on an enumerator line; empty if there is none.
    function enum_after_double_colon(line_text) result(rest)
        character(len=*), intent(in) :: line_text
        character(len=:), allocatable :: rest
        integer :: pos

        pos = index(line_text, '::')
        if (pos > 0) then
            rest = adjustl(line_text(pos + 2:))
        else
            rest = ''
        end if
        ! Drop a trailing comment so the last enumerator value parses cleanly.
        pos = index(rest, '!')
        if (pos > 0) rest = rest(1:pos - 1)
    end function enum_after_double_colon

    ! Parse a signed decimal integer enumerator initializer. Non-literal
    ! initializers (named constants, kinds) are reported unsupported by caller.
    subroutine parse_enum_value(text, value, ok)
        character(len=*), intent(in) :: text
        integer(c_int64_t), intent(out) :: value
        logical, intent(out) :: ok
        character(len=:), allocatable :: trimmed
        integer :: ios, i

        value = 0_c_int64_t
        ok = .false.
        trimmed = trim(adjustl(text))
        if (len(trimmed) == 0) return
        do i = 1, len(trimmed)
            if (i == 1 .and. (trimmed(i:i) == '-' .or. trimmed(i:i) == '+')) cycle
            if (trimmed(i:i) < '0' .or. trimmed(i:i) > '9') return
        end do
        read (trimmed, *, iostat=ios) value
        ok = ios == 0
    end subroutine parse_enum_value
