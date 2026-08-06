submodule (session_program_lowering_impl) session_program_lowering_reject_array_shape
    use session_program_lowering_reject_array_shape_order
    implicit none
contains

    ! Array indexing and shape-expression validation (F2018 9.5.3, 10.2.4,
    ! 11.1.7.4, C1010).
    !
    ! Four constraints of one family are enforced here, all before lowering:
    !   * a section-subscript or FORALL stride must not be zero,
    !   * a FORALL mask-expr must be a scalar LOGICAL expression,
    !   * a format specified as a character entity must not be a zero-sized
    !     array,
    !   * a pointer initialization target must be a named entity with the
    !     TARGET attribute, never a function or intrinsic result.
    !
    ! The stride rule runs twice: once over the typed nodes, so that a stride
    ! that folds to zero through a named constant is caught, and once over the
    ! source form, so that a literal zero stride is still rejected in a scope
    ! whose declarations the parser did not retain.

    subroutine check_array_shape_expressions(arena, error_msg)
        type(ast_arena_t), intent(in) :: arena
        character(len=:), allocatable, intent(out) :: error_msg

        call set_empty(error_msg)
        call check_zero_stride_nodes(arena, error_msg)
        if (len_trim(error_msg) > 0) return
        call check_zero_stride_source(arena, error_msg)
        if (len_trim(error_msg) > 0) return
        call check_forall_mask_scalar(arena, error_msg)
        if (len_trim(error_msg) > 0) return
        call check_zero_sized_format(arena, error_msg)
        if (len_trim(error_msg) > 0) return
        call check_pointer_init_targets(arena, error_msg)
    end subroutine check_array_shape_expressions

    ! A section subscript triplet stride whose constant value is zero is
    ! invalid: the section would have no defined extent.
    subroutine check_zero_stride_nodes(arena, error_msg)
        type(ast_arena_t), intent(in) :: arena
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: n, d, bidx

        call set_empty(error_msg)
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (nd => arena%entries(n)%node)
            type is (array_slice_node)
                do d = 1, min(nd%num_dimensions, size(nd%bounds_indices))
                    bidx = nd%bounds_indices(d)
                    call check_stride_node(arena, bidx, nd%line, nd%column, &
                                           error_msg)
                    if (len_trim(error_msg) > 0) return
                end do
            type is (forall_node)
                if (.not. allocated(nd%stride_indices)) cycle
                do d = 1, size(nd%stride_indices)
                    call check_stride_value(arena, nd%stride_indices(d), &
                                            nd%line, nd%column, error_msg)
                    if (len_trim(error_msg) > 0) return
                end do
            end select
        end do
    end subroutine check_zero_stride_nodes

    subroutine check_stride_node(arena, bounds_index, line, col, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: bounds_index
        integer, intent(in) :: line, col
        character(len=:), allocatable, intent(out) :: error_msg

        call set_empty(error_msg)
        if (bounds_index <= 0) return
        if (.not. node_exists(arena, bounds_index)) return
        select type (bd => arena%entries(bounds_index)%node)
        type is (range_expression_node)
            call check_stride_value(arena, bd%stride_index, line, col, error_msg)
        type is (array_bounds_node)
            call check_stride_value(arena, bd%stride_index, line, col, error_msg)
        end select
    end subroutine check_stride_node

    subroutine check_stride_value(arena, stride_index, line, col, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: stride_index
        integer, intent(in) :: line, col
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=64) :: location
        integer(c_int64_t) :: value
        logical :: ok

        call set_empty(error_msg)
        if (stride_index <= 0) return
        if (.not. node_exists(arena, stride_index)) return
        call try_const_int64(arena, stride_index, value, ok)
        if (.not. ok) call try_named_constant_int64(arena, stride_index, value, ok)
        if (.not. ok) return
        if (value /= 0_c_int64_t) return
        write (location, '(" at line ",I0,", column ",I0)') line, col
        error_msg = 'Illegal stride of zero'//trim(location)
    end subroutine check_stride_value

    ! Fold a reference to a named constant whose initializer is itself a
    ! constant integer expression.
    subroutine try_named_constant_int64(arena, idx, value, ok)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: idx
        integer(c_int64_t), intent(out) :: value
        logical, intent(out) :: ok
        character(len=:), allocatable :: wanted
        integer :: n

        value = 0_c_int64_t
        ok = .false.
        if (idx <= 0) return
        if (.not. node_exists(arena, idx)) return
        select type (nd => arena%entries(idx)%node)
        type is (identifier_node)
            wanted = trim(lowercase_text(nd%name))
        class default
            return
        end select
        if (len(wanted) == 0) return
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (nd => arena%entries(n)%node)
            type is (declaration_node)
                if (.not. nd%is_parameter) cycle
                if (nd%is_array) cycle
                if (.not. nd%has_initializer) cycle
                if (nd%initializer_index <= 0) cycle
                if (.not. allocated(nd%var_name)) cycle
                if (trim(lowercase_text(nd%var_name)) /= wanted) cycle
                call try_const_int64(arena, nd%initializer_index, value, ok)
                return
            end select
        end do
    end subroutine try_named_constant_int64

    ! Source-form backstop for the stride rule. A triplet whose third part is a
    ! literal zero is invalid however the enclosing statement parsed.
    subroutine check_zero_stride_source(arena, error_msg)
        type(ast_arena_t), intent(in) :: arena
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: source, line, code
        character(len=64) :: location
        integer :: pos, next_nl, line_no
        logical :: found

        call set_empty(error_msg)
        call get_source_text(arena, source, found)
        if (.not. found) return
        pos = 1
        line_no = 0
        do while (pos <= len(source))
            next_nl = index(source(pos:), new_line('a'))
            if (next_nl == 0) then
                line = source(pos:)
                pos = len(source) + 1
            else
                line = source(pos:pos + next_nl - 2)
                pos = pos + next_nl
            end if
            line_no = line_no + 1
            code = blank_strings_and_comment(line)
            if (line_has_zero_stride(code)) then
                write (location, '(" at line ",I0)') line_no
                error_msg = 'Illegal stride of zero'//trim(location)
                return
            end if
        end do
    end subroutine check_zero_stride_source

    ! Replace character-literal contents and any trailing comment by blanks so
    ! that only executable punctuation survives.
    function blank_strings_and_comment(line) result(code)
        character(len=*), intent(in) :: line
        character(len=:), allocatable :: code
        character(len=1) :: quote
        integer :: i
        logical :: in_string

        code = line
        in_string = .false.
        quote = ' '
        do i = 1, len(line)
            if (in_string) then
                code(i:i) = ' '
                if (line(i:i) == quote) in_string = .false.
            else if (line(i:i) == '''' .or. line(i:i) == '"') then
                quote = line(i:i)
                in_string = .true.
                code(i:i) = ' '
            else if (line(i:i) == '!') then
                code(i:) = ' '
                return
            end if
        end do
    end function blank_strings_and_comment

    ! Only a parenthesised list that directly follows a name can hold section
    ! subscripts or a FORALL triplet; array constructors and control-statement
    ! parentheses are skipped.
    logical function line_has_zero_stride(code) result(found)
        character(len=*), intent(in) :: code
        integer :: i, j, depth

        found = .false.
        i = 2
        do while (i <= len(code))
            if (code(i:i) == '(') then
                if (is_fortran_identifier_char(code(i - 1:i - 1))) then
                    depth = 0
                    j = i
                    do while (j <= len(code))
                        if (code(j:j) == '(') depth = depth + 1
                        if (code(j:j) == ')') then
                            depth = depth - 1
                            if (depth == 0) exit
                        end if
                        j = j + 1
                    end do
                    if (j > len(code)) return
                    if (j > i + 1) then
                        found = group_has_zero_stride(code(i + 1:j - 1))
                        if (found) return
                    end if
                    i = j
                end if
            end if
            i = i + 1
        end do
    end function line_has_zero_stride

    logical function group_has_zero_stride(inner) result(found)
        character(len=*), intent(in) :: inner
        integer :: i, depth, start

        found = .false.
        depth = 0
        start = 1
        do i = 1, len(inner)
            if (inner(i:i) == '(' .or. inner(i:i) == '[') depth = depth + 1
            if (inner(i:i) == ')' .or. inner(i:i) == ']') depth = depth - 1
            if (inner(i:i) == ',' .and. depth == 0) then
                if (i > start) found = part_has_zero_stride(inner(start:i - 1))
                if (found) return
                start = i + 1
            end if
        end do
        if (len(inner) >= start) found = part_has_zero_stride(inner(start:))
    end function group_has_zero_stride

    logical function part_has_zero_stride(part) result(found)
        character(len=*), intent(in) :: part
        integer :: i, depth, ncolon, second

        found = .false.
        if (index(part, '::') > 0) return
        depth = 0
        ncolon = 0
        second = 0
        do i = 1, len(part)
            if (part(i:i) == '(' .or. part(i:i) == '[') depth = depth + 1
            if (part(i:i) == ')' .or. part(i:i) == ']') depth = depth - 1
            if (part(i:i) == ':' .and. depth == 0) then
                ncolon = ncolon + 1
                if (ncolon == 2) second = i
            end if
        end do
        if (ncolon /= 2) return
        if (second >= len(part)) return
        found = is_zero_literal(part(second + 1:))
    end function part_has_zero_stride

    logical function is_zero_literal(text) result(is_zero)
        character(len=*), intent(in) :: text
        character(len=:), allocatable :: body
        integer :: i

        is_zero = .false.
        body = trim(adjustl(text))
        if (len(body) == 0) return
        if (body(1:1) == '+' .or. body(1:1) == '-') body = body(2:)
        if (len(body) == 0) return
        do i = 1, len(body)
            if (body(i:i) /= '0') return
        end do
        is_zero = .true.
    end function is_zero_literal

    ! F2018 11.1.7.4: the FORALL mask-expr is a scalar logical expression. A
    ! whole array reference in that position is invalid.
    subroutine check_forall_mask_scalar(arena, error_msg)
        type(ast_arena_t), intent(in) :: arena
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=64) :: location
        integer :: n

        call set_empty(error_msg)
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (nd => arena%entries(n)%node)
            type is (forall_node)
                if (.not. nd%has_mask) cycle
                if (nd%mask_expr_index <= 0) cycle
                if (.not. expr_is_array_valued(arena, nd%mask_expr_index)) cycle
                write (location, '(" at line ",I0,", column ",I0)') &
                    nd%line, nd%column
                error_msg = 'FORALL mask requires a scalar LOGICAL '// &
                    'expression'//trim(location)
                return
            end select
        end do
    end subroutine check_forall_mask_scalar

    logical function expr_is_array_valued(arena, idx) result(is_array)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: idx
        logical :: found, is_target, zero_sized

        is_array = .false.
        if (idx <= 0) return
        if (.not. node_exists(arena, idx)) return
        select type (nd => arena%entries(idx)%node)
        type is (array_literal_node)
            is_array = .true.
        type is (identifier_node)
            call named_declaration_facts(arena, nd%name, found, is_array, &
                                         is_target, zero_sized)
            if (.not. found) is_array = .false.
        end select
    end function expr_is_array_valued

    ! F2018 12.6.2.2: a format given as a character entity supplies the format
    ! string; a zero-sized array carries no format at all.
    subroutine check_zero_sized_format(arena, error_msg)
        type(ast_arena_t), intent(in) :: arena
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=64) :: location
        integer :: n

        call set_empty(error_msg)
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (nd => arena%entries(n)%node)
            type is (print_statement_node)
                if (.not. allocated(nd%format_spec)) cycle
                if (.not. named_format_is_zero_sized(arena, nd%format_spec)) cycle
                write (location, '(" at line ",I0,", column ",I0)') &
                    nd%line, nd%column
                error_msg = 'format specifier is a zero-sized array'// &
                    trim(location)
                return
            end select
        end do
    end subroutine check_zero_sized_format

    logical function named_format_is_zero_sized(arena, spec) result(is_zero)
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: spec
        character(len=:), allocatable :: name
        logical :: found, is_array, is_target
        integer :: i

        is_zero = .false.
        name = trim(adjustl(spec))
        if (len(name) == 0) return
        do i = 1, len(name)
            if (.not. is_fortran_identifier_char(name(i:i))) return
        end do
        if (scan(name(1:1), '0123456789_') /= 0) return
        call named_declaration_facts(arena, name, found, is_array, is_target, &
                                     is_zero)
        if (.not. found) is_zero = .false.
        if (.not. is_array) is_zero = .false.
    end function named_format_is_zero_sized

    ! F2018 C1010: the target of a pointer initialization must be a named
    ! entity with the TARGET or POINTER attribute and the SAVE attribute; a
    ! function or intrinsic result never qualifies.
    subroutine check_pointer_init_targets(arena, error_msg)
        type(ast_arena_t), intent(in) :: arena
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=64) :: location
        integer :: n

        call set_empty(error_msg)
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (nd => arena%entries(n)%node)
            type is (declaration_node)
                if (.not. nd%is_pointer) cycle
                if (.not. nd%has_initializer) cycle
                if (nd%initializer_index <= 0) cycle
                if (pointer_init_target_ok(arena, nd%initializer_index)) cycle
                write (location, '(" at line ",I0,", column ",I0)') &
                    nd%line, nd%column
                error_msg = 'pointer initialization target does not have '// &
                    'the TARGET attribute'//trim(location)
                return
            end select
        end do
    end subroutine check_pointer_init_targets

    logical function pointer_init_target_ok(arena, idx) result(ok)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: idx

        ok = .true.
        if (idx <= 0) return
        if (.not. node_exists(arena, idx)) return
        select type (nd => arena%entries(idx)%node)
        type is (identifier_node)
            ok = named_target_ok(arena, nd%name)
        type is (call_or_subscript_node)
            ok = named_target_ok(arena, nd%name)
        end select
    end function pointer_init_target_ok

    logical function named_target_ok(arena, name) result(ok)
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: name
        logical :: found, is_array, is_target, zero_sized

        ok = .true.
        if (len_trim(name) == 0) return
        if (trim(lowercase_text(name)) == 'null') return
        call named_declaration_facts(arena, name, found, is_array, is_target, &
                                     zero_sized)
        if (.not. found) then
            ! No declared entity of that name: the initializer references a
            ! function or intrinsic, whose result is never a valid target.
            ok = .false.
            return
        end if
        ok = is_target
    end function named_target_ok

    ! Collect the attributes of the declaration of NAME anywhere in the unit.
    subroutine named_declaration_facts(arena, name, found, is_array, is_target, &
                                       zero_sized)
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: name
        logical, intent(out) :: found, is_array, is_target, zero_sized
        character(len=:), allocatable :: wanted
        integer :: n, v

        found = .false.
        is_array = .false.
        is_target = .false.
        zero_sized = .false.
        wanted = trim(lowercase_text(name))
        if (len(wanted) == 0) return
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (nd => arena%entries(n)%node)
            type is (declaration_node)
                if (nd%is_multi_declaration) then
                    if (.not. allocated(nd%var_names)) cycle
                    found = .false.
                    do v = 1, size(nd%var_names)
                        if (trim(lowercase_text(nd%var_names(v))) == wanted) then
                            found = .true.
                        end if
                    end do
                    if (.not. found) cycle
                else
                    if (.not. allocated(nd%var_name)) cycle
                    if (trim(lowercase_text(nd%var_name)) /= wanted) cycle
                    found = .true.
                end if
                is_array = nd%is_array
                is_target = nd%is_target .or. nd%is_pointer
                zero_sized = declaration_is_zero_sized(arena, nd)
                return
            end select
        end do
    end subroutine named_declaration_facts

    logical function declaration_is_zero_sized(arena, decl) result(is_zero)
        type(ast_arena_t), intent(in) :: arena
        type(declaration_node), intent(in) :: decl
        integer(c_int64_t) :: value
        logical :: ok
        integer :: d

        is_zero = .false.
        if (.not. decl%is_array) return
        if (.not. allocated(decl%dimension_indices)) return
        do d = 1, size(decl%dimension_indices)
            if (decl%dimension_indices(d) <= 0) cycle
            call try_const_int64(arena, decl%dimension_indices(d), value, ok)
            if (.not. ok) cycle
            if (value == 0_c_int64_t) then
                is_zero = .true.
                return
            end if
        end do
    end function declaration_is_zero_sized

end submodule session_program_lowering_reject_array_shape
