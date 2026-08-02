submodule (session_program_lowering) session_program_lowering_reject_purity
    use session_program_lowering_reject_purity_order
    implicit none
contains
    ! PURE and ELEMENTAL attribute restrictions (#578).
    !
    ! Rules enforced:
    !   * F2018 C1518: a procedure pointer or a dummy procedure shall not be
    !     elemental. The interface named by a PROCEDURE declaration is
    !     elemental when the referenced interface body carries the ELEMENTAL
    !     prefix or names an elemental intrinsic; an interface body declaring
    !     a dummy argument of the host procedure is checked directly.
    !   * F2018 C1589: a local variable of a PURE subprogram shall not have
    !     the SAVE or VOLATILE attribute.
    !
    ! Both rules read declaration statements, several of which FortFront does
    ! not surface as typed nodes (a PROCEDURE declaration with an interface
    ! name, an interface body naming a dummy argument), so the scan works on
    ! the unit's own comment-stripped source text. Every rule fires only on a
    ! statement whose first line already shows the violation, so a
    ! continuation can only make a check silent, never wrong.

    module procedure check_purity_attribute_restrictions
        character(len=256), allocatable :: lines(:)
        character(len=64) :: elemental_names(256)
        integer :: line_count, elemental_count
        logical :: found

        call set_empty(error_msg)
        call storage_source_lines(arena, lines, line_count, found)
        if (.not. found) return
        call collect_elemental_procedure_names(lines, line_count, &
                                               elemental_names, elemental_count)
        call scan_purity_statements(lines, line_count, elemental_names, &
                                    elemental_count, error_msg)
    end procedure check_purity_attribute_restrictions

    ! Names of every procedure declared with an ELEMENTAL prefix anywhere in
    ! the unit, including interface bodies. Used to decide whether a
    ! PROCEDURE(name) declaration names an elemental interface.
    subroutine collect_elemental_procedure_names(lines, line_count, names, count)
        character(len=*), intent(in) :: lines(:)
        integer, intent(in) :: line_count
        character(len=*), intent(out) :: names(:)
        integer, intent(out) :: count
        character(len=:), allocatable :: name, args
        integer :: i
        logical :: is_proc, is_pure, is_elemental

        count = 0
        do i = 1, line_count
            call parse_procedure_statement(trim(lines(i)), is_proc, is_pure, &
                                           is_elemental, name, args)
            if (.not. is_proc) cycle
            if (.not. is_elemental) cycle
            call append_storage_name(names, count, name)
        end do
    end subroutine collect_elemental_procedure_names

    ! Walk the statements once, tracking the innermost open procedure scope so
    ! each rule sees the host's dummy argument list and PURE status.
    subroutine scan_purity_statements(lines, line_count, elemental_names, &
                                      elemental_count, error_msg)
        character(len=*), intent(in) :: lines(:)
        integer, intent(in) :: line_count
        character(len=*), intent(in) :: elemental_names(:)
        integer, intent(in) :: elemental_count
        character(len=:), allocatable, intent(out) :: error_msg
        integer, parameter :: max_depth = 32
        character(len=1024) :: scope_args(max_depth)
        logical :: scope_pure(max_depth)
        character(len=:), allocatable :: line, name, args, host_args
        integer :: i, depth, type_depth
        logical :: is_proc, is_pure, is_elemental, host_pure

        call set_empty(error_msg)
        depth = 0
        type_depth = 0
        scope_args = ''
        scope_pure = .false.
        do i = 1, line_count
            line = trim(lines(i))
            if (len(line) == 0) cycle
            ! A procedure pointer component of a derived type may name an
            ! elemental interface, so C1518 does not reach into a type
            ! definition body.
            if (purity_starts_derived_type(line)) then
                type_depth = type_depth + 1
                cycle
            end if
            if (purity_word_at(line, 'end', 1)) then
                if (purity_word_at(adjustl(line(4:)), 'type', 1)) then
                    if (type_depth > 0) type_depth = type_depth - 1
                    cycle
                end if
            end if
            if (type_depth > 0) cycle
            if (purity_ends_procedure(line)) then
                if (depth > 0) depth = depth - 1
                cycle
            end if
            host_args = ''
            host_pure = .false.
            if (depth > 0) then
                host_args = trim(scope_args(depth))
                host_pure = scope_pure(depth)
            end if
            call parse_procedure_statement(line, is_proc, is_pure, &
                                           is_elemental, name, args)
            if (is_proc) then
                if (is_elemental) then
                    if (purity_arg_listed(host_args, name)) then
                        error_msg = 'dummy procedure '''//name// &
                            ''' shall not be elemental'//purity_at_line(i)
                        return
                    end if
                end if
                if (depth < max_depth) then
                    depth = depth + 1
                    scope_args(depth) = args
                    scope_pure(depth) = is_pure
                end if
                cycle
            end if
            call check_procedure_declaration_purity(line, i, host_args, &
                                                    elemental_names, &
                                                    elemental_count, error_msg)
            if (len_trim(error_msg) > 0) return
            if (host_pure) then
                call check_pure_local_attributes(line, i, error_msg)
                if (len_trim(error_msg) > 0) return
            end if
        end do
    end subroutine scan_purity_statements

    ! A PROCEDURE(interface) declaration whose interface is elemental may
    ! declare neither a procedure pointer nor a dummy procedure (C1518). A
    ! plain external declaration of the same form stays valid.
    subroutine check_procedure_declaration_purity(line, line_no, host_args, &
                                                  elemental_names, &
                                                  elemental_count, error_msg)
        character(len=*), intent(in) :: line
        integer, intent(in) :: line_no
        character(len=*), intent(in) :: host_args
        character(len=*), intent(in) :: elemental_names(:)
        integer, intent(in) :: elemental_count
        character(len=:), allocatable, intent(inout) :: error_msg
        character(len=:), allocatable :: iface, attrs, rest, name
        integer :: open_paren, close_paren, dc_pos, start, comma

        if (.not. purity_word_at(line, 'procedure', 1)) return
        open_paren = index(line, '(')
        if (open_paren == 0) return
        if (len_trim(line(10:open_paren - 1)) /= 0) return
        close_paren = purity_matching_paren(line, open_paren)
        if (close_paren == 0) return
        iface = trim(adjustl(line(open_paren + 1:close_paren - 1)))
        if (len(iface) == 0) return
        if (.not. purity_interface_is_elemental(iface, elemental_names, &
                                                elemental_count)) return
        dc_pos = index(line(close_paren:), '::')
        if (dc_pos == 0) return
        dc_pos = close_paren + dc_pos - 1
        attrs = line(close_paren + 1:dc_pos - 1)
        if (purity_has_word(attrs, 'pointer')) then
            error_msg = 'procedure pointer declared with elemental interface '''// &
                iface//''' shall not be elemental'//purity_at_line(line_no)
            return
        end if
        rest = line(dc_pos + 2:)//','
        start = 1
        do
            comma = index(rest(start:), ',')
            if (comma == 0) exit
            name = trim(adjustl(rest(start:start + comma - 2)))
            if (len(name) > 0) then
                if (purity_arg_listed(host_args, name)) then
                    error_msg = 'dummy procedure '''//name// &
                        ''' shall not be elemental'//purity_at_line(line_no)
                    return
                end if
            end if
            start = start + comma
            if (start > len(rest)) exit
        end do
    end subroutine check_procedure_declaration_purity

    ! SAVE or VOLATILE specified on a declaration inside a PURE subprogram
    ! (C1589). Only an explicit attribute or attribute statement is flagged.
    subroutine check_pure_local_attributes(line, line_no, error_msg)
        character(len=*), intent(in) :: line
        integer, intent(in) :: line_no
        character(len=:), allocatable, intent(inout) :: error_msg
        character(len=:), allocatable :: attrs
        integer :: dc_pos

        if (purity_word_at(line, 'save', 1)) then
            error_msg = 'SAVE attribute cannot be specified in a PURE '// &
                'procedure'//purity_at_line(line_no)
            return
        end if
        if (purity_word_at(line, 'volatile', 1)) then
            error_msg = 'VOLATILE attribute cannot be specified in a PURE '// &
                'procedure'//purity_at_line(line_no)
            return
        end if
        dc_pos = index(line, '::')
        if (dc_pos <= 1) return
        attrs = line(:dc_pos - 1)
        if (purity_has_word(attrs, 'save')) then
            error_msg = 'SAVE attribute cannot be specified in a PURE '// &
                'procedure'//purity_at_line(line_no)
            return
        end if
        if (purity_has_word(attrs, 'volatile')) then
            error_msg = 'VOLATILE attribute cannot be specified in a PURE '// &
                'procedure'//purity_at_line(line_no)
        end if
    end subroutine check_pure_local_attributes

    ! Decompose a SUBROUTINE or FUNCTION statement into its purity prefix,
    ! name, and dummy argument list. is_proc is false for any statement that
    ! is not a procedure heading, including END statements and calls.
    subroutine parse_procedure_statement(line, is_proc, is_pure, is_elemental, &
                                         name, args)
        character(len=*), intent(in) :: line
        logical, intent(out) :: is_proc, is_pure, is_elemental
        character(len=:), allocatable, intent(out) :: name, args
        character(len=:), allocatable :: prefix, rest
        integer :: kw_pos, kw_len, open_paren, close_paren, name_end

        is_proc = .false.
        is_pure = .false.
        is_elemental = .false.
        name = ''
        args = ''
        kw_pos = purity_find_word(line, 'subroutine')
        kw_len = 10
        if (kw_pos == 0) then
            kw_pos = purity_find_word(line, 'function')
            kw_len = 8
        end if
        if (kw_pos == 0) return
        prefix = line(:kw_pos - 1)
        if (.not. purity_prefix_is_valid(prefix)) return
        rest = adjustl(line(kw_pos + kw_len:))
        if (len_trim(rest) == 0) return
        name_end = verify(rest, 'abcdefghijklmnopqrstuvwxyz0123456789_') - 1
        if (name_end == -1) name_end = len_trim(rest)
        if (name_end <= 0) return
        name = rest(:name_end)
        is_proc = .true.
        is_elemental = purity_has_word(prefix, 'elemental')
        is_pure = purity_has_word(prefix, 'pure')
        if (is_elemental .and. .not. purity_has_word(prefix, 'impure')) then
            is_pure = .true.
        end if
        open_paren = index(rest, '(')
        if (open_paren == 0) return
        close_paren = purity_matching_paren(rest, open_paren)
        if (close_paren == 0) return
        args = ','//purity_squeeze(rest(open_paren + 1:close_paren - 1))//','
    end subroutine parse_procedure_statement

    ! True when every word before SUBROUTINE/FUNCTION is a procedure prefix or
    ! part of a result type specification. Anything else means the keyword
    ! appeared inside an unrelated statement.
    logical function purity_prefix_is_valid(prefix) result(ok)
        character(len=*), intent(in) :: prefix
        character(len=:), allocatable :: text, word
        integer :: pos, depth, i, start

        ok = .false.
        text = ''
        depth = 0
        do i = 1, len(prefix)
            if (prefix(i:i) == '(') then
                depth = depth + 1
            else if (prefix(i:i) == ')') then
                if (depth > 0) depth = depth - 1
            else if (depth == 0) then
                text = text//prefix(i:i)
            end if
        end do
        if (index(text, '=') > 0) return
        pos = 1
        do
            start = verify(text(pos:), ' '//char(9))
            if (start == 0) exit
            pos = pos + start - 1
            i = verify(text(pos:), 'abcdefghijklmnopqrstuvwxyz0123456789_*') - 1
            if (i == -1) i = len(text) - pos + 1
            if (i <= 0) return
            word = text(pos:pos + i - 1)
            if (.not. purity_is_prefix_word(word)) return
            pos = pos + i
            if (pos > len(text)) exit
        end do
        ok = .true.
    end function purity_prefix_is_valid

    logical function purity_is_prefix_word(word) result(ok)
        character(len=*), intent(in) :: word

        select case (word)
        case ('pure', 'impure', 'elemental', 'recursive', 'non_recursive', &
              'module', 'real', 'integer', 'logical', 'complex', 'character', &
              'double', 'precision', 'type', 'class')
            ok = .true.
        case default
            ok = .false.
        end select
    end function purity_is_prefix_word

    ! True for END, END SUBROUTINE and END FUNCTION; false for every other
    ! END construct so only procedure scopes are popped.
    logical function purity_ends_procedure(line) result(ends)
        character(len=*), intent(in) :: line
        character(len=:), allocatable :: rest

        ends = .false.
        if (.not. purity_word_at(line, 'end', 1)) return
        rest = adjustl(line(4:))
        if (len_trim(rest) == 0) then
            ends = .true.
            return
        end if
        if (purity_word_at(rest, 'subroutine', 1)) ends = .true.
        if (purity_word_at(rest, 'function', 1)) ends = .true.
    end function purity_ends_procedure

    ! True for the opening statement of a derived type definition. A type
    ! declaration (TYPE(name) :: v), TYPE IS and CLASS IS guards are excluded.
    logical function purity_starts_derived_type(line) result(starts)
        character(len=*), intent(in) :: line
        character(len=:), allocatable :: rest

        starts = .false.
        if (.not. purity_word_at(line, 'type', 1)) return
        rest = adjustl(line(5:))
        if (len_trim(rest) == 0) return
        if (rest(1:1) == '(') return
        if (purity_word_at(rest, 'is', 1)) return
        starts = .true.
    end function purity_starts_derived_type

    logical function purity_interface_is_elemental(iface, names, count) &
            result(is_elem)
        character(len=*), intent(in) :: iface
        character(len=*), intent(in) :: names(:)
        integer, intent(in) :: count

        is_elem = storage_name_listed(names, count, iface)
        if (is_elem) return
        select case (iface)
        case ('sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'atan2', 'sinh', &
              'cosh', 'tanh', 'exp', 'log', 'log10', 'sqrt', 'abs', 'erf', &
              'erfc', 'gamma', 'log_gamma')
            is_elem = .true.
        end select
    end function purity_interface_is_elemental

    ! True when name appears in a ,-delimited dummy argument list.
    logical function purity_arg_listed(args, name) result(listed)
        character(len=*), intent(in) :: args
        character(len=*), intent(in) :: name

        listed = .false.
        if (len_trim(args) == 0) return
        if (len_trim(name) == 0) return
        listed = index(args, ','//trim(name)//',') > 0
    end function purity_arg_listed

    ! Position of word in text as a standalone identifier, or 0.
    integer function purity_find_word(text, word) result(pos)
        character(len=*), intent(in) :: text
        character(len=*), intent(in) :: word
        integer :: start, hit

        pos = 0
        start = 1
        do
            if (start > len(text)) return
            hit = index(text(start:), word)
            if (hit == 0) return
            hit = start + hit - 1
            if (purity_word_at(text, word, hit)) then
                pos = hit
                return
            end if
            start = hit + 1
        end do
    end function purity_find_word

    logical function purity_has_word(text, word) result(found)
        character(len=*), intent(in) :: text
        character(len=*), intent(in) :: word

        found = purity_find_word(text, word) > 0
    end function purity_has_word

    ! True when word occupies text(pos:) with identifier boundaries on both
    ! sides.
    logical function purity_word_at(text, word, pos) result(at)
        character(len=*), intent(in) :: text
        character(len=*), intent(in) :: word
        integer, intent(in) :: pos
        integer :: last

        at = .false.
        if (pos < 1) return
        last = pos + len(word) - 1
        if (last > len(text)) return
        if (text(pos:last) /= word) return
        if (pos > 1) then
            if (purity_is_name_char(text(pos - 1:pos - 1))) return
        end if
        if (last < len(text)) then
            if (purity_is_name_char(text(last + 1:last + 1))) return
        end if
        at = .true.
    end function purity_word_at

    logical function purity_is_name_char(c) result(is_name)
        character(len=1), intent(in) :: c

        is_name = (c >= 'a' .and. c <= 'z') .or. (c >= '0' .and. c <= '9') &
                  .or. c == '_'
    end function purity_is_name_char

    ! Index of the closing parenthesis matching the one at open_pos, or 0.
    integer function purity_matching_paren(text, open_pos) result(close_pos)
        character(len=*), intent(in) :: text
        integer, intent(in) :: open_pos
        integer :: i, depth

        close_pos = 0
        depth = 0
        do i = open_pos, len(text)
            if (text(i:i) == '(') then
                depth = depth + 1
            else if (text(i:i) == ')') then
                depth = depth - 1
                if (depth == 0) then
                    close_pos = i
                    return
                end if
            end if
        end do
    end function purity_matching_paren

    ! text with every blank and tab removed.
    function purity_squeeze(text) result(squeezed)
        character(len=*), intent(in) :: text
        character(len=:), allocatable :: squeezed
        integer :: i

        squeezed = ''
        do i = 1, len(text)
            if (text(i:i) == ' ') cycle
            if (text(i:i) == char(9)) cycle
            squeezed = squeezed//text(i:i)
        end do
    end function purity_squeeze

    function purity_at_line(line_no) result(location)
        integer, intent(in) :: line_no
        character(len=:), allocatable :: location
        character(len=32) :: buffer

        write (buffer, '(" at line ",I0)') line_no
        location = trim(buffer)
    end function purity_at_line
end submodule session_program_lowering_reject_purity
