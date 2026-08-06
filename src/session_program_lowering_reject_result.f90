submodule (session_program_lowering_impl) session_program_lowering_reject_result
    implicit none
contains
    ! Function-result and ENTRY rules (#379). Four families of invalid source
    ! lower cleanly today but are not conforming Fortran, so each is rejected
    ! from the earliest layer that still holds the information:
    !
    !   1. Assignment to a name that designates a procedure rather than the
    !      enclosing function's result variable (gfortran "is not a variable":
    !      func_assign.f90, entry_15.f90).
    !   2. A dummy argument of an ENTRY referenced before that ENTRY statement
    !      (gfortran "before the ENTRY statement": entry_dummy_ref_2.f90).
    !   3. An interface block inside a function redeclaring the function's own
    !      name, which is already the result variable (gfortran "cannot have a
    !      type": pr39695_2.f90, pr39695_3.f90).
    !   4. A POINTER or DIMENSION attribute statement naming the function
    !      itself while a RESULT clause is present (gfortran "RESULT
    !      variable": func_result_7.f90). Bare attribute statements never
    !      reach the typed AST, so this one is checked on the source text.
    module procedure check_result_and_entry_rules

        call check_function_name_in_interface(arena, error_msg)
        if (len_trim(error_msg) > 0) return
        call check_entry_dummy_before_entry(arena, error_msg)
        if (len_trim(error_msg) > 0) return
        call check_assignment_to_procedure_name(arena, error_msg)
        if (len_trim(error_msg) > 0) return
        call check_function_name_attribute_source(arena, error_msg)
    contains

    ! A function's own name always designates its result variable when no
    ! RESULT clause renames it. An interface block in that function's
    ! specification part that declares a procedure of the same name therefore
    ! gives the result variable a procedure identity it cannot have.
    subroutine check_function_name_in_interface(arena, error_msg)
        type(ast_arena_t), intent(in) :: arena
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: proc_name
        integer :: n, i, j, b, p
        character(len=64) :: location

        call set_empty(error_msg)
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (nd => arena%entries(n)%node)
            type is (function_def_node)
                if (.not. allocated(nd%name)) cycle
                if (.not. allocated(nd%body_indices)) cycle
                do i = 1, size(nd%body_indices)
                    b = nd%body_indices(i)
                    if (.not. node_exists(arena, b)) cycle
                    select type (ib => arena%entries(b)%node)
                    type is (interface_block_node)
                        if (.not. allocated(ib%procedure_indices)) cycle
                        do j = 1, size(ib%procedure_indices)
                            p = ib%procedure_indices(j)
                            call procedure_def_name(arena, p, proc_name)
                            if (len_trim(proc_name) == 0) cycle
                            if (.not. same_name(proc_name, nd%name)) cycle
                            write (location, '(" at line ",I0)') &
                                arena%entries(b)%node%line
                            error_msg = 'symbol '''//trim(proc_name)// &
                                ''' cannot have a type: an interface block '// &
                                'redeclares the enclosing function result'// &
                                trim(location)
                            return
                        end do
                    end select
                end do
            end select
        end do
    end subroutine check_function_name_in_interface

    ! ENTRY dummies only come into existence at the ENTRY statement. A dummy
    ! that belongs to an ENTRY but not to the host procedure is undefined in
    ! every statement that precedes the ENTRY, so any earlier reference to it
    ! is invalid (F2018 C1573, gfortran PR25058).
    subroutine check_entry_dummy_before_entry(arena, error_msg)
        type(ast_arena_t), intent(in) :: arena
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: n

        call set_empty(error_msg)
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (nd => arena%entries(n)%node)
            type is (function_def_node)
                if (.not. allocated(nd%body_indices)) cycle
                call check_entries_in_body(arena, nd%body_indices, &
                                           nd%param_indices, &
                                           arena%entries(n)%node%line, error_msg)
                if (len_trim(error_msg) > 0) return
            type is (subroutine_def_node)
                if (.not. allocated(nd%body_indices)) cycle
                call check_entries_in_body(arena, nd%body_indices, &
                                           nd%param_indices, &
                                           arena%entries(n)%node%line, error_msg)
                if (len_trim(error_msg) > 0) return
            end select
        end do
    end subroutine check_entry_dummy_before_entry

    subroutine check_entries_in_body(arena, body_indices, param_indices, &
                                     host_line, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: body_indices(:)
        integer, allocatable, intent(in) :: param_indices(:)
        integer, intent(in) :: host_line
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: dummies(:)
        integer :: i, k, dummy_count, entry_line, ref_line
        character(len=64) :: location

        call set_empty(error_msg)
        do i = 1, size(body_indices)
            if (.not. node_exists(arena, body_indices(i))) cycle
            select type (en => arena%entries(body_indices(i))%node)
            type is (entry_node)
                if (.not. allocated(en%params_text)) cycle
                call entry_dummy_names(en%params_text, dummies, dummy_count)
                entry_line = arena%entries(body_indices(i))%node%line
                do k = 1, dummy_count
                    if (host_dummy_named(arena, param_indices, &
                                         trim(dummies(k)))) cycle
                    call find_reference_before(arena, trim(dummies(k)), &
                                               host_line, entry_line, ref_line)
                    if (ref_line <= 0) cycle
                    write (location, '(" at line ",I0)') ref_line
                    error_msg = 'dummy argument '''//trim(dummies(k))// &
                        ''' of ENTRY '''//en%name//''' is referenced '// &
                        'before the ENTRY statement'//trim(location)
                    return
                end do
            end select
        end do
    end subroutine check_entries_in_body

    ! params_text holds the raw ENTRY suffix, e.g. "(K) result(RE1)". Only the
    ! parenthesised dummy list is wanted; the RESULT clause is not a dummy.
    subroutine entry_dummy_names(params_text, dummies, dummy_count)
        character(len=*), intent(in) :: params_text
        character(len=:), allocatable, intent(out) :: dummies(:)
        integer, intent(out) :: dummy_count
        integer :: open_pos, close_pos, depth, i

        dummy_count = 0
        allocate (character(len=1) :: dummies(1))
        dummies = ''
        open_pos = index(params_text, '(')
        if (open_pos == 0) return
        depth = 0
        close_pos = 0
        do i = open_pos, len(params_text)
            if (params_text(i:i) == '(') depth = depth + 1
            if (params_text(i:i) == ')') then
                depth = depth - 1
                if (depth == 0) then
                    close_pos = i
                    exit
                end if
            end if
        end do
        if (close_pos <= open_pos + 1) return
        call split_csv(params_text(open_pos + 1:close_pos - 1), dummies, &
                       dummy_count)
    end subroutine entry_dummy_names

    logical function host_dummy_named(arena, param_indices, name) result(found)
        type(ast_arena_t), intent(in) :: arena
        integer, allocatable, intent(in) :: param_indices(:)
        character(len=*), intent(in) :: name
        integer :: i

        found = .false.
        if (.not. allocated(param_indices)) return
        do i = 1, size(param_indices)
            if (.not. node_exists(arena, param_indices(i))) cycle
            select type (pd => arena%entries(param_indices(i))%node)
            type is (parameter_declaration_node)
                if (.not. allocated(pd%name)) cycle
                if (same_name(pd%name, name)) found = .true.
            type is (identifier_node)
                if (.not. allocated(pd%name)) cycle
                if (same_name(pd%name, name)) found = .true.
            type is (declaration_node)
                if (.not. allocated(pd%var_name)) cycle
                if (same_name(pd%var_name, name)) found = .true.
            end select
            if (found) return
        end do
    end function host_dummy_named

    subroutine find_reference_before(arena, name, host_line, entry_line, ref_line)
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: name
        integer, intent(in) :: host_line, entry_line
        integer, intent(out) :: ref_line
        integer :: n, line

        ref_line = 0
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (idn => arena%entries(n)%node)
            type is (identifier_node)
                if (.not. allocated(idn%name)) cycle
                if (.not. same_name(idn%name, name)) cycle
                line = arena%entries(n)%node%line
                if (line <= host_line) cycle
                if (line >= entry_line) cycle
                ref_line = line
                return
            end select
        end do
    end subroutine find_reference_before

    ! Assigning to a name that designates a procedure is never valid: only the
    ! enclosing function's own result variable may appear on the left of an
    ! assignment inside that function. Names declared locally (including
    ! dummies) are excluded so an ordinary variable that happens to share a
    ! spelling with an unrelated procedure stays untouched.
    subroutine check_assignment_to_procedure_name(arena, error_msg)
        type(ast_arena_t), intent(in) :: arena
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: n

        call set_empty(error_msg)
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (nd => arena%entries(n)%node)
            type is (function_def_node)
                if (.not. allocated(nd%body_indices)) cycle
                call check_scope_assignments(arena, nd%body_indices, &
                                             nd%param_indices, nd%name, &
                                             result_name_of(nd), error_msg)
                if (len_trim(error_msg) > 0) return
            type is (subroutine_def_node)
                if (.not. allocated(nd%body_indices)) cycle
                call check_scope_assignments(arena, nd%body_indices, &
                                             nd%param_indices, nd%name, '', &
                                             error_msg)
                if (len_trim(error_msg) > 0) return
            end select
        end do
    end subroutine check_assignment_to_procedure_name

    function result_name_of(nd) result(name)
        type(function_def_node), intent(in) :: nd
        character(len=:), allocatable :: name

        name = ''
        if (allocated(nd%result_variable)) name = trim(nd%result_variable)
        if (len_trim(name) == 0 .and. allocated(nd%name)) name = trim(nd%name)
    end function result_name_of

    subroutine check_scope_assignments(arena, body_indices, param_indices, &
                                       proc_name, result_name, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: body_indices(:)
        integer, allocatable, intent(in) :: param_indices(:)
        character(len=*), intent(in) :: proc_name, result_name
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: target_name
        integer :: i
        character(len=64) :: location

        call set_empty(error_msg)
        do i = 1, size(body_indices)
            if (.not. node_exists(arena, body_indices(i))) cycle
            select type (asg => arena%entries(body_indices(i))%node)
            type is (assignment_node)
                call assignment_target_name(arena, asg%target_index, target_name)
                if (len_trim(target_name) == 0) cycle
                if (same_name(target_name, proc_name)) cycle
                if (len_trim(result_name) > 0) then
                    if (same_name(target_name, result_name)) cycle
                end if
                if (host_dummy_named(arena, param_indices, target_name)) cycle
                if (declared_in_body(arena, body_indices, target_name)) cycle
                if (.not. names_a_procedure(arena, body_indices, target_name)) &
                    cycle
                write (location, '(" at line ",I0)') &
                    arena%entries(body_indices(i))%node%line
                error_msg = ''''//target_name//''' is not a variable: it '// &
                    'names a procedure and is not the function result'// &
                    trim(location)
                return
            end select
        end do
    end subroutine check_scope_assignments

    subroutine assignment_target_name(arena, target_index, name)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: target_index
        character(len=:), allocatable, intent(out) :: name

        call set_empty(name)
        if (.not. node_exists(arena, target_index)) return
        select type (tgt => arena%entries(target_index)%node)
        type is (identifier_node)
            if (allocated(tgt%name)) name = trim(tgt%name)
        end select
    end subroutine assignment_target_name

    logical function declared_in_body(arena, body_indices, name) result(found)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: body_indices(:)
        character(len=*), intent(in) :: name
        integer :: i, j

        found = .false.
        do i = 1, size(body_indices)
            if (.not. node_exists(arena, body_indices(i))) cycle
            select type (dcl => arena%entries(body_indices(i))%node)
            type is (declaration_node)
                if (allocated(dcl%var_name)) then
                    if (same_name(dcl%var_name, name)) then
                        found = .true.
                        return
                    end if
                end if
                if (.not. dcl%is_multi_declaration) cycle
                if (.not. allocated(dcl%var_names)) cycle
                do j = 1, size(dcl%var_names)
                    if (same_name(dcl%var_names(j), name)) then
                        found = .true.
                        return
                    end if
                end do
            end select
        end do
    end function declared_in_body

    ! A name is known to designate a procedure when an interface block in this
    ! scope declares it, when an ENTRY in this scope carries it together with
    ! its own RESULT clause, or when some procedure definition in the unit
    ! carries that name.
    logical function names_a_procedure(arena, body_indices, name) result(found)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: body_indices(:)
        character(len=*), intent(in) :: name
        character(len=:), allocatable :: proc_name
        integer :: i, j, n

        found = .false.
        do i = 1, size(body_indices)
            if (.not. node_exists(arena, body_indices(i))) cycle
            select type (bn => arena%entries(body_indices(i))%node)
            type is (interface_block_node)
                if (.not. allocated(bn%procedure_indices)) cycle
                do j = 1, size(bn%procedure_indices)
                    call procedure_def_name(arena, bn%procedure_indices(j), &
                                            proc_name)
                    if (len_trim(proc_name) == 0) cycle
                    if (same_name(proc_name, name)) then
                        found = .true.
                        return
                    end if
                end do
            type is (entry_node)
                if (.not. allocated(bn%name)) cycle
                if (.not. allocated(bn%params_text)) cycle
                if (index(lowercase_text(bn%params_text), 'result(') == 0) cycle
                if (same_name(bn%name, name)) then
                    found = .true.
                    return
                end if
            end select
        end do
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            call procedure_def_name(arena, n, proc_name)
            if (len_trim(proc_name) == 0) cycle
            if (same_name(proc_name, name)) then
                found = .true.
                return
            end if
        end do
    end function names_a_procedure

    ! POINTER and DIMENSION attribute statements are dropped by the parser, so
    ! the source text is the only layer that still records them. Inside a
    ! function that renames its result with RESULT(...), such a statement may
    ! not name the function itself.
    subroutine check_function_name_attribute_source(arena, error_msg)
        type(ast_arena_t), intent(in) :: arena
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: source, line, code, func_name, attr_name
        logical :: found
        integer :: pos, next_nl, line_no
        character(len=64) :: location

        call set_empty(error_msg)
        call get_source_text(arena, source, found)
        if (.not. found) return
        func_name = ''
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
            code = trim(adjustl(lowercase_text( &
                                strip_data_source_comment(line))))
            if (len(code) == 0) cycle
            if (starts_with_word(code, 'end')) then
                func_name = ''
                cycle
            end if
            call result_function_name(code, attr_name)
            if (len_trim(attr_name) > 0) then
                func_name = attr_name
                cycle
            end if
            if (len_trim(func_name) == 0) cycle
            call attribute_stmt_name(code, attr_name)
            if (len_trim(attr_name) == 0) cycle
            if (.not. same_name(attr_name, func_name)) cycle
            write (location, '(" at line ",I0)') line_no
            error_msg = 'the function name '''//trim(func_name)// &
                ''' cannot carry attributes when a RESULT variable is '// &
                'declared'//trim(location)
            return
        end do
    end subroutine check_function_name_attribute_source

    ! Returns the function name of a FUNCTION statement that carries a RESULT
    ! clause, and an empty string for every other line.
    subroutine result_function_name(code, name)
        character(len=*), intent(in) :: code
        character(len=:), allocatable, intent(out) :: name
        integer :: kw, rest

        call set_empty(name)
        kw = index(code, 'function ')
        if (kw == 0) return
        if (index(code, 'end function') > 0) return
        if (index(code, ' result') == 0) return
        if (index(code, '(') == 0) return
        rest = kw + len('function ')
        name = leading_identifier(trim(adjustl(code(rest:))))
    end subroutine result_function_name

    ! Returns the single name of a bare POINTER or DIMENSION attribute
    ! statement, and an empty string for any other line.
    subroutine attribute_stmt_name(code, name)
        character(len=*), intent(in) :: code
        character(len=:), allocatable, intent(out) :: name
        character(len=:), allocatable :: rest

        call set_empty(name)
        if (starts_with_word(code, 'pointer')) then
            rest = trim(adjustl(code(len('pointer') + 1:)))
        else if (starts_with_word(code, 'dimension')) then
            rest = trim(adjustl(code(len('dimension') + 1:)))
        else
            return
        end if
        if (len(rest) >= 2) then
            if (rest(1:2) == '::') rest = trim(adjustl(rest(3:)))
        end if
        name = leading_identifier(rest)
    end subroutine attribute_stmt_name
    end procedure check_result_and_entry_rules
end submodule session_program_lowering_reject_result
