submodule (session_program_lowering_impl) session_program_lowering_reject_pointer
    implicit none
contains
    ! Data and procedure pointer target contracts (#381).
    !
    ! A pointer only ever associates with something that can be a target:
    ! a data pointer needs a pointer or a valid target object, a procedure
    ! pointer needs an actual procedure that is visible here. The lowerer
    ! silently accepted several invalid associations, so this file collects
    ! the constraint family: PRESENT subobjects, ABSTRACT interfaces given
    ! the POINTER attribute, procedure-pointer targets that are data
    ! objects, function result variables or unknown names, INTENT(IN)
    ! pointer actuals in a pointer-association context, Cray pointer
    ! declarations, and parenthesised expressions where a target is
    ! required.
    module procedure check_pointer_target_contracts

        call set_empty(error_msg)
        call check_present_argument_subobject(arena, error_msg)
        if (len_trim(error_msg) > 0) return
        call check_abstract_interface_pointer(arena, error_msg)
        if (len_trim(error_msg) > 0) return
        call check_proc_pointer_targets(arena, error_msg)
        if (len_trim(error_msg) > 0) return
        call check_pointer_intent_actuals(arena, error_msg)
        if (len_trim(error_msg) > 0) return
        call check_pointer_source_forms(arena, error_msg)
    end procedure check_pointer_target_contracts

    ! The A argument of PRESENT shall be the name of an optional dummy
    ! argument, never a subobject of one (F2018 16.9.157, gfortran
    ! "must not be a subobject"). Anything but a bare identifier is a
    ! component reference, array element or expression.
    module procedure check_present_argument_subobject
        integer :: n
        character(len=64) :: location

        call set_empty(error_msg)
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (nd => arena%entries(n)%node)
            type is (call_or_subscript_node)
                if (.not. allocated(nd%name)) cycle
                if (.not. same_name(nd%name, 'present')) cycle
                if (.not. allocated(nd%arg_indices)) cycle
                if (size(nd%arg_indices) /= 1) cycle
                if (is_identifier(arena, nd%arg_indices(1))) cycle
                write (location, '(" at line ",I0,", column ",I0)') &
                    nd%line, nd%column
                error_msg = 'argument of PRESENT'//trim(location)// &
                    ' must be an optional dummy argument name and '// &
                    'must not be a subobject'
                return
            end select
        end do
    end procedure check_present_argument_subobject

    ! An interface body in an ABSTRACT INTERFACE names an abstract
    ! interface, not a procedure, so that name shall not also carry the
    ! POINTER attribute (F2018 C1213, gfortran "PROCEDURE POINTER
    ! attribute conflicts with ABSTRACT attribute").
    module procedure check_abstract_interface_pointer
        integer :: n, i
        character(len=:), allocatable :: proc_name
        character(len=64) :: location
        integer :: decl_line

        call set_empty(error_msg)
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (nd => arena%entries(n)%node)
            type is (interface_block_node)
                if (.not. nd%is_abstract) cycle
                if (.not. allocated(nd%procedure_indices)) cycle
                do i = 1, size(nd%procedure_indices)
                    call procedure_def_name(arena, nd%procedure_indices(i), &
                                            proc_name)
                    if (len_trim(proc_name) == 0) cycle
                    decl_line = pointer_declaration_line(arena, proc_name)
                    if (decl_line <= 0) then
                        decl_line = pointer_statement_line(arena, proc_name)
                    end if
                    if (decl_line <= 0) cycle
                    write (location, '(" at line ",I0)') decl_line
                    error_msg = 'POINTER attribute conflicts with ABSTRACT '// &
                        'attribute for '''//trim(proc_name)//''''//trim(location)
                    return
                end do
            end select
        end do
    end procedure check_abstract_interface_pointer

    ! Line of the first declaration giving name the POINTER attribute,
    ! or 0 when no such declaration exists.
    module procedure pointer_declaration_line
        integer :: n, i

        line_no = 0
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (nd => arena%entries(n)%node)
            type is (declaration_node)
                if (.not. nd%is_pointer) cycle
                if (allocated(nd%var_name)) then
                    if (same_name(nd%var_name, name)) then
                        line_no = nd%line
                        return
                    end if
                end if
                if (.not. nd%is_multi_declaration) cycle
                if (.not. allocated(nd%var_names)) cycle
                do i = 1, size(nd%var_names)
                    if (same_name(nd%var_names(i), name)) then
                        line_no = nd%line
                        return
                    end if
                end do
            end select
        end do
    end procedure pointer_declaration_line

    ! A proc-target shall be a procedure that is visible in this scope:
    ! a data object, the result variable of the enclosing function, or a
    ! name that denotes nothing at all are all invalid (F2018 10.2.2.4,
    ! gfortran "Invalid procedure pointer", "is invalid as proc-target",
    ! "must be either an intrinsic, host or use associated, referenced or
    ! have the EXTERNAL attribute").
    module procedure check_proc_pointer_targets
        character(len=:), allocatable :: ptr_name, target_name, ff_error
        character(len=:), allocatable :: enclosing_name
        integer :: n
        logical :: recursive_host
        character(len=64) :: location

        call set_empty(error_msg)
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (nd => arena%entries(n)%node)
            type is (pointer_assignment_node)
                if (.not. is_identifier(arena, nd%pointer_index)) cycle
                if (.not. is_identifier(arena, nd%target_index)) cycle
                call get_identifier_name(arena, nd%pointer_index, ptr_name, &
                                         ff_error)
                if (len_trim(ff_error) > 0) cycle
                if (.not. name_is_proc_pointer(arena, ptr_name)) cycle
                call get_identifier_name(arena, nd%target_index, target_name, &
                                         ff_error)
                if (len_trim(ff_error) > 0) cycle
                if (same_name(target_name, 'null')) cycle
                if (name_is_proc_pointer(arena, target_name)) cycle
                write (location, '(" at line ",I0)') nd%line
                if (name_is_data_object(arena, target_name)) then
                    error_msg = 'Invalid procedure pointer assignment'// &
                        trim(location)//': '''//trim(target_name)// &
                        ''' is a data object, not a procedure'
                    return
                end if
                call enclosing_procedure_of(arena, n, enclosing_name, &
                                            recursive_host)
                if (len_trim(enclosing_name) > 0 .and. .not. recursive_host) then
                    if (same_name(enclosing_name, target_name)) then
                        error_msg = ''''//trim(target_name)//''''//trim(location)// &
                            ' is invalid as proc-target in procedure '// &
                            'pointer assignment: it names the function '// &
                            'result variable'
                        return
                    end if
                end if
                if (name_is_visible_procedure(arena, target_name)) cycle
                error_msg = 'procedure pointer target '''//trim(target_name)// &
                    ''''//trim(location)//' must be either an intrinsic, '// &
                    'host or use associated, referenced or have the '// &
                    'EXTERNAL attribute'
                return
            end select
        end do
    end procedure check_proc_pointer_targets

    ! A procedure pointer is declared as procedure(...), pointer :: p.
    module procedure name_is_proc_pointer
        integer :: n, i
        logical :: names_it

        is_proc_ptr = .false.
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (nd => arena%entries(n)%node)
            type is (declaration_node)
                if (.not. nd%is_pointer) cycle
                if (.not. allocated(nd%type_name)) cycle
                if (.not. starts_with_word(lowercase_text(nd%type_name), &
                                           'procedure')) cycle
                names_it = .false.
                if (allocated(nd%var_name)) then
                    if (same_name(nd%var_name, name)) names_it = .true.
                end if
                if (allocated(nd%var_names)) then
                    do i = 1, size(nd%var_names)
                        if (same_name(nd%var_names(i), name)) names_it = .true.
                    end do
                end if
                if (names_it) then
                    is_proc_ptr = .true.
                    return
                end if
            end select
        end do
    end procedure name_is_proc_pointer

    ! A declaration with an intrinsic or derived type name and no
    ! PROCEDURE prefix declares a data object.
    module procedure name_is_data_object
        integer :: n, i
        logical :: names_it

        is_data = .false.
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (nd => arena%entries(n)%node)
            type is (declaration_node)
                if (nd%is_external) cycle
                if (.not. allocated(nd%type_name)) cycle
                if (len_trim(nd%type_name) == 0) cycle
                if (starts_with_word(lowercase_text(nd%type_name), 'procedure')) cycle
                names_it = .false.
                if (allocated(nd%var_name)) then
                    if (same_name(nd%var_name, name)) names_it = .true.
                end if
                if (allocated(nd%var_names)) then
                    do i = 1, size(nd%var_names)
                        if (same_name(nd%var_names(i), name)) names_it = .true.
                    end do
                end if
                if (names_it) then
                    is_data = .true.
                    return
                end if
            end select
        end do
    end procedure name_is_data_object

    ! A name is a usable proc-target when a procedure definition, an
    ! interface body, an EXTERNAL declaration or an INTRINSIC statement
    ! makes it visible.
    module procedure name_is_visible_procedure
        character(len=:), allocatable :: def_name
        integer :: n, i

        is_visible = .false.
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            call procedure_def_name(arena, n, def_name)
            if (len_trim(def_name) > 0) then
                if (same_name(def_name, name)) then
                    is_visible = .true.
                    return
                end if
            end if
            select type (nd => arena%entries(n)%node)
            type is (declaration_node)
                if (.not. nd%is_external) cycle
                if (allocated(nd%var_name)) then
                    if (same_name(nd%var_name, name)) then
                        is_visible = .true.
                        return
                    end if
                end if
                if (.not. allocated(nd%var_names)) cycle
                do i = 1, size(nd%var_names)
                    if (same_name(nd%var_names(i), name)) then
                        is_visible = .true.
                        return
                    end if
                end do
            type is (intrinsic_statement_node)
                if (.not. allocated(nd%procedure_names)) cycle
                do i = 1, size(nd%procedure_names)
                    if (.not. allocated(nd%procedure_names(i)%s)) cycle
                    if (same_name(nd%procedure_names(i)%s, name)) then
                        is_visible = .true.
                        return
                    end if
                end do
            type is (use_statement_node)
                ! A USE without an only-list can bring in any procedure.
                if (.not. nd%has_only) then
                    is_visible = .true.
                    return
                end if
            end select
        end do
    end procedure name_is_visible_procedure

    ! Name and recursiveness of the procedure whose body holds node_index.
    module procedure enclosing_procedure_of
        integer :: n, i

        call set_empty(name)
        is_recursive = .false.
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (nd => arena%entries(n)%node)
            type is (function_def_node)
                if (.not. allocated(nd%body_indices)) cycle
                do i = 1, size(nd%body_indices)
                    if (nd%body_indices(i) /= node_index) cycle
                    if (allocated(nd%name)) name = trim(nd%name)
                    is_recursive = nd%is_recursive
                    return
                end do
            end select
        end do
    end procedure enclosing_procedure_of

    ! An INTENT(IN) pointer shall not appear as the actual argument of an
    ! INTENT(OUT) or INTENT(INOUT) pointer dummy: that is a pointer
    ! association context (F2018 C844, gfortran "INTENT(IN) in pointer
    ! association context").
    module procedure check_pointer_intent_actuals
        character(len=:), allocatable :: call_name, actual_name, ff_error
        character(len=:), allocatable :: dummy_intent, actual_intent
        integer, allocatable :: arg_indices(:)
        integer :: n, k
        logical :: dummy_pointer, actual_pointer, actual_found
        character(len=64) :: location

        call set_empty(error_msg)
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            if (.not. is_subroutine_call_statement(arena, n)) cycle
            call get_subroutine_call_name(arena, n, call_name, ff_error)
            if (len_trim(ff_error) > 0) cycle
            call get_subroutine_call_arg_indices(arena, n, arg_indices, ff_error)
            if (len_trim(ff_error) > 0) cycle
            if (.not. allocated(arg_indices)) cycle
            do k = 1, size(arg_indices)
                if (.not. is_identifier(arena, arg_indices(k))) cycle
                call dummy_pointer_intent(arena, call_name, k, dummy_pointer, &
                                          dummy_intent)
                if (.not. dummy_pointer) cycle
                if (dummy_intent /= 'out' .and. dummy_intent /= 'inout') cycle
                call get_identifier_name(arena, arg_indices(k), actual_name, &
                                         ff_error)
                if (len_trim(ff_error) > 0) cycle
                call declared_pointer_intent(arena, actual_name, actual_found, &
                                             actual_pointer, actual_intent)
                if (.not. actual_found) cycle
                if (.not. actual_pointer) cycle
                if (actual_intent /= 'in') cycle
                write (location, '(" at line ",I0)') get_node_line(arena, n)
                error_msg = 'variable '''//trim(actual_name)// &
                    ''' is INTENT(IN) in pointer association context'// &
                    trim(location)//' (actual argument to INTENT = '// &
                    'OUT/INOUT dummy pointer)'
                return
            end do
        end do
    end procedure check_pointer_intent_actuals

    ! POINTER attribute and INTENT of the position-th dummy of proc_name.
    module procedure dummy_pointer_intent
        character(len=:), allocatable :: dummy_name
        integer :: n
        logical :: found

        is_pointer = .false.
        call set_empty(intent_text)
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (nd => arena%entries(n)%node)
            type is (subroutine_def_node)
                if (.not. allocated(nd%name)) cycle
                if (.not. same_name(nd%name, proc_name)) cycle
                if (.not. allocated(nd%param_indices)) return
                if (position > size(nd%param_indices)) return
                call param_node_name(arena, nd%param_indices(position), &
                                     dummy_name)
                if (len_trim(dummy_name) == 0) return
                if (.not. allocated(nd%body_indices)) return
                call body_declaration_attributes(arena, nd%body_indices, &
                    dummy_name, found, is_pointer, intent_text)
                return
            end select
        end do
    end procedure dummy_pointer_intent

    ! Declared name of a dummy argument node.
    module procedure param_node_name

        call set_empty(name)
        if (.not. node_exists(arena, node_index)) return
        select type (nd => arena%entries(node_index)%node)
        type is (parameter_declaration_node)
            if (allocated(nd%name)) name = trim(nd%name)
        type is (declaration_node)
            if (allocated(nd%var_name)) name = trim(nd%var_name)
        type is (identifier_node)
            if (allocated(nd%name)) name = trim(nd%name)
        end select
    end procedure param_node_name

    ! POINTER attribute and INTENT of a declaration inside a procedure body.
    module procedure body_declaration_attributes
        integer :: i, j
        logical :: names_it

        found = .false.
        is_pointer = .false.
        call set_empty(intent_text)
        do i = 1, size(body_indices)
            if (.not. node_exists(arena, body_indices(i))) cycle
            select type (nd => arena%entries(body_indices(i))%node)
            type is (declaration_node)
                names_it = .false.
                if (allocated(nd%var_name)) then
                    if (same_name(nd%var_name, name)) names_it = .true.
                end if
                if (allocated(nd%var_names)) then
                    do j = 1, size(nd%var_names)
                        if (same_name(nd%var_names(j), name)) names_it = .true.
                    end do
                end if
                if (.not. names_it) cycle
                found = .true.
                is_pointer = nd%is_pointer
                if (nd%has_intent .and. allocated(nd%intent)) then
                    intent_text = lowercase_text(trim(nd%intent))
                end if
                return
            end select
        end do
    end procedure body_declaration_attributes

    ! POINTER attribute and INTENT of the first declaration naming name.
    module procedure declared_pointer_intent
        integer :: n, i
        logical :: names_it

        found = .false.
        is_pointer = .false.
        call set_empty(intent_text)
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (nd => arena%entries(n)%node)
            type is (declaration_node)
                names_it = .false.
                if (allocated(nd%var_name)) then
                    if (same_name(nd%var_name, name)) names_it = .true.
                end if
                if (allocated(nd%var_names)) then
                    do i = 1, size(nd%var_names)
                        if (same_name(nd%var_names(i), name)) names_it = .true.
                    end do
                end if
                if (.not. names_it) cycle
                if (.not. nd%is_pointer) cycle
                found = .true.
                is_pointer = .true.
                if (nd%has_intent .and. allocated(nd%intent)) then
                    intent_text = lowercase_text(trim(nd%intent))
                end if
                return
            end select
        end do
    end procedure declared_pointer_intent

    ! Parentheses never survive into the typed AST, so the forms that turn
    ! on them are checked on the statement source: a Cray pointer
    ! declaration (POINTER (ptr, pointee), not accepted without
    ! -fcray-pointer), a parenthesised ASSOCIATED target, and a
    ! parenthesised actual argument passed to a POINTER dummy.
    module procedure check_pointer_source_forms
        character(len=:), allocatable :: source, line, code
        integer :: pos, next_nl, line_no

        call set_empty(error_msg)
        call get_pointer_source_lines(arena, source)
        if (len(source) == 0) return
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
            call strip_line_comment(line, code)
            call check_cray_pointer_line(code, line_no, error_msg)
            if (len_trim(error_msg) > 0) return
            call check_associated_target_line(code, line_no, error_msg)
            if (len_trim(error_msg) > 0) return
            call check_parenthesised_actual_line(arena, code, line_no, error_msg)
            if (len_trim(error_msg) > 0) return
            call check_present_source_line(code, line_no, error_msg)
            if (len_trim(error_msg) > 0) return
        end do
    end procedure check_pointer_source_forms

    ! Line of a POINTER attribute statement (POINTER :: a, b) naming name,
    ! or 0. The attribute-statement form carries no type and does not reach
    ! the typed AST as a declaration, so it is read from the source.
    module procedure pointer_statement_line
        character(len=:), allocatable :: source, line, code, low, list
        integer :: pos, next_nl, current, rest, starts(32), ends(32), n_names, k

        line_no = 0
        call get_pointer_source_lines(arena, source)
        if (len(source) == 0) return
        pos = 1
        current = 0
        do while (pos <= len(source))
            next_nl = index(source(pos:), new_line('a'))
            if (next_nl == 0) then
                line = source(pos:)
                pos = len(source) + 1
            else
                line = source(pos:pos + next_nl - 2)
                pos = pos + next_nl
            end if
            current = current + 1
            call strip_line_comment(line, code)
            low = trim(adjustl(lowercase_text(code)))
            if (len(low) < 9) cycle
            if (low(1:7) /= 'pointer') cycle
            if (is_fortran_identifier_char(low(8:8))) cycle
            rest = 8
            do while (rest <= len(low))
                if (low(rest:rest) /= ' ') exit
                rest = rest + 1
            end do
            if (rest > len(low)) cycle
            if (low(rest:rest) == '(') cycle
            if (rest + 1 <= len(low)) then
                if (low(rest:rest + 1) == '::') rest = rest + 2
            end if
            if (rest > len(low)) cycle
            list = low(rest:)
            call top_level_args(list, starts, ends, n_names)
            do k = 1, n_names
                if (.not. same_name(trim(adjustl(list(starts(k):ends(k)))), &
                                    name)) cycle
                line_no = current
                return
            end do
        end do
    end procedure pointer_statement_line

    ! PRESENT applied to anything but a bare name, read from the source.
    ! The typed AST does not expose the argument of PRESENT in every scope,
    ! so the constraint is also enforced on the statement text.
    module procedure check_present_source_line
        character(len=:), allocatable :: low, arg_text
        integer :: open_pos, close_pos, cursor
        character(len=64) :: location

        low = lowercase_text(code)
        cursor = 1
        do
            call find_call_paren(low, 'present', cursor, open_pos)
            if (open_pos <= 0) return
            cursor = open_pos + 1
            call matching_paren(low, open_pos, close_pos)
            if (close_pos <= open_pos) return
            arg_text = trim(adjustl(low(open_pos + 1:close_pos - 1)))
            if (is_plain_identifier(arg_text)) cycle
            if (len_trim(arg_text) == 0) cycle
            write (location, '(" at line ",I0)') line_no
            error_msg = 'argument of PRESENT'//trim(location)// &
                ' must be an optional dummy argument name and '// &
                'must not be a subobject'
            return
        end do
    end procedure check_present_source_line

    ! A name is a plain identifier: a letter followed by letters, digits
    ! or underscores, with nothing else attached.
    module procedure is_plain_identifier
        character(len=:), allocatable :: trimmed
        integer :: i

        plain = .false.
        trimmed = trim(adjustl(text))
        if (len(trimmed) == 0) return
        if (index('abcdefghijklmnopqrstuvwxyz', trimmed(1:1)) == 0) return
        do i = 2, len(trimmed)
            if (.not. is_fortran_identifier_char(trimmed(i:i))) return
        end do
        plain = .true.
    end procedure is_plain_identifier

    module procedure get_pointer_source_lines
        logical :: found

        call get_source_text(arena, source, found)
        if (.not. found) call set_empty(source)
        if (.not. allocated(source)) call set_empty(source)
    end procedure get_pointer_source_lines

    ! POINTER (ptr, pointee) is the Cray pointer extension, distinct from
    ! the standard POINTER attribute statement (gfortran: "Cray pointer
    ! declaration ... requires -fcray-pointer").
    module procedure check_cray_pointer_line
        character(len=:), allocatable :: low
        integer :: rest
        character(len=64) :: location

        low = trim(adjustl(lowercase_text(code)))
        if (len(low) < 9) return
        if (low(1:7) /= 'pointer') return
        rest = 8
        do while (rest <= len(low))
            if (low(rest:rest) /= ' ') exit
            rest = rest + 1
        end do
        if (rest > len(low)) return
        if (low(rest:rest) /= '(') return
        write (location, '(" at line ",I0)') line_no
        error_msg = 'Cray pointer declaration'//trim(location)// &
            ' requires -fcray-pointer; the Cray pointer extension is '// &
            'not accepted'
    end procedure check_cray_pointer_line

    ! The TARGET argument of ASSOCIATED shall be a pointer or a variable
    ! with the TARGET attribute; a parenthesised expression is neither
    ! (gfortran: "must be a VARIABLE or FUNCTION").
    module procedure check_associated_target_line
        character(len=:), allocatable :: low, arg_text
        integer :: open_pos, close_pos, starts(8), ends(8), n_args
        character(len=64) :: location

        low = lowercase_text(code)
        call find_call_paren(low, 'associated', 1, open_pos)
        if (open_pos <= 0) return
        call matching_paren(low, open_pos, close_pos)
        if (close_pos <= open_pos + 1) return
        arg_text = low(open_pos + 1:close_pos - 1)
        call top_level_args(arg_text, starts, ends, n_args)
        if (n_args /= 2) return
        if (.not. is_parenthesised(arg_text(starts(2):ends(2)))) return
        write (location, '(" at line ",I0)') line_no
        error_msg = 'TARGET argument of ASSOCIATED'//trim(location)// &
            ' must be a VARIABLE or FUNCTION, not a parenthesised expression'
    end procedure check_associated_target_line

    ! A parenthesised actual argument is an expression, so it can never
    ! associate with a POINTER dummy (gfortran: "must be a pointer or a
    ! valid target").
    module procedure check_parenthesised_actual_line
        character(len=:), allocatable :: low, call_name, arg_text
        integer :: open_pos, close_pos, starts(8), ends(8), n_args, k, name_end
        logical :: dummy_pointer
        character(len=:), allocatable :: dummy_intent
        character(len=64) :: location

        low = trim(adjustl(lowercase_text(code)))
        if (len(low) < 6) return
        if (low(1:5) /= 'call ') return
        open_pos = index(low, '(')
        if (open_pos <= 6) return
        name_end = open_pos - 1
        do while (name_end >= 6)
            if (low(name_end:name_end) /= ' ') exit
            name_end = name_end - 1
        end do
        if (name_end < 6) return
        call_name = trim(adjustl(low(6:name_end)))
        if (len_trim(call_name) == 0) return
        call matching_paren(low, open_pos, close_pos)
        if (close_pos <= open_pos + 1) return
        arg_text = low(open_pos + 1:close_pos - 1)
        call top_level_args(arg_text, starts, ends, n_args)
        do k = 1, n_args
            if (.not. is_parenthesised(arg_text(starts(k):ends(k)))) cycle
            call dummy_pointer_intent(arena, call_name, k, dummy_pointer, &
                                      dummy_intent)
            if (.not. dummy_pointer) cycle
            write (location, '(" at line ",I0)') line_no
            error_msg = 'actual argument '//trim(arg_text(starts(k):ends(k)))// &
                ' for '''//trim(call_name)//''''//trim(location)// &
                ' must be a pointer or a valid target'
            return
        end do
    end procedure check_parenthesised_actual_line

    ! Position of the opening parenthesis of a call to name, or 0.
    module procedure find_call_paren
        integer :: hit, cursor, after

        open_pos = 0
        cursor = from
        do while (cursor <= len(low))
            hit = index(low(cursor:), name)
            if (hit == 0) return
            hit = cursor + hit - 1
            cursor = hit + len(name)
            if (hit > 1) then
                if (is_fortran_identifier_char(low(hit - 1:hit - 1))) cycle
            end if
            after = hit + len(name)
            do while (after <= len(low))
                if (low(after:after) /= ' ') exit
                after = after + 1
            end do
            if (after > len(low)) return
            if (low(after:after) /= '(') cycle
            open_pos = after
            return
        end do
    end procedure find_call_paren

    ! Position of the parenthesis closing the one at open_pos, or 0.
    module procedure matching_paren
        integer :: i, depth

        close_pos = 0
        depth = 0
        do i = open_pos, len(text)
            if (text(i:i) == '(') depth = depth + 1
            if (text(i:i) == ')') then
                depth = depth - 1
                if (depth == 0) then
                    close_pos = i
                    return
                end if
            end if
        end do
    end procedure matching_paren

    ! Split an argument list on commas that sit outside parentheses.
    module procedure top_level_args
        integer :: i, depth, arg_start

        n_args = 0
        depth = 0
        arg_start = 1
        if (len_trim(text) == 0) return
        do i = 1, len(text)
            if (text(i:i) == '(') depth = depth + 1
            if (text(i:i) == ')') depth = depth - 1
            if (text(i:i) /= ',') cycle
            if (depth /= 0) cycle
            if (n_args >= size(starts)) return
            n_args = n_args + 1
            starts(n_args) = arg_start
            ends(n_args) = i - 1
            arg_start = i + 1
        end do
        if (n_args >= size(starts)) return
        n_args = n_args + 1
        starts(n_args) = arg_start
        ends(n_args) = len(text)
    end procedure top_level_args

    ! An argument is parenthesised when one outer pair of parentheses
    ! wraps the whole expression.
    module procedure is_parenthesised
        character(len=:), allocatable :: trimmed
        integer :: close_pos

        wrapped = .false.
        trimmed = trim(adjustl(text))
        if (len(trimmed) < 3) return
        if (trimmed(1:1) /= '(') return
        call matching_paren(trimmed, 1, close_pos)
        if (close_pos /= len(trimmed)) return
        wrapped = .true.
    end procedure is_parenthesised
end submodule session_program_lowering_reject_pointer
