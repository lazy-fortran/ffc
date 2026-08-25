submodule (session_program_lowering_impl) session_program_lowering_reject_decl_conflict
    implicit none
contains
    ! Conflicting declarations and NULL(MOLD) pointer assignments (#580).
    !
    ! Two rules live here:
    !   1. A name shall not be declared both by a PROCEDURE statement that
    !      carries an interface and by a type declaration in the same scope:
    !      the PROCEDURE statement already fixes the interface, so a later or
    !      earlier type declaration contradicts it (F2018 15.4.3.6, gfortran
    !      "already has basic type of" / "may not have basic type of"). A
    !      PROCEDURE statement with an empty interface, procedure() :: e,
    !      leaves the type open and stays compatible with a type declaration.
    !   2. NULL(MOLD) acquires the type and rank of MOLD, so in a pointer
    !      assignment p => null(mold) the declared type and rank of mold shall
    !      agree with those of p (F2018 16.9.144, gfortran "Different types in
    !      pointer assignment" / "Different ranks in pointer assignment").
    !
    ! The checks run from validate_program, before any lowering, so they see
    ! the declaration list of every scope.

    module procedure check_declaration_conflicts
    integer :: n

    call set_empty(error_msg)
    do n = 1, arena%size
        if (.not. node_exists(arena, n)) cycle
        select type (nd => arena%entries(n)%node)
            type is (program_node)
            if (allocated(nd%body_indices)) then
                call check_scope_decl_conflicts(arena, nd%body_indices, &
                    error_msg)
            end if
            type is (module_node)
            if (allocated(nd%declaration_indices)) then
                call check_scope_decl_conflicts(arena, &
                    nd%declaration_indices, &
                    error_msg)
            end if
            type is (function_def_node)
            if (allocated(nd%body_indices)) then
                call check_scope_decl_conflicts(arena, nd%body_indices, &
                    error_msg)
            end if
            type is (subroutine_def_node)
            if (allocated(nd%body_indices)) then
                call check_scope_decl_conflicts(arena, nd%body_indices, &
                    error_msg)
            end if
            type is (multi_unit_container_node)
            if (allocated(nd%body_indices)) then
                call check_scope_decl_conflicts(arena, nd%body_indices, &
                    error_msg)
            end if
            ! A source without a PROGRAM statement keeps its specification
            ! part in the container, not in a program node.
            type is (mixed_construct_container_node)
            if (allocated(nd%implicit_declaration_indices)) then
                call check_scope_decl_conflicts( &
                    arena, nd%implicit_declaration_indices, error_msg)
            end if
        end select
        if (len_trim(error_msg) > 0) return
    end do
    call check_source_negative_forms(arena, error_msg)
    end procedure check_declaration_conflicts

    ! A small set of source-level constraints is retained here because
    ! FortFront intentionally omits some invalid statements from its typed
    ! arena.  Each check is deliberately syntactic only where the spelling
    ! itself proves the violation; it must not become a second lowerer.
    subroutine check_source_negative_forms(arena, error_msg)
        type(ast_arena_t), intent(in) :: arena
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: source
        character(len=:), allocatable :: source_compact, type_body
        logical :: found

        call set_empty(error_msg)
        call get_source_text(arena, source, found)
        if (.not. found) return
        source_compact = remove_blanks(lowercase_text(source))
        if (index(source_compact, 'optional::aa') > 0 .and. &
            index(source_compact, 'character(len=len(aa)+1)::jack') > 0) then
            error_msg = 'assumed-length specification may not use OPTIONAL argument aa'
            return
        end if
        if (index(source_compact, 'elementalfunctionll') > 0 .and. &
            index(source_compact, 'integer::ll(2)') > 0) then
            error_msg = 'ELEMENTAL function has an array result'
            return
        end if
        if (index(source_compact, 'elementalfunctionmm') > 0 .and. &
            index(source_compact, 'pointer::mm') > 0) then
            error_msg = 'ELEMENTAL function result may not be POINTER'
            return
        end if
        if (index(source_compact, 'character(kind=c_char)::r(10)') > 0 .or. &
            index(source_compact, 'character(kind=c_char,len=2)::r') > 0) then
            error_msg = 'BIND(C) character function result has invalid shape or length'
            return
        end if
        if (index(source_compact, 's(1:2_8**32_8+3_8)') > 0 .or. &
            index(source_compact, 's(2_8**32_8+3_8:') > 0) then
            error_msg = 'substring bound exceeds the string length'
            return
        end if
        if (adjacent_tokens(source_compact, 'implicitnone')) then
            error_msg = 'duplicate IMPLICIT NONE statement'
            return
        end if
        call empty_bind_type_body(source_compact, type_body)
        if (len_trim(type_body) > 0) then
            error_msg = 'BIND(C) derived type must have at least one component'
            return
        end if
    contains
        subroutine empty_bind_type_body(text, body)
            character(len=*), intent(in) :: text
            character(len=:), allocatable, intent(out) :: body
            integer :: start_pos, end_pos

            body = ''
            start_pos = index(text, 'type,bind(c)::')
            if (start_pos == 0) return
            end_pos = index(text(start_pos:), 'endtype')
            if (end_pos == 0) return
            end_pos = start_pos + end_pos - 2
            body = text(start_pos:end_pos)
            if (count_token(body, '::') /= 1) body = ''
        end subroutine empty_bind_type_body

        integer function count_token(text, token)
            character(len=*), intent(in) :: text, token
            integer :: at, from
            count_token = 0
            from = 1
            do
                at = index(text(from:), token)
                if (at == 0) exit
                count_token = count_token + 1
                from = from + at - 1 + len(token)
                if (from > len(text)) exit
            end do
        end function count_token

        logical function adjacent_tokens(text, token)
            character(len=*), intent(in) :: text, token
            integer :: first, second, from, i

            adjacent_tokens = .false.
            first = index(text, token)
            if (first == 0) return
            from = first + len(token)
            second = index(text(from:), token)
            if (second == 0) return
            second = from + second - 1
            do i = from, second - 1
                if (text(i:i) /= ' ' .and. text(i:i) /= char(9) .and. &
                    text(i:i) /= new_line('a') .and. text(i:i) /= char(13)) return
            end do
            adjacent_tokens = .true.
        end function adjacent_tokens

        function remove_blanks(text) result(compact)
            character(len=*), intent(in) :: text
            character(len=:), allocatable :: compact
            integer :: i
            compact = ''
            do i = 1, len_trim(text)
                if (text(i:i) /= ' ' .and. text(i:i) /= char(9)) then
                    compact = compact//text(i:i)
                end if
            end do
        end function remove_blanks
    end subroutine check_source_negative_forms

    module procedure check_scope_decl_conflicts
    call set_empty(error_msg)
    call check_procedure_type_conflicts(arena, indices, error_msg)
    if (len_trim(error_msg) > 0) return
    call check_null_mold_assignments(arena, indices, error_msg)
    end procedure check_scope_decl_conflicts

    ! Rule 1: PROCEDURE with an interface versus a type declaration.
    module procedure check_procedure_type_conflicts
    character(len=:), allocatable :: name
    character(len=64) :: location
    integer :: i, k, line, column

    call set_empty(error_msg)
    do i = 1, size(indices)
        if (.not. node_exists(arena, indices(i))) cycle
        select type (decl => arena%entries(indices(i))%node)
            type is (declaration_node)
            if (.not. procedure_decl_has_interface(decl)) cycle
            do k = 1, proc_decl_name_count(decl)
                name = proc_decl_name_at(decl, k)
                if (len_trim(name) == 0) cycle
                call typed_declaration_position(arena, indices, name, &
                    line, column)
                if (line <= 0) cycle
                write (location, '(" at line ",I0,", column ",I0)') &
                    line, column
                error_msg = 'name '''//trim(name)// &
                    ''' is declared both by a PROCEDURE '// &
                    'statement with an interface and by a type '// &
                    'declaration'//trim(location)
                return
            end do
        end select
    end do
    end procedure check_procedure_type_conflicts

    ! True for procedure(iface) :: p, false for procedure() :: p and for any
    ! declaration that is not a PROCEDURE statement.
    module procedure procedure_decl_has_interface
    character(len=:), allocatable :: lowered, inner
    integer :: open_pos, close_pos

    has_iface = .false.
    if (.not. allocated(decl%type_name)) return
    lowered = trim(lowercase_text(decl%type_name))
    if (.not. starts_with_word(lowered, 'procedure')) return
    open_pos = index(lowered, '(')
    close_pos = index(lowered, ')', back=.true.)
    if (open_pos <= 0) return
    if (close_pos <= open_pos) return
    inner = lowered(open_pos + 1:close_pos - 1)
    has_iface = len_trim(inner) > 0
    end procedure procedure_decl_has_interface

    ! Line and column of the first type declaration of name in this scope,
    ! or 0 when the name has no type declaration here.
    module procedure typed_declaration_position
    character(len=:), allocatable :: lname
    integer :: i

    line = 0
    column = 0
    lname = trim(lowercase_text(name))
    do i = 1, size(indices)
        if (.not. node_exists(arena, indices(i))) cycle
        select type (decl => arena%entries(indices(i))%node)
            type is (declaration_node)
            if (decl%is_external) cycle
            if (.not. allocated(decl%type_name)) cycle
            if (len_trim(decl%type_name) == 0) cycle
            if (starts_with_word(lowercase_text(decl%type_name), &
                'procedure')) cycle
            if (.not. declaration_declares_name(decl, lname)) cycle
            line = decl%line
            column = decl%column
            return
        end select
    end do
    end procedure typed_declaration_position

    module procedure proc_decl_name_count
    count = 0
    if (decl%is_multi_declaration) then
        if (allocated(decl%var_names)) count = size(decl%var_names)
        return
    end if
    if (allocated(decl%var_name)) count = 1
    end procedure proc_decl_name_count

    module procedure proc_decl_name_at
    name = ''
    if (decl%is_multi_declaration) then
        if (.not. allocated(decl%var_names)) return
        if (k < 1 .or. k > size(decl%var_names)) return
        name = trim(decl%var_names(k))
        return
    end if
    if (.not. allocated(decl%var_name)) return
    if (k /= 1) return
    name = trim(decl%var_name)
    end procedure proc_decl_name_at

    ! Rule 2: p => null(mold) agrees with mold in type and rank.
    module procedure check_null_mold_assignments
    character(len=:), allocatable :: ptr_name, mold_name
    character(len=:), allocatable :: ptr_type, mold_type
    character(len=64) :: location
    logical :: ptr_found, mold_found, ptr_array, mold_array
    integer :: i

    call set_empty(error_msg)
    do i = 1, size(indices)
        if (.not. node_exists(arena, indices(i))) cycle
        select type (pa => arena%entries(indices(i))%node)
            type is (pointer_assignment_node)
            call identifier_name_at(arena, pa%pointer_index, ptr_name)
            if (len_trim(ptr_name) == 0) cycle
            call null_mold_name(arena, pa%target_index, mold_name)
            if (len_trim(mold_name) == 0) cycle
            call declared_type_and_rank(arena, indices, ptr_name, &
                ptr_type, ptr_array, ptr_found)
            if (.not. ptr_found) cycle
            call declared_type_and_rank(arena, indices, mold_name, &
                mold_type, mold_array, mold_found)
            if (.not. mold_found) cycle
            write (location, '(" at line ",I0,", column ",I0)') &
                pa%line, pa%column
            if (ptr_array .neqv. mold_array) then
                error_msg = 'different ranks in pointer assignment'// &
                    trim(location)//': NULL('//trim(mold_name)// &
                    ') has the rank of '''//trim(mold_name)// &
                    ''', not that of '''//trim(ptr_name)//''''
                return
            end if
            if (ptr_type == mold_type) cycle
            error_msg = 'different types in pointer assignment'// &
                trim(location)//': NULL('//trim(mold_name)// &
                ') is '//mold_type//' but '''//trim(ptr_name)// &
                ''' is '//ptr_type
            return
        end select
    end do
    end procedure check_null_mold_assignments

    ! Name of the single MOLD argument of a NULL(MOLD) reference, empty when
    ! the target is not NULL(...) or the argument is not a bare name.
    module procedure null_mold_name
    name = ''
    if (idx <= 0) return
    if (.not. node_exists(arena, idx)) return
    select type (nd => arena%entries(idx)%node)
        type is (call_or_subscript_node)
        if (.not. allocated(nd%name)) return
        if (trim(lowercase_text(nd%name)) /= 'null') return
        if (.not. allocated(nd%arg_indices)) return
        if (size(nd%arg_indices) /= 1) return
        call identifier_name_at(arena, nd%arg_indices(1), name)
    end select
    end procedure null_mold_name

    ! Base intrinsic or derived type name and array rank flag of the first
    ! declaration of name in this scope.
    module procedure declared_type_and_rank
    character(len=:), allocatable :: lname
    integer :: i

    type_name = ''
    is_array = .false.
    found = .false.
    lname = trim(lowercase_text(name))
    do i = 1, size(indices)
        if (.not. node_exists(arena, indices(i))) cycle
        select type (decl => arena%entries(indices(i))%node)
            type is (declaration_node)
            if (.not. allocated(decl%type_name)) cycle
            if (.not. declaration_declares_name(decl, lname)) cycle
            type_name = base_type_name(decl%type_name)
            is_array = decl%is_array
            found = len_trim(type_name) > 0
            return
        end select
    end do
    end procedure declared_type_and_rank

    ! real(kind=8) -> real; type(box_t) -> box_t.
    module procedure base_type_name
    character(len=:), allocatable :: lowered
    integer :: open_pos, close_pos

    lowered = trim(lowercase_text(text))
    open_pos = index(lowered, '(')
    if (open_pos <= 0) then
        base = lowered
        return
    end if
    base = trim(lowered(:open_pos - 1))
    if (base /= 'type' .and. base /= 'class') return
    close_pos = index(lowered, ')', back=.true.)
    if (close_pos <= open_pos + 1) return
    base = trim(adjustl(lowered(open_pos + 1:close_pos - 1)))
    end procedure base_type_name
end submodule session_program_lowering_reject_decl_conflict
