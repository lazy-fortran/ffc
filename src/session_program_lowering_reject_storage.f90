submodule (session_program_lowering_impl) session_program_lowering_reject_storage
    implicit none
contains
    ! Storage-association restrictions on COMMON, EQUIVALENCE, SAVE, and
    ! BLOCK DATA (#392). These constraints are about how named objects are
    ! placed in shared storage, so each rule works from the declared
    ! attributes of the objects involved rather than from any statement
    ! spelling. FortFront drops several of the statements involved (a COMMON
    ! statement in a mixed-construct unit, a SAVE statement inside a BLOCK
    ! construct, an EQUIVALENCE set in a module) from the arena, so the
    ! statement lists themselves are recovered from the unit's own source
    ! text; the attributes they are checked against (SEQUENCE, BIND(C),
    ! ALLOCATABLE, SAVE, DATA membership) come from the typed nodes.
    !
    ! Rules enforced:
    !   * F2018 C8117/8.10.2.4: a COMMON block object of derived type must
    !     have the SEQUENCE or BIND(C) attribute and must not have an
    !     ultimate allocatable component.
    !   * F2023 C1108: a SAVE statement in a BLOCK construct may not name a
    !     common block.
    !   * F2018 C8110: EQUIVALENCE conflicts with BIND(C).
    !   * A name that is also the name of a program unit may not appear in a
    !     COMMON block.
    !   * F2018 C8108: SAVE conflicts with COMMON block membership.
    !   * F2018 C8105: every DATA object in a BLOCK DATA unit must be in
    !     COMMON.
    !
    ! Continued statements are not stitched back together here: every rule
    ! fires only on a statement whose first line already shows the violation,
    ! so a continuation can only make a check silent, never wrong.

    module procedure check_storage_association_restrictions
        character(len=256), allocatable :: lines(:)
        character(len=64) :: common_names(256)
        integer :: line_count, common_count
        logical :: found

        call set_empty(error_msg)
        call storage_source_lines(arena, lines, line_count, found)
        common_count = 0
        if (found) call collect_common_member_names(lines, line_count, &
                                                    common_names, common_count)
        call collect_common_node_members(arena, common_names, common_count)
        call check_common_derived_type_objects(lines, line_count, common_names, &
                                              common_count, error_msg)
        if (len_trim(error_msg) > 0) return
        call check_common_program_unit_names(arena, common_names, common_count, &
                                            error_msg)
        if (len_trim(error_msg) > 0) return
        call check_common_save_conflict(arena, common_names, common_count, &
                                        error_msg)
        if (len_trim(error_msg) > 0) return
        call check_block_construct_common_save(lines, line_count, error_msg)
        if (len_trim(error_msg) > 0) return
        call check_equivalence_bind_c_conflict(arena, lines, line_count, error_msg)
        if (len_trim(error_msg) > 0) return
        call check_block_data_objects_in_common(arena, lines, line_count, &
                                                error_msg)
    end procedure check_storage_association_restrictions

    module procedure storage_source_lines
        ! Split the unit's source into comment-stripped, lower-cased,
        ! left-justified lines so the statement-level scans below all see the
        ! same normalised form.
        character(len=:), allocatable :: source, line, code
        integer :: pos, next_nl, total

        line_count = 0
        call get_source_text(arena, source, found)
        if (.not. found) then
            allocate (lines(0))
            return
        end if
        total = 1
        do pos = 1, len(source)
            if (source(pos:pos) == new_line('a')) total = total + 1
        end do
        allocate (lines(total))
        lines = ''
        pos = 1
        do while (pos <= len(source))
            next_nl = index(source(pos:), new_line('a'))
            if (next_nl == 0) then
                line = source(pos:)
                pos = len(source) + 1
            else
                line = source(pos:pos + next_nl - 2)
                pos = pos + next_nl
            end if
            call strip_line_comment(line, code)
            if (line_count >= total) exit
            line_count = line_count + 1
            lines(line_count) = adjustl(lowercase_text(code))
        end do
    end procedure storage_source_lines

    module procedure append_storage_name
        integer :: i

        if (len_trim(name) == 0) return
        do i = 1, count
            if (trim(names(i)) == trim(name)) return
        end do
        if (count >= size(names)) return
        count = count + 1
        names(count) = name
    end procedure append_storage_name

    module procedure storage_name_listed
        integer :: i

        listed = .false.
        do i = 1, count
            if (trim(names(i)) == trim(name)) then
                listed = .true.
                return
            end if
        end do
    end procedure storage_name_listed

    subroutine collect_common_member_names(lines, line_count, names, count)
        ! Gather every object named in a COMMON statement. Text between
        ! slashes is a block name, not an object, so it is skipped.
        character(len=*), intent(in) :: lines(:)
        integer, intent(in) :: line_count
        character(len=*), intent(inout) :: names(:)
        integer, intent(inout) :: count
        character(len=:), allocatable :: rest
        integer :: i

        do i = 1, line_count
            if (.not. starts_with_word(trim(lines(i)), 'common')) cycle
            rest = trim(lines(i))
            rest = rest(7:)
            call collect_common_list_names(rest, names, count)
        end do
    end subroutine collect_common_member_names

    subroutine collect_common_list_names(text, names, count)
        character(len=*), intent(in) :: text
        character(len=*), intent(inout) :: names(:)
        integer, intent(inout) :: count
        character(len=:), allocatable :: item
        integer :: i, depth, item_start
        logical :: in_block_name

        in_block_name = .false.
        depth = 0
        item_start = 1
        do i = 1, len(text)
            if (text(i:i) == '/' .and. depth == 0) then
                in_block_name = .not. in_block_name
                item_start = i + 1
                cycle
            end if
            if (in_block_name) cycle
            if (text(i:i) == '(') depth = depth + 1
            if (text(i:i) == ')') depth = max(depth - 1, 0)
            if (text(i:i) == ',' .and. depth == 0) then
                item = text(item_start:i - 1)
                call append_storage_name(names, count, leading_identifier(item))
                item_start = i + 1
            end if
        end do
        if (in_block_name) return
        if (item_start > len(text)) return
        item = text(item_start:)
        call append_storage_name(names, count, leading_identifier(item))
    end subroutine collect_common_list_names

    subroutine collect_common_node_members(arena, names, count)
        ! Union in the members FortFront did keep as common_block_nodes, so a
        ! unit whose source text is unavailable is still covered.
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(inout) :: names(:)
        integer, intent(inout) :: count
        integer :: n, i

        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (node => arena%entries(n)%node)
            type is (common_block_node)
                if (.not. allocated(node%member_names)) cycle
                do i = 1, size(node%member_names)
                    call append_storage_name(names, count, &
                        lowercase_text(trim(node%member_names(i)%s)))
                end do
            end select
        end do
    end subroutine collect_common_node_members

    subroutine check_common_program_unit_names(arena, names, count, error_msg)
        ! A program unit name is a global identifier; it cannot also name a
        ! variable placed in a COMMON block.
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: names(:)
        integer, intent(in) :: count
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: unit_name
        integer :: n

        call set_empty(error_msg)
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            call storage_program_unit_name(arena, n, unit_name)
            if (len_trim(unit_name) == 0) cycle
            if (.not. storage_name_listed(names, count, unit_name)) cycle
            error_msg = "'"//unit_name//"' is also the name of a program "// &
                'unit and cannot appear in a COMMON block'
            return
        end do
    end subroutine check_common_program_unit_names

    subroutine storage_program_unit_name(arena, node_index, unit_name)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        character(len=:), allocatable, intent(out) :: unit_name

        call set_empty(unit_name)
        select type (node => arena%entries(node_index)%node)
        type is (program_node)
            if (allocated(node%name)) unit_name = lowercase_text(trim(node%name))
        type is (module_node)
            if (allocated(node%name)) unit_name = lowercase_text(trim(node%name))
        type is (block_data_node)
            if (allocated(node%name)) unit_name = lowercase_text(trim(node%name))
        end select
    end subroutine storage_program_unit_name

    subroutine check_common_save_conflict(arena, names, count, error_msg)
        ! An object in a COMMON block already has the storage lifetime of the
        ! block; SAVE on it is a conflicting attribute.
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: names(:)
        integer, intent(in) :: count
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: var_name
        integer :: n, i

        call set_empty(error_msg)
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (decl => arena%entries(n)%node)
            type is (declaration_node)
                if (.not. decl%is_save) cycle
                do i = 1, declaration_name_count(decl)
                    var_name = lowercase_text(declaration_name_at(decl, i))
                    if (.not. storage_name_listed(names, count, var_name)) cycle
                    error_msg = "COMMON block object '"//var_name// &
                        "' conflicts with SAVE attribute"
                    return
                end do
            end select
        end do
    end subroutine check_common_save_conflict

    integer function declaration_name_count(decl) result(total)
        type(declaration_node), intent(in) :: decl

        total = 1
        if (.not. decl%is_multi_declaration) return
        if (.not. allocated(decl%var_names)) return
        total = size(decl%var_names)
    end function declaration_name_count

    function declaration_name_at(decl, position) result(name)
        type(declaration_node), intent(in) :: decl
        integer, intent(in) :: position
        character(len=:), allocatable :: name

        call set_empty(name)
        if (decl%is_multi_declaration) then
            if (.not. allocated(decl%var_names)) return
            if (position > size(decl%var_names)) return
            name = trim(decl%var_names(position))
            return
        end if
        if (allocated(decl%var_name)) name = trim(decl%var_name)
    end function declaration_name_at

    subroutine check_block_construct_common_save(lines, line_count, error_msg)
        ! F2023 C1108: the saved-entity-list of a SAVE statement in a BLOCK
        ! construct may not specify a common block name.
        character(len=*), intent(in) :: lines(:)
        integer, intent(in) :: line_count
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: line
        integer :: i, depth

        call set_empty(error_msg)
        depth = 0
        do i = 1, line_count
            line = trim(lines(i))
            if (starts_with_word(line, 'end')) then
                if (starts_with_word(trim(adjustl(line(4:))), 'block')) &
                    depth = max(depth - 1, 0)
                cycle
            end if
            if (starts_with_word(line, 'endblock')) then
                depth = max(depth - 1, 0)
                cycle
            end if
            if (starts_with_word(line, 'block')) then
                if (.not. starts_with_word(trim(adjustl(line(6:))), 'data')) &
                    depth = depth + 1
                cycle
            end if
            if (depth <= 0) cycle
            if (.not. starts_with_word(line, 'save')) cycle
            if (index(line, '/') == 0) cycle
            error_msg = 'SAVE of a COMMON block is not allowed in a '// &
                'BLOCK construct'
            return
        end do
    end subroutine check_block_construct_common_save

    subroutine check_equivalence_bind_c_conflict(arena, lines, line_count, &
                                                 error_msg)
        ! F2018 C8110: an object with the BIND(C) attribute has C storage
        ! association and may not be equivalenced.
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: lines(:)
        integer, intent(in) :: line_count
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=64) :: bind_names(256), equiv_names(256)
        integer :: bind_count, equiv_count, i

        call set_empty(error_msg)
        bind_count = 0
        equiv_count = 0
        call collect_bind_c_names(arena, lines, line_count, bind_names, bind_count)
        call collect_equivalence_names(lines, line_count, equiv_names, equiv_count)
        do i = 1, equiv_count
            if (.not. storage_name_listed(bind_names, bind_count, &
                                          trim(equiv_names(i)))) cycle
            error_msg = "EQUIVALENCE attribute conflicts with BIND(C) "// &
                "attribute in variable '"//trim(equiv_names(i))//"'"
            return
        end do
    end subroutine check_equivalence_bind_c_conflict

    subroutine collect_bind_c_names(arena, lines, line_count, names, count)
        ! BIND(C) reaches a variable either as a declaration attribute or as a
        ! standalone BIND statement naming already-declared objects.
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: lines(:)
        integer, intent(in) :: line_count
        character(len=*), intent(inout) :: names(:)
        integer, intent(inout) :: count
        character(len=:), allocatable :: line, rest
        integer :: i, n, sep

        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (decl => arena%entries(n)%node)
            type is (declaration_node)
                if (.not. decl%is_bind_c) cycle
                do i = 1, declaration_name_count(decl)
                    call append_storage_name(names, count, &
                        lowercase_text(declaration_name_at(decl, i)))
                end do
            end select
        end do
        do i = 1, line_count
            line = trim(lines(i))
            if (.not. starts_with_word(line, 'bind')) cycle
            sep = index(line, '::')
            if (sep == 0) cycle
            rest = line(sep + 2:)
            call collect_common_list_names(rest, names, count)
        end do
    end subroutine collect_bind_c_names

    subroutine collect_equivalence_names(lines, line_count, names, count)
        character(len=*), intent(in) :: lines(:)
        integer, intent(in) :: line_count
        character(len=*), intent(inout) :: names(:)
        integer, intent(inout) :: count
        character(len=:), allocatable :: line
        integer :: i, j
        logical :: at_name_start

        do i = 1, line_count
            line = trim(lines(i))
            if (.not. starts_with_word(line, 'equivalence')) cycle
            at_name_start = .true.
            do j = 12, len(line)
                if (.not. is_fortran_identifier_char(line(j:j))) then
                    at_name_start = line(j:j) /= ')'
                    cycle
                end if
                if (at_name_start) then
                    call append_storage_name(names, count, &
                                             leading_identifier(line(j:)))
                    at_name_start = .false.
                end if
            end do
        end do
    end subroutine collect_equivalence_names

    subroutine check_block_data_objects_in_common(arena, lines, line_count, &
                                                  error_msg)
        ! F2018 C8105: a BLOCK DATA unit initialises COMMON storage only, so
        ! every object it names in a DATA statement must be in a COMMON block
        ! of that unit, directly or through EQUIVALENCE association.
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: lines(:)
        integer, intent(in) :: line_count
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: n

        call set_empty(error_msg)
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (unit => arena%entries(n)%node)
            type is (block_data_node)
                if (.not. allocated(unit%statement_indices)) cycle
                call check_one_block_data_unit(arena, unit%statement_indices, &
                                               lines, line_count, error_msg)
                if (len_trim(error_msg) > 0) return
            end select
        end do
    end subroutine check_block_data_objects_in_common

    subroutine check_one_block_data_unit(arena, statement_indices, lines, &
                                         line_count, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: statement_indices(:)
        character(len=*), intent(in) :: lines(:)
        integer, intent(in) :: line_count
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=64) :: members(256)
        character(len=:), allocatable :: obj_name
        integer :: member_count, s, i, obj

        call set_empty(error_msg)
        member_count = 0
        do s = 1, size(statement_indices)
            if (.not. node_exists(arena, statement_indices(s))) cycle
            select type (node => arena%entries(statement_indices(s))%node)
            type is (common_block_node)
                if (.not. allocated(node%member_names)) cycle
                do i = 1, size(node%member_names)
                    call append_storage_name(members, member_count, &
                        lowercase_text(trim(node%member_names(i)%s)))
                end do
            end select
        end do
        call add_equivalenced_common_names(lines, line_count, members, &
                                           member_count)
        do s = 1, size(statement_indices)
            if (.not. node_exists(arena, statement_indices(s))) cycle
            select type (node => arena%entries(statement_indices(s))%node)
            type is (data_statement_node)
                if (.not. allocated(node%object_indices)) cycle
                do i = 1, size(node%object_indices)
                    obj = node%object_indices(i)
                    call data_object_base_name(arena, obj, obj_name)
                    if (len_trim(obj_name) == 0) cycle
                    if (storage_name_listed(members, member_count, obj_name)) cycle
                    error_msg = "DATA object '"//obj_name// &
                        "' in a BLOCK DATA unit must be in COMMON"
                    return
                end do
            end select
        end do
    end subroutine check_one_block_data_unit

    ! An object EQUIVALENCEd to a COMMON member is itself in COMMON storage,
    ! so a BLOCK DATA unit may initialise it (lazy-fortran/ffc#581). Association
    ! chains through several groups are followed to a fixed point.
    subroutine add_equivalenced_common_names(lines, line_count, members, &
                                             member_count)
        character(len=*), intent(in) :: lines(:)
        integer, intent(in) :: line_count
        character(len=*), intent(inout) :: members(:)
        integer, intent(inout) :: member_count
        character(len=64) :: group(64)
        character(len=:), allocatable :: line
        integer :: pass, i, j, group_count, pos
        logical :: changed, in_common

        do pass = 1, 16
            changed = .false.
            do i = 1, line_count
                line = trim(lines(i))
                if (.not. starts_with_word(line, 'equivalence')) cycle
                pos = 1
                do
                    call next_equivalence_group(line, pos, group, group_count)
                    if (group_count == 0) exit
                    in_common = .false.
                    do j = 1, group_count
                        if (storage_name_listed(members, member_count, &
                                                trim(group(j)))) in_common = .true.
                    end do
                    if (.not. in_common) cycle
                    do j = 1, group_count
                        if (storage_name_listed(members, member_count, &
                                                trim(group(j)))) cycle
                        call append_storage_name(members, member_count, &
                                                 trim(group(j)))
                        changed = .true.
                    end do
                end do
            end do
            if (.not. changed) exit
        end do
    end subroutine add_equivalenced_common_names

    subroutine next_equivalence_group(text, pos, names, count)
        !! Collect the object names of the next parenthesised EQUIVALENCE
        !! group starting at pos; subscripts nested inside a designator are
        !! skipped. Returns count == 0 when no further group is present.
        character(len=*), intent(in) :: text
        integer, intent(inout) :: pos
        character(len=*), intent(out) :: names(:)
        integer, intent(out) :: count
        character(len=:), allocatable :: name
        integer :: depth

        count = 0
        depth = 0
        do while (pos <= len(text))
            if (text(pos:pos) == '(') then
                depth = depth + 1
                pos = pos + 1
                cycle
            end if
            if (text(pos:pos) == ')') then
                depth = depth - 1
                pos = pos + 1
                if (depth <= 0) return
                cycle
            end if
            if (depth /= 1) then
                pos = pos + 1
                cycle
            end if
            if (.not. is_fortran_identifier_char(text(pos:pos))) then
                pos = pos + 1
                cycle
            end if
            name = leading_identifier(text(pos:))
            if (count < size(names)) then
                count = count + 1
                names(count) = lowercase_text(trim(name))
            end if
            pos = pos + len_trim(name)
        end do
    end subroutine next_equivalence_group

    subroutine data_object_base_name(arena, node_index, name)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        character(len=:), allocatable, intent(out) :: name

        call set_empty(name)
        if (.not. node_exists(arena, node_index)) return
        select type (node => arena%entries(node_index)%node)
        type is (identifier_node)
            if (allocated(node%name)) name = lowercase_text(trim(node%name))
        type is (call_or_subscript_node)
            if (allocated(node%name)) name = lowercase_text(trim(node%name))
        end select
    end subroutine data_object_base_name

    subroutine check_common_derived_type_objects(lines, line_count, names, &
                                                 count, error_msg)
        ! F2018 8.10.2.4: a COMMON block object of derived type must have the
        ! SEQUENCE or the BIND(C) attribute, and (C8117) must not have an
        ! ultimate component that is allocatable.
        character(len=*), intent(in) :: lines(:)
        integer, intent(in) :: line_count
        character(len=*), intent(in) :: names(:)
        integer, intent(in) :: count
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=64) :: type_names(64)
        character(len=64) :: comp_types(64, 32)
        logical :: type_seq(64), type_bind_c(64), type_alloc(64)
        integer :: comp_counts(64), type_count
        character(len=:), allocatable :: type_name
        integer :: i, t

        call set_empty(error_msg)
        call collect_derived_type_table(lines, line_count, type_names, &
                                        type_seq, type_bind_c, type_alloc, &
                                        comp_types, comp_counts, type_count)
        do i = 1, count
            call storage_declared_derived_type_of(lines, line_count, trim(names(i)), &
                                          type_name)
            if (len_trim(type_name) == 0) cycle
            t = derived_type_index(type_names, type_count, type_name)
            if (t == 0) cycle
            if (.not. type_seq(t) .and. .not. type_bind_c(t)) then
                error_msg = "COMMON block object '"//trim(names(i))// &
                    "' of derived type '"//type_name//"' has neither the "// &
                    'SEQUENCE nor the BIND(C) attribute'
                return
            end if
            if (derived_type_has_ultimate_allocatable(type_names, type_alloc, &
                                                      comp_types, comp_counts, &
                                                      type_count, t, 0)) then
                error_msg = "COMMON block object '"//trim(names(i))// &
                    "' of derived type '"//type_name//"' has an ultimate "// &
                    'component that is allocatable'
                return
            end if
        end do
    end subroutine check_common_derived_type_objects

    integer function derived_type_index(type_names, type_count, name) result(idx)
        character(len=*), intent(in) :: type_names(:)
        integer, intent(in) :: type_count
        character(len=*), intent(in) :: name
        integer :: i

        idx = 0
        do i = 1, type_count
            if (trim(type_names(i)) == trim(name)) then
                idx = i
                return
            end if
        end do
    end function derived_type_index

    recursive function derived_type_has_ultimate_allocatable(type_names, &
            type_alloc, comp_types, comp_counts, type_count, t, depth) &
            result(has_alloc)
        character(len=*), intent(in) :: type_names(:)
        logical, intent(in) :: type_alloc(:)
        character(len=*), intent(in) :: comp_types(:, :)
        integer, intent(in) :: comp_counts(:)
        integer, intent(in) :: type_count, t, depth
        logical :: has_alloc
        integer :: i, sub

        has_alloc = .false.
        if (depth > 32) return
        if (type_alloc(t)) then
            has_alloc = .true.
            return
        end if
        do i = 1, comp_counts(t)
            sub = derived_type_index(type_names, type_count, &
                                     trim(comp_types(t, i)))
            if (sub == 0) cycle
            if (sub == t) cycle
            if (derived_type_has_ultimate_allocatable(type_names, type_alloc, &
                    comp_types, comp_counts, type_count, sub, depth + 1)) then
                has_alloc = .true.
                return
            end if
        end do
    end function derived_type_has_ultimate_allocatable

    subroutine collect_derived_type_table(lines, line_count, type_names, &
                                          type_seq, type_bind_c, type_alloc, &
                                          comp_types, comp_counts, type_count)
        ! Build one entry per derived type definition in the source: whether it
        ! carries SEQUENCE or BIND(C), whether it declares an allocatable
        ! component directly, and the derived types of its components.
        character(len=*), intent(in) :: lines(:)
        integer, intent(in) :: line_count
        character(len=*), intent(out) :: type_names(:)
        logical, intent(out) :: type_seq(:), type_bind_c(:), type_alloc(:)
        character(len=*), intent(out) :: comp_types(:, :)
        integer, intent(out) :: comp_counts(:)
        integer, intent(out) :: type_count
        character(len=:), allocatable :: line, name, attrs
        integer :: i, current

        type_count = 0
        current = 0
        type_names = ''
        comp_types = ''
        comp_counts = 0
        type_seq = .false.
        type_bind_c = .false.
        type_alloc = .false.
        do i = 1, line_count
            line = trim(lines(i))
            if (is_end_type_line(line)) then
                current = 0
                cycle
            end if
            if (is_derived_type_header(line)) then
                if (type_count >= size(type_names)) then
                    current = 0
                    cycle
                end if
                call derived_type_header_parts(line, name, attrs)
                type_count = type_count + 1
                current = type_count
                type_names(current) = name
                type_bind_c(current) = index(attrs, 'bind') > 0
                cycle
            end if
            if (current == 0) cycle
            call record_type_component(line, current, type_seq, type_alloc, &
                                       comp_types, comp_counts)
        end do
    end subroutine collect_derived_type_table

    subroutine record_type_component(line, current, type_seq, type_alloc, &
                                     comp_types, comp_counts)
        character(len=*), intent(in) :: line
        integer, intent(in) :: current
        logical, intent(inout) :: type_seq(:), type_alloc(:)
        character(len=*), intent(inout) :: comp_types(:, :)
        integer, intent(inout) :: comp_counts(:)
        character(len=:), allocatable :: inner

        if (starts_with_word(line, 'sequence')) then
            type_seq(current) = .true.
            return
        end if
        if (index(line, 'allocatable') > 0) type_alloc(current) = .true.
        inner = type_spec_inner_name(line)
        if (len_trim(inner) == 0) return
        if (comp_counts(current) >= size(comp_types, 2)) return
        comp_counts(current) = comp_counts(current) + 1
        comp_types(current, comp_counts(current)) = inner
    end subroutine record_type_component

    logical function is_end_type_line(line) result(is_end)
        character(len=*), intent(in) :: line

        is_end = .false.
        if (starts_with_word(line, 'endtype')) then
            is_end = .true.
            return
        end if
        if (.not. starts_with_word(line, 'end')) return
        is_end = starts_with_word(trim(adjustl(line(4:))), 'type')
    end function is_end_type_line

    logical function is_derived_type_header(line) result(is_header)
        ! "type name", "type :: name", and "type, attrs :: name" open a
        ! definition; "type(x) ..." and "type is (...)" do not.
        character(len=*), intent(in) :: line
        character(len=:), allocatable :: rest

        is_header = .false.
        if (.not. starts_with_word(line, 'type')) return
        if (len(line) < 5) return
        rest = trim(adjustl(line(5:)))
        if (len(rest) == 0) return
        if (rest(1:1) == '(') return
        if (starts_with_word(rest, 'is')) return
        is_header = .true.
    end function is_derived_type_header

    subroutine derived_type_header_parts(line, name, attrs)
        character(len=*), intent(in) :: line
        character(len=:), allocatable, intent(out) :: name, attrs
        character(len=:), allocatable :: rest
        integer :: sep

        rest = trim(adjustl(line(5:)))
        sep = index(rest, '::')
        if (sep > 0) then
            attrs = rest(1:sep - 1)
            name = leading_identifier(rest(sep + 2:))
        else
            attrs = ''
            name = leading_identifier(rest)
        end if
    end subroutine derived_type_header_parts

    function type_spec_inner_name(line) result(inner)
        ! The type name inside a "type(name)" type specification, if the line
        ! opens with one.
        character(len=*), intent(in) :: line
        character(len=:), allocatable :: inner
        character(len=:), allocatable :: rest
        integer :: close_pos

        call set_empty(inner)
        if (.not. starts_with_word(line, 'type')) return
        if (len(line) < 5) return
        rest = trim(adjustl(line(5:)))
        if (len(rest) == 0) return
        if (rest(1:1) /= '(') return
        close_pos = index(rest, ')')
        if (close_pos < 3) return
        inner = leading_identifier(rest(2:close_pos - 1))
    end function type_spec_inner_name

    subroutine storage_declared_derived_type_of(lines, line_count, var_name, type_name)
        ! Find the derived type of var_name from a "type(x) :: ..." entity
        ! declaration outside any type definition body.
        character(len=*), intent(in) :: lines(:)
        integer, intent(in) :: line_count
        character(len=*), intent(in) :: var_name
        character(len=:), allocatable, intent(out) :: type_name
        character(len=64) :: entities(256)
        character(len=:), allocatable :: line, inner, rest
        integer :: i, sep, entity_count, close_pos
        logical :: in_type

        call set_empty(type_name)
        in_type = .false.
        do i = 1, line_count
            line = trim(lines(i))
            if (is_end_type_line(line)) then
                in_type = .false.
                cycle
            end if
            if (is_derived_type_header(line)) then
                in_type = .true.
                cycle
            end if
            if (in_type) cycle
            inner = type_spec_inner_name(line)
            if (len_trim(inner) == 0) cycle
            sep = index(line, '::')
            if (sep > 0) then
                rest = line(sep + 2:)
            else
                close_pos = index(line, ')')
                rest = line(close_pos + 1:)
            end if
            entity_count = 0
            call collect_common_list_names(rest, entities, entity_count)
            if (.not. storage_name_listed(entities, entity_count, var_name)) cycle
            type_name = inner
            return
        end do
    end subroutine storage_declared_derived_type_of
end submodule session_program_lowering_reject_storage
