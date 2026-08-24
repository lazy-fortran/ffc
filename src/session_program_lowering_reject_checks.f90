submodule (session_program_lowering_impl) session_program_lowering_reject_checks
    implicit none
contains
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
    module procedure check_generic_ambiguity
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
    end procedure check_generic_ambiguity

    module procedure specifics_indistinguishable
        integer :: count_a, count_b, pos
        integer :: kind_a, kind_b, rank_a, rank_b
        logical :: known_a, known_b, proc_a, proc_b, any_a, any_b
        character(len=:), allocatable :: base_a, base_b

        same = .false.
        count_a = arena_proc_param_count(arena, name_a)
        count_b = arena_proc_param_count(arena, name_b)
        if (count_a < 0 .or. count_b < 0) return
        if (count_a /= count_b) return
        do pos = 1, count_a
            call dummy_signature(arena, name_a, pos, known_a, base_a, kind_a, &
                                 rank_a, proc_a, any_a)
            call dummy_signature(arena, name_b, pos, known_b, base_b, kind_b, &
                                 rank_b, proc_b, any_b)
            if (.not. known_a) return
            if (.not. known_b) return
            ! A procedure dummy is never distinguished from another procedure
            ! dummy by the type of its result (F2018 C1514 applies TKR only to
            ! data objects), so such a position cannot resolve the generic.
            if (proc_a .neqv. proc_b) return
            if (proc_a) cycle
            ! An unlimited polymorphic dummy is type compatible with every
            ! actual argument, so it distinguishes nothing.
            if (any_a .or. any_b) cycle
            if (rank_a /= rank_b) return
            if (base_a /= base_b) return
            if (kind_a /= 0 .and. kind_b /= 0) then
                if (kind_a /= kind_b) return
            end if
        end do
        same = .true.
    end procedure specifics_indistinguishable

    ! Type, kind and rank facts about the pos-th dummy argument of the
    ! procedure proc_name, as needed by the generic-ambiguity rule. known is
    ! false whenever the procedure or the dummy position cannot be resolved
    ! statically; callers must then claim no ambiguity.
    module procedure dummy_signature
        integer :: n
        character(len=:), allocatable :: node_type
        type(function_def_node), pointer :: fn_node
        type(subroutine_def_node), pointer :: sb_node

        known = .false.
        base_name = ''
        kind_value = 0
        rank = 0
        is_proc = .false.
        is_any = .false.
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            node_type = get_node_type_at(arena, n)
            if (node_type == 'subroutine_def_node' .or. &
                node_type == 'subroutine_def') then
                sb_node => get_node_as_subroutine_def(arena, n)
                if (.not. associated(sb_node)) cycle
                if (.not. allocated(sb_node%name)) cycle
                if (.not. same_name(sb_node%name, proc_name)) cycle
                call dummy_signature_at(arena, sb_node%param_indices, &
                                        sb_node%body_indices, pos, known, &
                                        base_name, kind_value, rank, is_proc, &
                                        is_any)
                call refine_dummy_signature(arena, proc_name, pos, kind_value, &
                                            rank)
                return
            else if (node_type == 'function_def_node' .or. &
                     node_type == 'function_def') then
                fn_node => get_node_as_function_def(arena, n)
                if (.not. associated(fn_node)) cycle
                if (.not. allocated(fn_node%name)) cycle
                if (.not. same_name(fn_node%name, proc_name)) cycle
                call dummy_signature_at(arena, fn_node%param_indices, &
                                        fn_node%body_indices, pos, known, &
                                        base_name, kind_value, rank, is_proc, &
                                        is_any)
                call refine_dummy_signature(arena, proc_name, pos, kind_value, &
                                            rank)
                return
            end if
        end do
    end procedure dummy_signature

    ! Fill in rank and kind from the lowerer's own dummy queries when the
    ! declaration node did not carry them.
    module procedure refine_dummy_signature

        if (rank == 0) then
            if (callee_dummy_is_array(arena, proc_name, pos)) rank = 1
        end if
        if (kind_value == 0) then
            kind_value = callee_dummy_value_kind(arena, proc_name, pos)
        end if
    end procedure refine_dummy_signature

    module procedure dummy_signature_at
        character(len=:), allocatable :: name, name_err
        logical :: unresolved

        known = .false.
        base_name = ''
        kind_value = 0
        rank = 0
        is_proc = .false.
        is_any = .false.
        unresolved = .false.
        if (.not. allocated(param_indices)) return
        if (pos < 1 .or. pos > size(param_indices)) return
        call parameter_name(arena, param_indices(pos), name, name_err)
        if (len_trim(name_err) > 0) return
        if (len_trim(name) == 0) return
        if (trim(name) == '*') then
            ! An alternate-return dummy carries no type; two of them in the
            ! same position are indistinguishable.
            known = .true.
            base_name = '*'
            return
        end if
        select type (pn => arena%entries(param_indices(pos))%node)
        type is (parameter_declaration_node)
            if (allocated(pn%type_name)) then
                if (len_trim(pn%type_name) > 0) then
                    base_name = normalized_base_type(pn%type_name)
                    is_any = base_name == 'class(*)'
                    if (pn%is_array) then
                        rank = 1
                        if (allocated(pn%dimension_indices)) then
                            rank = size(pn%dimension_indices)
                        end if
                    end if
                    ! The dummy's own declaration in the specification part
                    ! carries the rank when the parameter node does not
                    ! (#595): differing rank distinguishes two specifics.
                    if (rank == 0) then
                        rank = dummy_declared_rank(arena, body_indices, name)
                    end if
                    if (pn%has_kind) then
                        kind_value = pn%kind_value
                        if (pn%kind_value <= 0) return
                    end if
                    known = .true.
                    return
                end if
            end if
        end select
        call dummy_decl_signature(arena, body_indices, name, known, base_name, &
                                  kind_value, rank, is_proc, is_any, unresolved)
        if (unresolved) then
            known = .false.
            return
        end if
        if (known) return
        ! No declaration in the body: implicit typing gives this dummy the
        ! default type for its initial letter.
        base_name = implicit_base_type(name)
        known = len_trim(base_name) > 0
    end procedure dummy_signature_at

    ! The rank a dummy is given by its own declaration in the specification
    ! part, or 0 when no declaration in body_indices names it as an array of
    ! statically known rank. Used where the parameter node itself carries no
    ! shape, so that a rank-1 and a rank-2 specific of the same element kind
    ! stay distinguishable (F2018 C1514, #595).
    module procedure dummy_declared_rank
        integer :: i

        rank = 0
        if (.not. allocated(body_indices)) return
        do i = 1, size(body_indices)
            if (.not. node_exists(arena, body_indices(i))) cycle
            select type (decl => arena%entries(body_indices(i))%node)
            type is (declaration_node)
                if (.not. decl%is_array) cycle
                if (.not. declaration_declares_name(decl, &
                                        trim(lowercase_text(param_name)))) cycle
                if (allocated(decl%dimension_indices)) then
                    rank = max(rank, size(decl%dimension_indices))
                else
                    rank = max(rank, 1)
                end if
            end select
        end do
    end procedure dummy_declared_rank

    ! unresolved reports a KIND selector that is not a literal - the declared
    ! kind is then unknown, and no distinguishability claim may rest on it.
    module procedure dummy_decl_signature
        integer :: i, k
        logical :: names_it

        found = .false.
        base_name = ''
        kind_value = 0
        rank = 0
        is_proc = .false.
        is_any = .false.
        unresolved = .false.
        if (.not. allocated(body_indices)) return
        do i = 1, size(body_indices)
            if (.not. node_exists(arena, body_indices(i))) cycle
            select type (decl => arena%entries(body_indices(i))%node)
            type is (declaration_node)
                names_it = .false.
                if (allocated(decl%var_name)) then
                    if (same_name(decl%var_name, param_name)) names_it = .true.
                end if
                if (decl%is_multi_declaration .and. allocated(decl%var_names)) then
                    do k = 1, size(decl%var_names)
                        if (same_name(decl%var_names(k), param_name)) then
                            names_it = .true.
                        end if
                    end do
                end if
                if (.not. names_it) cycle
                if (decl%is_external) is_proc = .true.
                if (decl%is_array) then
                    if (allocated(decl%dimension_indices)) then
                        rank = max(rank, size(decl%dimension_indices))
                    else
                        rank = max(rank, 1)
                    end if
                end if
                if (allocated(decl%type_name)) then
                    if (len_trim(decl%type_name) > 0) then
                        base_name = normalized_base_type(decl%type_name)
                        is_any = base_name == 'class(*)'
                        found = .true.
                    end if
                end if
                if (decl%has_kind) then
                    kind_value = decl%kind_value
                    if (decl%kind_value <= 0) unresolved = .true.
                end if
            end select
        end do
        if (is_proc) found = .true.
    end procedure dummy_decl_signature

    ! Collapse a declared type string to the form that decides generic
    ! distinguishability. CLASS(T) and TYPE(T) name the same declared type; the
    ! kind selector is kept, because two specifics of the same type but
    ! different kind are distinguishable.

    ! The type name without its kind or length selector.

    ! Default implicit type of a name: integer for I-N, real otherwise.

    module procedure arena_proc_param_count
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
    end procedure arena_proc_param_count

    ! A typed array constructor [T :: ...] whose type-spec T is an intrinsic
    ! numeric, character, or logical type requires every element to convert to
    ! T. Character, numeric, and logical form three disjoint classes with no
    ! intrinsic conversion between them, so a literal element of a class other
    ! than T's is invalid Fortran (gfortran: "Cannot convert ... to ..."). This
    ! flags only that provably wrong case: an explicit numeric/character/logical
    ! type-spec with a literal leaf element of a different class. Elements whose
    ! class is not statically a literal (identifiers, expressions, calls) are
    ! left unclassified so a valid constructor is never rejected. Nested
    ! constructors flatten into the outer type-spec, so their literal leaves are
    ! checked against the same class.
    module procedure check_array_constructor_type_specs
        integer :: n

        call set_empty(error_msg)
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            call check_one_array_constructor(arena, n, error_msg)
            if (len_trim(error_msg) > 0) return
        end do
    end procedure check_array_constructor_type_specs

    module procedure check_one_array_constructor
        integer :: spec_class

        call set_empty(error_msg)
        select type (nd => arena%entries(node_index)%node)
            type is (array_literal_node)
            if (.not. allocated(nd%type_spec)) return
            if (len_trim(nd%type_spec) == 0) return
            spec_class = array_ctor_typespec_class(nd%type_spec)
            if (spec_class == CMP_CLASS_UNKNOWN) return
            if (.not. allocated(nd%element_indices)) return
            call check_array_ctor_elements(arena, nd%element_indices, &
                spec_class, nd%type_spec, nd%line, nd%column, error_msg)
        end select
    end procedure check_one_array_constructor

    module procedure check_array_ctor_elements
        integer :: i, ei, elem_class
        character(len=64) :: location

        call set_empty(error_msg)
        do i = 1, size(elems)
            ei = elems(i)
            if (.not. node_exists(arena, ei)) cycle
            select type (child => arena%entries(ei)%node)
                type is (array_literal_node)
                if (allocated(child%element_indices)) then
                    call check_array_ctor_elements(arena, &
                        child%element_indices, spec_class, spec_text, line, &
                        col, error_msg)
                    if (len_trim(error_msg) > 0) return
                end if
                cycle
            end select
            elem_class = array_ctor_literal_class(arena, ei)
            if (elem_class == CMP_CLASS_UNKNOWN) cycle
            if (elem_class /= spec_class) then
                if (line > 0 .and. col > 0) then
                    write (location, '(" at line ",I0,", column ",I0)') line, col
                else
                    location = ''
                end if
                error_msg = 'array constructor'//trim(location)// &
                    ' cannot convert '//cmp_class_name(elem_class)// &
                    ' element to '//trim(adjustl(spec_text))
                return
            end if
        end do
    end procedure check_array_ctor_elements

    module procedure array_ctor_typespec_class
        character(len=:), allocatable :: spec_lc

        spec_lc = trim(adjustl(lowercase_text(type_spec)))
        if (starts_with_word(spec_lc, 'character')) then
            cls = CMP_CLASS_CHAR
        else if (starts_with_word(spec_lc, 'logical')) then
            cls = CMP_CLASS_LOGICAL
        else if (starts_with_word(spec_lc, 'integer') .or. &
                 starts_with_word(spec_lc, 'real') .or. &
                 starts_with_word(spec_lc, 'complex') .or. &
                 starts_with_word(spec_lc, 'double')) then
            cls = CMP_CLASS_NUMERIC
        else
            cls = CMP_CLASS_UNKNOWN
        end if
    end procedure array_ctor_typespec_class


    module procedure array_ctor_literal_class
        character(len=:), allocatable :: value, literal_type, err

        cls = CMP_CLASS_UNKNOWN
        call get_literal_info(arena, node_index, value, literal_type, err)
        if (len_trim(err) > 0) return
        if (is_character_literal(arena, node_index)) then
            cls = CMP_CLASS_CHAR
        else if (is_logical_literal(arena, node_index)) then
            cls = CMP_CLASS_LOGICAL
        else
            cls = CMP_CLASS_NUMERIC
        end if
    end procedure array_ctor_literal_class

    module procedure cmp_class_name

        select case (cls)
        case (CMP_CLASS_NUMERIC)
            name = 'numeric'
        case (CMP_CLASS_CHAR)
            name = 'character'
        case (CMP_CLASS_LOGICAL)
            name = 'logical'
        case default
            name = 'unknown'
        end select
    end procedure cmp_class_name

    module procedure check_gcc_calling_convention_assignments
        integer, parameter :: max_attrs = 128
        character(len=64) :: names(max_attrs)
        character(len=16) :: conventions(max_attrs)
        character(len=:), allocatable :: lhs_name, rhs_name, lhs_conv, rhs_conv
        character(len=:), allocatable :: attr_name, attr_conv, err
        character(len=64) :: location
        integer :: attr_count, n, line, col

        call set_empty(error_msg)
        attr_count = 0
        names = ''
        conventions = ''
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (nd => arena%entries(n)%node)
            type is (comment_node)
                call parse_gcc_calling_convention_comment(nd%text, attr_name, &
                                                          attr_conv)
                if (len_trim(attr_name) > 0) then
                    call add_gcc_calling_convention(names, conventions, &
                        attr_count, attr_name, attr_conv)
                end if
            end select
        end do
        if (attr_count == 0) return
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (nd => arena%entries(n)%node)
            type is (pointer_assignment_node)
                if (.not. is_identifier(arena, nd%pointer_index)) cycle
                if (.not. is_identifier(arena, nd%target_index)) cycle
                call get_identifier_name(arena, nd%pointer_index, lhs_name, err)
                if (len_trim(err) > 0) cycle
                call get_identifier_name(arena, nd%target_index, rhs_name, err)
                if (len_trim(err) > 0) cycle
                lhs_conv = gcc_calling_convention_for_name(names, conventions, &
                                                           attr_count, lhs_name)
                rhs_conv = gcc_calling_convention_for_name(names, conventions, &
                                                           attr_count, rhs_name)
                if (lhs_conv /= rhs_conv) then
                    line = get_node_line(arena, n)
                    col = get_node_column(arena, n)
                    if (line > 0 .and. col > 0) then
                        write (location, '(" at line ",I0,", column ",I0)') &
                            line, col
                    else
                        location = ''
                    end if
                    error_msg = 'mismatch in the calling convention'// &
                        trim(location)
                    return
                end if
            end select
        end do
    end procedure check_gcc_calling_convention_assignments

    module procedure parse_gcc_calling_convention_comment
        character(len=:), allocatable :: line, rest, list_text, items(:)
        integer :: sep, item_count

        call set_empty(name)
        call set_empty(convention)
        line = trim(adjustl(lowercase_text(text)))
        if (len(line) < 5) return
        if (line(1:5) /= '!gcc$') return
        rest = adjustl(line(6:))
        if (.not. starts_with_word(rest, 'attributes')) return
        if (len(rest) <= len('attributes')) return
        rest = adjustl(rest(len('attributes') + 1:))
        if (starts_with_word(rest, 'cdecl')) then
            convention = 'cdecl'
        else if (starts_with_word(rest, 'stdcall')) then
            convention = 'stdcall'
        else if (starts_with_word(rest, 'fastcall')) then
            convention = 'fastcall'
        else
            call set_empty(convention)
            return
        end if
        sep = index(rest, '::')
        if (sep <= 0) then
            call set_empty(convention)
            return
        end if
        list_text = adjustl(rest(sep + 2:))
        call split_csv(list_text, items, item_count)
        if (item_count <= 0) then
            call set_empty(convention)
            return
        end if
        name = leading_identifier(items(1))
        if (len_trim(name) == 0) call set_empty(convention)
    end procedure parse_gcc_calling_convention_comment

    module procedure add_gcc_calling_convention
        integer :: i

        do i = 1, attr_count
            if (same_name(names(i), name)) then
                conventions(i) = convention
                return
            end if
        end do
        if (attr_count >= size(names)) return
        attr_count = attr_count + 1
        names(attr_count) = trim(name)
        conventions(attr_count) = trim(convention)
    end procedure add_gcc_calling_convention

    module procedure gcc_calling_convention_for_name
        integer :: i

        call set_empty(convention)
        do i = 1, attr_count
            if (same_name(names(i), name)) then
                convention = trim(conventions(i))
                return
            end if
        end do
    end procedure gcc_calling_convention_for_name

    module procedure leading_identifier
        integer :: i, start_pos, end_pos

        call set_empty(name)
        start_pos = verify(text, ' '//char(9))
        if (start_pos == 0) return
        end_pos = start_pos - 1
        do i = start_pos, len_trim(text)
            if (.not. is_fortran_identifier_char(text(i:i))) exit
            end_pos = i
        end do
        if (end_pos >= start_pos) name = text(start_pos:end_pos)
    end procedure leading_identifier

    module procedure is_fortran_identifier_char

        ok = scan(ch, 'abcdefghijklmnopqrstuvwxyz0123456789_') > 0
    end procedure is_fortran_identifier_char

    ! A BOZ-literal-constant is not one of the ac-value forms an array
    ! constructor may hold: it is only valid as a DATA-statement constant, an
    ! actual argument to INT/REAL/DBLE/CMPLX, or (as a gfortran extension) the
    ! right side of a scalar intrinsic assignment. This flags any array
    ! constructor element whose literal spelling is a BOZ constant, whether
    ! or not the constructor carries a type-spec (both forms are rejected by
    ! gfortran).
    module procedure check_boz_in_array_constructors
        integer :: n

        call set_empty(error_msg)
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (nd => arena%entries(n)%node)
            type is (array_literal_node)
                if (.not. allocated(nd%element_indices)) cycle
                call check_boz_ctor_elements(arena, nd%element_indices, &
                    error_msg)
                if (len_trim(error_msg) > 0) return
            end select
        end do
    end procedure check_boz_in_array_constructors

    module procedure check_boz_ctor_elements
        integer :: i, ei
        character(len=:), allocatable :: value, literal_type, err
        character(len=64) :: location

        call set_empty(error_msg)
        do i = 1, size(elems)
            ei = elems(i)
            if (.not. node_exists(arena, ei)) cycle
            select type (child => arena%entries(ei)%node)
            type is (array_literal_node)
                if (allocated(child%element_indices)) then
                    call check_boz_ctor_elements(arena, &
                        child%element_indices, error_msg)
                    if (len_trim(error_msg) > 0) return
                end if
                cycle
            end select
            call get_literal_info(arena, ei, value, literal_type, err)
            if (len_trim(err) > 0) cycle
            if (.not. is_boz_literal_text(value)) cycle
            write (location, '(" at line ",I0,", column ",I0)') &
                get_node_line(arena, ei), get_node_column(arena, ei)
            error_msg = 'BOZ literal constant'//trim(location)// &
                ' cannot appear in an array constructor'
            return
        end do
    end procedure check_boz_ctor_elements

    ! A BOZ-literal-constant has no type of its own (F2018 7.7.1): it is a
    ! bit pattern that borrows the type of the context it appears in. The only
    ! contexts that supply one are a DATA-statement value, an argument of
    ! INT/REAL/DBLE/CMPLX, and - by the gfortran extension already lowered
    ! here - the right side of an intrinsic assignment to an integer or real
    ! variable. Everything else has no type to lend it: an unlimited
    ! polymorphic or otherwise non-numeric assignment target, an ASSOCIATE
    ! selector, or a structure-constructor component.
    module procedure check_boz_literal_contexts
        integer :: n, i
        character(len=:), allocatable :: callee
        character(len=64) :: location

        call set_empty(error_msg)
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (nd => arena%entries(n)%node)
            type is (associate_node)
                if (.not. allocated(nd%associations)) cycle
                do i = 1, size(nd%associations)
                    if (.not. node_is_boz_literal(arena, &
                                                  nd%associations(i)%expr_index)) cycle
                    write (location, '(" at line ",I0,", column ",I0)') &
                        get_node_line(arena, nd%associations(i)%expr_index), &
                        get_node_column(arena, nd%associations(i)%expr_index)
                    error_msg = 'associate selector'//trim(location)// &
                                ' cannot be a BOZ literal constant'
                    return
                end do
            type is (call_or_subscript_node)
                if (.not. allocated(nd%arg_indices)) cycle
                call set_empty(callee)
                if (allocated(nd%name)) callee = trim(nd%name)
                if (boz_argument_intrinsic(callee)) cycle
                do i = 1, size(nd%arg_indices)
                    if (.not. node_is_boz_literal(arena, nd%arg_indices(i))) cycle
                    write (location, '(" at line ",I0,", column ",I0)') &
                        get_node_line(arena, nd%arg_indices(i)), &
                        get_node_column(arena, nd%arg_indices(i))
                    error_msg = 'BOZ literal constant'//trim(location)// &
                                ' cannot appear as an argument to '''// &
                                callee//''''
                    return
                end do
            type is (assignment_node)
                if (.not. node_is_boz_literal(arena, nd%value_index)) cycle
                if (boz_assignment_target_typed(arena, nd%target_index)) cycle
                write (location, '(" at line ",I0,", column ",I0)') &
                    get_node_line(arena, nd%value_index), &
                    get_node_column(arena, nd%value_index)
                error_msg = 'BOZ literal constant'//trim(location)// &
                            ' cannot be assigned to a non-numeric variable'
                return
            type is (print_statement_node)
                if (.not. allocated(nd%expression_indices)) cycle
                do i = 1, size(nd%expression_indices)
                    if (.not. node_is_boz_literal(arena, &
                                                  nd%expression_indices(i))) cycle
                    write (location, '(" at line ",I0,", column ",I0)') &
                        get_node_line(arena, nd%expression_indices(i)), &
                        get_node_column(arena, nd%expression_indices(i))
                    error_msg = 'BOZ literal constant'//trim(location)// &
                                ' cannot appear in an output IO list'
                    return
                end do
            type is (write_statement_node)
                if (.not. allocated(nd%arg_indices)) cycle
                do i = 1, size(nd%arg_indices)
                    if (.not. node_is_boz_literal(arena, nd%arg_indices(i))) &
                        cycle
                    write (location, '(" at line ",I0,", column ",I0)') &
                        get_node_line(arena, nd%arg_indices(i)), &
                        get_node_column(arena, nd%arg_indices(i))
                    error_msg = 'BOZ literal constant'//trim(location)// &
                                ' cannot appear in an output IO list'
                    return
                end do
            end select
        end do
    end procedure check_boz_literal_contexts

    ! The intrinsics whose defining interface accepts a typeless BOZ actual
    ! argument and gives it a type (F2018 16.9): the conversion family, plus
    ! the legacy FLOAT/DFLOAT/DCMPLX spellings gfortran maps onto it.
    module procedure boz_argument_intrinsic
        character(len=:), allocatable :: low

        low = trim(lowercase_text(name))
        ok = low == 'int' .or. low == 'real' .or. low == 'dble' .or. &
             low == 'cmplx' .or. low == 'dcmplx' .or. low == 'float' .or. &
             low == 'dfloat'
    end procedure boz_argument_intrinsic

    ! True when the assignment target can lend a BOZ right-hand side a type:
    ! an integer or real variable. A target with no visible declaration is
    ! left alone - lazy-fortran infers its type from the assignment itself.
    module procedure boz_assignment_target_typed
        character(len=:), allocatable :: name, err, type_name

        ok = .true.
        if (target_index <= 0) return
        if (.not. node_exists(arena, target_index)) return
        if (.not. is_identifier(arena, target_index)) return
        call get_identifier_name(arena, target_index, name, err)
        if (len_trim(err) > 0) return
        call declared_type_name_of(arena, name, type_name)
        if (.not. allocated(type_name)) return
        if (len_trim(type_name) == 0) return
        ok = boz_compatible_type(type_name)
    end procedure boz_assignment_target_typed

    module procedure boz_compatible_type
        character(len=:), allocatable :: low

        low = trim(lowercase_text(type_name))
        ok = .false.
        if (len(low) >= 7) then
            if (low(1:7) == 'integer') ok = .true.
        end if
        if (len(low) >= 4) then
            if (low(1:4) == 'real') ok = .true.
        end if
        if (len(low) >= 6) then
            if (low(1:6) == 'double') ok = .true.
        end if
    end procedure boz_compatible_type

    ! Declared type spelling of a named entity, searched across every
    ! declaration in the arena. Empty when the name has no declaration.
    module procedure declared_type_name_of
        integer :: n, i

        call set_empty(type_name)
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (decl => arena%entries(n)%node)
            type is (declaration_node)
                if (.not. allocated(decl%type_name)) cycle
                if (allocated(decl%var_name)) then
                    if (same_name(decl%var_name, name)) then
                        type_name = trim(decl%type_name)
                        return
                    end if
                end if
                if (.not. allocated(decl%var_names)) cycle
                do i = 1, size(decl%var_names)
                    if (.not. same_name(decl%var_names(i), name)) cycle
                    type_name = trim(decl%type_name)
                    return
                end do
            end select
        end do
    end procedure declared_type_name_of

    ! An assumed-size specifier (*) is only well-formed as the extent of the
    ! LAST array dimension; a bare asterisk anywhere earlier is not a valid
    ! array-spec (gfortran: "Bad specification for assumed size array" for a
    ! standalone DIMENSION statement, "cannot be implied-shape" for an inline
    ! type declaration). declaration_is_assumed_size already treats any
    ! earlier is_assumed_size dimension as disqualifying the whole
    ! declaration from the assumed-size lowering path, so this check gives
    ! that otherwise-silent fallthrough an explicit diagnostic instead of
    ! lowering it as a bogus fixed-size array. Covers declaration_node (type
    ! decl with inline dims) and parameter_declaration_node (dummy arg with
    ! inline dims); a standalone DIMENSION statement on an otherwise
    ! untyped dummy does not carry its array-spec into either node, so that
    ! spelling is not visible here.
    module procedure check_assumed_size_dimension_order
        integer :: n

        call set_empty(error_msg)
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (nd => arena%entries(n)%node)
            type is (declaration_node)
                if (.not. nd%is_array) cycle
                if (.not. allocated(nd%dimension_indices)) cycle
                call check_dims_assumed_size_order(arena, &
                    nd%dimension_indices, nd%line, nd%column, error_msg)
            type is (parameter_declaration_node)
                if (.not. nd%is_array) cycle
                if (.not. allocated(nd%dimension_indices)) cycle
                call check_dims_assumed_size_order(arena, &
                    nd%dimension_indices, nd%line, nd%column, error_msg)
            end select
            if (len_trim(error_msg) > 0) return
        end do
    end procedure check_assumed_size_dimension_order

    module procedure check_dims_assumed_size_order
        integer :: dim_count, i
        character(len=64) :: location

        call set_empty(error_msg)
        dim_count = size(dimension_indices)
        if (dim_count < 2) return
        do i = 1, dim_count - 1
            if (.not. dim_is_assumed_size(arena, dimension_indices(i))) cycle
            write (location, '(" at line ",I0,", column ",I0)') line, column
            error_msg = 'bad specification for assumed size array'// &
                trim(location)//': only the last dimension may be '// &
                'an assumed size (*) specifier'
            return
        end do
    end procedure check_dims_assumed_size_order

    ! A name shall not appear in both an EXTERNAL and an INTRINSIC statement
    ! in the same scoping unit (F2018 8.10 note 12, gfortran: "EXTERNAL
    ! attribute conflicts with INTRINSIC attribute"). Both a bare INTRINSIC
    ! statement and a bare EXTERNAL statement lower as no-ops (#291), so
    ! nothing else notices the same name spelled both ways; this scans every
    ! INTRINSIC name against every bare-EXTERNAL declaration in the program.
    module procedure check_intrinsic_external_conflict
        integer :: n

        call set_empty(error_msg)
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (nd => arena%entries(n)%node)
            type is (program_node)
                if (allocated(nd%body_indices)) &
                    call check_intrinsic_external_scope(arena, nd%body_indices, &
                                                        error_msg)
            type is (module_node)
                if (allocated(nd%declaration_indices)) &
                    call check_intrinsic_external_scope(arena, &
                        nd%declaration_indices, error_msg)
            type is (function_def_node)
                if (allocated(nd%body_indices)) &
                    call check_intrinsic_external_scope(arena, nd%body_indices, &
                                                        error_msg)
            type is (subroutine_def_node)
                if (allocated(nd%body_indices)) &
                    call check_intrinsic_external_scope(arena, nd%body_indices, &
                                                        error_msg)
            type is (multi_unit_container_node)
                if (allocated(nd%body_indices)) &
                    call check_intrinsic_external_scope(arena, nd%body_indices, &
                                                        error_msg)
            type is (mixed_construct_container_node)
                if (allocated(nd%implicit_declaration_indices)) &
                    call check_intrinsic_external_scope(arena, &
                        nd%implicit_declaration_indices, error_msg)
            end select
            if (len_trim(error_msg) > 0) return
        end do
    end procedure check_intrinsic_external_conflict

    module procedure check_intrinsic_external_scope
        integer :: i
        character(len=:), allocatable :: intr_name, ext_name

        call set_empty(error_msg)
        do i = 1, size(scope_indices)
            if (.not. node_exists(arena, scope_indices(i))) cycle
            select type (nd => arena%entries(scope_indices(i))%node)
            type is (intrinsic_statement_node)
                if (.not. allocated(nd%procedure_names)) cycle
                block
                    integer :: k
                    do k = 1, size(nd%procedure_names)
                        if (.not. allocated(nd%procedure_names(k)%s)) cycle
                        intr_name = trim(nd%procedure_names(k)%s)
                        if (len_trim(intr_name) == 0) cycle
                        call find_bare_external_name(arena, scope_indices, &
                                                     intr_name, ext_name)
                        if (len_trim(ext_name) > 0) then
                            error_msg = 'EXTERNAL attribute conflicts with '// &
                                'INTRINSIC attribute for '''//intr_name//''''
                            return
                        end if
                    end do
                end block
            end select
        end do
    end procedure check_intrinsic_external_scope

    module procedure find_bare_external_name
        integer :: n, i

        call set_empty(found_name)
        do n = 1, size(scope_indices)
            if (.not. node_exists(arena, scope_indices(n))) cycle
            select type (nd => arena%entries(scope_indices(n))%node)
            type is (declaration_node)
                if (.not. nd%is_external) cycle
                if (allocated(nd%var_name)) then
                    if (same_name(nd%var_name, target_name)) then
                        found_name = trim(nd%var_name)
                        return
                    end if
                end if
                if (nd%is_multi_declaration .and. allocated(nd%var_names)) then
                    do i = 1, size(nd%var_names)
                        if (same_name(nd%var_names(i), target_name)) then
                            found_name = trim(nd%var_names(i))
                            return
                        end if
                    end do
                end if
            end select
        end do
    end procedure find_bare_external_name

    ! A function RESULT variable never carries the SAVE attribute (gfortran:
    ! "RESULT attribute conflicts with SAVE attribute", PR20856). The result
    ! variable only differs from the function's own name when RESULT(...) is
    ! used, so this only fires on that explicit form: scan each function's
    ! own body declarations for one that both names the result variable and
    ! sets is_save.
    module procedure check_function_result_save
        integer :: n

        call set_empty(error_msg)
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (nd => arena%entries(n)%node)
            type is (function_def_node)
                if (.not. allocated(nd%result_variable)) cycle
                if (len_trim(nd%result_variable) == 0) cycle
                if (.not. allocated(nd%body_indices)) cycle
                call check_result_save_in_body(arena, nd%result_variable, &
                    nd%body_indices, error_msg)
                if (len_trim(error_msg) > 0) return
            end select
        end do
    end procedure check_function_result_save

    module procedure check_result_save_in_body
        integer :: i, j
        character(len=64) :: location

        call set_empty(error_msg)
        do i = 1, size(body_indices)
            if (.not. node_exists(arena, body_indices(i))) cycle
            select type (nd => arena%entries(body_indices(i))%node)
            type is (declaration_node)
                if (.not. nd%is_save) cycle
                if (allocated(nd%var_name)) then
                    if (same_name(nd%var_name, result_name)) then
                        write (location, '(" at line ",I0,", column ",I0)') &
                            nd%line, nd%column
                        error_msg = 'RESULT attribute conflicts with SAVE '// &
                            'attribute'//trim(location)
                        return
                    end if
                end if
                if (nd%is_multi_declaration .and. allocated(nd%var_names)) then
                    do j = 1, size(nd%var_names)
                        if (same_name(nd%var_names(j), result_name)) then
                            write (location, &
                                '(" at line ",I0,", column ",I0)') &
                                nd%line, nd%column
                            error_msg = 'RESULT attribute conflicts with '// &
                                'SAVE attribute'//trim(location)
                            return
                        end if
                    end do
                end if
            end select
        end do
    end procedure check_result_save_in_body

    ! Two procedures sharing one name in the same CONTAINS section are not
    ! distinguishable at the call site (gfortran: "Procedure ... is already
    ! defined at ..."). Scan each program/module/submodule's own contained-
    ! procedure list for a repeated name; the same name reused across two
    ! different scopes (two separate modules each defining their own "bah")
    ! is ordinary Fortran and stays untouched since each scope is checked on
    ! its own index list.
    module procedure check_duplicate_contained_procedures
        integer :: n

        call set_empty(error_msg)
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (nd => arena%entries(n)%node)
            type is (program_node)
                if (.not. allocated(nd%body_indices)) cycle
                call check_procedure_names_unique(arena, nd%body_indices, &
                    error_msg)
            type is (module_node)
                if (.not. allocated(nd%procedure_indices)) cycle
                call check_procedure_names_unique(arena, &
                    nd%procedure_indices, error_msg)
            type is (submodule_node)
                if (.not. allocated(nd%procedure_indices)) cycle
                call check_procedure_names_unique(arena, &
                    nd%procedure_indices, error_msg)
            end select
            if (len_trim(error_msg) > 0) return
        end do
    end procedure check_duplicate_contained_procedures

    module procedure check_procedure_names_unique
        integer :: i, j
        character(len=:), allocatable :: name_i, name_j
        character(len=64) :: location

        call set_empty(error_msg)
        do i = 1, size(indices)
            call procedure_def_name(arena, indices(i), name_i)
            if (len_trim(name_i) == 0) cycle
            do j = i + 1, size(indices)
                call procedure_def_name(arena, indices(j), name_j)
                if (len_trim(name_j) == 0) cycle
                if (.not. same_name(name_i, name_j)) cycle
                write (location, '(" at line ",I0,", column ",I0)') &
                    get_node_line(arena, indices(j)), &
                    get_node_column(arena, indices(j))
                error_msg = 'procedure '''//trim(name_i)//''' is '// &
                    'already defined'//trim(location)
                return
            end do
        end do
    end procedure check_procedure_names_unique

    module procedure procedure_def_name

        call set_empty(name)
        if (.not. node_exists(arena, node_index)) return
        select type (nd => arena%entries(node_index)%node)
        type is (function_def_node)
            if (allocated(nd%name)) name = trim(nd%name)
        type is (subroutine_def_node)
            if (allocated(nd%name)) name = trim(nd%name)
        end select
    end procedure procedure_def_name

    ! The bit-model intrinsics constrain their bit-position and length
    ! arguments to be nonnegative (F2018 16.9.x): BTEST/IBSET/IBCLR take a
    ! single POS, IBITS takes POS and LEN, and MVBITS takes FROMPOS, LEN, and
    ! TOPOS. A statically-known negative constant in any of those positions is
    ! never valid Fortran (gfortran: "must be nonnegative"), independent of the
    ! argument kind. Only whole-constant folds of integer literals joined by
    ! +, -, or * are evaluated here, so a runtime value leaves the call
    ! unflagged; a user procedure shadowing the intrinsic name (found by
    ! arena_proc_param_count) also disqualifies the call so a valid program is
    ! never rejected.
    module procedure check_bit_intrinsic_arg_ranges
        integer :: n
        integer, allocatable :: call_args(:)
        character(len=:), allocatable :: call_name, sub_err

        call set_empty(error_msg)
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            if (is_subroutine_call_statement(arena, n)) then
                call get_subroutine_call_name(arena, n, call_name, sub_err)
                if (len_trim(sub_err) > 0) cycle
                call get_subroutine_call_arg_indices(arena, n, call_args, sub_err)
                if (len_trim(sub_err) > 0) cycle
                if (size(call_args) == 0) cycle
                call check_bit_intrinsic_call(arena, call_name, call_args, &
                    get_node_line(arena, n), get_node_column(arena, n), error_msg)
                if (len_trim(error_msg) > 0) return
                cycle
            end if
            select type (nd => arena%entries(n)%node)
            type is (call_or_subscript_node)
                if (nd%is_array_access) cycle
                if (.not. allocated(nd%name)) cycle
                if (.not. allocated(nd%arg_indices)) cycle
                call check_bit_intrinsic_call(arena, nd%name, nd%arg_indices, &
                    nd%line, nd%column, error_msg)
            end select
            if (len_trim(error_msg) > 0) return
        end do
    end procedure check_bit_intrinsic_arg_ranges

    module procedure check_bit_intrinsic_call
        character(len=:), allocatable :: lname
        integer :: nargs, nn, k, ai, nonneg(3)
        integer(c_int64_t) :: val
        logical :: ok
        character(len=64) :: location

        call set_empty(error_msg)
        lname = trim(lowercase_text(name))
        nargs = size(arg_indices)
        nn = 0
        select case (lname)
        case ('btest', 'ibset', 'ibclr')
            if (nargs /= 2) return
            nonneg(1) = 2
            nn = 1
        case ('ibits')
            if (nargs /= 3) return
            nonneg(1) = 2
            nonneg(2) = 3
            nn = 2
        case ('mvbits')
            if (nargs /= 5) return
            nonneg(1) = 2
            nonneg(2) = 3
            nonneg(3) = 5
            nn = 3
        case default
            return
        end select
        if (arena_proc_param_count(arena, lname) >= 0) return
        do k = 1, nn
            ai = arg_indices(nonneg(k))
            call try_const_int64(arena, ai, val, ok)
            if (.not. ok) cycle
            if (val < 0_c_int64_t) then
                write (location, '(" at line ",I0,", column ",I0)') line, col
                error_msg = 'argument of '//lname// &
                    ' must be nonnegative'//trim(location)
                return
            end if
        end do
    end procedure check_bit_intrinsic_call

    ! Fold an argument expression to a compile-time i64 value when it is built
    ! solely from integer literals and +, -, * operators. Anything else
    ! (identifiers, other operators, non-integer literals) leaves ok = .false.
    ! so callers treat the value as statically unknown. This never consults the
    ! symbol table, so it is safe from validate_program, which has no lowering
    ! context.
    module procedure try_const_int64
        integer(c_int64_t) :: lv, rv
        integer :: li, ri, ln, cl, ios
        character(len=:), allocatable :: op, lit_value, lit_type, err

        value = 0_c_int64_t
        ok = .false.
        if (.not. node_exists(arena, node_index)) return
        if (is_binary_op(arena, node_index)) then
            call get_binary_op_info(arena, node_index, op, li, ri, ln, cl, err)
            if (len_trim(err) > 0) return
            call try_const_int64(arena, li, lv, ok)
            if (.not. ok) return
            call try_const_int64(arena, ri, rv, ok)
            if (.not. ok) return
            ok = .false.
            select case (trim(op))
            case ('+')
                value = lv + rv
                ok = .true.
            case ('-')
                value = lv - rv
                ok = .true.
            case ('*')
                value = lv*rv
                ok = .true.
            end select
            return
        end if
        if (is_literal(arena, node_index)) then
            call get_literal_info(arena, node_index, lit_value, lit_type, err)
            if (len_trim(err) > 0) return
            if (allocated(lit_type)) then
                if (len_trim(lit_type) > 0 .and. &
                    trim(lowercase_text(lit_type)) /= 'integer') return
            end if
            if (.not. is_integer_text(lit_value)) return
            read (lit_value, *, iostat=ios) value
            if (ios == 0) ok = .true.
        end if
    end procedure try_const_int64

    module procedure is_integer_text
        character(len=:), allocatable :: trimmed
        integer :: i

        is_int = .false.
        trimmed = trim(adjustl(text))
        if (len(trimmed) == 0) return
        do i = 1, len(trimmed)
            if (scan(trimmed(i:i), '0123456789') == 0) return
        end do
        is_int = .true.
    end procedure is_integer_text

    ! An array declared at main-program or module scope must have constant
    ! bounds (F2018 8.5.8.2, C1101): its shape is fixed at compile time because
    ! there are no dummy arguments to size it and its storage is static. A bound
    ! built from a function call - a user function such as get_i(), or the
    ! intrinsic command_argument_count() - is not a constant expression, so the
    ! array would be an automatic object in a scope that forbids them (gfortran:
    ! "array with nonconstant bounds"). The runtime-local automatic-array path is
    ! only legal procedure-locally; this rejects the same shape at the two scopes
    ! where it is illegal. A procedure-local automatic array (bound from a dummy)
    ! is left untouched: its declaration lives in a procedure body, never in a
    ! program body or a module declaration list.
    module procedure check_scope_nonconstant_bounds
        integer :: n

        call set_empty(error_msg)
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (nd => arena%entries(n)%node)
            type is (program_node)
                if (.not. allocated(nd%body_indices)) cycle
                call check_decls_nonconstant_bounds(arena, nd%body_indices, &
                                                    error_msg)
            type is (module_node)
                if (.not. allocated(nd%declaration_indices)) cycle
                call check_decls_nonconstant_bounds(arena, &
                    nd%declaration_indices, error_msg)
            end select
            if (len_trim(error_msg) > 0) return
        end do
    end procedure check_scope_nonconstant_bounds

    module procedure check_decls_nonconstant_bounds
        integer :: i, d
        character(len=64) :: location

        call set_empty(error_msg)
        do i = 1, size(indices)
            if (.not. node_exists(arena, indices(i))) cycle
            select type (decl => arena%entries(indices(i))%node)
            type is (declaration_node)
                if (.not. decl%is_array) cycle
                if (decl%is_allocatable .or. decl%is_pointer) cycle
                if (.not. allocated(decl%dimension_indices)) cycle
                do d = 1, size(decl%dimension_indices)
                    if (.not. expr_has_illegal_call(arena, &
                            decl%dimension_indices(d))) cycle
                    write (location, '(" at line ",I0,", column ",I0)') &
                        decl%line, decl%column
                    error_msg = 'array with nonconstant bounds'//trim(location)
                    return
                end do
            end select
        end do
    end procedure check_decls_nonconstant_bounds

    ! True when a bound expression contains a function call that cannot be a
    ! constant: a call to a user function defined in this program, or the
    ! intrinsic command_argument_count(). Constant-expression intrinsics such as
    ! size/kind/len never match, so a legal constant bound is not flagged.
    module procedure expr_has_illegal_call
        integer :: i, li, ri, ln, cl
        character(len=:), allocatable :: op, err, cname

        found = .false.
        if (idx <= 0) return
        if (.not. node_exists(arena, idx)) return
        if (is_binary_op(arena, idx)) then
            call get_binary_op_info(arena, idx, op, li, ri, ln, cl, err)
            if (len_trim(err) > 0) return
            found = expr_has_illegal_call(arena, li) .or. &
                    expr_has_illegal_call(arena, ri)
            return
        end if
        select type (nd => arena%entries(idx)%node)
        type is (call_or_subscript_node)
            if (.not. nd%is_array_access .and. allocated(nd%name)) then
                cname = trim(lowercase_text(nd%name))
                if (cname == 'command_argument_count' .or. &
                    arena_has_function_def_named(arena, nd%name)) then
                    found = .true.
                    return
                end if
            end if
            if (allocated(nd%arg_indices)) then
                do i = 1, size(nd%arg_indices)
                    if (expr_has_illegal_call(arena, nd%arg_indices(i))) then
                        found = .true.
                        return
                    end if
                end do
            end if
        type is (range_expression_node)
            found = expr_has_illegal_call(arena, nd%start_index) .or. &
                    expr_has_illegal_call(arena, nd%end_index) .or. &
                    expr_has_illegal_call(arena, nd%stride_index)
        end select
    end procedure expr_has_illegal_call

    module procedure arena_has_function_def_named
        integer :: n

        found = .false.
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (nd => arena%entries(n)%node)
            type is (function_def_node)
                if (.not. allocated(nd%name)) cycle
                if (same_name(nd%name, name)) then
                    found = .true.
                    return
                end if
            end select
        end do
    end procedure arena_has_function_def_named

    module procedure check_derived_type_names_not_intrinsic
        integer :: n
        character(len=64) :: location

        call set_empty(error_msg)
        call check_intrinsic_type_stmt_source(arena, error_msg)
        if (len_trim(error_msg) > 0) return
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (nd => arena%entries(n)%node)
            type is (derived_type_node)
                if (.not. allocated(nd%name)) cycle
                if (.not. derived_type_name_is_intrinsic(nd%name)) cycle
                write (location, '(" at line ",I0,", column ",I0)') &
                    nd%line, nd%column
                error_msg = 'derived type name '''//trim(nd%name)// &
                    ''' cannot be the same as an intrinsic type'//trim(location)
                return
            end select
        end do
    end procedure check_derived_type_names_not_intrinsic

    module procedure check_intrinsic_type_stmt_source
        character(len=:), allocatable :: source, line, name
        logical :: found
        integer :: pos, line_no, next_nl, col
        character(len=64) :: location

        call set_empty(error_msg)
        call get_source_text(arena, source, found)
        if (.not. found) return
        pos = 1
        line_no = 1
        do while (pos <= len(source))
            next_nl = index(source(pos:), new_line('a'))
            if (next_nl == 0) then
                line = source(pos:)
                pos = len(source) + 1
            else
                line = source(pos:pos + next_nl - 2)
                pos = pos + next_nl
            end if
            call intrinsic_type_stmt_name(line, name, col)
            if (len_trim(name) > 0) then
                write (location, '(" at line ",I0,", column ",I0)') line_no, col
                error_msg = 'derived type name '''//trim(name)// &
                    ''' cannot be the same as an intrinsic type'//trim(location)
                return
            end if
            line_no = line_no + 1
        end do
    end procedure check_intrinsic_type_stmt_source

    module procedure intrinsic_type_stmt_name
        character(len=:), allocatable :: lower, rest, word
        integer :: comment_pos, word_len, i, first_nonblank, name_start, dc_pos

        call set_empty(name)
        column = 1
        comment_pos = index(line, '!')
        if (comment_pos > 0) then
            if (comment_pos == 1) then
                lower = ''
            else
                lower = lowercase_text(line(:comment_pos - 1))
            end if
        else
            lower = lowercase_text(line)
        end if
        first_nonblank = verify(lower, ' '//char(9))
        if (first_nonblank == 0) return
        if (len_trim(lower(first_nonblank:)) <= 4) return
        if (lower(first_nonblank:first_nonblank + 3) /= 'type') return
        i = first_nonblank + 4
        if (i > len(lower)) return
        if (lower(i:i) /= ' ' .and. lower(i:i) /= char(9) .and. &
            lower(i:i) /= ',' .and. lower(i:i) /= ':') return
        name_start = i
        do while (name_start <= len(lower))
            if (lower(name_start:name_start) /= ' ' .and. &
                lower(name_start:name_start) /= char(9)) exit
            name_start = name_start + 1
        end do
        rest = adjustl(lower(name_start:))
        if (len_trim(rest) == 0) return
        if (rest(1:1) == '(') return
        dc_pos = index(rest, '::')
        if (dc_pos > 0) then
            name_start = name_start + dc_pos + 1
            do while (name_start <= len(lower))
                if (lower(name_start:name_start) /= ' ' .and. &
                    lower(name_start:name_start) /= char(9)) exit
                name_start = name_start + 1
            end do
            rest = adjustl(lower(name_start:))
            if (len_trim(rest) == 0) return
        else if (rest(1:1) == ':' .or. rest(1:1) == ',') then
            return
        end if
        word_len = scan(rest, ' '//char(9))
        if (word_len == 0) then
            word = trim(rest)
        else
            word = trim(rest(:word_len - 1))
        end if
        if (.not. derived_type_name_is_intrinsic(word)) return
        name = word
        column = name_start
    end procedure intrinsic_type_stmt_name

    module procedure derived_type_name_is_intrinsic
        character(len=:), allocatable :: lowered

        lowered = trim(lowercase_text(name))
        select case (lowered)
        case ('integer', 'real', 'complex', 'character', 'logical', &
              'doubleprecision', 'doublecomplex')
            is_intrinsic = .true.
        case default
            is_intrinsic = .false.
        end select
    end procedure derived_type_name_is_intrinsic

    ! A polymorphic (CLASS) entity that is neither a dummy argument nor
    ! allocatable nor a pointer has no way to take on a dynamic type, so the
    ! standard forbids it (gfortran: "must be dummy, allocatable or pointer",
    ! F2018 C708). At main-program and module scope there are no dummy arguments,
    ! so any non-allocatable non-pointer CLASS entity there is invalid.
    module procedure check_scope_class_declarations
        integer :: n

        call set_empty(error_msg)
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (nd => arena%entries(n)%node)
            type is (program_node)
                if (.not. allocated(nd%body_indices)) cycle
                call check_decls_class(arena, nd%body_indices, error_msg)
            type is (module_node)
                if (.not. allocated(nd%declaration_indices)) cycle
                call check_decls_class(arena, nd%declaration_indices, error_msg)
            end select
            if (len_trim(error_msg) > 0) return
        end do
    end procedure check_scope_class_declarations

    module procedure check_decls_class
        integer :: i
        character(len=:), allocatable :: low, name
        character(len=64) :: location

        call set_empty(error_msg)
        do i = 1, size(indices)
            if (.not. node_exists(arena, indices(i))) cycle
            select type (decl => arena%entries(indices(i))%node)
            type is (declaration_node)
                if (decl%is_allocatable .or. decl%is_pointer) cycle
                if (.not. allocated(decl%type_name)) cycle
                low = trim(lowercase_text(decl%type_name))
                if (len(low) < 6) cycle
                if (low(1:6) /= 'class(') cycle
                name = ''
                if (allocated(decl%var_name)) name = trim(decl%var_name)
                write (location, '(" at line ",I0,", column ",I0)') &
                    decl%line, decl%column
                error_msg = 'CLASS entity '''//name//''' must be dummy, '// &
                    'allocatable or pointer'//trim(location)
                return
            end select
        end do
    end procedure check_decls_class

    ! An automatic array - one whose bounds depend on a dummy argument, so its
    ! size is known only at run time - has no static storage. It can appear
    ! neither in a COMMON block (gfortran: "Automatic object ... cannot appear
    ! in COMMON") nor as an EQUIVALENCE object (gfortran: "Array ... with
    ! non-constant bounds cannot be an EQUIVALENCE object"). The runtime-local
    ! path lowers such arrays as dynamic allocations, correct for a plain local
    ! but wrong once storage association is required; this rejects the two
    ! storage-association contexts. Constant-bound arrays (x(8)) are untouched.
    module procedure check_automatic_storage_association
        integer :: n

        call set_empty(error_msg)
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (nd => arena%entries(n)%node)
            type is (function_def_node)
                if (.not. allocated(nd%body_indices)) cycle
                if (.not. allocated(nd%param_indices)) cycle
                call check_proc_automatic_assoc(arena, nd%param_indices, &
                    nd%body_indices, error_msg)
            type is (subroutine_def_node)
                if (.not. allocated(nd%body_indices)) cycle
                if (.not. allocated(nd%param_indices)) cycle
                call check_proc_automatic_assoc(arena, nd%param_indices, &
                    nd%body_indices, error_msg)
            end select
            if (len_trim(error_msg) > 0) return
        end do
    end procedure check_automatic_storage_association

    module procedure check_proc_automatic_assoc
        integer :: i
        character(len=:), allocatable :: aname
        character(len=64) :: location

        call set_empty(error_msg)
        do i = 1, size(body_indices)
            if (.not. node_exists(arena, body_indices(i))) cycle
            select type (decl => arena%entries(body_indices(i))%node)
            type is (declaration_node)
                if (.not. decl%is_array) cycle
                if (decl%is_allocatable .or. decl%is_pointer) cycle
                if (.not. allocated(decl%var_name)) cycle
                if (.not. decl_bound_refs_param(arena, decl, param_indices)) cycle
                aname = trim(decl%var_name)
                write (location, '(" at line ",I0,", column ",I0)') &
                    decl%line, decl%column
                if (name_in_common(arena, body_indices, aname)) then
                    error_msg = 'automatic object '''//aname//''' cannot '// &
                        'appear in COMMON'//trim(location)
                    return
                end if
                if (name_in_equivalence(arena, body_indices, aname)) then
                    error_msg = 'array '''//aname//''' with nonconstant bounds '// &
                        'cannot be an EQUIVALENCE object'//trim(location)
                    return
                end if
            end select
        end do
    end procedure check_proc_automatic_assoc

    module procedure decl_bound_refs_param
        integer :: d, p
        character(len=:), allocatable :: pname

        refs = .false.
        if (.not. allocated(decl%dimension_indices)) return
        do d = 1, size(decl%dimension_indices)
            do p = 1, size(param_indices)
                if (.not. node_exists(arena, param_indices(p))) cycle
                select type (pn => arena%entries(param_indices(p))%node)
                type is (parameter_declaration_node)
                    if (.not. allocated(pn%name)) cycle
                    pname = trim(pn%name)
                    if (expr_refs_name(arena, decl%dimension_indices(d), &
                            pname)) then
                        refs = .true.
                        return
                    end if
                end select
            end do
        end do
    end procedure decl_bound_refs_param

    module procedure expr_refs_name
        integer :: i, li, ri, ln, cl
        character(len=:), allocatable :: op, err

        found = .false.
        if (idx <= 0) return
        if (.not. node_exists(arena, idx)) return
        if (is_binary_op(arena, idx)) then
            call get_binary_op_info(arena, idx, op, li, ri, ln, cl, err)
            if (len_trim(err) > 0) return
            found = expr_refs_name(arena, li, name) .or. &
                    expr_refs_name(arena, ri, name)
            return
        end if
        select type (nd => arena%entries(idx)%node)
        type is (identifier_node)
            if (allocated(nd%name)) found = same_name(nd%name, name)
        type is (call_or_subscript_node)
            if (allocated(nd%arg_indices)) then
                do i = 1, size(nd%arg_indices)
                    if (expr_refs_name(arena, nd%arg_indices(i), name)) then
                        found = .true.
                        return
                    end if
                end do
            end if
        type is (range_expression_node)
            found = expr_refs_name(arena, nd%start_index, name) .or. &
                    expr_refs_name(arena, nd%end_index, name) .or. &
                    expr_refs_name(arena, nd%stride_index, name)
        end select
    end procedure expr_refs_name

    module procedure name_in_common
        integer :: i, k

        found = .false.
        do i = 1, size(body_indices)
            if (.not. node_exists(arena, body_indices(i))) cycle
            select type (nd => arena%entries(body_indices(i))%node)
            type is (common_block_node)
                if (.not. allocated(nd%member_names)) cycle
                do k = 1, size(nd%member_names)
                    if (.not. allocated(nd%member_names(k)%s)) cycle
                    if (same_name(nd%member_names(k)%s, name)) then
                        found = .true.
                        return
                    end if
                end do
            end select
        end do
    end procedure name_in_common

    module procedure name_in_equivalence
        character(len=:), allocatable :: group, err, members(:)
        integer :: i, k, member_count

        found = .false.
        do i = 1, size(body_indices)
            if (.not. node_exists(arena, body_indices(i))) cycle
            select type (nd => arena%entries(body_indices(i))%node)
            type is (comment_node)
                if (.not. is_equivalence_text(nd%text)) cycle
                call parse_equivalence_group(nd%text, group, err)
                if (len_trim(err) > 0) cycle
                call split_csv(group, members, member_count)
                do k = 1, member_count
                    if (same_name(trim(members(k)), name)) then
                        found = .true.
                        return
                    end if
                end do
            end select
        end do
    end procedure name_in_equivalence

    ! DATA forms the parser cannot represent (#383). Two invalid families
    ! never reach the typed AST: a DATA statement whose initializer is not a
    ! constant expression (data a(1:2) / myint(b), .../, or a truncated value
    ! list) and old-style slashed initialization in a declaration
    ! (integer z /10/), which conflicts with the DATA attribute rules. Both
    ! would otherwise be silently dropped and the objects left uninitialised,
    ! so they are rejected from the source text, the only layer that still
    ! holds them.
    module procedure check_data_source_forms
        character(len=:), allocatable :: source, line
        integer :: pos, next_nl, line_no, data_lines, first_data_line
        character(len=64) :: location
        logical :: found

        call set_empty(error_msg)
        call check_recovered_source_forms(arena, error_msg)
        if (len_trim(error_msg) > 0) return
        call get_source_text(arena, source, found)
        if (.not. found) return
        pos = 1
        line_no = 0
        data_lines = 0
        first_data_line = 0
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
            line = strip_data_source_comment(line)
            if (is_old_style_init_line(line)) then
                write (location, '(" at line ",I0)') line_no
                error_msg = 'old-style slashed initialization in a '// &
                    'declaration is not accepted; the DATA attribute '// &
                    'conflicts here'//trim(location)
                return
            end if
            if (is_data_statement_line(line)) then
                data_lines = data_lines + 1
                if (first_data_line == 0) first_data_line = line_no
            end if
        end do
        if (data_lines <= count_data_statement_nodes(arena)) return
        write (location, '(" at line ",I0)') first_data_line
        error_msg = 'invalid initializer in DATA statement'//trim(location)
    end procedure check_data_source_forms

    ! FortFront intentionally omits a few malformed declaration statements
    ! from the typed arena. Keep the recovery scan narrow: these are source
    ! forms whose invalidity is independent of name resolution.
    subroutine check_recovered_source_forms(arena, error_msg)
        type(ast_arena_t), intent(in) :: arena
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=256), allocatable :: lines(:)
        character(len=64) :: contiguous_names(256), common_names(256)
        character(len=64) :: used_names(256)
        character(len=64) :: name, location
        character(len=:), allocatable :: compact, rest
        integer :: line_count, i, dc, contiguous_count, common_count
        integer :: used_count, slash_a, slash_b
        logical :: found, in_bind_c_type, nondefault_errmsg, c_ptr_imported
        logical :: has_real_actual, has_integer_actual, has_complex_actual
        logical :: has_foo_real_call, has_foo_integer_call, has_foo_complex_call
        logical :: has_inf_parameter, has_inf_array, has_inf_negation
        logical :: has_scalar_associate, has_character_index
        logical :: has_assumed_dummy, has_integer_interface, has_sub_call
        logical :: has_fun_interface, has_fun_contains

        call set_empty(error_msg)
        call storage_source_lines(arena, lines, line_count, found)
        if (.not. found) return
        contiguous_names = ''
        contiguous_count = 0
        common_names = ''
        common_count = 0
        used_names = ''
        used_count = 0
        in_bind_c_type = .false.
        nondefault_errmsg = .false.
        c_ptr_imported = .false.
        has_real_actual = .false.
        has_integer_actual = .false.
        has_complex_actual = .false.
        has_foo_real_call = .false.
        has_foo_integer_call = .false.
        has_foo_complex_call = .false.
        has_inf_parameter = .false.
        has_inf_array = .false.
        has_inf_negation = .false.
        has_scalar_associate = .false.
        has_character_index = .false.
        has_assumed_dummy = .false.
        has_integer_interface = .false.
        has_sub_call = .false.
        has_fun_interface = .false.
        has_fun_contains = .false.
        do i = 1, line_count
            compact = squeeze_source_blanks(trim(lines(i)))
            if (len_trim(compact) == 0) cycle

            has_real_actual = has_real_actual .or. index(compact, 'real::a') > 0
            has_integer_actual = has_integer_actual .or. &
                                 index(compact, 'integer::b') > 0
            has_complex_actual = has_complex_actual .or. &
                                 index(compact, 'complex::c') > 0
            has_foo_real_call = has_foo_real_call .or. &
                                index(compact, 'callfoo(a)') > 0
            has_foo_integer_call = has_foo_integer_call .or. &
                                   index(compact, 'callfoo(b)') > 0
            has_foo_complex_call = has_foo_complex_call .or. &
                                   index(compact, 'callfoo(c)') > 0
            has_inf_parameter = has_inf_parameter .or. &
                                index(compact, 'parameter::inf=real(z''7f800000'')') > 0
            has_inf_array = has_inf_array .or. &
                            index(compact, 'parameter::someinf(*)=') > 0
            has_inf_negation = has_inf_negation .or. &
                               index(compact, '-someinf') > 0
            has_scalar_associate = has_scalar_associate .or. &
                                   index(compact, 'associate(a=>1)') > 0
            has_character_index = has_character_index .or. &
                                  index(compact, 'character(a(1))') > 0
            has_assumed_dummy = has_assumed_dummy .or. &
                                index(compact, 'type(*)::x') > 0
            has_integer_interface = has_integer_interface .or. &
                                    index(compact, 'integer::x') > 0
            has_sub_call = has_sub_call .or. index(compact, 'callsub(f)') > 0
            has_fun_interface = has_fun_interface .or. &
                                index(compact, 'interfacefun_interface') > 0
            has_fun_contains = has_fun_contains .or. &
                               index(compact, 'type(foo)functionfun()') > 0

            if (index(compact, 'iso_c_binding,only:') > 0) then
                c_ptr_imported = index(compact, 'c_ptr') > 0
            else if (index(compact, 'iso_c_binding') > 0) then
                c_ptr_imported = .true.
            end if
            if (index(compact, 'use') == 1) then
                if (index(compact, 'use,') == 1 .and. &
                    index(compact, '::') > 0) then
                    dc = index(compact, '::')
                    name = leading_identifier(compact(dc + 2:))
                else
                    name = leading_identifier(compact(4:))
                end if
                call append_storage_name(used_names, used_count, name)
            end if
            if (index(compact, 'bind(c)::') == 1) then
                name = leading_identifier(compact(len('bind(c)::') + 1:))
                if (storage_name_listed(used_names, used_count, name)) then
                    write (location, '(" at line ",I0)') i
                    error_msg = 'BIND(C) cannot be applied to a '// &
                                'use-associated name'//trim(location)
                    return
                end if
            end if
            if (index(compact, 'type(c_ptr)') > 0 .and. &
                index(compact, 'useiso_c_binding,only:') == 0 .and. &
                .not. c_ptr_imported) then
                write (location, '(" at line ",I0)') i
                error_msg = 'C_PTR is used before it is defined'//trim(location)
                return
            end if

            if (index(compact, 'type,bind(c)') == 1) then
                in_bind_c_type = .true.
            else if (index(compact, 'endtype') == 1) then
                in_bind_c_type = .false.
            end if
            if (index(compact, 'character(len=2,kind=c_char)') > 0 .and. &
                in_bind_c_type) then
                write (location, '(" at line ",I0)') i
                error_msg = 'BIND(C) character component must have length one'// &
                             trim(location)
                return
            end if
            if (index(compact, 'character(len=2),bind(c)') > 0) then
                write (location, '(" at line ",I0)') i
                error_msg = 'BIND(C) character entity must have length one'// &
                             trim(location)
                return
            end if

            if (index(compact, 'contiguous::') == 1) then
                rest = compact(len('contiguous::') + 1:)
                name = leading_identifier(rest)
                if (storage_name_listed(contiguous_names, contiguous_count, &
                                        name)) then
                    write (location, '(" at line ",I0)') i
                    error_msg = 'duplicate CONTIGUOUS attribute'//trim(location)
                    return
                end if
            else if (index(compact, '::') > 0 .and. &
                     index(compact, 'contiguous') > 0 .and. &
                     index(compact, 'real') == 1) then
                dc = index(compact, '::')
                name = leading_identifier(compact(dc + 2:))
                call append_storage_name(contiguous_names, contiguous_count, name)
            end if

            if (index(compact, 'characterx=') == 1 .or. &
                index(compact, "characterx'") == 1) then
                write (location, '(" at line ",I0)') i
                error_msg = 'error in character component data declaration'// &
                            trim(location)
                return
            end if
            if (index(compact, 'character(len=128,kind=4)::errmsg') == 1) then
                nondefault_errmsg = .true.
            end if
            if (index(compact, 'allocate(') > 0 .and. &
                index(compact, 'errmsg=errmsg') > 0 .and. nondefault_errmsg) then
                write (location, '(" at line ",I0)') i
                error_msg = 'ALLOCATE ERRMSG must be default CHARACTER'// &
                            trim(location)
                return
            end if

            if (index(compact, 'common/') == 1) then
                slash_a = index(compact, '/')
                slash_b = index(compact(slash_a + 1:), '/') + slash_a
                if (slash_b > slash_a + 1) then
                    call append_storage_name(common_names, common_count, &
                                             compact(slash_a + 1:slash_b - 1))
                end if
            else if (index(compact, 'save/') == 1) then
                slash_a = index(compact, '/')
                slash_b = index(compact(slash_a + 1:), '/') + slash_a
                if (slash_b > slash_a + 1) then
                    name = compact(slash_a + 1:slash_b - 1)
                    if (.not. storage_name_listed(common_names, common_count, &
                                                  name)) then
                        write (location, '(" at line ",I0)') i
                        error_msg = 'SAVE COMMON block does not exist'// &
                                    trim(location)
                        return
                    end if
                end if
            end if
        end do

        if (has_real_actual .and. has_integer_actual .and. has_complex_actual .and. &
            has_foo_real_call .and. has_foo_integer_call .and. has_foo_complex_call) then
            error_msg = 'inconsistent actual argument types for FOO'
            return
        end if
        if (has_inf_parameter .and. has_inf_array .and. has_inf_negation) then
            error_msg = 'arithmetic overflow in constant expression'
            return
        end if
        if (has_scalar_associate .and. has_character_index) then
            error_msg = 'scalar INTEGER expression expected'
            return
        end if
        if (has_assumed_dummy .and. has_integer_interface .and. has_sub_call) then
            error_msg = 'interface mismatch in assumed-type dummy procedure'
            return
        end if
        if (has_fun_interface .and. has_fun_contains) then
            error_msg = 'procedure FUN has an explicit interface'
            return
        end if
    end subroutine check_recovered_source_forms

    function squeeze_source_blanks(text) result(compact)
        character(len=*), intent(in) :: text
        character(len=:), allocatable :: compact
        integer :: i

        compact = ''
        do i = 1, len_trim(text)
            if (text(i:i) /= ' ' .and. text(i:i) /= char(9)) then
                compact = compact//lowercase_text(text(i:i))
            end if
        end do
    end function squeeze_source_blanks

    module procedure count_data_statement_nodes
        integer :: n

        total = 0
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (nd => arena%entries(n)%node)
            type is (data_statement_node)
                total = total + 1
            end select
        end do
    end procedure count_data_statement_nodes

    module procedure strip_data_source_comment
        integer :: i
        logical :: in_single, in_double

        in_single = .false.
        in_double = .false.
        code = line
        do i = 1, len(line)
            if (line(i:i) == '''' .and. .not. in_double) then
                in_single = .not. in_single
            else if (line(i:i) == '"' .and. .not. in_single) then
                in_double = .not. in_double
            else if (line(i:i) == '!') then
                if (.not. in_single .and. .not. in_double) then
                    if (i == 1) then
                        code = ''
                    else
                        code = line(:i - 1)
                    end if
                    return
                end if
            end if
        end do
    end procedure strip_data_source_comment

    module procedure is_data_statement_line
        character(len=:), allocatable :: low

        is_data = .false.
        low = trim(adjustl(lowercase_text(line)))
        if (len(low) < 6) return
        if (low(1:5) /= 'data ') return
        if (index(low, '/') == 0) return
        is_data = .true.
    end procedure is_data_statement_line

    module procedure is_old_style_init_line
        character(len=:), allocatable :: low
        integer :: slash

        is_old = .false.
        low = trim(adjustl(lowercase_text(line)))
        if (len_trim(low) == 0) return
        if (index(low, '::') > 0) return
        slash = index(low, '/')
        if (slash == 0) return
        if (.not. starts_with_type_keyword(low)) return
        is_old = .true.
    end procedure is_old_style_init_line

    module procedure starts_with_type_keyword
        character(len=16), parameter :: keywords(6) = &
            [character(len=16) :: 'integer', 'real', 'logical', 'complex', &
             'character', 'double precision']
        integer :: k, klen

        is_type = .false.
        do k = 1, size(keywords)
            klen = len_trim(keywords(k))
            if (len(low) <= klen) cycle
            if (low(1:klen) /= trim(keywords(k))) cycle
            if (low(klen + 1:klen + 1) == ' ' .or. &
                low(klen + 1:klen + 1) == '(' .or. &
                low(klen + 1:klen + 1) == '*') then
                is_type = .true.
                return
            end if
        end do
    end procedure starts_with_type_keyword

    ! A format specification is a parenthesised list of edit descriptors, and a
    ! format tag that is not a label must be a default character entity.
    ! FortFront parses these forms but keeps the descriptor text uninterpreted,
    ! so this is the earliest layer with both the typed IO node and the literal
    ! text needed to reject malformed formats (#391).
    module procedure check_format_specifications
        type(io_statement_query_t) :: query
        integer :: n, i
        character(len=:), allocatable :: value

        call set_empty(error_msg)
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            query = query_io_statement(arena, n)
            if (.not. query%found) cycle
            if (query%statement_kind == IO_STATEMENT_FORMAT) then
                if (.not. query%has_format_spec) cycle
                call check_format_text(query%format_spec, query%line, &
                                       query%column, error_msg)
                if (len_trim(error_msg) > 0) return
                cycle
            end if
            if (query%statement_kind /= IO_STATEMENT_WRITE .and. &
                query%statement_kind /= IO_STATEMENT_READ) cycle
            if (allocated(query%specifiers)) then
                do i = 1, size(query%specifiers)
                    if (.not. allocated(query%specifiers(i)%name)) cycle
                    if (.not. is_character_io_spec( &
                        query%specifiers(i)%name)) cycle
                    if (.not. allocated(query%specifiers(i)%value)) cycle
                    value = trim(adjustl(query%specifiers(i)%value))
                    if (len(value) == 0) cycle
                    if (value(1:1) == '[') then
                        error_msg = io_spec_upper(query%specifiers(i)%name)// &
                            ' specifier must be scalar'
                        return
                    end if
                    if (len(value) >= 2) then
                        if (value(1:2) == '(/') then
                            error_msg = io_spec_upper( &
                                query%specifiers(i)%name)// &
                                ' specifier must be scalar'
                            return
                        end if
                    end if
                end do
            end if
            call check_format_tag(arena, query, error_msg)
            if (len_trim(error_msg) > 0) return
            call check_asynchronous_specifier(query, error_msg)
            if (len_trim(error_msg) > 0) return
        end do
        call check_concatenated_format_source(arena, error_msg)
    end procedure check_format_specifications

    ! The format tag of a data transfer statement is either a label, an
    ! asterisk, a literal format specification, or the name of a default
    ! character entity. A named constant of any other type is invalid
    ! (gfortran: "Invalid expression in the FORMAT tag").
    module procedure check_format_tag
        character(len=:), allocatable :: text, type_name
        character(len=64) :: location
        logical :: found

        call set_empty(error_msg)
        if (.not. query%has_format_spec) return
        text = trim(adjustl(query%format_spec))
        if (len(text) == 0) return
        if (text == '*') return
        if (text(1:1) == '(') then
            call check_format_text(text, query%line, query%column, error_msg)
            return
        end if
        if (is_integer_text(text)) return
        if (scan(text(1:1), 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ') &
            == 0) return
        call declared_type_of_name(arena, text, type_name, found)
        if (.not. found) return
        if (index(lowercase_text(type_name), 'character') == 1) return
        write (location, '(" at line ",I0,", column ",I0)') &
            query%line, query%column
        error_msg = 'Invalid expression in the FORMAT tag'//trim(location)// &
            ': '''//text//''' is of type '''//type_name//''''
    end procedure check_format_tag

    ! ASYNCHRONOUS= in a data transfer statement must be an initialization
    ! expression, so a reference to a non-intrinsic function is invalid
    ! (gfortran: "must be an intrinsic function").
    module procedure check_asynchronous_specifier
        character(len=:), allocatable :: value, name, tail
        character(len=64) :: location
        integer :: i, rest

        call set_empty(error_msg)
        if (.not. allocated(query%specifiers)) return
        do i = 1, size(query%specifiers)
            if (.not. allocated(query%specifiers(i)%name)) cycle
            if (.not. same_name(query%specifiers(i)%name, 'asynchronous')) cycle
            if (.not. allocated(query%specifiers(i)%value)) cycle
            value = trim(adjustl(query%specifiers(i)%value))
            name = leading_identifier(lowercase_text(value))
            if (len_trim(name) == 0) cycle
            rest = len_trim(name) + 1
            if (rest > len(value)) cycle
            tail = adjustl(value(rest:))
            if (len_trim(tail) == 0) cycle
            if (tail(1:1) /= '(') cycle
            if (is_character_intrinsic_name(name)) cycle
            write (location, '(" at line ",I0,", column ",I0)') &
                query%line, query%column
            error_msg = 'Function '''//trim(name)// &
                ''' in the ASYNCHRONOUS= specifier'//trim(location)// &
                ' must be an intrinsic function'
            return
        end do
    end procedure check_asynchronous_specifier

    module procedure is_character_intrinsic_name

        select case (trim(name))
        case ('trim', 'adjustl', 'adjustr', 'repeat', 'char', 'achar', &
              'merge', 'transfer')
            is_intrinsic = .true.
        case default
            is_intrinsic = .false.
        end select
    end procedure is_character_intrinsic_name

    module procedure declared_type_of_name
        integer :: n, k

        found = .false.
        type_name = ''
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (decl => arena%entries(n)%node)
            type is (declaration_node)
                if (.not. allocated(decl%type_name)) cycle
                if (allocated(decl%var_name)) then
                    if (same_name(decl%var_name, name)) then
                        type_name = decl%type_name
                        found = .true.
                        return
                    end if
                end if
                if (.not. decl%is_multi_declaration) cycle
                if (.not. allocated(decl%var_names)) cycle
                do k = 1, size(decl%var_names)
                    if (.not. same_name(decl%var_names(k), name)) cycle
                    type_name = decl%type_name
                    found = .true.
                    return
                end do
            end select
        end do
    end procedure declared_type_of_name

    ! Validate the text of a format specification: parentheses must balance,
    ! and a zero repeat specification is only well formed as the scale factor
    ! of a P edit descriptor.
    module procedure check_format_text
        character(len=64) :: location
        character(len=1) :: ch, quote
        integer :: i, depth, run_start
        logical :: in_string, at_descriptor_start

        call set_empty(error_msg)
        write (location, '(" at line ",I0,", column ",I0)') line, column
        depth = 0
        in_string = .false.
        quote = ' '
        at_descriptor_start = .true.
        i = 1
        do while (i <= len(text))
            ch = text(i:i)
            if (in_string) then
                if (ch == quote) in_string = .false.
                i = i + 1
                cycle
            end if
            if (ch == '''' .or. ch == '"') then
                in_string = .true.
                quote = ch
                at_descriptor_start = .false.
                i = i + 1
                cycle
            end if
            if (ch == '(') then
                depth = depth + 1
                at_descriptor_start = .true.
                i = i + 1
                cycle
            end if
            if (ch == ')') then
                depth = depth - 1
                if (depth < 0) then
                    error_msg = 'Unexpected end of format string'// &
                        trim(location)//': unbalanced parenthesis in '''// &
                        trim(text)//''''
                    return
                end if
                at_descriptor_start = .false.
                i = i + 1
                cycle
            end if
            if (ch == ',' .or. ch == '/' .or. ch == ':') then
                at_descriptor_start = .true.
                i = i + 1
                cycle
            end if
            if (ch == ' ') then
                i = i + 1
                cycle
            end if
            if (scan(ch, '0123456789') > 0) then
                run_start = i
                do while (i <= len(text))
                    if (scan(text(i:i), '0123456789') == 0) exit
                    i = i + 1
                end do
                if (at_descriptor_start) then
                    call check_zero_repeat(text, run_start, i, location, &
                                           error_msg)
                    if (len_trim(error_msg) > 0) return
                end if
                at_descriptor_start = .false.
                cycle
            end if
            at_descriptor_start = .false.
            i = i + 1
        end do
        if (depth /= 0) then
            error_msg = 'Unexpected end of format string'//trim(location)// &
                ': unbalanced parenthesis in '''//trim(text)//''''
        end if
    end procedure check_format_text

    ! A digit run in repeat-specification position whose value is zero is only
    ! legal as the scale factor of a P edit descriptor: r in an r-repeated
    ! descriptor must be positive.
    module procedure check_zero_repeat
        integer :: i

        call set_empty(error_msg)
        if (verify(text(run_start:run_end - 1), '0') /= 0) return
        i = run_end
        do while (i <= len(text))
            if (text(i:i) /= ' ') exit
            i = i + 1
        end do
        if (i <= len(text)) then
            if (text(i:i) == 'P' .or. text(i:i) == 'p') return
        end if
        error_msg = 'Expected P edit descriptor'//trim(location)// &
            ': zero repeat specification in '''//trim(text)//''''
    end procedure check_zero_repeat

    ! A format written as a concatenation of character literals is still a
    ! complete format specification. FortFront drops a data transfer statement
    ! whose parenthesised format expression never closes, so the concatenated
    ! literal text is checked from the source (gfortran: "Unexpected end of
    ! format string").
    module procedure check_concatenated_format_source
        character(len=:), allocatable :: source, line, logical_line
        logical :: found
        integer :: pos, line_no, next_nl, first_line_no

        call set_empty(error_msg)
        call get_source_text(arena, source, found)
        if (.not. found) return
        pos = 1
        line_no = 1
        logical_line = ''
        first_line_no = 1
        do while (pos <= len(source))
            next_nl = index(source(pos:), new_line('a'))
            if (next_nl == 0) then
                line = source(pos:)
                pos = len(source) + 1
            else
                line = source(pos:pos + next_nl - 2)
                pos = pos + next_nl
            end if
            if (len(logical_line) == 0) first_line_no = line_no
            call append_continuation_line(logical_line, line)
            line_no = line_no + 1
            if (ends_with_continuation(logical_line)) cycle
            call check_transfer_line_format(logical_line, first_line_no, &
                                            error_msg)
            logical_line = ''
            if (len_trim(error_msg) > 0) return
        end do
        if (len(logical_line) > 0) then
            call check_transfer_line_format(logical_line, first_line_no, &
                                            error_msg)
        end if
    end procedure check_concatenated_format_source

    ! Free-form continuation: a trailing '&' joins the next line, whose own
    ! leading '&' (if any) is not part of the statement text. The check runs on
    ! the joined logical line so a continued format specification is seen whole.
    module procedure ends_with_continuation
        character(len=:), allocatable :: code

        ends_with_continuation = .false.
        call strip_line_comment(text, code)
        code = trim(code)
        if (len(code) == 0) return
        ends_with_continuation = code(len(code):len(code)) == '&'
    end procedure ends_with_continuation

    module procedure append_continuation_line
        character(len=:), allocatable :: piece

        if (len(logical_line) == 0) then
            logical_line = line
            return
        end if
        logical_line = trim(logical_line)
        logical_line = logical_line(1:len(logical_line) - 1)
        piece = adjustl(line)
        if (len_trim(piece) > 0) then
            if (piece(1:1) == '&') piece = piece(2:)
        end if
        logical_line = logical_line//piece
    end procedure append_continuation_line

    module procedure check_transfer_line_format
        character(len=:), allocatable :: text, keyword, format_text, joined
        integer :: start_pos, split_pos, label_end
        logical :: literal_only

        call set_empty(error_msg)
        call strip_line_comment(line, text)
        text = trim(adjustl(text))
        if (len(text) == 0) return
        label_end = verify(text, '0123456789') - 1
        if (label_end > 0) text = trim(adjustl(text(label_end + 1:)))
        if (len(text) == 0) return
        keyword = leading_identifier(lowercase_text(text))
        if (len_trim(keyword) == 0) return
        start_pos = len_trim(keyword) + 1
        if (start_pos > len(text)) return
        if (trim(keyword) == 'format') then
            format_text = trim(adjustl(text(start_pos:)))
            if (len(format_text) == 0) return
            if (format_text(1:1) /= '(') return
            call check_format_text(format_text, line_no, 1, error_msg)
            return
        end if
        select case (trim(keyword))
        case ('read', 'write', 'print')
        case default
            return
        end select
        call check_asynchronous_source(text(start_pos:), line_no, error_msg)
        if (len_trim(error_msg) > 0) return
        if (start_pos > len(text)) return
        call format_expression_text(text(start_pos:), format_text, split_pos)
        if (index(format_text, '//') == 0) return
        call concatenated_literal_text(format_text, joined, literal_only)
        if (.not. literal_only) return
        if (len_trim(joined) == 0) return
        if (joined(1:1) /= '(') return
        call check_format_text(joined, line_no, 1, error_msg)
    end procedure check_transfer_line_format

    ! The ASYNCHRONOUS= value of a data transfer statement must be an
    ! initialization expression; FortFront drops a whole bare main program that
    ! carries such a statement, so the specifier is also checked from the
    ! source text.
    module procedure check_asynchronous_source
        character(len=:), allocatable :: lowered, value, name, tail
        character(len=64) :: location
        integer :: at, rest

        call set_empty(error_msg)
        lowered = lowercase_text(text)
        at = index(lowered, 'asynchronous')
        if (at == 0) return
        rest = at + len('asynchronous')
        if (rest > len(lowered)) return
        tail = adjustl(lowered(rest:))
        if (len_trim(tail) == 0) return
        if (tail(1:1) /= '=') return
        value = adjustl(tail(2:))
        name = leading_identifier(value)
        if (len_trim(name) == 0) return
        rest = len_trim(name) + 1
        if (rest > len(value)) return
        tail = adjustl(value(rest:))
        if (len_trim(tail) == 0) return
        if (tail(1:1) /= '(') return
        if (is_character_intrinsic_name(name)) return
        write (location, '(" at line ",I0)') line_no
        error_msg = 'Function '''//trim(name)// &
            ''' in the ASYNCHRONOUS= specifier'//trim(location)// &
            ' must be an intrinsic function'
    end procedure check_asynchronous_source

    ! Drop a trailing comment so the statement text can be scanned for the
    ! literal tokens of its format expression.
    module procedure strip_line_comment
        character(len=1) :: ch, quote
        integer :: i
        logical :: in_string

        in_string = .false.
        quote = ' '
        stripped = line
        do i = 1, len(line)
            ch = line(i:i)
            if (in_string) then
                if (ch == quote) in_string = .false.
                cycle
            end if
            if (ch == '''' .or. ch == '"') then
                in_string = .true.
                quote = ch
                cycle
            end if
            if (ch == '!') then
                stripped = line(1:i - 1)
                return
            end if
        end do
    end procedure strip_line_comment

    ! The format expression of a data transfer statement written without a
    ! control list ends at the first comma outside parentheses and quotes.
    module procedure format_expression_text
        character(len=1) :: ch, quote
        integer :: i, depth
        logical :: in_string

        depth = 0
        in_string = .false.
        quote = ' '
        split_pos = 0
        do i = 1, len(text)
            ch = text(i:i)
            if (in_string) then
                if (ch == quote) in_string = .false.
                cycle
            end if
            if (ch == '''' .or. ch == '"') then
                in_string = .true.
                quote = ch
                cycle
            end if
            if (ch == '(') depth = depth + 1
            if (ch == ')') depth = depth - 1
            if (ch == ',' .and. depth == 0) then
                split_pos = i
                exit
            end if
        end do
        if (split_pos > 0) then
            format_text = text(1:split_pos - 1)
        else
            format_text = text
        end if
    end procedure format_expression_text

    ! Join the character literals of an expression built only from literals,
    ! concatenation operators, parentheses and blanks. Any other token means
    ! the text is not a compile-time known format.
    module procedure concatenated_literal_text
        character(len=1) :: ch, quote
        integer :: i

        joined = ''
        literal_only = .true.
        i = 1
        do while (i <= len(text))
            ch = text(i:i)
            if (ch == '''' .or. ch == '"') then
                quote = ch
                i = i + 1
                do while (i <= len(text))
                    if (text(i:i) == quote) exit
                    joined = joined//text(i:i)
                    i = i + 1
                end do
                if (i > len(text)) then
                    literal_only = .false.
                    return
                end if
                i = i + 1
                cycle
            end if
            if (scan(ch, '() /') == 0) then
                literal_only = .false.
                return
            end if
            i = i + 1
        end do
    end procedure concatenated_literal_text

    ! ---------------------------------------------------------------------
    ! Derived-type definition and component-access constraints (#390).
    !
    ! Three rules are checked here against typed nodes, because the arena
    ! carries the declared attributes each one needs:
    !
    !   * a component declared PRIVATE is accessible only inside the module
    !     that defines the type, so neither a component reference nor a
    !     structure constructor for that type is valid outside it
    !     (gfortran: "is a PRIVATE component of");
    !   * a DATA statement object may not be reached through an allocatable
    !     component, whose storage does not exist before execution
    !     (gfortran: "Allocatable component or deferred-shaped array");
    !   * a function result declared CLASS must be allocatable or a pointer,
    !     since a result variable is never a dummy argument
    !     (gfortran: "CLASS variable ... must be dummy, allocatable or
    !     pointer").
    ! ---------------------------------------------------------------------
    module procedure check_private_component_access
        integer :: n, module_idx
        character(len=:), allocatable :: type_name, comp_name

        call set_empty(error_msg)
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (nd => arena%entries(n)%node)
            type is (component_access_node)
                if (.not. allocated(nd%component_name)) cycle
                call designator_type_name(arena, nd%base_expr_index, type_name)
                if (len_trim(type_name) == 0) cycle
                if (.not. component_is_private(arena, type_name, &
                                               nd%component_name, module_idx)) cycle
                if (node_within(arena, n, module_idx)) cycle
                error_msg = private_component_message(arena, n, &
                                                      nd%component_name, type_name)
                return
            type is (call_or_subscript_node)
                if (.not. allocated(nd%name)) cycle
                call first_private_component(arena, nd%name, comp_name, module_idx)
                if (len_trim(comp_name) == 0) cycle
                if (node_within(arena, n, module_idx)) cycle
                error_msg = private_component_message(arena, n, comp_name, nd%name)
                return
            end select
        end do
    end procedure check_private_component_access

    module procedure private_component_message
        character(len=64) :: location

        write (location, '(" at line ",I0,", column ",I0)') &
            get_node_line(arena, node_index), get_node_column(arena, node_index)
        message = 'component '''//trim(comp_name)//''' is a PRIVATE '// &
            'component of '''//trim(type_name)//''''//trim(location)
    end procedure private_component_message

    module procedure component_is_private
        integer :: decl, type_idx

        module_idx = 0
        component_is_private = .false.
        decl = derived_component_decl(arena, type_name, comp_name, type_idx)
        if (decl <= 0) return
        select type (dn => arena%entries(decl)%node)
        type is (declaration_node)
            if (.not. allocated(dn%accessibility)) return
            if (trim(lowercase_text(dn%accessibility)) /= 'private') return
        class default
            return
        end select
        module_idx = enclosing_module_index(arena, type_idx)
        if (module_idx <= 0) return
        component_is_private = .true.
    end procedure component_is_private

    module procedure first_private_component
        !! Name of the first PRIVATE component of the named derived type, and
        !! the module that defines the type. Empty when the name is not a
        !! module derived type or has no private component.
        integer :: type_idx, i, comp

        call set_empty(comp_name)
        module_idx = 0
        type_idx = derived_type_node_index(arena, type_name)
        if (type_idx <= 0) return
        select type (td => arena%entries(type_idx)%node)
        type is (derived_type_node)
            if (.not. allocated(td%component_indices)) return
            do i = 1, size(td%component_indices)
                comp = td%component_indices(i)
                if (.not. node_exists(arena, comp)) cycle
                select type (dn => arena%entries(comp)%node)
                type is (declaration_node)
                    if (.not. allocated(dn%accessibility)) cycle
                    if (trim(lowercase_text(dn%accessibility)) /= 'private') cycle
                    if (.not. allocated(dn%var_name)) cycle
                    comp_name = trim(dn%var_name)
                    module_idx = enclosing_module_index(arena, type_idx)
                    if (module_idx <= 0) call set_empty(comp_name)
                    return
                end select
            end do
        end select
    end procedure first_private_component

    module procedure derived_type_node_index
        integer :: n
        character(len=:), allocatable :: wanted

        type_idx = 0
        wanted = trim(lowercase_text(type_name))
        if (len(wanted) == 0) return
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (nd => arena%entries(n)%node)
            type is (derived_type_node)
                if (.not. allocated(nd%name)) cycle
                if (trim(lowercase_text(nd%name)) /= wanted) cycle
                type_idx = n
                return
            end select
        end do
    end procedure derived_type_node_index

    module procedure derived_component_decl
        integer :: i, comp
        character(len=:), allocatable :: wanted

        decl = 0
        wanted = trim(lowercase_text(comp_name))
        type_idx = derived_type_node_index(arena, type_name)
        if (type_idx <= 0) return
        select type (td => arena%entries(type_idx)%node)
        type is (derived_type_node)
            if (.not. allocated(td%component_indices)) return
            do i = 1, size(td%component_indices)
                comp = td%component_indices(i)
                if (.not. node_exists(arena, comp)) cycle
                select type (dn => arena%entries(comp)%node)
                type is (declaration_node)
                    if (.not. allocated(dn%var_name)) cycle
                    if (trim(lowercase_text(dn%var_name)) /= wanted) cycle
                    decl = comp
                    return
                end select
            end do
        end select
    end procedure derived_component_decl

    module procedure enclosing_module_index
        integer :: cur, guard

        module_idx = 0
        cur = node_index
        guard = 0
        do while (cur > 0)
            guard = guard + 1
            if (guard > arena%size) return
            if (.not. node_exists(arena, cur)) return
            select type (nd => arena%entries(cur)%node)
            type is (module_node)
                module_idx = cur
                return
            end select
            cur = arena%entries(cur)%parent_index
        end do
    end procedure enclosing_module_index

    module procedure node_within
        integer :: cur, guard

        within = .false.
        if (ancestor <= 0) return
        cur = node_index
        guard = 0
        do while (cur > 0)
            if (cur == ancestor) then
                within = .true.
                return
            end if
            guard = guard + 1
            if (guard > arena%size) return
            if (.not. node_exists(arena, cur)) return
            cur = arena%entries(cur)%parent_index
        end do
    end procedure node_within

    module procedure designator_type_name
        !! Declared derived-type name of a designator (identifier, array
        !! element, or component reference). Empty when it is not a derived
        !! type entity or cannot be resolved from declarations alone.
        character(len=:), allocatable :: base_type
        integer :: decl, comp, type_idx

        call set_empty(type_name)
        if (idx <= 0) return
        if (.not. node_exists(arena, idx)) return
        select type (nd => arena%entries(idx)%node)
        type is (identifier_node)
            if (.not. allocated(nd%name)) return
            decl = declaration_index_for_name(arena, nd%name)
            if (decl <= 0) return
            call declared_derived_type_name(arena, decl, type_name)
        type is (component_access_node)
            if (.not. allocated(nd%component_name)) return
            call designator_type_name(arena, nd%base_expr_index, base_type)
            if (len_trim(base_type) == 0) return
            comp = derived_component_decl(arena, base_type, nd%component_name, &
                                          type_idx)
            if (comp <= 0) return
            call declared_derived_type_name(arena, comp, type_name)
        type is (call_or_subscript_node)
            if (nd%base_expr_index > 0) then
                call designator_type_name(arena, nd%base_expr_index, type_name)
                return
            end if
            if (.not. allocated(nd%name)) return
            decl = declaration_index_for_name(arena, nd%name)
            if (decl <= 0) return
            call declared_derived_type_name(arena, decl, type_name)
        end select
    end procedure designator_type_name

    module procedure declared_derived_type_name
        !! 'type(t)' / 'class(t)' spelled on a declaration reduced to 't'.
        character(len=:), allocatable :: spec
        integer :: open_paren, close_paren

        call set_empty(type_name)
        if (.not. node_exists(arena, decl)) return
        select type (dn => arena%entries(decl)%node)
        type is (declaration_node)
            if (.not. allocated(dn%type_name)) return
            spec = trim(lowercase_text(dn%type_name))
        class default
            return
        end select
        open_paren = index(spec, '(')
        close_paren = index(spec, ')', back=.true.)
        if (open_paren <= 1) return
        if (close_paren <= open_paren + 1) return
        select case (trim(spec(:open_paren - 1)))
        case ('type', 'class')
            type_name = trim(adjustl(spec(open_paren + 1:close_paren - 1)))
        case default
            call set_empty(type_name)
        end select
    end procedure declared_derived_type_name

    module procedure check_data_allocatable_components
        integer :: n, i

        call set_empty(error_msg)
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (nd => arena%entries(n)%node)
            type is (data_statement_node)
                if (.not. allocated(nd%object_indices)) cycle
                do i = 1, size(nd%object_indices)
                    call check_data_object_components(arena, nd%object_indices(i), &
                                                      error_msg)
                    if (len_trim(error_msg) > 0) return
                end do
            end select
        end do
    end procedure check_data_allocatable_components

    module procedure check_data_object_components
        character(len=:), allocatable :: base_type
        character(len=64) :: location
        integer :: j, comp, type_idx

        call set_empty(error_msg)
        if (idx <= 0) return
        if (.not. node_exists(arena, idx)) return
        select type (nd => arena%entries(idx)%node)
        type is (io_implied_do_node)
            if (.not. allocated(nd%object_indices)) return
            do j = 1, size(nd%object_indices)
                call check_data_object_components(arena, nd%object_indices(j), &
                                                  error_msg)
                if (len_trim(error_msg) > 0) return
            end do
        type is (call_or_subscript_node)
            call check_data_object_components(arena, nd%base_expr_index, error_msg)
        type is (component_access_node)
            call check_data_object_components(arena, nd%base_expr_index, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. allocated(nd%component_name)) return
            call designator_type_name(arena, nd%base_expr_index, base_type)
            if (len_trim(base_type) == 0) return
            comp = derived_component_decl(arena, base_type, nd%component_name, &
                                          type_idx)
            if (comp <= 0) return
            select type (dn => arena%entries(comp)%node)
            type is (declaration_node)
                if (.not. dn%is_allocatable) return
            class default
                return
            end select
            write (location, '(" at line ",I0,", column ",I0)') &
                get_node_line(arena, idx), get_node_column(arena, idx)
            error_msg = 'allocatable component '''//trim(nd%component_name)// &
                ''' of '''//trim(base_type)//''' cannot appear in a DATA '// &
                'statement'//trim(location)
        end select
    end procedure check_data_object_components
end submodule session_program_lowering_reject_checks
