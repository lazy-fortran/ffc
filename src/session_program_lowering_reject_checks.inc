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
    subroutine check_generic_ambiguity(arena, context, generic_idx, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: generic_idx
        character(len=:), allocatable, intent(out) :: error_msg
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
    end subroutine check_generic_ambiguity

    logical function specifics_indistinguishable(arena, name_a, name_b) &
            result(same)
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: name_a, name_b
        integer :: count_a, count_b, pos, vk_a, vk_b
        logical :: char_a, char_b

        same = .false.
        count_a = arena_proc_param_count(arena, name_a)
        count_b = arena_proc_param_count(arena, name_b)
        if (count_a < 0 .or. count_b < 0) return
        if (count_a /= count_b) return
        do pos = 1, count_a
            if (callee_dummy_is_array(arena, name_a, pos)) return
            if (callee_dummy_is_array(arena, name_b, pos)) return
            char_a = callee_dummy_is_character(arena, name_a, pos)
            char_b = callee_dummy_is_character(arena, name_b, pos)
            if (char_a .neqv. char_b) return
            if (.not. char_a) then
                vk_a = callee_dummy_value_kind(arena, name_a, pos)
                vk_b = callee_dummy_value_kind(arena, name_b, pos)
                if (vk_a == 0 .or. vk_b == 0) return
                if (vk_a /= vk_b) return
            end if
        end do
        same = .true.
    end function specifics_indistinguishable

    integer function arena_proc_param_count(arena, proc_name) result(count)
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: proc_name
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
    end function arena_proc_param_count

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
    subroutine check_array_constructor_type_specs(arena, error_msg)
        type(ast_arena_t), intent(in) :: arena
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: n

        call set_empty(error_msg)
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            call check_one_array_constructor(arena, n, error_msg)
            if (len_trim(error_msg) > 0) return
        end do
    end subroutine check_array_constructor_type_specs

    subroutine check_one_array_constructor(arena, node_index, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        character(len=:), allocatable, intent(out) :: error_msg
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
    end subroutine check_one_array_constructor

    recursive subroutine check_array_ctor_elements(arena, elems, spec_class, &
                                                   spec_text, line, col, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: elems(:)
        integer, intent(in) :: spec_class, line, col
        character(len=*), intent(in) :: spec_text
        character(len=:), allocatable, intent(out) :: error_msg
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
    end subroutine check_array_ctor_elements

    integer function array_ctor_typespec_class(type_spec) result(cls)
        character(len=*), intent(in) :: type_spec
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
    end function array_ctor_typespec_class

    logical function starts_with_word(text, word) result(matches)
        character(len=*), intent(in) :: text, word
        integer :: nw

        nw = len(word)
        matches = .false.
        if (len(text) < nw) return
        if (text(1:nw) /= word) return
        if (len(text) == nw) then
            matches = .true.
        else
            matches = scan(text(nw + 1:nw + 1), 'abcdefghijklmnopqrstuvwxyz_') == 0
        end if
    end function starts_with_word

    integer function array_ctor_literal_class(arena, node_index) result(cls)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
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
    end function array_ctor_literal_class

    function cmp_class_name(cls) result(name)
        integer, intent(in) :: cls
        character(len=:), allocatable :: name

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
    end function cmp_class_name

    ! A BOZ-literal-constant is not one of the ac-value forms an array
    ! constructor may hold: it is only valid as a DATA-statement constant, an
    ! actual argument to INT/REAL/DBLE/CMPLX, or (as a gfortran extension) the
    ! right side of a scalar intrinsic assignment. This flags any array
    ! constructor element whose literal spelling is a BOZ constant, whether
    ! or not the constructor carries a type-spec (both forms are rejected by
    ! gfortran).
    subroutine check_boz_in_array_constructors(arena, error_msg)
        type(ast_arena_t), intent(in) :: arena
        character(len=:), allocatable, intent(out) :: error_msg
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
    end subroutine check_boz_in_array_constructors

    recursive subroutine check_boz_ctor_elements(arena, elems, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: elems(:)
        character(len=:), allocatable, intent(out) :: error_msg
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
    end subroutine check_boz_ctor_elements

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
    subroutine check_assumed_size_dimension_order(arena, error_msg)
        type(ast_arena_t), intent(in) :: arena
        character(len=:), allocatable, intent(out) :: error_msg
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
    end subroutine check_assumed_size_dimension_order

    subroutine check_dims_assumed_size_order(arena, dimension_indices, line, &
                                             column, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: dimension_indices(:)
        integer, intent(in) :: line, column
        character(len=:), allocatable, intent(out) :: error_msg
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
    end subroutine check_dims_assumed_size_order
