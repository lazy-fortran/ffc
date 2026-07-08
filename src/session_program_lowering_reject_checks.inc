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

    ! A name shall not appear in both an EXTERNAL and an INTRINSIC statement
    ! in the same scoping unit (F2018 8.10 note 12, gfortran: "EXTERNAL
    ! attribute conflicts with INTRINSIC attribute"). Both a bare INTRINSIC
    ! statement and a bare EXTERNAL statement lower as no-ops (#291), so
    ! nothing else notices the same name spelled both ways; this scans every
    ! INTRINSIC name against every bare-EXTERNAL declaration in the program.
    subroutine check_intrinsic_external_conflict(arena, error_msg)
        type(ast_arena_t), intent(in) :: arena
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: n, i
        character(len=:), allocatable :: intr_name, ext_name

        call set_empty(error_msg)
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (nd => arena%entries(n)%node)
            type is (intrinsic_statement_node)
                if (.not. allocated(nd%procedure_names)) cycle
                do i = 1, size(nd%procedure_names)
                    if (.not. allocated(nd%procedure_names(i)%s)) cycle
                    intr_name = trim(nd%procedure_names(i)%s)
                    if (len_trim(intr_name) == 0) cycle
                    call find_bare_external_name(arena, intr_name, ext_name)
                    if (len_trim(ext_name) > 0) then
                        error_msg = 'EXTERNAL attribute conflicts with '// &
                            'INTRINSIC attribute for '''//intr_name//''''
                        return
                    end if
                end do
            end select
        end do
    end subroutine check_intrinsic_external_conflict

    subroutine find_bare_external_name(arena, target_name, found_name)
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: target_name
        character(len=:), allocatable, intent(out) :: found_name
        integer :: n, i

        call set_empty(found_name)
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (nd => arena%entries(n)%node)
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
    end subroutine find_bare_external_name

    ! A function RESULT variable never carries the SAVE attribute (gfortran:
    ! "RESULT attribute conflicts with SAVE attribute", PR20856). The result
    ! variable only differs from the function's own name when RESULT(...) is
    ! used, so this only fires on that explicit form: scan each function's
    ! own body declarations for one that both names the result variable and
    ! sets is_save.
    subroutine check_function_result_save(arena, error_msg)
        type(ast_arena_t), intent(in) :: arena
        character(len=:), allocatable, intent(out) :: error_msg
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
    end subroutine check_function_result_save

    subroutine check_result_save_in_body(arena, result_name, body_indices, &
                                         error_msg)
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: result_name
        integer, intent(in) :: body_indices(:)
        character(len=:), allocatable, intent(out) :: error_msg
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
    end subroutine check_result_save_in_body

    ! Two procedures sharing one name in the same CONTAINS section are not
    ! distinguishable at the call site (gfortran: "Procedure ... is already
    ! defined at ..."). Scan each program/module/submodule's own contained-
    ! procedure list for a repeated name; the same name reused across two
    ! different scopes (two separate modules each defining their own "bah")
    ! is ordinary Fortran and stays untouched since each scope is checked on
    ! its own index list.
    subroutine check_duplicate_contained_procedures(arena, error_msg)
        type(ast_arena_t), intent(in) :: arena
        character(len=:), allocatable, intent(out) :: error_msg
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
    end subroutine check_duplicate_contained_procedures

    subroutine check_procedure_names_unique(arena, indices, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: indices(:)
        character(len=:), allocatable, intent(out) :: error_msg
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
    end subroutine check_procedure_names_unique

    subroutine procedure_def_name(arena, node_index, name)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        character(len=:), allocatable, intent(out) :: name

        call set_empty(name)
        if (.not. node_exists(arena, node_index)) return
        select type (nd => arena%entries(node_index)%node)
        type is (function_def_node)
            if (allocated(nd%name)) name = trim(nd%name)
        type is (subroutine_def_node)
            if (allocated(nd%name)) name = trim(nd%name)
        end select
    end subroutine procedure_def_name

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
    subroutine check_bit_intrinsic_arg_ranges(arena, error_msg)
        type(ast_arena_t), intent(in) :: arena
        character(len=:), allocatable, intent(out) :: error_msg
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
    end subroutine check_bit_intrinsic_arg_ranges

    subroutine check_bit_intrinsic_call(arena, name, arg_indices, line, col, &
                                        error_msg)
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: name
        integer, intent(in) :: arg_indices(:)
        integer, intent(in) :: line, col
        character(len=:), allocatable, intent(out) :: error_msg
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
    end subroutine check_bit_intrinsic_call

    ! Fold an argument expression to a compile-time i64 value when it is built
    ! solely from integer literals and +, -, * operators. Anything else
    ! (identifiers, other operators, non-integer literals) leaves ok = .false.
    ! so callers treat the value as statically unknown. This never consults the
    ! symbol table, so it is safe from validate_program, which has no lowering
    ! context.
    recursive subroutine try_const_int64(arena, node_index, value, ok)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        integer(c_int64_t), intent(out) :: value
        logical, intent(out) :: ok
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
    end subroutine try_const_int64

    logical function is_integer_text(text) result(is_int)
        character(len=*), intent(in) :: text
        character(len=:), allocatable :: trimmed
        integer :: i

        is_int = .false.
        trimmed = trim(adjustl(text))
        if (len(trimmed) == 0) return
        do i = 1, len(trimmed)
            if (scan(trimmed(i:i), '0123456789') == 0) return
        end do
        is_int = .true.
    end function is_integer_text

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
    subroutine check_scope_nonconstant_bounds(arena, error_msg)
        type(ast_arena_t), intent(in) :: arena
        character(len=:), allocatable, intent(out) :: error_msg
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
    end subroutine check_scope_nonconstant_bounds

    subroutine check_decls_nonconstant_bounds(arena, indices, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: indices(:)
        character(len=:), allocatable, intent(out) :: error_msg
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
    end subroutine check_decls_nonconstant_bounds

    ! True when a bound expression contains a function call that cannot be a
    ! constant: a call to a user function defined in this program, or the
    ! intrinsic command_argument_count(). Constant-expression intrinsics such as
    ! size/kind/len never match, so a legal constant bound is not flagged.
    recursive function expr_has_illegal_call(arena, idx) result(found)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: idx
        logical :: found
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
    end function expr_has_illegal_call

    logical function arena_has_function_def_named(arena, name) result(found)
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: name
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
    end function arena_has_function_def_named

    subroutine check_derived_type_names_not_intrinsic(arena, error_msg)
        type(ast_arena_t), intent(in) :: arena
        character(len=:), allocatable, intent(out) :: error_msg
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
    end subroutine check_derived_type_names_not_intrinsic

    subroutine check_intrinsic_type_stmt_source(arena, error_msg)
        type(ast_arena_t), intent(in) :: arena
        character(len=:), allocatable, intent(out) :: error_msg
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
    end subroutine check_intrinsic_type_stmt_source

    subroutine intrinsic_type_stmt_name(line, name, column)
        character(len=*), intent(in) :: line
        character(len=:), allocatable, intent(out) :: name
        integer, intent(out) :: column
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
    end subroutine intrinsic_type_stmt_name

    logical function derived_type_name_is_intrinsic(name) result(is_intrinsic)
        character(len=*), intent(in) :: name
        character(len=:), allocatable :: lowered

        lowered = trim(lowercase_text(name))
        select case (lowered)
        case ('integer', 'real', 'complex', 'character', 'logical', &
              'doubleprecision', 'doublecomplex')
            is_intrinsic = .true.
        case default
            is_intrinsic = .false.
        end select
    end function derived_type_name_is_intrinsic

    ! A polymorphic (CLASS) entity that is neither a dummy argument nor
    ! allocatable nor a pointer has no way to take on a dynamic type, so the
    ! standard forbids it (gfortran: "must be dummy, allocatable or pointer",
    ! F2018 C708). At main-program and module scope there are no dummy arguments,
    ! so any non-allocatable non-pointer CLASS entity there is invalid.
    subroutine check_scope_class_declarations(arena, error_msg)
        type(ast_arena_t), intent(in) :: arena
        character(len=:), allocatable, intent(out) :: error_msg
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
    end subroutine check_scope_class_declarations

    subroutine check_decls_class(arena, indices, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: indices(:)
        character(len=:), allocatable, intent(out) :: error_msg
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
    end subroutine check_decls_class

    ! An automatic array - one whose bounds depend on a dummy argument, so its
    ! size is known only at run time - has no static storage. It can appear
    ! neither in a COMMON block (gfortran: "Automatic object ... cannot appear
    ! in COMMON") nor as an EQUIVALENCE object (gfortran: "Array ... with
    ! non-constant bounds cannot be an EQUIVALENCE object"). The runtime-local
    ! path lowers such arrays as dynamic allocations, correct for a plain local
    ! but wrong once storage association is required; this rejects the two
    ! storage-association contexts. Constant-bound arrays (x(8)) are untouched.
    subroutine check_automatic_storage_association(arena, error_msg)
        type(ast_arena_t), intent(in) :: arena
        character(len=:), allocatable, intent(out) :: error_msg
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
    end subroutine check_automatic_storage_association

    subroutine check_proc_automatic_assoc(arena, param_indices, body_indices, &
                                          error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: param_indices(:)
        integer, intent(in) :: body_indices(:)
        character(len=:), allocatable, intent(out) :: error_msg
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
    end subroutine check_proc_automatic_assoc

    logical function decl_bound_refs_param(arena, decl, param_indices) &
            result(refs)
        type(ast_arena_t), intent(in) :: arena
        type(declaration_node), intent(in) :: decl
        integer, intent(in) :: param_indices(:)
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
    end function decl_bound_refs_param

    recursive function expr_refs_name(arena, idx, name) result(found)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: idx
        character(len=*), intent(in) :: name
        logical :: found
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
    end function expr_refs_name

    logical function name_in_common(arena, body_indices, name) result(found)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: body_indices(:)
        character(len=*), intent(in) :: name
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
    end function name_in_common

    logical function name_in_equivalence(arena, body_indices, name) result(found)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: body_indices(:)
        character(len=*), intent(in) :: name
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
    end function name_in_equivalence
