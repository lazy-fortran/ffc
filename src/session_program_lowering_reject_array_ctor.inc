    ! Array-constructor compatibility (F2018 7.8, 10.2.1.2, R769).
    !
    ! Four rules live here:
    !   1. An array constructor opened with (/ closes with /), and one opened
    !      with [ closes with ]. Mixing the delimiters is not an array
    !      constructor at all.
    !   2. A structure constructor cannot initialise an entity of intrinsic
    !      type, and an array constructor cannot initialise a scalar: the
    !      ranks are incompatible.
    !   3. The values of an array constructor must be assignment-compatible
    !      with the variable they are assigned to; LOGICAL values never
    !      convert to a numeric or character variable.
    !   4. Without a type-spec, every ac-value of an array constructor has the
    !      same intrinsic type: there is no single element type to lower
    !      otherwise. [1, 2.5] is not an integer constructor with a truncated
    !      element, it is invalid.

    subroutine check_array_constructor_compatibility(arena, error_msg, lazy_mode)
        type(ast_arena_t), intent(in) :: arena
        character(len=:), allocatable, intent(out) :: error_msg
        logical, intent(in), optional :: lazy_mode
        logical :: is_lazy

        is_lazy = .false.
        if (present(lazy_mode)) is_lazy = lazy_mode

        call set_empty(error_msg)
        call check_array_ctor_delimiters(arena, error_msg)
        if (len_trim(error_msg) > 0) return
        call check_array_ctor_initializers(arena, error_msg)
        if (len_trim(error_msg) > 0) return
        call check_array_ctor_init_source(arena, error_msg)
        if (len_trim(error_msg) > 0) return
        call check_array_ctor_assignments(arena, error_msg)
        if (len_trim(error_msg) > 0) return
        call check_array_ctor_element_agreement(arena, is_lazy, error_msg)
    end subroutine check_array_constructor_compatibility

    ! Rule 4: an array constructor written without a type-spec takes its
    ! element type from its values, so literal values of differing intrinsic
    ! type leave the element type undetermined. Only literals are classified;
    ! anything the driver cannot type statically is left alone.
    subroutine check_array_ctor_element_agreement(arena, is_lazy, error_msg)
        type(ast_arena_t), intent(in) :: arena
        logical, intent(in) :: is_lazy
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: n

        call set_empty(error_msg)
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (nd => arena%entries(n)%node)
            type is (array_literal_node)
                if (allocated(nd%type_spec)) then
                    if (len_trim(nd%type_spec) > 0) cycle
                end if
                if (.not. allocated(nd%element_indices)) cycle
                call check_ctor_elements_agree(arena, nd%element_indices, &
                                               is_lazy, nd%line, nd%column, &
                                               error_msg)
                if (len_trim(error_msg) > 0) return
                if (.not. is_lazy) then
                    call check_ctor_char_lengths(arena, nd%element_indices, &
                                                 nd%line, nd%column, error_msg)
                    if (len_trim(error_msg) > 0) return
                end if
            end select
        end do
    end subroutine check_array_ctor_element_agreement

    subroutine check_ctor_elements_agree(arena, elems, is_lazy, line, col, &
                                         error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: elems(:)
        logical, intent(in) :: is_lazy
        integer, intent(in) :: line, col
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=64) :: location
        integer :: i, elem_class, first_class, first_pos

        call set_empty(error_msg)
        first_class = CTOR_TYPE_UNKNOWN
        first_pos = 0
        do i = 1, size(elems)
            elem_class = ctor_literal_intrinsic_type(arena, elems(i))
            if (elem_class == CTOR_TYPE_UNKNOWN) cycle
            if (first_class == CTOR_TYPE_UNKNOWN) then
                first_class = elem_class
                first_pos = i
                cycle
            end if
            if (elem_class == first_class) cycle
            ! Lazy Fortran promotes a mixed integer/real constructor to real
            ! instead of rejecting it; only standard Fortran demands one type.
            if (is_lazy .and. ctor_types_are_numeric(first_class, elem_class)) &
                cycle
            if (line > 0 .and. col > 0) then
                write (location, '(" at line ",I0,", column ",I0)') line, col
            else
                location = ''
            end if
            error_msg = 'array constructor'//trim(location)// &
                ' mixes '//ctor_type_name(first_class)//' element '// &
                int_to_text(first_pos)//' with '//ctor_type_name(elem_class)// &
                ' element '//int_to_text(i)// &
                ': a constructor without a type-spec has one element type'
            return
        end do
    end subroutine check_ctor_elements_agree

    ! Rule 5 (standard mode only): without a type-spec the character length of
    ! an array
    ! constructor is that of its values, so character literals of differing
    ! length leave the element length undetermined (F2018 7.8, gfortran
    ! "Different CHARACTER lengths in array constructor"). A type-spec fixes
    ! the length and pads shorter values, which the caller already skips.
    subroutine check_ctor_char_lengths(arena, elems, line, col, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: elems(:)
        integer, intent(in) :: line, col
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=64) :: location
        integer :: i, elem_len, first_len, first_pos

        call set_empty(error_msg)
        first_len = -1
        first_pos = 0
        do i = 1, size(elems)
            elem_len = ctor_char_literal_length(arena, elems(i))
            if (elem_len < 0) cycle
            if (first_len < 0) then
                first_len = elem_len
                first_pos = i
                cycle
            end if
            if (elem_len == first_len) cycle
            if (line > 0 .and. col > 0) then
                write (location, '(" at line ",I0,", column ",I0)') line, col
            else
                location = ''
            end if
            error_msg = 'array constructor'//trim(location)// &
                        ' has different CHARACTER lengths ('// &
                        int_to_text(first_len)//'/'//int_to_text(elem_len)// &
                        ') in element '//int_to_text(first_pos)//' and '// &
                        'element '//int_to_text(i)// &
                        ': a constructor without a type-spec has one '// &
                        'element length'
            return
        end do
    end subroutine check_ctor_char_lengths

    ! Length of a character literal ac-value, or -1 when the value is not a
    ! character literal the driver can measure statically.
    integer function ctor_char_literal_length(arena, idx) result(char_len)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: idx
        character(len=:), allocatable :: value, literal_type, err, text

        char_len = -1
        if (idx <= 0) return
        if (.not. node_exists(arena, idx)) return
        if (.not. is_character_literal(arena, idx)) return
        call get_literal_info(arena, idx, value, literal_type, err)
        if (len_trim(err) > 0) return
        call strip_literal_quotes(value, text)
        char_len = len(text)
    end function ctor_char_literal_length

    ! The intrinsic type of a literal ac-value, distinguishing integer from
    ! real (cmp_class_name lumps both into "numeric", which is too coarse to
    ! decide a constructor's element type).
    integer function ctor_literal_intrinsic_type(arena, idx) result(cls)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: idx
        character(len=:), allocatable :: value, literal_type, err

        cls = CTOR_TYPE_UNKNOWN
        if (idx <= 0) return
        if (.not. node_exists(arena, idx)) return
        call get_literal_info(arena, idx, value, literal_type, err)
        if (len_trim(err) > 0) return
        if (is_character_literal(arena, idx)) then
            cls = CTOR_TYPE_CHAR
        else if (is_logical_literal(arena, idx)) then
            cls = CTOR_TYPE_LOGICAL
        else if (is_real_literal(arena, idx)) then
            cls = CTOR_TYPE_REAL
        else
            cls = CTOR_TYPE_INTEGER
        end if
    end function ctor_literal_intrinsic_type

    logical function ctor_types_are_numeric(a, b) result(numeric)
        integer, intent(in) :: a, b

        numeric = ctor_type_is_numeric(a)
        if (numeric) numeric = ctor_type_is_numeric(b)
    end function ctor_types_are_numeric

    logical function ctor_type_is_numeric(cls) result(numeric)
        integer, intent(in) :: cls

        numeric = cls == CTOR_TYPE_INTEGER .or. cls == CTOR_TYPE_REAL
    end function ctor_type_is_numeric

    function ctor_type_name(cls) result(name)
        integer, intent(in) :: cls
        character(len=:), allocatable :: name

        select case (cls)
        case (CTOR_TYPE_INTEGER); name = 'INTEGER'
        case (CTOR_TYPE_REAL); name = 'REAL'
        case (CTOR_TYPE_LOGICAL); name = 'LOGICAL'
        case (CTOR_TYPE_CHAR); name = 'CHARACTER'
        case default; name = 'UNKNOWN'
        end select
    end function ctor_type_name

    function int_to_text(value) result(text)
        integer, intent(in) :: value
        character(len=:), allocatable :: text
        character(len=16) :: buffer

        write (buffer, '(I0)') value
        text = trim(buffer)
    end function int_to_text

    ! Rule 1: scan the source once, tracking the open bracket delimiters and
    ! rejecting a close that does not match the delimiter it terminates.
    subroutine check_array_ctor_delimiters(arena, error_msg)
        type(ast_arena_t), intent(in) :: arena
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: source
        character(len=1) :: stack(256), quote
        character(len=64) :: location
        logical :: found, in_string
        integer :: pos, depth, line_no, col, source_len

        call set_empty(error_msg)
        call get_source_text(arena, source, found)
        if (.not. found) return
        source_len = len(source)
        depth = 0
        pos = 1
        line_no = 1
        col = 1
        in_string = .false.
        quote = ' '
        do while (pos <= source_len)
            if (source(pos:pos) == new_line('a')) then
                line_no = line_no + 1
                col = 0
                in_string = .false.
                pos = pos + 1
                col = col + 1
                cycle
            end if
            if (in_string) then
                if (source(pos:pos) == quote) in_string = .false.
                pos = pos + 1
                col = col + 1
                cycle
            end if
            select case (source(pos:pos))
            case ('''', '"')
                in_string = .true.
                quote = source(pos:pos)
            case ('!')
                do while (pos <= source_len)
                    if (source(pos:pos) == new_line('a')) exit
                    pos = pos + 1
                    col = col + 1
                end do
                cycle
            case ('(')
                if (pos < source_len .and. depth < size(stack)) then
                    depth = depth + 1
                    if (source(pos + 1:pos + 1) == '/') then
                        stack(depth) = '/'
                        pos = pos + 1
                        col = col + 1
                    else
                        stack(depth) = '('
                    end if
                end if
            case ('[')
                if (depth < size(stack)) then
                    depth = depth + 1
                    stack(depth) = '['
                end if
            case (']')
                if (depth > 0) then
                    if (stack(depth) /= '[') then
                        write (location, '(" at line ",I0,", column ",I0)') &
                            line_no, col
                        error_msg = 'array constructor'//trim(location)// &
                            ' closed with '']'' but opened with ''(/'''
                        return
                    end if
                    depth = depth - 1
                end if
            case (')')
                if (depth > 0) depth = depth - 1
            case ('/')
                if (pos < source_len) then
                    if (source(pos + 1:pos + 1) == ')') then
                        if (depth > 0) then
                            if (stack(depth) /= '/') then
                                write (location, &
                                       '(" at line ",I0,", column ",I0)') &
                                    line_no, col
                                error_msg = 'array constructor'// &
                                    trim(location)//' closed with ''/)'''// &
                                    ' but opened with ''['''
                                return
                            end if
                            depth = depth - 1
                        end if
                        pos = pos + 1
                        col = col + 1
                    end if
                end if
            end select
            pos = pos + 1
            col = col + 1
        end do
    end subroutine check_array_ctor_delimiters

    ! Rule 2: an initializer must match the rank and type of the entity.
    subroutine check_array_ctor_initializers(arena, error_msg)
        type(ast_arena_t), intent(in) :: arena
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: ctor_name
        character(len=64) :: location
        integer :: n, decl_class
        logical :: is_scalar

        call set_empty(error_msg)
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (nd => arena%entries(n)%node)
            type is (declaration_node)
                if (.not. nd%has_initializer) cycle
                if (nd%initializer_index <= 0) cycle
                if (nd%is_inferred) cycle
                if (.not. allocated(nd%type_name)) cycle
                decl_class = array_ctor_typespec_class(nd%type_name)
                if (decl_class == CMP_CLASS_UNKNOWN) cycle
                if (.not. node_exists(arena, nd%initializer_index)) cycle
                is_scalar = .not. nd%is_array
                if (allocated(nd%dimension_indices)) then
                    if (size(nd%dimension_indices) > 0) is_scalar = .false.
                end if
                write (location, '(" at line ",I0,", column ",I0)') &
                    nd%line, nd%column
                call structure_ctor_type_name(arena, nd%initializer_index, &
                                              ctor_name)
                if (len_trim(ctor_name) > 0) then
                    error_msg = 'cannot convert TYPE('//trim(ctor_name)// &
                        ') to '//trim(adjustl(nd%type_name))// &
                        ' in the initialization of '// &
                        trim(declaration_first_name(nd))//trim(location)
                    return
                end if
                if (.not. is_scalar) cycle
                select type (init => arena%entries(nd%initializer_index)%node)
                type is (array_literal_node)
                    error_msg = 'incompatible ranks 0 and 1 in the '// &
                        'initialization of '// &
                        trim(declaration_first_name(nd))//trim(location)
                    return
                end select
            end select
        end do
    end subroutine check_array_ctor_initializers

    function declaration_first_name(decl) result(name)
        type(declaration_node), intent(in) :: decl
        character(len=:), allocatable :: name

        name = 'a variable'
        if (allocated(decl%var_name)) then
            name = trim(decl%var_name)
        else if (allocated(decl%var_names)) then
            if (size(decl%var_names) > 0) name = trim(decl%var_names(1))
        end if
    end function declaration_first_name

    ! A call whose name is a derived type defined in this compilation is a
    ! structure constructor, not a function reference.
    subroutine structure_ctor_type_name(arena, idx, type_name)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: idx
        character(len=:), allocatable, intent(out) :: type_name
        character(len=:), allocatable :: candidate

        call set_empty(type_name)
        if (idx <= 0) return
        if (.not. node_exists(arena, idx)) return
        select type (nd => arena%entries(idx)%node)
        type is (call_or_subscript_node)
            if (.not. allocated(nd%name)) return
            candidate = nd%name
        class default
            return
        end select
        if (.not. arena_has_derived_type_named(arena, candidate)) return
        type_name = candidate
    end subroutine structure_ctor_type_name

    logical function arena_has_derived_type_named(arena, name) result(found)
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: name
        integer :: n

        found = .false.
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (nd => arena%entries(n)%node)
            type is (derived_type_node)
                if (.not. allocated(nd%name)) cycle
                if (.not. same_name(nd%name, name)) cycle
                found = .true.
                return
            end select
        end do
    end function arena_has_derived_type_named

    ! Rule 3: an array constructor assigned to a variable must carry values of
    ! a type that converts to the declared type of that variable.
    subroutine check_array_ctor_assignments(arena, error_msg)
        type(ast_arena_t), intent(in) :: arena
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: target_name, type_name
        character(len=64) :: location
        integer :: n, target_class, value_class
        logical :: found

        call set_empty(error_msg)
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (nd => arena%entries(n)%node)
            type is (assignment_node)
                call identifier_name_at(arena, nd%target_index, target_name)
                if (len_trim(target_name) == 0) cycle
                call declared_type_of_name(arena, target_name, type_name, found)
                if (.not. found) cycle
                target_class = array_ctor_typespec_class(type_name)
                if (target_class == CMP_CLASS_UNKNOWN) cycle
                value_class = array_ctor_value_class(arena, nd%value_index)
                if (value_class == CMP_CLASS_UNKNOWN) cycle
                if (value_class == target_class) cycle
                write (location, '(" at line ",I0,", column ",I0)') &
                    nd%line, nd%column
                error_msg = 'cannot convert '// &
                    array_ctor_class_upper(value_class)// &
                    ' array constructor to '//trim(adjustl(type_name))// &
                    ' variable '''//trim(target_name)//''''//trim(location)
                return
            end select
        end do
    end subroutine check_array_ctor_assignments

    ! gfortran spells the type class of a mismatched value in upper case.
    function array_ctor_class_upper(cls) result(name)
        integer, intent(in) :: cls
        character(len=:), allocatable :: name
        integer :: i

        name = cmp_class_name(cls)
        do i = 1, len(name)
            if (name(i:i) >= 'a' .and. name(i:i) <= 'z') then
                name(i:i) = achar(iachar(name(i:i)) - 32)
            end if
        end do
    end function array_ctor_class_upper

    subroutine identifier_name_at(arena, idx, name)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: idx
        character(len=:), allocatable, intent(out) :: name

        call set_empty(name)
        if (idx <= 0) return
        if (.not. node_exists(arena, idx)) return
        select type (nd => arena%entries(idx)%node)
        type is (identifier_node)
            if (allocated(nd%name)) name = nd%name
        end select
    end subroutine identifier_name_at

    ! The type class of an array constructor: its explicit type-spec when it
    ! has one, otherwise the class shared by its literal elements.
    integer function array_ctor_value_class(arena, idx) result(cls)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: idx
        integer :: i, elem_class

        cls = CMP_CLASS_UNKNOWN
        if (idx <= 0) return
        if (.not. node_exists(arena, idx)) return
        select type (nd => arena%entries(idx)%node)
        type is (array_literal_node)
            if (allocated(nd%type_spec)) then
                if (len_trim(nd%type_spec) > 0) then
                    cls = array_ctor_typespec_class(nd%type_spec)
                    return
                end if
            end if
            if (.not. allocated(nd%element_indices)) return
            do i = 1, size(nd%element_indices)
                elem_class = array_ctor_literal_class(arena, &
                                                      nd%element_indices(i))
                if (elem_class == CMP_CLASS_UNKNOWN) then
                    cls = CMP_CLASS_UNKNOWN
                    return
                end if
                if (i == 1) then
                    cls = elem_class
                else if (elem_class /= cls) then
                    cls = CMP_CLASS_UNKNOWN
                    return
                end if
            end do
        end select
    end function array_ctor_value_class

    ! Module declarations that follow a derived-type definition do not reach
    ! the typed arena, so the same two initializer rules are also enforced on
    ! the declaration statements of the source.
    subroutine check_array_ctor_init_source(arena, error_msg)
        type(ast_arena_t), intent(in) :: arena
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: source, line, type_names, name, owner
        character(len=64) :: location
        logical :: found
        integer :: pos, line_no, next_nl, kind

        call set_empty(error_msg)
        call get_source_text(arena, source, found)
        if (.not. found) return
        call collect_source_derived_types(source, type_names)
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
            call decl_init_violation(line, type_names, kind, name, owner)
            if (kind /= 0) then
                write (location, '(" at line ",I0)') line_no
                if (kind == 1) then
                    error_msg = 'cannot convert TYPE('//trim(name)// &
                        ') to an intrinsic type in the initialization of '// &
                        trim(owner)//trim(location)
                else
                    error_msg = 'incompatible ranks 0 and 1 in the '// &
                        'initialization of '//trim(owner)//trim(location)
                end if
                return
            end if
            line_no = line_no + 1
        end do
    end subroutine check_array_ctor_init_source

    ! '|name|' list of the derived types defined in the source.
    subroutine collect_source_derived_types(source, type_names)
        character(len=*), intent(in) :: source
        character(len=:), allocatable, intent(out) :: type_names
        character(len=:), allocatable :: line, name
        integer :: pos, next_nl

        type_names = '|'
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
            call source_derived_type_name(line, name)
            if (len_trim(name) > 0) type_names = type_names//trim(name)//'|'
        end do
    end subroutine collect_source_derived_types

    subroutine source_derived_type_name(line, name)
        character(len=*), intent(in) :: line
        character(len=:), allocatable, intent(out) :: name
        character(len=:), allocatable :: low, rest
        integer :: dc_pos

        call set_empty(name)
        call strip_line_comment(line, low)
        low = trim(adjustl(lowercase_text(low)))
        if (.not. starts_with_word(low, 'type')) return
        rest = trim(adjustl(low(5:)))
        if (len(rest) == 0) return
        if (rest(1:1) == '(') return
        if (starts_with_word(rest, 'is')) return
        dc_pos = index(rest, '::')
        if (dc_pos > 0) rest = trim(adjustl(rest(dc_pos + 2:)))
        name = leading_identifier(rest)
    end subroutine source_derived_type_name

    ! kind 1: a structure constructor initialises an entity of intrinsic type.
    ! kind 2: an array constructor initialises a scalar entity.
    subroutine decl_init_violation(line, type_names, kind, name, owner)
        character(len=*), intent(in) :: line, type_names
        integer, intent(out) :: kind
        character(len=:), allocatable, intent(out) :: name, owner
        character(len=:), allocatable :: low, attrs, rest, entity, init
        integer :: dc_pos, eq_pos, ident_len

        kind = 0
        call set_empty(name)
        call set_empty(owner)
        call strip_line_comment(line, low)
        low = trim(adjustl(lowercase_text(low)))
        if (.not. line_starts_intrinsic_type(low)) return
        dc_pos = index(low, '::')
        if (dc_pos == 0) return
        attrs = low(:dc_pos - 1)
        if (index(attrs, 'dimension') > 0) return
        rest = trim(adjustl(low(dc_pos + 2:)))
        eq_pos = index(rest, '=')
        if (eq_pos == 0) return
        if (eq_pos == len(rest)) return
        if (rest(eq_pos + 1:eq_pos + 1) == '>') return
        entity = trim(rest(:eq_pos - 1))
        if (len_trim(entity) == 0) return
        if (index(entity, '(') > 0) return
        if (index(entity, ',') > 0) return
        init = trim(adjustl(rest(eq_pos + 1:)))
        if (len(init) == 0) return
        owner = entity
        if (init(1:1) == '[') then
            kind = 2
            return
        end if
        if (len(init) >= 2) then
            if (init(1:2) == '(/') then
                kind = 2
                return
            end if
        end if
        name = leading_identifier(init)
        ident_len = len_trim(name)
        if (ident_len == 0) return
        if (ident_len >= len(init)) return
        if (init(ident_len + 1:ident_len + 1) /= '(') return
        if (index(type_names, '|'//trim(name)//'|') == 0) then
            call set_empty(name)
            return
        end if
        kind = 1
    end subroutine decl_init_violation

    logical function line_starts_intrinsic_type(low) result(is_type)
        character(len=*), intent(in) :: low

        is_type = starts_with_word(low, 'integer') .or. &
                  starts_with_word(low, 'real') .or. &
                  starts_with_word(low, 'logical') .or. &
                  starts_with_word(low, 'complex') .or. &
                  starts_with_word(low, 'character') .or. &
                  starts_with_word(low, 'double')
    end function line_starts_intrinsic_type
