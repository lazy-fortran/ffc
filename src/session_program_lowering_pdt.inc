    ! Parameterized derived types with constant integer KIND and LEN type
    ! parameters (#411). A PDT definition has no layout of its own: it is
    ! registered as a template, and every declaration that names it with
    ! constant actual parameters instantiates one concrete derived type whose
    ! components were laid out with the formals bound to those constants.
    ! Instances are cached under the mangled name base(v1,v2,...), so two
    ! declarations with the same actuals share one layout and two with
    ! different actuals get distinct layouts.

    logical function derived_type_is_parameterized(node) result(is_pdt)
        type(derived_type_node), intent(in) :: node

        is_pdt = .false.
        if (.not. allocated(node%param_names)) return
        is_pdt = size(node%param_names) > 0
    end function derived_type_is_parameterized

    subroutine record_pdt_template(context, name)
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: name
        character(len=64), allocatable :: tmp(:)
        integer :: slot

        if (find_pdt_template(context, name) > 0) return
        if (.not. allocated(context%pdt_template_names)) then
            allocate (context%pdt_template_names(8))
            context%pdt_template_names = ''
        else if (context%pdt_template_count >= size(context%pdt_template_names)) then
            call move_alloc(context%pdt_template_names, tmp)
            allocate (context%pdt_template_names(2*size(tmp)))
            context%pdt_template_names = ''
            context%pdt_template_names(1:size(tmp)) = tmp
        end if
        slot = context%pdt_template_count + 1
        context%pdt_template_names(slot) = trim(name)
        context%pdt_template_count = slot
    end subroutine record_pdt_template

    integer function find_pdt_template(context, name) result(slot)
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: name
        integer :: i

        slot = 0
        if (.not. allocated(context%pdt_template_names)) return
        do i = 1, context%pdt_template_count
            if (same_name(context%pdt_template_names(i), name)) then
                slot = i
                return
            end if
        end do
    end function find_pdt_template

    subroutine pdt_base_name(type_name, base)
        ! Base type name of a derived-type spec inner text, i.e. box_t for
        ! both "box_t" and "box_t(3, 8)".
        character(len=*), intent(in) :: type_name
        character(len=:), allocatable, intent(out) :: base
        integer :: paren

        paren = index(type_name, '(')
        if (paren > 0) then
            base = trim(adjustl(type_name(1:paren - 1)))
        else
            base = trim(adjustl(type_name))
        end if
    end subroutine pdt_base_name

    subroutine pdt_instance_name(base, values, name)
        ! Canonical instance name of a PDT with folded actual parameters.
        character(len=*), intent(in) :: base
        integer(c_int64_t), intent(in) :: values(:)
        character(len=:), allocatable, intent(out) :: name
        character(len=32) :: buffer
        integer :: i

        name = trim(base)//'('
        do i = 1, size(values)
            write (buffer, '(i0)') values(i)
            if (i > 1) name = name//','
            name = name//trim(buffer)
        end do
        name = name//')'
    end subroutine pdt_instance_name

    integer function pdt_template_node_index(arena, base) result(node_index)
        ! Arena index of the PDT definition named base.
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: base
        integer :: i

        node_index = 0
        do i = 1, arena%size
            if (.not. node_exists(arena, i)) cycle
            select type (node => arena%entries(i)%node)
            type is (derived_type_node)
                if (.not. allocated(node%name)) cycle
                if (.not. same_name(node%name, base)) cycle
                if (.not. derived_type_is_parameterized(node)) cycle
                node_index = i
                return
            end select
        end do
    end function pdt_template_node_index

    subroutine pdt_actual_values(arena, context, template_index, actuals, &
                                 line, column, values, error_msg)
        ! Fold the actual type parameters of one declaration to compile-time
        ! integers, filling any omitted trailing parameter from its default.
        type(ast_arena_t), intent(in) :: arena
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: template_index
        integer, intent(in) :: actuals(:)
        integer, intent(in) :: line, column
        integer(c_int64_t), allocatable, intent(out) :: values(:)
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: formal_count, i, source_index
        character(len=:), allocatable :: fold_error

        call set_empty(error_msg)
        select type (template => arena%entries(template_index)%node)
        type is (derived_type_node)
            formal_count = size(template%param_names)
            if (size(actuals) > formal_count) then
                call unsupported_feature_error('derived type parameters', &
                    line, column, 'too many actual type parameters for type "'// &
                    trim(template%name)//'"', error_msg)
                return
            end if
            allocate (values(formal_count))
            values = 0_c_int64_t
            do i = 1, formal_count
                if (i <= size(actuals)) then
                    source_index = actuals(i)
                else
                    source_index = template%param_defaults(i)
                end if
                if (source_index <= 0) then
                    call unsupported_feature_error('derived type parameters', &
                        line, column, 'type parameter "'// &
                        trim(template%param_names(i)%s)//'" of type "'// &
                        trim(template%name)//'" has no actual value and no '// &
                        'default', error_msg)
                    return
                end if
                call eval_i32_constant(arena, source_index, context, &
                                       values(i), fold_error)
                if (len_trim(fold_error) > 0) then
                    call unsupported_feature_error('derived type parameters', &
                        line, column, 'type parameter "'// &
                        trim(template%param_names(i)%s)//'" of type "'// &
                        trim(template%name)//'" must be a compile-time '// &
                        'integer constant', error_msg)
                    return
                end if
            end do
        class default
            error_msg = 'parameterized derived type template is not a type '// &
                        'definition'
        end select
    end subroutine pdt_actual_values

    subroutine pdt_declaration_instance(arena, context, node, instance_name, &
                                        found, error_msg)
        ! Instance name of the PDT a declaration names, e.g. box_t(3,4) for
        ! `type(box_t(3, 4)) :: a`. found is false when the declaration does
        ! not name a parameterized type.
        type(ast_arena_t), intent(in) :: arena
        type(lowering_context_t), intent(in) :: context
        type(declaration_node), intent(in) :: node
        character(len=:), allocatable, intent(out) :: instance_name
        logical, intent(out) :: found
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: spec_name, base
        integer(c_int64_t), allocatable :: values(:)
        integer :: template_index
        integer, allocatable :: actuals(:)

        found = .false.
        instance_name = ''
        call set_empty(error_msg)
        if (.not. allocated(node%type_name)) return
        call extracted_derived_type_name(node%type_name, spec_name)
        if (len_trim(spec_name) == 0) return
        call pdt_base_name(spec_name, base)
        if (find_pdt_template(context, base) <= 0) return
        template_index = pdt_template_node_index(arena, base)
        if (template_index <= 0) return
        if (allocated(node%type_param_indices)) then
            actuals = node%type_param_indices
        else
            allocate (actuals(0))
        end if
        call pdt_actual_values(arena, context, template_index, actuals, &
                               node%line, node%column, values, error_msg)
        if (len_trim(error_msg) > 0) return
        call pdt_instance_name(base, values, instance_name)
        found = .true.
    end subroutine pdt_declaration_instance

    subroutine instantiate_pdt_declarations(arena, context, error_msg)
        ! Instantiate a concrete layout for every distinct tuple of actual
        ! type parameters named by a declaration in the unit. Runs after the
        ! ordinary derived-type definitions are collected, so a PDT component
        ! of non-parameterized type resolves against them.
        type(ast_arena_t), intent(in) :: arena
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i

        call set_empty(error_msg)
        if (context%pdt_template_count <= 0) return
        do i = 1, arena%size
            if (.not. node_exists(arena, i)) cycle
            select type (node => arena%entries(i)%node)
            type is (declaration_node)
                call instantiate_pdt_for_declaration(arena, context, node, &
                                                     error_msg)
                if (len_trim(error_msg) > 0) return
            end select
        end do
    end subroutine instantiate_pdt_declarations

    subroutine instantiate_pdt_for_declaration(arena, context, node, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(lowering_context_t), intent(inout) :: context
        type(declaration_node), intent(in) :: node
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: spec_name, base, instance_name
        integer(c_int64_t), allocatable :: values(:)
        integer :: template_index
        integer, allocatable :: actuals(:)

        call set_empty(error_msg)
        if (.not. allocated(node%type_name)) return
        call extracted_derived_type_name(node%type_name, spec_name)
        if (len_trim(spec_name) == 0) return
        call pdt_base_name(spec_name, base)
        if (find_pdt_template(context, base) <= 0) return
        template_index = pdt_template_node_index(arena, base)
        if (template_index <= 0) return
        if (allocated(node%type_param_indices)) then
            actuals = node%type_param_indices
        else
            allocate (actuals(0))
        end if
        call pdt_actual_values(arena, context, template_index, actuals, &
                               node%line, node%column, values, error_msg)
        if (len_trim(error_msg) > 0) return
        call pdt_instance_name(base, values, instance_name)
        if (find_derived_type(context, instance_name) > 0) return
        call instantiate_pdt_layout(arena, context, template_index, values, &
                                    instance_name, error_msg)
    end subroutine instantiate_pdt_for_declaration

    subroutine instantiate_pdt_layout(arena, context, template_index, values, &
                                      instance_name, error_msg)
        ! Lay out one PDT instance: bind the formals to their actual constants
        ! so component kinds, character lengths, and array bounds fold against
        ! them, register the layout, then drop the bindings again.
        type(ast_arena_t), intent(in) :: arena
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: template_index
        integer(c_int64_t), intent(in) :: values(:)
        character(len=*), intent(in) :: instance_name
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: saved_symbol_count, i

        call set_empty(error_msg)
        saved_symbol_count = context%symbol_count
        select type (template => arena%entries(template_index)%node)
        type is (derived_type_node)
            do i = 1, size(values)
                call bind_pdt_parameter(context, trim(template%param_names(i)%s), &
                                        values(i))
            end do
            call register_derived_type_layout(template, context, instance_name, &
                                              error_msg)
        class default
            error_msg = 'parameterized derived type template is not a type '// &
                        'definition'
        end select
        call unbind_pdt_parameters(context, saved_symbol_count)
        if (len_trim(error_msg) > 0) return
        call emit_type_info_constant(context, context%derived_type_count, &
                                     error_msg)
    end subroutine instantiate_pdt_layout

    subroutine bind_pdt_parameter(context, name, value)
        ! Bind one type-parameter formal as a compile-time integer constant so
        ! specification expressions in the component declarations fold.
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: name
        integer(c_int64_t), intent(in) :: value
        integer :: index

        call grow_symbols(context)
        index = context%symbol_count + 1
        context%symbols(index)%name = trim(name)
        context%symbols(index)%value_kind = VALUE_I32
        context%symbols(index)%is_parameter = .true.
        context%symbols(index)%has_i32_constant = .true.
        context%symbols(index)%i32_constant = value
        context%symbols(index)%is_transient_i32_constant = .true.
        context%symbol_count = index
    end subroutine bind_pdt_parameter

    subroutine unbind_pdt_parameters(context, saved_symbol_count)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: saved_symbol_count
        integer :: i

        do i = saved_symbol_count + 1, context%symbol_count
            context%symbols(i)%name = ''
            context%symbols(i)%has_i32_constant = .false.
            context%symbols(i)%is_transient_i32_constant = .false.
            context%symbols(i)%is_parameter = .false.
        end do
        context%symbol_count = saved_symbol_count
    end subroutine unbind_pdt_parameters
