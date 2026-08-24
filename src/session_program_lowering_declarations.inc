    subroutine collect_resolved_declarations(arena, root_index, context, &
                                             error_msg)
        ! Build the declaration registry before any executable body is emitted.
        ! FortFront owns the name-to-entity decision; this pass records the
        ! binding returned at each declaration site, including a body
        ! redeclaration that FortFront maps to a dummy or result identity. The
        ! registry is metadata-only because storage operands can be created only
        ! after a procedure's LIRIC function is active (#457).
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: root_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i

        call set_empty(error_msg)
        context%declaration_record_count = 0
        context%declaration_collection_complete = .false.
        if (.not. allocated(context%declaration_records)) then
            allocate(context%declaration_records(64))
        end if
        if (.not. node_exists(arena, root_index)) then
            error_msg = 'declaration collection root is not an AST node'
            return
        end if

        do i = 1, arena%size
            if (.not. node_exists(arena, i)) cycle
            select type (node => arena%entries(i)%node)
            type is (declaration_node)
                if (.not. declaration_is_collectable(arena, i, node)) cycle
                call collect_declaration_node(arena, i, node, context, error_msg)
                if (len_trim(error_msg) > 0) return
            type is (parameter_declaration_node)
                call collect_parameter_declaration_node(arena, i, node, context, &
                                                        error_msg)
                if (len_trim(error_msg) > 0) return
            type is (function_def_node)
                call collect_function_result_declaration(arena, i, node, &
                                                         context, error_msg)
                if (len_trim(error_msg) > 0) return
            end select
        end do
        context%declaration_collection_complete = .true.
    end subroutine collect_resolved_declarations

    logical function declaration_is_collectable(arena, node_index, node) &
        result(is_collectable)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(declaration_node), intent(in) :: node
        integer :: parent_index

        is_collectable = .false.
        if (node_index <= 0) return
        ! FortFront-inserted declarations are legacy inferred symbols. They
        ! intentionally stay on the compatibility path until their lowering
        ! storage is migrated to binding identity (#332).
        if (node%is_inferred) return
        ! DIMENSION contributes shape to a later typed declaration, and EXTERNAL
        ! contributes a procedure name. Neither introduces scalar/array storage.
        if (declaration_is_bare_dimension(node)) return
        if (node%is_external) return
        ! A PROCEDURE declaration (procedure(iface) :: p) names a procedure with
        ! an explicit interface; it introduces no scalar/array storage (#364).
        if (declaration_names_procedure(node) .and. .not. node%is_pointer) return
        ! Derived-type components are layout metadata, not procedure-local
        ! entities. Their parent is not a lexical scope for name resolution.
        parent_index = arena%entries(node_index)%parent_index
        if (parent_index > 0 .and. node_exists(arena, parent_index)) then
            select type (parent => arena%entries(parent_index)%node)
            type is (derived_type_node)
                return
            end select
        end if
        if (node%is_multi_declaration) then
            is_collectable = allocated(node%var_names) .and. &
                             size(node%var_names) > 0
        else
            is_collectable = allocated(node%var_name) .and. &
                             len_trim(node%var_name) > 0
        end if
    end function declaration_is_collectable

    subroutine collect_declaration_node(arena, node_index, node, context, &
                                         error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(declaration_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i

        call set_empty(error_msg)
        if (node%is_multi_declaration .and. allocated(node%var_names)) then
            do i = 1, size(node%var_names)
                call collect_one_declaration_entity(arena, node_index, node, &
                    trim(node%var_names(i)), context, error_msg)
                if (len_trim(error_msg) > 0) return
            end do
        else if (allocated(node%var_name)) then
            call collect_one_declaration_entity(arena, node_index, node, &
                trim(node%var_name), context, error_msg)
        end if
    end subroutine collect_declaration_node

    subroutine collect_one_declaration_entity(arena, node_index, node, name, &
                                               context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(declaration_node), intent(in) :: node
        character(len=*), intent(in) :: name
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(declaration_binding_t) :: binding
        type(declaration_record_t) :: record
        character(len=:), allocatable :: resolve_error

        call set_empty(error_msg)
        if (len_trim(name) == 0) return
        call resolve_name_at_node(arena, node_index, name, binding, resolve_error)
        if (len_trim(resolve_error) > 0) return
        if (.not. binding%found) return
        if (binding%scope_node_index <= 0 .or. &
            (binding%binding_kind /= BINDING_DECLARATION .and. &
             binding%binding_kind /= BINDING_NAMED_CONSTANT .and. &
             binding%binding_kind /= BINDING_DUMMY_ARGUMENT .and. &
             binding%binding_kind /= BINDING_FUNCTION_RESULT)) then
            return
        end if
        if (allocated(node%type_name)) then
            block
                integer :: declared_kind
                call type_name_value_kind(node%type_name, node%line, node%column, &
                    declared_kind, error_msg, context, node_index)
                if (len_trim(error_msg) > 0) return
            end block
        end if
        call make_declaration_record(node, name, binding, context, record)
        call store_declaration_record(context, record, error_msg)
    end subroutine collect_one_declaration_entity

    subroutine collect_parameter_declaration_node(arena, node_index, node, &
                                                  context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(parameter_declaration_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(declaration_binding_t) :: binding
        type(declaration_record_t) :: record
        character(len=:), allocatable :: resolve_error

        call set_empty(error_msg)
        if (.not. allocated(node%name)) return
        if (len_trim(node%name) == 0) return
        call resolve_name_at_node(arena, node_index, trim(node%name), binding, &
                                  resolve_error)
        if (len_trim(resolve_error) > 0) return
        if (.not. binding%found) return
        if (binding%scope_node_index <= 0 .or. &
            (binding%binding_kind /= BINDING_DUMMY_ARGUMENT .and. &
             binding%binding_kind /= BINDING_DECLARATION)) then
            return
        end if
        call make_parameter_record(node, trim(node%name), binding, context, record)
        call store_declaration_record(context, record, error_msg)
    end subroutine collect_parameter_declaration_node

    subroutine collect_function_result_declaration(arena, node_index, node, &
                                                   context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(function_def_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(declaration_binding_t), allocatable :: bindings(:)
        type(declaration_binding_t) :: result_binding
        type(declaration_record_t) :: record
        character(len=:), allocatable :: resolve_error
        integer :: i

        call set_empty(error_msg)
        call get_scope_bindings(arena, node_index, bindings, resolve_error)
        if (len_trim(resolve_error) > 0) then
            error_msg = resolve_error
            return
        end if
        do i = 1, size(bindings)
            if (bindings(i)%binding_kind /= BINDING_FUNCTION_RESULT) cycle
            result_binding = bindings(i)
            call make_function_result_record(arena, node, result_binding, &
                                              context, record)
            call store_declaration_record(context, record, error_msg)
            if (len_trim(error_msg) > 0) return
        end do
    end subroutine collect_function_result_declaration

    subroutine make_declaration_record(node, name, binding, context, record)
        type(declaration_node), intent(in) :: node
        character(len=*), intent(in) :: name
        type(declaration_binding_t), intent(in) :: binding
        type(lowering_context_t), intent(in) :: context
        type(declaration_record_t), intent(out) :: record

        record = declaration_record_t()
        record%declaration_node_index = binding%declaration_node_index
        record%declaration_entity_index = binding%declaration_entity_index
        record%scope_node_index = binding%scope_node_index
        record%name = trim(name)
        if (allocated(node%type_name)) record%type_name = trim(node%type_name)
        record%type_kind = node%resolved_type_kind
        record%kind_value = node%resolved_kind_value
        record%rank = node%resolved_rank
        if (record%rank <= 0 .and. allocated(node%dimension_indices)) then
            record%rank = size(node%dimension_indices)
        end if
        record%is_array = node%is_array .or. record%rank > 0
        record%is_parameter = node%is_parameter .and. &
            binding%binding_kind == BINDING_NAMED_CONSTANT
        call apply_lazy_real_kind(context, record%type_kind, node%has_kind, &
                                  record%kind_value)
        record%value_kind = declaration_metadata_value_kind( &
            record%type_kind, record%kind_value, record%type_name, context, &
            node%line, node%column, node%resolved_derived_type_name)
    end subroutine make_declaration_record

    subroutine make_parameter_record(node, name, binding, context, record)
        type(parameter_declaration_node), intent(in) :: node
        character(len=*), intent(in) :: name
        type(declaration_binding_t), intent(in) :: binding
        type(lowering_context_t), intent(in) :: context
        type(declaration_record_t), intent(out) :: record

        record = declaration_record_t()
        record%declaration_node_index = binding%declaration_node_index
        record%declaration_entity_index = binding%declaration_entity_index
        record%scope_node_index = binding%scope_node_index
        record%name = trim(name)
        if (allocated(node%type_name)) record%type_name = trim(node%type_name)
        record%type_kind = node%resolved_type_kind
        record%kind_value = node%resolved_kind_value
        record%rank = node%resolved_rank
        if (record%rank <= 0 .and. allocated(node%dimension_indices)) then
            record%rank = size(node%dimension_indices)
        end if
        ! A parameter-list entry has no shape until its specification-part
        ! declaration is visited; rank zero here means "unknown", not scalar.
        if (.not. node%is_array .and. record%rank == 0) record%rank = -1
        record%is_array = node%is_array .or. record%rank > 0
        call apply_lazy_real_kind(context, record%type_kind, node%has_kind, &
                                  record%kind_value)
        record%value_kind = declaration_metadata_value_kind( &
            record%type_kind, record%kind_value, record%type_name, context)
    end subroutine make_parameter_record

    subroutine make_function_result_record(arena, node, binding, context, record)
        type(ast_arena_t), intent(in) :: arena
        type(function_def_node), intent(in) :: node
        type(declaration_binding_t), intent(in) :: binding
        type(lowering_context_t), intent(in) :: context
        type(declaration_record_t), intent(out) :: record
        integer :: value_kind, derived_index
        integer :: i, j
        character(len=:), allocatable :: kind_error
        logical :: found

        record = declaration_record_t()
        record%declaration_node_index = binding%declaration_node_index
        record%declaration_entity_index = binding%declaration_entity_index
        record%scope_node_index = binding%scope_node_index
        record%name = trim(binding%name)
        if (allocated(node%return_type)) record%type_name = trim(node%return_type)
        record%type_kind = node%resolved_type_kind
        record%kind_value = node%resolved_kind_value
        record%rank = 0
        record%is_array = .false.

        if (allocated(node%body_indices)) then
            do i = 1, size(node%body_indices)
                if (.not. node_exists(arena, node%body_indices(i))) cycle
                select type (decl => arena%entries(node%body_indices(i))%node)
                type is (declaration_node)
                    found = .false.
                    if (decl%is_multi_declaration .and. &
                        allocated(decl%var_names)) then
                        do j = 1, size(decl%var_names)
                            if (same_name(decl%var_names(j), record%name)) then
                                found = .true.
                                exit
                            end if
                        end do
                    else if (allocated(decl%var_name)) then
                        found = same_name(decl%var_name, record%name)
                    end if
                    if (.not. found) cycle
                    record%type_name = ''
                    if (allocated(decl%type_name)) &
                        record%type_name = trim(decl%type_name)
                    record%type_kind = decl%resolved_type_kind
                    record%kind_value = decl%resolved_kind_value
                    record%rank = decl%resolved_rank
                    if (record%rank < 0 .and. allocated(decl%dimension_indices)) &
                        record%rank = size(decl%dimension_indices)
                    record%is_array = decl%is_array .or. record%rank > 0
                    call apply_lazy_real_kind(context, record%type_kind, &
                                              decl%has_kind, record%kind_value)
                    exit
                end select
            end do
        end if

        value_kind = 0
        derived_index = 0
        call set_empty(kind_error)
        call function_value_kind(node, context, value_kind, derived_index, &
                                 kind_error, binding%scope_node_index)
        if (value_kind > 0) record%value_kind = value_kind
        if (record%type_kind == 0) then
            record%type_kind = value_kind_type_kind(record%value_kind)
        end if
        if (record%value_kind == VALUE_ARRAY_RESULT .or. &
            record%value_kind == VALUE_ALLOC_ARRAY_RESULT) record%is_array = .true.
    end subroutine make_function_result_record

    integer function declaration_metadata_value_kind(type_kind, kind_value, &
        type_name, context, line, column, derived_name) result(value_kind)
        integer, intent(in) :: type_kind
        integer, intent(in) :: kind_value
        character(len=*), intent(in) :: type_name
        type(lowering_context_t), intent(in), optional :: context
        integer, intent(in), optional :: line, column
        character(len=*), intent(in), optional :: derived_name
        character(len=:), allocatable :: kind_error
        integer :: candidate, source_line, source_column

        value_kind = 0
        source_line = 1
        source_column = 1
        if (present(line)) source_line = line
        if (present(column)) source_column = column
        select case (type_kind)
        case (TINT)
            select case (kind_value)
            case (1)
                value_kind = VALUE_I8
            case (2)
                value_kind = VALUE_I16
            case (8)
                value_kind = VALUE_I64
            case default
                value_kind = VALUE_I32
            end select
        case (TREAL)
            if (kind_value == 8) then
                value_kind = VALUE_F64
            else if (len_trim(type_name) > 0) then
                ! FortFront preserves a nested selector such as
                ! real(selected_real_kind(15, 307)) in the public type text,
                ! while its resolved metadata may still carry the default kind.
                ! Reconcile that text before classifying the storage ABI.
                candidate = VALUE_F32
                call set_empty(kind_error)
                if (present(context)) then
                    call type_name_value_kind(type_name, source_line, &
                        source_column, candidate, kind_error, context)
                else
                    call type_name_value_kind(type_name, source_line, &
                        source_column, candidate, kind_error)
                end if
                if (len_trim(kind_error) == 0) then
                    value_kind = candidate
                else
                    value_kind = VALUE_F32
                end if
            else
                value_kind = VALUE_F32
            end if
        case (TDOUBLE)
            value_kind = VALUE_F64
        case (TLOGICAL)
            value_kind = VALUE_LOGICAL
        case (TCHAR)
            value_kind = VALUE_CHARACTER
        case (TCOMPLEX)
            if (kind_value == 8) then
                value_kind = VALUE_C8
            else
                value_kind = VALUE_C4
            end if
        case (TDERIVED)
            value_kind = VALUE_DERIVED
        case default
            if (present(derived_name)) then
                if (len_trim(derived_name) > 0) then
                    value_kind = VALUE_DERIVED
                    return
                end if
            end if
            if (len_trim(type_name) > 0) then
                candidate = VALUE_I32
                call set_empty(kind_error)
                if (present(context)) then
                    call type_name_value_kind(type_name, source_line, &
                        source_column, candidate, kind_error, context)
                else
                    call type_name_value_kind(type_name, source_line, &
                        source_column, candidate, kind_error)
                end if
                if (len_trim(kind_error) == 0) value_kind = candidate
            end if
        end select
    end function declaration_metadata_value_kind

    integer function value_kind_type_kind(value_kind) result(type_kind)
        integer, intent(in) :: value_kind

        select case (value_kind)
        case (VALUE_F32, VALUE_F64)
            type_kind = TREAL
        case (VALUE_LOGICAL)
            type_kind = TLOGICAL
        case (VALUE_CHARACTER)
            type_kind = TCHAR
        case (VALUE_C4, VALUE_C8)
            type_kind = TCOMPLEX
        case (VALUE_DERIVED)
            type_kind = TDERIVED
        case default
            type_kind = TINT
        end select
    end function value_kind_type_kind

    subroutine store_declaration_record(context, record, error_msg)
        type(lowering_context_t), intent(inout) :: context
        type(declaration_record_t), intent(in) :: record
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i

        call set_empty(error_msg)
        if (record%declaration_node_index <= 0 .or. &
            record%scope_node_index <= 0) return
        do i = 1, context%declaration_record_count
            if (.not. same_declaration_identity( &
                context%declaration_records(i), record)) cycle
            if (.not. same_declaration_metadata( &
                context%declaration_records(i), record)) then
                error_msg = 'conflicting declaration metadata for '// &
                            trim(record%name)
            else
                call merge_declaration_metadata(context%declaration_records(i), &
                                                record)
            end if
            return
        end do
        if (context%declaration_record_count >= &
            size(context%declaration_records)) then
            call grow_declaration_records(context)
        end if
        context%declaration_record_count = context%declaration_record_count + 1
        context%declaration_records(context%declaration_record_count) = record
    end subroutine store_declaration_record

    subroutine grow_declaration_records(context)
        type(lowering_context_t), intent(inout) :: context
        type(declaration_record_t), allocatable :: tmp(:)
        integer :: old_size, new_size

        old_size = 0
        if (allocated(context%declaration_records)) &
            old_size = size(context%declaration_records)
        new_size = max(64, 2 * old_size)
        allocate(tmp(new_size))
        if (context%declaration_record_count > 0) then
            tmp(1:context%declaration_record_count) = &
                context%declaration_records(1:context%declaration_record_count)
        end if
        call move_alloc(tmp, context%declaration_records)
    end subroutine grow_declaration_records

    logical function same_declaration_identity(lhs, rhs) result(same)
        type(declaration_record_t), intent(in) :: lhs
        type(declaration_record_t), intent(in) :: rhs

        same = lhs%declaration_node_index == rhs%declaration_node_index .and. &
               lhs%declaration_entity_index == rhs%declaration_entity_index .and. &
               lhs%scope_node_index == rhs%scope_node_index
    end function same_declaration_identity

    logical function same_declaration_metadata(lhs, rhs) result(same)
        type(declaration_record_t), intent(in) :: lhs
        type(declaration_record_t), intent(in) :: rhs

        same = .true.
        if (lhs%type_kind > 0 .and. rhs%type_kind > 0) &
            same = same .and. lhs%type_kind == rhs%type_kind
        if (lhs%kind_value > 0 .and. rhs%kind_value > 0) &
            same = same .and. lhs%kind_value == rhs%kind_value
        if (lhs%value_kind > 0 .and. rhs%value_kind > 0) &
            same = same .and. lhs%value_kind == rhs%value_kind
        if (lhs%rank >= 0 .and. rhs%rank >= 0) then
            same = same .and. lhs%rank == rhs%rank
            same = same .and. (lhs%is_array .eqv. rhs%is_array)
        end if
        same = same .and. (lhs%is_parameter .eqv. rhs%is_parameter)
    end function same_declaration_metadata

    subroutine merge_declaration_metadata(destination, source)
        type(declaration_record_t), intent(inout) :: destination
        type(declaration_record_t), intent(in) :: source

        if (destination%type_kind <= 0) destination%type_kind = source%type_kind
        if (destination%kind_value <= 0) destination%kind_value = source%kind_value
        if (destination%value_kind <= 0) destination%value_kind = source%value_kind
        if (destination%rank < 0 .and. source%rank >= 0) then
            destination%rank = source%rank
            destination%is_array = source%is_array
        end if
        if (len_trim(destination%type_name) == 0) &
            destination%type_name = source%type_name
    end subroutine merge_declaration_metadata

    integer function find_declaration_record(context, binding) result(index)
        type(lowering_context_t), intent(in) :: context
        type(declaration_binding_t), intent(in) :: binding
        integer :: i

        index = 0
        if (.not. binding%found) return
        do i = context%declaration_record_count, 1, -1
            if (context%declaration_records(i)%declaration_node_index /= &
                binding%declaration_node_index) cycle
            if (context%declaration_records(i)%declaration_entity_index /= &
                binding%declaration_entity_index) cycle
            if (context%declaration_records(i)%scope_node_index /= &
                binding%scope_node_index) cycle
            index = i
            return
        end do
    end function find_declaration_record

    subroutine lower_constant_declaration(node, context, value_kind, error_msg)
        type(declaration_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: value_kind
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: name
        integer(c_int64_t) :: constant_value
        integer :: index

        if (node%is_array) then
            call unsupported_feature_error('array parameter declaration', &
                                           node%line, node%column, &
                                           'array parameters are not supported '// &
                                           'by direct LIRIC session', error_msg)
            return
        end if
        if (node%is_multi_declaration) then
            call unsupported_feature_error('parameter declaration', &
                                           node%line, node%column, &
                                           'direct LIRIC session only supports '// &
                                           'single scalar parameters', error_msg)
            return
        end if
        if (.not. allocated(node%var_name)) then
            error_msg = 'parameter declaration did not expose a variable name'
            return
        end if
        if (.not. node%has_initializer .or. node%initializer_index <= 0) then
            error_msg = 'parameter requires a compile-time initializer: '// &
                        trim(node%var_name)
            return
        end if

        ! Real/logical/character named constants bind a typed value into the
        ! symbol; integer scalars stay on the dedicated path because the
        ! derived-type bound pre-pass caches their folded i32 constant.
        if (value_kind /= VALUE_I32) then
            call lower_nonint_constant_declaration(node, context, value_kind, &
                                                   error_msg)
            return
        end if

        name = node%var_name
        index = find_symbol_compat(context, name)
        if (index > 0) then
            ! Already collected by the parameter pre-pass (used for derived-type
            ! array component bounds). Re-declaring the same integer parameter is
            ! a no-op; a genuine clash is a different symbol kind.
            if (context%symbols(index)%is_parameter .and. &
                context%symbols(index)%has_i32_constant) then
                call set_empty(error_msg)
                return
            end if
            error_msg = 'duplicate parameter declaration: '//trim(name)
            return
        end if
        call grow_symbols(context)

        call eval_i32_constant(context%arena, node%initializer_index, context, &
                               constant_value, error_msg)
        if (len_trim(error_msg) > 0) return

        index = context%symbol_count + 1
        context%symbols(index)%name = trim(name)
        context%symbols(index)%value_kind = VALUE_I32
        context%symbols(index)%value = i32_immediate(context%session, constant_value)
        context%symbols(index)%is_parameter = .true.
        context%symbols(index)%has_i32_constant = .true.
        context%symbols(index)%i32_constant = constant_value
        context%symbol_count = index
        call set_empty(error_msg)
    end subroutine lower_constant_declaration

    subroutine lower_nonint_constant_declaration(node, context, value_kind, &
                                                 error_msg)
        ! Bind a scalar REAL/LOGICAL/CHARACTER (or non-default integer) named
        ! constant: define a symbol of the declared kind, fold the literal
        ! initializer into its value operand, and mark it a parameter so later
        ! references read the constant directly.
        type(declaration_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: value_kind
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: name
        integer :: index

        name = node%var_name
        if (find_symbol_compat(context, name) > 0) then
            error_msg = 'duplicate parameter declaration: '//trim(name)
            return
        end if

        if (value_kind == VALUE_CHARACTER) then
            call lower_character_constant_declaration(node, context, name, &
                                                      error_msg)
            return
        end if

        call define_symbol(context, name, value_kind, error_msg)
        if (len_trim(error_msg) > 0) return
        index = find_symbol_compat(context, name)
        if (index <= 0) then
            error_msg = 'parameter declaration did not register a symbol: '// &
                        trim(name)
            return
        end if
        context%symbols(index)%is_parameter = .true.
        ! A non-default-kind integer parameter (integer(int64) etc.) folds
        ! directly to a compile-time value when eval_i32_constant reaches it
        ! (e.g. a bit-intrinsic call such as ibits() that the general scalar
        ! initializer lowerer does not know how to emit as runtime code);
        ! record that value and skip the runtime lowering path entirely.
        ! Otherwise fall back to lowering the initializer as executable code.
        if (value_kind == VALUE_I8 .or. value_kind == VALUE_I16 .or. &
            value_kind == VALUE_I64) then
            block
                integer(c_int64_t) :: int_const
                character(len=:), allocatable :: fold_err

                call eval_i32_constant(context%arena, node%initializer_index, &
                                       context, int_const, fold_err)
                if (len_trim(fold_err) == 0) then
                    context%symbols(index)%has_i32_constant = .true.
                    context%symbols(index)%i32_constant = int_const
                    select case (value_kind)
                    case (VALUE_I8)
                        context%symbols(index)%value = &
                            i8_immediate(context%session, int_const)
                    case (VALUE_I16)
                        context%symbols(index)%value = &
                            i16_immediate(context%session, int_const)
                    case (VALUE_I64)
                        context%symbols(index)%value = &
                            i64_immediate(context%session, int_const)
                    end select
                    call set_empty(error_msg)
                    return
                end if
            end block
        end if
        call lower_scalar_initializer(context, name, value_kind, &
                                      node%initializer_index, error_msg)
    end subroutine lower_nonint_constant_declaration

    subroutine lower_character_constant_declaration(node, context, name, error_msg)
        ! A character named constant with an assumed length
        ! (character(len=*), parameter :: s = "Hello") takes its length from
        ! the folded initializer text. A declared length
        ! (character(len=N), parameter :: s = "Hello") instead pads or
        ! truncates the folded text to N, matching gfortran's fixed-length
        ! assignment. The initializer may reference an earlier character
        ! named constant (z = x // y), not just literals. The value operand
        ! holds a materialized LIRIC string so references and print read it
        ! directly.
        type(declaration_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: name
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: literal_text
        character(len=64) :: string_name
        logical :: fold_ok
        integer :: index, declared_length

        call fold_char_constant_expr(context, node%initializer_index, &
                                     literal_text, fold_ok)
        if (.not. fold_ok) then
            call unsupported_feature_error('character parameter declaration', &
                node%line, node%column, 'only character-literal initializers '// &
                'are supported for named constants', error_msg)
            return
        end if

        call declaration_character_length(context, node, declared_length, &
                                          error_msg)
        if (len_trim(error_msg) > 0) return
        if (declared_length > 0) then
            call normalize_character_literal(literal_text, declared_length)
        end if

        call define_character_symbol(context, name, len(literal_text), error_msg)
        if (len_trim(error_msg) > 0) return
        index = find_symbol_compat(context, name)
        if (index <= 0) then
            error_msg = 'character parameter did not register a symbol: '// &
                        trim(name)
            return
        end if

        context%string_literal_count = context%string_literal_count + 1
        string_name = ffc_unit_global_name( &
            context, 'char.', context%string_literal_count)
        call materialize_liric_string(context%session, trim(string_name), &
                                      literal_text, &
                                      context%symbols(index)%value, error_msg)
        if (len_trim(error_msg) > 0) return
        context%symbols(index)%has_character_value = .true.
        context%symbols(index)%is_parameter = .true.
        context%symbols(index)%character_constant_text = literal_text
        call set_empty(error_msg)
    end subroutine lower_character_constant_declaration

    recursive subroutine fold_char_constant_expr(context, node_index, &
                                                 accumulated, ok)
        ! Fold a character-parameter initializer (a literal, a // chain of
        ! literals and/or earlier character named constants) into its
        ! compile-time text.
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: node_index
        character(len=:), allocatable, intent(out) :: accumulated
        logical, intent(out) :: ok
        character(len=:), allocatable :: left_text, right_text
        character(len=:), allocatable :: bin_op, bin_err
        integer :: bin_left, bin_right, bin_line, bin_col
        character(len=:), allocatable :: id_name, id_err
        integer :: symbol_index

        ok = .false.
        if (.not. node_exists(context%arena, node_index)) return

        if (is_binary_op(context%arena, node_index)) then
            call get_binary_op_info(context%arena, node_index, bin_op, &
                                    bin_left, bin_right, bin_line, bin_col, &
                                    bin_err)
            if (len_trim(bin_err) > 0) return
            if (trim(bin_op) /= '//') return
            call fold_char_constant_expr(context, bin_left, left_text, ok)
            if (.not. ok) return
            call fold_char_constant_expr(context, bin_right, right_text, ok)
            if (.not. ok) return
            accumulated = left_text//right_text
            ok = .true.
            return
        end if

        if (is_identifier(context%arena, node_index)) then
            call get_identifier_name(context%arena, node_index, id_name, &
                                     id_err)
            if (len_trim(id_err) > 0) return
            symbol_index = find_symbol_compat(context, id_name)
            if (symbol_index <= 0) return
            if (.not. &
                allocated(context%symbols(symbol_index)%character_constant_text)) &
                return
            accumulated = context%symbols(symbol_index)%character_constant_text
            ok = .true.
            return
        end if

        call concat_character_literals(context%arena, node_index, &
                                       accumulated, ok)
    end subroutine fold_char_constant_expr

    recursive subroutine type_name_value_kind(type_name, line, column, &
                                              value_kind, error_msg, context, &
                                              reference_index)
        character(len=*), intent(in) :: type_name
        integer, intent(in) :: line, column
        integer, intent(out) :: value_kind
        character(len=:), allocatable, intent(out) :: error_msg
        type(lowering_context_t), intent(in), optional :: context
        integer, intent(in), optional :: reference_index
        character(len=:), allocatable :: base
        character(len=:), allocatable :: kind_spec
        integer :: paren

        value_kind = VALUE_I32
        call set_empty(error_msg)
        if (is_character_type_name(type_name)) then
            value_kind = VALUE_CHARACTER
            return
        end if

        ! Split "integer(kind)" into the base type and the kind spec, which may
        ! be a numeric literal (4, 8) or an iso_c_binding kind name (c_int, ...).
        paren = index(type_name, '(')
        if (paren > 0) then
            base = lowercase_text(trim(adjustl(type_name(1:paren - 1))))
            kind_spec = lowercase_text(trim(adjustl( &
                type_name(paren + 1:index(type_name, ')', back=.true.) - 1))))
            ! Normalize the explicit "kind=" keyword form so kind-name matching
            ! below sees the bare selector (kind=8 -> 8, real(kind=8) -> 8).
            if (index(kind_spec, 'kind=') == 1) then
                kind_spec = trim(adjustl(kind_spec(6:)))
            end if
        else
            base = lowercase_text(trim(adjustl(type_name)))
            call set_empty(kind_spec)
        end if

        select case (base)
        case ('integer')
            call integer_kind_to_value_kind(kind_spec, line, column, &
                                            value_kind, error_msg, context, &
                                            reference_index)
        case ('real')
            call check_float_kind_supported('real', kind_spec, line, column, &
                                            error_msg, context, reference_index)
            if (len_trim(error_msg) > 0) return
            if (real_kind_is_f64(kind_spec, context, reference_index)) then
                value_kind = VALUE_F64
            else if (len_trim(kind_spec) == 0 .and. &
                     lazy_defaults_active(context)) then
                ! Lazy Fortran default real is 8 bytes (#438), so a kind-less
                ! `real` lowers exactly like the explicit `real(8)` a standard
                ! source would have to spell out.
                value_kind = VALUE_F64
            else
                value_kind = VALUE_F32
            end if
        case ('complex')
            call check_float_kind_supported('complex', kind_spec, line, &
                                            column, error_msg, context, &
                                            reference_index)
            if (len_trim(error_msg) > 0) return
            if (complex_kind_is_c8(kind_spec, context, reference_index)) then
                value_kind = VALUE_C8
            else
                value_kind = VALUE_C4
            end if
        case ('double precision')
            value_kind = VALUE_F64
        case ('logical')
            value_kind = VALUE_LOGICAL
        case ('procedure')
            ! PROCEDURE(interface), POINTER entities use the existing indirect
            ! call slot; the interface body supplies call compatibility.
            value_kind = VALUE_PROC_PTR
        case ('type')
            ! TYPE(*) is unlimited polymorphic, with the same descriptor ABI as
            ! CLASS(*).  FortFront represents the selector as the TYPE base
            ! here, so classify it before handling c_ptr and intrinsic specs.
            if (trim(kind_spec) == '*') then
                value_kind = VALUE_CLASS_STAR
            ! c_ptr is an opaque 8-byte pointer value; c_loc/c_associated/
            ! c_f_pointer operate on it (#139).
            else if (trim(kind_spec) == 'c_ptr' .or. &
                     trim(kind_spec) == 'c_funptr') then
                value_kind = VALUE_C_PTR
            else if (is_intrinsic_type_spec(kind_spec)) then
                ! F2008 TYPE(intrinsic-type-spec): type(real), type(integer),
                ! type(character(len=4)) name the intrinsic type itself, not a
                ! derived type, so lower them exactly like the bare intrinsic.
                call type_name_value_kind(kind_spec, line, column, value_kind, &
                                          error_msg, context, reference_index)
            else if (present(context)) then
                if (find_derived_type(context, kind_spec) > 0) then
                    value_kind = VALUE_DERIVED
                else
                    call unsupported_feature_error('scalar type', line, column, &
                        'direct LIRIC session only supports integer, real, '// &
                        'logical, character, and c_ptr scalars', error_msg)
                end if
            else
                call unsupported_feature_error('scalar type', line, column, &
                    'direct LIRIC session only supports integer, real, '// &
                    'logical, character, and c_ptr scalars', error_msg)
            end if
        case ('class')
            ! Unlimited polymorphic class(*) has its descriptor ABI. A named
            ! class(t) is a derived value and uses that descriptor path. This
            ! classifier also runs before the derived-type table is populated,
            ! so name lookup cannot be required here (#417).
            if (trim(kind_spec) == '*') then
                value_kind = VALUE_CLASS_STAR
            else
                value_kind = VALUE_DERIVED
            end if
        case default
            if (present(context)) then
                if (find_derived_type(context, base) > 0) then
                    value_kind = VALUE_DERIVED
                    return
                end if
            end if
            call unsupported_feature_error('scalar type', line, column, &
                                           'direct LIRIC session only supports '// &
                                           'integer, real, logical, and character '// &
                                           'scalars', error_msg)
        end select
    end subroutine type_name_value_kind

    ! True when text is an intrinsic type-spec (with an optional kind/len
    ! selector), e.g. real, integer, real(kind=4), character(len=4),
    ! double precision. Used to recognise the F2008 TYPE(intrinsic-type-spec)
    ! spelling, which names the intrinsic type rather than a derived type.
    logical function is_intrinsic_type_spec(text)
        character(len=*), intent(in) :: text
        character(len=:), allocatable :: base
        integer :: paren

        paren = index(text, '(')
        if (paren > 0) then
            base = lowercase_text(trim(adjustl(text(1:paren - 1))))
        else
            base = lowercase_text(trim(adjustl(text)))
        end if
        select case (base)
        case ('integer', 'real', 'logical', 'complex', 'character', &
              'double precision', 'doubleprecision')
            is_intrinsic_type_spec = .true.
        case default
            is_intrinsic_type_spec = .false.
        end select
    end function is_intrinsic_type_spec

    subroutine integer_kind_to_value_kind(kind_spec, line, column, &
                                          value_kind, error_msg, context, &
                                          reference_index)
        character(len=*), intent(in) :: kind_spec
        integer, intent(in) :: line, column
        integer, intent(out) :: value_kind
        character(len=:), allocatable, intent(out) :: error_msg
        type(lowering_context_t), intent(in), optional :: context
        integer, intent(in), optional :: reference_index
        logical :: resolved
        integer(c_int64_t) :: resolved_value
        character(len=32) :: resolved_text

        call set_empty(error_msg)
        select case (trim(kind_spec))
        case ('1', 'int8', 'c_int8_t', 'c_signed_char', &
              'c_int_least8_t', 'c_int_fast8_t')
            value_kind = VALUE_I8
            return
        case ('2', 'int16', 'c_int16_t', 'c_short', &
              'c_int_least16_t', 'c_int_fast16_t')
            value_kind = VALUE_I16
            return
        case ('', '4', 'c_int', 'c_int32_t', 'c_bool', 'c_char', 'int32', &
              'c_int_least32_t', 'c_int_fast32_t')
            value_kind = VALUE_I32
            return
        case ('8', 'int64', 'c_int64_t', 'c_long', 'c_long_long', &
              'c_int_least64_t', 'c_int_fast64_t', &
              'c_size_t', 'c_intptr_t', 'c_ptrdiff_t', 'c_intmax_t')
            ! LP64 Linux: size_t/intptr_t/ptrdiff_t/intmax_t are 8 bytes,
            ! matching the liric session's native 64-bit target.
            value_kind = VALUE_I64
            return
        end select

        ! Not a recognised literal or iso_c_binding name: try resolving it as a
        ! declared "integer, parameter" kind name (integer(prec) :: v).
        resolved = .false.
        if (present(context)) then
            call resolve_kind_parameter_value(context, kind_spec, resolved, &
                                              resolved_value, reference_index)
        end if
        if (resolved) then
            select case (resolved_value)
            case (1_c_int64_t)
                value_kind = VALUE_I8
                return
            case (2_c_int64_t)
                value_kind = VALUE_I16
                return
            case (4_c_int64_t)
                value_kind = VALUE_I32
                return
            case (8_c_int64_t)
                value_kind = VALUE_I64
                return
            end select
        end if

        value_kind = VALUE_I32
        if (resolved) then
            write (resolved_text, '(I0)') resolved_value
            call unsupported_feature_error('integer kind', line, column, &
                'direct LIRIC session supports integer kinds 1 (int8), '// &
                '2 (int16), 4 (default), and 8 (int64); kind '// &
                trim(kind_spec)//' resolves to unsupported byte size '// &
                trim(adjustl(resolved_text)), error_msg)
        else
            call unsupported_feature_error('integer kind', line, column, &
                'direct LIRIC session supports integer kinds 1 (int8), '// &
                '2 (int16), 4 (default), and 8 (int64); kind '// &
                trim(kind_spec)//' is not yet supported', &
                error_msg)
        end if
    end subroutine integer_kind_to_value_kind

    ! Fold a kind-spec token that names a declared "integer, parameter" symbol
    ! to its compile-time value (real(prec) :: v where prec is not a
    ! hardcoded name). Purely numeric and iso_c_binding kind specs are matched
    ! by their caller before this runs, so a lookup miss here just means the
    ! token is not a resolvable kind name.
    subroutine resolve_kind_parameter_value(context, kind_spec, resolved, &
                                            value, reference_index)
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: kind_spec
        logical, intent(out) :: resolved
        integer(c_int64_t), intent(out) :: value
        integer, intent(in), optional :: reference_index
        character(len=:), allocatable :: name
        character(len=:), allocatable :: error_msg
        logical :: found
        integer :: symbol_index

        resolved = .false.
        value = 0_c_int64_t
        name = trim(kind_spec)
        if (len_trim(name) == 0) return
        if (present(reference_index)) then
            call fold_scoped_i32_name(context%arena, context, reference_index, &
                                      name, value, found, error_msg)
            if (len_trim(error_msg) == 0) then
                resolved = .true.
                return
            end if
            if (found) return
        end if
        symbol_index = find_symbol_compat(context, name)
        if (symbol_index <= 0) return
        if (.not. context%symbols(symbol_index)%is_parameter) return
        if (.not. context%symbols(symbol_index)%has_i32_constant) return
        resolved = .true.
        value = context%symbols(symbol_index)%i32_constant
    end subroutine resolve_kind_parameter_value

    ! Reject a real/complex kind selector whose value has no exact machine
    ! representation in the direct LIRIC session. REAL(16) is accepted only
    ! through the explicit emulation path; silently narrowing complex(16) to a
    ! supported width would change the program's arithmetic.
    ! Only kind selectors that resolve to a definite value - numeric literals
    ! and declared integer parameters - are judged here; symbolic kind names
    ! keep their existing classification.
    subroutine check_float_kind_supported(base, kind_spec, line, column, &
                                          error_msg, context, reference_index)
        character(len=*), intent(in) :: base
        character(len=*), intent(in) :: kind_spec
        integer, intent(in) :: line, column
        character(len=:), allocatable, intent(out) :: error_msg
        type(lowering_context_t), intent(in), optional :: context
        integer, intent(in), optional :: reference_index
        logical :: resolved, has_value
        integer(c_int64_t) :: kind_value
        character(len=32) :: kind_text

        call set_empty(error_msg)
        if (len_trim(kind_spec) == 0) return
        has_value = .false.
        kind_value = 0_c_int64_t
        if (is_integer_literal_text(kind_spec)) then
            read (kind_spec, *) kind_value
            has_value = .true.
        else if (resolve_selected_real_kind_spec(kind_spec, kind_value)) then
            has_value = .true.
        else if (present(context)) then
            call resolve_kind_parameter_value(context, kind_spec, resolved, &
                                              kind_value, reference_index)
            has_value = resolved
        end if
        if (.not. has_value) return
        if (kind_value == 4_c_int64_t) return
        if (kind_value == 8_c_int64_t) return
        if (kind_value == 16_c_int64_t .and. trim(base) == 'real') return
        write (kind_text, '(I0)') kind_value
        if (trim(base) == 'real') then
            call unsupported_feature_error('real kind', line, column, &
                'direct LIRIC session supports real kinds 4 (single), 8 '// &
                '(double), and 16 (emulated quad); kind '// &
                trim(adjustl(kind_text))//' is not supported', error_msg)
        else
            call unsupported_feature_error('complex kind', line, column, &
                'direct LIRIC session supports complex kinds 4 (single) and '// &
                '8 (double); kind '//trim(adjustl(kind_text))// &
                ' is not supported', error_msg)
        end if
    end subroutine check_float_kind_supported

    ! True when text is a plain unsigned decimal integer literal.
    logical function is_integer_literal_text(text) result(is_literal)
        character(len=*), intent(in) :: text
        character(len=:), allocatable :: trimmed
        integer :: i

        trimmed = trim(adjustl(text))
        is_literal = len(trimmed) > 0
        if (.not. is_literal) return
        do i = 1, len(trimmed)
            if (trimmed(i:i) < '0' .or. trimmed(i:i) > '9') then
                is_literal = .false.
                return
            end if
        end do
    end function is_integer_literal_text

    ! Lazy Fortran dialect defaults apply only when the driver compiled this
    ! unit in lazy input mode (#438). Standard Fortran sources keep the
    ! processor-dependent default real kind ffc has always used.
    ! Lazy Fortran default real is 8 bytes (#438). Every metadata source for
    ! one symbol - specification declaration, dummy-argument entry, function
    ! result - must record that same kind, otherwise the records collide as
    ! conflicting declaration metadata (#571).
    subroutine apply_lazy_real_kind(context, type_kind, has_kind, kind_value)
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: type_kind
        logical, intent(in) :: has_kind
        integer, intent(inout) :: kind_value

        if (.not. context%lazy_mode) return
        if (type_kind /= TREAL) return
        if (has_kind) return
        kind_value = 8
    end subroutine apply_lazy_real_kind

    logical function lazy_defaults_active(context) result(active)
        type(lowering_context_t), intent(in), optional :: context

        active = .false.
        if (.not. present(context)) return
        active = context%lazy_mode
    end function lazy_defaults_active

    ! Returns true when the real kind spec selects f64 (kind 8). Bare real
    ! and real(4)/c_float/real32 are f32; a declared "integer, parameter"
    ! kind name (real(prec) :: v) is folded via the symbol table.
    logical function real_kind_is_f64(kind_spec, context, reference_index)
        character(len=*), intent(in) :: kind_spec
        type(lowering_context_t), intent(in), optional :: context
        integer, intent(in), optional :: reference_index
        logical :: resolved
        integer(c_int64_t) :: resolved_value

        real_kind_is_f64 = .false.
        if (resolve_selected_real_kind_spec(kind_spec, resolved_value)) then
            real_kind_is_f64 = resolved_value == 8_c_int64_t .or. &
                               resolved_value == 16_c_int64_t
            return
        end if
        if (present(context)) then
            call resolve_kind_parameter_value(context, kind_spec, resolved, &
                                              resolved_value, reference_index)
            if (resolved) then
                real_kind_is_f64 = (resolved_value == 8_c_int64_t .or. &
                                    resolved_value == 16_c_int64_t)
                return
            end if
        end if
        select case (trim(kind_spec))
        case ('8', '16', 'real64', 'c_double', 'c_long_double', 'dp', 'wp')
            ! dp/wp are the conventional double-precision kind parameters
            ! (dp/wp => real64 or selected_real_kind(15)); sp/default stay f32.
            real_kind_is_f64 = .true.
            return
        end select
    end function real_kind_is_f64

    logical function resolve_selected_real_kind_spec(kind_spec, kind_value)
        character(len=*), intent(in) :: kind_spec
        integer(c_int64_t), intent(out) :: kind_value
        character(len=:), allocatable :: compact, arguments, precision_text
        character(len=:), allocatable :: range_text
        integer(c_int64_t) :: precision, range
        integer :: i, comma_pos, close_pos, ios

        kind_value = 0_c_int64_t
        resolve_selected_real_kind_spec = .false.
        compact = ''
        do i = 1, len_trim(kind_spec)
            if (kind_spec(i:i) /= ' ') then
                compact = compact//lowercase_text(kind_spec(i:i))
            end if
        end do
        if (index(compact, 'selected_real_kind(') /= 1) return
        close_pos = index(compact, ')', back=.true.)
        if (close_pos /= len_trim(compact)) return
        arguments = compact(len('selected_real_kind(') + 1:close_pos - 1)
        comma_pos = index(arguments, ',')
        if (comma_pos > 0) then
            precision_text = arguments(:comma_pos - 1)
            range_text = arguments(comma_pos + 1:)
        else
            precision_text = arguments
            range_text = '0'
        end if
        read (precision_text, *, iostat=ios) precision
        if (ios /= 0) return
        read (range_text, *, iostat=ios) range
        if (ios /= 0) return
        kind_value = selected_real_kind_value(precision, range)
        resolve_selected_real_kind_spec = .true.
    end function resolve_selected_real_kind_spec

    ! Returns true when the complex kind spec selects double precision (kind
    ! 8). Bare complex and complex(4) are c4 (two f32s); a declared "integer,
    ! parameter" kind name is folded via the symbol table.
    logical function complex_kind_is_c8(kind_spec, context, reference_index)
        character(len=*), intent(in) :: kind_spec
        type(lowering_context_t), intent(in), optional :: context
        integer, intent(in), optional :: reference_index
        logical :: resolved
        integer(c_int64_t) :: resolved_value

        complex_kind_is_c8 = .false.
        if (present(context)) then
            call resolve_kind_parameter_value(context, kind_spec, resolved, &
                                              resolved_value, reference_index)
            if (resolved) then
                complex_kind_is_c8 = (resolved_value == 8_c_int64_t)
                return
            end if
        end if
        select case (trim(kind_spec))
        case ('8', 'real64', 'c_double', 'c_double_complex', 'dp', 'wp')
            ! dp/wp are the conventional double-precision kind parameters,
            ! matching real_kind_is_f64; sp/default stay c4.
            complex_kind_is_c8 = .true.
            return
        end select
    end function complex_kind_is_c8

    subroutine declaration_value_kind(node, value_kind, error_msg, context, &
                                      reference_index)
        type(declaration_node), intent(in) :: node
        integer, intent(out) :: value_kind
        character(len=:), allocatable, intent(out) :: error_msg
        type(lowering_context_t), intent(in), optional :: context
        integer, intent(in), optional :: reference_index

        value_kind = VALUE_I32
        call set_empty(error_msg)
        if (.not. allocated(node%type_name)) return
        ! A declared class(t) entity is a derived value with a polymorphic
        ! descriptor, not an intrinsic scalar spelling. Resolve it through
        ! the registered derived-type table before classifying the type text.
        if (present(context)) then
            if (declaration_derived_type_index(context, node) > 0) then
                value_kind = VALUE_DERIVED
                return
            end if
        end if
        call type_name_value_kind(node%type_name, node%line, node%column, &
                                  value_kind, error_msg, context, reference_index)
    end subroutine declaration_value_kind
