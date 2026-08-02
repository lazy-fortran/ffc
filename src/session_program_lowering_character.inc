    subroutine define_declared_character_symbol(context, node, name, error_msg)
        type(lowering_context_t), intent(inout) :: context
        type(declaration_node), intent(in) :: node
        character(len=*), intent(in) :: name
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: character_length
        integer :: existing_index
        integer :: source_index

        existing_index = find_symbol_compat(context, name)
        if (existing_index > 0) then
            if (context%in_internal_function .and. &
                existing_index == context%current_function_result_index) then
                call declaration_character_length(context, node, &
                                                  character_length, error_msg)
                ! A runtime length (character(len=n) where n is a dummy arg or
                ! other non-constant) keeps the deferred descriptor ABI: the
                ! length is resolved when the body assigns the result value.
                if (len_trim(error_msg) > 0) then
                    ! F2018 10.1.11: a nonconstant result length must be a
                    ! specification expression. A plain local variable of the
                    ! same function is not one, so reject it instead of letting
                    ! the deferred ABI silently invent a width (#410).
                    if (result_length_is_local_variable(context, node)) then
                        call unsupported_feature_error( &
                            'character length declaration', node%line, &
                            node%column, &
                            'nonconstant character result length must be a '// &
                            'specification expression: a local variable of '// &
                            'the same procedure is not permitted', error_msg)
                        return
                    end if
                    if (runtime_character_length(context, node)) then
                        if (character_length_is_len_trim(node)) then
                            call capture_runtime_len_trim_length(context, node, &
                                existing_index, error_msg)
                            if (len_trim(error_msg) > 0) return
                        else
                            call resolve_len_of_character_length(context, node, &
                                source_index)
                            if (source_index > 0) then
                                call capture_runtime_fixed_character_length( &
                                    context, existing_index, source_index, &
                                    error_msg)
                                if (len_trim(error_msg) > 0) return
                            end if
                        end if
                        call set_empty(error_msg)
                    end if
                    return
                end if
                ! A fixed-length result (character(len=N) :: s) reuses the
                ! descriptor-return ABI of the deferred result but pads its
                ! assigned value to N; record the length so the assignment
                ! lowering knows the fixed width.
                if (character_length > 0 .and. &
                    context%symbols(existing_index)%is_deferred_character) then
                    context%symbols(existing_index)%character_length = &
                        character_length
                end if
                call set_empty(error_msg)
                return
            end if
            if (context%symbols(existing_index)%is_parameter) then
                ! Both a fixed-length dummy (character(len=N)) and an
                ! assumed-length one (character(len=*)) read their data
                ! pointer from the caller's {data, length} descriptor; a
                ! fixed-length dummy keeps its own declared width N rather
                ! than the caller's runtime length.
                call declaration_character_length(context, node, &
                                                  character_length, error_msg)
                if (len_trim(error_msg) > 0) return
                ! An OPTIONAL dummy shares the presence ABI of every other
                ! scalar kind: absent is a null reference pointer. Binding a
                ! fixed-length dummy eagerly loads the data pointer out of
                ! that descriptor, which faults before present() can be
                ! evaluated, so an optional dummy keeps the descriptor form
                ! (address arithmetic only) and reads its data at each use,
                ! inside the guarded branch. The declared width N is still
                ! recorded so the dummy keeps its own length, not the
                ! caller's.
                if (character_length > 0 .and. .not. node%is_optional) then
                    call bind_fixed_character_parameter_symbol(context, &
                        existing_index, character_length, error_msg)
                else if (character_length > 0) then
                    call bind_character_parameter_symbol(context, &
                        existing_index, error_msg)
                    if (len_trim(error_msg) > 0) return
                    context%symbols(existing_index)%character_length = &
                        character_length
                else
                    call bind_character_parameter_symbol(context, &
                        existing_index, error_msg)
                end if
                return
            end if
        end if
        call declaration_character_length(context, node, character_length, &
                                          error_msg)
        if (len_trim(error_msg) > 0) then
            call resolve_len_of_character_length(context, node, source_index)
            if (source_index > 0) then
                call define_runtime_length_character_local(context, name, &
                    source_index, error_msg)
            end if
            return
        end if
        call define_character_symbol(context, name, character_length, error_msg)
    end subroutine define_declared_character_symbol

    logical function character_length_is_len_trim(node)
        type(declaration_node), intent(in) :: node
        character(len=:), allocatable :: expression

        character_length_is_len_trim = .false.
        if (.not. node%has_character_length) return
        if (.not. allocated(node%character_length_expr)) return
        expression = lowercase_text(trim(adjustl(node%character_length_expr)))
        if (len(expression) <= len('len_trim(')) return
        if (expression(1:len('len_trim(')) /= 'len_trim(') return
        character_length_is_len_trim = &
            expression(len(expression):len(expression)) == ')'
    end function character_length_is_len_trim

    subroutine capture_runtime_len_trim_length(context, node, symbol_index, &
                                                error_msg)
        type(lowering_context_t), intent(inout) :: context
        type(declaration_node), intent(in) :: node
        integer, intent(in) :: symbol_index
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: expression, inner
        integer :: source_index
        type(lr_operand_desc_t) :: source_data, source_length
        type(lr_operand_desc_t) :: trimmed_length, trimmed_length_i64

        call set_empty(error_msg)
        if (.not. allocated(node%character_length_expr)) return
        expression = trim(adjustl(node%character_length_expr))
        inner = trim(adjustl(expression(len('len_trim(') + 1:len(expression) - 1)))
        if (len(inner) == 0) return
        if (verify(inner, &
            'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_') &
            /= 0) return
        source_index = find_symbol_compat(context, inner)
        if (source_index <= 0) return
        if (context%symbols(source_index)%value_kind /= VALUE_CHARACTER) return

        ! The result descriptor is the destination, not the source of its own
        ! specification expression. Compute LEN_TRIM from the referenced
        ! character before any result bytes are assigned, so the result's
        ! {data,length} ABI is initialized with the actual width (#350).
        call char_length_operands(context, source_index, source_data, &
                                  source_length, error_msg)
        if (len_trim(error_msg) > 0) return
        call compute_len_trim(context, source_data, source_length, &
                              trimmed_length, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_liric_i32_to_i64(context%session, trimmed_length, &
                                        trimmed_length_i64, error_msg)) return
        if (.not. emit_i64_store(context%session, trimmed_length_i64, &
                                 context%symbols(symbol_index)%deferred_length, &
                                 error_msg)) return
        context%symbols(symbol_index)%is_runtime_fixed_character = .true.
        call set_empty(error_msg)
    end subroutine capture_runtime_len_trim_length

    subroutine define_runtime_length_character_local(context, name, &
                                                      source_index, error_msg)
        ! character(len=len(other)) :: name, where other is a declared
        ! character variable whose own length is only known at runtime (an
        ! assumed-length dummy, for instance). The local reuses the deferred
        ! {data, length} descriptor ABI, seeding its length from other's
        ! current runtime length; content stays unset until an assignment
        ! fills it in, matching an ordinary automatic-length local that is
        ! declared before use.
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: name
        integer, intent(in) :: source_index
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: symbol_index

        call define_deferred_character_symbol(context, name, error_msg)
        if (len_trim(error_msg) > 0) return
        symbol_index = find_symbol_compat(context, name)
        if (symbol_index <= 0) then
            error_msg = &
                'internal error defining runtime-length character local: '// &
                trim(name)
            return
        end if

        call capture_runtime_fixed_character_length(context, symbol_index, &
                                                    source_index, error_msg)
        if (len_trim(error_msg) > 0) return
        call set_empty(error_msg)
    end subroutine define_runtime_length_character_local

    subroutine capture_runtime_fixed_character_length(context, symbol_index, &
                                                       source_index, error_msg)
        ! Capture a `character(len=len(source))` width once at declaration.
        ! The storage layout stays the ordinary {data, length} descriptor, but
        ! the runtime-fixed flag makes later assignments retain this length.
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        integer, intent(in) :: source_index
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: source_data, source_len_i32, source_len_i64

        context%symbols(symbol_index)%is_runtime_fixed_character = .true.
        call char_length_operands(context, source_index, source_data, &
                                  source_len_i32, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_liric_i32_to_i64(context%session, source_len_i32, &
                source_len_i64, error_msg)) return
        if (.not. emit_i64_store(context%session, source_len_i64, &
                context%symbols(symbol_index)%deferred_length, error_msg)) &
            return
        call set_empty(error_msg)
    end subroutine capture_runtime_fixed_character_length

    subroutine resolve_len_of_character_length(context, node, source_index)
        ! Recognize a character length expression of the exact shape
        ! len(<name>), where <name> is a declared character variable.
        ! source_index is left at 0 when the expression does not match.
        type(lowering_context_t), intent(in) :: context
        type(declaration_node), intent(in) :: node
        integer, intent(out) :: source_index
        character(len=:), allocatable :: expr
        character(len=:), allocatable :: lowered
        character(len=:), allocatable :: inner

        source_index = 0
        if (.not. node%has_character_length) return
        if (.not. allocated(node%character_length_expr)) return
        expr = trim(adjustl(node%character_length_expr))
        if (len(expr) < 6) return
        lowered = lowercase_text(expr)
        if (lowered(1:4) /= 'len(') return
        if (lowered(len(lowered):len(lowered)) /= ')') return
        inner = trim(adjustl(expr(5:len(expr) - 1)))
        if (len(inner) == 0) return
        if (verify(inner, &
            'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_') &
            /= 0) return

        source_index = find_symbol_compat(context, inner)
        if (source_index <= 0) return
        if (context%symbols(source_index)%value_kind /= VALUE_CHARACTER) &
            source_index = 0
    end subroutine resolve_len_of_character_length

    ! #384: reject a character length specification that cannot be a scalar
    ! INTEGER expression: a real, complex, logical or character literal, or a
    ! LEN() of an entity that is not of type CHARACTER.
    subroutine validate_character_length_expression(node, error_msg)
        type(declaration_node), intent(in) :: node
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: expr
        character(len=:), allocatable :: reason
        character(len=64) :: location

        call set_empty(error_msg)
        if (.not. node%has_character_length) return
        if (.not. allocated(node%character_length_expr)) return
        expr = trim(adjustl(node%character_length_expr))
        if (len(expr) == 0) return

        reason = character_length_literal_reason(expr)
        if (len_trim(reason) == 0) return

        write (location, '(" at line ",I0,", column ",I0)') node%line, node%column
        error_msg = 'character length must be a scalar INTEGER expression: '// &
                    trim(reason)//trim(location)
    end subroutine validate_character_length_expression

    subroutine declaration_character_length(context, node, character_length, &
                                            error_msg, reference_index)
        type(lowering_context_t), intent(in) :: context
        type(declaration_node), intent(in) :: node
        integer, intent(out) :: character_length
        character(len=:), allocatable, intent(out) :: error_msg
        integer, intent(in), optional :: reference_index
        character(len=:), allocatable :: parse_error
        character(len=:), allocatable :: lowered

        character_length = -1
        call set_empty(error_msg)
        ! #384: a length that is not a scalar INTEGER expression is invalid at
        ! every declaration site, so it is rejected before any other handling.
        call validate_character_length_expression(node, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. allocated(node%type_name)) then
            return
        end if
        lowered = lowercase_text(node%type_name)
        if (trim(lowered) == 'character') then
            ! Deferred character: no len=N specified
            return
        end if
        call parse_character_length(node%type_name, character_length, parse_error)
        if (len_trim(parse_error) == 0) return

        ! A non-literal length such as character(max_len) keeps the length
        ! text in character_length_expr. Resolve a named integer parameter
        ! against the symbol table before reporting it unsupported.
        if (present(reference_index)) then
            call resolve_named_character_length(context, node, character_length, &
                                                parse_error, reference_index)
        else
            call resolve_named_character_length(context, node, character_length, &
                                                parse_error)
        end if
        if (len_trim(parse_error) > 0) then
            call unsupported_feature_error('character length declaration', &
                                           node%line, node%column, &
                                           trim(parse_error), error_msg)
        end if
    end subroutine declaration_character_length

    ! Resolve character(<name>) where <name> is a compile-time integer
    ! parameter. Leaves error_msg set when the length text is not a single
    ! resolvable parameter name.
    subroutine resolve_named_character_length(context, node, character_length, &
                                              error_msg, reference_index)
        type(lowering_context_t), intent(in) :: context
        type(declaration_node), intent(in) :: node
        integer, intent(out) :: character_length
        character(len=:), allocatable, intent(out) :: error_msg
        integer, intent(in), optional :: reference_index
        character(len=:), allocatable :: name
        character(len=:), allocatable :: fold_error
        integer(c_int64_t) :: folded
        logical :: folded_found
        integer :: anchor_index
        integer :: symbol_index

        character_length = -1
        error_msg = 'character length must be a positive integer literal'
        if (.not. node%has_character_length) return
        if (.not. allocated(node%character_length_expr)) return
        name = trim(adjustl(node%character_length_expr))
        if (len(name) == 0) return
        if (verify(name, &
            'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_') &
            /= 0) return

        anchor_index = context%current_declaration_index
        if (present(reference_index)) anchor_index = reference_index
        ! Ask FortFront which constant this name denotes at the declaration
        ! that spells it (#329). The length text carries no arena node, so the
        ! declaration is the anchor; without it a BLOCK-local parameter that
        ! shadows a host one of the same name silently takes the host width.
        call fold_named_constant_at_node(context, &
                                         anchor_index, &
                                         name, folded, folded_found, fold_error)
        if (len_trim(fold_error) == 0 .and. folded_found) then
            if (folded <= 0_c_int64_t) return
            character_length = int(folded)
            call set_empty(error_msg)
            return
        end if

        ! No FortFront binding: a symbol ffc synthesised without a declaration
        ! of its own still resolves by name, as it did before #329.
        symbol_index = find_symbol_compat(context, name)
        if (symbol_index <= 0) return
        if (.not. context%symbols(symbol_index)%has_i32_constant) return
        if (context%symbols(symbol_index)%i32_constant <= 0) return
        character_length = int(context%symbols(symbol_index)%i32_constant)
        call set_empty(error_msg)
    end subroutine resolve_named_character_length

    ! True when the character length is a runtime length rather than a literal
    ! or compile-time parameter. This covers a bare dummy identifier
    ! (character(len=k)) and a length expression that references a runtime
    ! quantity (character(len=len(name)), character(len=len(name)+7)). Such a
    ! length is only valid for a function result here, lowered through the
    ! deferred descriptor ABI: the actual width comes from the assigned result
    ! value, so the declared expression itself is never evaluated.
    ! True when a function result's character length is written as a bare
    ! identifier naming a local variable of the same procedure: not a dummy
    ! argument, not a named constant, and not host- or module-associated
    ! storage. Such a length is not a valid specification expression.
    logical function result_length_is_local_variable(context, node)
        type(lowering_context_t), intent(in) :: context
        type(declaration_node), intent(in) :: node
        character(len=:), allocatable :: expr
        integer :: symbol_index

        result_length_is_local_variable = .false.
        if (.not. context%in_internal_function) return
        if (.not. node%has_character_length) return
        if (.not. allocated(node%character_length_expr)) return
        expr = trim(adjustl(node%character_length_expr))
        if (len(expr) == 0) return
        if (verify(expr, &
            'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_') &
            /= 0) return
        symbol_index = find_symbol_compat(context, expr)
        if (symbol_index <= 0) return
        if (context%symbols(symbol_index)%is_parameter) return
        if (context%symbols(symbol_index)%has_i32_constant) return
        if (symbol_has_global_storage(context%symbols(symbol_index))) return
        result_length_is_local_variable = .true.
    end function result_length_is_local_variable

    logical function runtime_character_length(context, node)
        type(lowering_context_t), intent(in) :: context
        type(declaration_node), intent(in) :: node
        character(len=:), allocatable :: expr
        character(len=:), allocatable :: name
        character(len=:), allocatable :: invalid_msg
        integer :: symbol_index

        runtime_character_length = .false.
        if (.not. node%has_character_length) return
        if (.not. allocated(node%character_length_expr)) return
        ! An invalid length (#384) is never a runtime length: the deferred ABI
        ! must not absorb it and hide the diagnostic.
        call validate_character_length_expression(node, invalid_msg)
        if (len_trim(invalid_msg) > 0) return
        expr = trim(adjustl(node%character_length_expr))
        if (len(expr) == 0) return

        ! A bare identifier that is a declared non-constant symbol (a dummy).
        if (verify(expr, &
            'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_') &
            == 0) then
            symbol_index = find_symbol_compat(context, expr)
            if (symbol_index <= 0) return
            runtime_character_length = &
                .not. context%symbols(symbol_index)%has_i32_constant
            return
        end if

        ! A length expression such as len(name) or len(name)+7 is a runtime
        ! length when at least one identifier it references is a declared
        ! non-constant symbol (a runtime dummy or variable). The deferred ABI
        ! resolves the width from the assigned value, so the expression is not
        ! evaluated here.
        runtime_character_length = &
            length_expr_references_runtime_symbol(context, expr)
    end function runtime_character_length

    ! Scan an integer length expression for identifiers and report whether any
    ! names a declared non-constant symbol. Intrinsic names like len that are
    ! not declared symbols are ignored.
    logical function length_expr_references_runtime_symbol(context, expr)
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: expr
        character(len=:), allocatable :: ident_chars
        integer :: i, start, symbol_index
        logical :: in_ident

        length_expr_references_runtime_symbol = .false.
        ident_chars = &
            'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_'
        in_ident = .false.
        start = 0
        do i = 1, len(expr)
            if (verify(expr(i:i), ident_chars) == 0) then
                if (.not. in_ident) then
                    in_ident = .true.
                    start = i
                end if
            else
                if (in_ident) then
                    if (token_is_runtime_symbol(context, expr(start:i - 1))) then
                        length_expr_references_runtime_symbol = .true.
                        return
                    end if
                    in_ident = .false.
                end if
            end if
        end do
        if (in_ident) then
            length_expr_references_runtime_symbol = &
                token_is_runtime_symbol(context, expr(start:))
        end if
    end function length_expr_references_runtime_symbol

    ! True when token is a declared symbol that is not a compile-time integer
    ! parameter. A leading digit (a numeric literal) is rejected outright.
    logical function token_is_runtime_symbol(context, token)
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: token
        integer :: symbol_index

        token_is_runtime_symbol = .false.
        if (len(token) == 0) return
        if (verify(token(1:1), &
            'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_') /= 0) return
        symbol_index = find_symbol_compat(context, token)
        if (symbol_index <= 0) return
        token_is_runtime_symbol = &
            .not. context%symbols(symbol_index)%has_i32_constant
    end function token_is_runtime_symbol

    subroutine define_character_symbol(context, name, character_length, &
                                       error_msg)
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: name
        integer, intent(in) :: character_length
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: index

        if (find_symbol_compat(context, name) > 0) then
            error_msg = 'duplicate character declaration: '//trim(name)
            return
        end if
       call grow_symbols(context)
        if (character_length <= 0) then
            call define_deferred_character_symbol(context, name, error_msg)
            return
        end if

        index = context%symbol_count + 1
        context%symbols(index)%name = trim(name)
        context%symbols(index)%value_kind = VALUE_CHARACTER
        context%symbols(index)%character_length = character_length
        context%symbols(index)%has_character_value = .false.
        context%symbol_count = index
        call set_empty(error_msg)
    end subroutine define_character_symbol

    subroutine lower_character_assignment(arena, node, context, symbol_index, &
                                          error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(assignment_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: literal_text
        character(len=64) :: string_name
        logical :: fold_ok
        character(len=:), allocatable :: bin_op
        integer :: bin_left, bin_right, bin_line, bin_col
        character(len=:), allocatable :: lit_value, lit_type
        type(lr_operand_desc_t) :: literal_ptr

        if (.not. node_exists(arena, node%value_index)) then
            error_msg = 'character assignment value does not reference an AST node'
            return
        end if

        if (context%symbols(symbol_index)%is_runtime_fixed_character) then
            call lower_runtime_fixed_char_assignment(arena, node%value_index, &
                context, symbol_index, error_msg)
            return
        end if

       ! Check for deferred character function call result or trim() result
        if (context%symbols(symbol_index)%is_deferred_character) then
            if (is_char_expr_call(arena, node%value_index, context)) then
                call lower_deferred_trim_assignment(arena, node%value_index, &
                    context, symbol_index, error_msg)
                return
            end if
            if (node_exists(arena, node%value_index)) then
                select type (val => arena%entries(node%value_index)%node)
                type is (call_or_subscript_node)
                    if (.not. val%is_array_access .and. &
                        allocated(val%name)) then
                        if (is_contained_deferred_char_function(context, &
                            val%name)) then
                            call lower_deferred_char_result_call(arena, node, &
                                context, symbol_index, error_msg)
                            return
                        end if
                    end if
                end select
            end if
        end if

        ! A fixed-length scalar may receive the result of a character-returning
        ! contained function (e.g. character(len=5) :: r; r = make(...)). The
        ! callee returns a {data,len} descriptor; bind the destination's value
        ! to the returned data so later uses (print) see the result text.
        if (.not. context%symbols(symbol_index)%is_deferred_character) then
            if (node_exists(arena, node%value_index)) then
                select type (val => arena%entries(node%value_index)%node)
                type is (call_or_subscript_node)
                    if (.not. val%is_array_access .and. allocated(val%name)) then
                        if (is_contained_deferred_char_function(context, &
                            val%name)) then
                            call lower_fixed_char_result_call(arena, node, &
                                context, symbol_index, error_msg)
                            return
                        end if
                    end if
                end select
            end if
        end if

        ! Check for deferred character with binary // operator or literal
        if (context%symbols(symbol_index)%is_deferred_character) then
            if (is_binary_op(arena, node%value_index)) then
                call get_binary_op_info(arena, node%value_index, bin_op, &
                                        bin_left, bin_right, bin_line, bin_col, &
                                        error_msg)
                if (len_trim(error_msg) > 0) return
                if (trim(bin_op) == '//') then
                    ! An all-literal // chain (any depth) folds to one literal.
                    call concat_character_literals(arena, node%value_index, &
                                                   literal_text, fold_ok)
                    if (fold_ok) then
                        call lower_deferred_folded_literal_assignment( &
                            literal_text, context, symbol_index, error_msg)
                        return
                    end if
                    call lower_deferred_concat_assignment(arena, node, &
                        context, symbol_index, error_msg)
                    return
                end if
            else if (is_literal(arena, node%value_index)) then
                if (is_character_literal(arena, node%value_index)) then
                    call get_literal_info(arena, node%value_index, lit_value, &
                                          lit_type, error_msg)
                    if (len_trim(error_msg) > 0) return
                    call lower_deferred_literal_assignment(lit_value, &
                        context, symbol_index, error_msg)
                    return
                end if
            else if (is_character_substring(arena, node%value_index, context)) &
                then
                ! Deferred target receives a substring view: intrinsic
                ! assignment copies the view's bytes at the view's length, so
                ! the target never aliases its parent's storage.
                call lower_deferred_char_view_assignment(arena, &
                    node%value_index, context, symbol_index, error_msg)
                return
            else if (is_identifier(arena, node%value_index)) then
                ! Deferred target receives another character variable: copy the
                ! source {data, len} into the target descriptor (str = other).
                call lower_deferred_identifier_assignment(arena, &
                    node%value_index, context, symbol_index, error_msg)
                return
            end if
        end if

        if (is_binary_op(arena, node%value_index)) then
            call concat_character_literals(arena, node%value_index, &
                                           literal_text, fold_ok)
            if (.not. fold_ok) then
                ! A // chain with variable or intrinsic operands concatenates at
                ! runtime into a fresh buffer bound to the target's value.
                call lower_fixed_concat_assignment(arena, node, context, &
                                                   symbol_index, error_msg)
                return
            end if
        else if (is_literal(arena, node%value_index)) then
            if (.not. is_character_literal(arena, node%value_index)) then
                call unsupported_feature_error('character assignment', &
                                               node%line, node%column, &
                                               'only character literal '// &
                                               'assignment is supported', &
                                               error_msg)
                return
            end if
            call get_literal_info(arena, node%value_index, lit_value, lit_type, &
                                  error_msg)
            if (len_trim(error_msg) > 0) return
            call strip_literal_quotes(lit_value, literal_text)
        else if (is_identifier(arena, node%value_index) .or. &
                 is_char_expr_call(arena, node%value_index, context) .or. &
                 is_character_substring(arena, node%value_index, context)) then
            ! A character variable, a substring view, or a
            ! trim()/adjustl()/adjustr()/achar()/repeat() result assigned to a
            ! fixed-length target: pad or truncate to the target's declared
            ! width like gfortran's fixed-length assignment.
            call lower_fixed_char_expr_assignment(arena, node%value_index, &
                context, symbol_index, error_msg)
            return
        else
            call unsupported_feature_error('character assignment', &
                                           node%line, node%column, &
                                           'only character literal '// &
                                           'assignment is supported', &
                                           error_msg)
            return
        end if

        call normalize_character_literal( &
            literal_text, context%symbols(symbol_index)%character_length)

        context%string_literal_count = context%string_literal_count + 1
        string_name = ffc_unit_global_name( &
            context, 'char.', context%string_literal_count)
        call materialize_liric_string(context%session, trim(string_name), &
                                      literal_text, &
                                      literal_ptr, &
                                      error_msg)
        if (len_trim(error_msg) > 0) return

        ! A scalar character target/pointer owns a stable mutable buffer. Keep
        ! the symbol's data pointer bound to that buffer instead of rebinding it
        ! to an immutable literal/global on every assignment.
        if ((context%symbols(symbol_index)%is_target .or. &
             context%symbols(symbol_index)%is_pointer) .and. &
            context%symbols(symbol_index)%has_address) then
            if (.not. emit_memcpy(context%session, &
                context%symbols(symbol_index)%address, literal_ptr, &
                i64_immediate(context%session, int( &
                    context%symbols(symbol_index)%character_length + 1, &
                    c_int64_t)), error_msg)) return
            context%symbols(symbol_index)%value = &
                context%symbols(symbol_index)%address
        else
            context%symbols(symbol_index)%value = literal_ptr
        end if

        context%symbols(symbol_index)%has_character_value = .true.
        call set_empty(error_msg)
    end subroutine lower_character_assignment

    subroutine lower_deferred_identifier_assignment(arena, value_index, context, &
                                                    symbol_index, error_msg)
        ! str = other, where str is character(len=:), allocatable and other is a
        ! character variable (fixed or deferred). Fortran intrinsic assignment
        ! gives str a value, not a view: str takes its own copy of other's
        ! bytes at other's current length. Aliasing the source pointer would
        ! make str observe later changes to other, and would leave str
        ! dangling once other's storage is released.
        use, intrinsic :: iso_c_binding, only: c_int64_t
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: value_index
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: src_name
        integer :: src_index
        type(lr_operand_desc_t) :: data_ptr, len_i32, len_i64, buf

        call get_identifier_name(arena, value_index, src_name, error_msg)
        if (len_trim(error_msg) > 0) return
        src_index = find_symbol_compat(context, src_name)
        if (src_index <= 0 .or. &
            context%symbols(max(src_index, 1))%value_kind /= VALUE_CHARACTER) then
            call unsupported_feature_error('character assignment', 0, 0, &
                'character assignment source is not a character variable: '// &
                trim(src_name), error_msg)
            return
        end if

        if (src_index == symbol_index) then
            ! str = str keeps the descriptor exactly as it is. Releasing and
            ! reinstalling the same pointer would be a use after free.
            call set_empty(error_msg)
            return
        end if

        call char_length_operands(context, src_index, data_ptr, len_i32, &
                                  error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_liric_i32_to_i64(context%session, len_i32, len_i64, &
                error_msg)) return

        ! Copy into storage the destination owns, before releasing the block it
        ! held: the source may be a different variable that shares that block,
        ! and the copy has to be made while it is still valid.
        call copy_character_bytes_to_owned(context, data_ptr, len_i64, buf, &
                                           error_msg)
        if (len_trim(error_msg) > 0) return
        call release_owned_character_storage(context, symbol_index, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_ptr_store(context%session, buf, &
                context%symbols(symbol_index)%deferred_data, error_msg)) return
        if (.not. emit_i64_store(context%session, len_i64, &
                context%symbols(symbol_index)%deferred_length, error_msg)) return
        call set_character_storage(context, symbol_index, len_i64, &
                                   LOWERING_CHARACTER_STORAGE_OWNED, error_msg)
        if (len_trim(error_msg) > 0) return
        context%symbols(symbol_index)%value = buf
        context%symbols(symbol_index)%has_character_value = .true.
        call set_empty(error_msg)
    end subroutine lower_deferred_identifier_assignment

    subroutine copy_character_bytes_to_owned(context, src_data, len_i64, buf, &
                                             error_msg)
        ! Allocate len + 1 bytes the caller's descriptor will own, copy len
        ! bytes of source text into them, and null-terminate so the print path
        ! stops at the Fortran length. The copy is made before any release, so
        ! it is safe even when the source aliases the storage being replaced.
        use, intrinsic :: iso_c_binding, only: c_int64_t
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(in) :: src_data
        type(lr_operand_desc_t), intent(in) :: len_i64
        type(lr_operand_desc_t), intent(out) :: buf
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: bytes, null_pos

        if (.not. emit_i64_binary(context%session, LR_OP_ADD, len_i64, &
                i64_immediate(context%session, 1_c_int64_t), bytes, &
                error_msg)) return
        if (.not. emit_malloc(context%session, bytes, buf, error_msg)) return
        if (.not. emit_memcpy(context%session, buf, src_data, len_i64, &
                error_msg)) return
        if (.not. emit_i64_binary(context%session, LR_OP_ADD, buf, len_i64, &
                null_pos, error_msg)) return
        if (.not. emit_liric_store_char_byte(context%session, null_pos, &
                i32_immediate(context%session, 0_c_int64_t), &
                i32_immediate(context%session, 0_c_int64_t), error_msg)) return
        call set_empty(error_msg)
    end subroutine copy_character_bytes_to_owned

    recursive subroutine concat_character_literals(arena, node_index, &
                                                   accumulated, ok)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        character(len=:), allocatable, intent(out) :: accumulated
        logical, intent(out) :: ok
        character(len=:), allocatable :: left_text, right_text, piece
        character(len=:), allocatable :: bin_op, bin_err
        integer :: bin_left, bin_right, bin_line, bin_col
        character(len=:), allocatable :: lit_value, lit_type, lit_err

        ok = .false.
        if (.not. node_exists(arena, node_index)) return

        if (is_binary_op(arena, node_index)) then
            call get_binary_op_info(arena, node_index, bin_op, bin_left, &
                                    bin_right, bin_line, bin_col, bin_err)
            if (trim(bin_op) /= '//') return
            call concat_character_literals(arena, bin_left, left_text, ok)
            if (.not. ok) return
            call concat_character_literals(arena, bin_right, right_text, ok)
            if (.not. ok) return
            accumulated = left_text//right_text
            ok = .true.
            return
        end if

        if (is_literal(arena, node_index)) then
            if (.not. is_character_literal(arena, node_index)) return
            call get_literal_info(arena, node_index, lit_value, lit_type, lit_err)
            if (len_trim(lit_err) > 0) return
            call strip_literal_quotes(lit_value, piece)
            accumulated = piece
            ok = .true.
        end if
    end subroutine concat_character_literals

    subroutine normalize_character_literal(literal_text, character_length)
        character(len=:), allocatable, intent(inout) :: literal_text
        integer, intent(in) :: character_length
        character(len=:), allocatable :: fixed_text

        allocate (character(len=character_length) :: fixed_text)
        fixed_text(:) = literal_text
        call move_alloc(fixed_text, literal_text)
    end subroutine normalize_character_literal

    logical function is_character_type_name(name)
        character(len=*), intent(in) :: name
        character(len=:), allocatable :: lowered

        lowered = lowercase_text(name)
        is_character_type_name = lowered == 'character' .or. &
                                 index(lowered, 'character(') == 1
    end function is_character_type_name

    subroutine parse_character_length(type_name, character_length, error_msg)
        character(len=*), intent(in) :: type_name
        integer, intent(out) :: character_length
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: lowered
        character(len=:), allocatable :: length_text
        integer :: close_pos
        integer :: io_stat
        integer :: len_pos
        integer :: start_pos

        character_length = 1
        lowered = lowercase_text(type_name)
        if (trim(lowered) == 'character') then
            call set_empty(error_msg)
            return
        end if

        len_pos = index(lowered, 'len=')
        if (len_pos <= 0) then
            ! Positional form: character(N), character(*), character(:).
            start_pos = index(lowered, '(')
            close_pos = index(lowered, ')', back=.true.)
            if (start_pos <= 0 .or. close_pos <= start_pos) then
                error_msg = 'only character(len=N) declarations are supported'
                return
            end if
            length_text = adjustl(lowered(start_pos + 1:close_pos - 1))
            if (index(length_text, '=') > 0) then
                error_msg = 'only character(len=N) declarations are supported'
                return
            end if
            if (trim(length_text) == ':' .or. trim(length_text) == '*') then
                character_length = -1
                call set_empty(error_msg)
                return
            end if
            read (length_text, *, iostat=io_stat) character_length
            if (io_stat /= 0 .or. character_length <= 0) then
                error_msg = 'only character(len=N) declarations are supported'
                return
            end if
            call set_empty(error_msg)
            return
        end if

        start_pos = len_pos + len('len=')
        close_pos = index(lowered(start_pos:), ')')
        if (close_pos <= 0) then
            error_msg = 'character length declaration is missing ")"'
            return
        end if

        close_pos = start_pos + close_pos - 2
        length_text = adjustl(lowered(start_pos:close_pos))
        if (trim(length_text) == ':' .or. trim(length_text) == '*') then
            character_length = -1
            call set_empty(error_msg)
            return
        end if
        read (length_text, *, iostat=io_stat) character_length
        if (io_stat /= 0 .or. character_length <= 0) then
            error_msg = 'character length must be a positive integer literal'
            return
        end if

        call set_empty(error_msg)
    end subroutine parse_character_length

    subroutine define_deferred_character_symbol(context, name, error_msg)
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: name
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: index
        type(lr_operand_desc_t) :: descriptor
        type(lr_operand_desc_t) :: data_addr
        type(lr_operand_desc_t) :: len_addr
        type(lr_operand_desc_t) :: capacity_addr
        type(lr_operand_desc_t) :: storage_addr

        if (find_symbol_compat(context, name) > 0) then
            error_msg = 'duplicate character declaration: '//trim(name)
            return
        end if
     call grow_symbols(context)

        index = context%symbol_count + 1
        context%symbols(index)%name = trim(name)
        context%symbols(index)%value_kind = VALUE_CHARACTER
        context%symbols(index)%is_deferred_character = .true.
        context%symbols(index)%has_character_value = .false.

        ! One contiguous 32-byte canonical character descriptor
        ! {i8* data; i64 length; i64 capacity; i32 storage_class}, laid out as
        ! docs/CHARACTER_DESCRIPTOR_ABI.md specifies. The {data, length} prefix
        ! is unchanged, so the descriptor's address is still what a
        ! character(len=*) dummy binds against (bind_character_parameter_symbol)
        ! for an intent(out)/(inout) actual. Every field starts zeroed, which is
        ! the ABI's null state: no data, zero length, zero capacity, storage
        ! class CHARACTER_STORAGE_NULL.
        if (.not. emit_alloca_bytes(context%session, &
                i64_immediate(context%session, 32_c_int64_t), descriptor, &
                error_msg)) return
        if (.not. emit_ptr_offset(context%session, descriptor, 0_c_int64_t, &
                                  data_addr, error_msg)) return
        if (.not. emit_ptr_offset(context%session, descriptor, 8_c_int64_t, &
                                  len_addr, error_msg)) return
        if (.not. emit_ptr_offset(context%session, descriptor, 16_c_int64_t, &
                                  capacity_addr, error_msg)) return
        if (.not. emit_ptr_offset(context%session, descriptor, 24_c_int64_t, &
                                  storage_addr, error_msg)) return
        if (.not. emit_i64_store(context%session, &
                                 i64_immediate(context%session, 0_c_int64_t), &
                                 data_addr, error_msg)) return
        if (.not. emit_i64_store(context%session, &
                                 i64_immediate(context%session, 0_c_int64_t), &
                                 len_addr, error_msg)) return
        if (.not. emit_i64_store(context%session, &
                                 i64_immediate(context%session, 0_c_int64_t), &
                                 capacity_addr, error_msg)) return
        if (.not. emit_i64_store(context%session, &
                                 i64_immediate(context%session, 0_c_int64_t), &
                                 storage_addr, error_msg)) return

        context%symbols(index)%deferred_data = data_addr
        context%symbols(index)%deferred_length = len_addr
        context%symbols(index)%deferred_capacity = capacity_addr
        context%symbols(index)%deferred_storage = storage_addr
        context%symbols(index)%has_character_ownership = .true.
        context%symbol_count = index
        call set_empty(error_msg)
    end subroutine define_deferred_character_symbol

    subroutine set_character_storage(context, symbol_index, capacity, &
                                     storage_class, error_msg)
        ! Record the ownership half of the canonical character descriptor:
        ! how many bytes are usable at data without reallocating, and which
        ! storage class owns them (docs/CHARACTER_DESCRIPTOR_ABI.md). Symbols
        ! whose descriptor is only the 16-byte {data, length} prefix (dummies,
        ! runtime-fixed automatics) carry no ownership slots and are skipped.
        use, intrinsic :: iso_c_binding, only: c_int64_t
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        type(lr_operand_desc_t), intent(in) :: capacity
        integer(c_int64_t), intent(in) :: storage_class
        character(len=:), allocatable, intent(out) :: error_msg

        call set_empty(error_msg)
        if (symbol_index <= 0) return
        if (.not. context%symbols(symbol_index)%has_character_ownership) return

        if (.not. emit_i64_store(context%session, capacity, &
                context%symbols(symbol_index)%deferred_capacity, error_msg)) return
        if (.not. emit_i64_store(context%session, &
                i64_immediate(context%session, storage_class), &
                context%symbols(symbol_index)%deferred_storage, error_msg)) return
    end subroutine set_character_storage

    pure function character_expression_storage_class(context) result(storage_class)
        ! The character intrinsic helpers (trim, adjustl, adjustr, repeat, ...)
        ! materialise their result on the heap inside an internal function, so
        ! it can outlive the frame, and on the stack at program scope. The
        ! descriptor that adopts such a buffer records the matching class.
        use, intrinsic :: iso_c_binding, only: c_int64_t
        type(lowering_context_t), intent(in) :: context
        integer(c_int64_t) :: storage_class

        if (context%in_internal_function) then
            storage_class = LOWERING_CHARACTER_STORAGE_OWNED
        else
            storage_class = LOWERING_CHARACTER_STORAGE_STACK
        end if
    end function character_expression_storage_class

    subroutine release_owned_character_storage(context, symbol_index, error_msg)
        ! Return the descriptor's heap block to the allocator when, and only
        ! when, this descriptor owns it (storage class
        ! CHARACTER_STORAGE_OWNED). Borrowed static and stack data is never
        ! freed through the descriptor. The class is reset to
        ! CHARACTER_STORAGE_NULL afterwards, so a second release, or a
        ! deallocate of an already-released variable, frees nothing.
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        character(len=:), allocatable, intent(out) :: error_msg

        call set_empty(error_msg)
        if (symbol_index <= 0) return
        if (.not. context%symbols(symbol_index)%has_character_ownership) return

        call release_character_descriptor_storage(context, &
            context%symbols(symbol_index)%deferred_data, &
            context%symbols(symbol_index)%deferred_storage, error_msg)
    end subroutine release_owned_character_storage

    subroutine bind_character_parameter_symbol(context, symbol_index, error_msg)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        character(len=:), allocatable, intent(out) :: error_msg

        context%symbols(symbol_index)%value_kind = VALUE_CHARACTER
        context%symbols(symbol_index)%is_deferred_character = .true.
        context%symbols(symbol_index)%has_character_value = .true.
        if (.not. emit_ptr_offset(context%session, &
            context%symbols(symbol_index)%address, 0_c_int64_t, &
            context%symbols(symbol_index)%deferred_data, error_msg)) return
        if (.not. emit_ptr_offset(context%session, &
            context%symbols(symbol_index)%address, 8_c_int64_t, &
            context%symbols(symbol_index)%deferred_length, error_msg)) return
        call set_empty(error_msg)
    end subroutine bind_character_parameter_symbol

    subroutine bind_fixed_character_parameter_symbol(context, symbol_index, &
                                                      character_length, &
                                                      error_msg)
        ! A fixed-length dummy (character(len=N)) is associated with exactly
        ! the first N bytes of the caller's actual: its own length is the
        ! declared N, not the caller's runtime length, so it binds as a
        ! non-deferred symbol whose data pointer is read from the shared
        ! {data, length} descriptor's data slot.
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        integer, intent(in) :: character_length
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: data_addr

        context%symbols(symbol_index)%value_kind = VALUE_CHARACTER
        context%symbols(symbol_index)%is_deferred_character = .false.
        context%symbols(symbol_index)%character_length = character_length
        context%symbols(symbol_index)%has_character_value = .true.
        if (.not. emit_ptr_offset(context%session, &
            context%symbols(symbol_index)%address, 0_c_int64_t, data_addr, &
            error_msg)) return
        if (.not. emit_ptr_load(context%session, data_addr, &
            context%symbols(symbol_index)%value, error_msg)) return
        call set_empty(error_msg)
    end subroutine bind_fixed_character_parameter_symbol

    subroutine char_length_arg_symbol(arena, node, context, symbol_index, &
                                       error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        integer, intent(out) :: symbol_index
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: id_name

        symbol_index = 0
        if (.not. allocated(node%arg_indices)) then
            error_msg = 'len/len_trim require one character argument'
            return
        end if
        if (size(node%arg_indices) /= 1) then
            error_msg = 'len/len_trim require exactly one argument'
            return
        end if
        if (.not. node_exists(arena, node%arg_indices(1))) then
            error_msg = 'len/len_trim argument does not reference an AST node'
            return
        end if
        if (is_identifier(arena, node%arg_indices(1))) then
            call get_identifier_name(arena, node%arg_indices(1), id_name, &
                                     error_msg)
            if (len_trim(error_msg) > 0) return
            symbol_index = find_symbol_compat(context, id_name)
        else
            call unsupported_feature_error('len/len_trim argument', node%line, &
                node%column, 'direct LIRIC session only supports len/len_trim '// &
                'of a character variable', error_msg)
            return
        end if
        if (symbol_index <= 0) then
            error_msg = 'len/len_trim argument is not a declared character: '// &
                        trim(node%name)
            return
        end if
        if (context%symbols(symbol_index)%value_kind /= VALUE_CHARACTER) then
            call unsupported_feature_error('len/len_trim argument', node%line, &
                node%column, 'direct LIRIC session only supports len/len_trim '// &
                'of a character variable', error_msg)
            return
        end if
        call set_empty(error_msg)
    end subroutine char_length_arg_symbol

    subroutine char_length_operands(context, symbol_index, data_ptr, length, &
                                     error_msg)
        ! Produce the data pointer and i32 length for a character symbol.
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        type(lr_operand_desc_t), intent(out) :: data_ptr
        type(lr_operand_desc_t), intent(out) :: length
        character(len=:), allocatable, intent(out) :: error_msg

        if (context%symbols(symbol_index)%is_deferred_character) then
            if (.not. emit_ptr_load(context%session, &
                context%symbols(symbol_index)%deferred_data, data_ptr, &
                error_msg)) return
            ! Length lives in an i64 slot; load its low word as i32 (lengths
            ! never approach 2**31 here).
            if (.not. emit_i32_load(context%session, &
                context%symbols(symbol_index)%deferred_length, length, &
                error_msg)) return
        else
            data_ptr = context%symbols(symbol_index)%value
            length = i32_immediate(context%session, &
                int(context%symbols(symbol_index)%character_length, c_int64_t))
            call set_empty(error_msg)
        end if
    end subroutine char_length_operands

    subroutine lower_fixed_char_expr_assignment(arena, value_index, context, &
                                                dest_symbol_index, error_msg)
        ! Assign a character expression (a variable, or a trim()/adjustl()/
        ! adjustr()/achar()/repeat() result) to a fixed-length target: copy
        ! min(source length, target length) bytes into a blank-filled buffer
        ! of the target's declared width, matching gfortran's fixed-length
        ! assignment padding/truncation.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: value_index
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: dest_symbol_index
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: src_data, src_len

        call char_expr_operands(arena, value_index, context, src_data, src_len, &
                                error_msg)
        if (len_trim(error_msg) > 0) return
        call store_fixed_character_value(context, dest_symbol_index, src_data, &
                                         src_len, error_msg)
    end subroutine lower_fixed_char_expr_assignment

    subroutine store_fixed_character_value(context, dest_symbol_index, src_data, &
                                           src_len, error_msg)
        ! Copy a character value into a fixed-length destination: blank-fill a
        ! buffer of the declared width, copy min(source length, declared
        ! width) bytes, and null-terminate at the fixed end. The destination
        ! holds a copy, never an alias of the source buffer, so a source that
        ! is a temporary can be released straight after this call.
        use, intrinsic :: iso_c_binding, only: c_int64_t
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: dest_symbol_index
        type(lr_operand_desc_t), intent(in) :: src_data
        type(lr_operand_desc_t), intent(in) :: src_len
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: dest_len_i32, cmp
        type(lr_operand_desc_t) :: copy_len, copy_len64, buf, offset, null_ptr
        character(len=64) :: null_name
        integer :: dest_len

        dest_len = context%symbols(dest_symbol_index)%character_length
        dest_len_i32 = i32_immediate(context%session, int(dest_len, c_int64_t))

        if (.not. emit_alloca_bytes(context%session, &
            i64_immediate(context%session, int(dest_len + 1, c_int64_t)), buf, &
            error_msg)) return
        call fill_spaces(context, buf, dest_len_i32, error_msg)
        if (len_trim(error_msg) > 0) return

        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_SLE, src_len, &
                dest_len_i32, cmp, error_msg)) return
        call select_value(context, cmp, src_len, dest_len_i32, copy_len, &
                          error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_liric_i32_to_i64(context%session, copy_len, copy_len64, &
                error_msg)) return
        if (.not. emit_memcpy(context%session, buf, src_data, copy_len64, &
                error_msg)) return

        ! Null-terminate at the fixed end so the print path stops after dest_len.
        if (.not. emit_i64_binary(context%session, LR_OP_ADD, buf, &
            i64_immediate(context%session, int(dest_len, c_int64_t)), offset, &
            error_msg)) return
        context%string_literal_count = context%string_literal_count + 1
        null_name = ffc_unit_global_name( &
            context, 'fassign.term.', context%string_literal_count)
        call materialize_liric_string(context%session, trim(null_name), '', &
                                      null_ptr, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_memcpy(context%session, offset, null_ptr, &
            i64_immediate(context%session, 1_c_int64_t), error_msg)) return

        if ((context%symbols(dest_symbol_index)%is_target .or. &
             context%symbols(dest_symbol_index)%is_pointer) .and. &
            context%symbols(dest_symbol_index)%has_address) then
            if (.not. emit_memcpy(context%session, &
                context%symbols(dest_symbol_index)%address, buf, &
                i64_immediate(context%session, int(dest_len + 1, c_int64_t)), &
                error_msg)) return
            context%symbols(dest_symbol_index)%value = &
                context%symbols(dest_symbol_index)%address
        else
            context%symbols(dest_symbol_index)%value = buf
        end if
        context%symbols(dest_symbol_index)%has_character_value = .true.
        call set_empty(error_msg)
    end subroutine store_fixed_character_value

    subroutine lower_runtime_fixed_char_assignment(arena, value_index, context, &
                                                    dest_symbol_index, error_msg)
        ! Assignment to an automatic character whose declared width was captured
        ! at runtime. The descriptor length remains that declaration-time width;
        ! only its data pointer changes. This is deliberately distinct from an
        ! allocatable deferred character, whose assignment changes its length.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: value_index
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: dest_symbol_index
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: source_data, source_len, target_len
        type(lr_operand_desc_t) :: target_len64, buffer_size, buffer, copy_len
        type(lr_operand_desc_t) :: copy_len64, source_fits, null_pos
        integer(c_int64_t) :: buffer_storage_class

        buffer_storage_class = LOWERING_CHARACTER_STORAGE_STACK
        call char_expr_operands(arena, value_index, context, source_data, &
                                source_len, error_msg)
        if (len_trim(error_msg) > 0) return

        ! The data slot may be null before the first assignment. The length slot
        ! was initialized at declaration time and remains immutable here.
        if (.not. emit_i32_load(context%session, &
                context%symbols(dest_symbol_index)%deferred_length, target_len, &
                error_msg)) return
        if (.not. emit_liric_i32_to_i64(context%session, target_len, &
                target_len64, error_msg)) return
        if (.not. emit_i64_binary(context%session, LR_OP_ADD, target_len64, &
                i64_immediate(context%session, 1_c_int64_t), buffer_size, &
                error_msg)) return

        ! A runtime-fixed local used to form a function result must outlive the
        ! function frame. Use the same heap allocation convention as the other
        ! runtime character-expression helpers in internal functions.
        if (context%in_internal_function) then
            if (.not. emit_malloc(context%session, buffer_size, buffer, &
                    error_msg)) return
            buffer_storage_class = LOWERING_CHARACTER_STORAGE_OWNED
        else
            if (.not. emit_alloca_bytes(context%session, buffer_size, buffer, &
                    error_msg)) return
            buffer_storage_class = LOWERING_CHARACTER_STORAGE_STACK
        end if
        call fill_spaces(context, buffer, target_len, error_msg)
        if (len_trim(error_msg) > 0) return

        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_SLE, source_len, &
                target_len, source_fits, error_msg)) return
        call select_value(context, source_fits, source_len, target_len, copy_len, &
                          error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_liric_i32_to_i64(context%session, copy_len, copy_len64, &
                error_msg)) return
        if (.not. emit_memcpy(context%session, buffer, source_data, copy_len64, &
                error_msg)) return

        if (.not. emit_i64_binary(context%session, LR_OP_ADD, buffer, &
                target_len64, null_pos, error_msg)) return
        if (.not. emit_liric_store_char_byte(context%session, null_pos, &
                i32_immediate(context%session, 0_c_int64_t), &
                i32_immediate(context%session, 0_c_int64_t), error_msg)) return

        ! The declared width is immutable here, so the freshly built buffer
        ! replaces whatever the descriptor held. Release the former block when
        ! this descriptor owned it, then record the new block's class so the
        ! consumer of a returned result knows whether to free it.
        call release_owned_character_storage(context, dest_symbol_index, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_ptr_store(context%session, buffer, &
                context%symbols(dest_symbol_index)%deferred_data, error_msg)) return
        call set_character_storage(context, dest_symbol_index, target_len64, &
                                   buffer_storage_class, error_msg)
        if (len_trim(error_msg) > 0) return
        context%symbols(dest_symbol_index)%value = buffer
        context%symbols(dest_symbol_index)%has_character_value = .true.
        call set_empty(error_msg)
    end subroutine lower_runtime_fixed_char_assignment

    subroutine lower_deferred_trim_assignment(arena, trim_index, context, &
                                              dest_symbol_index, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: trim_index
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: dest_symbol_index
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: data_ptr
        type(lr_operand_desc_t) :: length
        type(lr_operand_desc_t) :: length64

        call char_expr_operands(arena, trim_index, context, data_ptr, length, &
                                error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_liric_i32_to_i64(context%session, length, length64, &
                                        error_msg)) return
        ! char_expr_operands has already materialised the buffer, so the
        ! former owned block can be released before the rebind. Inside an
        ! internal function that buffer is heap storage, because it has to
        ! outlive the frame that produced it; at program scope it is a stack
        ! temporary the descriptor only borrows.
        call release_owned_character_storage(context, dest_symbol_index, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_ptr_store(context%session, data_ptr, &
            context%symbols(dest_symbol_index)%deferred_data, error_msg)) return
        if (.not. emit_i64_store(context%session, length64, &
            context%symbols(dest_symbol_index)%deferred_length, error_msg)) return
        call set_character_storage(context, dest_symbol_index, length64, &
                                   character_expression_storage_class(context), &
                                   error_msg)
        if (len_trim(error_msg) > 0) return
        context%symbols(dest_symbol_index)%value = data_ptr
        context%symbols(dest_symbol_index)%has_character_value = .true.
        call set_empty(error_msg)
    end subroutine lower_deferred_trim_assignment

    subroutine lower_i32_len_intrinsic(arena, node, context, value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: data_ptr

        if (.not. allocated(node%arg_indices)) then
            error_msg = 'len requires one character argument'
            return
        end if
        ! len of a character expression (variable or trim()): the descriptor
        ! length. char_expr_operands materialises trim() as a trimmed buffer.
        call char_expr_operands(arena, node%arg_indices(1), context, data_ptr, &
                                value, error_msg)
    end subroutine lower_i32_len_intrinsic

    logical function is_trim_call(arena, node_index, context)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context

        is_trim_call = .false.
        if (.not. node_exists(arena, node_index)) return
        select type (n => arena%entries(node_index)%node)
        type is (call_or_subscript_node)
            if (.not. n%is_array_access .and. allocated(n%name)) then
                is_trim_call = same_name(n%name, 'trim') .and. &
                               .not. is_contained_char_result_call( &
                                   arena, node_index, context)
            end if
        end select
    end function is_trim_call

    subroutine character_array_symbol_element_operands(context, symbol_index, &
                                                       linear_index, data_ptr, &
                                                       length, error_msg)
        use, intrinsic :: iso_c_binding, only: c_int64_t
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        integer, intent(in) :: linear_index
        type(lr_operand_desc_t), intent(out) :: data_ptr
        type(lr_operand_desc_t), intent(out) :: length
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: element_index

        element_index = i32_immediate(context%session, &
                                      int(linear_index, c_int64_t))
        call character_array_element_address(context, symbol_index, element_index, &
                                             data_ptr, error_msg)
        if (len_trim(error_msg) > 0) return
        length = i32_immediate(context%session, &
                               int(context%symbols(symbol_index)%character_length, &
                                   c_int64_t))
        call set_empty(error_msg)
    end subroutine character_array_symbol_element_operands

    recursive subroutine char_expr_operands(arena, node_index, context, &
                                            data_ptr, length, error_msg)
        ! Resolve a scalar character expression (a variable, a // concatenation,
        ! or a character intrinsic such as trim()) to its data pointer and i32
        ! length. Concatenations and trim() materialise a fresh null-terminated
        ! buffer holding the result text.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: data_ptr
        type(lr_operand_desc_t), intent(out) :: length
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: symbol_index
        character(len=:), allocatable :: lit_value, lit_type
        character(len=:), allocatable :: id_name

        if (.not. node_exists(arena, node_index)) then
            error_msg = 'character expression does not reference an AST node'
            return
        end if
        if (is_binary_op(arena, node_index)) then
            call concat_char_operands(arena, node_index, context, data_ptr, &
                                      length, error_msg)
            return
        end if
        if (is_trim_call(arena, node_index, context)) then
            select type (n => arena%entries(node_index)%node)
            type is (call_or_subscript_node)
                call lower_trim_deferred(arena, n, context, data_ptr, length, &
                                         error_msg)
            end select
            return
        end if
        if (is_named_char_call(arena, node_index, 'achar') .or. &
            is_named_char_call(arena, node_index, 'char')) then
            select type (n => arena%entries(node_index)%node)
            type is (call_or_subscript_node)
                call lower_achar_deferred(arena, n, context, data_ptr, length, &
                                          error_msg)
            end select
            return
        end if
        if (is_named_char_call(arena, node_index, 'adjustl')) then
            select type (n => arena%entries(node_index)%node)
            type is (call_or_subscript_node)
                call lower_adjust_deferred(arena, n, context, .true., data_ptr, &
                                           length, error_msg)
            end select
            return
        end if
        if (is_named_char_call(arena, node_index, 'adjustr')) then
            select type (n => arena%entries(node_index)%node)
            type is (call_or_subscript_node)
                call lower_adjust_deferred(arena, n, context, .false., data_ptr, &
                                           length, error_msg)
            end select
            return
        end if
        if (is_named_char_call(arena, node_index, 'repeat')) then
            select type (n => arena%entries(node_index)%node)
            type is (call_or_subscript_node)
                call lower_repeat_deferred(arena, n, context, data_ptr, length, &
                                           error_msg)
            end select
            return
        end if
        if (is_character_substring(arena, node_index, context)) then
            call substring_operands(arena, node_index, context, data_ptr, &
                                    length, error_msg)
            return
        end if
        if (is_character_array_element(arena, node_index, context)) then
            select type (n => arena%entries(node_index)%node)
            type is (call_or_subscript_node)
                call char_array_element_operands(arena, n, context, data_ptr, &
                                                 length, error_msg)
            end select
            return
        end if
        if (is_contained_char_result_call(arena, node_index, context)) then
            call char_result_call_operands(arena, node_index, context, data_ptr, &
                                           length, error_msg)
            return
        end if
        if (node_exists(arena, node_index)) then
            select type (n => arena%entries(node_index)%node)
            type is (component_access_node)
                if (derived_component_access_kind(arena, n, context) == &
                    VALUE_CHARACTER) then
                    call char_component_operands(arena, n, context, data_ptr, &
                                                 length, error_msg)
                    return
                end if
            end select
        end if
        if (is_literal(arena, node_index)) then
            if (.not. is_character_literal(arena, node_index)) then
                call unsupported_feature_error('character expression', 0, 0, &
                    'direct LIRIC session only supports a character variable, '// &
                    'literal, or trim() here', error_msg)
                return
            end if
            call get_literal_info(arena, node_index, lit_value, lit_type, &
                                  error_msg)
            if (len_trim(error_msg) > 0) return
            call char_literal_operands(context, lit_value, data_ptr, length, &
                                       error_msg)
            return
        end if

        if (is_identifier(arena, node_index)) then
            call get_identifier_name(arena, node_index, id_name, error_msg)
            if (len_trim(error_msg) > 0) return
            ! C_NULL_CHAR is a named ISO_C_BINDING constant, but its semantic
            ! value is a single zero byte rather than a regular character
            ! variable in the lowering symbol table.  Materialising an empty
            ! literal gives the same pointer-to-terminator representation and
            ! lets concatenation build the required C string (#584).
            if (same_name(id_name, 'c_null_char')) then
                call char_literal_operands(context, '""', data_ptr, length, &
                                           error_msg)
                return
            end if
            symbol_index = find_symbol_compat(context, id_name)
            if (symbol_index <= 0 .or. &
                context%symbols(max(symbol_index, 1))%value_kind /= &
                VALUE_CHARACTER) then
                call unsupported_feature_error('character expression', 0, 0, &
                    'direct LIRIC session only supports a character variable '// &
                    'or trim() here', error_msg)
                return
            end if
            call char_length_operands(context, symbol_index, data_ptr, length, &
                                      error_msg)
            return
        end if

        call unsupported_feature_error('character expression', 0, 0, &
            'direct LIRIC session only supports a character variable or '// &
            'trim() here', error_msg)
    end subroutine char_expr_operands

    subroutine reject_constant_substring_overrun(arena, context, symbol_index, &
                                                 lower_index, upper_index, &
                                                 error_msg)
        ! Reject s(l:u) whose constant bounds fall outside a parent of known
        ! declared width. Only a fixed-length parent has a compile-time width;
        ! a deferred-length parent's length is a run-time value, so nothing is
        ! decidable here and the substring is accepted. Non-constant bounds are
        ! likewise left alone.
        use, intrinsic :: iso_c_binding, only: c_int64_t
        type(ast_arena_t), intent(in) :: arena
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: symbol_index
        integer, intent(in) :: lower_index
        integer, intent(in) :: upper_index
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int64_t) :: lower_value, upper_value
        logical :: lower_known, upper_known
        integer :: declared_length

        call set_empty(error_msg)
        if (context%symbols(symbol_index)%is_deferred_character) return
        declared_length = context%symbols(symbol_index)%character_length
        if (declared_length <= 0) return

        call constant_i64_value(arena, lower_index, lower_value, lower_known)
        call constant_i64_value(arena, upper_index, upper_value, upper_known)

        if (lower_known) then
            if (lower_value < 1_c_int64_t) then
                error_msg = 'substring start index is below the string length'
                return
            end if
        end if
        if (upper_known) then
            if (upper_value > int(declared_length, c_int64_t)) then
                error_msg = 'substring end index exceeds the string length'
                return
            end if
        end if
    end subroutine reject_constant_substring_overrun

    subroutine constant_i64_value(arena, node_index, value, is_known)
        ! Fold a substring bound to a compile-time integer when it is a plain
        ! integer literal. Anything else is a run-time expression.
        use, intrinsic :: iso_c_binding, only: c_int64_t
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        integer(c_int64_t), intent(out) :: value
        logical, intent(out) :: is_known
        character(len=:), allocatable :: lit_value, lit_type, lit_err
        integer :: ios

        value = 0_c_int64_t
        is_known = .false.
        if (node_index <= 0) return
        if (.not. node_exists(arena, node_index)) return
        if (.not. is_literal(arena, node_index)) return
        call get_literal_info(arena, node_index, lit_value, lit_type, lit_err)
        if (len_trim(lit_err) > 0) return
        if (verify(trim(adjustl(lit_value)), '0123456789') /= 0) return
        read (lit_value, *, iostat=ios) value
        is_known = ios == 0
    end subroutine constant_i64_value

    subroutine lower_deferred_char_view_assignment(arena, value_index, context, &
                                                   symbol_index, error_msg)
        ! str = other(l:u), where str is character(len=:), allocatable. The
        ! right-hand side is a borrowed view into another variable's storage,
        ! so the destination takes its own copy at the view's length. The copy
        ! is made before the destination releases the block it held, which is
        ! what keeps `s = s(2:4)` correct.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: value_index
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: data_ptr, len_i32, len_i64, buf

        call substring_operands(arena, value_index, context, data_ptr, len_i32, &
                                error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_liric_i32_to_i64(context%session, len_i32, len_i64, &
                error_msg)) return

        call copy_character_bytes_to_owned(context, data_ptr, len_i64, buf, &
                                           error_msg)
        if (len_trim(error_msg) > 0) return
        call release_owned_character_storage(context, symbol_index, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_ptr_store(context%session, buf, &
                context%symbols(symbol_index)%deferred_data, error_msg)) return
        if (.not. emit_i64_store(context%session, len_i64, &
                context%symbols(symbol_index)%deferred_length, error_msg)) return
        call set_character_storage(context, symbol_index, len_i64, &
                                   LOWERING_CHARACTER_STORAGE_OWNED, error_msg)
        if (len_trim(error_msg) > 0) return
        context%symbols(symbol_index)%value = buf
        context%symbols(symbol_index)%has_character_value = .true.
        call set_empty(error_msg)
    end subroutine lower_deferred_char_view_assignment

    subroutine lower_character_substring_assignment(arena, node, context, error_msg)
        ! Assign a scalar character expression to s(l:u).  FortFront uses the
        ! array_slice_node representation for both array sections and
        ! substrings; the assignment dispatcher selects this path after the
        ! scalar-character test.  Evaluate the RHS into a private buffer before
        ! blank-filling the target so overlapping views retain Fortran's
        ! evaluate-before-store semantics.
        type(ast_arena_t), intent(in) :: arena
        type(assignment_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: target_data, target_len
        type(lr_operand_desc_t) :: source_data, source_len
        type(lr_operand_desc_t) :: source_copy, copy_len, copy_len64, fits

        call substring_operands(arena, node%target_index, context, target_data, &
                                target_len, error_msg)
        if (len_trim(error_msg) > 0) return
        call char_expr_operands(arena, node%value_index, context, source_data, &
                                source_len, error_msg)
        if (len_trim(error_msg) > 0) return
        call materialize_character_view(context, source_data, source_len, &
                                        source_copy, error_msg)
        if (len_trim(error_msg) > 0) return

        call fill_spaces(context, target_data, target_len, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_SLE, source_len, &
                target_len, fits, error_msg)) return
        call select_value(context, fits, source_len, target_len, copy_len, &
                          error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_liric_i32_to_i64(context%session, copy_len, copy_len64, &
                error_msg)) return
        if (.not. emit_memcpy(context%session, target_data, source_copy, &
                copy_len64, error_msg)) return
        call set_empty(error_msg)
    end subroutine lower_character_substring_assignment

    subroutine materialize_character_view(context, data_ptr, length, buf, &
                                          error_msg)
        ! Copy a borrowed character view into a fresh null-terminated buffer of
        ! exactly its Fortran length. A view has no terminator of its own, so
        ! anything that reads it as a C string needs this first. The buffer
        ! follows the same convention as the other character expression
        ! helpers: heap inside an internal function, where it has to outlive
        ! the frame, and stack at program scope.
        use, intrinsic :: iso_c_binding, only: c_int64_t
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(in) :: data_ptr
        type(lr_operand_desc_t), intent(in) :: length
        type(lr_operand_desc_t), intent(out) :: buf
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: length64, bytes, null_pos

        if (.not. emit_liric_i32_to_i64(context%session, length, length64, &
                error_msg)) return
        if (.not. emit_i64_binary(context%session, LR_OP_ADD, length64, &
                i64_immediate(context%session, 1_c_int64_t), bytes, &
                error_msg)) return
        if (context%in_internal_function) then
            if (.not. emit_malloc(context%session, bytes, buf, error_msg)) return
        else
            if (.not. emit_alloca_bytes(context%session, bytes, buf, error_msg)) &
                return
        end if
        if (.not. emit_memcpy(context%session, buf, data_ptr, length64, &
                error_msg)) return
        if (.not. emit_i64_binary(context%session, LR_OP_ADD, buf, length64, &
                null_pos, error_msg)) return
        if (.not. emit_liric_store_char_byte(context%session, null_pos, &
                i32_immediate(context%session, 0_c_int64_t), &
                i32_immediate(context%session, 0_c_int64_t), error_msg)) return
        call set_empty(error_msg)
    end subroutine materialize_character_view

    logical function is_character_substring(arena, node_index, context) &
            result(is_substring)
        ! True when node is s(l:u) on a scalar character variable. The parser
        ! produces the same array_slice_node shape as an array section, so the
        ! base symbol's rank is what separates a substring from a section.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        character(len=:), allocatable :: base_name, name_err
        integer :: symbol_index

        is_substring = .false.
        if (.not. node_exists(arena, node_index)) return
        select type (n => arena%entries(node_index)%node)
        type is (array_slice_node)
            if (n%num_dimensions /= 1) return
            call identifier_name(arena, n%array_index, base_name, name_err)
            if (len_trim(name_err) > 0) return
            symbol_index = find_symbol_compat(context, base_name)
            if (symbol_index <= 0) return
            if (context%symbols(symbol_index)%value_kind /= VALUE_CHARACTER) return
            if (context%symbols(symbol_index)%is_array) return
            is_substring = .true.
        end select
    end function is_character_substring

    subroutine substring_operands(arena, node_index, context, data_ptr, length, &
                                  error_msg)
        ! s(l:u) on a scalar character: a borrowed view into s's own storage.
        ! The view's data pointer is s's data advanced by l - 1 bytes and its
        ! Fortran length is u - l + 1; nothing is allocated and nothing is
        ! copied, so the view stays valid exactly as long as s's storage does.
        ! An omitted bound defaults to 1 for the lower and len(s) for the upper.
        use, intrinsic :: iso_c_binding, only: c_int64_t
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: data_ptr
        type(lr_operand_desc_t), intent(out) :: length
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: base_name
        integer :: symbol_index, bounds_index, lower_index, upper_index
        type(lr_operand_desc_t) :: base_data, base_len, lower_i32, upper_i32
        type(lr_operand_desc_t) :: zero_based, span

        call set_empty(error_msg)
        bounds_index = 0
        lower_index = -1
        upper_index = -1
        select type (n => arena%entries(node_index)%node)
        type is (array_slice_node)
            call identifier_name(arena, n%array_index, base_name, error_msg)
            if (len_trim(error_msg) > 0) return
            bounds_index = n%bounds_indices(1)
        class default
            error_msg = 'expected a substring reference'
            return
        end select

        symbol_index = find_symbol_compat(context, base_name)
        if (symbol_index <= 0) then
            error_msg = 'substring base was not declared: '//trim(base_name)
            return
        end if
        call char_length_operands(context, symbol_index, base_data, base_len, &
                                  error_msg)
        if (len_trim(error_msg) > 0) return

        if (.not. node_exists(arena, bounds_index)) then
            error_msg = 'substring bounds do not reference an AST node'
            return
        end if
        ! The parser produces either shape for l:u depending on context.
        select type (b => arena%entries(bounds_index)%node)
        type is (array_bounds_node)
            if (b%stride_index > 0) then
                call unsupported_feature_error('substring', 0, 0, &
                    'a substring has no stride', error_msg)
                return
            end if
            lower_index = b%lower_bound_index
            upper_index = b%upper_bound_index
        type is (range_expression_node)
            if (b%stride_index > 0) then
                call unsupported_feature_error('substring', 0, 0, &
                    'a substring has no stride', error_msg)
                return
            end if
            lower_index = b%start_index
            upper_index = b%end_index
        class default
            error_msg = 'substring bounds are not a range'
            return
        end select

        ! A fixed-length parent has a known width, so constant bounds outside
        ! it are rejected here rather than reading past the variable, matching
        ! gfortran's "substring end index exceeds the string length".
        call reject_constant_substring_overrun(arena, context, symbol_index, &
                                               lower_index, upper_index, &
                                               error_msg)
        if (len_trim(error_msg) > 0) return

        if (lower_index > 0) then
            call lower_i32_expression(arena, lower_index, context, lower_i32, &
                                      error_msg)
            if (len_trim(error_msg) > 0) return
        else
            lower_i32 = i32_immediate(context%session, 1_c_int64_t)
        end if
        if (upper_index > 0) then
            call lower_i32_expression(arena, upper_index, context, upper_i32, &
                                      error_msg)
            if (len_trim(error_msg) > 0) return
        else
            upper_i32 = base_len
        end if

        ! data = base + (lower - 1)
        if (.not. emit_i32_binary(context%session, LR_OP_SUB, lower_i32, &
                i32_immediate(context%session, 1_c_int64_t), zero_based, &
                error_msg)) return
        call ptr_plus_i32(context, base_data, zero_based, data_ptr, error_msg)
        if (len_trim(error_msg) > 0) return

        ! length = upper - lower + 1
        if (.not. emit_i32_binary(context%session, LR_OP_SUB, upper_i32, &
                zero_based, span, error_msg)) return
        length = span
        call set_empty(error_msg)
    end subroutine substring_operands

    logical function is_character_array_element(arena, node_index, context) &
            result(is_elem)
        ! True when node is a subscripted reference into a declared character
        ! array (arr(i)), producing a scalar character element.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context

        integer :: sym

        is_elem = .false.
        if (.not. node_exists(arena, node_index)) return
        select type (n => arena%entries(node_index)%node)
        type is (call_or_subscript_node)
            ! The symbol table decides what the name is. The parser's
            ! is_array_access flag is not set on every subscripted reference
            ! (an implied-do object, for one), so trusting it made the same
            ! reference a character element in one context and not in another.
            if (n%is_array_access) then
                is_elem = array_access_value_kind(n, context) == VALUE_CHARACTER
                return
            end if
            ! An implied-do object reaches here without the flag set, so fall
            ! back on the symbol table for a fixed character array, whose
            ! elements live in this file's contiguous block. An allocatable
            ! character array keeps its own descriptor-based element path.
            if (.not. allocated(n%name)) return
            sym = find_symbol_compat(context, n%name)
            if (sym <= 0) return
            if (.not. context%symbols(sym)%is_array) return
            if (context%symbols(sym)%is_allocatable) return
            is_elem = context%symbols(sym)%value_kind == VALUE_CHARACTER
        end select
    end function is_character_array_element

    subroutine char_array_element_operands(arena, node, context, data_ptr, &
                                           length, error_msg)
        ! Resolve a character array element to its data pointer and the array's
        ! declared element length.
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: data_ptr
        type(lr_operand_desc_t), intent(out) :: length
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: slot_addr
        integer :: symbol_index

        symbol_index = 0
        if (allocated(node%name)) symbol_index = find_symbol_compat(context, node%name)
        if (symbol_index > 0) then
            if (context%symbols(symbol_index)%is_allocatable) then
                ! An allocatable character array addresses its element pointer
                ! slot through its runtime descriptor, not the fixed [N x ptr]
                ! element_address buffer, so route to the allocatable path.
                call lower_allocatable_element_address(arena, node, symbol_index, &
                                                       context, slot_addr, error_msg)
                if (len_trim(error_msg) > 0) return
                if (.not. emit_ptr_load(context%session, slot_addr, data_ptr, &
                                        error_msg)) return
                length = i32_immediate(context%session, &
                    int(context%symbols(symbol_index)%character_length, c_int64_t))
                call set_empty(error_msg)
                return
            end if
        end if

        ! A fixed character array stores its elements contiguously, so the
        ! element's address is its data pointer: the element is a borrowed view
        ! of character_length bytes inside the array's own block.
        call lower_character_array_element_slot(arena, node, context, &
                                                symbol_index, data_ptr, error_msg)
        if (len_trim(error_msg) > 0) return
        length = i32_immediate(context%session, &
            int(context%symbols(symbol_index)%character_length, c_int64_t))
        call set_empty(error_msg)
    end subroutine char_array_element_operands

    logical function is_contained_char_result_call(arena, node_index, context) &
            result(is_call)
        ! True when node is a plain call to a character-returning contained
        ! function (descriptor-ABI result), not an array subscript.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context

        is_call = .false.
        if (.not. node_exists(arena, node_index)) return
        select type (n => arena%entries(node_index)%node)
        type is (call_or_subscript_node)
            if (.not. n%is_array_access .and. allocated(n%name)) &
                is_call = is_contained_deferred_char_function(context, n%name)
        end select
    end function is_contained_char_result_call

    recursive logical function is_character_concat_actual(arena, node_index, &
                                                          context) result(is_char)
        ! Character actuals are recognized by the argument path through
        ! is_char_expr_call().  Keep this classifier separate from
        ! is_character_concat(), whose operand walk calls is_char_expr_call()
        ! and would therefore recurse if it were used here directly.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        character(len=:), allocatable :: op, op_error
        integer :: left_index, right_index, line, column

        is_char = .false.
        if (.not. node_exists(arena, node_index)) return
        if (.not. is_binary_op(arena, node_index)) return
        call get_binary_op_info(arena, node_index, op, left_index, right_index, &
                                line, column, op_error)
        if (len_trim(op_error) > 0) return
        if (trim(op) /= '//') return
        if (.not. is_direct_character_operand(arena, left_index, context)) return
        is_char = is_direct_character_operand(arena, right_index, context)
    end function is_character_concat_actual

    recursive logical function is_direct_character_operand(arena, node_index, &
                                                            context) result(is_char)
        ! A deliberately non-general operand test for the actual-argument
        ! bridge above.  It recognizes the scalar character values that
        ! char_expr_operands() can materialize, including C_NULL_CHAR.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        character(len=:), allocatable :: name, name_error
        integer :: symbol_index

        is_char = .false.
        if (.not. node_exists(arena, node_index)) return
        if (is_character_literal(arena, node_index)) then
            is_char = .true.
            return
        end if
        if (is_character_concat_actual(arena, node_index, context)) then
            is_char = .true.
            return
        end if
        if (is_character_substring(arena, node_index, context)) then
            is_char = .true.
            return
        end if
        if (is_contained_char_result_call(arena, node_index, context)) then
            is_char = .true.
            return
        end if
        if (is_named_char_call(arena, node_index, 'achar') .or. &
            is_named_char_call(arena, node_index, 'char') .or. &
            is_named_char_call(arena, node_index, 'adjustl') .or. &
            is_named_char_call(arena, node_index, 'adjustr') .or. &
            is_named_char_call(arena, node_index, 'repeat')) then
            is_char = .true.
            return
        end if
        if (.not. is_identifier(arena, node_index)) return
        call get_identifier_name(arena, node_index, name, name_error)
        if (len_trim(name_error) > 0) return
        if (same_name(name, 'c_null_char')) then
            is_char = .true.
            return
        end if
        symbol_index = resolve_symbol_at_node(context, node_index, name)
        if (symbol_index > 0) is_char = &
            context%symbols(symbol_index)%value_kind == VALUE_CHARACTER
    end function is_direct_character_operand

    logical function is_char_expr_call(arena, node_index, context)
        ! A call that produces a character value usable in print/assign/len.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context

        is_char_expr_call = is_contained_char_result_call(arena, node_index, &
                                                          context) .or. &
                            is_trim_call(arena, node_index, context) .or. &
                            is_named_char_call(arena, node_index, 'achar') .or. &
                            is_named_char_call(arena, node_index, 'char') .or. &
                            is_named_char_call(arena, node_index, 'adjustl') .or. &
                            is_named_char_call(arena, node_index, 'adjustr') .or. &
                            is_named_char_call(arena, node_index, 'repeat') .or. &
                            is_character_concat_actual(arena, node_index, context)
    end function is_char_expr_call

    recursive logical function is_character_concat(arena, node_index, context) &
            result(is_concat)
        ! True when node is a // binary op whose operands resolve to character
        ! data (a literal, a character variable, a character intrinsic, or a
        ! nested // chain). Used to route print/assign through the runtime
        ! concatenation path instead of the integer fallback.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        character(len=:), allocatable :: bin_op, bin_err
        integer :: left_idx, right_idx, bin_line, bin_col

        is_concat = .false.
        if (.not. node_exists(arena, node_index)) return
        if (.not. is_binary_op(arena, node_index)) return
        call get_binary_op_info(arena, node_index, bin_op, left_idx, right_idx, &
                                bin_line, bin_col, bin_err)
        if (len_trim(bin_err) > 0) return
        if (trim(bin_op) /= '//') return
        is_concat = is_character_operand(arena, left_idx, context) .or. &
                    is_character_operand(arena, right_idx, context)
    end function is_character_concat

    recursive logical function is_character_operand(arena, node_index, context) &
            result(is_char)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        character(len=:), allocatable :: id_name, id_err
        integer :: symbol_index

        is_char = .false.
        if (.not. node_exists(arena, node_index)) return
        if (is_literal(arena, node_index)) then
            is_char = is_character_literal(arena, node_index)
            return
        end if
        if (is_char_expr_call(arena, node_index, context)) then
            is_char = .true.
            return
        end if
        if (is_character_concat(arena, node_index, context)) then
            is_char = .true.
            return
        end if
        if (is_character_array_element(arena, node_index, context)) then
            is_char = .true.
            return
        end if
        if (is_contained_char_result_call(arena, node_index, context)) then
            is_char = .true.
            return
        end if
        select type (n => arena%entries(node_index)%node)
        type is (component_access_node)
            is_char = derived_component_access_kind(arena, n, context) == &
                      VALUE_CHARACTER
            return
        end select
        if (is_identifier(arena, node_index)) then
            call get_identifier_name(arena, node_index, id_name, id_err)
            if (len_trim(id_err) > 0) return
            symbol_index = find_symbol_compat(context, id_name)
            if (symbol_index > 0) &
                is_char = context%symbols(symbol_index)%value_kind == &
                          VALUE_CHARACTER
        end if
    end function is_character_operand

    subroutine lower_character_condition(arena, bin_op, left_idx, right_idx, &
                                         context, value, error_msg)
        ! A character comparison (==, /=, <, <=, >, >=) with Fortran's
        ! blank-padded lexical ordering. char_compare reduces both operands
        ! to a three-way -1/0/1 result; the same integer predicate used for
        ! numeric comparisons then tests that result against zero.
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: bin_op
        integer, intent(in) :: left_idx, right_idx
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: a_data, a_len, b_data, b_len, cmp
        integer(c_int) :: pred

        call char_expr_operands(arena, left_idx, context, a_data, a_len, &
                                error_msg)
        if (len_trim(error_msg) > 0) return
        call char_expr_operands(arena, right_idx, context, b_data, b_len, &
                                error_msg)
        if (len_trim(error_msg) > 0) return
        call char_compare(context, a_data, a_len, b_data, b_len, cmp, &
                          error_msg)
        if (len_trim(error_msg) > 0) return
        call integer_compare_predicate(bin_op, pred, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_liric_i32_icmp(context%session, pred, cmp, &
                i32_immediate(context%session, 0_c_int64_t), value, &
                error_msg)) return
    end subroutine lower_character_condition

    logical function is_named_char_call(arena, node_index, name)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        character(len=*), intent(in) :: name

        is_named_char_call = .false.
        if (.not. node_exists(arena, node_index)) return
        select type (n => arena%entries(node_index)%node)
        type is (call_or_subscript_node)
            if (.not. n%is_array_access .and. allocated(n%name)) &
                is_named_char_call = same_name(n%name, name)
        end select
    end function is_named_char_call

    subroutine lower_achar_deferred(arena, node, context, out_data, out_length, &
                                    error_msg)
        ! achar(i): a length-1 character whose byte is the low 8 bits of i.
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: out_data
        type(lr_operand_desc_t), intent(out) :: out_length
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: code
        type(lr_operand_desc_t) :: buf

        if (.not. allocated(node%arg_indices)) then
            error_msg = 'achar requires one integer argument'
            return
        end if
        call lower_i32_expression(arena, node%arg_indices(1), context, code, &
                                  error_msg)
        if (len_trim(error_msg) > 0) return
        ! 2 bytes: the character plus a null terminator.
        if (context%in_internal_function) then
            if (.not. emit_malloc(context%session, &
                    i64_immediate(context%session, 2_c_int64_t), buf, error_msg)) &
                return
        else
            if (.not. emit_alloca_bytes(context%session, &
                    i64_immediate(context%session, 2_c_int64_t), buf, error_msg)) &
                return
        end if
        if (.not. emit_liric_store_char_byte(context%session, buf, &
                i32_immediate(context%session, 0_c_int64_t), code, error_msg)) &
            return
        if (.not. emit_liric_store_char_byte(context%session, buf, &
                i32_immediate(context%session, 1_c_int64_t), &
                i32_immediate(context%session, 0_c_int64_t), error_msg)) return
        out_data = buf
        out_length = i32_immediate(context%session, 1_c_int64_t)
        call set_empty(error_msg)
    end subroutine lower_achar_deferred

    subroutine lower_iachar_intrinsic(arena, node, context, value, error_msg)
        ! iachar(c): the ASCII code of the first byte of c.
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: data_ptr
        type(lr_operand_desc_t) :: length

        if (.not. allocated(node%arg_indices)) then
            error_msg = 'iachar requires one character argument'
            return
        end if
        call char_expr_operands(arena, node%arg_indices(1), context, data_ptr, &
                                length, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_liric_char_byte_zext(context%session, data_ptr, &
                i32_immediate(context%session, 0_c_int64_t), value, error_msg)) &
            return
    end subroutine lower_iachar_intrinsic

    subroutine char_literal_operands(context, literal_value, data_ptr, length, &
                                      error_msg)
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: literal_value
        type(lr_operand_desc_t), intent(out) :: data_ptr
        type(lr_operand_desc_t), intent(out) :: length
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: literal_text
        character(len=64) :: string_name

        call strip_literal_quotes(literal_value, literal_text)
        context%string_literal_count = context%string_literal_count + 1
        string_name = ffc_unit_global_name( &
            context, 'cexpr.', context%string_literal_count)
        call materialize_liric_string(context%session, trim(string_name), &
                                      literal_text, data_ptr, error_msg)
        if (len_trim(error_msg) > 0) return
        length = i32_immediate(context%session, int(len(literal_text), c_int64_t))
    end subroutine char_literal_operands

    subroutine compute_leading_blanks(context, data_ptr, length, value, error_msg)
        ! Count leading space bytes in data[0:length).
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(in) :: data_ptr
        type(lr_operand_desc_t), intent(in) :: length
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: idx_addr, i_val, byte_value, cond, tmp
        integer(c_int32_t) :: hdr, body, inc, done

        if (.not. emit_i32_alloca(context%session, idx_addr, error_msg)) return
        if (.not. emit_i32_store(context%session, &
                i32_immediate(context%session, 0_c_int64_t), idx_addr, &
                error_msg)) return
        hdr = create_liric_block(context%session)
        body = create_liric_block(context%session)
        inc = create_liric_block(context%session)
        done = create_liric_block(context%session)
        if (.not. emit_liric_br(context%session, hdr, error_msg)) return

        if (.not. set_liric_block(context%session, hdr, error_msg)) return
        context%current_block_id = hdr
        if (.not. emit_i32_load(context%session, idx_addr, i_val, error_msg)) return
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_SLT, i_val, &
                length, cond, error_msg)) return
        if (.not. emit_liric_condbr(context%session, cond, body, done, &
                error_msg)) return

        if (.not. set_liric_block(context%session, body, error_msg)) return
        context%current_block_id = body
        if (.not. emit_liric_char_byte_zext(context%session, data_ptr, i_val, &
                byte_value, error_msg)) return
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_EQ, byte_value, &
                i32_immediate(context%session, 32_c_int64_t), cond, error_msg)) &
            return
        if (.not. emit_liric_condbr(context%session, cond, inc, done, &
                error_msg)) return

        if (.not. set_liric_block(context%session, inc, error_msg)) return
        context%current_block_id = inc
        if (.not. emit_i32_binary(context%session, LR_OP_ADD, i_val, &
                i32_immediate(context%session, 1_c_int64_t), tmp, error_msg)) &
            return
        if (.not. emit_i32_store(context%session, tmp, idx_addr, error_msg)) return
        if (.not. emit_liric_br(context%session, hdr, error_msg)) return

        if (.not. set_liric_block(context%session, done, error_msg)) return
        context%current_block_id = done
        context%current_block_terminated = .false.
        if (.not. emit_i32_load(context%session, idx_addr, value, error_msg)) return
        call set_empty(error_msg)
    end subroutine compute_leading_blanks

    subroutine fill_spaces(context, buf, count, error_msg)
        ! Store `count` space bytes into buf[0:count).
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(in) :: buf
        type(lr_operand_desc_t), intent(in) :: count
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: idx_addr, i_val, cond, tmp
        integer(c_int32_t) :: hdr, body, done

        if (.not. emit_i32_alloca(context%session, idx_addr, error_msg)) return
        if (.not. emit_i32_store(context%session, &
                i32_immediate(context%session, 0_c_int64_t), idx_addr, &
                error_msg)) return
        hdr = create_liric_block(context%session)
        body = create_liric_block(context%session)
        done = create_liric_block(context%session)
        if (.not. emit_liric_br(context%session, hdr, error_msg)) return

        if (.not. set_liric_block(context%session, hdr, error_msg)) return
        context%current_block_id = hdr
        if (.not. emit_i32_load(context%session, idx_addr, i_val, error_msg)) return
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_SLT, i_val, count, &
                cond, error_msg)) return
        if (.not. emit_liric_condbr(context%session, cond, body, done, &
                error_msg)) return

        if (.not. set_liric_block(context%session, body, error_msg)) return
        context%current_block_id = body
        if (.not. emit_liric_store_char_byte(context%session, buf, i_val, &
                i32_immediate(context%session, 32_c_int64_t), error_msg)) return
        if (.not. emit_i32_binary(context%session, LR_OP_ADD, i_val, &
                i32_immediate(context%session, 1_c_int64_t), tmp, error_msg)) &
            return
        if (.not. emit_i32_store(context%session, tmp, idx_addr, error_msg)) return
        if (.not. emit_liric_br(context%session, hdr, error_msg)) return

        if (.not. set_liric_block(context%session, done, error_msg)) return
        context%current_block_id = done
        context%current_block_terminated = .false.
        call set_empty(error_msg)
    end subroutine fill_spaces

    include 'session_program_lowering_character_tail.inc'

    ! #384: every CHARACTER length specification must be a scalar INTEGER
    ! expression. The parser drops a length it cannot read as an integer, so
    ! the specification text is re-read from the source the arena carries;
    ! this mirrors check_intrinsic_type_stmt_source.
    subroutine check_character_length_specs(arena, error_msg)
        type(ast_arena_t), intent(in) :: arena
        character(len=:), allocatable :: source
        character(len=:), allocatable :: line
        character(len=:), allocatable :: reason
        character(len=*), parameter :: PREFIX = &
            'character length must be a scalar INTEGER expression: '
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=64) :: location
        logical :: found
        integer :: pos, line_no, next_nl, column

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
            call character_length_spec_reason(line, reason, column)
            if (len_trim(reason) > 0) then
                write (location, '(" at line ",I0,", column ",I0)') &
                    line_no, column
                error_msg = PREFIX//trim(reason)//trim(location)
                return
            end if
            line_no = line_no + 1
        end do
    end subroutine check_character_length_specs

    ! Report why the first CHARACTER length specification on one source line
    ! is not a scalar INTEGER expression, and where it starts. An empty reason
    ! means the line carries no invalid specification.
    subroutine character_length_spec_reason(line, reason, column)
        character(len=*), intent(in) :: line
        character(len=:), allocatable, intent(out) :: reason
        integer, intent(out) :: column
        character(len=:), allocatable :: code
        character(len=:), allocatable :: lowered
        character(len=:), allocatable :: spec
        integer :: i, spec_start, spec_end

        reason = ''
        column = 1
        call strip_char_length_source_comment(line, code)
        if (len_trim(code) == 0) return
        lowered = lowercase_text(mask_string_literals(code))

        i = 1
        do while (i > 0)
            i = next_character_keyword(lowered, i)
            if (i <= 0) exit
            call character_spec_bounds(lowered, i, spec_start, spec_end)
            if (spec_start <= 0) then
                i = i + len('character')
                cycle
            end if
            spec = character_length_spec_text(code(spec_start + 1:spec_end - 1))
            reason = character_length_literal_reason(spec)
            if (len_trim(reason) == 0) then
                if (statement_is_implicit(lowered)) then
                    if (index(spec, '(') > 0) then
                        reason = 'an IMPLICIT length needs a constant '// &
                                 'expression'
                    end if
                end if
            end if
            if (len_trim(reason) > 0) then
                column = spec_start
                return
            end if
            i = spec_end + 1
        end do
    end subroutine character_length_spec_reason

    ! Position of the next CHARACTER keyword in lowered source outside of
    ! character literals and not embedded in a longer identifier; 0 when none.
    integer function next_character_keyword(lowered, from) result(at)
        character(len=*), intent(in) :: lowered
        integer, intent(in) :: from
        character(len=:), allocatable :: ident_chars
        integer :: i

        ident_chars = 'abcdefghijklmnopqrstuvwxyz0123456789_'
        at = 0
        i = from
        do while (i + len('character') - 1 <= len(lowered))
            if (lowered(i:i + len('character') - 1) == 'character') then
                if (identifier_boundary(lowered, i, len('character'))) then
                    at = i
                    return
                end if
            end if
            i = i + 1
        end do
    end function next_character_keyword

    ! True when the word of length word_len starting at position start is not
    ! part of a longer identifier.
    logical function identifier_boundary(lowered, start, word_len)
        character(len=*), intent(in) :: lowered
        integer, intent(in) :: start
        integer, intent(in) :: word_len
        character(len=:), allocatable :: ident_chars
        integer :: after

        ident_chars = 'abcdefghijklmnopqrstuvwxyz0123456789_'
        identifier_boundary = .true.
        if (start > 1) then
            if (verify(lowered(start - 1:start - 1), ident_chars) == 0) then
                identifier_boundary = .false.
                return
            end if
        end if
        after = start + word_len
        if (after <= len(lowered)) then
            if (verify(lowered(after:after), ident_chars) == 0) then
                identifier_boundary = .false.
            end if
        end if
    end function identifier_boundary

    ! Bounds of the parenthesised type-parameter list that follows the
    ! CHARACTER keyword at position start; spec_start is 0 when the keyword is
    ! not followed by one.
    subroutine character_spec_bounds(code, start, spec_start, spec_end)
        character(len=*), intent(in) :: code
        integer, intent(in) :: start
        integer, intent(out) :: spec_start
        integer, intent(out) :: spec_end
        integer :: i, depth

        spec_start = 0
        spec_end = 0
        i = start + len('character')
        do while (i <= len(code))
            if (code(i:i) /= ' ') exit
            i = i + 1
        end do
        if (i > len(code)) return
        if (code(i:i) /= '(') return
        spec_start = i
        depth = 0
        do i = spec_start, len(code)
            if (code(i:i) == '(') depth = depth + 1
            if (code(i:i) == ')') then
                depth = depth - 1
                if (depth == 0) then
                    spec_end = i
                    return
                end if
            end if
        end do
        spec_start = 0
    end subroutine character_spec_bounds

    ! The length value inside a CHARACTER type-parameter list: the text after
    ! LEN= when the list is keyword-spelled, otherwise the whole list. A list
    ! that only sets KIND carries no length.
    function character_length_spec_text(spec) result(text)
        character(len=*), intent(in) :: spec
        character(len=:), allocatable :: text
        character(len=:), allocatable :: lowered
        integer :: len_pos, i, depth, value_end

        text = trim(adjustl(spec))
        lowered = lowercase_text(text)
        if (index(lowered, '=') == 0) return

        len_pos = index(lowered, 'len')
        do while (len_pos > 0)
            if (identifier_boundary(lowered, len_pos, 3)) exit
            i = index(lowered(len_pos + 1:), 'len')
            if (i == 0) then
                len_pos = 0
            else
                len_pos = len_pos + i
            end if
        end do
        if (len_pos <= 0) then
            text = ''
            return
        end if
        i = len_pos + 3
        do while (i <= len(text))
            if (text(i:i) /= ' ') exit
            i = i + 1
        end do
        if (i > len(text)) then
            text = ''
            return
        end if
        if (text(i:i) /= '=') then
            text = ''
            return
        end if
        i = i + 1
        depth = 0
        do value_end = i, len(text)
            if (text(value_end:value_end) == '(') depth = depth + 1
            if (text(value_end:value_end) == ')') depth = depth - 1
            if (text(value_end:value_end) == ',' .and. depth == 0) exit
        end do
        text = trim(adjustl(text(i:value_end - 1)))
    end function character_length_spec_text

    ! True when the statement on this lowered line starts with IMPLICIT.
    logical function statement_is_implicit(lowered)
        character(len=*), intent(in) :: lowered
        character(len=:), allocatable :: head

        statement_is_implicit = .false.
        head = trim(adjustl(lowered))
        if (len(head) < len('implicit')) return
        if (head(1:len('implicit')) /= 'implicit') return
        statement_is_implicit = identifier_boundary(head, 1, len('implicit'))
    end function statement_is_implicit

    ! Drop a trailing comment, honouring character literals.
    subroutine strip_char_length_source_comment(line, code)
        character(len=*), intent(in) :: line
        character(len=:), allocatable, intent(out) :: code
        character :: quote
        logical :: in_string
        integer :: i

        in_string = .false.
        quote = ' '
        code = line
        do i = 1, len(line)
            if (in_string) then
                if (line(i:i) == quote) in_string = .false.
            else if (line(i:i) == '''' .or. line(i:i) == '"') then
                in_string = .true.
                quote = line(i:i)
            else if (line(i:i) == '!') then
                code = line(1:i - 1)
                return
            end if
        end do
    end subroutine strip_char_length_source_comment

    ! Replace the contents of character literals with a filler so that a scan
    ! for keywords and parentheses never reads inside a literal. The quotes
    ! themselves stay in place, and positions are preserved.
    function mask_string_literals(code) result(masked)
        character(len=*), intent(in) :: code
        character(len=:), allocatable :: masked
        character :: quote
        logical :: in_string
        integer :: i

        masked = code
        in_string = .false.
        quote = ' '
        do i = 1, len(code)
            if (in_string) then
                if (code(i:i) == quote) then
                    in_string = .false.
                else
                    masked(i:i) = 'x'
                end if
            else if (code(i:i) == '''' .or. code(i:i) == '"') then
                in_string = .true.
                quote = code(i:i)
            end if
        end do
    end function mask_string_literals
