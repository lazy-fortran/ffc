    subroutine lower_allocatable_declaration(node, context, error_msg)
        ! integer/real/logical, allocatable :: a(:) or a(:,:)
        type(declaration_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i, rank, value_kind

        call declaration_value_kind(node, value_kind, error_msg)
        if (len_trim(error_msg) > 0) return
        if (value_kind /= VALUE_I32 .and. value_kind /= VALUE_F32 .and. &
            value_kind /= VALUE_F64 .and. value_kind /= VALUE_LOGICAL) then
            call unsupported_feature_error('allocatable declaration', &
                node%line, node%column, &
                'direct LIRIC session supports integer, real, and logical '// &
                'allocatable arrays only', error_msg)
            return
        end if

        rank = 1
        if (allocated(node%dimension_indices)) then
            rank = size(node%dimension_indices)
            if (rank < 1 .or. rank > 2) then
                call unsupported_feature_error('allocatable declaration', &
                    node%line, node%column, &
                    'direct LIRIC session supports rank-1 and rank-2 '// &
                    'allocatables', error_msg)
                return
            end if
            ! Only integer rank-2 is wired through allocate/element/print.
            if (rank == 2 .and. value_kind /= VALUE_I32) then
                call unsupported_feature_error('allocatable declaration', &
                    node%line, node%column, &
                    'direct LIRIC session supports rank-2 allocatables for '// &
                    'integer elements only', error_msg)
                return
            end if
        end if

        if (node%is_multi_declaration .and. allocated(node%var_names)) then
            do i = 1, size(node%var_names)
                call declare_allocatable(context, node%var_names(i), rank, &
                                         value_kind, error_msg)
                if (len_trim(error_msg) > 0) return
            end do
        else if (allocated(node%var_name)) then
            call declare_allocatable(context, node%var_name, rank, value_kind, &
                                     error_msg)
        else
            error_msg = 'allocatable declaration did not expose a variable name'
        end if
    end subroutine lower_allocatable_declaration

    subroutine lower_allocate_statement(arena, node, context, error_msg)
        ! allocate(a(N)) or allocate(a(M,N)) for one integer allocatable.
        ! N and M may be runtime integer expressions. Multi-variable and
        ! lower:upper forms are rejected.
        type(ast_arena_t), intent(in) :: arena
        type(allocate_statement_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: name
        integer :: size_index1, size_index2, symbol_index

        if (.not. allocated(node%var_indices) .or. size(node%var_indices) /= 1) then
            call unsupported_feature_error('allocate statement', node%line, &
                node%column, 'direct LIRIC session supports allocating exactly '// &
                'one variable per allocate', error_msg)
            return
        end if

        if (node%mold_expr_index > 0 .or. node%source_expr_index > 0) then
            call lower_allocate_mold_source(arena, node, context, error_msg)
            return
        end if

        ! allocate(<type-spec> :: x): no shape, the element type/length lives in
        ! node%type_spec as a string. Route by the type-spec text.
        if (allocated(node%type_spec)) then
            if (len_trim(node%type_spec) > 0 .and. &
                (.not. allocated(node%shape_indices) .or. &
                 size(node%shape_indices) == 0)) then
                call lower_allocate_type_spec(arena, node, context, error_msg)
                return
            end if
        end if

        ! allocate(x) for a bare scalar allocatable: no shape at all.
        if (is_identifier(arena, node%var_indices(1)) .and. &
            (.not. allocated(node%shape_indices) .or. &
             size(node%shape_indices) == 0)) then
            call get_identifier_name(arena, node%var_indices(1), name, error_msg)
            if (len_trim(error_msg) > 0) return
            symbol_index = find_symbol(context, name)
            if (symbol_index > 0) then
                if (context%symbols(symbol_index)%is_allocatable .and. &
                    context%symbols(symbol_index)%array_rank == 0) then
                    call lower_allocate_scalar(symbol_index, context, error_msg)
                    return
                end if
            end if
        end if

        call allocate_target(arena, node, name, size_index1, size_index2, error_msg)
        if (len_trim(error_msg) > 0) then
            call unsupported_feature_error('allocate statement', node%line, &
                node%column, trim(error_msg), error_msg)
            return
        end if

        symbol_index = find_symbol(context, name)
        if (symbol_index <= 0) then
            call unsupported_feature_error('allocate statement', node%line, &
                node%column, 'allocate target is not declared: '//trim(name), &
                error_msg)
            return
        end if
        if (.not. context%symbols(symbol_index)%is_allocatable) then
            call unsupported_feature_error('allocate statement', node%line, &
                node%column, 'allocate target is not an allocatable array: '// &
                trim(name), error_msg)
            return
        end if

        if (context%symbols(symbol_index)%array_rank == 2) then
            call lower_allocate_i32_2d(arena, size_index1, size_index2, &
                                       symbol_index, context, error_msg)
        else
            call lower_allocate_i32_1d(arena, size_index1, symbol_index, &
                                       context, error_msg)
            if (len_trim(error_msg) == 0) &
                call record_allocatable_static_size(arena, size_index1, &
                                                    symbol_index, context)
        end if
    end subroutine lower_allocate_statement

    subroutine lower_allocate_type_spec(arena, node, context, error_msg)
        ! allocate(<type-spec> :: x): the element type/length comes from the
        ! type-spec STRING, not a shape expression. A character type-spec sizes
        ! a deferred-length character scalar; a derived type-spec allocates one
        ! instance of a scalar allocatable derived variable.
        type(ast_arena_t), intent(in) :: arena
        type(allocate_statement_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: name, spec_lc
        integer :: symbol_index

        call set_empty(error_msg)
        if (.not. node_exists(arena, node%var_indices(1)) .or. &
            .not. is_identifier(arena, node%var_indices(1))) then
            call unsupported_feature_error('allocate statement', node%line, &
                node%column, 'type-spec allocate target must be a scalar name', &
                error_msg)
            return
        end if
        call get_identifier_name(arena, node%var_indices(1), name, error_msg)
        if (len_trim(error_msg) > 0) return

        symbol_index = find_symbol(context, name)
        if (symbol_index <= 0) then
            call unsupported_feature_error('allocate statement', node%line, &
                node%column, 'allocate target is not declared: '//trim(name), &
                error_msg)
            return
        end if

        spec_lc = lowercase_text(trim(adjustl(node%type_spec)))
        if (len(spec_lc) >= 9) then
            if (spec_lc(1:9) == 'character') then
                call lower_allocate_char_type_spec(context, node, symbol_index, &
                                                   name, error_msg)
                return
            end if
        end if

        ! Otherwise treat the type-spec as a derived type name. A scalar
        ! allocatable derived variable already received its storage at
        ! declaration (define_derived_symbol), so the allocate just establishes
        ! the allocated state; the existing storage is reused.
        if (context%symbols(symbol_index)%is_derived) then
            call set_empty(error_msg)
            return
        end if
        call unsupported_feature_error('allocate statement', node%line, &
            node%column, 'unsupported type-spec allocate for: '//trim(name), &
            error_msg)
    end subroutine lower_allocate_type_spec

    subroutine lower_allocate_char_type_spec(context, node, symbol_index, name, &
                                             error_msg)
        ! allocate(character(len=<expr>) :: str): malloc len+1 bytes (room for a
        ! trailing NUL so %s prints stay bounded), store the data pointer and the
        ! length into str's deferred {data,len} descriptor. Mirrors the malloc /
        ! store path in lower_deferred_concat_assignment.
        use, intrinsic :: iso_c_binding, only: c_int64_t
        type(lowering_context_t), intent(inout) :: context
        type(allocate_statement_node), intent(in) :: node
        integer, intent(in) :: symbol_index
        character(len=*), intent(in) :: name
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: len_i32, len_i64, bytes_i64, data_ptr
        character(len=:), allocatable :: len_expr

        if (.not. context%symbols(symbol_index)%is_deferred_character) then
            call unsupported_feature_error('allocate statement', node%line, &
                node%column, 'character type-spec allocate requires a '// &
                'character(len=:), allocatable target: '//trim(name), error_msg)
            return
        end if

        call extract_type_spec_len(node%type_spec, len_expr, error_msg)
        if (len_trim(error_msg) > 0) then
            call unsupported_feature_error('allocate statement', node%line, &
                node%column, trim(error_msg), error_msg)
            return
        end if

        call lower_type_spec_len_value(context, len_expr, len_i32, error_msg)
        if (len_trim(error_msg) > 0) then
            call unsupported_feature_error('allocate statement', node%line, &
                node%column, trim(error_msg), error_msg)
            return
        end if

        ! bytes = len + 1 (trailing NUL), computed in i64.
        if (.not. emit_liric_i32_to_i64(context%session, len_i32, len_i64, &
                error_msg)) return
        if (.not. emit_i64_binary(context%session, LR_OP_ADD, len_i64, &
                i64_immediate(context%session, 1_c_int64_t), bytes_i64, &
                error_msg)) return
        if (.not. emit_malloc(context%session, bytes_i64, data_ptr, error_msg)) &
            return

        if (.not. emit_ptr_store(context%session, data_ptr, &
                context%symbols(symbol_index)%deferred_data, error_msg)) return
        if (.not. emit_i64_store(context%session, len_i64, &
                context%symbols(symbol_index)%deferred_length, error_msg)) return
        context%symbols(symbol_index)%value = data_ptr
        context%symbols(symbol_index)%has_character_value = .true.
        call set_empty(error_msg)
    end subroutine lower_allocate_char_type_spec

    subroutine extract_type_spec_len(type_spec, len_expr, error_msg)
        ! Pull the length expression out of a "character(len=<expr>)" or
        ! "character(<expr>)" type-spec string, balancing parentheses so a nested
        ! call like len(input) survives intact.
        character(len=*), intent(in) :: type_spec
        character(len=:), allocatable, intent(out) :: len_expr
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: spec_lc
        integer :: open_paren, start, depth, i, content_end

        call set_empty(error_msg)
        len_expr = ''
        spec_lc = lowercase_text(trim(adjustl(type_spec)))
        open_paren = index(spec_lc, '(')
        if (open_paren <= 0) then
            error_msg = 'character type-spec without a length'
            return
        end if

        ! Find the inner content up to the matching close paren.
        start = open_paren + 1
        depth = 1
        content_end = 0
        do i = start, len(spec_lc)
            if (spec_lc(i:i) == '(') then
                depth = depth + 1
            else if (spec_lc(i:i) == ')') then
                depth = depth - 1
                if (depth == 0) then
                    content_end = i - 1
                    exit
                end if
            end if
        end do
        if (content_end < start) then
            error_msg = 'malformed character type-spec length'
            return
        end if

        ! Drop a leading "len=" if present; keep the original (cased) text so a
        ! variable name resolves with its real spelling.
        block
            character(len=:), allocatable :: inner, inner_lc
            integer :: eq_pos
            inner = trim(adjustl(type_spec(open_paren + 1:open_paren + 1 + &
                                            (content_end - start))))
            inner_lc = lowercase_text(inner)
            if (len(inner_lc) >= 4) then
                if (inner_lc(1:4) == 'len=') then
                    inner = trim(adjustl(inner(5:)))
                end if
            end if
            len_expr = inner
        end block
        if (len_trim(len_expr) == 0) error_msg = 'empty character type-spec length'
    end subroutine extract_type_spec_len

    subroutine lower_type_spec_len_value(context, len_expr, value, error_msg)
        ! Lower a type-spec length expression string to an i32 operand. Supports
        ! an integer literal, a bare integer variable, and len(<name>) of a
        ! character symbol. More complex expressions stay unsupported here.
        use, intrinsic :: iso_c_binding, only: c_int64_t
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: len_expr
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: expr, expr_lc, inner_name
        type(lr_operand_desc_t) :: data_ptr
        integer(c_int64_t) :: literal
        integer :: symbol_index, ios, open_paren, close_paren

        call set_empty(error_msg)
        expr = trim(adjustl(len_expr))
        expr_lc = lowercase_text(expr)

        ! Integer literal.
        if (verify(trim(expr), '0123456789') == 0 .and. len_trim(expr) > 0) then
            read (expr, *, iostat=ios) literal
            if (ios /= 0) then
                error_msg = 'invalid integer length in character type-spec'
                return
            end if
            value = i32_immediate(context%session, literal)
            return
        end if

        ! len(<name>) of a character symbol.
        if (len(expr_lc) > 4) then
            if (expr_lc(1:4) == 'len(') then
                open_paren = index(expr, '(')
                close_paren = index(expr, ')', back=.true.)
                if (close_paren <= open_paren + 1) then
                    error_msg = 'malformed len() in character type-spec'
                    return
                end if
                inner_name = trim(adjustl(expr(open_paren + 1:close_paren - 1)))
                symbol_index = find_symbol(context, inner_name)
                if (symbol_index <= 0 .or. &
                    context%symbols(max(symbol_index, 1))%value_kind /= &
                    VALUE_CHARACTER) then
                    error_msg = 'len() argument is not a character variable: '// &
                                trim(inner_name)
                    return
                end if
                call char_length_operands(context, symbol_index, data_ptr, &
                                          value, error_msg)
                return
            end if
        end if

        ! Bare integer variable.
        if (verify(trim(expr), &
            'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_') &
            == 0) then
            symbol_index = find_symbol(context, trim(expr))
            if (symbol_index <= 0) then
                error_msg = 'length variable is not declared: '//trim(expr)
                return
            end if
            if (context%symbols(symbol_index)%value_kind /= VALUE_I32) then
                error_msg = 'length variable is not integer: '//trim(expr)
                return
            end if
            if (context%symbols(symbol_index)%has_address .and. &
                context%symbols(symbol_index)%is_reference) then
                if (.not. emit_i32_load(context%session, &
                    context%symbols(symbol_index)%address, value, error_msg)) &
                    return
            else
                value = context%symbols(symbol_index)%value
            end if
            return
        end if

        error_msg = 'unsupported character type-spec length expression: '// &
                    trim(expr)
    end subroutine lower_type_spec_len_value

    subroutine lower_allocate_mold_source(arena, node, context, error_msg)
        ! allocate(b, mold=a) / allocate(c, source=a): give the target the same
        ! rank-1 extent as the mold/source allocatable. SOURCE also copies the
        ! element bytes; MOLD leaves contents undefined. The source extent must
        ! be a compile-time constant (allocatable_static_size), which covers the
        ! straight-line allocate(a(N)); a=... case in the corpus.
        type(ast_arena_t), intent(in) :: arena
        type(allocate_statement_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: target_name, source_name
        integer :: target_sym, source_sym, source_expr, n
        logical :: copy_values
        type(lr_operand_desc_t) :: n_i32, dst_desc, src_desc, dst_data, src_data

        copy_values = node%source_expr_index > 0
        if (copy_values) then
            source_expr = node%source_expr_index
        else
            source_expr = node%mold_expr_index
        end if

        if (.not. is_identifier(arena, node%var_indices(1))) then
            error_msg = 'mold/source allocate target must be a scalar array name'
            return
        end if
        call get_identifier_name(arena, node%var_indices(1), target_name, error_msg)
        if (len_trim(error_msg) > 0) return
        target_sym = find_symbol(context, target_name)
        if (target_sym > 0) then
            if (context%symbols(target_sym)%is_allocatable .and. &
                context%symbols(target_sym)%array_rank == 0) then
                call lower_scalar_allocate_mold_source(arena, node, target_sym, &
                    context, error_msg)
                return
            end if
        end if
        if (.not. is_identifier(arena, source_expr)) then
            error_msg = 'mold/source must name an allocatable array'
            return
        end if
        call get_identifier_name(arena, source_expr, source_name, error_msg)
        if (len_trim(error_msg) > 0) return

        target_sym = find_symbol(context, target_name)
        source_sym = find_symbol(context, source_name)
        if (target_sym <= 0 .or. source_sym <= 0) then
            error_msg = 'mold/source allocate references an undeclared variable'
            return
        end if
        if (.not. context%symbols(target_sym)%is_allocatable .or. &
            .not. context%symbols(source_sym)%is_allocatable) then
            error_msg = 'mold/source allocate requires allocatable operands'
            return
        end if
        if (context%symbols(source_sym)%array_rank == 2) then
            error_msg = 'mold/source allocate supports only rank-1 arrays'
            return
        end if
        n = context%symbols(source_sym)%allocatable_static_size
        if (n <= 0) then
            error_msg = 'mold/source allocate requires a constant source extent'
            return
        end if

        n_i32 = i32_immediate(context%session, int(n, c_int64_t))
        call lower_allocate_i32_1d_operand(n_i32, target_sym, context, error_msg)
        if (len_trim(error_msg) > 0) return
        context%symbols(target_sym)%allocatable_static_size = n

        if (.not. copy_values) then
            call set_empty(error_msg)
            return
        end if

        dst_desc = context%symbols(target_sym)%allocatable_descriptor_address
        src_desc = context%symbols(source_sym)%allocatable_descriptor_address
        if (.not. emit_ptr_load(context%session, dst_desc, dst_data, error_msg)) &
            return
        if (.not. emit_ptr_load(context%session, src_desc, src_data, error_msg)) &
            return
        if (.not. emit_memcpy(context%session, dst_data, src_data, &
                i64_immediate(context%session, int(n, c_int64_t) * &
                    allocatable_elem_size(context%symbols(source_sym)%value_kind)), &
                error_msg)) return
        call set_empty(error_msg)
    end subroutine lower_allocate_mold_source

    subroutine record_allocatable_static_size(arena, size_index, symbol_index, &
                                              context)
        ! Remember the rank-1 allocation extent when it is a compile-time
        ! constant; this lets whole-array print unroll without a runtime loop.
        ! A non-constant size leaves the size unknown (0), keeping print xfail.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: size_index
        integer, intent(in) :: symbol_index
        type(lowering_context_t), intent(inout) :: context
        integer(c_int64_t) :: n
        character(len=:), allocatable :: eval_err

        context%symbols(symbol_index)%allocatable_static_size = 0
        call eval_i32_constant(arena, size_index, context, n, eval_err)
        if (len_trim(eval_err) == 0 .and. n >= 0_c_int64_t) &
            context%symbols(symbol_index)%allocatable_static_size = int(n)
    end subroutine record_allocatable_static_size

    subroutine allocate_target(arena, node, name, size_index1, size_index2, &
                               error_msg)
        ! Extract the variable name and size expression(s). allocate parses
        ! a(N) or a(M,N) as a call_or_subscript in var_indices (or as an
        ! identifier in var_indices with shape in shape_indices for rank-1).
        ! size_index2 is 0 for rank-1 allocations.
        type(ast_arena_t), intent(in) :: arena
        type(allocate_statement_node), intent(in) :: node
        character(len=:), allocatable, intent(out) :: name
        integer, intent(out) :: size_index1, size_index2
        character(len=:), allocatable, intent(out) :: error_msg

        call set_empty(error_msg)
        size_index1 = 0
        size_index2 = 0
        if (.not. node_exists(arena, node%var_indices(1))) then
            error_msg = 'allocate target is not an AST node'
            return
        end if
        if (is_identifier(arena, node%var_indices(1))) then
            call get_identifier_name(arena, node%var_indices(1), name, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. allocated(node%shape_indices) .or. &
                size(node%shape_indices) /= 1) then
                error_msg = 'only one-dimensional allocate is supported '// &
                            'for bare-identifier form'
                return
            end if
            size_index1 = node%shape_indices(1)
            return
        end if
        select type (v => arena%entries(node%var_indices(1))%node)
        type is (call_or_subscript_node)
            if (.not. allocated(v%name) .or. .not. allocated(v%arg_indices)) then
                error_msg = 'allocate target must name an array with a size'
                return
            end if
            if (size(v%arg_indices) < 1 .or. size(v%arg_indices) > 2) then
                error_msg = 'only rank-1 and rank-2 allocate is supported'
                return
            end if
            name = v%name
            size_index1 = v%arg_indices(1)
            if (size(v%arg_indices) == 2) size_index2 = v%arg_indices(2)
        class default
            error_msg = 'allocate target must be a scalar array name'
        end select
    end subroutine allocate_target

    subroutine lower_allocate_i32_1d(arena, size_index, symbol_index, context, &
                                     error_msg)
        ! data = malloc(N*4); descriptor.data = data; .lower = 1; .upper = N
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: size_index
        integer, intent(in) :: symbol_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: n_i32

        call lower_i32_expression(arena, size_index, context, n_i32, error_msg)
        if (len_trim(error_msg) > 0) return
        call lower_allocate_i32_1d_operand(n_i32, symbol_index, context, error_msg)
    end subroutine lower_allocate_i32_1d

    subroutine lower_allocate_i32_2d(arena, size_index1, size_index2, &
                                     symbol_index, context, error_msg)
        ! data = malloc(M*N*4); descriptor: ptr, lower1=1, upper1=M,
        !                                       lower2=1, upper2=N.
        ! Convert M and N to i64 before arithmetic so the i32 values are
        ! not consumed by the multiply before they are stored in the descriptor.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: size_index1, size_index2
        integer, intent(in) :: symbol_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: m_i32, n_i32, m_i64, n_i64
        type(lr_operand_desc_t) :: total_i64, bytes_i64, data_ptr, descriptor

        call lower_i32_expression(arena, size_index1, context, m_i32, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_liric_i32_to_i64(context%session, m_i32, m_i64, error_msg)) &
            return
        call lower_i32_expression(arena, size_index2, context, n_i32, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_liric_i32_to_i64(context%session, n_i32, n_i64, error_msg)) &
            return

        ! bytes = M * N * 4 (all in i64 to avoid overflow and reuse issues)
        if (.not. emit_i64_binary(context%session, LR_OP_MUL, m_i64, n_i64, &
                total_i64, error_msg)) return
        if (.not. emit_i64_binary(context%session, LR_OP_MUL, total_i64, &
                i64_immediate(context%session, &
                    allocatable_elem_size(context%symbols(symbol_index)%value_kind)), &
                bytes_i64, error_msg)) return
        if (.not. emit_malloc(context%session, bytes_i64, data_ptr, error_msg)) &
            return

        descriptor = context%symbols(symbol_index)%allocatable_descriptor_address
        if (.not. emit_ptr_store(context%session, data_ptr, descriptor, &
                error_msg)) return
        ! lower1 = 1 at offset 8
        if (.not. emit_i64_store_at(context%session, &
                i64_immediate(context%session, 1_c_int64_t), descriptor, &
                8_c_int64_t, error_msg)) return
        ! upper1 = M at offset 16
        if (.not. emit_i64_store_at(context%session, m_i64, descriptor, &
                16_c_int64_t, error_msg)) return
        ! lower2 = 1 at offset 24
        if (.not. emit_i64_store_at(context%session, &
                i64_immediate(context%session, 1_c_int64_t), descriptor, &
                24_c_int64_t, error_msg)) return
        ! upper2 = N at offset 32
        if (.not. emit_i64_store_at(context%session, n_i64, descriptor, &
                32_c_int64_t, error_msg)) return
        call set_empty(error_msg)
    end subroutine lower_allocate_i32_2d

    subroutine lower_allocate_i32_1d_operand(n_i32, symbol_index, context, error_msg)
        ! data = malloc(N*4); descriptor.data = data; .lower = 1; .upper = N
        ! n_i32 is the already-lowered size operand (i32).
        type(lr_operand_desc_t), intent(in) :: n_i32
        integer, intent(in) :: symbol_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: bytes_i32, bytes_i64, n_i64, data_ptr
        type(lr_operand_desc_t) :: descriptor

        descriptor = context%symbols(symbol_index)%allocatable_descriptor_address
        if (.not. emit_i32_binary(context%session, LR_OP_MUL, n_i32, &
                i32_immediate(context%session, &
                    allocatable_elem_size(context%symbols(symbol_index)%value_kind)), &
                bytes_i32, error_msg)) return
        if (.not. emit_liric_i32_to_i64(context%session, bytes_i32, bytes_i64, &
                error_msg)) return
        if (.not. emit_malloc(context%session, bytes_i64, data_ptr, error_msg)) &
            return
        if (.not. emit_liric_i32_to_i64(context%session, n_i32, n_i64, error_msg)) &
            return

        if (.not. emit_ptr_store(context%session, data_ptr, descriptor, &
                error_msg)) return
        if (.not. emit_i64_store_at(context%session, &
                i64_immediate(context%session, 1_c_int64_t), descriptor, &
                8_c_int64_t, error_msg)) return
        if (.not. emit_i64_store_at(context%session, n_i64, descriptor, &
                16_c_int64_t, error_msg)) return
        call set_empty(error_msg)
    end subroutine lower_allocate_i32_1d_operand

    subroutine lower_allocatable_constructor_assignment(arena, ctor, symbol_index, &
                                                       context, error_msg)
        ! a = [e1, e2, ...] where a is a 1-D allocatable of integer, real, or
        ! logical elements (B2c). Deallocates old storage, allocates for N
        ! elements, then stores each at the kind-specific stride.
        type(ast_arena_t), intent(in) :: arena
        type(array_literal_node), intent(in) :: ctor
        integer, intent(in) :: symbol_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: n_i32, elem_addr, value
        type(lr_operand_desc_t) :: descriptor, data_ptr
        integer :: i, n, vk
        integer(c_int64_t) :: stride
        integer, allocatable :: flat(:)

        if (.not. allocated(ctor%element_indices)) then
            error_msg = 'array constructor has no elements'
            return
        end if
        allocate(flat(0))
        call flatten_constructor_elements(arena, ctor%element_indices, flat, &
                                          error_msg)
        if (len_trim(error_msg) > 0) return
        n = size(flat)
        if (n <= 0) then
            error_msg = 'array constructor has no elements'
            return
        end if

        ! Free old allocation if any.
        descriptor = context%symbols(symbol_index)%allocatable_descriptor_address
        if (.not. emit_ptr_load(context%session, descriptor, data_ptr, error_msg)) &
            return
        if (.not. emit_free(context%session, data_ptr, error_msg)) return

        ! Allocate N elements.
        n_i32 = i32_immediate(context%session, int(n, c_int64_t))
        call lower_allocate_i32_1d_operand(n_i32, symbol_index, context, error_msg)
        if (len_trim(error_msg) > 0) return
        context%symbols(symbol_index)%allocatable_static_size = n

        ! Store each element. After lower_allocate_i32_1d_operand the descriptor
        ! data pointer points to the fresh allocation with lower = 1.
        vk = context%symbols(symbol_index)%value_kind
        stride = allocatable_elem_size(vk)
        if (.not. emit_ptr_load(context%session, descriptor, data_ptr, error_msg)) &
            return
        do i = 1, n
            call lower_allocatable_element_value(arena, flat(i), vk, context, &
                                                 value, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_ptr_offset(context%session, data_ptr, &
                    int(i - 1, c_int64_t) * stride, elem_addr, error_msg)) return
            if (.not. emit_allocatable_element_store(context, vk, value, &
                                                     elem_addr, error_msg)) return
        end do
        call set_empty(error_msg)
    end subroutine lower_allocatable_constructor_assignment

    subroutine lower_allocatable_scalar_broadcast(arena, value_index, &
                                                  symbol_index, context, error_msg)
        ! a = scalar where a is a rank-1 allocatable: store the scalar into every
        ! element. Requires the current extent to be a compile-time constant
        ! (allocatable_static_size), set by the preceding allocate.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: value_index
        integer, intent(in) :: symbol_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: value, descriptor, data_ptr, addr
        integer :: vk, n, i
        integer(c_int64_t) :: stride

        if (context%symbols(symbol_index)%array_rank == 2) then
            error_msg = 'allocatable scalar broadcast supports only rank-1 arrays'
            return
        end if
        n = context%symbols(symbol_index)%allocatable_static_size
        if (n <= 0) then
            error_msg = 'allocatable scalar broadcast requires a constant extent'
            return
        end if

        vk = context%symbols(symbol_index)%value_kind
        call lower_allocatable_element_value(arena, value_index, vk, context, &
                                             value, error_msg)
        if (len_trim(error_msg) > 0) return

        stride = allocatable_elem_size(vk)
        descriptor = context%symbols(symbol_index)%allocatable_descriptor_address
        if (.not. emit_ptr_load(context%session, descriptor, data_ptr, error_msg)) &
            return
        do i = 0, n - 1
            if (.not. emit_ptr_offset(context%session, data_ptr, &
                    int(i, c_int64_t) * stride, addr, error_msg)) return
            if (.not. emit_allocatable_element_store(context, vk, value, addr, &
                                                     error_msg)) return
        end do
        call set_empty(error_msg)
    end subroutine lower_allocatable_scalar_broadcast

    subroutine lower_allocatable_elementwise_assignment(arena, node, &
                                                         symbol_index, context, &
                                                         error_msg)
        ! a = expr where a is a rank-1 allocatable and expr is a general
        ! elementwise expression (arithmetic over other arrays and/or
        ! scalars). Requires the current extent to be a compile-time
        ! constant (allocatable_static_size), set by a preceding allocate.
        type(ast_arena_t), intent(in) :: arena
        type(assignment_node), intent(in) :: node
        integer, intent(in) :: symbol_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: value
        integer :: i
        integer :: array_size

        if (context%symbols(symbol_index)%array_rank /= 1) then
            error_msg = 'allocatable whole-array assignment from an expression '// &
                        'supports only rank-1 arrays'
            return
        end if
        array_size = context%symbols(symbol_index)%allocatable_static_size
        if (array_size <= 0) then
            error_msg = 'allocatable whole-array assignment from an expression '// &
                        'requires a previously allocated constant extent'
            return
        end if
        do i = 1, array_size
            call lower_array_elementwise_value(arena, node%value_index, &
                                               symbol_index, i - 1, context, &
                                               value, error_msg)
            if (len_trim(error_msg) > 0) return
            call store_array_linear_element(context, symbol_index, &
                                            int(i - 1, c_int64_t), value, error_msg)
            if (len_trim(error_msg) > 0) return
        end do
        call set_empty(error_msg)
    end subroutine lower_allocatable_elementwise_assignment

    logical function is_scalar_broadcast_to_allocatable(arena, node_index, &
                                                        context) result(scalar)
        ! True when the RHS broadcasts into an allocatable: a literal, or an
        ! identifier naming a genuine scalar. An allocatable, array, or pointer
        ! identifier is not a scalar, so a = b (both allocatable) is rejected.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        character(len=:), allocatable :: name, err
        integer :: sym

        scalar = .false.
        if (.not. node_exists(arena, node_index)) return
        select type (n => arena%entries(node_index)%node)
        type is (literal_node)
            scalar = .true.
            return
        end select
        if (.not. is_identifier(arena, node_index)) return
        call get_identifier_name(arena, node_index, name, err)
        if (len_trim(err) > 0) return
        sym = find_symbol(context, name)
        if (sym <= 0) return
        scalar = .not. context%symbols(sym)%is_array .and. &
                 .not. context%symbols(sym)%is_allocatable .and. &
                 .not. context%symbols(sym)%is_pointer
    end function is_scalar_broadcast_to_allocatable

    subroutine lower_allocatable_element_value(arena, expr_index, vk, context, &
                                               value, error_msg)
        ! Lower one element-typed value for an allocatable, by element kind.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: expr_index
        integer, intent(in) :: vk
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg

        select case (vk)
        case (VALUE_F32)
            call lower_f32_expression(arena, expr_index, context, value, error_msg)
        case (VALUE_F64)
            call lower_f64_expression(arena, expr_index, context, value, error_msg)
        case (VALUE_LOGICAL)
            call lower_logical_expression(arena, expr_index, context, value, &
                                          error_msg)
        case default
            call lower_i32_expression(arena, expr_index, context, value, error_msg)
        end select
    end subroutine lower_allocatable_element_value

    logical function emit_allocatable_element_store(context, vk, value, addr, &
                                                    error_msg) result(ok)
        ! Typed store of one allocatable element by element kind.
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: vk
        type(lr_operand_desc_t), intent(in) :: value, addr
        character(len=:), allocatable, intent(out) :: error_msg
        select case (vk)
        case (VALUE_F32)
            ok = emit_liric_f32_store(context%session, value, addr, error_msg)
        case (VALUE_F64)
            ok = emit_liric_f64_store(context%session, value, addr, error_msg)
        case default
            ok = emit_i32_store(context%session, value, addr, error_msg)
        end select
    end function emit_allocatable_element_store

    logical function emit_allocatable_element_load(context, vk, addr, value, &
                                                   error_msg) result(ok)
        ! Typed load of one allocatable element by element kind.
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: vk
        type(lr_operand_desc_t), intent(in) :: addr
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        select case (vk)
        case (VALUE_F32)
            ok = emit_liric_f32_load(context%session, addr, value, error_msg)
        case (VALUE_F64)
            ok = emit_liric_f64_load(context%session, addr, value, error_msg)
        case default
            ok = emit_i32_load(context%session, addr, value, error_msg)
        end select
    end function emit_allocatable_element_load

    subroutine lower_deallocate_statement(arena, node, context, error_msg)
        ! deallocate(a) for one integer 1-D allocatable.
        type(ast_arena_t), intent(in) :: arena
        type(deallocate_statement_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: name
        integer :: symbol_index

        if (.not. allocated(node%var_indices) .or. size(node%var_indices) /= 1) then
            call unsupported_feature_error('deallocate statement', node%line, &
                node%column, 'direct LIRIC session supports deallocating exactly '// &
                'one variable per deallocate', error_msg)
            return
        end if
        if (.not. node_exists(arena, node%var_indices(1))) then
            error_msg = 'deallocate target is not an AST node'
            return
        end if
        if (is_identifier(arena, node%var_indices(1))) then
            call get_identifier_name(arena, node%var_indices(1), name, error_msg)
            if (len_trim(error_msg) > 0) return
        else
            call unsupported_feature_error('deallocate statement', node%line, &
                node%column, 'deallocate target must be a scalar array name', &
                error_msg)
            return
        end if

        symbol_index = find_symbol(context, name)
        if (symbol_index <= 0) then
            call unsupported_feature_error('deallocate statement', node%line, &
                node%column, 'deallocate target is not declared: '//trim(name), &
                error_msg)
            return
        end if
        if (context%symbols(symbol_index)%is_deferred_character) then
            call lower_deallocate_deferred_char(symbol_index, context, error_msg)
            return
        end if
        if (.not. context%symbols(symbol_index)%is_allocatable) then
            call unsupported_feature_error('deallocate statement', node%line, &
                node%column, 'deallocate target is not an allocatable array: '// &
                trim(name), error_msg)
            return
        end if
        if (context%symbols(symbol_index)%array_rank == 0) then
            call lower_deallocate_scalar(symbol_index, context, error_msg)
            return
        end if

        call lower_deallocate_i32(symbol_index, context, error_msg)
    end subroutine lower_deallocate_statement

    subroutine lower_deallocate_deferred_char(symbol_index, context, error_msg)
        ! deallocate(str) for a character(len=:), allocatable scalar. Mark the
        ! descriptor unallocated by zeroing its data pointer and length. The data
        ! slot may currently hold a static string pointer (after str = literal),
        ! so it is not free()d here; the descriptor reset is what later reads see.
        use, intrinsic :: iso_c_binding, only: c_int64_t
        integer, intent(in) :: symbol_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg

        if (.not. emit_i64_store(context%session, &
                i64_immediate(context%session, 0_c_int64_t), &
                context%symbols(symbol_index)%deferred_data, error_msg)) return
        if (.not. emit_i64_store(context%session, &
                i64_immediate(context%session, 0_c_int64_t), &
                context%symbols(symbol_index)%deferred_length, error_msg)) return
        context%symbols(symbol_index)%has_character_value = .false.
        call set_empty(error_msg)
    end subroutine lower_deallocate_deferred_char

    subroutine lower_deallocate_i32(symbol_index, context, error_msg)
        ! free(descriptor.data); zero descriptor. Works for rank-1 and rank-2.
        ! free(NULL) is a no-op, so an unallocated variable is handled safely.
        integer, intent(in) :: symbol_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: descriptor, data_ptr, zero

        descriptor = context%symbols(symbol_index)%allocatable_descriptor_address
        if (.not. emit_ptr_load(context%session, descriptor, data_ptr, &
                error_msg)) return
        if (.not. emit_free(context%session, data_ptr, error_msg)) return

        zero = i64_immediate(context%session, 0_c_int64_t)
        if (.not. emit_i64_store_at(context%session, zero, descriptor, &
                0_c_int64_t, error_msg)) return
        if (.not. emit_i64_store_at(context%session, zero, descriptor, &
                16_c_int64_t, error_msg)) return
        call set_empty(error_msg)
    end subroutine lower_deallocate_i32

    subroutine lower_move_alloc(arena, arg_indices, context, error_msg)
        ! move_alloc(from, to): copy descriptor from -> to, null from.
        ! Copies all five i64 slots (40 bytes); the rank-1 tail is zero and
        ! stays zero, so this works for both ranks.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: arg_indices(:)
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: from_name, to_name
        integer :: from_sym, to_sym, k
        type(lr_operand_desc_t) :: from_desc, to_desc, val, zero

        if (size(arg_indices) /= 2) then
            error_msg = 'move_alloc requires exactly two arguments'
            return
        end if
        if (.not. is_identifier(arena, arg_indices(1)) .or. &
            .not. is_identifier(arena, arg_indices(2))) then
            error_msg = 'move_alloc arguments must be allocatable array names'
            return
        end if
        call get_identifier_name(arena, arg_indices(1), from_name, error_msg)
        if (len_trim(error_msg) > 0) return
        call get_identifier_name(arena, arg_indices(2), to_name, error_msg)
        if (len_trim(error_msg) > 0) return
        from_sym = find_symbol(context, from_name)
        to_sym   = find_symbol(context, to_name)
        if (from_sym <= 0 .or. .not. context%symbols(from_sym)%is_allocatable) then
            error_msg = 'move_alloc: from argument is not an allocatable: '// &
                        trim(from_name)
            return
        end if
        if (to_sym <= 0 .or. .not. context%symbols(to_sym)%is_allocatable) then
            error_msg = 'move_alloc: to argument is not an allocatable: '// &
                        trim(to_name)
            return
        end if

        from_desc = context%symbols(from_sym)%allocatable_descriptor_address
        to_desc   = context%symbols(to_sym)%allocatable_descriptor_address
        zero = i64_immediate(context%session, 0_c_int64_t)

        ! Copy ptr + four i64 bounds fields from -> to.
        do k = 0, 4
            if (.not. emit_i64_load_at(context%session, from_desc, &
                    int(k * 8, c_int64_t), val, error_msg)) return
            if (.not. emit_i64_store_at(context%session, val, to_desc, &
                    int(k * 8, c_int64_t), error_msg)) return
        end do

        ! Null the from descriptor.
        do k = 0, 4
            if (.not. emit_i64_store_at(context%session, zero, from_desc, &
                    int(k * 8, c_int64_t), error_msg)) return
        end do
        call set_empty(error_msg)
    end subroutine lower_move_alloc

    logical function is_allocatable_element_ref(node, context) result(is_ref)
        ! a(i) or a(i,j) where a is an allocatable, even when FortFront did not
        ! flag the call_or_subscript node as an array access. The subscript count
        ! must match the allocatable rank.
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        integer :: symbol_index, rank, nargs

        is_ref = .false.
        if (.not. allocated(node%name)) return
        if (.not. allocated(node%arg_indices)) return
        nargs = size(node%arg_indices)
        if (nargs < 1 .or. nargs > 2) return
        symbol_index = find_symbol(context, node%name)
        if (symbol_index <= 0) return
        if (.not. context%symbols(symbol_index)%is_allocatable) return
        rank = context%symbols(symbol_index)%array_rank
        if (rank == 0) rank = 1   ! pre-B2e rank-1 default
        is_ref = (nargs == rank)
    end function is_allocatable_element_ref

    logical function is_declared_array_element_ref(node, context) result(is_ref)
        ! a(i) or a(i,j) where a is a declared (non-allocatable) array symbol,
        ! even when FortFront did not flag the node as an array access. This gap
        ! shows up for array dummy arguments inside module procedures, where the
        ! usage node keeps is_array_access = .false. The subscript count must
        ! match the array rank and every subscript must be scalar (a range
        ! subscript is an array section, handled elsewhere).
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        integer :: symbol_index, rank, nargs, i

        is_ref = .false.
        if (.not. allocated(node%name)) return
        if (.not. allocated(node%arg_indices)) return
        nargs = size(node%arg_indices)
        if (nargs < 1 .or. nargs > 2) return
        symbol_index = find_symbol(context, node%name)
        if (symbol_index <= 0) return
        if (.not. context%symbols(symbol_index)%is_array) return
        if (context%symbols(symbol_index)%is_allocatable) return
        do i = 1, nargs
            if (.not. node_exists(context%arena, node%arg_indices(i))) return
            select type (arg => context%arena%entries(node%arg_indices(i))%node)
            type is (range_expression_node)
                return
            end select
        end do
        rank = context%symbols(symbol_index)%array_rank
        if (rank == 0) rank = 1
        is_ref = (nargs == rank)
    end function is_declared_array_element_ref

    subroutine lower_allocatable_element_address(arena, node, symbol_index, &
                                                 context, element_address, error_msg)
        ! Runtime element address for a rank-1 or rank-2 integer allocatable.
        ! Descriptor layout (all fields are i64 at the given byte offsets):
        !   offset  0: data pointer
        !   offset  8: lower1
        !   offset 16: upper1
        !   offset 24: lower2  (rank-2 only)
        !   offset 32: upper2  (rank-2 only)
        ! rank-1: linear = i - lower1
        ! rank-2: linear = (i - lower1) + (upper1 - lower1 + 1) * (j - lower2)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        integer, intent(in) :: symbol_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: element_address
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: descriptor, data_ptr
        type(lr_operand_desc_t) :: lower1, upper1, lower2
        type(lr_operand_desc_t) :: idx1_i32, idx1_i64, offset1_i64
        type(lr_operand_desc_t) :: idx2_i32, idx2_i64, offset2_i64
        type(lr_operand_desc_t) :: extent1, linear_i64, byte_offset, tmp
        integer :: rank, arg_count

        rank = context%symbols(symbol_index)%array_rank
        if (rank == 0) rank = 1   ! pre-B2e rank-1 default
        arg_count = 0
        if (allocated(node%arg_indices)) arg_count = size(node%arg_indices)
        if (arg_count /= rank) then
            call unsupported_feature_error('allocatable element access', node%line, &
                node%column, 'subscript count does not match allocatable rank', &
                error_msg)
            return
        end if

        call validate_i32_array_subscript(arena, node%arg_indices(1), context, &
                                          error_msg)
        if (len_trim(error_msg) > 0) return
        call lower_i32_expression(arena, node%arg_indices(1), context, idx1_i32, &
                                  error_msg)
        if (len_trim(error_msg) > 0) return

        descriptor = context%symbols(symbol_index)%allocatable_descriptor_address
        if (.not. emit_ptr_load(context%session, descriptor, data_ptr, error_msg)) &
            return
        if (.not. emit_i64_load_at(context%session, descriptor, 8_c_int64_t, &
                lower1, error_msg)) return
        if (.not. emit_liric_i32_to_i64(context%session, idx1_i32, idx1_i64, &
                error_msg)) return
        if (.not. emit_i64_binary(context%session, LR_OP_SUB, idx1_i64, lower1, &
                offset1_i64, error_msg)) return

        if (rank == 1) then
            linear_i64 = offset1_i64
        else
            ! rank == 2: linear = offset1 + extent1 * (j - lower2)
            if (.not. emit_i64_load_at(context%session, descriptor, 16_c_int64_t, &
                    upper1, error_msg)) return
            if (.not. emit_i64_load_at(context%session, descriptor, 24_c_int64_t, &
                    lower2, error_msg)) return
            ! extent1 = upper1 - lower1 + 1
            ! Use tmp to avoid aliasing (extent1 as both lhs and result is UB).
            if (.not. emit_i64_binary(context%session, LR_OP_SUB, upper1, lower1, &
                    tmp, error_msg)) return
            if (.not. emit_i64_binary(context%session, LR_OP_ADD, tmp, &
                    i64_immediate(context%session, 1_c_int64_t), extent1, &
                    error_msg)) return
            call validate_i32_array_subscript(arena, node%arg_indices(2), context, &
                                              error_msg)
            if (len_trim(error_msg) > 0) return
            call lower_i32_expression(arena, node%arg_indices(2), context, &
                                      idx2_i32, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_liric_i32_to_i64(context%session, idx2_i32, idx2_i64, &
                    error_msg)) return
            if (.not. emit_i64_binary(context%session, LR_OP_SUB, idx2_i64, lower2, &
                    offset2_i64, error_msg)) return
            ! linear = offset1 + extent1 * offset2
            ! Use tmp to avoid aliasing (linear_i64 as both rhs and result is UB).
            if (.not. emit_i64_binary(context%session, LR_OP_MUL, extent1, &
                    offset2_i64, tmp, error_msg)) return
            if (.not. emit_i64_binary(context%session, LR_OP_ADD, offset1_i64, &
                    tmp, linear_i64, error_msg)) return
        end if

        if (.not. emit_i64_binary(context%session, LR_OP_MUL, linear_i64, &
                i64_immediate(context%session, &
                    allocatable_elem_size(context%symbols(symbol_index)%value_kind)), &
                byte_offset, error_msg)) return
        if (.not. emit_ptr_offset_dyn(context%session, data_ptr, byte_offset, &
                element_address, error_msg)) return
        call set_empty(error_msg)
    end subroutine lower_allocatable_element_address

    pure integer(c_int64_t) function allocatable_elem_size(value_kind) result(sz)
        ! Byte stride of one allocatable element. i32/f32/logical occupy 4 bytes;
        ! i64/f64 occupy 8. Logical uses an i32 slot, matching the array path.
        integer, intent(in) :: value_kind
        select case (value_kind)
        case (VALUE_F64, VALUE_I64)
            sz = 8_c_int64_t
        case default
            sz = 4_c_int64_t
        end select
    end function allocatable_elem_size

    subroutine declare_allocatable(context, name, rank, value_kind, error_msg)
        ! Stack-allocate a 40-byte descriptor and zero it. Layout:
        !   offset  0: data pointer (i64)
        !   offset  8: lower1 (i64)
        !   offset 16: upper1 (i64)
        !   offset 24: lower2 (i64, rank-2; 0 for rank-1)
        !   offset 32: upper2 (i64, rank-2; 0 for rank-1)
        ! data == 0 marks the array unallocated. The element kind lives on the
        ! symbol's value_kind; the descriptor itself is kind-agnostic.
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: name
        integer, intent(in) :: rank
        integer, intent(in) :: value_kind
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: descriptor, zero
        integer :: index, k

        index = find_symbol(context, name)
        if (index > 0) then
            ! An allocatable dummy argument: the parameter symbol was already
            ! registered generically at call-entry binding. The incoming
            ! parameter pointer IS the caller's own descriptor address
            ! (passed by allocatable_array_actual_descriptor_address), so
            ! adopt it directly instead of allocating fresh storage -- writes
            ! to the descriptor (allocate/deallocate) then land in the
            ! caller's frame too.
            if (context%symbols(index)%is_parameter .and. &
                .not. context%symbols(index)%is_array .and. &
                .not. context%symbols(index)%is_allocatable) then
                context%symbols(index)%value_kind = value_kind
                context%symbols(index)%is_allocatable = .true.
                context%symbols(index)%array_rank = rank
                context%symbols(index)%allocatable_descriptor_address = &
                    context%symbols(index)%address
                call set_empty(error_msg)
                return
            end if
            error_msg = 'duplicate allocatable declaration: '//trim(name)
            return
        end if
        call grow_symbols(context)

        if (.not. emit_alloca_bytes(context%session, &
                i64_immediate(context%session, 40_c_int64_t), descriptor, &
                error_msg)) return
        zero = i64_immediate(context%session, 0_c_int64_t)
        do k = 0, 4
            if (.not. emit_i64_store_at(context%session, zero, descriptor, &
                    int(k * 8, c_int64_t), error_msg)) return
        end do

        index = context%symbol_count + 1
        context%symbols(index)%name = trim(name)
        context%symbols(index)%value_kind = value_kind
        context%symbols(index)%is_allocatable = .true.
        context%symbols(index)%array_rank = rank
        context%symbols(index)%allocatable_descriptor_address = descriptor
        context%symbol_count = index
        call set_empty(error_msg)
    end subroutine declare_allocatable

    subroutine lower_allocated(arena, arg_indices, context, value, error_msg)
        ! allocated(a): true when a's data pointer is non-null. Works for a
        ! rank-1/2 allocatable array (data ptr in allocatable_descriptor_address
        ! offset 0) and a deferred-length character allocatable (data ptr in
        ! deferred_data).
        type(ast_arena_t), intent(in) :: arena
        integer, allocatable, intent(in) :: arg_indices(:)
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: name
        type(lr_operand_desc_t) :: data_ptr
        integer :: symbol_index

        if (.not. allocated(arg_indices)) then
            error_msg = 'allocated requires exactly one argument'
            return
        end if
        if (size(arg_indices) /= 1) then
            error_msg = 'allocated requires exactly one argument'
            return
        end if
        if (.not. is_identifier(arena, arg_indices(1))) then
            error_msg = 'allocated argument must be an allocatable variable'
            return
        end if
        call identifier_name(arena, arg_indices(1), name, error_msg)
        if (len_trim(error_msg) > 0) return

        symbol_index = find_symbol(context, name)
        if (symbol_index <= 0) then
            error_msg = 'allocated argument was not declared: '//trim(name)
            return
        end if
        if (context%symbols(symbol_index)%is_deferred_character) then
            if (.not. emit_ptr_load(context%session, &
                    context%symbols(symbol_index)%deferred_data, data_ptr, &
                    error_msg)) return
        else if (context%symbols(symbol_index)%is_allocatable .and. &
                 context%symbols(symbol_index)%array_rank == 0) then
            ! A scalar allocatable folds its straight-line association state,
            ! mirroring associated()'s compile-time limitation.
            if (context%symbols(symbol_index)%has_address) then
                value = i32_immediate(context%session, 1_c_int64_t)
            else
                value = i32_immediate(context%session, 0_c_int64_t)
            end if
            call set_empty(error_msg)
            return
        else if (context%symbols(symbol_index)%is_allocatable) then
            if (.not. emit_ptr_load(context%session, &
                    context%symbols(symbol_index)%allocatable_descriptor_address, &
                    data_ptr, error_msg)) return
        else
            error_msg = 'allocated argument is not an allocatable variable: '// &
                        trim(name)
            return
        end if
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_NE, data_ptr, &
                null_ptr_operand(context), value, error_msg)) return
        call set_empty(error_msg)
    end subroutine lower_allocated
