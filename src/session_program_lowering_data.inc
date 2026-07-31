    ! DATA statement lowering. A DATA statement gives variables their initial
    ! value before execution begins, independent of its textual position; the
    ! body walk applies it ahead of the first executable statement (#2349,
    ! #2251, #2252). Objects and values are flat parallel lists. An array object
    ! consumes one value per element in storage order. Implied-do objects and
    ! values are unrolled over compile-time-constant control bounds: the loop
    ! variable's SSA value is bound to each constant in turn, so subscript
    ! expressions like coeff(i+2) resolve to the right element.

    subroutine flush_data_statements(arena, body_indices, context, error_msg)
        !! Apply every DATA statement in the body, in textual order, as program
        !! initialisation. Called once at the specification/execution boundary so
        !! DATA values precede execution regardless of textual position.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: body_indices(:)
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i

        call set_empty(error_msg)
        do i = 1, size(body_indices)
            if (.not. is_data_statement_at(arena, body_indices(i))) cycle
            select type (node => arena%entries(body_indices(i))%node)
            type is (data_statement_node)
                call lower_data_statement(arena, node, context, error_msg)
                if (len_trim(error_msg) > 0) return
            end select
        end do
    end subroutine flush_data_statements

    logical function is_data_statement_at(arena, idx)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: idx
        is_data_statement_at = .false.
        if (.not. node_exists(arena, idx)) return
        select type (node => arena%entries(idx)%node)
        type is (data_statement_node)
            is_data_statement_at = .true.
        end select
    end function is_data_statement_at

    logical function is_suppressed_assignment_at(arena, idx)
        !! A hidden DATA-expansion assignment (suppress_codegen=.true.). FortFront
        !! emits these at the statement's textual position; ffc applies DATA via
        !! flush_data_statements instead, so they are skipped during the walk.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: idx
        is_suppressed_assignment_at = .false.
        if (.not. node_exists(arena, idx)) return
        select type (node => arena%entries(idx)%node)
        type is (assignment_node)
            is_suppressed_assignment_at = node%suppress_codegen
        end select
    end function is_suppressed_assignment_at

    logical function is_specification_statement_at(arena, idx)
        !! Specification-class statements precede execution and so do not force
        !! the deferred DATA flush. They must all be lowered (defining symbols)
        !! before DATA initialisers run; the flush fires only when the first
        !! executable statement appears.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: idx
        character(len=:), allocatable :: t

        is_specification_statement_at = .false.
        if (.not. node_exists(arena, idx)) return
        t = get_node_type_at(arena, idx)
        select case (t)
        case ('declaration', 'declaration_node', &
              'parameter_declaration', 'parameter_declaration_node', &
              'data_statement', 'data_statement_node', &
              'implicit_statement', 'implicit_statement_node', &
              'use_statement', 'use_statement_node', &
              'include_statement', 'include_statement_node', &
              'derived_type', 'derived_type_node', &
              'interface_block', 'interface_block_node', &
              'visibility_statement', 'visibility_statement_node')
            is_specification_statement_at = .true.
        end select
    end function is_specification_statement_at

    subroutine lower_data_statement(arena, node, context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(data_statement_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(data_value_cursor_t) :: cursor
        integer :: oi

        call set_empty(error_msg)
        if (.not. allocated(node%object_indices) .or. &
            .not. allocated(node%value_indices)) return
        cursor%vpos = 0
        do oi = 1, size(node%object_indices)
            call lower_data_object(arena, node, node%object_indices(oi), &
                                   cursor, context, error_msg)
            if (len_trim(error_msg) > 0) return
        end do
    end subroutine lower_data_statement

    recursive subroutine lower_data_object(arena, node, obj_index, cursor, &
                                           context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(data_statement_node), intent(in) :: node
        integer, intent(in) :: obj_index
        type(data_value_cursor_t), intent(inout) :: cursor
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: name
        integer :: sym, k, asz

        call set_empty(error_msg)
        if (is_data_implied_do(arena, obj_index)) then
            call lower_data_implied_do_object(arena, node, obj_index, cursor, &
                                              context, error_msg)
            return
        end if
        if (is_array_element_object(arena, obj_index)) then
            call lower_data_array_element_object(arena, node, obj_index, cursor, &
                                                 context, error_msg)
            return
        end if
        call identifier_name(arena, obj_index, name, error_msg)
        if (len_trim(error_msg) > 0) return
        sym = find_symbol_compat(context, name)
        if (sym <= 0) then
            error_msg = 'data statement target was not declared: '//trim(name)
            return
        end if
        if (context%symbols(sym)%is_array) then
            asz = context%symbols(sym)%array_size
            do k = 0, asz - 1
                call store_next_data_value_to_element(arena, node, cursor, sym, &
                                                      int(k, c_int64_t), context, &
                                                      error_msg)
                if (len_trim(error_msg) > 0) return
            end do
        else
            call store_next_data_value_to_scalar(arena, node, cursor, sym, name, &
                                                  context, error_msg)
        end if
    end subroutine lower_data_object

    subroutine store_next_data_value_to_scalar(arena, node, cursor, sym, name, &
                                               context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(data_statement_node), intent(in) :: node
        type(data_value_cursor_t), intent(inout) :: cursor
        integer, intent(in) :: sym
        character(len=*), intent(in) :: name
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: vidx

        call next_data_value(arena, node, cursor, context, vidx, error_msg)
        if (len_trim(error_msg) > 0) return
        call lower_scalar_initializer(context, name, &
                                      context%symbols(sym)%value_kind, vidx, &
                                      error_msg)
    end subroutine store_next_data_value_to_scalar

    subroutine store_next_data_value_to_element(arena, node, cursor, sym, &
                                                linear_index, context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(data_statement_node), intent(in) :: node
        type(data_value_cursor_t), intent(inout) :: cursor
        integer, intent(in) :: sym
        integer(c_int64_t), intent(in) :: linear_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: value
        integer :: vidx

        call next_data_value(arena, node, cursor, context, vidx, error_msg)
        if (len_trim(error_msg) > 0) return
        call lower_data_value_operand(arena, vidx, &
                                      context%symbols(sym)%value_kind, context, &
                                      value, error_msg)
        if (len_trim(error_msg) > 0) return
        call store_array_linear_element(context, sym, linear_index, value, &
                                        error_msg)
    end subroutine store_next_data_value_to_element

    subroutine lower_data_value_operand(arena, vidx, vk, context, value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: vidx, vk
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg

        if (vk == VALUE_F32) then
            call lower_f32_expression(arena, vidx, context, value, error_msg)
        else if (vk == VALUE_F64) then
            call lower_f64_expression(arena, vidx, context, value, error_msg)
        else if (vk == VALUE_LOGICAL) then
            call lower_logical_expression(arena, vidx, context, value, error_msg)
        else
            call lower_i32_expression(arena, vidx, context, value, error_msg)
        end if
    end subroutine lower_data_value_operand

    ! Bind the loop variable to each constant in the control range and, for
    ! every inner object expression, drive the array-element assignment path so
    ! coeff(i)/coeff(i+2) resolve against the bound value.
    recursive subroutine lower_data_implied_do_object(arena, node, obj_index, cursor, &
                                            context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(data_statement_node), intent(in) :: node
        integer, intent(in) :: obj_index
        type(data_value_cursor_t), intent(inout) :: cursor
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: start_v, end_v, step_v, var_sym, iv
        type(lr_operand_desc_t) :: saved_value
        logical :: saved_has_const
        integer(c_int64_t) :: saved_const

        ! Control bounds may reference an enclosing implied-do variable (nested
        ! ((arr(i,j),i=1,j),j=1,3)); fold them after the outer iteration bound
        ! its variable, so resolve the control inside the loop, not before it.
        select type (idn => arena%entries(obj_index)%node)
        type is (io_implied_do_node)
            var_sym = find_symbol_compat(context, idn%var_name)
        class default
            error_msg = 'data implied-do node expected'
            return
        end select
        if (var_sym <= 0) then
            error_msg = 'data implied-do control variable not declared'
            return
        end if
        saved_value = context%symbols(var_sym)%value
        saved_has_const = context%symbols(var_sym)%has_i32_constant
        saved_const = context%symbols(var_sym)%i32_constant
        call data_implied_do_control(arena, obj_index, context, var_sym, &
                                     start_v, end_v, step_v, error_msg)
        if (len_trim(error_msg) > 0) return
        iv = start_v
        do
            if (step_v > 0 .and. iv > end_v) exit
            if (step_v < 0 .and. iv < end_v) exit
            call bind_loop_var_constant(context, var_sym, iv, error_msg)
            if (len_trim(error_msg) > 0) return
            call lower_implied_do_inner_objects(arena, node, obj_index, cursor, &
                                                context, error_msg)
            if (len_trim(error_msg) > 0) return
            iv = iv + step_v
        end do
        context%symbols(var_sym)%value = saved_value
        context%symbols(var_sym)%has_i32_constant = saved_has_const
        context%symbols(var_sym)%i32_constant = saved_const
    end subroutine lower_data_implied_do_object

    recursive subroutine lower_implied_do_inner_objects(arena, node, obj_index, cursor, &
                                              context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(data_statement_node), intent(in) :: node
        integer, intent(in) :: obj_index
        type(data_value_cursor_t), intent(inout) :: cursor
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: j

        select type (idn => arena%entries(obj_index)%node)
        type is (io_implied_do_node)
            if (.not. allocated(idn%object_indices)) return
            do j = 1, size(idn%object_indices)
                call lower_data_object(arena, node, idn%object_indices(j), &
                                       cursor, context, error_msg)
                if (len_trim(error_msg) > 0) return
            end do
        end select
    end subroutine lower_implied_do_inner_objects

    ! A non-implied-do array element target such as coeff(i) from an unrolled
    ! implied-do: consume one value, lower it, and store to the resolved slot.
    subroutine lower_data_array_element_object(arena, node, obj_index, cursor, &
                                               context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(data_statement_node), intent(in) :: node
        integer, intent(in) :: obj_index
        type(data_value_cursor_t), intent(inout) :: cursor
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: value, addr
        integer :: vidx, sym, vk

        select type (tgt => arena%entries(obj_index)%node)
        type is (call_or_subscript_node)
            sym = find_symbol_compat(context, tgt%name)
            if (sym <= 0) then
                error_msg = 'data statement target was not declared: '//tgt%name
                return
            end if
            vk = context%symbols(sym)%value_kind
            call next_data_value(arena, node, cursor, context, vidx, error_msg)
            if (len_trim(error_msg) > 0) return
            call lower_data_value_operand(arena, vidx, vk, context, value, &
                                          error_msg)
            if (len_trim(error_msg) > 0) return
            if (vk == VALUE_F32 .or. vk == VALUE_F64) then
                call lower_real_array_element_address(arena, tgt, context, vk, &
                                                      addr, error_msg, &
                                                      'data implied-do target')
            else
                call lower_i32_array_element_address(arena, tgt, context, addr, &
                                                     error_msg, &
                                                     'data implied-do target')
            end if
            if (len_trim(error_msg) > 0) return
            call store_data_element_value(context, vk, value, addr, error_msg)
        class default
            error_msg = 'data statement object form not supported'
        end select
    end subroutine lower_data_array_element_object

    subroutine store_data_element_value(context, vk, value, addr, error_msg)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: vk
        type(lr_operand_desc_t), intent(in) :: value, addr
        character(len=:), allocatable, intent(out) :: error_msg

        call set_empty(error_msg)
        if (vk == VALUE_F32) then
            if (.not. emit_liric_f32_store(context%session, value, addr, &
                                           error_msg)) return
        else if (vk == VALUE_F64) then
            if (.not. emit_liric_f64_store(context%session, value, addr, &
                                           error_msg)) return
        else
            if (.not. emit_i32_store(context%session, value, addr, error_msg)) return
        end if
    end subroutine store_data_element_value

    subroutine bind_loop_var_constant(context, var_sym, iv, error_msg)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: var_sym, iv
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: value

        call set_empty(error_msg)
        value = i32_immediate(context%session, int(iv, c_int64_t))
        context%symbols(var_sym)%value = value
        ! Expose the bound value to eval_i32_constant so nested implied-do
        ! bounds that name this variable fold to a compile-time constant.
        context%symbols(var_sym)%has_i32_constant = .true.
        context%symbols(var_sym)%i32_constant = int(iv, c_int64_t)
        context%symbols(var_sym)%is_transient_i32_constant = .true.
        if (context%symbols(var_sym)%has_address .and. &
            context%symbols(var_sym)%is_reference) then
            call store_reference_value(context, var_sym, value, error_msg)
        end if
    end subroutine bind_loop_var_constant

    subroutine data_implied_do_control(arena, obj_index, context, var_sym, &
                                       start_v, end_v, step_v, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: obj_index
        type(lowering_context_t), intent(inout) :: context
        integer, intent(out) :: var_sym, start_v, end_v, step_v
        character(len=:), allocatable, intent(out) :: error_msg

        var_sym = 0
        start_v = 0
        end_v = 0
        step_v = 1
        call set_empty(error_msg)
        select type (idn => arena%entries(obj_index)%node)
        type is (io_implied_do_node)
            if (.not. allocated(idn%var_name)) then
                error_msg = 'data implied-do has no control variable'
                return
            end if
            var_sym = find_symbol_compat(context, idn%var_name)
            if (var_sym <= 0) then
                error_msg = 'data implied-do control variable not declared: '// &
                            idn%var_name
                return
            end if
            call data_fold_const_int(arena, idn%start_expr_index, context, &
                                     start_v, error_msg)
            if (len_trim(error_msg) > 0) return
            call data_fold_const_int(arena, idn%end_expr_index, context, end_v, &
                                     error_msg)
            if (len_trim(error_msg) > 0) return
            if (idn%step_expr_index > 0) then
                call data_fold_const_int(arena, idn%step_expr_index, context, &
                                         step_v, error_msg)
                if (len_trim(error_msg) > 0) return
            end if
            if (step_v == 0) error_msg = 'data implied-do step is zero'
        class default
            error_msg = 'data implied-do node expected'
        end select
    end subroutine data_implied_do_control

    ! Pull the next value AST index. When the cursor enters an implied-do value
    ! it binds the control variable for each inner evaluation so value
    ! expressions (i*1.0) read the right constant.
    recursive subroutine next_data_value(arena, node, cursor, context, vidx, &
                                         error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(data_statement_node), intent(in) :: node
        type(data_value_cursor_t), intent(inout) :: cursor
        type(lowering_context_t), intent(inout) :: context
        integer, intent(out) :: vidx
        character(len=:), allocatable, intent(out) :: error_msg

        vidx = 0
        call set_empty(error_msg)
        if (cursor%count > 0) then
            call bind_loop_var_constant(context, cursor%var_sym, cursor%cur, &
                                        error_msg)
            if (len_trim(error_msg) > 0) return
            vidx = cursor%inner
            cursor%cur = cursor%cur + cursor%step
            cursor%count = cursor%count - 1
            return
        end if
        cursor%vpos = cursor%vpos + 1
        if (cursor%vpos > size(node%value_indices)) then
            error_msg = 'data statement value count does not match objects'
            return
        end if
        vidx = node%value_indices(cursor%vpos)
        if (is_data_implied_do(arena, vidx)) then
            call open_value_implied_do(arena, vidx, cursor, context, error_msg)
            if (len_trim(error_msg) > 0) return
            call next_data_value(arena, node, cursor, context, vidx, error_msg)
        end if
    end subroutine next_data_value

    subroutine open_value_implied_do(arena, vidx, cursor, context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: vidx
        type(data_value_cursor_t), intent(inout) :: cursor
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: start_v, end_v, step_v, var_sym, n

        call data_implied_do_control(arena, vidx, context, var_sym, start_v, &
                                     end_v, step_v, error_msg)
        if (len_trim(error_msg) > 0) return
        select type (idn => arena%entries(vidx)%node)
        type is (io_implied_do_node)
            if (.not. allocated(idn%object_indices)) then
                error_msg = 'data value implied-do has no expression'
                return
            end if
            if (size(idn%object_indices) /= 1) then
                error_msg = 'data value implied-do must hold one expression'
                return
            end if
            cursor%inner = idn%object_indices(1)
        end select
        n = (end_v - start_v)/step_v + 1
        if (n < 0) n = 0
        cursor%count = n
        cursor%var_sym = var_sym
        cursor%cur = start_v
        cursor%step = step_v
    end subroutine open_value_implied_do

    logical function is_data_implied_do(arena, idx)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: idx
        is_data_implied_do = .false.
        if (.not. node_exists(arena, idx)) return
        select type (n => arena%entries(idx)%node)
        type is (io_implied_do_node)
            is_data_implied_do = .true.
        end select
    end function is_data_implied_do

    logical function is_array_element_object(arena, idx)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: idx
        is_array_element_object = .false.
        if (.not. node_exists(arena, idx)) return
        select type (n => arena%entries(idx)%node)
        type is (call_or_subscript_node)
            is_array_element_object = allocated(n%arg_indices)
        end select
    end function is_array_element_object

    subroutine data_fold_const_int(arena, idx, context, value, error_msg)
        !! Fold an implied-do control bound to a compile-time integer. Reuses
        !! eval_i32_constant, which handles literals, integer arithmetic, named
        !! parameters, and enclosing implied-do variables bound to a constant.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: idx
        type(lowering_context_t), intent(in) :: context
        integer, intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int64_t) :: v64

        value = 0
        call set_empty(error_msg)
        if (.not. node_exists(arena, idx)) then
            error_msg = 'data implied-do bound is not a constant'
            return
        end if
        call eval_i32_constant(arena, idx, context, v64, error_msg)
        if (len_trim(error_msg) > 0) return
        value = int(v64)
    end subroutine data_fold_const_int

    ! DATA statement object and initializer restrictions (#383).
    !
    ! A data-stmt-object must be a variable that is definable at
    ! initialization time: not a named constant, not a pointer, and not an
    ! entity that already carries a declaration initializer. A data-stmt-value
    ! must be a constant expression, so every name it references has to be a
    ! named constant, and its type must agree with the object's type class.
    ! Each of these is checked before lowering, where the typed declarations
    ! are the earliest layer that carries the attributes.
    subroutine check_data_statement_restrictions(arena, error_msg)
        type(ast_arena_t), intent(in) :: arena
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: n

        call set_empty(error_msg)
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (nd => arena%entries(n)%node)
            type is (data_statement_node)
                call check_one_data_statement(arena, nd, error_msg)
            end select
            if (len_trim(error_msg) > 0) return
        end do
    end subroutine check_data_statement_restrictions

    subroutine check_one_data_statement(arena, node, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(data_statement_node), intent(in) :: node
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: bound, obj_class
        integer :: i

        call set_empty(error_msg)
        if (.not. allocated(node%object_indices)) return
        bound = '|'
        do i = 1, size(node%object_indices)
            call collect_implied_do_names(arena, node%object_indices(i), bound)
        end do
        if (allocated(node%value_indices)) then
            do i = 1, size(node%value_indices)
                call collect_implied_do_names(arena, node%value_indices(i), bound)
            end do
        end if
        obj_class = ''
        do i = 1, size(node%object_indices)
            call check_data_object_node(arena, node%object_indices(i), &
                                        obj_class, error_msg)
            if (len_trim(error_msg) > 0) return
        end do
        if (.not. allocated(node%value_indices)) return
        do i = 1, size(node%value_indices)
            call check_data_value_node(arena, node%value_indices(i), obj_class, &
                                       bound, error_msg)
            if (len_trim(error_msg) > 0) return
        end do
    end subroutine check_one_data_statement

    recursive subroutine collect_implied_do_names(arena, idx, bound)
        !! Append every implied-do control variable name reachable from idx to
        !! the '|'-delimited membership string. Those names are bound by the
        !! DATA statement itself, so they are not required to be constants.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: idx
        character(len=:), allocatable, intent(inout) :: bound
        integer :: j

        if (.not. node_exists(arena, idx)) return
        select type (nd => arena%entries(idx)%node)
        type is (io_implied_do_node)
            if (allocated(nd%var_name)) then
                bound = bound//trim(lowercase_text(nd%var_name))//'|'
            end if
            if (.not. allocated(nd%object_indices)) return
            do j = 1, size(nd%object_indices)
                call collect_implied_do_names(arena, nd%object_indices(j), bound)
            end do
        end select
    end subroutine collect_implied_do_names

    recursive subroutine check_data_object_node(arena, idx, obj_class, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: idx
        character(len=:), allocatable, intent(inout) :: obj_class
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: name
        integer :: j, decl

        call set_empty(error_msg)
        if (.not. node_exists(arena, idx)) return
        select type (nd => arena%entries(idx)%node)
        type is (io_implied_do_node)
            if (.not. allocated(nd%object_indices)) return
            do j = 1, size(nd%object_indices)
                call check_data_object_node(arena, nd%object_indices(j), &
                                            obj_class, error_msg)
                if (len_trim(error_msg) > 0) return
            end do
            return
        end select
        call data_base_name(arena, idx, name)
        if (len_trim(name) == 0) return
        decl = declaration_index_for_name(arena, name)
        if (decl <= 0) return
        obj_class = data_type_class_of_declaration(arena, decl)
        call check_data_object_attributes(arena, decl, name, idx, error_msg)
    end subroutine check_data_object_node

    subroutine check_data_object_attributes(arena, decl, name, idx, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: decl, idx
        character(len=*), intent(in) :: name
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=64) :: location

        call set_empty(error_msg)
        write (location, '(" at line ",I0,", column ",I0)') &
            get_node_line(arena, idx), get_node_column(arena, idx)
        select type (dn => arena%entries(decl)%node)
        type is (declaration_node)
            if (dn%is_parameter) then
                error_msg = 'named constant '''//trim(name)//''' shall not '// &
                    'appear in a DATA statement'//trim(location)
            else if (dn%is_pointer) then
                error_msg = 'data-stmt-object '''//trim(name)//''' has the '// &
                    'POINTER attribute'//trim(location)
            else if (dn%has_initializer) then
                error_msg = 'symbol '''//trim(name)//''' already is '// &
                    'initialized in its declaration'//trim(location)
            end if
        end select
    end subroutine check_data_object_attributes

    recursive subroutine check_data_value_node(arena, idx, obj_class, bound, &
                                               error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: idx
        character(len=*), intent(in) :: obj_class, bound
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: name, val_class
        character(len=64) :: location
        integer :: j, decl

        call set_empty(error_msg)
        if (.not. node_exists(arena, idx)) return
        call check_data_value_children(arena, idx, obj_class, bound, error_msg)
        if (len_trim(error_msg) > 0) return
        val_class = data_value_type_class(arena, idx)
        write (location, '(" at line ",I0,", column ",I0)') &
            get_node_line(arena, idx), get_node_column(arena, idx)
        if (len(obj_class) > 0 .and. len(val_class) > 0) then
            if (obj_class /= val_class) then
                error_msg = 'incompatible types in DATA statement: '// &
                    obj_class//' object with '//val_class//' value'// &
                    trim(location)
                return
            end if
        end if
        call data_base_name(arena, idx, name)
        if (len_trim(name) == 0) return
        if (index(bound, '|'//trim(lowercase_text(name))//'|') > 0) return
        decl = declaration_index_for_name(arena, name)
        if (decl <= 0) return
        j = decl
        select type (dn => arena%entries(j)%node)
        type is (declaration_node)
            if (.not. dn%is_parameter) then
                error_msg = 'symbol '''//trim(name)//''' must be a '// &
                    'PARAMETER in DATA statement'//trim(location)
            end if
        end select
    end subroutine check_data_value_node

    recursive subroutine check_data_value_children(arena, idx, obj_class, bound, &
                                                   error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: idx
        character(len=*), intent(in) :: obj_class, bound
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: op, err
        integer :: j, li, ri, ln, cl

        call set_empty(error_msg)
        if (is_binary_op(arena, idx)) then
            call get_binary_op_info(arena, idx, op, li, ri, ln, cl, err)
            if (len_trim(err) > 0) return
            call check_data_value_node(arena, li, '', bound, error_msg)
            if (len_trim(error_msg) > 0) return
            call check_data_value_node(arena, ri, obj_class, bound, error_msg)
            return
        end if
        select type (nd => arena%entries(idx)%node)
        type is (call_or_subscript_node)
            if (.not. allocated(nd%arg_indices)) return
            do j = 1, size(nd%arg_indices)
                call check_data_value_node(arena, nd%arg_indices(j), '', bound, &
                                           error_msg)
                if (len_trim(error_msg) > 0) return
            end do
        type is (io_implied_do_node)
            if (.not. allocated(nd%object_indices)) return
            do j = 1, size(nd%object_indices)
                call check_data_value_node(arena, nd%object_indices(j), &
                                           obj_class, bound, error_msg)
                if (len_trim(error_msg) > 0) return
            end do
        end select
    end subroutine check_data_value_children

    subroutine data_base_name(arena, idx, name)
        !! The name a DATA object or value expression is rooted at: the
        !! identifier itself, or the array name of an element reference.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: idx
        character(len=:), allocatable, intent(out) :: name

        name = ''
        if (.not. node_exists(arena, idx)) return
        select type (nd => arena%entries(idx)%node)
        type is (identifier_node)
            if (allocated(nd%name)) name = trim(nd%name)
        type is (call_or_subscript_node)
            if (allocated(nd%name)) name = trim(nd%name)
        end select
    end subroutine data_base_name

    integer function declaration_index_for_name(arena, name) result(idx)
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: name
        integer :: n, k

        idx = 0
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (nd => arena%entries(n)%node)
            type is (declaration_node)
                if (allocated(nd%var_name)) then
                    if (same_name(nd%var_name, name)) then
                        idx = n
                        return
                    end if
                end if
                if (.not. allocated(nd%var_names)) cycle
                do k = 1, size(nd%var_names)
                    if (same_name(trim(nd%var_names(k)), name)) then
                        idx = n
                        return
                    end if
                end do
            end select
        end do
    end function declaration_index_for_name

    function data_type_class_of_declaration(arena, decl) result(cls)
        !! 'character' or 'numeric' for a declaration, '' when unknown.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: decl
        character(len=:), allocatable :: cls

        cls = ''
        if (.not. node_exists(arena, decl)) return
        select type (dn => arena%entries(decl)%node)
        type is (declaration_node)
            if (.not. allocated(dn%type_name)) return
            cls = data_type_class_of_name(dn%type_name)
        end select
    end function data_type_class_of_declaration

    function data_type_class_of_name(type_name) result(cls)
        character(len=*), intent(in) :: type_name
        character(len=:), allocatable :: cls, low

        cls = ''
        low = trim(lowercase_text(type_name))
        if (len(low) >= 9) then
            if (low(1:9) == 'character') then
                cls = 'character'
                return
            end if
        end if
        select case (low)
        case ('integer', 'real', 'logical', 'complex', 'double precision')
            cls = 'numeric'
        end select
    end function data_type_class_of_name

    function data_value_type_class(arena, idx) result(cls)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: idx
        character(len=:), allocatable :: cls, name
        integer :: decl

        cls = ''
        if (.not. node_exists(arena, idx)) return
        select type (nd => arena%entries(idx)%node)
        type is (literal_node)
            if (nd%literal_kind == LITERAL_STRING) then
                cls = 'character'
            else
                cls = 'numeric'
            end if
            return
        end select
        call data_base_name(arena, idx, name)
        if (len_trim(name) == 0) return
        decl = declaration_index_for_name(arena, name)
        if (decl <= 0) return
        cls = data_type_class_of_declaration(arena, decl)
    end function data_value_type_class
