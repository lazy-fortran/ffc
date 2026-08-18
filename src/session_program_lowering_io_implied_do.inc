    logical function is_io_implied_do_item(arena, item_index) result(is_implied)
        !! True when an I/O list item is an implied-do.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: item_index

        is_implied = .false.
        if (.not. node_exists(arena, item_index)) return
        select type (item => arena%entries(item_index)%node)
        type is (io_implied_do_node)
            is_implied = .true.
        class default
        end select
    end function is_io_implied_do_item

    recursive subroutine emit_io_implied_do_print_items(arena, item_index, &
                                                        context, handled, &
                                                        error_msg, prev_is_char)
        !! List-directed print of an I/O implied-do item,
        !! print *, (obj, ..., i = lo, hi [, step]).
        !! The trip count must fold at compile time; the control variable is
        !! bound as a transient i32 constant for each iteration so every object
        !! lowers as an ordinary scalar print item. A nested implied-do object
        !! recurses through the same path.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: item_index
        type(lowering_context_t), intent(inout) :: context
        logical, intent(out) :: handled
        character(len=:), allocatable, intent(out) :: error_msg
        !! Carries the list-directed separator state in and out, so the rule
        !! that no blank is written between two consecutive character values
        !! holds across the boundary of this item as well as inside it.
        logical, intent(inout), optional :: prev_is_char
        logical :: char_state
        integer(c_int64_t) :: lo, hi, step, ival
        integer :: vsym, b
        logical :: created_temp
        type(symbol_t) :: saved_sym
        integer, allocatable :: objects(:)

        handled = .false.
        call set_empty(error_msg)
        char_state = .false.
        if (present(prev_is_char)) char_state = prev_is_char
        if (.not. node_exists(arena, item_index)) return

        select type (idn => arena%entries(item_index)%node)
        type is (io_implied_do_node)
            if (.not. allocated(idn%var_name)) return
            ! fortfront fills either the object list or the single expression
            ! form; normalise both into one object index list.
            if (allocated(idn%object_indices)) then
                objects = idn%object_indices
            else if (idn%expr_index > 0) then
                allocate (objects(1))
                objects(1) = idn%expr_index
            else
                return
            end if
            if (size(objects) == 0) return
            call eval_i32_constant(arena, idn%start_expr_index, context, lo, &
                                   error_msg)
            if (len_trim(error_msg) > 0) return
            call eval_i32_constant(arena, idn%end_expr_index, context, hi, &
                                   error_msg)
            if (len_trim(error_msg) > 0) return
            step = 1_c_int64_t
            if (idn%step_expr_index > 0) then
                call eval_i32_constant(arena, idn%step_expr_index, context, &
                                       step, error_msg)
                if (len_trim(error_msg) > 0) return
            end if
            if (step == 0_c_int64_t) then
                error_msg = 'io implied-do step is zero'
                return
            end if
            call bind_io_implied_do_var(context, idn%var_name, vsym, &
                                        created_temp, saved_sym)
            ival = lo
            do while ((step > 0 .and. ival <= hi) .or. &
                      (step < 0 .and. ival >= hi))
                context%symbols(vsym)%i32_constant = ival
                context%symbols(vsym)%value = i32_immediate(context%session, ival)
                context%symbols(vsym)%has_address = .false.
                context%symbols(vsym)%is_reference = .false.
                do b = 1, size(objects)
                    call emit_io_implied_do_object(arena, objects(b), &
                                                   context, char_state, error_msg)
                    if (len_trim(error_msg) > 0) then
                        call release_io_implied_do_var(context, vsym, &
                                                       created_temp, saved_sym)
                        return
                    end if
                end do
                ival = ival + step
            end do
            call release_io_implied_do_var(context, vsym, created_temp, saved_sym)
            if (present(prev_is_char)) prev_is_char = char_state
            handled = .true.
        end select
    end subroutine emit_io_implied_do_print_items

    recursive subroutine emit_array_constructor_implied_do_print_items(arena, &
                                                                       ctor, &
                                                                       context, &
                                                                       handled, &
                                                                       error_msg)
        !! List-directed print of an array-constructor implied-do,
        !! print *, (/(obj, ..., i = lo, hi)/).  The constructor node wraps a
        !! do_loop_node, so bind its index exactly as the I/O implied-do path
        !! does and lower each body expression as an ordinary scalar item.
        type(ast_arena_t), intent(in) :: arena
        type(array_literal_node), intent(in) :: ctor
        type(lowering_context_t), intent(inout) :: context
        logical, intent(out) :: handled
        character(len=:), allocatable, intent(out) :: error_msg
        logical :: prev_is_char

        handled = .false.
        call set_empty(error_msg)
        prev_is_char = .false.
        if (.not. allocated(ctor%element_indices)) return
        if (size(ctor%element_indices) /= 1) return
        if (.not. node_exists(arena, ctor%element_indices(1))) return
        select type (loop => arena%entries(ctor%element_indices(1))%node)
        type is (do_loop_node)
            call emit_array_constructor_implied_do_loop(arena, loop, context, &
                prev_is_char, handled, error_msg)
        class default
        end select
    end subroutine emit_array_constructor_implied_do_print_items

    recursive subroutine emit_array_constructor_implied_do_loop(arena, loop, &
                                                                context, &
                                                                prev_is_char, &
                                                                handled, &
                                                                error_msg)
        !! Expand one array-constructor implied-DO loop for list-directed output.
        type(ast_arena_t), intent(in) :: arena
        type(do_loop_node), intent(in) :: loop
        type(lowering_context_t), intent(inout) :: context
        logical, intent(inout) :: prev_is_char
        logical, intent(out) :: handled
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int64_t) :: lo, hi, step, ival
        integer :: vsym, b
        logical :: created_temp
        type(symbol_t) :: saved_sym

        handled = .false.
        call set_empty(error_msg)
        if (.not. allocated(loop%var_name)) return
        if (.not. allocated(loop%body_indices)) return
        if (size(loop%body_indices) == 0) return
        call eval_i32_constant(arena, loop%start_expr_index, context, lo, error_msg)
        if (len_trim(error_msg) > 0) return
        call eval_i32_constant(arena, loop%end_expr_index, context, hi, error_msg)
        if (len_trim(error_msg) > 0) return
        step = 1_c_int64_t
        if (loop%step_expr_index > 0) then
            call eval_i32_constant(arena, loop%step_expr_index, context, step, &
                                   error_msg)
            if (len_trim(error_msg) > 0) return
        end if
        if (step == 0_c_int64_t) then
            error_msg = 'array-constructor implied-do step is zero'
            return
        end if

        call bind_io_implied_do_var(context, loop%var_name, vsym, created_temp, &
                                    saved_sym)
        ival = lo
        do while ((step > 0_c_int64_t .and. ival <= hi) .or. &
                  (step < 0_c_int64_t .and. ival >= hi))
            context%symbols(vsym)%i32_constant = ival
            context%symbols(vsym)%value = i32_immediate(context%session, ival)
            context%symbols(vsym)%has_address = .false.
            context%symbols(vsym)%is_reference = .false.
            do b = 1, size(loop%body_indices)
                call emit_array_constructor_implied_do_object(arena, &
                    loop%body_indices(b), context, prev_is_char, handled, &
                    error_msg)
                if (len_trim(error_msg) > 0) then
                    call release_io_implied_do_var(context, vsym, created_temp, &
                                                   saved_sym)
                    return
                end if
                if (.not. handled) then
                    call release_io_implied_do_var(context, vsym, created_temp, &
                                                   saved_sym)
                    return
                end if
            end do
            ival = ival + step
        end do
        call release_io_implied_do_var(context, vsym, created_temp, saved_sym)
        handled = .true.
    end subroutine emit_array_constructor_implied_do_loop

    recursive subroutine emit_array_constructor_implied_do_object(arena, &
                                                                  object_index, &
                                                                  context, &
                                                                  prev_is_char, &
                                                                  handled, &
                                                                  error_msg)
        !! Emit one body object, recursing for a nested implied-DO loop.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: object_index
        type(lowering_context_t), intent(inout) :: context
        logical, intent(inout) :: prev_is_char
        logical, intent(out) :: handled
        character(len=:), allocatable, intent(out) :: error_msg
        logical :: this_is_char

        handled = .false.
        call set_empty(error_msg)
        if (.not. node_exists(arena, object_index)) return
        select type (object => arena%entries(object_index)%node)
        type is (array_literal_node)
            if (is_implied_do_constructor(arena, object)) then
                call emit_array_constructor_implied_do_print_items(arena, object, &
                    context, handled, error_msg)
                return
            end if
        type is (do_loop_node)
            call emit_array_constructor_implied_do_loop(arena, object, context, &
                prev_is_char, handled, error_msg)
            return
        end select

        this_is_char = char_print_item(arena, object_index, context)
        if (.not. (this_is_char .and. prev_is_char)) then
            if (.not. emit_liric_print_space(context%session, error_msg)) return
        end if
        if (this_is_char) then
            call lower_print_char_expr(arena, object_index, context, error_msg)
        else
            call lower_print_expression_value(arena, object_index, context, &
                                              error_msg)
        end if
        if (len_trim(error_msg) > 0) return
        prev_is_char = this_is_char
        handled = .true.
    end subroutine emit_array_constructor_implied_do_object

    recursive subroutine emit_io_implied_do_object(arena, object_index, context, &
                                                   prev_is_char, error_msg)
        !! One object of an unrolled I/O implied-do: a nested implied-do
        !! recurses through the same path, anything else prints as a
        !! list-directed scalar item. The item's type decides how it prints,
        !! which is the one place that decision is made for an implied-do.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: object_index
        type(lowering_context_t), intent(inout) :: context
        logical, intent(inout) :: prev_is_char
        character(len=:), allocatable, intent(out) :: error_msg
        logical :: nested_handled
        logical :: this_is_char

        call set_empty(error_msg)
        if (node_exists(arena, object_index)) then
            select type (inner => arena%entries(object_index)%node)
            type is (io_implied_do_node)
                call emit_io_implied_do_print_items(arena, object_index, &
                                                    context, nested_handled, &
                                                    error_msg, prev_is_char)
                if (len_trim(error_msg) > 0) return
                if (.not. nested_handled) &
                    error_msg = 'nested io implied-do is not supported'
                return
            end select
        end if
        this_is_char = char_print_item(arena, object_index, context)
        if (.not. (this_is_char .and. prev_is_char)) then
            if (.not. emit_liric_print_space(context%session, error_msg)) return
        end if
        if (this_is_char) then
            ! A character object prints its bytes at its own length, not the
            ! integer those bytes would be read as.
            call lower_print_char_expr(arena, object_index, context, error_msg)
        else
            call lower_print_expression_value(arena, object_index, context, &
                                              error_msg)
        end if
        if (len_trim(error_msg) > 0) return
        prev_is_char = this_is_char
    end subroutine emit_io_implied_do_object

    subroutine bind_io_implied_do_var(context, var_name, vsym, created_temp, &
                                      saved_sym)
        !! Bind the implied-do control variable as a transient i32 constant,
        !! creating a temporary symbol when the name is not already declared.
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: var_name
        integer, intent(out) :: vsym
        logical, intent(out) :: created_temp
        type(symbol_t), intent(out) :: saved_sym

        created_temp = .false.
        vsym = find_symbol_compat(context, var_name)
        if (vsym <= 0) then
            call grow_symbols(context)
            vsym = context%symbol_count + 1
            context%symbols(vsym)%name = trim(var_name)
            context%symbol_count = vsym
            created_temp = .true.
        end if
        saved_sym = context%symbols(vsym)
        context%symbols(vsym)%has_i32_constant = .true.
        context%symbols(vsym)%is_transient_i32_constant = .true.
        context%symbols(vsym)%value_kind = VALUE_I32
    end subroutine bind_io_implied_do_var

    subroutine release_io_implied_do_var(context, vsym, created_temp, saved_sym)
        !! Undo bind_io_implied_do_var: drop the temporary or restore the
        !! previous binding of an existing symbol.
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: vsym
        logical, intent(in) :: created_temp
        type(symbol_t), intent(in) :: saved_sym

        if (created_temp) then
            context%symbol_count = context%symbol_count - 1
        else
            context%symbols(vsym) = saved_sym
        end if
    end subroutine release_io_implied_do_var
