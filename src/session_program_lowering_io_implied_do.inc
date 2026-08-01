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
                                                        error_msg)
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
        integer(c_int64_t) :: lo, hi, step, ival
        integer :: vsym, b
        logical :: created_temp
        type(symbol_t) :: saved_sym
        integer, allocatable :: objects(:)

        handled = .false.
        call set_empty(error_msg)
        if (.not. node_exists(arena, item_index)) return

        select type (idn => arena%entries(item_index)%node)
        type is (io_implied_do_node)
            if (.not. allocated(idn%var_name)) return
            ! fortfront fills either the object list or the single expression
            ! form; normalise both into one object index list.
            if (allocated(idn%object_indices)) then
                objects = idn%object_indices
            else if (idn%expr_index > 0) then
                objects = [idn%expr_index]
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
                                                   context, error_msg)
                    if (len_trim(error_msg) > 0) then
                        call release_io_implied_do_var(context, vsym, &
                                                       created_temp, saved_sym)
                        return
                    end if
                end do
                ival = ival + step
            end do
            call release_io_implied_do_var(context, vsym, created_temp, saved_sym)
            handled = .true.
        end select
    end subroutine emit_io_implied_do_print_items

    recursive subroutine emit_io_implied_do_object(arena, object_index, context, &
                                                   error_msg)
        !! One object of an unrolled I/O implied-do: a nested implied-do
        !! recurses, anything else prints as a list-directed scalar item.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: object_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        logical :: nested_handled

        call set_empty(error_msg)
        if (node_exists(arena, object_index)) then
            select type (inner => arena%entries(object_index)%node)
            type is (io_implied_do_node)
                call emit_io_implied_do_print_items(arena, object_index, &
                                                    context, nested_handled, &
                                                    error_msg)
                if (len_trim(error_msg) > 0) return
                if (.not. nested_handled) &
                    error_msg = 'nested io implied-do is not supported'
                return
            end select
        end if
        if (.not. emit_liric_print_space(context%session, error_msg)) return
        call lower_print_expression_value(arena, object_index, context, error_msg)
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
