    subroutine collect_lazy_specializations(arena, root_index, context, error_msg)
        ! Lazy Fortran lets a procedure leave its dummies untyped and take their
        ! types from use. When every call site agrees on one concrete type,
        ! FortFront resolves the dummies in place and the ordinary path lowers a
        ! single body. When call sites disagree there is no one type to give the
        ! dummies, so FortFront monomorphizes: it leaves the written procedure
        ! untyped and adds one fully typed copy per concrete signature.
        !
        ! This pass records which procedures were monomorphized. It does not
        ! name, key, or emit specializations: FortFront owns all of that. What
        ! it establishes is that the untyped original is not a callable body,
        ! so a call that still names it is refused instead of bound to whichever
        ! instance happened to be emitted (#437).
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: root_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i

        call set_empty(error_msg)
        if (.not. context%lazy_mode) return
        do i = 1, arena%size
            if (.not. node_exists(arena, i)) cycle
            call record_monomorphized_procedure(arena, i, context, error_msg)
            if (len_trim(error_msg) > 0) return
        end do
    end subroutine collect_lazy_specializations

    subroutine record_monomorphized_procedure(arena, node_index, context, &
                                              error_msg)
        ! Record one Lazy procedure that FortFront replaced with typed copies.
        ! The presence of copies is the fact that matters: FortFront makes them
        ! only when the written procedure cannot serve every call site, so the
        ! written name no longer denotes a callable body.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(function_def_node), pointer :: fn_node
        integer :: arg_count

        call set_empty(error_msg)
        fn_node => get_node_as_function_def(arena, node_index)
        if (.not. associated(fn_node)) return
        if (.not. allocated(fn_node%name)) return
        if (.not. allocated(fn_node%param_indices)) return
        arg_count = size(fn_node%param_indices)
        if (arg_count < 1 .or. arg_count > MAX_PROC_ARGS) return
        if (.not. has_monomorphic_copies(arena, trim(fn_node%name), arg_count)) &
            return
        call add_monomorphized_name(context, trim(fn_node%name))
    end subroutine record_monomorphized_procedure

    logical function has_monomorphic_copies(arena, proc_name, arg_count) &
            result(has_copies)
        ! Whether FortFront added typed copies of this procedure. A copy is a
        ! procedure of the same dummy count whose name FortFront derived from
        ! this one; the name only associates copy with original, and no type
        ! fact is read from it.
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: proc_name
        integer, intent(in) :: arg_count
        type(function_def_node), pointer :: copy_node
        integer :: i

        has_copies = .false.
        do i = 1, arena%size
            if (.not. node_exists(arena, i)) cycle
            copy_node => get_node_as_function_def(arena, i)
            if (.not. associated(copy_node)) cycle
            if (.not. allocated(copy_node%name)) cycle
            if (.not. name_is_monomorphic_copy(copy_node%name, proc_name)) cycle
            if (.not. allocated(copy_node%param_indices)) cycle
            if (size(copy_node%param_indices) /= arg_count) cycle
            has_copies = .true.
            return
        end do
    end function has_monomorphic_copies

    logical function name_is_monomorphic_copy(candidate, proc_name) result(is_copy)
        ! FortFront names a monomorphic copy <procedure>__<kind tokens>.
        character(len=*), intent(in) :: candidate
        character(len=*), intent(in) :: proc_name
        character(len=:), allocatable :: prefix, lowered

        is_copy = .false.
        prefix = lowercase_text(trim(proc_name))//'__'
        lowered = lowercase_text(trim(candidate))
        if (len(lowered) <= len(prefix)) return
        is_copy = lowered(1:len(prefix)) == prefix
    end function name_is_monomorphic_copy

    subroutine add_monomorphized_name(context, proc_name)
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: proc_name
        type(lazy_specialization_t), allocatable :: bigger(:)
        integer :: current, next

        if (lazy_procedure_is_monomorphized(context, proc_name)) return
        if (.not. allocated(context%lazy_specializations)) then
            allocate (context%lazy_specializations(8))
        else
            current = size(context%lazy_specializations)
            if (context%lazy_specialization_count >= current) then
                allocate (bigger(2 * current))
                bigger(1:current) = context%lazy_specializations
                call move_alloc(bigger, context%lazy_specializations)
            end if
        end if
        next = context%lazy_specialization_count + 1
        context%lazy_specializations(next)%procedure_name = trim(proc_name)
        context%lazy_specialization_count = next
    end subroutine add_monomorphized_name

    logical function lazy_procedure_is_monomorphized(context, proc_name) &
            result(is_mono)
        ! Whether the written name is an untyped original that FortFront
        ! replaced with typed copies.
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: proc_name
        integer :: i

        is_mono = .false.
        do i = 1, context%lazy_specialization_count
            if (same_name(context%lazy_specializations(i)%procedure_name, &
                          proc_name)) then
                is_mono = .true.
                return
            end if
        end do
    end function lazy_procedure_is_monomorphized

    subroutine reject_monomorphized_call(context, call_name, line, column, &
                                         error_msg)
        ! A call that still names a monomorphized original names no callable
        ! body: the original was left untyped and each concrete signature lives
        ! in its own copy. Which copy this call means is FortFront's mapping to
        ! make, so binding one here would be a guess -- and binding the wrong
        ! one silently returns a wrong value. Refuse it instead (#437,
        ! FortFront #2971).
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: call_name
        integer, intent(in) :: line, column
        character(len=:), allocatable, intent(out) :: error_msg

        call set_empty(error_msg)
        if (.not. lazy_procedure_is_monomorphized(context, call_name)) return
        call unsupported_feature_error('Lazy procedure call', line, column, &
            'call to '//trim(call_name)//' was not bound to one of its '// &
            'monomorphic specializations; the procedure leaves its dummy '// &
            'arguments untyped and is used at more than one concrete '// &
            'signature', error_msg)
    end subroutine reject_monomorphized_call
