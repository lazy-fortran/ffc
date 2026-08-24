    ! Procedure dummy arguments (#467, #606).
    !
    ! A dummy whose signature is declared by an interface body inside the
    ! receiving procedure is a *procedure* dummy: the caller passes a callable
    ! address, not data storage. FortFront #2950 stopped fabricating a scalar
    ! data declaration for such a name, so the address now has to be lowered as
    ! one. It reuses the procedure-pointer representation already in place for
    ! `procedure(iface), pointer :: fp` (#245 B3d): the dummy owns a ptr slot
    ! holding the callee address, and every call through it is an indirect call.
    ! There is exactly one callable-address convention in the lowerer.

    logical function dummy_is_procedure(arena, body_indices, name) result(is_proc)
        ! True when `name` is declared by an interface body in this scope, or by
        ! a PROCEDURE(interface) declaration.  The latter is how a dummy names
        ! an ABSTRACT INTERFACE declared in a host/module scope; it is still a
        ! callable address and must not be lowered as data.
        type(ast_arena_t), intent(in) :: arena
        integer, allocatable, intent(in) :: body_indices(:)
        character(len=*), intent(in) :: name
        integer :: i, j, k
        character(len=:), allocatable :: body_name

        is_proc = .false.
        if (.not. allocated(body_indices)) return
        do i = 1, size(body_indices)
            if (.not. node_exists(arena, body_indices(i))) cycle
            select type (iface => arena%entries(body_indices(i))%node)
            type is (declaration_node)
                if (.not. declaration_names_procedure(iface)) cycle
                if (allocated(iface%var_name)) then
                    if (same_name(iface%var_name, name)) then
                        is_proc = .true.
                        return
                    end if
                end if
                if (iface%is_multi_declaration .and. &
                    allocated(iface%var_names)) then
                    do k = 1, size(iface%var_names)
                        if (same_name(iface%var_names(k), name)) then
                            is_proc = .true.
                            return
                        end if
                    end do
                end if
            type is (interface_block_node)
                if (iface%is_abstract) cycle
                if (allocated(iface%name)) then
                    if (len_trim(iface%name) > 0) cycle
                end if
                if (.not. allocated(iface%procedure_indices)) cycle
                do j = 1, size(iface%procedure_indices)
                    call interface_body_procedure_name(arena, &
                        iface%procedure_indices(j), body_name)
                    if (len_trim(body_name) == 0) cycle
                    if (same_name(body_name, name)) then
                        is_proc = .true.
                        return
                    end if
                end do
            end select
        end do
    end function dummy_is_procedure

    subroutine define_procedure_dummy_symbol(context, name, param_index, &
                                             error_msg)
        ! Bind a procedure dummy to a ptr slot holding the passed callee
        ! address. The incoming parameter carries that address by value.
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: name
        integer, intent(in) :: param_index
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: idx

        call grow_symbols(context)
        idx = context%symbol_count + 1
        context%symbols(idx)%name = trim(name)
        context%symbols(idx)%value_kind = VALUE_PROC_PTR
        context%symbols(idx)%is_pointer = .true.
        context%symbols(idx)%is_proc_pointer = .true.
        context%symbols(idx)%is_associated = .true.
        context%symbols(idx)%is_dummy_argument = .true.
        context%symbols(idx)%is_parameter = .true.
        if (.not. emit_ptr_alloca(context%session, &
                                  context%symbols(idx)%address, error_msg)) return
        if (.not. emit_ptr_store(context%session, &
                                 ptr_param(context%session, param_index), &
                                 context%symbols(idx)%address, error_msg)) return
        context%symbols(idx)%has_address = .true.
        context%symbol_count = idx
        call set_empty(error_msg)
    end subroutine define_procedure_dummy_symbol

    logical function procedure_actual_name(arena, node_index, context, name) &
            result(is_proc)
        ! True when the actual argument at node_index is a bare identifier that
        ! names a procedure rather than data: another procedure dummy or
        ! procedure pointer, a procedure defined in this unit, or a supported
        ! double-precision elemental intrinsic. A declared data symbol always
        ! wins, so a function result variable never reaches this path.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        character(len=:), allocatable, intent(out) :: name
        character(len=:), allocatable :: local_error
        integer :: idx

        is_proc = .false.
        call set_empty(name)
        if (node_index <= 0) return
        if (.not. node_exists(arena, node_index)) return
        if (.not. is_identifier(arena, node_index)) return
        call get_identifier_name(arena, node_index, name, local_error)
        if (len_trim(local_error) > 0) return
        if (len_trim(name) == 0) return
        idx = find_symbol_compat(context, name)
        if (idx > 0) then
            is_proc = context%symbols(idx)%is_proc_pointer
            return
        end if
        if (find_function_index(context, name) > 0) then
            is_proc = .true.
            return
        end if
        is_proc = len_trim(intrinsic_thunk_libm_name(name)) > 0
    end function procedure_actual_name

    subroutine procedure_actual_operand(arena, context, name, operand, error_msg)
        ! Callable address for a procedure passed as an actual argument.
        type(ast_arena_t), intent(in) :: arena
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: name
        type(lr_operand_desc_t), intent(out) :: operand
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: idx

        idx = find_symbol_compat(context, name)
        if (idx > 0) then
            ! Forwarding a procedure dummy or pointer: pass on the address it
            ! holds, so the receiving procedure sees the original target.
            if (.not. emit_ptr_load(context%session, &
                                    context%symbols(idx)%address, operand, &
                                    error_msg)) return
            call set_empty(error_msg)
            return
        end if
        if (find_function_index(context, name) > 0) then
            call procedure_symbol_operand(context, &
                call_emit_name(arena, name, context), operand, error_msg)
            return
        end if
        call procedure_symbol_operand(context, &
            intrinsic_thunk_name(name), operand, error_msg)
    end subroutine procedure_actual_operand

    subroutine procedure_symbol_operand(context, emit_name, operand, error_msg)
        ! Intern a procedure symbol and describe its address as a ptr operand.
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: emit_name
        type(lr_operand_desc_t), intent(out) :: operand
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int32_t) :: symbol_id
        character(kind=c_char), allocatable :: c_name(:)

        call to_c_chars(trim(emit_name), c_name)
        symbol_id = lr_session_intern(context%session%handle, c_name)
        if (symbol_id < 0_c_int32_t) then
            error_msg = 'could not intern procedure symbol: '//trim(emit_name)
            return
        end if
        operand%kind = LR_OP_KIND_GLOBAL
        operand%payload = int(symbol_id, c_int64_t)
        operand%typ = lr_type_ptr_s(context%session%handle)
        operand%global_offset = 0_c_int64_t
        call set_empty(error_msg)
    end subroutine procedure_symbol_operand

    subroutine lower_f64_proc_ptr_call(arena, node, context, value, error_msg)
        ! Indirect double-precision function call through a procedure dummy or
        ! procedure pointer.
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: callee_ptr
        type(lr_operand_desc_t), allocatable :: args(:)
        integer, allocatable :: copyback_indices(:)
        integer :: idx
        character(len=:), allocatable :: callee_name

        idx = find_symbol_compat(context, node%name)
        call resolve_proc_pointer_callee_name(arena, node%name, callee_name)
        ! A procedure dummy carries its result ABI in the interface that
        ! created the dummy.  An explicit procedure pointer has no runtime
        ! result descriptor, so only a single, direct, same-unit real(8)
        ! target is safe to lower here.  In particular, do not infer an ABI
        ! from the first assignment when the pointer is unresolved or can be
        ! rebound through control flow.
        if (.not. context%symbols(idx)%is_dummy_argument) then
            if (.not. proc_pointer_f64_target_is_stable(arena, node%name, &
                                                        context)) then
                error_msg = 'procedure pointer call result is unsupported or '// &
                            'ambiguous: '//trim(node%name)
                return
            end if
        end if
        if (.not. emit_ptr_load(context%session, context%symbols(idx)%address, &
                                callee_ptr, error_msg)) return
        if (allocated(node%arg_indices)) then
            call prepare_reference_args(arena, node%arg_indices, context, &
                                        VALUE_F64, callee_name, args, &
                                        copyback_indices, error_msg)
            if (len_trim(error_msg) > 0) return
        else
            allocate (args(0))
            allocate (copyback_indices(0))
        end if
        if (.not. emit_f64_indirect_call(context%session, callee_ptr, args, &
                                         value, error_msg)) return
        call copy_back_reference_args(context, args, copyback_indices, error_msg)
    end subroutine lower_f64_proc_ptr_call

    logical function proc_pointer_f64_target_is_stable(arena, ptr_name, context) &
        result(is_stable)
        ! Return true only for the deliberately small f64 procedure-pointer
        ! boundary: one direct assignment to a same-unit real(8) function,
        ! outside all flow constructs.  A procedure dummy is handled by the
        ! caller and does not use this predicate.
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: ptr_name
        type(lowering_context_t), intent(in) :: context
        integer :: n, parent, assignment_count
        character(len=:), allocatable :: lhs_name, rhs_name, local_error
        character(len=:), allocatable :: parent_type

        is_stable = .false.
        assignment_count = 0
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (pnode => arena%entries(n)%node)
            type is (pointer_assignment_node)
                if (.not. is_identifier(arena, pnode%pointer_index)) cycle
                call get_identifier_name(arena, pnode%pointer_index, lhs_name, &
                                          local_error)
                if (len_trim(local_error) > 0) cycle
                if (.not. same_name(lhs_name, ptr_name)) cycle
                assignment_count = assignment_count + 1
                if (assignment_count /= 1) return
                if (.not. is_identifier(arena, pnode%target_index)) return
                call get_identifier_name(arena, pnode%target_index, rhs_name, &
                                          local_error)
                if (len_trim(local_error) > 0) return
                if (.not. is_contained_f64_function(context, rhs_name)) return

                parent = arena%entries(n)%parent_index
                do while (parent > 0 .and. parent <= arena%size)
                    parent_type = get_node_type_at(arena, parent)
                    if (parent_type == 'if_node' .or. parent_type == 'if' .or. &
                        parent_type == 'do_loop_node' .or. &
                        parent_type == 'do_loop' .or. &
                        parent_type == 'do_while_node' .or. &
                        parent_type == 'do_while' .or. &
                        parent_type == 'select_case_node' .or. &
                        parent_type == 'select_case' .or. &
                        parent_type == 'where_node' .or. parent_type == 'where' .or. &
                        parent_type == 'forall_node' .or. parent_type == 'forall') then
                        return
                    end if
                    parent = arena%entries(parent)%parent_index
                end do
            end select
        end do
        is_stable = assignment_count == 1
    end function proc_pointer_f64_target_is_stable

    subroutine lower_f32_proc_ptr_call(arena, node, context, value, error_msg)
        ! Indirect default-real function call through a procedure pointer.
        ! Only a same-unit real(4) target reaches this path; unresolved and
        ! other-kind procedure pointers remain explicit unsupported calls.
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: callee_ptr
        type(lr_operand_desc_t), allocatable :: args(:)
        integer, allocatable :: copyback_indices(:)
        integer :: idx
        character(len=:), allocatable :: callee_name

        idx = find_symbol_compat(context, node%name)
        call resolve_proc_pointer_callee_name(arena, node%name, callee_name)
        if (.not. context%symbols(idx)%is_dummy_argument .and. &
            contained_function_kind(context, callee_name) /= VALUE_F32 .and. &
            contained_function_kind(context, callee_name) /= VALUE_PROC_PTR) then
            error_msg = 'procedure pointer call result is unsupported or ambiguous: '// &
                        trim(node%name)
            return
        end if
        if (.not. emit_ptr_load(context%session, context%symbols(idx)%address, &
                                callee_ptr, error_msg)) return
        if (allocated(node%arg_indices)) then
            call prepare_reference_args(arena, node%arg_indices, context, &
                                        VALUE_F32, callee_name, args, &
                                        copyback_indices, error_msg)
            if (len_trim(error_msg) > 0) return
        else
            allocate (args(0))
            allocate (copyback_indices(0))
        end if
        if (.not. emit_f32_indirect_call(context%session, callee_ptr, args, &
                                         value, error_msg)) return
        call copy_back_reference_args(context, args, copyback_indices, error_msg)
    end subroutine lower_f32_proc_ptr_call

    ! ---------------------------------------------------------------------
    ! Intrinsics passed as actual arguments.
    !
    ! A libm entry point takes its argument by value, while a Fortran dummy
    ! procedure is called with the reference ABI. Passing the libm symbol
    ! itself would therefore be an ABI mismatch, so each intrinsic used as an
    ! actual argument gets one by-reference thunk emitted up front.

    function intrinsic_thunk_libm_name(name) result(libm_name)
        ! libm entry point behind a double-precision intrinsic that may be
        ! passed as an actual argument, or '' when the name is not one.
        character(len=*), intent(in) :: name
        character(len=:), allocatable :: libm_name
        integer :: i
        character(len=9), parameter :: THUNKABLE(15) = &
            [character(len=9) :: 'sqrt', 'exp', 'log', 'log10', 'sin', 'cos', &
            'tan', 'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh', 'erf', &
            'erfc']

        libm_name = ''
        do i = 1, size(THUNKABLE)
            if (same_name(name, trim(THUNKABLE(i)))) then
                libm_name = trim(THUNKABLE(i))
                return
            end if
            if (same_name(name, 'd'//trim(THUNKABLE(i)))) then
                libm_name = trim(THUNKABLE(i))
                return
            end if
        end do
    end function intrinsic_thunk_libm_name

    function intrinsic_thunk_name(name) result(thunk_name)
        ! Emitted symbol of the by-reference thunk for an intrinsic actual.
        character(len=*), intent(in) :: name
        character(len=:), allocatable :: thunk_name

        thunk_name = '.ffc.intrinsic.'//trim(intrinsic_thunk_libm_name(name))// &
                     '.r8'
    end function intrinsic_thunk_name

    subroutine emit_intrinsic_procedure_thunks(arena, context, error_msg)
        ! Emit one by-reference thunk per intrinsic that this unit passes as an
        ! actual argument. Runs before any procedure body so the session has no
        ! function open.
        type(ast_arena_t), intent(in) :: arena
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=16) :: emitted(16)
        integer :: emitted_count
        integer :: i, j, k
        integer, allocatable :: arg_indices(:)
        character(len=:), allocatable :: actual_name, libm_name

        call set_empty(error_msg)
        emitted_count = 0
        do i = 1, arena%size
            if (.not. node_exists(arena, i)) cycle
            call procedure_call_actual_indices(arena, i, arg_indices)
            if (size(arg_indices) == 0) cycle
            do j = 1, size(arg_indices)
                if (arg_indices(j) <= 0) cycle
                if (.not. node_exists(arena, arg_indices(j))) cycle
                if (.not. is_identifier(arena, arg_indices(j))) cycle
                call get_identifier_name(arena, arg_indices(j), actual_name, &
                                         error_msg)
                if (len_trim(error_msg) > 0) then
                    call set_empty(error_msg)
                    cycle
                end if
                libm_name = intrinsic_thunk_libm_name(actual_name)
                if (len_trim(libm_name) == 0) cycle
                if (find_function_index(context, actual_name) > 0) cycle
                do k = 1, emitted_count
                    if (trim(emitted(k)) == libm_name) exit
                end do
                if (k <= emitted_count) cycle
                if (emitted_count >= size(emitted)) cycle
                emitted_count = emitted_count + 1
                emitted(emitted_count) = libm_name
                call emit_one_intrinsic_thunk(context, libm_name, error_msg)
                if (len_trim(error_msg) > 0) return
            end do
        end do
    end subroutine emit_intrinsic_procedure_thunks

    subroutine procedure_call_actual_indices(arena, node_index, arg_indices)
        ! Actual argument indices of a CALL statement or a function reference,
        ! or a zero-size list for any other node.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        integer, allocatable, intent(out) :: arg_indices(:)
        character(len=:), allocatable :: local_error

        if (is_subroutine_call_statement(arena, node_index)) then
            call get_subroutine_call_arg_indices(arena, node_index, &
                                                 arg_indices, local_error)
            if (len_trim(local_error) > 0) then
                if (allocated(arg_indices)) deallocate (arg_indices)
                allocate (arg_indices(0))
            end if
            return
        end if
        allocate (arg_indices(0))
        select type (n => arena%entries(node_index)%node)
        type is (call_or_subscript_node)
            if (allocated(n%arg_indices)) then
                if (size(n%arg_indices) > 0) arg_indices = n%arg_indices
            end if
        end select
    end subroutine procedure_call_actual_indices

    subroutine emit_one_intrinsic_thunk(context, libm_name, error_msg)
        ! real(kind=8) function .ffc.intrinsic.<libm>.r8(x) with a reference
        ! dummy: load the argument, call libm by value, return the result.
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: libm_name
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: argument, result_value
        type(lr_operand_desc_t) :: call_args(1)

        if (.not. begin_liric_f64_function(context%session, &
                '.ffc.intrinsic.'//trim(libm_name)//'.r8', 1, error_msg)) return
        if (.not. emit_liric_f64_load(context%session, &
                ptr_param(context%session, 0), argument, error_msg)) return
        call_args(1) = argument
        if (.not. emit_liric_f64_call(context%session, trim(libm_name), &
                                      call_args, result_value, error_msg)) return
        if (.not. emit_ret_i32_operand(context%session, result_value, &
                                       error_msg)) return
        if (.not. finish_function(context%session, error_msg)) return
        call set_empty(error_msg)
    end subroutine emit_one_intrinsic_thunk
