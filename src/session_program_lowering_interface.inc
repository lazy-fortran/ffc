    subroutine lower_interface_block(arena, node, context, error_msg)
        ! Register bind(c) external integer functions declared in an interface
        ! block. Calls are lowered separately (#192). Non-bind(c) interfaces,
        ! generic/operator/assignment interfaces, and non-integer results are
        ! rejected.
        type(ast_arena_t), intent(in) :: arena
        type(interface_block_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i

        call set_empty(error_msg)
        if (allocated(node%kind)) then
            if (trim(lowercase_text(node%kind)) /= 'interface') then
                call unsupported_feature_error('interface block', node%line, &
                    node%column, 'only plain bind(c) interfaces are supported', &
                    error_msg)
                return
            end if
        end if
        if (.not. allocated(node%procedure_indices)) return

        do i = 1, size(node%procedure_indices)
            call register_interface_procedure(arena, node%procedure_indices(i), &
                                              node, context, error_msg)
            if (len_trim(error_msg) > 0) return
        end do
    end subroutine lower_interface_block

    subroutine register_interface_procedure(arena, proc_index, block, context, &
                                            error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: proc_index
        type(interface_block_node), intent(in) :: block
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: ret_kind, j, n
        character(len=:), allocatable :: c_name

        if (.not. node_exists(arena, proc_index)) then
            error_msg = 'interface procedure index does not reference a node'
            return
        end if
        select type (proc => arena%entries(proc_index)%node)
        type is (function_def_node)
            if (.not. has_bind_c(proc%bind_c_clause)) then
                call unsupported_feature_error('interface declaration', &
                    block%line, block%column, 'only bind(c) interface '// &
                    'procedures are supported', error_msg)
                return
            end if
            ret_kind = VALUE_I32
            if (allocated(proc%return_type)) then
                call type_name_value_kind(proc%return_type, proc%line, &
                                          proc%column, ret_kind, error_msg)
                if (len_trim(error_msg) > 0) return
            end if
            if (ret_kind /= VALUE_I32) then
                call unsupported_feature_error('interface declaration', &
                    proc%line, proc%column, 'only integer bind(c) interface '// &
                    'functions are supported', error_msg)
                return
            end if
            call bind_c_name(proc%bind_c_clause, proc%name, c_name)
            n = 0
            if (allocated(proc%param_indices)) n = size(proc%param_indices)
            if (n > MAX_PROC_ARGS) then
                error_msg = 'bind(c) interface has too many arguments'
                return
            end if
            call add_external_procedure(context, proc%name, c_name, VALUE_I32, n)
        type is (subroutine_def_node)
            if (.not. has_bind_c(proc%bind_c_clause)) then
                call unsupported_feature_error('interface declaration', &
                    block%line, block%column, 'only bind(c) interface '// &
                    'procedures are supported', error_msg)
                return
            end if
            call bind_c_name(proc%bind_c_clause, proc%name, c_name)
            n = 0
            if (allocated(proc%param_indices)) n = size(proc%param_indices)
            if (n > MAX_PROC_ARGS) then
                error_msg = 'bind(c) interface has too many arguments'
                return
            end if
            call add_external_procedure(context, proc%name, c_name, &
                                        VALUE_SUBROUTINE, n)
        class default
            call unsupported_feature_error('interface declaration', block%line, &
                block%column, 'only bind(c) interface functions are supported', &
                error_msg)
        end select
        call set_empty(error_msg)
    end subroutine register_interface_procedure

    subroutine add_external_procedure(context, fortran_name, c_name, &
                                      return_kind, arg_count)
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: fortran_name, c_name
        integer, intent(in) :: return_kind, arg_count
        integer :: j

        call grow_external_procedures(context)
        context%external_procedure_count = context%external_procedure_count + 1
        associate (ep => context%external_procedures( &
                   context%external_procedure_count))
            ep%fortran_name = trim(fortran_name)
            ep%c_name = trim(c_name)
            ep%return_value_kind = return_kind
            ep%arg_count = arg_count
            do j = 1, arg_count
                ep%arg_value_kinds(j) = VALUE_I32
            end do
        end associate
    end subroutine add_external_procedure

    logical function has_bind_c(clause)
        character(len=:), allocatable, intent(in) :: clause

        has_bind_c = .false.
        if (.not. allocated(clause)) return
        has_bind_c = index(lowercase_text(clause), 'bind') > 0
    end function has_bind_c

    subroutine bind_c_name(clause, fortran_name, c_name)
        ! Extract the C name from a bind(c, name="...") clause; default to the
        ! lowercased Fortran name when no name= is present.
        character(len=*), intent(in) :: clause
        character(len=*), intent(in) :: fortran_name
        character(len=:), allocatable, intent(out) :: c_name
        integer :: name_pos, first_quote, second_quote

        name_pos = index(lowercase_text(clause), 'name')
        if (name_pos > 0) then
            first_quote = index(clause(name_pos:), '"')
            if (first_quote > 0) then
                first_quote = name_pos + first_quote - 1
                second_quote = index(clause(first_quote + 1:), '"')
                if (second_quote > 0) then
                    c_name = clause(first_quote + 1:first_quote + second_quote - 1)
                    return
                end if
            end if
        end if
        c_name = trim(lowercase_text(fortran_name))
    end subroutine bind_c_name

    subroutine prepare_external_args(arena, arg_indices, context, args, error_msg)
        ! Lower bind(c) call arguments by value. A c_null_ptr actual becomes a
        ! null pointer operand (8 bytes); other actuals lower as i32 values.
        type(ast_arena_t), intent(in) :: arena
        integer, allocatable, intent(in) :: arg_indices(:)
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), allocatable, intent(out) :: args(:)
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: n, i

        n = 0
        if (allocated(arg_indices)) n = size(arg_indices)
        allocate (args(n))
        do i = 1, n
            if (is_c_null_ptr_arg(arena, arg_indices(i))) then
                args(i)%kind = LR_OP_KIND_IMM_I64
                args(i)%payload = 0_c_int64_t
                args(i)%typ = lr_type_ptr_s(context%session%handle)
                args(i)%global_offset = 0_c_int64_t
            else
                call lower_i32_expression(arena, arg_indices(i), context, &
                                          args(i), error_msg)
                if (len_trim(error_msg) > 0) return
            end if
        end do
        call set_empty(error_msg)
    end subroutine prepare_external_args

    logical function is_c_null_ptr_arg(arena, node_index)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index

        is_c_null_ptr_arg = .false.
        if (.not. node_exists(arena, node_index)) return
        select type (node => arena%entries(node_index)%node)
        type is (identifier_node)
            if (allocated(node%name)) is_c_null_ptr_arg = &
                same_name(node%name, 'c_null_ptr')
        end select
    end function is_c_null_ptr_arg

    subroutine lower_external_i32_call(arena, node, ext_idx, context, value, &
                                       error_msg)
        ! Lower a call to a registered bind(c) integer function: each argument
        ! is passed by value (bind(c) value semantics) and the call targets the
        ! C symbol name.
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        integer, intent(in) :: ext_idx
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t), allocatable :: args(:)
        integer :: n

        n = 0
        if (allocated(node%arg_indices)) n = size(node%arg_indices)
        if (n /= context%external_procedures(ext_idx)%arg_count) then
            call unsupported_feature_error('external function call', node%line, &
                node%column, 'argument count does not match the bind(c) '// &
                'interface for '//trim(node%name), error_msg)
            return
        end if

        call prepare_external_args(arena, node%arg_indices, context, args, &
                                   error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_i32_call(context%session, &
                trim(context%external_procedures(ext_idx)%c_name), args, value, &
                error_msg)) return
        call set_empty(error_msg)
    end subroutine lower_external_i32_call

    subroutine lower_external_void_call(arena, node_index, ext_idx, context, &
                                        error_msg)
        ! Lower a CALL to a registered bind(c) subroutine.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        integer, intent(in) :: ext_idx
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer, allocatable :: arg_indices(:)
        type(lr_operand_desc_t), allocatable :: args(:)
        integer :: n

        call get_subroutine_call_arg_indices(arena, node_index, arg_indices, &
                                             error_msg)
        if (len_trim(error_msg) > 0) return
        n = 0
        if (allocated(arg_indices)) n = size(arg_indices)
        if (n /= context%external_procedures(ext_idx)%arg_count) then
            error_msg = 'argument count does not match the bind(c) interface'
            return
        end if
        call prepare_external_args(arena, arg_indices, context, args, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_void_call(context%session, &
                trim(context%external_procedures(ext_idx)%c_name), args, &
                error_msg)) return
        call set_empty(error_msg)
    end subroutine lower_external_void_call

    integer function external_procedure_index(context, name) result(idx)
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: name
        integer :: i

        idx = 0
        do i = 1, context%external_procedure_count
            if (same_name(context%external_procedures(i)%fortran_name, name)) then
                idx = i
                return
            end if
        end do
    end function external_procedure_index

    subroutine grow_external_procedures(context)
        type(lowering_context_t), intent(inout) :: context
        type(external_procedure_t), allocatable :: old(:)

        if (.not. allocated(context%external_procedures)) then
            allocate (context%external_procedures(8))
            return
        end if
        if (context%external_procedure_count < size(context%external_procedures)) &
            return
        call move_alloc(context%external_procedures, old)
        allocate (context%external_procedures(2 * size(old)))
        context%external_procedures(1:size(old)) = old
    end subroutine grow_external_procedures
