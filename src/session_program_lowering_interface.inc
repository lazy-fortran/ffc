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
            call grow_external_procedures(context)
            context%external_procedure_count = context%external_procedure_count + 1
            associate (ep => context%external_procedures( &
                       context%external_procedure_count))
                ep%fortran_name = trim(proc%name)
                ep%c_name = trim(c_name)
                ep%return_value_kind = VALUE_I32
                ep%arg_count = n
                do j = 1, n
                    ep%arg_value_kinds(j) = VALUE_I32
                end do
            end associate
        class default
            call unsupported_feature_error('interface declaration', block%line, &
                block%column, 'only bind(c) interface functions are supported', &
                error_msg)
        end select
        call set_empty(error_msg)
    end subroutine register_interface_procedure

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
