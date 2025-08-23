module fortran_backend_statements
    use backend_interface
    use ast_core
    use type_system_unified, only: TINT, TREAL, TCHAR, TLOGICAL
    use ast_nodes_data, only: intent_type_to_string, INTENT_NONE
    use string_type, only: string_t
    use codegen_indent
    implicit none
    private

    public :: generate_code_assignment, generate_code_print_statement
    public :: generate_code_declaration, generate_code_parameter_declaration
    public :: generate_code_if, generate_code_do_loop, generate_code_do_while
    public :: generate_code_select_case, generate_code_stop, generate_code_return
    public :: generate_code_cycle, generate_code_exit, generate_code_where
    public :: generate_code_use_statement, generate_code_subroutine_call
    public :: generate_grouped_body, can_group_declarations, can_group_parameters
    public :: generate_grouped_declaration, build_param_name_with_dims

contains

    ! Local placeholder for generate_code_from_arena to avoid circular dependency
    function generate_code_from_arena(arena, node_index) result(code)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        character(len=:), allocatable :: code
        
        associate(associate_placeholder => arena)
        associate(associate_placeholder2 => node_index)
        end associate
        end associate
        
        code = "! statement placeholder"
    end function generate_code_from_arena

    ! Local helper function for indentation at specific level
    function indent(level) result(indent_str)
        integer, intent(in) :: level
        character(len=:), allocatable :: indent_str
        integer :: i
        
        indent_str = ""
        do i = 1, level * 4  ! 4 spaces per indent level
            indent_str = indent_str//" "
        end do
    end function indent

    ! Generate code for assignment node
    function generate_code_assignment(arena, node, node_index) result(code)
        type(ast_arena_t), intent(in) :: arena
        type(assignment_node), intent(in) :: node
        integer, intent(in) :: node_index
        character(len=:), allocatable :: code
        character(len=:), allocatable :: target_code, value_code

        associate(associate_placeholder => node_index)
        end associate

        ! Generate code for target
        if (node%target_index > 0 .and. node%target_index <= arena%size) then
            target_code = generate_code_from_arena(arena, node%target_index)
        else
            target_code = "???"
        end if

        ! Generate code for value
        if (node%value_index > 0 .and. node%value_index <= arena%size) then
            value_code = generate_code_from_arena(arena, node%value_index)
        else
            value_code = "???"
        end if

        ! Combine target and value
        code = target_code//" = "//value_code
    end function generate_code_assignment

    ! Generate code for print statement node
    function generate_code_print_statement(arena, node, node_index) result(code)
        type(ast_arena_t), intent(in) :: arena
        type(print_statement_node), intent(in) :: node
        integer, intent(in) :: node_index
        character(len=:), allocatable :: code
        character(len=:), allocatable :: args_code
        integer :: i

        associate(associate_placeholder => node_index)
        end associate

        if (allocated(node%format_spec)) then
            if (allocated(node%expression_indices)) then
                ! Format string with arguments
                args_code = ""
                do i = 1, size(node%expression_indices)
                    if (i > 1) args_code = args_code // ", "
                    if (node%expression_indices(i) > 0 .and. node%expression_indices(i) <= arena%size) then
                        args_code = args_code // generate_code_from_arena(arena, node%expression_indices(i))
                    end if
                end do
                code = "print " // trim(node%format_spec) // ", " // args_code
            else
                ! Format string only
                code = "print " // trim(node%format_spec)
            end if
        else
            if (allocated(node%expression_indices)) then
                ! Arguments without format string (use * format)
                args_code = ""
                do i = 1, size(node%expression_indices)
                    if (i > 1) args_code = args_code // ", "
                    if (node%expression_indices(i) > 0 .and. node%expression_indices(i) <= arena%size) then
                        args_code = args_code // generate_code_from_arena(arena, node%expression_indices(i))
                    end if
                end do
                code = "print *, " // args_code
            else
                ! Empty print statement
                code = "print *"
            end if
        end if
    end function generate_code_print_statement

    ! Generate code for declaration node
    function generate_code_declaration(arena, node, node_index) result(code)
        type(ast_arena_t), intent(in) :: arena
        type(declaration_node), intent(in) :: node
        integer, intent(in) :: node_index
        character(len=:), allocatable :: code
        character(len=:), allocatable :: type_str, kind_str, intent_str, dimension_str

        associate(associate_placeholder => arena)
        associate(associate_placeholder2 => node_index)
        end associate
        end associate

        ! Build type string
        type_str = node%type_name
        
        ! Add kind specification if present
        kind_str = ""
        if (node%kind_value > 0 .and. node%kind_value /= 4) then
            kind_str = "(kind=" // int_to_string(node%kind_value) // ")"
        end if

        ! Build intent string - TODO: Fix type mismatch for node%intent
        intent_str = ""
        ! if (node%intent /= INTENT_NONE) then
        !     intent_str = ", intent(" // intent_type_to_string(node%intent) // ")"
        ! end if

        ! Build dimension string
        dimension_str = ""
        if (node%is_array .and. allocated(node%dimension_indices)) then
            dimension_str = ", dimension("
            ! For now, just indicate array without specific bounds
            dimension_str = dimension_str // ":"
            dimension_str = dimension_str // ")"
        end if

        ! Combine all parts
        code = type_str // kind_str // intent_str // dimension_str // " :: " // node%var_name

        ! Add initialization if present - TODO: implement when value_index field is available
        ! if (node%value_index > 0) then
        !     code = code // " = " // generate_code_from_arena(arena, node%value_index)
        ! end if
    end function generate_code_declaration

    ! Generate code for parameter declaration node
    function generate_code_parameter_declaration(arena, node, node_index) result(code)
        type(ast_arena_t), intent(in) :: arena
        type(parameter_declaration_node), intent(in) :: node
        integer, intent(in) :: node_index
        character(len=:), allocatable :: code
        character(len=:), allocatable :: name_with_dims, value_str

        associate(associate_placeholder => node_index)
        end associate

        ! Build parameter name with dimensions if it's an array
        name_with_dims = build_param_name_with_dims(arena, node)

        ! Generate value expression - TODO: implement when value_index field is available
        ! if (node%value_index > 0) then
        !     value_str = generate_code_from_arena(arena, node%value_index)
        ! else
        value_str = "0"  ! Default value - TODO: get actual value
        ! end if

        code = node%type_name // ", parameter :: " // name_with_dims // " = " // value_str
    end function generate_code_parameter_declaration

    ! Generate code for if statement
    function generate_code_if(arena, node, node_index) result(code)
        type(ast_arena_t), intent(in) :: arena
        type(if_node), intent(in) :: node
        integer, intent(in) :: node_index
        character(len=:), allocatable :: code
        character(len=:), allocatable :: condition_code, then_code, else_code
        integer :: i

        associate(associate_placeholder => node_index)
        end associate

        ! Generate condition
        if (node%condition_index > 0) then
            condition_code = generate_code_from_arena(arena, node%condition_index)
        else
            condition_code = ".true."
        end if

        code = "if (" // condition_code // ") then" // new_line('A')

        ! Generate then body
        if (allocated(node%then_body_indices)) then
            then_code = generate_grouped_body(arena, node%then_body_indices, 1)
            code = code // then_code
        end if

        ! Generate else body if present
        if (allocated(node%else_body_indices) .and. size(node%else_body_indices) > 0) then
            code = code // "else" // new_line('A')
            else_code = generate_grouped_body(arena, node%else_body_indices, 1)
            code = code // else_code
        end if

        code = code // "end if"
    end function generate_code_if

    ! Generate code for do loop
    function generate_code_do_loop(arena, node, node_index) result(code)
        type(ast_arena_t), intent(in) :: arena
        type(do_loop_node), intent(in) :: node
        integer, intent(in) :: node_index
        character(len=:), allocatable :: code
        character(len=:), allocatable :: var_code, start_code, end_code, step_code, body_code

        associate(associate_placeholder => node_index)
        end associate

        ! Generate loop variable
        if (allocated(node%var_name)) then
            var_code = trim(node%var_name)
        else
            var_code = "i"  ! default loop variable
        end if

        ! Generate start value
        start_code = generate_code_from_arena(arena, node%start_expr_index)

        ! Generate end value
        end_code = generate_code_from_arena(arena, node%end_expr_index)

        ! Generate step value if present
        step_code = ""
        if (node%step_expr_index > 0) then
            step_code = ", " // generate_code_from_arena(arena, node%step_expr_index)
        end if

        code = "do " // var_code // " = " // start_code // ", " // end_code // step_code // new_line('A')

        ! Generate body
        if (allocated(node%body_indices)) then
            body_code = generate_grouped_body(arena, node%body_indices, 1)
            code = code // body_code
        end if

        code = code // "end do"
    end function generate_code_do_loop

    ! Generate code for do while loop
    function generate_code_do_while(arena, node, node_index) result(code)
        type(ast_arena_t), intent(in) :: arena
        type(do_while_node), intent(in) :: node
        integer, intent(in) :: node_index
        character(len=:), allocatable :: code
        character(len=:), allocatable :: condition_code, body_code

        associate(associate_placeholder => node_index)
        end associate

        ! Generate condition
        condition_code = generate_code_from_arena(arena, node%condition_index)

        code = "do while (" // condition_code // ")" // new_line('A')

        ! Generate body
        if (allocated(node%body_indices)) then
            body_code = generate_grouped_body(arena, node%body_indices, 1)
            code = code // body_code
        end if

        code = code // "end do"
    end function generate_code_do_while

    ! Generate code for select case
    function generate_code_select_case(arena, node, node_index) result(code)
        type(ast_arena_t), intent(in) :: arena
        type(select_case_node), intent(in) :: node
        integer, intent(in) :: node_index
        character(len=:), allocatable :: code

        associate(associate_placeholder => arena)
        associate(associate_placeholder2 => node)
        associate(associate_placeholder3 => node_index)
        end associate
        end associate
        end associate

        code = "! TODO: Implement select case"
    end function generate_code_select_case

    ! Generate code for stop statement
    function generate_code_stop(arena, node, node_index) result(code)
        type(ast_arena_t), intent(in) :: arena
        type(stop_node), intent(in) :: node
        integer, intent(in) :: node_index
        character(len=:), allocatable :: code
        character(len=:), allocatable :: message_code

        associate(associate_placeholder => node_index)
        end associate

        ! TODO: Fix when message_index field is available
        ! if (node%message_index > 0) then
        !     message_code = generate_code_from_arena(arena, node%message_index)
        !     code = "stop " // message_code
        ! else
        code = "stop"  ! Default stop statement
        ! end if
    end function generate_code_stop

    ! Generate code for return statement
    function generate_code_return(arena, node, node_index) result(code)
        type(ast_arena_t), intent(in) :: arena
        type(return_node), intent(in) :: node
        integer, intent(in) :: node_index
        character(len=:), allocatable :: code
        character(len=:), allocatable :: value_code

        associate(associate_placeholder => node_index)
        end associate

        ! TODO: Fix when value_index field is available
        ! if (node%value_index > 0) then
        !     value_code = generate_code_from_arena(arena, node%value_index)
        !     code = "return " // value_code
        ! else
        code = "return"  ! Default return statement
        ! end if
    end function generate_code_return

    ! Generate code for cycle statement
    function generate_code_cycle(arena, node, node_index) result(code)
        type(ast_arena_t), intent(in) :: arena
        type(cycle_node), intent(in) :: node
        integer, intent(in) :: node_index
        character(len=:), allocatable :: code

        associate(associate_placeholder => arena)
        associate(associate_placeholder2 => node)
        associate(associate_placeholder3 => node_index)
        end associate
        end associate
        end associate

        code = "cycle"
    end function generate_code_cycle

    ! Generate code for exit statement
    function generate_code_exit(arena, node, node_index) result(code)
        type(ast_arena_t), intent(in) :: arena
        type(exit_node), intent(in) :: node
        integer, intent(in) :: node_index
        character(len=:), allocatable :: code

        associate(associate_placeholder => arena)
        associate(associate_placeholder2 => node)
        associate(associate_placeholder3 => node_index)
        end associate
        end associate
        end associate

        code = "exit"
    end function generate_code_exit

    ! Generate code for where statement
    function generate_code_where(arena, node, node_index) result(code)
        type(ast_arena_t), intent(in) :: arena
        type(where_node), intent(in) :: node
        integer, intent(in) :: node_index
        character(len=:), allocatable :: code

        associate(associate_placeholder => arena)
        associate(associate_placeholder2 => node)
        associate(associate_placeholder3 => node_index)
        end associate
        end associate
        end associate

        code = "! TODO: Implement where statement"
    end function generate_code_where

    ! Generate code for use statement
    function generate_code_use_statement(node) result(code)
        type(use_statement_node), intent(in) :: node
        character(len=:), allocatable :: code
        integer :: i

        code = "use " // trim(node%module_name)
        
        if (node%has_only .and. allocated(node%only_list)) then
            code = code // ", only: "
            do i = 1, size(node%only_list)
                if (i > 1) code = code // ", "
                code = code // trim(node%only_list(i)%s)
            end do
        end if
    end function generate_code_use_statement

    ! Generate code for subroutine call
    function generate_code_subroutine_call(arena, node, node_index) result(code)
        type(ast_arena_t), intent(in) :: arena
        type(subroutine_call_node), intent(in) :: node
        integer, intent(in) :: node_index
        character(len=:), allocatable :: code

        associate(associate_placeholder => arena)
        associate(associate_placeholder2 => node_index)
        end associate
        end associate

        code = "call " // trim(node%name)
        
        ! TODO: Add arguments if present when argument_indices field is available
        ! if (allocated(node%argument_indices)) then
        !     code = code // "(...)"  ! Placeholder for arguments
        ! end if
    end function generate_code_subroutine_call

    ! Generate grouped body with proper indentation
    function generate_grouped_body(arena, body_indices, indent_level) result(code)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: body_indices(:)
        integer, intent(in) :: indent_level
        character(len=:), allocatable :: code
        character(len=:), allocatable :: stmt_code
        integer :: i

        code = ""
        
        do i = 1, size(body_indices)
            if (body_indices(i) > 0 .and. body_indices(i) <= arena%size) then
                stmt_code = generate_code_from_arena(arena, body_indices(i))
                if (len_trim(stmt_code) > 0) then
                    if (len_trim(code) > 0) then
                        code = code // new_line('A')
                    end if
                    code = code // indent(indent_level) // stmt_code
                end if
            end if
        end do
    end function generate_grouped_body

    ! Check if two declarations can be grouped
    function can_group_declarations(node1, node2) result(can_group)
        type(declaration_node), intent(in) :: node1, node2
        logical :: can_group

        can_group = (trim(node1%type_name) == trim(node2%type_name) .and. &
                    node1%kind_value == node2%kind_value .and. &
                    node1%intent == node2%intent .and. &
                    node1%is_array .eqv. node2%is_array)
    end function can_group_declarations

    ! Check if two parameters can be grouped
    function can_group_parameters(node1, node2) result(can_group)
        type(parameter_declaration_node), intent(in) :: node1, node2
        logical :: can_group

        can_group = (trim(node1%type_name) == trim(node2%type_name))
    end function can_group_parameters

    ! Generate grouped declaration
    function generate_grouped_declaration(type_name, kind_value, has_kind, intent, var_list) result(stmt)
        character(len=*), intent(in) :: type_name
        integer, intent(in) :: kind_value
        logical, intent(in) :: has_kind
        integer, intent(in) :: intent
        character(len=*), intent(in) :: var_list
        character(len=:), allocatable :: stmt
        character(len=:), allocatable :: kind_str, intent_str

        kind_str = ""
        if (has_kind .and. kind_value > 0 .and. kind_value /= 4) then
            kind_str = "(kind=" // int_to_string(kind_value) // ")"
        end if

        intent_str = ""
        if (intent /= INTENT_NONE) then
            intent_str = ", intent(" // intent_type_to_string(intent) // ")"
        end if

        stmt = type_name // kind_str // intent_str // " :: " // var_list
    end function generate_grouped_declaration

    ! Build parameter name with dimensions
    function build_param_name_with_dims(arena, param_node) result(name_with_dims)
        type(ast_arena_t), intent(in) :: arena
        type(parameter_declaration_node), intent(in) :: param_node
        character(len=:), allocatable :: name_with_dims

        associate(associate_placeholder => arena)
        end associate

        name_with_dims = trim(param_node%name)
        
        ! TODO: Add dimension handling for array parameters
        if (allocated(param_node%dimension_indices)) then
            name_with_dims = name_with_dims // "(...)"  ! Placeholder
        end if
    end function build_param_name_with_dims

    ! Utility function to convert integer to string
    function int_to_string(num) result(str)
        integer, intent(in) :: num
        character(len=:), allocatable :: str
        character(len=20) :: temp_str
        
        write(temp_str, '(I0)') num
        str = trim(temp_str)
    end function int_to_string

end module fortran_backend_statements