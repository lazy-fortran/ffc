module fortran_backend_core
    use backend_interface
    use ast_core
    use type_system_unified, only: TINT, TREAL, TCHAR, TLOGICAL
    use ast_nodes_data, only: intent_type_to_string, INTENT_NONE
    use string_type, only: string_t
    use codegen_indent
    use fortran_backend_statements, only: generate_grouped_body
    implicit none
    private

    public :: fortran_backend_t
    public :: generate_code_from_arena, generate_code_program
    public :: generate_code_function_def, generate_code_subroutine_def
    public :: int_to_string

    ! Context for function indentation
    logical :: context_has_executable_before_contains = .false.

    ! Fortran backend implementation
    type, extends(backend_t) :: fortran_backend_t
    contains
        procedure :: generate_code => fortran_generate_code
        procedure :: get_name => fortran_get_name
        procedure :: get_version => fortran_get_version
    end type fortran_backend_t

contains

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

    ! Local placeholders for expression functions to avoid circular dependencies
    function generate_code_literal(node) result(code)
        class(*), intent(in) :: node
        character(len=:), allocatable :: code
        associate(associate_placeholder => node)
        end associate
        code = "literal_placeholder"
    end function generate_code_literal

    function generate_code_identifier(node) result(code)
        class(*), intent(in) :: node
        character(len=:), allocatable :: code
        associate(associate_placeholder => node)
        end associate
        code = "identifier_placeholder"
    end function generate_code_identifier

    function generate_code_binary_op(arena, node, node_index) result(code)
        type(ast_arena_t), intent(in) :: arena
        class(*), intent(in) :: node
        integer, intent(in) :: node_index
        character(len=:), allocatable :: code
        associate(associate_placeholder => arena)
        associate(associate_placeholder2 => node)
        associate(associate_placeholder3 => node_index)
        end associate
        end associate
        end associate
        code = "binary_op_placeholder"
    end function generate_code_binary_op

    function generate_code_call_or_subscript(arena, node, node_index) result(code)
        type(ast_arena_t), intent(in) :: arena
        class(*), intent(in) :: node
        integer, intent(in) :: node_index
        character(len=:), allocatable :: code
        associate(associate_placeholder => arena)
        associate(associate_placeholder2 => node)
        associate(associate_placeholder3 => node_index)
        end associate
        end associate
        end associate
        code = "call_or_subscript_placeholder"
    end function generate_code_call_or_subscript

    function generate_code_array_literal(arena, node, node_index) result(code)
        type(ast_arena_t), intent(in) :: arena
        class(*), intent(in) :: node
        integer, intent(in) :: node_index
        character(len=:), allocatable :: code
        associate(associate_placeholder => arena)
        associate(associate_placeholder2 => node)
        associate(associate_placeholder3 => node_index)
        end associate
        end associate
        end associate
        code = "[array_placeholder]"
    end function generate_code_array_literal

    ! Local placeholders for statement functions to avoid circular dependencies
    function generate_code_assignment(arena, node, node_index) result(code)
        type(ast_arena_t), intent(in) :: arena
        class(*), intent(in) :: node
        integer, intent(in) :: node_index
        character(len=:), allocatable :: code
        associate(associate_placeholder => arena)
        associate(associate_placeholder2 => node)
        associate(associate_placeholder3 => node_index)
        end associate
        end associate
        end associate
        code = "assignment_placeholder"
    end function generate_code_assignment

    function generate_code_print_statement(arena, node, node_index) result(code)
        type(ast_arena_t), intent(in) :: arena
        class(*), intent(in) :: node
        integer, intent(in) :: node_index
        character(len=:), allocatable :: code
        associate(associate_placeholder => arena)
        associate(associate_placeholder2 => node)
        associate(associate_placeholder3 => node_index)
        end associate
        end associate
        end associate
        code = "print *, 'placeholder'"
    end function generate_code_print_statement

    function generate_code_declaration(arena, node, node_index) result(code)
        type(ast_arena_t), intent(in) :: arena
        class(*), intent(in) :: node
        integer, intent(in) :: node_index
        character(len=:), allocatable :: code
        associate(associate_placeholder => arena)
        associate(associate_placeholder2 => node)
        associate(associate_placeholder3 => node_index)
        end associate
        end associate
        end associate
        code = "integer :: placeholder"
    end function generate_code_declaration

    function generate_code_parameter_declaration(arena, node, node_index) result(code)
        type(ast_arena_t), intent(in) :: arena
        class(*), intent(in) :: node
        integer, intent(in) :: node_index
        character(len=:), allocatable :: code
        associate(associate_placeholder => arena)
        associate(associate_placeholder2 => node)
        associate(associate_placeholder3 => node_index)
        end associate
        end associate
        end associate
        code = "integer, parameter :: placeholder = 0"
    end function generate_code_parameter_declaration

    ! Additional statement placeholders
    function generate_code_subroutine_call(arena, node, node_index) result(code)
        type(ast_arena_t), intent(in) :: arena
        class(*), intent(in) :: node
        integer, intent(in) :: node_index
        character(len=:), allocatable :: code
        associate(associate_placeholder => arena)
        associate(associate_placeholder2 => node)
        associate(associate_placeholder3 => node_index)
        end associate
        end associate
        end associate
        code = "call placeholder_subroutine"
    end function generate_code_subroutine_call

    function generate_code_if(arena, node, node_index) result(code)
        type(ast_arena_t), intent(in) :: arena
        class(*), intent(in) :: node
        integer, intent(in) :: node_index
        character(len=:), allocatable :: code
        associate(associate_placeholder => arena)
        associate(associate_placeholder2 => node)
        associate(associate_placeholder3 => node_index)
        end associate
        end associate
        end associate
        code = "if (.true.) then"//new_line('A')//"end if"
    end function generate_code_if

    function generate_code_do_loop(arena, node, node_index) result(code)
        type(ast_arena_t), intent(in) :: arena
        class(*), intent(in) :: node
        integer, intent(in) :: node_index
        character(len=:), allocatable :: code
        associate(associate_placeholder => arena)
        associate(associate_placeholder2 => node)
        associate(associate_placeholder3 => node_index)
        end associate
        end associate
        end associate
        code = "do i = 1, 10"//new_line('A')//"end do"
    end function generate_code_do_loop

    function generate_code_do_while(arena, node, node_index) result(code)
        type(ast_arena_t), intent(in) :: arena
        class(*), intent(in) :: node
        integer, intent(in) :: node_index
        character(len=:), allocatable :: code
        associate(associate_placeholder => arena)
        associate(associate_placeholder2 => node)
        associate(associate_placeholder3 => node_index)
        end associate
        end associate
        end associate
        code = "do while (.true.)"//new_line('A')//"end do"
    end function generate_code_do_while

    function generate_code_select_case(arena, node, node_index) result(code)
        type(ast_arena_t), intent(in) :: arena
        class(*), intent(in) :: node
        integer, intent(in) :: node_index
        character(len=:), allocatable :: code
        associate(associate_placeholder => arena)
        associate(associate_placeholder2 => node)
        associate(associate_placeholder3 => node_index)
        end associate
        end associate
        end associate
        code = "select case (1)"//new_line('A')//"end select"
    end function generate_code_select_case

    ! More statement placeholders
    function generate_code_stop(arena, node, node_index) result(code)
        type(ast_arena_t), intent(in) :: arena
        class(*), intent(in) :: node
        integer, intent(in) :: node_index
        character(len=:), allocatable :: code
        associate(associate_placeholder => arena)
        associate(associate_placeholder2 => node)
        associate(associate_placeholder3 => node_index)
        end associate
        end associate
        end associate
        code = "stop"
    end function generate_code_stop

    function generate_code_return(arena, node, node_index) result(code)
        type(ast_arena_t), intent(in) :: arena
        class(*), intent(in) :: node
        integer, intent(in) :: node_index
        character(len=:), allocatable :: code
        associate(associate_placeholder => arena)
        associate(associate_placeholder2 => node)
        associate(associate_placeholder3 => node_index)
        end associate
        end associate
        end associate
        code = "return"
    end function generate_code_return

    function generate_code_cycle(arena, node, node_index) result(code)
        type(ast_arena_t), intent(in) :: arena
        class(*), intent(in) :: node
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

    function generate_code_exit(arena, node, node_index) result(code)
        type(ast_arena_t), intent(in) :: arena
        class(*), intent(in) :: node
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

    function generate_code_where(arena, node, node_index) result(code)
        type(ast_arena_t), intent(in) :: arena
        class(*), intent(in) :: node
        integer, intent(in) :: node_index
        character(len=:), allocatable :: code
        associate(associate_placeholder => arena)
        associate(associate_placeholder2 => node)
        associate(associate_placeholder3 => node_index)
        end associate
        end associate
        end associate
        code = "where (.true.) ! placeholder"
    end function generate_code_where

    function generate_code_use_statement(node) result(code)
        class(*), intent(in) :: node
        character(len=:), allocatable :: code
        associate(associate_placeholder => node)
        end associate
        code = "use placeholder_module"
    end function generate_code_use_statement

    subroutine fortran_generate_code(this, arena, prog_index, options, &
                                     output, error_msg)
        class(fortran_backend_t), intent(inout) :: this
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: prog_index
        type(backend_options_t), intent(in) :: options
        character(len=:), allocatable, intent(out) :: output
        character(len=*), intent(out) :: error_msg

        associate(associate_placeholder => options)
        end associate

        ! Clear error message
        error_msg = ""

        ! Use the moved code generation logic
        output = generate_code_from_arena(arena, prog_index)

        ! Check for errors
        if (.not. allocated(output)) then
            error_msg = "Failed to generate Fortran code"
            output = ""
        end if
    end subroutine fortran_generate_code

    function fortran_get_name(this) result(name)
        class(fortran_backend_t), intent(in) :: this
        character(len=:), allocatable :: name

        associate(associate_placeholder => this)
        end associate

        name = "Fortran"
    end function fortran_get_name

    function fortran_get_version(this) result(version)
        class(fortran_backend_t), intent(in) :: this
        character(len=:), allocatable :: version

        associate(associate_placeholder => this)
        end associate

        version = "1.0.0"
    end function fortran_get_version

    ! Generate code from AST arena
    function generate_code_from_arena(arena, node_index) result(code)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        character(len=:), allocatable :: code

        code = ""
        if (node_index <= 0 .or. node_index > arena%size) return
        if (.not. allocated(arena%entries(node_index)%node)) return

        select type (node => arena%entries(node_index)%node)
        type is (literal_node)
            code = generate_code_literal(node)
        type is (identifier_node)
            code = generate_code_identifier(node)
        type is (assignment_node)
            code = generate_code_assignment(arena, node, node_index)
        type is (binary_op_node)
            code = generate_code_binary_op(arena, node, node_index)
        type is (program_node)
            code = generate_code_program(arena, node, node_index)
        type is (call_or_subscript_node)
            code = generate_code_call_or_subscript(arena, node, node_index)
        type is (subroutine_call_node)
            code = generate_code_subroutine_call(arena, node, node_index)
        type is (function_def_node)
            code = generate_code_function_def(arena, node, node_index)
        type is (subroutine_def_node)
            code = generate_code_subroutine_def(arena, node, node_index)
        type is (print_statement_node)
            code = generate_code_print_statement(arena, node, node_index)
        type is (declaration_node)
            code = generate_code_declaration(arena, node, node_index)
        type is (parameter_declaration_node)
            code = generate_code_parameter_declaration(arena, node, node_index)
        type is (if_node)
            code = generate_code_if(arena, node, node_index)
        type is (do_loop_node)
            code = generate_code_do_loop(arena, node, node_index)
        type is (do_while_node)
            code = generate_code_do_while(arena, node, node_index)
        type is (select_case_node)
            code = generate_code_select_case(arena, node, node_index)
        type is (stop_node)
            code = generate_code_stop(arena, node, node_index)
        type is (return_node)
            code = generate_code_return(arena, node, node_index)
        type is (cycle_node)
            code = generate_code_cycle(arena, node, node_index)
        type is (exit_node)
            code = generate_code_exit(arena, node, node_index)
        type is (where_node)
            code = generate_code_where(arena, node, node_index)
        type is (use_statement_node)
            code = generate_code_use_statement(node)
        type is (array_literal_node)
            code = generate_code_array_literal(arena, node, node_index)
        class default
            code = "! Unsupported node type"
        end select
    end function generate_code_from_arena

    ! Generate code for program
    function generate_code_program(arena, node, node_index) result(code)
        type(ast_arena_t), intent(in) :: arena
        type(program_node), intent(in) :: node
        integer, intent(in) :: node_index
        character(len=:), allocatable :: code
        integer :: i
        character(len=:), allocatable :: body_code, contains_code

        associate(associate_placeholder => node_index)
        end associate

        code = "program " // trim(node%name) // new_line('A')

        ! Generate body (declarations and executable statements)
        if (allocated(node%body_indices)) then
            body_code = generate_grouped_body(arena, node%body_indices, 1)
            code = code // body_code
        end if

        ! Add contains section if there are internal procedures
        contains_code = ""
        if (allocated(node%body_indices)) then
            do i = 1, size(node%body_indices)
                if (node%body_indices(i) > 0 .and. node%body_indices(i) <= arena%size) then
                    select type (body_node => arena%entries(node%body_indices(i))%node)
                    type is (function_def_node)
                        if (len_trim(contains_code) == 0) then
                            contains_code = contains_code // indent(1) // "contains" // new_line('A')
                        end if
                        contains_code = contains_code // generate_code_from_arena(arena, node%body_indices(i))
                    type is (subroutine_def_node)
                        if (len_trim(contains_code) == 0) then
                            contains_code = contains_code // indent(1) // "contains" // new_line('A')
                        end if
                        contains_code = contains_code // generate_code_from_arena(arena, node%body_indices(i))
                    end select
                end if
            end do
        end if

        code = code // contains_code
        code = code // "end program " // trim(node%name) // new_line('A')
    end function generate_code_program

    ! Generate code for function definition
    function generate_code_function_def(arena, node, node_index) result(code)
        type(ast_arena_t), intent(in) :: arena
        type(function_def_node), intent(in) :: node
        integer, intent(in) :: node_index
        character(len=:), allocatable :: code
        character(len=:), allocatable :: result_clause, params, body_code
        integer :: i

        associate(associate_placeholder => node_index)
        end associate

        ! Generate parameter list
        params = ""
        if (allocated(node%param_indices)) then
            do i = 1, size(node%param_indices)
                if (node%param_indices(i) > 0 .and. node%param_indices(i) <= arena%size) then
                    if (i > 1) params = params // ", "
                    select type (param_node => arena%entries(node%param_indices(i))%node)
                    type is (declaration_node)
                        params = params // trim(param_node%var_name)
                    type is (parameter_declaration_node)
                        params = params // trim(param_node%name)
                    end select
                end if
            end do
        end if

        ! Add result clause if return type is specified
        result_clause = ""
        if (allocated(node%return_type)) then
            result_clause = " result(" // trim(node%name) // "_result)"
        end if

        ! Function header
        code = indent(1) // "function " // trim(node%name) // "(" // params // ")" // result_clause // new_line('A')

        ! Generate body
        if (allocated(node%body_indices)) then
            body_code = generate_grouped_body(arena, node%body_indices, 2)
            code = code // body_code
        end if

        code = code // indent(1) // "end function " // trim(node%name) // new_line('A')
    end function generate_code_function_def

    ! Generate code for subroutine definition
    function generate_code_subroutine_def(arena, node, node_index) result(code)
        type(ast_arena_t), intent(in) :: arena
        type(subroutine_def_node), intent(in) :: node
        integer, intent(in) :: node_index
        character(len=:), allocatable :: code
        character(len=:), allocatable :: params, body_code
        integer :: i

        associate(associate_placeholder => node_index)
        end associate

        ! Generate parameter list
        params = ""
        if (allocated(node%param_indices)) then
            do i = 1, size(node%param_indices)
                if (node%param_indices(i) > 0 .and. node%param_indices(i) <= arena%size) then
                    if (i > 1) params = params // ", "
                    select type (param_node => arena%entries(node%param_indices(i))%node)
                    type is (declaration_node)
                        params = params // trim(param_node%var_name)
                    type is (parameter_declaration_node)
                        params = params // trim(param_node%name)
                    end select
                end if
            end do
        end if

        ! Subroutine header
        code = indent(1) // "subroutine " // trim(node%name) // "(" // params // ")" // new_line('A')

        ! Generate body
        if (allocated(node%body_indices)) then
            body_code = generate_grouped_body(arena, node%body_indices, 2)
            code = code // body_code
        end if

        code = code // indent(1) // "end subroutine " // trim(node%name) // new_line('A')
    end function generate_code_subroutine_def

    ! Utility function to convert integer to string
    function int_to_string(num) result(str)
        integer, intent(in) :: num
        character(len=:), allocatable :: str
        character(len=20) :: temp_str
        
        write(temp_str, '(I0)') num
        str = trim(temp_str)
    end function int_to_string

end module fortran_backend_core