module fortran_backend
    use fortran_backend_core, only: fortran_backend_t, generate_code_from_arena, generate_code_program, &
                                    generate_code_function_def, generate_code_subroutine_def
    use fortran_backend_statements, only: generate_code_assignment, generate_code_print_statement, &
                                          generate_code_declaration, generate_code_parameter_declaration, &
                                          generate_code_if, generate_code_do_loop, generate_code_do_while, &
                                          generate_code_select_case, generate_code_stop, generate_code_return, &
                                          generate_code_cycle, generate_code_exit, generate_code_where, &
                                          generate_code_use_statement, generate_code_subroutine_call, &
                                          generate_grouped_body, can_group_declarations, can_group_parameters, &
                                          generate_grouped_declaration, build_param_name_with_dims
    use fortran_backend_expressions, only: generate_code_literal, generate_code_identifier, &
                                           generate_code_binary_op, generate_code_call_or_subscript, &
                                           generate_code_array_literal, find_node_index_in_arena, same_node
    implicit none
    private

    ! Re-export main backend type
    public :: fortran_backend_t

    ! From fortran_backend_core
    public :: generate_code_from_arena, generate_code_program
    public :: generate_code_function_def, generate_code_subroutine_def

    ! From fortran_backend_statements  
    public :: generate_code_assignment, generate_code_print_statement
    public :: generate_code_declaration, generate_code_parameter_declaration
    public :: generate_code_if, generate_code_do_loop, generate_code_do_while
    public :: generate_code_select_case, generate_code_stop, generate_code_return
    public :: generate_code_cycle, generate_code_exit, generate_code_where
    public :: generate_code_use_statement, generate_code_subroutine_call
    public :: generate_grouped_body, can_group_declarations, can_group_parameters
    public :: generate_grouped_declaration, build_param_name_with_dims

    ! From fortran_backend_expressions
    public :: generate_code_literal, generate_code_identifier
    public :: generate_code_binary_op, generate_code_call_or_subscript
    public :: generate_code_array_literal
    public :: find_node_index_in_arena, same_node

end module fortran_backend