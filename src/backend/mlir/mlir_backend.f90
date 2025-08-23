module mlir_backend
    use mlir_backend_core, only: generate_mlir_program, generate_mlir_node, generate_mlir_module, &
                                 generate_mlir_interface_block, generate_mlir_function_declaration, &
                                 generate_mlir_subroutine_declaration, generate_mlir_module_node, &
                                 generate_mlir_use_statement, resolve_module_symbols, &
                                 generate_mlir_where_construct, generate_mlir_derived_type
    use mlir_backend_helpers, only: format_string_to_array, generate_hlfir_constant, generate_hlfir_load, &
                                    generate_hlfir_string_literal, int_to_char
    use mlir_backend_functions, only: generate_mlir_program_functions, generate_mlir_function, &
                                      generate_mlir_subroutine, generate_function_parameter_list, &
                                      generate_function_call_with_args, is_function_name
    use mlir_backend_statements, only: generate_mlir_declaration, generate_mlir_assignment, &
                                       generate_mlir_pointer_assignment, is_pointer_variable, &
                                       generate_mlir_subroutine_call, generate_mlir_return, &
                                       generate_mlir_exit, generate_mlir_cycle, &
                                       generate_mlir_allocate_statement, generate_array_size_calculation, &
                                       generate_mlir_deallocate_statement
    use mlir_backend_operators, only: is_generic_procedure_call, generate_mlir_generic_procedure_call, &
                                      find_interface_block, resolve_generic_procedure, &
                                      is_operator_overloaded, generate_mlir_operator_overload_call, &
                                      generate_mlir_assignment_overload_call, resolve_operator_procedure, &
                                      resolve_assignment_procedure, get_procedure_name_from_node
    use mlir_backend_output, only: generate_string_runtime_output, generate_integer_runtime_output, &
                                   generate_real_runtime_output, generate_logical_runtime_output, &
                                   generate_variable_runtime_output, generate_expression_runtime_output, &
                                   generate_call_runtime_output
    implicit none
    private

    ! Re-export all public interfaces from decomposed modules
    public :: generate_mlir_program, generate_mlir_node, generate_mlir_module

    ! From mlir_backend_helpers
    public :: format_string_to_array, generate_hlfir_constant, generate_hlfir_load
    public :: generate_hlfir_string_literal, int_to_char

    ! From mlir_backend_functions  
    public :: generate_mlir_program_functions, generate_mlir_function
    public :: generate_mlir_subroutine, generate_function_parameter_list
    public :: generate_function_call_with_args, is_function_name

    ! From mlir_backend_statements
    public :: generate_mlir_declaration, generate_mlir_assignment
    public :: generate_mlir_pointer_assignment, is_pointer_variable
    public :: generate_mlir_subroutine_call, generate_mlir_return
    public :: generate_mlir_exit, generate_mlir_cycle
    public :: generate_mlir_allocate_statement, generate_array_size_calculation
    public :: generate_mlir_deallocate_statement

    ! From mlir_backend_operators
    public :: is_generic_procedure_call, generate_mlir_generic_procedure_call
    public :: find_interface_block, resolve_generic_procedure
    public :: is_operator_overloaded, generate_mlir_operator_overload_call
    public :: generate_mlir_assignment_overload_call, resolve_operator_procedure
    public :: resolve_assignment_procedure, get_procedure_name_from_node

    ! From mlir_backend_output
    public :: generate_string_runtime_output, generate_integer_runtime_output
    public :: generate_real_runtime_output, generate_logical_runtime_output
    public :: generate_variable_runtime_output, generate_expression_runtime_output
    public :: generate_call_runtime_output

    ! From mlir_backend_core (additional functions)
    public :: generate_mlir_interface_block, generate_mlir_function_declaration
    public :: generate_mlir_subroutine_declaration, generate_mlir_module_node
    public :: generate_mlir_use_statement, resolve_module_symbols
    public :: generate_mlir_where_construct, generate_mlir_derived_type

end module mlir_backend