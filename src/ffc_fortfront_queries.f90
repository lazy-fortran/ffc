module ffc_fortfront_queries
    use fortfront_compiler, only: ast_arena_t, get_program_body_info, &
        get_module_body_info, &
        get_function_body_info, &
        get_subroutine_body_info, &
        get_select_case_info, get_case_block_info, &
        get_case_default_body, get_case_range_info, &
        get_select_type_info, get_type_guard_info
    use fortfront_compiler, only: node_exists, get_node_type_at, &
        get_type_for_node, mono_type_t, &
        TINT, TREAL, TCHAR, TLOGICAL, TARRAY, TCOMPLEX, TDOUBLE, TDERIVED, &
        is_derived_type_node, is_declaration_node, &
        get_derived_type_name, get_derived_type_components, &
        get_declaration_var_name, get_declaration_type_name, &
        get_declaration_has_initializer, &
        get_declaration_initializer_index, &
        get_node_stmt_label, get_goto_label, goto_is_computed, &
        get_goto_label_list, get_goto_selector_index
    implicit none
    private
    public :: ast_arena_t, node_exists, get_node_type_at
    public :: get_type_for_node
    public :: mono_type_t
    public :: TINT, TREAL, TCHAR, TLOGICAL, TARRAY, TCOMPLEX, TDOUBLE, TDERIVED
    public :: get_node_stmt_label, get_goto_label, goto_is_computed
    public :: get_goto_label_list, get_goto_selector_index
    public :: get_program_body_info, get_module_body_info
    public :: get_function_body_info, get_subroutine_body_info
    public :: get_select_case_info, get_case_block_info
    public :: get_case_default_body, get_case_range_info
    public :: get_select_type_info, get_type_guard_info
    public :: is_derived_type_node, is_declaration_node
    public :: get_derived_type_name
    public :: get_derived_type_components
    public :: get_declaration_var_name
    public :: get_declaration_type_name
    public :: get_declaration_has_initializer
    public :: get_declaration_initializer_index

end module ffc_fortfront_queries
