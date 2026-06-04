module ffc_fortfront_queries
    use fortfront_compiler, only: ast_arena_t, get_program_body_info, &
                                  get_module_body_info, &
                                  get_function_body_info, &
                                  get_subroutine_body_info, &
                                  get_select_case_info, get_case_block_info, &
                                  get_case_default_body, get_case_range_info, &
                                  get_select_type_info, get_type_guard_info
    use fortfront_utils, only: node_exists, get_node_type_at
    implicit none
    private
    public :: ast_arena_t, node_exists, get_node_type_at
    public :: get_program_body_info, get_module_body_info
    public :: get_function_body_info, get_subroutine_body_info
    public :: get_select_case_info, get_case_block_info
    public :: get_case_default_body, get_case_range_info
    public :: get_select_type_info, get_type_guard_info

end module ffc_fortfront_queries
