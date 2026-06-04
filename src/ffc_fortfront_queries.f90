module ffc_fortfront_queries
    use fortfront_compiler, only: ast_arena_t, get_program_body_info, &
                                  get_module_body_info, &
                                  get_function_body_info, &
                                  get_subroutine_body_info, &
                                  get_select_case_info, get_case_block_info, &
                                  get_case_default_body, get_case_range_info, &
                                  get_select_type_info, get_type_guard_info
    use fortfront_utils, only: node_exists, get_node_type_at
    use fortfront, only: is_derived_type_node, derived_type_node, &
                         is_declaration_node, declaration_node
    implicit none
    private
    public :: ast_arena_t, node_exists, get_node_type_at
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

contains

    subroutine get_derived_type_name(arena, type_index, name, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: type_index
        character(len=:), allocatable, intent(out) :: name
        character(len=:), allocatable, intent(out) :: error_msg

        call set_empty(name)
        if (.not. node_exists(arena, type_index)) then
            error_msg = 'derived type index does not reference an AST node'
            return
        end if
        select type (node => arena%entries(type_index)%node)
        type is (derived_type_node)
            if (allocated(node%name)) name = node%name
            call set_empty(error_msg)
        class default
            error_msg = 'AST node is not a derived type definition'
        end select
    end subroutine get_derived_type_name

    subroutine get_derived_type_components(arena, type_index, components)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: type_index
        integer, allocatable, intent(out) :: components(:)

        if (.not. node_exists(arena, type_index)) then
            allocate (components(0))
            return
        end if
        select type (node => arena%entries(type_index)%node)
        type is (derived_type_node)
            if (allocated(node%component_indices)) then
                components = node%component_indices
            else
                allocate (components(0))
            end if
        class default
            allocate (components(0))
        end select
    end subroutine get_derived_type_components

    subroutine get_declaration_var_name(arena, decl_index, var_name, &
                                        error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: decl_index
        character(len=:), allocatable, intent(out) :: var_name
        character(len=:), allocatable, intent(out) :: error_msg

        call set_empty(var_name)
        if (.not. node_exists(arena, decl_index)) then
            error_msg = 'declaration index does not reference an AST node'
            return
        end if
        select type (node => arena%entries(decl_index)%node)
        type is (declaration_node)
            if (allocated(node%var_name)) var_name = node%var_name
            call set_empty(error_msg)
        class default
            error_msg = 'AST node is not a declaration'
        end select
    end subroutine get_declaration_var_name

    subroutine get_declaration_type_name(arena, decl_index, type_name, &
                                         error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: decl_index
        character(len=:), allocatable, intent(out) :: type_name
        character(len=:), allocatable, intent(out) :: error_msg

        call set_empty(type_name)
        if (.not. node_exists(arena, decl_index)) then
            error_msg = 'declaration index does not reference an AST node'
            return
        end if
        select type (node => arena%entries(decl_index)%node)
        type is (declaration_node)
            if (allocated(node%type_name)) type_name = node%type_name
            call set_empty(error_msg)
        class default
            error_msg = 'AST node is not a declaration'
        end select
    end subroutine get_declaration_type_name

    logical function get_declaration_has_initializer(arena, decl_index) &
        result(has_init)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: decl_index

        has_init = .false.
        if (.not. node_exists(arena, decl_index)) return
        select type (node => arena%entries(decl_index)%node)
        type is (declaration_node)
            has_init = node%has_initializer
        end select
    end function get_declaration_has_initializer

    function get_declaration_initializer_index(arena, decl_index) &
        result(init_index)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: decl_index
        integer :: init_index

        init_index = 0
        if (.not. node_exists(arena, decl_index)) return
        select type (node => arena%entries(decl_index)%node)
        type is (declaration_node)
            if (node%has_initializer .and. node%initializer_index > 0) then
                init_index = node%initializer_index
            end if
        end select
    end function get_declaration_initializer_index

    subroutine set_empty(value)
        character(len=:), allocatable, intent(out) :: value
        allocate (character(len=0) :: value)
    end subroutine set_empty

end module ffc_fortfront_queries
