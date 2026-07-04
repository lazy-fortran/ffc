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
    use ast_nodes_control, only: goto_node
    implicit none
    private
    public :: ast_arena_t, node_exists, get_node_type_at
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
            if (len_trim(var_name) == 0 .and. node%is_multi_declaration .and. &
                allocated(node%var_names)) then
                if (size(node%var_names) == 1) var_name = node%var_names(1)
            end if
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

    ! Statement label attached to any node (e.g. the 100 in `100 continue`).
    ! Empty string when the statement carries no label.
    function get_node_stmt_label(arena, node_index) result(label)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        character(len=:), allocatable :: label

        call set_empty(label)
        if (.not. node_exists(arena, node_index)) return
        if (allocated(arena%entries(node_index)%node%stmt_label)) then
            label = arena%entries(node_index)%node%stmt_label
        end if
    end function get_node_stmt_label

    ! Target label of a simple `goto N`. Empty string for a non-goto node or a
    ! computed goto (which carries label_list instead).
    function get_goto_label(arena, node_index) result(label)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        character(len=:), allocatable :: label

        call set_empty(label)
        if (.not. node_exists(arena, node_index)) return
        select type (node => arena%entries(node_index)%node)
            type is (goto_node)
            if (allocated(node%label)) label = node%label
        end select
    end function get_goto_label

    ! A computed goto carries a comma-separated label_list and a selector;
    ! it is out of scope for the simple branch lowering.
    logical function goto_is_computed(arena, node_index) result(is_computed)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index

        is_computed = .false.
        if (.not. node_exists(arena, node_index)) return
        select type (node => arena%entries(node_index)%node)
            type is (goto_node)
            is_computed = node%selector_index /= 0 .or. &
                allocated(node%label_list)
        end select
    end function goto_is_computed

    ! Comma-separated target labels of a computed `goto (L1, L2, ...), expr`.
    ! Empty string for a non-computed goto.
    function get_goto_label_list(arena, node_index) result(label_list)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        character(len=:), allocatable :: label_list

        call set_empty(label_list)
        if (.not. node_exists(arena, node_index)) return
        select type (node => arena%entries(node_index)%node)
            type is (goto_node)
            if (allocated(node%label_list)) label_list = node%label_list
        end select
    end function get_goto_label_list

    ! Arena index of the selector expression of a computed goto; 0 otherwise.
    integer function get_goto_selector_index(arena, node_index) result(idx)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index

        idx = 0
        if (.not. node_exists(arena, node_index)) return
        select type (node => arena%entries(node_index)%node)
            type is (goto_node)
            idx = node%selector_index
        end select
    end function get_goto_selector_index

    subroutine set_empty(value)
        character(len=:), allocatable, intent(out) :: value
        allocate (character(len=0) :: value)
    end subroutine set_empty

end module ffc_fortfront_queries
