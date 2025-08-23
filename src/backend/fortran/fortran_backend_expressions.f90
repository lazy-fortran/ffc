module fortran_backend_expressions
    use backend_interface
    use ast_core
    use type_system_unified, only: TINT, TREAL, TCHAR, TLOGICAL
    use ast_nodes_data, only: intent_type_to_string, INTENT_NONE
    use string_type, only: string_t
    use codegen_indent
    use fortran_backend_core, only: generate_code_from_arena
    implicit none
    private

    public :: generate_code_literal, generate_code_identifier
    public :: generate_code_binary_op, generate_code_call_or_subscript
    public :: generate_code_array_literal
    public :: find_node_index_in_arena, same_node

contains

    ! Generate code for literal node
    function generate_code_literal(node) result(code)
        type(literal_node), intent(in) :: node
        character(len=:), allocatable :: code

        ! Handle different literal types
        select case (node%literal_kind)
        case (LITERAL_STRING)
            ! Check for special case of bare string (not enclosed in quotes)
            if (allocated(node%value) .and. len_trim(node%value) > 0) then
                if (node%value(1:1) /= '"' .and. node%value(1:1) /= "'") then
                    ! Bare string - just return as is (for things like "implicit none")
                    code = node%value
                else
                    ! Quoted string - return as is
                    code = node%value
                end if
            else
                code = "''"
            end if
        case (LITERAL_INTEGER)
            ! Ensure integer literals don't have leading/trailing spaces
            if (allocated(node%value) .and. len_trim(node%value) > 0) then
                code = trim(node%value)
            else
                code = "0"
            end if
        case (LITERAL_REAL)
            ! For real literals, ensure double precision by adding 'd0' suffix if needed
            if (index(node%value, 'd') == 0 .and. index(node%value, 'D') == 0 .and. &
                index(node%value, '_') == 0) then
                code = node%value//"d0"
            else
                code = node%value
            end if
        case default
            ! Handle invalid/empty literals safely
            if (allocated(node%value) .and. len_trim(node%value) > 0) then
                code = node%value
            else
                code = "! Invalid literal node"
            end if
        end select
    end function generate_code_literal

    ! Generate code for identifier node
    function generate_code_identifier(node) result(code)
        type(identifier_node), intent(in) :: node
        character(len=:), allocatable :: code

        ! Simply return the identifier name
        code = node%name
    end function generate_code_identifier

    ! Generate code for binary operation node
    function generate_code_binary_op(arena, node, node_index) result(code)
        type(ast_arena_t), intent(in) :: arena
        type(binary_op_node), intent(in) :: node
        integer, intent(in) :: node_index
        character(len=:), allocatable :: code
        character(len=:), allocatable :: left_code, right_code

        associate(associate_placeholder => node_index)
        end associate

        ! Generate code for operands
        if (node%left_index > 0 .and. node%left_index <= arena%size) then
            left_code = generate_code_from_arena(arena, node%left_index)
        else
            left_code = ""
        end if

        if (node%right_index > 0 .and. node%right_index <= arena%size) then
            right_code = generate_code_from_arena(arena, node%right_index)
        else
            right_code = ""
        end if

        ! Combine with operator - match fprettify spacing rules
        ! fprettify: * and / get no spaces, +/- and comparisons get spaces
        if (trim(node%operator) == ':') then
            ! Array slicing operator
            if (len(left_code) == 0) then
                ! Empty lower bound: :upper
                code = ":"//right_code
            else if (len(right_code) == 0) then
                ! Empty upper bound: lower:
                code = left_code//":"
            else
                ! Both bounds: lower:upper
                code = left_code//":"//right_code
            end if
        else if (trim(node%operator) == '*' .or. trim(node%operator) == '/') then
            code = left_code//node%operator//right_code
        else
            code = left_code//" "//node%operator//" "//right_code
        end if
    end function generate_code_binary_op

    ! Generate code for call_or_subscript node (handles both function calls and array indexing)
    function generate_code_call_or_subscript(arena, node, node_index) result(code)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        integer, intent(in) :: node_index
        character(len=:), allocatable :: code
        character(len=:), allocatable :: args_code
        character(len=:), allocatable :: arg_code
        integer :: i
        logical :: is_array_slice
        type(binary_op_node), pointer :: bin_op

        associate(associate_placeholder => node_index)
        end associate

        ! Generate arguments
        args_code = ""
        if (allocated(node%arg_indices)) then
            ! Check if this might be array slicing (single argument that's a binary op with ":")
            is_array_slice = .false.
            if (size(node%arg_indices) == 1 .and. node%arg_indices(1) > 0) then
                select type (arg_node => arena%entries(node%arg_indices(1))%node)
                type is (binary_op_node)
                    bin_op => arg_node
                    if (trim(bin_op%operator) == ':') then
                        is_array_slice = .true.
                    end if
                end select
            end if

            do i = 1, size(node%arg_indices)
                if (node%arg_indices(i) > 0 .and. node%arg_indices(i) <= arena%size) then
                    if (i > 1) args_code = args_code // ", "
                    arg_code = generate_code_from_arena(arena, node%arg_indices(i))
                    args_code = args_code // arg_code
                end if
            end do
        end if

        ! Combine name with arguments/indices
        if (len(args_code) > 0) then
            code = node%name // "(" // args_code // ")"
        else
            code = node%name // "()"
        end if
    end function generate_code_call_or_subscript

    ! Generate code for array literal
    function generate_code_array_literal(arena, node, node_index) result(code)
        type(ast_arena_t), intent(in) :: arena
        type(array_literal_node), intent(in) :: node
        integer, intent(in) :: node_index
        character(len=:), allocatable :: code
        character(len=:), allocatable :: elements_code
        integer :: i

        associate(associate_placeholder => node_index)
        end associate

        if (allocated(node%element_indices)) then
            elements_code = ""
            do i = 1, size(node%element_indices)
                if (node%element_indices(i) > 0 .and. node%element_indices(i) <= arena%size) then
                    if (i > 1) elements_code = elements_code // ", "

                    ! Generate code for array element
                    elements_code = elements_code // generate_code_from_arena(arena, node%element_indices(i))
                end if
            end do
            code = "[" // elements_code // "]"
        else
            code = "[]"
        end if
    end function generate_code_array_literal


    ! Find the index of a node in the arena
    function find_node_index_in_arena(arena, target_node) result(index)
        type(ast_arena_t), intent(in) :: arena
        class(ast_node), intent(in) :: target_node
        integer :: index
        integer :: i

        index = 0
        do i = 1, arena%size
            if (allocated(arena%entries(i)%node)) then
                if (same_node(arena%entries(i)%node, target_node)) then
                    index = i
                    return
                end if
            end if
        end do
    end function find_node_index_in_arena

    ! Check if two nodes are the same (simple pointer comparison)
    function same_node(node1, node2) result(is_same)
        class(ast_node), intent(in) :: node1, node2
        logical :: is_same

        ! For now, just use a simple type and basic content comparison
        ! This is a simplified implementation
        is_same = .false.
        
        select type (node1)
        type is (identifier_node)
            select type (node2)
            type is (identifier_node)
                is_same = (trim(node1%name) == trim(node2%name))
            end select
        type is (literal_node)
            select type (node2)
            type is (literal_node)
                is_same = (node1%literal_kind == node2%literal_kind .and. &
                          trim(node1%value) == trim(node2%value))
            end select
        end select
    end function same_node

end module fortran_backend_expressions