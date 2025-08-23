module mlir_backend_operators
    use backend_interface
    use ast_core
    use mlir_backend_types
    use mlir_utils, mlir_int_to_str => int_to_str
    use mlir_expressions
    implicit none
    private

    public :: is_generic_procedure_call, generate_mlir_generic_procedure_call
    public :: find_interface_block, resolve_generic_procedure
    public :: is_operator_overloaded, generate_mlir_operator_overload_call
    public :: generate_mlir_assignment_overload_call, resolve_operator_procedure
    public :: resolve_assignment_procedure, get_procedure_name_from_node

contains

    ! Check if a function call is to a generic procedure
    function is_generic_procedure_call(backend, arena, name) result(is_generic)
        class(mlir_backend_t), intent(in) :: backend
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: name
        logical :: is_generic
        integer :: i

        associate(associate_placeholder => backend)
        end associate

        is_generic = .false.

        ! Search through the arena for interface blocks with this name
        do i = 1, arena%size
            if (allocated(arena%entries(i)%node)) then
                select type (node => arena%entries(i)%node)
                type is (interface_block_node)
                    if (allocated(node%name) .and. trim(node%name) == trim(name)) then
                        ! Found a generic interface with this name
                        is_generic = .true.
                        return
                    end if
                end select
            end if
        end do
    end function is_generic_procedure_call

    ! Generate MLIR for generic procedure call with runtime dispatch
    function generate_mlir_generic_procedure_call(backend, arena, node, indent_str) result(mlir)
        class(mlir_backend_t), intent(inout) :: backend
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        character(len=*), intent(in) :: indent_str
        character(len=:), allocatable :: mlir
        integer :: interface_idx, resolved_proc_idx
        character(len=:), allocatable :: resolved_name, result_ssa

        mlir = ""

        ! Find the interface block for this generic procedure
        interface_idx = find_interface_block(arena, trim(node%name))
        if (interface_idx == 0) then
            ! No interface found - generate error or fallback
            mlir = indent_str//"// Error: Generic interface '"//trim(node%name)//"' not found"//new_line('a')
            return
        end if

        ! For now, implement simple static resolution (first procedure in interface)
        ! TODO: Implement full type-based resolution with argument matching
        resolved_proc_idx = resolve_generic_procedure(backend, arena, interface_idx, node)

        if (resolved_proc_idx == 0) then
            ! Resolution failed - generate error
            mlir = indent_str//"// Error: Cannot resolve generic procedure '"//trim(node%name)//"'"//new_line('a')
            return
        end if

        ! Get the resolved procedure name
        if (resolved_proc_idx > 0 .and. resolved_proc_idx <= arena%size .and. &
            allocated(arena%entries(resolved_proc_idx)%node)) then
            resolved_name = get_procedure_name_from_node(arena%entries(resolved_proc_idx)%node)
        else
            resolved_name = "unknown_procedure"
        end if

        ! Generate call to the resolved specific procedure
        backend%ssa_counter = backend%ssa_counter + 1
        result_ssa = "%"//mlir_int_to_str(backend%ssa_counter)
        backend%last_ssa_value = result_ssa

        if (allocated(node%arg_indices) .and. size(node%arg_indices) > 0) then
            ! Function call with arguments - delegate to existing function call generation
            mlir = indent_str//result_ssa//" = fir.call @"//trim(resolved_name)//"() : () -> i32"//new_line('a')
        else
            ! Function call without arguments
            mlir = indent_str//result_ssa//" = fir.call @"//trim(resolved_name)//"() : () -> i32"//new_line('a')
        end if
    end function generate_mlir_generic_procedure_call

    ! Find interface block by name
    function find_interface_block(arena, name) result(interface_idx)
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: name
        integer :: interface_idx
        integer :: i

        interface_idx = 0

        do i = 1, arena%size
            if (allocated(arena%entries(i)%node)) then
                select type (node => arena%entries(i)%node)
                type is (interface_block_node)
                    if (allocated(node%name) .and. trim(node%name) == trim(name)) then
                        interface_idx = i
                        return
                    end if
                end select
            end if
        end do
    end function find_interface_block

    ! Resolve generic procedure based on arguments (simplified version)
    function resolve_generic_procedure(backend, arena, interface_idx, call_node) result(resolved_idx)
        class(mlir_backend_t), intent(in) :: backend
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: interface_idx
        type(call_or_subscript_node), intent(in) :: call_node
        integer :: resolved_idx
        integer :: i

        associate(associate_placeholder => backend)
        associate(associate_placeholder2 => call_node)
        end associate
        end associate

        resolved_idx = 0

        ! Get the interface block
        select type (iface_node => arena%entries(interface_idx)%node)
        type is (interface_block_node)
            if (allocated(iface_node%procedure_indices) .and. &
                size(iface_node%procedure_indices) > 0) then

                ! For now, simple resolution: return the first procedure
                ! TODO: Implement proper type-based resolution by examining argument types
                do i = 1, size(iface_node%procedure_indices)
                    if (iface_node%procedure_indices(i) > 0 .and. &
                        iface_node%procedure_indices(i) <= arena%size) then
                        resolved_idx = iface_node%procedure_indices(i)
                        return
                    end if
                end do
            end if
        end select
    end function resolve_generic_procedure

    ! Check if an operator is overloaded
    function is_operator_overloaded(backend, arena, operator_name) result(is_overloaded)
        class(mlir_backend_t), intent(in) :: backend
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: operator_name
        logical :: is_overloaded
        integer :: i
        character(len=:), allocatable :: full_operator_name

        associate(associate_placeholder => backend)
        end associate

        is_overloaded = .false.

        ! Construct the full operator interface name
        if (trim(operator_name) == "assignment(=)") then
            full_operator_name = "assignment(=)"
        else
            full_operator_name = "operator("//trim(operator_name)//")"
        end if

        ! Search through the arena for interface blocks with operator names
        do i = 1, arena%size
            if (allocated(arena%entries(i)%node)) then
                select type (node => arena%entries(i)%node)
                type is (interface_block_node)
                    if (allocated(node%name) .and. trim(node%name) == trim(full_operator_name)) then
                        is_overloaded = .true.
                        return
                    end if
                end select
            end if
        end do
    end function is_operator_overloaded

    ! Generate MLIR for operator overload call (binary operations)
    function generate_mlir_operator_overload_call(backend, arena, node, indent_str) result(mlir)
        class(mlir_backend_t), intent(inout) :: backend
        type(ast_arena_t), intent(in) :: arena
        type(binary_op_node), intent(in) :: node
        character(len=*), intent(in) :: indent_str
        character(len=:), allocatable :: mlir
        character(len=:), allocatable :: full_operator_name, resolved_name, result_ssa
        integer :: interface_idx, resolved_proc_idx

        mlir = ""

        ! Construct the full operator interface name
        full_operator_name = "operator("//trim(node%operator)//")"

        ! Find the interface block for this operator
        interface_idx = find_interface_block(arena, trim(full_operator_name))
        if (interface_idx == 0) then
            mlir = indent_str//"// Error: Operator interface '"//trim(full_operator_name)//"' not found"//new_line('a')
            return
        end if

        ! Resolve to the specific procedure (simplified - take first one)
        resolved_proc_idx = resolve_operator_procedure(backend, arena, interface_idx, node)
        if (resolved_proc_idx == 0) then
            mlir = indent_str//"// Error: Cannot resolve operator procedure"//new_line('a')
            return
        end if

        ! Get the resolved procedure name
        if (resolved_proc_idx > 0 .and. resolved_proc_idx <= arena%size .and. &
            allocated(arena%entries(resolved_proc_idx)%node)) then
            resolved_name = get_procedure_name_from_node(arena%entries(resolved_proc_idx)%node)
        else
            resolved_name = "unknown_procedure"
        end if

        ! Generate the operator arguments and function call
        mlir = mlir//generate_mlir_expression(backend, arena, node%left_index, indent_str)
        mlir = mlir//generate_mlir_expression(backend, arena, node%right_index, indent_str)

        ! Generate function call to the overloaded operator
        backend%ssa_counter = backend%ssa_counter + 1
        result_ssa = "%"//mlir_int_to_str(backend%ssa_counter)
        backend%last_ssa_value = result_ssa

        ! For now, generate simple function call (TODO: handle argument passing)
        mlir = mlir//indent_str//result_ssa//" = fir.call @"//trim(resolved_name)//"() : () -> i32"//new_line('a')
    end function generate_mlir_operator_overload_call

    ! Generate MLIR for assignment overload call
    function generate_mlir_assignment_overload_call(backend, arena, node, indent_str) result(mlir)
        class(mlir_backend_t), intent(inout) :: backend
        type(ast_arena_t), intent(in) :: arena
        type(assignment_node), intent(in) :: node
        character(len=*), intent(in) :: indent_str
        character(len=:), allocatable :: mlir
        character(len=:), allocatable :: resolved_name, target_ref, value_ref
        character(len=:), allocatable :: target_ssa, value_ssa
        integer :: interface_idx, resolved_proc_idx

        mlir = ""

        ! Find the assignment interface block
        interface_idx = find_interface_block(arena, "assignment(=)")
        if (interface_idx == 0) then
            mlir = indent_str//"// Error: Assignment interface not found"//new_line('a')
            return
        end if

        ! Resolve to the specific procedure (simplified - take first one)
        resolved_proc_idx = resolve_assignment_procedure(backend, arena, interface_idx, node)
        if (resolved_proc_idx == 0) then
            mlir = indent_str//"// Error: Cannot resolve assignment procedure"//new_line('a')
            return
        end if

        ! Get the resolved procedure name
        if (resolved_proc_idx > 0 .and. resolved_proc_idx <= arena%size .and. &
            allocated(arena%entries(resolved_proc_idx)%node)) then
            resolved_name = get_procedure_name_from_node(arena%entries(resolved_proc_idx)%node)
        else
            resolved_name = "unknown_procedure"
        end if

        ! Generate target expression (lhs) and capture SSA value
        target_ref = generate_mlir_expression(backend, arena, node%target_index, indent_str)
        mlir = mlir//target_ref
        target_ssa = backend%last_ssa_value

        ! Generate value expression (rhs) and capture SSA value
        value_ref = generate_mlir_expression(backend, arena, node%value_index, indent_str)
        mlir = mlir//value_ref
        value_ssa = backend%last_ssa_value

        ! Generate subroutine call to the overloaded assignment with proper HLFIR types
        ! Assignment overloading uses fir.call but with proper !fir.* types instead of !fir.ref<none>
        mlir = mlir//indent_str//"fir.call @"//trim(resolved_name)//"("//trim(target_ssa)//", "//trim(value_ssa)// &
               ") : (!fir.ref<!fir.type<vector>>, !fir.ref<!fir.type<vector>>) -> ()"//new_line('a')
    end function generate_mlir_assignment_overload_call

    ! Resolve operator procedure (simplified version)
    function resolve_operator_procedure(backend, arena, interface_idx, op_node) result(resolved_idx)
        class(mlir_backend_t), intent(in) :: backend
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: interface_idx
        type(binary_op_node), intent(in) :: op_node
        integer :: resolved_idx

        associate(associate_placeholder => backend)
        associate(associate_placeholder2 => op_node)
        end associate
        end associate

        ! For now, simple resolution - use the generic procedure resolver
        ! TODO: Implement operator-specific resolution based on operand types
        resolved_idx = 0

        select type (iface_node => arena%entries(interface_idx)%node)
        type is (interface_block_node)
            if (allocated(iface_node%procedure_indices) .and. &
                size(iface_node%procedure_indices) > 0) then
                resolved_idx = iface_node%procedure_indices(1)
            end if
        end select
    end function resolve_operator_procedure

    ! Resolve assignment procedure (simplified version)
    function resolve_assignment_procedure(backend, arena, interface_idx, assign_node) result(resolved_idx)
        class(mlir_backend_t), intent(in) :: backend
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: interface_idx
        type(assignment_node), intent(in) :: assign_node
        integer :: resolved_idx

        associate(associate_placeholder => backend)
        associate(associate_placeholder2 => assign_node)
        end associate
        end associate

        ! For now, simple resolution - use the first procedure in interface
        resolved_idx = 0

        select type (iface_node => arena%entries(interface_idx)%node)
        type is (interface_block_node)
            if (allocated(iface_node%procedure_indices) .and. &
                size(iface_node%procedure_indices) > 0) then
                resolved_idx = iface_node%procedure_indices(1)
            end if
        end select
    end function resolve_assignment_procedure

    ! Get procedure name from various node types
    function get_procedure_name_from_node(node) result(name)
        class(ast_node), intent(in) :: node
        character(len=:), allocatable :: name

        name = "unknown"

        select type (node)
        type is (function_def_node)
            name = trim(node%name)
        type is (subroutine_def_node)
            name = trim(node%name)
        class default
            name = "unknown_procedure"
        end select
    end function get_procedure_name_from_node

end module mlir_backend_operators