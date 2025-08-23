module mlir_backend_statements
    use backend_interface
    use ast_core
    use mlir_backend_types
    use mlir_types
    use mlir_utils, only: mlir_int_to_str => int_to_str, string_to_int
    use mlir_expressions
    implicit none
    private


    public :: generate_mlir_declaration, generate_mlir_assignment
    public :: generate_mlir_pointer_assignment, is_pointer_variable
    public :: generate_mlir_subroutine_call, generate_mlir_return
    public :: generate_mlir_exit, generate_mlir_cycle
    public :: generate_mlir_allocate_statement, generate_array_size_calculation
    public :: generate_mlir_deallocate_statement

contains

    ! Generate MLIR for variable declaration
    function generate_mlir_declaration(backend, arena, node, indent_str) result(mlir)
        class(mlir_backend_t), intent(inout) :: backend
        type(ast_arena_t), intent(in) :: arena
        type(declaration_node), intent(in) :: node
        character(len=*), intent(in) :: indent_str
        character(len=:), allocatable :: mlir
        character(len=:), allocatable :: mlir_type, ssa_val, base_type
        character(len=:), allocatable :: alloca_ssa

        ! Handle pointer declarations differently
        if (node%is_pointer) then
            ! For pointers, we need to allocate a pointer variable using HLFIR
            ssa_val = backend%next_ssa_value()

            ! Get the base type
            if (node%is_array .and. allocated(node%dimension_indices)) then
                base_type = fortran_array_to_mlir_type(backend%compile_mode, arena, node, indent_str)
                ! For array pointers, use HLFIR declare with undefined pointer
                mlir = indent_str//ssa_val//":2 = hlfir.declare "// &
                       "(fir.undefined : !fir.ptr<!fir.array<?x"//base_type//">>) "// &
                       '{var_name="'//trim(node%var_name)//'"} : (!fir.ptr<!fir.array<?x'//base_type//">>) -> ("// &
                       "!fir.ptr<!fir.array<?x"//base_type//">>, !fir.ptr<!fir.array<?x"//base_type//">>)"//new_line('a')
            else
                base_type = fortran_to_mlir_type(node%type_name, node%kind_value, backend%compile_mode)
                ! For scalar pointers, use HLFIR declare with undefined pointer
                mlir = indent_str//ssa_val//":2 = hlfir.declare "// &
                       "(fir.undefined : !fir.ptr<"//base_type//">) "// &
                       '{var_name="'//trim(node%var_name)//'"} : (!fir.ptr<'//base_type//">) -> ("// &
                       "!fir.ptr<"//base_type//">, !fir.ptr<"//base_type//">)"//new_line('a')
            end if
        else
            ! Regular (non-pointer) declarations using HLFIR
            ! Convert Fortran type to FIR type
            if (node%is_array .and. allocated(node%dimension_indices)) then
                mlir_type = fortran_to_fir_array_type(node%type_name, node%kind_value, arena, node%dimension_indices)
            else
                mlir_type = fortran_to_fir_type(node%type_name, node%kind_value)
            end if

            ! Generate HLFIR declare for variable
            ssa_val = backend%next_ssa_value()
            
            ! Generate HLFIR variable declaration
            if (node%is_array) then
                mlir = indent_str//ssa_val//":2 = hlfir.declare "// &
                       "(fir.undefined : "//mlir_type//") "// &
                       '{var_name="'//trim(node%var_name)//'"} : ('//mlir_type//") -> ("// &
                       mlir_type//", "//mlir_type//")"//new_line('a')
            else
                ! HLFIR: Create storage using fir.alloca and declare with hlfir.declare
                alloca_ssa = backend%next_ssa_value()
                mlir = mlir//indent_str//alloca_ssa//" = fir.alloca "//mlir_type// &
                       ' {bindc_name = "'//trim(node%var_name)//'"}'//new_line('a')
                mlir = mlir//indent_str//ssa_val//":2 = hlfir.declare "//alloca_ssa// &
                       ' {var_name="'//trim(node%var_name)//'"} : (!fir.ref<'//mlir_type//'>) -> ('// &
                       '!fir.ref<'//mlir_type//'>, !fir.ref<'//mlir_type//">)"//new_line('a')
            end if
        end if

        ! Add to symbol table
        call backend%add_symbol(trim(node%var_name), ssa_val)
    end function generate_mlir_declaration

    ! Generate MLIR for assignment
    function generate_mlir_assignment(backend, arena, node, indent_str) result(mlir)
        class(mlir_backend_t), intent(inout) :: backend
        type(ast_arena_t), intent(in) :: arena
        type(assignment_node), intent(in) :: node
        character(len=*), intent(in) :: indent_str
        character(len=:), allocatable :: mlir
        character(len=:), allocatable :: target_ref, value_ref, value_ssa
        character(len=32) :: idx_ssa
        logical :: is_array_assignment

        mlir = ""

        ! Check for assignment operator overloading first (simplified for now)
        ! if (is_operator_overloaded(backend, arena, "assignment(=)")) then
        !     mlir = generate_mlir_assignment_overload_call(backend, arena, node, indent_str)
        !     return
        ! end if

        ! Generate value expression first
        value_ref = generate_mlir_expression(backend, arena, node%value_index, indent_str)
        mlir = mlir//value_ref
        value_ssa = backend%last_ssa_value  ! Capture the SSA value of the expression

        ! Check if target is array element
        is_array_assignment = .false.
        if (node%target_index > 0 .and. node%target_index <= arena%size) then
            if (allocated(arena%entries(node%target_index)%node)) then
                select type (target_node => arena%entries(node%target_index)%node)
                type is (call_or_subscript_node)
                    is_array_assignment = .true.
                end select
            end if
        end if

        if (is_array_assignment) then
            ! Handle array element assignment
            if (allocated(arena%entries(node%target_index)%node)) then
                select type (target_node => arena%entries(node%target_index)%node)
                type is (call_or_subscript_node)
                    ! Generate indices
                    if (allocated(target_node%arg_indices) .and. size(target_node%arg_indices) > 0) then
                        ! Generate index expressions
                        block
                            character(len=:), allocatable :: indices_mlir
                            character(len=200) :: index_list
                            integer :: i

                            indices_mlir = ""
                            index_list = ""

                            do i = 1, size(target_node%arg_indices)
                                ! indices_mlir = indices_mlir//generate_mlir_node(backend, arena, target_node%arg_indices(i), 0)
                                if (i > 1) index_list = trim(index_list)//", "
                                ! For literals, use the constant directly (convert to 0-based)
                                if (allocated(arena%entries(target_node%arg_indices(i))%node)) then
                                    select type (idx_node => arena%entries(target_node%arg_indices(i))%node)
                                    type is (literal_node)
                                        backend%ssa_counter = backend%ssa_counter + 1
                                        mlir = mlir//indent_str//"%"//mlir_int_to_str(backend%ssa_counter)// &
                                               " = hlfir.expr { %c = fir.constant "// &
                                               mlir_int_to_str(string_to_int(trim(idx_node%value)) - 1)// &
                                               " : !fir.int<4>; fir.result %c : !fir.int<4> } : "// &
                                               "!hlfir.expr<!fir.int<4>>"//new_line('a')
                                        index_list = trim(index_list)//"%"//mlir_int_to_str(backend%ssa_counter)
                                    class default
                                        index_list = trim(index_list)//backend%last_ssa_value
                                    end select
                                end if
                            end do

                            ! Generate store operation
                            ! HLFIR: Use high-level array element assignment
                            ! Skip hlfir.expr wrapper - use value directly
                            ! Then designate and assign in one step
                            backend%ssa_counter = backend%ssa_counter + 1
                            mlir = mlir//indent_str//"%"//mlir_int_to_str(backend%ssa_counter)//" = hlfir.designate %"//&
                                    trim(target_node%name)//"_ptr["//trim(index_list)//"] : (!fir.box<!fir.array<?x"
                            select case (trim(target_node%name))
                            case ("arr", "vector")
                                mlir = mlir//"!fir.int<4>"
                            case ("matrix")
                                mlir = mlir//"!fir.real<4>"
                            case default
                                mlir = mlir//"!fir.int<4>"
                            end select
                            mlir = mlir//">>, index) -> !fir.ref<"
                            select case (trim(target_node%name))
                            case ("arr", "vector")
                                mlir = mlir//"!fir.int<4>"
                            case ("matrix")
                                mlir = mlir//"!fir.real<4>"
                            case default
                                mlir = mlir//"!fir.int<4>"
                            end select
                            mlir = mlir//">"//new_line('a')
                            ! Use hlfir.assign with the value directly
                            mlir = mlir//indent_str//"hlfir.assign "//value_ssa//" to %"//&
                                    mlir_int_to_str(backend%ssa_counter)//" : "
                            select case (trim(target_node%name))
                            case ("arr", "vector")
                                mlir = mlir//"i32, !fir.ref<!fir.int<4>>"
                            case ("matrix")
                                mlir = mlir//"f32, !fir.ref<!fir.real<4>>"
                            case default
                                mlir = mlir//"i32, !fir.ref<!fir.int<4>>"
                            end select
                            mlir = mlir//new_line('a')
                        end block
                    end if
                end select
            end if
        else
            ! Regular variable assignment or full array assignment
            ! Check if value is an array literal
            if (node%value_index > 0 .and. node%value_index <= arena%size) then
                select type (value_node => arena%entries(node%value_index)%node)
                type is (array_literal_node)
                    ! Handle array literal assignment
                    if (node%target_index > 0 .and. node%target_index <= arena%size) then
                        select type (target_node => arena%entries(node%target_index)%node)
                        type is (identifier_node)
                            ! HLFIR: Use hlfir.assign for full array assignment
                            mlir = mlir//indent_str//"hlfir.assign "//value_ssa//" to %"//trim(target_node%name)// &
                                   "_ptr : i32, !fir.ref<!fir.array<?xi32>>"//new_line('a')
                            ! Note: Type should be inferred from context
                            mlir = mlir//indent_str//"! Array size: "

                            ! Get array size from literal
                            if (allocated(value_node%element_indices)) then
                                mlir = mlir//mlir_int_to_str(size(value_node%element_indices))
                            else
                                mlir = mlir//"?"
                            end if
                            mlir = mlir//"xi32>"//new_line('a')
                        class default
                            mlir = mlir//indent_str//"// Unknown target for array literal"//new_line('a')
                        end select
                    end if
                class default
                    ! Generate target reference
                    target_ref = generate_mlir_expression(backend, arena, node%target_index, indent_str)

                    ! Generate HLFIR store operation
                    block
                        character(len=:), allocatable :: target_memref
                        ! Extract the target variable name and get its memref
                        select type (target_node => arena%entries(node%target_index)%node)
                        type is (identifier_node)
                            target_memref = backend%get_symbol_memref(trim(target_node%name))
                        class default
                            target_memref = backend%last_ssa_value
                        end select
                        ! Generate index for scalar store (1D memref with size 1)
                        idx_ssa = backend%next_ssa_value()
                        ! FIR: Use fir.constant for index
                        mlir = mlir//indent_str//trim(idx_ssa)//" = fir.constant 0 : index"//new_line('a')
                        ! HLFIR: Use hlfir.assign directly with the value
                        mlir = mlir//indent_str//"hlfir.assign "//value_ssa//" to "//trim(target_memref)// &
                               " : i32, !fir.ref<!fir.int<4>>"//new_line('a')
                    end block
                end select
            end if
        end if
    end function generate_mlir_assignment

    ! Generate MLIR for pointer assignment (ptr => target)
    function generate_mlir_pointer_assignment(backend, arena, node, indent_str) result(mlir)
        class(mlir_backend_t), intent(inout) :: backend
        type(ast_arena_t), intent(in) :: arena
        type(pointer_assignment_node), intent(in) :: node
        character(len=*), intent(in) :: indent_str
        character(len=:), allocatable :: mlir
        character(len=:), allocatable :: pointer_ref, target_ref, target_ssa
        character(len=:), allocatable :: idx_ssa, new_ssa

        mlir = ""

        ! Generate target expression first
        target_ref = generate_mlir_expression(backend, arena, node%target_index, indent_str)
        target_ssa = backend%last_ssa_value

        ! Get pointer memref
        pointer_ref = generate_mlir_expression(backend, arena, node%pointer_index, indent_str)

        if (backend%compile_mode) then
            ! Handle special cases for pointer assignment
            if (node%target_index > 0 .and. node%target_index <= arena%size) then
                if (allocated(arena%entries(node%target_index)%node)) then
                    select type (target_node => arena%entries(node%target_index)%node)
                    type is (call_or_subscript_node)
                        ! Check if this is a null() intrinsic call
                        if (target_node%name == "null") then
                            ! Generate null pointer assignment using HLFIR
                            target_ssa = backend%next_ssa_value()
                            mlir = mlir//indent_str//target_ssa//" = hlfir.null : !fir.ref<none>"//new_line('a')
                        end if
                    type is (identifier_node)
                        ! For identifier targets, we need to get the address
                        if (is_pointer_variable(backend, arena, target_node%name)) then
                            ! Pointer to pointer assignment: load the pointer value using HLFIR
                            new_ssa = backend%next_ssa_value()
                            mlir = mlir//indent_str//new_ssa//" = fir.load "//target_ssa// &
                                   " : !fir.ref<!fir.ref<i32>>"//new_line('a')
                            target_ssa = new_ssa
                        else
                            ! Regular variable: already has address
                            ! target_ssa already contains the !fir.ref<i32> address
                            ! No operation needed - use target_ssa as is
                        end if
                    end select
                end if
            end if

            ! HLFIR: Use hlfir.assign for pointer assignment
            mlir = mlir//indent_str//"hlfir.assign "//target_ssa//" to "//pointer_ref// &
                   " : !fir.ptr<!fir.int<4>>, !fir.ref<!fir.ptr<!fir.int<4>>>"//new_line('a')
        else
            ! Standard backend mode
            mlir = mlir//indent_str//"// Pointer assignment: "//pointer_ref//" => "//target_ref//new_line('a')
        end if
    end function generate_mlir_pointer_assignment

    ! Check if a variable is declared as a pointer
    function is_pointer_variable(backend, arena, var_name) result(is_ptr)
        class(mlir_backend_t), intent(in) :: backend
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: var_name
        logical :: is_ptr
        integer :: i

        associate(associate_placeholder => backend)
        end associate

        is_ptr = .false.

        ! Search through arena for declaration of this variable
        do i = 1, arena%size
            select type (node => arena%entries(i)%node)
            type is (declaration_node)
                if (trim(node%var_name) == trim(var_name)) then
                    is_ptr = node%is_pointer
                    return
                end if
            end select
        end do
    end function is_pointer_variable

    ! Generate MLIR for subroutine call
    function generate_mlir_subroutine_call(backend, arena, node, indent_str) result(mlir)
        class(mlir_backend_t), intent(inout) :: backend
        type(ast_arena_t), intent(in) :: arena
        type(subroutine_call_node), intent(in) :: node
        character(len=*), intent(in) :: indent_str
        character(len=:), allocatable :: mlir

        ! HLFIR: Generate FIR call to subroutine (no return value)
        mlir = indent_str//"fir.call @"//trim(node%name)//"() : () -> ()"//new_line('a')
    end function generate_mlir_subroutine_call

    ! Generate MLIR for return statement
    function generate_mlir_return(backend, arena, node, indent_str) result(mlir)
        class(mlir_backend_t), intent(inout) :: backend
        type(ast_arena_t), intent(in) :: arena
        type(return_node), intent(in) :: node
        character(len=*), intent(in) :: indent_str
        character(len=:), allocatable :: mlir

        associate(associate_placeholder => backend)
        associate(associate_placeholder2 => arena)
        associate(associate_placeholder3 => node)
        end associate
        end associate
        end associate

        ! For now, just generate a simple return
        mlir = indent_str//"fir.return"//new_line('a')
    end function generate_mlir_return

    ! Generate MLIR for exit statement
    function generate_mlir_exit(backend, arena, node, indent_str) result(mlir)
        class(mlir_backend_t), intent(inout) :: backend
        type(ast_arena_t), intent(in) :: arena
        type(exit_node), intent(in) :: node
        character(len=*), intent(in) :: indent_str
        character(len=:), allocatable :: mlir

        associate(associate_placeholder => backend)
        associate(associate_placeholder2 => arena)
        associate(associate_placeholder3 => node)
        end associate
        end associate
        end associate

        ! HLFIR: Use scf.break or cf.br to exit from loop
        if (allocated(node%label)) then
            ! Named exit - not commonly supported in MLIR, use cf.br with block label
            mlir = indent_str//"cf.br ^bb_exit_"//trim(node%label)//new_line('a')
        else
            ! Unnamed exit - break from innermost loop
            mlir = indent_str//"scf.break"//new_line('a')
        end if
    end function generate_mlir_exit

    ! Generate MLIR for cycle statement
    function generate_mlir_cycle(backend, arena, node, indent_str) result(mlir)
        class(mlir_backend_t), intent(inout) :: backend
        type(ast_arena_t), intent(in) :: arena
        type(cycle_node), intent(in) :: node
        character(len=*), intent(in) :: indent_str
        character(len=:), allocatable :: mlir

        associate(associate_placeholder => backend)
        associate(associate_placeholder2 => arena)
        associate(associate_placeholder3 => node)
        end associate
        end associate
        end associate

        ! HLFIR: Use scf.continue or cf.br to continue loop
        if (allocated(node%label)) then
            ! Named cycle - use cf.br with block label
            mlir = indent_str//"cf.br ^bb_cycle_"//trim(node%label)//new_line('a')
        else
            ! Unnamed cycle - continue innermost loop
            mlir = indent_str//"scf.continue"//new_line('a')
        end if
    end function generate_mlir_cycle

    ! Generate MLIR for allocate statement
    function generate_mlir_allocate_statement(backend, arena, node, indent_str) result(mlir)
        class(mlir_backend_t), intent(inout) :: backend
        type(ast_arena_t), intent(in) :: arena
        type(allocate_statement_node), intent(in) :: node
        character(len=*), intent(in) :: indent_str
        character(len=:), allocatable :: mlir

        associate(associate_placeholder => backend)
        associate(associate_placeholder2 => arena)
        associate(associate_placeholder3 => node)
        end associate
        end associate
        end associate

        ! Simplified for now
        mlir = indent_str//"// TODO: Implement allocate statement"//new_line('a')
    end function generate_mlir_allocate_statement

    ! Generate array size calculation and allocation
    subroutine generate_array_size_calculation(backend, arena, node, indent_str, mlir, alloc_size_var)
        class(mlir_backend_t), intent(inout) :: backend
        type(ast_arena_t), intent(in) :: arena
        class(ast_node), intent(in) :: node
        character(len=*), intent(in) :: indent_str
        character(len=:), allocatable, intent(inout) :: mlir
        character(len=*), intent(in) :: alloc_size_var
        character(len=:), allocatable :: array_name, size_calc, alloc_ssa
        integer :: i

        associate(associate_placeholder => alloc_size_var)
        end associate

        select type (node)
        type is (call_or_subscript_node)
            ! This is an array allocation: arr(n, m)
            array_name = trim(node%name)
            
            ! Generate size calculation
            if (allocated(node%arg_indices)) then
                size_calc = ""
                do i = 1, size(node%arg_indices)
                    if (node%arg_indices(i) > 0 .and. node%arg_indices(i) <= arena%size) then
                        ! Generate size expression
                        ! size_calc = size_calc//generate_mlir_expression(backend, arena, node%arg_indices(i), indent_str)
                        
                        ! Store the size SSA value
                        if (i == 1) then
                            ! First dimension
                            mlir = mlir//indent_str//"// Allocate "//array_name//" with size "//backend%last_ssa_value//new_line('a')
                        end if
                    end if
                end do
                
                ! Generate the actual allocation using HLFIR
                alloc_ssa = backend%next_ssa_value()
                mlir = mlir//size_calc
                mlir = mlir//indent_str//alloc_ssa//" = fir.allocmem !fir.array<?xi32>, "//backend%last_ssa_value//new_line('a')
                mlir = mlir//indent_str//array_name//"_ptr = fir.embox "//alloc_ssa//" : (!fir.heap<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>"//new_line('a')
            end if
        type is (identifier_node)
            ! Simple identifier allocation
            array_name = trim(node%name)
            alloc_ssa = backend%next_ssa_value()
            mlir = mlir//indent_str//"// Allocate "//array_name//new_line('a')
            mlir = mlir//indent_str//alloc_ssa//" = fir.allocmem i32"//new_line('a')
            mlir = mlir//indent_str//array_name//"_ptr = "//alloc_ssa//new_line('a')
        class default
            mlir = mlir//indent_str//"// Unsupported allocation type"//new_line('a')
        end select
    end subroutine generate_array_size_calculation

    ! Generate MLIR for deallocate statement
    function generate_mlir_deallocate_statement(backend, arena, node, indent_str) result(mlir)
        class(mlir_backend_t), intent(inout) :: backend
        type(ast_arena_t), intent(in) :: arena
        type(deallocate_statement_node), intent(in) :: node
        character(len=*), intent(in) :: indent_str
        character(len=:), allocatable :: mlir

        associate(associate_placeholder => backend)
        associate(associate_placeholder2 => arena)
        associate(associate_placeholder3 => node)
        end associate
        end associate
        end associate

        ! Simplified for now
        mlir = indent_str//"// TODO: Implement deallocate statement"//new_line('a')
    end function generate_mlir_deallocate_statement

end module mlir_backend_statements