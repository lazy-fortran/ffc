module mlir_backend_functions
    use backend_interface
    use ast_core
    use mlir_backend_types
    use mlir_types
    use mlir_expressions
    implicit none
    private

    public :: generate_mlir_program_functions, generate_mlir_function
    public :: generate_mlir_subroutine, generate_function_parameter_list
    public :: generate_function_call_with_args, is_function_name

contains

    ! Placeholder for generate_mlir_node - this should be properly implemented
    ! For now, use a simplified approach for function/subroutine body generation
    function generate_mlir_node(backend, arena, node_index, indent_level) result(mlir)
        class(mlir_backend_t), intent(inout) :: backend
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index, indent_level
        character(len=:), allocatable :: mlir
        character(len=:), allocatable :: indent_str
        integer :: i
        
        ! Create indent string
        indent_str = ""
        do i = 1, indent_level * 2
            indent_str = indent_str//" "
        end do
        
        mlir = indent_str//"! Function body statement"//new_line('a')
    end function generate_mlir_node

    ! Generate all functions within a program node at module level
    function generate_mlir_program_functions(backend, arena, node, indent_level) result(mlir)
        class(mlir_backend_t), intent(inout) :: backend
        type(ast_arena_t), intent(in) :: arena
        type(program_node), intent(in) :: node
        integer, intent(in) :: indent_level
        character(len=:), allocatable :: mlir
        character(len=:), allocatable :: body_content, indent_str
        integer :: i, j

        mlir = ""

        ! Create indentation
        indent_str = ""
        do j = 1, indent_level
            indent_str = indent_str//"  "
        end do

        ! Generate body for the program function
        if (allocated(node%body_indices)) then
            do i = 1, size(node%body_indices)
                select type (nested_node => arena%entries(node%body_indices(i))%node)
                type is (function_def_node)
                    continue  ! Skip functions for this iteration - they'll be handled at module level
                type is (subroutine_def_node)
                    continue  ! Skip subroutines for this iteration - they'll be handled at module level
                class default
                    body_content = generate_mlir_node(backend, arena, node%body_indices(i), indent_level)
                    mlir = mlir//body_content
                end select
            end do
        end if

        ! HLFIR: Always use fir.func for main
        mlir = mlir//new_line('a')//indent_str(3:)//"fir.func @main() {"//new_line('a')

        ! Generate body content
        if (allocated(node%body_indices)) then
            do i = 1, size(node%body_indices)
                select type (nested_node => arena%entries(node%body_indices(i))%node)
                type is (function_def_node)
                    continue  ! Functions handled separately at module level
                type is (subroutine_def_node)
                    continue  ! Subroutines handled separately at module level
                class default
                    body_content = generate_mlir_node(backend, arena, node%body_indices(i), indent_level + 1)
                    mlir = mlir//body_content
                end select
            end do
        end if

        mlir = mlir//indent_str//"  fir.return"//new_line('a')
        mlir = mlir//indent_str//"}"//new_line('a')

        ! Now generate any nested functions and subroutines at module level
        if (allocated(node%body_indices)) then
            do i = 1, size(node%body_indices)
                select type (nested_node => arena%entries(node%body_indices(i))%node)
                type is (function_def_node)
                    body_content = generate_mlir_function(backend, arena, nested_node, indent_str(3:))
                    mlir = mlir//new_line('a')//body_content
                type is (subroutine_def_node)
                    body_content = generate_mlir_subroutine(backend, arena, nested_node, indent_str(3:))
                    mlir = mlir//new_line('a')//body_content
                end select
            end do
        end if
    end function generate_mlir_program_functions

    ! Generate MLIR for function definition
    function generate_mlir_function(backend, arena, node, indent_str) result(mlir)
        class(mlir_backend_t), intent(inout) :: backend
        type(ast_arena_t), intent(in) :: arena
        type(function_def_node), intent(in) :: node
        character(len=*), intent(in) :: indent_str
        character(len=:), allocatable :: mlir
        character(len=:), allocatable :: return_type, body_content, param_list, func_name
        integer :: i

        ! Determine return type using FIR types
        if (allocated(node%return_type)) then
            return_type = fortran_to_mlir_type(node%return_type, 4, backend%compile_mode)
        else
            return_type = "i32"  ! Default
        end if

        ! Generate parameter list
        param_list = generate_function_parameter_list(backend, arena, node%param_indices)

        ! Add module namespace if we're inside a module
        if (allocated(backend%current_module_name)) then
            func_name = trim(backend%current_module_name)//"."//trim(node%name)
        else
            func_name = trim(node%name)
        end if

        ! HLFIR: Use fir.func with optional enzyme attributes
        if (backend%enable_ad) then
            mlir = indent_str//"fir.func @"//func_name//"("//trim(param_list)//") -> "//return_type// &
                   " attributes {enzyme.differentiable} {"//new_line('a')
        else
            mlir = indent_str // "fir.func @" // func_name // "(" // trim(param_list) // ") -> " // &
                   return_type // " {" // new_line('a')
        end if

        ! Generate body
        if (allocated(node%body_indices)) then
            do i = 1, size(node%body_indices)
                body_content = generate_mlir_node(backend, arena, node%body_indices(i), 3)
                mlir = mlir//body_content
            end do
        end if

        ! Return the last computed value or default
        if (allocated(backend%last_ssa_value)) then
            mlir = mlir//indent_str//"  fir.return "//backend%last_ssa_value//" : "//return_type//new_line('a')
        else
            ! Default return if no value was computed
            mlir = mlir//indent_str//"  %0 = fir.constant 0 : "//return_type//new_line('a')
            mlir = mlir//indent_str//"  fir.return %0 : "//return_type//new_line('a')
        end if
        mlir = mlir//indent_str//"}"//new_line('a')
    end function generate_mlir_function

    ! Generate MLIR for subroutine definition
    function generate_mlir_subroutine(backend, arena, node, indent_str) result(mlir)
        class(mlir_backend_t), intent(inout) :: backend
        type(ast_arena_t), intent(in) :: arena
        type(subroutine_def_node), intent(in) :: node
        character(len=*), intent(in) :: indent_str
        character(len=:), allocatable :: mlir
        character(len=:), allocatable :: body_content, param_list
        integer :: i

        ! Generate parameter list
        param_list = generate_function_parameter_list(backend, arena, node%param_indices)

        ! Generate subroutine signature - subroutines have no return value
        ! HLFIR: Always use fir.func
        mlir = indent_str//"fir.func @"//trim(node%name)//"("//trim(param_list)//") {"//new_line('a')

        ! Generate body
        if (allocated(node%body_indices)) then
            do i = 1, size(node%body_indices)
                body_content = generate_mlir_node(backend, arena, node%body_indices(i), 2)
                mlir = mlir//body_content
            end do
        end if

        ! End subroutine
        mlir = mlir//indent_str//"  fir.return"//new_line('a')
        mlir = mlir//indent_str//"}"//new_line('a')
    end function generate_mlir_subroutine

    ! Generate parameter list for function/subroutine signatures
    function generate_function_parameter_list(backend, arena, param_indices) result(param_list)
        class(mlir_backend_t), intent(inout) :: backend
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in), optional :: param_indices(:)
        character(len=:), allocatable :: param_list
        character(len=:), allocatable :: param_type, param_name
        integer :: i

        param_list = ""

        if (.not. present(param_indices)) return
        if (size(param_indices) == 0) return

        do i = 1, size(param_indices)
            if (param_indices(i) > 0 .and. param_indices(i) <= arena%size) then
                select type (param_node => arena%entries(param_indices(i))%node)
                type is (declaration_node)
                    ! Convert Fortran type to HLFIR type
                    param_type = fortran_to_mlir_type(param_node%type_name, 4, .false.)

                    ! Generate parameter entry: %param_name : type
                    param_name = trim(param_node%var_name)

                    if (i > 1) then
                        param_list = param_list//", "
                    end if
                    param_list = param_list//"%"//param_name//" : "//param_type

                type is (parameter_declaration_node)
                    ! Handle parameter declaration nodes
                    param_type = fortran_to_mlir_type(param_node%type_name, 4, .false.)

                    param_name = trim(param_node%name)

                    if (i > 1) then
                        param_list = param_list//", "
                    end if
                    param_list = param_list//"%"//param_name//" : "//param_type

                class default
                    ! Skip unsupported parameter types
                    continue
                end select
            end if
        end do
    end function generate_function_parameter_list

    ! Generate function call with arguments
    function generate_function_call_with_args(backend, arena, node, result_ssa, indent_str) result(mlir)
        class(mlir_backend_t), intent(inout) :: backend
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        character(len=*), intent(in) :: result_ssa
        character(len=*), intent(in) :: indent_str
        character(len=:), allocatable :: mlir
        character(len=:), allocatable :: arg_list, arg_types, arg_mlir
        integer :: i

        mlir = ""
        arg_list = ""
        arg_types = ""

        ! Generate arguments
        do i = 1, size(node%arg_indices)
            if (node%arg_indices(i) > 0 .and. node%arg_indices(i) <= arena%size) then
                ! Generate the argument expression first
                arg_mlir = generate_mlir_expression(backend, arena, node%arg_indices(i), indent_str)
                mlir = mlir//arg_mlir

                ! Add to argument list
                if (i > 1) then
                    arg_list = arg_list//", "
                    arg_types = arg_types//", "
                end if

                arg_list = arg_list//trim(backend%last_ssa_value)

                ! Determine argument type (simple heuristic for now)
                select type (arg_node => arena%entries(node%arg_indices(i))%node)
                type is (literal_node)
                    select case (arg_node%literal_kind)
                    case (LITERAL_INTEGER)
                        arg_types = arg_types//"i32"
                    case (LITERAL_REAL)
                        arg_types = arg_types//"f32"
                    case (LITERAL_STRING)
                        arg_types = arg_types//"!fir.char<1>"
                    case default
                        arg_types = arg_types//"i32"
                    end select
                class default
                    ! Default to i32 for other types
                    arg_types = arg_types//"i32"
                end select
            end if
        end do

        ! Generate the function call using FIR
        mlir = mlir//indent_str//result_ssa//" = fir.call @"//trim(node%name)//"("// &
               trim(arg_list)//") : ("//trim(arg_types)//") -> i32"//new_line('a')
    end function generate_function_call_with_args

    ! Check if a name corresponds to a function definition in the current scope
    function is_function_name(backend, arena, name) result(is_function)
        class(mlir_backend_t), intent(inout) :: backend
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: name
        logical :: is_function
        integer :: i

        associate(associate_placeholder => backend)
        end associate

        is_function = .false.

        ! Search through the arena for function definitions with this name
        do i = 1, arena%size
            select type (node => arena%entries(i)%node)
            type is (function_def_node)
                if (trim(node%name) == trim(name)) then
                    is_function = .true.
                    return
                end if
            end select
        end do

        ! Check if it's a built-in intrinsic function
        select case (trim(name))
        case ("sin", "cos", "tan", "exp", "log", "sqrt", "abs", "max", "min", &
              "real", "int", "nint", "aint", "anint", "mod", "sign")
            is_function = .true.
        case default
            ! Check for more intrinsics or user-defined functions in modules
            continue
        end select
    end function is_function_name

end module mlir_backend_functions