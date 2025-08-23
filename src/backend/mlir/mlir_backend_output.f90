module mlir_backend_output
    use backend_interface
    use ast_core
    use mlir_backend_types
    use mlir_backend_helpers, only: format_string_to_array
    use mlir_utils, mlir_int_to_str => int_to_str
    use mlir_expressions, only: generate_mlir_call_or_subscript
    implicit none
    private

    public :: generate_string_runtime_output, generate_integer_runtime_output
    public :: generate_real_runtime_output, generate_logical_runtime_output
    public :: generate_variable_runtime_output, generate_expression_runtime_output
    public :: generate_call_runtime_output

contains

    ! Helper subroutines for I/O runtime generation
    subroutine generate_string_runtime_output(backend, value, cookie_var, mlir, indent_str)
        class(mlir_backend_t), intent(inout) :: backend
        character(len=*), intent(in) :: value, cookie_var, indent_str
        character(len=:), allocatable, intent(inout) :: mlir
        character(len=:), allocatable :: str_global, str_ptr, str_len, result_var
        character(len=:), allocatable :: clean_value

        associate(associate_placeholder => cookie_var)
        end associate

        ! Remove quotes if present and add newline for proper formatting
        clean_value = trim(value)
        if (len(clean_value) >= 2) then
            if (clean_value(1:1) == '"' .and. clean_value(len(clean_value):len(clean_value)) == '"') then
                clean_value = clean_value(2:len(clean_value) - 1)
            end if
        end if

        ! Create global string constant with null terminator
        backend%ssa_counter = backend%ssa_counter + 1
        str_global = "@str_"//mlir_int_to_str(backend%ssa_counter)
        str_ptr = backend%next_ssa_value()
        str_len = backend%next_ssa_value()
        result_var = backend%next_ssa_value()

        ! Add global declaration
        if (.not. allocated(backend%global_declarations)) then
            backend%global_declarations = ""
        end if

        block
            character(len=:), allocatable :: global_decl
            integer :: str_len_val
            str_len_val = len(clean_value) + 1  ! +1 for null terminator
            global_decl = "  fir.global private constant "//str_global// &
                         "(dense<["//format_string_to_array(clean_value)//", 0]> : "// &
                          "tensor<"//mlir_int_to_str(str_len_val)//"xi8>) : "// &
                          "!fir.array<"//mlir_int_to_str(str_len_val)//" x i8>"
            backend%global_declarations = backend%global_declarations//global_decl//new_line('a')
        end block

        ! Generate string pointer and length
        mlir = mlir//indent_str//str_ptr//" = fir.address_of "//str_global// &
               " : !fir.ref<!fir.array<? x i8>>"//new_line('a')
        mlir = mlir//indent_str//str_len//" = fir.constant "// &
               mlir_int_to_str(len(clean_value))//" : i64"//new_line('a')

        ! Output string using FIR runtime call
        mlir = mlir//indent_str//"fir.call @_FortranAioOutputAscii("//str_ptr//", "//str_len// &
               &") : (!fir.ref<!fir.array<? x i8>>, i64) -> ()"//new_line('a')
    end subroutine generate_string_runtime_output

    subroutine generate_integer_runtime_output(backend, value, cookie_var, mlir, indent_str)
        class(mlir_backend_t), intent(inout) :: backend
        character(len=*), intent(in) :: value, cookie_var, indent_str
        character(len=:), allocatable, intent(inout) :: mlir
        character(len=:), allocatable :: const_var, result_var

        associate(associate_placeholder => cookie_var)
        end associate

        const_var = backend%next_ssa_value()
        result_var = backend%next_ssa_value()

        ! Generate integer constant
        mlir = mlir//indent_str//const_var//" = fir.constant "//trim(value)//" : i32"//new_line('a')

        ! Output integer using FIR runtime call
        mlir = mlir//indent_str//"fir.call @_FortranAioOutputInteger32("//const_var//") : (i32) -> ()"//new_line('a')
    end subroutine generate_integer_runtime_output

    subroutine generate_real_runtime_output(backend, value, cookie_var, mlir, indent_str)
        class(mlir_backend_t), intent(inout) :: backend
        character(len=*), intent(in) :: value, cookie_var, indent_str
        character(len=:), allocatable, intent(inout) :: mlir
        character(len=:), allocatable :: const_var, result_var

        associate(associate_placeholder => cookie_var)
        end associate

        const_var = backend%next_ssa_value()
        result_var = backend%next_ssa_value()

        ! Generate real constant
        mlir = mlir//indent_str//const_var//" = fir.constant "//trim(value)//" : f32"//new_line('a')

        ! Output real using FIR runtime call
        mlir = mlir//indent_str//"fir.call @_FortranAioOutputReal32("//const_var//") : (f32) -> ()"//new_line('a')
    end subroutine generate_real_runtime_output

    subroutine generate_logical_runtime_output(backend, value, cookie_var, mlir, indent_str)
        class(mlir_backend_t), intent(inout) :: backend
        character(len=*), intent(in) :: value, cookie_var, indent_str
        character(len=:), allocatable, intent(inout) :: mlir
        character(len=:), allocatable :: const_var, result_var

        associate(associate_placeholder => cookie_var)
        end associate

        const_var = backend%next_ssa_value()
        result_var = backend%next_ssa_value()

        ! Generate logical constant (convert to i1)
        if (trim(value) == ".true." .or. trim(value) == "T") then
            mlir = mlir//indent_str//const_var//" = fir.constant true : i1"//new_line('a')
        else
            mlir = mlir//indent_str//const_var//" = fir.constant false : i1"//new_line('a')
        end if

        ! Output logical using FIR runtime call
        mlir = mlir//indent_str//"fir.call @_FortranAioOutputLogical("//const_var//") : (i1) -> ()"//new_line('a')
    end subroutine generate_logical_runtime_output

    subroutine generate_variable_runtime_output(backend, arena, node, cookie_var, mlir, indent_str)
        class(mlir_backend_t), intent(inout) :: backend
        type(ast_arena_t), intent(in) :: arena
        type(identifier_node), intent(in) :: node
        character(len=*), intent(in) :: cookie_var, indent_str
        character(len=:), allocatable, intent(inout) :: mlir
        character(len=:), allocatable :: load_var, result_var
        character(len=:), allocatable :: index_var, zero_var

        associate(associate_placeholder => arena)
        associate(associate_placeholder2 => cookie_var)
        end associate
        end associate

        ! Generate code to load the variable (simplified version)
        load_var = backend%next_ssa_value()
        index_var = backend%next_ssa_value()
        zero_var = backend%next_ssa_value()
        result_var = backend%next_ssa_value()

        ! Create index for variable access
        mlir = mlir//indent_str//zero_var//" = fir.constant 0 : index"//new_line('a')
        ! Load the value from variable
        mlir = mlir//indent_str//load_var//" = fir.load %"//trim(node%name)// &
               " : !fir.ref<i32>"//new_line('a')

        ! Output using FIR runtime call
        mlir = mlir//indent_str//"fir.call @_FortranAioOutputInteger32("//load_var//") : (i32) -> ()"//new_line('a')
    end subroutine generate_variable_runtime_output

    subroutine generate_expression_runtime_output(backend, arena, node, cookie_var, mlir, indent_str)
        class(mlir_backend_t), intent(inout) :: backend
        type(ast_arena_t), intent(in) :: arena
        class(*), intent(in) :: node
        character(len=*), intent(in) :: cookie_var, indent_str
        character(len=:), allocatable, intent(inout) :: mlir
        character(len=:), allocatable :: expr_var, result_var

        associate(associate_placeholder => arena)
        associate(associate_placeholder2 => node)
        associate(associate_placeholder3 => cookie_var)
        end associate
        end associate
        end associate

        ! For now, generate a placeholder (TODO: implement proper expression evaluation)
        expr_var = backend%next_ssa_value()
        result_var = backend%next_ssa_value()

        ! Generate a simple constant for expressions (temporary implementation)
        mlir = mlir//indent_str//expr_var//" = fir.constant 42 : i32"//new_line('a')

        ! Output using FIR runtime call
        mlir = mlir//indent_str//"fir.call @_FortranAioOutputInteger32("//expr_var//") : (i32) -> ()"//new_line('a')
    end subroutine generate_expression_runtime_output

    ! Generate runtime output for function calls
    subroutine generate_call_runtime_output(backend, arena, node, cookie_var, mlir, indent_str)
        class(mlir_backend_t), intent(inout) :: backend
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        character(len=*), intent(in) :: cookie_var, indent_str
        character(len=:), allocatable, intent(inout) :: mlir
        character(len=:), allocatable :: call_mlir, result_var

        associate(associate_placeholder => cookie_var)
        end associate

        ! Generate the function call first
        call_mlir = generate_mlir_call_or_subscript(backend, arena, node, indent_str)
        mlir = mlir//call_mlir

        ! Get the SSA value returned by the call
        result_var = backend%last_ssa_value

        ! Output based on the function type
        select case (trim(node%name))
        case ("associated")
            ! Associated returns a logical value - output the computed result directly
            block
                character(len=:), allocatable :: output_result
                output_result = backend%next_ssa_value()
                mlir = mlir//indent_str//"fir.call @_FortranAioOutputLogical("//result_var//") : (i1) -> ()"//new_line('a')
            end block
        case default
            ! Default to integer output for unknown functions
            block
                character(len=:), allocatable :: output_result
                output_result = backend%next_ssa_value()
                mlir = mlir//indent_str//"fir.call @_FortranAioOutputInteger32("//result_var//") : (i32) -> ()"//new_line('a')
            end block
        end select
    end subroutine generate_call_runtime_output

end module mlir_backend_output