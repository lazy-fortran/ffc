! MLIR I/O Operations Module - Pure HLFIR Implementation
! This module contains all I/O statement handlers for MLIR code generation using HLFIR
module mlir_io_operations
    use fortfront
    use mlir_backend_types
    implicit none

    private
    public :: generate_mlir_print_statement, generate_mlir_write_statement, generate_mlir_read_statement
    public :: generate_io_error_check, generate_simple_print_for_compile, generate_print_with_printf
    public :: generate_io_runtime_decls, generate_hlfir_io_runtime_decls
    public :: generate_print_runtime_calls, generate_write_runtime_calls, generate_read_runtime_calls

contains

    ! Helper function for integer to string conversion
    function io_int_to_str(i) result(str)
        integer, intent(in) :: i
        character(len=20) :: str
        write(str, '(I0)') i
        str = trim(adjustl(str))
    end function io_int_to_str

    ! Generate MLIR for print statement - HLFIR-compliant with FIR runtime calls
    function generate_mlir_print_statement(backend, arena, node, indent_str) result(mlir)
        class(mlir_backend_t), intent(inout) :: backend
        type(ast_arena_t), intent(in) :: arena
        type(print_statement_node), intent(in) :: node
        character(len=*), intent(in) :: indent_str
        character(len=:), allocatable :: mlir
        character(len=:), allocatable :: cookie_ssa, format_ssa, arg_ssa
        integer :: i

        mlir = ""
        
        ! DEBUG: Verify this function is being called
        print *, "DEBUG: generate_mlir_print_statement called - using FIR runtime calls"

        ! HLFIR-compliant I/O: Use FIR runtime calls @_FortranAio* (not hlfir.print)
        ! Step 1: Begin external list output for print statement
        cookie_ssa = backend%next_ssa_value()
        mlir = mlir//indent_str//cookie_ssa//" = fir.call @_FortranAioBeginExternalListOutput("// &
               "fir.constant 6 : i32, fir.constant 0 : !fir.ref<!fir.char<1,0>>, "// &
               "fir.constant 0 : i32) : (i32, !fir.ref<!fir.char<1,0>>, i32) -> !fir.llvm_ptr"//new_line('a')

        ! Step 2: Output each argument using appropriate FIR runtime call
        if (allocated(node%expression_indices)) then
            do i = 1, size(node%expression_indices)
                ! Generate expression for each argument
                mlir = mlir//generate_mlir_expression(backend, arena, node%expression_indices(i), indent_str)
                arg_ssa = backend%last_ssa_value
                
                ! Output the argument using FIR runtime call (assuming string for now)
                mlir = mlir//indent_str//"fir.call @_FortranAioOutputCharacter("//cookie_ssa//", "//arg_ssa// &
                       ", fir.constant 1 : i64) : (!fir.llvm_ptr, !fir.ref<!fir.char<1,?>>, i64) -> i1"//new_line('a')
            end do
        end if

        ! Step 3: End external list output
        mlir = mlir//indent_str//"fir.call @_FortranAioEndIoStatement("//cookie_ssa// &
               ") : (!fir.llvm_ptr) -> i32"//new_line('a')
    end function generate_mlir_print_statement

    ! Generate MLIR for write statement - HLFIR-compliant with FIR runtime calls
    function generate_mlir_write_statement(backend, arena, node, indent_str) result(mlir)
        class(mlir_backend_t), intent(inout) :: backend
        type(ast_arena_t), intent(in) :: arena
        type(write_statement_node), intent(in) :: node
        character(len=*), intent(in) :: indent_str
        character(len=:), allocatable :: mlir
        character(len=:), allocatable :: unit_ssa, cookie_ssa, arg_ssa, format_ssa
        integer :: i

        mlir = ""

        ! HLFIR-compliant I/O: Use FIR runtime calls @_FortranAio* (not hlfir.write)
        ! Step 1: Create unit SSA value
        unit_ssa = backend%next_ssa_value()
        if (allocated(node%unit_spec) .and. len_trim(node%unit_spec) > 0) then
            mlir = mlir//indent_str//unit_ssa//" = fir.constant "//trim(node%unit_spec)//" : i32"//new_line('a')
        else
            ! Default to unit 6 (standard output)
            mlir = mlir//indent_str//unit_ssa//" = fir.constant 6 : i32"//new_line('a')
        end if

        ! Step 2: Begin external output - check if formatted or list-directed
        cookie_ssa = backend%next_ssa_value()
        if (node%is_formatted .and. allocated(node%format_spec)) then
            ! Formatted output
            mlir = mlir//generate_format_setup(backend, node%format_spec, indent_str)
            mlir = mlir//indent_str//cookie_ssa//" = fir.call @_FortranAioBeginExternalFormattedOutput("// &
                   unit_ssa//", "//backend%last_ssa_value//", "// &
                   "fir.constant "//io_int_to_str(len_trim(node%format_spec))//" : i64"// &
                   ") : (i32, !fir.ref<!fir.char<1,?>>, i64) -> !fir.llvm_ptr"//new_line('a')
        else if (node%format_expr_index > 0) then
            ! Runtime format expression
            mlir = mlir//generate_mlir_expression(backend, arena, node%format_expr_index, indent_str)
            format_ssa = backend%last_ssa_value
            mlir = mlir//indent_str//cookie_ssa//" = fir.call @_FortranAioBeginExternalFormattedOutput("// &
                   unit_ssa//", "//format_ssa//", "// &
                   "fir.constant -1 : i64) : (i32, !fir.ref<!fir.char<1,?>>, i64) -> !fir.llvm_ptr"//new_line('a')
        else
            ! List-directed output
            mlir = mlir//indent_str//cookie_ssa//" = fir.call @_FortranAioBeginExternalListOutput("// &
                   unit_ssa//", fir.constant 0 : !fir.ref<!fir.char<1,0>>, "// &
                   "fir.constant 0 : i32) : (i32, !fir.ref<!fir.char<1,0>>, i32) -> !fir.llvm_ptr"//new_line('a')
        end if

        ! Step 3: Output each argument using appropriate FIR runtime call
        if (allocated(node%arg_indices)) then
            do i = 1, size(node%arg_indices)
                ! Generate expression for each argument
                mlir = mlir//generate_mlir_expression(backend, arena, node%arg_indices(i), indent_str)
                arg_ssa = backend%last_ssa_value
                
                ! Determine type from expression and use appropriate output function
                mlir = mlir//generate_typed_output_call(backend, arena, node%arg_indices(i), cookie_ssa, arg_ssa, indent_str)
            end do
        end if

        ! Step 4: End external list output and handle error specifiers
        mlir = mlir//generate_io_error_handling(backend, arena, node, cookie_ssa, indent_str)
    end function generate_mlir_write_statement

    ! Generate MLIR for read statement - HLFIR-compliant with FIR runtime calls
    function generate_mlir_read_statement(backend, arena, node, indent_str) result(mlir)
        class(mlir_backend_t), intent(inout) :: backend
        type(ast_arena_t), intent(in) :: arena
        type(read_statement_node), intent(in) :: node
        character(len=*), intent(in) :: indent_str
        character(len=:), allocatable :: mlir
        character(len=:), allocatable :: unit_ssa, cookie_ssa, var_ssa
        integer :: i

        mlir = ""

        ! HLFIR-compliant I/O: Use FIR runtime calls @_FortranAio* (not hlfir.read)
        ! Step 1: Create unit SSA value
        unit_ssa = backend%next_ssa_value()
        if (allocated(node%unit_spec) .and. len_trim(node%unit_spec) > 0) then
            mlir = mlir//indent_str//unit_ssa//" = fir.constant "//trim(node%unit_spec)//" : i32"//new_line('a')
        else
            ! Default to unit 5 (standard input)
            mlir = mlir//indent_str//unit_ssa//" = fir.constant 5 : i32"//new_line('a')
        end if

        ! Step 2: Begin external list input for read statement
        cookie_ssa = backend%next_ssa_value()
        mlir = mlir//indent_str//cookie_ssa//" = fir.call @_FortranAioBeginExternalListInput("// &
               unit_ssa//", fir.constant 0 : !fir.ref<!fir.char<1,0>>, "// &
               "fir.constant 0 : i32) : (i32, !fir.ref<!fir.char<1,0>>, i32) -> !fir.llvm_ptr"//new_line('a')

        ! Step 3: Input each variable using appropriate FIR runtime call
        if (allocated(node%var_indices)) then
            do i = 1, size(node%var_indices)
                ! Generate variable reference
                mlir = mlir//generate_mlir_expression(backend, arena, node%var_indices(i), indent_str)
                var_ssa = backend%last_ssa_value
                
                ! Input the variable using FIR runtime call (assuming string for now)
                mlir = mlir//indent_str//"fir.call @_FortranAioInputCharacter("//cookie_ssa//", "//var_ssa// &
                       ", fir.constant 1 : i64) : (!fir.llvm_ptr, !fir.ref<!fir.char<1,?>>, i64) -> i1"//new_line('a')
            end do
        end if

        ! Step 4: End external list input and handle error specifiers
        mlir = mlir//generate_io_error_handling(backend, arena, node, cookie_ssa, indent_str)
    end function generate_mlir_read_statement

    ! Generate I/O error checking - Pure HLFIR
    function generate_io_error_check(backend, status_var, error_label, indent_str) result(mlir)
        class(mlir_backend_t), intent(inout) :: backend
        character(len=*), intent(in) :: status_var, error_label, indent_str
        character(len=:), allocatable :: mlir
        character(len=:), allocatable :: status_expr, cond_ssa, branch_ssa

        ! Generate status check expression
        status_expr = backend%next_ssa_value()
        mlir = indent_str//status_expr//" = hlfir.expr { %zero = fir.constant 0 : i32; "// &
               "fir.result %zero : i32 } : !hlfir.expr<i32>"//new_line('a')

        ! Generate condition check using HLFIR
        cond_ssa = backend%next_ssa_value()
        mlir = mlir//indent_str//cond_ssa//" = hlfir.ne "//status_var//", "//status_expr// &
               " : (!hlfir.expr<i32>, !hlfir.expr<i32>) -> !hlfir.expr<i1>"//new_line('a')

        ! Generate conditional branch using HLFIR
        branch_ssa = backend%next_ssa_value()
        mlir = mlir//indent_str//branch_ssa//" = hlfir.if "//cond_ssa//" {"//new_line('a')
        mlir = mlir//indent_str//"  hlfir.goto ^"//error_label//new_line('a')
        mlir = mlir//indent_str//"}"//new_line('a')
    end function generate_io_error_check

    ! Simple print for compile mode using HLFIR
    function generate_simple_print_for_compile(backend, arena, node, indent_str) result(mlir)
        class(mlir_backend_t), intent(inout) :: backend
        type(ast_arena_t), intent(in) :: arena
        type(print_statement_node), intent(in) :: node
        character(len=*), intent(in) :: indent_str
        character(len=:), allocatable :: mlir
        character(len=:), allocatable :: arg_expr, output_ssa
        integer :: i

        mlir = ""

        ! HLFIR: Use hlfir.print for compile mode
        if (allocated(node%expression_indices)) then
            do i = 1, size(node%expression_indices)
                ! Generate expression for argument
                arg_expr = generate_mlir_expression(backend, arena, node%expression_indices(i), indent_str)
                mlir = mlir//arg_expr
                
                ! Generate HLFIR print operation
                output_ssa = backend%next_ssa_value()
                mlir = mlir//indent_str//output_ssa//" = hlfir.print "//backend%last_ssa_value// &
                       " : (!hlfir.expr<*>) -> ()"//new_line('a')
            end do
        end if
    end function generate_simple_print_for_compile

    ! Printf-based print (deprecated - use HLFIR operations instead)
    function generate_print_with_printf(backend, arena, node, indent_str) result(mlir)
        class(mlir_backend_t), intent(inout) :: backend
        type(ast_arena_t), intent(in) :: arena
        type(print_statement_node), intent(in) :: node
        character(len=*), intent(in) :: indent_str
        character(len=:), allocatable :: mlir

        ! Redirect to HLFIR print operations
        mlir = generate_mlir_print_statement(backend, arena, node, indent_str)
    end function generate_print_with_printf

    ! I/O runtime declarations - HLFIR operations don't need external declarations
    function generate_io_runtime_decls(backend, indent_str) result(mlir)
        class(mlir_backend_t), intent(inout) :: backend
        character(len=*), intent(in) :: indent_str
        character(len=:), allocatable :: mlir
        
        ! HLFIR I/O operations are built into the dialect - no external declarations needed
        mlir = indent_str//"// HLFIR I/O operations are built-in - no runtime declarations needed"//new_line('a')
    end function generate_io_runtime_decls

    ! HLFIR I/O runtime declarations - Generate FIR runtime function declarations
    function generate_hlfir_io_runtime_decls(backend, indent_str) result(mlir)
        class(mlir_backend_t), intent(inout) :: backend
        character(len=*), intent(in) :: indent_str
        character(len=:), allocatable :: mlir
        
        ! HLFIR-compliant FIR runtime function declarations using proper !fir.* types
        mlir = ""
        
        ! FIR runtime function for beginning external list output
        mlir = mlir//indent_str//"func.func private @_FortranAioBeginExternalListOutput("// &
               "i32, !fir.ref<!fir.char<1,0>>, i32) -> !fir.llvm_ptr"//new_line('a')
        
        ! FIR runtime function for ending I/O statement
        mlir = mlir//indent_str//"func.func private @_FortranAioEndIoStatement("// &
               "!fir.llvm_ptr) -> i32"//new_line('a')
        
        ! FIR runtime function for outputting character data
        mlir = mlir//indent_str//"func.func private @_FortranAioOutputCharacter("// &
               "!fir.llvm_ptr, !fir.ref<!fir.char<1,?>>, i64) -> i1"//new_line('a')
        
        ! FIR runtime function for beginning external list input
        mlir = mlir//indent_str//"func.func private @_FortranAioBeginExternalListInput("// &
               "i32, !fir.ref<!fir.char<1,0>>, i32) -> !fir.llvm_ptr"//new_line('a')
        
        ! FIR runtime function for inputting character data
        mlir = mlir//indent_str//"func.func private @_FortranAioInputCharacter("// &
               "!fir.llvm_ptr, !fir.ref<!fir.char<1,?>>, i64) -> i1"//new_line('a')
        
        ! FIR runtime function for beginning external formatted output
        mlir = mlir//indent_str//"func.func private @_FortranAioBeginExternalFormattedOutput("// &
               "i32, !fir.ref<!fir.char<1,?>>, i64) -> !fir.llvm_ptr"//new_line('a')
        
        ! FIR runtime function for setting format at runtime
        mlir = mlir//indent_str//"func.func private @_FortranAioSetFormat("// &
               "!fir.llvm_ptr, !fir.ref<!fir.char<1,?>>, i64) -> ()"//new_line('a')
        
        ! Additional FIR runtime functions for integer and real I/O
        mlir = mlir//indent_str//"func.func private @_FortranAioOutputInteger32("// &
               "!fir.llvm_ptr, i32) -> i1"//new_line('a')
        
        mlir = mlir//indent_str//"func.func private @_FortranAioOutputReal32("// &
               "!fir.llvm_ptr, f32) -> i1"//new_line('a')
        
        mlir = mlir//indent_str//"func.func private @_FortranAioOutputReal64("// &
               "!fir.llvm_ptr, f64) -> i1"//new_line('a')
        
        mlir = mlir//indent_str//"func.func private @_FortranAioInputInteger32("// &
               "!fir.llvm_ptr, !fir.ref<i32>) -> i1"//new_line('a')
        
        mlir = mlir//indent_str//"func.func private @_FortranAioInputReal32("// &
               "!fir.llvm_ptr, !fir.ref<f32>) -> i1"//new_line('a')
        
        mlir = mlir//indent_str//"func.func private @_FortranAioInputReal64("// &
               "!fir.llvm_ptr, !fir.ref<f64>) -> i1"//new_line('a')
        
        ! Add blank line after declarations
        mlir = mlir//new_line('a')
    end function generate_hlfir_io_runtime_decls

    ! Generate print runtime calls using HLFIR
    function generate_print_runtime_calls(backend, arena, node, indent_str) result(mlir)
        class(mlir_backend_t), intent(inout) :: backend
        type(ast_arena_t), intent(in) :: arena
        type(print_statement_node), intent(in) :: node
        character(len=*), intent(in) :: indent_str
        character(len=:), allocatable :: mlir
        
        ! Delegate to HLFIR print implementation
        mlir = generate_mlir_print_statement(backend, arena, node, indent_str)
    end function generate_print_runtime_calls

    ! Generate write runtime calls using HLFIR
    function generate_write_runtime_calls(backend, arena, node, indent_str) result(mlir)
        class(mlir_backend_t), intent(inout) :: backend
        type(ast_arena_t), intent(in) :: arena
        type(write_statement_node), intent(in) :: node
        character(len=*), intent(in) :: indent_str
        character(len=:), allocatable :: mlir
        
        ! Delegate to HLFIR write implementation
        mlir = generate_mlir_write_statement(backend, arena, node, indent_str)
    end function generate_write_runtime_calls

    ! Generate read runtime calls using HLFIR
    function generate_read_runtime_calls(backend, arena, node, indent_str) result(mlir)
        class(mlir_backend_t), intent(inout) :: backend
        type(ast_arena_t), intent(in) :: arena
        type(read_statement_node), intent(in) :: node
        character(len=*), intent(in) :: indent_str
        character(len=:), allocatable :: mlir
        
        ! Delegate to HLFIR read implementation
        mlir = generate_mlir_read_statement(backend, arena, node, indent_str)
    end function generate_read_runtime_calls

    ! Generate HLFIR-compliant I/O error handling for iostat/err/end specifiers
    function generate_io_error_handling(backend, arena, io_node, cookie_ssa, indent_str) result(mlir)
        class(mlir_backend_t), intent(inout) :: backend
        type(ast_arena_t), intent(in) :: arena
        class(*), intent(in) :: io_node  ! Can be read_statement_node or write_statement_node
        character(len=*), intent(in) :: cookie_ssa, indent_str
        character(len=:), allocatable :: mlir
        character(len=:), allocatable :: status_ssa, iostat_ssa, err_cond_ssa, end_cond_ssa
        integer :: iostat_var_index, err_label_index, end_label_index

        mlir = ""
        iostat_var_index = 0
        err_label_index = 0 
        end_label_index = 0

        ! Extract error handling fields from the I/O node
        select type (io_node)
        type is (read_statement_node)
            iostat_var_index = io_node%iostat_var_index
            err_label_index = io_node%err_label_index
            end_label_index = io_node%end_label_index
        type is (write_statement_node)
            iostat_var_index = io_node%iostat_var_index
            err_label_index = io_node%err_label_index
            end_label_index = io_node%end_label_index
        end select

        ! Step 1: End I/O statement and get status
        status_ssa = backend%next_ssa_value()
        mlir = mlir//indent_str//status_ssa//" = fir.call @_FortranAioEndIoStatement("//cookie_ssa// &
               ") : (!fir.llvm_ptr) -> i32"//new_line('a')

        ! Step 2: Handle iostat specifier (assign status to variable)
        if (iostat_var_index > 0) then
            mlir = mlir//generate_mlir_expression(backend, arena, iostat_var_index, indent_str)
            iostat_ssa = backend%last_ssa_value
            mlir = mlir//indent_str//"fir.store "//status_ssa//" to "//iostat_ssa// &
                   " : !fir.ref<i32>"//new_line('a')
        end if

        ! Step 3: Handle err specifier (branch on any error)
        if (err_label_index > 0) then
            err_cond_ssa = backend%next_ssa_value()
            mlir = mlir//indent_str//err_cond_ssa//" = arith.cmpi sgt, "//status_ssa//", "// &
                   "fir.constant 0 : i32 : i32, i32"//new_line('a')
            
            ! Generate label from node index
            mlir = mlir//indent_str//"cf.cond_br "//err_cond_ssa//", ^bb"// &
                   io_int_to_str(err_label_index)//", ^bb_continue"//new_line('a')
        end if

        ! Step 4: Handle end specifier (branch on EOF - status < 0)
        if (end_label_index > 0) then
            end_cond_ssa = backend%next_ssa_value()
            mlir = mlir//indent_str//end_cond_ssa//" = arith.cmpi slt, "//status_ssa//", "// &
                   "fir.constant 0 : i32 : i32, i32"//new_line('a')
            
            ! Generate label from node index  
            mlir = mlir//indent_str//"cf.cond_br "//end_cond_ssa//", ^bb"// &
                   io_int_to_str(end_label_index)//", ^bb_continue"//new_line('a')
        end if

        ! Step 5: Continue label for normal execution
        if (err_label_index > 0 .or. end_label_index > 0) then
            mlir = mlir//indent_str//"^bb_continue:"//new_line('a')
        end if
    end function generate_io_error_handling

    ! Generate format string setup for HLFIR-compliant formatted I/O
    function generate_format_setup(backend, format_spec, indent_str) result(mlir)
        class(mlir_backend_t), intent(inout) :: backend
        character(len=*), intent(in) :: format_spec, indent_str
        character(len=:), allocatable :: mlir
        character(len=:), allocatable :: format_global, format_ssa
        integer :: i
        
        mlir = ""
        
        ! Create global format string constant
        format_global = "@.str.fmt."//io_int_to_str(backend%format_counter)
        backend%format_counter = backend%format_counter + 1
        
        ! Add to global declarations
        backend%global_declarations = backend%global_declarations// &
            "fir.global "//format_global//" constant : !fir.char<1,"// &
            io_int_to_str(len_trim(format_spec))//"> {"//new_line('a')// &
            "  fir.string_lit """//trim(format_spec)//""" : !fir.char<1,"// &
            io_int_to_str(len_trim(format_spec))//">"//new_line('a')// &
            "}"//new_line('a')
        
        ! Get address of format string
        format_ssa = backend%next_ssa_value()
        mlir = mlir//indent_str//format_ssa//" = fir.address_of("//format_global// &
               ") : !fir.ref<!fir.char<1,"//io_int_to_str(len_trim(format_spec))//">>"//new_line('a')
        
        backend%last_ssa_value = format_ssa
    end function generate_format_setup

    ! Forward declaration placeholder for generate_mlir_expression
    function generate_mlir_expression(backend, arena, node_index, indent_str) result(mlir)
        class(mlir_backend_t), intent(inout) :: backend
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        character(len=*), intent(in) :: indent_str
        character(len=:), allocatable :: mlir
        
        mlir = "! generate_mlir_expression placeholder"//new_line('a')
    end function generate_mlir_expression

    ! Generate type-specific output call based on expression type
    function generate_typed_output_call(backend, arena, expr_index, cookie_ssa, arg_ssa, indent_str) result(mlir)
        class(mlir_backend_t), intent(inout) :: backend
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: expr_index
        character(len=*), intent(in) :: cookie_ssa, arg_ssa, indent_str
        character(len=:), allocatable :: mlir
        
        ! Simple type detection based on expression node type
        ! This is a basic implementation - full type analysis would require semantic information
        if (expr_index > 0 .and. expr_index <= arena%size) then
            ! Check the actual node type using polymorphism
            select type (node => arena%entries(expr_index)%node)
            type is (literal_node)
                ! Check the literal type
                select case (node%literal_type)
                case ("integer")
                    mlir = indent_str//"fir.call @_FortranAioOutputInteger32("//cookie_ssa//", "//arg_ssa// &
                           ") : (!fir.llvm_ptr, i32) -> i1"//new_line('a')
                case ("real")
                    mlir = indent_str//"fir.call @_FortranAioOutputReal64("//cookie_ssa//", "//arg_ssa// &
                           ") : (!fir.llvm_ptr, f64) -> i1"//new_line('a')
                case ("character")
                    mlir = indent_str//"fir.call @_FortranAioOutputCharacter("//cookie_ssa//", "//arg_ssa// &
                           ", fir.constant 1 : i64) : (!fir.llvm_ptr, !fir.ref<!fir.char<1,?>>, i64) -> i1"//new_line('a')
                case default
                    ! Default to integer output for unknown literal types
                    mlir = indent_str//"fir.call @_FortranAioOutputInteger32("//cookie_ssa//", "//arg_ssa// &
                           ") : (!fir.llvm_ptr, i32) -> i1"//new_line('a')
                end select
            class default
                ! For non-literal nodes, default to integer output
                ! In a complete implementation, this would query the type system
                mlir = indent_str//"fir.call @_FortranAioOutputInteger32("//cookie_ssa//", "//arg_ssa// &
                       ") : (!fir.llvm_ptr, i32) -> i1"//new_line('a')
            end select
        else
            ! Invalid expression index - use character output as fallback
            mlir = indent_str//"fir.call @_FortranAioOutputCharacter("//cookie_ssa//", "//arg_ssa// &
                   ", fir.constant 1 : i64) : (!fir.llvm_ptr, !fir.ref<!fir.char<1,?>>, i64) -> i1"//new_line('a')
        end if
    end function generate_typed_output_call

end module mlir_io_operations