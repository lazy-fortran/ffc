module mlir_backend_core
    use backend_interface
    use ast_core
    use ast_core, only: LITERAL_INTEGER, LITERAL_REAL, LITERAL_STRING, LITERAL_COMPLEX
    use mlir_backend_types
    use mlir_backend_helpers
    use mlir_backend_functions
    use mlir_backend_statements
    use mlir_backend_operators
    use mlir_backend_output
    use mlir_types
    use mlir_compile, only: compile_mlir_to_output, apply_mlir_lowering_passes
    use mlir_hlfir_helpers
    use mlir_io_operations
    use mlir_intrinsics
    use mlir_control_flow
    use mlir_expressions
    implicit none
    private

    public :: generate_mlir_program, generate_mlir_node, generate_mlir_module
    public :: generate_mlir_interface_block, generate_mlir_function_declaration
    public :: generate_mlir_subroutine_declaration, generate_mlir_module_node
    public :: generate_mlir_use_statement, resolve_module_symbols
    public :: generate_mlir_where_construct, generate_mlir_derived_type

contains

    ! Generate complete MLIR module
    function generate_mlir_module(backend, arena, prog_index, options) result(mlir)
        class(mlir_backend_t), intent(inout) :: backend
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: prog_index
        type(backend_options_t), intent(in) :: options
        character(len=:), allocatable :: mlir
        character(len=:), allocatable :: body_content

        ! Store options in backend for use by generation functions
        ! Pure HLFIR mode - no standard dialects
        backend%enable_ad = options%enable_ad
        backend%compile_mode = options%compile_mode
        backend%emit_hlfir = .true.  ! Always emit HLFIR

        ! Initialize global declarations, error messages, and symbol table
        backend%global_declarations = ""
        backend%error_messages = ""  ! Reset errors for each generation
        if (.not. allocated(backend%symbol_names)) then
            allocate (backend%symbol_names(0))
            allocate (backend%symbol_memrefs(0))
        end if

        ! Debug output
        if (options%debug_info) then
            print *, "Backend compile_mode:", backend%compile_mode
            print *, "Backend emit_hlfir: true (pure HLFIR mode)"
        end if

        ! Start MLIR module
        mlir = "module {"//new_line('a')

        ! HLFIR: Use built-in HLFIR I/O operations (no external declarations needed)
        mlir = mlir//generate_hlfir_io_runtime_decls(backend, "  ")
        print *, "DEBUG: After FIR runtime decls, mlir length:", len_trim(mlir)
        mlir = mlir//new_line('a')

        ! Add optimization attributes if enabled (only when not compiling)
        if (options%optimize .and. .not. backend%compile_mode) then
            mlir = mlir//"  // Optimization enabled"//new_line('a')
        end if

        ! Generate the program node - this will create all necessary functions
        select type (prog_node => arena%entries(prog_index)%node)
        type is (program_node)
            ! Generate program function(s) at module level
            body_content = generate_mlir_program_functions(backend, arena, prog_node, 2)

            ! Add global declarations (string constants) before function declarations
            if (allocated(backend%global_declarations) .and. len_trim(backend%global_declarations) > 0) then
                mlir = mlir//new_line('a')//backend%global_declarations
            end if

            mlir = mlir//body_content
        class default
            ! For other node types, use regular generation
            body_content = generate_mlir_node(backend, arena, prog_index, 2)

            ! Add global declarations for other node types too
            if (allocated(backend%global_declarations) .and. len_trim(backend%global_declarations) > 0) then
                mlir = mlir//new_line('a')//backend%global_declarations
            end if

            mlir = mlir//body_content
        end select

        ! Add optimization passes as comments (only when not compiling)
        if (options%optimize .and. .not. backend%compile_mode) then
            mlir = mlir//"  // Apply optimization passes:"//new_line('a')
            mlir = mlir//"  // - Constant folding"//new_line('a')
            mlir = mlir//"  // - Dead code elimination"//new_line('a')
            mlir = mlir//"  // - Loop optimization"//new_line('a')
        end if

        ! Add AD-specific annotations
        if (options%enable_ad) then
            mlir = mlir//"  // Enzyme AD support:"//new_line('a')
            if (options%generate_gradients) then
                mlir = mlir//"  // - Gradient generation enabled"//new_line('a')
            end if
            if (options%ad_annotations) then
                mlir = mlir//"  // - AD annotations enabled"//new_line('a')
            end if
            if (options%validate_gradients) then
                mlir = mlir//"  // - Gradient validation enabled"//new_line('a')
            end if
        end if

        ! Close module
        mlir = mlir//"}"//new_line('a')
    end function generate_mlir_module

    ! Generate MLIR for any AST node
    recursive function generate_mlir_node(backend, arena, node_index, indent_level) result(mlir)
        class(mlir_backend_t), intent(inout) :: backend
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        integer, intent(in) :: indent_level
        character(len=:), allocatable :: mlir
        character(len=:), allocatable :: indent_str
        integer :: i

        mlir = ""

        if (node_index < 1 .or. node_index > arena%size) then
            return
        end if

        ! Create indentation
        indent_str = ""
        do i = 1, indent_level
            indent_str = indent_str//"  "
        end do

        ! Access the node through the arena
        select type (node => arena%entries(node_index)%node)
        type is (program_node)
            ! Program nodes should not be generated here - they're handled at module level
            mlir = ""
        type is (function_def_node)
            mlir = generate_mlir_function(backend, arena, node, indent_str)
        type is (subroutine_def_node)
            mlir = generate_mlir_subroutine(backend, arena, node, indent_str)
        type is (declaration_node)
            mlir = generate_mlir_declaration(backend, arena, node, indent_str)
        type is (assignment_node)
            mlir = generate_mlir_assignment(backend, arena, node, indent_str)
        type is (pointer_assignment_node)
            mlir = generate_mlir_pointer_assignment(backend, arena, node, indent_str)
        type is (binary_op_node)
            mlir = generate_mlir_binary_op(backend, arena, node, indent_str)
        type is (do_loop_node)
            mlir = generate_mlir_do_loop(backend, arena, node, indent_str)
        type is (do_while_node)
            mlir = generate_mlir_do_while(backend, arena, node, indent_str)
        type is (forall_node)
            mlir = generate_mlir_forall(backend, arena, node, indent_str)
        type is (select_case_node)
            mlir = generate_mlir_select_case(backend, arena, node, indent_str)
        type is (case_block_node)
            mlir = generate_mlir_case_block(backend, arena, node, indent_str)
        type is (case_range_node)
            mlir = generate_mlir_case_range(backend, arena, node, indent_str)
        type is (case_default_node)
            mlir = generate_mlir_case_default(backend, arena, node, indent_str)
        type is (if_node)
            mlir = generate_mlir_if(backend, arena, node, indent_str)
        type is (subroutine_call_node)
            mlir = generate_mlir_subroutine_call(backend, arena, node, indent_str)
        type is (identifier_node)
            mlir = generate_mlir_identifier(backend, arena, node, indent_str)
        type is (literal_node)
            mlir = generate_mlir_literal(backend, arena, node, indent_str)
        type is (return_node)
            mlir = generate_mlir_return(backend, arena, node, indent_str)
        type is (exit_node)
            mlir = generate_mlir_exit(backend, arena, node, indent_str)
        type is (cycle_node)
            mlir = generate_mlir_cycle(backend, arena, node, indent_str)
        type is (call_or_subscript_node)
            mlir = generate_mlir_call_or_subscript(backend, arena, node, indent_str)
        type is (print_statement_node)
            mlir = generate_mlir_print_statement(backend, arena, node, indent_str)
        type is (write_statement_node)
            mlir = generate_mlir_write_statement(backend, arena, node, indent_str)
        type is (read_statement_node)
            mlir = generate_mlir_read_statement(backend, arena, node, indent_str)
        type is (module_node)
            mlir = generate_mlir_module_node(backend, arena, node, indent_str)
        type is (use_statement_node)
            mlir = generate_mlir_use_statement(backend, arena, node, indent_str)
        type is (array_literal_node)
            mlir = generate_mlir_array_literal(backend, arena, node, indent_str)
        type is (complex_literal_node)
            ! Generate complex literal
            block
                character(len=:), allocatable :: real_part, imag_part, complex_ssa
                ! Generate real part
                real_part = generate_mlir_node(backend, arena, node%real_index, indent_level)
                mlir = mlir//real_part
                ! Generate imaginary part
                imag_part = generate_mlir_node(backend, arena, node%imag_index, indent_level)
                mlir = mlir//imag_part
                ! Create complex value
                complex_ssa = backend%next_ssa_value()
                backend%last_ssa_value = complex_ssa
                mlir = mlir//indent_str//complex_ssa//" = hlfir.expr { %c = fir.undefined : !fir.complex<4>; "// &
                       "fir.result %c : !fir.complex<4> } : !hlfir.expr<!fir.complex<4>>"//new_line('a')
            end block
        type is (where_node)
            mlir = generate_mlir_where_construct(backend, arena, node, indent_str)
        type is (derived_type_node)
            mlir = generate_mlir_derived_type(backend, arena, node, indent_str)
        type is (allocate_statement_node)
            mlir = generate_mlir_allocate_statement(backend, arena, node, indent_str)
        type is (deallocate_statement_node)
            mlir = generate_mlir_deallocate_statement(backend, arena, node, indent_str)
        type is (interface_block_node)
            mlir = generate_mlir_interface_block(backend, arena, node, indent_str)
        class default
            mlir = indent_str//"// Unsupported node type"//new_line('a')
        end select
    end function generate_mlir_node

    ! Generate MLIR for program node
    function generate_mlir_program(backend, arena, node, indent_str) result(mlir)
        class(mlir_backend_t), intent(inout) :: backend
        type(ast_arena_t), intent(in) :: arena
        type(program_node), intent(in) :: node
        character(len=*), intent(in) :: indent_str
        character(len=:), allocatable :: mlir
        character(len=:), allocatable :: body_content
        integer :: i

        ! For programs, generate the program function and a main wrapper
        ! First generate the program function if not named "main"
        if (trim(node%name) /= "main") then
            ! HLFIR: All functions are fir.func
            mlir = indent_str//"fir.func @"//trim(node%name)//"() {"//new_line('a')

            ! Generate body for the program function
            if (allocated(node%body_indices)) then
                do i = 1, size(node%body_indices)
                    select type (nested_node => arena%entries(node%body_indices(i))%node)
                    type is (function_def_node)
                        continue  ! Skip functions for module-level handling
                    type is (subroutine_def_node)
                        continue  ! Skip subroutines for module-level handling
                    class default
                        body_content = generate_mlir_node(backend, arena, node%body_indices(i), 3)
                        mlir = mlir//body_content
                    end select
                end do
            end if

            ! End program function
            mlir = mlir//indent_str//"  fir.return"//new_line('a')
            mlir = mlir//indent_str//"}"//new_line('a')//new_line('a')

            ! Now generate main wrapper that calls the program function
            mlir = mlir//indent_str//"fir.func @main() {"//new_line('a')
            mlir = mlir//indent_str//"  fir.call @"//trim(node%name)//"() : () -> ()"//new_line('a')
            mlir = mlir//indent_str//"  fir.return"//new_line('a')
            mlir = mlir//indent_str//"}"//new_line('a')
        else
            ! Program is already named "main"
            mlir = generate_mlir_program_functions(backend, arena, node, 1)
        end if
    end function generate_mlir_program

    ! Generate MLIR for interface block node
    function generate_mlir_interface_block(backend, arena, node, indent_str) result(mlir)
        class(mlir_backend_t), intent(inout) :: backend
        type(ast_arena_t), intent(in) :: arena
        type(interface_block_node), intent(in) :: node
        character(len=*), intent(in) :: indent_str
        character(len=:), allocatable :: mlir
        character(len=:), allocatable :: proc_mlir
        integer :: i

        mlir = ""

        ! Interface blocks in MLIR are represented as function declarations
        ! (forward declarations without bodies)
        if (allocated(node%procedure_indices)) then
            do i = 1, size(node%procedure_indices)
                if (node%procedure_indices(i) > 0 .and. &
                    node%procedure_indices(i) <= arena%size) then

                    select type (proc_node => arena%entries(node%procedure_indices(i))%node)
                    type is (function_def_node)
                        ! Generate function declaration without body
                        proc_mlir = generate_mlir_function_declaration(backend, arena, proc_node, indent_str)
                        mlir = mlir//proc_mlir
                    type is (subroutine_def_node)
                        ! Generate subroutine declaration without body
                        proc_mlir = generate_mlir_subroutine_declaration(backend, arena, proc_node, indent_str)
                        mlir = mlir//proc_mlir
                    class default
                        ! Skip unsupported procedure types
                        mlir = mlir//indent_str//"// Unsupported procedure type in interface"//new_line('a')
                    end select
                end if
            end do
        end if
    end function generate_mlir_interface_block

    ! Generate MLIR function declaration (without body) for interface blocks
    function generate_mlir_function_declaration(backend, arena, node, indent_str) result(mlir)
        class(mlir_backend_t), intent(inout) :: backend
        type(ast_arena_t), intent(in) :: arena
        type(function_def_node), intent(in) :: node
        character(len=*), intent(in) :: indent_str
        character(len=:), allocatable :: mlir
        character(len=:), allocatable :: return_type, param_list, func_name

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

        ! Generate function declaration using FIR
        mlir = indent_str//"fir.func private @"//trim(func_name)//"("//trim(param_list)//") -> "//return_type//new_line('a')
    end function generate_mlir_function_declaration

    ! Generate MLIR subroutine declaration (without body) for interface blocks
    function generate_mlir_subroutine_declaration(backend, arena, node, indent_str) result(mlir)
        class(mlir_backend_t), intent(inout) :: backend
        type(ast_arena_t), intent(in) :: arena
        type(subroutine_def_node), intent(in) :: node
        character(len=*), intent(in) :: indent_str
        character(len=:), allocatable :: mlir
        character(len=:), allocatable :: param_list

        ! Generate parameter list
        param_list = generate_function_parameter_list(backend, arena, node%param_indices)

        ! Generate subroutine declaration (private means it's just a declaration)
        mlir = indent_str//"fir.func private @"//trim(node%name)//"("//trim(param_list)//")"//new_line('a')
    end function generate_mlir_subroutine_declaration

    ! Generate MLIR for module node
    function generate_mlir_module_node(backend, arena, node, indent_str) result(mlir)
        class(mlir_backend_t), intent(inout) :: backend
        type(ast_arena_t), intent(in) :: arena
        type(module_node), intent(in) :: node
        character(len=*), intent(in) :: indent_str
        character(len=:), allocatable :: mlir
        character(len=:), allocatable :: decl_mlir, proc_mlir, inner_indent
        integer :: i

        mlir = ""
        inner_indent = indent_str//"  "

        ! Generate module scope with proper symbol namespacing
        ! In compile mode, generate actual MLIR symbol table
        if (backend%compile_mode) then
            ! Use fir.func with visibility attributes for module representation
            mlir = indent_str//"// ======== Fortran Module: "//trim(node%name)//" ========="//new_line('a')

            ! Mark the beginning of module scope for symbol table
            backend%current_module_name = trim(node%name)
        else
            mlir = indent_str//"// ======== Fortran Module: "//trim(node%name)//" ========="//new_line('a')
        end if

        ! Generate module variable declarations as global variables with module prefix
        if (allocated(node%declaration_indices)) then
            mlir = mlir//indent_str//"// Module variables"//new_line('a')
            do i = 1, size(node%declaration_indices)
                if (node%declaration_indices(i) > 0 .and. node%declaration_indices(i) <= arena%size) then
                    select type (decl_node => arena%entries(node%declaration_indices(i))%node)
                    type is (declaration_node)
                        ! Generate global variable with module prefix for proper namespacing
                        mlir = mlir//indent_str//"fir.global @"//trim(node%name)//"_"//trim(decl_node%var_name)//" : "
                        if (decl_node%is_array .and. allocated(decl_node%dimension_indices)) then
                            mlir = mlir//fortran_array_to_mlir_type(backend%compile_mode, arena, decl_node, indent_str)
                        else
                            mlir = mlir//fortran_to_mlir_type(decl_node%type_name, decl_node%kind_value, backend%compile_mode)
                        end if
                        mlir = mlir//new_line('a')
                    class default
                        ! Other declarations (parameters, etc.)
                        decl_mlir = generate_mlir_node(backend, arena, node%declaration_indices(i), len(indent_str)/2)
                        mlir = mlir//decl_mlir
                    end select
                end if
            end do
        end if

        ! Generate module procedures with proper namespacing
        if (node%has_contains .and. allocated(node%procedure_indices)) then
            mlir = mlir//indent_str//"// Module procedures"//new_line('a')
            do i = 1, size(node%procedure_indices)
                if (node%procedure_indices(i) > 0 .and. node%procedure_indices(i) <= arena%size) then
                    select type (proc_node => arena%entries(node%procedure_indices(i))%node)
                    type is (function_def_node)
                        ! Generate function with module scope (TODO: add proper namespacing)
                        proc_mlir = generate_mlir_function(backend, arena, proc_node, indent_str)
                        mlir = mlir//proc_mlir
                    type is (subroutine_def_node)
                        ! Generate subroutine with module scope (TODO: add proper namespacing)
                        proc_mlir = generate_mlir_subroutine(backend, arena, proc_node, indent_str)
                        mlir = mlir//proc_mlir
                    class default
                        proc_mlir = generate_mlir_node(backend, arena, node%procedure_indices(i), len(indent_str)/2)
                        mlir = mlir//proc_mlir
                    end select
                end if
            end do
        end if

        ! Clear module name when exiting module scope
        if (backend%compile_mode) then
            if (allocated(backend%current_module_name)) then
                deallocate (backend%current_module_name)
            end if
        end if

        mlir = mlir//indent_str//"// ======== End Module: "//trim(node%name)//" ========="//new_line('a')
    end function generate_mlir_module_node

    ! Generate MLIR for use statement
    function generate_mlir_use_statement(backend, arena, node, indent_str) result(mlir)
        class(mlir_backend_t), intent(inout) :: backend
        type(ast_arena_t), intent(in) :: arena
        type(use_statement_node), intent(in) :: node
        character(len=*), intent(in) :: indent_str
        character(len=:), allocatable :: mlir
        integer :: i

        associate(associate_placeholder => arena)
        end associate

        mlir = ""

        ! Generate symbol import declarations
        ! In MLIR, we need to declare external symbols that will be resolved at link time
        mlir = mlir//indent_str//"// Import from module: "//trim(node%module_name)//new_line('a')

        if (backend%compile_mode) then
            if (node%has_only .and. allocated(node%only_list)) then
                ! Generate explicit symbol imports for "only" clause using symbol resolution
                block
                    character(len=:), allocatable :: resolved_symbols
                    character(len=32), allocatable :: only_strings(:)

                    ! Convert string_t array to character array for symbol resolution
                    allocate (only_strings(size(node%only_list)))
                    do i = 1, size(node%only_list)
                        only_strings(i) = trim(node%only_list(i)%s)
                    end do

                    resolved_symbols = resolve_module_symbols(backend, node%module_name, .true., only_strings)

                    ! Add declarations to global declarations instead of inline for proper MLIR structure
                    if (index(resolved_symbols, "fir.func private") > 0 .or. &
                        index(resolved_symbols, "fir.global external") > 0) then
                        backend%global_declarations = backend%global_declarations//resolved_symbols
                        mlir = mlir//indent_str//"// External symbols from "//trim(node%module_name)// &
                               " declared at module level"//new_line('a')
                    end if
                end block
            else
                ! Import all symbols from the module
                block
                    character(len=:), allocatable :: resolved_symbols
                    resolved_symbols = resolve_module_symbols(backend, node%module_name, .false., [character(len=0) ::])

                    if (index(resolved_symbols, "fir.func private") > 0 .or. &
                        index(resolved_symbols, "fir.global external") > 0) then
                        backend%global_declarations = backend%global_declarations//resolved_symbols
                        mlir = mlir//indent_str//"// All symbols from "//trim(node%module_name)// &
                               " declared at module level"//new_line('a')
                    end if
                end block
            end if
        else
            ! Standard mode: just generate comments
            if (node%has_only .and. allocated(node%only_list)) then
                mlir = mlir//indent_str//"// Use only: "
                do i = 1, size(node%only_list)
                    if (i > 1) mlir = mlir//", "
                    mlir = mlir//trim(node%only_list(i)%s)
                end do
                mlir = mlir//new_line('a')
            else
                mlir = mlir//indent_str//"// Use all symbols from module"//new_line('a')
            end if
        end if
    end function generate_mlir_use_statement

    ! Resolve symbols from a module and generate external declarations
    function resolve_module_symbols(this, module_name, has_only, only_list) result(symbol_declarations)
        class(mlir_backend_t), intent(in) :: this
        character(len=*), intent(in) :: module_name
        logical, intent(in) :: has_only
        character(len=*), intent(in) :: only_list(:)
        character(len=:), allocatable :: symbol_declarations
        integer :: i

        associate(associate_placeholder => this)
        end associate

        symbol_declarations = ""

        ! For well-known standard modules, provide specific symbol declarations
        select case (trim(module_name))
        case ("iso_fortran_env")
            ! Generate FIR declarations for ISO_FORTRAN_ENV symbols
            if (has_only) then
                ! Only import requested symbols
                do i = 1, size(only_list)
                    select case (trim(only_list(i)))
                    case ("real64")
                        symbol_declarations = symbol_declarations// &
                                              "  fir.global external constant @iso_fortran_env_real64 : i32"//new_line('a')
                    case ("int32")
                        symbol_declarations = symbol_declarations// &
                                              "  fir.global external constant @iso_fortran_env_int32 : i32"//new_line('a')
                    case ("error_unit")
                        symbol_declarations = symbol_declarations// &
                                              "  fir.global external constant @iso_fortran_env_error_unit : i32"//new_line('a')
                    case default
                        ! Unknown symbol - add generic declaration
                        symbol_declarations = symbol_declarations// &
                                              "  // Unknown ISO_FORTRAN_ENV symbol: "//trim(only_list(i))//new_line('a')
                    end select
                end do
            else
                ! Import common ISO_FORTRAN_ENV symbols
                symbol_declarations = "  // ISO_FORTRAN_ENV symbols (common subset)"//new_line('a')// &
                                     "  fir.global external constant @iso_fortran_env_real64 : i32"//new_line('a')// &
                                     "  fir.global external constant @iso_fortran_env_int32 : i32"//new_line('a')// &
                                     "  fir.global external constant @iso_fortran_env_error_unit : i32"//new_line('a')
            end if

        case ("iso_c_binding")
            ! Generate FIR declarations for ISO_C_BINDING symbols
            if (has_only) then
                do i = 1, size(only_list)
                    select case (trim(only_list(i)))
                    case ("c_ptr")
                        symbol_declarations = symbol_declarations// &
                                              "  fir.global external constant @iso_c_binding_c_ptr : !fir.type<c_ptr>"//new_line('a')
                    case ("c_int")
                        symbol_declarations = symbol_declarations// &
                                              "  fir.global external constant @iso_c_binding_c_int : i32"//new_line('a')
                    case ("c_double")
                        symbol_declarations = symbol_declarations// &
                                              "  fir.global external constant @iso_c_binding_c_double : f64"//new_line('a')
                    case default
                        symbol_declarations = symbol_declarations// &
                                              "  // Unknown ISO_C_BINDING symbol: "//trim(only_list(i))//new_line('a')
                    end select
                end do
            else
                symbol_declarations = "  // ISO_C_BINDING symbols (common subset)"//new_line('a')// &
                                     "  fir.global external constant @iso_c_binding_c_ptr : !fir.type<c_ptr>"//new_line('a')// &
                                     "  fir.global external constant @iso_c_binding_c_int : i32"//new_line('a')// &
                                     "  fir.global external constant @iso_c_binding_c_double : f64"//new_line('a')
            end if

        case default
            ! For user-defined or unknown modules, generate generic external declarations
            if (has_only .and. size(only_list) > 0) then
                symbol_declarations = "  // External symbols from "//trim(module_name)//":"//new_line('a')
                do i = 1, size(only_list)
                    ! Assume procedures for now (could be enhanced with more sophisticated analysis)
                    symbol_declarations = symbol_declarations// &
                                         "  fir.func private @"//trim(module_name)//"_"//trim(only_list(i))// &
                                         "() : () -> i32"//new_line('a')
                end do
            else
                ! Add a generic external declaration to satisfy potential symbol references
                symbol_declarations = symbol_declarations// &
                                     "// External module interface (symbols resolved by linker)"//new_line('a')
            end if
        end select
    end function resolve_module_symbols

    ! Generate MLIR for where construct
    function generate_mlir_where_construct(backend, arena, node, indent_str) result(mlir)
        class(mlir_backend_t), intent(inout) :: backend
        type(ast_arena_t), intent(in) :: arena
        type(where_node), intent(in) :: node
        character(len=*), intent(in) :: indent_str
        character(len=:), allocatable :: mlir

        associate(associate_placeholder => backend)
        associate(associate_placeholder2 => arena)
        associate(associate_placeholder3 => node)
        end associate
        end associate
        end associate

        mlir = indent_str//"// TODO: Implement WHERE construct"//new_line('a')
    end function generate_mlir_where_construct

    ! Generate MLIR for derived type definition
    function generate_mlir_derived_type(backend, arena, node, indent_str) result(mlir)
        class(mlir_backend_t), intent(inout) :: backend
        type(ast_arena_t), intent(in) :: arena
        type(derived_type_node), intent(in) :: node
        character(len=*), intent(in) :: indent_str
        character(len=:), allocatable :: mlir

        associate(associate_placeholder => backend)
        associate(associate_placeholder2 => arena)
        associate(associate_placeholder3 => node)
        end associate
        end associate
        end associate

        mlir = indent_str//"// TODO: Implement derived type definition"//new_line('a')
    end function generate_mlir_derived_type

end module mlir_backend_core