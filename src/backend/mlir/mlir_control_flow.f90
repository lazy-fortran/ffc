! MLIR Control Flow Functions Module
! This module contains all control flow statement handlers for MLIR code generation
module mlir_control_flow
    use fortfront
    use fortfront, only: LITERAL_INTEGER, LITERAL_REAL, LITERAL_STRING, LITERAL_COMPLEX
    use stdlib_strings, only: to_string
    use mlir_backend_types
    use mlir_hlfir_helpers
    implicit none

    private
    public :: generate_mlir_do_loop, generate_mlir_do_while, generate_mlir_forall
    public :: generate_mlir_select_case, generate_mlir_case_block, generate_mlir_case_range
    public :: generate_mlir_case_default, generate_mlir_if

contains

    ! Generate MLIR for do loop
    function generate_mlir_do_loop(backend, arena, node, indent_str) result(mlir)
        class(mlir_backend_t), intent(inout) :: backend
        type(ast_arena_t), intent(in) :: arena
        type(do_loop_node), intent(in) :: node
        character(len=*), intent(in) :: indent_str
        character(len=:), allocatable :: mlir
        character(len=:), allocatable :: start_ref, end_ref, step_ref, body_content
        character(len=:), allocatable :: ssa_val, label_comment
        character(len=:), allocatable :: start_ssa, end_ssa, step_ssa
        character(len=:), allocatable :: start_index_ssa, end_index_ssa, step_index_ssa
        integer :: i

        mlir = ""

        ! Add label comment if present
        if (allocated(node%label)) then
            label_comment = "// Named do loop: "//node%label
            mlir = mlir//indent_str//label_comment//new_line('a')
        end if

        ! Generate bounds
 start_ref = generate_mlir_expression(backend, arena, node%start_expr_index, indent_str)
     end_ref = generate_mlir_expression(backend, arena, node%end_expr_index, indent_str)
        mlir = mlir//start_ref//end_ref

        ! Generate step (default to 1 if not present)
        if (node%step_expr_index > 0) then
   step_ref = generate_mlir_expression(backend, arena, node%step_expr_index, indent_str)
            mlir = mlir//step_ref
        else
            ! HLFIR: Use helper function for constants
            mlir = mlir//generate_hlfir_constant_code(backend%next_ssa_value(), "1", "!fir.int<4>", indent_str)
            ssa_val = backend%last_ssa_value
        end if

        ! Get the actual SSA values for loop bounds and convert to index type
        start_ssa = "%"//to_string(backend%ssa_counter - 2)  ! start value
        end_ssa = "%"//to_string(backend%ssa_counter - 1)    ! end value
        step_ssa = "%"//to_string(backend%ssa_counter)       ! step value

        ! HLFIR Implementation: Use fir.do_loop instead of fir.do_loop
        ! Set loop context to track loop variable
        if (allocated(node%var_name)) then
            backend%current_loop_var = trim(node%var_name)
            backend%current_loop_ssa = "%iv"
        end if

        ! HLFIR: Use hlfir.elemental for loop expressions where possible, otherwise fir.do_loop
        if (allocated(node%label)) then
            mlir = mlir//indent_str//"// Loop label: "//node%label//new_line('a')
        end if
        
        ! Check if this is an elemental operation that can be simplified
        if (allocated(node%body_indices) .and. size(node%body_indices) == 1) then
            ! Try to use hlfir.elemental for simple loops
            mlir = mlir//indent_str//"%loop_result = hlfir.elemental (%iv) = "//start_ssa//" to "//end_ssa// &
                   " step "//step_ssa//" : !hlfir.expr<?xi32> {"//new_line('a')
        else
            ! Fall back to fir.do_loop for complex loops
            mlir = mlir//indent_str//"fir.do_loop %iv = "//start_ssa//" to "//end_ssa//" step "//step_ssa//" {"//new_line('a')
        end if

        ! Generate body
        if (allocated(node%body_indices)) then
            do i = 1, size(node%body_indices)
              body_content = generate_mlir_statement(backend, arena, node%body_indices(i), indent_str//"  ")
                mlir = mlir//body_content
            end do
        end if

        if (allocated(node%body_indices) .and. size(node%body_indices) == 1) then
            ! Complete hlfir.elemental
            mlir = mlir//indent_str//"  fir.result %body_result : i32"//new_line('a')
            mlir = mlir//indent_str//"}"//new_line('a')
        else
            ! Complete fir.do_loop
            mlir = mlir//indent_str//"}"//new_line('a')
        end if

        ! Clear loop context
        backend%current_loop_var = ""
        backend%current_loop_ssa = ""

        ! Add end label if present
        if (allocated(node%label)) then
            mlir = mlir//indent_str//"^"//node%label//"_end:"//new_line('a')
        end if
    end function generate_mlir_do_loop

    ! Generate MLIR for do-while loop
    function generate_mlir_do_while(backend, arena, node, indent_str) result(mlir)
        class(mlir_backend_t), intent(inout) :: backend
        type(ast_arena_t), intent(in) :: arena
        type(do_while_node), intent(in) :: node
        character(len=*), intent(in) :: indent_str
        character(len=:), allocatable :: mlir
        character(len=:), allocatable :: cond_mlir, body_content
       character(len=:), allocatable :: loop_start, loop_body, loop_end, condition_check
        integer :: i

        mlir = ""

        ! Generate do-while loop using control flow dialect
        ! Structure:
        !   fir.br ^loop_start
        ! ^loop_start:
        !   [body statements]
        !   [condition evaluation]
        !   fir.cond_br %condition, ^loop_start, ^loop_end
        ! ^loop_end:

        ! Branch to loop start
        mlir = mlir//indent_str//"fir.br ^loop_start"//new_line('a')

        ! Loop start block
        mlir = mlir//indent_str//"^loop_start:"//new_line('a')

        ! Generate body statements
        if (allocated(node%body_indices)) then
            do i = 1, size(node%body_indices)
                body_content = generate_mlir_statement(backend, arena, node%body_indices(i), indent_str//"  ")
                mlir = mlir//body_content
            end do
        end if

        ! Generate condition evaluation
        cond_mlir = generate_mlir_expression(backend, arena, node%condition_index, indent_str//"  ")
        mlir = mlir//cond_mlir

        ! Conditional branch: continue if condition true, exit if false
        mlir = mlir//indent_str//"  fir.cond_br "//trim(backend%last_ssa_value)//", ^loop_start, ^loop_end"//new_line('a')

        ! Loop end block
        mlir = mlir//indent_str//"^loop_end:"//new_line('a')
    end function generate_mlir_do_while

    ! Generate MLIR for forall statement
    function generate_mlir_forall(backend, arena, node, indent_str) result(mlir)
        class(mlir_backend_t), intent(inout) :: backend
        type(ast_arena_t), intent(in) :: arena
        type(forall_node), intent(in) :: node
        character(len=*), intent(in) :: indent_str
        character(len=:), allocatable :: mlir
        character(len=:), allocatable :: start_mlir, end_mlir, body_mlir
        character(len=:), allocatable :: start_ssa, end_ssa, loop_var_ssa, result_ssa
        integer :: i

        mlir = ""

        ! Validate we have at least one index specification
        if (node%num_indices < 1 .or. .not. allocated(node%lower_bound_indices) .or. &
            .not. allocated(node%upper_bound_indices)) then
            mlir = indent_str//"! Error: FORALL has no index specifications"//new_line('a')
            return
        end if

        ! Generate start value (use first index)
        start_mlir = generate_mlir_expression(backend, arena, &
            node%lower_bound_indices(1), indent_str)
        mlir = mlir//start_mlir
        start_ssa = backend%last_ssa_value

        ! Generate end value (use first index)
        end_mlir = generate_mlir_expression(backend, arena, &
            node%upper_bound_indices(1), indent_str)
        mlir = mlir//end_mlir
        end_ssa = backend%last_ssa_value

        ! HLFIR: Generate hlfir.elemental for parallel execution
        loop_var_ssa = backend%next_ssa_value()
        result_ssa = backend%next_ssa_value()

        mlir = mlir//indent_str//result_ssa//" = hlfir.elemental ("//loop_var_ssa//") = "//start_ssa// &
               " to "//end_ssa//" : !hlfir.expr<?xi32> {"//new_line('a')

        ! Store current loop context (use first index name)
        if (allocated(node%index_names) .and. size(node%index_names) > 0) then
            backend%current_loop_var = node%index_names(1)
        else
            backend%current_loop_var = "i"
        end if
        backend%current_loop_ssa = loop_var_ssa

        ! Generate forall body
        if (allocated(node%body_indices)) then
            do i = 1, size(node%body_indices)
                body_mlir = generate_mlir_statement(backend, arena, node%body_indices(i), indent_str//"  ")
                mlir = mlir//body_mlir
            end do
        end if

        mlir = mlir//indent_str//"  fir.result %elem : i32"//new_line('a')
        mlir = mlir//indent_str//"}"//new_line('a')

        backend%last_ssa_value = result_ssa
        
        ! Clear loop context
        backend%current_loop_var = ""
        backend%current_loop_ssa = ""
    end function generate_mlir_forall

    ! Generate MLIR for select case statement
    function generate_mlir_select_case(backend, arena, node, indent_str) result(mlir)
        class(mlir_backend_t), intent(inout) :: backend
        type(ast_arena_t), intent(in) :: arena
        type(select_case_node), intent(in) :: node
        character(len=*), intent(in) :: indent_str
        character(len=:), allocatable :: mlir
        character(len=:), allocatable :: expr_mlir, case_mlir
        character(len=:), allocatable :: expr_ssa, result_ssa
        integer :: i

        mlir = ""

        ! Generate selector expression
        expr_mlir = generate_mlir_expression(backend, arena, node%selector_index, indent_str)
        mlir = mlir//expr_mlir
        expr_ssa = backend%last_ssa_value

        ! HLFIR: Use fir.select for select case
        result_ssa = backend%next_ssa_value()
        mlir = mlir//indent_str//result_ssa//" = fir.select "//expr_ssa//" {"//new_line('a')

        ! Generate case blocks
        if (allocated(node%case_indices)) then
            do i = 1, size(node%case_indices)
                case_mlir = generate_mlir_case_block(backend, arena, &
                    arena%entries(node%case_indices(i))%node, indent_str//"  ")
                mlir = mlir//case_mlir
            end do
        end if

        mlir = mlir//indent_str//"}"//new_line('a')

        backend%last_ssa_value = result_ssa
    end function generate_mlir_select_case

    ! Generate MLIR for case block
    function generate_mlir_case_block(backend, arena, node, indent_str) result(mlir)
        class(mlir_backend_t), intent(inout) :: backend
        type(ast_arena_t), intent(in) :: arena
        class(*), intent(in) :: node
        character(len=*), intent(in) :: indent_str
        character(len=:), allocatable :: mlir
        character(len=:), allocatable :: body_mlir
        integer :: i

        mlir = ""

        select type (case_node => node)
        type is (case_block_node)
            ! Generate case value(s)
            if (allocated(case_node%value_indices)) then
                do i = 1, size(case_node%value_indices)
                    if (i == 1) then
                        mlir = mlir//indent_str//"case "
                    else
                        mlir = mlir//", "
                    end if
                    
                    if (allocated(arena%entries(case_node%value_indices(i))%node)) then
                        select type (val_node => arena%entries(case_node%value_indices(i))%node)
                        type is (literal_node)
                            mlir = mlir//trim(val_node%value)
                        class default
                            mlir = mlir//"???"
                        end select
                    end if
                end do
                mlir = mlir//" {"//new_line('a')
            end if

            ! Generate case body
            if (allocated(case_node%body_indices)) then
                do i = 1, size(case_node%body_indices)
                    body_mlir = generate_mlir_statement(backend, arena, case_node%body_indices(i), indent_str//"  ")
                    mlir = mlir//body_mlir
                end do
            end if

            mlir = mlir//indent_str//"  fir.result"//new_line('a')
            mlir = mlir//indent_str//"}"//new_line('a')

        type is (case_default_node)
            mlir = mlir//indent_str//"default {"//new_line('a')

            ! Generate default body
            if (allocated(case_node%body_indices)) then
                do i = 1, size(case_node%body_indices)
                    body_mlir = generate_mlir_statement(backend, arena, case_node%body_indices(i), indent_str//"  ")
                    mlir = mlir//body_mlir
                end do
            end if

            mlir = mlir//indent_str//"  fir.result"//new_line('a')
            mlir = mlir//indent_str//"}"//new_line('a')
        end select
    end function generate_mlir_case_block

    ! Generate MLIR for case range
    function generate_mlir_case_range(backend, arena, node, indent_str) result(mlir)
        class(mlir_backend_t), intent(inout) :: backend
        type(ast_arena_t), intent(in) :: arena
        type(case_range_node), intent(in) :: node
        character(len=*), intent(in) :: indent_str
        character(len=:), allocatable :: mlir
        character(len=:), allocatable :: start_val, end_val

        mlir = ""

        ! Get start and end values - they are integer values in the node
        start_val = trim(to_string(node%start_value))
        end_val = trim(to_string(node%end_value))

        ! Generate range syntax (MLIR doesn't have native range support, so expand)
        mlir = mlir//start_val//" to "//end_val
    end function generate_mlir_case_range

    ! Generate MLIR for case default
    function generate_mlir_case_default(backend, arena, node, indent_str) result(mlir)
        class(mlir_backend_t), intent(inout) :: backend
        type(ast_arena_t), intent(in) :: arena
        type(case_default_node), intent(in) :: node
        character(len=*), intent(in) :: indent_str
        character(len=:), allocatable :: mlir
        character(len=:), allocatable :: body_mlir
        integer :: i

        mlir = ""

        mlir = mlir//indent_str//"default {"//new_line('a')

        ! Generate default body
        if (allocated(node%body_indices)) then
            do i = 1, size(node%body_indices)
                body_mlir = generate_mlir_statement(backend, arena, node%body_indices(i), indent_str//"  ")
                mlir = mlir//body_mlir
            end do
        end if

        mlir = mlir//indent_str//"  fir.result"//new_line('a')
        mlir = mlir//indent_str//"}"//new_line('a')
    end function generate_mlir_case_default

    ! Generate MLIR for if statement
    function generate_mlir_if(backend, arena, node, indent_str) result(mlir)
        class(mlir_backend_t), intent(inout) :: backend
        type(ast_arena_t), intent(in) :: arena
        type(if_node), intent(in) :: node
        character(len=*), intent(in) :: indent_str
        character(len=:), allocatable :: mlir
        character(len=:), allocatable :: condition_mlir, then_mlir, else_mlir
        character(len=:), allocatable :: condition_ssa, result_ssa
        integer :: i

        mlir = ""

        ! Generate condition
        condition_mlir = generate_mlir_expression(backend, arena, node%condition_index, indent_str)
        mlir = mlir//condition_mlir
        condition_ssa = backend%last_ssa_value

        ! HLFIR: Generate fir.if
        result_ssa = backend%next_ssa_value()
        mlir = mlir//indent_str//result_ssa//" = fir.if "//condition_ssa//" {"//new_line('a')

        ! Generate then block
        if (allocated(node%then_body_indices)) then
            do i = 1, size(node%then_body_indices)
                then_mlir = generate_mlir_statement(backend, arena, node%then_body_indices(i), indent_str//"  ")
                mlir = mlir//then_mlir
            end do
        end if

        mlir = mlir//indent_str//"  fir.result"//new_line('a')

        ! Generate else block if present
        if (allocated(node%else_body_indices)) then
            mlir = mlir//indent_str//"} else {"//new_line('a')
            
            do i = 1, size(node%else_body_indices)
                else_mlir = generate_mlir_statement(backend, arena, node%else_body_indices(i), indent_str//"  ")
                mlir = mlir//else_mlir
            end do
            
            mlir = mlir//indent_str//"  fir.result"//new_line('a')
        end if

        mlir = mlir//indent_str//"}"//new_line('a')

        backend%last_ssa_value = result_ssa
    end function generate_mlir_if

    ! Forward declaration placeholder for generate_mlir_expression and generate_mlir_statement
    function generate_mlir_expression(backend, arena, node_index, indent_str) result(mlir)
        class(mlir_backend_t), intent(inout) :: backend
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        character(len=*), intent(in) :: indent_str
        character(len=:), allocatable :: mlir
        
        ! This should be implemented in the main backend or as an interface
        mlir = "! generate_mlir_expression placeholder"//new_line('a')
    end function generate_mlir_expression

    function generate_mlir_statement(backend, arena, node_index, indent_str) result(mlir)
        class(mlir_backend_t), intent(inout) :: backend
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        character(len=*), intent(in) :: indent_str
        character(len=:), allocatable :: mlir
        
        ! This should be implemented in the main backend or as an interface
        mlir = "! generate_mlir_statement placeholder"//new_line('a')
    end function generate_mlir_statement

end module mlir_control_flow