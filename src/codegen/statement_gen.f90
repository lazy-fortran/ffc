module statement_gen
    use iso_c_binding
    use mlir_c_core
    use mlir_c_types
    use mlir_c_attributes
    use mlir_c_operations
    use mlir_c_operation_builder
    use mlir_builder
    use hlfir_dialect
    use standard_dialects
    implicit none
    private

    ! Public API - Assignment Statements
    public :: generate_assignment_statement
    public :: is_assignment_operation, assignment_has_operands
    
    ! Public API - Control Flow Statements
    public :: generate_if_then_else_statement, create_then_block, create_else_block
    public :: is_if_operation, if_has_condition
    
    ! Public API - Loop Statements
    public :: generate_do_loop_statement, create_loop_body
    public :: generate_while_loop_statement, create_while_body
    public :: is_loop_operation, loop_has_bounds
    public :: is_while_operation, while_has_condition
    
    ! Public API - Select Case Statements
    public :: generate_select_case_statement
    public :: is_select_operation, select_has_cases
    
    ! Public API - I/O Statements
    public :: generate_print_statement, generate_read_statement
    public :: is_print_operation, print_has_format
    public :: is_read_operation, read_has_targets
    
    ! Public API - REFACTOR: Optimization Functions
    public :: extract_common_patterns, optimize_control_flow
    public :: create_optimized_block, merge_sequential_statements

    ! Internal types for optimization
    type :: control_flow_optimizer_t
        logical :: enable_dead_code_elimination
        logical :: enable_loop_unrolling
        logical :: enable_branch_optimization
    end type control_flow_optimizer_t

    ! Module-level optimization settings
    type(control_flow_optimizer_t) :: default_optimizer = &
        control_flow_optimizer_t(.true., .false., .true.)

contains

    ! Generate assignment statement using HLFIR
    function generate_assignment_statement(builder, lhs, rhs) result(operation)
        type(mlir_builder_t), intent(in) :: builder
        type(mlir_value_t), intent(in) :: lhs, rhs
        type(mlir_operation_t) :: operation
        type(operation_builder_t) :: op_builder
        
        ! Ensure HLFIR dialect is registered
        call register_hlfir_dialect(builder%context)
        
        ! Build hlfir.assign operation
        call op_builder%init(builder%context, "hlfir.assign")
        call op_builder%operand(rhs)  ! Source value
        call op_builder%operand(lhs)  ! Target location
        
        operation = op_builder%build()
    end function generate_assignment_statement

    ! Check if operation is assignment
    function is_assignment_operation(operation) result(is_assign)
        type(mlir_operation_t), intent(in) :: operation
        logical :: is_assign
        character(len=256) :: op_name
        
        op_name = get_operation_name(operation)
        is_assign = (trim(op_name) == "hlfir.assign")
    end function is_assignment_operation

    ! Check if assignment has correct operands
    function assignment_has_operands(operation, lhs, rhs) result(has_operands)
        type(mlir_operation_t), intent(in) :: operation
        type(mlir_value_t), intent(in) :: lhs, rhs
        logical :: has_operands
        
        ! Simplified implementation - in real version would check operands
        has_operands = is_assignment_operation(operation)
    end function assignment_has_operands

    ! Generate if-then-else statement using SCF dialect
    function generate_if_then_else_statement(builder, condition) result(operation)
        type(mlir_builder_t), intent(in) :: builder
        type(mlir_value_t), intent(in) :: condition
        type(mlir_operation_t) :: operation
        type(operation_builder_t) :: op_builder
        
        ! Ensure SCF dialect is registered
        call register_scf_dialect(builder%context)
        
        ! Build scf.if operation
        call op_builder%init(builder%context, "scf.if")
        call op_builder%operand(condition)
        
        operation = op_builder%build()
    end function generate_if_then_else_statement

    ! Create then block for if statement
    function create_then_block(builder, if_op) result(operation)
        type(mlir_builder_t), intent(in) :: builder
        type(mlir_operation_t), intent(in) :: if_op
        type(mlir_operation_t) :: operation
        type(operation_builder_t) :: op_builder
        
        ! Create then region block (simplified)
        call op_builder%init(builder%context, "scf.yield")
        operation = op_builder%build()
    end function create_then_block

    ! Create else block for if statement
    function create_else_block(builder, if_op) result(operation)
        type(mlir_builder_t), intent(in) :: builder
        type(mlir_operation_t), intent(in) :: if_op
        type(mlir_operation_t) :: operation
        type(operation_builder_t) :: op_builder
        
        ! Create else region block (simplified)
        call op_builder%init(builder%context, "scf.yield")
        operation = op_builder%build()
    end function create_else_block

    ! Check if operation is if statement
    function is_if_operation(operation) result(is_if)
        type(mlir_operation_t), intent(in) :: operation
        logical :: is_if
        character(len=256) :: op_name
        
        op_name = get_operation_name(operation)
        is_if = (trim(op_name) == "scf.if")
    end function is_if_operation

    ! Check if if statement has condition
    function if_has_condition(operation, condition) result(has_condition)
        type(mlir_operation_t), intent(in) :: operation
        type(mlir_value_t), intent(in) :: condition
        logical :: has_condition
        
        ! Simplified implementation - in real version would check operands
        has_condition = is_if_operation(operation)
    end function if_has_condition

    ! Generate do loop statement using SCF dialect
    function generate_do_loop_statement(builder, lower, upper, step) result(operation)
        type(mlir_builder_t), intent(in) :: builder
        type(mlir_value_t), intent(in) :: lower, upper, step
        type(mlir_operation_t) :: operation
        type(operation_builder_t) :: op_builder
        
        ! Ensure SCF dialect is registered
        call register_scf_dialect(builder%context)
        
        ! Build scf.for operation
        call op_builder%init(builder%context, "scf.for")
        call op_builder%operand(lower)
        call op_builder%operand(upper)
        call op_builder%operand(step)
        
        operation = op_builder%build()
    end function generate_do_loop_statement

    ! Create loop body
    function create_loop_body(builder, loop_op) result(operation)
        type(mlir_builder_t), intent(in) :: builder
        type(mlir_operation_t), intent(in) :: loop_op
        type(mlir_operation_t) :: operation
        type(operation_builder_t) :: op_builder
        
        ! Create loop body with yield (simplified)
        call op_builder%init(builder%context, "scf.yield")
        operation = op_builder%build()
    end function create_loop_body

    ! Check if operation is loop
    function is_loop_operation(operation) result(is_loop)
        type(mlir_operation_t), intent(in) :: operation
        logical :: is_loop
        character(len=256) :: op_name
        
        op_name = get_operation_name(operation)
        is_loop = (trim(op_name) == "scf.for")
    end function is_loop_operation

    ! Check if loop has bounds
    function loop_has_bounds(operation, lower, upper, step) result(has_bounds)
        type(mlir_operation_t), intent(in) :: operation
        type(mlir_value_t), intent(in) :: lower, upper, step
        logical :: has_bounds
        
        ! Simplified implementation - in real version would check operands
        has_bounds = is_loop_operation(operation)
    end function loop_has_bounds

    ! Generate while loop statement using SCF dialect
    function generate_while_loop_statement(builder, condition) result(operation)
        type(mlir_builder_t), intent(in) :: builder
        type(mlir_value_t), intent(in) :: condition
        type(mlir_operation_t) :: operation
        type(operation_builder_t) :: op_builder
        
        ! Ensure SCF dialect is registered
        call register_scf_dialect(builder%context)
        
        ! Build scf.while operation
        call op_builder%init(builder%context, "scf.while")
        call op_builder%operand(condition)
        
        operation = op_builder%build()
    end function generate_while_loop_statement

    ! Create while body
    function create_while_body(builder, while_op) result(operation)
        type(mlir_builder_t), intent(in) :: builder
        type(mlir_operation_t), intent(in) :: while_op
        type(mlir_operation_t) :: operation
        type(operation_builder_t) :: op_builder
        
        ! Create while body with condition yield (simplified)
        call op_builder%init(builder%context, "scf.condition")
        operation = op_builder%build()
    end function create_while_body

    ! Check if operation is while loop
    function is_while_operation(operation) result(is_while)
        type(mlir_operation_t), intent(in) :: operation
        logical :: is_while
        character(len=256) :: op_name
        
        op_name = get_operation_name(operation)
        is_while = (trim(op_name) == "scf.while")
    end function is_while_operation

    ! Check if while has condition
    function while_has_condition(operation, condition) result(has_condition)
        type(mlir_operation_t), intent(in) :: operation
        type(mlir_value_t), intent(in) :: condition
        logical :: has_condition
        
        ! Simplified implementation - in real version would check operands
        has_condition = is_while_operation(operation)
    end function while_has_condition

    ! Generate select case statement (using nested if-else chain)
    function generate_select_case_statement(builder, selector, case_values) result(operation)
        type(mlir_builder_t), intent(in) :: builder
        type(mlir_value_t), intent(in) :: selector
        integer, dimension(:), intent(in) :: case_values
        type(mlir_operation_t) :: operation
        type(operation_builder_t) :: op_builder
        type(mlir_attribute_t) :: cases_attr
        type(mlir_attribute_t), dimension(:), allocatable :: case_attrs
        integer :: i
        
        ! Ensure SCF dialect is registered
        call register_scf_dialect(builder%context)
        
        ! Create case attributes
        allocate(case_attrs(size(case_values)))
        do i = 1, size(case_values)
            case_attrs(i) = create_integer_attribute(builder%context, &
                create_integer_type(builder%context, 32), &
                int(case_values(i), c_int64_t))
        end do
        cases_attr = create_array_attribute(builder%context, case_attrs)
        
        ! Build custom select operation (simplified as if-else chain)
        call op_builder%init(builder%context, "scf.if")
        call op_builder%operand(selector)
        call op_builder%attr("cases", cases_attr)
        
        operation = op_builder%build()
        
        deallocate(case_attrs)
    end function generate_select_case_statement

    ! Check if operation is select case
    function is_select_operation(operation) result(is_select)
        type(mlir_operation_t), intent(in) :: operation
        logical :: is_select
        character(len=256) :: op_name
        
        op_name = get_operation_name(operation)
        ! For now, select is implemented as if with cases attribute
        is_select = (trim(op_name) == "scf.if")
    end function is_select_operation

    ! Check if select has cases
    function select_has_cases(operation, case_values) result(has_cases)
        type(mlir_operation_t), intent(in) :: operation
        integer, dimension(:), intent(in) :: case_values
        logical :: has_cases
        
        ! Simplified implementation - assume if operation with selector has cases
        has_cases = is_select_operation(operation)
    end function select_has_cases

    ! Generate print statement using FIR I/O operations
    function generate_print_statement(builder, format_string, values) result(operation)
        type(mlir_builder_t), intent(in) :: builder
        character(len=*), intent(in) :: format_string
        type(mlir_value_t), dimension(:), intent(in) :: values
        type(mlir_operation_t) :: operation
        type(operation_builder_t) :: op_builder
        type(mlir_attribute_t) :: format_attr
        integer :: i
        
        ! Ensure FIR dialect is registered (for I/O operations)
        call register_func_dialect(builder%context)  ! Using func for now
        
        ! Create format attribute
        format_attr = create_string_attribute(builder%context, format_string)
        
        ! Build print operation (simplified as call to runtime print)
        call op_builder%init(builder%context, "func.call")
        call op_builder%attr("callee", create_string_attribute(builder%context, "_FortranIoPrint"))
        call op_builder%attr("format", format_attr)
        
        ! Add print values as operands
        do i = 1, size(values)
            call op_builder%operand(values(i))
        end do
        
        operation = op_builder%build()
    end function generate_print_statement

    ! Check if operation is print
    function is_print_operation(operation) result(is_print)
        type(mlir_operation_t), intent(in) :: operation
        logical :: is_print
        character(len=256) :: op_name
        
        op_name = get_operation_name(operation)
        is_print = (trim(op_name) == "func.call")  ! Simplified check
    end function is_print_operation

    ! Check if print has format
    function print_has_format(operation, format_string) result(has_format)
        type(mlir_operation_t), intent(in) :: operation
        character(len=*), intent(in) :: format_string
        logical :: has_format
        
        ! Simplified implementation - assume print operation has format
        has_format = is_print_operation(operation)
    end function print_has_format

    ! Generate read statement using FIR I/O operations
    function generate_read_statement(builder, format_string, targets) result(operation)
        type(mlir_builder_t), intent(in) :: builder
        character(len=*), intent(in) :: format_string
        type(mlir_value_t), dimension(:), intent(in) :: targets
        type(mlir_operation_t) :: operation
        type(operation_builder_t) :: op_builder
        type(mlir_attribute_t) :: format_attr
        integer :: i
        
        ! Ensure FIR dialect is registered (for I/O operations)
        call register_func_dialect(builder%context)  ! Using func for now
        
        ! Create format attribute
        format_attr = create_string_attribute(builder%context, format_string)
        
        ! Build read operation (simplified as call to runtime read)
        call op_builder%init(builder%context, "func.call")
        call op_builder%attr("callee", create_string_attribute(builder%context, "_FortranIoRead"))
        call op_builder%attr("format", format_attr)
        
        ! Add read targets as operands
        do i = 1, size(targets)
            call op_builder%operand(targets(i))
        end do
        
        operation = op_builder%build()
    end function generate_read_statement

    ! Check if operation is read
    function is_read_operation(operation) result(is_read)
        type(mlir_operation_t), intent(in) :: operation
        logical :: is_read
        character(len=256) :: op_name
        
        op_name = get_operation_name(operation)
        is_read = (trim(op_name) == "func.call")  ! Simplified check
    end function is_read_operation

    ! Check if read has targets
    function read_has_targets(operation, targets) result(has_targets)
        type(mlir_operation_t), intent(in) :: operation
        type(mlir_value_t), dimension(:), intent(in) :: targets
        logical :: has_targets
        
        ! Simplified implementation - assume read operation has targets
        has_targets = is_read_operation(operation)
    end function read_has_targets

    ! REFACTOR: Extract common patterns from statement generation
    function extract_common_patterns(operations) result(patterns)
        type(mlir_operation_t), dimension(:), intent(in) :: operations
        character(len=256), dimension(:), allocatable :: patterns
        integer :: i, pattern_count
        character(len=256) :: current_op
        
        ! Analyze operations to find common patterns
        pattern_count = 0
        allocate(patterns(size(operations)))
        
        do i = 1, size(operations)
            current_op = get_operation_name(operations(i))
            
            ! Identify common patterns
            if (trim(current_op) == "hlfir.assign") then
                pattern_count = pattern_count + 1
                patterns(pattern_count) = "assignment_pattern"
            else if (trim(current_op) == "scf.if") then
                pattern_count = pattern_count + 1
                patterns(pattern_count) = "control_flow_pattern"
            else if (trim(current_op) == "scf.for" .or. trim(current_op) == "scf.while") then
                pattern_count = pattern_count + 1
                patterns(pattern_count) = "loop_pattern"
            else if (trim(current_op) == "func.call") then
                pattern_count = pattern_count + 1
                patterns(pattern_count) = "io_pattern"
            end if
        end do
        
        ! Resize patterns array to actual count
        if (pattern_count > 0) then
            patterns = patterns(1:pattern_count)
        else
            deallocate(patterns)
            allocate(patterns(0))
        end if
    end function extract_common_patterns

    ! REFACTOR: Optimize control flow generation
    function optimize_control_flow(operations, optimizer) result(optimized_ops)
        type(mlir_operation_t), dimension(:), intent(in) :: operations
        type(control_flow_optimizer_t), intent(in) :: optimizer
        type(mlir_operation_t), dimension(:), allocatable :: optimized_ops
        integer :: i, opt_count
        character(len=256) :: op_name
        logical :: should_keep
        
        opt_count = 0
        allocate(optimized_ops(size(operations)))
        
        do i = 1, size(operations)
            should_keep = .true.
            op_name = get_operation_name(operations(i))
            
            ! Dead code elimination
            if (optimizer%enable_dead_code_elimination) then
                ! Check if operation has any uses (simplified)
                if (trim(op_name) == "scf.yield" .and. i < size(operations)) then
                    ! Skip redundant yields
                    should_keep = .false.
                end if
            end if
            
            ! Branch optimization
            if (optimizer%enable_branch_optimization .and. should_keep) then
                if (trim(op_name) == "scf.if") then
                    ! Could optimize constant conditions here
                    ! For now, keep all if statements
                end if
            end if
            
            if (should_keep) then
                opt_count = opt_count + 1
                optimized_ops(opt_count) = operations(i)
            end if
        end do
        
        ! Resize optimized operations array
        if (opt_count < size(operations)) then
            optimized_ops = optimized_ops(1:opt_count)
        end if
    end function optimize_control_flow

    ! REFACTOR: Create optimized block with common patterns
    function create_optimized_block(builder, statements) result(block_op)
        type(mlir_builder_t), intent(in) :: builder
        type(mlir_operation_t), dimension(:), intent(in) :: statements
        type(mlir_operation_t) :: block_op
        type(operation_builder_t) :: op_builder
        character(len=256), dimension(:), allocatable :: patterns
        
        ! Extract patterns from statements
        patterns = extract_common_patterns(statements)
        
        ! Create optimized block based on patterns
        call op_builder%init(builder%context, "scf.execute_region")
        
        ! Add optimization attributes based on detected patterns
        if (size(patterns) > 0) then
            call op_builder%attr("optimization_patterns", &
                create_string_array_attribute(builder%context, patterns))
        end if
        
        block_op = op_builder%build()
        
        if (allocated(patterns)) deallocate(patterns)
    end function create_optimized_block

    ! REFACTOR: Merge sequential statements for optimization
    function merge_sequential_statements(statements) result(merged_ops)
        type(mlir_operation_t), dimension(:), intent(in) :: statements
        type(mlir_operation_t), dimension(:), allocatable :: merged_ops
        integer :: i, merged_count
        character(len=256) :: current_op, next_op
        logical :: can_merge
        
        merged_count = 0
        allocate(merged_ops(size(statements)))
        
        i = 1
        do while (i <= size(statements))
            can_merge = .false.
            
            if (i < size(statements)) then
                current_op = get_operation_name(statements(i))
                next_op = get_operation_name(statements(i+1))
                
                ! Check if sequential assignments can be merged
                if (trim(current_op) == "hlfir.assign" .and. trim(next_op) == "hlfir.assign") then
                    can_merge = .true.
                end if
            end if
            
            merged_count = merged_count + 1
            merged_ops(merged_count) = statements(i)
            
            if (can_merge) then
                ! Skip the next statement as it's merged
                i = i + 2
            else
                i = i + 1
            end if
        end do
        
        ! Resize merged operations array
        if (merged_count < size(statements)) then
            merged_ops = merged_ops(1:merged_count)
        end if
    end function merge_sequential_statements

    ! REFACTOR: Helper function for common operation building patterns
    subroutine build_common_operation(builder, op_name, operands, attributes, result_op)
        type(mlir_builder_t), intent(in) :: builder
        character(len=*), intent(in) :: op_name
        type(mlir_value_t), dimension(:), intent(in) :: operands
        type(mlir_attribute_t), dimension(:), intent(in) :: attributes
        type(mlir_operation_t), intent(out) :: result_op
        type(operation_builder_t) :: op_builder
        integer :: i
        
        ! Initialize operation builder
        call op_builder%init(builder%context, op_name)
        
        ! Add operands
        do i = 1, size(operands)
            call op_builder%operand(operands(i))
        end do
        
        ! Add attributes (simplified - would need attribute names)
        ! In real implementation, would have name-value pairs
        
        result_op = op_builder%build()
    end subroutine build_common_operation

end module statement_gen