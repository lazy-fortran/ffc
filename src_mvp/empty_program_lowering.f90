module empty_program_lowering
    use fortfront, only: ast_arena_t, assignment_node, binary_op_node, &
                         declaration_node, do_loop_node, identifier_node, &
                         literal_node, if_node, print_statement_node, program_node
    implicit none
    private

    public :: lower_empty_program_to_llvm

    integer, parameter :: MAX_SYMBOLS = 64

    type :: lowering_context_t
        character(len=:), allocatable :: body
        character(len=64) :: names(MAX_SYMBOLS) = ''
        character(len=128) :: values(MAX_SYMBOLS) = ''
        integer :: symbol_count = 0
        integer :: temp_count = 0
        integer :: block_count = 0
    end type lowering_context_t

contains

    subroutine lower_empty_program_to_llvm(arena, root_index, llvm_ir, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: root_index
        character(len=:), allocatable, intent(out) :: llvm_ir
        character(len=:), allocatable, intent(out) :: error_msg
        type(lowering_context_t) :: context

        if (root_index <= 0) then
            error_msg = 'FortFront did not return a root program index'
            call set_empty(llvm_ir)
            return
        end if

        if (.not. arena%has_node_at(root_index)) then
            error_msg = 'FortFront root index does not reference an AST node'
            call set_empty(llvm_ir)
            return
        end if

        select type (program => arena%entries(root_index)%node)
        type is (program_node)
            call set_empty(context%body)
            call lower_program_body(arena, program, context, error_msg)
            if (len_trim(error_msg) > 0) then
                call set_empty(llvm_ir)
                return
            end if
        class default
            error_msg = 'ffc MVP only supports a top-level program unit'
            call set_empty(llvm_ir)
            return
        end select

        llvm_ir = '@.fmt_i32 = private constant [4 x i8] c"%d\0A\00"'// &
                  new_line('a')//new_line('a')// &
                  'declare i32 @printf(ptr, ...)'//new_line('a')// &
                  new_line('a')// &
                  'define i32 @main() {'//new_line('a')// &
                  'entry:'//new_line('a')// &
                  context%body// &
                  '  ret i32 0'//new_line('a')// &
                  '}'//new_line('a')
        call set_empty(error_msg)
    end subroutine lower_empty_program_to_llvm

    subroutine lower_program_body(arena, program, context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(program_node), intent(in) :: program
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i

        call set_empty(error_msg)
        if (.not. allocated(program%body_indices)) return

        do i = 1, size(program%body_indices)
            call lower_statement(arena, program%body_indices(i), context, error_msg)
            if (len_trim(error_msg) > 0) return
        end do
    end subroutine lower_program_body

    subroutine lower_statement(arena, node_index, context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: value
        integer :: i

        call set_empty(error_msg)
        if (.not. arena%has_node_at(node_index)) return

        select type (node => arena%entries(node_index)%node)
        type is (declaration_node)
            call lower_declaration(node, context, error_msg)
        type is (assignment_node)
            call lower_assignment(arena, node, context, error_msg)
        type is (if_node)
            call lower_if(arena, node_index, node, context, error_msg)
        type is (do_loop_node)
            call lower_do_loop(arena, node, context, error_msg)
        type is (print_statement_node)
            if (.not. allocated(node%expression_indices)) return
            do i = 1, size(node%expression_indices)
                call lower_expr(arena, node%expression_indices(i), context, value, &
                                error_msg)
                if (len_trim(error_msg) > 0) return
                call append_line(context, next_print_temp(context)//' = '// &
                                 'call i32 (ptr, ...) @printf(ptr @.fmt_i32, '// &
                                 'i32 '//value//')')
            end do
        class default
            error_msg = 'ffc MVP does not support statement kind: '// &
                        trim(arena%entries(node_index)%node_type)
        end select
    end subroutine lower_statement

    subroutine lower_if(arena, node_index, node, context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(if_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: condition
        character(len=:), allocatable :: label_id
        integer :: condition_index
        integer :: i

        condition_index = node%condition_index
        if (condition_index <= 0) then
            condition_index = first_child_index(arena, node_index)
        end if
        if (condition_index <= 0) then
            error_msg = 'FortFront if node did not expose a condition index'
            return
        end if

        call lower_condition(arena, condition_index, context, condition, &
                             error_msg)
        if (len_trim(error_msg) > 0) return

        label_id = next_block_id(context)
        call append_line(context, 'br i1 '//condition//', label %then'// &
                         label_id//', label %else'//label_id)
        call append_label(context, 'then'//label_id)
        if (allocated(node%then_body_indices)) then
            do i = 1, size(node%then_body_indices)
                call lower_statement(arena, node%then_body_indices(i), context, &
                                     error_msg)
                if (len_trim(error_msg) > 0) return
            end do
        end if
        call append_line(context, 'br label %endif'//label_id)

        call append_label(context, 'else'//label_id)
        if (allocated(node%else_body_indices)) then
            do i = 1, size(node%else_body_indices)
                call lower_statement(arena, node%else_body_indices(i), context, &
                                     error_msg)
                if (len_trim(error_msg) > 0) return
            end do
        end if
        call append_line(context, 'br label %endif'//label_id)
        call append_label(context, 'endif'//label_id)
    end subroutine lower_if

    subroutine lower_do_loop(arena, node, context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(do_loop_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: start_value
        integer :: end_value
        integer :: step_value
        integer :: value
        integer :: i

        call set_empty(error_msg)
        if (.not. allocated(node%var_name)) then
            error_msg = 'ffc MVP requires a named do-loop variable'
            return
        end if

        call constant_integer(arena, node%start_expr_index, start_value, error_msg)
        if (len_trim(error_msg) > 0) return
        call constant_integer(arena, node%end_expr_index, end_value, error_msg)
        if (len_trim(error_msg) > 0) return
        if (node%step_expr_index > 0) then
            call constant_integer(arena, node%step_expr_index, step_value, error_msg)
            if (len_trim(error_msg) > 0) return
        else
            step_value = 1
        end if
        if (step_value == 0) then
            error_msg = 'ffc MVP does not support zero do-loop step'
            return
        end if

        value = start_value
        do while (loop_continues(value, end_value, step_value))
            call set_symbol(context, node%var_name, int_to_text(value), error_msg)
            if (len_trim(error_msg) > 0) return
            if (allocated(node%body_indices)) then
                do i = 1, size(node%body_indices)
                    call lower_statement(arena, node%body_indices(i), context, &
                                         error_msg)
                    if (len_trim(error_msg) > 0) return
                end do
            end if
            value = value + step_value
        end do
    end subroutine lower_do_loop

    integer function first_child_index(arena, node_index) result(child_index)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index

        child_index = 0
        if (.not. arena%has_node_at(node_index)) return
        if (.not. allocated(arena%entries(node_index)%child_indices)) return
        if (arena%entries(node_index)%child_count <= 0) return

        child_index = arena%entries(node_index)%child_indices(1)
    end function first_child_index

    subroutine lower_declaration(node, context, error_msg)
        type(declaration_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i

        call set_empty(error_msg)
        if (allocated(node%type_name)) then
            if (trim(node%type_name) /= 'integer') then
                error_msg = 'ffc MVP only supports integer declarations'
                return
            end if
        end if

        if (node%is_multi_declaration .and. allocated(node%var_names)) then
            do i = 1, size(node%var_names)
                call define_symbol(context, node%var_names(i), '0', error_msg)
                if (len_trim(error_msg) > 0) return
            end do
        else if (allocated(node%var_name)) then
            call define_symbol(context, node%var_name, '0', error_msg)
        end if
    end subroutine lower_declaration

    subroutine lower_assignment(arena, node, context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(assignment_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: name
        character(len=:), allocatable :: value

        call identifier_name(arena, node%target_index, name, error_msg)
        if (len_trim(error_msg) > 0) return

        call lower_expr(arena, node%value_index, context, value, error_msg)
        if (len_trim(error_msg) > 0) return

        call set_symbol(context, name, value, error_msg)
    end subroutine lower_assignment

    recursive subroutine lower_expr(arena, node_index, context, value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: left
        character(len=:), allocatable :: right
        character(len=:), allocatable :: op

        call set_empty(error_msg)
        call set_empty(value)
        if (.not. arena%has_node_at(node_index)) then
            error_msg = 'Invalid expression node'
            return
        end if

        select type (node => arena%entries(node_index)%node)
        type is (literal_node)
            if (.not. is_integer_literal(node%value)) then
                error_msg = 'ffc MVP only supports integer literals'
                return
            end if
            value = trim(node%value)
        type is (identifier_node)
            call lookup_symbol(context, node%name, value, error_msg)
        type is (binary_op_node)
            call lower_expr(arena, node%left_index, context, left, error_msg)
            if (len_trim(error_msg) > 0) return
            call lower_expr(arena, node%right_index, context, right, error_msg)
            if (len_trim(error_msg) > 0) return
            call llvm_binary_op(node%operator, op, error_msg)
            if (len_trim(error_msg) > 0) return
            value = next_temp(context)
            call append_line(context, value//' = '//op//' i32 '//left//', '//right)
        class default
            error_msg = 'ffc MVP does not support expression kind: '// &
                        trim(arena%entries(node_index)%node_type)
        end select
    end subroutine lower_expr

    recursive subroutine lower_condition(arena, node_index, context, value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: left
        character(len=:), allocatable :: right
        character(len=:), allocatable :: predicate

        call set_empty(error_msg)
        call set_empty(value)
        if (.not. arena%has_node_at(node_index)) then
            error_msg = 'Invalid condition node'
            return
        end if

        select type (node => arena%entries(node_index)%node)
        type is (binary_op_node)
            call llvm_compare_predicate(node%operator, predicate, error_msg)
            if (len_trim(error_msg) > 0) return
            call lower_expr(arena, node%left_index, context, left, error_msg)
            if (len_trim(error_msg) > 0) return
            call lower_expr(arena, node%right_index, context, right, error_msg)
            if (len_trim(error_msg) > 0) return
            value = next_temp(context)
            call append_line(context, value//' = icmp '//predicate//' i32 '// &
                             left//', '//right)
        class default
            call lower_expr(arena, node_index, context, left, error_msg)
            if (len_trim(error_msg) > 0) return
            value = next_temp(context)
            call append_line(context, value//' = icmp ne i32 '//left//', 0')
        end select
    end subroutine lower_condition

    logical function is_integer_literal(text) result(is_integer)
        character(len=*), intent(in) :: text
        integer :: i
        integer :: start

        is_integer = len_trim(text) > 0
        if (.not. is_integer) return

        start = 1
        if (text(1:1) == '-' .or. text(1:1) == '+') start = 2
        if (start > len_trim(text)) then
            is_integer = .false.
            return
        end if

        do i = start, len_trim(text)
            if (text(i:i) < '0' .or. text(i:i) > '9') then
                is_integer = .false.
                return
            end if
        end do
    end function is_integer_literal

    subroutine llvm_binary_op(source_op, llvm_op, error_msg)
        character(len=*), intent(in) :: source_op
        character(len=:), allocatable, intent(out) :: llvm_op
        character(len=:), allocatable, intent(out) :: error_msg

        call set_empty(error_msg)
        select case (trim(source_op))
        case ('+')
            llvm_op = 'add'
        case ('-')
            llvm_op = 'sub'
        case ('*')
            llvm_op = 'mul'
        case ('/')
            llvm_op = 'sdiv'
        case default
            error_msg = 'ffc MVP does not support binary operator: '// &
                        trim(source_op)
            call set_empty(llvm_op)
        end select
    end subroutine llvm_binary_op

    subroutine llvm_compare_predicate(source_op, predicate, error_msg)
        character(len=*), intent(in) :: source_op
        character(len=:), allocatable, intent(out) :: predicate
        character(len=:), allocatable, intent(out) :: error_msg

        call set_empty(error_msg)
        select case (trim(source_op))
        case ('==', '.eq.', '.EQ.')
            predicate = 'eq'
        case ('/=', '.ne.', '.NE.')
            predicate = 'ne'
        case ('>')
            predicate = 'sgt'
        case ('>=')
            predicate = 'sge'
        case ('<')
            predicate = 'slt'
        case ('<=')
            predicate = 'sle'
        case default
            error_msg = 'ffc MVP does not support comparison operator: '// &
                        trim(source_op)
            call set_empty(predicate)
        end select
    end subroutine llvm_compare_predicate

    subroutine identifier_name(arena, node_index, name, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        character(len=:), allocatable, intent(out) :: name
        character(len=:), allocatable, intent(out) :: error_msg

        call set_empty(error_msg)
        call set_empty(name)
        if (.not. arena%has_node_at(node_index)) then
            error_msg = 'Invalid assignment target'
            return
        end if

        select type (node => arena%entries(node_index)%node)
        type is (identifier_node)
            name = node%name
        class default
            error_msg = 'ffc MVP only supports identifier assignment targets'
        end select
    end subroutine identifier_name

    subroutine constant_integer(arena, node_index, value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        integer, intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: io_stat

        value = 0
        call set_empty(error_msg)
        if (.not. arena%has_node_at(node_index)) then
            error_msg = 'Invalid integer constant expression'
            return
        end if

        select type (node => arena%entries(node_index)%node)
        type is (literal_node)
            if (.not. is_integer_literal(node%value)) then
                error_msg = 'ffc MVP requires integer do-loop bounds'
                return
            end if
            read (node%value, *, iostat=io_stat) value
            if (io_stat /= 0) then
                error_msg = 'Invalid integer literal in do-loop bound'
            end if
        class default
            error_msg = 'ffc MVP only supports literal do-loop bounds'
        end select
    end subroutine constant_integer

    function next_temp(context) result(name)
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable :: name
        character(len=32) :: buffer

        context%temp_count = context%temp_count + 1
        write (buffer, '("%t",I0)') context%temp_count
        name = trim(buffer)
    end function next_temp

    function next_print_temp(context) result(name)
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable :: name
        character(len=32) :: buffer

        context%temp_count = context%temp_count + 1
        write (buffer, '("%p",I0)') context%temp_count
        name = trim(buffer)
    end function next_print_temp

    function next_block_id(context) result(name)
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable :: name
        character(len=32) :: buffer

        context%block_count = context%block_count + 1
        write (buffer, '(I0)') context%block_count
        name = trim(buffer)
    end function next_block_id

    logical function loop_continues(value, end_value, step_value) result(continues)
        integer, intent(in) :: value
        integer, intent(in) :: end_value
        integer, intent(in) :: step_value

        if (step_value > 0) then
            continues = value <= end_value
        else
            continues = value >= end_value
        end if
    end function loop_continues

    function int_to_text(value) result(text)
        integer, intent(in) :: value
        character(len=:), allocatable :: text
        character(len=32) :: buffer

        write (buffer, '(I0)') value
        text = trim(buffer)
    end function int_to_text

    subroutine append_line(context, line)
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: line

        context%body = context%body//'  '//trim(line)//new_line('a')
    end subroutine append_line

    subroutine append_label(context, label)
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: label

        context%body = context%body//trim(label)//':'//new_line('a')
    end subroutine append_label

    subroutine define_symbol(context, name, value, error_msg)
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: name
        character(len=*), intent(in) :: value
        character(len=:), allocatable, intent(out) :: error_msg

        call set_empty(error_msg)
        if (context%symbol_count >= MAX_SYMBOLS) then
            error_msg = 'Too many scalar variables for ffc MVP'
            return
        end if
        context%symbol_count = context%symbol_count + 1
        context%names(context%symbol_count) = trim(name)
        context%values(context%symbol_count) = trim(value)
    end subroutine define_symbol

    subroutine set_symbol(context, name, value, error_msg)
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: name
        character(len=*), intent(in) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: index

        call set_empty(error_msg)
        index = find_symbol(context, name)
        if (index == 0) then
            call define_symbol(context, name, value, error_msg)
            return
        end if
        context%values(index) = trim(value)
    end subroutine set_symbol

    subroutine lookup_symbol(context, name, value, error_msg)
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: name
        character(len=:), allocatable, intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: index

        call set_empty(error_msg)
        index = find_symbol(context, name)
        if (index == 0) then
            error_msg = 'Undefined scalar variable in ffc MVP: '//trim(name)
            call set_empty(value)
            return
        end if
        value = trim(context%values(index))
    end subroutine lookup_symbol

    integer function find_symbol(context, name) result(index)
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: name
        integer :: i

        index = 0
        do i = 1, context%symbol_count
            if (trim(context%names(i)) == trim(name)) then
                index = i
                return
            end if
        end do
    end function find_symbol

    subroutine set_empty(value)
        character(len=:), allocatable, intent(out) :: value

        allocate (character(len=0) :: value)
    end subroutine set_empty

end module empty_program_lowering
