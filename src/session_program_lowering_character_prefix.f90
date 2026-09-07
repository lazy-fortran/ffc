submodule(session_program_lowering_impl) session_program_lowering_character_prefix
    implicit none
contains
    module subroutine initialize_character_prefix_result(node, binding, &
                                                         context, error_msg)
        type(function_def_node), intent(in) :: node
        type(declaration_binding_t), intent(in) :: binding
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(declaration_node) :: declaration
        logical :: found
        integer :: previous_declaration, result_index

        call character_prefix_declaration(node, declaration, found, error_msg)
        if (len_trim(error_msg) > 0 .or. .not. found) return
        ! An explicit result declaration remains authoritative when present.
        if (node_exists(context%arena, binding%declaration_node_index)) then
            select type (existing => &
                         context%arena%entries(binding%declaration_node_index)%node)
            type is (declaration_node)
                return
            end select
        end if
        result_index = context%current_function_result_index
        declaration%var_name = context%symbols(result_index)%name
        previous_declaration = context%current_declaration_index
        context%current_declaration_index = context%current_proc_node_index
        call define_declared_character_symbol(context, declaration, &
                                              declaration%var_name, error_msg)
        context%current_declaration_index = previous_declaration
        if (len_trim(error_msg) > 0) return
        ! Do not silently infer a prefix specification expression from the RHS
        ! when its runtime length cannot yet be evaluated by this lowerer.
        if (context%symbols(result_index)%is_runtime_fixed_character) return
        if (context%symbols(result_index)%character_length > 0) return
        if (.not. declaration%has_character_length) return
        if (declaration%character_length_expr == ':' .or. &
            declaration%character_length_expr == '*') return
        error_msg = 'unsupported character function result length expression'
    end subroutine initialize_character_prefix_result

    module subroutine character_prefix_host_reference(arena, node_index, &
                                                      name, procedure_index)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        character(len=:), allocatable, intent(out) :: name
        integer, intent(out) :: procedure_index
        type(declaration_node) :: declaration
        character(len=:), allocatable :: error_msg
        logical :: found

        name = ''
        procedure_index = 0
        select type (node => arena%entries(node_index)%node)
        type is (function_def_node)
            call character_prefix_declaration(node, declaration, found, error_msg)
            if (len_trim(error_msg) > 0 .or. .not. found) return
            if (.not. declaration%has_character_length) return
            if (verify(declaration%character_length_expr, &
                    'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_') &
                /= 0) return
            name = declaration%character_length_expr
            procedure_index = node_index
        end select
    end subroutine character_prefix_host_reference

    subroutine character_prefix_declaration(node, declaration, found, error_msg)
        ! FortFront stores a typed function prefix on return_type, without a
        ! body declaration. Adapt that AST metadata to the ordinary result
        ! declaration path; never inspect the original source text.
        type(function_def_node), intent(in) :: node
        type(declaration_node), intent(out) :: declaration
        logical, intent(out) :: found
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: selector, literal_error
        integer :: open_pos, close_pos, literal_length

        call set_empty(error_msg)
        found = .false.
        if (.not. node%has_return_type_in_header) return
        if (.not. allocated(node%return_type)) return
        if (.not. is_character_type_name(node%return_type)) return
        found = .true.
        declaration%type_name = node%return_type
        declaration%line = node%line
        declaration%column = node%column
        ! Retain existing literal/default/deferred selector handling, including
        ! an explicit default kind alongside a literal length.
        call parse_character_length(node%return_type, literal_length, literal_error)
        if (len_trim(literal_error) == 0) return
        open_pos = index(node%return_type, '(')
        if (open_pos == 0) return
        close_pos = index(node%return_type, ')', back=.true.)
        if (close_pos <= open_pos) then
            error_msg = 'character function result length is missing ")"'
            return
        end if
        selector = trim(adjustl(node%return_type(open_pos + 1:close_pos - 1)))
        if (index(selector, ',') > 0) then
            error_msg = 'unsupported combined character result length/kind selectors'
            return
        end if
        if (index(lowercase_text(selector), 'len=') == 1) then
            selector = trim(adjustl(selector(5:)))
        else if (index(selector, '=') > 0) then
            error_msg = 'unsupported character function result selector'
            return
        end if
        declaration%has_character_length = .true.
        declaration%character_length_expr = selector
    end subroutine character_prefix_declaration
end submodule session_program_lowering_character_prefix
