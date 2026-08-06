submodule (session_program_lowering_impl) session_program_lowering_reject_const_init
    implicit none
contains
    ! Constant-expression validation (F2018 10.1.12).
    !
    ! A declaration initializer, and the ASYNCHRONOUS= specifier of a data
    ! transfer statement, must be an initialization expression: every primary
    ! must be a constant, a named constant, an inquiry whose result is fixed at
    ! compile time, or an implied-do index of the same constructor. A reference
    ! to a variable, to a user function, or to the shape of an assumed-shape or
    ! deferred-shape array is not. A constant expression whose folded value
    ! leaves the representable range of its type is invalid as well.
    !
    ! The check runs from validate_program, before any lowering, so it sees the
    ! declaration list of every scope and never needs a lowering context.

    module procedure check_constant_initialization_exprs
        integer :: n

        call set_empty(error_msg)
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (nd => arena%entries(n)%node)
            type is (program_node)
                if (allocated(nd%body_indices)) then
                    call check_scope_const_inits(arena, nd%body_indices, error_msg)
                end if
            type is (module_node)
                if (allocated(nd%declaration_indices)) then
                    call check_scope_const_inits(arena, nd%declaration_indices, &
                                                 error_msg)
                end if
            type is (function_def_node)
                if (allocated(nd%body_indices)) then
                    call check_scope_const_inits(arena, nd%body_indices, error_msg)
                end if
            type is (subroutine_def_node)
                if (allocated(nd%body_indices)) then
                    call check_scope_const_inits(arena, nd%body_indices, error_msg)
                end if
            end select
            if (len_trim(error_msg) > 0) return
        end do
    end procedure check_constant_initialization_exprs

    module procedure check_scope_const_inits
        character(len=:), allocatable :: reason
        character(len=64) :: location
        integer :: i

        call set_empty(error_msg)
        do i = 1, size(indices)
            if (.not. node_exists(arena, indices(i))) cycle
            select type (nd => arena%entries(indices(i))%node)
            type is (declaration_node)
                if (.not. nd%has_initializer) cycle
                if (nd%initializer_index <= 0) cycle
                call const_expr_reason(arena, indices, nd%initializer_index, &
                                       '|', reason)
                if (len_trim(reason) > 0) then
                    write (location, '(" at line ",I0,", column ",I0)') &
                        nd%line, nd%column
                    error_msg = trim(reason)//trim(location)
                    return
                end if
            type is (write_statement_node)
                if (.not. allocated(nd%specifiers)) cycle
                call check_async_specifiers(arena, indices, nd%specifiers, &
                                            nd%line, nd%column, error_msg)
            type is (read_statement_node)
                if (.not. allocated(nd%specifiers)) cycle
                call check_async_specifiers(arena, indices, nd%specifiers, &
                                            nd%line, nd%column, error_msg)
            end select
            if (len_trim(error_msg) > 0) return
        end do
    end procedure check_scope_const_inits

    ! The ASYNCHRONOUS= specifier of READ/WRITE must be an initialization
    ! expression (F2018 12.6.2.2); OPEN takes an ordinary scalar expression and
    ! is therefore left alone.
    module procedure check_async_specifiers
        character(len=:), allocatable :: reason
        character(len=64) :: location
        integer :: i

        call set_empty(error_msg)
        do i = 1, size(specifiers)
            if (.not. allocated(specifiers(i)%name)) cycle
            if (trim(lowercase_text(specifiers(i)%name)) /= 'asynchronous') cycle
            call set_empty(reason)
            if (specifiers(i)%value_node_index > 0) then
                call const_expr_reason(arena, indices, &
                                       specifiers(i)%value_node_index, '|', reason)
            else if (allocated(specifiers(i)%value)) then
                call bare_name_const_reason(arena, indices, specifiers(i)%value, &
                                            reason)
            end if
            if (len_trim(reason) > 0) then
                write (location, '(" at line ",I0,", column ",I0)') line, col
                error_msg = trim(reason)//trim(location)
                return
            end if
        end do
    end procedure check_async_specifiers

    ! A specifier value kept as source text: only a bare name can name a
    ! variable, so anything quoted or punctuated is left to the expression path.
    module procedure bare_name_const_reason
        character(len=:), allocatable :: name
        integer :: i

        call set_empty(reason)
        name = trim(adjustl(text))
        if (len(name) == 0) return
        do i = 1, len(name)
            if (scan(name(i:i), 'abcdefghijklmnopqrstuvwxyz'// &
                     'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_') == 0) return
        end do
        if (scan(name(1:1), '0123456789_') /= 0) return
        call identifier_const_reason(arena, indices, name, '|', reason)
    end procedure bare_name_const_reason

    module procedure const_expr_reason
        character(len=:), allocatable :: op, err, inner
        integer :: i, li, ri, ln, cl

        call set_empty(reason)
        if (idx <= 0) return
        if (.not. node_exists(arena, idx)) return
        if (is_binary_op(arena, idx)) then
            call get_binary_op_info(arena, idx, op, li, ri, ln, cl, err)
            if (len_trim(err) > 0) return
            call const_expr_reason(arena, indices, li, loop_names, reason)
            if (len_trim(reason) > 0) return
            call const_expr_reason(arena, indices, ri, loop_names, reason)
            return
        end if
        select type (nd => arena%entries(idx)%node)
        type is (identifier_node)
            if (.not. allocated(nd%name)) return
            call identifier_const_reason(arena, indices, nd%name, loop_names, &
                                         reason)
        type is (array_literal_node)
            if (.not. allocated(nd%element_indices)) return
            do i = 1, size(nd%element_indices)
                call const_expr_reason(arena, indices, nd%element_indices(i), &
                                       loop_names, reason)
                if (len_trim(reason) > 0) return
            end do
        type is (do_loop_node)
            ! An array-constructor implied-do; its index is constant inside.
            inner = loop_names
            if (allocated(nd%var_name)) then
                inner = inner//trim(lowercase_text(nd%var_name))//'|'
            end if
            if (allocated(nd%body_indices)) then
                do i = 1, size(nd%body_indices)
                    call const_expr_reason(arena, indices, nd%body_indices(i), &
                                           inner, reason)
                    if (len_trim(reason) > 0) return
                end do
            end if
            call const_expr_reason(arena, indices, nd%start_expr_index, &
                                   loop_names, reason)
            if (len_trim(reason) > 0) return
            call const_expr_reason(arena, indices, nd%end_expr_index, &
                                   loop_names, reason)
            if (len_trim(reason) > 0) return
            call const_expr_reason(arena, indices, nd%step_expr_index, &
                                   loop_names, reason)
        type is (io_implied_do_node)
            inner = loop_names
            if (allocated(nd%var_name)) then
                inner = inner//trim(lowercase_text(nd%var_name))//'|'
            end if
            call const_expr_reason(arena, indices, nd%expr_index, inner, reason)
            if (len_trim(reason) > 0) return
            if (allocated(nd%object_indices)) then
                do i = 1, size(nd%object_indices)
                    call const_expr_reason(arena, indices, &
                                           nd%object_indices(i), inner, reason)
                    if (len_trim(reason) > 0) return
                end do
            end if
            call const_expr_reason(arena, indices, nd%start_expr_index, &
                                   loop_names, reason)
            if (len_trim(reason) > 0) return
            call const_expr_reason(arena, indices, nd%end_expr_index, &
                                   loop_names, reason)
            if (len_trim(reason) > 0) return
            call const_expr_reason(arena, indices, nd%step_expr_index, &
                                   loop_names, reason)
        type is (call_or_subscript_node)
            call call_const_reason(arena, indices, nd, loop_names, reason)
        end select
    end procedure const_expr_reason

    module procedure call_const_reason
        character(len=:), allocatable :: cname
        integer :: i, first_arg

        call set_empty(reason)
        if (.not. allocated(nd%name)) return
        cname = trim(lowercase_text(nd%name))
        first_arg = 1
        select case (cname)
        case ('size', 'shape', 'lbound', 'ubound')
            ! Array inquiries are constant only when the shape itself is.
            if (allocated(nd%arg_indices)) then
                if (size(nd%arg_indices) >= 1) then
                    call shape_inquiry_reason(arena, indices, &
                                              nd%arg_indices(1), reason)
                    if (len_trim(reason) > 0) return
                end if
            end if
            first_arg = 2
        case ('kind', 'len', 'bit_size', 'huge', 'tiny', 'epsilon', 'digits', &
              'radix', 'maxexponent', 'minexponent', 'precision', 'range', &
              'storage_size', 'new_line', 'selected_int_kind', &
              'selected_real_kind', 'selected_char_kind')
            ! Type-parameter inquiries never read the argument's value.
            return
        case default
            if (.not. nd%is_array_access) then
                if (arena_has_function_def_named(arena, nd%name)) then
                    reason = 'function reference '''//trim(nd%name)// &
                             ''' is not a constant expression'
                    return
                end if
            end if
            call identifier_const_reason(arena, indices, nd%name, loop_names, &
                                         reason)
            if (len_trim(reason) > 0) return
        end select
        if (.not. allocated(nd%arg_indices)) return
        do i = first_arg, size(nd%arg_indices)
            call const_expr_reason(arena, indices, nd%arg_indices(i), &
                                   loop_names, reason)
            if (len_trim(reason) > 0) return
        end do
    end procedure call_const_reason

    ! A name is constant when it is an implied-do index of the enclosing
    ! constructor, a named constant, or not declared in this scope at all (a
    ! host- or use-associated name is left to the later lowering passes).
    module procedure identifier_const_reason
        character(len=:), allocatable :: lname
        integer :: i

        call set_empty(reason)
        lname = trim(lowercase_text(name))
        if (len(lname) == 0) return
        if (index(loop_names, '|'//lname//'|') > 0) return
        do i = 1, size(indices)
            if (.not. node_exists(arena, indices(i))) cycle
            select type (decl => arena%entries(indices(i))%node)
            type is (declaration_node)
                if (.not. declaration_declares_name(decl, lname)) cycle
                if (decl%is_parameter) return
                reason = 'variable '''//trim(name)// &
                         ''' does not reduce to a constant expression'
                return
            end select
        end do
    end procedure identifier_const_reason

    module procedure shape_inquiry_reason
        character(len=:), allocatable :: lname
        integer :: i, d
        logical :: assumed

        call set_empty(reason)
        if (arg_index <= 0) return
        if (.not. node_exists(arena, arg_index)) return
        select type (nd => arena%entries(arg_index)%node)
        type is (identifier_node)
            if (.not. allocated(nd%name)) return
            lname = trim(lowercase_text(nd%name))
        class default
            return
        end select
        do i = 1, size(indices)
            if (.not. node_exists(arena, indices(i))) cycle
            select type (decl => arena%entries(indices(i))%node)
            type is (declaration_node)
                if (.not. declaration_declares_name(decl, lname)) cycle
                if (.not. decl%is_array) return
                if (decl%is_allocatable .or. decl%is_pointer) then
                    reason = 'deferred-shape array '''//trim(lname)// &
                             ''' has no constant shape in an '// &
                             'initialization expression'
                    return
                end if
                if (.not. allocated(decl%dimension_indices)) return
                assumed = .false.
                do d = 1, size(decl%dimension_indices)
                    if (dim_is_assumed_shape(arena, decl%dimension_indices(d))) &
                        assumed = .true.
                end do
                if (assumed) then
                    reason = 'assumed-shape array '''//trim(lname)// &
                             ''' has no constant shape in an '// &
                             'initialization expression'
                end if
                return
            end select
        end do
    end procedure shape_inquiry_reason

    module procedure declaration_declares_name
        integer :: k

        declares = .false.
        if (allocated(decl%var_name)) then
            if (trim(lowercase_text(decl%var_name)) == lname) then
                declares = .true.
                return
            end if
        end if
        if (.not. decl%is_multi_declaration) return
        if (.not. allocated(decl%var_names)) return
        do k = 1, size(decl%var_names)
            if (trim(lowercase_text(decl%var_names(k))) == lname) then
                declares = .true.
                return
            end if
        end do
    end procedure declaration_declares_name
end submodule session_program_lowering_reject_const_init
