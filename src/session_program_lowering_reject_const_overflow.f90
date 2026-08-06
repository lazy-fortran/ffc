submodule (session_program_lowering_impl) session_program_lowering_reject_const_overflow
    implicit none
contains
    ! Arithmetic overflow in a constant expression (F2018 10.1.12).
    !
    ! A constant expression is evaluated by the compiler, so a folded value that
    ! leaves the representable range of its type makes the program invalid
    ! rather than merely wrapping at run time. Two contexts are folded here:
    ! integer array bounds and declaration initializers, and a REAL() kind
    ! conversion whose constant argument does not fit the requested kind.

    subroutine check_constant_expression_overflow(arena, error_msg)
        type(ast_arena_t), intent(in) :: arena
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=64) :: location
        integer :: n, d

        call set_empty(error_msg)
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (nd => arena%entries(n)%node)
            type is (declaration_node)
                if (allocated(nd%dimension_indices)) then
                    do d = 1, size(nd%dimension_indices)
                        if (.not. expr_overflows_integer(arena, &
                                nd%dimension_indices(d))) cycle
                        write (location, '(" at line ",I0,", column ",I0)') &
                            nd%line, nd%column
                        error_msg = 'arithmetic overflow in constant '// &
                                    'expression'//trim(location)
                        return
                    end do
                end if
                if (nd%has_initializer .and. nd%initializer_index > 0) then
                    if (expr_overflows_integer(arena, nd%initializer_index)) then
                        write (location, '(" at line ",I0,", column ",I0)') &
                            nd%line, nd%column
                        error_msg = 'arithmetic overflow in constant '// &
                                    'expression'//trim(location)
                        return
                    end if
                end if
            type is (call_or_subscript_node)
                if (.not. real_conversion_overflows(arena, nd)) cycle
                write (location, '(" at line ",I0,", column ",I0)') &
                    nd%line, nd%column
                error_msg = 'arithmetic overflow in constant '// &
                            'expression'//trim(location)
                return
            end select
        end do
    end subroutine check_constant_expression_overflow

    logical function expr_overflows_integer(arena, idx) result(overflows)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: idx
        integer(c_int64_t) :: value
        integer :: kind_bytes
        logical :: ok

        call const_int_fold(arena, idx, value, kind_bytes, ok, overflows)
    end function expr_overflows_integer

    ! Fold an integer constant expression built from literals, huge(), and the
    ! +, -, * operators, tracking the kind so the result range is checked
    ! against the type the expression actually has.
    recursive subroutine const_int_fold(arena, idx, value, kind_bytes, ok, ovf)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: idx
        integer(c_int64_t), intent(out) :: value
        integer, intent(out) :: kind_bytes
        logical, intent(out) :: ok
        logical, intent(out) :: ovf
        character(len=:), allocatable :: op, err, lit_value, lit_type, cname
        integer(c_int64_t) :: lv, rv
        integer :: li, ri, ln, cl, lk, rk

        value = 0_c_int64_t
        kind_bytes = 4
        ok = .false.
        ovf = .false.
        if (idx <= 0) return
        if (.not. node_exists(arena, idx)) return
        if (is_binary_op(arena, idx)) then
            call get_binary_op_info(arena, idx, op, li, ri, ln, cl, err)
            if (len_trim(err) > 0) return
            call const_int_fold(arena, li, lv, lk, ok, ovf)
            if (ovf) return
            if (.not. ok) return
            call const_int_fold(arena, ri, rv, rk, ok, ovf)
            if (ovf) return
            if (.not. ok) return
            kind_bytes = max(lk, rk)
            ok = .false.
            select case (trim(op))
            case ('+')
                call i64_add_checked(lv, rv, value, ovf)
                ok = .not. ovf
            case ('-')
                call i64_add_checked(lv, -rv, value, ovf)
                ok = .not. ovf
            case ('*')
                call i64_mul_checked(lv, rv, value, ovf)
                ok = .not. ovf
            case default
                return
            end select
            if (ovf) return
            if (abs(value) > integer_kind_limit(kind_bytes)) then
                ovf = .true.
                ok = .false.
            end if
            return
        end if
        select type (nd => arena%entries(idx)%node)
        type is (call_or_subscript_node)
            if (.not. allocated(nd%name)) return
            cname = trim(lowercase_text(nd%name))
            if (cname /= 'huge') return
            if (.not. allocated(nd%arg_indices)) return
            if (size(nd%arg_indices) /= 1) return
            call const_int_fold(arena, nd%arg_indices(1), lv, lk, ok, ovf)
            ovf = .false.
            if (.not. ok) return
            kind_bytes = lk
            value = integer_kind_limit(lk)
            ok = .true.
        class default
            if (.not. is_literal(arena, idx)) return
            call get_literal_info(arena, idx, lit_value, lit_type, err)
            if (len_trim(err) > 0) return
            if (allocated(lit_type)) then
                if (len_trim(lit_type) > 0) then
                    if (trim(lowercase_text(lit_type)) /= 'integer') return
                end if
            end if
            call parse_integer_literal(lit_value, value, kind_bytes, ok)
        end select
    end subroutine const_int_fold

    subroutine parse_integer_literal(text, value, kind_bytes, ok)
        character(len=*), intent(in) :: text
        integer(c_int64_t), intent(out) :: value
        integer, intent(out) :: kind_bytes
        logical, intent(out) :: ok
        character(len=:), allocatable :: body, suffix
        integer :: us, ios

        value = 0_c_int64_t
        kind_bytes = 4
        ok = .false.
        body = trim(adjustl(text))
        if (len(body) == 0) return
        us = index(body, '_')
        if (us > 0) then
            suffix = trim(body(us + 1:))
            body = trim(body(:us - 1))
            if (.not. is_integer_text(suffix)) return
            read (suffix, *, iostat=ios) kind_bytes
            if (ios /= 0) return
        end if
        if (.not. is_integer_text(body)) return
        read (body, *, iostat=ios) value
        if (ios /= 0) return
        ok = .true.
    end subroutine parse_integer_literal

    integer(c_int64_t) function integer_kind_limit(kind_bytes) result(limit)
        integer, intent(in) :: kind_bytes

        select case (kind_bytes)
        case (1)
            limit = 127_c_int64_t
        case (2)
            limit = 32767_c_int64_t
        case (8)
            limit = huge(0_c_int64_t)
        case default
            limit = 2147483647_c_int64_t
        end select
    end function integer_kind_limit

    subroutine i64_add_checked(a, b, r, ovf)
        integer(c_int64_t), intent(in) :: a, b
        integer(c_int64_t), intent(out) :: r
        logical, intent(out) :: ovf

        ovf = .false.
        r = 0_c_int64_t
        if (b > 0_c_int64_t) then
            if (a > huge(0_c_int64_t) - b) then
                ovf = .true.
                return
            end if
        else if (b < 0_c_int64_t) then
            if (a < -huge(0_c_int64_t) - b) then
                ovf = .true.
                return
            end if
        end if
        r = a + b
    end subroutine i64_add_checked

    subroutine i64_mul_checked(a, b, r, ovf)
        integer(c_int64_t), intent(in) :: a, b
        integer(c_int64_t), intent(out) :: r
        logical, intent(out) :: ovf

        ovf = .false.
        r = 0_c_int64_t
        if (a /= 0_c_int64_t) then
            if (abs(b) > huge(0_c_int64_t)/abs(a)) then
                ovf = .true.
                return
            end if
        end if
        r = a*b
    end subroutine i64_mul_checked

    ! real(x, kind) with a constant x whose magnitude exceeds the target kind.
    logical function real_conversion_overflows(arena, nd) result(overflows)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: nd
        character(len=:), allocatable :: cname
        real(c_double) :: value
        integer(c_int64_t) :: kind_value
        integer :: kind_bytes
        logical :: ok, ovf

        overflows = .false.
        if (.not. allocated(nd%name)) return
        cname = trim(lowercase_text(nd%name))
        if (cname /= 'real') return
        if (.not. allocated(nd%arg_indices)) return
        if (size(nd%arg_indices) /= 2) return
        call const_int_fold(arena, nd%arg_indices(2), kind_value, kind_bytes, &
                            ok, ovf)
        if (.not. ok) return
        if (kind_value /= 4_c_int64_t) return
        call const_real_fold(arena, nd%arg_indices(1), value, ok)
        if (.not. ok) return
        overflows = abs(value) > real(huge(0.0_c_float), c_double)
    end function real_conversion_overflows

    ! Fold a real constant expression built from real literals and huge().
    recursive subroutine const_real_fold(arena, idx, value, ok)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: idx
        real(c_double), intent(out) :: value
        logical, intent(out) :: ok
        character(len=:), allocatable :: lit_value, lit_type, err, cname
        integer :: kind_bytes

        value = 0.0_c_double
        ok = .false.
        if (idx <= 0) return
        if (.not. node_exists(arena, idx)) return
        select type (nd => arena%entries(idx)%node)
        type is (call_or_subscript_node)
            if (.not. allocated(nd%name)) return
            cname = trim(lowercase_text(nd%name))
            if (cname /= 'huge') return
            if (.not. allocated(nd%arg_indices)) return
            if (size(nd%arg_indices) /= 1) return
            call real_literal_kind(arena, nd%arg_indices(1), kind_bytes, ok)
            if (.not. ok) return
            if (kind_bytes == 8) then
                value = huge(0.0_c_double)
            else
                value = real(huge(0.0_c_float), c_double)
            end if
        class default
            if (.not. is_literal(arena, idx)) return
            call get_literal_info(arena, idx, lit_value, lit_type, err)
            if (len_trim(err) > 0) return
            if (allocated(lit_type)) then
                if (len_trim(lit_type) > 0) then
                    if (trim(lowercase_text(lit_type)) /= 'real') return
                end if
            end if
            call parse_real_literal(lit_value, value, kind_bytes, ok)
        end select
    end subroutine const_real_fold

    subroutine real_literal_kind(arena, idx, kind_bytes, ok)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: idx
        integer, intent(out) :: kind_bytes
        logical, intent(out) :: ok
        character(len=:), allocatable :: lit_value, lit_type, err
        real(c_double) :: value

        kind_bytes = 4
        ok = .false.
        if (idx <= 0) return
        if (.not. node_exists(arena, idx)) return
        if (.not. is_literal(arena, idx)) return
        call get_literal_info(arena, idx, lit_value, lit_type, err)
        if (len_trim(err) > 0) return
        if (allocated(lit_type)) then
            if (len_trim(lit_type) > 0) then
                if (trim(lowercase_text(lit_type)) /= 'real') return
            end if
        end if
        call parse_real_literal(lit_value, value, kind_bytes, ok)
    end subroutine real_literal_kind

    subroutine parse_real_literal(text, value, kind_bytes, ok)
        character(len=*), intent(in) :: text
        real(c_double), intent(out) :: value
        integer, intent(out) :: kind_bytes
        logical, intent(out) :: ok
        character(len=:), allocatable :: body, suffix
        integer :: us, ios

        value = 0.0_c_double
        kind_bytes = 4
        ok = .false.
        body = trim(adjustl(text))
        if (len(body) == 0) return
        us = index(body, '_')
        if (us > 0) then
            suffix = trim(body(us + 1:))
            body = trim(body(:us - 1))
            if (.not. is_integer_text(suffix)) return
            read (suffix, *, iostat=ios) kind_bytes
            if (ios /= 0) return
        else if (index(lowercase_text(body), 'd') > 0) then
            kind_bytes = 8
        end if
        read (body, *, iostat=ios) value
        if (ios /= 0) return
        ok = .true.
    end subroutine parse_real_literal
end submodule session_program_lowering_reject_const_overflow
