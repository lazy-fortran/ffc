submodule (session_program_lowering_impl) print_boz
    use liric_session_format_bindings, only: emit_boz_write_call
    implicit none

contains

    module procedure lower_boz_descriptor
    integer :: width, min_digits, radix, i

    call set_empty(error_msg)
    exhausted = .false.
    call parse_boz_field(format_body, pos, kind_char, width, min_digits, error_msg)
    if (len_trim(error_msg) > 0) return
    select case (kind_char)
    case ('B')
        radix = 2
    case ('O')
        radix = 8
    case ('Z')
        radix = 16
    end select
    do i = 1, repeat_count
        if (.not. allocated(node%expression_indices)) then
            exhausted = .true.
            return
        end if
        if (item_index > size(node%expression_indices)) then
            exhausted = .true.
            return
        end if
        call lower_boz_item(arena, node%expression_indices(item_index), &
            context, radix, width, min_digits, error_msg)
        if (len_trim(error_msg) > 0) return
        item_index = item_index + 1
    end do
    end procedure lower_boz_descriptor

    module procedure lower_boz_item
    type(lr_operand_desc_t) :: value, wide_value
    integer :: value_kind, bits

    if (.not. node_exists(arena, node_index)) then
        error_msg = 'B/O/Z item does not reference an AST node'
        return
    end if
    value_kind = expression_value_kind(arena, node_index, context, VALUE_I32)
    if (arena%entries(node_index)%node%resolved_type_found) then
        if (arena%entries(node_index)%node%resolved_type_kind /= TINT) then
            error_msg = 'B/O/Z edit descriptors require an integer item'
            return
        end if
        bits = 8 * arena%entries(node_index)%node%resolved_kind_value
    else
        select case (value_kind)
        case (VALUE_I8); bits = 8
        case (VALUE_I16); bits = 16
        case (VALUE_I32); bits = 32
        case (VALUE_I64); bits = 64
        case default; bits = 0
        end select
    end if
    select case (bits)
    case (8, 16, 32, 64)
    case default
        error_msg = 'B/O/Z item has an unsupported integer kind'
        return
    end select
    if (bits == 64) then
        call lower_i64_expression(arena, node_index, context, wide_value, error_msg)
    else
        call lower_i32_expression(arena, node_index, context, value, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_liric_i32_to_i64(context%session, value, wide_value, &
            error_msg)) return
    end if
    if (len_trim(error_msg) > 0) return
    if (.not. emit_boz_write_call(context%session, wide_value, radix, width, &
        min_digits, bits, error_msg)) return
    end procedure lower_boz_item

    subroutine parse_boz_field(format_body, pos, kind_char, width, min_digits, &
            error_msg)
        character(len=*), intent(in) :: format_body
        integer, intent(inout) :: pos
        character, intent(in) :: kind_char
        integer, intent(out) :: width, min_digits
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: digits

        call parse_decimal_digits(format_body, pos, digits)
        if (len(digits) == 0) then
            error_msg = kind_char//' edit descriptor requires width'
            return
        end if
        call read_decimal_value(digits, width, error_msg)
        if (len_trim(error_msg) > 0) return
        min_digits = 1
        if (pos <= len_trim(format_body)) then
            if (format_body(pos:pos) == '.') then
                pos = pos + 1
                call parse_decimal_digits(format_body, pos, digits)
                if (len(digits) == 0) then
                    error_msg = kind_char//' edit descriptor requires minimum digits'
                    return
                end if
                call read_decimal_value(digits, min_digits, error_msg)
                if (len_trim(error_msg) > 0) return
            end if
        end if
        if (width > 0) then
            if (min_digits > width) then
                error_msg = kind_char//' minimum digits exceed field width'
                return
            end if
        end if
    end subroutine parse_boz_field

end submodule print_boz
