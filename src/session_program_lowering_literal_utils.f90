submodule (session_program_lowering_impl) session_program_lowering_literal_utils
    use session_program_lowering_literal_utils_order
contains

    module procedure real_intrinsic_is_f64
        !! Result kind of a REAL(A [,KIND]) conversion (F2018 16.9.160). With a
        !! KIND selector the width follows KIND; a literal 8 is f64 and a literal
        !! 4 is f32, while an unfoldable selector (named constant, kind() inquiry)
        !! stays f64 to preserve the historical conservative width. With no KIND,
        !! the result is default real (f32) for integer or real A, and keeps the
        !! argument kind only when A is complex, so real(z) on a complex(8) is the
        !! one no-KIND case that is f64.
        integer(c_int64_t) :: kind_value
        character(len=:), allocatable :: kind_error

        is_f64 = .false.
        if (.not. allocated(node%arg_indices)) return
        if (size(node%arg_indices) >= 2) then
            call parse_i32_constant(arena, node%arg_indices(2), kind_value, &
                                    'real kind', kind_error)
            if (len_trim(kind_error) > 0) then
                is_f64 = .true.
            else
                is_f64 = kind_value == 8_c_int64_t
            end if
            return
        end if
        is_f64 = is_complex_valued(arena, node%arg_indices(1), context, VALUE_C8)
    end procedure real_intrinsic_is_f64

    module procedure real_conversion_intrinsic
        !! True for the AINT/ANINT real conversions, which accept an optional
        !! KIND selector that fixes the result width.

        is_conversion = same_name(name, 'aint') .or. same_name(name, 'anint')
    end procedure real_conversion_intrinsic

    module procedure real_conversion_kind_is_f64
        !! Result width of an AINT/ANINT KIND selector. A foldable 8 selects
        !! f64; anything unfoldable keeps the conservative wide result, matching
        !! how REAL(A, KIND) treats named constants.
        integer(c_int64_t) :: kind_value
        character(len=:), allocatable :: kind_error

        call parse_i32_constant(arena, kind_index, kind_value, 'real kind', &
                                kind_error)
        if (len_trim(kind_error) > 0) then
            is_f64 = .true.
        else
            is_f64 = kind_value == 8_c_int64_t
        end if
    end procedure real_conversion_kind_is_f64

    module procedure is_real_array_reduction
        !! True when node is a sum/product/maxval/minval/norm2 call whose first
        !! argument is a fixed-size array of value kind vk.
        character(len=:), allocatable :: arg_name, err
        integer :: sym, elem_kind, rank, extent
        logical :: alloc_ok

        ok = .false.
        if (.not. allocated(node%name)) return
        select case (trim(node%name))
        case ('sum', 'product', 'maxval', 'minval', 'norm2')
        case default
            return
        end select
        if (.not. allocated(node%arg_indices)) return
        if (size(node%arg_indices) /= 1) return
        if (node_exists(arena, node%arg_indices(1))) then
            ! sum(a(lo:hi)) and friends: the section reduces over its base
            ! array's element kind, so gate on that base's value kind.
            select type (arg => arena%entries(node%arg_indices(1))%node)
            type is (array_slice_node)
                call get_identifier_name(arena, arg%array_index, arg_name, err)
                if (len_trim(err) > 0) return
                sym = find_symbol_compat(context, arg_name)
                if (sym <= 0) return
                ok = context%symbols(sym)%is_array .and. &
                     context%symbols(sym)%value_kind == vk
                return
            end select
            ! sum(f()): a bare call to a contained function returning an
            ! allocatable array reduces over that result's element kind.
            if (is_alloc_array_result_call(arena, node%arg_indices(1), context)) then
                call alloc_array_result_call_info(arena, node%arg_indices(1), &
                    context, elem_kind, rank, alloc_ok)
                ok = alloc_ok .and. rank == 1 .and. elem_kind == vk
                return
            end if
            ! sum(a + b) and friends: a binary-op array expression reduces
            ! over its own element kind, determined recursively.
            if (is_binary_op(arena, node%arg_indices(1))) then
                ok = reduction_arg_extent(arena, node%arg_indices(1), context, &
                                          vk, extent)
                return
            end if
            if (reduction_expression_is_abs_call(arena, node%arg_indices(1))) then
                ok = reduction_expression_has_kind(arena, node%arg_indices(1), &
                                                   context, vk)
                return
            end if
        end if
        call get_identifier_name(arena, node%arg_indices(1), arg_name, err)
        if (len_trim(err) > 0) return
        sym = find_symbol_compat(context, arg_name)
        if (sym <= 0) return
        ! A 1-D allocatable reduces over its element kind just like a fixed
        ! array, so the result print kind follows the allocatable element kind.
        ok = (context%symbols(sym)%is_array .or. &
              context%symbols(sym)%is_allocatable) .and. &
             context%symbols(sym)%value_kind == vk
    end procedure is_real_array_reduction

    module procedure is_real_inquiry_intrinsic
        !! True for the real numeric-model inquiry intrinsics whose result is a
        !! scalar real constant determined solely by the argument's kind.

        select case (trim(name))
        case ('tiny', 'huge', 'epsilon')
            ok = .true.
        case default
            ok = .false.
        end select
    end procedure is_real_inquiry_intrinsic

    module procedure inquiry_arg_real_kind
        !! Real kind (4 or 8) of a tiny/huge/epsilon argument, or -1 when the
        !! argument is not a real literal, real array constructor, or real
        !! variable of a supported kind.
        integer :: arg_idx, sym
        character(len=:), allocatable :: id_name, err, lv, lt, le

        kind_num = -1
        if (.not. allocated(node%arg_indices)) return
        if (size(node%arg_indices) < 1) return
        arg_idx = node%arg_indices(1)
        if (.not. node_exists(arena, arg_idx)) return
        select type (arg => arena%entries(arg_idx)%node)
        type is (array_literal_node)
            if (.not. allocated(arg%element_indices)) return
            if (size(arg%element_indices) < 1) return
            arg_idx = arg%element_indices(1)
        end select
        if (.not. node_exists(arena, arg_idx)) return
        if (is_real_literal(arena, arg_idx)) then
            call get_literal_info(arena, arg_idx, lv, lt, le)
            if (literal_is_f64(lv, context, arg_idx)) then
                kind_num = 8
            else
                kind_num = 4
            end if
            return
        end if
        if (is_identifier(arena, arg_idx)) then
            call get_identifier_name(arena, arg_idx, id_name, err)
            if (len_trim(err) > 0) return
            sym = find_symbol_compat(context, id_name)
            if (sym <= 0) return
            select case (context%symbols(sym)%value_kind)
            case (VALUE_F32)
                kind_num = 4
            case (VALUE_F64)
                kind_num = 8
            end select
        end if
    end procedure inquiry_arg_real_kind

    module procedure is_real_dot_product
        !! True when node is a dot_product call whose two arguments are
        !! rank-1 arrays of value kind vk.
        character(len=:), allocatable :: a_name, b_name, err
        integer :: a_sym, b_sym

        ok = .false.
        if (.not. allocated(node%name)) return
        if (trim(node%name) /= 'dot_product') return
        if (.not. allocated(node%arg_indices)) return
        if (size(node%arg_indices) /= 2) return
        call get_identifier_name(arena, node%arg_indices(1), a_name, err)
        if (len_trim(err) > 0) return
        call get_identifier_name(arena, node%arg_indices(2), b_name, err)
        if (len_trim(err) > 0) return
        a_sym = find_symbol_compat(context, a_name)
        b_sym = find_symbol_compat(context, b_name)
        if (a_sym <= 0 .or. b_sym <= 0) return
        ok = context%symbols(a_sym)%is_array .and. &
             context%symbols(a_sym)%array_rank == 1 .and. &
             context%symbols(a_sym)%value_kind == vk .and. &
             context%symbols(b_sym)%is_array .and. &
             context%symbols(b_sym)%array_rank == 1 .and. &
             context%symbols(b_sym)%value_kind == vk
    end procedure is_real_dot_product

    module procedure real_opcode

        call set_empty(error_msg)
        select case (trim(source_op))
        case ('+')
            opcode = 18
        case ('-')
            opcode = 19
        case ('*')
            opcode = 20
        case ('/')
            opcode = 21
        case default
            call unsupported_feature_error('real operator', line, column, &
                                           'direct LIRIC session supports '// &
                                           '+, -, *, and / for real expressions', &
                                           error_msg)
        end select
    end procedure real_opcode

    module procedure is_real_literal
        character(len=:), allocatable :: value, literal_type, err

        is_real_literal = .false.
        call get_literal_info(arena, node_index, value, literal_type, err)
        if (len_trim(err) > 0) return
        ! A logical literal (.true./.false.) contains a '.' but is not real.
        if (trim(literal_type) == 'logical') return
        if (trim(value) == '.true.' .or. trim(value) == '.false.') return
        ! A BOZ constant (B'..'/O'..'/Z'..'/X'..' or postfix '..'B/'..'O/'..'Z)
        ! is always integer-valued, even when its hex digits spell a letter
        ! ('e', 'd') that the substring heuristic below would mistake for a
        ! real exponent marker.
        if (is_boz_literal_text(value)) return
        is_real_literal = trim(literal_type) == 'real' .or. &
                          index(value, '.') > 0 .or. index(value, 'e') > 0 .or. &
                          index(value, 'E') > 0
    end procedure is_real_literal

    module procedure is_boz_literal_text
        !! True when text is a BOZ-literal-constant spelling: a b/o/z/x radix
        !! designator adjacent to a quoted digit string, prefix or postfix.
        character(len=:), allocatable :: trimmed, lo
        integer :: n

        is_boz = .false.
        trimmed = trim(adjustl(text))
        n = len(trimmed)
        if (n < 3) return
        lo = lowercase_text(trimmed)
        if (trimmed(1:1) == "'" .or. trimmed(1:1) == '"') then
            is_boz = is_boz_designator(lo(n:n))
        else
            is_boz = is_boz_designator(lo(1:1)) .and. &
                     (trimmed(2:2) == "'" .or. trimmed(2:2) == '"')
        end if
    end procedure is_boz_literal_text

    module procedure node_is_boz_literal
        !! True when the arena node is a literal whose spelling is a
        !! BOZ-literal-constant. Non-literal nodes and missing nodes are not.
        character(len=:), allocatable :: value, literal_type, err

        is_boz = .false.
        if (node_index <= 0) return
        if (.not. node_exists(arena, node_index)) return
        call get_literal_info(arena, node_index, value, literal_type, err)
        if (len_trim(err) > 0) return
        is_boz = is_boz_literal_text(value)
    end procedure node_is_boz_literal

    module procedure is_boz_designator

        is_boz_designator = c == 'b' .or. c == 'o' .or. c == 'z' .or. c == 'x'
    end procedure is_boz_designator

    module procedure boz_bits_i32
        !! Two's-complement-safe narrowing of an i64 bit pattern to its low
        !! 32 bits, for reinterpreting a BOZ literal's magnitude as the raw
        !! bits of an f32 value (REAL()'s BOZ-argument bit-transfer rule).
        integer(c_int64_t) :: masked
        integer(c_int64_t), parameter :: mask32 = int(z'00000000FFFFFFFF', c_int64_t)
        integer(c_int64_t), parameter :: sign_bit = int(z'0000000080000000', c_int64_t)
        integer(c_int64_t), parameter :: wrap = int(z'0000000100000000', c_int64_t)

        masked = iand(v, mask32)
        if (masked >= sign_bit) masked = masked - wrap
        bits = int(masked, c_int32_t)
    end procedure boz_bits_i32

    module procedure lower_boz_real_bits
        !! REAL()/DBLE() reinterpret a BOZ-literal-constant argument's bit
        !! pattern directly as the result kind's representation (F2008
        !! 13.7.128-129), unlike an ordinary integer argument which converts
        !! numerically. handled is false (with no error) for any other
        !! argument form, so the caller falls back to normal lowering.
        character(len=:), allocatable :: lit_value, lit_type
        integer(c_int64_t) :: raw_bits

        handled = .false.
        call set_empty(error_msg)
        if (.not. is_literal(arena, node_index)) return
        call get_literal_info(arena, node_index, lit_value, lit_type, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. is_boz_literal_text(lit_value)) then
            call set_empty(error_msg)
            return
        end if

        call parse_i32_literal(lit_value, raw_bits, error_msg)
        if (len_trim(error_msg) > 0) return

        if (want_f64) then
            value = liric_f64_immediate(context%session, &
                transfer(raw_bits, 1.0_c_double))
        else
            value = liric_f32_immediate(context%session, &
                transfer(boz_bits_i32(raw_bits), 1.0_c_float))
        end if
        handled = .true.
    end procedure lower_boz_real_bits

    module procedure is_character_literal
        character(len=:), allocatable :: value, literal_type, err

        is_character_literal = .false.
        call get_literal_info(arena, node_index, value, literal_type, err)
        if (len_trim(err) > 0) return
        is_character_literal = trim(literal_type) == 'character' .or. &
                               starts_with_quote(value)
    end procedure is_character_literal

    module procedure is_logical_literal
        character(len=:), allocatable :: value, literal_type, err
        character(len=:), allocatable :: lowered_value

        is_logical_literal = .false.
        call get_literal_info(arena, node_index, value, literal_type, err)
        if (len_trim(err) > 0) return
        lowered_value = lowercase_text(trim(value))
        is_logical_literal = lowered_value == '.true.' .or. &
                             lowered_value == '.false.' .or. &
                             index(lowered_value, '.true._') == 1 .or. &
                             index(lowered_value, '.false._') == 1 .or. &
                             lowered_value == 'true' .or. &
                             lowered_value == 'false' .or. &
                             trim(literal_type) == 'logical'
    end procedure is_logical_literal

    module procedure starts_with_quote

        ! .and. does not short-circuit, so the length guard cannot protect the
        ! substring reference in the same expression.
        starts_with_quote = .false.
        if (len_trim(text) < 2) return
        starts_with_quote = text(1:1) == '"' .or. text(1:1) == "'"
    end procedure starts_with_quote

    module procedure strip_literal_quotes
        integer :: text_len

        ! .and. does not short-circuit, so the length guard is a separate test:
        ! an empty text would otherwise index text(1:1) out of bounds.
        text_len = len_trim(text)
        value = trim(text)
        if (text_len < 2) return
        if (text(1:1) == '"' .and. text(text_len:text_len) == '"') then
            value = text(2:text_len - 1)
        else if (text(1:1) == "'" .and. text(text_len:text_len) == "'") then
            value = text(2:text_len - 1)
        end if
    end procedure strip_literal_quotes

    module procedure logical_i32_value
        character(len=:), allocatable :: lowered

        lowered = lowercase_text(trim(adjustl(text)))
        if (lowered == '.true.' .or. lowered == 'true' .or. &
            index(lowered, '.true._') == 1) then
            value = 1_c_int64_t
        else
            value = 0_c_int64_t
        end if
    end procedure logical_i32_value

    module procedure literal_is_f64
        ! Returns true when a real literal is explicitly f64 — i.e. its numeric
        ! part carries a 'd'/'D' exponent, or it carries a _8 / _real64 /
        ! _c_double / _c_long_double kind suffix, or a suffix naming a
        ! compile-time integer PARAMETER whose folded value is 8 (e.g.
        ! `1.0_dp` with `integer, parameter :: dp = kind(1.0d0)`). A bare
        ! literal such as "2.5" or "3.0e0" is default real (f32). The exponent
        ! check is scoped to the numeric part so a suffix that merely contains
        ! the letter 'd' (e.g. a BOZ digit string reaching this function by
        ! mistake) cannot misfire.
        character(len=:), allocatable :: lo, numeric_part, suffix
        integer :: uscore

        lo = lowercase_text(trim(text))
        uscore = scan(lo, '_')
        if (uscore > 1) then
            if (scan(lo(uscore - 1:uscore - 1), '0123456789.de') > 0) then
                numeric_part = lo(1:uscore - 1)
                suffix = lo(uscore + 1:)
            else
                numeric_part = lo
                call set_empty(suffix)
            end if
        else
            numeric_part = lo
            call set_empty(suffix)
        end if

        if (scan(numeric_part, 'd') > 0) then
            literal_is_f64 = .true.
            return
        end if

        select case (trim(suffix))
        case ('8', 'real64', 'c_double', 'c_long_double')
            literal_is_f64 = .true.
        case ('')
            literal_is_f64 = .false.
        case default
            literal_is_f64 = named_kind_suffix_is_f64(suffix, context, &
                                                      reference_index)
        end select
    end procedure literal_is_f64

    module procedure named_kind_suffix_is_f64
        integer :: sym
        integer(c_int64_t) :: folded
        character(len=:), allocatable :: fold_error
        logical :: found

        if (present(reference_index)) then
            call fold_scoped_i32_name(context%arena, context, reference_index, &
                                      suffix, folded, found, fold_error)
            if (len_trim(fold_error) == 0) then
                is_f64 = folded == 8_c_int64_t
                return
            end if
            if (found) then
                is_f64 = .false.
                return
            end if
        end if
        sym = find_symbol_compat(context, suffix)
        if (sym > 0) then
            if (context%symbols(sym)%has_i32_constant) then
                is_f64 = context%symbols(sym)%i32_constant == 8_c_int64_t
                return
            end if
        end if
        select case (trim(suffix))
        case ('dp', 'wp')
            is_f64 = .true.
        case default
            is_f64 = .false.
        end select
    end procedure named_kind_suffix_is_f64

    module procedure parse_f64_literal
        character(len=:), allocatable :: clean
        integer :: io_stat, uscore
        integer(c_int64_t) :: int_value
        real(c_float) :: single_value

        ! Strip the kind suffix (_8, _dp, _real64, _c_double, etc.) which Fortran
        ! internal reads do not accept. The suffix begins at the first underscore
        ! that follows a digit, dot, or exponent letter; a kind name such as
        ! c_double contains further underscores that must not be the split point.
        clean = trim(text)
        uscore = scan(clean, '_')
        if (uscore > 1) then
            if (scan(clean(uscore - 1:uscore - 1), '0123456789.dDeE') > 0) then
                clean = clean(1:uscore - 1)
            end if
        end if

        ! A literal without an explicit f64 marker is default real (f32): it
        ! rounds to single precision first, and only that rounded value widens
        ! to f64, matching gfortran (e.g. 1.05 -> 1.0499999523162842d0, not the
        ! double-precision-accurate 1.05d0).
        if (literal_is_f64(text, context, reference_index)) then
            read (clean, *, iostat=io_stat) value
        else
            read (clean, *, iostat=io_stat) single_value
            value = real(single_value, c_double)
        end if
        if (io_stat == 0) then
            call set_empty(error_msg)
            return
        end if

        ! If real parsing failed, try parsing as a BOZ literal (e.g. Z'ABC')
        ! and convert the integer to real. This handles real(b'010101') expressions.
        call parse_i32_literal(text, int_value, error_msg)
        if (len_trim(error_msg) == 0) then
            value = real(int_value, c_double)
            return
        end if

        error_msg = 'invalid real literal for direct LIRIC session: '// &
                    trim(text)
    end procedure parse_f64_literal

end submodule session_program_lowering_literal_utils
