submodule (liric_session_format_bindings) liric_session_format_boz
    implicit none

contains

    module procedure emit_boz_write_call
    type(lr_operand_desc_t) :: args(6)

    args(1) = i32_immediate(session, 6_c_int64_t)
    args(2) = i32_immediate(session, int(radix, c_int64_t))
    args(3) = i32_immediate(session, int(width, c_int64_t))
    args(4) = i32_immediate(session, int(min_digits, c_int64_t))
    args(5) = i32_immediate(session, int(bits, c_int64_t))
    args(6) = value
    ok = emit_c_call(session, '_ffc_write_boz', args, &
        lr_type_i32_s(session%handle), 6_c_int32_t, &
        c_false, error_msg)
    end procedure emit_boz_write_call

end submodule liric_session_format_boz
