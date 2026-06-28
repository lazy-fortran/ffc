module liric_session_io_bindings
    use liric_session_format_bindings, only: prepare_liric_print_runtime, &
        LR_OP_FSUB
    use liric_session_io_emission_bindings, only: emit_liric_f32_binary, &
        emit_liric_i32_to_f32, &
        emit_liric_f32_to_i32, &
        emit_liric_f32_to_f64, &
        emit_liric_print_f32, &
        emit_liric_print_f32_value, &
        emit_liric_f64_binary, &
        materialize_liric_string, &
        emit_liric_i32_to_f64, &
        emit_liric_f64_to_i32, &
        emit_liric_char_byte_zext, &
        emit_liric_i32_to_i64, &
        emit_liric_store_char_byte, &
        emit_liric_print_f64, &
        emit_liric_print_f64_value, &
        emit_liric_print_i32, &
        emit_liric_print_i32_value, &
        emit_liric_print_i64, &
        emit_liric_print_i64_value, &
        emit_liric_print_newline, &
        emit_liric_print_space, &
        emit_liric_print_string, &
        emit_liric_print_string_operand, &
        emit_liric_print_string_operand_value, &
        emit_liric_print_string_value, &
        liric_f32_immediate, &
        liric_f64_immediate, &
        emit_liric_i8_to_i32, &
        emit_liric_i16_to_i32, &
        emit_liric_print_i8, &
        emit_liric_print_i8_value, &
        emit_liric_print_i16, &
        emit_liric_print_i16_value
    implicit none
    private

    public :: emit_liric_f32_binary
    public :: emit_liric_i32_to_f32
    public :: emit_liric_f32_to_i32
    public :: emit_liric_f32_to_f64
    public :: emit_liric_print_f32
    public :: emit_liric_print_f32_value
    public :: emit_liric_f64_binary
    public :: emit_liric_i32_to_f64
    public :: emit_liric_f64_to_i32
    public :: emit_liric_char_byte_zext
    public :: emit_liric_i32_to_i64
    public :: emit_liric_store_char_byte
    public :: emit_liric_print_f64
    public :: emit_liric_print_f64_value
    public :: emit_liric_print_i32
    public :: emit_liric_print_i32_value
    public :: emit_liric_print_i64
    public :: emit_liric_print_i64_value
    public :: emit_liric_print_newline
    public :: emit_liric_print_space
    public :: emit_liric_print_string
    public :: emit_liric_print_string_operand
    public :: emit_liric_print_string_operand_value
    public :: emit_liric_print_string_value
    public :: liric_f32_immediate
    public :: liric_f64_immediate
    public :: materialize_liric_string
    public :: emit_liric_i8_to_i32
    public :: emit_liric_i16_to_i32
    public :: emit_liric_print_i8
    public :: emit_liric_print_i8_value
    public :: emit_liric_print_i16
    public :: emit_liric_print_i16_value
    public :: prepare_liric_print_runtime
    public :: LR_OP_FSUB

end module liric_session_io_bindings
