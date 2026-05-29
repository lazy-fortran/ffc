module liric_session_io_bindings
    use liric_session_format_bindings, only: prepare_liric_print_runtime, &
                                              LR_OP_FSUB
    use liric_session_io_emission_bindings, only: emit_liric_f64_binary, &
                                                  materialize_liric_string, &
                                                  emit_liric_i32_to_f64, &
                                                  emit_liric_f64_to_i32, &
                                                  emit_liric_char_byte_zext, &
                                                  emit_liric_print_f64, &
                                                  emit_liric_print_f64_value, &
                                                  emit_liric_print_i32, &
                                                  emit_liric_print_i32_value, &
                                                  emit_liric_print_newline, &
                                                  emit_liric_print_space, &
                                                  emit_liric_print_string, &
                                                  emit_liric_print_string_operand, &
                                                  emit_liric_print_string_operand_value, &
                                                  emit_liric_print_string_value, &
                                                  liric_f64_immediate
    implicit none
    private

    public :: emit_liric_f64_binary
    public :: emit_liric_i32_to_f64
    public :: emit_liric_f64_to_i32
    public :: emit_liric_char_byte_zext
    public :: emit_liric_print_f64
    public :: emit_liric_print_f64_value
    public :: emit_liric_print_i32
    public :: emit_liric_print_i32_value
    public :: emit_liric_print_newline
    public :: emit_liric_print_space
    public :: emit_liric_print_string
    public :: emit_liric_print_string_operand
    public :: emit_liric_print_string_operand_value
    public :: emit_liric_print_string_value
    public :: liric_f64_immediate
    public :: materialize_liric_string
    public :: prepare_liric_print_runtime
    public :: LR_OP_FSUB

end module liric_session_io_bindings
