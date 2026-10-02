module ffc_suite_group_08
    use ffc_case_test_session_integer_kind64_compiler, only: &
        case_test_session_integer_kind64_compiler
    use ffc_case_test_session_integer_kind_compare_compiler, only: &
        case_test_session_integer_kind_compare_compiler
    use ffc_case_test_session_integer_kind_i8_i16_compiler, only: &
        case_test_session_integer_kind_i8_i16_compiler
    use ffc_case_test_session_integer_pow_compiler, only: &
        case_test_session_integer_pow_compiler
    use ffc_case_test_session_integer_subroutine_compiler, only: &
        case_test_session_integer_subroutine_compiler
    use ffc_case_test_session_integer_variable_compiler, only: &
        case_test_session_integer_variable_compiler
    use ffc_case_test_session_interface_external_call_compiler, only: &
        case_test_session_interface_external_call_compiler
    use ffc_case_test_session_internal_ex_format_compiler, only: &
        case_test_session_internal_ex_format_compiler
    use ffc_case_test_session_internal_read_compiler, only: &
        case_test_session_internal_read_compiler
    use ffc_case_test_session_internal_read_list_directed_compiler, only: &
        case_test_session_internal_read_list_directed_compiler
    use ffc_case_test_session_internal_write_compiler, only: &
        case_test_session_internal_write_compiler
    use ffc_case_test_session_internal_write_compound_compiler, only: &
        case_test_session_internal_write_compound_compiler
    use ffc_case_test_session_intrinsic_dispatch_compiler, only: &
        case_test_session_intrinsic_dispatch_compiler
    use ffc_case_test_session_intrinsics_extra_compiler, only: &
        case_test_session_intrinsics_extra_compiler
    use ffc_case_test_session_io_char_spec_compiler, only: &
        case_test_session_io_char_spec_compiler
    use ffc_case_test_session_io_implied_do_print_compiler, only: &
        case_test_session_io_implied_do_print_compiler
    use ffc_case_test_session_iostat_compiler, only: &
        case_test_session_iostat_compiler
    use ffc_case_test_session_iostat_iomsg_compiler, only: &
        case_test_session_iostat_iomsg_compiler
    use ffc_case_test_session_iso_c_binding_compiler, only: &
        case_test_session_iso_c_binding_compiler
    use ffc_case_test_session_keyword_arguments_compiler, only: &
        case_test_session_keyword_arguments_compiler
    use ffc_case_test_session_large_symbol_count_compiler, only: &
        case_test_session_large_symbol_count_compiler
    use ffc_case_test_session_lazy_array_inference_compiler, only: &
        case_test_session_lazy_array_inference_compiler
    use ffc_case_test_session_lazy_defaults_compiler, only: &
        case_test_session_lazy_defaults_compiler
    use ffc_case_test_session_lazy_derived_inference_compiler, only: &
        case_test_session_lazy_derived_inference_compiler
    use ffc_case_test_session_lazy_monomorph_compiler, only: &
        case_test_session_lazy_monomorph_compiler
    use ffc_case_test_session_lazy_toplevel_function_compiler, only: &
        case_test_session_lazy_toplevel_function_compiler
    use ffc_case_test_session_libm_extra_compiler, only: &
        case_test_session_libm_extra_compiler
    use ffc_case_test_session_literal_kind_parameter_compiler, only: &
        case_test_session_literal_kind_parameter_compiler
    use ffc_case_test_session_logical_allocatable_compiler, only: &
        case_test_session_logical_allocatable_compiler
    use ffc_case_test_session_logical_array_compiler, only: &
        case_test_session_logical_array_compiler
    use ffc_case_test_session_logical_function_print_compiler, only: &
        case_test_session_logical_function_print_compiler
    use ffc_case_test_session_logical_if_compiler, only: &
        case_test_session_logical_if_compiler
    implicit none
    private
    public :: run_group
contains
    subroutine run_group(name, matched)
        character(len=*), intent(in) :: name
        logical, intent(out) :: matched

        matched = .true.
        select case (name)
        case ("test_session_integer_kind64_compiler")
            call case_test_session_integer_kind64_compiler()
        case ("test_session_integer_kind_compare_compiler")
            call case_test_session_integer_kind_compare_compiler()
        case ("test_session_integer_kind_i8_i16_compiler")
            call case_test_session_integer_kind_i8_i16_compiler()
        case ("test_session_integer_pow_compiler")
            call case_test_session_integer_pow_compiler()
        case ("test_session_integer_subroutine_compiler")
            call case_test_session_integer_subroutine_compiler()
        case ("test_session_integer_variable_compiler")
            call case_test_session_integer_variable_compiler()
        case ("test_session_interface_external_call_compiler")
            call case_test_session_interface_external_call_compiler()
        case ("test_session_internal_ex_format_compiler")
            call case_test_session_internal_ex_format_compiler()
        case ("test_session_internal_read_compiler")
            call case_test_session_internal_read_compiler()
        case ("test_session_internal_read_list_directed_compiler")
            call case_test_session_internal_read_list_directed_compiler()
        case ("test_session_internal_write_compiler")
            call case_test_session_internal_write_compiler()
        case ("test_session_internal_write_compound_compiler")
            call case_test_session_internal_write_compound_compiler()
        case ("test_session_intrinsic_dispatch_compiler")
            call case_test_session_intrinsic_dispatch_compiler()
        case ("test_session_intrinsics_extra_compiler")
            call case_test_session_intrinsics_extra_compiler()
        case ("test_session_io_char_spec_compiler")
            call case_test_session_io_char_spec_compiler()
        case ("test_session_io_implied_do_print_compiler")
            call case_test_session_io_implied_do_print_compiler()
        case ("test_session_iostat_compiler")
            call case_test_session_iostat_compiler()
        case ("test_session_iostat_iomsg_compiler")
            call case_test_session_iostat_iomsg_compiler()
        case ("test_session_iso_c_binding_compiler")
            call case_test_session_iso_c_binding_compiler()
        case ("test_session_keyword_arguments_compiler")
            call case_test_session_keyword_arguments_compiler()
        case ("test_session_large_symbol_count_compiler")
            call case_test_session_large_symbol_count_compiler()
        case ("test_session_lazy_array_inference_compiler")
            call case_test_session_lazy_array_inference_compiler()
        case ("test_session_lazy_defaults_compiler")
            call case_test_session_lazy_defaults_compiler()
        case ("test_session_lazy_derived_inference_compiler")
            call case_test_session_lazy_derived_inference_compiler()
        case ("test_session_lazy_monomorph_compiler")
            call case_test_session_lazy_monomorph_compiler()
        case ("test_session_lazy_toplevel_function_compiler")
            call case_test_session_lazy_toplevel_function_compiler()
        case ("test_session_libm_extra_compiler")
            call case_test_session_libm_extra_compiler()
        case ("test_session_literal_kind_parameter_compiler")
            call case_test_session_literal_kind_parameter_compiler()
        case ("test_session_logical_allocatable_compiler")
            call case_test_session_logical_allocatable_compiler()
        case ("test_session_logical_array_compiler")
            call case_test_session_logical_array_compiler()
        case ("test_session_logical_function_print_compiler")
            call case_test_session_logical_function_print_compiler()
        case ("test_session_logical_if_compiler")
            call case_test_session_logical_if_compiler()
        case default
            matched = .false.
        end select
    end subroutine run_group
end module ffc_suite_group_08
