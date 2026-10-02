module ffc_suite_group_10
    use ffc_case_test_session_multi_allocate_compiler, only: &
        case_test_session_multi_allocate_compiler
    use ffc_case_test_session_multi_declaration_compiler, only: &
        case_test_session_multi_declaration_compiler
    use ffc_case_test_session_multi_value_print_compiler, only: &
        case_test_session_multi_value_print_compiler
    use ffc_case_test_session_namelist_compiler, only: &
        case_test_session_namelist_compiler
    use ffc_case_test_session_namelist_module_compiler, only: &
        case_test_session_namelist_module_compiler
    use ffc_case_test_session_namelist_read_compiler, only: &
        case_test_session_namelist_read_compiler
    use ffc_case_test_session_narrow_int_array_compiler, only: &
        case_test_session_narrow_int_array_compiler
    use ffc_case_test_session_nested_character_substring_compiler, only: &
        case_test_session_nested_character_substring_compiler
    use ffc_case_test_session_nested_do_tail_compiler, only: &
        case_test_session_nested_do_tail_compiler
    use ffc_case_test_session_nested_implied_do_compiler, only: &
        case_test_session_nested_implied_do_compiler
    use ffc_case_test_session_non_integer_procedure_compiler, only: &
        case_test_session_non_integer_procedure_compiler
    use ffc_case_test_session_norm2_compiler, only: &
        case_test_session_norm2_compiler
    use ffc_case_test_session_open_close_file_compiler, only: &
        case_test_session_open_close_file_compiler
    use ffc_case_test_session_open_file_variable_compiler, only: &
        case_test_session_open_file_variable_compiler
    use ffc_case_test_session_open_positional_unit_compiler, only: &
        case_test_session_open_positional_unit_compiler
    use ffc_case_test_session_open_status_variable_compiler, only: &
        case_test_session_open_status_variable_compiler
    use ffc_case_test_session_operator_overload_compiler, only: &
        case_test_session_operator_overload_compiler
    use ffc_case_test_session_optional_args_compiler, only: &
        case_test_session_optional_args_compiler
    use ffc_case_test_session_optional_scalar_kinds_compiler, only: &
        case_test_session_optional_scalar_kinds_compiler
    use ffc_case_test_session_pack_unpack_compiler, only: &
        case_test_session_pack_unpack_compiler
    use ffc_case_test_session_pause_compiler, only: &
        case_test_session_pause_compiler
    use ffc_case_test_session_pdt_constant_compiler, only: &
        case_test_session_pdt_constant_compiler
    use ffc_case_test_session_pdt_inheritance_compiler, only: &
        case_test_session_pdt_inheritance_compiler
    use ffc_case_test_session_plain_derived_value_compiler, only: &
        case_test_session_plain_derived_value_compiler
    use ffc_case_test_session_pointer_array_compiler, only: &
        case_test_session_pointer_array_compiler
    use ffc_case_test_session_pointer_array_descriptor_compiler, only: &
        case_test_session_pointer_array_descriptor_compiler
    use ffc_case_test_session_pointer_array_rank2_compiler, only: &
        case_test_session_pointer_array_rank2_compiler
    use ffc_case_test_session_pointer_associated2_compiler, only: &
        case_test_session_pointer_associated2_compiler
    use ffc_case_6060197a4bcf6129f05a0604, only: &
        case_test_session_pointer_derived_component_rank234_compiler
    use ffc_case_test_session_pointer_function_result_compiler, only: &
        case_test_session_pointer_function_result_compiler
    use ffc_case_test_session_pointer_intent_in_target_arg_compiler, only: &
        case_test_session_pointer_intent_in_target_arg_compiler
    use ffc_case_test_session_pointer_proc_compiler, only: &
        case_test_session_pointer_proc_compiler
    implicit none
    private
    public :: run_group
contains
    subroutine run_group(name, matched)
        character(len=*), intent(in) :: name
        logical, intent(out) :: matched

        matched = .true.
        select case (name)
        case ("test_session_multi_allocate_compiler")
            call case_test_session_multi_allocate_compiler()
        case ("test_session_multi_declaration_compiler")
            call case_test_session_multi_declaration_compiler()
        case ("test_session_multi_value_print_compiler")
            call case_test_session_multi_value_print_compiler()
        case ("test_session_namelist_compiler")
            call case_test_session_namelist_compiler()
        case ("test_session_namelist_module_compiler")
            call case_test_session_namelist_module_compiler()
        case ("test_session_namelist_read_compiler")
            call case_test_session_namelist_read_compiler()
        case ("test_session_narrow_int_array_compiler")
            call case_test_session_narrow_int_array_compiler()
        case ("test_session_nested_character_substring_compiler")
            call case_test_session_nested_character_substring_compiler()
        case ("test_session_nested_do_tail_compiler")
            call case_test_session_nested_do_tail_compiler()
        case ("test_session_nested_implied_do_compiler")
            call case_test_session_nested_implied_do_compiler()
        case ("test_session_non_integer_procedure_compiler")
            call case_test_session_non_integer_procedure_compiler()
        case ("test_session_norm2_compiler")
            call case_test_session_norm2_compiler()
        case ("test_session_open_close_file_compiler")
            call case_test_session_open_close_file_compiler()
        case ("test_session_open_file_variable_compiler")
            call case_test_session_open_file_variable_compiler()
        case ("test_session_open_positional_unit_compiler")
            call case_test_session_open_positional_unit_compiler()
        case ("test_session_open_status_variable_compiler")
            call case_test_session_open_status_variable_compiler()
        case ("test_session_operator_overload_compiler")
            call case_test_session_operator_overload_compiler()
        case ("test_session_optional_args_compiler")
            call case_test_session_optional_args_compiler()
        case ("test_session_optional_scalar_kinds_compiler")
            call case_test_session_optional_scalar_kinds_compiler()
        case ("test_session_pack_unpack_compiler")
            call case_test_session_pack_unpack_compiler()
        case ("test_session_pause_compiler")
            call case_test_session_pause_compiler()
        case ("test_session_pdt_constant_compiler")
            call case_test_session_pdt_constant_compiler()
        case ("test_session_pdt_inheritance_compiler")
            call case_test_session_pdt_inheritance_compiler()
        case ("test_session_plain_derived_value_compiler")
            call case_test_session_plain_derived_value_compiler()
        case ("test_session_pointer_array_compiler")
            call case_test_session_pointer_array_compiler()
        case ("test_session_pointer_array_descriptor_compiler")
            call case_test_session_pointer_array_descriptor_compiler()
        case ("test_session_pointer_array_rank2_compiler")
            call case_test_session_pointer_array_rank2_compiler()
        case ("test_session_pointer_associated2_compiler")
            call case_test_session_pointer_associated2_compiler()
        case ("test_session_pointer_derived_component_rank234_compiler")
            call case_test_session_pointer_derived_component_rank234_compiler()
        case ("test_session_pointer_function_result_compiler")
            call case_test_session_pointer_function_result_compiler()
        case ("test_session_pointer_intent_in_target_arg_compiler")
            call case_test_session_pointer_intent_in_target_arg_compiler()
        case ("test_session_pointer_proc_compiler")
            call case_test_session_pointer_proc_compiler()
        case default
            matched = .false.
        end select
    end subroutine run_group
end module ffc_suite_group_10
