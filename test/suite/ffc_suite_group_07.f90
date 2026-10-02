module ffc_suite_group_07
    use ffc_case_test_session_formatted_print_compiler, only: &
        case_test_session_formatted_print_compiler
    use ffc_case_test_session_formatted_read_al_compiler, only: &
        case_test_session_formatted_read_al_compiler
    use ffc_case_test_session_formatted_read_compiler, only: &
        case_test_session_formatted_read_compiler
    use ffc_case_test_session_function_result_output_item_compiler, only: &
        case_test_session_function_result_output_item_compiler
    use ffc_case_test_session_function_return_compiler, only: &
        case_test_session_function_return_compiler
    use ffc_case_test_session_generic_interface_compiler, only: &
        case_test_session_generic_interface_compiler
    use ffc_case_test_session_goto_compiler, only: &
        case_test_session_goto_compiler
    use ffc_case_test_session_hoisted_contained_name_compiler, only: &
        case_test_session_hoisted_contained_name_compiler
    use ffc_case_test_session_host_shadowed_dummy_compiler, only: &
        case_test_session_host_shadowed_dummy_compiler
    use ffc_case_test_session_huge_intrinsic_compiler, only: &
        case_test_session_huge_intrinsic_compiler
    use ffc_case_test_session_if_condition_general_compiler, only: &
        case_test_session_if_condition_general_compiler
    use ffc_case_test_session_if_merge_character_compiler, only: &
        case_test_session_if_merge_character_compiler
    use ffc_case_test_session_if_merge_compiler, only: &
        case_test_session_if_merge_compiler
    use ffc_case_test_session_if_merge_derived_compiler, only: &
        case_test_session_if_merge_derived_compiler
    use ffc_case_test_session_if_merge_fixed_array_compiler, only: &
        case_test_session_if_merge_fixed_array_compiler
    use ffc_case_test_session_implicit_dimension_data_compiler, only: &
        case_test_session_implicit_dimension_data_compiler
    use ffc_case_test_session_implicit_function_result_compiler, only: &
        case_test_session_implicit_function_result_compiler
    use ffc_case_test_session_imported_module_kind_compiler, only: &
        case_test_session_imported_module_kind_compiler
    use ffc_case_test_session_include_compiler, only: &
        case_test_session_include_compiler
    use ffc_case_test_session_inferred_integer_compiler, only: &
        case_test_session_inferred_integer_compiler
    use ffc_case_test_session_inferred_logical_compiler, only: &
        case_test_session_inferred_logical_compiler
    use ffc_case_test_session_inferred_module_compiler, only: &
        case_test_session_inferred_module_compiler
    use ffc_case_test_session_inferred_real_compiler, only: &
        case_test_session_inferred_real_compiler
    use ffc_case_test_session_inquire_compiler, only: &
        case_test_session_inquire_compiler
    use ffc_case_test_session_inquire_file_expression_compiler, only: &
        case_test_session_inquire_file_expression_compiler
    use ffc_case_test_session_inquiry_fold_compiler, only: &
        case_test_session_inquiry_fold_compiler
    use ffc_case_test_session_integer8_array_reduction_compiler, only: &
        case_test_session_integer8_array_reduction_compiler
    use ffc_case_test_session_integer8_function_compiler, only: &
        case_test_session_integer8_function_compiler
    use ffc_case_test_session_integer8_loop_accumulation_compiler, only: &
        case_test_session_integer8_loop_accumulation_compiler
    use ffc_case_test_session_integer_call_real_operand_compiler, only: &
        case_test_session_integer_call_real_operand_compiler
    use ffc_case_test_session_integer_character_width_compiler, only: &
        case_test_session_integer_character_width_compiler
    use ffc_case_test_session_integer_function_compiler, only: &
        case_test_session_integer_function_compiler
    implicit none
    private
    public :: run_group
contains
    subroutine run_group(name, matched)
        character(len=*), intent(in) :: name
        logical, intent(out) :: matched

        matched = .true.
        select case (name)
        case ("test_session_formatted_print_compiler")
            call case_test_session_formatted_print_compiler()
        case ("test_session_formatted_read_al_compiler")
            call case_test_session_formatted_read_al_compiler()
        case ("test_session_formatted_read_compiler")
            call case_test_session_formatted_read_compiler()
        case ("test_session_function_result_output_item_compiler")
            call case_test_session_function_result_output_item_compiler()
        case ("test_session_function_return_compiler")
            call case_test_session_function_return_compiler()
        case ("test_session_generic_interface_compiler")
            call case_test_session_generic_interface_compiler()
        case ("test_session_goto_compiler")
            call case_test_session_goto_compiler()
        case ("test_session_hoisted_contained_name_compiler")
            call case_test_session_hoisted_contained_name_compiler()
        case ("test_session_host_shadowed_dummy_compiler")
            call case_test_session_host_shadowed_dummy_compiler()
        case ("test_session_huge_intrinsic_compiler")
            call case_test_session_huge_intrinsic_compiler()
        case ("test_session_if_condition_general_compiler")
            call case_test_session_if_condition_general_compiler()
        case ("test_session_if_merge_character_compiler")
            call case_test_session_if_merge_character_compiler()
        case ("test_session_if_merge_compiler")
            call case_test_session_if_merge_compiler()
        case ("test_session_if_merge_derived_compiler")
            call case_test_session_if_merge_derived_compiler()
        case ("test_session_if_merge_fixed_array_compiler")
            call case_test_session_if_merge_fixed_array_compiler()
        case ("test_session_implicit_dimension_data_compiler")
            call case_test_session_implicit_dimension_data_compiler()
        case ("test_session_implicit_function_result_compiler")
            call case_test_session_implicit_function_result_compiler()
        case ("test_session_imported_module_kind_compiler")
            call case_test_session_imported_module_kind_compiler()
        case ("test_session_include_compiler")
            call case_test_session_include_compiler()
        case ("test_session_inferred_integer_compiler")
            call case_test_session_inferred_integer_compiler()
        case ("test_session_inferred_logical_compiler")
            call case_test_session_inferred_logical_compiler()
        case ("test_session_inferred_module_compiler")
            call case_test_session_inferred_module_compiler()
        case ("test_session_inferred_real_compiler")
            call case_test_session_inferred_real_compiler()
        case ("test_session_inquire_compiler")
            call case_test_session_inquire_compiler()
        case ("test_session_inquire_file_expression_compiler")
            call case_test_session_inquire_file_expression_compiler()
        case ("test_session_inquiry_fold_compiler")
            call case_test_session_inquiry_fold_compiler()
        case ("test_session_integer8_array_reduction_compiler")
            call case_test_session_integer8_array_reduction_compiler()
        case ("test_session_integer8_function_compiler")
            call case_test_session_integer8_function_compiler()
        case ("test_session_integer8_loop_accumulation_compiler")
            call case_test_session_integer8_loop_accumulation_compiler()
        case ("test_session_integer_call_real_operand_compiler")
            call case_test_session_integer_call_real_operand_compiler()
        case ("test_session_integer_character_width_compiler")
            call case_test_session_integer_character_width_compiler()
        case ("test_session_integer_function_compiler")
            call case_test_session_integer_function_compiler()
        case default
            matched = .false.
        end select
    end subroutine run_group
end module ffc_suite_group_07
