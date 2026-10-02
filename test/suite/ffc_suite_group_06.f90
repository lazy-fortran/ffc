module ffc_suite_group_06
    use ffc_case_test_session_dimension_statement_compiler, only: &
        case_test_session_dimension_statement_compiler
    use ffc_case_test_session_do_while_compiler, only: &
        case_test_session_do_while_compiler
    use ffc_case_test_session_do_while_conditional_stop_compiler, only: &
        case_test_session_do_while_conditional_stop_compiler
    use ffc_case_test_session_do_while_gfortran_compiler, only: &
        case_test_session_do_while_gfortran_compiler
    use ffc_case_test_session_dogfood_select_type_compiler, only: &
        case_test_session_dogfood_select_type_compiler
    use ffc_case_test_session_dotted_operators_compiler, only: &
        case_test_session_dotted_operators_compiler
    use ffc_case_test_session_dummy_array_bound_call_site_compiler, only: &
        case_test_session_dummy_array_bound_call_site_compiler
    use ffc_case_test_session_e_descriptor_compiler, only: &
        case_test_session_e_descriptor_compiler
    use ffc_case_test_session_elemental_class_compiler, only: &
        case_test_session_elemental_class_compiler
    use ffc_case_test_session_elemental_procedure_compiler, only: &
        case_test_session_elemental_procedure_compiler
    use ffc_case_test_session_elseif_chain_compiler, only: &
        case_test_session_elseif_chain_compiler
    use ffc_case_test_session_emit_fmod_compiler, only: &
        case_test_session_emit_fmod_compiler
    use ffc_case_test_session_empty_derived_type_581_compiler, only: &
        case_test_session_empty_derived_type_581_compiler
    use ffc_case_test_session_empty_print_compiler, only: &
        case_test_session_empty_print_compiler
    use ffc_case_test_session_empty_program_compiler, only: &
        case_test_session_empty_program_compiler
    use ffc_case_test_session_empty_program_object_compiler, only: &
        case_test_session_empty_program_object_compiler
    use ffc_case_test_session_enum_compiler, only: &
        case_test_session_enum_compiler
    use ffc_case_test_session_equivalence_compiler, only: &
        case_test_session_equivalence_compiler
    use ffc_case_test_session_external_dummy_callback_compiler, only: &
        case_test_session_external_dummy_callback_compiler
    use ffc_case_test_session_external_interface_procedure_compiler, only: &
        case_test_session_external_interface_procedure_compiler
    use ffc_case_test_session_external_only_unit_compiler, only: &
        case_test_session_external_only_unit_compiler
    use ffc_case_test_session_external_statement_compiler, only: &
        case_test_session_external_statement_compiler
    use ffc_case_test_session_f32_call_f64_function_compiler, only: &
        case_test_session_f32_call_f64_function_compiler
    use ffc_case_test_session_file_unit_io_compiler, only: &
        case_test_session_file_unit_io_compiler
    use ffc_case_test_session_file_unit_read_character_compiler, only: &
        case_test_session_file_unit_read_character_compiler
    use ffc_case_test_session_file_write_character_literal_compiler, only: &
        case_test_session_file_write_character_literal_compiler
    use ffc_case_test_session_fixed_char_function_result_compiler, only: &
        case_test_session_fixed_char_function_result_compiler
    use ffc_case_test_session_fixed_concat_compiler, only: &
        case_test_session_fixed_concat_compiler
    use ffc_case_test_session_fixed_size_array_compiler, only: &
        case_test_session_fixed_size_array_compiler
    use ffc_case_test_session_forall_alias_compiler, only: &
        case_test_session_forall_alias_compiler
    use ffc_case_test_session_forall_compiler, only: &
        case_test_session_forall_compiler
    use ffc_case_test_session_formatted_output_compiler, only: &
        case_test_session_formatted_output_compiler
    implicit none
    private
    public :: run_group
contains
    subroutine run_group(name, matched)
        character(len=*), intent(in) :: name
        logical, intent(out) :: matched

        matched = .true.
        select case (name)
        case ("test_session_dimension_statement_compiler")
            call case_test_session_dimension_statement_compiler()
        case ("test_session_do_while_compiler")
            call case_test_session_do_while_compiler()
        case ("test_session_do_while_conditional_stop_compiler")
            call case_test_session_do_while_conditional_stop_compiler()
        case ("test_session_do_while_gfortran_compiler")
            call case_test_session_do_while_gfortran_compiler()
        case ("test_session_dogfood_select_type_compiler")
            call case_test_session_dogfood_select_type_compiler()
        case ("test_session_dotted_operators_compiler")
            call case_test_session_dotted_operators_compiler()
        case ("test_session_dummy_array_bound_call_site_compiler")
            call case_test_session_dummy_array_bound_call_site_compiler()
        case ("test_session_e_descriptor_compiler")
            call case_test_session_e_descriptor_compiler()
        case ("test_session_elemental_class_compiler")
            call case_test_session_elemental_class_compiler()
        case ("test_session_elemental_procedure_compiler")
            call case_test_session_elemental_procedure_compiler()
        case ("test_session_elseif_chain_compiler")
            call case_test_session_elseif_chain_compiler()
        case ("test_session_emit_fmod_compiler")
            call case_test_session_emit_fmod_compiler()
        case ("test_session_empty_derived_type_581_compiler")
            call case_test_session_empty_derived_type_581_compiler()
        case ("test_session_empty_print_compiler")
            call case_test_session_empty_print_compiler()
        case ("test_session_empty_program_compiler")
            call case_test_session_empty_program_compiler()
        case ("test_session_empty_program_object_compiler")
            call case_test_session_empty_program_object_compiler()
        case ("test_session_enum_compiler")
            call case_test_session_enum_compiler()
        case ("test_session_equivalence_compiler")
            call case_test_session_equivalence_compiler()
        case ("test_session_external_dummy_callback_compiler")
            call case_test_session_external_dummy_callback_compiler()
        case ("test_session_external_interface_procedure_compiler")
            call case_test_session_external_interface_procedure_compiler()
        case ("test_session_external_only_unit_compiler")
            call case_test_session_external_only_unit_compiler()
        case ("test_session_external_statement_compiler")
            call case_test_session_external_statement_compiler()
        case ("test_session_f32_call_f64_function_compiler")
            call case_test_session_f32_call_f64_function_compiler()
        case ("test_session_file_unit_io_compiler")
            call case_test_session_file_unit_io_compiler()
        case ("test_session_file_unit_read_character_compiler")
            call case_test_session_file_unit_read_character_compiler()
        case ("test_session_file_write_character_literal_compiler")
            call case_test_session_file_write_character_literal_compiler()
        case ("test_session_fixed_char_function_result_compiler")
            call case_test_session_fixed_char_function_result_compiler()
        case ("test_session_fixed_concat_compiler")
            call case_test_session_fixed_concat_compiler()
        case ("test_session_fixed_size_array_compiler")
            call case_test_session_fixed_size_array_compiler()
        case ("test_session_forall_alias_compiler")
            call case_test_session_forall_alias_compiler()
        case ("test_session_forall_compiler")
            call case_test_session_forall_compiler()
        case ("test_session_formatted_output_compiler")
            call case_test_session_formatted_output_compiler()
        case default
            matched = .false.
        end select
    end subroutine run_group
end module ffc_suite_group_06
