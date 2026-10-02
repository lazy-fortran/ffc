module ffc_suite_group_05
    use ffc_case_test_session_data_statement_compiler, only: &
        case_test_session_data_statement_compiler
    use ffc_case_test_session_declaration_collection_compiler, only: &
        case_test_session_declaration_collection_compiler
    use ffc_case_test_session_declaration_kind_parameter_compiler, only: &
        case_test_session_declaration_kind_parameter_compiler
    use ffc_case_test_session_deferred_char_array_compiler, only: &
        case_test_session_deferred_char_array_compiler
    use ffc_case_test_session_deferred_char_compiler, only: &
        case_test_session_deferred_char_compiler
    use ffc_case_test_session_deferred_intent_out_compiler, only: &
        case_test_session_deferred_intent_out_compiler
    use ffc_case_test_session_derived_alloc_array_compiler, only: &
        case_test_session_derived_alloc_array_compiler
    use ffc_case_test_session_derived_alloc_array_dummy_compiler, only: &
        case_test_session_derived_alloc_array_dummy_compiler
    use ffc_case_test_session_derived_alloc_array_finalizer_compiler, only: &
        case_test_session_derived_alloc_array_finalizer_compiler
    use ffc_case_42f771abaa503c9dfb375775, only: &
        case_test_session_derived_alloc_array_finalizer_rank4_compiler
    use ffc_case_test_session_derived_alloc_char_component_compiler, only: &
        case_test_session_derived_alloc_char_component_compiler
    use ffc_case_4c0ecdb22b27d86d1d791509, only: &
        case_test_session_derived_alloc_component_assignment_compiler
    use ffc_case_test_session_derived_alloc_component_compiler, only: &
        case_test_session_derived_alloc_component_compiler
    use ffc_case_test_session_derived_alloc_component_stack_compiler, only: &
        case_test_session_derived_alloc_component_stack_compiler
    use ffc_case_test_session_derived_array_component_compiler, only: &
        case_test_session_derived_array_component_compiler
    use ffc_case_test_session_derived_array_section_compiler, only: &
        case_test_session_derived_array_section_compiler
    use ffc_case_test_session_derived_character_component_compiler, only: &
        case_test_session_derived_character_component_compiler
    use ffc_case_test_session_derived_component_index_rank34_compiler, only: &
        case_test_session_derived_component_index_rank34_compiler
    use ffc_case_test_session_derived_component_section_rank34_compiler, only: &
        case_test_session_derived_component_section_rank34_compiler
    use ffc_case_test_session_derived_constructor_compiler, only: &
        case_test_session_derived_constructor_compiler
    use ffc_case_test_session_derived_element_rank234_compiler, only: &
        case_test_session_derived_element_rank234_compiler
    use ffc_case_test_session_derived_empty_type_compiler, only: &
        case_test_session_derived_empty_type_compiler
    use ffc_case_test_session_derived_nested_array_component_compiler, only: &
        case_test_session_derived_nested_array_component_compiler
    use ffc_case_test_session_derived_pointer_component_compiler, only: &
        case_test_session_derived_pointer_component_compiler
    use ffc_case_test_session_derived_real_component_compiler, only: &
        case_test_session_derived_real_component_compiler
    use ffc_case_test_session_derived_scalar_initializer_compiler, only: &
        case_test_session_derived_scalar_initializer_compiler
    use ffc_case_test_session_derived_section_assumed_shape_compiler, only: &
        case_test_session_derived_section_assumed_shape_compiler
    use ffc_case_test_session_derived_type_compiler, only: &
        case_test_session_derived_type_compiler
    use ffc_case_test_session_derived_typed_defaults_compiler, only: &
        case_test_session_derived_typed_defaults_compiler
    use ffc_case_test_session_dim_numeric_mask_compiler, only: &
        case_test_session_dim_numeric_mask_compiler
    use ffc_case_test_session_dim_reduction_compiler, only: &
        case_test_session_dim_reduction_compiler
    use ffc_case_test_session_dimension_statement_compiler, only: &
        case_test_session_dimension_statement_compiler
    implicit none
    private
    public :: run_group
contains
    subroutine run_group(name, matched)
        character(len=*), intent(in) :: name
        logical, intent(out) :: matched

        matched = .true.
        select case (name)
        case ("test_session_data_statement_compiler")
            call case_test_session_data_statement_compiler()
        case ("test_session_declaration_collection_compiler")
            call case_test_session_declaration_collection_compiler()
        case ("test_session_declaration_kind_parameter_compiler")
            call case_test_session_declaration_kind_parameter_compiler()
        case ("test_session_deferred_char_array_compiler")
            call case_test_session_deferred_char_array_compiler()
        case ("test_session_deferred_char_compiler")
            call case_test_session_deferred_char_compiler()
        case ("test_session_deferred_intent_out_compiler")
            call case_test_session_deferred_intent_out_compiler()
        case ("test_session_derived_alloc_array_compiler")
            call case_test_session_derived_alloc_array_compiler()
        case ("test_session_derived_alloc_array_dummy_compiler")
            call case_test_session_derived_alloc_array_dummy_compiler()
        case ("test_session_derived_alloc_array_finalizer_compiler")
            call case_test_session_derived_alloc_array_finalizer_compiler()
        case ("test_session_derived_alloc_array_finalizer_rank4_compiler")
            call case_test_session_derived_alloc_array_finalizer_rank4_compiler()
        case ("test_session_derived_alloc_char_component_compiler")
            call case_test_session_derived_alloc_char_component_compiler()
        case ("test_session_derived_alloc_component_assignment_compiler")
            call case_test_session_derived_alloc_component_assignment_compiler()
        case ("test_session_derived_alloc_component_compiler")
            call case_test_session_derived_alloc_component_compiler()
        case ("test_session_derived_alloc_component_stack_compiler")
            call case_test_session_derived_alloc_component_stack_compiler()
        case ("test_session_derived_array_component_compiler")
            call case_test_session_derived_array_component_compiler()
        case ("test_session_derived_array_section_compiler")
            call case_test_session_derived_array_section_compiler()
        case ("test_session_derived_character_component_compiler")
            call case_test_session_derived_character_component_compiler()
        case ("test_session_derived_component_index_rank34_compiler")
            call case_test_session_derived_component_index_rank34_compiler()
        case ("test_session_derived_component_section_rank34_compiler")
            call case_test_session_derived_component_section_rank34_compiler()
        case ("test_session_derived_constructor_compiler")
            call case_test_session_derived_constructor_compiler()
        case ("test_session_derived_element_rank234_compiler")
            call case_test_session_derived_element_rank234_compiler()
        case ("test_session_derived_empty_type_compiler")
            call case_test_session_derived_empty_type_compiler()
        case ("test_session_derived_nested_array_component_compiler")
            call case_test_session_derived_nested_array_component_compiler()
        case ("test_session_derived_pointer_component_compiler")
            call case_test_session_derived_pointer_component_compiler()
        case ("test_session_derived_real_component_compiler")
            call case_test_session_derived_real_component_compiler()
        case ("test_session_derived_scalar_initializer_compiler")
            call case_test_session_derived_scalar_initializer_compiler()
        case ("test_session_derived_section_assumed_shape_compiler")
            call case_test_session_derived_section_assumed_shape_compiler()
        case ("test_session_derived_type_compiler")
            call case_test_session_derived_type_compiler()
        case ("test_session_derived_typed_defaults_compiler")
            call case_test_session_derived_typed_defaults_compiler()
        case ("test_session_dim_numeric_mask_compiler")
            call case_test_session_dim_numeric_mask_compiler()
        case ("test_session_dim_reduction_compiler")
            call case_test_session_dim_reduction_compiler()
        case ("test_session_dimension_statement_compiler")
            call case_test_session_dimension_statement_compiler()
        case default
            matched = .false.
        end select
    end subroutine run_group
end module ffc_suite_group_05
