module ffc_suite_group_01
    use ffc_case_test_session_abstract_interface_compiler, only: &
        case_test_session_abstract_interface_compiler
    use ffc_case_2b23d77d694bf09939b9d7c7, only: &
        case_test_session_abstract_interface_procedure_dummy_compiler
    use ffc_case_test_session_accept_reject_false_01_compiler, only: &
        case_test_session_accept_reject_false_01_compiler
    use ffc_case_test_session_accept_use_generic_heuristics_compiler, only: &
        case_test_session_accept_use_generic_heuristics_compiler
    use ffc_case_test_session_aint_anint_compiler, only: &
        case_test_session_aint_anint_compiler
    use ffc_case_test_session_alloc_array_component_compiler, only: &
        case_test_session_alloc_array_component_compiler
    use ffc_case_test_session_alloc_array_function_result_compiler, only: &
        case_test_session_alloc_array_function_result_compiler
    use ffc_case_test_session_alloc_derived_array_component_compiler, only: &
        case_test_session_alloc_derived_array_component_compiler
    use ffc_case_test_session_alloc_rank2_component_compiler, only: &
        case_test_session_alloc_rank2_component_compiler
    use ffc_case_test_session_alloc_rank3_component_compiler, only: &
        case_test_session_alloc_rank3_component_compiler
    use ffc_case_test_session_alloc_rank4_component_compiler, only: &
        case_test_session_alloc_rank4_component_compiler
    use ffc_case_test_session_allocatable_constructor_compiler, only: &
        case_test_session_allocatable_constructor_compiler
    use ffc_case_test_session_allocatable_dummy_array_compiler, only: &
        case_test_session_allocatable_dummy_array_compiler
    use ffc_case_test_session_allocatable_element_compiler, only: &
        case_test_session_allocatable_element_compiler
    use ffc_case_test_session_allocatable_elementwise_compiler, only: &
        case_test_session_allocatable_elementwise_compiler
    use ffc_case_07b9231cdeeb7e9f84c56b59, only: &
        case_test_session_allocatable_function_result_rank34_compiler
    use ffc_case_test_session_allocatable_lifecycle_compiler, only: &
        case_test_session_allocatable_lifecycle_compiler
    use ffc_case_test_session_allocatable_lower_bounds_compiler, only: &
        case_test_session_allocatable_lower_bounds_compiler
    use ffc_case_test_session_allocatable_move_alloc_compiler, only: &
        case_test_session_allocatable_move_alloc_compiler
    use ffc_case_test_session_allocatable_rank2_compiler, only: &
        case_test_session_allocatable_rank2_compiler
    use ffc_case_test_session_allocatable_rank2_typed_compiler, only: &
        case_test_session_allocatable_rank2_typed_compiler
    use ffc_case_test_session_allocatable_rank3_compiler, only: &
        case_test_session_allocatable_rank3_compiler
    use ffc_case_test_session_allocatable_rank4_compiler, only: &
        case_test_session_allocatable_rank4_compiler
    use ffc_case_test_session_allocatable_reduction_compiler, only: &
        case_test_session_allocatable_reduction_compiler
    use ffc_case_test_session_allocate_mold_source_compiler, only: &
        case_test_session_allocate_mold_source_compiler
    use ffc_case_test_session_allocate_mold_source_rank34_compiler, only: &
        case_test_session_allocate_mold_source_rank34_compiler
    use ffc_case_e768d7228655406e1c7f15f8, only: &
        case_test_session_allocate_source_expression_rank234_compiler
    use ffc_case_test_session_allocate_typespec_compiler, only: &
        case_test_session_allocate_typespec_compiler
    use ffc_case_test_session_allocated_keyword_compiler, only: &
        case_test_session_allocated_keyword_compiler
    use ffc_case_test_session_alternate_return_compiler, only: &
        case_test_session_alternate_return_compiler
    use ffc_case_test_session_ambiguous_interface_compiler, only: &
        case_test_session_ambiguous_interface_compiler
    use ffc_case_test_session_arg_count_mismatch_compiler, only: &
        case_test_session_arg_count_mismatch_compiler
    implicit none
    private
    public :: run_group
contains
    subroutine run_group(name, matched)
        character(len=*), intent(in) :: name
        logical, intent(out) :: matched

        matched = .true.
        select case (name)
        case ("test_session_abstract_interface_compiler")
            call case_test_session_abstract_interface_compiler()
        case ("test_session_abstract_interface_procedure_dummy_compiler")
            call case_test_session_abstract_interface_procedure_dummy_compiler()
        case ("test_session_accept_reject_false_01_compiler")
            call case_test_session_accept_reject_false_01_compiler()
        case ("test_session_accept_use_generic_heuristics_compiler")
            call case_test_session_accept_use_generic_heuristics_compiler()
        case ("test_session_aint_anint_compiler")
            call case_test_session_aint_anint_compiler()
        case ("test_session_alloc_array_component_compiler")
            call case_test_session_alloc_array_component_compiler()
        case ("test_session_alloc_array_function_result_compiler")
            call case_test_session_alloc_array_function_result_compiler()
        case ("test_session_alloc_derived_array_component_compiler")
            call case_test_session_alloc_derived_array_component_compiler()
        case ("test_session_alloc_rank2_component_compiler")
            call case_test_session_alloc_rank2_component_compiler()
        case ("test_session_alloc_rank3_component_compiler")
            call case_test_session_alloc_rank3_component_compiler()
        case ("test_session_alloc_rank4_component_compiler")
            call case_test_session_alloc_rank4_component_compiler()
        case ("test_session_allocatable_constructor_compiler")
            call case_test_session_allocatable_constructor_compiler()
        case ("test_session_allocatable_dummy_array_compiler")
            call case_test_session_allocatable_dummy_array_compiler()
        case ("test_session_allocatable_element_compiler")
            call case_test_session_allocatable_element_compiler()
        case ("test_session_allocatable_elementwise_compiler")
            call case_test_session_allocatable_elementwise_compiler()
        case ("test_session_allocatable_function_result_rank34_compiler")
            call case_test_session_allocatable_function_result_rank34_compiler()
        case ("test_session_allocatable_lifecycle_compiler")
            call case_test_session_allocatable_lifecycle_compiler()
        case ("test_session_allocatable_lower_bounds_compiler")
            call case_test_session_allocatable_lower_bounds_compiler()
        case ("test_session_allocatable_move_alloc_compiler")
            call case_test_session_allocatable_move_alloc_compiler()
        case ("test_session_allocatable_rank2_compiler")
            call case_test_session_allocatable_rank2_compiler()
        case ("test_session_allocatable_rank2_typed_compiler")
            call case_test_session_allocatable_rank2_typed_compiler()
        case ("test_session_allocatable_rank3_compiler")
            call case_test_session_allocatable_rank3_compiler()
        case ("test_session_allocatable_rank4_compiler")
            call case_test_session_allocatable_rank4_compiler()
        case ("test_session_allocatable_reduction_compiler")
            call case_test_session_allocatable_reduction_compiler()
        case ("test_session_allocate_mold_source_compiler")
            call case_test_session_allocate_mold_source_compiler()
        case ("test_session_allocate_mold_source_rank34_compiler")
            call case_test_session_allocate_mold_source_rank34_compiler()
        case ("test_session_allocate_source_expression_rank234_compiler")
            call case_test_session_allocate_source_expression_rank234_compiler()
        case ("test_session_allocate_typespec_compiler")
            call case_test_session_allocate_typespec_compiler()
        case ("test_session_allocated_keyword_compiler")
            call case_test_session_allocated_keyword_compiler()
        case ("test_session_alternate_return_compiler")
            call case_test_session_alternate_return_compiler()
        case ("test_session_ambiguous_interface_compiler")
            call case_test_session_ambiguous_interface_compiler()
        case ("test_session_arg_count_mismatch_compiler")
            call case_test_session_arg_count_mismatch_compiler()
        case default
            matched = .false.
        end select
    end subroutine run_group
end module ffc_suite_group_01
