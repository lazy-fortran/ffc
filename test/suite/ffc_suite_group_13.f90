module ffc_suite_group_13
    use ffc_case_test_session_runtime_any_all_compiler, only: &
        case_test_session_runtime_any_all_compiler
    use ffc_case_test_session_runtime_archive_compiler, only: &
        case_test_session_runtime_archive_compiler
    use ffc_case_test_session_runtime_array_section_broadcast_compiler, only: &
        case_test_session_runtime_array_section_broadcast_compiler
    use ffc_case_test_session_runtime_array_section_multidim_compiler, only: &
        case_test_session_runtime_array_section_multidim_compiler
    use ffc_case_test_session_runtime_array_section_rank34_compiler, only: &
        case_test_session_runtime_array_section_rank34_compiler
    use ffc_case_test_session_runtime_bound_array_compiler, only: &
        case_test_session_runtime_bound_array_compiler
    use ffc_case_test_session_runtime_character_result_compiler, only: &
        case_test_session_runtime_character_result_compiler
    use ffc_case_test_session_runtime_count_compiler, only: &
        case_test_session_runtime_count_compiler
    use ffc_case_test_session_runtime_extreme_compiler, only: &
        case_test_session_runtime_extreme_compiler
    use ffc_case_test_session_runtime_fixed_character_compiler, only: &
        case_test_session_runtime_fixed_character_compiler
    use ffc_case_7f5152b0a55651ce7b58b5e4, only: &
        case_7f5152b0a55651ce7b58b5e4
    use ffc_case_test_session_runtime_local_array_compiler, only: &
        case_test_session_runtime_local_array_compiler
    use ffc_case_test_session_runtime_norm2_compiler, only: &
        case_test_session_runtime_norm2_compiler
    use ffc_case_test_session_runtime_product_compiler, only: &
        case_test_session_runtime_product_compiler
    use ffc_case_test_session_runtime_rank2_print_compiler, only: &
        case_test_session_runtime_rank2_print_compiler
    use ffc_case_test_session_runtime_rank2_sum_compiler, only: &
        case_test_session_runtime_rank2_sum_compiler
    use ffc_case_test_session_runtime_rank4_array_compiler, only: &
        case_test_session_runtime_rank4_array_compiler
    use ffc_case_test_session_save_attribute_compiler, only: &
        case_test_session_save_attribute_compiler
    use ffc_case_test_session_save_module_shadow_compiler, only: &
        case_test_session_save_module_shadow_compiler
    use ffc_case_test_session_scalar_allocatable_compiler, only: &
        case_test_session_scalar_allocatable_compiler
    use ffc_case_test_session_scalar_allocatable_derived_compiler, only: &
        case_test_session_scalar_allocatable_derived_compiler
    use ffc_case_test_session_scalar_expression_compiler, only: &
        case_test_session_scalar_expression_compiler
    use ffc_case_test_session_scalar_finalizer_compiler, only: &
        case_test_session_scalar_finalizer_compiler
    use ffc_case_test_session_scalar_intrinsics_compiler, only: &
        case_test_session_scalar_intrinsics_compiler
    use ffc_case_test_session_scalar_pointer_compiler, only: &
        case_test_session_scalar_pointer_compiler
    use ffc_case_test_session_scalar_print_compiler, only: &
        case_test_session_scalar_print_compiler
    use ffc_case_test_session_scalar_procedure_call_compiler, only: &
        case_test_session_scalar_procedure_call_compiler
    use ffc_case_test_session_scalar_reduction_compiler, only: &
        case_test_session_scalar_reduction_compiler
    use ffc_case_test_session_scope_binding_negative_control_compiler, only: &
        case_test_session_scope_binding_negative_control_compiler
    use ffc_case_test_session_scope_resolution_compiler, only: &
        case_test_session_scope_resolution_compiler
    use ffc_case_test_session_scratch_unit_compiler, only: &
        case_test_session_scratch_unit_compiler
    use ffc_case_test_session_select_case_compiler, only: &
        case_test_session_select_case_compiler
    implicit none
    private
    public :: run_group
contains
    subroutine run_group(name, matched)
        character(len=*), intent(in) :: name
        logical, intent(out) :: matched

        matched = .true.
        select case (name)
        case ("test_session_runtime_any_all_compiler")
            call case_test_session_runtime_any_all_compiler()
        case ("test_session_runtime_archive_compiler")
            call case_test_session_runtime_archive_compiler()
        case ("test_session_runtime_array_section_broadcast_compiler")
            call case_test_session_runtime_array_section_broadcast_compiler()
        case ("test_session_runtime_array_section_multidim_compiler")
            call case_test_session_runtime_array_section_multidim_compiler()
        case ("test_session_runtime_array_section_rank34_compiler")
            call case_test_session_runtime_array_section_rank34_compiler()
        case ("test_session_runtime_bound_array_compiler")
            call case_test_session_runtime_bound_array_compiler()
        case ("test_session_runtime_character_result_compiler")
            call case_test_session_runtime_character_result_compiler()
        case ("test_session_runtime_count_compiler")
            call case_test_session_runtime_count_compiler()
        case ("test_session_runtime_extreme_compiler")
            call case_test_session_runtime_extreme_compiler()
        case ("test_session_runtime_fixed_character_compiler")
            call case_test_session_runtime_fixed_character_compiler()
        case ("test_session_runtime_length_expression_character_result_compiler")
            call case_7f5152b0a55651ce7b58b5e4()
        case ("test_session_runtime_local_array_compiler")
            call case_test_session_runtime_local_array_compiler()
        case ("test_session_runtime_norm2_compiler")
            call case_test_session_runtime_norm2_compiler()
        case ("test_session_runtime_product_compiler")
            call case_test_session_runtime_product_compiler()
        case ("test_session_runtime_rank2_print_compiler")
            call case_test_session_runtime_rank2_print_compiler()
        case ("test_session_runtime_rank2_sum_compiler")
            call case_test_session_runtime_rank2_sum_compiler()
        case ("test_session_runtime_rank4_array_compiler")
            call case_test_session_runtime_rank4_array_compiler()
        case ("test_session_save_attribute_compiler")
            call case_test_session_save_attribute_compiler()
        case ("test_session_save_module_shadow_compiler")
            call case_test_session_save_module_shadow_compiler()
        case ("test_session_scalar_allocatable_compiler")
            call case_test_session_scalar_allocatable_compiler()
        case ("test_session_scalar_allocatable_derived_compiler")
            call case_test_session_scalar_allocatable_derived_compiler()
        case ("test_session_scalar_expression_compiler")
            call case_test_session_scalar_expression_compiler()
        case ("test_session_scalar_finalizer_compiler")
            call case_test_session_scalar_finalizer_compiler()
        case ("test_session_scalar_intrinsics_compiler")
            call case_test_session_scalar_intrinsics_compiler()
        case ("test_session_scalar_pointer_compiler")
            call case_test_session_scalar_pointer_compiler()
        case ("test_session_scalar_print_compiler")
            call case_test_session_scalar_print_compiler()
        case ("test_session_scalar_procedure_call_compiler")
            call case_test_session_scalar_procedure_call_compiler()
        case ("test_session_scalar_reduction_compiler")
            call case_test_session_scalar_reduction_compiler()
        case ("test_session_scope_binding_negative_control_compiler")
            call case_test_session_scope_binding_negative_control_compiler()
        case ("test_session_scope_resolution_compiler")
            call case_test_session_scope_resolution_compiler()
        case ("test_session_scratch_unit_compiler")
            call case_test_session_scratch_unit_compiler()
        case ("test_session_select_case_compiler")
            call case_test_session_select_case_compiler()
        case default
            matched = .false.
        end select
    end subroutine run_group
end module ffc_suite_group_13
