module ffc_suite_group_09
    use ffc_case_test_session_logical_if_compiler, only: &
        case_test_session_logical_if_compiler
    use ffc_case_test_session_logical_kind_compiler, only: &
        case_test_session_logical_kind_compiler
    use ffc_case_test_session_logical_literal_print_compiler, only: &
        case_test_session_logical_literal_print_compiler
    use ffc_case_test_session_logical_not_reduction_oracle_compiler, only: &
        case_test_session_logical_not_reduction_oracle_compiler
    use ffc_case_test_session_logical_reduction_expression_compiler, only: &
        case_test_session_logical_reduction_expression_compiler
    use ffc_case_test_session_logical_result_call_compiler, only: &
        case_test_session_logical_result_call_compiler
    use ffc_case_test_session_logical_transfer_compiler, only: &
        case_test_session_logical_transfer_compiler
    use ffc_case_test_session_logical_variable_compiler, only: &
        case_test_session_logical_variable_compiler
    use ffc_case_test_session_loop_cycle_compiler, only: &
        case_test_session_loop_cycle_compiler
    use ffc_case_test_session_loop_exit_compiler, only: &
        case_test_session_loop_exit_compiler
    use ffc_case_test_session_many_contained_procedures_compiler, only: &
        case_test_session_many_contained_procedures_compiler
    use ffc_case_test_session_mask_reduction_compiler, only: &
        case_test_session_mask_reduction_compiler
    use ffc_case_test_session_matmul_rank34_compiler, only: &
        case_test_session_matmul_rank34_compiler
    use ffc_case_test_session_matmul_vector_compiler, only: &
        case_test_session_matmul_vector_compiler
    use ffc_case_test_session_maxloc_minloc_rank234_compiler, only: &
        case_test_session_maxloc_minloc_rank234_compiler
    use ffc_case_test_session_merge_character_compiler, only: &
        case_test_session_merge_character_compiler
    use ffc_case_test_session_merge_reduction_oracle_compiler, only: &
        case_test_session_merge_reduction_oracle_compiler
    use ffc_case_test_session_mixed_kind_real_expr_compiler, only: &
        case_test_session_mixed_kind_real_expr_compiler
    use ffc_case_test_session_module_allocatable_rank34_compiler, only: &
        case_test_session_module_allocatable_rank34_compiler
    use ffc_case_test_session_module_array_variable_compiler, only: &
        case_test_session_module_array_variable_compiler
    use ffc_case_test_session_module_char_result_compiler, only: &
        case_test_session_module_char_result_compiler
    use ffc_case_test_session_module_constant_scope_compiler, only: &
        case_test_session_module_constant_scope_compiler
    use ffc_case_test_session_module_derived_arg_compiler, only: &
        case_test_session_module_derived_arg_compiler
    use ffc_case_test_session_module_derived_variable_compiler, only: &
        case_test_session_module_derived_variable_compiler
    use ffc_case_test_session_module_exports_derived_type_compiler, only: &
        case_test_session_module_exports_derived_type_compiler
    use ffc_case_test_session_module_fixed_rank4_compiler, only: &
        case_test_session_module_fixed_rank4_compiler
    use ffc_case_test_session_module_internal_proc_compiler, only: &
        case_test_session_module_internal_proc_compiler
    use ffc_case_test_session_module_multi_name_var_compiler, only: &
        case_test_session_module_multi_name_var_compiler
    use ffc_case_test_session_module_parameter_kind_compiler, only: &
        case_test_session_module_parameter_kind_compiler
    use ffc_case_test_session_module_procedure_compiler, only: &
        case_test_session_module_procedure_compiler
    use ffc_case_test_session_module_variable_compiler, only: &
        case_test_session_module_variable_compiler
    use ffc_case_test_session_module_visibility_compiler, only: &
        case_test_session_module_visibility_compiler
    implicit none
    private
    public :: run_group
contains
    subroutine run_group(name, matched)
        character(len=*), intent(in) :: name
        logical, intent(out) :: matched

        matched = .true.
        select case (name)
        case ("test_session_logical_if_compiler")
            call case_test_session_logical_if_compiler()
        case ("test_session_logical_kind_compiler")
            call case_test_session_logical_kind_compiler()
        case ("test_session_logical_literal_print_compiler")
            call case_test_session_logical_literal_print_compiler()
        case ("test_session_logical_not_reduction_oracle_compiler")
            call case_test_session_logical_not_reduction_oracle_compiler()
        case ("test_session_logical_reduction_expression_compiler")
            call case_test_session_logical_reduction_expression_compiler()
        case ("test_session_logical_result_call_compiler")
            call case_test_session_logical_result_call_compiler()
        case ("test_session_logical_transfer_compiler")
            call case_test_session_logical_transfer_compiler()
        case ("test_session_logical_variable_compiler")
            call case_test_session_logical_variable_compiler()
        case ("test_session_loop_cycle_compiler")
            call case_test_session_loop_cycle_compiler()
        case ("test_session_loop_exit_compiler")
            call case_test_session_loop_exit_compiler()
        case ("test_session_many_contained_procedures_compiler")
            call case_test_session_many_contained_procedures_compiler()
        case ("test_session_mask_reduction_compiler")
            call case_test_session_mask_reduction_compiler()
        case ("test_session_matmul_rank34_compiler")
            call case_test_session_matmul_rank34_compiler()
        case ("test_session_matmul_vector_compiler")
            call case_test_session_matmul_vector_compiler()
        case ("test_session_maxloc_minloc_rank234_compiler")
            call case_test_session_maxloc_minloc_rank234_compiler()
        case ("test_session_merge_character_compiler")
            call case_test_session_merge_character_compiler()
        case ("test_session_merge_reduction_oracle_compiler")
            call case_test_session_merge_reduction_oracle_compiler()
        case ("test_session_mixed_kind_real_expr_compiler")
            call case_test_session_mixed_kind_real_expr_compiler()
        case ("test_session_module_allocatable_rank34_compiler")
            call case_test_session_module_allocatable_rank34_compiler()
        case ("test_session_module_array_variable_compiler")
            call case_test_session_module_array_variable_compiler()
        case ("test_session_module_char_result_compiler")
            call case_test_session_module_char_result_compiler()
        case ("test_session_module_constant_scope_compiler")
            call case_test_session_module_constant_scope_compiler()
        case ("test_session_module_derived_arg_compiler")
            call case_test_session_module_derived_arg_compiler()
        case ("test_session_module_derived_variable_compiler")
            call case_test_session_module_derived_variable_compiler()
        case ("test_session_module_exports_derived_type_compiler")
            call case_test_session_module_exports_derived_type_compiler()
        case ("test_session_module_fixed_rank4_compiler")
            call case_test_session_module_fixed_rank4_compiler()
        case ("test_session_module_internal_proc_compiler")
            call case_test_session_module_internal_proc_compiler()
        case ("test_session_module_multi_name_var_compiler")
            call case_test_session_module_multi_name_var_compiler()
        case ("test_session_module_parameter_kind_compiler")
            call case_test_session_module_parameter_kind_compiler()
        case ("test_session_module_procedure_compiler")
            call case_test_session_module_procedure_compiler()
        case ("test_session_module_variable_compiler")
            call case_test_session_module_variable_compiler()
        case ("test_session_module_visibility_compiler")
            call case_test_session_module_visibility_compiler()
        case default
            matched = .false.
        end select
    end subroutine run_group
end module ffc_suite_group_09
