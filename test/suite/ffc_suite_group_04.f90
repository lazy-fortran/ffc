module ffc_suite_group_04
    use ffc_case_test_session_character_prefix_compiler, only: &
        case_test_session_character_prefix_compiler
    use ffc_case_test_session_character_variable_compiler, only: &
        case_test_session_character_variable_compiler
    use ffc_case_test_session_class_allocatable_source_compiler, only: &
        case_test_session_class_allocatable_source_compiler
    use ffc_case_test_session_class_pointer_dispatch_compiler, only: &
        case_test_session_class_pointer_dispatch_compiler
    use ffc_case_test_session_class_scalar_identity_compiler, only: &
        case_test_session_class_scalar_identity_compiler
    use ffc_case_test_session_class_star_assumed_shape_compiler, only: &
        case_test_session_class_star_assumed_shape_compiler
    use ffc_case_test_session_class_star_rank2_assumed_shape_compiler, only: &
        case_test_session_class_star_rank2_assumed_shape_compiler
    use ffc_case_test_session_cli_backend, only: &
        case_test_session_cli_backend
    use ffc_case_test_session_cli_include_paths, only: &
        case_test_session_cli_include_paths
    use ffc_case_test_session_close_status_compiler, only: &
        case_test_session_close_status_compiler
    use ffc_case_test_session_command_argument_compiler, only: &
        case_test_session_command_argument_compiler
    use ffc_case_test_session_common_mixed_layout_compiler, only: &
        case_test_session_common_mixed_layout_compiler
    use ffc_case_test_session_comparison_typecheck_compiler, only: &
        case_test_session_comparison_typecheck_compiler
    use ffc_case_test_session_complex_arith_compiler, only: &
        case_test_session_complex_arith_compiler
    use ffc_case_test_session_complex_array_compiler, only: &
        case_test_session_complex_array_compiler
    use ffc_case_test_session_complex_array_rank34_compiler, only: &
        case_test_session_complex_array_rank34_compiler
    use ffc_case_test_session_complex_array_whole_compiler, only: &
        case_test_session_complex_array_whole_compiler
    use ffc_case_test_session_complex_cast_compiler, only: &
        case_test_session_complex_cast_compiler
    use ffc_case_test_session_complex_compiler, only: &
        case_test_session_complex_compiler
    use ffc_case_test_session_complex_component_compiler, only: &
        case_test_session_complex_component_compiler
    use ffc_case_test_session_complex_function_result_compiler, only: &
        case_test_session_complex_function_result_compiler
    use ffc_case_test_session_complex_intrinsics_compiler, only: &
        case_test_session_complex_intrinsics_compiler
    use ffc_case_test_session_complex_literal_compiler, only: &
        case_test_session_complex_literal_compiler
    use ffc_case_test_session_complex_real_mixed_compiler, only: &
        case_test_session_complex_real_mixed_compiler
    use ffc_case_test_session_compound_declarations_compiler, only: &
        case_test_session_compound_declarations_compiler
    use ffc_case_test_session_const_fold_compiler, only: &
        case_test_session_const_fold_compiler
    use ffc_case_test_session_const_fold_intrinsics_compiler, only: &
        case_test_session_const_fold_intrinsics_compiler
    use ffc_case_test_session_construct_name_branches_compiler, only: &
        case_test_session_construct_name_branches_compiler
    use ffc_case_test_session_contained_call_operand_compiler, only: &
        case_test_session_contained_call_operand_compiler
    use ffc_case_test_session_container_contained_fn_compiler, only: &
        case_test_session_container_contained_fn_compiler
    use ffc_case_test_session_corpus_gaps_609_compiler, only: &
        case_test_session_corpus_gaps_609_compiler
    use ffc_case_test_session_cycle_branch_values_compiler, only: &
        case_test_session_cycle_branch_values_compiler
    implicit none
    private
    public :: run_group
contains
    subroutine run_group(name, matched)
        character(len=*), intent(in) :: name
        logical, intent(out) :: matched

        matched = .true.
        select case (name)
        case ("test_session_character_prefix_compiler")
            call case_test_session_character_prefix_compiler()
        case ("test_session_character_variable_compiler")
            call case_test_session_character_variable_compiler()
        case ("test_session_class_allocatable_source_compiler")
            call case_test_session_class_allocatable_source_compiler()
        case ("test_session_class_pointer_dispatch_compiler")
            call case_test_session_class_pointer_dispatch_compiler()
        case ("test_session_class_scalar_identity_compiler")
            call case_test_session_class_scalar_identity_compiler()
        case ("test_session_class_star_assumed_shape_compiler")
            call case_test_session_class_star_assumed_shape_compiler()
        case ("test_session_class_star_rank2_assumed_shape_compiler")
            call case_test_session_class_star_rank2_assumed_shape_compiler()
        case ("test_session_cli_backend")
            call case_test_session_cli_backend()
        case ("test_session_cli_include_paths")
            call case_test_session_cli_include_paths()
        case ("test_session_close_status_compiler")
            call case_test_session_close_status_compiler()
        case ("test_session_command_argument_compiler")
            call case_test_session_command_argument_compiler()
        case ("test_session_common_mixed_layout_compiler")
            call case_test_session_common_mixed_layout_compiler()
        case ("test_session_comparison_typecheck_compiler")
            call case_test_session_comparison_typecheck_compiler()
        case ("test_session_complex_arith_compiler")
            call case_test_session_complex_arith_compiler()
        case ("test_session_complex_array_compiler")
            call case_test_session_complex_array_compiler()
        case ("test_session_complex_array_rank34_compiler")
            call case_test_session_complex_array_rank34_compiler()
        case ("test_session_complex_array_whole_compiler")
            call case_test_session_complex_array_whole_compiler()
        case ("test_session_complex_cast_compiler")
            call case_test_session_complex_cast_compiler()
        case ("test_session_complex_compiler")
            call case_test_session_complex_compiler()
        case ("test_session_complex_component_compiler")
            call case_test_session_complex_component_compiler()
        case ("test_session_complex_function_result_compiler")
            call case_test_session_complex_function_result_compiler()
        case ("test_session_complex_intrinsics_compiler")
            call case_test_session_complex_intrinsics_compiler()
        case ("test_session_complex_literal_compiler")
            call case_test_session_complex_literal_compiler()
        case ("test_session_complex_real_mixed_compiler")
            call case_test_session_complex_real_mixed_compiler()
        case ("test_session_compound_declarations_compiler")
            call case_test_session_compound_declarations_compiler()
        case ("test_session_const_fold_compiler")
            call case_test_session_const_fold_compiler()
        case ("test_session_const_fold_intrinsics_compiler")
            call case_test_session_const_fold_intrinsics_compiler()
        case ("test_session_construct_name_branches_compiler")
            call case_test_session_construct_name_branches_compiler()
        case ("test_session_contained_call_operand_compiler")
            call case_test_session_contained_call_operand_compiler()
        case ("test_session_container_contained_fn_compiler")
            call case_test_session_container_contained_fn_compiler()
        case ("test_session_corpus_gaps_609_compiler")
            call case_test_session_corpus_gaps_609_compiler()
        case ("test_session_cycle_branch_values_compiler")
            call case_test_session_cycle_branch_values_compiler()
        case default
            matched = .false.
        end select
    end subroutine run_group
end module ffc_suite_group_04
