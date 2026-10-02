module ffc_suite_group_02
    use ffc_case_test_session_array_alias_assignment_compiler, only: &
        case_test_session_array_alias_assignment_compiler
    use ffc_case_test_session_array_constructor_compiler, only: &
        case_test_session_array_constructor_compiler
    use ffc_case_test_session_array_ctor_length_compiler, only: &
        case_test_session_array_ctor_length_compiler
    use ffc_case_test_session_array_ctor_typecheck_compiler, only: &
        case_test_session_array_ctor_typecheck_compiler
    use ffc_case_test_session_array_element_rank34_compiler, only: &
        case_test_session_array_element_rank34_compiler
    use ffc_case_test_session_array_elemental_intrinsic_print_compiler, only: &
        case_test_session_array_elemental_intrinsic_print_compiler
    use ffc_case_test_session_array_expr_reduction_compiler, only: &
        case_test_session_array_expr_reduction_compiler
    use ffc_case_test_session_array_function_rank34_compiler, only: &
        case_test_session_array_function_rank34_compiler
    use ffc_case_test_session_array_function_result_compiler, only: &
        case_test_session_array_function_result_compiler
    use ffc_case_test_session_array_intrinsics_cluster_compiler, only: &
        case_test_session_array_intrinsics_cluster_compiler
    use ffc_case_test_session_array_intrinsics_compiler, only: &
        case_test_session_array_intrinsics_compiler
    use ffc_case_test_session_array_literal_print_compiler, only: &
        case_test_session_array_literal_print_compiler
    use ffc_case_test_session_array_mask_reduction_compiler, only: &
        case_test_session_array_mask_reduction_compiler
    use ffc_case_test_session_array_product_compiler, only: &
        case_test_session_array_product_compiler
    use ffc_case_test_session_array_section_compiler, only: &
        case_test_session_array_section_compiler
    use ffc_case_test_session_array_section_copy_rank4_compiler, only: &
        case_test_session_array_section_copy_rank4_compiler
    use ffc_case_test_session_array_section_descriptor_compiler, only: &
        case_test_session_array_section_descriptor_compiler
    use ffc_case_test_session_array_section_rank4_compiler, only: &
        case_test_session_array_section_rank4_compiler
    use ffc_case_test_session_array_shape_module_compiler, only: &
        case_test_session_array_shape_module_compiler
    use ffc_case_test_session_array_shift_intrinsics_compiler, only: &
        case_test_session_array_shift_intrinsics_compiler
    use ffc_case_test_session_array_transform_intrinsics_compiler, only: &
        case_test_session_array_transform_intrinsics_compiler
    use ffc_case_test_session_array_unsupported_diagnostics, only: &
        case_test_session_array_unsupported_diagnostics
    use ffc_case_test_session_associate_compiler, only: &
        case_test_session_associate_compiler
    use ffc_case_test_session_associate_expression_rank4_compiler, only: &
        case_test_session_associate_expression_rank4_compiler
    use ffc_case_test_session_associate_selector_rank34_compiler, only: &
        case_test_session_associate_selector_rank34_compiler
    use ffc_case_test_session_associate_selectors_compiler, only: &
        case_test_session_associate_selectors_compiler
    use ffc_case_test_session_assumed_rank_select_rank2_compiler, only: &
        case_test_session_assumed_rank_select_rank2_compiler
    use ffc_case_test_session_assumed_rank_select_rank3_compiler, only: &
        case_test_session_assumed_rank_select_rank3_compiler
    use ffc_case_test_session_assumed_rank_select_rank4_compiler, only: &
        case_test_session_assumed_rank_select_rank4_compiler
    use ffc_case_test_session_assumed_rank_select_rank_compiler, only: &
        case_test_session_assumed_rank_select_rank_compiler
    use ffc_case_test_session_assumed_shape_compiler, only: &
        case_test_session_assumed_shape_compiler
    use ffc_case_test_session_assumed_shape_derived_bounds_compiler, only: &
        case_test_session_assumed_shape_derived_bounds_compiler
    implicit none
    private
    public :: run_group
contains
    subroutine run_group(name, matched)
        character(len=*), intent(in) :: name
        logical, intent(out) :: matched

        matched = .true.
        select case (name)
        case ("test_session_array_alias_assignment_compiler")
            call case_test_session_array_alias_assignment_compiler()
        case ("test_session_array_constructor_compiler")
            call case_test_session_array_constructor_compiler()
        case ("test_session_array_ctor_length_compiler")
            call case_test_session_array_ctor_length_compiler()
        case ("test_session_array_ctor_typecheck_compiler")
            call case_test_session_array_ctor_typecheck_compiler()
        case ("test_session_array_element_rank34_compiler")
            call case_test_session_array_element_rank34_compiler()
        case ("test_session_array_elemental_intrinsic_print_compiler")
            call case_test_session_array_elemental_intrinsic_print_compiler()
        case ("test_session_array_expr_reduction_compiler")
            call case_test_session_array_expr_reduction_compiler()
        case ("test_session_array_function_rank34_compiler")
            call case_test_session_array_function_rank34_compiler()
        case ("test_session_array_function_result_compiler")
            call case_test_session_array_function_result_compiler()
        case ("test_session_array_intrinsics_cluster_compiler")
            call case_test_session_array_intrinsics_cluster_compiler()
        case ("test_session_array_intrinsics_compiler")
            call case_test_session_array_intrinsics_compiler()
        case ("test_session_array_literal_print_compiler")
            call case_test_session_array_literal_print_compiler()
        case ("test_session_array_mask_reduction_compiler")
            call case_test_session_array_mask_reduction_compiler()
        case ("test_session_array_product_compiler")
            call case_test_session_array_product_compiler()
        case ("test_session_array_section_compiler")
            call case_test_session_array_section_compiler()
        case ("test_session_array_section_copy_rank4_compiler")
            call case_test_session_array_section_copy_rank4_compiler()
        case ("test_session_array_section_descriptor_compiler")
            call case_test_session_array_section_descriptor_compiler()
        case ("test_session_array_section_rank4_compiler")
            call case_test_session_array_section_rank4_compiler()
        case ("test_session_array_shape_module_compiler")
            call case_test_session_array_shape_module_compiler()
        case ("test_session_array_shift_intrinsics_compiler")
            call case_test_session_array_shift_intrinsics_compiler()
        case ("test_session_array_transform_intrinsics_compiler")
            call case_test_session_array_transform_intrinsics_compiler()
        case ("test_session_array_unsupported_diagnostics")
            call case_test_session_array_unsupported_diagnostics()
        case ("test_session_associate_compiler")
            call case_test_session_associate_compiler()
        case ("test_session_associate_expression_rank4_compiler")
            call case_test_session_associate_expression_rank4_compiler()
        case ("test_session_associate_selector_rank34_compiler")
            call case_test_session_associate_selector_rank34_compiler()
        case ("test_session_associate_selectors_compiler")
            call case_test_session_associate_selectors_compiler()
        case ("test_session_assumed_rank_select_rank2_compiler")
            call case_test_session_assumed_rank_select_rank2_compiler()
        case ("test_session_assumed_rank_select_rank3_compiler")
            call case_test_session_assumed_rank_select_rank3_compiler()
        case ("test_session_assumed_rank_select_rank4_compiler")
            call case_test_session_assumed_rank_select_rank4_compiler()
        case ("test_session_assumed_rank_select_rank_compiler")
            call case_test_session_assumed_rank_select_rank_compiler()
        case ("test_session_assumed_shape_compiler")
            call case_test_session_assumed_shape_compiler()
        case ("test_session_assumed_shape_derived_bounds_compiler")
            call case_test_session_assumed_shape_derived_bounds_compiler()
        case default
            matched = .false.
        end select
    end subroutine run_group
end module ffc_suite_group_02
