module ffc_suite_group_14
    use ffc_case_test_session_select_rank_compiler, only: &
        case_test_session_select_rank_compiler
    use ffc_case_test_session_select_rank_trailing_compiler, only: &
        case_test_session_select_rank_trailing_compiler
    use ffc_case_test_session_select_type_array_compiler, only: &
        case_test_session_select_type_array_compiler
    use ffc_case_test_session_select_type_compiler, only: &
        case_test_session_select_type_compiler
    use ffc_case_test_session_select_type_derived_compiler, only: &
        case_test_session_select_type_derived_compiler
    use ffc_case_test_session_select_type_runtime_compiler, only: &
        case_test_session_select_type_runtime_compiler
    use ffc_case_test_session_select_type_trailing_compiler, only: &
        case_test_session_select_type_trailing_compiler
    use ffc_case_test_session_selected_kind_compiler, only: &
        case_test_session_selected_kind_compiler
    use ffc_case_test_session_separate_compilation_compiler, only: &
        case_test_session_separate_compilation_compiler
    use ffc_case_test_session_separate_generic_compiler, only: &
        case_test_session_separate_generic_compiler
    use ffc_case_test_session_shift_merge_compiler, only: &
        case_test_session_shift_merge_compiler
    use ffc_case_test_session_shift_rank34_compiler, only: &
        case_test_session_shift_rank34_compiler
    use ffc_case_test_session_size_i64_compiler, only: &
        case_test_session_size_i64_compiler
    use ffc_case_test_session_size_kind_compiler, only: &
        case_test_session_size_kind_compiler
    use ffc_case_test_session_spec_expression_scope_compiler, only: &
        case_test_session_spec_expression_scope_compiler
    use ffc_case_test_session_spread_compiler, only: &
        case_test_session_spread_compiler
    use ffc_case_test_session_statement_function_compiler, only: &
        case_test_session_statement_function_compiler
    use ffc_case_test_session_static_typebound_override_compiler, only: &
        case_test_session_static_typebound_override_compiler
    use ffc_case_test_session_stop_code_compiler, only: &
        case_test_session_stop_code_compiler
    use ffc_case_test_session_stop_message_compiler, only: &
        case_test_session_stop_message_compiler
    use ffc_case_test_session_structured_control_compiler, only: &
        case_test_session_structured_control_compiler
    use ffc_case_test_session_submodule_compiler, only: &
        case_test_session_submodule_compiler
    use ffc_case_test_session_subroutine_return_compiler, only: &
        case_test_session_subroutine_return_compiler
    use ffc_case_test_session_sum_expr_compiler, only: &
        case_test_session_sum_expr_compiler
    use ffc_case_test_session_symbol_table, only: &
        case_test_session_symbol_table
    use ffc_case_test_session_target_attribute_compiler, only: &
        case_test_session_target_attribute_compiler
    use ffc_case_test_session_timing_intrinsics_compiler, only: &
        case_test_session_timing_intrinsics_compiler
    use ffc_case_test_session_transfer_array_compiler, only: &
        case_test_session_transfer_array_compiler
    use ffc_case_test_session_transfer_compiler, only: &
        case_test_session_transfer_compiler
    use ffc_case_test_session_transfer_descriptor_compiler, only: &
        case_test_session_transfer_descriptor_compiler
    use ffc_case_test_session_type_bound_compiler, only: &
        case_test_session_type_bound_compiler
    use ffc_case_test_session_type_extends_compiler, only: &
        case_test_session_type_extends_compiler
    implicit none
    private
    public :: run_group
contains
    subroutine run_group(name, matched)
        character(len=*), intent(in) :: name
        logical, intent(out) :: matched

        matched = .true.
        select case (name)
        case ("test_session_select_rank_compiler")
            call case_test_session_select_rank_compiler()
        case ("test_session_select_rank_trailing_compiler")
            call case_test_session_select_rank_trailing_compiler()
        case ("test_session_select_type_array_compiler")
            call case_test_session_select_type_array_compiler()
        case ("test_session_select_type_compiler")
            call case_test_session_select_type_compiler()
        case ("test_session_select_type_derived_compiler")
            call case_test_session_select_type_derived_compiler()
        case ("test_session_select_type_runtime_compiler")
            call case_test_session_select_type_runtime_compiler()
        case ("test_session_select_type_trailing_compiler")
            call case_test_session_select_type_trailing_compiler()
        case ("test_session_selected_kind_compiler")
            call case_test_session_selected_kind_compiler()
        case ("test_session_separate_compilation_compiler")
            call case_test_session_separate_compilation_compiler()
        case ("test_session_separate_generic_compiler")
            call case_test_session_separate_generic_compiler()
        case ("test_session_shift_merge_compiler")
            call case_test_session_shift_merge_compiler()
        case ("test_session_shift_rank34_compiler")
            call case_test_session_shift_rank34_compiler()
        case ("test_session_size_i64_compiler")
            call case_test_session_size_i64_compiler()
        case ("test_session_size_kind_compiler")
            call case_test_session_size_kind_compiler()
        case ("test_session_spec_expression_scope_compiler")
            call case_test_session_spec_expression_scope_compiler()
        case ("test_session_spread_compiler")
            call case_test_session_spread_compiler()
        case ("test_session_statement_function_compiler")
            call case_test_session_statement_function_compiler()
        case ("test_session_static_typebound_override_compiler")
            call case_test_session_static_typebound_override_compiler()
        case ("test_session_stop_code_compiler")
            call case_test_session_stop_code_compiler()
        case ("test_session_stop_message_compiler")
            call case_test_session_stop_message_compiler()
        case ("test_session_structured_control_compiler")
            call case_test_session_structured_control_compiler()
        case ("test_session_submodule_compiler")
            call case_test_session_submodule_compiler()
        case ("test_session_subroutine_return_compiler")
            call case_test_session_subroutine_return_compiler()
        case ("test_session_sum_expr_compiler")
            call case_test_session_sum_expr_compiler()
        case ("test_session_symbol_table")
            call case_test_session_symbol_table()
        case ("test_session_target_attribute_compiler")
            call case_test_session_target_attribute_compiler()
        case ("test_session_timing_intrinsics_compiler")
            call case_test_session_timing_intrinsics_compiler()
        case ("test_session_transfer_array_compiler")
            call case_test_session_transfer_array_compiler()
        case ("test_session_transfer_compiler")
            call case_test_session_transfer_compiler()
        case ("test_session_transfer_descriptor_compiler")
            call case_test_session_transfer_descriptor_compiler()
        case ("test_session_type_bound_compiler")
            call case_test_session_type_bound_compiler()
        case ("test_session_type_extends_compiler")
            call case_test_session_type_extends_compiler()
        case default
            matched = .false.
        end select
    end subroutine run_group
end module ffc_suite_group_14
