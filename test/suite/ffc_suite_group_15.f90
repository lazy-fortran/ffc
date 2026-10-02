module ffc_suite_group_15
    use ffc_case_test_session_type_extends_compiler, only: &
        case_test_session_type_extends_compiler
    use ffc_case_test_session_type_info_compiler, only: &
        case_test_session_type_info_compiler
    use ffc_case_test_session_type_intrinsic_spec_compiler, only: &
        case_test_session_type_intrinsic_spec_compiler
    use ffc_case_test_session_typebound_override_dispatch_compiler, only: &
        case_test_session_typebound_override_dispatch_compiler
    use ffc_case_test_session_typed_array_parameter_compiler, only: &
        case_test_session_typed_array_parameter_compiler
    use ffc_case_test_session_undefined_source_oracles_compiler, only: &
        case_test_session_undefined_source_oracles_compiler
    use ffc_case_test_session_unit_boundary_diagnostics, only: &
        case_test_session_unit_boundary_diagnostics
    use ffc_case_test_session_unit_runtime_compiler, only: &
        case_test_session_unit_runtime_compiler
    use ffc_case_test_session_unlinked_external_reference_compiler, only: &
        case_test_session_unlinked_external_reference_compiler
    use ffc_case_test_session_unsupported_diagnostics, only: &
        case_test_session_unsupported_diagnostics
    use ffc_case_test_session_unsupported_kind_rejection_compiler, only: &
        case_test_session_unsupported_kind_rejection_compiler
    use ffc_case_test_session_use_association_valid_compiler, only: &
        case_test_session_use_association_valid_compiler
    use ffc_case_test_session_use_empty_module_compiler, only: &
        case_test_session_use_empty_module_compiler
    use ffc_case_test_session_use_module_constants_compiler, only: &
        case_test_session_use_module_constants_compiler
    use ffc_case_test_session_use_module_derived_type_compiler, only: &
        case_test_session_use_module_derived_type_compiler
    use ffc_case_test_session_use_only_compiler, only: &
        case_test_session_use_only_compiler
    use ffc_case_test_session_vector_subscript_assignment_compiler, only: &
        case_test_session_vector_subscript_assignment_compiler
    use ffc_case_test_session_vector_subscript_compiler, only: &
        case_test_session_vector_subscript_compiler
    use ffc_case_test_session_where_compiler, only: &
        case_test_session_where_compiler
    use ffc_case_test_session_whole_array_arithmetic_compiler, only: &
        case_test_session_whole_array_arithmetic_compiler
    use ffc_case_test_session_whole_array_compare_compiler, only: &
        case_test_session_whole_array_compare_compiler
    use ffc_case_test_session_whole_array_div_pow_compiler, only: &
        case_test_session_whole_array_div_pow_compiler
    use ffc_case_test_session_whole_array_minmax_compiler, only: &
        case_test_session_whole_array_minmax_compiler
    use ffc_case_test_session_whole_array_not_compiler, only: &
        case_test_session_whole_array_not_compiler
    use ffc_case_test_session_whole_array_ops_compiler, only: &
        case_test_session_whole_array_ops_compiler
    use ffc_case_test_session_write_stdout_compiler, only: &
        case_test_session_write_stdout_compiler
    implicit none
    private
    public :: run_group
contains
    subroutine run_group(name, matched)
        character(len=*), intent(in) :: name
        logical, intent(out) :: matched

        matched = .true.
        select case (name)
        case ("test_session_type_extends_compiler")
            call case_test_session_type_extends_compiler()
        case ("test_session_type_info_compiler")
            call case_test_session_type_info_compiler()
        case ("test_session_type_intrinsic_spec_compiler")
            call case_test_session_type_intrinsic_spec_compiler()
        case ("test_session_typebound_override_dispatch_compiler")
            call case_test_session_typebound_override_dispatch_compiler()
        case ("test_session_typed_array_parameter_compiler")
            call case_test_session_typed_array_parameter_compiler()
        case ("test_session_undefined_source_oracles_compiler")
            call case_test_session_undefined_source_oracles_compiler()
        case ("test_session_unit_boundary_diagnostics")
            call case_test_session_unit_boundary_diagnostics()
        case ("test_session_unit_runtime_compiler")
            call case_test_session_unit_runtime_compiler()
        case ("test_session_unlinked_external_reference_compiler")
            call case_test_session_unlinked_external_reference_compiler()
        case ("test_session_unsupported_diagnostics")
            call case_test_session_unsupported_diagnostics()
        case ("test_session_unsupported_kind_rejection_compiler")
            call case_test_session_unsupported_kind_rejection_compiler()
        case ("test_session_use_association_valid_compiler")
            call case_test_session_use_association_valid_compiler()
        case ("test_session_use_empty_module_compiler")
            call case_test_session_use_empty_module_compiler()
        case ("test_session_use_module_constants_compiler")
            call case_test_session_use_module_constants_compiler()
        case ("test_session_use_module_derived_type_compiler")
            call case_test_session_use_module_derived_type_compiler()
        case ("test_session_use_only_compiler")
            call case_test_session_use_only_compiler()
        case ("test_session_vector_subscript_assignment_compiler")
            call case_test_session_vector_subscript_assignment_compiler()
        case ("test_session_vector_subscript_compiler")
            call case_test_session_vector_subscript_compiler()
        case ("test_session_where_compiler")
            call case_test_session_where_compiler()
        case ("test_session_whole_array_arithmetic_compiler")
            call case_test_session_whole_array_arithmetic_compiler()
        case ("test_session_whole_array_compare_compiler")
            call case_test_session_whole_array_compare_compiler()
        case ("test_session_whole_array_div_pow_compiler")
            call case_test_session_whole_array_div_pow_compiler()
        case ("test_session_whole_array_minmax_compiler")
            call case_test_session_whole_array_minmax_compiler()
        case ("test_session_whole_array_not_compiler")
            call case_test_session_whole_array_not_compiler()
        case ("test_session_whole_array_ops_compiler")
            call case_test_session_whole_array_ops_compiler()
        case ("test_session_write_stdout_compiler")
            call case_test_session_write_stdout_compiler()
        case default
            matched = .false.
        end select
    end subroutine run_group
end module ffc_suite_group_15
