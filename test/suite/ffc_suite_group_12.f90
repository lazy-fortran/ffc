module ffc_suite_group_12
    use ffc_case_test_session_real_variable_compiler, only: &
        case_test_session_real_variable_compiler
    use ffc_case_test_session_recursive_compiler, only: &
        case_test_session_recursive_compiler
    use ffc_case_test_session_recursive_function_compiler, only: &
        case_test_session_recursive_function_compiler
    use ffc_case_test_session_reduction_expr_oracle_compiler, only: &
        case_test_session_reduction_expr_oracle_compiler
    use ffc_case_test_session_reject_alloc_01_compiler, only: &
        case_test_session_reject_alloc_01_compiler
    use ffc_case_test_session_reject_alloc_02_compiler, only: &
        case_test_session_reject_alloc_02_compiler
    use ffc_case_test_session_reject_array_01_compiler, only: &
        case_test_session_reject_array_01_compiler
    use ffc_case_test_session_reject_array_02_compiler, only: &
        case_test_session_reject_array_02_compiler
    use ffc_case_test_session_reject_assumed_size_order_compiler, only: &
        case_test_session_reject_assumed_size_order_compiler
    use ffc_case_test_session_reject_automatic_scope_compiler, only: &
        case_test_session_reject_automatic_scope_compiler
    use ffc_case_test_session_reject_bit_intrinsic_range_compiler, only: &
        case_test_session_reject_bit_intrinsic_range_compiler
    use ffc_case_test_session_reject_boz_01_compiler, only: &
        case_test_session_reject_boz_01_compiler
    use ffc_case_test_session_reject_boz_array_constructor_compiler, only: &
        case_test_session_reject_boz_array_constructor_compiler
    use ffc_case_test_session_reject_c_pointer_01_compiler, only: &
        case_test_session_reject_c_pointer_01_compiler
    use ffc_case_test_session_reject_charlen_01_compiler, only: &
        case_test_session_reject_charlen_01_compiler
    use ffc_case_test_session_reject_const_01_compiler, only: &
        case_test_session_reject_const_01_compiler
    use ffc_case_test_session_reject_data_01_compiler, only: &
        case_test_session_reject_data_01_compiler
    use ffc_case_test_session_reject_decl_02_compiler, only: &
        case_test_session_reject_decl_02_compiler
    use ffc_case_test_session_reject_derived_01_compiler, only: &
        case_test_session_reject_derived_01_compiler
    use ffc_case_test_session_reject_derived_type_name_compiler, only: &
        case_test_session_reject_derived_type_name_compiler
    use ffc_case_test_session_reject_format_01_compiler, only: &
        case_test_session_reject_format_01_compiler
    use ffc_case_test_session_reject_generic_01_compiler, only: &
        case_test_session_reject_generic_01_compiler
    use ffc_case_test_session_reject_io_01_compiler, only: &
        case_test_session_reject_io_01_compiler
    use ffc_case_test_session_reject_namelist_01_compiler, only: &
        case_test_session_reject_namelist_01_compiler
    use ffc_case_test_session_reject_pointer_01_compiler, only: &
        case_test_session_reject_pointer_01_compiler
    use ffc_case_test_session_reject_purity_01_compiler, only: &
        case_test_session_reject_purity_01_compiler
    use ffc_case_test_session_reject_result_01_compiler, only: &
        case_test_session_reject_result_01_compiler
    use ffc_case_test_session_reject_round2_compiler, only: &
        case_test_session_reject_round2_compiler
    use ffc_case_test_session_reject_storage_01_compiler, only: &
        case_test_session_reject_storage_01_compiler
    use ffc_case_test_session_reject_text_helpers, only: &
        case_test_session_reject_text_helpers
    use ffc_case_test_session_reshape_compiler, only: &
        case_test_session_reshape_compiler
    use ffc_case_test_session_reshape_rank4_compiler, only: &
        case_test_session_reshape_rank4_compiler
    implicit none
    private
    public :: run_group
contains
    subroutine run_group(name, matched)
        character(len=*), intent(in) :: name
        logical, intent(out) :: matched

        matched = .true.
        select case (name)
        case ("test_session_real_variable_compiler")
            call case_test_session_real_variable_compiler()
        case ("test_session_recursive_compiler")
            call case_test_session_recursive_compiler()
        case ("test_session_recursive_function_compiler")
            call case_test_session_recursive_function_compiler()
        case ("test_session_reduction_expr_oracle_compiler")
            call case_test_session_reduction_expr_oracle_compiler()
        case ("test_session_reject_alloc_01_compiler")
            call case_test_session_reject_alloc_01_compiler()
        case ("test_session_reject_alloc_02_compiler")
            call case_test_session_reject_alloc_02_compiler()
        case ("test_session_reject_array_01_compiler")
            call case_test_session_reject_array_01_compiler()
        case ("test_session_reject_array_02_compiler")
            call case_test_session_reject_array_02_compiler()
        case ("test_session_reject_assumed_size_order_compiler")
            call case_test_session_reject_assumed_size_order_compiler()
        case ("test_session_reject_automatic_scope_compiler")
            call case_test_session_reject_automatic_scope_compiler()
        case ("test_session_reject_bit_intrinsic_range_compiler")
            call case_test_session_reject_bit_intrinsic_range_compiler()
        case ("test_session_reject_boz_01_compiler")
            call case_test_session_reject_boz_01_compiler()
        case ("test_session_reject_boz_array_constructor_compiler")
            call case_test_session_reject_boz_array_constructor_compiler()
        case ("test_session_reject_c_pointer_01_compiler")
            call case_test_session_reject_c_pointer_01_compiler()
        case ("test_session_reject_charlen_01_compiler")
            call case_test_session_reject_charlen_01_compiler()
        case ("test_session_reject_const_01_compiler")
            call case_test_session_reject_const_01_compiler()
        case ("test_session_reject_data_01_compiler")
            call case_test_session_reject_data_01_compiler()
        case ("test_session_reject_decl_02_compiler")
            call case_test_session_reject_decl_02_compiler()
        case ("test_session_reject_derived_01_compiler")
            call case_test_session_reject_derived_01_compiler()
        case ("test_session_reject_derived_type_name_compiler")
            call case_test_session_reject_derived_type_name_compiler()
        case ("test_session_reject_format_01_compiler")
            call case_test_session_reject_format_01_compiler()
        case ("test_session_reject_generic_01_compiler")
            call case_test_session_reject_generic_01_compiler()
        case ("test_session_reject_io_01_compiler")
            call case_test_session_reject_io_01_compiler()
        case ("test_session_reject_namelist_01_compiler")
            call case_test_session_reject_namelist_01_compiler()
        case ("test_session_reject_pointer_01_compiler")
            call case_test_session_reject_pointer_01_compiler()
        case ("test_session_reject_purity_01_compiler")
            call case_test_session_reject_purity_01_compiler()
        case ("test_session_reject_result_01_compiler")
            call case_test_session_reject_result_01_compiler()
        case ("test_session_reject_round2_compiler")
            call case_test_session_reject_round2_compiler()
        case ("test_session_reject_storage_01_compiler")
            call case_test_session_reject_storage_01_compiler()
        case ("test_session_reject_text_helpers")
            call case_test_session_reject_text_helpers()
        case ("test_session_reshape_compiler")
            call case_test_session_reshape_compiler()
        case ("test_session_reshape_rank4_compiler")
            call case_test_session_reshape_rank4_compiler()
        case default
            matched = .false.
        end select
    end subroutine run_group
end module ffc_suite_group_12
