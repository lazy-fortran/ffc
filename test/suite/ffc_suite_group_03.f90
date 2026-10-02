module ffc_suite_group_03
    use ffc_case_test_session_assumed_shape_module_compiler, only: &
        case_test_session_assumed_shape_module_compiler
    use ffc_case_test_session_assumed_shape_rank2_runtime_compiler, only: &
        case_test_session_assumed_shape_rank2_runtime_compiler
    use ffc_case_test_session_assumed_shape_runtime_extent_compiler, only: &
        case_test_session_assumed_shape_runtime_extent_compiler
    use ffc_case_test_session_assumed_shape_section_compiler, only: &
        case_test_session_assumed_shape_section_compiler
    use ffc_case_test_session_assumed_shape_target_compiler, only: &
        case_test_session_assumed_shape_target_compiler
    use ffc_case_test_session_assumed_size_array_compiler, only: &
        case_test_session_assumed_size_array_compiler
    use ffc_case_test_session_bindc_module_scalar_compiler, only: &
        case_test_session_bindc_module_scalar_compiler
    use ffc_case_test_session_block_data_compiler, only: &
        case_test_session_block_data_compiler
    use ffc_case_test_session_block_if_compiler, only: &
        case_test_session_block_if_compiler
    use ffc_case_test_session_block_shadow_compiler, only: &
        case_test_session_block_shadow_compiler
    use ffc_case_test_session_boz_output_compiler, only: &
        case_test_session_boz_output_compiler
    use ffc_case_test_session_c_interop_compiler, only: &
        case_test_session_c_interop_compiler
    use ffc_case_test_session_c_pointer_compiler, only: &
        case_test_session_c_pointer_compiler
    use ffc_case_test_session_c_ptr_module_compiler, only: &
        case_test_session_c_ptr_module_compiler
    use ffc_case_test_session_case_overlap_compiler, only: &
        case_test_session_case_overlap_compiler
    use ffc_case_test_session_char_array_compare_compiler, only: &
        case_test_session_char_array_compare_compiler
    use ffc_case_test_session_char_array_compiler, only: &
        case_test_session_char_array_compiler
    use ffc_case_test_session_char_array_initializer_compiler, only: &
        case_test_session_char_array_initializer_compiler
    use ffc_case_test_session_char_initializer_compiler, only: &
        case_test_session_char_initializer_compiler
    use ffc_case_test_session_char_intrinsic_compare_compiler, only: &
        case_test_session_char_intrinsic_compare_compiler
    use ffc_case_test_session_char_result_compare_compiler, only: &
        case_test_session_char_result_compare_compiler
    use ffc_case_test_session_character_allocatable_compiler, only: &
        case_test_session_character_allocatable_compiler
    use ffc_case_test_session_character_array_compiler, only: &
        case_test_session_character_array_compiler
    use ffc_case_test_session_character_array_rank34_compiler, only: &
        case_test_session_character_array_rank34_compiler
    use ffc_case_test_session_character_fixed_ops_compiler, only: &
        case_test_session_character_fixed_ops_compiler
    use ffc_case_test_session_character_function_result_compiler, only: &
        case_test_session_character_function_result_compiler
    use ffc_case_test_session_character_intrinsics_compiler, only: &
        case_test_session_character_intrinsics_compiler
    use ffc_case_test_session_character_length_runtime_compiler, only: &
        case_test_session_character_length_runtime_compiler
    use ffc_case_test_session_character_literal_print_compiler, only: &
        case_test_session_character_literal_print_compiler
    use ffc_case_test_session_character_locate_compiler, only: &
        case_test_session_character_locate_compiler
    use ffc_case_test_session_character_module_intrinsics_compiler, only: &
        case_test_session_character_module_intrinsics_compiler
    use ffc_case_test_session_character_prefix_compiler, only: &
        case_test_session_character_prefix_compiler
    implicit none
    private
    public :: run_group
contains
    subroutine run_group(name, matched)
        character(len=*), intent(in) :: name
        logical, intent(out) :: matched

        matched = .true.
        select case (name)
        case ("test_session_assumed_shape_module_compiler")
            call case_test_session_assumed_shape_module_compiler()
        case ("test_session_assumed_shape_rank2_runtime_compiler")
            call case_test_session_assumed_shape_rank2_runtime_compiler()
        case ("test_session_assumed_shape_runtime_extent_compiler")
            call case_test_session_assumed_shape_runtime_extent_compiler()
        case ("test_session_assumed_shape_section_compiler")
            call case_test_session_assumed_shape_section_compiler()
        case ("test_session_assumed_shape_target_compiler")
            call case_test_session_assumed_shape_target_compiler()
        case ("test_session_assumed_size_array_compiler")
            call case_test_session_assumed_size_array_compiler()
        case ("test_session_bindc_module_scalar_compiler")
            call case_test_session_bindc_module_scalar_compiler()
        case ("test_session_block_data_compiler")
            call case_test_session_block_data_compiler()
        case ("test_session_block_if_compiler")
            call case_test_session_block_if_compiler()
        case ("test_session_block_shadow_compiler")
            call case_test_session_block_shadow_compiler()
        case ("test_session_boz_output_compiler")
            call case_test_session_boz_output_compiler()
        case ("test_session_c_interop_compiler")
            call case_test_session_c_interop_compiler()
        case ("test_session_c_pointer_compiler")
            call case_test_session_c_pointer_compiler()
        case ("test_session_c_ptr_module_compiler")
            call case_test_session_c_ptr_module_compiler()
        case ("test_session_case_overlap_compiler")
            call case_test_session_case_overlap_compiler()
        case ("test_session_char_array_compare_compiler")
            call case_test_session_char_array_compare_compiler()
        case ("test_session_char_array_compiler")
            call case_test_session_char_array_compiler()
        case ("test_session_char_array_initializer_compiler")
            call case_test_session_char_array_initializer_compiler()
        case ("test_session_char_initializer_compiler")
            call case_test_session_char_initializer_compiler()
        case ("test_session_char_intrinsic_compare_compiler")
            call case_test_session_char_intrinsic_compare_compiler()
        case ("test_session_char_result_compare_compiler")
            call case_test_session_char_result_compare_compiler()
        case ("test_session_character_allocatable_compiler")
            call case_test_session_character_allocatable_compiler()
        case ("test_session_character_array_compiler")
            call case_test_session_character_array_compiler()
        case ("test_session_character_array_rank34_compiler")
            call case_test_session_character_array_rank34_compiler()
        case ("test_session_character_fixed_ops_compiler")
            call case_test_session_character_fixed_ops_compiler()
        case ("test_session_character_function_result_compiler")
            call case_test_session_character_function_result_compiler()
        case ("test_session_character_intrinsics_compiler")
            call case_test_session_character_intrinsics_compiler()
        case ("test_session_character_length_runtime_compiler")
            call case_test_session_character_length_runtime_compiler()
        case ("test_session_character_literal_print_compiler")
            call case_test_session_character_literal_print_compiler()
        case ("test_session_character_locate_compiler")
            call case_test_session_character_locate_compiler()
        case ("test_session_character_module_intrinsics_compiler")
            call case_test_session_character_module_intrinsics_compiler()
        case ("test_session_character_prefix_compiler")
            call case_test_session_character_prefix_compiler()
        case default
            matched = .false.
        end select
    end subroutine run_group
end module ffc_suite_group_03
