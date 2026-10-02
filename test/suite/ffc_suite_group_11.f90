module ffc_suite_group_11
    use ffc_case_test_session_pointer_scalar_compiler, only: &
        case_test_session_pointer_scalar_compiler
    use ffc_case_test_session_pointer_scalar_kinds_compiler, only: &
        case_test_session_pointer_scalar_kinds_compiler
    use ffc_case_test_session_pointer_section_rank4_compiler, only: &
        case_test_session_pointer_section_rank4_compiler
    use ffc_case_test_session_polymorphic_allocatable_array_compiler, only: &
        case_test_session_polymorphic_allocatable_array_compiler
    use ffc_case_test_session_polymorphic_array_compiler, only: &
        case_test_session_polymorphic_array_compiler
    use ffc_case_test_session_print_runtime_compiler, only: &
        case_test_session_print_runtime_compiler
    use ffc_case_5c140e4c93b8d4c194246fb3, only: &
        case_test_session_proc_pointer_component_associated_compiler
    use ffc_case_test_session_proc_ptr_scalar_f32_compiler, only: &
        case_test_session_proc_ptr_scalar_f32_compiler
    use ffc_case_test_session_proc_ptr_scalar_f64_compiler, only: &
        case_test_session_proc_ptr_scalar_f64_compiler
    use ffc_case_test_session_procedure_dummy_argument_compiler, only: &
        case_test_session_procedure_dummy_argument_compiler
    use ffc_case_test_session_program_units_compiler, only: &
        case_test_session_program_units_compiler
    use ffc_case_test_session_random_number_compiler, only: &
        case_test_session_random_number_compiler
    use ffc_case_test_session_random_seed_compiler, only: &
        case_test_session_random_seed_compiler
    use ffc_case_test_session_rank2_array_compiler, only: &
        case_test_session_rank2_array_compiler
    use ffc_case_test_session_rank3_array_compiler, only: &
        case_test_session_rank3_array_compiler
    use ffc_case_test_session_rank_mismatch_arg_compiler, only: &
        case_test_session_rank_mismatch_arg_compiler
    use ffc_case_test_session_read_fmod_compiler, only: &
        case_test_session_read_fmod_compiler
    use ffc_case_test_session_read_stdin_compiler, only: &
        case_test_session_read_stdin_compiler
    use ffc_case_test_session_real8_function_compiler, only: &
        case_test_session_real8_function_compiler
    use ffc_case_test_session_real_allocatable_compiler, only: &
        case_test_session_real_allocatable_compiler
    use ffc_case_test_session_real_array_b1f_compiler, only: &
        case_test_session_real_array_b1f_compiler
    use ffc_case_test_session_real_array_compiler, only: &
        case_test_session_real_array_compiler
    use ffc_case_test_session_real_conversion_kind_compiler, only: &
        case_test_session_real_conversion_kind_compiler
    use ffc_case_test_session_real_inquiry_compiler, only: &
        case_test_session_real_inquiry_compiler
    use ffc_case_test_session_real_kind_expr_compiler, only: &
        case_test_session_real_kind_expr_compiler
    use ffc_case_test_session_real_list_directed_compiler, only: &
        case_test_session_real_list_directed_compiler
    use ffc_case_test_session_real_literal_print_compiler, only: &
        case_test_session_real_literal_print_compiler
    use ffc_case_test_session_real_loop_accumulation_compiler, only: &
        case_test_session_real_loop_accumulation_compiler
    use ffc_case_test_session_real_parameter_compiler, only: &
        case_test_session_real_parameter_compiler
    use ffc_case_test_session_real_pow_compiler, only: &
        case_test_session_real_pow_compiler
    use ffc_case_test_session_real_to_integer_literal_compiler, only: &
        case_test_session_real_to_integer_literal_compiler
    use ffc_case_test_session_real_transcendental_compiler, only: &
        case_test_session_real_transcendental_compiler
    implicit none
    private
    public :: run_group
contains
    subroutine run_group(name, matched)
        character(len=*), intent(in) :: name
        logical, intent(out) :: matched

        matched = .true.
        select case (name)
        case ("test_session_pointer_scalar_compiler")
            call case_test_session_pointer_scalar_compiler()
        case ("test_session_pointer_scalar_kinds_compiler")
            call case_test_session_pointer_scalar_kinds_compiler()
        case ("test_session_pointer_section_rank4_compiler")
            call case_test_session_pointer_section_rank4_compiler()
        case ("test_session_polymorphic_allocatable_array_compiler")
            call case_test_session_polymorphic_allocatable_array_compiler()
        case ("test_session_polymorphic_array_compiler")
            call case_test_session_polymorphic_array_compiler()
        case ("test_session_print_runtime_compiler")
            call case_test_session_print_runtime_compiler()
        case ("test_session_proc_pointer_component_associated_compiler")
            call case_test_session_proc_pointer_component_associated_compiler()
        case ("test_session_proc_ptr_scalar_f32_compiler")
            call case_test_session_proc_ptr_scalar_f32_compiler()
        case ("test_session_proc_ptr_scalar_f64_compiler")
            call case_test_session_proc_ptr_scalar_f64_compiler()
        case ("test_session_procedure_dummy_argument_compiler")
            call case_test_session_procedure_dummy_argument_compiler()
        case ("test_session_program_units_compiler")
            call case_test_session_program_units_compiler()
        case ("test_session_random_number_compiler")
            call case_test_session_random_number_compiler()
        case ("test_session_random_seed_compiler")
            call case_test_session_random_seed_compiler()
        case ("test_session_rank2_array_compiler")
            call case_test_session_rank2_array_compiler()
        case ("test_session_rank3_array_compiler")
            call case_test_session_rank3_array_compiler()
        case ("test_session_rank_mismatch_arg_compiler")
            call case_test_session_rank_mismatch_arg_compiler()
        case ("test_session_read_fmod_compiler")
            call case_test_session_read_fmod_compiler()
        case ("test_session_read_stdin_compiler")
            call case_test_session_read_stdin_compiler()
        case ("test_session_real8_function_compiler")
            call case_test_session_real8_function_compiler()
        case ("test_session_real_allocatable_compiler")
            call case_test_session_real_allocatable_compiler()
        case ("test_session_real_array_b1f_compiler")
            call case_test_session_real_array_b1f_compiler()
        case ("test_session_real_array_compiler")
            call case_test_session_real_array_compiler()
        case ("test_session_real_conversion_kind_compiler")
            call case_test_session_real_conversion_kind_compiler()
        case ("test_session_real_inquiry_compiler")
            call case_test_session_real_inquiry_compiler()
        case ("test_session_real_kind_expr_compiler")
            call case_test_session_real_kind_expr_compiler()
        case ("test_session_real_list_directed_compiler")
            call case_test_session_real_list_directed_compiler()
        case ("test_session_real_literal_print_compiler")
            call case_test_session_real_literal_print_compiler()
        case ("test_session_real_loop_accumulation_compiler")
            call case_test_session_real_loop_accumulation_compiler()
        case ("test_session_real_parameter_compiler")
            call case_test_session_real_parameter_compiler()
        case ("test_session_real_pow_compiler")
            call case_test_session_real_pow_compiler()
        case ("test_session_real_to_integer_literal_compiler")
            call case_test_session_real_to_integer_literal_compiler()
        case ("test_session_real_transcendental_compiler")
            call case_test_session_real_transcendental_compiler()
        case default
            matched = .false.
        end select
    end subroutine run_group
end module ffc_suite_group_11
