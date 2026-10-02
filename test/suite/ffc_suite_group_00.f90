module ffc_suite_group_00
    use ffc_case_test_array_descriptor_layout, only: &
        case_test_array_descriptor_layout
    use ffc_case_test_character_descriptor_layout, only: &
        case_test_character_descriptor_layout
    use ffc_case_test_cli_version_help, only: &
        case_test_cli_version_help
    use ffc_case_test_concat_literal_and_deferred, only: &
        case_test_concat_literal_and_deferred
    use ffc_case_test_concat_three_distinct_deferred_variables, only: &
        case_test_concat_three_distinct_deferred_variables
    use ffc_case_test_conformance_execution_evidence, only: &
        case_test_conformance_execution_evidence
    use ffc_case_test_conformance_flake_detection, only: &
        case_test_conformance_flake_detection
    use ffc_case_test_conformance_gauntlet_smoke, only: &
        case_test_conformance_gauntlet_smoke
    use ffc_case_test_conformance_include_snapshot, only: &
        case_test_conformance_include_snapshot
    use ffc_case_test_conformance_isolation, only: &
        case_test_conformance_isolation
    use ffc_case_test_conformance_manifest_validation, only: &
        case_test_conformance_manifest_validation
    use ffc_case_test_conformance_observation_once, only: &
        case_test_conformance_observation_once
    use ffc_case_test_conformance_oracles, only: &
        case_test_conformance_oracles
    use ffc_case_test_conformance_report_worktree, only: &
        case_test_conformance_report_worktree
    use ffc_case_test_conformance_sampling, only: &
        case_test_conformance_sampling
    use ffc_case_test_conformance_shard_merge, only: &
        case_test_conformance_shard_merge
    use ffc_case_test_counted_do_compiler, only: &
        case_test_counted_do_compiler
    use ffc_case_test_counted_do_negative_step_compiler, only: &
        case_test_counted_do_negative_step_compiler
    use ffc_case_test_fortfront_corpus_conformance, only: &
        case_test_fortfront_corpus_conformance
    use ffc_case_test_liric_memory_submodule_api, only: &
        case_test_liric_memory_submodule_api
    use ffc_case_test_liric_session_bindings, only: &
        case_test_liric_session_bindings
    use ffc_case_test_minimal_ext1, only: &
        case_test_minimal_ext1
    use ffc_case_test_parity_dashboard, only: &
        case_test_parity_dashboard
    use ffc_case_test_polymorphic_descriptor_layout, only: &
        case_test_polymorphic_descriptor_layout
    use ffc_case_test_rank2_loop_debug, only: &
        case_test_rank2_loop_debug
    use ffc_case_test_runtime_archives, only: &
        case_test_runtime_archives
    use ffc_case_test_runtime_link_compiler, only: &
        case_test_runtime_link_compiler
    use ffc_case_test_runtime_mn, only: &
        case_test_runtime_mn
    use ffc_case_test_scalar_kind_engine_is_sole_authority, only: &
        case_test_scalar_kind_engine_is_sole_authority
    use ffc_case_test_self_concat_appends_literal, only: &
        case_test_self_concat_appends_literal
    use ffc_case_test_self_concat_three_times, only: &
        case_test_self_concat_three_times
    use ffc_case_test_session_abstract_interface_compiler, only: &
        case_test_session_abstract_interface_compiler
    implicit none
    private
    public :: run_group
contains
    subroutine run_group(name, matched)
        character(len=*), intent(in) :: name
        logical, intent(out) :: matched

        matched = .true.
        select case (name)
        case ("test_array_descriptor_layout")
            call case_test_array_descriptor_layout()
        case ("test_character_descriptor_layout")
            call case_test_character_descriptor_layout()
        case ("test_cli_version_help")
            call case_test_cli_version_help()
        case ("test_concat_literal_and_deferred")
            call case_test_concat_literal_and_deferred()
        case ("test_concat_three_distinct_deferred_variables")
            call case_test_concat_three_distinct_deferred_variables()
        case ("test_conformance_execution_evidence")
            call case_test_conformance_execution_evidence()
        case ("test_conformance_flake_detection")
            call case_test_conformance_flake_detection()
        case ("test_conformance_gauntlet_smoke")
            call case_test_conformance_gauntlet_smoke()
        case ("test_conformance_include_snapshot")
            call case_test_conformance_include_snapshot()
        case ("test_conformance_isolation")
            call case_test_conformance_isolation()
        case ("test_conformance_manifest_validation")
            call case_test_conformance_manifest_validation()
        case ("test_conformance_observation_once")
            call case_test_conformance_observation_once()
        case ("test_conformance_oracles")
            call case_test_conformance_oracles()
        case ("test_conformance_report_worktree")
            call case_test_conformance_report_worktree()
        case ("test_conformance_sampling")
            call case_test_conformance_sampling()
        case ("test_conformance_shard_merge")
            call case_test_conformance_shard_merge()
        case ("test_counted_do_compiler")
            call case_test_counted_do_compiler()
        case ("test_counted_do_negative_step_compiler")
            call case_test_counted_do_negative_step_compiler()
        case ("test_fortfront_corpus_conformance")
            call case_test_fortfront_corpus_conformance()
        case ("test_liric_memory_submodule_api")
            call case_test_liric_memory_submodule_api()
        case ("test_liric_session_bindings")
            call case_test_liric_session_bindings()
        case ("test_minimal_ext1")
            call case_test_minimal_ext1()
        case ("test_parity_dashboard")
            call case_test_parity_dashboard()
        case ("test_polymorphic_descriptor_layout")
            call case_test_polymorphic_descriptor_layout()
        case ("test_rank2_loop_debug")
            call case_test_rank2_loop_debug()
        case ("test_runtime_archives")
            call case_test_runtime_archives()
        case ("test_runtime_link_compiler")
            call case_test_runtime_link_compiler()
        case ("test_runtime_mn")
            call case_test_runtime_mn()
        case ("test_scalar_kind_engine_is_sole_authority")
            call case_test_scalar_kind_engine_is_sole_authority()
        case ("test_self_concat_appends_literal")
            call case_test_self_concat_appends_literal()
        case ("test_self_concat_three_times")
            call case_test_self_concat_three_times()
        case ("test_session_abstract_interface_compiler")
            call case_test_session_abstract_interface_compiler()
        case default
            matched = .false.
        end select
    end subroutine run_group
end module ffc_suite_group_00
