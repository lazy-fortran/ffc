program test_lowering_pipeline
    use iso_c_binding
    use mlir_c_core
    use mlir_c_types
    use mlir_c_operations
    use mlir_builder
    use hlfir_dialect
    use fir_dialect
    use standard_dialects
    use pass_manager
    use lowering_pipeline
    implicit none

    logical :: all_tests_passed

    print *, "=== MLIR Lowering Pipeline Tests (C API) ==="
    print *

    all_tests_passed = .true.

    ! Run all tests - these will fail initially (RED phase)
    if (.not. test_hlfir_to_fir_lowering()) all_tests_passed = .false.
    if (.not. test_fir_to_llvm_lowering()) all_tests_passed = .false.
    if (.not. test_optimization_passes()) all_tests_passed = .false.
    if (.not. test_debug_info_preservation()) all_tests_passed = .false.

    print *
    if (all_tests_passed) then
        print *, "All lowering pipeline tests passed!"
        stop 0
    else
        print *, "Some lowering pipeline tests failed!"
        stop 1
    end if

contains

    function test_hlfir_to_fir_lowering() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_builder_t) :: builder
        type(mlir_module_t) :: module
        type(mlir_location_t) :: location
        type(mlir_lowering_pipeline_t) :: pipeline
        type(mlir_operation_t) :: hlfir_op, fir_op
        logical :: success
        
        passed = .true.
        
        ! Create MLIR context and builder
        context = create_mlir_context()
        builder = create_mlir_builder(context)
        location = create_unknown_location(context)
        module = create_empty_module(location)
        
        passed = passed .and. context%is_valid()
        passed = passed .and. builder%is_valid()
        passed = passed .and. module%is_valid()
        
        ! Test: Create lowering pipeline for HLFIR to FIR
        pipeline = create_lowering_pipeline(context, "hlfir-to-fir")
        passed = passed .and. pipeline%is_valid()
        passed = passed .and. pipeline_has_pass(pipeline, "convert-hlfir-to-fir")
        
        ! Create HLFIR operation to test lowering
        hlfir_op = create_test_hlfir_operation(builder)
        passed = passed .and. hlfir_op%is_valid()
        passed = passed .and. is_hlfir_operation(hlfir_op)
        
        ! Test: Apply lowering pipeline
        success = apply_lowering_pipeline(pipeline, module)
        passed = passed .and. success
        
        ! Verify lowering produced FIR operations
        fir_op = get_lowered_operation(module, hlfir_op)
        passed = passed .and. fir_op%is_valid()
        passed = passed .and. is_fir_operation(fir_op)
        passed = passed .and. (.not. has_hlfir_operations(module))
        
        if (passed) then
            print *, "PASS: test_hlfir_to_fir_lowering"
        else
            print *, "FAIL: test_hlfir_to_fir_lowering"
        end if
        
        ! Cleanup
        call destroy_lowering_pipeline(pipeline)
        call destroy_operation(hlfir_op)
        call destroy_mlir_builder(builder)
        call destroy_mlir_context(context)
    end function test_hlfir_to_fir_lowering

    function test_fir_to_llvm_lowering() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_builder_t) :: builder
        type(mlir_module_t) :: module
        type(mlir_location_t) :: location
        type(mlir_lowering_pipeline_t) :: pipeline
        type(mlir_operation_t) :: fir_op, llvm_op
        logical :: success
        
        passed = .true.
        
        ! Create MLIR context and builder
        context = create_mlir_context()
        builder = create_mlir_builder(context)
        location = create_unknown_location(context)
        module = create_empty_module(location)
        
        passed = passed .and. context%is_valid()
        passed = passed .and. builder%is_valid()
        passed = passed .and. module%is_valid()
        
        ! Test: Create lowering pipeline for FIR to LLVM
        pipeline = create_lowering_pipeline(context, "fir-to-llvm")
        passed = passed .and. pipeline%is_valid()
        passed = passed .and. pipeline_has_pass(pipeline, "convert-fir-to-llvm")
        
        ! Create FIR operation to test lowering
        fir_op = create_test_fir_operation(builder)
        passed = passed .and. fir_op%is_valid()
        passed = passed .and. is_fir_operation(fir_op)
        
        ! Test: Configure target triple and data layout
        call configure_target_info(pipeline, "x86_64-unknown-linux-gnu")
        passed = passed .and. pipeline_has_target_info(pipeline)
        
        ! Test: Apply lowering pipeline
        success = apply_lowering_pipeline(pipeline, module)
        passed = passed .and. success
        
        ! Verify lowering produced LLVM operations
        llvm_op = get_lowered_operation(module, fir_op)
        passed = passed .and. llvm_op%is_valid()
        passed = passed .and. is_llvm_operation(llvm_op)
        passed = passed .and. (.not. has_fir_operations(module))
        
        if (passed) then
            print *, "PASS: test_fir_to_llvm_lowering"
        else
            print *, "FAIL: test_fir_to_llvm_lowering"
        end if
        
        ! Cleanup
        call destroy_lowering_pipeline(pipeline)
        call destroy_operation(fir_op)
        call destroy_mlir_builder(builder)
        call destroy_mlir_context(context)
    end function test_fir_to_llvm_lowering

    function test_optimization_passes() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_builder_t) :: builder
        type(mlir_module_t) :: module, optimized_module
        type(mlir_location_t) :: location
        type(mlir_lowering_pipeline_t) :: pipeline
        character(len=256), dimension(5) :: optimization_passes
        logical :: success
        integer :: initial_op_count, optimized_op_count
        
        passed = .true.
        
        ! Create MLIR context and builder
        context = create_mlir_context()
        builder = create_mlir_builder(context)
        location = create_unknown_location(context)
        module = create_empty_module(location)
        
        passed = passed .and. context%is_valid()
        passed = passed .and. builder%is_valid()
        passed = passed .and. module%is_valid()
        
        ! Test: Create optimization pipeline
        optimization_passes = [character(len=256) :: &
            "canonicalize", "cse", "sccp", "loop-invariant-code-motion", "inline"]
        
        pipeline = create_optimization_pipeline(context, optimization_passes)
        passed = passed .and. pipeline%is_valid()
        
        ! Verify all optimization passes are registered
        passed = passed .and. pipeline_has_passes(pipeline, optimization_passes)
        passed = passed .and. pipeline_pass_count(pipeline) >= 5
        
        ! Create test module with optimizable patterns
        call create_optimizable_test_module(builder, module)
        initial_op_count = count_operations(module)
        passed = passed .and. initial_op_count > 0
        
        ! Test: Set optimization level
        call set_optimization_level(pipeline, 2)  ! -O2
        passed = passed .and. pipeline_has_optimization_level(pipeline, 2)
        
        ! Test: Apply optimization pipeline
        success = apply_lowering_pipeline(pipeline, module)
        passed = passed .and. success
        
        ! Verify optimizations were applied
        optimized_op_count = count_operations(module)
        passed = passed .and. optimized_op_count < initial_op_count  ! Some ops eliminated
        passed = passed .and. module_is_optimized(module)
        
        ! Test: Verify specific optimizations
        passed = passed .and. (.not. has_dead_code(module))
        passed = passed .and. (.not. has_redundant_operations(module))
        
        if (passed) then
            print *, "PASS: test_optimization_passes"
        else
            print *, "FAIL: test_optimization_passes"
        end if
        
        ! Cleanup
        call destroy_lowering_pipeline(pipeline)
        call destroy_mlir_builder(builder)
        call destroy_mlir_context(context)
    end function test_optimization_passes

    function test_debug_info_preservation() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_builder_t) :: builder
        type(mlir_module_t) :: module
        type(mlir_location_t) :: location, file_location
        type(mlir_lowering_pipeline_t) :: pipeline
        type(mlir_operation_t) :: op_with_debug
        logical :: success
        character(len=256) :: debug_info
        
        passed = .true.
        
        ! Create MLIR context and builder
        context = create_mlir_context()
        builder = create_mlir_builder(context)
        location = create_unknown_location(context)
        module = create_empty_module(location)
        
        passed = passed .and. context%is_valid()
        passed = passed .and. builder%is_valid()
        passed = passed .and. module%is_valid()
        
        ! Test: Create complete lowering pipeline with debug info preservation
        pipeline = create_complete_lowering_pipeline(context)
        passed = passed .and. pipeline%is_valid()
        
        ! Enable debug info preservation
        call enable_debug_info_preservation(pipeline)
        passed = passed .and. pipeline_preserves_debug_info(pipeline)
        
        ! Create operation with debug info
        file_location = create_file_location(context, "test.f90", 42, 10)
        op_with_debug = create_operation_with_location(builder, file_location)
        passed = passed .and. op_with_debug%is_valid()
        passed = passed .and. operation_has_debug_info(op_with_debug)
        
        ! Test: Apply complete lowering (HLFIR -> FIR -> LLVM)
        success = apply_lowering_pipeline(pipeline, module)
        passed = passed .and. success
        
        ! Verify debug info is preserved through lowering
        debug_info = get_operation_debug_info(op_with_debug)
        passed = passed .and. len_trim(debug_info) > 0
        passed = passed .and. index(debug_info, "test.f90") > 0
        passed = passed .and. index(debug_info, "42") > 0
        
        ! Test: Verify DWARF debug info generation
        passed = passed .and. module_has_dwarf_info(module)
        passed = passed .and. verify_debug_info_integrity(module)
        
        if (passed) then
            print *, "PASS: test_debug_info_preservation"
        else
            print *, "FAIL: test_debug_info_preservation"
        end if
        
        ! Cleanup
        call destroy_lowering_pipeline(pipeline)
        call destroy_operation(op_with_debug)
        call destroy_mlir_builder(builder)
        call destroy_mlir_context(context)
    end function test_debug_info_preservation


end program test_lowering_pipeline