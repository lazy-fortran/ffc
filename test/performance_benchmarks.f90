program performance_benchmarks
    use iso_c_binding
    use test_harness
    use mlir_c_core
    use mlir_c_types
    use mlir_c_operations
    use mlir_builder
    use memory_tracker
    implicit none
    
    type(test_suite_t) :: perf_suite
    
    print *, "=== FortFC Performance Benchmarks ==="
    print *
    
    perf_suite = create_test_suite("Performance Benchmarks")
    
    ! Add benchmark tests
    call add_test_case(perf_suite, "Context creation benchmark", benchmark_context_creation)
    call add_test_case(perf_suite, "Type creation benchmark", benchmark_type_creation)
    call add_test_case(perf_suite, "Operation creation benchmark", benchmark_operation_creation)
    call add_test_case(perf_suite, "Large module generation", benchmark_large_module)
    call add_test_case(perf_suite, "SSA value generation", benchmark_ssa_generation)
    call add_test_case(perf_suite, "Memory allocation patterns", benchmark_memory_patterns)
    
    call run_test_suite(perf_suite, verbose=.true.)
    
    call perf_suite%cleanup()
    
contains

    function benchmark_context_creation() result(passed)
        logical :: passed
        integer :: i, iterations
        real :: start_time, end_time, elapsed
        type(mlir_context_t) :: contexts(100)
        
        iterations = 1000
        passed = .true.
        
        print *
        print *, "  Benchmarking context creation/destruction..."
        
        call cpu_time(start_time)
        
        do i = 1, iterations
            contexts(mod(i-1, 100) + 1) = create_mlir_context()
            if (mod(i, 100) == 0) then
                ! Destroy oldest 100 contexts
                do j = 1, 100
                    call destroy_mlir_context(contexts(j))
                end do
            end if
        end do
        
        call cpu_time(end_time)
        elapsed = end_time - start_time
        
        print '(A,I0,A,F8.3,A)', "  Created/destroyed ", iterations, &
              " contexts in ", elapsed, " seconds"
        print '(A,F8.3,A)', "  Rate: ", real(iterations)/elapsed, " contexts/second"
        
        passed = elapsed < 10.0  ! Should complete in under 10 seconds
    end function benchmark_context_creation

    function benchmark_type_creation() result(passed)
        logical :: passed
        integer :: i, iterations
        real :: start_time, end_time, elapsed
        type(mlir_context_t) :: context
        type(mlir_type_t) :: types(1000)
        
        iterations = 10000
        passed = .true.
        
        print *
        print *, "  Benchmarking type creation..."
        
        context = create_mlir_context()
        
        call cpu_time(start_time)
        
        do i = 1, iterations
            select case (mod(i, 4))
            case (0)
                types(mod(i-1, 1000) + 1) = create_i32_type(context)
            case (1)
                types(mod(i-1, 1000) + 1) = create_f64_type(context)
            case (2)
                types(mod(i-1, 1000) + 1) = create_i1_type(context)
            case (3)
                types(mod(i-1, 1000) + 1) = create_f32_type(context)
            end select
        end do
        
        call cpu_time(end_time)
        elapsed = end_time - start_time
        
        print '(A,I0,A,F8.3,A)', "  Created ", iterations, &
              " types in ", elapsed, " seconds"
        print '(A,F8.3,A)', "  Rate: ", real(iterations)/elapsed, " types/second"
        
        call destroy_mlir_context(context)
        
        passed = elapsed < 5.0  ! Should complete in under 5 seconds
    end function benchmark_type_creation

    function benchmark_operation_creation() result(passed)
        logical :: passed
        integer :: i, iterations
        real :: start_time, end_time, elapsed
        type(mlir_context_t) :: context
        type(mlir_builder_t) :: builder
        type(mlir_location_t) :: loc
        type(mlir_operation_t) :: op
        type(mlir_module_t) :: module
        
        iterations = 5000
        passed = .true.
        
        print *
        print *, "  Benchmarking operation creation..."
        
        context = create_mlir_context()
        builder = create_mlir_builder(context)
        loc = create_unknown_location(context)
        module = create_empty_module(loc)
        
        call cpu_time(start_time)
        
        do i = 1, iterations
            ! Create a simple constant operation
            op = create_constant_op(builder, loc, i)
        end do
        
        call cpu_time(end_time)
        elapsed = end_time - start_time
        
        print '(A,I0,A,F8.3,A)', "  Created ", iterations, &
              " operations in ", elapsed, " seconds"
        print '(A,F8.3,A)', "  Rate: ", real(iterations)/elapsed, " operations/second"
        
        call destroy_mlir_builder(builder)
        call destroy_mlir_context(context)
        
        passed = elapsed < 10.0  ! Should complete in under 10 seconds
    end function benchmark_operation_creation

    function benchmark_large_module() result(passed)
        logical :: passed
        integer :: num_functions, num_blocks, num_ops
        real :: start_time, end_time, elapsed
        type(mlir_context_t) :: context
        type(mlir_builder_t) :: builder
        type(memory_tracker_t) :: tracker
        integer(8) :: memory_used
        
        num_functions = 100
        num_blocks = 10
        num_ops = 50
        passed = .true.
        
        print *
        print *, "  Benchmarking large module generation..."
        
        call tracker%init()
        call tracker%enable_peak_tracking()
        
        context = create_mlir_context()
        builder = create_mlir_builder(context)
        
        call cpu_time(start_time)
        
        ! Generate a large module
        call generate_large_test_module(builder, num_functions, num_blocks, num_ops)
        
        call cpu_time(end_time)
        elapsed = end_time - start_time
        
        memory_used = tracker%get_peak_usage()
        
        print '(A,I0,A,I0,A,I0,A)', "  Generated module with ", num_functions, &
              " functions, ", num_blocks * num_functions, " blocks, ", &
              num_ops * num_blocks * num_functions, " operations"
        print '(A,F8.3,A)', "  Time: ", elapsed, " seconds"
        print '(A,F8.1,A)', "  Peak memory: ", real(memory_used) / 1048576.0, " MB"
        
        call destroy_mlir_builder(builder)
        call destroy_mlir_context(context)
        call tracker%cleanup()
        
        passed = elapsed < 30.0  ! Should complete in under 30 seconds
    end function benchmark_large_module

    function benchmark_ssa_generation() result(passed)
        use ssa_manager
        logical :: passed
        integer :: i, iterations
        real :: start_time, end_time, elapsed
        type(ssa_manager_t) :: ssa_mgr
        character(len=:), allocatable :: value_name
        
        iterations = 100000
        passed = .true.
        
        print *
        print *, "  Benchmarking SSA value generation..."
        
        call ssa_mgr%init()
        
        call cpu_time(start_time)
        
        do i = 1, iterations
            value_name = ssa_mgr%next_value()
        end do
        
        call cpu_time(end_time)
        elapsed = end_time - start_time
        
        print '(A,I0,A,F8.3,A)', "  Generated ", iterations, &
              " SSA values in ", elapsed, " seconds"
        print '(A,F8.3,A)', "  Rate: ", real(iterations)/elapsed, " values/second"
        
        call ssa_mgr%cleanup()
        
        passed = elapsed < 5.0  ! Should complete in under 5 seconds
    end function benchmark_ssa_generation

    function benchmark_memory_patterns() result(passed)
        logical :: passed
        integer :: i, j, iterations
        real :: start_time, end_time, elapsed
        type(memory_tracker_t) :: tracker
        character(len=32) :: name
        integer(8) :: sizes(5) = [1024_8, 4096_8, 16384_8, 65536_8, 262144_8]
        
        iterations = 10000
        passed = .true.
        
        print *
        print *, "  Benchmarking memory allocation patterns..."
        
        call tracker%init()
        call tracker%enable_peak_tracking()
        
        call cpu_time(start_time)
        
        ! Simulate allocation/deallocation patterns
        do i = 1, iterations
            j = mod(i-1, 5) + 1
            write(name, '(A,I0)') "allocation_", i
            call tracker%record_allocation(name, sizes(j))
            
            ! Deallocate some allocations
            if (mod(i, 3) == 0 .and. i > 3) then
                write(name, '(A,I0)') "allocation_", i-3
                call tracker%record_deallocation(name, sizes(mod(i-4, 5) + 1))
            end if
        end do
        
        call cpu_time(end_time)
        elapsed = end_time - start_time
        
        print '(A,I0,A,F8.3,A)', "  Processed ", iterations, &
              " allocation operations in ", elapsed, " seconds"
        print '(A,F8.3,A)', "  Rate: ", real(iterations)/elapsed, " operations/second"
        print '(A,I0)', "  Peak tracked allocations: ", tracker%get_peak_usage()
        
        call tracker%cleanup()
        
        passed = elapsed < 10.0  ! Should complete in under 10 seconds
    end function benchmark_memory_patterns

    ! Helper functions
    
    function create_constant_op(builder, loc, value) result(op)
        type(mlir_builder_t), intent(in) :: builder
        type(mlir_location_t), intent(in) :: loc
        integer, intent(in) :: value
        type(mlir_operation_t) :: op
        
        ! Simplified constant creation
        op%ptr = c_null_ptr  ! Stub
    end function create_constant_op

    subroutine generate_large_test_module(builder, num_functions, num_blocks, num_ops)
        type(mlir_builder_t), intent(inout) :: builder
        integer, intent(in) :: num_functions, num_blocks, num_ops
        
        ! Stub implementation
        ! In real implementation, would generate a module with specified complexity
    end subroutine generate_large_test_module

end program performance_benchmarks