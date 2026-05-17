program test_builder_minimal
    use iso_c_binding
    use mlir_c_core
    use mlir_builder
    implicit none

    type(mlir_context_t) :: context
    type(mlir_builder_t) :: builder
    logical :: passed

    print *, "=== Minimal Builder Test ==="

    ! Create context
    context = create_mlir_context()
    
    ! Try to create builder
    builder = create_mlir_builder(context)
    passed = builder%is_valid()
    
    if (passed) then
        print *, "PASS: Builder creation works"
        call destroy_mlir_builder(builder)
    else
        print *, "FAIL: Builder creation failed"
    end if
    
    call destroy_mlir_context(context)
    
    if (passed) then
        stop 0
    else
        stop 1
    end if

end program test_builder_minimal