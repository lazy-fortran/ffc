program test_array_intrinsics
    use iso_fortran_env, only: error_unit
    use mlir_backend
    use backend_factory
    use backend_interface
    use fortfront, only: ast_arena_t, create_ast_arena, LITERAL_INTEGER
    use ast_factory
    use mlir_utils, only: int_to_str
    implicit none

    logical :: all_tests_passed

    print *, "=== Array Intrinsics Tests ==="
    print *

    all_tests_passed = .true.

    ! Run all array intrinsic tests
    if (.not. test_size_intrinsic()) all_tests_passed = .false.
    if (.not. test_shape_intrinsic()) all_tests_passed = .false.
    if (.not. test_lbound_ubound_intrinsics()) all_tests_passed = .false.
    if (.not. test_sum_intrinsic()) all_tests_passed = .false.
    if (.not. test_product_intrinsic()) all_tests_passed = .false.
    if (.not. test_maxval_minval_intrinsics()) all_tests_passed = .false.

    print *
    if (all_tests_passed) then
        print *, "All array intrinsic tests passed!"
        stop 0
    else
        print *, "Some array intrinsic tests failed!"
        stop 1
    end if

contains

    function test_size_intrinsic() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: arr_id, size_call_idx, assign_idx, prog_idx, n_id

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        error_msg = ""

        ! Test: n = size(arr)
        arena = create_ast_arena()

        ! Create identifiers
        arr_id = push_identifier(arena, "arr")
        n_id = push_identifier(arena, "n")

        ! Create size intrinsic call
        size_call_idx = push_call_or_subscript(arena, "size", [arr_id])

        ! Create assignment
        assign_idx = push_assignment(arena, n_id, size_call_idx)
        prog_idx = push_program(arena, "test", [assign_idx])

        ! Generate MLIR code
        options%compile_mode = .true.
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0 .and. allocated(output)) then
            ! Check for proper size intrinsic generation
            if (index(output, "memref.dim") > 0 .or. &
                index(output, "shape.shape_of") > 0 .or. &
                index(output, "tensor.dim") > 0) then
                print *, "PASS: Size intrinsic generates proper MLIR"
                passed = .true.
            else
                print *, "FAIL: Missing proper size intrinsic generation"
                print *, "Expected: memref.dim or shape operations"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating MLIR: ", trim(error_msg)
        end if
    end function test_size_intrinsic

    function test_shape_intrinsic() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: arr_id, shape_call_idx, assign_idx, prog_idx, s_id

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        error_msg = ""

        ! Test: s = shape(matrix)
        arena = create_ast_arena()

        ! Create identifiers
        arr_id = push_identifier(arena, "matrix")
        s_id = push_identifier(arena, "s")

        ! Create shape intrinsic call
        shape_call_idx = push_call_or_subscript(arena, "shape", [arr_id])

        ! Create assignment
        assign_idx = push_assignment(arena, s_id, shape_call_idx)
        prog_idx = push_program(arena, "test", [assign_idx])

        ! Generate MLIR code
        options%compile_mode = .true.
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0 .and. allocated(output)) then
            ! Check for proper shape intrinsic generation
            if (index(output, "shape.shape_of") > 0 .or. &
                index(output, "memref.rank") > 0 .or. &
                index(output, "memref.dim") > 0) then
                print *, "PASS: Shape intrinsic generates proper MLIR"
                passed = .true.
            else
                print *, "FAIL: Missing proper shape intrinsic generation"
                print *, "Expected: shape.shape_of or rank/dim operations"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating MLIR: ", trim(error_msg)
        end if
    end function test_shape_intrinsic

    function test_lbound_ubound_intrinsics() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: arr_id, lbound_idx, ubound_idx, assign1_idx, assign2_idx
        integer :: prog_idx, lb_id, ub_id, dim_idx

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        error_msg = ""

        ! Test: lb = lbound(arr, 1); ub = ubound(arr, 1)
        arena = create_ast_arena()

        ! Create identifiers and literals
        arr_id = push_identifier(arena, "arr")
        lb_id = push_identifier(arena, "lb")
        ub_id = push_identifier(arena, "ub")
        dim_idx = push_literal(arena, "1", LITERAL_INTEGER)

        ! Create lbound/ubound calls
        lbound_idx = push_call_or_subscript(arena, "lbound", [arr_id, dim_idx])
        ubound_idx = push_call_or_subscript(arena, "ubound", [arr_id, dim_idx])

        ! Create assignments
        assign1_idx = push_assignment(arena, lb_id, lbound_idx)
        assign2_idx = push_assignment(arena, ub_id, ubound_idx)
        prog_idx = push_program(arena, "test", [assign1_idx, assign2_idx])

        ! Generate MLIR code
        options%compile_mode = .true.
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0 .and. allocated(output)) then
            ! Check for proper bound intrinsic generation
            if ((index(output, "arith.constant 1") > 0 .or. &
                 index(output, "arith.constant 0") > 0) .and. &
                (index(output, "memref.dim") > 0 .or. &
                 index(output, "tensor.dim") > 0)) then
                print *, "PASS: Lbound/ubound intrinsics generate proper MLIR"
                passed = .true.
            else
                print *, "FAIL: Missing proper bound intrinsic generation"
                print *, "Expected: constants for bounds and dim operations"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating MLIR: ", trim(error_msg)
        end if
    end function test_lbound_ubound_intrinsics

    function test_sum_intrinsic() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: arr_id, sum_call_idx, assign_idx, prog_idx, total_id

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        error_msg = ""

        ! Test: total = sum(arr)
        arena = create_ast_arena()

        ! Create identifiers
        arr_id = push_identifier(arena, "arr")
        total_id = push_identifier(arena, "total")

        ! Create sum intrinsic call
        sum_call_idx = push_call_or_subscript(arena, "sum", [arr_id])

        ! Create assignment
        assign_idx = push_assignment(arena, total_id, sum_call_idx)
        prog_idx = push_program(arena, "test", [assign_idx])

        ! Generate MLIR code
        options%compile_mode = .true.
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0 .and. allocated(output)) then
            ! Check for proper sum reduction
            if (index(output, "linalg.reduce") > 0 .or. &
                index(output, "scf.for") > 0 .or. &
                index(output, "arith.addi") > 0 .or. &
                index(output, "arith.addf") > 0) then
                print *, "PASS: Sum intrinsic generates proper MLIR"
                passed = .true.
            else
                print *, "FAIL: Missing proper sum reduction"
                print *, "Expected: linalg.reduce or loop with addition"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating MLIR: ", trim(error_msg)
        end if
    end function test_sum_intrinsic

    function test_product_intrinsic() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: arr_id, prod_call_idx, assign_idx, prog_idx, result_id

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        error_msg = ""

        ! Test: result = product(arr)
        arena = create_ast_arena()

        ! Create identifiers
        arr_id = push_identifier(arena, "arr")
        result_id = push_identifier(arena, "result")

        ! Create product intrinsic call
        prod_call_idx = push_call_or_subscript(arena, "product", [arr_id])

        ! Create assignment
        assign_idx = push_assignment(arena, result_id, prod_call_idx)
        prog_idx = push_program(arena, "test", [assign_idx])

        ! Generate MLIR code
        options%compile_mode = .true.
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0 .and. allocated(output)) then
            ! Check for proper product reduction
            if (index(output, "linalg.reduce") > 0 .or. &
                index(output, "scf.for") > 0 .or. &
                index(output, "arith.muli") > 0 .or. &
                index(output, "arith.mulf") > 0) then
                print *, "PASS: Product intrinsic generates proper MLIR"
                passed = .true.
            else
                print *, "FAIL: Missing proper product reduction"
                print *, "Expected: linalg.reduce or loop with multiplication"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating MLIR: ", trim(error_msg)
        end if
    end function test_product_intrinsic

    function test_maxval_minval_intrinsics() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: arr_id, max_idx, min_idx, assign1_idx, assign2_idx
        integer :: prog_idx, max_id, min_id

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        error_msg = ""

        ! Test: max_val = maxval(arr); min_val = minval(arr)
        arena = create_ast_arena()

        ! Create identifiers
        arr_id = push_identifier(arena, "arr")
        max_id = push_identifier(arena, "max_val")
        min_id = push_identifier(arena, "min_val")

        ! Create maxval/minval calls
        max_idx = push_call_or_subscript(arena, "maxval", [arr_id])
        min_idx = push_call_or_subscript(arena, "minval", [arr_id])

        ! Create assignments
        assign1_idx = push_assignment(arena, max_id, max_idx)
        assign2_idx = push_assignment(arena, min_id, min_idx)
        prog_idx = push_program(arena, "test", [assign1_idx, assign2_idx])

        ! Generate MLIR code
        options%compile_mode = .true.
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0 .and. allocated(output)) then
            ! Check for proper max/min reduction
            if (index(output, "arith.maxsi") > 0 .or. &
                index(output, "arith.minsi") > 0 .or. &
                index(output, "arith.maxf") > 0 .or. &
                index(output, "arith.minf") > 0 .or. &
                index(output, "linalg.reduce") > 0) then
                print *, "PASS: Maxval/minval intrinsics generate proper MLIR"
                passed = .true.
            else
                print *, "FAIL: Missing proper max/min reduction"
                print *, "Expected: arith.max/min operations or linalg.reduce"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating MLIR: ", trim(error_msg)
        end if
    end function test_maxval_minval_intrinsics

end program test_array_intrinsics
