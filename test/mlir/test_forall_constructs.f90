program test_forall_constructs
    use iso_fortran_env, only: error_unit
    use mlir_backend
    use backend_factory
    use backend_interface
    use ast_core, only: ast_arena_t, create_ast_arena, LITERAL_INTEGER
    use ast_factory
    use mlir_utils, only: int_to_str
    implicit none

    logical :: all_tests_passed

    print *, "=== Forall Construct Tests ==="
    print *

    all_tests_passed = .true.

    ! Run all forall construct tests
    if (.not. test_simple_forall()) all_tests_passed = .false.
    if (.not. test_forall_with_mask()) all_tests_passed = .false.
    if (.not. test_nested_forall()) all_tests_passed = .false.
    if (.not. test_forall_assignment()) all_tests_passed = .false.

    print *
    if (all_tests_passed) then
        print *, "All forall construct tests passed!"
        stop 0
    else
        print *, "Some forall construct tests failed!"
        stop 1
    end if

contains

    function test_simple_forall() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: forall_idx, prog_idx, one_id, ten_id, i_id, assign_idx

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        error_msg = ""

        ! Test: forall (i = 1:10) a(i) = i
        arena = create_ast_arena()

        ! Create bounds
        one_id = push_literal(arena, "1", LITERAL_INTEGER)
        ten_id = push_literal(arena, "10", LITERAL_INTEGER)

        ! Create assignment: a(i) = i
        i_id = push_identifier(arena, "i")
        assign_idx = push_assignment(arena, i_id, i_id)

        ! Create forall construct
        forall_idx = push_forall(arena, "i", one_id, ten_id, &
                                 body_indices=[assign_idx])
        prog_idx = push_program(arena, "test", [forall_idx])

        ! Generate MLIR code
        options%compile_mode = .true.
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0 .and. allocated(output)) then
            ! Check for proper forall generation
            if ((index(output, "scf.parallel") > 0 .or. &
                 index(output, "scf.for") > 0) .and. &
                (index(output, "forall") > 0 .or. &
                 index(output, "parallel") > 0)) then
                print *, "PASS: Simple forall generates proper MLIR"
                passed = .true.
            else
                print *, "FAIL: Missing proper forall generation"
                print *, "Expected: parallel loop or forall construct"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating MLIR: ", trim(error_msg)
        end if
    end function test_simple_forall

    function test_forall_with_mask() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: forall_idx, prog_idx, one_id, ten_id, i_id, assign_idx
        integer :: mask_idx, five_id, gt_op_idx

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        error_msg = ""

        ! Test: forall (i = 1:10, i > 5) a(i) = i
        arena = create_ast_arena()

        ! Create bounds
        one_id = push_literal(arena, "1", LITERAL_INTEGER)
        ten_id = push_literal(arena, "10", LITERAL_INTEGER)
        five_id = push_literal(arena, "5", LITERAL_INTEGER)

        ! Create mask condition: i > 5
        i_id = push_identifier(arena, "i")
        gt_op_idx = push_binary_op(arena, i_id, five_id, ">")

        ! Create assignment: a(i) = i
        assign_idx = push_assignment(arena, i_id, i_id)

        ! Create forall construct with mask
        forall_idx = push_forall(arena, "i", one_id, ten_id, &
                                 mask_index=gt_op_idx, body_indices=[assign_idx])
        prog_idx = push_program(arena, "test", [forall_idx])

        ! Generate MLIR code
        options%compile_mode = .true.
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0 .and. allocated(output)) then
            ! Check for proper forall with mask
            if ((index(output, "scf.parallel") > 0 .or. &
                 index(output, "scf.for") > 0) .and. &
                (index(output, "scf.if") > 0 .or. &
                 index(output, "mask") > 0)) then
                print *, "PASS: Forall with mask generates proper MLIR"
                passed = .true.
            else
                print *, "FAIL: Missing proper forall with mask generation"
                print *, "Expected: parallel loop with conditional"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating MLIR: ", trim(error_msg)
        end if
    end function test_forall_with_mask

    function test_nested_forall() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: outer_forall_idx, inner_forall_idx, prog_idx
        integer :: one_id, ten_id, five_id, i_id, j_id, assign_idx

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        error_msg = ""

        ! Test: forall (i = 1:10)
        !         forall (j = 1:5) a(i,j) = i + j
        !       end forall
        arena = create_ast_arena()

        ! Create bounds
        one_id = push_literal(arena, "1", LITERAL_INTEGER)
        ten_id = push_literal(arena, "10", LITERAL_INTEGER)
        five_id = push_literal(arena, "5", LITERAL_INTEGER)

        ! Create assignment: a(i,j) = i + j
        i_id = push_identifier(arena, "i")
        j_id = push_identifier(arena, "j")
        assign_idx = push_binary_op(arena, i_id, j_id, "+")

        ! Create inner forall
        inner_forall_idx = push_forall(arena, "j", one_id, five_id, &
                                       body_indices=[assign_idx])

        ! Create outer forall
        outer_forall_idx = push_forall(arena, "i", one_id, ten_id, &
                                       body_indices=[inner_forall_idx])
        prog_idx = push_program(arena, "test", [outer_forall_idx])

        ! Generate MLIR code
        options%compile_mode = .true.
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0 .and. allocated(output)) then
            ! Check for proper nested forall
            if ((index(output, "scf.parallel") > 0 .or. &
                 index(output, "scf.for") > 0) .and. &
                (index(output, "forall") > 0 .or. &
                 count_occurrences(output, "parallel") >= 2)) then
                print *, "PASS: Nested forall generates proper MLIR"
                passed = .true.
            else
                print *, "FAIL: Missing proper nested forall generation"
                print *, "Expected: nested parallel loops"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating MLIR: ", trim(error_msg)
        end if
    end function test_nested_forall

    function test_forall_assignment() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: forall_idx, prog_idx, one_id, n_id, i_id
        integer :: a_ref_idx, b_ref_idx, assign_idx

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        error_msg = ""

        ! Test: forall (i = 1:n) a(i) = b(i) * 2
        arena = create_ast_arena()

        ! Create bounds
        one_id = push_literal(arena, "1", LITERAL_INTEGER)
        n_id = push_identifier(arena, "n")

        ! Create array references and assignment
        i_id = push_identifier(arena, "i")
        a_ref_idx = push_call_or_subscript(arena, "a", [i_id])
        b_ref_idx = push_call_or_subscript(arena, "b", [i_id])
        assign_idx = push_assignment(arena, a_ref_idx, b_ref_idx)

        ! Create forall construct
        forall_idx = push_forall(arena, "i", one_id, n_id, &
                                 body_indices=[assign_idx])
        prog_idx = push_program(arena, "test", [forall_idx])

        ! Generate MLIR code
        options%compile_mode = .true.
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0 .and. allocated(output)) then
            ! Check for proper forall array assignment
            if ((index(output, "scf.parallel") > 0 .or. &
                 index(output, "scf.for") > 0) .and. &
                (index(output, "memref.load") > 0 .and. &
                 index(output, "memref.store") > 0)) then
                print *, "PASS: Forall with array assignment generates proper MLIR"
                passed = .true.
            else
                print *, "FAIL: Missing proper forall array assignment"
                print *, "Expected: parallel loop with memory operations"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating MLIR: ", trim(error_msg)
        end if
    end function test_forall_assignment

    ! Helper function to count occurrences of a substring
    function count_occurrences(string, substring) result(count)
        character(len=*), intent(in) :: string, substring
        integer :: count
        integer :: pos, start

        count = 0
        start = 1
        do
            pos = index(string(start:), substring)
            if (pos == 0) exit
            count = count + 1
            start = start + pos + len(substring) - 1
        end do
    end function count_occurrences

end program test_forall_constructs
