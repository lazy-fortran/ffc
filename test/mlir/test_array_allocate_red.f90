program test_array_allocate_red
    use ast_core, only: ast_arena_t, create_ast_arena
    use ast_factory
    use backend_interface
    use backend_factory
    implicit none

    logical :: passed
    type(ast_arena_t) :: arena
    class(backend_t), allocatable :: backend
    type(backend_options_t) :: backend_opts
    character(len=:), allocatable :: mlir_code
    character(len=256) :: error_msg
    integer :: prog_idx, decl_idx, allocate_idx, n_idx, m_idx
    integer :: n_expr_idx, m_expr_idx, arr_idx

    print *, "=== RED phase: Array Allocate with Dimensions Test ==="

    passed = .false.
    arena = create_ast_arena()

    ! Create AST for:
    ! program test
    !   integer, allocatable :: arr(:,:)
    !   integer :: n, m
    !   n = 10
    !   m = 20
    !   allocate(arr(n,m))
    ! end program

    ! Create variable declarations
    decl_idx = push_declaration(arena, "integer", "arr")
    ! TODO: Need to mark as allocatable array with dimensions
    
    n_idx = push_declaration(arena, "integer", "n")
    m_idx = push_declaration(arena, "integer", "m")

    ! Create dimension expressions for allocate
    n_expr_idx = push_identifier(arena, "n")
    m_expr_idx = push_identifier(arena, "m")
    arr_idx = push_identifier(arena, "arr")

    ! Create array allocate statement: allocate(arr(n,m))
    allocate_idx = push_allocate(arena, [arr_idx], shape_indices=[n_expr_idx, m_expr_idx])

    prog_idx = push_program(arena, "test_array_allocate", &
                           [decl_idx, n_idx, m_idx, allocate_idx])

    ! Configure backend for compile mode
    backend_opts%compile_mode = .true.

    ! Create MLIR backend
    call create_backend("mlir", backend, error_msg)
    if (len_trim(error_msg) > 0) then
        print *, "FAIL: Error creating backend:", trim(error_msg)
        stop 1
    end if

    ! Generate MLIR code
    call backend%generate_code(arena, prog_idx, backend_opts, mlir_code, &
                               error_msg)

    ! Check if array allocate with dimensions generates proper MLIR
    ! This SHOULD FAIL in RED phase - look for runtime dimension calculation
    if (index(mlir_code, "arith.constant 8 : i64") > 0) then
        print *, "FAIL: Array allocate uses hardcoded size (expected in RED phase)"
        print *, "  Found: arith.constant 8 : i64 (hardcoded size)"
        print *, "  Expected: Runtime dimension calculation using n and m"
        passed = .false.
    else if (index(mlir_code, "memref.load") > 0 .and. &
             index(mlir_code, "arith.muli") > 0) then
        print *, "PASS: Array allocate with runtime dimensions works (unexpected in RED phase)"
        passed = .true.
    else
        print *, "FAIL: Array allocate doesn't use runtime dimensions (expected in RED phase)"
        print *, "  Expected: Load n and m values, multiply for size calculation"
        if (len_trim(error_msg) > 0) then
            print *, "  Error: ", trim(error_msg)
        end if
        passed = .false.
    end if

    if (.not. passed) then
        print *, "RED PHASE CONFIRMED: Array allocate needs dimension support"
        stop 0  ! This is expected failure in TDD RED phase
    else
        print *, "RED PHASE UNEXPECTED: Array allocate already works"
        stop 1
    end if

end program test_array_allocate_red