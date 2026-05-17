program test_array_allocate_debug
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

    print *, "=== DEBUG: Array Allocate with Dimensions ==="

    arena = create_ast_arena()

    ! Create variable declarations
    decl_idx = push_declaration(arena, "integer", "arr")
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

    print *, "MLIR code length:", len(mlir_code)
    if (len_trim(mlir_code) > 0) then
        print *, "MLIR code:"
        print *, mlir_code
    end if

    if (len_trim(error_msg) > 0) then
        print *, "Error message:", trim(error_msg)
    end if

    stop 0

end program test_array_allocate_debug