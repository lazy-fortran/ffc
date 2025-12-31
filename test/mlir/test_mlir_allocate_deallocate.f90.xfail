program test_mlir_allocate_deallocate
    use iso_fortran_env, only: error_unit
    use backend_interface
    use backend_factory
    use fortfront
    use ast_factory
    implicit none

    logical :: all_tests_passed

    print *, "=== MLIR Allocate/Deallocate Tests (HLFIR) ==="
    print *

    all_tests_passed = .true.

    ! Run all tests
    if (.not. test_simple_allocate()) all_tests_passed = .false.
    if (.not. test_array_allocate()) all_tests_passed = .false.
    if (.not. test_allocate_with_stat()) all_tests_passed = .false.
    if (.not. test_deallocate()) all_tests_passed = .false.
    if (.not. test_deallocate_with_stat()) all_tests_passed = .false.

    print *
    if (all_tests_passed) then
        print *, "All MLIR allocate/deallocate tests passed!"
        stop 0
    else
        print *, "Some MLIR allocate/deallocate tests failed!"
        stop 1
    end if

contains

    function test_simple_allocate() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: ptr_idx, alloc_idx, prog_idx

        passed = .false.
        error_msg = ""
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        ! Test that simple allocate uses fir.allocmem (not malloc)
        arena = create_ast_arena()

        ! Create: integer, pointer :: p
        ptr_idx = push_declaration(arena, "integer", ["p"], is_pointer=.true.)
        ! Create: allocate(p)
        alloc_idx = push_allocate(arena, [ptr_idx])
        prog_idx = push_program(arena, "test", [ptr_idx, alloc_idx])

        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0 .and. allocated(output)) then
            ! Should use fir.allocmem for HLFIR memory allocation
            if (index(output, "fir.allocmem") > 0 .and. &
                index(output, "malloc") == 0) then
                print *, "PASS: Simple allocate uses fir.allocmem"
                passed = .true.
            else
                print *, "FAIL: Simple allocate should use fir.allocmem, not malloc"
         print *, "  Output contains 'fir.allocmem':", index(output, "fir.allocmem") > 0
                print *, "  Output contains 'malloc':", index(output, "malloc") > 0
            end if
        else
            print *, "FAIL: Error generating MLIR: ", trim(error_msg)
        end if
    end function test_simple_allocate

    function test_array_allocate() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: arr_idx, dim_idx, alloc_idx, prog_idx

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        error_msg = ""

        ! Test that array allocate uses fir.allocmem with proper shape
        arena = create_ast_arena()

        ! Create: real, allocatable :: arr(:)
        arr_idx = push_declaration(arena, "real", ["arr"], is_allocatable=.true.)
        ! Create dimension expression: 100
        dim_idx = push_literal(arena, "100", LITERAL_INTEGER)
        ! Create: allocate(arr(100))
        alloc_idx = push_allocate(arena, [arr_idx], shape_indices=[dim_idx])
        prog_idx = push_program(arena, "test", [arr_idx, alloc_idx])

        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0 .and. allocated(output)) then
            ! Should use fir.allocmem with array type
            if (index(output, "fir.allocmem") > 0 .and. &
                index(output, "!fir.array") > 0 .and. &
                index(output, "malloc") == 0) then
                print *, "PASS: Array allocate uses fir.allocmem with array type"
                passed = .true.
            else
                print *, "FAIL: Array allocate should use fir.allocmem with !fir.array"
         print *, "  Output contains 'fir.allocmem':", index(output, "fir.allocmem") > 0
             print *, "  Output contains '!fir.array':", index(output, "!fir.array") > 0
            end if
        else
            print *, "FAIL: Error generating MLIR: ", trim(error_msg)
        end if
    end function test_array_allocate

    function test_allocate_with_stat() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: ptr_idx, stat_idx, alloc_idx, prog_idx

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        error_msg = ""

        ! Test that allocate with stat parameter generates proper error handling
        arena = create_ast_arena()

        ! Create: integer, pointer :: p
        ptr_idx = push_declaration(arena, "integer", ["p"], is_pointer=.true.)
        ! Create: integer :: stat
        stat_idx = push_declaration(arena, "integer", ["stat"])
        ! Create: allocate(p, stat=stat)
        alloc_idx = push_allocate(arena, [ptr_idx], stat_var_index=stat_idx)
        prog_idx = push_program(arena, "test", [ptr_idx, stat_idx, alloc_idx])

        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0 .and. allocated(output)) then
            ! Should use fir.allocmem and store status
            if (index(output, "fir.allocmem") > 0 .and. &
                index(output, "fir.store") > 0) then
                print *, "PASS: Allocate with stat uses fir.allocmem and stores status"
                passed = .true.
            else
            print *, "FAIL: Allocate with stat should use fir.allocmem and store status"
            end if
        else
            print *, "FAIL: Error generating MLIR: ", trim(error_msg)
        end if
    end function test_allocate_with_stat

    function test_deallocate() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: ptr_idx, dealloc_idx, prog_idx

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        error_msg = ""

        ! Test that deallocate uses fir.freemem (not free)
        arena = create_ast_arena()

        ! Create: integer, pointer :: p
        ptr_idx = push_declaration(arena, "integer", ["p"], is_pointer=.true.)
        ! Create: deallocate(p)
        dealloc_idx = push_deallocate(arena, [ptr_idx])
        prog_idx = push_program(arena, "test", [ptr_idx, dealloc_idx])

        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0 .and. allocated(output)) then
            ! Should use fir.freemem for HLFIR memory deallocation
            if (index(output, "fir.freemem") > 0 .and. &
                index(output, "free") > 0 .and. &
                index(output, "llvm.call @free") == 0) then
                print *, "PASS: Deallocate uses fir.freemem"
                passed = .true.
            else
                print *, "FAIL: Deallocate should use fir.freemem, not free"
           print *, "  Output contains 'fir.freemem':", index(output, "fir.freemem") > 0
   print *, "  Output contains 'llvm.call @free':", index(output, "llvm.call @free") > 0
            end if
        else
            print *, "FAIL: Error generating MLIR: ", trim(error_msg)
        end if
    end function test_deallocate

    function test_deallocate_with_stat() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: ptr_idx, stat_idx, dealloc_idx, prog_idx

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        error_msg = ""

        ! Test that deallocate with stat parameter generates proper error handling
        arena = create_ast_arena()

        ! Create: integer, pointer :: p
        ptr_idx = push_declaration(arena, "integer", ["p"], is_pointer=.true.)
        ! Create: integer :: stat
        stat_idx = push_declaration(arena, "integer", ["stat"])
        ! Create: deallocate(p, stat=stat)
        dealloc_idx = push_deallocate(arena, [ptr_idx], stat_var_index=stat_idx)
        prog_idx = push_program(arena, "test", [ptr_idx, stat_idx, dealloc_idx])

        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0 .and. allocated(output)) then
            ! Should use fir.freemem and store status
            if (index(output, "fir.freemem") > 0 .and. &
                index(output, "fir.store") > 0) then
                print *, "PASS: Deallocate with stat uses fir.freemem and stores status"
                passed = .true.
            else
           print *, "FAIL: Deallocate with stat should use fir.freemem and store status"
            end if
        else
            print *, "FAIL: Error generating MLIR: ", trim(error_msg)
        end if
    end function test_deallocate_with_stat

end program test_mlir_allocate_deallocate
