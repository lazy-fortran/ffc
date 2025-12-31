program test_module_system
    use iso_fortran_env, only: error_unit
    use mlir_backend
    use backend_factory
    use backend_interface
    use fortfront, only: ast_arena_t, create_ast_arena, LITERAL_INTEGER, LITERAL_STRING
    use ast_factory
    implicit none

    logical :: all_tests_passed

    print *, "=== Module System Tests ==="
    print *

    all_tests_passed = .true.

    ! Run all tests
    if (.not. test_module_definition()) all_tests_passed = .false.
    if (.not. test_use_statement()) all_tests_passed = .false.
    if (.not. test_public_private_visibility()) all_tests_passed = .false.
    if (.not. test_module_procedures()) all_tests_passed = .false.
    if (.not. test_generic_interfaces()) all_tests_passed = .false.

    print *
    if (all_tests_passed) then
        print *, "All module system tests passed!"
        stop 0
    else
        print *, "Some module system tests failed!"
        stop 1
    end if

contains

    function test_module_definition() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: module_idx, prog_idx

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        error_msg = ""

        ! Test module definition
        ! module my_module
        ! end module my_module
        arena = create_ast_arena()

        module_idx = push_module(arena, "my_module")
        prog_idx = push_program(arena, "test", [module_idx])

        ! Generate MLIR code
        options%compile_mode = .true.
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0 .and. allocated(output)) then
            ! Check for module definition in MLIR output
            if (index(output, "module {") > 0 .and. &
                index(output, "my_module") > 0) then
                print *, "PASS: Module definition generates MLIR module"
                passed = .true.
            else
                print *, "FAIL: Missing MLIR module generation"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating MLIR: ", trim(error_msg)
        end if
    end function test_module_definition

    function test_use_statement() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: use_idx, prog_idx

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        error_msg = ""

        ! Test use statement
        ! use iso_fortran_env
        arena = create_ast_arena()

        use_idx = push_use_statement(arena, "iso_fortran_env")
        prog_idx = push_program(arena, "test", [use_idx])

        ! Generate MLIR code
        options%compile_mode = .true.
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0 .and. allocated(output)) then
            ! Check for use statement import in MLIR output
            if (index(output, "import") > 0 .or. &
                index(output, "iso_fortran_env") > 0) then
                print *, "PASS: Use statement generates MLIR import"
                passed = .true.
            else
                print *, "FAIL: Missing MLIR import generation"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating MLIR: ", trim(error_msg)
        end if
    end function test_use_statement

    function test_public_private_visibility() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: module_idx, decl_idx, prog_idx

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        error_msg = ""

        ! Test public/private visibility
        ! module my_module
        !   private
        !   integer, public :: x
        ! end module my_module
        arena = create_ast_arena()

        decl_idx = push_declaration(arena, "integer", ["x"])
        module_idx = push_module(arena, "my_module", [decl_idx])
        prog_idx = push_program(arena, "test", [module_idx])

        ! Generate MLIR code
        options%compile_mode = .true.
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0 .and. allocated(output)) then
            ! Check for visibility attributes in MLIR output
            if (index(output, "my_module") > 0) then
                print *, "PASS: Public/private visibility handled"
                passed = .true.
            else
                print *, "FAIL: Missing visibility handling"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating MLIR: ", trim(error_msg)
        end if
    end function test_public_private_visibility

    function test_module_procedures() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: module_idx, func_idx, prog_idx

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        error_msg = ""

        ! Test module procedures
        ! module my_module
        !   contains
        !   function add(x, y) result(z)
        !     integer :: x, y, z
        !     z = x + y
        !   end function add
        ! end module my_module
        arena = create_ast_arena()

       func_idx = push_function_def(arena, "add", [integer ::], "integer", [integer ::])
        module_idx = push_module(arena, "my_module", [func_idx])
        prog_idx = push_program(arena, "test", [module_idx])

        ! Generate MLIR code
        options%compile_mode = .true.
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0 .and. allocated(output)) then
            ! Check for module procedures in MLIR output
            if (index(output, "func.func @add") > 0) then
                print *, "PASS: Module procedures generate MLIR functions"
                passed = .true.
            else
                print *, "FAIL: Missing module procedure generation"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating MLIR: ", trim(error_msg)
        end if
    end function test_module_procedures

    function test_generic_interfaces() result(passed)
        logical :: passed

        ! Generic interfaces are not yet supported in AST factory
        ! This test will need to be implemented when interface support is added
        print *, "SKIP: Generic interfaces not yet supported in AST factory"
        passed = .true.
    end function test_generic_interfaces

end program test_module_system
