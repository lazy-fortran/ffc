program test_derived_types
    use iso_fortran_env, only: error_unit
    use mlir_backend
    use backend_factory
    use backend_interface
    use ast_core, only: ast_arena_t, create_ast_stack, LITERAL_INTEGER, LITERAL_REAL, LITERAL_STRING
    use ast_factory
    use mlir_utils, only: int_to_str
    implicit none

    logical :: all_tests_passed

    print *, "=== Derived Types Tests ==="
    print *

    all_tests_passed = .true.

    ! Run all derived type tests
    if (.not. test_simple_type_definition()) all_tests_passed = .false.
    if (.not. test_type_with_multiple_components()) all_tests_passed = .false.
    if (.not. test_type_constructor()) all_tests_passed = .false.
    if (.not. test_component_access()) all_tests_passed = .false.
    if (.not. test_nested_types()) all_tests_passed = .false.

    print *
    if (all_tests_passed) then
        print *, "All derived types tests passed!"
        stop 0
    else
        print *, "Some derived types tests failed!"
        stop 1
    end if

contains

    function test_simple_type_definition() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: type_idx, prog_idx, x_comp_idx, y_comp_idx
        integer :: x_decl_idx, y_decl_idx

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        error_msg = ""

        ! Test: type :: point_t
        !         integer :: x
        !         integer :: y
        !       end type point_t
        arena = create_ast_stack()

        ! Create component declarations
        x_decl_idx = push_declaration(arena, "integer", "x")
        y_decl_idx = push_declaration(arena, "integer", "y")

        ! Create derived type
        type_idx = push_derived_type(arena, "point_t", [x_decl_idx, y_decl_idx])
        prog_idx = push_program(arena, "test", [type_idx])

        ! Generate MLIR code
        options%compile_mode = .true.
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0 .and. allocated(output)) then
            ! Check for proper derived type generation
            if ((index(output, "llvm.struct") > 0 .or. &
                 index(output, "!llvm.struct") > 0) .and. &
                (index(output, "point_t") > 0 .or. &
                 index(output, "type") > 0)) then
                print *, "PASS: Simple derived type generates proper MLIR"
                passed = .true.
            else
                print *, "FAIL: Missing proper derived type generation"
                print *, "Expected: LLVM struct type definition"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating MLIR: ", trim(error_msg)
        end if
    end function test_simple_type_definition

    function test_type_with_multiple_components() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: type_idx, prog_idx
        integer :: x_decl_idx, y_decl_idx, z_decl_idx, name_decl_idx

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        error_msg = ""

        ! Test: type :: person_t
        !         integer :: x, y, z
        !         character(len=20) :: name
        !       end type person_t
        arena = create_ast_stack()

        ! Create component declarations
        x_decl_idx = push_declaration(arena, "integer", "x")
        y_decl_idx = push_declaration(arena, "integer", "y")
        z_decl_idx = push_declaration(arena, "integer", "z")
        name_decl_idx = push_declaration(arena, "character", "name", kind_value=20)

        ! Create derived type
        type_idx = push_derived_type(arena, "person_t", [x_decl_idx, y_decl_idx, z_decl_idx, name_decl_idx])
        prog_idx = push_program(arena, "test", [type_idx])

        ! Generate MLIR code
        options%compile_mode = .true.
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0 .and. allocated(output)) then
            ! Check for proper multi-component type generation
            if ((index(output, "llvm.struct") > 0 .or. &
                 index(output, "!llvm.struct") > 0) .and. &
                (index(output, "i32") > 0 .and. &
                 index(output, "person_t") > 0)) then
                print *, "PASS: Multi-component derived type generates proper MLIR"
                passed = .true.
            else
                print *, "FAIL: Missing proper multi-component type generation"
                print *, "Expected: LLVM struct with multiple component types"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating MLIR: ", trim(error_msg)
        end if
    end function test_type_with_multiple_components

    function test_type_constructor() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: constructor_idx, prog_idx, type_idx
        integer :: x_val_idx, y_val_idx, point_var_idx, assign_idx
        integer :: x_decl_idx, y_decl_idx

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        error_msg = ""

        ! Test: type :: point_t
        !         integer :: x, y
        !       end type
        !       type(point_t) :: p
        !       p = point_t(10, 20)
        arena = create_ast_stack()

        ! Create type definition
        x_decl_idx = push_declaration(arena, "integer", "x")
        y_decl_idx = push_declaration(arena, "integer", "y")
        type_idx = push_derived_type(arena, "point_t", [x_decl_idx, y_decl_idx])

        ! Create constructor call values
        x_val_idx = push_literal(arena, "10", LITERAL_INTEGER)
        y_val_idx = push_literal(arena, "20", LITERAL_INTEGER)

        ! Create type constructor
       constructor_idx = push_type_constructor(arena, "point_t", [x_val_idx, y_val_idx])

        ! Create variable and assignment
        point_var_idx = push_identifier(arena, "p")
        assign_idx = push_assignment(arena, point_var_idx, constructor_idx)

        prog_idx = push_program(arena, "test", [type_idx, assign_idx])

        ! Generate MLIR code
        options%compile_mode = .true.
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0 .and. allocated(output)) then
            ! Check for proper type constructor generation
            if ((index(output, "llvm.undef") > 0 .or. &
                 index(output, "llvm.insertvalue") > 0) .and. &
                (index(output, "point_t") > 0 .or. &
                 index(output, "struct") > 0)) then
                print *, "PASS: Type constructor generates proper MLIR"
                passed = .true.
            else
                print *, "FAIL: Missing proper type constructor generation"
                print *, "Expected: struct initialization with insertvalue"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating MLIR: ", trim(error_msg)
        end if
    end function test_type_constructor

    function test_component_access() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: access_idx, prog_idx, type_idx, assign_idx
        integer :: point_var_idx, x_var_idx, x_decl_idx, y_decl_idx

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        error_msg = ""

        ! Test: type :: point_t
        !         integer :: x, y
        !       end type
        !       type(point_t) :: p
        !       x = p%x
        arena = create_ast_stack()

        ! Create type definition
        x_decl_idx = push_declaration(arena, "integer", "x")
        y_decl_idx = push_declaration(arena, "integer", "y")
        type_idx = push_derived_type(arena, "point_t", [x_decl_idx, y_decl_idx])

        ! Create component access: p%x
        point_var_idx = push_identifier(arena, "p")
        access_idx = push_component_access(arena, point_var_idx, "x")

        ! Create assignment: x = p%x
        x_var_idx = push_identifier(arena, "x")
        assign_idx = push_assignment(arena, x_var_idx, access_idx)

        prog_idx = push_program(arena, "test", [type_idx, assign_idx])

        ! Generate MLIR code
        options%compile_mode = .true.
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0 .and. allocated(output)) then
            ! Check for proper component access generation
            if ((index(output, "llvm.extractvalue") > 0 .or. &
                 index(output, "llvm.getelementptr") > 0) .and. &
                (index(output, "component") > 0 .or. &
                 index(output, "access") > 0 .or. &
                 index(output, "struct") > 0)) then
                print *, "PASS: Component access generates proper MLIR"
                passed = .true.
            else
                print *, "FAIL: Missing proper component access generation"
                print *, "Expected: extractvalue or getelementptr for component access"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating MLIR: ", trim(error_msg)
        end if
    end function test_component_access

    function test_nested_types() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: point_type_idx, circle_type_idx, prog_idx
        integer :: x_decl_idx, y_decl_idx, center_decl_idx, radius_decl_idx

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        error_msg = ""

        ! Test: type :: point_t
        !         integer :: x, y
        !       end type
        !       type :: circle_t
        !         type(point_t) :: center
        !         real :: radius
        !       end type
        arena = create_ast_stack()

        ! Create point type
        x_decl_idx = push_declaration(arena, "integer", "x")
        y_decl_idx = push_declaration(arena, "integer", "y")
        point_type_idx = push_derived_type(arena, "point_t", [x_decl_idx, y_decl_idx])

        ! Create circle type with nested point type
        center_decl_idx = push_declaration(arena, "point_t", "center")
        radius_decl_idx = push_declaration(arena, "real", "radius")
        circle_type_idx = push_derived_type(arena, "circle_t", [center_decl_idx, radius_decl_idx])

        prog_idx = push_program(arena, "test", [point_type_idx, circle_type_idx])

        ! Generate MLIR code
        options%compile_mode = .true.
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0 .and. allocated(output)) then
            ! Check for proper nested type generation
            if ((index(output, "point_t") > 0 .and. &
                 index(output, "circle_t") > 0) .and. &
                (index(output, "llvm.struct") > 0 .or. &
                 index(output, "!llvm.struct") > 0)) then
                print *, "PASS: Nested types generate proper MLIR"
                passed = .true.
            else
                print *, "FAIL: Missing proper nested type generation"
                print *, "Expected: multiple struct definitions with nesting"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating MLIR: ", trim(error_msg)
        end if
    end function test_nested_types

end program test_derived_types
