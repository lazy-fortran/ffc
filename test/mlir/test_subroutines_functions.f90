program test_subroutines_functions
    use iso_fortran_env, only: error_unit
    use mlir_backend
    use backend_factory
    use backend_interface
    use fortfront, only: ast_arena_t, create_ast_arena, LITERAL_INTEGER
    use ast_factory
    implicit none

    logical :: all_tests_passed

    print *, "=== Subroutines and Functions Tests ==="
    print *

    all_tests_passed = .true.

    ! Run all tests
    if (.not. test_subroutine_definition()) all_tests_passed = .false.
    if (.not. test_subroutine_call()) all_tests_passed = .false.
    if (.not. test_function_definition()) all_tests_passed = .false.
    if (.not. test_function_call()) all_tests_passed = .false.
    if (.not. test_parameter_passing()) all_tests_passed = .false.
    if (.not. test_intent_attributes()) all_tests_passed = .false.
    if (.not. test_optional_parameters()) all_tests_passed = .false.
    if (.not. test_function_result_variables()) all_tests_passed = .false.

    print *
    if (all_tests_passed) then
        print *, "All subroutines and functions tests passed!"
        stop 0
    else
        print *, "Some subroutines and functions tests failed!"
        stop 1
    end if

contains

    function test_subroutine_definition() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: sub_idx, prog_idx

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        error_msg = ""

        ! Test subroutine definition
        arena = create_ast_arena()

        ! subroutine hello()
        !   print *, 'Hello from subroutine!'
        ! end subroutine hello
        sub_idx = push_subroutine_def(arena, "hello")
        prog_idx = push_program(arena, "test", [sub_idx])

        ! Generate MLIR code
        options%compile_mode = .true.
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0 .and. allocated(output)) then
            ! Check for subroutine function in MLIR output
            if (index(output, "func.func @hello") > 0) then
                print *, "PASS: Subroutine definition generates func.func"
                passed = .true.
            else
                print *, "FAIL: Missing func.func for subroutine"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating MLIR: ", trim(error_msg)
        end if
    end function test_subroutine_definition

    function test_subroutine_call() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: sub_idx, call_idx, prog_idx

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        error_msg = ""

        ! Test subroutine call
        arena = create_ast_arena()

        ! subroutine hello()
        ! end subroutine hello
        !
        ! program test
        !   call hello()
        ! end program test
        sub_idx = push_subroutine_def(arena, "hello")
        call_idx = push_subroutine_call(arena, "hello", [integer ::])
        prog_idx = push_program(arena, "test", [sub_idx, call_idx])

        ! Generate MLIR code
        options%compile_mode = .true.
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0 .and. allocated(output)) then
            ! Check for function call in MLIR output
            if (index(output, "call @hello") > 0) then
                print *, "PASS: Subroutine call generates func.call"
                passed = .true.
            else
                print *, "FAIL: Missing func.call for subroutine"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating MLIR: ", trim(error_msg)
        end if
    end function test_subroutine_call

    function test_function_definition() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: func_idx, ret_idx, prog_idx

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        error_msg = ""

        ! Test function definition
        arena = create_ast_arena()

        ! integer function add_one(x)
        !   integer :: x
        !   add_one = x + 1
        ! end function add_one
        ret_idx = push_literal(arena, "1", LITERAL_INTEGER)
      func_idx = push_function_def(arena, "add_one", [integer ::], "integer", [ret_idx])
        prog_idx = push_program(arena, "test", [func_idx])

        ! Generate MLIR code
        options%compile_mode = .true.
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0 .and. allocated(output)) then
            ! Check for function with return type in MLIR output
            if (index(output, "func.func @add_one") > 0 .and. &
                index(output, "-> i32") > 0) then
               print *, "PASS: Function definition generates func.func with return type"
                passed = .true.
            else
                print *, "FAIL: Missing func.func with return type for function"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating MLIR: ", trim(error_msg)
        end if
    end function test_function_definition

    function test_function_call() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: func_idx, call_idx, ret_idx, prog_idx

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        error_msg = ""

        ! Test function call
        arena = create_ast_arena()

        ! integer function get_five()
        !   get_five = 5
        ! end function get_five
        !
        ! program test
        !   integer :: result
        !   result = get_five()
        ! end program test
        ret_idx = push_literal(arena, "5", LITERAL_INTEGER)
     func_idx = push_function_def(arena, "get_five", [integer ::], "integer", [ret_idx])
        call_idx = push_call_or_subscript(arena, "get_five", [integer ::])
        prog_idx = push_program(arena, "test", [func_idx, call_idx])

        ! Generate MLIR code
        options%compile_mode = .true.
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0 .and. allocated(output)) then
            ! Check for function call in MLIR output
            if (index(output, "call @get_five") > 0) then
                print *, "PASS: Function call generates func.call"
                passed = .true.
            else
                print *, "FAIL: Missing func.call for function"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating MLIR: ", trim(error_msg)
        end if
    end function test_function_call

    function test_parameter_passing() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: param_idx, func_idx, call_idx, arg_idx, prog_idx

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        error_msg = ""

        ! Test parameter passing
        arena = create_ast_arena()

        ! integer function double_value(x)
        !   integer :: x
        !   double_value = x * 2
        ! end function double_value
        !
        ! program test
        !   integer :: result
        !   result = double_value(42)
        ! end program test
        param_idx = push_declaration(arena, "integer", ["x"])
        func_idx = push_function_def(arena, "double_value", [param_idx], "integer", [integer ::])
        arg_idx = push_literal(arena, "42", LITERAL_INTEGER)
        call_idx = push_call_or_subscript(arena, "double_value", [arg_idx])
        prog_idx = push_program(arena, "test", [func_idx, call_idx])

        ! Generate MLIR code
        options%compile_mode = .true.
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0 .and. allocated(output)) then
            ! Check for function with parameters in MLIR output
            if (index(output, "func.func @double_value") > 0 .and. &
                index(output, "call @double_value") > 0) then
                print *, "PASS: Parameter passing generates correct function signature"
                passed = .true.
            else
                print *, "FAIL: Missing correct parameter passing"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating MLIR: ", trim(error_msg)
        end if
    end function test_parameter_passing

    function test_intent_attributes() result(passed)
        logical :: passed

        ! Intent attributes are not yet supported in AST factory
        ! This test will need to be implemented when intent support is added
        print *, "SKIP: Intent attributes not yet supported in AST factory"
        passed = .true.
    end function test_intent_attributes

    function test_optional_parameters() result(passed)
        logical :: passed

        ! Optional parameters are not yet supported in AST factory
        ! This test will need to be implemented when optional support is added
        print *, "SKIP: Optional parameters not yet supported in AST factory"
        passed = .true.
    end function test_optional_parameters

    function test_function_result_variables() result(passed)
        logical :: passed

        ! Function result variables are not yet supported in AST factory
        ! This test will need to be implemented when result variable support is added
        print *, "SKIP: Function result variables not yet supported in AST factory"
        passed = .true.
    end function test_function_result_variables

end program test_subroutines_functions
