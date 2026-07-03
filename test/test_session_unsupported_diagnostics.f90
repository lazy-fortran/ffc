program test_session_unsupported_diagnostics
    use ffc_test_support, only: expect_error_contains, &
        expect_cli_error_contains, &
        expect_output, expect_cli_no_error
    implicit none

    logical :: all_passed

    print *, '=== direct session unsupported diagnostic test ==='

    all_passed = .true.
    if (.not. test_character_expression_diagnostic()) all_passed = .false.
    if (.not. test_unassigned_character_print_diagnostic()) all_passed = .false.
    if (.not. test_module_diagnostic()) all_passed = .false.
    if (.not. test_character_parameter_lowers()) all_passed = .false.
    if (.not. test_complex_scalar_lowers()) all_passed = .false.
    if (.not. test_select_case_no_default_lowers()) all_passed = .false.
    if (.not. test_do_zero_step_diagnostic()) all_passed = .false.
    if (.not. test_do_terminating_body_diagnostic()) all_passed = .false.
    if (.not. test_derived_type_diagnostic()) all_passed = .false.
    if (.not. test_component_assignment_target_diagnostic()) &
        all_passed = .false.
    if (.not. test_component_expression_diagnostic()) all_passed = .false.
    if (.not. test_unsupported_intrinsic_diagnostic()) all_passed = .false.
    if (.not. test_external_function_call_diagnostic()) all_passed = .false.
    if (.not. test_external_real_function_call_diagnostic()) all_passed = .false.
    if (.not. test_external_logical_function_call_diagnostic()) &
        all_passed = .false.
    if (.not. test_integer_exponent_operator_diagnostic()) all_passed = .false.
    if (.not. test_allocate_statement_diagnostic()) all_passed = .false.
    if (.not. test_deallocate_statement_diagnostic()) all_passed = .false.
    if (.not. test_return_statement_diagnostic()) all_passed = .false.
    if (.not. test_cli_character_expression_diagnostic()) all_passed = .false.
    if (.not. test_cli_unassigned_character_print_diagnostic()) &
        all_passed = .false.
    if (.not. test_cli_module_diagnostic()) all_passed = .false.
    if (.not. test_cli_character_parameter_lowers()) all_passed = .false.
    if (.not. test_cli_complex_scalar_lowers()) &
        all_passed = .false.
    if (.not. test_cli_select_case_no_default_lowers()) all_passed = .false.
    if (.not. test_cli_do_zero_step_diagnostic()) all_passed = .false.
    if (.not. test_cli_do_terminating_body_diagnostic()) all_passed = .false.
    if (.not. test_cli_derived_type_diagnostic()) all_passed = .false.
    if (.not. test_cli_component_assignment_target_diagnostic()) &
        all_passed = .false.
    if (.not. test_cli_component_expression_diagnostic()) &
        all_passed = .false.
    if (.not. test_cli_unsupported_intrinsic_diagnostic()) all_passed = .false.
    if (.not. test_cli_external_function_call_diagnostic()) all_passed = .false.
    if (.not. test_cli_external_real_function_call_diagnostic()) &
        all_passed = .false.
    if (.not. test_cli_external_logical_function_call_diagnostic()) &
        all_passed = .false.
    if (.not. test_cli_integer_exponent_operator_diagnostic()) &
        all_passed = .false.
    if (.not. test_cli_allocate_statement_diagnostic()) all_passed = .false.
    if (.not. test_cli_deallocate_statement_diagnostic()) all_passed = .false.
    if (.not. test_cli_return_statement_diagnostic()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: unsupported direct-session features emit diagnostics'

contains

    logical function test_character_expression_diagnostic()
        ! A substring target keeps the unsupported diagnostic; // concatenation
        ! into a fixed-length scalar is now lowered (see the fixed-concat test).
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=5) :: name'//new_line('a')// &
            '  name = "ab"'//new_line('a')// &
            '  name = name(1:2)'//new_line('a')// &
            'end program main'

        test_character_expression_diagnostic = expect_error_contains( &
            source, 'unsupported character assignment', &
            '/tmp/ffc_session_character_diagnostic_test')
    end function test_character_expression_diagnostic

    logical function test_unassigned_character_print_diagnostic()
        character(len=*), parameter :: expected = &
            'unsupported character variable print'
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=5) :: name'//new_line('a')// &
            '  print *, name'//new_line('a')// &
            'end program main'

        test_unassigned_character_print_diagnostic = &
            expect_error_contains(source, expected, &
            '/tmp/ffc_session_unassigned_character_test')
    end function test_unassigned_character_print_diagnostic

    logical function test_module_diagnostic()
        ! Scalar module variables lower since #263; a module ARRAY variable still
        ! needs storage that is not yet emitted, so it stays a clean diagnostic.
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '  integer :: counter(3)'//new_line('a')// &
            'end module m'

        test_module_diagnostic = expect_error_contains( &
            source, 'unsupported module variable', &
            '/tmp/ffc_session_module_diagnostic_test')
    end function test_module_diagnostic

    ! A fixed-length character dummy (character(len=N)) now lowers: it reads
    ! the first N bytes of the caller's actual through the shared descriptor,
    ! keeping its own declared width rather than the caller's runtime length.
    logical function test_character_parameter_lowers()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  call show("hi")'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine show(s)'//new_line('a')// &
            '    character(len=2), intent(in) :: '// &
            's'// &
            new_line('a')// &
            '    print *, s'//new_line('a')// &
            '  end subroutine show'//new_line('a')// &
            'end program main'

        test_character_parameter_lowers = expect_output( &
            source, ' hi'//new_line('a'), &
            '/tmp/ffc_session_char_param_test')
    end function test_character_parameter_lowers

    ! complex scalars are supported now; verify they lower and match gfortran.
    logical function test_complex_scalar_lowers()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  complex :: z'//new_line('a')// &
            '  z = (1.0, 2.0)'//new_line('a')// &
            '  print *, z'//new_line('a')// &
            'end program main'

        test_complex_scalar_lowers = expect_output( &
            source, '             (1.00000000,2.00000000)'//new_line('a'), &
            '/tmp/ffc_session_scalar_type_test')
    end function test_complex_scalar_lowers

    logical function test_select_case_no_default_lowers()
        ! SELECT CASE without a CASE DEFAULT arm now lowers; the matching arm
        ! runs and a non-matching selector simply falls through.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: x'//new_line('a')// &
            '  x = 1'//new_line('a')// &
            '  select case (x)'//new_line('a')// &
            '  case (1)'//new_line('a')// &
            '    print *, 1'//new_line('a')// &
            '  end select'//new_line('a')// &
            'end program main'

        test_select_case_no_default_lowers = expect_output( &
            source, '           1'//new_line('a'), &
            '/tmp/ffc_session_select_case_test')
    end function test_select_case_no_default_lowers

    logical function test_do_zero_step_diagnostic()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  do i = 1, 3, 0'//new_line('a')// &
            '    print *, i'//new_line('a')// &
            'end program main'

        test_do_zero_step_diagnostic = expect_error_contains( &
            source, 'nonzero literal step', &
            '/tmp/ffc_session_do_zero_step_test')
    end function test_do_zero_step_diagnostic

    logical function test_do_terminating_body_diagnostic()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  do i = 1, 3'//new_line('a')// &
            '    stop 1'//new_line('a')// &
            '  end do'//new_line('a')// &
            'end program main'

        test_do_terminating_body_diagnostic = expect_error_contains( &
            source, 'unsupported terminating do loop body', &
            '/tmp/ffc_session_do_terminating_body_test')
    end function test_do_terminating_body_diagnostic

    logical function test_derived_type_diagnostic()
        ! A fixed-length scalar character component reads and writes through
        ! obj%comp (#265); a character ARRAY component stays unsupported since
        ! the flat slot layout only reserves inline byte storage for a scalar.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: point'//new_line('a')// &
            '    character(len=4) :: name(2)'//new_line('a')// &
            '  end type point'//new_line('a')// &
            '  type(point) :: p'//new_line('a')// &
            '  p%name(1) = "abcd"'//new_line('a')// &
            'end program main'

        test_derived_type_diagnostic = expect_error_contains( &
            source, &
            'character array', &
            '/tmp/ffc_session_derived_type_test')
    end function test_derived_type_diagnostic

    logical function test_component_assignment_target_diagnostic()
        character(len=*), parameter :: expected = &
            'unsupported derived type component '// &
            'assignment target'
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  p%x = 1'//new_line('a')// &
            'end program main'

        test_component_assignment_target_diagnostic = expect_error_contains( &
            source, expected, &
            '/tmp/ffc_session_component_target_test')
    end function test_component_assignment_target_diagnostic

    logical function test_component_expression_diagnostic()
        character(len=*), parameter :: expected = &
            'unsupported derived type component '// &
            'expression'
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  print *, p%x'//new_line('a')// &
            'end program main'

        test_component_expression_diagnostic = expect_error_contains( &
            source, expected, &
            '/tmp/ffc_session_component_expr_test')
    end function test_component_expression_diagnostic

    logical function test_unsupported_intrinsic_diagnostic()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  print *, sqrt(4)'//new_line('a')// &
            'end program main'

        test_unsupported_intrinsic_diagnostic = expect_error_contains( &
            source, 'unsupported scalar intrinsic: sqrt', &
            '/tmp/ffc_session_intrinsic_diagnostic_test')
    end function test_unsupported_intrinsic_diagnostic

    logical function test_external_function_call_diagnostic()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: x'//new_line('a')// &
            '  x = external_value(1)'//new_line('a')// &
            'end program main'

        test_external_function_call_diagnostic = expect_error_contains( &
            source, 'unsupported scalar function call', &
            '/tmp/ffc_session_external_call_test')
    end function test_external_function_call_diagnostic

    logical function test_external_real_function_call_diagnostic()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real :: x'//new_line('a')// &
            '  x = external_real(1.0)'//new_line('a')// &
            'end program main'

        test_external_real_function_call_diagnostic = &
            expect_error_contains(source, &
            'unsupported scalar real(4) function call', &
            '/tmp/ffc_session_external_real_call_test')
    end function test_external_real_function_call_diagnostic

    logical function test_external_logical_function_call_diagnostic()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  logical :: flag'//new_line('a')// &
            '  flag = external_flag(.true.)'// &
            new_line('a')// &
            'end program main'

        test_external_logical_function_call_diagnostic = &
            expect_error_contains(source, &
            'unsupported scalar logical function call', &
            '/tmp/ffc_session_external_logical_call_test')
    end function test_external_logical_function_call_diagnostic

    logical function test_integer_exponent_operator_diagnostic()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: x'//new_line('a')// &
            '  x = 2 ** (-1)'//new_line('a')// &
            'end program main'

        test_integer_exponent_operator_diagnostic = expect_error_contains( &
            source, 'unsupported negative integer exponent', &
            '/tmp/ffc_session_integer_exponent_operator_test')
    end function test_integer_exponent_operator_diagnostic

    logical function test_allocate_statement_diagnostic()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  allocate(values(3))'//new_line('a')// &
            'end program main'

        test_allocate_statement_diagnostic = expect_error_contains( &
            source, &
            'unsupported allocate statement', &
            '/tmp/ffc_session_allocate_test')
    end function test_allocate_statement_diagnostic

    logical function test_deallocate_statement_diagnostic()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  deallocate(values)'//new_line('a')// &
            'end program main'

        test_deallocate_statement_diagnostic = expect_error_contains( &
            source, &
            'unsupported deallocate statement', &
            '/tmp/ffc_session_deallocate_test')
    end function test_deallocate_statement_diagnostic

    logical function test_return_statement_diagnostic()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  return'//new_line('a')// &
            'end program main'

        test_return_statement_diagnostic = expect_error_contains( &
            source, &
            'unsupported return statement', &
            '/tmp/ffc_session_return_test')
    end function test_return_statement_diagnostic

    logical function test_cli_character_expression_diagnostic()
        ! A substring target keeps the unsupported diagnostic; // concatenation
        ! into a fixed-length scalar is now lowered (see the fixed-concat test).
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=5) :: name'//new_line('a')// &
            '  name = "ab"'//new_line('a')// &
            '  name = name(1:2)'//new_line('a')// &
            'end program main'

        test_cli_character_expression_diagnostic = expect_cli_error_contains( &
            source, 'unsupported character assignment', &
            '/tmp/ffc_cli_character_diagnostic_test')
    end function test_cli_character_expression_diagnostic

    logical function test_cli_unassigned_character_print_diagnostic()
        character(len=*), parameter :: expected = &
            'unsupported character variable print'
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=5) :: name'//new_line('a')// &
            '  print *, name'//new_line('a')// &
            'end program main'

        test_cli_unassigned_character_print_diagnostic = &
            expect_cli_error_contains(source, expected, &
            '/tmp/ffc_cli_unassigned_character_test')
    end function test_cli_unassigned_character_print_diagnostic

    logical function test_cli_module_diagnostic()
        ! Scalar module variables lower since #263; a module ARRAY variable still
        ! needs storage that is not yet emitted, so it stays a clean diagnostic.
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '  integer :: counter(3)'//new_line('a')// &
            'end module m'

        test_cli_module_diagnostic = expect_cli_error_contains( &
            source, 'unsupported module variable', &
            '/tmp/ffc_cli_module_diagnostic_test')
    end function test_cli_module_diagnostic

    ! Same construct through the CLI: it compiles cleanly with no diagnostic.
    logical function test_cli_character_parameter_lowers()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  call show("hi")'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine show(s)'//new_line('a')// &
            '    character(len=2), intent(in) :: '// &
            's'// &
            new_line('a')// &
            '    print *, s'//new_line('a')// &
            '  end subroutine show'//new_line('a')// &
            'end program main'

        test_cli_character_parameter_lowers = expect_cli_no_error( &
            source, '/tmp/ffc_cli_char_param_test')
    end function test_cli_character_parameter_lowers

    logical function test_cli_complex_scalar_lowers()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  complex :: z'//new_line('a')// &
            '  z = (1.0, 2.0)'//new_line('a')// &
            '  print *, z'//new_line('a')// &
            'end program main'

        test_cli_complex_scalar_lowers = expect_cli_no_error( &
            source, '/tmp/ffc_cli_scalar_type_test')
    end function test_cli_complex_scalar_lowers


    logical function test_cli_select_case_no_default_lowers()
        ! Same construct through the CLI: it compiles cleanly with no diagnostic.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: x'//new_line('a')// &
            '  x = 1'//new_line('a')// &
            '  select case (x)'//new_line('a')// &
            '  case (1)'//new_line('a')// &
            '    print *, 1'//new_line('a')// &
            '  end select'//new_line('a')// &
            'end program main'

        test_cli_select_case_no_default_lowers = expect_cli_no_error( &
            source, '/tmp/ffc_cli_select_case_test')
    end function test_cli_select_case_no_default_lowers

    logical function test_cli_do_zero_step_diagnostic()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  do i = 1, 3, 0'//new_line('a')// &
            '    print *, i'//new_line('a')// &
            '  end do'//new_line('a')// &
            'end program main'

        test_cli_do_zero_step_diagnostic = expect_cli_error_contains( &
            source, 'nonzero literal step', &
            '/tmp/ffc_cli_do_zero_step_test')
    end function test_cli_do_zero_step_diagnostic

    logical function test_cli_do_terminating_body_diagnostic()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  do i = 1, 3'//new_line('a')// &
            '    stop 1'//new_line('a')// &
            '  end do'//new_line('a')// &
            'end program main'

        test_cli_do_terminating_body_diagnostic = expect_cli_error_contains( &
            source, 'unsupported terminating do loop body', &
            '/tmp/ffc_cli_do_terminating_body_test')
    end function test_cli_do_terminating_body_diagnostic

    logical function test_cli_derived_type_diagnostic()
        ! A fixed-length scalar character component reads and writes through
        ! obj%comp (#265); a character ARRAY component stays unsupported since
        ! the flat slot layout only reserves inline byte storage for a scalar.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: point'//new_line('a')// &
            '    character(len=4) :: name(2)'//new_line('a')// &
            '  end type point'//new_line('a')// &
            '  type(point) :: p'//new_line('a')// &
            '  p%name(1) = "abcd"'//new_line('a')// &
            'end program main'

        test_cli_derived_type_diagnostic = expect_cli_error_contains( &
            source, &
            'character array', &
            '/tmp/ffc_cli_derived_type_test')
    end function test_cli_derived_type_diagnostic

    logical function test_cli_component_assignment_target_diagnostic()
        character(len=*), parameter :: expected = &
            'unsupported derived type component '// &
            'assignment target'
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  p%x = 1'//new_line('a')// &
            'end program main'

        test_cli_component_assignment_target_diagnostic = &
            expect_cli_error_contains(source, expected, &
            '/tmp/ffc_cli_component_target_test')
    end function test_cli_component_assignment_target_diagnostic

    logical function test_cli_component_expression_diagnostic()
        character(len=*), parameter :: expected = &
            'unsupported derived type component '// &
            'expression'
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  print *, p%x'//new_line('a')// &
            'end program main'

        test_cli_component_expression_diagnostic = expect_cli_error_contains( &
            source, expected, &
            '/tmp/ffc_cli_component_expr_test')
    end function test_cli_component_expression_diagnostic

    logical function test_cli_unsupported_intrinsic_diagnostic()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  print *, sqrt(4)'//new_line('a')// &
            'end program main'

        test_cli_unsupported_intrinsic_diagnostic = expect_cli_error_contains( &
            source, 'unsupported scalar intrinsic: sqrt', &
            '/tmp/ffc_cli_intrinsic_diagnostic_test')
    end function test_cli_unsupported_intrinsic_diagnostic

    logical function test_cli_external_function_call_diagnostic()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: x'//new_line('a')// &
            '  x = external_value(1)'//new_line('a')// &
            'end program main'

        test_cli_external_function_call_diagnostic = expect_cli_error_contains( &
            source, 'unsupported scalar function call', &
            '/tmp/ffc_cli_external_call_test')
    end function test_cli_external_function_call_diagnostic

    logical function test_cli_external_real_function_call_diagnostic()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real :: x'//new_line('a')// &
            '  x = external_real(1.0)'//new_line('a')// &
            'end program main'

        test_cli_external_real_function_call_diagnostic = &
            expect_cli_error_contains(source, &
            'unsupported scalar real(4) function call', &
            '/tmp/ffc_cli_external_real_call_test')
    end function test_cli_external_real_function_call_diagnostic

    logical function test_cli_external_logical_function_call_diagnostic()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  logical :: flag'//new_line('a')// &
            '  flag = external_flag(.true.)'// &
            new_line('a')// &
            'end program main'

        test_cli_external_logical_function_call_diagnostic = &
            expect_cli_error_contains(source, &
            'unsupported scalar logical function call', &
            '/tmp/ffc_cli_external_logical_call_test')
    end function test_cli_external_logical_function_call_diagnostic

    logical function test_cli_integer_exponent_operator_diagnostic()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: x'//new_line('a')// &
            '  x = 2 ** (-1)'//new_line('a')// &
            'end program main'

        test_cli_integer_exponent_operator_diagnostic = &
            expect_cli_error_contains(source, &
            'unsupported negative integer exponent', &
            '/tmp/ffc_cli_integer_exponent_operator_test')
    end function test_cli_integer_exponent_operator_diagnostic

    logical function test_cli_allocate_statement_diagnostic()
        ! The CLI retries an undeclared allocate target under lazy inference,
        ! which now declares values as an allocatable rank-1 integer and
        ! lowers the allocate, so the source compiles.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  allocate(values(3))'//new_line('a')// &
            'end program main'

        test_cli_allocate_statement_diagnostic = expect_cli_no_error( &
            source, &
            '/tmp/ffc_cli_allocate_test')
    end function test_cli_allocate_statement_diagnostic

    logical function test_cli_deallocate_statement_diagnostic()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  deallocate(values)'//new_line('a')// &
            'end program main'

        test_cli_deallocate_statement_diagnostic = expect_cli_error_contains( &
            source, &
            'unsupported deallocate statement', &
            '/tmp/ffc_cli_deallocate_test')
    end function test_cli_deallocate_statement_diagnostic

    logical function test_cli_return_statement_diagnostic()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  return'//new_line('a')// &
            'end program main'

        test_cli_return_statement_diagnostic = expect_cli_error_contains( &
            source, &
            'unsupported return statement', &
            '/tmp/ffc_cli_return_test')
    end function test_cli_return_statement_diagnostic

end program test_session_unsupported_diagnostics
