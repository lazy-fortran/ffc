program test_session_unsupported_diagnostics
    use fortfront, only: compiler_frontend_options_t, &
                         compiler_frontend_result_t, &
                         compile_frontend_from_string, INPUT_MODE_STANDARD
    use session_program_lowering, only: lower_program_to_liric_exe
    implicit none

    logical :: all_passed

    print *, '=== direct session unsupported diagnostic test ==='

    all_passed = .true.
    if (.not. test_array_declaration_diagnostic()) all_passed = .false.
    if (.not. test_array_assignment_target_diagnostic()) all_passed = .false.
    if (.not. test_array_expression_diagnostic()) all_passed = .false.
    if (.not. test_character_expression_diagnostic()) all_passed = .false.
    if (.not. test_module_diagnostic()) all_passed = .false.
    if (.not. test_character_parameter_diagnostic()) all_passed = .false.
    if (.not. test_unsupported_scalar_type_diagnostic()) all_passed = .false.
    if (.not. test_exit_diagnostic()) all_passed = .false.
    if (.not. test_cycle_diagnostic()) all_passed = .false.
    if (.not. test_select_case_diagnostic()) all_passed = .false.
    if (.not. test_do_step_diagnostic()) all_passed = .false.
    if (.not. test_do_zero_step_diagnostic()) all_passed = .false.
    if (.not. test_derived_type_diagnostic()) all_passed = .false.
    if (.not. test_component_assignment_target_diagnostic()) &
        all_passed = .false.
    if (.not. test_component_expression_diagnostic()) all_passed = .false.
    if (.not. test_unsupported_intrinsic_diagnostic()) all_passed = .false.
    if (.not. test_external_function_call_diagnostic()) all_passed = .false.
    if (.not. test_external_real_function_call_diagnostic()) all_passed = .false.
    if (.not. test_external_logical_function_call_diagnostic()) &
        all_passed = .false.
    if (.not. test_real_exponent_operator_diagnostic()) all_passed = .false.
    if (.not. test_cli_array_declaration_diagnostic()) all_passed = .false.
    if (.not. test_cli_array_assignment_target_diagnostic()) all_passed = .false.
    if (.not. test_cli_array_expression_diagnostic()) all_passed = .false.
    if (.not. test_cli_character_expression_diagnostic()) all_passed = .false.
    if (.not. test_cli_module_diagnostic()) all_passed = .false.
    if (.not. test_cli_character_parameter_diagnostic()) all_passed = .false.
    if (.not. test_cli_unsupported_scalar_type_diagnostic()) &
        all_passed = .false.
    if (.not. test_cli_exit_diagnostic()) all_passed = .false.
    if (.not. test_cli_cycle_diagnostic()) all_passed = .false.
    if (.not. test_cli_select_case_diagnostic()) all_passed = .false.
    if (.not. test_cli_do_step_diagnostic()) all_passed = .false.
    if (.not. test_cli_do_zero_step_diagnostic()) all_passed = .false.
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
    if (.not. test_cli_real_exponent_operator_diagnostic()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: unsupported direct-session features emit diagnostics'

contains

    logical function test_array_declaration_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: values(3)'//new_line('a')// &
                                       'end program main'

        test_array_declaration_diagnostic = expect_error_contains( &
                                            source, 'unsupported array declaration', &
                                            '/tmp/ffc_session_array_diagnostic_test')
    end function test_array_declaration_diagnostic

    logical function test_array_assignment_target_diagnostic()
        character(len=*), parameter :: expected = &
                                       'unsupported array assignment target'
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  values(1) = 1'//new_line('a')// &
                                       'end program main'

        test_array_assignment_target_diagnostic = &
            expect_error_contains(source, expected, &
                                  '/tmp/ffc_session_array_target_test')
    end function test_array_assignment_target_diagnostic

    logical function test_array_expression_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  print *, values(1)'//new_line('a')// &
                                       'end program main'

        test_array_expression_diagnostic = expect_error_contains( &
                                           source, 'array expression', &
                                           '/tmp/ffc_session_array_expr_test')
    end function test_array_expression_diagnostic

    logical function test_character_expression_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  character(len=5) :: name'//new_line('a')// &
                                       '  name = "he" // "llo"'//new_line('a')// &
                                       'end program main'

        test_character_expression_diagnostic = expect_error_contains( &
                                           source, 'unsupported character assignment', &
                                           '/tmp/ffc_session_character_diagnostic_test')
    end function test_character_expression_diagnostic

    logical function test_module_diagnostic()
        character(len=*), parameter :: source = &
                                       'module m'//new_line('a')// &
                                       'end module m'

        test_module_diagnostic = expect_error_contains( &
                                 source, 'unsupported module program unit', &
                                 '/tmp/ffc_session_module_diagnostic_test')
    end function test_module_diagnostic

    logical function test_character_parameter_diagnostic()
        character(len=*), parameter :: expected = &
                                       'unsupported character parameter '// &
                                       'declaration'
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

        test_character_parameter_diagnostic = expect_error_contains( &
                                              source, expected, &
                                              '/tmp/ffc_session_char_param_test')
    end function test_character_parameter_diagnostic

    logical function test_unsupported_scalar_type_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  complex :: z'//new_line('a')// &
                                       'end program main'

        test_unsupported_scalar_type_diagnostic = expect_error_contains( &
                                                  source, 'unsupported scalar type', &
                                                  '/tmp/ffc_session_scalar_type_test')
    end function test_unsupported_scalar_type_diagnostic

    logical function test_exit_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: i'//new_line('a')// &
                                       '  do i = 1, 3'//new_line('a')// &
                                       '    exit'//new_line('a')// &
                                       '  end do'//new_line('a')// &
                                       'end program main'

        test_exit_diagnostic = expect_error_contains( &
                               source, 'unsupported exit statement', &
                               '/tmp/ffc_session_exit_diagnostic_test')
    end function test_exit_diagnostic

    logical function test_cycle_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: i'//new_line('a')// &
                                       '  do i = 1, 3'//new_line('a')// &
                                       '    cycle'//new_line('a')// &
                                       '  end do'//new_line('a')// &
                                       'end program main'

        test_cycle_diagnostic = expect_error_contains( &
                                source, 'unsupported cycle statement', &
                                '/tmp/ffc_session_cycle_diagnostic_test')
    end function test_cycle_diagnostic

    logical function test_select_case_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: x'//new_line('a')// &
                                       '  x = 1'//new_line('a')// &
                                       '  select case (x)'//new_line('a')// &
                                       '  case (1)'//new_line('a')// &
                                       '    print *, 1'//new_line('a')// &
                                       '  end select'//new_line('a')// &
                                       'end program main'

        test_select_case_diagnostic = expect_error_contains( &
                                    source, 'unsupported select case statement', &
                                    '/tmp/ffc_session_select_case_test')
    end function test_select_case_diagnostic

    logical function test_do_step_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: i, step'//new_line('a')// &
                                       '  step = 1'//new_line('a')// &
                                       '  do i = 1, 3, step'//new_line('a')// &
                                       '    print *, i'//new_line('a')// &
                                       '  end do'//new_line('a')// &
                                       'end program main'

        test_do_step_diagnostic = expect_error_contains( &
                                  source, 'unsupported do loop step', &
                                  '/tmp/ffc_session_do_step_test')
    end function test_do_step_diagnostic

    logical function test_do_zero_step_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: i'//new_line('a')// &
                                       '  do i = 1, 3, 0'//new_line('a')// &
                                       '    print *, i'//new_line('a')// &
                                       '  end do'//new_line('a')// &
                                       'end program main'

        test_do_zero_step_diagnostic = expect_error_contains( &
                                       source, 'nonzero literal step', &
                                       '/tmp/ffc_session_do_zero_step_test')
    end function test_do_zero_step_diagnostic

    logical function test_derived_type_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  type :: point'//new_line('a')// &
                                       '    integer :: x'//new_line('a')// &
                                       '  end type point'//new_line('a')// &
                                       'end program main'

        test_derived_type_diagnostic = expect_error_contains( &
                                       source, &
                                       'unsupported derived type definition', &
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
                                  'unsupported scalar real function call', &
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

    logical function test_real_exponent_operator_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  real :: x'//new_line('a')// &
                                       '  x = 2.0 ** 3.0'//new_line('a')// &
                                       'end program main'

        test_real_exponent_operator_diagnostic = expect_error_contains( &
            source, 'unsupported real operator', &
            '/tmp/ffc_session_real_exponent_operator_test')
    end function test_real_exponent_operator_diagnostic

    logical function test_cli_array_declaration_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: values(3)'//new_line('a')// &
                                       'end program main'

        test_cli_array_declaration_diagnostic = expect_cli_error_contains( &
                                              source, 'unsupported array declaration', &
                                                '/tmp/ffc_cli_array_diagnostic_test')
    end function test_cli_array_declaration_diagnostic

    logical function test_cli_array_assignment_target_diagnostic()
        character(len=*), parameter :: expected = &
                                       'unsupported array assignment target'
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  values(1) = 1'//new_line('a')// &
                                       'end program main'

        test_cli_array_assignment_target_diagnostic = &
            expect_cli_error_contains(source, expected, &
                                      '/tmp/ffc_cli_array_target_test')
    end function test_cli_array_assignment_target_diagnostic

    logical function test_cli_array_expression_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  print *, values(1)'//new_line('a')// &
                                       'end program main'

        test_cli_array_expression_diagnostic = expect_cli_error_contains( &
                                               source, 'array expression', &
                                               '/tmp/ffc_cli_array_expr_test')
    end function test_cli_array_expression_diagnostic

    logical function test_cli_character_expression_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  character(len=5) :: name'//new_line('a')// &
                                       '  name = "he" // "llo"'//new_line('a')// &
                                       'end program main'

        test_cli_character_expression_diagnostic = expect_cli_error_contains( &
                                           source, 'unsupported character assignment', &
                                               '/tmp/ffc_cli_character_diagnostic_test')
    end function test_cli_character_expression_diagnostic

    logical function test_cli_module_diagnostic()
        character(len=*), parameter :: source = &
                                       'module m'//new_line('a')// &
                                       'end module m'

        test_cli_module_diagnostic = expect_cli_error_contains( &
                                     source, 'unsupported module program unit', &
                                     '/tmp/ffc_cli_module_diagnostic_test')
    end function test_cli_module_diagnostic

    logical function test_cli_character_parameter_diagnostic()
        character(len=*), parameter :: expected = &
                                       'unsupported character parameter '// &
                                       'declaration'
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

        test_cli_character_parameter_diagnostic = expect_cli_error_contains( &
                                                  source, expected, &
                                                  '/tmp/ffc_cli_char_param_test')
    end function test_cli_character_parameter_diagnostic

    logical function test_cli_unsupported_scalar_type_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  complex :: z'//new_line('a')// &
                                       'end program main'

        test_cli_unsupported_scalar_type_diagnostic = expect_cli_error_contains( &
                                                    source, 'unsupported scalar type', &
                                                      '/tmp/ffc_cli_scalar_type_test')
    end function test_cli_unsupported_scalar_type_diagnostic

    logical function test_cli_exit_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: i'//new_line('a')// &
                                       '  do i = 1, 3'//new_line('a')// &
                                       '    exit'//new_line('a')// &
                                       '  end do'//new_line('a')// &
                                       'end program main'

        test_cli_exit_diagnostic = expect_cli_error_contains( &
                                   source, 'unsupported exit statement', &
                                   '/tmp/ffc_cli_exit_diagnostic_test')
    end function test_cli_exit_diagnostic

    logical function test_cli_cycle_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: i'//new_line('a')// &
                                       '  do i = 1, 3'//new_line('a')// &
                                       '    cycle'//new_line('a')// &
                                       '  end do'//new_line('a')// &
                                       'end program main'

        test_cli_cycle_diagnostic = expect_cli_error_contains( &
                                    source, 'unsupported cycle statement', &
                                    '/tmp/ffc_cli_cycle_diagnostic_test')
    end function test_cli_cycle_diagnostic

    logical function test_cli_select_case_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: x'//new_line('a')// &
                                       '  x = 1'//new_line('a')// &
                                       '  select case (x)'//new_line('a')// &
                                       '  case (1)'//new_line('a')// &
                                       '    print *, 1'//new_line('a')// &
                                       '  end select'//new_line('a')// &
                                       'end program main'

        test_cli_select_case_diagnostic = expect_cli_error_contains( &
                                    source, 'unsupported select case statement', &
                                    '/tmp/ffc_cli_select_case_test')
    end function test_cli_select_case_diagnostic

    logical function test_cli_do_step_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: i, step'//new_line('a')// &
                                       '  step = 1'//new_line('a')// &
                                       '  do i = 1, 3, step'//new_line('a')// &
                                       '    print *, i'//new_line('a')// &
                                       '  end do'//new_line('a')// &
                                       'end program main'

        test_cli_do_step_diagnostic = expect_cli_error_contains( &
                                      source, 'unsupported do loop step', &
                                      '/tmp/ffc_cli_do_step_test')
    end function test_cli_do_step_diagnostic

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

    logical function test_cli_derived_type_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  type :: point'//new_line('a')// &
                                       '    integer :: x'//new_line('a')// &
                                       '  end type point'//new_line('a')// &
                                       'end program main'

        test_cli_derived_type_diagnostic = expect_cli_error_contains( &
                                           source, &
                                           'unsupported derived type definition', &
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
                                      'unsupported scalar real function call', &
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

    logical function test_cli_real_exponent_operator_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  real :: x'//new_line('a')// &
                                       '  x = 2.0 ** 3.0'//new_line('a')// &
                                       'end program main'

        test_cli_real_exponent_operator_diagnostic = expect_cli_error_contains( &
            source, 'unsupported real operator', &
            '/tmp/ffc_cli_real_exponent_operator_test')
    end function test_cli_real_exponent_operator_diagnostic

    logical function expect_error_contains(source, expected, exe_path)
        character(len=*), intent(in) :: source
        character(len=*), intent(in) :: expected
        character(len=*), intent(in) :: exe_path
        character(len=:), allocatable :: error_msg

        expect_error_contains = .false.
        call compile_and_lower(source, exe_path, error_msg)
        call execute_command_line('rm -f '//exe_path)

        if (len_trim(error_msg) == 0) then
            print *, 'FAIL: unsupported source lowered without error'
            return
        end if
        if (index(error_msg, expected) <= 0) then
            print *, 'FAIL: expected diagnostic substring ', expected
            print *, '  got ', trim(error_msg)
            return
        end if
        if (index(error_msg, 'line ') <= 0 .or. &
            index(error_msg, 'column ') <= 0) then
            print *, 'FAIL: expected line/column diagnostic, got ', &
                trim(error_msg)
            return
        end if

        expect_error_contains = .true.
    end function expect_error_contains

    logical function expect_cli_error_contains(source, expected, stem)
        character(len=*), intent(in) :: source
        character(len=*), intent(in) :: expected
        character(len=*), intent(in) :: stem
        character(len=:), allocatable :: output
        character(len=:), allocatable :: command
        character(len=:), allocatable :: source_path
        character(len=:), allocatable :: output_path
        character(len=:), allocatable :: exe_path
        integer :: exit_stat
        integer :: cmd_stat

        expect_cli_error_contains = .false.
        source_path = stem//'.f90'
        output_path = stem//'.out'
        exe_path = stem//'.exe'
        call execute_command_line('rm -f '//source_path//' '//output_path//' '// &
                                  exe_path)
        if (.not. write_source_file(source_path, source)) return

        command = "sh -c 'exe=$(ls -t build/*/app/ffc 2>/dev/null | "// &
                  "head -n 1); "// &
                  "test -n ""$exe"" && ""$exe"" "// &
                  source_path//' -o '//exe_path//' > '//output_path//" 2>&1'"
        call execute_command_line(command, exitstat=exit_stat, cmdstat=cmd_stat)
        output = read_text_file(output_path)
        call execute_command_line('rm -f '//source_path//' '//output_path//' '// &
                                  exe_path)

        if (cmd_stat /= 0) then
            print *, 'FAIL: ffc CLI command could not be executed'
            return
        end if
        if (exit_stat == 0) then
            print *, 'FAIL: unsupported source CLI exited successfully'
            return
        end if
        if (index(output, expected) <= 0) then
            print *, 'FAIL: expected CLI diagnostic substring ', expected
            print *, '  got ', trim(output)
            return
        end if
        if (index(output, 'line ') <= 0 .or. index(output, 'column ') <= 0) then
            print *, 'FAIL: expected CLI line/column diagnostic, got ', &
                trim(output)
            return
        end if

        expect_cli_error_contains = .true.
    end function expect_cli_error_contains

    logical function write_source_file(path, source)
        character(len=*), intent(in) :: path
        character(len=*), intent(in) :: source
        integer :: unit
        integer :: io_stat

        write_source_file = .false.
        open (newunit=unit, file=path, status='replace', action='write', &
              iostat=io_stat)
        if (io_stat /= 0) then
            print *, 'FAIL: could not create source file ', path
            return
        end if

        write (unit, '(A)', iostat=io_stat) source
        close (unit)
        if (io_stat /= 0) then
            print *, 'FAIL: could not write source file ', path
            return
        end if

        write_source_file = .true.
    end function write_source_file

    function read_text_file(path) result(text)
        character(len=*), intent(in) :: path
        character(len=:), allocatable :: text
        character(len=512) :: line
        integer :: unit
        integer :: io_stat

        text = ''
        open (newunit=unit, file=path, status='old', action='read', &
              iostat=io_stat)
        if (io_stat /= 0) return

        do
            read (unit, '(A)', iostat=io_stat) line
            if (io_stat /= 0) exit
            text = text//trim(line)//new_line('a')
        end do
        close (unit)
    end function read_text_file

    subroutine compile_and_lower(source, exe_path, error_msg)
        character(len=*), intent(in) :: source
        character(len=*), intent(in) :: exe_path
        character(len=:), allocatable, intent(out) :: error_msg
        type(compiler_frontend_options_t) :: options
        type(compiler_frontend_result_t) :: frontend_result

        options = compiler_frontend_options_t()
        options%run_semantics = .true.
        options%input_mode = INPUT_MODE_STANDARD

        call compile_frontend_from_string(source, frontend_result, options)
        if (.not. frontend_result%success()) then
            error_msg = 'FortFront rejected source: '// &
                        trim(frontend_result%diagnostic_text)
            return
        end if

        call lower_program_to_liric_exe(frontend_result%arena, &
                                        frontend_result%root_index, exe_path, &
                                        error_msg)
    end subroutine compile_and_lower
end program test_session_unsupported_diagnostics
