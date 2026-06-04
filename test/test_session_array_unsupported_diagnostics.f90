program test_session_array_unsupported_diagnostics
    use ffc_test_support, only: expect_error_contains, &
                                expect_cli_error_contains
    implicit none

    logical :: all_passed

    print *, '=== direct session array unsupported diagnostic test ==='

    all_passed = .true.
    if (.not. test_array_rank3_declaration_diagnostic()) all_passed = .false.
    if (.not. test_array_zero_extent_diagnostic()) all_passed = .false.
    if (.not. test_array_reversed_extent_diagnostic()) all_passed = .false.
    if (.not. test_noninteger_array_declaration_diagnostic()) &
        all_passed = .false.
    if (.not. test_array_assignment_target_diagnostic()) all_passed = .false.
    if (.not. test_array_expression_diagnostic()) all_passed = .false.
    if (.not. test_whole_array_expression_diagnostic()) all_passed = .false.
    if (.not. test_array_rhs_assignment_diagnostic()) all_passed = .false.
    if (.not. test_array_slice_subscript_diagnostic()) all_passed = .false.
    if (.not. test_array_real_subscript_diagnostic()) all_passed = .false.
    if (.not. test_whole_array_assignment_target_diagnostic()) &
        all_passed = .false.
    if (.not. test_whole_array_argument_diagnostic()) all_passed = .false.
    if (.not. test_cli_array_rank3_declaration_diagnostic()) all_passed = .false.
    if (.not. test_cli_array_zero_extent_diagnostic()) all_passed = .false.
    if (.not. test_cli_array_reversed_extent_diagnostic()) all_passed = .false.
    if (.not. test_cli_noninteger_array_declaration_diagnostic()) &
        all_passed = .false.
    if (.not. test_cli_array_assignment_target_diagnostic()) all_passed = .false.
    if (.not. test_cli_array_expression_diagnostic()) all_passed = .false.
    if (.not. test_cli_whole_array_expression_diagnostic()) &
        all_passed = .false.
    if (.not. test_cli_array_rhs_assignment_diagnostic()) all_passed = .false.
    if (.not. test_cli_array_slice_subscript_diagnostic()) all_passed = .false.
    if (.not. test_cli_array_real_subscript_diagnostic()) all_passed = .false.
    if (.not. test_cli_whole_array_assignment_target_diagnostic()) &
        all_passed = .false.
    if (.not. test_cli_whole_array_argument_diagnostic()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: unsupported direct-session array features emit diagnostics'

contains

    logical function test_array_rank3_declaration_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: values(2, 2, 2)'//new_line('a')// &
                                       'end program main'

        test_array_rank3_declaration_diagnostic = expect_error_contains( &
                                            source, 'unsupported array declaration', &
                                            '/tmp/ffc_session_array_diagnostic_test')
    end function test_array_rank3_declaration_diagnostic

    logical function test_array_zero_extent_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: values(0)'//new_line('a')// &
                                       'end program main'

        test_array_zero_extent_diagnostic = expect_error_contains( &
                                            source, 'non-positive extent', &
                                            '/tmp/ffc_session_array_zero_test')
    end function test_array_zero_extent_diagnostic

    logical function test_array_reversed_extent_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: values(2:1)'//new_line('a')// &
                                       'end program main'

        test_array_reversed_extent_diagnostic = expect_error_contains( &
                                                source, 'non-positive extent', &
                                                '/tmp/ffc_session_array_reversed_test')
    end function test_array_reversed_extent_diagnostic

    logical function test_noninteger_array_declaration_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  real :: values(3)'//new_line('a')// &
                                       'end program main'

        test_noninteger_array_declaration_diagnostic = expect_error_contains( &
                                                    source, 'supports integer arrays', &
                                                     '/tmp/ffc_session_real_array_test')
    end function test_noninteger_array_declaration_diagnostic

    logical function test_array_assignment_target_diagnostic()
        character(len=*), parameter :: expected = &
                                       'unsupported array assignment target'
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: values(3)'//new_line('a')// &
                                       '  values(1, 1) = 1'//new_line('a')// &
                                       'end program main'

        test_array_assignment_target_diagnostic = &
            expect_error_contains(source, expected, &
                                  '/tmp/ffc_session_array_target_test')
    end function test_array_assignment_target_diagnostic

    logical function test_array_expression_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: values(3)'//new_line('a')// &
                                       '  print *, values(1, 1)'//new_line('a')// &
                                       'end program main'

        test_array_expression_diagnostic = expect_error_contains( &
                                           source, 'array expression', &
                                           '/tmp/ffc_session_array_expr_test')
    end function test_array_expression_diagnostic

    logical function test_whole_array_expression_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: values(3)'//new_line('a')// &
                                       '  print *, values'//new_line('a')// &
                                       'end program main'

        test_whole_array_expression_diagnostic = expect_error_contains( &
                                               source, 'unsupported array expression', &
                                               '/tmp/ffc_session_whole_array_expr_test')
    end function test_whole_array_expression_diagnostic

    logical function test_array_rhs_assignment_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: values(3)'//new_line('a')// &
                                       '  integer :: x'//new_line('a')// &
                                       '  x = values'//new_line('a')// &
                                       'end program main'

        test_array_rhs_assignment_diagnostic = expect_error_contains( &
                                               source, 'unsupported array expression', &
                                           '/tmp/ffc_session_array_rhs_assignment_test')
    end function test_array_rhs_assignment_diagnostic

    logical function test_array_slice_subscript_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: values(3)'//new_line('a')// &
                                       '  print *, values(1:2, 3:4)'//new_line('a')// &
                                       'end program main'

        test_array_slice_subscript_diagnostic = expect_error_contains( &
                                                source, 'too many subscripts', &
                                                '/tmp/ffc_session_array_slice_test')
    end function test_array_slice_subscript_diagnostic

    logical function test_array_real_subscript_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: values(3)'//new_line('a')// &
                                       '  print *, values(1.0)'//new_line('a')// &
                                       'end program main'

        test_array_real_subscript_diagnostic = expect_error_contains( &
                                               source, 'unsupported array subscript', &
                                               '/tmp/ffc_session_array_real_index_test')
    end function test_array_real_subscript_diagnostic

    logical function test_whole_array_assignment_target_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: values(3)'//new_line('a')// &
                                       '  values = 4'//new_line('a')// &
                                       'end program main'

        test_whole_array_assignment_target_diagnostic = expect_error_contains( &
                                        source, 'whole-array assignment requires array operands', &
                                         '/tmp/ffc_session_whole_array_assignment_test')
    end function test_whole_array_assignment_target_diagnostic

    logical function test_whole_array_argument_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: values(1)'//new_line('a')// &
                                       '  values(1) = 1'//new_line('a')// &
                                       '  call bump(values)'//new_line('a')// &
                                       'contains'//new_line('a')// &
                                       '  subroutine bump(x)'//new_line('a')// &
                                       '    integer :: x'//new_line('a')// &
                                       '    x = x + 4'//new_line('a')// &
                                       '  end subroutine bump'//new_line('a')// &
                                       'end program main'

        test_whole_array_argument_diagnostic = expect_error_contains( &
                                               source, 'unsupported array argument', &
                                               '/tmp/ffc_session_array_arg_test')
    end function test_whole_array_argument_diagnostic

    logical function test_cli_array_rank3_declaration_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: values(2, 2, 2)'//new_line('a')// &
                                       'end program main'

        test_cli_array_rank3_declaration_diagnostic = expect_cli_error_contains( &
                                              source, 'unsupported array declaration', &
                                                '/tmp/ffc_cli_array_diagnostic_test')
    end function test_cli_array_rank3_declaration_diagnostic

    logical function test_cli_array_zero_extent_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: values(0)'//new_line('a')// &
                                       'end program main'

        test_cli_array_zero_extent_diagnostic = expect_cli_error_contains( &
                                                source, 'non-positive extent', &
                                                '/tmp/ffc_cli_array_zero_test')
    end function test_cli_array_zero_extent_diagnostic

    logical function test_cli_array_reversed_extent_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: values(2:1)'//new_line('a')// &
                                       'end program main'

        test_cli_array_reversed_extent_diagnostic = expect_cli_error_contains( &
                                                    source, 'non-positive extent', &
                                                    '/tmp/ffc_cli_array_reversed_test')
    end function test_cli_array_reversed_extent_diagnostic

    logical function test_cli_noninteger_array_declaration_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  real :: values(3)'//new_line('a')// &
                                       'end program main'

        test_cli_noninteger_array_declaration_diagnostic = &
            expect_cli_error_contains(source, 'supports integer arrays', &
                                      '/tmp/ffc_cli_real_array_test')
    end function test_cli_noninteger_array_declaration_diagnostic

    logical function test_cli_array_assignment_target_diagnostic()
        character(len=*), parameter :: expected = &
                                       'unsupported array assignment target'
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: values(3)'//new_line('a')// &
                                       '  values(1, 1) = 1'//new_line('a')// &
                                       'end program main'

        test_cli_array_assignment_target_diagnostic = &
            expect_cli_error_contains(source, expected, &
                                      '/tmp/ffc_cli_array_target_test')
    end function test_cli_array_assignment_target_diagnostic

    logical function test_cli_array_expression_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: values(3)'//new_line('a')// &
                                       '  print *, values(1, 1)'//new_line('a')// &
                                       'end program main'

        test_cli_array_expression_diagnostic = expect_cli_error_contains( &
                                               source, 'array expression', &
                                               '/tmp/ffc_cli_array_expr_test')
    end function test_cli_array_expression_diagnostic

    logical function test_cli_whole_array_expression_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: values(3)'//new_line('a')// &
                                       '  print *, values'//new_line('a')// &
                                       'end program main'

        test_cli_whole_array_expression_diagnostic = expect_cli_error_contains( &
                                               source, 'unsupported array expression', &
                                                   '/tmp/ffc_cli_whole_array_expr_test')
    end function test_cli_whole_array_expression_diagnostic

    logical function test_cli_array_rhs_assignment_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: values(3)'//new_line('a')// &
                                       '  integer :: x'//new_line('a')// &
                                       '  x = values'//new_line('a')// &
                                       'end program main'

        test_cli_array_rhs_assignment_diagnostic = expect_cli_error_contains( &
                                               source, 'unsupported array expression', &
                                               '/tmp/ffc_cli_array_rhs_assignment_test')
    end function test_cli_array_rhs_assignment_diagnostic

    logical function test_cli_array_slice_subscript_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: values(3)'//new_line('a')// &
                                       '  print *, values(1:2, 3:4)'//new_line('a')// &
                                       'end program main'

        test_cli_array_slice_subscript_diagnostic = expect_cli_error_contains( &
                                                source, 'too many subscripts', &
                                                    '/tmp/ffc_cli_array_slice_test')
    end function test_cli_array_slice_subscript_diagnostic

    logical function test_cli_array_real_subscript_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: values(3)'//new_line('a')// &
                                       '  print *, values(1.0)'//new_line('a')// &
                                       'end program main'

        test_cli_array_real_subscript_diagnostic = expect_cli_error_contains( &
                                                source, 'unsupported array subscript', &
                                                   '/tmp/ffc_cli_array_real_index_test')
    end function test_cli_array_real_subscript_diagnostic

    logical function test_cli_whole_array_assignment_target_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: values(3)'//new_line('a')// &
                                       '  values = 4'//new_line('a')// &
                                       'end program main'

        test_cli_whole_array_assignment_target_diagnostic = &
            expect_cli_error_contains(source, &
                                      'whole-array assignment requires array operands', &
                                      '/tmp/ffc_cli_whole_array_assignment_test')
    end function test_cli_whole_array_assignment_target_diagnostic

    logical function test_cli_whole_array_argument_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: values(1)'//new_line('a')// &
                                       '  values(1) = 1'//new_line('a')// &
                                       '  call bump(values)'//new_line('a')// &
                                       'contains'//new_line('a')// &
                                       '  subroutine bump(x)'//new_line('a')// &
                                       '    integer :: x'//new_line('a')// &
                                       '    x = x + 4'//new_line('a')// &
                                       '  end subroutine bump'//new_line('a')// &
                                       'end program main'

        test_cli_whole_array_argument_diagnostic = expect_cli_error_contains( &
                                                 source, 'unsupported array argument', &
                                                   '/tmp/ffc_cli_array_arg_test')
    end function test_cli_whole_array_argument_diagnostic

end program test_session_array_unsupported_diagnostics
