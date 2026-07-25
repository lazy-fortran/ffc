program test_session_scope_resolution_compiler
    use ffc_test_support, only: expect_error_contains, expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session scoped declaration resolution compiler test ==='

    all_passed = .true.
    if (.not. test_host_variable_bound_shadows_unrelated_module()) &
        all_passed = .false.
    if (.not. test_host_kind_parameter_shadows_unrelated_module()) &
        all_passed = .false.
    if (.not. test_function_header_kind_parameter_shadows_dp()) &
        all_passed = .false.
    if (.not. test_function_result_kind_parameter_shadows_dp()) &
        all_passed = .false.
    if (.not. test_allocatable_kind_parameter_shadows_dp()) &
        all_passed = .false.
    if (.not. test_pointer_target_array_kind_parameter_shadows_dp()) &
        all_passed = .false.
    if (.not. test_binding_identity_spans_block_shadow_and_host()) &
        all_passed = .false.
    if (.not. test_unresolved_binding_is_not_synthesized_from_text()) &
        all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: specification expressions resolve names lexically'

contains

    logical function test_host_variable_bound_shadows_unrelated_module()
        character(len=*), parameter :: source = &
            'module unrelated'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer :: n = 2'//new_line('a')// &
            'end module unrelated'//new_line('a')// &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer :: n = 4'//new_line('a')// &
            '  call show()'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine show()'//new_line('a')// &
            '    integer :: a(n)'//new_line('a')// &
            '    print *, size(a)'//new_line('a')// &
            '  end subroutine show'//new_line('a')// &
            'end program main'

        test_host_variable_bound_shadows_unrelated_module = expect_output( &
            source, '           4'//new_line('a'), &
            '/tmp/ffc_session_scope_host_bound_test')
    end function test_host_variable_bound_shadows_unrelated_module

    logical function test_host_kind_parameter_shadows_unrelated_module()
        character(len=*), parameter :: source = &
            'module unrelated'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer, parameter :: rk = 8'//new_line('a')// &
            'end module unrelated'//new_line('a')// &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer, parameter :: rk = 4'//new_line('a')// &
            '  call show()'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine show()'//new_line('a')// &
            '    print *, 1.5_rk'//new_line('a')// &
            '  end subroutine show'//new_line('a')// &
            'end program main'

        test_host_kind_parameter_shadows_unrelated_module = expect_output( &
            source, '   1.50000000    '//new_line('a'), &
            '/tmp/ffc_session_scope_host_kind_test')
    end function test_host_kind_parameter_shadows_unrelated_module

    logical function test_function_header_kind_parameter_shadows_dp()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer, parameter :: dp = 4'//new_line('a')// &
            '  print *, value()'//new_line('a')// &
            'contains'//new_line('a')// &
            '  real(dp) function value()'//new_line('a')// &
            '    value = 1.5_dp'//new_line('a')// &
            '  end function value'//new_line('a')// &
            'end program main'

        test_function_header_kind_parameter_shadows_dp = expect_output( &
            source, '   1.50000000    '//new_line('a'), &
            '/tmp/ffc_session_scope_function_header_kind_test')
    end function test_function_header_kind_parameter_shadows_dp

    logical function test_function_result_kind_parameter_shadows_dp()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer, parameter :: dp = 4'//new_line('a')// &
            '  print *, value()'//new_line('a')// &
            'contains'//new_line('a')// &
            '  function value() result(r)'//new_line('a')// &
            '    real(dp) :: r'//new_line('a')// &
            '    r = 1.5_dp'//new_line('a')// &
            '  end function value'//new_line('a')// &
            'end program main'

        test_function_result_kind_parameter_shadows_dp = expect_output( &
            source, '   1.50000000    '//new_line('a'), &
            '/tmp/ffc_session_scope_function_result_kind_test')
    end function test_function_result_kind_parameter_shadows_dp

    logical function test_allocatable_kind_parameter_shadows_dp()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer, parameter :: dp = 4'//new_line('a')// &
            '  real(dp), allocatable :: a(:)'//new_line('a')// &
            '  allocate(a(1))'//new_line('a')// &
            '  a(1) = 1.5_dp'//new_line('a')// &
            '  print *, a(1)'//new_line('a')// &
            'end program main'

        test_allocatable_kind_parameter_shadows_dp = expect_output( &
            source, '   1.50000000    '//new_line('a'), &
            '/tmp/ffc_session_scope_allocatable_kind_test')
    end function test_allocatable_kind_parameter_shadows_dp

    logical function test_pointer_target_array_kind_parameter_shadows_dp()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer, parameter :: dp = 4'//new_line('a')// &
            '  real(dp), target :: a(1)'//new_line('a')// &
            '  a(1) = 1.5_dp'//new_line('a')// &
            '  print *, a(1)'//new_line('a')// &
            'end program main'

        test_pointer_target_array_kind_parameter_shadows_dp = expect_output( &
            source, '   1.50000000    '//new_line('a'), &
            '/tmp/ffc_session_scope_pointer_target_kind_test')
    end function test_pointer_target_array_kind_parameter_shadows_dp

    logical function test_binding_identity_spans_block_shadow_and_host()
        ! Three references, three different binding identities (#327):
        !   `later`  a named constant declared later in `report`'s
        !            specification part than the variable `n` it is printed
        !            beside, referenced from an inner BLOCK  -> 4
        !   `n`      the BLOCK-local shadow, not `report`'s own `n` = 7 -> 2
        !   `hosted` host-associated from the program's specification part,
        !            which has no symbol in `report`'s lowering context -> 1
        ! Keying by text cannot distinguish these: `hosted` resolves to no
        ! symbol at all, and only the FortFront binding says which
        ! declaration it names.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer :: shadowed'//new_line('a')// &
            '  integer, parameter :: hosted = 1'//new_line('a')// &
            '  shadowed = 9'//new_line('a')// &
            '  call report()'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine report()'//new_line('a')// &
            '    integer :: n'//new_line('a')// &
            '    integer, parameter :: later = 4'//new_line('a')// &
            '    n = 7'//new_line('a')// &
            '    block'//new_line('a')// &
            '      integer :: n'//new_line('a')// &
            '      n = 2'//new_line('a')// &
            '      print *, later, n, hosted'//new_line('a')// &
            '    end block'//new_line('a')// &
            '  end subroutine report'//new_line('a')// &
            'end program main'

        test_binding_identity_spans_block_shadow_and_host = expect_output( &
            source, '           4           2           1'//new_line('a'), &
            '/tmp/ffc_session_scope_binding_identity_test')
    end function test_binding_identity_spans_block_shadow_and_host

    logical function test_unresolved_binding_is_not_synthesized_from_text()
        ! `token` is a named constant of a module the program never USEs, so
        ! FortFront resolves no binding for it. ffc must report the existing
        ! undeclared-name diagnostic rather than fold the identically spelled
        ! constant it can see elsewhere in the arena (#327).
        character(len=*), parameter :: source = &
            'module hidden'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer, parameter :: token = 4'//new_line('a')// &
            'end module hidden'//new_line('a')// &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer :: v'//new_line('a')// &
            '  v = token'//new_line('a')// &
            '  print *, v'//new_line('a')// &
            'end program main'

        test_unresolved_binding_is_not_synthesized_from_text = &
            expect_error_contains(source, &
            'integer identifier was not declared: token', &
            '/tmp/ffc_session_scope_unresolved_binding_test')
    end function test_unresolved_binding_is_not_synthesized_from_text

end program test_session_scope_resolution_compiler
