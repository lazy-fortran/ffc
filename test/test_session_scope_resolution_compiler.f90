program test_session_scope_resolution_compiler
    use ffc_test_support, only: expect_output
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

end program test_session_scope_resolution_compiler
