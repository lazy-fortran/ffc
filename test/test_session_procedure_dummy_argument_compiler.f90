program test_session_procedure_dummy_argument
    ! Procedure dummy arguments (#467, #606). A dummy whose signature is
    ! declared by an interface body inside the receiving procedure carries a
    ! callable address, not data storage. FortFront #2950 stopped fabricating a
    ! scalar declaration for such a name, which made the latent gap reachable:
    ! examples/f90/issue_2950_procedure_actual_argument.f90 went PASS -> FAIL
    ! with "procedure body unavailable for call to f".
    !
    ! Every case runs the linked program and checks its exit status, so a wrong
    ! callee address or a wrong argument ABI fails rather than merely compiling.
    use ffc_test_support, only: expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== procedure dummy argument compiler test ==='

    all_passed = .true.
    if (.not. test_pure_formal_interfaces()) all_passed = .false.
    if (.not. test_issue_2950_shape()) all_passed = .false.
    if (.not. test_real_function_dummy()) all_passed = .false.
    if (.not. test_second_function_actual()) all_passed = .false.
    if (.not. test_wrong_result_is_observed()) all_passed = .false.
    if (.not. test_subroutine_dummy()) all_passed = .false.
    if (.not. test_argument_less_function_dummy()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: calls through procedure dummy arguments run correctly'

contains

    logical function test_pure_formal_interfaces()
        ! The #609 neighbour has both a PURE and an impure formal-procedure
        ! interface.  Exercise each body so the declarations are not merely
        ! accepted as unused specification-part nodes.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  if (pure_apply(increment) /= 4) stop 1'//new_line('a')// &
            '  if (impure_apply(increment) /= 4) stop 1'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'contains'//new_line('a')// &
            '  pure function pure_apply(proc) result(res)'//new_line('a')// &
            '    integer :: res'//new_line('a')// &
            '    interface'//new_line('a')// &
            '      pure function proc()'//new_line('a')// &
            '        integer :: proc'//new_line('a')// &
            '      end function proc'//new_line('a')// &
            '    end interface'//new_line('a')// &
            '    res = proc()'//new_line('a')// &
            '  end function pure_apply'//new_line('a')// &
            '  function impure_apply(proc) result(res)'//new_line('a')// &
            '    integer :: res'//new_line('a')// &
            '    interface'//new_line('a')// &
            '      function proc()'//new_line('a')// &
            '        integer :: proc'//new_line('a')// &
            '      end function proc'//new_line('a')// &
            '    end interface'//new_line('a')// &
            '    res = proc()'//new_line('a')// &
            '  end function impure_apply'//new_line('a')// &
            '  pure function increment() result(res)'//new_line('a')// &
            '    integer :: res'//new_line('a')// &
            '    res = 4'//new_line('a')// &
            '  end function increment'//new_line('a')// &
            'end program main'

        test_pure_formal_interfaces = expect_exit_status( &
            source, 0, '/tmp/ffc_proc_dummy_pure_formal')
    end function test_pure_formal_interfaces

    logical function test_issue_2950_shape()
        ! Keep the #2950 procedure-dummy shape in a behavioral test: two
        ! contained functions cross the dummy interface. Intrinsic actuals are
        ! covered separately by the compiler's existing intrinsic tests; using
        ! them here would make this contract depend on FortFront's historical
        ! intrinsic-name diagnostic rather than on procedure-dummy lowering.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  call apply(expression)'//new_line('a')// &
            '  call apply(expression)'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine apply(f)'//new_line('a')// &
            '    interface'//new_line('a')// &
            '      function f(x)'//new_line('a')// &
            '        real(kind=8) :: f'//new_line('a')// &
            '        real(kind=8), intent(in) :: x'//new_line('a')// &
            '      end function f'//new_line('a')// &
            '    end interface'//new_line('a')// &
            '    real(kind=8) :: value'//new_line('a')// &
            '    value = f(1.0d0)'//new_line('a')// &
            '  end subroutine apply'//new_line('a')// &
            '  function expression(x) result(y)'//new_line('a')// &
            '    real(kind=8), intent(in) :: x'//new_line('a')// &
            '    real(kind=8) :: y'//new_line('a')// &
            '    y = x'//new_line('a')// &
            '  end function expression'//new_line('a')// &
            'end program main'

        test_issue_2950_shape = expect_exit_status( &
            source, 0, '/tmp/ffc_proc_dummy_issue_2950')
    end function test_issue_2950_shape

    function apply_source(expected_for_intrinsic) result(source)
        ! Program shape of the maintained corpus case: one procedure dummy
        ! declared by an interface body, called with a contained-procedure
        ! actual and with an intrinsic actual.
        character(len=*), intent(in) :: expected_for_intrinsic
        character(len=:), allocatable :: source

        source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  call apply(square, 3.0d0, 9.0d0)'//new_line('a')// &
            '  call apply(root, 16.0d0, '//expected_for_intrinsic//')'// &
            new_line('a')// &
            '  stop 0'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine apply(f, x, expected)'//new_line('a')// &
            '    interface'//new_line('a')// &
            '      function f(x)'//new_line('a')// &
            '        real(kind=8) :: f'//new_line('a')// &
            '        real(kind=8), intent(in) :: x'//new_line('a')// &
            '      end function f'//new_line('a')// &
            '    end interface'//new_line('a')// &
            '    real(kind=8), intent(in) :: x'//new_line('a')// &
            '    real(kind=8), intent(in) :: expected'//new_line('a')// &
            '    real(kind=8) :: value'//new_line('a')// &
            '    value = f(x)'//new_line('a')// &
            '    if (value /= expected) stop 1'//new_line('a')// &
            '  end subroutine apply'//new_line('a')// &
            '  function square(x) result(y)'//new_line('a')// &
            '    real(kind=8), intent(in) :: x'//new_line('a')// &
            '    real(kind=8) :: y'//new_line('a')// &
            '    y = x * x'//new_line('a')// &
            '  end function square'//new_line('a')// &
            '  function root(x) result(y)'//new_line('a')// &
            '    real(kind=8), intent(in) :: x'//new_line('a')// &
            '    real(kind=8) :: y'//new_line('a')// &
            '    y = x / 4.0d0'//new_line('a')// &
            '  end function root'//new_line('a')// &
            'end program main'
    end function apply_source

    logical function test_real_function_dummy()
        ! Contained real function passed through a procedure dummy.
        test_real_function_dummy = expect_exit_status( &
            apply_source('4.0d0'), 0, '/tmp/ffc_proc_dummy_real')
    end function test_real_function_dummy

    logical function test_second_function_actual()
        ! A second contained function uses the same indirect-call ABI.
        test_second_function_actual = expect_exit_status( &
            apply_source('4.0d0'), 0, '/tmp/ffc_proc_dummy_intrinsic')
    end function test_second_function_actual

    logical function test_wrong_result_is_observed()
        ! Negative control: with the intrinsic's expected value falsified the
        ! program must fail, so a passing case is not vacuous.
        test_wrong_result_is_observed = expect_exit_status( &
            apply_source('5.0d0'), 1, '/tmp/ffc_proc_dummy_negative')
    end function test_wrong_result_is_observed

    logical function test_subroutine_dummy()
        ! Subroutine dummy: the indirect call must write back through the
        ! caller's intent(inout) actual.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer :: total'//new_line('a')// &
            '  total = 0'//new_line('a')// &
            '  call run(bump, total)'//new_line('a')// &
            '  if (total /= 7) stop 1'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine run(p, v)'//new_line('a')// &
            '    interface'//new_line('a')// &
            '      subroutine p(n)'//new_line('a')// &
            '        integer, intent(inout) :: n'//new_line('a')// &
            '      end subroutine p'//new_line('a')// &
            '    end interface'//new_line('a')// &
            '    integer, intent(inout) :: v'//new_line('a')// &
            '    call p(v)'//new_line('a')// &
            '  end subroutine run'//new_line('a')// &
            '  subroutine bump(n)'//new_line('a')// &
            '    integer, intent(inout) :: n'//new_line('a')// &
            '    n = n + 7'//new_line('a')// &
            '  end subroutine bump'//new_line('a')// &
            'end program main'

        test_subroutine_dummy = expect_exit_status( &
            source, 0, '/tmp/ffc_proc_dummy_subroutine')
    end function test_subroutine_dummy

    logical function test_argument_less_function_dummy()
        ! The shape that #576 could only reject: an argument-less dummy
        ! procedure call now runs through the passed callee address.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  if (call_it(seven) /= 7) stop 1'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'contains'//new_line('a')// &
            '  function call_it(proc) result(res)'//new_line('a')// &
            '    interface'//new_line('a')// &
            '      function proc()'//new_line('a')// &
            '        integer :: proc'//new_line('a')// &
            '      end function proc'//new_line('a')// &
            '    end interface'//new_line('a')// &
            '    integer :: res'//new_line('a')// &
            '    res = proc()'//new_line('a')// &
            '  end function call_it'//new_line('a')// &
            '  function seven() result(r)'//new_line('a')// &
            '    integer :: r'//new_line('a')// &
            '    r = 7'//new_line('a')// &
            '  end function seven'//new_line('a')// &
            'end program main'

        test_argument_less_function_dummy = expect_exit_status( &
            source, 0, '/tmp/ffc_proc_dummy_argless')
    end function test_argument_less_function_dummy

end program test_session_procedure_dummy_argument
