program test_session_reject_result_01_compiler
    ! Function-result and ENTRY rules (#379). Each invalid form below is taken
    ! from the gfortran.dg fixtures entry_15.f90, entry_dummy_ref_2.f90,
    ! func_assign.f90, func_result_7.f90, pr39695_2.f90 and pr39695_3.f90 and
    ! must be rejected with a source diagnostic; every corrected neighbour must
    ! still compile and run.
    use ffc_test_support, only: expect_error_contains, expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== function-result and ENTRY rejection test ==='

    all_passed = .true.
    if (.not. test_assign_to_interface_procedure_rejected()) all_passed = .false.
    if (.not. test_assign_to_sibling_procedure_rejected()) all_passed = .false.
    if (.not. test_assign_to_entry_name_rejected()) all_passed = .false.
    if (.not. test_entry_dummy_before_entry_rejected()) all_passed = .false.
    if (.not. test_self_named_interface_rejected()) all_passed = .false.
    if (.not. test_function_name_attribute_rejected()) all_passed = .false.
    if (.not. test_assign_to_own_result_accepted()) all_passed = .false.
    if (.not. test_foreign_interface_name_accepted()) all_passed = .false.
    if (.not. test_result_without_attributes_accepted()) all_passed = .false.
    if (.not. test_local_named_like_procedure_accepted()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: function-result and ENTRY rules enforced'

contains

    logical function test_assign_to_interface_procedure_rejected()
        character(len=*), parameter :: source = &
            'module mod'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine a()'//new_line('a')// &
            '    implicit none'//new_line('a')// &
            '    interface'//new_line('a')// &
            '      subroutine sub()'//new_line('a')// &
            '      end subroutine sub'//new_line('a')// &
            '    end interface'//new_line('a')// &
            '    sub = 3'//new_line('a')// &
            '  end subroutine a'//new_line('a')// &
            'end module mod'

        test_assign_to_interface_procedure_rejected = expect_error_contains( &
            source, 'is not a variable', '/tmp/ffc_reject_result_iface_assign')
    end function test_assign_to_interface_procedure_rejected

    logical function test_assign_to_sibling_procedure_rejected()
        character(len=*), parameter :: source = &
            'module mod'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            'contains'//new_line('a')// &
            '  integer function bar()'//new_line('a')// &
            '    bar = 4'//new_line('a')// &
            '  end function bar'//new_line('a')// &
            '  subroutine a()'//new_line('a')// &
            '    implicit none'//new_line('a')// &
            '    bar = 5'//new_line('a')// &
            '  end subroutine a'//new_line('a')// &
            'end module mod'

        test_assign_to_sibling_procedure_rejected = expect_error_contains( &
            source, 'is not a variable', '/tmp/ffc_reject_result_sibling_assign')
    end function test_assign_to_sibling_procedure_rejected

    logical function test_assign_to_entry_name_rejected()
        character(len=*), parameter :: source = &
            'module m2'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            'contains'//new_line('a')// &
            'function func(a)'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer :: a, func'//new_line('a')// &
            '  real :: func2'//new_line('a')// &
            '  func = a*8'//new_line('a')// &
            '  return'//new_line('a')// &
            'entry ent(a) result(func2)'//new_line('a')// &
            '  ent = -a*4.0'//new_line('a')// &
            '  return'//new_line('a')// &
            'end function func'//new_line('a')// &
            'end module m2'

        test_assign_to_entry_name_rejected = expect_error_contains( &
            source, 'is not a variable', '/tmp/ffc_reject_result_entry_assign')
    end function test_assign_to_entry_name_rejected

    logical function test_entry_dummy_before_entry_rejected()
        character(len=*), parameter :: source = &
            'MODULE M1'//new_line('a')// &
            'CONTAINS'//new_line('a')// &
            'FUNCTION F1(I) RESULT(RF1)'//new_line('a')// &
            ' INTEGER :: I,K,RE1,RF1'//new_line('a')// &
            ' RE1=K'//new_line('a')// &
            ' RETURN'//new_line('a')// &
            ' ENTRY E1(K) RESULT(RE1)'//new_line('a')// &
            ' RE1=-I'//new_line('a')// &
            ' RETURN'//new_line('a')// &
            'END FUNCTION F1'//new_line('a')// &
            'END MODULE M1'

        test_entry_dummy_before_entry_rejected = expect_error_contains( &
            source, 'before the ENTRY statement', &
            '/tmp/ffc_reject_result_entry_dummy')
    end function test_entry_dummy_before_entry_rejected

    logical function test_self_named_interface_rejected()
        character(len=*), parameter :: source = &
            'function g()'//new_line('a')// &
            '  interface'//new_line('a')// &
            '    subroutine g()'//new_line('a')// &
            '    end subroutine g'//new_line('a')// &
            '  end interface'//new_line('a')// &
            '  real g'//new_line('a')// &
            'end function'

        test_self_named_interface_rejected = expect_error_contains( &
            source, 'outside its INTERFACE body', &
            '/tmp/ffc_reject_result_self_iface')
    end function test_self_named_interface_rejected

    logical function test_function_name_attribute_rejected()
        character(len=*), parameter :: source = &
            'function fun() result(f)'//new_line('a')// &
            '  pointer fun'//new_line('a')// &
            '  dimension fun(1)'//new_line('a')// &
            '  f=0'//new_line('a')// &
            'end'

        test_function_name_attribute_rejected = expect_error_contains( &
            source, 'cannot carry attributes', &
            '/tmp/ffc_reject_result_name_attr')
    end function test_function_name_attribute_rejected

    logical function test_assign_to_own_result_accepted()
        ! Corrected neighbour of func_assign.f90 and entry_15.f90: the only
        ! assignable name inside a function is its own result variable.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  stop bar() + baz()'//new_line('a')// &
            'contains'//new_line('a')// &
            '  integer function bar()'//new_line('a')// &
            '    bar = 4'//new_line('a')// &
            '  end function bar'//new_line('a')// &
            '  integer function baz() result(r)'//new_line('a')// &
            '    r = 3'//new_line('a')// &
            '  end function baz'//new_line('a')// &
            'end program main'

        test_assign_to_own_result_accepted = expect_exit_status( &
            source, 7, '/tmp/ffc_reject_result_own_result_ok')
    end function test_assign_to_own_result_accepted

    logical function test_foreign_interface_name_accepted()
        ! Corrected neighbour of pr39695_2.f90 and pr39695_3.f90: an interface
        ! block naming a procedure other than the enclosing function is fine.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  stop g()'//new_line('a')// &
            'contains'//new_line('a')// &
            '  integer function g()'//new_line('a')// &
            '    interface'//new_line('a')// &
            '      subroutine h()'//new_line('a')// &
            '      end subroutine h'//new_line('a')// &
            '    end interface'//new_line('a')// &
            '    g = 5'//new_line('a')// &
            '  end function g'//new_line('a')// &
            'end program main'

        test_foreign_interface_name_accepted = expect_exit_status( &
            source, 5, '/tmp/ffc_reject_result_foreign_iface_ok')
    end function test_foreign_interface_name_accepted

    logical function test_result_without_attributes_accepted()
        ! Corrected neighbour of func_result_7.f90: with the attribute
        ! statements on the function name removed the unit is valid.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  stop fun()'//new_line('a')// &
            'contains'//new_line('a')// &
            '  integer function fun() result(f)'//new_line('a')// &
            '    f = 6'//new_line('a')// &
            '  end function fun'//new_line('a')// &
            'end program main'

        test_result_without_attributes_accepted = expect_exit_status( &
            source, 6, '/tmp/ffc_reject_result_plain_result_ok')
    end function test_result_without_attributes_accepted

    logical function test_local_named_like_procedure_accepted()
        ! A locally declared variable that happens to share its spelling with
        ! a procedure elsewhere in the unit stays assignable.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  stop use_it()'//new_line('a')// &
            'contains'//new_line('a')// &
            '  integer function helper()'//new_line('a')// &
            '    helper = 1'//new_line('a')// &
            '  end function helper'//new_line('a')// &
            '  integer function use_it()'//new_line('a')// &
            '    integer :: helper'//new_line('a')// &
            '    helper = 9'//new_line('a')// &
            '    use_it = helper'//new_line('a')// &
            '  end function use_it'//new_line('a')// &
            'end program main'

        test_local_named_like_procedure_accepted = expect_exit_status( &
            source, 9, '/tmp/ffc_reject_result_local_shadow_ok')
    end function test_local_named_like_procedure_accepted

end program test_session_reject_result_01_compiler
