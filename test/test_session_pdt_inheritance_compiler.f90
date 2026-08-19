program test_session_pdt_inheritance_compiler
    ! A parameterized derived type that EXTENDS another parameterized derived
    ! type (#621). The child's full formal type-parameter list is the parent's
    ! formals followed by its own, so one instance of the child pins one
    ! instance of the parent: the parent instance is laid out first and the
    ! child inherits its components ahead of its own.
    use ffc_test_support, only: expect_output, &
        expect_error_contains
    implicit none

    logical :: all_passed

    print *, '=== direct session PDT inheritance compiler test ==='

    all_passed = .true.
    if (.not. test_defaulted_parameters_inherit_components()) all_passed = .false.
    if (.not. test_explicit_parent_and_own_actuals()) all_passed = .false.
    if (.not. test_excess_actuals_still_rejected()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: PDT inheritance lowers through direct LIRIC session'

contains

    logical function test_defaulted_parameters_inherit_components()
        ! Reduced from lfortran/pdt_13.f90: a child PDT in a second module
        ! extends a parent PDT and both type parameters take their defaults.
        character(len=*), parameter :: source = &
            'module pdt_parent_m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  type :: parent_t(k)'//new_line('a')// &
            '    integer, kind :: k = kind(1.0)'//new_line('a')// &
            '    integer :: x = 0'//new_line('a')// &
            '  end type'//new_line('a')// &
            'end module'//new_line('a')// &
            'module pdt_child_m'//new_line('a')// &
            '  use pdt_parent_m, only: parent_t'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  type, extends(parent_t) :: child_t(m)'//new_line('a')// &
            '    integer, kind :: m = kind(1.0)'//new_line('a')// &
            '    integer :: y = 0'//new_line('a')// &
            '  end type'//new_line('a')// &
            'end module'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use pdt_child_m, only: child_t'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  type(child_t) :: child'//new_line('a')// &
            '  child%x = 42'//new_line('a')// &
            '  child%y = 7'//new_line('a')// &
            '  if (child%x /= 42) error stop 1'//new_line('a')// &
            '  if (child%y /= 7) error stop 2'//new_line('a')// &
            '  print *, child%x'//new_line('a')// &
            'end program main'

        test_defaulted_parameters_inherit_components = expect_output( &
            source, '          42'//new_line('a'), &
            '/tmp/ffc_pdt_inherit_defaults')
    end function test_defaulted_parameters_inherit_components

    logical function test_explicit_parent_and_own_actuals()
        ! The child names the inherited formal first, then its own, and two
        ! distinct tuples get distinct layouts.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: base_t(n)'//new_line('a')// &
            '    integer, len :: n'//new_line('a')// &
            '    integer :: head(n)'//new_line('a')// &
            '  end type base_t'//new_line('a')// &
            '  type, extends(base_t) :: leaf_t(p)'//new_line('a')// &
            '    integer, len :: p'//new_line('a')// &
            '    integer :: tail(p)'//new_line('a')// &
            '  end type leaf_t'//new_line('a')// &
            '  type(leaf_t(2, 3)) :: a'//new_line('a')// &
            '  type(leaf_t(4, 2)) :: b'//new_line('a')// &
            '  a%head(2) = 20'//new_line('a')// &
            '  a%tail(3) = 300'//new_line('a')// &
            '  b%head(4) = 40'//new_line('a')// &
            '  b%tail(1) = 100'//new_line('a')// &
            '  print *, a%head(2) + a%tail(3)'//new_line('a')// &
            '  print *, b%head(4) + b%tail(1)'//new_line('a')// &
            'end program main'

        test_explicit_parent_and_own_actuals = expect_output( &
            source, &
            '         320'//new_line('a')// &
            '         140'//new_line('a'), &
            '/tmp/ffc_pdt_inherit_actuals')
    end function test_explicit_parent_and_own_actuals

    logical function test_excess_actuals_still_rejected()
        ! The inherited formals widen the accepted arity but do not remove the
        ! arity check.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: base_t(n)'//new_line('a')// &
            '    integer, len :: n'//new_line('a')// &
            '    integer :: head(n)'//new_line('a')// &
            '  end type base_t'//new_line('a')// &
            '  type, extends(base_t) :: leaf_t(p)'//new_line('a')// &
            '    integer, len :: p'//new_line('a')// &
            '    integer :: tail(p)'//new_line('a')// &
            '  end type leaf_t'//new_line('a')// &
            '  type(leaf_t(2, 3, 4)) :: a'//new_line('a')// &
            '  a%head(1) = 1'//new_line('a')// &
            'end program main'

        test_excess_actuals_still_rejected = expect_error_contains( &
            source, 'too many actual type parameters', &
            '/tmp/ffc_pdt_inherit_excess')
    end function test_excess_actuals_still_rejected

end program test_session_pdt_inheritance_compiler
