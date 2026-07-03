program test_session_derived_constructor_compiler
    use ffc_test_support, only: expect_exit_status, expect_output, &
        expect_error_contains
    implicit none

    logical :: all_passed

    print *, '=== direct session derived constructor compiler test ==='

    all_passed = .true.
    if (.not. test_positional_multi_component()) all_passed = .false.
    if (.not. test_real_component_constructor()) all_passed = .false.
    if (.not. test_logical_and_double_constructor()) all_passed = .false.
    if (.not. test_omitted_default_component()) all_passed = .false.
    if (.not. test_whole_derived_copy()) all_passed = .false.
    if (.not. test_inherited_component_constructor()) all_passed = .false.
    if (.not. test_nested_constructor_unsupported()) all_passed = .false.
    if (.not. test_interface_constructor_unsupported()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: derived constructors lower through direct LIRIC'

contains

    logical function test_positional_multi_component()
        ! t(3, 4): positional arguments fill components in declaration order.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: t'//new_line('a')// &
            '    integer :: a'//new_line('a')// &
            '    integer :: b'//new_line('a')// &
            '  end type t'//new_line('a')// &
            '  type(t) :: x'//new_line('a')// &
            '  x = t(3, 4)'//new_line('a')// &
            '  stop x%a + x%b'//new_line('a')// &
            'end program main'

        test_positional_multi_component = expect_exit_status( &
            source, 7, '/tmp/ffc_session_ctor_positional')
    end function test_positional_multi_component

    logical function test_real_component_constructor()
        ! pt(1.5, 2.5): real components take their positional real values.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: pt'//new_line('a')// &
            '    real :: x'//new_line('a')// &
            '    real :: y'//new_line('a')// &
            '  end type pt'//new_line('a')// &
            '  type(pt) :: p'//new_line('a')// &
            '  p = pt(1.5, 2.5)'//new_line('a')// &
            '  print *, p%x, p%y'//new_line('a')// &
            'end program main'

        test_real_component_constructor = expect_output( &
            source, '   1.50000000       2.50000000    '//new_line('a'), &
            '/tmp/ffc_session_ctor_real')
    end function test_real_component_constructor

    logical function test_logical_and_double_constructor()
        ! Mixed logical / real(8) / integer components store at their own kinds.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, parameter :: dp = kind(0.d0)'//new_line('a')// &
            '  type :: t'//new_line('a')// &
            '    logical :: flag'//new_line('a')// &
            '    real(dp) :: d'//new_line('a')// &
            '    integer :: n'//new_line('a')// &
            '  end type t'//new_line('a')// &
            '  type(t) :: x'//new_line('a')// &
            '  x = t(.true., 3.25_dp, 9)'//new_line('a')// &
            '  if (.not. x%flag) error stop'//new_line('a')// &
            '  if (x%d /= 3.25_dp) error stop'//new_line('a')// &
            '  stop x%n'//new_line('a')// &
            'end program main'

        test_logical_and_double_constructor = expect_exit_status( &
            source, 9, '/tmp/ffc_session_ctor_logical_double')
    end function test_logical_and_double_constructor

    logical function test_omitted_default_component()
        ! t(3): the omitted component keeps its compile-time default.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: t'//new_line('a')// &
            '    integer :: a'//new_line('a')// &
            '    integer :: b = 7'//new_line('a')// &
            '  end type t'//new_line('a')// &
            '  type(t) :: x'//new_line('a')// &
            '  x = t(3)'//new_line('a')// &
            '  if (x%a /= 3) error stop'//new_line('a')// &
            '  stop x%b'//new_line('a')// &
            'end program main'

        test_omitted_default_component = expect_exit_status( &
            source, 7, '/tmp/ffc_session_ctor_default')
    end function test_omitted_default_component

    logical function test_whole_derived_copy()
        ! y = x copies every component of a derived scalar.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: t'//new_line('a')// &
            '    integer :: a'//new_line('a')// &
            '    integer :: b'//new_line('a')// &
            '  end type t'//new_line('a')// &
            '  type(t) :: x, y'//new_line('a')// &
            '  x%a = 5'//new_line('a')// &
            '  x%b = 6'//new_line('a')// &
            '  y = x'//new_line('a')// &
            '  stop y%a + y%b'//new_line('a')// &
            'end program main'

        test_whole_derived_copy = expect_exit_status( &
            source, 11, '/tmp/ffc_session_derived_copy')
    end function test_whole_derived_copy

    logical function test_inherited_component_constructor()
        ! derived2(1, 2, 3, 4): inherited components construct positionally in
        ! parent-then-child declaration order.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: base'//new_line('a')// &
            '    integer :: a'//new_line('a')// &
            '  end type base'//new_line('a')// &
            '  type, extends(base) :: derived'//new_line('a')// &
            '    integer :: b'//new_line('a')// &
            '  end type derived'//new_line('a')// &
            '  type, extends(derived) :: derived2'//new_line('a')// &
            '    integer :: c'//new_line('a')// &
            '    integer :: d'//new_line('a')// &
            '  end type derived2'//new_line('a')// &
            '  type(derived2) :: x'//new_line('a')// &
            '  x = derived2(1, 2, 3, 4)'//new_line('a')// &
            '  if (x%a /= 1) error stop'//new_line('a')// &
            '  if (x%b /= 2) error stop'//new_line('a')// &
            '  if (x%c /= 3) error stop'//new_line('a')// &
            '  stop x%d'//new_line('a')// &
            'end program main'

        test_inherited_component_constructor = expect_exit_status( &
            source, 4, '/tmp/ffc_session_ctor_inherited')
    end function test_inherited_component_constructor

    logical function test_nested_constructor_unsupported()
        ! A nested derived argument is not part of the scalar constructor path;
        ! it must report the unsupported diagnostic, never a wrong value.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: inner'//new_line('a')// &
            '    integer :: a'//new_line('a')// &
            '  end type inner'//new_line('a')// &
            '  type :: outer'//new_line('a')// &
            '    type(inner) :: i'//new_line('a')// &
            '    integer :: b'//new_line('a')// &
            '  end type outer'//new_line('a')// &
            '  type(outer) :: o'//new_line('a')// &
            '  o = outer(inner(1), 2)'//new_line('a')// &
            '  stop o%b'//new_line('a')// &
            'end program main'

        test_nested_constructor_unsupported = expect_error_contains( &
            source, 'unsupported derived type constructor', &
            '/tmp/ffc_session_ctor_nested')
    end function test_nested_constructor_unsupported

    logical function test_interface_constructor_unsupported()
        ! When a generic interface overloads the type name, t(...) must not be
        ! treated as a default structure constructor.
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '  type :: twoint'//new_line('a')// &
            '    integer :: m1'//new_line('a')// &
            '    integer :: m2'//new_line('a')// &
            '  end type twoint'//new_line('a')// &
            '  interface twoint'//new_line('a')// &
            '    procedure :: reverse_constructor'//new_line('a')// &
            '  end interface twoint'//new_line('a')// &
            'contains'//new_line('a')// &
            '  function reverse_constructor(i, j) result(n)'//new_line('a')// &
            '    integer, intent(in) :: i, j'//new_line('a')// &
            '    type(twoint) :: n'//new_line('a')// &
            '    n%m1 = j'//new_line('a')// &
            '    n%m2 = i'//new_line('a')// &
            '  end function reverse_constructor'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  type(twoint) :: x'//new_line('a')// &
            '  x = twoint(1, 2)'//new_line('a')// &
            '  stop x%m1'//new_line('a')// &
            'end program main'

        test_interface_constructor_unsupported = expect_error_contains( &
            source, 'unsupported derived type constructor', &
            '/tmp/ffc_session_ctor_interface')
    end function test_interface_constructor_unsupported

end program test_session_derived_constructor_compiler
