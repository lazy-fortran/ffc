program test_type_extends
    use ffc_test_support, only: expect_exit_status, expect_error_contains
    implicit none

    logical :: all_passed

    print *, '=== direct session type extends compiler test ==='

    all_passed = .true.
    if (.not. test_parent_first_layout()) all_passed = .false.
    if (.not. test_child_with_no_own_components()) all_passed = .false.
    if (.not. test_array_parent_component_on_child()) all_passed = .false.
    if (.not. test_inherited_type_bound_call()) all_passed = .false.
    if (.not. test_missing_parent_diagnostic()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: type extension lowers through direct LIRIC'

contains

    logical function test_parent_first_layout()
        ! The child lays the parent component first, then its own; writing both
        ! and reading both back proves the two slots do not alias (#248 B6a).
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: base_t'//new_line('a')// &
            '    integer :: a'//new_line('a')// &
            '    integer :: b'//new_line('a')// &
            '  end type base_t'//new_line('a')// &
            '  type, extends(base_t) :: child_t'//new_line('a')// &
            '    integer :: c'//new_line('a')// &
            '  end type child_t'//new_line('a')// &
            '  type(child_t) :: x'//new_line('a')// &
            '  x%a = 1'//new_line('a')// &
            '  x%b = 2'//new_line('a')// &
            '  x%c = 4'//new_line('a')// &
            '  stop x%a + x%b * 4 + x%c * 16'//new_line('a')// &
            'end program main'

        ! 1 + 8 + 64 = 73; distinct positional weights catch any slot aliasing.
        test_parent_first_layout = expect_exit_status( &
            source, 73, '/tmp/ffc_extends_layout_test')
    end function test_parent_first_layout

    logical function test_child_with_no_own_components()
        ! A child that adds no components still inherits the parent's storage
        ! and the parent component is reachable on the child instance.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: base_t'//new_line('a')// &
            '    integer :: v'//new_line('a')// &
            '  end type base_t'//new_line('a')// &
            '  type, extends(base_t) :: child_t'//new_line('a')// &
            '  end type child_t'//new_line('a')// &
            '  type(child_t) :: x'//new_line('a')// &
            '  x%v = 42'//new_line('a')// &
            '  stop x%v'//new_line('a')// &
            'end program main'

        test_child_with_no_own_components = expect_exit_status( &
            source, 42, '/tmp/ffc_extends_empty_child_test')
    end function test_child_with_no_own_components

    logical function test_array_parent_component_on_child()
        ! An inherited array component keeps its element count in the child
        ! layout and stays addressable element-by-element.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: base_t'//new_line('a')// &
            '    integer :: arr(3)'//new_line('a')// &
            '  end type base_t'//new_line('a')// &
            '  type, extends(base_t) :: child_t'//new_line('a')// &
            '    integer :: tail'//new_line('a')// &
            '  end type child_t'//new_line('a')// &
            '  type(child_t) :: x'//new_line('a')// &
            '  x%arr(1) = 5'//new_line('a')// &
            '  x%arr(3) = 9'//new_line('a')// &
            '  x%tail = 1'//new_line('a')// &
            '  stop x%arr(1) + x%arr(3) + x%tail'//new_line('a')// &
            'end program main'

        ! 5 + 9 + 1 = 15
        test_array_parent_component_on_child = expect_exit_status( &
            source, 15, '/tmp/ffc_extends_array_test')
    end function test_array_parent_component_on_child

    logical function test_inherited_type_bound_call()
        ! A parent type-bound procedure is callable on a child instance via the
        ! inherited static binding (default pass routes the child as the object).
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: base_t'//new_line('a')// &
            '    integer :: n'//new_line('a')// &
            '  contains'//new_line('a')// &
            '    procedure :: get => base_get'//new_line('a')// &
            '  end type base_t'//new_line('a')// &
            '  type, extends(base_t) :: child_t'//new_line('a')// &
            '    integer :: extra'//new_line('a')// &
            '  end type child_t'//new_line('a')// &
            '  type(child_t) :: c'//new_line('a')// &
            '  c%n = 8'//new_line('a')// &
            '  stop c%get()'//new_line('a')// &
            'contains'//new_line('a')// &
            '  integer function base_get(self)'//new_line('a')// &
            '    type(base_t), intent(in) :: self'//new_line('a')// &
            '    base_get = self%n'//new_line('a')// &
            '  end function base_get'//new_line('a')// &
            'end program main'

        test_inherited_type_bound_call = expect_exit_status( &
            source, 8, '/tmp/ffc_extends_tbp_test')
    end function test_inherited_type_bound_call

    logical function test_missing_parent_diagnostic()
        ! Extending a type that was never defined is a clear diagnostic.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type, extends(nope_t) :: child_t'//new_line('a')// &
            '    integer :: x'//new_line('a')// &
            '  end type child_t'//new_line('a')// &
            '  type(child_t) :: c'//new_line('a')// &
            '  c%x = 1'//new_line('a')// &
            '  stop c%x'//new_line('a')// &
            'end program main'

        test_missing_parent_diagnostic = expect_error_contains( &
            source, 'unsupported derived type inheritance', &
            '/tmp/ffc_extends_missing_parent_test')
    end function test_missing_parent_diagnostic

end program test_type_extends
