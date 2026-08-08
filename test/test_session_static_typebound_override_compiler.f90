program test_session_static_typebound_override_compiler
    use ffc_test_support, only: expect_exit_status, expect_error_contains
    implicit none

    logical :: all_passed

    print *, '=== static type-bound override tests ==='

    all_passed = .true.
    if (.not. test_default_pass_override()) all_passed = .false.
    if (.not. test_nopass_override()) all_passed = .false.
    if (.not. test_class_pointer_reassociation_is_rejected()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: static type-bound overrides'

contains

    logical function test_default_pass_override()
        ! The declared child type fixes dispatch at compile time.  The result
        ! uses the child-only component, so selecting the parent binding would
        ! produce 3 instead of the analytical value 13.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: parent_t'//new_line('a')// &
            '    integer :: base'//new_line('a')// &
            '  contains'//new_line('a')// &
            '    procedure :: Value => parent_value'//new_line('a')// &
            '  end type parent_t'//new_line('a')// &
            '  type, extends(parent_t) :: child_t'//new_line('a')// &
            '    integer :: child'//new_line('a')// &
            '  contains'//new_line('a')// &
            '    procedure :: value => child_value'//new_line('a')// &
            '  end type child_t'//new_line('a')// &
            '  type(child_t) :: model'//new_line('a')// &
            '  model%base = 3'//new_line('a')// &
            '  model%child = 10'//new_line('a')// &
            '  stop model%value()'//new_line('a')// &
            'contains'//new_line('a')// &
            '  integer function parent_value(self)'//new_line('a')// &
            '    type(parent_t), intent(in) :: self'//new_line('a')// &
            '    parent_value = self%base'//new_line('a')// &
            '  end function parent_value'//new_line('a')// &
            '  integer function child_value(self)'//new_line('a')// &
            '    type(child_t), intent(in) :: self'//new_line('a')// &
            '    child_value = self%base + self%child'//new_line('a')// &
            '  end function child_value'//new_line('a')// &
            'end program main'

        test_default_pass_override = expect_exit_status(source, 13, &
            '/tmp/ffc_static_tbp_default_pass')
    end function test_default_pass_override

    logical function test_nopass_override()
        ! NOPASS keeps the same static child-slot selection while omitting the
        ! receiver from both the binding and the implementation call.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: parent_t'//new_line('a')// &
            '  contains'//new_line('a')// &
            '    procedure, nopass :: value => parent_value'//new_line('a')// &
            '  end type parent_t'//new_line('a')// &
            '  type, extends(parent_t) :: child_t'//new_line('a')// &
            '  contains'//new_line('a')// &
            '    procedure, nopass :: value => child_value'//new_line('a')// &
            '  end type child_t'//new_line('a')// &
            '  type(child_t) :: model'//new_line('a')// &
            '  stop model%value(7)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  integer function parent_value(x)'//new_line('a')// &
            '    integer, intent(in) :: x'//new_line('a')// &
            '    parent_value = x'//new_line('a')// &
            '  end function parent_value'//new_line('a')// &
            '  integer function child_value(x)'//new_line('a')// &
            '    integer, intent(in) :: x'//new_line('a')// &
            '    child_value = x + 10'//new_line('a')// &
            '  end function child_value'//new_line('a')// &
            'end program main'

        test_nopass_override = expect_exit_status(source, 17, &
            '/tmp/ffc_static_tbp_nopass')
    end function test_nopass_override

    logical function test_class_pointer_reassociation_is_rejected()
        ! Scalar class-pointer reassociation remains outside the allocation /
        ! dispatch slice and must fail before any storage is adopted.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: parent_t'//new_line('a')// &
            '  contains'//new_line('a')// &
            '    procedure :: value => parent_value'//new_line('a')// &
            '  end type parent_t'//new_line('a')// &
            '  type, extends(parent_t) :: child_t'//new_line('a')// &
            '  contains'//new_line('a')// &
            '    procedure :: value => child_value'//new_line('a')// &
            '  end type child_t'//new_line('a')// &
            '  type(parent_t), target :: base'//new_line('a')// &
            '  class(parent_t), pointer :: model'//new_line('a')// &
            '  model => base'//new_line('a')// &
            '  print *, model%value()'//new_line('a')// &
            'contains'//new_line('a')// &
            '  integer function parent_value(self)'//new_line('a')// &
            '    class(parent_t), intent(in) :: self'//new_line('a')// &
            '    parent_value = 1'//new_line('a')// &
            '  end function parent_value'//new_line('a')// &
            '  integer function child_value(self)'//new_line('a')// &
            '    class(child_t), intent(in) :: self'//new_line('a')// &
            '    child_value = 2'//new_line('a')// &
            '  end function child_value'//new_line('a')// &
            'end program main'

        test_class_pointer_reassociation_is_rejected = expect_error_contains( &
            source, 'class pointer reassociation', &
            '/tmp/ffc_static_tbp_dynamic_reject')
    end function test_class_pointer_reassociation_is_rejected

end program test_session_static_typebound_override_compiler
