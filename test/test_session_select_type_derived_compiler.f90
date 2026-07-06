program test_session_select_type_derived_compiler
    use ffc_test_support, only: expect_exit_status, expect_error_contains
    implicit none

    logical :: all_passed

    print *, '=== monomorphic select type (class(t)) compiler test ==='

    all_passed = .true.
    if (.not. test_select_type_self_type_is()) all_passed = .false.
    if (.not. test_select_type_assoc_name()) all_passed = .false.
    if (.not. test_select_type_class_is()) all_passed = .false.
    if (.not. test_select_type_default_over_other_type()) all_passed = .false.
    if (.not. test_select_type_discriminates_declines()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: monomorphic select type dispatches to the declared type'

contains

    logical function test_select_type_self_type_is()
        ! A class(t) dummy whose dynamic type is its declared type t: the
        ! `type is (t)` arm runs and reads the passed object's component.
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '  type :: t'//new_line('a')// &
            '     integer :: i'//new_line('a')// &
            '  end type t'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine show(self, out)'//new_line('a')// &
            '     class(t), intent(in) :: self'//new_line('a')// &
            '     integer, intent(out) :: out'//new_line('a')// &
            '     out = 0'//new_line('a')// &
            '     select type (self)'//new_line('a')// &
            '     type is (t)'//new_line('a')// &
            '        out = self%i'//new_line('a')// &
            '     end select'//new_line('a')// &
            '  end subroutine show'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  type(t) :: x'//new_line('a')// &
            '  integer :: r'//new_line('a')// &
            '  x%i = 42'//new_line('a')// &
            '  call show(x, r)'//new_line('a')// &
            '  stop r'//new_line('a')// &
            'end program main'

        test_select_type_self_type_is = expect_exit_status( &
            source, 42, '/tmp/ffc_session_st_derived_self')
    end function test_select_type_self_type_is

    logical function test_select_type_assoc_name()
        ! select type (a => obj): the associate name aliases the selector's
        ! derived storage so component access reads the same slots.
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '  type :: t'//new_line('a')// &
            '     integer :: i'//new_line('a')// &
            '  end type t'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine show(obj, out)'//new_line('a')// &
            '     class(t), intent(in) :: obj'//new_line('a')// &
            '     integer, intent(out) :: out'//new_line('a')// &
            '     out = 0'//new_line('a')// &
            '     select type (a => obj)'//new_line('a')// &
            '     type is (t)'//new_line('a')// &
            '        out = a%i'//new_line('a')// &
            '     class default'//new_line('a')// &
            '        out = 99'//new_line('a')// &
            '     end select'//new_line('a')// &
            '  end subroutine show'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  type(t) :: x'//new_line('a')// &
            '  integer :: r'//new_line('a')// &
            '  x%i = 7'//new_line('a')// &
            '  call show(x, r)'//new_line('a')// &
            '  stop r'//new_line('a')// &
            'end program main'

        test_select_type_assoc_name = expect_exit_status( &
            source, 7, '/tmp/ffc_session_st_derived_assoc')
    end function test_select_type_assoc_name

    logical function test_select_type_class_is()
        ! A `class is (t)` guard naming the declared type also runs.
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '  type :: t'//new_line('a')// &
            '     integer :: i'//new_line('a')// &
            '  end type t'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine show(self, out)'//new_line('a')// &
            '     class(t), intent(in) :: self'//new_line('a')// &
            '     integer, intent(out) :: out'//new_line('a')// &
            '     out = 0'//new_line('a')// &
            '     select type (self)'//new_line('a')// &
            '     class is (t)'//new_line('a')// &
            '        out = self%i'//new_line('a')// &
            '     end select'//new_line('a')// &
            '  end subroutine show'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  type(t) :: x'//new_line('a')// &
            '  integer :: r'//new_line('a')// &
            '  x%i = 5'//new_line('a')// &
            '  call show(x, r)'//new_line('a')// &
            '  stop r'//new_line('a')// &
            'end program main'

        test_select_type_class_is = expect_exit_status( &
            source, 5, '/tmp/ffc_session_st_derived_classis')
    end function test_select_type_class_is

    logical function test_select_type_default_over_other_type()
        ! A guard naming an INTRINSIC kind never matches a derived selector; the
        ! class default arm runs instead (no subtype discrimination involved).
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '  type :: t'//new_line('a')// &
            '     integer :: i'//new_line('a')// &
            '  end type t'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine show(self, out)'//new_line('a')// &
            '     class(t), intent(in) :: self'//new_line('a')// &
            '     integer, intent(out) :: out'//new_line('a')// &
            '     out = 0'//new_line('a')// &
            '     select type (self)'//new_line('a')// &
            '     type is (integer)'//new_line('a')// &
            '        out = 1'//new_line('a')// &
            '     class default'//new_line('a')// &
            '        out = self%i'//new_line('a')// &
            '     end select'//new_line('a')// &
            '  end subroutine show'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  type(t) :: x'//new_line('a')// &
            '  integer :: r'//new_line('a')// &
            '  x%i = 3'//new_line('a')// &
            '  call show(x, r)'//new_line('a')// &
            '  stop r'//new_line('a')// &
            'end program main'

        test_select_type_default_over_other_type = expect_exit_status( &
            source, 3, '/tmp/ffc_session_st_derived_default')
    end function test_select_type_default_over_other_type

    logical function test_select_type_discriminates_declines()
        ! A construct that discriminates a runtime subtype needs a vtable; the
        ! monomorphic path must decline gracefully rather than statically
        ! mispick, so ffc emits the unsupported diagnostic.
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '  type :: base'//new_line('a')// &
            '     integer :: i'//new_line('a')// &
            '  end type base'//new_line('a')// &
            '  type, extends(base) :: child'//new_line('a')// &
            '     integer :: j'//new_line('a')// &
            '  end type child'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine show(self)'//new_line('a')// &
            '     class(base), intent(in) :: self'//new_line('a')// &
            '     select type (self)'//new_line('a')// &
            '     type is (child)'//new_line('a')// &
            '        print *, self%j'//new_line('a')// &
            '     type is (base)'//new_line('a')// &
            '        print *, self%i'//new_line('a')// &
            '     end select'//new_line('a')// &
            '  end subroutine show'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  type(base) :: x'//new_line('a')// &
            '  x%i = 1'//new_line('a')// &
            '  call show(x)'//new_line('a')// &
            'end program main'

        test_select_type_discriminates_declines = expect_error_contains( &
            source, 'monomorphic select type', &
            '/tmp/ffc_session_st_derived_decline')
    end function test_select_type_discriminates_declines

end program test_session_select_type_derived_compiler
