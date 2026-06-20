program test_session_type_bound_compiler
    use ffc_test_support, only: expect_exit_status, expect_output
    implicit none

    print *, '=== direct session type-bound procedure compiler test ==='

    if (.not. test_bound_sub_and_fn()) stop 1
    if (.not. test_bound_call_output()) stop 1
    if (.not. test_bound_pass_rename()) stop 1

    print *, 'PASS: type-bound procedures lower through direct LIRIC'

contains

    ! A class(t) passed-object dummy on a non-polymorphic declared-type object:
    ! the bound subroutine writes a component, the bound function reads it back.
    ! Standard Fortran requires the passed-object dummy to be class(t); for a
    ! concrete-type object the dispatch is static (by reference).
    logical function test_bound_sub_and_fn()
        character(len=*), parameter :: source = &
           'module m'//new_line('a')// &
           '  implicit none'//new_line('a')// &
           '  type :: counter'//new_line('a')// &
           '    integer :: n'//new_line('a')// &
           '  contains'//new_line('a')// &
           '    procedure :: set => counter_set'//new_line('a')// &
           '    procedure :: get => counter_get'//new_line('a')// &
           '  end type counter'//new_line('a')// &
           'contains'//new_line('a')// &
           '  subroutine counter_set(this, v)'//new_line('a')// &
           '    class(counter), intent(inout) :: this'//new_line('a')// &
           '    integer, intent(in) :: v'//new_line('a')// &
           '    this%n = v'//new_line('a')// &
           '  end subroutine counter_set'//new_line('a')// &
           '  integer function counter_get(this)'//new_line('a')// &
           '    class(counter), intent(in) :: this'//new_line('a')// &
           '    counter_get = this%n'//new_line('a')// &
           '  end function counter_get'//new_line('a')// &
           'end module m'//new_line('a')// &
           'program main'//new_line('a')// &
           '  use m'//new_line('a')// &
           '  implicit none'//new_line('a')// &
           '  type(counter) :: c'//new_line('a')// &
           '  call c%set(42)'//new_line('a')// &
           '  stop c%get()'//new_line('a')// &
           'end program main'

        test_bound_sub_and_fn = expect_exit_status( &
           source, 42, '/tmp/ffc_type_bound_sub_fn_test')
    end function test_bound_sub_and_fn

    ! Bound function with an explicit argument, result combined and printed.
    logical function test_bound_call_output()
        character(len=*), parameter :: source = &
           'module m'//new_line('a')// &
           '  implicit none'//new_line('a')// &
           '  type :: box'//new_line('a')// &
           '    integer :: base'//new_line('a')// &
           '  contains'//new_line('a')// &
           '    procedure :: plus => box_plus'//new_line('a')// &
           '  end type box'//new_line('a')// &
           'contains'//new_line('a')// &
           '  integer function box_plus(this, k)'//new_line('a')// &
           '    class(box), intent(in) :: this'//new_line('a')// &
           '    integer, intent(in) :: k'//new_line('a')// &
           '    box_plus = this%base + k'//new_line('a')// &
           '  end function box_plus'//new_line('a')// &
           'end module m'//new_line('a')// &
           'program main'//new_line('a')// &
           '  use m'//new_line('a')// &
           '  implicit none'//new_line('a')// &
           '  type(box) :: b'//new_line('a')// &
           '  b%base = 100'//new_line('a')// &
           '  print *, b%plus(23)'//new_line('a')// &
           'end program main'

        test_bound_call_output = expect_output( &
           source, '         123'//new_line('a'), &
           '/tmp/ffc_type_bound_out_test')
    end function test_bound_call_output

    ! A pass(self) renamed passed-object dummy: the bound subroutine receives
    ! the object in the position named by pass, not necessarily the first.
    logical function test_bound_pass_rename()
        character(len=*), parameter :: source = &
           'module m'//new_line('a')// &
           '  implicit none'//new_line('a')// &
           '  type :: acc'//new_line('a')// &
           '    integer :: total'//new_line('a')// &
           '  contains'//new_line('a')// &
           '    procedure, pass(self) :: add => acc_add'//new_line('a')// &
           '  end type acc'//new_line('a')// &
           'contains'//new_line('a')// &
           '  subroutine acc_add(self, v)'//new_line('a')// &
           '    class(acc), intent(inout) :: self'//new_line('a')// &
           '    integer, intent(in) :: v'//new_line('a')// &
           '    self%total = self%total + v'//new_line('a')// &
           '  end subroutine acc_add'//new_line('a')// &
           'end module m'//new_line('a')// &
           'program main'//new_line('a')// &
           '  use m'//new_line('a')// &
           '  implicit none'//new_line('a')// &
           '  type(acc) :: a'//new_line('a')// &
           '  a%total = 5'//new_line('a')// &
           '  call a%add(7)'//new_line('a')// &
           '  stop a%total'//new_line('a')// &
           'end program main'

        test_bound_pass_rename = expect_exit_status( &
           source, 12, '/tmp/ffc_type_bound_pass_test')
    end function test_bound_pass_rename

end program test_session_type_bound_compiler
