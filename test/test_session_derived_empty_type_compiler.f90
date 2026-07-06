program test_session_derived_empty_type_compiler
    use ffc_test_support, only: expect_exit_status, expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session empty derived type compiler test ==='

    all_passed = .true.
    if (.not. test_bare_empty_type()) all_passed = .false.
    if (.not. test_behavioral_only_type_bound_call()) all_passed = .false.
    if (.not. test_empty_parent_extended_with_data()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: empty derived types register and lower'

contains

    logical function test_bare_empty_type()
        ! A type with no data components (bare `type :: t; end type`) registers
        ! with a hidden placeholder slot and can be declared and used.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: mt'//new_line('a')// &
            '  end type mt'//new_line('a')// &
            '  type(mt) :: x'//new_line('a')// &
            '  integer :: v'//new_line('a')// &
            '  v = 3'//new_line('a')// &
            '  stop v'//new_line('a')// &
            'end program main'

        test_bare_empty_type = expect_exit_status( &
            source, 3, '/tmp/ffc_session_empty_bare_test')
    end function test_bare_empty_type

    logical function test_behavioral_only_type_bound_call()
        ! A behavioral-only type (only a `contains` type-bound-procedure block,
        ! no data) registers so class(t) self dummies and the bound call resolve.
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  type :: greeter_t'//new_line('a')// &
            '  contains'//new_line('a')// &
            '    procedure :: value => greeter_value'//new_line('a')// &
            '  end type greeter_t'//new_line('a')// &
            'contains'//new_line('a')// &
            '  integer function greeter_value(self)'//new_line('a')// &
            '    class(greeter_t), intent(in) :: self'//new_line('a')// &
            '    greeter_value = 9'//new_line('a')// &
            '  end function greeter_value'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  type(greeter_t) :: g'//new_line('a')// &
            '  stop g%value()'//new_line('a')// &
            'end program main'

        test_behavioral_only_type_bound_call = expect_exit_status( &
            source, 9, '/tmp/ffc_session_empty_behavioral_test')
    end function test_behavioral_only_type_bound_call

    logical function test_empty_parent_extended_with_data()
        ! An empty base type extended by a child carrying data: the child
        ! inherits the placeholder slot and adds its own component.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: base_t'//new_line('a')// &
            '  end type base_t'//new_line('a')// &
            '  type, extends(base_t) :: child_t'//new_line('a')// &
            '    integer :: code'//new_line('a')// &
            '  end type child_t'//new_line('a')// &
            '  type(child_t) :: c'//new_line('a')// &
            '  c%code = 5'//new_line('a')// &
            '  stop c%code'//new_line('a')// &
            'end program main'

        test_empty_parent_extended_with_data = expect_exit_status( &
            source, 5, '/tmp/ffc_session_empty_extends_test')
    end function test_empty_parent_extended_with_data

end program test_session_derived_empty_type_compiler
