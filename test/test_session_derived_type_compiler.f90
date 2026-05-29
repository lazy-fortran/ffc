program test_session_derived_type_compiler
    use ffc_test_support, only: expect_exit_status, expect_output, &
                                expect_error_contains
    implicit none

    logical :: all_passed

    print *, '=== direct session derived type compiler test ==='

    all_passed = .true.
    if (.not. test_component_slots()) all_passed = .false.
    if (.not. test_two_variables_do_not_alias()) all_passed = .false.
    if (.not. test_component_stop_code()) all_passed = .false.
    if (.not. test_real_component_diagnostic()) all_passed = .false.
    if (.not. test_nested_component_diagnostic()) all_passed = .false.
    if (.not. test_constructor_diagnostic()) all_passed = .false.
    if (.not. test_inheritance_diagnostic()) all_passed = .false.
    if (.not. test_type_bound_binding_compiles()) all_passed = .false.
    if (.not. test_generic_binding_diagnostic()) all_passed = .false.
    if (.not. test_component_default_initialiser()) all_passed = .false.
    if (.not. test_component_default_overridden()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: derived types lower through direct LIRIC'

contains

    logical function test_component_slots()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  type :: cell_t'//new_line('a')// &
                                       '    integer :: lo'//new_line('a')// &
                                       '    integer :: mid'//new_line('a')// &
                                       '    integer :: hi'//new_line('a')// &
                                       '  end type cell_t'//new_line('a')// &
                                       '  type(cell_t) :: node'//new_line('a')// &
                                       '  node%lo = 4'//new_line('a')// &
                                       '  node%mid = 6'//new_line('a')// &
                                     '  node%hi = node%lo + node%mid'//new_line('a')// &
                                       '  print *, node%lo + node%mid + node%hi'// &
                                       new_line('a')// &
                                       'end program main'

        test_component_slots = expect_output(source, '          20'//new_line('a'), &
                                             '/tmp/ffc_session_derived_slots_test')
    end function test_component_slots

    logical function test_two_variables_do_not_alias()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  type :: pair_t'//new_line('a')// &
                                       '    integer :: left'//new_line('a')// &
                                       '    integer :: right'//new_line('a')// &
                                       '  end type pair_t'//new_line('a')// &
                                       '  type(pair_t) :: first'//new_line('a')// &
                                       '  type(pair_t) :: second'//new_line('a')// &
                                       '  first%left = 3'//new_line('a')// &
                                       '  first%right = 8'//new_line('a')// &
                                       '  second%left = 11'//new_line('a')// &
                                       '  second%right = 13'//new_line('a')// &
                                       '  print *, first%left + second%right'// &
                                       new_line('a')// &
                                       'end program main'

        test_two_variables_do_not_alias = expect_output( &
                                          source, '          16'//new_line('a'), &
                                          '/tmp/ffc_session_derived_alias_test')
    end function test_two_variables_do_not_alias

    logical function test_component_stop_code()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  type :: status_t'//new_line('a')// &
                                       '    integer :: code'//new_line('a')// &
                                       '  end type status_t'//new_line('a')// &
                                       '  type(status_t) :: result'//new_line('a')// &
                                       '  result%code = 7'//new_line('a')// &
                                       '  stop result%code'//new_line('a')// &
                                       'end program main'

        test_component_stop_code = expect_exit_status( &
                                   source, 7, &
                                   '/tmp/ffc_session_derived_stop_test')
    end function test_component_stop_code

    logical function test_real_component_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  type :: sample_t'//new_line('a')// &
                                       '    real :: value'//new_line('a')// &
                                       '  end type sample_t'//new_line('a')// &
                                       'end program main'

        test_real_component_diagnostic = expect_error_contains( &
                                         source, 'unsupported derived type component', &
                                         '/tmp/ffc_session_derived_real_test')
    end function test_real_component_diagnostic

    logical function test_nested_component_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  type :: inner_t'//new_line('a')// &
                                       '    integer :: value'//new_line('a')// &
                                       '  end type inner_t'//new_line('a')// &
                                       '  type :: outer_t'//new_line('a')// &
                                       '    type(inner_t) :: inner'//new_line('a')// &
                                       '  end type outer_t'//new_line('a')// &
                                       'end program main'

        test_nested_component_diagnostic = expect_error_contains( &
                                           source, 'unsupported nested derived type', &
                                           '/tmp/ffc_session_derived_nested_test')
    end function test_nested_component_diagnostic

    logical function test_constructor_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  type :: item_t'//new_line('a')// &
                                       '    integer :: count'//new_line('a')// &
                                       '  end type item_t'//new_line('a')// &
                                       '  type(item_t) :: item'//new_line('a')// &
                                       '  item = item_t(4)'//new_line('a')// &
                                       'end program main'

        test_constructor_diagnostic = expect_error_contains( &
                                      source, 'unsupported derived type constructor', &
                                      '/tmp/ffc_session_derived_constructor_test')
    end function test_constructor_diagnostic

    logical function test_inheritance_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  type :: base_t'//new_line('a')// &
                                       '    integer :: code'//new_line('a')// &
                                       '  end type base_t'//new_line('a')// &
                                       '  type, extends(base_t) :: child_t'// &
                                       new_line('a')// &
                                       '    integer :: extra'//new_line('a')// &
                                       '  end type child_t'//new_line('a')// &
                                       'end program main'

        test_inheritance_diagnostic = expect_error_contains( &
                                      source, 'unsupported derived type inheritance', &
                                      '/tmp/ffc_session_derived_extends_test')
    end function test_inheritance_diagnostic

    logical function test_type_bound_binding_compiles()
        ! A plain procedure binding is recorded (not yet lowered as a call,
        ! #146), so a type carrying one compiles and its data is usable.
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  type :: counter_t'//new_line('a')// &
                                       '    integer :: n'//new_line('a')// &
                                       '  contains'//new_line('a')// &
                                       '    procedure :: bump => bump_impl'// &
                                       new_line('a')// &
                                       '  end type counter_t'//new_line('a')// &
                                       '  type(counter_t) :: c'//new_line('a')// &
                                       '  c%n = 5'//new_line('a')// &
                                       '  stop c%n'//new_line('a')// &
                                       'contains'//new_line('a')// &
                                       '  subroutine bump_impl(x)'//new_line('a')// &
                                       '    integer :: x'//new_line('a')// &
                                       '    x = x + 1'//new_line('a')// &
                                       '  end subroutine bump_impl'//new_line('a')// &
                                       'end program main'

        test_type_bound_binding_compiles = expect_exit_status( &
                                     source, 5, &
                                     '/tmp/ffc_session_derived_bound_test')
    end function test_type_bound_binding_compiles

    logical function test_component_default_initialiser()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  type :: t'//new_line('a')// &
                                       '    integer :: x = 5'//new_line('a')// &
                                       '  end type t'//new_line('a')// &
                                       '  type(t) :: a'//new_line('a')// &
                                       '  stop a%x'//new_line('a')// &
                                       'end program main'

        test_component_default_initialiser = expect_exit_status( &
            source, 5, '/tmp/ffc_session_comp_default_test')
    end function test_component_default_initialiser

    logical function test_component_default_overridden()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  type :: t'//new_line('a')// &
                                       '    integer :: x = 5'//new_line('a')// &
                                       '    integer :: y = 10'//new_line('a')// &
                                       '  end type t'//new_line('a')// &
                                       '  type(t) :: a'//new_line('a')// &
                                       '  a%x = 1'//new_line('a')// &
                                       '  stop a%x + a%y'//new_line('a')// &
                                       'end program main'

        ! x overridden to 1, y keeps its default 10 -> 11
        test_component_default_overridden = expect_exit_status( &
            source, 11, '/tmp/ffc_session_comp_default_over_test')
    end function test_component_default_overridden

    logical function test_generic_binding_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  type :: t'//new_line('a')// &
                                       '    integer :: n'//new_line('a')// &
                                       '  contains'//new_line('a')// &
                                       '    generic :: g => f1, f2'//new_line('a')// &
                                       '  end type t'//new_line('a')// &
                                       '  type(t) :: x'//new_line('a')// &
                                       '  x%n = 1'//new_line('a')// &
                                       '  stop x%n'//new_line('a')// &
                                       'end program main'

        test_generic_binding_diagnostic = expect_error_contains( &
                                     source, 'type-bound procedure', &
                                     '/tmp/ffc_session_generic_bound_test')
    end function test_generic_binding_diagnostic

end program test_session_derived_type_compiler
