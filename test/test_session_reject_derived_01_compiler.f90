program test_session_reject_derived_01_compiler
    use ffc_test_support, only: expect_error_contains, expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== derived-type definition constraint rejection test ==='

    all_passed = .true.
    if (.not. test_empty_bind_c_type_rejected()) all_passed = .false.
    if (.not. test_bind_c_type_with_component_accepted()) all_passed = .false.
    if (.not. test_assumed_size_default_init_rejected()) all_passed = .false.
    if (.not. test_explicit_shape_default_init_accepted()) all_passed = .false.
    if (.not. test_mangled_type_guard_rejected()) all_passed = .false.
    if (.not. test_spaced_type_guard_accepted()) all_passed = .false.
    if (.not. test_private_component_rejected()) all_passed = .false.
    if (.not. test_public_component_accepted()) all_passed = .false.
    if (.not. test_data_allocatable_component_rejected()) all_passed = .false.
    if (.not. test_data_plain_component_accepted()) all_passed = .false.
    if (.not. test_class_function_result_rejected()) all_passed = .false.
    if (.not. test_allocatable_class_result_accepted()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: derived-type definition constraints enforced'

contains

    logical function test_empty_bind_c_type_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type, bind(C) :: t'//new_line('a')// &
            '  end type t'//new_line('a')// &
            'end program main'

        test_empty_bind_c_type_rejected = expect_error_contains( &
            source, 'has no components', '/tmp/ffc_reject_derived_01_bindc')
    end function test_empty_bind_c_type_rejected

    logical function test_bind_c_type_with_component_accepted()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  use, intrinsic :: iso_c_binding, only: c_int'//new_line('a')// &
            '  type, bind(C) :: t'//new_line('a')// &
            '    integer(c_int) :: i'//new_line('a')// &
            '  end type t'//new_line('a')// &
            '  type(t) :: x'//new_line('a')// &
            '  x%i = 5'//new_line('a')// &
            '  stop x%i'//new_line('a')// &
            'end program main'

        test_bind_c_type_with_component_accepted = expect_exit_status( &
            source, 5, '/tmp/ffc_reject_derived_01_bindc_ok')
    end function test_bind_c_type_with_component_accepted

    logical function test_assumed_size_default_init_rejected()
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '  type init'//new_line('a')// &
            '    integer :: i = 0'//new_line('a')// &
            '  end type init'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine try(a)'//new_line('a')// &
            '    type(init), dimension(*), intent(out) :: a'//new_line('a')// &
            '  end subroutine try'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            'end program main'

        test_assumed_size_default_init_rejected = expect_error_contains( &
            source, 'cannot have a default initializer', &
            '/tmp/ffc_reject_derived_01_assumed')
    end function test_assumed_size_default_init_rejected

    logical function test_explicit_shape_default_init_accepted()
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '  type init'//new_line('a')// &
            '    integer :: i = 0'//new_line('a')// &
            '  end type init'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine try(a)'//new_line('a')// &
            '    type(init), dimension(2), intent(out) :: a'//new_line('a')// &
            '    a(1)%i = 7'//new_line('a')// &
            '  end subroutine try'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  type(init) :: b(2)'//new_line('a')// &
            '  call try(b)'//new_line('a')// &
            '  stop b(1)%i'//new_line('a')// &
            'end program main'

        test_explicit_shape_default_init_accepted = expect_exit_status( &
            source, 7, '/tmp/ffc_reject_derived_01_assumed_ok')
    end function test_explicit_shape_default_init_accepted

    logical function test_mangled_type_guard_rejected()
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '  type t'//new_line('a')// &
            '    integer :: i'//new_line('a')// &
            '  end type t'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine s(x)'//new_line('a')// &
            '    class(t), pointer :: x'//new_line('a')// &
            '    select type (x)'//new_line('a')// &
            '    typeis (t)'//new_line('a')// &
            '    end select'//new_line('a')// &
            '  end subroutine s'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            'end program main'

        test_mangled_type_guard_rejected = expect_error_contains( &
            source, 'mangled derived type definition', &
            '/tmp/ffc_reject_derived_01_typeis')
    end function test_mangled_type_guard_rejected

    logical function test_spaced_type_guard_accepted()
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '  type t'//new_line('a')// &
            '    integer :: i'//new_line('a')// &
            '  end type t'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine s(x)'//new_line('a')// &
            '    class(t), pointer :: x'//new_line('a')// &
            '    select type (x)'//new_line('a')// &
            '    type is (t)'//new_line('a')// &
            '      x%i = 1'//new_line('a')// &
            '    end select'//new_line('a')// &
            '  end subroutine s'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  stop 4'//new_line('a')// &
            'end program main'

        test_spaced_type_guard_accepted = expect_exit_status( &
            source, 4, '/tmp/ffc_reject_derived_01_typeis_ok')
    end function test_spaced_type_guard_accepted

    logical function test_private_component_rejected()
        character(len=*), parameter :: source = &
            'module foomod'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  type :: bartype'//new_line('a')// &
            '    integer :: dummy'//new_line('a')// &
            '    integer, private :: dummy2'//new_line('a')// &
            '  end type bartype'//new_line('a')// &
            'end module foomod'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use foomod'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  type(bartype) :: foo2'//new_line('a')// &
            '  foo2%dummy2 = 5'//new_line('a')// &
            'end program main'

        test_private_component_rejected = expect_error_contains( &
            source, 'is a PRIVATE component', &
            '/tmp/ffc_reject_derived_01_private')
    end function test_private_component_rejected

    logical function test_public_component_accepted()
        character(len=*), parameter :: source = &
            'module foomod'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  type :: bartype'//new_line('a')// &
            '    integer :: dummy'//new_line('a')// &
            '    integer :: dummy2'//new_line('a')// &
            '  end type bartype'//new_line('a')// &
            'end module foomod'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use foomod'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  type(bartype) :: foo2'//new_line('a')// &
            '  foo2%dummy2 = 5'//new_line('a')// &
            '  stop foo2%dummy2'//new_line('a')// &
            'end program main'

        test_public_component_accepted = expect_exit_status( &
            source, 5, '/tmp/ffc_reject_derived_01_private_ok')
    end function test_public_component_accepted

    logical function test_data_allocatable_component_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type t'//new_line('a')// &
            '    integer, allocatable :: a(:)'//new_line('a')// &
            '  end type t'//new_line('a')// &
            '  type(t) :: z'//new_line('a')// &
            '  data z%a(1) / 789 /'//new_line('a')// &
            'end program main'

        test_data_allocatable_component_rejected = expect_error_contains( &
            source, 'cannot appear in a DATA statement', &
            '/tmp/ffc_reject_derived_01_data')
    end function test_data_allocatable_component_rejected

    logical function test_data_plain_component_accepted()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type t'//new_line('a')// &
            '    integer, allocatable :: a(:)'//new_line('a')// &
            '  end type t'//new_line('a')// &
            '  type(t) :: z'//new_line('a')// &
            '  allocate (z%a(1))'//new_line('a')// &
            '  z%a(1) = 6'//new_line('a')// &
            '  stop z%a(1)'//new_line('a')// &
            'end program main'

        test_data_plain_component_accepted = expect_exit_status( &
            source, 6, '/tmp/ffc_reject_derived_01_data_ok')
    end function test_data_plain_component_accepted

    logical function test_class_function_result_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type t'//new_line('a')// &
            '    integer :: i'//new_line('a')// &
            '  end type t'//new_line('a')// &
            'contains'//new_line('a')// &
            '  function f() result(z)'//new_line('a')// &
            '    class(t) :: z'//new_line('a')// &
            '  end function f'//new_line('a')// &
            'end program main'

        test_class_function_result_rejected = expect_error_contains( &
            source, 'must be dummy, allocatable or pointer', &
            '/tmp/ffc_reject_derived_01_class')
    end function test_class_function_result_rejected

    logical function test_allocatable_class_result_accepted()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type t'//new_line('a')// &
            '    integer :: i'//new_line('a')// &
            '  end type t'//new_line('a')// &
            '  type(t) :: r'//new_line('a')// &
            '  r = f()'//new_line('a')// &
            '  stop r%i'//new_line('a')// &
            'contains'//new_line('a')// &
            '  function f() result(z)'//new_line('a')// &
            '    type(t) :: z'//new_line('a')// &
            '    z%i = 9'//new_line('a')// &
            '  end function f'//new_line('a')// &
            'end program main'

        test_allocatable_class_result_accepted = expect_exit_status( &
            source, 9, '/tmp/ffc_reject_derived_01_class_ok')
    end function test_allocatable_class_result_accepted

end program test_session_reject_derived_01_compiler
