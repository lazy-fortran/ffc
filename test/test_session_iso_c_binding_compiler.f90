program test_session_iso_c_binding_compiler
    use ffc_test_support, only: expect_exit_status, expect_error_contains
    implicit none

    logical :: all_passed

    print *, '=== direct session iso_c_binding compiler test ==='

    all_passed = .true.
    if (.not. test_use_iso_c_binding_alone_compiles()) all_passed = .false.
    if (.not. test_use_iso_c_binding_intrinsic_form()) all_passed = .false.
    if (.not. test_c_int_kind_used_in_declaration()) all_passed = .false.
    if (.not. test_c_ptr_declaration_and_null_init()) all_passed = .false.
    if (.not. test_c_ptr_assignment_from_other_c_ptr()) all_passed = .false.
    if (.not. test_bind_c_derived_type_with_int_and_ptr()) all_passed = .false.
    if (.not. test_bind_c_derived_type_rejects_deferred_character()) &
        all_passed = .false.
    if (.not. test_interface_bind_c_integer_function_compiles()) &
        all_passed = .false.
    if (.not. test_non_bind_c_interface_rejected()) all_passed = .false.
    if (.not. test_call_bind_c_integer_function()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: iso_c_binding kind constants lower through direct LIRIC'

contains

    logical function test_use_iso_c_binding_alone_compiles()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  use iso_c_binding'//new_line('a')// &
            '  integer(c_int32_t) :: x'//new_line('a')// &
            '  x = 7'//new_line('a')// &
            '  stop x'//new_line('a')// &
            'end program main'

        test_use_iso_c_binding_alone_compiles = expect_exit_status( &
            source, 7, '/tmp/ffc_session_iso_c_c_int32_test')
    end function test_use_iso_c_binding_alone_compiles

    logical function test_use_iso_c_binding_intrinsic_form()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  use, intrinsic :: iso_c_binding'//new_line('a')// &
            '  integer(c_int32_t) :: x'//new_line('a')// &
            '  x = 7'//new_line('a')// &
            '  stop x'//new_line('a')// &
            'end program main'

        test_use_iso_c_binding_intrinsic_form = expect_exit_status( &
            source, 7, '/tmp/ffc_session_iso_c_intrinsic_test')
    end function test_use_iso_c_binding_intrinsic_form

    logical function test_c_int_kind_used_in_declaration()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  use iso_c_binding'//new_line('a')// &
            '  integer(c_int) :: x'//new_line('a')// &
            '  x = 5'//new_line('a')// &
            '  stop x'//new_line('a')// &
            'end program main'

        test_c_int_kind_used_in_declaration = expect_exit_status( &
            source, 5, '/tmp/ffc_session_iso_c_c_int_test')
    end function test_c_int_kind_used_in_declaration

    logical function test_c_ptr_declaration_and_null_init()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  use iso_c_binding'//new_line('a')// &
            '  type(c_ptr) :: p'//new_line('a')// &
            '  p = c_null_ptr'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_c_ptr_declaration_and_null_init = expect_exit_status( &
            source, 0, '/tmp/ffc_session_c_ptr_null_test')
    end function test_c_ptr_declaration_and_null_init

    logical function test_c_ptr_assignment_from_other_c_ptr()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  use iso_c_binding'//new_line('a')// &
            '  type(c_ptr) :: a, b'//new_line('a')// &
            '  a = c_null_ptr'//new_line('a')// &
            '  b = a'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_c_ptr_assignment_from_other_c_ptr = expect_exit_status( &
            source, 0, '/tmp/ffc_session_c_ptr_copy_test')
    end function test_c_ptr_assignment_from_other_c_ptr

    logical function test_bind_c_derived_type_with_int_and_ptr()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  use iso_c_binding'//new_line('a')// &
            '  type, bind(c) :: pair_t'//new_line('a')// &
            '    integer(c_int32_t) :: a'//new_line('a')// &
            '    type(c_ptr) :: b'//new_line('a')// &
            '  end type pair_t'//new_line('a')// &
            '  type(pair_t) :: p'//new_line('a')// &
            '  p%a = 42'//new_line('a')// &
            '  p%b = c_null_ptr'//new_line('a')// &
            '  stop p%a'//new_line('a')// &
            'end program main'

        test_bind_c_derived_type_with_int_and_ptr = expect_exit_status( &
            source, 42, '/tmp/ffc_session_bind_c_pair_test')
    end function test_bind_c_derived_type_with_int_and_ptr

    logical function test_bind_c_derived_type_rejects_deferred_character()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  use iso_c_binding'//new_line('a')// &
            '  type, bind(c) :: bad_t'//new_line('a')// &
            '    character(len=:), allocatable :: name'//new_line('a')// &
            '  end type bad_t'//new_line('a')// &
            '  type(bad_t) :: x'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_bind_c_derived_type_rejects_deferred_character = &
            expect_error_contains(source, 'derived type component', &
            '/tmp/ffc_session_bind_c_bad_test')
    end function test_bind_c_derived_type_rejects_deferred_character

    logical function test_interface_bind_c_integer_function_compiles()
        ! A bind(c) integer interface is registered; declaring it without
        ! calling it compiles and runs.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  use iso_c_binding'//new_line('a')// &
            '  interface'//new_line('a')// &
            '    integer(c_int) function abs_c(x) bind(c, name="abs")'// &
            new_line('a')// &
            '      import :: c_int'//new_line('a')// &
            '      integer(c_int), value :: x'//new_line('a')// &
            '    end function abs_c'//new_line('a')// &
            '  end interface'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_interface_bind_c_integer_function_compiles = expect_exit_status( &
            source, 0, '/tmp/ffc_session_iface_bindc_test')
    end function test_interface_bind_c_integer_function_compiles

    logical function test_non_bind_c_interface_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  interface'//new_line('a')// &
            '    integer function plain(x)'//new_line('a')// &
            '      integer :: x'//new_line('a')// &
            '    end function plain'//new_line('a')// &
            '  end interface'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_non_bind_c_interface_rejected = expect_error_contains( &
            source, 'interface declaration', '/tmp/ffc_session_iface_plain_test')
    end function test_non_bind_c_interface_rejected

    logical function test_call_bind_c_integer_function()
        ! Calling a registered bind(c) integer function targets the C symbol
        ! (libc abs) with by-value arguments.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  use iso_c_binding'//new_line('a')// &
            '  integer :: r'//new_line('a')// &
            '  interface'//new_line('a')// &
            '    integer(c_int) function abs_c(x) bind(c, name="abs")'// &
            new_line('a')// &
            '      import :: c_int'//new_line('a')// &
            '      integer(c_int), value :: x'//new_line('a')// &
            '    end function abs_c'//new_line('a')// &
            '  end interface'//new_line('a')// &
            '  r = abs_c(-3) + abs_c(10)'//new_line('a')// &
            '  stop r'//new_line('a')// &
            'end program main'

        ! abs(-3) + abs(10) = 13
        test_call_bind_c_integer_function = expect_exit_status( &
            source, 13, '/tmp/ffc_session_call_bindc_test')
    end function test_call_bind_c_integer_function

end program test_session_iso_c_binding_compiler
