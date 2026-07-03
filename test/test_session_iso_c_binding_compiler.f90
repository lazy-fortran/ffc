program test_session_iso_c_binding_compiler
    use ffc_test_support, only: expect_exit_status, expect_error_contains, &
        expect_exe_has_symbol
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
    if (.not. test_non_bind_c_interface_lowers()) all_passed = .false.
    if (.not. test_call_bind_c_integer_function()) all_passed = .false.
    if (.not. test_call_external_void_subroutine_with_c_ptr()) &
        all_passed = .false.
    if (.not. test_internal_function_bind_c_emits_named_symbol()) &
        all_passed = .false.
    if (.not. test_internal_bind_c_function_is_callable()) all_passed = .false.
    if (.not. test_c_loc_of_integer_round_trip_with_c_associated()) &
        all_passed = .false.
    if (.not. test_c_associated_on_null_returns_false()) all_passed = .false.
    if (.not. test_c_f_pointer_assigns_storage()) all_passed = .false.
    if (.not. test_interface_bind_c_with_import_compiles()) all_passed = .false.

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
            expect_error_contains(source, 'component', &
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

    logical function test_non_bind_c_interface_lowers()
        ! A plain (non-bind(c)) explicit interface records its signature and
        ! emits no code; an unused declaration lowers cleanly.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  interface'//new_line('a')// &
            '    integer function plain(x)'//new_line('a')// &
            '      integer :: x'//new_line('a')// &
            '    end function plain'//new_line('a')// &
            '  end interface'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_non_bind_c_interface_lowers = expect_exit_status( &
            source, 0, '/tmp/ffc_session_iface_plain_test')
    end function test_non_bind_c_interface_lowers

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

        logical function test_call_external_void_subroutine_with_c_ptr()
            ! Void bind(c) subroutine with a c_ptr argument; passing c_null_ptr
            ! sends a null pointer (free(NULL) is a no-op), exits 0.
            character(len=*), parameter :: source = &
                'program main'//new_line('a')// &
                '  use iso_c_binding'//new_line('a')// &
                '  interface'//new_line('a')// &
                '    subroutine free_c(p) bind(c, name="free")'//new_line('a')// &
                '      import :: c_ptr'//new_line('a')// &
                '      type(c_ptr), value :: p'//new_line('a')// &
                '    end subroutine free_c'//new_line('a')// &
                '  end interface'//new_line('a')// &
                '  call free_c(c_null_ptr)'//new_line('a')// &
                '  stop 0'//new_line('a')// &
                'end program main'

            test_call_external_void_subroutine_with_c_ptr = expect_exit_status( &
                source, 0, '/tmp/ffc_session_extern_void_test')
        end function test_call_external_void_subroutine_with_c_ptr

        logical function test_internal_function_bind_c_emits_named_symbol()
            ! A contained function with bind(c, name="...") is emitted under the
            ! unmangled C symbol so other languages can link against it.
            character(len=*), parameter :: source = &
                'program main'//new_line('a')// &
                '  integer :: r'//new_line('a')// &
                '  r = my_add(2, 3)'//new_line('a')// &
                '  stop r'//new_line('a')// &
                'contains'//new_line('a')// &
                '  integer function my_add(a, b) bind(c, name="my_add")'// &
                new_line('a')// &
                '    integer, intent(in) :: a, b'//new_line('a')// &
                '    my_add = a + b'//new_line('a')// &
                '  end function my_add'//new_line('a')// &
                'end program main'

            test_internal_function_bind_c_emits_named_symbol = &
                expect_exe_has_symbol(source, '/tmp/ffc_session_bindc_sym_test', &
                'my_add')
        end function test_internal_function_bind_c_emits_named_symbol

        logical function test_internal_bind_c_function_is_callable()
            ! The bind(c) function is still callable internally under its C name.
            character(len=*), parameter :: source = &
                'program main'//new_line('a')// &
                '  integer :: r'//new_line('a')// &
                '  r = my_add(2, 3)'//new_line('a')// &
                '  stop r'//new_line('a')// &
                'contains'//new_line('a')// &
                '  integer function my_add(a, b) bind(c, name="my_add")'// &
                new_line('a')// &
                '    integer, intent(in) :: a, b'//new_line('a')// &
                '    my_add = a + b'//new_line('a')// &
                '  end function my_add'//new_line('a')// &
                'end program main'

            test_internal_bind_c_function_is_callable = expect_exit_status( &
                source, 5, '/tmp/ffc_session_bindc_call_test')
        end function test_internal_bind_c_function_is_callable

        logical function test_c_loc_of_integer_round_trip_with_c_associated()
            ! c_loc(x) returns a non-null pointer; c_associated reports it true.
            character(len=*), parameter :: source = &
                'program main'//new_line('a')// &
                '  use iso_c_binding'//new_line('a')// &
                '  integer, target :: x'//new_line('a')// &
                '  type(c_ptr) :: p'//new_line('a')// &
                '  x = 42'//new_line('a')// &
                '  p = c_loc(x)'//new_line('a')// &
                '  if (c_associated(p)) then'//new_line('a')// &
                '    stop 1'//new_line('a')// &
                '  else'//new_line('a')// &
                '    stop 2'//new_line('a')// &
                '  end if'//new_line('a')// &
                'end program main'

            test_c_loc_of_integer_round_trip_with_c_associated = &
                expect_exit_status(source, 1, '/tmp/ffc_session_c_loc_test')
        end function test_c_loc_of_integer_round_trip_with_c_associated

        logical function test_c_associated_on_null_returns_false()
            ! c_associated(c_null_ptr) is false.
            character(len=*), parameter :: source = &
                'program main'//new_line('a')// &
                '  use iso_c_binding'//new_line('a')// &
                '  type(c_ptr) :: p'//new_line('a')// &
                '  p = c_null_ptr'//new_line('a')// &
                '  if (c_associated(p)) then'//new_line('a')// &
                '    stop 1'//new_line('a')// &
                '  else'//new_line('a')// &
                '    stop 2'//new_line('a')// &
                '  end if'//new_line('a')// &
                'end program main'

            test_c_associated_on_null_returns_false = &
                expect_exit_status(source, 2, '/tmp/ffc_session_c_assoc_null_test')
        end function test_c_associated_on_null_returns_false

        logical function test_c_f_pointer_assigns_storage()
            ! c_f_pointer(p, q) crosses p into q; q then reports associated.
            character(len=*), parameter :: source = &
                'program main'//new_line('a')// &
                '  use iso_c_binding'//new_line('a')// &
                '  integer, target :: x'//new_line('a')// &
                '  type(c_ptr) :: p, q'//new_line('a')// &
                '  x = 7'//new_line('a')// &
                '  p = c_loc(x)'//new_line('a')// &
                '  call c_f_pointer(p, q)'//new_line('a')// &
                '  if (c_associated(q)) then'//new_line('a')// &
                '    stop 3'//new_line('a')// &
                '  else'//new_line('a')// &
                '    stop 4'//new_line('a')// &
                '  end if'//new_line('a')// &
                'end program main'

            test_c_f_pointer_assigns_storage = &
                expect_exit_status(source, 3, '/tmp/ffc_session_c_f_pointer_test')
        end function test_c_f_pointer_assigns_storage

        logical function test_interface_bind_c_with_import_compiles()
            ! An interface body may carry import :: of a host-scope type used as a
            ! value argument; it parses and registers without error.
            character(len=*), parameter :: source = &
                'program main'//new_line('a')// &
                '  use iso_c_binding'//new_line('a')// &
                '  type, bind(c) :: my_type'//new_line('a')// &
                '    integer(c_int) :: a'//new_line('a')// &
                '  end type my_type'//new_line('a')// &
                '  interface'//new_line('a')// &
                '    integer(c_int) function use_it(v) bind(c, name="use_it")'// &
                new_line('a')// &
                '      import :: my_type, c_int'//new_line('a')// &
                '      type(my_type), value :: v'//new_line('a')// &
                '    end function use_it'//new_line('a')// &
                '  end interface'//new_line('a')// &
                '  stop 0'//new_line('a')// &
                'end program main'

            test_interface_bind_c_with_import_compiles = expect_exit_status( &
                source, 0, '/tmp/ffc_session_iface_import_test')
        end function test_interface_bind_c_with_import_compiles

    end program test_session_iso_c_binding_compiler
