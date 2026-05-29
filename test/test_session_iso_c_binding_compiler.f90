program test_session_iso_c_binding_compiler
    use ffc_test_support, only: expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== direct session iso_c_binding compiler test ==='

    all_passed = .true.
    if (.not. test_use_iso_c_binding_alone_compiles()) all_passed = .false.
    if (.not. test_use_iso_c_binding_intrinsic_form()) all_passed = .false.
    if (.not. test_c_int_kind_used_in_declaration()) all_passed = .false.
    if (.not. test_c_ptr_declaration_and_null_init()) all_passed = .false.
    if (.not. test_c_ptr_assignment_from_other_c_ptr()) all_passed = .false.

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

end program test_session_iso_c_binding_compiler
