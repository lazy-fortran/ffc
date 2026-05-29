program test_session_module_visibility_compiler
    use ffc_test_support, only: expect_exit_status, expect_error_contains
    implicit none

    logical :: all_passed

    print *, '=== module visibility (public/private) compiler test ==='

    all_passed = .true.
    if (.not. test_private_constant_not_visible_outside_module()) &
        all_passed = .false.
    if (.not. test_public_constant_visible_through_default_private_module()) &
        all_passed = .false.
    if (.not. test_attribute_form_private_on_decl()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: module visibility honored'

contains

    logical function test_private_constant_not_visible_outside_module()
        ! private :: HIDDEN keeps HIDDEN out of the caller's scope.
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  private :: hidden'//new_line('a')// &
            '  integer, parameter :: hidden = 7'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  stop hidden'//new_line('a')// &
            'end program main'

        test_private_constant_not_visible_outside_module = &
            expect_error_contains(source, 'hidden', &
            '/tmp/ffc_session_vis_private_list_test')
    end function test_private_constant_not_visible_outside_module

    logical function test_public_constant_visible_through_default_private_module()
        ! A bare `private` flips the default; `public :: visible` re-exports it.
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  private'//new_line('a')// &
            '  integer, parameter :: hidden = 7'//new_line('a')// &
            '  public :: visible'//new_line('a')// &
            '  integer, parameter :: visible = 9'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  stop visible'//new_line('a')// &
            'end program main'

        test_public_constant_visible_through_default_private_module = &
            expect_exit_status(source, 9, '/tmp/ffc_session_vis_default_test')
    end function test_public_constant_visible_through_default_private_module

    logical function test_attribute_form_private_on_decl()
        ! integer, private, parameter :: HIDDEN is not exported.
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer, private, parameter :: hidden = 7'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  stop hidden'//new_line('a')// &
            'end program main'

        test_attribute_form_private_on_decl = &
            expect_error_contains(source, 'hidden', &
            '/tmp/ffc_session_vis_attr_test')
    end function test_attribute_form_private_on_decl

end program test_session_module_visibility_compiler
