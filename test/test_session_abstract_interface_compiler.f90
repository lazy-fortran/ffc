program test_session_abstract_interface_compiler
    use ffc_test_support, only: expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== direct session abstract interface compiler test ==='

    all_passed = .true.
    if (.not. test_abstract_interface_is_noop()) all_passed = .false.
    if (.not. test_explicit_interface_subroutine_call()) all_passed = .false.
    if (.not. test_explicit_interface_function_decl()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: abstract and explicit interface blocks lower correctly'

contains

    logical function test_abstract_interface_is_noop()
        ! An abstract interface declares a signature template only; it binds no
        ! symbol and emits no code, so a program carrying one runs unchanged.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: v'//new_line('a')// &
            '  abstract interface'//new_line('a')// &
            '    function transform(x) result(y)'//new_line('a')// &
            '      integer, intent(in) :: x'//new_line('a')// &
            '      integer :: y'//new_line('a')// &
            '    end function transform'//new_line('a')// &
            '  end interface'//new_line('a')// &
            '  v = 5'//new_line('a')// &
            '  stop v'//new_line('a')// &
            'end program main'

        test_abstract_interface_is_noop = expect_exit_status( &
            source, 5, '/tmp/ffc_abstract_interface_noop_test')
    end function test_abstract_interface_is_noop

    logical function test_explicit_interface_subroutine_call()
        ! A plain explicit interface for an external subroutine lets the CALL
        ! lower by reference; the symbol resolves to the separate definition.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: v'//new_line('a')// &
            '  interface'//new_line('a')// &
            '    subroutine setup(x)'//new_line('a')// &
            '      integer, intent(inout) :: x'//new_line('a')// &
            '    end subroutine setup'//new_line('a')// &
            '  end interface'//new_line('a')// &
            '  v = 3'//new_line('a')// &
            '  call setup(v)'//new_line('a')// &
            '  stop v'//new_line('a')// &
            'end program main'//new_line('a')// &
            'subroutine setup(x)'//new_line('a')// &
            '  integer, intent(inout) :: x'//new_line('a')// &
            '  x = x + 1'//new_line('a')// &
            'end subroutine setup'

        test_explicit_interface_subroutine_call = expect_exit_status( &
            source, 4, '/tmp/ffc_explicit_interface_sub_test')
    end function test_explicit_interface_subroutine_call

    logical function test_explicit_interface_function_decl()
        ! A plain explicit interface for an external function records its
        ! signature and emits no code; an unused declaration lowers cleanly.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  interface'//new_line('a')// &
            '    integer function triple(x)'//new_line('a')// &
            '      integer, intent(in) :: x'//new_line('a')// &
            '    end function triple'//new_line('a')// &
            '  end interface'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_explicit_interface_function_decl = expect_exit_status( &
            source, 0, '/tmp/ffc_explicit_interface_func_test')
    end function test_explicit_interface_function_decl

end program test_session_abstract_interface_compiler
