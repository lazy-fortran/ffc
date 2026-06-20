program test_session_use_only_compiler
    use ffc_test_support, only: expect_exit_status, expect_error_contains
    implicit none

    logical :: all_passed

    print *, '=== direct session use-only compiler test ==='

    all_passed = .true.
    if (.not. test_use_only_admits_listed_constant()) all_passed = .false.
    if (.not. test_use_only_filters_to_one_constant()) all_passed = .false.
    if (.not. test_use_only_rejects_unknown_name()) all_passed = .false.
    if (.not. test_use_only_rename()) all_passed = .false.
    if (.not. test_use_only_rename_hides_original()) all_passed = .false.
    if (.not. test_use_only_admits_module_function()) all_passed = .false.
    if (.not. test_use_only_admits_module_subroutine()) all_passed = .false.
    if (.not. test_use_rename_module_variable()) all_passed = .false.
    if (.not. test_use_rename_module_function()) all_passed = .false.
    if (.not. test_use_rename_module_subroutine()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: use ... only: filters imported symbols'

contains

    logical function test_use_only_admits_listed_constant()
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '  integer, parameter :: A = 1'//new_line('a')// &
            '  integer, parameter :: B = 2'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m, only: A'//new_line('a')// &
            '  stop A'//new_line('a')// &
            'end program main'

        test_use_only_admits_listed_constant = expect_exit_status( &
            source, 1, '/tmp/ffc_session_use_only_admit_test')
    end function test_use_only_admits_listed_constant

    logical function test_use_only_filters_to_one_constant()
        ! B is not imported, so referencing it is rejected.
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '  integer, parameter :: A = 1'//new_line('a')// &
            '  integer, parameter :: B = 2'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m, only: A'//new_line('a')// &
            '  stop A + B'//new_line('a')// &
            'end program main'

        test_use_only_filters_to_one_constant = expect_error_contains( &
            source, 'B', '/tmp/ffc_session_use_only_filter_test')
    end function test_use_only_filters_to_one_constant

    logical function test_use_only_rejects_unknown_name()
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '  integer, parameter :: A = 1'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m, only: ZZZ'//new_line('a')// &
            '  stop A'//new_line('a')// &
            'end program main'

        test_use_only_rejects_unknown_name = expect_error_contains( &
            source, 'not exported by module', &
            '/tmp/ffc_session_use_only_unknown_test')
    end function test_use_only_rejects_unknown_name

    logical function test_use_only_rename()
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '  integer, parameter :: WIDTH = 80'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m, only: COLS => WIDTH'//new_line('a')// &
            '  stop COLS'//new_line('a')// &
            'end program main'

        test_use_only_rename = expect_exit_status( &
            source, 80, '/tmp/ffc_session_use_rename_test')
    end function test_use_only_rename

    logical function test_use_only_rename_hides_original()
        ! Under the rename, the original module name is not in local scope.
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '  integer, parameter :: WIDTH = 80'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m, only: COLS => WIDTH'//new_line('a')// &
            '  stop WIDTH'//new_line('a')// &
            'end program main'

        test_use_only_rename_hides_original = expect_error_contains( &
            source, 'WIDTH', '/tmp/ffc_session_use_rename_hide_test')
    end function test_use_only_rename_hides_original

    logical function test_use_only_admits_module_function()
        ! A module procedure named in only: is a valid export (#263).
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            'contains'//new_line('a')// &
            '  integer function square(x) result(res)'//new_line('a')// &
            '    integer, intent(in) :: x'//new_line('a')// &
            '    res = x * x'//new_line('a')// &
            '  end function square'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m, only: square'//new_line('a')// &
            '  stop square(5)'//new_line('a')// &
            'end program main'

        test_use_only_admits_module_function = expect_exit_status( &
            source, 25, '/tmp/ffc_session_use_only_function_test')
    end function test_use_only_admits_module_function

    logical function test_use_only_admits_module_subroutine()
        ! A module subroutine named in only: is a valid export (#263).
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine no_op()'//new_line('a')// &
            '  end subroutine no_op'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m, only: no_op'//new_line('a')// &
            '  call no_op()'//new_line('a')// &
            '  stop 7'//new_line('a')// &
            'end program main'

        test_use_only_admits_module_subroutine = expect_exit_status( &
            source, 7, '/tmp/ffc_session_use_only_subroutine_test')
    end function test_use_only_admits_module_subroutine

    logical function test_use_rename_module_variable()
        ! Renaming a module variable binds the local name to the shared
        ! global (#274).
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer :: counter = 42'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m, only: local_counter => counter'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  stop local_counter'//new_line('a')// &
            'end program main'

        test_use_rename_module_variable = expect_exit_status( &
            source, 42, '/tmp/ffc_session_use_rename_var_test')
    end function test_use_rename_module_variable

    logical function test_use_rename_module_function()
        ! Renaming a module function aliases the call site (#274).
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            'contains'//new_line('a')// &
            '  integer function square(x) result(res)'//new_line('a')// &
            '    integer, intent(in) :: x'//new_line('a')// &
            '    res = x * x'//new_line('a')// &
            '  end function square'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m, only: sq => square'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  stop sq(6)'//new_line('a')// &
            'end program main'

        test_use_rename_module_function = expect_exit_status( &
            source, 36, '/tmp/ffc_session_use_rename_fn_test')
    end function test_use_rename_module_function

    logical function test_use_rename_module_subroutine()
        ! Renaming a module subroutine aliases the call site (#274).
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine no_op()'//new_line('a')// &
            '  end subroutine no_op'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m, only: run => no_op'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  call run()'//new_line('a')// &
            '  stop 9'//new_line('a')// &
            'end program main'

        test_use_rename_module_subroutine = expect_exit_status( &
            source, 9, '/tmp/ffc_session_use_rename_sub_test')
    end function test_use_rename_module_subroutine

end program test_session_use_only_compiler
