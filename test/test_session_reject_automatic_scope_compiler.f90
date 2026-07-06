program test_session_reject_automatic_scope_compiler
    use ffc_test_support, only: expect_error_contains, expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== automatic/interface scope rejection compiler test ==='

    all_passed = .true.
    if (.not. test_main_scope_call_bound_rejected()) all_passed = .false.
    if (.not. test_class_scalar_rejected()) all_passed = .false.
    if (.not. test_result_rank_mismatch_rejected()) all_passed = .false.
    if (.not. test_procedure_local_automatic_accepted()) all_passed = .false.
    if (.not. test_parameter_bound_accepted()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: illegal automatic/class/interface forms rejected, ' // &
        'valid procedure-local and parameter-bound arrays still accepted'

contains

    logical function test_main_scope_call_bound_rejected()
        ! An array at main-program scope sized by a function call is not a
        ! constant-bound array; it would be an automatic object where none is
        ! allowed (gfortran: "array with nonconstant bounds").
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(get_n())'//new_line('a')// &
            '  print *, size(a)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  pure integer function get_n()'//new_line('a')// &
            '    get_n = 3'//new_line('a')// &
            '  end function'//new_line('a')// &
            'end program main'

        test_main_scope_call_bound_rejected = expect_error_contains( &
            source, 'array with nonconstant bounds', &
            '/tmp/ffc_session_nonconst_bound_reject')
    end function test_main_scope_call_bound_rejected

    logical function test_class_scalar_rejected()
        ! A polymorphic entity that is neither dummy nor allocatable nor pointer
        ! is invalid (gfortran: "must be dummy, allocatable or pointer").
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type t'//new_line('a')// &
            '    integer :: a'//new_line('a')// &
            '  end type t'//new_line('a')// &
            '  class(t) :: x'//new_line('a')// &
            'end program main'

        test_class_scalar_rejected = expect_error_contains( &
            source, 'must be dummy, allocatable or pointer', &
            '/tmp/ffc_session_class_scalar_reject')
    end function test_class_scalar_rejected

    logical function test_result_rank_mismatch_rejected()
        ! An interface body whose function result rank disagrees with the real
        ! definition is a genuine signature mismatch, not a reconcilable
        ! duplicate (gfortran: "Rank mismatch in function result").
        character(len=*), parameter :: source = &
            'function cntf(a) result(s)'//new_line('a')// &
            '  integer, intent(in) :: a(:)'//new_line('a')// &
            '  integer :: s(3)'//new_line('a')// &
            '  s = [1, 2, 3]'//new_line('a')// &
            'end function cntf'//new_line('a')// &
            'program main'//new_line('a')// &
            '  interface'//new_line('a')// &
            '    function cntf(a) result(s)'//new_line('a')// &
            '      integer, intent(in) :: a(:)'//new_line('a')// &
            '      integer :: s'//new_line('a')// &
            '    end function cntf'//new_line('a')// &
            '  end interface'//new_line('a')// &
            '  integer :: arr(9), s(3)'//new_line('a')// &
            '  s = cntf(arr)'//new_line('a')// &
            'end program main'

        test_result_rank_mismatch_rejected = expect_error_contains( &
            source, 'rank mismatch in function result', &
            '/tmp/ffc_session_result_rank_reject')
    end function test_result_rank_mismatch_rejected

    logical function test_procedure_local_automatic_accepted()
        ! A rank-1 automatic array local to a procedure, sized by a dummy, is
        ! legal Fortran and must still compile and run.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  call fill(4)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine fill(n)'//new_line('a')// &
            '    integer, intent(in) :: n'//new_line('a')// &
            '    integer :: a(n)'//new_line('a')// &
            '    a = 5'//new_line('a')// &
            '    print *, a(2)'//new_line('a')// &
            '  end subroutine'//new_line('a')// &
            'end program main'

        test_procedure_local_automatic_accepted = expect_exit_status( &
            source, 0, '/tmp/ffc_session_proc_automatic_accept')
    end function test_procedure_local_automatic_accepted

    logical function test_parameter_bound_accepted()
        ! A main-scope array with a named-constant bound is a constant-bound
        ! array and must still compile and run.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, parameter :: n = 3'//new_line('a')// &
            '  integer :: a(n)'//new_line('a')// &
            '  a = 7'//new_line('a')// &
            '  print *, a(1)'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_parameter_bound_accepted = expect_exit_status( &
            source, 0, '/tmp/ffc_session_parameter_bound_accept')
    end function test_parameter_bound_accepted

end program test_session_reject_automatic_scope_compiler
