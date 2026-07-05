program test_session_reject_assumed_size_order_compiler
    use ffc_test_support, only: expect_error_contains, expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== assumed-size dimension order rejection compiler test ==='

    all_passed = .true.
    if (.not. test_leading_assumed_size_rejected()) all_passed = .false.
    if (.not. test_trailing_assumed_size_accepted()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: leading assumed-size (*) dimension rejected, ' // &
        'trailing assumed-size dimension still accepted'

contains

    logical function test_leading_assumed_size_rejected()
        ! An assumed-size specifier (*) is only well-formed as the extent of
        ! the last array dimension (gfortran: "Bad specification for
        ! assumed size array" / "cannot be implied-shape").
        character(len=*), parameter :: source = &
            'subroutine foo(a)'//new_line('a')// &
            '  integer :: a(*,*)'//new_line('a')// &
            'end subroutine foo'//new_line('a')// &
            'program main'//new_line('a')// &
            'end program main'

        test_leading_assumed_size_rejected = expect_error_contains( &
            source, 'assumed size', '/tmp/ffc_session_assumed_size_reject')
    end function test_leading_assumed_size_rejected

    logical function test_trailing_assumed_size_accepted()
        ! The nearest valid form: only the last dimension is assumed-size.
        character(len=*), parameter :: source = &
            'subroutine foo(a)'//new_line('a')// &
            '  integer :: a(10,*)'//new_line('a')// &
            '  print *, a(1,1)'//new_line('a')// &
            'end subroutine foo'//new_line('a')// &
            'program main'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_trailing_assumed_size_accepted = expect_exit_status( &
            source, 0, '/tmp/ffc_session_assumed_size_accept')
    end function test_trailing_assumed_size_accepted

end program test_session_reject_assumed_size_order_compiler
