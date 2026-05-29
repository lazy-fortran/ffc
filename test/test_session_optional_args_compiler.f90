program test_session_optional_args_compiler
    ! optional dummy arguments and the present() intrinsic. Absence is a null
    ! reference pointer; present(x) is a null check.
    use ffc_test_support, only: expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== direct session optional argument compiler test ==='

    all_passed = .true.
    if (.not. test_optional_present_and_used()) all_passed = .false.
    if (.not. test_optional_absent_takes_default()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: optional arguments and present() lower through direct LIRIC'

contains

    logical function test_optional_present_and_used()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  call sub(7)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine sub(x)'//new_line('a')// &
            '    integer, optional, intent(in) :: x'//new_line('a')// &
            '    if (present(x)) stop x'//new_line('a')// &
            '    stop 99'//new_line('a')// &
            '  end subroutine sub'//new_line('a')// &
            'end program main'

        test_optional_present_and_used = expect_exit_status( &
            source, 7, '/tmp/ffc_optional_present_test')
    end function test_optional_present_and_used

    logical function test_optional_absent_takes_default()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  call sub()'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine sub(x)'//new_line('a')// &
            '    integer, optional, intent(in) :: x'//new_line('a')// &
            '    if (.not. present(x)) stop 99'//new_line('a')// &
            '    stop x'//new_line('a')// &
            '  end subroutine sub'//new_line('a')// &
            'end program main'

        test_optional_absent_takes_default = expect_exit_status( &
            source, 99, '/tmp/ffc_optional_absent_test')
    end function test_optional_absent_takes_default

end program test_session_optional_args_compiler
