! fo: dispatcher
module ffc_case_test_session_read_stdin_compiler
    implicit none
    private
    public :: case_test_session_read_stdin_compiler
    interface
        subroutine case_test_session_read_stdin_compiler()
        end subroutine case_test_session_read_stdin_compiler
    end interface
end module ffc_case_test_session_read_stdin_compiler

subroutine case_test_session_read_stdin_compiler()
    ! read(*,*) var  reads list-directed input from stdin using scanf.
    use ffc_test_support, only: expect_output_with_stdin
    implicit none
    save

    logical :: all_passed

    print *, '=== direct session read(*,*) compiler test ==='

    all_passed = .true.
    if (.not. test_read_integer_roundtrip()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: read(*,*) lowers through direct LIRIC session'

contains

    logical function test_read_integer_roundtrip()
        ! read(*,*) x then write(*,*) x should echo the value.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: x'//new_line('a')// &
            '  read (*, *) x'//new_line('a')// &
            '  write (*, *) x'//new_line('a')// &
            'end program main'

        test_read_integer_roundtrip = expect_output_with_stdin( &
            source, '17', '          17'//new_line('a'), &
            '/tmp/ffc_read_stdin_int_test')
    end function test_read_integer_roundtrip

end subroutine case_test_session_read_stdin_compiler
