program test_session_deferred_intent_out_compiler
    use ffc_test_support, only: expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== deferred-length intent(out) character dummy tests ==='

    all_passed = .true.
    if (.not. test_alloc_intent_out_returns_string()) all_passed = .false.
    if (.not. test_alloc_intent_out_length_via_caller()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: deferred-length intent(out) character dummy'

contains

    logical function test_alloc_intent_out_returns_string()
        ! A subroutine assigns its character(len=:), allocatable, intent(out)
        ! dummy; the caller sees the assigned length.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=:), allocatable :: msg'//new_line('a')// &
            '  call make_msg(msg)'//new_line('a')// &
            '  stop len(msg)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine make_msg(msg)'//new_line('a')// &
            '    character(len=:), allocatable, intent(out) :: msg'// &
            new_line('a')// &
            '    msg = "hello"'//new_line('a')// &
            '  end subroutine make_msg'//new_line('a')// &
            'end program main'

        test_alloc_intent_out_returns_string = expect_exit_status( &
            source, 5, '/tmp/ffc_session_alloc_intent_out_test')
    end function test_alloc_intent_out_returns_string

    logical function test_alloc_intent_out_length_via_caller()
        ! len_trim on the returned value reflects the callee's concatenation.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=:), allocatable :: msg'//new_line('a')// &
            '  call build(msg)'//new_line('a')// &
            '  stop len_trim(msg)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine build(msg)'//new_line('a')// &
            '    character(len=:), allocatable, intent(out) :: msg'// &
            new_line('a')// &
            '    msg = "ab" // "cde"'//new_line('a')// &
            '  end subroutine build'//new_line('a')// &
            'end program main'

        test_alloc_intent_out_length_via_caller = expect_exit_status( &
            source, 5, '/tmp/ffc_session_alloc_intent_out_concat_test')
    end function test_alloc_intent_out_length_via_caller

end program test_session_deferred_intent_out_compiler
