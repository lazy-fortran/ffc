program test_session_pointer_array
    use ffc_test_support, only: expect_exit_status, expect_output
    implicit none

    print *, '=== direct session rank-1 pointer/target array compiler test ==='

    ! p => t aliases the target array's storage: a write through p(i) mutates
    ! t(i), and size(p) reports the target extent. stop returns t(2), set to 99
    ! through the pointer.
    if (.not. expect_exit_status( &
        'program main'//new_line('a')// &
        'integer, target :: t(3)'//new_line('a')// &
        'integer, pointer :: p(:)'//new_line('a')// &
        't = [10, 20, 30]'//new_line('a')// &
        'p => t'//new_line('a')// &
        'if (size(p) /= 3) stop 1'//new_line('a')// &
        'p(2) = 99'//new_line('a')// &
        'if (t(2) /= 99) stop 2'//new_line('a')// &
        'stop t(2)'//new_line('a')// &
        'end program main', 99, &
        '/tmp/ffc_session_pointer_array_alias')) stop 1

    ! A read through the pointer observes a later write to the target element.
    if (.not. expect_exit_status( &
        'program main'//new_line('a')// &
        'integer, target :: t(4)'//new_line('a')// &
        'integer, pointer :: p(:)'//new_line('a')// &
        'p => t'//new_line('a')// &
        't(3) = 7'//new_line('a')// &
        'stop p(3)'//new_line('a')// &
        'end program main', 7, &
        '/tmp/ffc_session_pointer_array_read')) stop 2

    ! A real target array: pointer element arithmetic flows back to the target.
    if (.not. expect_exit_status( &
        'program main'//new_line('a')// &
        'real, target :: r(3)'//new_line('a')// &
        'real, pointer :: q(:)'//new_line('a')// &
        'r = [1.0, 2.0, 3.0]'//new_line('a')// &
        'q => r'//new_line('a')// &
        'q(1) = q(1) + q(3)'//new_line('a')// &
        'if (abs(r(1) - 4.0) > 1.0e-6) stop 3'//new_line('a')// &
        'stop 0'//new_line('a')// &
        'end program main', 0, &
        '/tmp/ffc_session_pointer_array_real')) stop 3

    ! Whole-array print through the pointer prints the target's elements.
    if (.not. expect_output( &
        'program main'//new_line('a')// &
        'integer, target :: t(3)'//new_line('a')// &
        'integer, pointer :: p(:)'//new_line('a')// &
        't = [4, 5, 6]'//new_line('a')// &
        'p => t'//new_line('a')// &
        'print *, p'//new_line('a')// &
        'end program main', &
        '           4           5           6'//new_line('a'), &
        '/tmp/ffc_session_pointer_array_print')) stop 4

    print *, 'PASS: rank-1 integer/real pointer/target array, => , element access'
end program test_session_pointer_array
