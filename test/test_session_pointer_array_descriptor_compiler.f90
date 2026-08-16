program test_session_pointer_array_descriptor
    use ffc_test_support, only: expect_output, expect_exit_status
    implicit none

    print *, '=== direct session pointer-array descriptor compiler test ==='

    ! p => a copies the whole-array view; q => a(2:8:2) is a strided descriptor
    ! view. Elements through both pointers reach the target storage.
    if (.not. expect_output( &
        'program main'//new_line('a')// &
        'integer, target :: a(8)'//new_line('a')// &
        'integer, pointer :: p(:), q(:)'//new_line('a')// &
        'a = [1, 2, 3, 4, 5, 6, 7, 8]'//new_line('a')// &
        'p => a'//new_line('a')// &
        'q => a(2:8:2)'//new_line('a')// &
        'print *, p(5), q(1), q(2), q(3), q(4)'//new_line('a')// &
        'end program main', &
        '           5           2           4           6           8'//new_line('a'), &
        '/tmp/ffc_session_pointer_array_descriptor')) stop 1

    ! associated(p) is true once p => a has run.
    if (.not. expect_exit_status( &
        'program main'//new_line('a')// &
        'integer, target :: a(3)'//new_line('a')// &
        'integer, pointer :: p(:)'//new_line('a')// &
        'p => a'//new_line('a')// &
        'if (.not. associated(p)) stop 1'//new_line('a')// &
        'stop 0'//new_line('a')// &
        'end program main', 0, &
        '/tmp/ffc_session_pointer_array_descriptor_associated')) stop 2

    print *, 'PASS: pointer array descriptor, whole array and strided section'
end program test_session_pointer_array_descriptor
