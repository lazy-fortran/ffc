program test_session_c_interop
    use ffc_test_support, only: expect_exit_status, expect_output
    implicit none

    print *, '=== direct session ISO_C_BINDING pointer round-trip test ==='

    ! c_loc(x) yields x's address; c_associated(cp) is true for a non-null
    ! pointer; c_f_pointer(cp, p) binds p to that address so a read of p
    ! dereferences it and observes x's value (issue #2820).
    if (.not. expect_exit_status( &
        'program main'//new_line('a')// &
        'use, intrinsic :: iso_c_binding, only: c_ptr, c_loc, c_f_pointer, '// &
        'c_associated, c_int'//new_line('a')// &
        'integer(c_int), target :: x'//new_line('a')// &
        'integer(c_int), pointer :: p'//new_line('a')// &
        'type(c_ptr) :: cp'//new_line('a')// &
        'x = 5'//new_line('a')// &
        'cp = c_loc(x)'//new_line('a')// &
        'if (.not. c_associated(cp)) stop 1'//new_line('a')// &
        'call c_f_pointer(cp, p)'//new_line('a')// &
        'stop p'//new_line('a')// &
        'end program main', 5, &
        '/tmp/ffc_session_c_interop_status_test')) stop 1

    ! The same round-trip prints the dereferenced value.
    if (.not. expect_output( &
        'program main'//new_line('a')// &
        'use, intrinsic :: iso_c_binding, only: c_ptr, c_loc, c_f_pointer, '// &
        'c_associated, c_int'//new_line('a')// &
        'integer(c_int), target :: x'//new_line('a')// &
        'integer(c_int), pointer :: p'//new_line('a')// &
        'type(c_ptr) :: cp'//new_line('a')// &
        'x = 7'//new_line('a')// &
        'cp = c_loc(x)'//new_line('a')// &
        'if (c_associated(cp)) then'//new_line('a')// &
        'call c_f_pointer(cp, p)'//new_line('a')// &
        'print *, p'//new_line('a')// &
        'end if'//new_line('a')// &
        'end program main', &
        '           7'//new_line('a'), &
        '/tmp/ffc_session_c_interop_output_test')) stop 2

    print *, 'PASS: c_loc / c_associated / c_f_pointer scalar round-trip'
end program test_session_c_interop
