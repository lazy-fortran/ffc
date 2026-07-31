program test_session_c_interop
    use ffc_test_support, only: expect_error_contains, expect_exit_status, &
                                expect_output
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

    ! c_f_strpointer(cptr, fptr, nchars) associates a deferred-length
    ! character pointer with borrowed C character storage: len(fptr) is the
    ! number of characters before the first C null, capped at nchars, and the
    ! view is borrowed, so writes through the target are observed (#361).
    if (.not. expect_exit_status( &
        'program main'//new_line('a')// &
        'use, intrinsic :: iso_c_binding, only: c_ptr, c_loc, '// &
        'c_f_strpointer, c_char'//new_line('a')// &
        'character(len=12, kind=c_char), target :: s'//new_line('a')// &
        'character(len=:), pointer :: fp, fp5'//new_line('a')// &
        'type(c_ptr) :: cp'//new_line('a')// &
        's = "hello world!"'//new_line('a')// &
        'cp = c_loc(s)'//new_line('a')// &
        'call c_f_strpointer(cp, fp, 12)'//new_line('a')// &
        'call c_f_strpointer(cp, fp5, 5)'//new_line('a')// &
        'if (len(fp) /= 12) stop 100'//new_line('a')// &
        'if (len(fp5) /= 5) stop 101'//new_line('a')// &
        'if (fp /= "hello world!") stop 102'//new_line('a')// &
        'if (fp5 /= "hello") stop 103'//new_line('a')// &
        's = "Xello world!"'//new_line('a')// &
        'if (fp /= "Xello world!") stop 104'//new_line('a')// &
        'stop 12'//new_line('a')// &
        'end program main', 12, &
        '/tmp/ffc_session_c_strpointer_status_test')) stop 3

    ! The associated length is a runtime value usable in expressions.
    if (.not. expect_output( &
        'program main'//new_line('a')// &
        'use, intrinsic :: iso_c_binding, only: c_ptr, c_loc, '// &
        'c_f_strpointer, c_char'//new_line('a')// &
        'character(len=12, kind=c_char), target :: s'//new_line('a')// &
        'character(len=:), pointer :: fp'//new_line('a')// &
        'type(c_ptr) :: cp'//new_line('a')// &
        's = "hello world!"'//new_line('a')// &
        'cp = c_loc(s)'//new_line('a')// &
        'call c_f_strpointer(cp, fp, 7)'//new_line('a')// &
        'print *, len(fp)'//new_line('a')// &
        'end program main', &
        '           7'//new_line('a'), &
        '/tmp/ffc_session_c_strpointer_output_test')) stop 4

    ! A c_ptr cstring carries no length metadata, so omitting nchars must
    ! produce a diagnostic instead of an unbounded view (#361).
    if (.not. expect_error_contains( &
        'program main'//new_line('a')// &
        'use, intrinsic :: iso_c_binding, only: c_ptr, c_null_ptr, '// &
        'c_f_strpointer'//new_line('a')// &
        'character(len=:), pointer :: fp'//new_line('a')// &
        'type(c_ptr) :: cp'//new_line('a')// &
        'cp = c_null_ptr'//new_line('a')// &
        'call c_f_strpointer(cp, fp)'//new_line('a')// &
        'end program main', 'c_f_strpointer', &
        '/tmp/ffc_session_c_strpointer_reject_test')) stop 5

    print *, 'PASS: c_loc / c_associated / c_f_pointer / c_f_strpointer'
end program test_session_c_interop
