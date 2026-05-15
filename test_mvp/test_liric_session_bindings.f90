program test_liric_session_bindings
    use liric_session_bindings, only: liric_session_t, liric_session_create
    implicit none

    type(liric_session_t) :: session
    character(len=:), allocatable :: error_msg
    character(len=*), parameter :: exe_path = '/tmp/ffc_liric_session_ret_0'
    integer :: exit_stat
    integer :: cmd_stat
    logical :: ok

    print *, '=== LIRIC session binding tests ==='

    call execute_command_line('rm -f '//exe_path)

    call liric_session_create(session, error_msg)
    if (len_trim(error_msg) > 0) then
        print *, 'FAIL: create returned ', trim(error_msg)
        stop 1
    end if
    if (.not. session%is_open()) then
        print *, 'FAIL: session handle was not opened'
        stop 1
    end if

    ok = session%emit_ret_i32_main_exe(0, exe_path, error_msg)
    if (.not. ok) then
        print *, 'FAIL: direct session emit returned ', trim(error_msg)
        call session%destroy()
        stop 1
    end if

    call execute_command_line(exe_path, exitstat=exit_stat, &
                              cmdstat=cmd_stat)
    if (cmd_stat /= 0) then
        print *, 'FAIL: could not run emitted executable'
        call session%destroy()
        stop 1
    end if
    if (exit_stat /= 0) then
        print *, 'FAIL: executable returned ', exit_stat
        call session%destroy()
        stop 1
    end if

    call session%destroy()
    if (session%is_open()) then
        print *, 'FAIL: session handle was not closed'
        stop 1
    end if

    call execute_command_line('rm -f '//exe_path)

    print *, 'PASS: LIRIC session binding tests'
end program test_liric_session_bindings
