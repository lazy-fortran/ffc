program test_liric_session_bindings
    use liric_session_bindings, only: liric_session_t, &
        liric_session_create, destroy, is_open, &
        emit_ret_i32_main_exe, emit_ret_i32_operand, &
        finish_and_emit_exe, begin_i32_main, i32_immediate
    use liric_session_memory_bindings, only: emit_i32_alloca, &
        emit_i32_store, emit_i32_load
    use liric_session_procedure_bindings, only: &
        begin_liric_f64_function, emit_liric_f64_alloca, &
        emit_liric_f64_store, emit_liric_f64_load
    use liric_session_common, only: lr_operand_desc_t
    use, intrinsic :: iso_c_binding, only: c_int64_t
    implicit none

    type(liric_session_t) :: session
    character(len=:), allocatable :: error_msg
    character(len=*), parameter :: exe_path = '/tmp/ffc_liric_session_ret_0'
    character(len=*), parameter :: exe_path2 = '/tmp/ffc_liric_session_mem_test'
    integer :: exit_stat
    integer :: cmd_stat
    logical :: ok
    type(lr_operand_desc_t) :: addr, val, src, tmp

    print *, '=== LIRIC session binding tests ==='

    call execute_command_line('rm -f '//exe_path)

    call liric_session_create(session, error_msg)
    if (len_trim(error_msg) > 0) then
        print *, 'FAIL: create returned ', trim(error_msg)
        stop 1
    end if
    if (.not. is_open(session)) then
        print *, 'FAIL: session handle was not opened'
        stop 1
    end if

    ok = emit_ret_i32_main_exe(session, 0, exe_path, error_msg)
    if (.not. ok) then
        print *, 'FAIL: direct session emit returned ', trim(error_msg)
        call destroy(session)
        stop 1
    end if

    call execute_command_line(exe_path, exitstat=exit_stat, &
                              cmdstat=cmd_stat)
    if (cmd_stat /= 0) then
        print *, 'FAIL: could not run emitted executable'
        call destroy(session)
        stop 1
    end if
    if (exit_stat /= 0) then
        print *, 'FAIL: executable returned ', exit_stat
        call destroy(session)
        stop 1
    end if

    call destroy(session)
    if (is_open(session)) then
        print *, 'FAIL: session handle was not closed'
        stop 1
    end if

    call execute_command_line('rm -f '//exe_path)

    ! Test i32 alloca/store/load path (issue #234)
    call execute_command_line('rm -f '//exe_path2)

    call liric_session_create(session, error_msg)
    if (len_trim(error_msg) > 0) then
        print *, 'FAIL: create (mem) returned ', trim(error_msg)
        call destroy(session)
        stop 1
    end if

    ok = begin_i32_main(session, error_msg)
    if (.not. ok) then
        print *, 'FAIL: begin_i32_main (mem) returned ', trim(error_msg)
        call destroy(session)
        stop 1
    end if

    ok = emit_i32_alloca(session, addr, error_msg)
    if (.not. ok) then
        print *, 'FAIL: emit_i32_alloca returned ', trim(error_msg)
        call destroy(session)
        stop 1
    end if

    val = i32_immediate(session, 42_c_int64_t)
    ok = emit_i32_store(session, val, addr, error_msg)
    if (.not. ok) then
        print *, 'FAIL: emit_i32_store returned ', trim(error_msg)
        call destroy(session)
        stop 1
    end if

    ok = emit_i32_load(session, addr, val, error_msg)
    if (.not. ok) then
        print *, 'FAIL: emit_i32_load returned ', trim(error_msg)
        call destroy(session)
        stop 1
    end if

    ok = emit_ret_i32_operand(session, val, error_msg)
    if (.not. ok) then
        print *, 'FAIL: emit_ret_i32_operand returned ', trim(error_msg)
        call destroy(session)
        stop 1
    end if

    ok = finish_and_emit_exe(session, exe_path2, error_msg)
    if (.not. ok) then
        print *, 'FAIL: finish_and_emit_exe (mem) returned ', trim(error_msg)
        call destroy(session)
        stop 1
    end if

    call destroy(session)

    call execute_command_line(exe_path2, exitstat=exit_stat, &
                              cmdstat=cmd_stat)
    if (cmd_stat /= 0) then
        print *, 'FAIL: could not run mem test executable'
        call execute_command_line('rm -f '//exe_path2)
        stop 1
    end if
    if (exit_stat /= 42) then
        print *, 'FAIL: mem test executable returned ', exit_stat, ' (expected 42)'
        call execute_command_line('rm -f '//exe_path2)
        stop 1
    end if

    call execute_command_line('rm -f '//exe_path2)

    ! Test f64 alloca/store/load path (issue #234)
    call execute_command_line('rm -f '//exe_path2)

    call liric_session_create(session, error_msg)
    if (len_trim(error_msg) > 0) then
        print *, 'FAIL: create (f64) returned ', trim(error_msg)
        call destroy(session)
        stop 1
    end if

    ok = begin_liric_f64_function(session, 'f64_helper', 0, error_msg)
    if (.not. ok) then
        print *, 'FAIL: begin_liric_f64_function returned ', trim(error_msg)
        call destroy(session)
        stop 1
    end if

    ok = emit_liric_f64_alloca(session, addr, error_msg)
    if (.not. ok) then
        print *, 'FAIL: emit_liric_f64_alloca returned ', trim(error_msg)
        call destroy(session)
        stop 1
    end if

    ! f64 smoke: alloca'd pointer can be loaded, stored, and re-loaded.
    ! begin_liric_f64_function creates an f64-returning function, so we
    ! cannot emit an executable from it. We verify the helpers succeed.
    ok = emit_liric_f64_load(session, addr, val, error_msg)
    if (.not. ok) then
        print *, 'FAIL: emit_liric_f64_load returned ', trim(error_msg)
        call destroy(session)
        stop 1
    end if

    src = val
    ok = emit_liric_f64_store(session, src, addr, error_msg)
    if (.not. ok) then
        print *, 'FAIL: emit_liric_f64_store returned ', trim(error_msg)
        call destroy(session)
        stop 1
    end if

    ok = emit_liric_f64_load(session, addr, tmp, error_msg)
    if (.not. ok) then
        print *, 'FAIL: emit_liric_f64_load(2) returned ', trim(error_msg)
        call destroy(session)
        stop 1
    end if

    ! f64 helpers exercised; no executable emit from function context.

    call destroy(session)

    call execute_command_line('rm -f '//exe_path2)

    print *, 'PASS: LIRIC session binding tests'
end program test_liric_session_bindings
