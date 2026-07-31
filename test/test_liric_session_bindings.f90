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
    use liric_session_common, only: lr_session_abi_info_t, &
        lr_session_get_abi_info, verify_liric_abi, &
        set_liric_abi_override_for_testing, &
        clear_liric_abi_override_for_testing, &
        FFC_EXPECTED_LIRIC_ABI_VERSION, FFC_EXPECTED_OPCODE_COUNT, &
        FFC_EXPECTED_OPERAND_KIND_COUNT, lr_session_config_t, lr_error_t, &
        lr_inst_desc_t, LR_OK
    use, intrinsic :: iso_c_binding, only: c_int64_t, c_size_t
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

    call check_liric_abi_contract()

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
contains

    ! Issue #375: ffc's mirrored LIRIC structures and opcode constants are
    ! verified against the library's published session ABI before any
    ! instruction is emitted. A mismatch must stop us here, not corrupt
    ! descriptors later.
    subroutine check_liric_abi_contract()
        type(lr_session_abi_info_t) :: info
        type(lr_session_abi_info_t) :: bad
        type(lr_session_config_t) :: config_probe
        type(lr_error_t) :: error_probe
        type(lr_operand_desc_t) :: operand_probe
        type(lr_inst_desc_t) :: inst_probe
        character(len=:), allocatable :: error_msg
        integer :: status

        ! Positive: the linked LIRIC reports exactly what ffc mirrors.
        status = int(lr_session_get_abi_info(info, &
                                             int(storage_size(info)/8, c_size_t)))
        if (status /= int(LR_OK)) then
            print *, 'FAIL: lr_session_get_abi_info returned ', status
            stop 1
        end if
        if (info%abi_version /= int(FFC_EXPECTED_LIRIC_ABI_VERSION, &
                                    kind(info%abi_version))) then
            print *, 'FAIL: unexpected LIRIC ABI version ', info%abi_version
            stop 1
        end if
        if (info%config_size /= int(storage_size(config_probe)/8, &
                                    kind(info%config_size))) then
            print *, 'FAIL: lr_session_config_t size differs from LIRIC'
            stop 1
        end if
        if (info%error_size /= int(storage_size(error_probe)/8, &
                                   kind(info%error_size))) then
            print *, 'FAIL: lr_error_t size differs from LIRIC'
            stop 1
        end if
        if (info%operand_size /= int(storage_size(operand_probe)/8, &
                                     kind(info%operand_size))) then
            print *, 'FAIL: lr_operand_desc_t size differs from LIRIC'
            stop 1
        end if
        if (info%inst_size /= int(storage_size(inst_probe)/8, &
                                  kind(info%inst_size))) then
            print *, 'FAIL: lr_inst_desc_t size differs from LIRIC'
            stop 1
        end if
        if (info%opcode_count /= int(FFC_EXPECTED_OPCODE_COUNT, &
                                     kind(info%opcode_count))) then
            print *, 'FAIL: opcode count differs from LIRIC'
            stop 1
        end if
        if (info%operand_kind_count /= int(FFC_EXPECTED_OPERAND_KIND_COUNT, &
                                           kind(info%operand_kind_count))) then
            print *, 'FAIL: operand kind count differs from LIRIC'
            stop 1
        end if
        if (.not. verify_liric_abi(error_msg)) then
            print *, 'FAIL: real LIRIC rejected by ABI check: ', trim(error_msg)
            stop 1
        end if

        ! Negative: injected metadata must be refused before emission, with a
        ! diagnostic naming both the expected and the observed value.
        bad = info
        bad%abi_version = info%abi_version + 1
        call expect_reject(bad, 'version mismatch', 'wrong ABI version')

        bad = info
        bad%operand_size = info%operand_size + 8
        call expect_reject(bad, 'lr_operand_desc_t size', 'wrong operand size')

        bad = info
        bad%inst_size = info%inst_size - 4
        call expect_reject(bad, 'lr_inst_desc_t size', 'wrong inst size')

        bad = info
        bad%config_size = info%config_size + 4
        call expect_reject(bad, 'lr_session_config_t size', 'wrong config size')

        bad = info
        bad%opcode_count = info%opcode_count - 1
        call expect_reject(bad, 'opcode count', 'wrong opcode count')

        bad = info
        bad%operand_kind_count = 1
        call expect_reject(bad, 'operand kind count', 'wrong kind count')

        bad = info
        bad%operand_kind_offset = 8
        call expect_reject(bad, 'operand kind offset', 'shifted kind offset')

        call clear_liric_abi_override_for_testing()
        if (.not. verify_liric_abi(error_msg)) then
            print *, 'FAIL: ABI check did not recover after override'
            stop 1
        end if

        print *, 'PASS: LIRIC ABI verification contract'
    end subroutine check_liric_abi_contract

    subroutine expect_reject(bad, expected_text, label)
        type(lr_session_abi_info_t), intent(in) :: bad
        character(len=*), intent(in) :: expected_text
        character(len=*), intent(in) :: label
        character(len=:), allocatable :: error_msg

        call set_liric_abi_override_for_testing(bad)
        if (verify_liric_abi(error_msg)) then
            call clear_liric_abi_override_for_testing()
            print *, 'FAIL: ', label, ' was accepted'
            stop 1
        end if
        if (index(error_msg, expected_text) == 0) then
            call clear_liric_abi_override_for_testing()
            print *, 'FAIL: ', label, ' diagnostic lacks "', expected_text, &
                '": ', trim(error_msg)
            stop 1
        end if
        call clear_liric_abi_override_for_testing()
    end subroutine expect_reject

end program test_liric_session_bindings
