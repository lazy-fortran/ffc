submodule (liric_session_memory_bindings) liric_session_integer_copy_bindings
    implicit none
contains
    module function emit_i64_copy_to(session, value, dest_vreg, result, &
            error_msg) result(ok)
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: value
        integer(c_int32_t), intent(in) :: dest_vreg
        type(lr_operand_desc_t), intent(out) :: result
        character(len=:), allocatable, intent(out) :: error_msg
        logical :: ok
        type(lr_error_t) :: error
        integer(c_int32_t) :: vreg

        ok = .false.
        if (.not. require_open_session(session, error_msg)) return
        if (dest_vreg <= 0_c_int32_t) then
            error_msg = 'explicit LIRIC destination vreg must be positive'
            return
        end if
        vreg = emit_binary_with_dest(session%handle, LR_OP_ADD, value, &
            i64_immediate(session, 0_c_int64_t), error, dest_vreg)
        if (.not. status_ok(error%code, error, error_msg)) return
        if (vreg /= dest_vreg) then
            error_msg = 'LIRIC did not honor explicit binary destination vreg'
            return
        end if
        result = i64_vreg(session, vreg)
        call set_empty(error_msg)
        ok = .true.
    end function emit_i64_copy_to
end submodule liric_session_integer_copy_bindings
