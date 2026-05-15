module liric_session_control_bindings
    use, intrinsic :: iso_c_binding, only: c_associated
    use, intrinsic :: iso_c_binding, only: c_bool
    use, intrinsic :: iso_c_binding, only: c_int, c_int32_t, c_int64_t
    use, intrinsic :: iso_c_binding, only: c_loc, c_null_ptr, c_ptr
    use liric_session_bindings, only: liric_session_t, lr_error_t, &
                                      lr_inst_desc_t, lr_operand_desc_t, &
                                      LR_OK, LR_OP_KIND_BLOCK, &
                                      LR_OP_KIND_VREG, liric_session_error_message
    implicit none
    private

    integer(c_int), parameter, public :: LR_CMP_EQ = 0_c_int
    integer(c_int), parameter, public :: LR_CMP_NE = 1_c_int
    integer(c_int), parameter, public :: LR_CMP_SGT = 2_c_int
    integer(c_int), parameter, public :: LR_CMP_SGE = 3_c_int
    integer(c_int), parameter, public :: LR_CMP_SLT = 4_c_int
    integer(c_int), parameter, public :: LR_CMP_SLE = 5_c_int

    integer(c_int), parameter :: LR_OP_CONDBR = 3_c_int
    integer(c_int), parameter :: LR_OP_ICMP = 24_c_int
    logical(c_bool), parameter :: c_false = .false.

    public :: create_liric_block
    public :: set_liric_block
    public :: emit_liric_i32_icmp
    public :: emit_liric_condbr

    interface
        function lr_type_i1_s(handle) result(typ) bind(c)
            import :: c_ptr
            type(c_ptr), value :: handle
            type(c_ptr) :: typ
        end function lr_type_i1_s

        function lr_session_block(handle) result(block_id) bind(c)
            import :: c_int32_t, c_ptr
            type(c_ptr), value :: handle
            integer(c_int32_t) :: block_id
        end function lr_session_block

        function lr_session_set_block(handle, block_id, err) result(status) &
            bind(c)
            import :: c_int, c_int32_t, c_ptr, lr_error_t
            type(c_ptr), value :: handle
            integer(c_int32_t), value :: block_id
            type(lr_error_t), intent(inout) :: err
            integer(c_int) :: status
        end function lr_session_set_block

        function lr_session_emit(handle, inst, err) result(vreg) bind(c)
            import :: c_int32_t, c_ptr, lr_error_t, lr_inst_desc_t
            type(c_ptr), value :: handle
            type(lr_inst_desc_t), intent(in) :: inst
            type(lr_error_t), intent(inout) :: err
            integer(c_int32_t) :: vreg
        end function lr_session_emit
    end interface

contains

    function create_liric_block(session) result(block_id)
        type(liric_session_t), intent(in) :: session
        integer(c_int32_t) :: block_id

        block_id = lr_session_block(session%handle)
    end function create_liric_block

    logical function set_liric_block(session, block_id, error_msg)
        type(liric_session_t), intent(inout) :: session
        integer(c_int32_t), intent(in) :: block_id
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_error_t) :: error
        integer(c_int) :: status

        set_liric_block = .false.
        if (.not. require_open_session(session, error_msg)) return

        call clear_liric_error(error)
        status = lr_session_set_block(session%handle, block_id, error)
        if (.not. status_ok(status, error, error_msg)) return

        call set_empty(error_msg)
        set_liric_block = .true.
    end function set_liric_block

    logical function emit_liric_i32_icmp(session, pred, lhs, rhs, result, &
                                         error_msg)
        type(liric_session_t), intent(inout) :: session
        integer(c_int), intent(in) :: pred
        type(lr_operand_desc_t), intent(in) :: lhs
        type(lr_operand_desc_t), intent(in) :: rhs
        type(lr_operand_desc_t), intent(out) :: result
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_error_t) :: error
        integer(c_int32_t) :: vreg

        emit_liric_i32_icmp = .false.
        if (.not. require_open_session(session, error_msg)) return

        vreg = emit_icmp(session%handle, pred, lhs, rhs, error)
        if (.not. status_ok(error%code, error, error_msg)) return

        result = i1_vreg(session, vreg)
        call set_empty(error_msg)
        emit_liric_i32_icmp = .true.
    end function emit_liric_i32_icmp

    logical function emit_liric_condbr(session, condition, true_block, &
                                       false_block, error_msg)
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(in) :: condition
        integer(c_int32_t), intent(in) :: true_block
        integer(c_int32_t), intent(in) :: false_block
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_error_t) :: error
        integer(c_int32_t) :: unused_vreg

        emit_liric_condbr = .false.
        if (.not. require_open_session(session, error_msg)) return

        unused_vreg = emit_condbr_inst(session%handle, condition, &
                                       block_operand(true_block), &
                                       block_operand(false_block), error)
        if (.not. status_ok(error%code, error, error_msg)) return

        call set_empty(error_msg)
        emit_liric_condbr = .true.
    end function emit_liric_condbr

    function i1_vreg(session, vreg) result(operand)
        type(liric_session_t), intent(in) :: session
        integer(c_int32_t), intent(in) :: vreg
        type(lr_operand_desc_t) :: operand

        operand%kind = LR_OP_KIND_VREG
        operand%payload = int(vreg, c_int64_t)
        operand%typ = lr_type_i1_s(session%handle)
        operand%global_offset = 0_c_int64_t
    end function i1_vreg

    function block_operand(block_id) result(operand)
        integer(c_int32_t), intent(in) :: block_id
        type(lr_operand_desc_t) :: operand

        operand%kind = LR_OP_KIND_BLOCK
        operand%payload = int(block_id, c_int64_t)
        operand%typ = c_null_ptr
        operand%global_offset = 0_c_int64_t
    end function block_operand

    function emit_icmp(handle, pred, lhs, rhs, error) result(vreg)
        type(c_ptr), intent(in) :: handle
        integer(c_int), intent(in) :: pred
        type(lr_operand_desc_t), intent(in) :: lhs
        type(lr_operand_desc_t), intent(in) :: rhs
        type(lr_error_t), intent(inout) :: error
        integer(c_int32_t) :: vreg
        type(lr_operand_desc_t), target :: operands(2)
        type(lr_inst_desc_t) :: inst

        operands = [lhs, rhs]

        call clear_inst(inst)
        inst%op = LR_OP_ICMP
        inst%operands = c_loc(operands)
        inst%num_operands = 2_c_int32_t
        inst%icmp_pred = pred

        call clear_liric_error(error)
        vreg = lr_session_emit(handle, inst, error)
    end function emit_icmp

    function emit_condbr_inst(handle, condition, true_target, false_target, &
                              error) result(vreg)
        type(c_ptr), intent(in) :: handle
        type(lr_operand_desc_t), intent(in) :: condition
        type(lr_operand_desc_t), intent(in) :: true_target
        type(lr_operand_desc_t), intent(in) :: false_target
        type(lr_error_t), intent(inout) :: error
        integer(c_int32_t) :: vreg
        type(lr_operand_desc_t), target :: operands(3)
        type(lr_inst_desc_t) :: inst

        operands = [condition, true_target, false_target]

        call clear_inst(inst)
        inst%op = LR_OP_CONDBR
        inst%operands = c_loc(operands)
        inst%num_operands = 3_c_int32_t

        call clear_liric_error(error)
        vreg = lr_session_emit(handle, inst, error)
    end function emit_condbr_inst

    subroutine clear_inst(inst)
        type(lr_inst_desc_t), intent(out) :: inst

        inst%op = 0_c_int
        inst%typ = c_null_ptr
        inst%dest = 0_c_int32_t
        inst%operands = c_null_ptr
        inst%num_operands = 0_c_int32_t
        inst%indices = c_null_ptr
        inst%num_indices = 0_c_int32_t
        inst%align = 0_c_int32_t
        inst%icmp_pred = 0_c_int
        inst%fcmp_pred = 0_c_int
        inst%call_external_abi = c_false
        inst%call_vararg = c_false
        inst%call_fixed_args = 0_c_int32_t
    end subroutine clear_inst

    logical function require_open_session(session, error_msg)
        type(liric_session_t), intent(in) :: session
        character(len=:), allocatable, intent(out) :: error_msg

        require_open_session = c_associated(session%handle)
        if (require_open_session) then
            call set_empty(error_msg)
        else
            error_msg = 'LIRIC session handle is not open'
        end if
    end function require_open_session

    logical function status_ok(status, error, error_msg)
        integer(c_int), intent(in) :: status
        type(lr_error_t), intent(in) :: error
        character(len=:), allocatable, intent(out) :: error_msg

        status_ok = status == LR_OK
        if (status_ok) then
            call set_empty(error_msg)
        else
            error_msg = liric_session_error_message(error)
        end if
    end function status_ok

    subroutine clear_liric_error(error)
        type(lr_error_t), intent(out) :: error

        error%code = LR_OK
        error%msg = char(0)
    end subroutine clear_liric_error

    subroutine set_empty(value)
        character(len=:), allocatable, intent(out) :: value

        allocate (character(len=0) :: value)
    end subroutine set_empty

end module liric_session_control_bindings
