module liric_session_common
    use, intrinsic :: iso_c_binding, only: c_associated, c_bool, c_char
    use, intrinsic :: iso_c_binding, only: c_int, c_int32_t, c_int64_t
    use, intrinsic :: iso_c_binding, only: c_null_char, c_null_ptr, c_ptr, &
        c_size_t
    implicit none
    private

    public :: lr_session_config_t, lr_error_t, lr_operand_desc_t, &
        lr_inst_desc_t, liric_session_t
    public :: LR_OK, LR_OP_KIND_VREG, LR_OP_KIND_IMM_I64, LR_OP_KIND_GLOBAL, &
        LR_OP_KIND_BLOCK
    public :: require_open_session, status_ok, liric_session_error_message, &
        clear_liric_error, to_c_chars, set_empty
    public :: lr_session_abi_info_t, lr_session_get_abi_info
    public :: FFC_EXPECTED_LIRIC_ABI_VERSION, FFC_EXPECTED_OPCODE_COUNT, &
        FFC_EXPECTED_OPERAND_KIND_COUNT
    public :: verify_liric_abi, set_liric_abi_override_for_testing, &
        clear_liric_abi_override_for_testing

    ! Published LIRIC session ABI this build of ffc mirrors (LIRIC #527).
    ! ffc refuses to emit instructions against a library that reports anything
    ! else, because a shifted layout would silently corrupt every descriptor
    ! passed across the ISO_C_BINDING boundary.
    integer(c_int), parameter :: FFC_EXPECTED_LIRIC_ABI_VERSION = 2_c_int
    integer(c_int), parameter :: FFC_EXPECTED_OPCODE_COUNT = 47_c_int
    integer(c_int), parameter :: FFC_EXPECTED_OPERAND_KIND_COUNT = 8_c_int

    integer(c_int), parameter :: LR_OK = 0_c_int
    integer(c_int), parameter :: LR_OP_RET = 0_c_int
    integer(c_int), parameter :: LR_OP_KIND_VREG = 0_c_int
    integer(c_int), parameter :: LR_OP_KIND_IMM_I64 = 1_c_int
    integer(c_int), parameter :: LR_OP_KIND_GLOBAL = 4_c_int
    integer(c_int), parameter :: LR_OP_KIND_BLOCK = 3_c_int

    type, bind(c), public :: lr_session_config_t
        integer(c_int) :: mode = 0_c_int
        type(c_ptr) :: target = c_null_ptr
        integer(c_int) :: backend = 0_c_int
        integer(c_int) :: opt_level = 0_c_int
    end type lr_session_config_t

    type, bind(c), public :: lr_error_t
        integer(c_int) :: code = LR_OK
        character(kind=c_char) :: msg(256)
    end type lr_error_t

    type, bind(c), public :: lr_operand_desc_t
        integer(c_int) :: kind = LR_OP_KIND_IMM_I64
        integer(c_int64_t) :: payload = 0_c_int64_t
        integer(c_int64_t) :: payload_hi = 0_c_int64_t
        type(c_ptr) :: typ = c_null_ptr
        integer(c_int64_t) :: global_offset = 0_c_int64_t
    end type lr_operand_desc_t

    type, bind(c), public :: lr_inst_desc_t
        integer(c_int) :: op = LR_OP_RET
        type(c_ptr) :: typ = c_null_ptr
        integer(c_int32_t) :: dest = 0_c_int32_t
        type(c_ptr) :: operands = c_null_ptr
        integer(c_int32_t) :: num_operands = 0_c_int32_t
        type(c_ptr) :: indices = c_null_ptr
        integer(c_int32_t) :: num_indices = 0_c_int32_t
        integer(c_int32_t) :: align = 0_c_int32_t
        integer(c_int) :: icmp_pred = 0_c_int
        integer(c_int) :: fcmp_pred = 0_c_int
        logical(c_bool) :: call_external_abi = .false.
        logical(c_bool) :: call_vararg = .false.
        integer(c_int32_t) :: call_fixed_args = 0_c_int32_t
    end type lr_inst_desc_t

    type, public :: liric_session_t
        type(c_ptr) :: handle = c_null_ptr
    end type liric_session_t

    ! Mirror of LIRIC's lr_session_abi_info_t (liric_session.h). Field order
    ! and types must track the C definition exactly; the verification below is
    ! what proves they still do.
    type, bind(c), public :: lr_session_abi_info_t
        integer(c_int32_t) :: abi_version = 0_c_int32_t
        integer(c_size_t) :: struct_size = 0_c_size_t
        integer(c_size_t) :: config_size = 0_c_size_t
        integer(c_size_t) :: error_size = 0_c_size_t
        integer(c_size_t) :: operand_size = 0_c_size_t
        integer(c_size_t) :: inst_size = 0_c_size_t
        integer(c_size_t) :: config_mode_offset = 0_c_size_t
        integer(c_size_t) :: config_target_offset = 0_c_size_t
        integer(c_size_t) :: config_backend_offset = 0_c_size_t
        integer(c_size_t) :: config_opt_level_offset = 0_c_size_t
        integer(c_size_t) :: error_code_offset = 0_c_size_t
        integer(c_size_t) :: error_msg_offset = 0_c_size_t
        integer(c_size_t) :: error_msg_size = 0_c_size_t
        integer(c_size_t) :: operand_kind_offset = 0_c_size_t
        integer(c_size_t) :: operand_type_offset = 0_c_size_t
        integer(c_size_t) :: operand_global_offset_offset = 0_c_size_t
        integer(c_size_t) :: inst_op_offset = 0_c_size_t
        integer(c_size_t) :: inst_type_offset = 0_c_size_t
        integer(c_size_t) :: inst_dest_offset = 0_c_size_t
        integer(c_size_t) :: inst_operands_offset = 0_c_size_t
        integer(c_size_t) :: inst_num_operands_offset = 0_c_size_t
        integer(c_int32_t) :: opcode_count = 0_c_int32_t
        integer(c_int32_t) :: operand_kind_count = 0_c_int32_t
    end type lr_session_abi_info_t

    ! Test seam: forces the next verification to see injected metadata, so the
    ! pre-emission refusal can be exercised without a rebuilt LIRIC.
    logical :: abi_override_active = .false.
    type(lr_session_abi_info_t) :: abi_override_info

    interface
        function lr_session_get_abi_info(info, info_size) result(status) &
            bind(c, name='lr_session_get_abi_info')
            import :: lr_session_abi_info_t, c_size_t, c_int
            type(lr_session_abi_info_t), intent(inout) :: info
            integer(c_size_t), value :: info_size
            integer(c_int) :: status
        end function lr_session_get_abi_info
    end interface

contains

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

    function liric_session_error_message(error) result(message)
        type(lr_error_t), intent(in) :: error
        character(len=:), allocatable :: message
        integer :: i
        integer :: message_len

        message_len = 0
        do i = 1, size(error%msg)
            if (error%msg(i) == c_null_char) exit
            message_len = message_len + 1
        end do

        if (message_len == 0) then
            allocate (character(len=32) :: message)
            write (message, '(A,I0)') 'LIRIC error code ', error%code
            return
        end if

        allocate (character(len=message_len) :: message)
        do i = 1, message_len
            message(i:i) = error%msg(i)
        end do
    end function liric_session_error_message

    subroutine clear_liric_error(error)
        type(lr_error_t), intent(out) :: error

        error%code = LR_OK
        error%msg = c_null_char
    end subroutine clear_liric_error

    subroutine to_c_chars(text, chars)
        character(len=*), intent(in) :: text
        character(kind=c_char), allocatable, intent(out) :: chars(:)
        integer :: i

        allocate (chars(len(text) + 1))
        do i = 1, len(text)
            chars(i) = text(i:i)
        end do
        chars(len(text) + 1) = c_null_char
    end subroutine to_c_chars

    subroutine set_empty(value)
        character(len=:), allocatable, intent(out) :: value

        allocate (character(len=0) :: value)
    end subroutine set_empty

    subroutine set_liric_abi_override_for_testing(info)
        type(lr_session_abi_info_t), intent(in) :: info

        abi_override_info = info
        abi_override_active = .true.
    end subroutine set_liric_abi_override_for_testing

    subroutine clear_liric_abi_override_for_testing()

        abi_override_active = .false.
    end subroutine clear_liric_abi_override_for_testing

    function size_mismatch(label, observed, expected) result(message)
        character(len=*), intent(in) :: label
        integer(c_size_t), intent(in) :: observed
        integer(c_size_t), intent(in) :: expected
        character(len=:), allocatable :: message
        character(len=32) :: got_text, want_text

        write (got_text, '(i0)') observed
        write (want_text, '(i0)') expected
        message = label//': expected '//trim(want_text)// &
            ', observed '//trim(got_text)
    end function size_mismatch

    function count_mismatch(label, observed, expected) result(message)
        character(len=*), intent(in) :: label
        integer(c_int32_t), intent(in) :: observed
        integer(c_int), intent(in) :: expected
        character(len=:), allocatable :: message
        character(len=32) :: got_text, want_text

        write (got_text, '(i0)') observed
        write (want_text, '(i0)') expected
        message = label//': expected '//trim(want_text)// &
            ', observed '//trim(got_text)
    end function count_mismatch

    ! Verifies ffc's mirrored LIRIC structures and opcode constants against the
    ! library's published session ABI. Returns .false. with a diagnostic that
    ! names both the expected and observed value on any mismatch; the caller
    ! must not emit a single instruction after that.
    logical function verify_liric_abi(error_msg) result(ok)
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_session_abi_info_t) :: info
        type(lr_session_config_t) :: config_probe
        type(lr_error_t) :: error_probe
        type(lr_operand_desc_t) :: operand_probe
        type(lr_inst_desc_t) :: inst_probe
        integer(c_int) :: status
        character(len=32) :: got_text, want_text

        ok = .false.
        call set_empty(error_msg)

        if (abi_override_active) then
            info = abi_override_info
            status = LR_OK
        else
            status = lr_session_get_abi_info(info, &
                                             int(storage_size(info)/8, &
                                                 c_size_t))
        end if

        if (status /= LR_OK) then
            write (got_text, '(i0)') status
            error_msg = 'LIRIC ABI query failed with status '//trim(got_text)// &
                '; refusing to emit instructions'
            return
        end if

        if (info%abi_version /= int(FFC_EXPECTED_LIRIC_ABI_VERSION, &
                                    c_int32_t)) then
            write (got_text, '(i0)') info%abi_version
            write (want_text, '(i0)') FFC_EXPECTED_LIRIC_ABI_VERSION
            error_msg = 'LIRIC session ABI version mismatch: expected '// &
                trim(want_text)//', observed '//trim(got_text)// &
                '; refusing to emit instructions'
            return
        end if

        ! Every structure ffc mirrors must have the same size the library
        ! reports, or an argument written on one side is read at the wrong
        ! offset on the other.
        if (info%config_size /= int(storage_size(config_probe)/8, c_size_t)) then
            error_msg = size_mismatch('LIRIC lr_session_config_t size', &
                                      info%config_size, &
                                      int(storage_size(config_probe)/8, &
                                          c_size_t))
            return
        end if
        if (info%error_size /= int(storage_size(error_probe)/8, c_size_t)) then
            error_msg = size_mismatch('LIRIC lr_error_t size', &
                                      info%error_size, &
                                      int(storage_size(error_probe)/8, &
                                          c_size_t))
            return
        end if
        if (info%operand_size /= int(storage_size(operand_probe)/8, &
                                     c_size_t)) then
            error_msg = size_mismatch('LIRIC lr_operand_desc_t size', &
                                      info%operand_size, &
                                      int(storage_size(operand_probe)/8, &
                                          c_size_t))
            return
        end if
        if (info%inst_size /= int(storage_size(inst_probe)/8, c_size_t)) then
            error_msg = size_mismatch('LIRIC lr_inst_desc_t size', &
                                      info%inst_size, &
                                      int(storage_size(inst_probe)/8, c_size_t))
            return
        end if

        ! Leading-field offsets must be zero on both sides, and the operand
        ! kind discriminator must be the first field ffc writes.
        if (info%config_mode_offset /= 0_c_size_t) then
            error_msg = size_mismatch('LIRIC config mode offset', &
                                      info%config_mode_offset, 0_c_size_t)
            return
        end if
        if (info%error_code_offset /= 0_c_size_t) then
            error_msg = size_mismatch('LIRIC error code offset', &
                                      info%error_code_offset, 0_c_size_t)
            return
        end if
        if (info%operand_kind_offset /= 0_c_size_t) then
            error_msg = size_mismatch('LIRIC operand kind offset', &
                                      info%operand_kind_offset, 0_c_size_t)
            return
        end if
        if (info%inst_op_offset /= 0_c_size_t) then
            error_msg = size_mismatch('LIRIC instruction op offset', &
                                      info%inst_op_offset, 0_c_size_t)
            return
        end if

        ! The opcode and operand-kind constants ffc hard-codes must still be
        ! inside the library's published range.
        if (info%opcode_count /= int(FFC_EXPECTED_OPCODE_COUNT, c_int32_t)) then
            error_msg = count_mismatch('LIRIC opcode count', &
                                       info%opcode_count, &
                                       FFC_EXPECTED_OPCODE_COUNT)
            return
        end if
        if (info%operand_kind_count /= int(FFC_EXPECTED_OPERAND_KIND_COUNT, &
                                           c_int32_t)) then
            error_msg = count_mismatch('LIRIC operand kind count', &
                                       info%operand_kind_count, &
                                       FFC_EXPECTED_OPERAND_KIND_COUNT)
            return
        end if
        if (LR_OP_KIND_GLOBAL >= int(info%operand_kind_count, c_int)) then
            error_msg = count_mismatch( &
                'LIRIC operand kind LR_OP_KIND_GLOBAL is out of range; count', &
                info%operand_kind_count, LR_OP_KIND_GLOBAL + 1_c_int)
            return
        end if

        call set_empty(error_msg)
        ok = .true.
    end function verify_liric_abi

end module liric_session_common
