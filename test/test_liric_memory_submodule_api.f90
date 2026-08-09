program test_liric_memory_submodule_api
    use, intrinsic :: iso_c_binding, only: c_associated, c_int64_t
    use liric_session_bindings, only: destroy, liric_session_create, &
        liric_session_t, lr_operand_desc_t
    use liric_session_memory_bindings, only: i8_immediate, i16_immediate
    implicit none

    type(liric_session_t) :: session
    type(lr_operand_desc_t) :: i8_value, i16_value
    character(len=:), allocatable :: error_msg

    call liric_session_create(session, error_msg)
    if (allocated(error_msg)) then
        if (len_trim(error_msg) > 0) then
            write (*, '(a)') 'FAIL: '//trim(error_msg)
            stop 1
        end if
    end if

    i8_value = i8_immediate(session, 42_c_int64_t)
    i16_value = i16_immediate(session, 1234_c_int64_t)
    if (i8_value%payload /= 42_c_int64_t .or. &
            i16_value%payload /= 1234_c_int64_t .or. &
            .not. c_associated(i8_value%typ) .or. &
            .not. c_associated(i16_value%typ)) then
        call destroy(session)
        write (*, '(a)') 'FAIL: narrow integer operands disagree with LIRIC API'
        stop 1
    end if

    call destroy(session)
    write (*, '(a)') 'PASS: NVHPC narrow-integer submodule API'
end program test_liric_memory_submodule_api
