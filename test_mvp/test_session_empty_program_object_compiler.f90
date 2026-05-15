program test_session_empty_program_object_compiler
    use fortfront, only: compiler_frontend_options_t, &
                         compiler_frontend_result_t, &
                         compile_frontend_from_string, INPUT_MODE_STANDARD
    use session_program_lowering, only: lower_program_to_liric_object
    implicit none

    type(compiler_frontend_options_t) :: options
    type(compiler_frontend_result_t) :: frontend_result
    character(len=:), allocatable :: error_msg
    character(len=*), parameter :: object_path = &
        '/tmp/ffc_session_empty_program_test.o'
    integer :: object_size
    logical :: object_exists
    character(len=*), parameter :: source = &
        'program main'//new_line('a')// &
        'end program main'

    print *, '=== direct session empty program object compiler test ==='

    options = compiler_frontend_options_t()
    options%run_semantics = .true.
    options%input_mode = INPUT_MODE_STANDARD

    call compile_frontend_from_string(source, frontend_result, options)
    if (.not. frontend_result%success()) then
        print *, 'FAIL: FortFront rejected source: ', &
            trim(frontend_result%diagnostic_text)
        stop 1
    end if

    call execute_command_line('rm -f '//object_path)
    call lower_program_to_liric_object(frontend_result%arena, &
                                       frontend_result%root_index, &
                                       object_path, error_msg)
    if (len_trim(error_msg) > 0) then
        print *, 'FAIL: direct LIRIC object lowering failed: ', trim(error_msg)
        stop 1
    end if

    inquire (file=object_path, exist=object_exists, size=object_size)
    call execute_command_line('rm -f '//object_path)

    if (.not. object_exists .or. object_size <= 0) then
        print *, 'FAIL: expected non-empty object file'
        stop 1
    end if

    print *, 'PASS: empty program emits object through direct LIRIC session'
end program test_session_empty_program_object_compiler
