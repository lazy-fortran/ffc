program test_self_concat_appends_literal
    use fortfront, only: compiler_frontend_options_t, &
                         compiler_frontend_result_t, &
                         compile_frontend_from_string, INPUT_MODE_STANDARD
    use session_program_lowering, only: lower_program_to_liric_exe
    implicit none

    logical :: all_passed

    print *, '=== self-aliasing concat test ==='

    all_passed = .true.
    if (.not. test_self_concat_appends_literal_case()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: self-aliasing deferred char concat'

contains

    logical function test_self_concat_appends_literal_case()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character :: s'//new_line('a')// &
            '  s = "hi"'//new_line('a')// &
            '  s = s // "!"'//new_line('a')// &
            '  print *, s'//new_line('a')// &
            'end program main'

        test_self_concat_appends_literal_case = expect_output( &
            source, 'hi!'//new_line('a'), &
            '/tmp/ffc_self_concat_literal_test')
    end function test_self_concat_appends_literal_case

    logical function expect_output(source, expected, stem)
        character(len=*), intent(in) :: source
        character(len=*), intent(in) :: expected
        character(len=*), intent(in) :: stem
        character(len=:), allocatable :: error_msg
        character(len=:), allocatable :: exe_path
        character(len=:), allocatable :: output_text
        character(len=:), allocatable :: out_path
        integer :: exit_stat
        integer :: file_size
        integer :: io_stat
        integer :: unit

        expect_output = .false.
        exe_path = stem
        out_path = stem//'.out'
        call execute_command_line('rm -f '//exe_path//' '//out_path)
        call compile_and_lower(source, exe_path, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL: direct LIRIC lowering failed: ', trim(error_msg)
            return
        end if

        call execute_command_line(exe_path//' > '//out_path, exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL: executable exit status ', exit_stat
            call execute_command_line('rm -f '//exe_path//' '//out_path)
            return
        end if

        inquire (file=out_path, size=file_size)
        if (file_size /= len(expected)) then
            print *, 'FAIL: expected output size ', len(expected), &
                ', got ', file_size
            call execute_command_line('rm -f '//exe_path//' '//out_path)
            return
        end if

        allocate (character(len=file_size) :: output_text)
        open (newunit=unit, file=out_path, status='old', access='stream', &
              form='unformatted', action='read', iostat=io_stat)
        if (io_stat /= 0) then
            print *, 'FAIL: could not open captured output as stream'
            call execute_command_line('rm -f '//exe_path//' '//out_path)
            return
        end if
        read (unit, iostat=io_stat) output_text
        close (unit)
        call execute_command_line('rm -f '//exe_path//' '//out_path)

        if (io_stat /= 0 .or. output_text /= expected) then
            print *, 'FAIL: unexpected character output bytes'
            return
        end if

        expect_output = .true.
    end function expect_output

    subroutine compile_and_lower(source, exe_path, error_msg)
        character(len=*), intent(in) :: source
        character(len=*), intent(in) :: exe_path
        character(len=:), allocatable, intent(out) :: error_msg
        type(compiler_frontend_options_t) :: options
        type(compiler_frontend_result_t) :: frontend_result

        options = compiler_frontend_options_t()
        options%run_semantics = .true.
        options%input_mode = INPUT_MODE_STANDARD

        call compile_frontend_from_string(source, frontend_result, options)
        if (.not. frontend_result%success()) then
            error_msg = 'FortFront rejected source: '// &
                        trim(frontend_result%diagnostic_text)
            return
        end if

        call lower_program_to_liric_exe(frontend_result%arena, &
                                        frontend_result%root_index, exe_path, &
                                        error_msg)
    end subroutine compile_and_lower
end program test_self_concat_appends_literal
