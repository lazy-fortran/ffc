program test_session_if_merge_compiler
    use fortfront, only: compiler_frontend_options_t, &
                         compiler_frontend_result_t, &
                         compile_frontend_from_string, INPUT_MODE_STANDARD
    use session_program_lowering, only: lower_program_to_liric_exe
    implicit none

    logical :: all_passed

    print *, '=== direct session if merge compiler test ==='

    all_passed = .true.
    if (.not. test_integer_if_merge()) all_passed = .false.
    if (.not. test_real_if_merge()) all_passed = .false.
    if (.not. test_logical_if_merge()) all_passed = .false.
    if (.not. test_nested_if_in_do_merge()) all_passed = .false.
    if (.not. test_do_in_if_merge()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: fallthrough IF merges scalar values through direct LIRIC'

contains

    logical function test_integer_if_merge()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       'integer :: x'//new_line('a')// &
                                       'if (2 < 3) then'//new_line('a')// &
                                       '    x = 9'//new_line('a')// &
                                       'else'//new_line('a')// &
                                       '    x = 4'//new_line('a')// &
                                       'end if'//new_line('a')// &
                                       'stop x'//new_line('a')// &
                                       'end program main'

        test_integer_if_merge = compile_and_expect_exit( &
                                source, &
                                '/tmp/ffc_session_if_integer_merge_test', 9)
    end function test_integer_if_merge

    logical function test_real_if_merge()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       'real :: x'//new_line('a')// &
                                       'if (2 < 3) then'//new_line('a')// &
                                       '    x = 4.5'//new_line('a')// &
                                       'else'//new_line('a')// &
                                       '    x = 1.25'//new_line('a')// &
                                       'end if'//new_line('a')// &
                                       'print *, x'//new_line('a')// &
                                       'end program main'

        test_real_if_merge = compile_and_expect_output( &
                             source, '/tmp/ffc_session_if_real_merge_test', &
                             '4.500000')
    end function test_real_if_merge

    logical function test_logical_if_merge()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       'logical :: flag'//new_line('a')// &
                                       'if (2 < 3) then'//new_line('a')// &
                                       '    flag = .true.'//new_line('a')// &
                                       'else'//new_line('a')// &
                                       '    flag = .false.'//new_line('a')// &
                                       'end if'//new_line('a')// &
                                       'if (flag) then'//new_line('a')// &
                                       '    stop 7'//new_line('a')// &
                                       'else'//new_line('a')// &
                                       '    stop 3'//new_line('a')// &
                                       'end if'//new_line('a')// &
                                       'end program main'

        test_logical_if_merge = compile_and_expect_exit( &
                                source, &
                                '/tmp/ffc_session_if_logical_merge_test', 7)
    end function test_logical_if_merge

    logical function test_nested_if_in_do_merge()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       'integer :: i'//new_line('a')// &
                                       'integer :: total'//new_line('a')// &
                                       'logical :: found'//new_line('a')// &
                                       'total = 0'//new_line('a')// &
                                       'found = .false.'//new_line('a')// &
                                       'do i = 1, 3'//new_line('a')// &
                                       '    if (i < 3) then'//new_line('a')// &
                                       '        total = total + i'//new_line('a')// &
                                       '        found = .true.'//new_line('a')// &
                                       '    else'//new_line('a')// &
                                       '        total = total + 4'//new_line('a')// &
                                       '        found = found'//new_line('a')// &
                                       '    end if'//new_line('a')// &
                                       'end do'//new_line('a')// &
                                       'if (found) then'//new_line('a')// &
                                       '    stop total'//new_line('a')// &
                                       'else'//new_line('a')// &
                                       '    stop 1'//new_line('a')// &
                                       'end if'//new_line('a')// &
                                       'end program main'

        test_nested_if_in_do_merge = compile_and_expect_exit( &
                                     source, &
                                     '/tmp/ffc_session_nested_if_do_test', 7)
    end function test_nested_if_in_do_merge

    logical function test_do_in_if_merge()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       'integer :: i'//new_line('a')// &
                                       'integer :: total'//new_line('a')// &
                                       'logical :: flag'//new_line('a')// &
                                       'total = 0'//new_line('a')// &
                                       'flag = .false.'//new_line('a')// &
                                       'if (2 < 3) then'//new_line('a')// &
                                       '    do i = 1, 2'//new_line('a')// &
                                       '        total = total + i'//new_line('a')// &
                                       '        flag = .true.'//new_line('a')// &
                                       '    end do'//new_line('a')// &
                                       'else'//new_line('a')// &
                                       '    total = 9'//new_line('a')// &
                                       '    flag = .false.'//new_line('a')// &
                                       'end if'//new_line('a')// &
                                       'if (flag) then'//new_line('a')// &
                                       '    stop total'//new_line('a')// &
                                       'else'//new_line('a')// &
                                       '    stop 8'//new_line('a')// &
                                       'end if'//new_line('a')// &
                                       'end program main'

        test_do_in_if_merge = compile_and_expect_exit( &
                              source, '/tmp/ffc_session_do_if_test', 3)
    end function test_do_in_if_merge

    logical function compile_and_expect_exit(source, exe_path, expected_status)
        character(len=*), intent(in) :: source
        character(len=*), intent(in) :: exe_path
        integer, intent(in) :: expected_status
        integer :: cmd_stat
        integer :: exit_stat

        compile_and_expect_exit = .false.
        if (.not. compile_source(source, exe_path)) return

        call execute_command_line(exe_path, exitstat=exit_stat, &
                                  cmdstat=cmd_stat)
        call execute_command_line('rm -f '//exe_path)
        if (cmd_stat /= 0) then
            print *, 'FAIL: emitted executable did not run'
            return
        end if
        if (exit_stat /= expected_status) then
            print *, 'FAIL: executable exit status ', exit_stat
            return
        end if

        compile_and_expect_exit = .true.
    end function compile_and_expect_exit

    logical function compile_and_expect_output(source, exe_path, expected_output)
        character(len=*), intent(in) :: source
        character(len=*), intent(in) :: exe_path
        character(len=*), intent(in) :: expected_output
        character(len=:), allocatable :: out_path
        character(len=64) :: output_line
        integer :: exit_stat
        integer :: io_stat
        integer :: unit

        compile_and_expect_output = .false.
        out_path = exe_path//'.out'
        if (.not. compile_source(source, exe_path)) return

        call execute_command_line(exe_path//' > '//out_path, exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL: executable exit status ', exit_stat
            call execute_command_line('rm -f '//exe_path//' '//out_path)
            return
        end if

        open (newunit=unit, file=out_path, status='old', action='read', &
              iostat=io_stat)
        if (io_stat == 0) read (unit, '(A)', iostat=io_stat) output_line
        if (io_stat == 0) close (unit)
        call execute_command_line('rm -f '//exe_path//' '//out_path)

        if (io_stat /= 0) then
            print *, 'FAIL: could not read executable output'
            return
        end if
        if (trim(adjustl(output_line)) /= expected_output) then
            print *, 'FAIL: expected output ', expected_output, ', got ', &
                trim(output_line)
            return
        end if

        compile_and_expect_output = .true.
    end function compile_and_expect_output

    logical function compile_source(source, exe_path)
        character(len=*), intent(in) :: source
        character(len=*), intent(in) :: exe_path
        type(compiler_frontend_options_t) :: options
        type(compiler_frontend_result_t) :: frontend_result
        character(len=:), allocatable :: error_msg

        compile_source = .false.
        options = compiler_frontend_options_t()
        options%run_semantics = .true.
        options%input_mode = INPUT_MODE_STANDARD

        call compile_frontend_from_string(source, frontend_result, options)
        if (.not. frontend_result%success()) then
            print *, 'FAIL: FortFront rejected source: ', &
                trim(frontend_result%diagnostic_text)
            return
        end if

        call execute_command_line('rm -f '//exe_path)
        call lower_program_to_liric_exe(frontend_result%arena, &
                                        frontend_result%root_index, exe_path, &
                                        error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL: direct LIRIC lowering failed: ', trim(error_msg)
            return
        end if

        compile_source = .true.
    end function compile_source
end program test_session_if_merge_compiler
