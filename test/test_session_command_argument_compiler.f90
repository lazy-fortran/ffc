program test_session_command_argument_compiler
    ! command_argument_count() and get_command_argument(i, value). The
    ! compiled binaries are invoked with arguments to check the runtime
    ! behaviour against gfortran.
    use fortfront_compiler, only: compiler_frontend_options_t, &
                                  compiler_frontend_result_t, &
                                  compile_frontend_from_string, INPUT_MODE_STANDARD
    use session_program_lowering, only: lower_program_to_liric_exe
    implicit none

    logical :: all_passed

    print *, '=== direct session command argument compiler test ==='

    all_passed = .true.
    if (.not. test_count_no_args()) all_passed = .false.
    if (.not. test_count_three_args()) all_passed = .false.
    if (.not. test_get_first_argument()) all_passed = .false.
    if (.not. test_get_argument_blank_padded()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: command-line argument intrinsics lower through direct LIRIC'

contains

    logical function build(source, exe)
        character(len=*), intent(in) :: source
        character(len=*), intent(in) :: exe
        type(compiler_frontend_options_t) :: options
        type(compiler_frontend_result_t) :: frontend_result
        character(len=:), allocatable :: error_msg

        build = .false.
        options = compiler_frontend_options_t()
        options%run_semantics = .true.
        options%input_mode = INPUT_MODE_STANDARD
        call compile_frontend_from_string(source, frontend_result, options)
        if (.not. frontend_result%success()) then
            print *, 'FAIL: FortFront rejected source: ', &
                trim(frontend_result%diagnostic_text)
            return
        end if
        call lower_program_to_liric_exe(frontend_result%arena, &
                                        frontend_result%root_index, exe, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL: ffc lowering failed: ', trim(error_msg)
            return
        end if
        build = .true.
    end function build

    logical function test_count_no_args()
        character(len=*), parameter :: exe = '/tmp/ffc_argc_none'
        integer :: status
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  stop command_argument_count()'//new_line('a')// &
            'end program main'

        test_count_no_args = .false.
        if (.not. build(source, exe)) return
        call execute_command_line(exe, exitstat=status)
        if (status /= 0) then
            print *, 'FAIL: count without args expected 0, got ', status
            return
        end if
        test_count_no_args = .true.
    end function test_count_no_args

    logical function test_count_three_args()
        character(len=*), parameter :: exe = '/tmp/ffc_argc_three'
        integer :: status
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  stop command_argument_count()'//new_line('a')// &
            'end program main'

        test_count_three_args = .false.
        if (.not. build(source, exe)) return
        call execute_command_line(exe//' a b c', exitstat=status)
        if (status /= 3) then
            print *, 'FAIL: count with three args expected 3, got ', status
            return
        end if
        test_count_three_args = .true.
    end function test_count_three_args

    logical function test_get_first_argument()
        character(len=*), parameter :: exe = '/tmp/ffc_getarg_first'
        character(len=*), parameter :: out = '/tmp/ffc_getarg_first.out'
        integer :: status
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=32) :: arg'//new_line('a')// &
            '  call get_command_argument(1, arg)'//new_line('a')// &
            "  print '(A)', trim(arg)"//new_line('a')// &
            'end program main'

        test_get_first_argument = .false.
        if (.not. build(source, exe)) return
        call execute_command_line(exe//' hello world > '//out, exitstat=status)
        test_get_first_argument = file_first_line_is(out, 'hello')
        if (.not. test_get_first_argument) &
            print *, 'FAIL: get_command_argument(1) did not yield "hello"'
        call execute_command_line('rm -f '//out)
    end function test_get_first_argument

    logical function test_get_argument_blank_padded()
        character(len=*), parameter :: exe = '/tmp/ffc_getarg_pad'
        character(len=*), parameter :: out = '/tmp/ffc_getarg_pad.out'
        integer :: status
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=8) :: arg'//new_line('a')// &
            '  call get_command_argument(1, arg)'//new_line('a')// &
            "  print '(A)', arg"//new_line('a')// &
            'end program main'

        test_get_argument_blank_padded = .false.
        if (.not. build(source, exe)) return
        call execute_command_line(exe//' hi > '//out, exitstat=status)
        ! len=8 "hi" -> "hi      " (six trailing blanks)
        test_get_argument_blank_padded = file_first_line_is(out, 'hi      ')
        if (.not. test_get_argument_blank_padded) &
            print *, 'FAIL: get_command_argument did not blank-pad to length 8'
        call execute_command_line('rm -f '//out)
    end function test_get_argument_blank_padded

    logical function file_first_line_is(path, expected)
        character(len=*), intent(in) :: path
        character(len=*), intent(in) :: expected
        integer :: unit, io_stat
        character(len=256) :: line

        file_first_line_is = .false.
        open (newunit=unit, file=path, status='old', action='read', &
              iostat=io_stat)
        if (io_stat /= 0) return
        read (unit, '(A)', iostat=io_stat) line
        close (unit)
        if (io_stat /= 0) return
        file_first_line_is = line == expected
    end function file_first_line_is

end program test_session_command_argument_compiler
