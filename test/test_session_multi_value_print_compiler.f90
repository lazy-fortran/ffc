program test_session_multi_value_print_compiler
    use fortfront_compiler, only: compiler_frontend_options_t, &
        compiler_frontend_result_t, &
        compile_frontend_from_string, INPUT_MODE_STANDARD
    use session_program_lowering, only: lower_program_to_liric_exe
    implicit none

    type(compiler_frontend_options_t) :: options
    type(compiler_frontend_result_t) :: frontend_result
    character(len=:), allocatable :: error_msg
    character(len=256) :: output_line
    character(len=*), parameter :: exe_path = &
        '/tmp/ffc_session_multi_value_print_test'
    character(len=*), parameter :: out_path = &
        '/tmp/ffc_session_multi_value_print_test.out'
    character(len=*), parameter :: ref_path = &
        '/tmp/ffc_session_multi_value_print_test.ref'
    integer :: exit_stat
    integer :: io_stat
    integer :: unit
    character(len=*), parameter :: source = &
        'program main'//new_line('a')// &
        '  print *, 1, 2, 3'//new_line('a')// &
        '  print *, 7, "hello"'//new_line('a')// &
        'end program main'
    character(len=*), parameter :: src_path = &
        '/tmp/ffc_multi_value_print_src.f90'

    print *, '=== direct session multi-value print compiler test ==='

    options = compiler_frontend_options_t()
    options%run_semantics = .true.
    options%input_mode = INPUT_MODE_STANDARD

    call compile_frontend_from_string(source, frontend_result, options)
    if (.not. frontend_result%success()) then
        print *, 'FAIL: FortFront rejected source: ', &
            trim(frontend_result%diagnostic_text)
        stop 1
    end if

    call execute_command_line('rm -f '//exe_path//' '//out_path//' '//ref_path)
    call execute_command_line('rm -f '//src_path)
    open (newunit=unit, file=src_path, status='replace', action='write')
    write (unit, '(A)') source
    close (unit)

    call lower_program_to_liric_exe(frontend_result%arena, &
        frontend_result%root_index, exe_path, &
        error_msg)
    if (len_trim(error_msg) > 0) then
        print *, 'FAIL: direct LIRIC lowering failed: ', trim(error_msg)
        stop 1
    end if

    call execute_command_line(exe_path//' > '//out_path, exitstat=exit_stat)
    if (exit_stat /= 0) then
        print *, 'FAIL: executable exit status ', exit_stat
        call execute_command_line('rm -f '//exe_path//' '//out_path//' '//ref_path)
        stop 1
    end if

    call execute_command_line('gfortran -w '//src_path//' -o '//ref_path, &
        exitstat=exit_stat)
    if (exit_stat /= 0) then
        print *, 'FAIL: gfortran failed to compile ffc output'
        call execute_command_line('rm -f '//exe_path//' '//out_path//' '//ref_path//' '//src_path)
        stop 1
    end if
    call execute_command_line(ref_path//' > '//ref_path//'.out', exitstat=exit_stat)

    if (.not. outputs_match(out_path, ref_path//'.out')) then
        print *, 'FAIL: ffc output differs from gfortran'
        print *, '--- ffc output ---'
        call dump_file(out_path)
        print *, '--- gfortran output ---'
        call dump_file(ref_path//'.out')
        print *, '---'
        call execute_command_line('rm -f '//exe_path//' '//out_path//' '//ref_path)
        stop 1
    end if

    call execute_command_line('rm -f '//exe_path//' '//out_path//' '//ref_path//' '//src_path)

    print *, 'PASS: multi-value print matches gfortran byte-for-byte'

contains

    logical function files_equal(left, right)
        character(len=*), intent(in) :: left
        character(len=*), intent(in) :: right
        integer :: status

        call execute_command_line('diff -q '//left//' '//right// &
            ' > /dev/null 2>&1', exitstat=status)
        files_equal = status == 0
    end function files_equal

    logical function outputs_match(left, right)
        character(len=*), intent(in) :: left
        character(len=*), intent(in) :: right
        character(len=:), allocatable :: norm_left
        character(len=:), allocatable :: norm_right
        integer :: status

        outputs_match = .false.

        call normalize_file(left, norm_left, .true.)
        call normalize_file(right, norm_right, .false.)

        call execute_command_line('diff -q '//norm_left//' '//norm_right// &
            ' > /dev/null 2>&1', exitstat=status)
        outputs_match = status == 0

        call execute_command_line('rm -f '//norm_left//' '//norm_right)
    end function outputs_match

    subroutine normalize_file(in_path, out_path, is_first)
        character(len=*), intent(in) :: in_path
        character(len=:), allocatable, intent(out) :: out_path
        logical, intent(in) :: is_first
        integer :: io_stat
        character(len=256) :: cmd
        character(len=64) :: norm_path

        if (is_first) then
            norm_path = '/tmp/ffc_norm_a'
        else
            norm_path = '/tmp/ffc_norm_b'
        end if

        cmd = 'sed "s/^[[:space:]]*//;s/[[:space:]]*$//;s/[[:space:]]\{1,\}/ /g" '// &
            in_path//' > '//norm_path//' 2>/dev/null'
        call execute_command_line(cmd, exitstat=io_stat)
        allocate (character(len=len_trim(norm_path)) :: out_path)
        out_path = trim(norm_path)
    end subroutine normalize_file

    subroutine dump_file(path)
        character(len=*), intent(in) :: path
        integer :: unit, io_stat

        open (newunit=unit, file=path, status='old', action='read', &
            iostat=io_stat)
        if (io_stat /= 0) return
        do
            read (unit, '(A)', iostat=io_stat) output_line
            if (io_stat /= 0) exit
            print '(A)', trim(output_line)
        end do
        close (unit)
    end subroutine dump_file

end program test_session_multi_value_print_compiler
