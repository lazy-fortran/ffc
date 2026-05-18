program test_session_multi_value_print_compiler
    use fortfront, only: compiler_frontend_options_t, &
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
        character(len=256) :: left_line, right_line
        integer :: unit_l, unit_r, io_stat_l, io_stat_r
        logical :: done_l, done_r
        character(len=256) :: norm_left, norm_right
        integer :: ll, lr, li, lc

        outputs_match = .false.
        open (newunit=unit_l, file=left, status='old', action='read', &
              iostat=io_stat_l)
        if (io_stat_l /= 0) return
        open (newunit=unit_r, file=right, status='old', action='read', &
              iostat=io_stat_r)
        if (io_stat_r /= 0) then
            close (unit_l)
            return
        end if

        done_l = .true.
        done_r = .true.
        do
            read (unit_l, '(A)', iostat=io_stat_l) left_line
            if (io_stat_l /= 0) then
                done_l = .true.
            else
                done_l = .false.
            end if
            read (unit_r, '(A)', iostat=io_stat_r) right_line
            if (io_stat_r /= 0) then
                done_r = .true.
            else
                done_r = .false.
            end if
            if (done_l .and. done_r) then
                outputs_match = .true.
                exit
            end if
            if (done_l .neqv. done_r) exit

            norm_left = normalize_ws(trim(left_line))
            norm_right = normalize_ws(trim(right_line))
            if (norm_left /= norm_right) exit
        end do

        close (unit_l)
        close (unit_r)
    end function outputs_match

    function normalize_ws(text) result(normalized)
        character(len=*), intent(in) :: text
        character(len=:), allocatable :: normalized
        character(len=256) :: buf
        integer :: li, lc, len_text

        len_text = len_trim(text)
        if (len_text == 0) then
            allocate (character(len=0) :: normalized)
            return
        end if

        buf = ''
        lc = 0
        do li = 1, len_text
            if (text(li:li) == ' ') then
                if (lc > 0) then
                    lc = lc + 1
                    buf(lc:lc) = ' '
                end if
            else
                lc = lc + 1
                buf(lc:lc) = text(li:li)
            end if
        end do
        allocate (character(len=lc) :: normalized)
        normalized = buf(1:lc)
    end function normalize_ws

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
