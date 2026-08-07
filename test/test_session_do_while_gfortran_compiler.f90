program test_session_do_while_gfortran_compiler
    ! Independent oracle for the typed DO WHILE descendant: compile the same
    ! source with gfortran and require byte-identical list-directed output.
    use fortfront_compiler, only: compiler_frontend_options_t, &
        compiler_frontend_result_t, compile_frontend_from_string, &
        INPUT_MODE_STANDARD
    use session_program_lowering, only: lower_program_to_liric_exe
    implicit none

    type(compiler_frontend_options_t) :: options
    type(compiler_frontend_result_t) :: frontend_result
    character(len=:), allocatable :: source, base, src, ffc_exe, gfortran_exe
    character(len=:), allocatable :: ffc_out, gfortran_out, error_msg
    integer :: unit, exit_stat, diff_status

    print *, '=== do while module gfortran oracle test ==='
    source = 'program main'//new_line('a')// &
        'integer :: i, total'//new_line('a')// &
        'i = 0'//new_line('a')// &
        'total = 0'//new_line('a')// &
        'do while (i < 4)'//new_line('a')// &
        '    i = i + 1'//new_line('a')// &
        '    if (i == 2) cycle'//new_line('a')// &
        '    total = total + i'//new_line('a')// &
        'end do'//new_line('a')// &
        'print *, i, total'//new_line('a')// &
        'end program main'
    base = '/var/tmp/ert/ffc_do_while_module_oracle'
    src = base//'.f90'
    ffc_exe = base//'.ffc'
    gfortran_exe = base//'.gfortran'
    ffc_out = base//'.ffc.out'
    gfortran_out = base//'.gfortran.out'

    call execute_command_line('mkdir -p /var/tmp/ert')
    call execute_command_line('rm -f '//src//' '//ffc_exe//' '//gfortran_exe// &
        ' '//ffc_out//' '//gfortran_out)
    options = compiler_frontend_options_t()
    options%run_semantics = .true.
    options%input_mode = INPUT_MODE_STANDARD
    call compile_frontend_from_string(source, frontend_result, options)
    if (.not. frontend_result%success()) then
        print *, 'FAIL: FortFront rejected DO WHILE oracle source: ', &
            trim(frontend_result%diagnostic_text)
        stop 1
    end if
    call lower_program_to_liric_exe(frontend_result%arena, &
        frontend_result%root_index, ffc_exe, error_msg)
    if (len_trim(error_msg) > 0) then
        print *, 'FAIL: ffc DO WHILE lowering failed: ', trim(error_msg)
        stop 1
    end if

    open (newunit=unit, file=src, status='replace', action='write')
    write (unit, '(A)') source
    close (unit)
    call execute_command_line('gfortran -w '//src//' -o '//gfortran_exe, &
        exitstat=exit_stat)
    if (exit_stat /= 0) then
        print *, 'FAIL: gfortran rejected DO WHILE oracle source'
        stop 1
    end if
    call execute_command_line(ffc_exe//' > '//ffc_out, exitstat=exit_stat)
    if (exit_stat /= 0) then
        print *, 'FAIL: ffc DO WHILE oracle executable failed'
        stop 1
    end if
    call execute_command_line(gfortran_exe//' > '//gfortran_out, &
        exitstat=exit_stat)
    if (exit_stat /= 0) then
        print *, 'FAIL: gfortran DO WHILE oracle executable failed'
        stop 1
    end if
    call execute_command_line('diff '//ffc_out//' '//gfortran_out// &
        ' > /dev/null 2>&1', exitstat=diff_status)
    if (diff_status /= 0) then
        print *, 'FAIL: typed DO WHILE output differs from gfortran'
        call execute_command_line('diff '//ffc_out//' '//gfortran_out)
        stop 1
    end if
    call execute_command_line('rm -f '//src//' '//ffc_exe//' '//gfortran_exe// &
        ' '//ffc_out//' '//gfortran_out)
    print *, 'PASS: typed DO WHILE module matches gfortran'
end program test_session_do_while_gfortran_compiler
