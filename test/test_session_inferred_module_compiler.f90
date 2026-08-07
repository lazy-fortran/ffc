program test_session_inferred_module_compiler
    ! The inferred-symbol pass is a typed descendant of the lowering module.
    ! Keep an accepted Lazy fragment against gfortran so the extraction cannot
    ! silently change implicit integer binding or generated output.
    use fortfront_compiler, only: compiler_frontend_options_t, &
        compiler_frontend_result_t, compile_frontend_from_string, INPUT_MODE_LAZY
    use session_program_lowering, only: lower_program_to_liric_exe
    implicit none

    type(compiler_frontend_options_t) :: options
    type(compiler_frontend_result_t) :: frontend_result
    character(len=:), allocatable :: error_msg
    character(len=:), allocatable :: source, base, src, exe, reference
    character(len=:), allocatable :: ffc_out, reference_out
    integer :: unit, exit_stat, diff_status

    print *, '=== inferred-symbol module compiler test ==='
    source = 'program main'//new_line('a')// &
        '  i = 17'//new_line('a')// &
        '  j = 25'//new_line('a')// &
        '  k = i + j'//new_line('a')// &
        '  print *, k'//new_line('a')// &
        'end program main'
    base = '/var/tmp/ert/ffc_inferred_module_oracle'
    src = base//'.f90'
    exe = base//'.ffc'
    reference = base//'.gfortran'
    ffc_out = base//'.ffc.out'
    reference_out = base//'.gfortran.out'

    call execute_command_line('mkdir -p /var/tmp/ert')
    call execute_command_line('rm -f '//src//' '//exe//' '//reference//' '// &
        ffc_out//' '//reference_out)
    options = compiler_frontend_options_t()
    options%run_semantics = .true.
    options%input_mode = INPUT_MODE_LAZY
    call compile_frontend_from_string(source, frontend_result, options)
    if (.not. frontend_result%success()) then
        print *, 'FAIL: FortFront rejected inferred source: ', &
            trim(frontend_result%diagnostic_text)
        stop 1
    end if
    call lower_program_to_liric_exe(frontend_result%arena, &
        frontend_result%root_index, exe, error_msg)
    if (len_trim(error_msg) > 0) then
        print *, 'FAIL: ffc inferred lowering failed: ', trim(error_msg)
        stop 1
    end if

    open (newunit=unit, file=src, status='replace', action='write')
    write (unit, '(A)') source
    close (unit)
    call execute_command_line('gfortran -w '//src//' -o '//reference, &
        exitstat=exit_stat)
    if (exit_stat /= 0) then
        print *, 'FAIL: gfortran rejected inferred source'
        stop 1
    end if
    call execute_command_line(exe//' > '//ffc_out, exitstat=exit_stat)
    if (exit_stat /= 0) then
        print *, 'FAIL: ffc inferred executable failed'
        stop 1
    end if
    call execute_command_line(reference//' > '//reference_out, &
        exitstat=exit_stat)
    if (exit_stat /= 0) then
        print *, 'FAIL: gfortran inferred executable failed'
        stop 1
    end if
    call execute_command_line('diff '//ffc_out//' '//reference_out// &
        ' > /dev/null 2>&1', exitstat=diff_status)
    if (diff_status /= 0) then
        print *, 'FAIL: inferred output differs from gfortran'
        call execute_command_line('diff '//ffc_out//' '//reference_out)
        stop 1
    end if
    call execute_command_line('rm -f '//src//' '//exe//' '//reference//' '// &
        ffc_out//' '//reference_out)
    print *, 'PASS: inferred-symbol module matches gfortran'
end program test_session_inferred_module_compiler
