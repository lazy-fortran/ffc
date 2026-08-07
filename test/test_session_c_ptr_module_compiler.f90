program test_session_c_ptr_module_compiler
    ! The ISO_C_BINDING lowering now lives in a typed descendant.  Compare one
    ! scalar/array pointer round-trip with gfortran so the move cannot silently
    ! change address, association, or shape semantics.
    use fortfront_compiler, only: compiler_frontend_options_t, &
        compiler_frontend_result_t, compile_frontend_from_string, &
        INPUT_MODE_STANDARD
    use session_program_lowering, only: lower_program_to_liric_exe
    implicit none

    type(compiler_frontend_options_t) :: options
    type(compiler_frontend_result_t) :: frontend_result
    character(len=:), allocatable :: source, base, src, ffc_exe, ref_exe
    character(len=:), allocatable :: ffc_out, ref_out, error_msg
    integer :: unit, exit_stat, diff_status

    print *, '=== c_ptr module compiler test ==='
    source = 'program main'//new_line('a')// &
        '  use, intrinsic :: iso_c_binding, only: c_ptr, c_loc, '// &
        'c_f_pointer, c_associated, c_int'//new_line('a')// &
        '  integer(c_int), target :: a(2)'//new_line('a')// &
        '  integer(c_int), pointer :: p(:)'//new_line('a')// &
        '  type(c_ptr) :: cp'//new_line('a')// &
        '  a(1) = 7'//new_line('a')// &
        '  a(2) = 9'//new_line('a')// &
        '  cp = c_loc(a)'//new_line('a')// &
        '  if (.not. c_associated(cp)) stop 1'//new_line('a')// &
        '  call c_f_pointer(cp, p, [2])'//new_line('a')// &
        '  print *, p(1), p(2), size(p), lbound(p,1), ubound(p,1)'// &
        new_line('a')// &
        'end program main'
    base = '/var/tmp/ert/ffc_c_ptr_module_oracle'
    src = base//'.f90'
    ffc_exe = base//'.ffc'
    ref_exe = base//'.gfortran'
    ffc_out = base//'.ffc.out'
    ref_out = base//'.gfortran.out'

    call execute_command_line('mkdir -p /var/tmp/ert')
    call execute_command_line('rm -f '//src//' '//ffc_exe//' '//ref_exe//' '// &
        ffc_out//' '//ref_out)
    options = compiler_frontend_options_t()
    options%run_semantics = .true.
    options%input_mode = INPUT_MODE_STANDARD
    call compile_frontend_from_string(source, frontend_result, options)
    if (.not. frontend_result%success()) then
        print *, 'FAIL: FortFront rejected c_ptr source: ', &
            trim(frontend_result%diagnostic_text)
        stop 1
    end if
    call lower_program_to_liric_exe(frontend_result%arena, &
        frontend_result%root_index, ffc_exe, error_msg)
    if (len_trim(error_msg) > 0) then
        print *, 'FAIL: ffc c_ptr lowering failed: ', trim(error_msg)
        stop 1
    end if

    open (newunit=unit, file=src, status='replace', action='write')
    write (unit, '(A)') source
    close (unit)
    call execute_command_line('gfortran -w '//src//' -o '//ref_exe, &
        exitstat=exit_stat)
    if (exit_stat /= 0) then
        print *, 'FAIL: gfortran rejected c_ptr source'
        stop 1
    end if
    call execute_command_line(ffc_exe//' > '//ffc_out, exitstat=exit_stat)
    if (exit_stat /= 0) then
        print *, 'FAIL: ffc c_ptr executable failed'
        stop 1
    end if
    call execute_command_line(ref_exe//' > '//ref_out, exitstat=exit_stat)
    if (exit_stat /= 0) then
        print *, 'FAIL: gfortran c_ptr executable failed'
        stop 1
    end if
    call execute_command_line('diff '//ffc_out//' '//ref_out// &
        ' > /dev/null 2>&1', exitstat=diff_status)
    if (diff_status /= 0) then
        print *, 'FAIL: c_ptr output differs from gfortran'
        call execute_command_line('diff '//ffc_out//' '//ref_out)
        stop 1
    end if
    call execute_command_line('rm -f '//src//' '//ffc_exe//' '//ref_exe//' '// &
        ffc_out//' '//ref_out)
    print *, 'PASS: c_ptr module matches gfortran'
end program test_session_c_ptr_module_compiler
