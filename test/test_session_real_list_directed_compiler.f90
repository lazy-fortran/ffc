program test_session_real_list_directed_compiler
    ! gfortran-exact list-directed real(8) output: fixed form, exponential
    ! form, the F/E boundary, multi-item records, and non-finite values.
    ! Each program is compiled by ffc and by gfortran and the stdout is
    ! compared byte-for-byte (no whitespace normalisation).
    use fortfront_compiler, only: compiler_frontend_options_t, &
        compiler_frontend_result_t, &
        compile_frontend_from_string, INPUT_MODE_STANDARD
    use session_program_lowering, only: lower_program_to_liric_exe
    implicit none

    logical :: all_passed

    print *, '=== direct session real list-directed output test ==='

    all_passed = .true.
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  real(8) :: x'//new_line('a')// &
        '  x = 0.01d0'//new_line('a')// &
        '  print *, x'//new_line('a')// &
        'end program main', 'rld_eform')) all_passed = .false.
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  real(8) :: x, y, z'//new_line('a')// &
        '  x = 1.0d0'//new_line('a')// &
        '  y = 2.5d0'//new_line('a')// &
        '  z = -3.0d0'//new_line('a')// &
        '  print *, x, y, z'//new_line('a')// &
        'end program main', 'rld_multi')) all_passed = .false.
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  real(8) :: x'//new_line('a')// &
        '  x = 1.5d10'//new_line('a')// &
        '  print *, x'//new_line('a')// &
        '  x = 1.5d-10'//new_line('a')// &
        '  print *, x'//new_line('a')// &
        'end program main', 'rld_boundary')) all_passed = .false.
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  real(8) :: x, z'//new_line('a')// &
        '  z = 0.0d0'//new_line('a')// &
        '  x = 1.0d0 / z'//new_line('a')// &
        '  print *, x'//new_line('a')// &
        '  x = -1.0d0 / z'//new_line('a')// &
        '  print *, x'//new_line('a')// &
        '  x = z / z'//new_line('a')// &
        '  print *, x'//new_line('a')// &
        'end program main', 'rld_nonfinite')) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: real list-directed output matches gfortran byte-for-byte'

contains

    logical function matches_gfortran(source, stem)
        character(len=*), intent(in) :: source
        character(len=*), intent(in) :: stem
        type(compiler_frontend_options_t) :: options
        type(compiler_frontend_result_t) :: frontend_result
        character(len=:), allocatable :: error_msg
        character(len=:), allocatable :: base, src, exe, ref, ffc_out, ref_out
        integer :: unit, exit_stat, status

        matches_gfortran = .false.
        base = '/tmp/ffc_'//stem
        src = base//'.f90'
        exe = base//'.ffc'
        ref = base//'.gf'
        ffc_out = base//'.ffc.out'
        ref_out = base//'.gf.out'

        options = compiler_frontend_options_t()
        options%run_semantics = .true.
        options%input_mode = INPUT_MODE_STANDARD
        call compile_frontend_from_string(source, frontend_result, options)
        if (.not. frontend_result%success()) then
            print *, 'FAIL[', stem, ']: FortFront rejected source: ', &
                trim(frontend_result%diagnostic_text)
            return
        end if

        call lower_program_to_liric_exe(frontend_result%arena, &
            frontend_result%root_index, exe, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL[', stem, ']: ffc lowering failed: ', trim(error_msg)
            return
        end if

        open (newunit=unit, file=src, status='replace', action='write')
        write (unit, '(A)') source
        close (unit)
        call execute_command_line('gfortran -w '//src//' -o '//ref, &
            exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL[', stem, ']: gfortran rejected source'
            return
        end if

        call execute_command_line(exe//' > '//ffc_out, exitstat=exit_stat)
        call execute_command_line(ref//' > '//ref_out, exitstat=exit_stat)
        call execute_command_line('diff '//ffc_out//' '//ref_out// &
            ' > /dev/null 2>&1', exitstat=status)
        if (status /= 0) then
            print *, 'FAIL[', stem, ']: ffc output differs from gfortran'
            call execute_command_line('diff '//ffc_out//' '//ref_out)
        else
            matches_gfortran = .true.
        end if
        call execute_command_line('rm -f '//src//' '//exe//' '//ref//' '// &
            ffc_out//' '//ref_out)
    end function matches_gfortran

end program test_session_real_list_directed_compiler
