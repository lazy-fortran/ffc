program test_session_complex_intrinsics_compiler
    ! Verify scalar complex intrinsics lower through the direct LIRIC session and
    ! print byte-for-byte like gfortran: conjg/dconjg, abs(complex) -> real
    ! magnitude, cmplx/dcmplx with a kind selector (keyword or positional), the
    ! single-argument cmplx (zero imaginary part), real(z, kind) extraction, and
    ! complex(dp) kind resolution.
    use session_program_lowering, only: lower_program_to_liric_exe
    use fortfront_compiler, only: compiler_frontend_options_t, &
        compiler_frontend_result_t, &
        compile_frontend_from_string, &
        INPUT_MODE_STANDARD
    implicit none

    logical :: all_passed

    print *, '=== direct session complex intrinsics compiler test ==='

    all_passed = .true.

    ! conjg(a) = (3, -4)
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  complex :: a, r'//new_line('a')// &
        '  a = (3.0, 4.0)'//new_line('a')// &
        '  r = conjg(a)'//new_line('a')// &
        '  print *, r'//new_line('a')// &
        'end program main', &
        'conjg4')) all_passed = .false.

    ! abs((3,4)) = 5.0, complex(4) magnitude assigned to real
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  complex :: a'//new_line('a')// &
        '  real :: r'//new_line('a')// &
        '  a = (3.0, 4.0)'//new_line('a')// &
        '  r = abs(a)'//new_line('a')// &
        '  print *, r'//new_line('a')// &
        'end program main', &
        'abs4')) all_passed = .false.

    ! abs((3,4)) = 5.0d0, complex(8) magnitude assigned to real(8)
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  complex(8) :: a'//new_line('a')// &
        '  real(8) :: r'//new_line('a')// &
        '  a = (3.0d0, 4.0d0)'//new_line('a')// &
        '  r = abs(a)'//new_line('a')// &
        '  print *, r'//new_line('a')// &
        'end program main', &
        'abs8')) all_passed = .false.

    ! cmplx with keyword kind selector on two real arguments
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  complex(8) :: r'//new_line('a')// &
        '  r = cmplx(5.0d0, 6.0d0, kind=8)'//new_line('a')// &
        '  print *, r'//new_line('a')// &
        'end program main', &
        'cmplx_kw_kind')) all_passed = .false.

    ! cmplx with positional kind selector
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  complex(8) :: r'//new_line('a')// &
        '  r = cmplx(1.0d0, 2.0d0, 8)'//new_line('a')// &
        '  print *, r'//new_line('a')// &
        'end program main', &
        'cmplx_pos_kind')) all_passed = .false.

    ! single-argument cmplx gives a zero imaginary part
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  complex :: r'//new_line('a')// &
        '  r = cmplx(7.0)'//new_line('a')// &
        '  print *, r'//new_line('a')// &
        'end program main', &
        'cmplx_single')) all_passed = .false.

    ! dcmplx and dconjg on real(8) arguments
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  complex(8) :: r'//new_line('a')// &
        '  r = dconjg(dcmplx(1.5d0, 2.5d0))'//new_line('a')// &
        '  print *, r'//new_line('a')// &
        'end program main', &
        'dcmplx_dconjg')) all_passed = .false.

    ! complex(dp) kind resolution plus real(z, dp) extraction
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  use, intrinsic :: iso_fortran_env, only: dp => real64'//new_line('a')// &
        '  complex(dp) :: z'//new_line('a')// &
        '  z = dcmplx(1.5d0, 2.5d0)'//new_line('a')// &
        '  print *, real(z, dp), aimag(z)'//new_line('a')// &
        'end program main', &
        'complex_dp_extract')) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: scalar complex intrinsics match gfortran'

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
        base = '/tmp/ffc_cxi_'//trim(stem)
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
            print *, 'FAIL[', trim(stem), ']: FortFront rejected source: ', &
                trim(frontend_result%diagnostic_text)
            return
        end if

        call lower_program_to_liric_exe(frontend_result%arena, &
            frontend_result%root_index, exe, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL[', trim(stem), ']: ffc lowering failed: ', trim(error_msg)
            return
        end if

        open (newunit=unit, file=src, status='replace', action='write')
        write (unit, '(A)') source
        close (unit)
        call execute_command_line('gfortran -w '//src//' -o '//ref, &
            exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL[', trim(stem), ']: gfortran rejected source'
            return
        end if

        call execute_command_line(exe//' > '//ffc_out//' 2>&1', exitstat=exit_stat)
        call execute_command_line(ref//' > '//ref_out//' 2>&1', exitstat=exit_stat)
        call execute_command_line('diff '//ffc_out//' '//ref_out// &
            ' > /dev/null 2>&1', exitstat=status)
        if (status /= 0) then
            print *, 'FAIL[', trim(stem), ']: ffc output differs from gfortran'
            call execute_command_line('diff '//ffc_out//' '//ref_out)
            return
        end if
        call execute_command_line('rm -f '//src//' '//exe//' '//ref// &
            ' '//ffc_out//' '//ref_out)
        matches_gfortran = .true.
    end function matches_gfortran

end program test_session_complex_intrinsics_compiler
