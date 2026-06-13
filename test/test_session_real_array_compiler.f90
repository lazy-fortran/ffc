program test_session_real_array_compiler
    use fortfront_compiler, only: compiler_frontend_options_t, &
                                  compiler_frontend_result_t, &
                                  compile_frontend_from_string, INPUT_MODE_STANDARD
    use session_program_lowering, only: lower_program_to_liric_exe
    implicit none

    logical :: all_passed

    print *, '=== direct session real array compiler test ==='

    all_passed = .true.
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  real :: a(3)'//new_line('a')// &
        '  a(1) = 1.5'//new_line('a')// &
        '  a(2) = 2.5'//new_line('a')// &
        '  a(3) = 3.5'//new_line('a')// &
        '  print *, a(2)'//new_line('a')// &
        'end program main', 'ra_elem_read')) all_passed = .false.

    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  real :: a(4)'//new_line('a')// &
        '  a = 3.0'//new_line('a')// &
        '  print *, a(1), a(4)'//new_line('a')// &
        'end program main', 'ra_broadcast')) all_passed = .false.

    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  real :: a(3)'//new_line('a')// &
        '  a = [1.0, 2.0, 3.0]'//new_line('a')// &
        '  print *, a(1), a(2), a(3)'//new_line('a')// &
        'end program main', 'ra_ctor')) all_passed = .false.

    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  real :: a(3)'//new_line('a')// &
        '  a = [1.0, 2.0, 3.0]'//new_line('a')// &
        '  print *, a'//new_line('a')// &
        'end program main', 'ra_print')) all_passed = .false.

    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  real :: a(3), b(3), c(3)'//new_line('a')// &
        '  a = [1.0, 2.0, 3.0]'//new_line('a')// &
        '  b = [4.0, 5.0, 6.0]'//new_line('a')// &
        '  c = a + b'//new_line('a')// &
        '  print *, c(1), c(2), c(3)'//new_line('a')// &
        'end program main', 'ra_add')) all_passed = .false.

    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  real :: a(3), b(3), c(3)'//new_line('a')// &
        '  a = [3.0, 6.0, 9.0]'//new_line('a')// &
        '  b = [1.0, 2.0, 3.0]'//new_line('a')// &
        '  c = a - b'//new_line('a')// &
        '  c = c * b'//new_line('a')// &
        '  print *, c(1), c(2), c(3)'//new_line('a')// &
        'end program main', 'ra_submul')) all_passed = .false.

    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  real :: a(4)'//new_line('a')// &
        '  a = [1.0, 2.0, 3.0, 4.0]'//new_line('a')// &
        '  print *, sum(a)'//new_line('a')// &
        '  print *, product(a)'//new_line('a')// &
        '  print *, maxval(a)'//new_line('a')// &
        '  print *, minval(a)'//new_line('a')// &
        'end program main', 'ra_reductions')) all_passed = .false.

    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  real :: a(4)'//new_line('a')// &
        '  integer :: i'//new_line('a')// &
        '  do i = 1, 4'//new_line('a')// &
        '    a(i) = real(i) * 0.5'//new_line('a')// &
        '  end do'//new_line('a')// &
        '  print *, a(1), a(4)'//new_line('a')// &
        'end program main', 'ra_loop')) all_passed = .false.

    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  real(8) :: a(3)'//new_line('a')// &
        '  a(1) = 1.5d0'//new_line('a')// &
        '  a(2) = 2.5d0'//new_line('a')// &
        '  a(3) = 3.5d0'//new_line('a')// &
        '  print *, a(2)'//new_line('a')// &
        'end program main', 'ra_f64_elem')) all_passed = .false.

    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  real(8) :: a(3)'//new_line('a')// &
        '  a = 2.0d0'//new_line('a')// &
        '  print *, a(1), a(3)'//new_line('a')// &
        'end program main', 'ra_f64_broadcast')) all_passed = .false.

    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  real(8) :: a(2), b(2), c(2)'//new_line('a')// &
        '  a = [1.0d0, 2.0d0]'//new_line('a')// &
        '  b = [10.0d0, 20.0d0]'//new_line('a')// &
        '  c = a + b'//new_line('a')// &
        '  print *, c(1), c(2)'//new_line('a')// &
        'end program main', 'ra_f64_add')) all_passed = .false.

    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  real(8) :: a(3)'//new_line('a')// &
        '  a = [1.0d0, 2.0d0, 3.0d0]'//new_line('a')// &
        '  print *, sum(a)'//new_line('a')// &
        '  print *, maxval(a)'//new_line('a')// &
        '  print *, minval(a)'//new_line('a')// &
        'end program main', 'ra_f64_reductions')) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: real(4) and real(8) arrays lower through direct LIRIC'

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
        base = '/tmp/ffc_ra_'//stem
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

end program test_session_real_array_compiler
