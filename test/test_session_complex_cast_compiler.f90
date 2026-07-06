program test_session_complex_cast_compiler
    ! Verify complex scalar assignments that widen an integer/real component
    ! or convert between complex kinds match gfortran: cmplx() with integer
    ! and mixed integer/real arguments, integer/real-to-complex implicit
    ! assignment, and complex(4)<->complex(8) kind conversion.
    use session_program_lowering, only: lower_program_to_liric_exe
    use fortfront_compiler, only: compiler_frontend_options_t, &
        compiler_frontend_result_t, &
        compile_frontend_from_string, &
        INPUT_MODE_STANDARD
    implicit none

    logical :: all_passed

    print *, '=== direct session complex cast compiler test ==='

    all_passed = .true.

    ! cmplx() with an integer real part and a real imaginary part
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  integer :: i = 5'//new_line('a')// &
        '  real :: x = 2.5'//new_line('a')// &
        '  complex :: z'//new_line('a')// &
        '  z = cmplx(i, x)'//new_line('a')// &
        '  print *, z'//new_line('a')// &
        'end program main', &
        'cmplx_int_real')) all_passed = .false.

    ! integer scalar assigned to a complex variable (imag part zero)
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  integer :: i = 7'//new_line('a')// &
        '  complex :: z'//new_line('a')// &
        '  z = i'//new_line('a')// &
        '  print *, z'//new_line('a')// &
        'end program main', &
        'int_to_complex')) all_passed = .false.

    ! cmplx() into complex(8) from a real and an integer imaginary part
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  real(8) :: r = 6.5d0'//new_line('a')// &
        '  integer :: i = 3'//new_line('a')// &
        '  complex(8) :: z'//new_line('a')// &
        '  z = cmplx(r, i, kind=8)'//new_line('a')// &
        '  print *, z'//new_line('a')// &
        'end program main', &
        'cmplx_real8_int')) all_passed = .false.

    ! complex(8) = complex(4) promotes both components to double
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  complex :: a'//new_line('a')// &
        '  complex(8) :: b'//new_line('a')// &
        '  a = (1.5, 2.5)'//new_line('a')// &
        '  b = a'//new_line('a')// &
        '  print *, real(b), aimag(b)'//new_line('a')// &
        'end program main', &
        'c4_to_c8')) all_passed = .false.

    ! complex(4) = complex(8) demotes both components to single
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  complex(8) :: a'//new_line('a')// &
        '  complex :: b'//new_line('a')// &
        '  a = (3.5d0, 4.5d0)'//new_line('a')// &
        '  b = a'//new_line('a')// &
        '  print *, real(b), aimag(b)'//new_line('a')// &
        'end program main', &
        'c8_to_c4')) all_passed = .false.

    ! integer expression assigned to a complex variable (real combine, im=0)
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  integer :: i = 1'//new_line('a')// &
        '  complex :: z'//new_line('a')// &
        '  z = i + 10'//new_line('a')// &
        '  print *, real(z), aimag(z)'//new_line('a')// &
        'end program main', &
        'int_expr_to_complex')) all_passed = .false.

    if (all_passed) then
        print *, 'PASS: all complex cast cases match gfortran'
    else
        print *, 'FAIL: some complex cast cases diverged'
        error stop 1
    end if

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
        base = '/tmp/ffc_cxcast_'//trim(stem)
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

end program test_session_complex_cast_compiler
