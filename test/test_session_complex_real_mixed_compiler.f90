program test_session_complex_real_mixed_compiler
    ! Verify a complex(4)/complex(8) binary expression with one real operand
    ! promotes the real side with a zero imaginary part and matches gfortran
    ! byte-for-byte, both in a scalar assignment and a declaration initializer.
    use session_program_lowering, only: lower_program_to_liric_exe
    use fortfront_compiler, only: compiler_frontend_options_t, &
        compiler_frontend_result_t, &
        compile_frontend_from_string, &
        INPUT_MODE_STANDARD
    implicit none

    logical :: all_passed

    print *, '=== direct session mixed real/complex arithmetic compiler test ==='

    all_passed = .true.

    ! complex(4) = real + complex(4)*real, in a declaration initializer.
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  implicit none'//new_line('a')// &
        '  real, parameter :: a = 3.0, b = 4.0'//new_line('a')// &
        '  complex, parameter :: i_ = (0, 1)'//new_line('a')// &
        '  complex :: z = a + i_*b'//new_line('a')// &
        '  complex :: w = a+b + i_*(a-b)'//new_line('a')// &
        '  print *, z, w'//new_line('a')// &
        'end program main', &
        'c4_mixed_decl')) all_passed = .false.

    ! complex(4) assignment statement with a real identifier operand.
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  implicit none'//new_line('a')// &
        '  real :: r'//new_line('a')// &
        '  complex :: c, s'//new_line('a')// &
        '  r = 2.0'//new_line('a')// &
        '  c = (1.0, 3.0)'//new_line('a')// &
        '  s = c + r'//new_line('a')// &
        '  print *, s'//new_line('a')// &
        'end program main', &
        'c4_mixed_assign')) all_passed = .false.

    ! complex(8) assignment with a real subtraction operand mixed in.
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  implicit none'//new_line('a')// &
        '  real(8) :: r'//new_line('a')// &
        '  complex(8) :: c, s'//new_line('a')// &
        '  r = 2.0d0'//new_line('a')// &
        '  c = (1.0d0, 3.0d0)'//new_line('a')// &
        '  s = c * (r - 1.0d0)'//new_line('a')// &
        '  print *, s'//new_line('a')// &
        'end program main', &
        'c8_mixed_assign')) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: mixed real/complex arithmetic matches gfortran'

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
        base = '/tmp/ffc_cxrm_'//trim(stem)
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

end program test_session_complex_real_mixed_compiler
