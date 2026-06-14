program test_session_complex_compiler
    ! Verify complex(4) and complex(8) declarations, assignment, and
    ! list-directed print match gfortran byte-for-byte (#246).
    use session_program_lowering, only: lower_program_to_liric_exe
    use fortfront_compiler, only: compiler_frontend_options_t, &
                                   compiler_frontend_result_t, &
                                   compile_frontend_from_string, &
                                   INPUT_MODE_STANDARD
    implicit none

    logical :: all_passed

    print *, '=== direct session complex compiler test ==='

    all_passed = .true.

    ! complex(4) basic declaration and print
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  complex :: z'//new_line('a')// &
        '  z = (1.0, 2.0)'//new_line('a')// &
        '  print *, z'//new_line('a')// &
        'end program main', &
        'c4_basic')) all_passed = .false.

    ! complex(4) negative imaginary part
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  complex :: z'//new_line('a')// &
        '  z = (3.5, -1.25)'//new_line('a')// &
        '  print *, z'//new_line('a')// &
        'end program main', &
        'c4_neg_imag')) all_passed = .false.

    ! complex(4) zero
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  complex :: z'//new_line('a')// &
        '  z = (0.0, 0.0)'//new_line('a')// &
        '  print *, z'//new_line('a')// &
        'end program main', &
        'c4_zero')) all_passed = .false.

    ! complex(8) declaration and print
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  complex(8) :: z'//new_line('a')// &
        '  z = (1.0d0, 2.0d0)'//new_line('a')// &
        '  print *, z'//new_line('a')// &
        'end program main', &
        'c8_basic')) all_passed = .false.

    ! complex(8) large exponent (exponential notation)
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  complex(8) :: z'//new_line('a')// &
        '  z = (1.0d10, -2.5d-5)'//new_line('a')// &
        '  print *, z'//new_line('a')// &
        'end program main', &
        'c8_exp')) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: complex(4)/complex(8) lower through direct LIRIC session'

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
        base = '/tmp/ffc_cx_'//trim(stem)
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

end program test_session_complex_compiler
