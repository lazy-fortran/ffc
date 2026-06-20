program test_session_complex_arith_compiler
    ! Verify complex(4) and complex(8) multiply and divide lower through the
    ! direct LIRIC session and print byte-for-byte like gfortran.
    !   (a+bi)*(c+di) = (ac-bd) + (ad+bc)i
    !   (a+bi)/(c+di) = ((ac+bd) + (bc-ad)i) / (c*c+d*d)
    use session_program_lowering, only: lower_program_to_liric_exe
    use fortfront_compiler, only: compiler_frontend_options_t, &
                                   compiler_frontend_result_t, &
                                   compile_frontend_from_string, &
                                   INPUT_MODE_STANDARD
    implicit none

    logical :: all_passed

    print *, '=== direct session complex arithmetic compiler test ==='

    all_passed = .true.

    ! complex(4) multiply: (2+3i)*(1-i) = 5 + i
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  complex :: a, b, r'//new_line('a')// &
        '  a = (2.0, 3.0)'//new_line('a')// &
        '  b = (1.0, -1.0)'//new_line('a')// &
        '  r = a * b'//new_line('a')// &
        '  print *, r'//new_line('a')// &
        'end program main', &
        'c4_mul')) all_passed = .false.

    ! complex(4) divide: (2+3i)/(1-i) = -0.5 + 2.5i
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  complex :: a, b, r'//new_line('a')// &
        '  a = (2.0, 3.0)'//new_line('a')// &
        '  b = (1.0, -1.0)'//new_line('a')// &
        '  r = a / b'//new_line('a')// &
        '  print *, r'//new_line('a')// &
        'end program main', &
        'c4_div')) all_passed = .false.

    ! complex(4) multiply by complex literal directly
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  complex :: a, r'//new_line('a')// &
        '  a = (2.0, 3.0)'//new_line('a')// &
        '  r = a * (1.0, -1.0)'//new_line('a')// &
        '  print *, r'//new_line('a')// &
        'end program main', &
        'c4_mul_lit')) all_passed = .false.

    ! complex(8) multiply: (2+3i)*(1-i) = 5 + i
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  complex(8) :: a, b, r'//new_line('a')// &
        '  a = (2.0d0, 3.0d0)'//new_line('a')// &
        '  b = (1.0d0, -1.0d0)'//new_line('a')// &
        '  r = a * b'//new_line('a')// &
        '  print *, r'//new_line('a')// &
        'end program main', &
        'c8_mul')) all_passed = .false.

    ! complex(8) divide: (2+3i)/(1-i) = -0.5 + 2.5i
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  complex(8) :: a, b, r'//new_line('a')// &
        '  a = (2.0d0, 3.0d0)'//new_line('a')// &
        '  b = (1.0d0, -1.0d0)'//new_line('a')// &
        '  r = a / b'//new_line('a')// &
        '  print *, r'//new_line('a')// &
        'end program main', &
        'c8_div')) all_passed = .false.

    ! complex(8) chained multiply: a * b * b
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  complex(8) :: a, b, r'//new_line('a')// &
        '  a = (2.0d0, 3.0d0)'//new_line('a')// &
        '  b = (1.0d0, -1.0d0)'//new_line('a')// &
        '  r = a * b * b'//new_line('a')// &
        '  print *, r'//new_line('a')// &
        'end program main', &
        'c8_chain_mul')) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: complex(4)/complex(8) multiply and divide match gfortran'

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
        base = '/tmp/ffc_cxa_'//trim(stem)
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

end program test_session_complex_arith_compiler
