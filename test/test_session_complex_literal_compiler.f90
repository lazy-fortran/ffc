program test_session_complex_literal_compiler
    ! Verify complex literal lowering: (re, im) literals reach memory with the
    ! right components (#267), declaration initializers, cmplx(re, im), complex
    ! +/- arithmetic, and real(z)/aimag(z) extraction match gfortran exactly.
    use session_program_lowering, only: lower_program_to_liric_exe
    use fortfront_compiler, only: compiler_frontend_options_t, &
        compiler_frontend_result_t, &
        compile_frontend_from_string, &
        INPUT_MODE_STANDARD
    implicit none

    logical :: all_passed

    print *, '=== direct session complex literal compiler test ==='

    all_passed = .true.

    ! complex(4) literal assignment must keep both parts (the #267 bug).
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  complex :: z'//new_line('a')// &
        '  z = (1.0, 2.0)'//new_line('a')// &
        '  print *, z'//new_line('a')// &
        'end program main', &
        'lit_c4')) all_passed = .false.

    ! complex(8) literal assignment.
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  complex(8) :: z'//new_line('a')// &
        '  z = (1.0d0, 2.0d0)'//new_line('a')// &
        '  print *, z'//new_line('a')// &
        'end program main', &
        'lit_c8')) all_passed = .false.

    ! complex(kind=8) declaration initializer with a literal.
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  complex(kind=8) :: w = (1.0d0, 2.0d0)'//new_line('a')// &
        '  print *, w'//new_line('a')// &
        'end program main', &
        'init_c8')) all_passed = .false.

    ! cmplx(re, im) intrinsic plus complex addition and real/aimag extraction.
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  complex :: z1, z2, z3'//new_line('a')// &
        '  z1 = (1.0, 2.0)'//new_line('a')// &
        '  z2 = cmplx(3.0, 4.0)'//new_line('a')// &
        '  z3 = z1 + z2'//new_line('a')// &
        '  print *, z3'//new_line('a')// &
        '  print *, real(z3)'//new_line('a')// &
        '  print *, aimag(z3)'//new_line('a')// &
        'end program main', &
        'cmplx_add')) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: complex literals, cmplx, add, and real/aimag match gfortran'

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
        base = '/tmp/ffc_clit_'//trim(stem)
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

end program test_session_complex_literal_compiler
