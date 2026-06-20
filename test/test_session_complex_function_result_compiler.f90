program test_session_complex_function_result_compiler
    ! Verify contained functions returning complex(4)/complex(8) lower through
    ! the sret result ABI and print byte-for-byte like gfortran (#266 E5).
    use session_program_lowering, only: lower_program_to_liric_exe
    use fortfront_compiler, only: compiler_frontend_options_t, &
                                   compiler_frontend_result_t, &
                                   compile_frontend_from_string, &
                                   INPUT_MODE_STANDARD
    implicit none

    logical :: all_passed

    print *, '=== direct session complex function result test ==='

    all_passed = .true.

    ! complex(4) result from a literal
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  print *, make_c()'//new_line('a')// &
        'contains'//new_line('a')// &
        '  function make_c() result(z)'//new_line('a')// &
        '    complex :: z'//new_line('a')// &
        '    z = (1.0, 2.0)'//new_line('a')// &
        '  end function make_c'//new_line('a')// &
        'end program main', &
        'c4_result')) all_passed = .false.

    ! complex(4) result with a negative imaginary part
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  print *, make_c()'//new_line('a')// &
        'contains'//new_line('a')// &
        '  function make_c() result(z)'//new_line('a')// &
        '    complex :: z'//new_line('a')// &
        '    z = (3.5, -1.25)'//new_line('a')// &
        '  end function make_c'//new_line('a')// &
        'end program main', &
        'c4_result_neg')) all_passed = .false.

    ! complex(8) result from a double-precision literal
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  print *, make_c()'//new_line('a')// &
        'contains'//new_line('a')// &
        '  function make_c() result(z)'//new_line('a')// &
        '    complex(8) :: z'//new_line('a')// &
        '    z = (3.5d0, -1.25d0)'//new_line('a')// &
        '  end function make_c'//new_line('a')// &
        'end program main', &
        'c8_result')) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: complex(4)/complex(8) function results lower through '// &
        'direct LIRIC session'

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
        base = '/tmp/ffc_cxfn_'//trim(stem)
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

end program test_session_complex_function_result_compiler
