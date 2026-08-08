program test_session_runtime_array_section_multidim_compiler
    ! The reference executable is compiled independently by gfortran.  This
    ! catches both the section trip-count and column-major coordinate mapping;
    ! a compile-only check could miss either one.
    use fortfront_compiler, only: compiler_frontend_options_t, &
        compiler_frontend_result_t, compile_frontend_from_string, &
        INPUT_MODE_STANDARD
    use session_program_lowering, only: lower_program_to_liric_exe
    implicit none

    character(len=*), parameter :: source = &
        'program main'//new_line('a')// &
        '  integer :: n, m'//new_line('a')// &
        '  n = 4'//new_line('a')// &
        '  m = 3'//new_line('a')// &
        '  call work(n, m)'//new_line('a')// &
        'contains'//new_line('a')// &
        '  subroutine work(n, m)'//new_line('a')// &
        '    integer, intent(in) :: n, m'//new_line('a')// &
        '    integer :: a(n,m), i, j'//new_line('a')// &
        '    do j = 1, m'//new_line('a')// &
        '      do i = 1, n'//new_line('a')// &
        '        a(i,j) = 10*j + i'//new_line('a')// &
        '      end do'//new_line('a')// &
        '    end do'//new_line('a')// &
        '    a(2:n,1:m) = 7'//new_line('a')// &
        '    print *, a(1,1), a(2,1), a(1,2), a(4,3)'//new_line('a')// &
        '  end subroutine work'//new_line('a')// &
        'end program main'

    print *, '=== direct session multidimensional runtime section test ==='
    if (.not. matches_gfortran(source)) stop 1
    print *, 'PASS: runtime array-section broadcast matches gfortran for two retained dimensions'

contains

    logical function matches_gfortran(program_source) result(ok)
        character(len=*), intent(in) :: program_source
        type(compiler_frontend_options_t) :: options
        type(compiler_frontend_result_t) :: frontend_result
        character(len=:), allocatable :: error_msg
        character(len=:), allocatable :: base, src, exe, ref, ffc_out, ref_out
        integer :: unit, exit_stat, diff_status

        ok = .false.
        base = '/var/tmp/ert/ffc_runtime_section_multidim'
        src = base//'.f90'
        exe = base//'.ffc'
        ref = base//'.gfortran'
        ffc_out = base//'.ffc.out'
        ref_out = base//'.gfortran.out'

        options = compiler_frontend_options_t()
        options%run_semantics = .true.
        options%input_mode = INPUT_MODE_STANDARD
        call compile_frontend_from_string(program_source, frontend_result, options)
        if (.not. frontend_result%success()) then
            print *, 'FAIL: FortFront rejected runtime section source: ', &
                trim(frontend_result%diagnostic_text)
            return
        end if

        call lower_program_to_liric_exe(frontend_result%arena, &
            frontend_result%root_index, exe, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL: ffc lowering failed: ', trim(error_msg)
            return
        end if

        open (newunit=unit, file=src, status='replace', action='write')
        write (unit, '(A)') program_source
        close (unit)
        call execute_command_line('gfortran -w '//src//' -o '//ref, &
            exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL: gfortran rejected runtime section source'
            return
        end if

        call execute_command_line(exe//' > '//ffc_out, exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL: ffc runtime section executable failed'
            return
        end if
        call execute_command_line(ref//' > '//ref_out, exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL: gfortran runtime section executable failed'
            return
        end if
        call execute_command_line('diff '//ffc_out//' '//ref_out// &
            ' > /dev/null 2>&1', exitstat=diff_status)
        if (diff_status /= 0) then
            print *, 'FAIL: ffc runtime section output differs from gfortran'
            call execute_command_line('diff '//ffc_out//' '//ref_out)
            return
        end if

        call execute_command_line('rm -f '//src//' '//exe//' '//ref//' '// &
            ffc_out//' '//ref_out)
        ok = .true.
    end function matches_gfortran

end program test_session_runtime_array_section_multidim_compiler
