program test_session_intrinsics_extra_compiler
    use fortfront, only: compile_frontend_from_string, &
        compiler_frontend_options_t, compiler_frontend_result_t
    use session_program_lowering, only: lower_program_to_liric_exe
    implicit none

    character(len=*), parameter :: source = &
        'program main'//new_line('a')// &
        '  integer :: a(4) = [3, 1, 4, 2]'//new_line('a')// &
        '  logical :: m(4) = [.false., .true., .true., .false.]'//new_line('a')// &
        '  character(len=4) :: s'//new_line('a')// &
        '  s = repeat("ab", 2)'//new_line('a')// &
        '  print *, lbound(a, 1)'//new_line('a')// &
        '  print *, ubound(a, 1)'//new_line('a')// &
        '  print *, count(m)'//new_line('a')// &
        '  print *, minloc(a, dim=1), maxloc(a, dim=1, mask=m)'//new_line('a')// &
        '  print *, scan("abcde", "dx"), verify("abc", "abc")'//new_line('a')// &
        '  print *, s'//new_line('a')// &
        'end program main'

    print *, '=== intrinsic-extra gfortran differential test ==='
    if (.not. matches_gfortran(source)) stop 1
    print *, 'PASS: intrinsic-extra lowering matches gfortran'

contains

    logical function matches_gfortran(program_source)
        character(len=*), intent(in) :: program_source
        type(compiler_frontend_options_t) :: options
        type(compiler_frontend_result_t) :: frontend_result
        character(len=:), allocatable :: error_msg
        character(len=:), allocatable :: base, src, exe, ref
        character(len=:), allocatable :: ffc_out, ref_out
        integer :: unit, exit_stat, diff_status

        matches_gfortran = .false.
        call execute_command_line('mkdir -p /var/tmp/ert')
        base = '/var/tmp/ert/ffc_intrinsics_extra_differential'
        src = base//'.f90'
        exe = base//'.ffc'
        ref = base//'.gfortran'
        ffc_out = base//'.ffc.out'
        ref_out = base//'.gfortran.out'

        options = compiler_frontend_options_t()
        options%run_semantics = .true.
        call compile_frontend_from_string(program_source, frontend_result, options)
        if (.not. frontend_result%success()) then
            print *, 'FAIL: FortFront rejected intrinsic-extra source'
            print *, trim(frontend_result%diagnostic_text)
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
            print *, 'FAIL: gfortran rejected intrinsic-extra source'
            return
        end if

        call execute_command_line(exe//' > '//ffc_out//' 2>&1', &
            exitstat=exit_stat)
        call execute_command_line(ref//' > '//ref_out//' 2>&1', &
            exitstat=exit_stat)
        call execute_command_line('diff '//ffc_out//' '//ref_out// &
            ' > /dev/null 2>&1', exitstat=diff_status)
        if (diff_status /= 0) then
            print *, 'FAIL: ffc intrinsic-extra output differs from gfortran'
            call execute_command_line('diff '//ffc_out//' '//ref_out)
            return
        end if

        call execute_command_line('rm -f '//src//' '//exe//' '//ref//' '// &
            ffc_out//' '//ref_out)
        matches_gfortran = .true.
    end function matches_gfortran

end program test_session_intrinsics_extra_compiler
