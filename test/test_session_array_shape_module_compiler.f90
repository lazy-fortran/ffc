program test_session_array_shape_module_compiler
    ! The array-shape classifier is a typed descendant of the lowering
    ! module.  Keep one differential oracle over both assumed-shape and
    ! assumed-size dummies so the extraction cannot silently alter ABI shape.
    use fortfront_compiler, only: compiler_frontend_options_t, &
        compiler_frontend_result_t, compile_frontend_from_string, &
        INPUT_MODE_STANDARD
    use session_program_lowering, only: lower_program_to_liric_exe
    implicit none

    logical :: all_passed

    print *, '=== array-shape module compiler test ==='
    all_passed = matches_gfortran( &
        'program main'//new_line('a')// &
        '  integer :: a(3,2)'//new_line('a')// &
        '  integer :: i, j'//new_line('a')// &
        '  do j = 1, 2'//new_line('a')// &
        '    do i = 1, 3'//new_line('a')// &
        '      a(i,j) = 10*j + i'//new_line('a')// &
        '    end do'//new_line('a')// &
        '  end do'//new_line('a')// &
        '  call inspect_shape(a)'//new_line('a')// &
        '  call inspect_size(a)'//new_line('a')// &
        'contains'//new_line('a')// &
        '  subroutine inspect_shape(x)'//new_line('a')// &
        '    integer, intent(in) :: x(:,:)'//new_line('a')// &
        '    print *, size(x), lbound(x,1), ubound(x,1), '// &
        'lbound(x,2), ubound(x,2)'//new_line('a')// &
        '    print *, x(1,1), x(3,2)'//new_line('a')// &
        '  end subroutine inspect_shape'//new_line('a')// &
        '  subroutine inspect_size(x)'//new_line('a')// &
        '    integer, intent(in) :: x(3,*)'//new_line('a')// &
        '    print *, x(1,1), x(3,2)'//new_line('a')// &
        '  end subroutine inspect_size'//new_line('a')// &
        'end program main', 'shape_contract')

    if (.not. all_passed) stop 1
    print *, 'PASS: array-shape module matches gfortran'

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
        base = '/var/tmp/ert/ffc_array_shape_'//trim(stem)
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
            print *, 'FAIL: FortFront rejected array-shape source: ', &
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
        write (unit, '(A)') source
        close (unit)
        call execute_command_line('gfortran -w '//src//' -o '//ref, &
            exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL: gfortran rejected array-shape source'
            return
        end if

        call execute_command_line(exe//' > '//ffc_out//' 2>&1', &
            exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL: ffc array-shape executable failed'
            return
        end if
        call execute_command_line(ref//' > '//ref_out//' 2>&1', &
            exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL: gfortran array-shape executable failed'
            return
        end if
        call execute_command_line('diff '//ffc_out//' '//ref_out// &
            ' > /dev/null 2>&1', exitstat=status)
        if (status /= 0) then
            print *, 'FAIL: ffc array-shape output differs from gfortran'
            call execute_command_line('diff '//ffc_out//' '//ref_out)
            return
        end if
        call execute_command_line('rm -f '//src//' '//exe//' '//ref//' '// &
            ffc_out//' '//ref_out)
        matches_gfortran = .true.
    end function matches_gfortran

end program test_session_array_shape_module_compiler
