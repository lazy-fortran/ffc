program test_session_runtime_rank2_sum_compiler
    ! Runtime reductions must walk the complete contiguous descriptor, not only
    ! the leading dimension. The positive rank-2 and rank-3 cases compare ffc
    ! with independently compiled gfortran executables.
    use fortfront_compiler, only: compiler_frontend_options_t, &
        compiler_frontend_result_t, compile_frontend_from_string, &
        INPUT_MODE_STANDARD
    use session_program_lowering, only: lower_program_to_liric_exe
    implicit none

    character(len=*), parameter :: source = &
        'program main'//new_line('a')// &
        '  integer, allocatable :: values(:,:)'//new_line('a')// &
        '  integer :: i, j'//new_line('a')// &
        '  allocate(values(2,3))'//new_line('a')// &
        '  do j = 1, 3'//new_line('a')// &
        '    do i = 1, 2'//new_line('a')// &
        '      values(i,j) = 10*j + i'//new_line('a')// &
        '    end do'//new_line('a')// &
        '  end do'//new_line('a')// &
        '  call consume(values)'//new_line('a')// &
        '  call automatic(2, 3)'//new_line('a')// &
        'contains'//new_line('a')// &
        '  subroutine consume(a)'//new_line('a')// &
        '    integer, intent(in) :: a(:,:)'//new_line('a')// &
        '    print *, sum(a)'//new_line('a')// &
        '  end subroutine consume'//new_line('a')// &
        '  subroutine automatic(n, m)'//new_line('a')// &
        '    integer, intent(in) :: n, m'//new_line('a')// &
        '    integer :: a(n,m), i, j'//new_line('a')// &
        '    do j = 1, m'//new_line('a')// &
        '      do i = 1, n'//new_line('a')// &
        '        a(i,j) = 10*j + i'//new_line('a')// &
        '      end do'//new_line('a')// &
        '    end do'//new_line('a')// &
        '    print *, sum(a)'//new_line('a')// &
        '  end subroutine automatic'//new_line('a')// &
        'end program main'

    character(len=*), parameter :: rank3_source = &
        'program main'//new_line('a')// &
        '  integer, allocatable :: values(:,:,:)'//new_line('a')// &
        '  allocate(values(2,2,2))'//new_line('a')// &
        '  values = 3'//new_line('a')// &
        '  call consume(values)'//new_line('a')// &
        '  call automatic(2,2,2)'//new_line('a')// &
        'contains'//new_line('a')// &
        '  subroutine consume(a)'//new_line('a')// &
        '    integer, intent(in) :: a(:,:,:)'//new_line('a')// &
        '    print *, sum(a)'//new_line('a')// &
        '  end subroutine consume'//new_line('a')// &
        '  subroutine automatic(n,m,k)'//new_line('a')// &
        '    integer, intent(in) :: n,m,k'//new_line('a')// &
        '    integer :: a(n,m,k)'//new_line('a')// &
        '    a = 2'//new_line('a')// &
        '    print *, sum(a)'//new_line('a')// &
        '  end subroutine automatic'//new_line('a')// &
        'end program main'

    logical :: all_passed

    print *, '=== direct session runtime rank-2/rank-3 sum compiler test ==='
    all_passed = matches_gfortran(source)
    if (.not. matches_gfortran(rank3_source)) all_passed = .false.
    if (.not. all_passed) stop 1
    print *, 'PASS: runtime rank-2 and rank-3 sum match gfortran'

contains

    logical function matches_gfortran(program_source) result(ok)
        character(len=*), intent(in) :: program_source
        type(compiler_frontend_options_t) :: options
        type(compiler_frontend_result_t) :: frontend_result
        character(len=:), allocatable :: error_msg
        character(len=:), allocatable :: base, src, exe, ref
        character(len=:), allocatable :: ffc_out, ref_out
        integer :: unit, exit_stat, diff_status

        ok = .false.
        base = '/var/tmp/ert/ffc_runtime_rank2_sum'
        src = base//'.f90'
        exe = base//'.ffc'
        ref = base//'.gfortran'
        ffc_out = base//'.ffc.out'
        ref_out = base//'.gfortran.out'

        options = compiler_frontend_options_t()
        options%run_semantics = .true.
        options%input_mode = INPUT_MODE_STANDARD
        call compile_frontend_from_string(program_source, frontend_result, &
                                           options)
        if (.not. frontend_result%success()) then
            print *, 'FAIL: FortFront rejected runtime sum source: ', &
                trim(frontend_result%diagnostic_text)
            return
        end if

        call lower_program_to_liric_exe(frontend_result%arena, &
                                        frontend_result%root_index, exe, &
                                        error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL: ffc runtime sum lowering failed: ', &
                trim(error_msg)
            return
        end if

        open (newunit=unit, file=src, status='replace', action='write')
        write (unit, '(A)') program_source
        close (unit)
        call execute_command_line('gfortran -w '//src//' -o '//ref, &
                                  exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL: gfortran rejected runtime sum source'
            return
        end if

        call execute_command_line(exe//' > '//ffc_out, exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL: ffc runtime sum executable failed'
            return
        end if
        call execute_command_line(ref//' > '//ref_out, exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL: gfortran runtime sum executable failed'
            return
        end if
        call execute_command_line('diff '//ffc_out//' '//ref_out// &
                                  ' > /dev/null 2>&1', &
                                  exitstat=diff_status)
        if (diff_status /= 0) then
            print *, 'FAIL: ffc runtime sum differs from gfortran'
            call execute_command_line('diff '//ffc_out//' '//ref_out)
            return
        end if

        call execute_command_line('rm -f '//src//' '//exe//' '//ref//' '// &
                                  ffc_out//' '//ref_out)
        ok = .true.
    end function matches_gfortran

end program test_session_runtime_rank2_sum_compiler
