program test_session_runtime_rank2_sum_compiler
    ! Runtime rank-2 reductions must walk the complete contiguous descriptor,
    ! not only the leading dimension.  The positive case compares ffc with an
    ! independently compiled gfortran executable; the rank-3 case is a valid
    ! Fortran program that remains outside this bounded lowering contract.
    use fortfront_compiler, only: compiler_frontend_options_t, &
        compiler_frontend_result_t, compile_frontend_from_string, &
        INPUT_MODE_STANDARD
    use ffc_test_support, only: compile_to_exe
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
        '  call work(2, 2, 2)'//new_line('a')// &
        'contains'//new_line('a')// &
        '  subroutine work(n, m, k)'//new_line('a')// &
        '    integer, intent(in) :: n, m, k'//new_line('a')// &
        '    integer :: a(n,m,k)'//new_line('a')// &
        '    a = 1'//new_line('a')// &
        '    print *, sum(a)'//new_line('a')// &
        '  end subroutine work'//new_line('a')// &
        'end program main'

    logical :: all_passed

    print *, '=== direct session runtime rank-2 sum compiler test ==='
    all_passed = matches_gfortran(source)
    if (.not. test_rank3_refusal(rank3_source)) all_passed = .false.
    if (.not. all_passed) stop 1
    print *, 'PASS: runtime rank-2 sum matches gfortran and rank-3 remains refused'

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
            print *, 'FAIL: FortFront rejected runtime rank-2 sum source: ', &
                trim(frontend_result%diagnostic_text)
            return
        end if

        call lower_program_to_liric_exe(frontend_result%arena, &
                                        frontend_result%root_index, exe, &
                                        error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL: ffc runtime rank-2 sum lowering failed: ', &
                trim(error_msg)
            return
        end if

        open (newunit=unit, file=src, status='replace', action='write')
        write (unit, '(A)') program_source
        close (unit)
        call execute_command_line('gfortran -w '//src//' -o '//ref, &
                                  exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL: gfortran rejected runtime rank-2 sum source'
            return
        end if

        call execute_command_line(exe//' > '//ffc_out, exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL: ffc runtime rank-2 sum executable failed'
            return
        end if
        call execute_command_line(ref//' > '//ref_out, exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL: gfortran runtime rank-2 sum executable failed'
            return
        end if
        call execute_command_line('diff '//ffc_out//' '//ref_out// &
                                  ' > /dev/null 2>&1', &
                                  exitstat=diff_status)
        if (diff_status /= 0) then
            print *, 'FAIL: ffc runtime rank-2 sum differs from gfortran'
            call execute_command_line('diff '//ffc_out//' '//ref_out)
            return
        end if

        call execute_command_line('rm -f '//src//' '//exe//' '//ref//' '// &
                                  ffc_out//' '//ref_out)
        ok = .true.
    end function matches_gfortran

    logical function test_rank3_refusal(program_source) result(ok)
        character(len=*), intent(in) :: program_source
        character(len=*), parameter :: base = &
            '/var/tmp/ert/ffc_runtime_rank3_sum_refusal'
        character(len=*), parameter :: src = base//'.f90'
        character(len=*), parameter :: ref = base//'.gfortran'
        character(len=*), parameter :: exe = base//'.ffc'
        character(len=:), allocatable :: error_msg
        integer :: unit, exit_stat

        ok = .false.
        open (newunit=unit, file=src, status='replace', action='write')
        write (unit, '(A)') program_source
        close (unit)
        call execute_command_line('gfortran -w '//src//' -o '//ref, &
                                  exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL: gfortran rejected valid rank-3 refusal fixture'
            call execute_command_line('rm -f '//src//' '//ref//' '//exe)
            return
        end if

        call compile_to_exe(program_source, exe, error_msg)
        call execute_command_line('rm -f '//src//' '//ref//' '//exe)
        if (len_trim(error_msg) == 0) then
            print *, 'FAIL: rank-3 runtime sum lowered without a diagnostic'
            return
        end if
        if (index(error_msg, &
                  'sum over runtime-extent arrays supports rank-1 and rank-2 only') == 0) then
            print *, 'FAIL: rank-3 refusal diagnostic changed: ', trim(error_msg)
            return
        end if
        ok = .true.
    end function test_rank3_refusal

end program test_session_runtime_rank2_sum_compiler
