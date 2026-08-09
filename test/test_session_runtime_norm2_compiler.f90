program test_session_runtime_norm2_compiler
    ! NORM2 over runtime-shaped real arrays must traverse every descriptor
    ! element. Positive cases are compared with an independently compiled
    ! gfortran executable; rank, DIM, and KIND boundaries have exact errors.
    use fortfront_compiler, only: compiler_frontend_options_t, &
        compiler_frontend_result_t, compile_frontend_from_string, &
        INPUT_MODE_STANDARD
    use ffc_test_support, only: compile_to_exe
    use session_program_lowering, only: lower_program_to_liric_exe
    implicit none

    character(len=*), parameter :: source = &
        'program main'//new_line('a')// &
        '  call automatic_rank1(5)'//new_line('a')// &
        '  call automatic_rank4(2,2,2,2)'//new_line('a')// &
        '  call assumed_rank4()'//new_line('a')// &
        'contains'//new_line('a')// &
        '  subroutine automatic_rank1(n)'//new_line('a')// &
        '    integer, intent(in) :: n'//new_line('a')// &
        '    real :: values(n)'//new_line('a')// &
        '    values = 0.0'//new_line('a')// &
        '    values(2) = 3.0'//new_line('a')// &
        '    values(5) = 4.0'//new_line('a')// &
        '    print *, norm2(values)'//new_line('a')// &
        '  end subroutine automatic_rank1'//new_line('a')// &
        '  subroutine automatic_rank4(n,m,k,l)'//new_line('a')// &
        '    integer, intent(in) :: n,m,k,l'//new_line('a')// &
        '    real(8) :: values(n,m,k,l)'//new_line('a')// &
        '    values = 0.0d0'//new_line('a')// &
        '    values(2,1,2,2) = 3.0d0'//new_line('a')// &
        '    values(1,2,1,1) = 4.0d0'//new_line('a')// &
        '    print *, norm2(values)'//new_line('a')// &
        '  end subroutine automatic_rank4'//new_line('a')// &
        '  subroutine assumed_rank4()'//new_line('a')// &
        '    real :: values(2,2,2,2)'//new_line('a')// &
        '    values = 0.0'//new_line('a')// &
        '    values(2,2,2,2) = 3.0'//new_line('a')// &
        '    values(1,1,1,1) = 4.0'//new_line('a')// &
        '    call consume(values)'//new_line('a')// &
        '  end subroutine assumed_rank4'//new_line('a')// &
        '  subroutine consume(values)'//new_line('a')// &
        '    real, intent(in) :: values(:,:,:,:)'//new_line('a')// &
        '    print *, norm2(values)'//new_line('a')// &
        '  end subroutine consume'//new_line('a')// &
        'end program main'

    character(len=*), parameter :: rank5_source = &
        'program main'//new_line('a')// &
        '  call work(2,2,2,2,2)'//new_line('a')// &
        'contains'//new_line('a')// &
        '  subroutine work(n,m,k,l,q)'//new_line('a')// &
        '    integer, intent(in) :: n,m,k,l,q'//new_line('a')// &
        '    real :: values(n,m,k,l,q)'//new_line('a')// &
        '    values = 1.0'//new_line('a')// &
        '    print *, norm2(values)'//new_line('a')// &
        '  end subroutine work'//new_line('a')// &
        'end program main'

    character(len=*), parameter :: dim_source = &
        'program main'//new_line('a')// &
        '  call work(2,2,2,2)'//new_line('a')// &
        'contains'//new_line('a')// &
        '  subroutine work(n,m,k,l)'//new_line('a')// &
        '    integer, intent(in) :: n,m,k,l'//new_line('a')// &
        '    real :: values(n,m,k,l)'//new_line('a')// &
        '    values = 1.0'//new_line('a')// &
        '    print *, norm2(values, dim=1)'//new_line('a')// &
        '  end subroutine work'//new_line('a')// &
        'end program main'

    character(len=*), parameter :: kind_source = &
        'program main'//new_line('a')// &
        '  call work(2)'//new_line('a')// &
        'contains'//new_line('a')// &
        '  subroutine work(n)'//new_line('a')// &
        '    integer, intent(in) :: n'//new_line('a')// &
        '    real :: values(n)'//new_line('a')// &
        '    values = 1.0'//new_line('a')// &
        '    print *, norm2(values, kind=8)'//new_line('a')// &
        '  end subroutine work'//new_line('a')// &
        'end program main'

    logical :: all_passed

    print *, '=== direct session runtime norm2 compiler test ==='
    all_passed = matches_gfortran(source)
    if (.not. test_refusal(rank5_source, &
        'runtime-sized array supports ranks 1 through 4 only', &
        '/var/tmp/ert/ffc_runtime_norm2_rank5', .true.)) all_passed = .false.
    if (.not. test_refusal(dim_source, &
        'norm2 supports exactly one real array argument; DIM and KIND forms are not supported', &
        '/var/tmp/ert/ffc_runtime_norm2_dim', .true.)) all_passed = .false.
    if (.not. test_refusal(kind_source, &
        'norm2 supports exactly one real array argument; DIM and KIND forms are not supported', &
        '/var/tmp/ert/ffc_runtime_norm2_kind', .false.)) all_passed = .false.
    if (.not. all_passed) stop 1
    print *, 'PASS: runtime NORM2 matches gfortran; rank/DIM/KIND boundaries refused'

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
        base = '/var/tmp/ert/ffc_runtime_norm2'
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
            print *, 'FAIL: FortFront rejected runtime norm2 source: ', &
                trim(frontend_result%diagnostic_text)
            return
        end if

        call lower_program_to_liric_exe(frontend_result%arena, &
            frontend_result%root_index, exe, &
            error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL: ffc runtime norm2 lowering failed: ', trim(error_msg)
            return
        end if

        open (newunit=unit, file=src, status='replace', action='write')
        write (unit, '(A)') program_source
        close (unit)
        call execute_command_line('gfortran -w '//src//' -o '//ref, &
            exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL: gfortran rejected runtime norm2 source'
            return
        end if

        call execute_command_line(exe//' > '//ffc_out, exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL: ffc runtime norm2 executable failed'
            return
        end if
        call execute_command_line(ref//' > '//ref_out, exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL: gfortran runtime norm2 executable failed'
            return
        end if
        call execute_command_line('diff '//ffc_out//' '//ref_out// &
            ' > /dev/null 2>&1', exitstat=diff_status)
        if (diff_status /= 0) then
            print *, 'FAIL: ffc runtime norm2 differs from gfortran'
            call execute_command_line('diff '//ffc_out//' '//ref_out)
            return
        end if

        call execute_command_line('rm -f '//src//' '//exe//' '//ref//' '// &
            ffc_out//' '//ref_out)
        ok = .true.
    end function matches_gfortran

    logical function test_refusal(program_source, expected, base, &
            gfortran_accepts) result(ok)
        character(len=*), intent(in) :: program_source, expected, base
        logical, intent(in) :: gfortran_accepts
        character(len=:), allocatable :: error_msg
        character(len=:), allocatable :: src, ref, exe
        integer :: unit, exit_stat

        ok = .false.
        src = base//'.f90'
        ref = base//'.gfortran'
        exe = base//'.ffc'
        open (newunit=unit, file=src, status='replace', action='write')
        write (unit, '(A)') program_source
        close (unit)
        call execute_command_line('gfortran -w '//src//' -o '//ref, &
            exitstat=exit_stat)
        if (gfortran_accepts .and. exit_stat /= 0) then
            print *, 'FAIL: gfortran rejected valid norm2 refusal fixture'
            call execute_command_line('rm -f '//src//' '//ref//' '//exe)
            return
        end if
        if (.not. gfortran_accepts .and. exit_stat == 0) then
            print *, 'FAIL: gfortran accepted invalid norm2 KIND fixture'
            call execute_command_line('rm -f '//src//' '//ref//' '//exe)
            return
        end if

        call compile_to_exe(program_source, exe, error_msg)
        call execute_command_line('rm -f '//src//' '//ref//' '//exe)
        if (len_trim(error_msg) == 0) then
            print *, 'FAIL: unsupported runtime norm2 form lowered without a diagnostic'
            return
        end if
        if (index(error_msg, expected) == 0) then
            print *, 'FAIL: norm2 refusal diagnostic changed: ', trim(error_msg)
            return
        end if
        ok = .true.
    end function test_refusal

end program test_session_runtime_norm2_compiler
