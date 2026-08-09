program test_session_runtime_count_compiler
    ! COUNT over a runtime-shaped logical mask must walk all descriptor elements
    ! for rank one through rank three.  The positive fixture is compared with
    ! an independently compiled gfortran executable; rank four remains a
    ! precise bounded lowering refusal.
    use fortfront_compiler, only: compiler_frontend_options_t, &
        compiler_frontend_result_t, compile_frontend_from_string, &
        INPUT_MODE_STANDARD
    use ffc_test_support, only: compile_to_exe
    use session_program_lowering, only: lower_program_to_liric_exe
    implicit none

    character(len=*), parameter :: source = &
        'program main'//new_line('a')// &
        '  logical, allocatable :: mask(:,:)'//new_line('a')// &
        '  allocate(mask(2,3))'//new_line('a')// &
        '  mask = .false.'//new_line('a')// &
        '  mask(1,1) = .true.'//new_line('a')// &
        '  mask(2,3) = .true.'//new_line('a')// &
        '  call consume(mask)'//new_line('a')// &
        '  call automatic(5)'//new_line('a')// &
        '  call automatic_rank3(2,3,2)'//new_line('a')// &
        'contains'//new_line('a')// &
        '  subroutine consume(a)'//new_line('a')// &
        '    logical, intent(in) :: a(:,:)'//new_line('a')// &
        '    print *, count(a)'//new_line('a')// &
        '  end subroutine consume'//new_line('a')// &
        '  subroutine automatic(n)'//new_line('a')// &
        '    integer, intent(in) :: n'//new_line('a')// &
        '    logical :: a(n)'//new_line('a')// &
        '    a = .false.'//new_line('a')// &
        '    a(1) = .true.'//new_line('a')// &
        '    a(n) = .true.'//new_line('a')// &
        '    print *, count(a)'//new_line('a')// &
        '  end subroutine automatic'//new_line('a')// &
        '  subroutine automatic_rank3(n,m,k)'//new_line('a')// &
        '    integer, intent(in) :: n,m,k'//new_line('a')// &
        '    logical :: a(n,m,k)'//new_line('a')// &
        '    a = .false.'//new_line('a')// &
        '    a(1,1,1) = .true.'//new_line('a')// &
        '    a(n,m,k) = .true.'//new_line('a')// &
        '    a(1,2,1) = .true.'//new_line('a')// &
        '    print *, count(a)'//new_line('a')// &
        '  end subroutine automatic_rank3'//new_line('a')// &
        'end program main'

    character(len=*), parameter :: rank4_source = &
        'program main'//new_line('a')// &
        '  call work(2,2,2,2)'//new_line('a')// &
        'contains'//new_line('a')// &
        '  subroutine work(n,m,k,l)'//new_line('a')// &
        '    integer, intent(in) :: n,m,k,l'//new_line('a')// &
        '    logical :: a(n,m,k,l)'//new_line('a')// &
        '    a = .true.'//new_line('a')// &
        '    print *, count(a)'//new_line('a')// &
        '  end subroutine work'//new_line('a')// &
        'end program main'

    logical :: all_passed

    print *, '=== direct session runtime count compiler test ==='
    all_passed = matches_gfortran(source)
    if (.not. test_rank4_refusal(rank4_source)) all_passed = .false.
    if (.not. all_passed) stop 1
    print *, 'PASS: runtime count matches gfortran and rank-4 remains refused'

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
        base = '/var/tmp/ert/ffc_runtime_count'
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
            print *, 'FAIL: FortFront rejected runtime count source: ', &
                trim(frontend_result%diagnostic_text)
            return
        end if

        call lower_program_to_liric_exe(frontend_result%arena, &
                                        frontend_result%root_index, exe, &
                                        error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL: ffc runtime count lowering failed: ', trim(error_msg)
            return
        end if

        open (newunit=unit, file=src, status='replace', action='write')
        write (unit, '(A)') program_source
        close (unit)
        call execute_command_line('gfortran -w '//src//' -o '//ref, &
                                  exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL: gfortran rejected runtime count source'
            return
        end if

        call execute_command_line(exe//' > '//ffc_out, exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL: ffc runtime count executable failed'
            return
        end if
        call execute_command_line(ref//' > '//ref_out, exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL: gfortran runtime count executable failed'
            return
        end if
        call execute_command_line('diff '//ffc_out//' '//ref_out// &
                                  ' > /dev/null 2>&1', exitstat=diff_status)
        if (diff_status /= 0) then
            print *, 'FAIL: ffc runtime count differs from gfortran'
            call execute_command_line('diff '//ffc_out//' '//ref_out)
            return
        end if

        call execute_command_line('rm -f '//src//' '//exe//' '//ref//' '// &
                                  ffc_out//' '//ref_out)
        ok = .true.
    end function matches_gfortran

    logical function test_rank4_refusal(program_source) result(ok)
        character(len=*), intent(in) :: program_source
        character(len=*), parameter :: base = &
            '/var/tmp/ert/ffc_runtime_count_rank4_refusal'
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
            print *, 'FAIL: gfortran rejected valid rank-4 count fixture'
            call execute_command_line('rm -f '//src//' '//ref//' '//exe)
            return
        end if

        call compile_to_exe(program_source, exe, error_msg)
        call execute_command_line('rm -f '//src//' '//ref//' '//exe)
        if (len_trim(error_msg) == 0) then
            print *, 'FAIL: rank-4 runtime count lowered without a diagnostic'
            return
        end if
        if (index(error_msg, &
            'count over runtime-extent arrays supports rank-1 through rank-3 only') == 0) then
            print *, 'FAIL: rank-4 count refusal diagnostic changed: ', &
                trim(error_msg)
            return
        end if
        ok = .true.
    end function test_rank4_refusal

end program test_session_runtime_count_compiler
