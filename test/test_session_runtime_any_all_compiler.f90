program test_session_runtime_any_all_compiler
    ! Scalar ANY/ALL over a bare logical runtime array must walk every element
    ! for both automatic arrays and assumed-shape dummies.  The positive case
    ! is compared with an independently compiled gfortran executable.
    use fortfront_compiler, only: compiler_frontend_options_t, &
        compiler_frontend_result_t, compile_frontend_from_string, &
        INPUT_MODE_STANDARD
    use ffc_test_support, only: compile_to_exe
    use session_program_lowering, only: lower_program_to_liric_exe
    implicit none

    character(len=*), parameter :: source = &
        'program main'//new_line('a')// &
        '  logical, allocatable :: mask(:,:,:)'//new_line('a')// &
        '  allocate(mask(2,3,2))'//new_line('a')// &
        '  mask = .false.'//new_line('a')// &
        '  mask(1,1,1) = .true.'//new_line('a')// &
        '  mask(2,3,2) = .true.'//new_line('a')// &
        '  call consume(mask)'//new_line('a')// &
        '  call automatic(2,3,2)'//new_line('a')// &
        'contains'//new_line('a')// &
        '  subroutine consume(a)'//new_line('a')// &
        '    logical, intent(in) :: a(:,:,:)'//new_line('a')// &
        '    print *, any(a), all(a)'//new_line('a')// &
        '  end subroutine consume'//new_line('a')// &
        '  subroutine automatic(n,m,k)'//new_line('a')// &
        '    integer, intent(in) :: n,m,k'//new_line('a')// &
        '    logical :: a(n,m,k)'//new_line('a')// &
        '    a = .true.'//new_line('a')// &
        '    a(1,1,1) = .false.'//new_line('a')// &
        '    print *, any(a), all(a)'//new_line('a')// &
        '  end subroutine automatic'//new_line('a')// &
        'end program main'

    character(len=*), parameter :: rank4_source = &
        'program main'//new_line('a')// &
        '  call work(2,2,2,2)'//new_line('a')// &
        'contains'//new_line('a')// &
        '  subroutine work(n,m,k,l)'//new_line('a')// &
        '    integer, intent(in) :: n,m,k,l'//new_line('a')// &
        '    logical :: a(n,m,k,l)'//new_line('a')// &
        '    a = .true.'//new_line('a')// &
        '    print *, any(a)'//new_line('a')// &
        '  end subroutine work'//new_line('a')// &
        'end program main'

    character(len=*), parameter :: dim_source = &
        'program main'//new_line('a')// &
        '  call work(2,2,2)'//new_line('a')// &
        'contains'//new_line('a')// &
        '  subroutine work(n,m,k)'//new_line('a')// &
        '    integer, intent(in) :: n,m,k'//new_line('a')// &
        '    logical :: a(n,m,k)'//new_line('a')// &
        '    a = .true.'//new_line('a')// &
        '    print *, any(a,1)'//new_line('a')// &
        '  end subroutine work'//new_line('a')// &
        'end program main'

    logical :: all_passed

    print *, '=== direct session runtime any/all compiler test ==='
    all_passed = matches_gfortran(source)
    if (.not. test_refusal(rank4_source, &
            'any over runtime-extent arrays supports rank-1 through rank-3 only', &
            '/var/tmp/ert/ffc_runtime_any_rank4')) all_passed = .false.
    if (.not. test_refusal(dim_source, &
            'any requires exactly one logical array argument; DIM and MASK forms are not supported', &
            '/var/tmp/ert/ffc_runtime_any_dim')) all_passed = .false.
    if (.not. all_passed) stop 1
    print *, 'PASS: runtime any/all match gfortran and unsupported forms are refused'

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
        base = '/var/tmp/ert/ffc_runtime_any_all'
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
            print *, 'FAIL: FortFront rejected runtime any/all source: ', &
                trim(frontend_result%diagnostic_text)
            return
        end if

        call lower_program_to_liric_exe(frontend_result%arena, &
                                        frontend_result%root_index, exe, &
                                        error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL: ffc runtime any/all lowering failed: ', &
                trim(error_msg)
            return
        end if

        open (newunit=unit, file=src, status='replace', action='write')
        write (unit, '(A)') program_source
        close (unit)
        call execute_command_line('gfortran -w '//src//' -o '//ref, &
                                  exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL: gfortran rejected runtime any/all source'
            return
        end if

        call execute_command_line(exe//' > '//ffc_out, exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL: ffc runtime any/all executable failed'
            return
        end if
        call execute_command_line(ref//' > '//ref_out, exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL: gfortran runtime any/all executable failed'
            return
        end if
        call execute_command_line('diff '//ffc_out//' '//ref_out// &
                                  ' > /dev/null 2>&1', exitstat=diff_status)
        if (diff_status /= 0) then
            print *, 'FAIL: ffc runtime any/all differs from gfortran'
            call execute_command_line('diff '//ffc_out//' '//ref_out)
            return
        end if

        call execute_command_line('rm -f '//src//' '//exe//' '//ref//' '// &
                                  ffc_out//' '//ref_out)
        ok = .true.
    end function matches_gfortran

    logical function test_refusal(program_source, expected, base) result(ok)
        character(len=*), intent(in) :: program_source, expected, base
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
        if (exit_stat /= 0) then
            print *, 'FAIL: gfortran rejected valid refusal fixture'
            call execute_command_line('rm -f '//src//' '//ref//' '//exe)
            return
        end if

        call compile_to_exe(program_source, exe, error_msg)
        call execute_command_line('rm -f '//src//' '//ref//' '//exe)
        if (len_trim(error_msg) == 0) then
            print *, 'FAIL: unsupported any/all form lowered without a diagnostic'
            return
        end if
        if (index(error_msg, expected) == 0) then
            print *, 'FAIL: any/all refusal diagnostic changed: ', trim(error_msg)
            return
        end if
        ok = .true.
    end function test_refusal

end program test_session_runtime_any_all_compiler
