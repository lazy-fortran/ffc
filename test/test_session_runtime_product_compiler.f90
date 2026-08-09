program test_session_runtime_product_compiler
    ! Runtime automatic and assumed-shape arrays have no compile-time extent to
    ! unroll. PRODUCT must still walk their complete contiguous storage through
    ! rank three; unsupported rank, element-kind, and DIM forms stay precise.
    use fortfront_compiler, only: compiler_frontend_options_t, &
        compiler_frontend_result_t, compile_frontend_from_string, &
        INPUT_MODE_STANDARD
    use ffc_test_support, only: compile_to_exe
    use session_program_lowering, only: lower_program_to_liric_exe
    implicit none

    character(len=*), parameter :: source = &
        'program main'//new_line('a')// &
        '  integer, allocatable :: input(:,:,:)'//new_line('a')// &
        '  real, allocatable :: real_input(:,:,:)'//new_line('a')// &
        '  integer :: i, j, k'//new_line('a')// &
        '  allocate(input(2,2,2))'//new_line('a')// &
        '  allocate(real_input(2,2,2))'//new_line('a')// &
        '  do j = 1, 2'//new_line('a')// &
        '    do k = 1, 2'//new_line('a')// &
        '      do i = 1, 2'//new_line('a')// &
        '        input(i,k,j) = i + j + k'//new_line('a')// &
        '        real_input(i,k,j) = real(i + j + k)'//new_line('a')// &
        '      end do'//new_line('a')// &
        '    end do'//new_line('a')// &
        '  end do'//new_line('a')// &
        '  call consume(input)'//new_line('a')// &
        '  call consume_real(real_input)'//new_line('a')// &
        '  call automatic(2,2,2)'//new_line('a')// &
        '  call automatic_real(2,2,2)'//new_line('a')// &
        '  call automatic64(3)'//new_line('a')// &
        '  deallocate(input)'//new_line('a')// &
        '  deallocate(real_input)'//new_line('a')// &
        'contains'//new_line('a')// &
        '  subroutine consume(values)'//new_line('a')// &
        '    integer, intent(in) :: values(:,:,:)'//new_line('a')// &
        '    print *, product(values)'//new_line('a')// &
        '  end subroutine consume'//new_line('a')// &
        '  subroutine consume_real(values)'//new_line('a')// &
        '    real, intent(in) :: values(:,:,:)'//new_line('a')// &
        '    print *, product(values)'//new_line('a')// &
        '  end subroutine consume_real'//new_line('a')// &
        '  subroutine automatic(n,m,k)'//new_line('a')// &
        '    integer, intent(in) :: n,m,k'//new_line('a')// &
        '    integer :: values(n,m,k)'//new_line('a')// &
        '    values = 2'//new_line('a')// &
        '    print *, product(values)'//new_line('a')// &
        '  end subroutine automatic'//new_line('a')// &
        '  subroutine automatic_real(n,m,k)'//new_line('a')// &
        '    integer, intent(in) :: n,m,k'//new_line('a')// &
        '    real :: values(n,m,k)'//new_line('a')// &
        '    values = 2.0'//new_line('a')// &
        '    print *, product(values)'//new_line('a')// &
        '  end subroutine automatic_real'//new_line('a')// &
        '  subroutine automatic64(n)'//new_line('a')// &
        '    integer, intent(in) :: n'//new_line('a')// &
        '    real(8) :: values(n)'//new_line('a')// &
        '    values = 3.0d0'//new_line('a')// &
        '    print *, product(values)'//new_line('a')// &
        '  end subroutine automatic64'//new_line('a')// &
        'end program main'

    character(len=*), parameter :: rank4_source = &
        'program main'//new_line('a')// &
        '  call work(2, 2, 2, 2)'//new_line('a')// &
        'contains'//new_line('a')// &
        '  subroutine work(n, m, k, l)'//new_line('a')// &
        '    integer, intent(in) :: n, m, k, l'//new_line('a')// &
        '    integer :: values(n,m,k,l)'//new_line('a')// &
        '    values = 2'//new_line('a')// &
        '    print *, product(values)'//new_line('a')// &
        '  end subroutine work'//new_line('a')// &
        'end program main'

    character(len=*), parameter :: kind_source = &
        'program main'//new_line('a')// &
        '  integer(kind=8), allocatable :: actual(:,:,:)'//new_line('a')// &
        '  allocate(actual(2,2,2))'//new_line('a')// &
        '  actual(1,1,1) = 2_8'//new_line('a')// &
        '  call work(actual)'//new_line('a')// &
        'contains'//new_line('a')// &
        '  subroutine work(values)'//new_line('a')// &
        '    integer(kind=8), intent(in) :: values(:,:,:)'//new_line('a')// &
        '    print *, product(values)'//new_line('a')// &
        '  end subroutine work'//new_line('a')// &
        'end program main'

    character(len=*), parameter :: dim_source = &
        'program main'//new_line('a')// &
        '  call work(2)'//new_line('a')// &
        'contains'//new_line('a')// &
        '  subroutine work(n)'//new_line('a')// &
        '    integer, intent(in) :: n'//new_line('a')// &
        '    integer :: values(n)'//new_line('a')// &
        '    values = 2'//new_line('a')// &
        '    print *, product(values, 1)'//new_line('a')// &
        '  end subroutine work'//new_line('a')// &
        'end program main'

    logical :: all_passed

    print *, '=== direct session runtime product compiler test ==='
    all_passed = matches_gfortran(source)
    if (.not. test_rank4_refusal(rank4_source)) all_passed = .false.
    if (.not. test_refusal(kind_source, &
            'product over runtime-extent arrays supports default integer, real, and real(8) elements only', &
            '/var/tmp/ert/ffc_runtime_product_kind')) all_passed = .false.
    if (.not. test_refusal(dim_source, &
            'product requires exactly one array argument', &
            '/var/tmp/ert/ffc_runtime_product_dim')) all_passed = .false.
    if (.not. all_passed) stop 1
    print *, 'PASS: runtime PRODUCT matches gfortran; unsupported rank/kind/DIM refused'

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
        base = '/var/tmp/ert/ffc_runtime_product'
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
            print *, 'FAIL: FortFront rejected runtime product source: ', &
                trim(frontend_result%diagnostic_text)
            return
        end if

        call lower_program_to_liric_exe(frontend_result%arena, &
            frontend_result%root_index, exe, &
            error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL: ffc runtime product lowering failed: ', &
                trim(error_msg)
            return
        end if

        open (newunit=unit, file=src, status='replace', action='write')
        write (unit, '(A)') program_source
        close (unit)
        call execute_command_line('gfortran -w '//src//' -o '//ref, &
            exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL: gfortran rejected runtime product source'
            return
        end if

        call execute_command_line(exe//' > '//ffc_out, exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL: ffc runtime product executable failed'
            return
        end if
        call execute_command_line(ref//' > '//ref_out, exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL: gfortran runtime product executable failed'
            return
        end if
        call execute_command_line('diff '//ffc_out//' '//ref_out// &
            ' > /dev/null 2>&1', &
            exitstat=diff_status)
        if (diff_status /= 0) then
            print *, 'FAIL: ffc runtime product differs from gfortran'
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
            '/var/tmp/ert/ffc_runtime_product_rank4_refusal'
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
            print *, 'FAIL: gfortran rejected valid rank-4 product refusal fixture'
            call execute_command_line('rm -f '//src//' '//ref//' '//exe)
            return
        end if

        call compile_to_exe(program_source, exe, error_msg)
        call execute_command_line('rm -f '//src//' '//ref//' '//exe)
        if (len_trim(error_msg) == 0) then
            print *, 'FAIL: rank-4 runtime product lowered without a diagnostic'
            return
        end if
        if (index(error_msg, &
            'product over runtime-extent arrays supports rank-1 through rank-3 only') == 0) then
            print *, 'FAIL: rank-4 product refusal diagnostic changed: ', &
                trim(error_msg)
            return
        end if
        ok = .true.
    end function test_rank4_refusal

    logical function test_refusal(program_source, expected, base) result(ok)
        character(len=*), intent(in) :: program_source, expected, base
        character(len=:), allocatable :: error_msg, src, ref, exe
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
            print *, 'FAIL: gfortran rejected valid PRODUCT refusal fixture'
            call execute_command_line('rm -f '//src//' '//ref//' '//exe)
            return
        end if

        call compile_to_exe(program_source, exe, error_msg)
        call execute_command_line('rm -f '//src//' '//ref//' '//exe)
        if (len_trim(error_msg) == 0) then
            print *, 'FAIL: unsupported PRODUCT form lowered without a diagnostic'
            return
        end if
        if (index(error_msg, expected) == 0) then
            print *, 'FAIL: PRODUCT refusal diagnostic changed: ', trim(error_msg)
            return
        end if
        ok = .true.
    end function test_refusal

end program test_session_runtime_product_compiler
