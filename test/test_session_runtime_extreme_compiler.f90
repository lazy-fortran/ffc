program test_session_runtime_extreme_compiler
    ! Runtime reductions over descriptor-backed arrays must traverse the
    ! complete contiguous storage. The positive fixtures are compared with an
    ! independently compiled gfortran executable across rank-3 and rank-4
    ! automatic and assumed-shape dummies; unsupported rank/kind/form
    ! combinations retain precise diagnostics.
    use fortfront_compiler, only: compiler_frontend_options_t, &
        compiler_frontend_result_t, compile_frontend_from_string, &
        INPUT_MODE_STANDARD
    use ffc_test_support, only: compile_to_exe
    use session_program_lowering, only: lower_program_to_liric_exe
    implicit none

    character(len=*), parameter :: source = &
        'program main'//new_line('a')// &
        '  integer, allocatable :: one(:), two(:,:)'//new_line('a')// &
        '  integer :: i, j'//new_line('a')// &
        '  allocate(one(3), two(2,3))'//new_line('a')// &
        '  one = [7, -2, 5]'//new_line('a')// &
        '  do j = 1, 3'//new_line('a')// &
        '    do i = 1, 2'//new_line('a')// &
        '      two(i,j) = 10*j + i'//new_line('a')// &
        '    end do'//new_line('a')// &
        '  end do'//new_line('a')// &
        '  call consume_one(one)'//new_line('a')// &
        '  call consume_two(two)'//new_line('a')// &
        '  call automatic(4)'//new_line('a')// &
        '  call automatic64(3)'//new_line('a')// &
        'contains'//new_line('a')// &
        '  subroutine consume_one(values)'//new_line('a')// &
        '    integer, intent(in) :: values(:)'//new_line('a')// &
        '    print *, maxval(values), minval(values)'//new_line('a')// &
        '  end subroutine consume_one'//new_line('a')// &
        '  subroutine consume_two(values)'//new_line('a')// &
        '    integer, intent(in) :: values(:,:)'//new_line('a')// &
        '    print *, maxval(values), minval(values)'//new_line('a')// &
        '  end subroutine consume_two'//new_line('a')// &
        '  subroutine automatic(n)'//new_line('a')// &
        '    integer, intent(in) :: n'//new_line('a')// &
        '    real :: values(n)'//new_line('a')// &
        '    values = 2.0'//new_line('a')// &
        '    values(2) = -5.0'//new_line('a')// &
        '    values(4) = 7.0'//new_line('a')// &
        '    print *, maxval(values), minval(values)'//new_line('a')// &
        '  end subroutine automatic'//new_line('a')// &
        '  subroutine automatic64(n)'//new_line('a')// &
        '    integer, intent(in) :: n'//new_line('a')// &
        '    real(8) :: values(n)'//new_line('a')// &
        '    values = 3.0d0'//new_line('a')// &
        '    values(2) = -8.0d0'//new_line('a')// &
        '    values(3) = 4.0d0'//new_line('a')// &
        '    print *, maxval(values), minval(values)'//new_line('a')// &
        '  end subroutine automatic64'//new_line('a')// &
        'end program main'

    character(len=*), parameter :: rank3_source = &
        'program main'//new_line('a')// &
        '  integer, allocatable :: actual(:,:,:)'//new_line('a')// &
        '  integer :: i, j, k'//new_line('a')// &
        '  allocate(actual(2,3,2))'//new_line('a')// &
        '  do k = 1, 2'//new_line('a')// &
        '    do j = 1, 3'//new_line('a')// &
        '      do i = 1, 2'//new_line('a')// &
        '        actual(i,j,k) = 100*k + 10*j + i'//new_line('a')// &
        '      end do'//new_line('a')// &
        '    end do'//new_line('a')// &
        '  end do'//new_line('a')// &
        '  call consume(actual)'//new_line('a')// &
        '  call automatic(2, 2, 2)'//new_line('a')// &
        'contains'//new_line('a')// &
        '  subroutine consume(values)'//new_line('a')// &
        '    integer, intent(in) :: values(:,:,:)'//new_line('a')// &
        '    print *, maxval(values), minval(values), sum(values)'//new_line('a')// &
        '  end subroutine consume'//new_line('a')// &
        '  subroutine automatic(n, m, k)'//new_line('a')// &
        '    integer, intent(in) :: n, m, k'//new_line('a')// &
        '    real :: values(n,m,k)'//new_line('a')// &
        '    values = 3.0'//new_line('a')// &
        '    values(2,1,2) = -9.0'//new_line('a')// &
        '    values(1,2,1) = 8.0'//new_line('a')// &
        '    print *, maxval(values), minval(values), sum(values)'//new_line('a')// &
        '  end subroutine automatic'//new_line('a')// &
        'end program main'

    character(len=*), parameter :: rank4_sum_source = &
        'program main'//new_line('a')// &
        '  integer, allocatable :: actual(:,:,:,:)'//new_line('a')// &
        '  allocate(actual(2, 2, 2, 2))'//new_line('a')// &
        '  actual = 2'//new_line('a')// &
        '  call consume(actual)'//new_line('a')// &
        '  call work(2, 2, 2, 2)'//new_line('a')// &
        'contains'//new_line('a')// &
        '  subroutine consume(values)'//new_line('a')// &
        '    integer, intent(in) :: values(:,:,:,:)'//new_line('a')// &
        '    print *, sum(values)'//new_line('a')// &
        '  end subroutine consume'//new_line('a')// &
        '  subroutine work(n, m, k, l)'//new_line('a')// &
        '    integer, intent(in) :: n, m, k, l'//new_line('a')// &
        '    integer :: values(n,m,k,l)'//new_line('a')// &
        '    values = 2'//new_line('a')// &
        '    print *, sum(values)'//new_line('a')// &
        '  end subroutine work'//new_line('a')// &
        'end program main'

    character(len=*), parameter :: rank4_extreme_source = &
        'program main'//new_line('a')// &
        '  integer, allocatable :: actual(:,:,:,:)'//new_line('a')// &
        '  real, allocatable :: real_actual(:,:,:,:)'//new_line('a')// &
        '  real(8), allocatable :: double_actual(:,:,:,:)'//new_line('a')// &
        '  allocate(actual(2,2,2,2), real_actual(2,2,2,2), '// &
        'double_actual(2,2,2,2))'//new_line('a')// &
        '  actual = 2'//new_line('a')// &
        '  actual(2,1,2,2) = -12'//new_line('a')// &
        '  actual(1,2,1,1) = 17'//new_line('a')// &
        '  real_actual = 2.0'//new_line('a')// &
        '  real_actual(2,1,2,2) = -9.0'//new_line('a')// &
        '  real_actual(1,2,1,1) = 8.0'//new_line('a')// &
        '  double_actual = 3.0d0'//new_line('a')// &
        '  double_actual(2,1,2,2) = -11.0d0'//new_line('a')// &
        '  double_actual(1,2,1,1) = 13.0d0'//new_line('a')// &
        '  call consume(actual)'//new_line('a')// &
        '  call consume_real(real_actual)'//new_line('a')// &
        '  call consume_double(double_actual)'//new_line('a')// &
        '  call automatic(2,2,2,2)'//new_line('a')// &
        '  call automatic_real(2,2,2,2)'//new_line('a')// &
        '  call automatic_double(2,2,2,2)'//new_line('a')// &
        '  call empty4(0)'//new_line('a')// &
        'contains'//new_line('a')// &
        '  subroutine consume(values)'//new_line('a')// &
        '    integer, intent(in) :: values(:,:,:,:)'//new_line('a')// &
        '    print *, maxval(values), minval(values)'//new_line('a')// &
        '  end subroutine consume'//new_line('a')// &
        '  subroutine consume_real(values)'//new_line('a')// &
        '    real, intent(in) :: values(:,:,:,:)'//new_line('a')// &
        '    print *, maxval(values), minval(values)'//new_line('a')// &
        '  end subroutine consume_real'//new_line('a')// &
        '  subroutine consume_double(values)'//new_line('a')// &
        '    real(8), intent(in) :: values(:,:,:,:)'//new_line('a')// &
        '    print *, maxval(values), minval(values)'//new_line('a')// &
        '  end subroutine consume_double'//new_line('a')// &
        '  subroutine automatic(n,m,k,l)'//new_line('a')// &
        '    integer, intent(in) :: n,m,k,l'//new_line('a')// &
        '    integer :: values(n,m,k,l)'//new_line('a')// &
        '    values = 2'//new_line('a')// &
        '    values(2,1,2,2) = -10'//new_line('a')// &
        '    values(1,2,1,1) = 19'//new_line('a')// &
        '    print *, maxval(values), minval(values)'//new_line('a')// &
        '  end subroutine automatic'//new_line('a')// &
        '  subroutine automatic_real(n,m,k,l)'//new_line('a')// &
        '    integer, intent(in) :: n,m,k,l'//new_line('a')// &
        '    real :: values(n,m,k,l)'//new_line('a')// &
        '    values = 2.0'//new_line('a')// &
        '    values(2,1,2,2) = -7.0'//new_line('a')// &
        '    values(1,2,1,1) = 6.0'//new_line('a')// &
        '    print *, maxval(values), minval(values)'//new_line('a')// &
        '  end subroutine automatic_real'//new_line('a')// &
        '  subroutine automatic_double(n,m,k,l)'//new_line('a')// &
        '    integer, intent(in) :: n,m,k,l'//new_line('a')// &
        '    real(8) :: values(n,m,k,l)'//new_line('a')// &
        '    values = 3.0d0'//new_line('a')// &
        '    values(2,1,2,2) = -5.0d0'//new_line('a')// &
        '    values(1,2,1,1) = 14.0d0'//new_line('a')// &
        '    print *, maxval(values), minval(values)'//new_line('a')// &
        '  end subroutine automatic_double'//new_line('a')// &
        '  subroutine empty4(n)'//new_line('a')// &
        '    integer, intent(in) :: n'//new_line('a')// &
        '    integer :: values(n,2,2,2)'//new_line('a')// &
        '    print *, maxval(values), minval(values)'//new_line('a')// &
        '  end subroutine empty4'//new_line('a')// &
        'end program main'

    character(len=*), parameter :: rank5_product_source = &
        'program main'//new_line('a')// &
        '  call work(2, 2, 2, 2, 2)'//new_line('a')// &
        'contains'//new_line('a')// &
        '  subroutine work(n, m, k, l, q)'//new_line('a')// &
        '    integer, intent(in) :: n, m, k, l, q'//new_line('a')// &
        '    integer :: values(n,m,k,l,q)'//new_line('a')// &
        '    print *, product(values)'//new_line('a')// &
        '  end subroutine work'//new_line('a')// &
        'end program main'

    character(len=*), parameter :: rank4_kind_source = &
        'program main'//new_line('a')// &
        '  integer(8), allocatable :: actual(:,:,:,:)'//new_line('a')// &
        '  allocate(actual(2,2,2,2))'//new_line('a')// &
        '  call work(actual)'//new_line('a')// &
        'contains'//new_line('a')// &
        '  subroutine work(values)'//new_line('a')// &
        '    integer(8), intent(in) :: values(:,:,:,:)'//new_line('a')// &
        '    print *, maxval(values)'//new_line('a')// &
        '  end subroutine work'//new_line('a')// &
        'end program main'

    character(len=*), parameter :: rank4_min_kind_source = &
        'program main'//new_line('a')// &
        '  integer(8), allocatable :: actual(:,:,:,:)'//new_line('a')// &
        '  allocate(actual(2,2,2,2))'//new_line('a')// &
        '  call work(actual)'//new_line('a')// &
        'contains'//new_line('a')// &
        '  subroutine work(values)'//new_line('a')// &
        '    integer(8), intent(in) :: values(:,:,:,:)'//new_line('a')// &
        '    print *, minval(values)'//new_line('a')// &
        '  end subroutine work'//new_line('a')// &
        'end program main'

    character(len=*), parameter :: rank3_form_source = &
        'program main'//new_line('a')// &
        '  call work(2, 2, 2)'//new_line('a')// &
        'contains'//new_line('a')// &
        '  subroutine work(n, m, k)'//new_line('a')// &
        '    integer, intent(in) :: n, m, k'//new_line('a')// &
        '    real :: values(n,m,k)'//new_line('a')// &
        '    values = 2.0'//new_line('a')// &
        '    print *, sum(values, 1)'//new_line('a')// &
        '  end subroutine work'//new_line('a')// &
        'end program main'

    character(len=*), parameter :: rank4_max_dim_source = &
        'program main'//new_line('a')// &
        '  call work(2,2,2,2)'//new_line('a')// &
        'contains'//new_line('a')// &
        '  subroutine work(n,m,k,l)'//new_line('a')// &
        '    integer, intent(in) :: n,m,k,l'//new_line('a')// &
        '    real :: values(n,m,k,l)'//new_line('a')// &
        '    values = 2.0'//new_line('a')// &
        '    print *, maxval(values, 1)'//new_line('a')// &
        '  end subroutine work'//new_line('a')// &
        'end program main'

    character(len=*), parameter :: rank4_min_dim_source = &
        'program main'//new_line('a')// &
        '  call work(2,2,2,2)'//new_line('a')// &
        'contains'//new_line('a')// &
        '  subroutine work(n,m,k,l)'//new_line('a')// &
        '    integer, intent(in) :: n,m,k,l'//new_line('a')// &
        '    real :: values(n,m,k,l)'//new_line('a')// &
        '    values = 2.0'//new_line('a')// &
        '    print *, minval(values, 1)'//new_line('a')// &
        '  end subroutine work'//new_line('a')// &
        'end program main'

    character(len=*), parameter :: rank5_max_source = &
        'program main'//new_line('a')// &
        '  call work(2, 2, 2, 2, 2)'//new_line('a')// &
        'contains'//new_line('a')// &
        '  subroutine work(n, m, k, l, q)'//new_line('a')// &
        '    integer, intent(in) :: n, m, k, l, q'//new_line('a')// &
        '    real :: values(n,m,k,l,q)'//new_line('a')// &
        '    values = 2.0'//new_line('a')// &
        '    print *, maxval(values)'//new_line('a')// &
        '  end subroutine work'//new_line('a')// &
        'end program main'

    character(len=*), parameter :: rank5_min_source = &
        'program main'//new_line('a')// &
        '  call work(2,2,2,2,2)'//new_line('a')// &
        'contains'//new_line('a')// &
        '  subroutine work(n,m,k,l,q)'//new_line('a')// &
        '    integer, intent(in) :: n,m,k,l,q'//new_line('a')// &
        '    real :: values(n,m,k,l,q)'//new_line('a')// &
        '    values = 2.0'//new_line('a')// &
        '    print *, minval(values)'//new_line('a')// &
        '  end subroutine work'//new_line('a')// &
        'end program main'

    character(len=*), parameter :: rank3_assumed_count_source = &
        'program main'//new_line('a')// &
        '  logical, allocatable :: mask(:,:,:)'//new_line('a')// &
        '  allocate(mask(2,2,2))'//new_line('a')// &
        '  mask = .false.'//new_line('a')// &
        '  call work(mask)'//new_line('a')// &
        'contains'//new_line('a')// &
        '  subroutine work(values)'//new_line('a')// &
        '    logical, intent(in) :: values(:,:,:)'//new_line('a')// &
        '    print *, count(values)'//new_line('a')// &
        '  end subroutine work'//new_line('a')// &
        'end program main'

    character(len=*), parameter :: empty_source = &
        'program main'//new_line('a')// &
        '  call work(0)'//new_line('a')// &
        'contains'//new_line('a')// &
        '  subroutine work(n)'//new_line('a')// &
        '    integer, intent(in) :: n'//new_line('a')// &
        '    integer :: values(n)'//new_line('a')// &
        '    print *, maxval(values), minval(values)'//new_line('a')// &
        '  end subroutine work'//new_line('a')// &
        'end program main'

    logical :: all_passed

    print *, '=== direct session runtime reduction compiler test ==='
    all_passed = matches_gfortran(source, 'filled')
    if (.not. matches_gfortran(empty_source, 'empty')) &
        all_passed = .false.
    if (.not. matches_gfortran(rank3_source, 'rank3')) &
        all_passed = .false.
    if (.not. matches_gfortran(rank4_sum_source, 'rank4_sum')) &
        all_passed = .false.
    if (.not. matches_gfortran(rank4_extreme_source, 'rank4_extreme')) &
        all_passed = .false.
    if (.not. test_runtime_refusal(rank5_product_source, 'product_rank5', &
        'runtime-sized array supports ranks 1 through 4 only')) &
        all_passed = .false.
    if (.not. test_runtime_refusal(rank5_max_source, 'maxval_rank5', &
        'runtime-sized array supports ranks 1 through 4 only')) &
        all_passed = .false.
    if (.not. test_runtime_refusal(rank5_min_source, 'minval_rank5', &
        'runtime-sized array supports ranks 1 through 4 only')) &
        all_passed = .false.
    if (.not. test_runtime_refusal(rank4_kind_source, 'maxval_kind', &
        'maxval over runtime-extent arrays supports default integer, real, and real(8) elements only')) &
        all_passed = .false.
    if (.not. test_runtime_refusal(rank4_min_kind_source, 'minval_kind', &
        'minval over runtime-extent arrays supports default integer, real, and real(8) elements only')) &
        all_passed = .false.
    if (.not. test_runtime_refusal(rank3_form_source, 'sum_form', &
        'sum requires exactly one array argument')) &
        all_passed = .false.
    if (.not. test_runtime_refusal(rank4_max_dim_source, 'maxval_dim', &
        'maxval requires exactly one array argument')) &
        all_passed = .false.
    if (.not. test_runtime_refusal(rank4_min_dim_source, 'minval_dim', &
        'minval requires exactly one array argument')) &
        all_passed = .false.
    if (.not. matches_gfortran(rank3_assumed_count_source, 'count_rank3')) &
        all_passed = .false.
    if (.not. all_passed) stop 1
    print *, 'PASS: runtime MAXVAL/MINVAL match gfortran through rank four; rank/kind/DIM boundaries refused'

contains

    logical function matches_gfortran(program_source, stem) result(ok)
        character(len=*), intent(in) :: program_source, stem
        type(compiler_frontend_options_t) :: options
        type(compiler_frontend_result_t) :: frontend_result
        character(len=:), allocatable :: error_msg
        character(len=:), allocatable :: base, src, exe, ref
        character(len=:), allocatable :: ffc_out, ref_out
        integer :: unit, exit_stat, diff_status

        ok = .false.
        base = '/var/tmp/ert/ffc_runtime_extreme_'//trim(stem)
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
            print *, 'FAIL: FortFront rejected runtime maxval/minval source: ', &
                trim(frontend_result%diagnostic_text)
            return
        end if

        call lower_program_to_liric_exe(frontend_result%arena, &
            frontend_result%root_index, exe, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL: ffc runtime maxval/minval lowering failed: ', &
                trim(error_msg)
            return
        end if

        open (newunit=unit, file=src, status='replace', action='write')
        write (unit, '(A)') program_source
        close (unit)
        call execute_command_line('gfortran -w '//src//' -o '//ref, &
            exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL: gfortran rejected runtime maxval/minval source'
            return
        end if

        call execute_command_line(exe//' > '//ffc_out, exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL: ffc runtime maxval/minval executable failed'
            return
        end if
        call execute_command_line(ref//' > '//ref_out, exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL: gfortran runtime maxval/minval executable failed'
            return
        end if
        call execute_command_line('diff '//ffc_out//' '//ref_out// &
            ' > /dev/null 2>&1', &
            exitstat=diff_status)
        if (diff_status /= 0) then
            print *, 'FAIL: ffc runtime maxval/minval differs from gfortran'
            call execute_command_line('diff '//ffc_out//' '//ref_out)
            return
        end if

        call execute_command_line('rm -f '//src//' '//exe//' '//ref//' '// &
            ffc_out//' '//ref_out)
        ok = .true.
    end function matches_gfortran

    logical function test_runtime_refusal(program_source, intrinsic_name, &
            expected_message) result(ok)
        character(len=*), intent(in) :: program_source, intrinsic_name
        character(len=*), intent(in) :: expected_message
        character(len=:), allocatable :: error_msg
        character(len=:), allocatable :: base, src, ref, exe
        integer :: unit, exit_stat

        ok = .false.
        base = '/var/tmp/ert/ffc_runtime_extreme_refusal_'//trim(intrinsic_name)
        src = base//'.f90'
        ref = base//'.gfortran'
        exe = base//'.ffc'
        open (newunit=unit, file=src, status='replace', action='write')
        write (unit, '(A)') program_source
        close (unit)
        call execute_command_line('gfortran -w '//src//' -o '//ref, &
            exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL: gfortran rejected valid rank-3 '//trim(intrinsic_name)// &
                ' refusal fixture'
            call execute_command_line('rm -f '//src//' '//ref//' '//exe)
            return
        end if

        call compile_to_exe(program_source, exe, error_msg)
        call execute_command_line('rm -f '//src//' '//ref//' '//exe)
        if (len_trim(error_msg) == 0) then
            print *, 'FAIL: refused runtime '//trim(intrinsic_name)// &
                ' lowered without a diagnostic'
            return
        end if
        if (index(error_msg, trim(expected_message)) == 0) then
            print *, 'FAIL: runtime '//trim(intrinsic_name)// &
                ' refusal diagnostic changed: ', trim(error_msg)
            return
        end if
        ok = .true.
    end function test_runtime_refusal

end program test_session_runtime_extreme_compiler
