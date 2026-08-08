program test_session_alloc_rank2_component_compiler
    ! Rank-two intrinsic allocatable components through the direct LIRIC session.
    ! The positive case is compared against a separately compiled gfortran
    ! executable; the negative cases lock the deliberately narrow contract.
    use ffc_test_support, only: expect_error_contains
    use fortfront_compiler, only: compiler_frontend_options_t, &
        compiler_frontend_result_t, compile_frontend_from_string, &
        INPUT_MODE_STANDARD
    use session_program_lowering, only: lower_program_to_liric_exe
    implicit none

    logical :: all_passed

    print *, '=== direct session rank-2 allocatable component test ==='
    all_passed = .true.
    if (.not. test_runtime_lifecycle()) all_passed = .false.
    if (.not. test_whole_component_assignment_rejected()) all_passed = .false.
    if (.not. test_actual_argument_rejected()) all_passed = .false.
    if (.not. test_alias_rejected()) all_passed = .false.
    if (.not. test_unsupported_kind_rejected()) all_passed = .false.
    if (.not. test_rank5_rejected()) all_passed = .false.
    if (.not. all_passed) stop 1
    print *, 'PASS: rank-2 intrinsic allocatable components lower through LIRIC'

contains

    logical function test_runtime_lifecycle()
        ! Runtime extents, allocated/size inquiries, element reads/writes, and
        ! deallocation are all checked by differential execution against
        ! gfortran. Integer, real, and logical descriptors share the same type
        ! layout but use their native element widths and loads/stores.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: box_t'//new_line('a')// &
            '    integer, allocatable :: iv(:,:)'//new_line('a')// &
            '    real, allocatable :: rv(:,:)'//new_line('a')// &
            '    logical, allocatable :: lv(:,:)'//new_line('a')// &
            '  end type box_t'//new_line('a')// &
            '  type(box_t) :: x'//new_line('a')// &
            '  integer :: i, j, m, n, isum, lsum'//new_line('a')// &
            '  real :: rsum'//new_line('a')// &
            '  m = 2'//new_line('a')// &
            '  n = 3'//new_line('a')// &
            '  if (allocated(x%iv)) error stop 1'//new_line('a')// &
            '  if (allocated(x%rv)) error stop 2'//new_line('a')// &
            '  if (allocated(x%lv)) error stop 3'//new_line('a')// &
            '  allocate(x%iv(m,n))'//new_line('a')// &
            '  allocate(x%rv(m,n))'//new_line('a')// &
            '  allocate(x%lv(m,n))'//new_line('a')// &
            '  if (.not. allocated(x%iv)) error stop 4'//new_line('a')// &
            '  if (.not. allocated(x%rv)) error stop 5'//new_line('a')// &
            '  if (.not. allocated(x%lv)) error stop 6'//new_line('a')// &
            '  if (size(x%iv) /= 6) error stop 7'//new_line('a')// &
            '  if (size(x%iv, 1) /= m) error stop 8'//new_line('a')// &
            '  if (size(x%iv, 2) /= n) error stop 9'//new_line('a')// &
            '  if (size(x%rv, 1) /= m .or. size(x%rv, 2) /= n) error stop 10'// &
            new_line('a')// &
            '  if (size(x%lv) /= m*n) error stop 11'//new_line('a')// &
            '  do j = 1, n'//new_line('a')// &
            '    do i = 1, m'//new_line('a')// &
            '      x%iv(i,j) = 10*i + j'//new_line('a')// &
            '      x%rv(i,j) = real(i) + 0.25*real(j)'//new_line('a')// &
            '      x%lv(i,j) = mod(i+j, 2) == 0'//new_line('a')// &
            '    end do'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  isum = x%iv(1,1) + x%iv(2,1) + x%iv(1,2) + x%iv(2,2) + '// &
            'x%iv(1,3) + x%iv(2,3)'//new_line('a')// &
            '  rsum = x%rv(1,1) + x%rv(2,1) + x%rv(1,2) + x%rv(2,2) + '// &
            'x%rv(1,3) + x%rv(2,3)'//new_line('a')// &
            '  lsum = 0'//new_line('a')// &
            '  do j = 1, n'//new_line('a')// &
            '    do i = 1, m'//new_line('a')// &
            '      if (x%lv(i,j)) lsum = lsum + 1'//new_line('a')// &
            '    end do'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  print *, size(x%iv), size(x%iv,1), size(x%iv,2), isum'// &
            new_line('a')// &
            '  print *, rsum, lsum, x%iv(2,3), x%lv(1,1)'//new_line('a')// &
            '  deallocate(x%iv)'//new_line('a')// &
            '  deallocate(x%rv)'//new_line('a')// &
            '  deallocate(x%lv)'//new_line('a')// &
            '  if (allocated(x%iv)) error stop 12'//new_line('a')// &
            '  if (allocated(x%rv)) error stop 13'//new_line('a')// &
            '  if (allocated(x%lv)) error stop 14'//new_line('a')// &
            'end program main'

        test_runtime_lifecycle = matches_gfortran(source, 'runtime_lifecycle')
    end function test_runtime_lifecycle

    logical function test_whole_component_assignment_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: t'//new_line('a')// &
            '    integer, allocatable :: a(:,:)'//new_line('a')// &
            '  end type t'//new_line('a')// &
            '  type(t) :: x'//new_line('a')// &
            '  x%a = 1'//new_line('a')// &
            'end program main'

        test_whole_component_assignment_rejected = expect_error_contains(source, &
            'whole-component assignment supports rank-1 components only', &
            '/tmp/ffc_alloc_rank2_component_whole_reject')
    end function test_whole_component_assignment_rejected

    logical function test_actual_argument_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: t'//new_line('a')// &
            '    integer, allocatable :: a(:,:)'//new_line('a')// &
            '  end type t'//new_line('a')// &
            '  type(t) :: x'//new_line('a')// &
            '  call consume(x%a)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine consume(a)'//new_line('a')// &
            '    integer :: a(:,:)'//new_line('a')// &
            '  end subroutine consume'//new_line('a')// &
            'end program main'

        test_actual_argument_rejected = expect_error_contains(source, &
            'passing a rank-2 allocatable component as an actual argument', &
            '/tmp/ffc_alloc_rank2_component_actual_reject')
    end function test_actual_argument_rejected

    logical function test_alias_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: t'//new_line('a')// &
            '    integer, allocatable :: a(:,:)'//new_line('a')// &
            '  end type t'//new_line('a')// &
            '  type(t) :: x'//new_line('a')// &
            '  associate(y => x%a)'//new_line('a')// &
            '    print *, allocated(y)'//new_line('a')// &
            '  end associate'//new_line('a')// &
            'end program main'

        test_alias_rejected = expect_error_contains(source, &
            'rank-2 allocatable array component aliases are not supported', &
            '/tmp/ffc_alloc_rank2_component_alias_reject')
    end function test_alias_rejected

    logical function test_unsupported_kind_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: t'//new_line('a')// &
            '    complex, allocatable :: a(:,:)'//new_line('a')// &
            '  end type t'//new_line('a')// &
            'end program main'

        test_unsupported_kind_rejected = expect_error_contains(source, &
            'only integer, real, and logical allocatable array components', &
            '/tmp/ffc_alloc_rank2_component_kind_reject')
    end function test_unsupported_kind_rejected

    logical function test_rank5_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: t'//new_line('a')// &
            '    integer, allocatable :: a(:,:,:,:,:)'//new_line('a')// &
            '  end type t'//new_line('a')// &
            'end program main'

        test_rank5_rejected = expect_error_contains(source, &
            'rank-1 through rank-4 intrinsic allocatable components', &
            '/tmp/ffc_alloc_rank2_component_rank5_reject')
    end function test_rank5_rejected

    logical function matches_gfortran(source, stem)
        ! gfortran is the independent behavioural oracle: compile and run the
        ! exact same source, then compare complete stdout/stderr byte-for-byte.
        character(len=*), intent(in) :: source, stem
        type(compiler_frontend_options_t) :: options
        type(compiler_frontend_result_t) :: frontend_result
        character(len=:), allocatable :: error_msg
        character(len=:), allocatable :: base, src, exe, ref, ffc_out, ref_out
        integer :: unit, exit_stat, status

        matches_gfortran = .false.
        base = '/var/tmp/ert/ffc_alloc_rank2_component_'//trim(stem)
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
            print *, 'FAIL[', trim(stem), ']: FortFront rejected source: ', &
                trim(frontend_result%diagnostic_text)
            return
        end if
        call lower_program_to_liric_exe(frontend_result%arena, &
            frontend_result%root_index, exe, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL[', trim(stem), ']: ffc lowering failed: ', trim(error_msg)
            return
        end if

        open (newunit=unit, file=src, status='replace', action='write')
        write (unit, '(A)') source
        close (unit)
        call execute_command_line('gfortran -w '//src//' -o '//ref, &
            exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL[', trim(stem), ']: gfortran rejected source'
            return
        end if
        call execute_command_line(exe//' > '//ffc_out//' 2>&1', &
            exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL[', trim(stem), ']: ffc executable failed'
            return
        end if
        call execute_command_line(ref//' > '//ref_out//' 2>&1', &
            exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL[', trim(stem), ']: gfortran executable failed'
            return
        end if
        call execute_command_line('diff '//ffc_out//' '//ref_out// &
            ' > /dev/null 2>&1', exitstat=status)
        if (status /= 0) then
            print *, 'FAIL[', trim(stem), ']: ffc output differs from gfortran'
            call execute_command_line('diff '//ffc_out//' '//ref_out)
            return
        end if
        call execute_command_line('rm -f '//src//' '//exe//' '//ref//' '// &
            ffc_out//' '//ref_out)
        matches_gfortran = .true.
    end function matches_gfortran

end program test_session_alloc_rank2_component_compiler
