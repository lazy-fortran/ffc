program test_session_allocatable_rank3_compiler
    ! Rank-three intrinsic allocatable owners through the direct LIRIC session.
    ! Positive behavior is compared with an independently compiled gfortran
    ! executable; negative cases pin the deliberately narrow owner contract.
    use ffc_test_support, only: expect_error_contains
    use fortfront_compiler, only: compiler_frontend_options_t, &
        compiler_frontend_result_t, compile_frontend_from_string, &
        INPUT_MODE_STANDARD
    use session_program_lowering, only: lower_program_to_liric_exe
    implicit none

    logical :: all_passed

    print *, '=== direct session rank-3 allocatable owner test ==='
    all_passed = .true.
    if (.not. test_runtime_lifecycle_and_dummy()) all_passed = .false.
    if (.not. test_runtime_whole_owner_copy()) all_passed = .false.
    if (.not. test_rank5_rejected()) all_passed = .false.
    if (.not. test_derived_component_rank5_rejected()) all_passed = .false.
    if (.not. test_unsupported_kind_rejected()) all_passed = .false.
    if (.not. test_pointer_rejected()) all_passed = .false.
    if (.not. test_target_rejected()) all_passed = .false.
    if (.not. test_alias_rejected()) all_passed = .false.
    if (.not. all_passed) stop 1
    print *, 'PASS: rank-3 intrinsic allocatable owners lower through LIRIC'

contains

    logical function test_runtime_lifecycle_and_dummy()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer, allocatable :: iv(:,:,:)'//new_line('a')// &
            '  real, allocatable :: rv(:,:,:)'//new_line('a')// &
            '  logical, allocatable :: lv(:,:,:)'//new_line('a')// &
            '  integer :: i, j, k, m, n, p, isum, lsum'//new_line('a')// &
            '  real :: rsum'//new_line('a')// &
            '  m = 2'//new_line('a')// &
            '  n = 2'//new_line('a')// &
            '  p = 3'//new_line('a')// &
            '  if (allocated(iv)) error stop 1'//new_line('a')// &
            '  allocate(iv(m,n,p))'//new_line('a')// &
            '  allocate(rv(m,n,p))'//new_line('a')// &
            '  allocate(lv(m,n,p))'//new_line('a')// &
            '  if (.not. allocated(iv)) error stop 2'//new_line('a')// &
            '  if (size(iv) /= m*n*p) error stop 3'//new_line('a')// &
            '  if (size(iv,1) /= m .or. size(iv,2) /= n .or. '// &
            'size(iv,3) /= p) error stop 4'//new_line('a')// &
            '  do k = 1, p'//new_line('a')// &
            '    do j = 1, n'//new_line('a')// &
            '      do i = 1, m'//new_line('a')// &
            '        iv(i,j,k) = 100*i + 10*j + k'//new_line('a')// &
            '        rv(i,j,k) = real(i) + 0.25*real(j) + 0.5*real(k)'// &
            new_line('a')// &
            '        lv(i,j,k) = mod(i+j+k, 2) == 0'//new_line('a')// &
            '      end do'//new_line('a')// &
            '    end do'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  isum = iv(1,1,1) + iv(2,2,3)'//new_line('a')// &
            '  rsum = rv(1,1,1) + rv(2,2,3)'//new_line('a')// &
            '  lsum = 0'//new_line('a')// &
            '  do k = 1, p'//new_line('a')// &
            '    do j = 1, n'//new_line('a')// &
            '      do i = 1, m'//new_line('a')// &
            '        if (lv(i,j,k)) lsum = lsum + 1'//new_line('a')// &
            '      end do'//new_line('a')// &
            '    end do'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  call touch(iv)'//new_line('a')// &
            '  print *, size(iv), size(iv,1), size(iv,2), size(iv,3), isum'// &
            new_line('a')// &
            '  print *, rsum, lsum, iv(2,2,3)'//new_line('a')// &
            '  deallocate(iv)'//new_line('a')// &
            '  deallocate(rv)'//new_line('a')// &
            '  deallocate(lv)'//new_line('a')// &
            '  if (allocated(iv) .or. allocated(rv) .or. allocated(lv)) '// &
            'error stop 5'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine touch(x)'//new_line('a')// &
            '    integer, allocatable, intent(inout) :: x(:,:,:)'//new_line('a')// &
            '    x(2,2,3) = x(2,2,3) + 1'//new_line('a')// &
            '  end subroutine touch'//new_line('a')// &
            'end program main'

        test_runtime_lifecycle_and_dummy = matches_gfortran(source, &
            'runtime_lifecycle_and_dummy')
    end function test_runtime_lifecycle_and_dummy

    logical function test_runtime_whole_owner_copy()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer, allocatable :: src(:,:,:), dst(:,:,:)'// &
            new_line('a')// &
            '  integer :: i, j, k, m, n, p'//new_line('a')// &
            '  m = 2'//new_line('a')// &
            '  n = 3'//new_line('a')// &
            '  p = 2'//new_line('a')// &
            '  allocate(src(m,n,p))'//new_line('a')// &
            '  allocate(dst(1,1,1))'//new_line('a')// &
            '  do k = 1, p'//new_line('a')// &
            '    do j = 1, n'//new_line('a')// &
            '      do i = 1, m'//new_line('a')// &
            '        src(i,j,k) = 100*i + 10*j + k'//new_line('a')// &
            '      end do'//new_line('a')// &
            '    end do'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  dst = src'//new_line('a')// &
            '  print *, allocated(dst), size(dst), size(dst,1), size(dst,2), '// &
            'size(dst,3)'//new_line('a')// &
            '  print *, dst(1,1,1), dst(2,3,2)'//new_line('a')// &
            '  deallocate(src)'//new_line('a')// &
            '  deallocate(dst)'//new_line('a')// &
            'end program main'

        test_runtime_whole_owner_copy = matches_gfortran(source, &
            'runtime_whole_owner_copy')
    end function test_runtime_whole_owner_copy

    logical function test_rank5_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, allocatable :: a(:,:,:,:,:)'//new_line('a')// &
            'end program main'

        test_rank5_rejected = expect_error_contains(source, &
            'rank-1 through rank-4 allocatables', &
            '/tmp/ffc_alloc_rank3_rank5_reject')
    end function test_rank5_rejected

    logical function test_derived_component_rank5_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: box_t'//new_line('a')// &
            '    integer, allocatable :: a(:,:,:,:,:)'//new_line('a')// &
            '  end type box_t'//new_line('a')// &
            '  type(box_t) :: x'//new_line('a')// &
            '  allocate(x%a(2,2,2,2))'//new_line('a')// &
            'end program main'

        test_derived_component_rank5_rejected = expect_error_contains(source, &
            'rank-1 through rank-4 intrinsic allocatable components', &
            '/tmp/ffc_alloc_rank3_derived_component_rank5_reject')
    end function test_derived_component_rank5_rejected

    logical function test_unsupported_kind_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  complex, allocatable :: a(:,:,:)'//new_line('a')// &
            'end program main'

        test_unsupported_kind_rejected = expect_error_contains(source, &
            'rank-3 allocatables support only integer, real, and logical', &
            '/tmp/ffc_alloc_rank3_kind_reject')
    end function test_unsupported_kind_rejected

    logical function test_pointer_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, pointer :: a(:,:,:,:)'//new_line('a')// &
            'end program main'

        test_pointer_rejected = expect_error_contains(source, &
            'supports rank-1, rank-2, and rank-3 fixed-size integer, real, logical, and complex pointer/target arrays only', &
            '/tmp/ffc_alloc_rank3_pointer_reject')
    end function test_pointer_rejected

    logical function test_target_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, allocatable, target :: a(:,:,:,:)'//new_line('a')// &
            'end program main'

        test_target_rejected = expect_error_contains(source, &
            'supports rank-1, rank-2, and rank-3 fixed-size integer, real, logical, and complex pointer/target arrays only', &
            '/tmp/ffc_alloc_rank3_target_reject')
    end function test_target_rejected

    logical function test_alias_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, allocatable :: a(:,:,:)'//new_line('a')// &
            '  allocate(a(2,2,2))'//new_line('a')// &
            '  associate(b => a)'//new_line('a')// &
            '    print *, size(b)'//new_line('a')// &
            '  end associate'//new_line('a')// &
            'end program main'

        test_alias_rejected = expect_error_contains(source, &
            'allocatable array aliases are not supported', &
            '/tmp/ffc_alloc_rank3_alias_reject')
    end function test_alias_rejected

    logical function matches_gfortran(source, stem)
        ! gfortran is an independent behavioral oracle: compile and run the
        ! exact same source with both compilers, then compare complete output.
        character(len=*), intent(in) :: source, stem
        type(compiler_frontend_options_t) :: options
        type(compiler_frontend_result_t) :: frontend_result
        character(len=:), allocatable :: error_msg
        character(len=:), allocatable :: base, src, exe, ref, ffc_out, ref_out
        integer :: unit, exit_stat, status

        matches_gfortran = .false.
        base = '/var/tmp/ert/ffc_alloc_rank3_owner_'//trim(stem)
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

end program test_session_allocatable_rank3_compiler
