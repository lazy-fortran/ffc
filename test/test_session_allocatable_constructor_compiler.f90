program test_session_allocatable_constructor
    ! Auto-reallocation on array constructor assignment (#244 slice B2c).
    ! Assigns [e1, e2, ...] to an integer 1-D allocatable: frees old data,
    ! allocates fresh storage, fills in order, then reads back via element access.
    use ffc_test_support, only: expect_exit_status
    use fortfront_compiler, only: compiler_frontend_options_t, &
        compiler_frontend_result_t, compile_frontend_from_string, &
        INPUT_MODE_STANDARD
    use session_program_lowering, only: lower_program_to_liric_exe
    implicit none

    logical :: all_passed

    print *, '=== direct session allocatable constructor compiler test ==='

    all_passed = .true.
    if (.not. test_constructor_assign_and_read()) all_passed = .false.
    if (.not. test_constructor_reassign()) all_passed = .false.
    if (.not. test_identifier_copy()) all_passed = .false.
    if (.not. test_lazy_209_array_operands()) all_passed = .false.
    if (.not. test_runtime_rank2_descriptor_copy_oracle()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: allocatable constructor assignment lowers through LIRIC'

contains

    logical function test_constructor_assign_and_read()
        ! a = [10, 20, 30]; stop a(2) -> exit status 20.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, allocatable :: a(:)'//new_line('a')// &
            '  allocate(a(3))'//new_line('a')// &
            '  a = [10, 20, 30]'//new_line('a')// &
            '  stop a(2)'//new_line('a')// &
            'end program main'

        test_constructor_assign_and_read = expect_exit_status( &
            source, 20, '/tmp/ffc_alloc_ctor_read')
    end function test_constructor_assign_and_read

    logical function test_constructor_reassign()
        ! Reassign to a different size; a(1) of the new [5, 6] should be 5.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, allocatable :: a(:)'//new_line('a')// &
            '  allocate(a(3))'//new_line('a')// &
            '  a = [10, 20, 30]'//new_line('a')// &
            '  a = [5, 6]'//new_line('a')// &
            '  stop a(1)'//new_line('a')// &
            'end program main'

        test_constructor_reassign = expect_exit_status( &
            source, 5, '/tmp/ffc_alloc_ctor_reassign')
    end function test_constructor_reassign

    logical function test_identifier_copy()
        ! a = b, both allocated rank-1 allocatables: elementwise copy.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, allocatable :: a(:)'//new_line('a')// &
            '  integer, allocatable :: b(:)'//new_line('a')// &
            '  allocate(a(3))'//new_line('a')// &
            '  allocate(b(3))'//new_line('a')// &
            '  b = [10, 20, 30]'//new_line('a')// &
            '  a = b'//new_line('a')// &
            '  stop a(2)'//new_line('a')// &
            'end program main'

        test_identifier_copy = expect_exit_status( &
            source, 20, '/tmp/ffc_alloc_ctor_copy')
    end function test_identifier_copy

    logical function test_lazy_209_array_operands()
        ! These are the Fortran forms emitted by FortFront for the two
        ! fortfront-lf/test_209_* corpus cases. Compare ffc's output with an
        ! independently compiled gfortran executable, including the values
        ! after each whole-array constructor reallocation.
        character(len=*), parameter :: all_source = &
            'program main'//new_line('a')// &
            '  integer, allocatable :: x(:)'//new_line('a')// &
            '  integer, allocatable :: y(:)'//new_line('a')// &
            '  integer, allocatable :: z(:)'//new_line('a')// &
            '  x = [1, 2]'//new_line('a')// &
            '  y = [3, 4]'//new_line('a')// &
            '  z = [5, 6]'//new_line('a')// &
            '  x = [x, 3]'//new_line('a')// &
            '  y = [y, 7]'//new_line('a')// &
            '  z = [z, 8]'//new_line('a')// &
            '  print *, x'//new_line('a')// &
            '  print *, y'//new_line('a')// &
            '  print *, z'//new_line('a')// &
            'end program main'
        character(len=*), parameter :: complex_source = &
            'program main'//new_line('a')// &
            '  real(8), allocatable :: x(:)'//new_line('a')// &
            '  real(8), allocatable :: y(:)'//new_line('a')// &
            '  x = [1.0_8, 2.0_8]'//new_line('a')// &
            '  y = [3.0_8, 4.0_8]'//new_line('a')// &
            '  x = [x, 5.0_8]'//new_line('a')// &
            '  y = [y, 6.0_8]'//new_line('a')// &
            '  print *, x'//new_line('a')// &
            '  print *, y'//new_line('a')// &
            'end program main'

        test_lazy_209_array_operands = &
            matches_gfortran(all_source, 'lazy_209_all') .and. &
            matches_gfortran(complex_source, 'lazy_209_complex') .and. &
            test_runtime_descriptor_copy_oracle()
    end function test_lazy_209_array_operands

    logical function test_runtime_descriptor_copy_oracle()
        ! The source extent is available only through b's runtime descriptor.
        ! The output checks allocation, extents, values after each reallocation,
        ! and that the copy owns independent storage after b is changed.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, allocatable :: a(:), b(:)'//new_line('a')// &
            '  integer :: n, i'//new_line('a')// &
            '  n = 4'//new_line('a')// &
            '  allocate(b(n))'//new_line('a')// &
            '  do i = 1, n'//new_line('a')// &
            '    b(i) = i * 10'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  allocate(a(2))'//new_line('a')// &
            '  a = [-1, -2]'//new_line('a')// &
            '  a = b'//new_line('a')// &
            '  print *, allocated(a), size(a), sum(a), a(1), a(4)'//new_line('a')// &
            '  deallocate(b)'//new_line('a')// &
            '  n = 2'//new_line('a')// &
            '  allocate(b(n))'//new_line('a')// &
            '  do i = 1, n'//new_line('a')// &
            '    b(i) = i + 6'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  a = b'//new_line('a')// &
            '  print *, allocated(a), size(a), sum(a), a(1), a(2)'//new_line('a')// &
            '  b(1) = 99'//new_line('a')// &
            '  print *, a(1), b(1)'//new_line('a')// &
            'end program main'

        test_runtime_descriptor_copy_oracle = matches_gfortran( &
            source, 'runtime_descriptor_copy')
    end function test_runtime_descriptor_copy_oracle

    logical function test_runtime_rank2_descriptor_copy_oracle()
        ! Differentially compare rank-2 integer, real, and logical copies with
        ! gfortran. Every source extent is runtime-only, each target changes
        ! shape twice, and the final source mutation proves the target owns a
        ! separate allocation rather than sharing the source data pointer.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, allocatable :: ai(:,:), bi(:,:)'//new_line('a')// &
            '  real, allocatable :: ar(:,:), br(:,:)'//new_line('a')// &
            '  logical, allocatable :: al(:,:), bl(:,:)'//new_line('a')// &
            '  integer :: m, n, i, j'//new_line('a')// &
            '  m = 2'//new_line('a')// &
            '  n = 3'//new_line('a')// &
            '  allocate(bi(m,n), br(m,n), bl(m,n))'//new_line('a')// &
            '  do j = 1, n'//new_line('a')// &
            '    do i = 1, m'//new_line('a')// &
            '      bi(i,j) = i * 10 + j'//new_line('a')// &
            '      br(i,j) = i * 10.0 + j'//new_line('a')// &
            '      bl(i,j) = mod(i + j, 2) == 0'//new_line('a')// &
            '    end do'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  allocate(ai(1,1), ar(1,1), al(1,1))'//new_line('a')// &
            '  al(1,1) = .false.'//new_line('a')// &
            '  ai = bi'//new_line('a')// &
            '  ar = br'//new_line('a')// &
            '  al = bl'//new_line('a')// &
            '  print *, allocated(ai), size(ai,1), size(ai,2), ai(1,1), ai(2,3)'// &
            new_line('a')// &
            '  print *, allocated(ar), size(ar,1), size(ar,2), ar(1,1), ar(2,3)'// &
            new_line('a')// &
            '  print *, allocated(al), size(al,1), size(al,2), al(1,1), al(2,3)'// &
            new_line('a')// &
            '  deallocate(bi, br, bl)'//new_line('a')// &
            '  m = 3'//new_line('a')// &
            '  n = 2'//new_line('a')// &
            '  allocate(bi(m,n), br(m,n), bl(m,n))'//new_line('a')// &
            '  do j = 1, n'//new_line('a')// &
            '    do i = 1, m'//new_line('a')// &
            '      bi(i,j) = 100 + i * 10 + j'//new_line('a')// &
            '      br(i,j) = 100.0 + i * 10.0 + j'//new_line('a')// &
            '      if (i + j == 2) then'//new_line('a')// &
            '        bl(i,j) = .false.'//new_line('a')// &
            '      else'//new_line('a')// &
            '        bl(i,j) = .true.'//new_line('a')// &
            '      end if'//new_line('a')// &
            '    end do'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  ai = bi'//new_line('a')// &
            '  ar = br'//new_line('a')// &
            '  al = bl'//new_line('a')// &
            '  print *, allocated(ai), size(ai,1), size(ai,2), ai(1,1), ai(3,2)'// &
            new_line('a')// &
            '  print *, allocated(ar), size(ar,1), size(ar,2), ar(1,1), ar(3,2)'// &
            new_line('a')// &
            '  print *, allocated(al), size(al,1), size(al,2), al(1,1), al(3,2)'// &
            new_line('a')// &
            '  bi(1,1) = -999'//new_line('a')// &
            '  br(1,1) = -999.0'//new_line('a')// &
            '  bl(1,1) = .not. bl(1,1)'//new_line('a')// &
            '  print *, ai(1,1), bi(1,1)'//new_line('a')// &
            '  print *, ar(1,1), br(1,1)'//new_line('a')// &
            '  print *, al(1,1), bl(1,1)'//new_line('a')// &
            'end program main'

        test_runtime_rank2_descriptor_copy_oracle = matches_gfortran( &
            source, 'runtime_rank2_descriptor_copy')
    end function test_runtime_rank2_descriptor_copy_oracle

    logical function matches_gfortran(source, stem)
        character(len=*), intent(in) :: source, stem
        type(compiler_frontend_options_t) :: options
        type(compiler_frontend_result_t) :: frontend_result
        character(len=:), allocatable :: error_msg, base, src, exe, ref
        character(len=:), allocatable :: ffc_out, ref_out
        integer :: unit, exit_stat, status

        matches_gfortran = .false.
        base = '/tmp/ffc_alloc_ctor_'//trim(stem)
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
            print *, 'FAIL[', trim(stem), ']: FortFront rejected source'
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
        call execute_command_line(exe//' > '//ffc_out, exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL[', trim(stem), ']: ffc executable failed'
            return
        end if
        call execute_command_line(ref//' > '//ref_out, exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL[', trim(stem), ']: gfortran executable failed'
            return
        end if
        call execute_command_line('diff -b '//ffc_out//' '//ref_out// &
            ' > /dev/null 2>&1', exitstat=status)
        if (status /= 0) then
            print *, 'FAIL[', trim(stem), ']: ffc output differs from gfortran'
            return
        end if
        call execute_command_line('rm -f '//src//' '//exe//' '//ref//' '// &
            ffc_out//' '//ref_out)
        matches_gfortran = .true.
    end function matches_gfortran

end program test_session_allocatable_constructor
